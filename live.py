import cv2
import numpy as np
import pytesseract
from pytesseract import Output

LANG = "por+eng"
CFG = "--psm 6"
ESCALA = 0.5          # reduz a imagem para o OCR ficar mais rápido
CONF_MIN = 60
OCR_A_CADA = 15       # frames entre cada OCR automático (0 = só manual)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("Câmera não encontrada. Confira com: ls /dev/video*")

def preprocessar(frame):
    pequeno = cv2.resize(frame, None, fx=ESCALA, fy=ESCALA)
    gray = cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.dilate(binaria, np.ones((2, 2), np.uint8))

def rodar_ocr(frame):
    d = pytesseract.image_to_data(preprocessar(frame), lang=LANG,
                                  config=CFG, output_type=Output.DICT)
    caixas = []
    for i in range(len(d["text"])):
        texto = d["text"][i].strip()
        if texto and float(d["conf"][i]) > CONF_MIN:
            x, y, w, h = (int(d[k][i] / ESCALA) for k in ("left", "top", "width", "height"))
            caixas.append((texto, x, y, w, h))
    return caixas

caixas, n = [], 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao ler a câmera")
        break

    n += 1
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("q"):
        break
    if tecla == ord(" ") or (OCR_A_CADA and n % OCR_A_CADA == 0):
        caixas = rodar_ocr(frame)
        if caixas:
            print(" ".join(c[0] for c in caixas))

    for texto, x, y, w, h in caixas:   # desenha o último resultado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, texto, (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("OCR", frame)

cap.release()
cv2.destroyAllWindows()
