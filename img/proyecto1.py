import cv2
import pytesseract
import numpy as np
import streamlit as st
from PIL import Image
import csv
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\juanp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def preprocesar_imagen_mejorado(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 100, 200)    # Aplicar el detector de bordes Canny
    return edges

def detectar_placa_mejorada(image):
    preprocesada = preprocesar_imagen_mejorado(image)
    contornos, _ = cv2.findContours(preprocesada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contorno)
            if 1.5 <= w / h <= 5:  # Ampliamos un poco el rango de relación de aspecto
                if w > 80 and h > 20:  # Ajustamos loimport cv2s requisitos de tamaño mínimo
                    placa = image[y:y+h, x:x+w]
                    return placa
    return None

def extraer_texto_placa(image):
    placa_bin = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY_INV)[1]
    texto = pytesseract.image_to_string(placa_bin, config='--psm 8')
    return texto.strip()

# Función para guardar los resultados en un archivo CSV
def save_to_csv(data, filename="placas_escaneadas.csv"):
    # Verificar si el archivo CSV existe, si no, crear uno nuevo con los encabezados
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            # Escribir los encabezados si el archivo no existe
            writer.writerow(["Imagen", "Texto de la Placa"])
        
        # Escribir los datos en el archivo CSV
        writer.writerow(data)

def main():
    st.title("Detección de Placa del Auto")
    uploaded_file = st.file_uploader("Sube una imagen del auto", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Abrir la imagen subida y convertirla a formato numpy (array)
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        placa_mejorada = detectar_placa_mejorada(image_np)
        
        if placa_mejorada is not None:
            texto_placa_mejorada = extraer_texto_placa(placa_mejorada)
            st.write(f"Texto extraído de la placa: {texto_placa_mejorada}")
            
            st.image(image, caption=f"Ticket N° 12\nPlaca: {texto_placa_mejorada}", use_column_width=True)
            
            save_to_csv([uploaded_file.name, texto_placa_mejorada])
            st.success(f"Resultado guardado en el archivo CSV.")
        else:
            st.write("No se detectó ninguna placa en la imagen.")

if __name__ == "__main__":
    main()
