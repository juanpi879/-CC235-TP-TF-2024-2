
import cv2
import pytesseract
import numpy as np
import streamlit as st
from PIL import Image
import random
from datetime import datetime
import csv
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
                if w > 80 and h > 20:  # Ajustamos los requisitos de tamaño mínimo
                    placa = image[y:y+h, x:x+w]
                    return placa
    return None

def extraer_texto_placa(image):
    placa_bin = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY_INV)[1]
    texto = pytesseract.image_to_string(placa_bin, config='--psm 8')
    return texto.strip()

def generar_ticket(placa):
    ticket_numero = random.randint(1000, 9999)
    hora_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    costo_por_hora = 6
    ticket = f"""TICKET N° {ticket_numero}
Placa: {placa}
Fecha y Hora: {hora_actual}
Costo: S/. {costo_por_hora}.00 por hora"""
    return ticket_numero, placa, hora_actual, costo_por_hora

def guardar_ticket_csv(ticket_numero, placa, hora_actual, costo_por_hora):
    archivo_csv = "tickets.csv"
    existe = os.path.isfile(archivo_csv)
    with open(archivo_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not existe:
            writer.writerow(["Ticket N°", "Placa", "Fecha y Hora", "Costo"])
        writer.writerow([ticket_numero, placa, hora_actual, f"S/. {costo_por_hora}.00"])

st.title("Sistema de ticket para estacionamiento")

# Subir la imagen
uploaded_file = st.file_uploader("Sube una imagen de la placa", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Mostrar la imagen original
    st.image(image, caption="Imagen original subida", use_column_width=True)
    
    # Detectar la placa
    placa_detectada = detectar_placa_mejorada(image_np)
    
    if placa_detectada is not None:
        # Extraer texto de la placa
        texto_placa = extraer_texto_placa(placa_detectada)
        
        if texto_placa:
            st.success(f"Placa detectada: {texto_placa}")
            
            # Generar el ticket
            ticket_numero, placa, hora_actual, costo_por_hora = generar_ticket(texto_placa)
            ticket = f"""TICKET N° {ticket_numero}
Placa: {placa}
Fecha y Hora: {hora_actual}
Costo: S/. {costo_por_hora}.00 por hora"""
            st.text_area("Ticket generado:", ticket, height=150)
            
            # Guardar en CSV
            guardar_ticket_csv(ticket_numero, placa, hora_actual, costo_por_hora)
            st.success("El ticket se ha guardado en el archivo 'tickets.csv'.")
        else:
            st.error("No se pudo extraer texto de la placa detectada.")
    else:
        st.error("No se detectó ninguna placa en la imagen.")
