import os
from fpdf import FPDF
import resend
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configurar API Key de Resend
resend.api_key = os.getenv("RESEND_API_KEY")

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 18)
        self.set_text_color(34, 139, 34)  # Forest Green
        self.cell(0, 12, 'CropAdvisor - Historial de Predicciones', 0, 1, 'C')
        self.ln(3)
        self.set_draw_color(34, 139, 34)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def generate_history_pdf(history_data, filename="historial.pdf", model_filter=None):
    """
    Genera PDF del historial.
    Si model_filter está presente, usa diseño específico para ese modelo.
    """
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    if model_filter:
        # === DISEÑO ESPECÍFICO PARA UN MODELO ===
        _generate_single_model_pdf(pdf, history_data, model_filter)
    else:
        # === DISEÑO COMPLETO (TODOS LOS MODELOS) ===
        _generate_all_models_pdf(pdf, history_data)
    
    pdf.output(filename)
    return filename

def _generate_single_model_pdf(pdf, history_data, model_name):
    """Diseño específico cuando se filtra por un solo modelo"""
    
    # Banner del modelo
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(103, 58, 183)  # Morado
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, f'Modelo: {model_name}', 0, 1, 'C', 1)
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f'Total de predicciones: {len(history_data)}', 0, 1, 'C')
    pdf.ln(3)
    
    for idx, item in enumerate(history_data, 1):
        if pdf.get_y() > 250:
            pdf.add_page()
        
        # Header de predicción
        pdf.set_font('Arial', 'B', 11)
        pdf.set_fill_color(240, 240, 250)
        pdf.set_text_color(103, 58, 183)
        fecha_str = item['fecha'].strftime('%d/%m/%Y %H:%M')
        pdf.cell(0, 8, f'Prediccion #{idx} - {fecha_str}', 0, 1, 'L', 1)
        pdf.ln(2)
        
        # Resultado del modelo (DESTACADO)
        prediccion_modelo = next((p for p in item['predicciones'] if p['modelo'] == model_name), None)
        cultivo_modelo = prediccion_modelo['cultivo'] if prediccion_modelo else 'N/A'
        
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(255, 235, 59)  # Amarillo
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f'Prediccion {model_name}: {cultivo_modelo}', 1, 1, 'C', 1)
        pdf.ln(3)
        
        # Tabla de datos de entrada (compacta)
        pdf.set_font('Arial', 'B', 9)
        pdf.set_fill_color(76, 175, 80)
        pdf.set_text_color(255, 255, 255)
        
        col_width = 27.14  # 190 / 7
        pdf.cell(col_width, 7, 'N', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'P', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'K', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'Temp', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'Hum', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'pH', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'Lluvia', 1, 1, 'C', 1)
        
        # Valores
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(255, 255, 255)
        
        inp = item['input']
        pdf.cell(col_width, 7, str(inp['N']), 1, 0, 'C')
        pdf.cell(col_width, 7, str(inp['P']), 1, 0, 'C')
        pdf.cell(col_width, 7, str(inp['K']), 1, 0, 'C')
        pdf.cell(col_width, 7, f"{inp['temperature']:.1f}", 1, 0, 'C')
        pdf.cell(col_width, 7, f"{inp['humidity']:.1f}", 1, 0, 'C')
        pdf.cell(col_width, 7, f"{inp['ph']:.1f}", 1, 0, 'C')
        pdf.cell(col_width, 7, f"{inp['rainfall']:.1f}", 1, 1, 'C')
        pdf.ln(2)
        
        # Info adicional: Resultado final del sistema
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, f'Resultado final del sistema: {item["cultivo_final"]} (Confianza: {item["confianza"]:.2f})', 0, 1, 'L')
        
        # Separador
        pdf.ln(3)
        pdf.set_draw_color(200, 200, 200)
        pdf.set_line_width(0.2)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

def _generate_all_models_pdf(pdf, history_data):
    """Diseño completo cuando se muestran todos los modelos"""
    
    pdf.set_font("Arial", size=9)
    
    for idx, item in enumerate(history_data, 1):
        if pdf.get_y() > 240:
            pdf.add_page()
        
        # Registro header
        pdf.set_font('Arial', 'B', 11)
        pdf.set_fill_color(240, 248, 255)
        pdf.set_text_color(0, 102, 204)
        fecha_str = item['fecha'].strftime('%d/%m/%Y %H:%M')
        pdf.cell(0, 8, f'Prediccion #{idx} - {fecha_str}', 0, 1, 'L', 1)
        pdf.ln(2)
        
        # TABLA 1: Datos de Entrada + Confianza
        pdf.set_font('Arial', 'B', 9)
        pdf.set_fill_color(76, 175, 80)
        pdf.set_text_color(255, 255, 255)
        
        col_width = 24
        pdf.cell(col_width, 7, 'N', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'P', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'K', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'Temp', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'Hum', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'pH', 1, 0, 'C', 1)
        pdf.cell(col_width, 7, 'Lluvia', 1, 0, 'C', 1)
        pdf.cell(24, 7, 'Conf.', 1, 1, 'C', 1)
        
        # Valores
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(255, 255, 255)
        
        inp = item['input']
        pdf.cell(col_width, 7, str(inp['N']), 1, 0, 'C')
        pdf.cell(col_width, 7, str(inp['P']), 1, 0, 'C')
        pdf.cell(col_width, 7, str(inp['K']), 1, 0, 'C')
        pdf.cell(col_width, 7, f"{inp['temperature']:.1f}", 1, 0, 'C')
        pdf.cell(col_width, 7, f"{inp['humidity']:.1f}", 1, 0, 'C')
        pdf.cell(col_width, 7, f"{inp['ph']:.1f}", 1, 0, 'C')
        pdf.cell(col_width, 7, f"{inp['rainfall']:.1f}", 1, 0, 'C')
        pdf.cell(24, 7, f"{item['confianza']:.2f}", 1, 1, 'C')
        
        # Predicción Final
        pdf.set_font('Arial', 'B', 10)
        pdf.set_fill_color(255, 235, 59)
        pdf.cell(0, 8, f'Cultivo Recomendado: {item["cultivo_final"]}', 1, 1, 'C', 1)
        pdf.ln(3)
        
        # TABLA 2: Predicciones de Modelos
        pdf.set_font('Arial', 'B', 9)
        pdf.set_fill_color(33, 150, 243)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 7, 'Predicciones Individuales por Modelo', 1, 1, 'C', 1)
        
        pdf.set_fill_color(200, 230, 255)
        pdf.set_text_color(0, 0, 0)
        model_col_width = 63.33
        pdf.cell(model_col_width, 7, 'Random Forest', 1, 0, 'C', 1)
        pdf.cell(model_col_width, 7, 'XGBoost', 1, 0, 'C', 1)
        pdf.cell(model_col_width, 7, 'SVM', 1, 1, 'C', 1)
        
        pdf.set_font('Arial', '', 9)
        pdf.set_fill_color(255, 255, 255)
        
        preds_dict = {p['modelo']: p['cultivo'] for p in item['predicciones']}
        
        pdf.cell(model_col_width, 7, preds_dict.get('Random Forest', 'N/A'), 1, 0, 'C')
        pdf.cell(model_col_width, 7, preds_dict.get('XGBoost', 'N/A'), 1, 0, 'C')
        pdf.cell(model_col_width, 7, preds_dict.get('SVM', 'N/A'), 1, 1, 'C')
        
        # Separador
        pdf.ln(5)
        pdf.set_draw_color(200, 200, 200)
        pdf.set_line_width(0.2)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

def send_history_email(to_email, pdf_path, user_name="Usuario", model_filter=None):
    if not resend.api_key:
        print("ADVERTENCIA: No se encontro RESEND_API_KEY. El correo no se enviara.")
        return

    with open(pdf_path, "rb") as f:
        pdf_content = f.read()

    # Personalizar mensaje según si hay filtro de modelo
    if model_filter:
        subject_line = f"Tu Reporte de {model_filter} - CropAdvisor"
        report_description = f"<strong>reporte exclusivo</strong> de las predicciones del modelo <strong>{model_filter}</strong>"
        model_info = f"<p style='background: #f0f0f0; padding: 15px; border-radius: 8px; margin: 15px 0;'><strong>Modelo seleccionado:</strong> {model_filter}</p>"
    else:
        subject_line = "Tu Reporte Completo - CropAdvisor"
        report_description = "<strong>reporte completo</strong> de tu historial de predicciones de cultivos"
        model_info = ""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 40px 20px;
                line-height: 1.6;
            }}
            .container {{ 
                max-width: 600px; 
                margin: 0 auto; 
                background: white;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}
            .header {{ 
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                padding: 40px 30px;
                text-align: center;
                color: white;
            }}
            .header h1 {{ 
                font-size: 32px;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }}
            .header p {{ 
                font-size: 16px;
                opacity: 0.95;
            }}
            .content {{ 
                padding: 40px 30px;
                background: #ffffff;
            }}
            .greeting {{ 
                font-size: 24px;
                color: #333;
                margin-bottom: 20px;
                font-weight: 600;
            }}
            .message {{ 
                color: #555;
                font-size: 16px;
                margin-bottom: 15px;
            }}
            .highlight-box {{ 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-left: 4px solid #4CAF50;
                padding: 20px;
                margin: 25px 0;
                border-radius: 8px;
            }}
            .highlight-box h3 {{ 
                color: #4CAF50;
                margin-bottom: 10px;
                font-size: 18px;
            }}
            .highlight-box p {{ 
                color: #555;
                font-size: 14px;
            }}
            .icon {{ 
                font-size: 48px;
                margin-bottom: 15px;
            }}
            .footer {{ 
                background: #f8f9fa;
                padding: 30px;
                text-align: center;
                border-top: 1px solid #e0e0e0;
            }}
            .footer p {{ 
                color: #888;
                font-size: 13px;
                margin-bottom: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="icon">🌱</div>
                <h1>CropAdvisor</h1>
                <p>Inteligencia Artificial para Agricultura Sostenible</p>
            </div>
            
            <div class="content">
                <p class="greeting">¡Hola, {user_name}! 👋</p>
                
                <p class="message">
                    Nos complace enviarte el {report_description}.
                </p>
                
                {model_info}
                
                <div class="highlight-box">
                    <h3>📊 ¿Qué encontrarás en el reporte?</h3>
                    <p>
                        ✓ Todas tus predicciones organizadas cronológicamente<br>
                        ✓ Análisis detallado de condiciones de suelo (N, P, K, pH)<br>
                        ✓ Factores climáticos evaluados (temperatura, humedad, lluvia)<br>
                        {"✓ Predicciones del modelo " + model_filter + "<br>" if model_filter else "✓ Recomendaciones de nuestros 3 modelos de IA<br>"}
                        ✓ Nivel de confianza de cada predicción
                    </p>
                </div>
                
                <p class="message">
                    Este documento ha sido generado por nuestros modelos de inteligencia artificial de última generación, 
                    analizando las condiciones específicas de suelo y clima que ingresaste.
                </p>
                
                <p class="message" style="margin-top: 30px; color: #4CAF50; font-weight: 600;">
                    ¡Gracias por confiar en CropAdvisor para optimizar tus cultivos! 🚜
                </p>
            </div>
            
            <div class="footer">
                <p style="font-weight: 600; color: #4CAF50;">CropAdvisor</p>
                <p>&copy; 2025 CropAdvisor. Todos los derechos reservados.</p>
                <p style="margin-top: 10px; font-size: 12px;">
                    Este correo fue generado automáticamente. Por favor no responder.
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    params = {
        "from": "CropAdvisor <onboarding@resend.dev>",
        "to": [to_email],
        "subject": f"📊 {subject_line}",
        "html": html_content,
        "attachments": [
            {
                "filename": "Reporte_CropAdvisor.pdf",
                "content": list(pdf_content)
            }
        ]
    }

    try:
        email = resend.Emails.send(params)
        return email
    except Exception as e:
        print(f"Error enviando correo: {e}")
        raise e