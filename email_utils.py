import os
from fpdf import FPDF
import resend
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configurar API Key de Resend (asegúrate de tenerla en tus variables de entorno)
resend.api_key = os.getenv("RESEND_API_KEY")

class PDF(FPDF):
    def header(self):
        # Logo placeholder or Title
        self.set_font('Arial', 'B', 16)
        self.set_text_color(34, 139, 34)  # Forest Green
        self.cell(0, 10, 'CropAdvisor - Historial de Predicciones', 0, 1, 'C')
        self.ln(5)
        
        # Table Header
        self.set_fill_color(200, 220, 255)
        self.set_text_color(0)
        self.set_font('Arial', 'B', 10)
        
        # Columns: Fecha, Cultivo, Conf, Condiciones, Modelos
        self.cell(35, 10, 'Fecha', 1, 0, 'C', 1)
        self.cell(30, 10, 'Cultivo', 1, 0, 'C', 1)
        self.cell(20, 10, 'Conf.', 1, 0, 'C', 1)
        self.cell(65, 10, 'Condiciones (N-P-K-T-H-pH-R)', 1, 0, 'C', 1)
        self.cell(40, 10, 'Modelos', 1, 1, 'C', 1)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

def generate_history_pdf(history_data, filename="historial.pdf"):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Arial", size=9)

    for item in history_data:
        # Calculate height needed
        # We need to wrap text for 'Condiciones' and 'Modelos'
        
        fecha_str = item['fecha'].strftime('%Y-%m-%d\n%H:%M')
        cultivo = item['cultivo_final']
        confianza = f"{item['confianza']}"
        
        condiciones = (f"N:{item['input']['N']} P:{item['input']['P']} K:{item['input']['K']}\n"
                       f"T:{item['input']['temperature']} H:{item['input']['humidity']}\n"
                       f"pH:{item['input']['ph']} R:{item['input']['rainfall']}")
        
        modelos = "\n".join([f"{p['modelo'][:10]}..: {p['cultivo']}" for p in item['predicciones']])

        # Determine max height of the row based on content
        # Estimate lines needed
        lines_fecha = 2
        lines_cultivo = 1
        lines_conf = 1
        lines_cond = 3
        lines_mod = len(item['predicciones'])
        
        max_lines = max(lines_fecha, lines_cultivo, lines_conf, lines_cond, lines_mod)
        row_height = max_lines * 5  # 5mm per line approx
        
        # Check page break
        if pdf.get_y() + row_height > 270:
            pdf.add_page()
            
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        
        # Draw cells
        # Fecha
        pdf.multi_cell(35, row_height / lines_fecha if lines_fecha > 0 else row_height, fecha_str, 1, 'C')
        pdf.set_xy(x_start + 35, y_start)
        
        # Cultivo
        pdf.cell(30, row_height, cultivo, 1, 0, 'C')
        
        # Confianza
        pdf.cell(20, row_height, confianza, 1, 0, 'C')
        
        # Condiciones
        x_cond = pdf.get_x()
        pdf.multi_cell(65, row_height / 3, condiciones, 1, 'L')
        pdf.set_xy(x_cond + 65, y_start)
        
        # Modelos
        pdf.multi_cell(40, row_height / max(1, len(item['predicciones'])), modelos, 1, 'L')
        
        # Move to next row
        pdf.set_y(y_start + row_height)

    pdf.output(filename)
    return filename

def send_history_email(to_email, pdf_path):
    if not resend.api_key:
        print("ADVERTENCIA: No se encontró RESEND_API_KEY. El correo no se enviará.")
        return

    with open(pdf_path, "rb") as f:
        pdf_content = f.read()

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; color: #333; line-height: 1.6; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #f9f9f9; }
            .header { text-align: center; padding-bottom: 20px; border-bottom: 2px solid #4CAF50; }
            .header h1 { color: #4CAF50; margin: 0; }
            .content { padding: 20px 0; }
            .footer { text-align: center; font-size: 12px; color: #888; margin-top: 20px; border-top: 1px solid #e0e0e0; padding-top: 10px; }
            .button { display: inline-block; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌱 CropAdvisor</h1>
            </div>
            <div class="content">
                <p>Hola,</p>
                <p>Adjunto encontrarás el reporte detallado de tu historial de predicciones de cultivos.</p>
                <p>Este documento contiene un resumen de todas las recomendaciones generadas por nuestros modelos de inteligencia artificial, basándose en las condiciones de suelo y clima que ingresaste.</p>
                <p>¡Gracias por usar CropAdvisor!</p>
            </div>
            <div class="footer">
                <p>&copy; 2025 CropAdvisor. Todos los derechos reservados.</p>
            </div>
        </div>
    </body>
    </html>
    """

    params = {
        "from": "CropAdvisor <onboarding@resend.dev>",
        "to": [to_email],
        "subject": "📊 Tu Reporte de Predicciones - CropAdvisor",
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
