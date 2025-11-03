#!/usr/bin/env python3
"""
Generate PDF documentation from HTML
Requires: pip install weasyprint
"""

from weasyprint import HTML, CSS
import os

def generate_pdf():
    """Generate professional PDF from HTML documentation"""
    
    print("🔄 Generating PDF documentation...")
    
    # Input and output paths
    html_file = "api_documentation.html"
    output_pdf = "UIUC_CHAT_API_Documentation.pdf"
    
    # Check if HTML file exists
    if not os.path.exists(html_file):
        print(f"❌ Error: {html_file} not found!")
        print("   Please create the HTML file first.")
        return False
    
    try:
        # Generate PDF
        HTML(filename=html_file).write_pdf(
            output_pdf,
            stylesheets=[CSS(string='''
                @page {
                    size: A4;
                    margin: 2cm;
                }
            ''')]
        )
        
        print(f"✅ PDF generated successfully: {output_pdf}")
        print(f"📊 File size: {os.path.getsize(output_pdf) / 1024:.2f} KB")
        return True
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False

if __name__ == "__main__":
    generate_pdf()