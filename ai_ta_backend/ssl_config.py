import os
import ssl
import warnings
import urllib3

warnings.filterwarnings('ignore', message='Unverified HTTPS request')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['SSL_CERT_FILE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

ssl._create_default_https_context = ssl._create_unverified_context

print("SSL verification disabled")
