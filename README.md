# Projeto reconhecimento facial utilizando python.

## Passos

### 1. Instale o pacote para ler faces
  python -m pip install --upgrade pip

  pip install opencv-python

  pip install deepface

  pip install tf-keras
 
### 2. Crie uma pasta para imagens e coloque uma foto que será validada.s
  Ex: img/reference.jpg

### 3. Verifique o indice da sua Camêra de teste
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   ##0,1,2,3.....


