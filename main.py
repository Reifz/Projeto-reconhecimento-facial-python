import threading
import cv2
from deepface import DeepFace

# Inicia a captura de vídeo pela webcam (usando DirectShow no Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Define a resolução da captura de vídeo
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0   # Contador de frames
face_match = False   # Variável para indicar se o rosto foi reconhecido ou não

# Carrega a imagem de referência (que será comparada com os rostos detectados)
reference_img = cv2.imread("img/reference.jpg")

# Função para verificar se o rosto atual corresponde ao da referência
def check_face(frame):
    global face_match
    try:
        # Usa o DeepFace para comparar a imagem do frame com a imagem de referência
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        # Caso não seja possível detectar rosto no frame, mantém como não correspondente
        face_match = False

# Loop principal de captura de vídeo
while True:
    ret, frame = cap.read()   # Lê um frame da webcam

    if ret:
        # A cada 30 frames, inicia uma thread para verificar o rosto
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        # Exibe na tela a mensagem de acordo com o resultado da verificação
        if face_match:
            cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_COMPLEX,
                        2, (0, 255, 0), 3)   # Texto verde se reconhecer
        else:
            cv2.putText(frame, "NO Match!", (20, 450), cv2.FONT_HERSHEY_COMPLEX,
                        2, (0, 0, 255), 3)   # Texto vermelho se não reconhecer

        # Mostra o vídeo em uma janela
        cv2.imshow("video", frame)

    # Pressionar 'q' encerra o programa
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Libera recursos
cv2.destroyAllWindows()
