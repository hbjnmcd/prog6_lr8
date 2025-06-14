import cv2
import os

def highlightFace(net, frame, ageNet, genderNet, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

            face_img = frameOpencvDnn[max(0,y1):min(y2, frameHeight-1), max(0,x1):min(x2, frameWidth-1)]
            face_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(face_blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(face_blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            label = f"{gender}, {age}"
            cv2.putText(frameOpencvDnn, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    return frameOpencvDnn, faceBoxes

def process_image_file(net, ageNet, genderNet, filename, scale=0.5):
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден.")
        return
    image = cv2.imread(filename)
    if image is None:
        print(f"Не удалось загрузить изображение {filename}.")
        return
    resultImg, faceBoxes = highlightFace(net, image, ageNet, genderNet)
    if not faceBoxes:
        print("Лица не распознаны")
    else:
        print(f"Найдено лиц: {len(faceBoxes)}")

    # Масштабируем изображение перед показом
    width = int(resultImg.shape[1] * scale)
    height = int(resultImg.shape[0] * scale)
    resized_img = cv2.resize(resultImg, (width, height))
    cv2.imshow("Face detection (image)", resized_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_camera(net, ageNet, genderNet, scale=0.5):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Не удалось открыть камеру")
        return
    print("Нажмите любую клавишу для выхода...")
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            print("Кадр не получен, завершаем работу")
            break
        resultImg, faceBoxes = highlightFace(net, frame, ageNet, genderNet)
        if not faceBoxes:
            print("Лица не распознаны")

        width = int(resultImg.shape[1] * scale)
        height = int(resultImg.shape[0] * scale)
        resized_img = cv2.resize(resultImg, (width, height))
        cv2.imshow("Face detection (camera)", resized_img)

        if cv2.waitKey(1) >= 0:
            break
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys

    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    scale = 0.7 # масштаб окна (например, 0.5 — уменьшение в 2 раза)

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        process_image_file(faceNet, ageNet, genderNet, filename, scale)
    else:
        process_camera(faceNet, ageNet, genderNet, scale)
