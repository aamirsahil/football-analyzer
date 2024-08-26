from ultralytics import YOLO

def main():
    # data loc
    model_name = 'models/best.pt'
    root = 'data/'
    file_name = '08fd33_4.mp4'
    path = root + file_name
    # define model and predict
    model = YOLO(model_name)
    results = model.predict(path, save=True)
    # print result
    print(results[0])
    print('+++++++++++++++++++++++++++++++++++++++++++++++')
    for box in results[0].boxes:
        print(box)
        print('================================================')

if __name__ == "__main__":
    main()