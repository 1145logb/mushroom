# mushroom
these codes include mushroom object, white wdge, edge, bite by bug and get mouldy detection by use yolov8

necessary install library:
cuda & cudnn install: https://medium.com/@zera.tseng888/%E5%9C%A8windows11%E7%92%B0%E5%A2%83%E4%B8%8B%E5%AE%89%E8%A3%9Dcuda%E8%88%87cudnn-dd85575187ae

python version:3.12.8

after finish install use bash to install yolov8 and pytorch:

installation code:

  install pytorch with gpu version
  
    conda install pytorch torchvision torchaudio pytorch-cuda=your cuda version -c pytorch -c nvidia
    
  install pytorch with cpu version
  
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

  install yolov8

    pip install ultralytics

If all installed use python code to check whether they install successful:

  check pytorch

    import torch
    print(torch.cuda.is_available())  # True 表示可以使用 GPU
    print(torch.cuda.get_device_name(0))  # 應該會顯示顯卡

  check yolov8

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # 載入 YOLOv8 最小模型
    results = model("path/to/your/image.jpg")  # 推論
    results.show()



