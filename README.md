# Vehicle Speed and Number Plate Detection System

Hey there! To get this project running follow the following steps:
1. Open the database_script.sql file in MySQL
2. Install all the dependencies and packages required in vsnpd.py and app.py
3. Update your database credentials in vsnpd.py
4. Run by typing this command in the terminal: python app.py
5. Register with username, email id and password.
5. I have uploaded a test video which you can upload and test.
		Real-World Distance (in meter) = 10
6. When you click on start processing for the first time, wait for a few seconds as the Yolov8M.pt file will be downloaded. 
7. Enjoy :)

In this project, I've used YoloV8M for vehicle detection and trained YoloV8s on custom dataset from Kaggle which include the number plates of vehicles. 

For number plate recognition, the system employs a custom YOLOv8 model for localization and PaddleOCR for accurate text extraction. All data—including vehicle speed, number plate, and timestamp is stored in a MySQL database and displayed on a web dashboard for real-time monitoring.

![Image](https://github.com/user-attachments/assets/a79a044c-ba75-4cad-b87d-f373199b636b)

![Image](https://github.com/user-attachments/assets/70fa589d-3d9f-4bf3-b9e2-8539ca1db606)

For any further suggestions or queries, mail me at: liandabre2262003@gmail.com
