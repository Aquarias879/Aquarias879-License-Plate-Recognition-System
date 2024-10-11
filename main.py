import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import messagebox
import cv2
import threading
import queue
import os
from PIL import Image, ImageTk, ImageDraw, ImageFont
import logging
from datetime import datetime
import time
from license_detect import ObjectDetector as Ort
import json
import sys
import pyodbc
import requests

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Global variables for RTSP URLs and license address
rtsp_urls = []  # List of RTSP URLs
license_address = None
video_streams = []  # List of VideoCapture instances
root = None
update_frame_id = None
cctv_labels = []  # List of labels (cctv1, cctv2, cctv3, cctv4)

# Global variables for plate frames and detections
plate_frame1, plate_frame2, plate_frame3 = None, None, None
latest_detections1, latest_detections2, latest_detections3 = [], [], []
detected_classes = []
detected_classes_lock = threading.Lock()
url_entry = None
url_exit1 = None
url_exit2 = None

door_status = {
    '1': False,
    '2': False,
    '3': False
}

boot_called = False

base_dir = os.path.dirname(__file__)
icon_path = os.path.join(base_dir,"static","icon", "cctv.ico")
onnx_model1 = os.path.join(base_dir,"static","models", "plate.onnx")
onnx_model2 = os.path.join(base_dir,"static","models", "text.onnx")

class_names = ['-']
class_names2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8',
                '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize the object detectors
plate_detector = Ort(onnx_model1, class_names,  conf_thres=0.6,  iou_thres=0.8)  #0.6 0.8
text_detector  = Ort(onnx_model2, class_names2, conf_thres=0.65, iou_thres=0.8)
error_handled_event = threading.Event()

# Define VideoCapture class to handle video capture and detection
class VideoCapture:
    def __init__(self, rtsp_url, label, text_detector, plate_detector, plate_frame, latest_detections, ch, url_gate, license_address):
        self.rtsp_url = rtsp_url
        self.plate_detector = plate_detector
        self.text_detector = text_detector
        self.cap = None
        self.q = queue.Queue()
        self.running = True
        self.thread_started = False
        self.label = label  # Reference to the label to display error messages
        self.plate_frame = plate_frame  # Reference to the scrollable frame
        self.detection_labels = []  # List to keep track of labels
        self.latest_detections = latest_detections  # List to store latest detections
        self.url_gate = url_gate
        self.license_address = license_address
    
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()
        self.ch = ch

        self.boot_called = boot_called
        self.boot_lock = threading.Lock()

    def _reader(self):
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_FPS, 29.99)
            while not self.cap.isOpened():
                logging.error(f"Cannot open RTSP stream: {self.rtsp_url}")
                self.update_label_error("Cannot open stream")
                self.running = False
                return
    
            self.thread_started = True
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error(f"Failed to read frame from {self.rtsp_url}.")
                    self.update_label_error("Failed to read frame")
                    self.running = False
                    return
                if not self.q.empty():
                    self.q.get_nowait()
                self.q.put(frame)

        except Exception as e:
            logging.error(f"Exception in _reader: {e}")
            # Use the shared error event
            if self.running and not error_handled_event.is_set():
                error_handled_event.set()  # Set the event to indicate the error has been handled
                self.handle_loading()  # Show the error popup only once
            self.running = False  # Stop the thread

        finally:
            if self.cap:
                self.cap.release()

    '''
    def read(self):
        if not self.q.empty():
            return self.q.get()
        else:
            return None'''
    
    def read(self):
        try:
            return self.q.get()
        except Exception as e:
            logging.error(f"Error in read: {e}")
            return None

    def stop(self):
        try:
            logging.debug(f"Stopping VideoCapture for {self.rtsp_url}")
            self.running = False
            if threading.current_thread() != self.t and self.thread_started:
                self.t.join()
        
        except Exception as e:
            logging.error(f"Error in stopping VideoCapture: {e}")
        
        finally:
            if not self.boot_called:  # Check the flag before calling
                self.boot_called = True
                logging.debug(f"Calling reboot....")
        
    def detect_objects(self, frame):
        try:
            height, width, _ = frame.shape
            #roi_x, roi_y, roi_width, roi_height = 0, height // 2, width // 1, height //1
            roi_x, roi_y = 0, height // 4  # Start a bit lower
            roi_width, roi_height = width, height * 3 // 4  # Use 3/4 of the height
            
            cropped_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 1)

            # Detect plates
            boxes, scores, class_ids = self.plate_detector(cropped_frame)

            for j, box in enumerate(boxes):
                x, y, w, h = box
                cv2.rectangle(frame, (int(x) + roi_x, int(y) + roi_y), (int(w) + roi_x, int(h) + roi_y),
                              (0, 255, 0), 1)
                plate_image = cropped_frame[max(0, int(y - 10)):int(y + h + 10),
                                            max(0, int(x - 10)):int(x + w + 10)]

                if plate_image.size == 0:
                    logging.error("Empty plate image, skipping detection.")
                    continue

                try:
                    plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                except cv2.error as e:
                    logging.error(f"Failed to convert plate image to grayscale: {e}")
                    continue

                # Detect text (license number) on the plate
                boxes_text, scores_text, class_ids_text = self.text_detector(plate_gray)
                try:
                    sorted_preds = sorted(zip(boxes_text, class_ids_text), key=lambda x: x[0][0])
                    license_number = ''.join([class_names2[class_id] for _, class_id in sorted_preds])

                    # Add to detected classes if not already detected
                    with detected_classes_lock:
                        if license_number not in detected_classes:
                            detected_classes.append(license_number)
                            now = datetime.now()
                            taiwan_year = now.year - 1911
                            date = f"{taiwan_year}{now.strftime('%m%d')}"
                            time = datetime.now().strftime("%H%M%S")

                            # Create a JSON object with license number, channel, and timestamp
                            data = {
                                "car_no": license_number,
                                "type": self.ch,  # Channel (e.g., "Entry", "Exit1", "Exit2")
                                "add_date": date,
                                "add_time": time
                            }
                            self.latest_detections.append(data)
                            # Limit to the last 2 detections
                            if len(self.latest_detections) > 1:
                                self.latest_detections.pop(0)
                            
                            # Update the corresponding plate frame
                            self.update_plate_listbox()
                            pos = {
                                    "x"     : x + roi_x,
                                    "y"     : y + roi_y,
                                  }
                                
                            #controling(frame, self.license_address, data, self.url_gate,pos)
                            threading.Thread(target=controling, args=(frame, license_address, data, self.url_gate, pos)).start()
                            detected_classes.clear()

                except Exception as e:
                    logging.error(f"Error processing detected text on plate: {e}")
        except Exception as e:
            logging.error(f"Error in detect_objects: {e}")

        return frame

    def update_label_error(self, message):
        img = Image.new('RGB', (640, 480), color='red')
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        text_x = 100
        text_y = 180
        draw.text((text_x, text_y), message, font=font, fill='white')

        img_ctk = CTkImage(light_image=img, size=(640, 480))  # Use CTkImage
        self.label.config(image=img_ctk, text="")
        self.label.image = img_ctk

    def update_plate_listbox(self):
        """ Updates the plate frame with the latest detections """
        # Clear existing labels
        for label in self.detection_labels:
            label.destroy()
        self.detection_labels.clear()

        for detection in self.latest_detections:
            add_date = detection['add_date']  # e.g., '1130925'
            add_time = detection['add_time'] 
            # Convert the date and time strings to datetime objects
            date = f"{add_date[:3]}/{add_date[3:5]}/{add_date[5:]}"
            time = datetime.strptime(add_time, '%H%M%S').strftime('%H:%M:%S')

            num_label = ctk.CTkLabel(self.plate_frame, text=f"車牌號碼: {detection['car_no']}", font=("SimSun", 20))
            num_label.pack(anchor='w')
            self.detection_labels.append(num_label)

            ch_label = ctk.CTkLabel(self.plate_frame, text=f"頻道: {detection['type']}", font=("SimSun", 18))
            ch_label.pack(anchor='w')
            self.detection_labels.append(ch_label)

            date_label = ctk.CTkLabel(self.plate_frame, text=f"日期: {date}", font=("SimSun", 18))
            date_label.pack(anchor='w')
            self.detection_labels.append(date_label)

            time_label = ctk.CTkLabel(self.plate_frame, text=f"時間: {time}", font=("SimSun", 18))
            time_label.pack(anchor='w')
            self.detection_labels.append(time_label)

            empty_label = ctk.CTkLabel(self.plate_frame, text="")
            empty_label.pack(anchor='w')
            self.detection_labels.append(empty_label)

    def handle_loading(self):
        dialog = ctk.CTkToplevel()
        dialog.title("警告!")
        dialog.geometry("300x150")
        dialog.iconbitmap(icon_path)
        dialog.resizable(False, False)
        dialog.grab_set()  # Make the dialog modal
        ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

        label = ctk.CTkLabel(dialog, text="CCTV 出現問題，您需要重新啟動嗎？", font=("SimSun", 12))
        label.pack(pady=20)

        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(pady=10)

        def on_no_button_click():
            on_closing()
            dialog.destroy()

        def on_yes_button_click():
            restart_program()
            dialog.destroy()

        yes_button = ctk.CTkButton(button_frame, text="是", command=on_yes_button_click)
        yes_button.pack(side=ctk.LEFT, padx=10)

        no_button  = ctk.CTkButton(button_frame, text="否", command=on_no_button_click)
        no_button.pack(side=ctk.LEFT, padx=10)

        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')

        dialog.mainloop()

def save_output(frame, data, pos):
    try:
        save_path = os.path.join('license_frame', f"{data['car_no']}_{data['type']}_{data['add_date']}_{data['add_time']}.jpg")
        
        os.makedirs('license_frame', exist_ok=True)
        
        pos_x = int(pos['x'])
        pos_y = int(pos['y'])
        
        cv2.putText(frame, data['car_no'], (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imwrite(save_path, frame)

    except OSError as e:
        # Handle file or directory errors
        print(f"File system error occurred: {e}")
    except cv2.error as e:
        # Handle OpenCV-related errors
        print(f"OpenCV error occurred: {e}")
    except KeyError as e:
        # Handle missing keys in 'data' or 'pos' dictionaries
        print(f"Missing key in data or position: {e}")
    except Exception as e:
        # Catch-all for any other unexpected errors
        print(f"An unexpected error occurred: {e}")

def controling(frame, addr, data, url_gate,pos):
    channel = data['type']
    license_number = data['car_no']
    #save_output(frame,data,pos)
    try:
        conn = connection()
        cursor = conn.cursor()

        # Check if the door is already open (based on the flag)
        if door_status.get(channel):
            print(f"{channel} gate is already open. Skipping further actions.")
            return
        else:
            #query = "SELECT COUNT(1) FROM carout_List WHERE car_no = ?"
            #cursor.execute(query, (license_number,))
            #result = cursor.fetchone()  # Fetch the result
            # Handle "Entry" and "Exit" cases differently
            if channel == "1":  # Entry
                #license_number = data['car_no']
                query = "SELECT COUNT(1) FROM canin_List WHERE car_no = ?"
                cursor.execute(query, (license_number,))
                result = cursor.fetchone()  # Fetch the result
                if result and result[0] == 1:
                    # License number exists, proceed to post the data to the API
                    print(f"License {license_number} exists in the database for {channel}. Posting to API.")
                    door_open(data, url_gate, channel)  # Pass url_map and channel to door_open
                    insert_data(data)
                    save_output(frame,data,pos)
                else:
                    # License number doesn't exist in the database for "Entry"
                    print(f"License {license_number} not found in the database for {channel}.")
            
            # For Exit1 and Exit2
            elif channel in ["2", "3"]:
                #license_number = data['car_no']
                query = "SELECT COUNT(1) FROM canout_List WHERE car_no = ?"
                cursor.execute(query, (license_number,))
                result = cursor.fetchone()  # Fetch the result
                #if license_number == "6481Q8":
                if result and result[0] == 1:
                    # Special case for license number 6481Q8
                    print(f"Special case: License {license_number} found. Posting to API for {channel}.")
                    door_open(data, url_gate, channel)  # Pass url_map and channel to door_open
                    insert_data(data)
                    save_output(frame,data,pos)
                else:
                    # License number doesn't exist in the database for Exit1 or Exit2
                    print(f"License {license_number} not found in the database for {channel}.")

        cursor.close()
        conn.close()

    except pyodbc.Error as e:
        print(f"Database error: {e}")
    except requests.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def insert_data(data):
    try:
        # Get the connection (ensure the 'connection()' function is defined elsewhere)
        conn = connection()
        cursor = conn.cursor()

        # SQL query to insert data
        query = """
                INSERT INTO nlog0001 (car_no, type, add_date, add_time)
                VALUES (?, ?, ?, ?)
                """
        
        # Extract data from the dictionary
        values = (
            data['car_no'],  # license_number
            data['type'],    # channel
            data['add_date'],  # date
            data['add_time']   # time
        )
        
        # Execute the query with the data
        cursor.execute(query, values)
        
        # Commit the transaction
        conn.commit()

        #print(f"Successfully inserted {data['car_no']} into carin_List.")

    except pyodbc.Error as e:
        # Handle any database errors
        print(f"Database error occurred: {e}")
    
    finally:
        # Ensure the cursor and connection are closed
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def door_open(data, url_gate, channel):
    global door_status  # Declare door_status as global
    try:
        open_signal = {'open': '1'}  # Signal to open the gate

        # Check if the door is already open
        if door_status[channel]:
            print(f"{channel} gate is already open. Skipping further actions.")
            return False

        # Set the door flag to True to indicate it's open
        door_status[channel] = True
        print(f"Sending open signal to {channel} gate.")

        # Send the open signal request
        response = requests.post(url_gate, json=open_signal, headers={'Content-Type': 'application/json'})
        response.raise_for_status()

        if response.status_code == 200:
            print(f"{channel} gate opened successfully.")

            # Wait for 10 seconds and then reset the door status to allow future openings
            for remaining in range(10, 0, -1):
                print(f"{channel}: Waiting... {remaining} seconds remaining", end="\r")
                time.sleep(1)

            print(f"\n{channel}: Ready for the next signal.")

            # Reset the door status after waiting
            door_status[channel] = False
            return True
        else:
            print(f"Failed to open {channel} gate.")
            door_status[channel] = False
            return False

    except requests.RequestException as e:
        print(f"Error in sending signal to {data['type']}: {e}")
        door_status[channel] = False
        return False

def connection():
    connection_str= (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "Server=192.168.2.102,1435;"  # Server and port
        "Database=PKM;"                # Database name
        "UID=sa;"                      # User ID (SQL Server Authentication)
        "PWD=$j53272162;"              # Password
        "MultipleActiveResultSets=True;" 
        #"TrustServerCertificate=True;"
        "Trusted_Connection=no;" 
    )

    return pyodbc.connect(connection_str)

# Convert OpenCV image to Tkinter-compatible format
def convert_to_tk(frame, target_size=(640, 480)):
    try:
        frame = cv2.resize(frame, target_size)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_ctk = CTkImage(light_image=img, size=target_size)  # Use CTkImage
        return img_ctk
    
    except Exception as e:
        logging.error(f"Error in convert_to_tk: {e}")
        return None

'''
def update_frame():
    global video_streams, cctv_labels, root, update_frame_id
    try:
        def update_stream(vs, label):
            try:
                if vs.running:
                    frame = vs.read()
                    if frame is not None:
                        frame = vs.detect_objects(frame)
                        frame_tk = convert_to_tk(frame)
                        label.configure(image=frame_tk, text="")
                        label.image = frame_tk
            except Exception as e:
                logging.error(f"Error in update_stream: {e}")

        # Launch threads to handle each video stream
        threads = []
        for idx, vs in enumerate(video_streams):
            t = threading.Thread(target=update_stream, args=(vs, vs.label))
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        update_frame_id = root.after(10, update_frame)  # Adjust timing to 30ms to reduce workload
    except Exception as e:
        logging.error(f"Error in update_frame: {e}")

'''
# Update frames in the Tkinter window
def update_frame():
    global video_streams, cctv_labels, root, update_frame_id
    try:
        for idx, vs in enumerate(video_streams):
            if not vs.running:
                continue
            frame = vs.read()
            if frame is None:
                continue
            frame = vs.detect_objects(frame)
            frame_tk = convert_to_tk(frame)
            vs.label.configure(image=frame_tk, text="")
            vs.label.image = frame_tk

        update_frame_id = root.after(10, update_frame)
    except Exception as e:
        logging.error(f"Error in update_frame: {e}")

def restart_program():
    # Restart the program using execv
    logging.debug("booting program...")
    python = sys.executable
    os.execl(python, python, *sys.argv)
    
# Start the main program
def start_program():
    global rtsp_urls, license_address, video_streams, root, update_frame_id, cctv_labels
    global plate_frame1, plate_frame2, plate_frame3
    global latest_detections1, latest_detections2, latest_detections3
    
    def on_escape(event):
        # This function will be called when the Esc key is pressed
        root.destroy()  # Close the window

    root = ctk.CTk()
    root.title("攝影機")
    width  = 1920
    height = 1080
    root.resizable(False, False)
    root.iconbitmap(icon_path)
    screen_width= root.winfo_screenwidth() 
    screen_height= root.winfo_screenheight()

    # calculate position x and y coordinates
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))
    root.bind('<Escape>', on_escape)

    # Set default appearance
    ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

    # Create frames
    sidebar_frame = ctk.CTkFrame(root, width=550)
    sidebar_frame.pack(side=ctk.LEFT, fill=ctk.Y)

    cctv_frame = ctk.CTkFrame(root)
    cctv_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)

    # Sidebar (Recent Plates)
    ctk.CTkLabel(sidebar_frame, text="最近車", font=("SimSun", 20)).pack(padx=20,anchor="w")

    # Create ScrollableFrames for each CCTV feed
    plate_frame1 = ctk.CTkScrollableFrame(sidebar_frame, width=520, height=250)
    plate_frame1.pack(padx=20, pady=5, fill=ctk.BOTH)
    ctk.CTkLabel(sidebar_frame, text="入口攝影機", font=("SimSun", 18)).pack(padx=20,anchor="w")

    plate_frame2 = ctk.CTkScrollableFrame(sidebar_frame, width=520, height=250)
    plate_frame2.pack(padx=20, pady=5, fill=ctk.BOTH)
    ctk.CTkLabel(sidebar_frame, text="出口1攝影機", font=("SimSun", 18)).pack(padx=20,anchor="w")

    plate_frame3 = ctk.CTkScrollableFrame(sidebar_frame, width=520, height=250)
    plate_frame3.pack(padx=20, pady=5, fill=ctk.BOTH)
    ctk.CTkLabel(sidebar_frame, text="出口2攝影機", font=("SimSun", 18)).pack(padx=20,anchor="w")

    # Settings Button
    def open_settings():
        restart_program()

    settings_button = ctk.CTkButton(sidebar_frame, text="設定", command=open_settings, font=("SimSun", 16))
    settings_button.pack(pady=20)

    # CCTV Feed Section
    ctk.CTkLabel(cctv_frame, text="攝影機", font=("SimSun", 20)).grid(row=0, column=0, columnspan=2, sticky="w")

    # Initialize VideoCapture instances for each RTSP URL
    video_streams.clear()
    cctv_labels.clear()

    if len(rtsp_urls) >= 1:
        label1 = ctk.CTkLabel(cctv_frame, text="入口", width=640, height=480)
        label1.grid(row=1, column=0, padx=10, pady=5)
        label1.configure(anchor='center')
        cctv_labels.append(label1)
        vs1 = VideoCapture(rtsp_urls[0], label1, text_detector, plate_detector, plate_frame1, latest_detections1, ch="1", url_gate=url_entry, license_address=license_address)
        video_streams.append(vs1)

    if len(rtsp_urls) >= 2:
        label2 = ctk.CTkLabel(cctv_frame, text="出口2", width=640, height=480)
        label2.grid(row=1, column=1, padx=10, pady=5)
        label2.configure(anchor='center')
        cctv_labels.append(label2)
        vs2 = VideoCapture(rtsp_urls[1], label2, text_detector, plate_detector, plate_frame2, latest_detections2, ch="2", url_gate=url_exit1, license_address=license_address)
        video_streams.append(vs2)

    if len(rtsp_urls) >= 3:
        label3 = ctk.CTkLabel(cctv_frame, text="出口2", width=640, height=480)
        label3.grid(row=2, column=0, padx=10, pady=5)
        label3.configure(anchor='center')
        cctv_labels.append(label3)
        vs3 = VideoCapture(rtsp_urls[2], label3, text_detector, plate_detector, plate_frame3, latest_detections3, ch="3", url_gate=url_exit2, license_address=license_address)
        video_streams.append(vs3)

    update_frame()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

# Handle window closing
def on_closing():
    global root, video_streams
    for vs in video_streams:
        vs.stop()
    root.destroy()

# Input RTSP URLs
def input_rtsp_urls():
    global rtsp_urls, license_address, door_entry, door_exit1, door_exit2
    global url_entry, url_exit1, url_exit2, license_address

    def upload_rtsp_file():
        # Auto find RTSP file in the specified directory
        try:
            file_path = "./_internal/static/Setting.txt"
            with open(file_path, "r") as file:
                lines = file.readlines()

                # Ensure the file has at least 7 lines (3 RTSP URLs and door controls)
                if len(lines) >= 6:
                    rtsp1 = int(lines[0].strip()) if lines[0].strip().isdigit() else lines[0].strip()
                    rtsp2 = int(lines[1].strip()) if lines[1].strip().isdigit() else lines[1].strip()
                    rtsp3 = int(lines[2].strip()) if lines[2].strip().isdigit() else lines[2].strip()
                    rtsp_urls[:] = [rtsp1, rtsp2, rtsp3]
                    global license_address, door_entry, door_exit1, door_exit2
                    global url_entry, url_exit1, url_exit2, license_address
                    door_entry = lines[3].strip()
                    door_exit1 = lines[4].strip()
                    door_exit2 = lines[5].strip()

                    url_entry = door_entry
                    url_exit1 = door_exit1
                    url_exit2 = door_exit2
                    license_address = license_address

                    #print("RTSP URLs and API address successfully loaded from file.")
                    #for i, line in enumerate(lines):
                    #    print(f"Line {i}: {line}")
                    input_window.destroy()  # Close the window before starting the program
                    start_program()  # Start the main program
                else:
                    print("Error: The file must contain at least 7 lines (3 RTSP URLs, 1 API address, and 3 door settings).")
        except Exception as e:
            print(f"Error reading the file: {e}")

    def submit_rtsp():
        rtsp1 = int(entry_rtsp1.get().strip()) if entry_rtsp1.get().strip().isdigit() else entry_rtsp1.get().strip()
        rtsp2 = int(entry_rtsp2.get().strip()) if entry_rtsp2.get().strip().isdigit() else entry_rtsp2.get().strip()
        rtsp3 = int(entry_rtsp3.get().strip()) if entry_rtsp3.get().strip().isdigit() else entry_rtsp3.get().strip()
        rtsp_urls[:] = [rtsp1, rtsp2, rtsp3]  # Replace existing URLs with the new ones

        # Remove empty strings from rtsp_urls
        rtsp_urls[:] = [url for url in rtsp_urls if url]

        global license_address, door_entry, door_exit1, door_exit2
        global url_entry, url_exit1, url_exit2, license_address
        door_entry = entry_entry.get().strip()
        door_exit1 = entry_exit1.get().strip()
        door_exit2 = entry_exit2.get().strip()

        url_entry = door_entry
        url_exit1 = door_exit1
        url_exit2 = door_exit2
        license_address = license_address

        input_window.destroy()  # Close the window
        start_program()  # Restart the main program with new settings

    input_window = ctk.CTk()
    input_window.title("設定攝影機")
    input_window.geometry("320x480")
    input_window.resizable(False, False)
    input_window.iconbitmap(icon_path)

    # Set default appearance
    ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

    ctk.CTkLabel(input_window, text="RTSP URL 入口:").grid(row=0, column=0, padx=10, sticky="w")
    entry_rtsp1 = ctk.CTkEntry(input_window, width=300)
    entry_rtsp1.grid(row=1, column=0, padx=10, pady=1)

    ctk.CTkLabel(input_window, text="RTSP URL 出口1:").grid(row=2, column=0, padx=10, sticky="w")
    entry_rtsp2 = ctk.CTkEntry(input_window, width=300)
    entry_rtsp2.grid(row=3, column=0, padx=10, pady=1)

    ctk.CTkLabel(input_window, text="RTSP URL 出口2:").grid(row=4, column=0, padx=10, sticky="w")
    entry_rtsp3 = ctk.CTkEntry(input_window, width=300)
    entry_rtsp3.grid(row=5, column=0, padx=10, pady=1)

    ctk.CTkLabel(input_window, text="入口控制 API:").grid(row=8, column=0, padx=10, sticky="w")
    entry_entry = ctk.CTkEntry(input_window, width=300)
    entry_entry.grid(row=9, column=0, padx=10, pady=1)

    ctk.CTkLabel(input_window, text="出口1控制 API:").grid(row=10, column=0, padx=10, sticky="w")
    entry_exit1 = ctk.CTkEntry(input_window, width=300)
    entry_exit1.grid(row=11, column=0, padx=10, pady=1)

    ctk.CTkLabel(input_window, text="出口2控制 API:").grid(row=12, column=0, padx=10, sticky="w")
    entry_exit2 = ctk.CTkEntry(input_window, width=300)
    entry_exit2.grid(row=13, column=0, padx=10, pady=1)

    ctk.CTkButton(input_window, text="確認", command=submit_rtsp, width=80).grid(row=14, column=0, padx=20, pady=20, sticky="e")

    # Option to load RTSP URLs from file
    ctk.CTkButton(input_window, text="掃描", command=upload_rtsp_file, width=80).grid(row=14, column=0, padx=20, pady=20)

    input_window.mainloop()

# Entry point of the application
if __name__ == "__main__":
    input_rtsp_urls()
