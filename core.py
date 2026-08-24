# coding:utf-8
import os
import sys
import warnings
import cv2
import json
import re
import numpy as np
import torch
from datetime import datetime, timedelta
from PIL import Image
import astropy.constants as const
from scipy import optimize
from ultralytics import YOLO

warnings.filterwarnings("ignore")


sys.path.append("MobileSAM-master")
sys.path.append("yolo11")
from mobile_encoder.setup_mobile_sam import setup_model
from segment_anything import SamPredictor

# ==========================================
# Part 1
# ==========================================
def add_mask(image, mask, alpha=0.5):
    
    int_mask = (mask.astype(np.uint8) * 255)
    mask_3d = np.dstack((int_mask, int_mask, int_mask))
    mask_area = mask > 0
    res = image.copy()
    blended = cv2.addWeighted(image, 1 - alpha, mask_3d, alpha, 0)
    res[mask_area] = blended[mask_area]
    return res

def show_box(image, box, color=(0, 255, 0), thickness=2):
    
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
    return image

# ==========================================
# Part 2
# ==========================================
def process_events(result, time_threshold):
    if len(result) == 0:
        return []
    
    events_list = []
    for r in result:
        events_list.append({
            'd_datetime': r[0],
            'start_time': r[1].strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': r[2].strftime("%Y-%m-%d %H:%M:%S"),
            'event_type': r[3],
            'start_freq': float(r[4]),
            'end_freq': float(r[5]),
            'B': float(r[6]),
            'R_B': float(r[7]),
            'yita': float(r[8]),
            'event_level': int(1),
            'filename': r[9]
        })

    events_list.sort(key=lambda x: x['start_time'])
    merged_events = []
    current_event = events_list[0]
    
    for event in events_list[1:]:
        time_diff = datetime.strptime(event['start_time'], "%Y-%m-%d %H:%M:%S") - datetime.strptime(current_event['end_time'], "%Y-%m-%d %H:%M:%S")
        if time_diff.total_seconds() <= time_threshold:
            current_event['end_time'] = event['end_time']
        else:
            merged_events.append(current_event)
            current_event = event

    merged_events.append(current_event)

    filtered_events = [merged_events[0]]
    for event in merged_events[1:]:
        if event['start_time'] != filtered_events[-1]['start_time'] or event['end_time'] != filtered_events[-1]['end_time']:
            filtered_events.append(event)
    return filtered_events

def save_event_to_json(event, output_file_path):
    with open(output_file_path, "w", encoding='utf-8') as file:
        json.dump(event, file, default=str, ensure_ascii=False, indent=4)

def data2jsonf(data, fname):
    with open(fname, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

def copy_and_rename_files(result_events, source_image_path):
    target_root_dir = "SpaceWeather/ybfx/product/WARN/SOLAR/DS/CBSM"
    for event in result_events:
        start_time_datetime = datetime.strptime(event['start_time'], "%Y-%m-%d %H:%M:%S")
        year = start_time_datetime.strftime("%Y")
        date_str = start_time_datetime.strftime("%Y%m%d")
        
        start_time_str = event['start_time'].replace(" ", "_").replace(":", "")
        end_time_str = event['end_time'].replace(" ", "_").replace(":", "")
        new_filename = f"Z_NAFP_C_WARN-_{start_time_str}_P_SOLA_00_RADI_L0_{end_time_str}_000STP_CBSM-_ZXGC_NCSW_V1A.PNG"

        target_year_dir = os.path.join(target_root_dir, year, date_str)
        target_filename = os.path.join(target_year_dir, new_filename)
        os.makedirs(target_year_dir, exist_ok=True)

       
        img = Image.open(source_image_path)
        resized_img = img.resize((1920, 1440), resample=Image.Resampling.BICUBIC)
        resized_img.save(target_filename, format="PNG", dpi=(240, 240))

def Newkirkmodel(freqx, Nharm=1, Nfold=1):
    lgN0 = np.log10(4.2) + 4.0
    freqp = freqx / Nharm 
    kk = (const.e.value * const.e.value) / (4.0 * np.pi * np.pi * const.eps0.value * const.m_e.value)
    e_density = freqp * freqp / kk
    RR = 4.32 / (np.log10(e_density) + 6.0 - np.log10(Nfold) - lgN0)
    return RR

def typeIIradioSource(t, f, dens_model='Newkirk', Nharmset=2, Nfoldset=1, dates='2022-08-29'):
    tt = np.ravel(np.double(t))
    freqx = np.ravel(np.double(f))
    typeIIndot = len(tt)
    
    if typeIIndot < 1:
        return
        
    if dens_model == 'Newkirk':
        RR = Newkirkmodel(freqx, Nharm=Nharmset, Nfold=Nfoldset)
        
    Rsunkm = const.R_sun.value / 1000.0  # km
    if typeIIndot == 1:
        return RR
    if typeIIndot == 2:
        vshock = Rsunkm * (RR[-1] - RR[0]) / (tt[-1] - tt[0])
        return vshock, RR

    def fx_1(x, A, B): return A*x + B
    def fx_2(x, A, B, C): return A * x**2 + B*x + C

    A1, B1 = optimize.curve_fit(fx_1, tt, RR)[0]
    v1 = A1 * Rsunkm
    if typeIIndot == 3:
        return v1
    
    A2, B2, C2 = optimize.curve_fit(fx_2, tt, RR)[0]
    aa2 = A2 * Rsunkm * 2
    v2 = B2 * Rsunkm + aa2 * tt[0]
    return v1, v2, aa2

# ==========================================
# Part 3
# ==========================================
def process_physics_from_yolo(yolo_boxes, class_names, image_path):
   
    if len(yolo_boxes) == 0:
        print("no target")
        return

    
    left_array = yolo_boxes[:, 0]
    top_array = yolo_boxes[:, 1]
    right_array = yolo_boxes[:, 2]
    bottom_array = yolo_boxes[:, 3]
    
   
    top_array[top_array < 50] = 50
    bottom_array[bottom_array > 560] = 560
    
    type_array = class_names
    img_name = os.path.basename(image_path)
    name_array = np.array([img_name] * len(yolo_boxes))
    
   
    kt = 15 * 60 / 900
    kf = (600 - 90) / 510
    
    t_range = (right_array - left_array) * kt
    f_range = (bottom_array - top_array) * kf
    t_start = (left_array - 80) * kt 
    
    start_delta = np.array([timedelta(seconds=int(interval)) for interval in t_start])
    time_delta = np.array([timedelta(seconds=int(interval)) for interval in t_range])
    
    pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})'
    match = re.search(pattern, img_name)
    if not match:
        print(f"fail")
        return
        
    base_date_time = datetime.strptime(match.group(0), "%Y-%m-%dT%H-%M-%S")
    date_time_array = [base_date_time] * len(yolo_boxes)
    
    start_time = date_time_array + start_delta
    numberId = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in start_time]
    end_time = start_time + time_delta 
    
    f_end = (600 - (bottom_array - 50)) * kf 
    f_start = (600 - (top_array - 50)) * kf
    
    B = f_end - f_start
    R_B = 2 * B / (f_end + f_start)
    yita = -f_range / t_range
    
   
    result = np.vstack((numberId, start_time, end_time, type_array, f_start, f_end, B, R_B, yita, name_array)).T
    
  
    unique_events = process_events(result, time_threshold=2)
    result_events = np.array(unique_events)
    
  
    copy_and_rename_files(result_events, image_path)
    
   
    for event in result_events:
        start_time_dt = datetime.strptime(event['start_time'], "%Y-%m-%d %H:%M:%S")
        year = start_time_dt.strftime("%Y")
        date_str = start_time_dt.strftime("%Y%m%d")
        
       
        if event['event_type'].lower() == 'typeii':
            typeIIoutdir = os.path.join('SpaceWeather/ybfx/product/WARN/SOLAR/RB/CBSM', year)
            os.makedirs(typeIIoutdir, exist_ok=True)
            
            data = event.copy()
            dates = event['start_time'][0:10]
            date0 = datetime.fromisoformat(dates + 'T00:00:00')

            t = np.array([
                (datetime.fromisoformat(data['start_time']) - date0).total_seconds(), 
                (datetime.fromisoformat(data['end_time']) - date0).total_seconds()
            ])
            f = np.array([data['start_freq'], data['end_freq']])        

            vII, RR = typeIIradioSource(t, f, dens_model='Newkirk', Nharmset=2, Nfoldset=6, dates=dates)
            data['velocity'] = vII
            data['R_start'] = float(RR[0])
            data['R_end'] = float(RR[1])
            
            fname = os.path.join(typeIIoutdir, f"typeII_{dates}_Velocity.json")
            data2jsonf(data, fname)

       
        start_time_str = event['start_time'].replace(" ", "_").replace(":", "").replace("-", "")
        end_time_str = event['end_time'].replace(" ", "_").replace(":", "").replace("-", "")
        
        output_dir = os.path.join('SpaceWeather/ybfx/product/WARN/SOLAR/RB/CBSM', year, date_str)
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"Z_NAFP_C_WARN-_{start_time_str}_P_SOLA_LJ_RADI_L0_{end_time_str}_00015M_CBSM-_ZXGC_NCSW_V1A.json")

        save_event_to_json({
            "Attributes": {
                "Global": {
                    "QualityFlag": 0,
                    "TimeRes": "1 Day",
                    "Create_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Level": "L4",
                    "Project": "ZXGC",
                    "Construction": "NCSW",
                    "Format": "JSON"
                }
            },
            "Data": [event]
        }, output_file_path)

    print(f" {len(result_events)} events")

# ==========================================
# Part 4
# ==========================================
def main():
    
    yolo_model_path = "./YOLOv10_MobileSAM/weights/best.pt"
    source = "img/SynRadiospec_CSO_2022-11-11T01-27-06UT11_Part.jpg"
    
    if not os.path.exists(source):
        raise FileNotFoundError(f"no iamge: {source}")

    print("start...")
    model = YOLO(yolo_model_path)
    results = model(source)
    res = results[0]
    
   
    box_locations = res.boxes.xyxy.cpu().numpy()
    class_indices = res.boxes.cls.cpu().numpy().astype(int)
    class_names = np.array([model.names[idx] for idx in class_indices])

   
   
    process_physics_from_yolo(box_locations, class_names, source)

    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_checkpoint = "MobileSAM-master/weights/mobile_sam.pt"
    
    mobile_sam = setup_model()
    checkpoint = torch.load(sam_checkpoint, map_location=torch.device(device))
    mobile_sam.load_state_dict(checkpoint, strict=True)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    
    predictor = SamPredictor(mobile_sam)
    
    image = cv2.imread(source)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    mask_only_image = np.zeros_like(image)
    image_result = image.copy()

    for box in box_locations:
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )
        current_mask = masks[0]
        image_result = add_mask(image_result, current_mask, alpha=0.4)
        mask_only_image[current_mask] = image[current_mask]
        image_result = show_box(image_result, box)

    cv2.imwrite("res.jpg", image_result)
    cv2.imwrite("mask_only.jpg", mask_only_image)
    print("done")

if __name__ == "__main__":
    main()
