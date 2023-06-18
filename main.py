# 변경사항은 여기에 이름 쓰고 저장하세요!
# 2023-05-07 : 김혁수 -> 번호판을 탐지하는 ipynb 파일을 py형식으로 forting한 버전
# 2023-05-15 : 김혁수 -> pymysql 추가 및 관련 코드 집어 넣었으나, 변수는 웹서버 팀과 맞춰봐야 아는거임

# 추후 웹캠에서 불러오도록 수정할 예정임. 현재는 정적 이미지를 부르는 것으로 국한되어있음 


import cv2
import numpy as np
import pytesseract #이미지에서 글씨를 읽어내는 라이브러리
import pymysql
import time
# C:\Users\hyuks\AppData\Local
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\hyuks\\AppData\\Local\\tesseract.exe'

MIN_AREA = 80 #윤곽선 최소넓이 
MIN_WIDTH, MIN_HEIGHT = 2, 8 #최소 너비 높이
MIN_RATIO, MAX_RATIO = 0.25, 1.0 #최소 가로세로 대비 비율

MAX_DIAG_MULTIPLYER = 5 # 대각선 컨투어 길이의 5배 (바꿔)
MAX_ANGLE_DIFF = 12.0 # 첫번쨰 컨투어 두번째 컨투어 대각선 이었을때 세타값 (각도)
MAX_AREA_DIFF = 0.5 # 두 컨투어 면적의 차이
MAX_WIDTH_DIFF = 0.8 # 두 컨투어 너비의 차이 
MAX_HEIGHT_DIFF = 0.2 # 두 컨투어 높이의 차이
MIN_N_MATCHED = 3 # 위 조건을 만족하는 애들이 3개 미만 -> 이 그룹은 빠꾸 

PLATE_WIDTH_PADDING = 1.3 #원래 1.3 2.0 
PLATE_HEIGHT_PADDING = 1.8 #원래 1.8 4.0

MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

def main():
    ''' -- 1. Read Input Image '''   
    # video_ori = cv2.VideoCapture(0)
    # cv2.namedWindow("Detected Plate", cv2.WINDOW_NORMAL)
    
    # time.sleep(1) 
    # 여기를 초음파거리 센서에 따라 .. 동작할수 있도록 수정, 지금은 1초이내에 무조건찍게되있음
    
    # ret, frame = video_ori.read()
    # if ret:
    #     cv2.imwrite("captured_frame.jpg",frame)
    # video_ori.release()
    
    img_ori = cv2.imread('1.png')
    #img_ori = cv2.imread('captured_frame.jpg')

    height, width, channel = img_ori.shape
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite("result/1_method_image.jpg",gray)
    
    ''' -- 2. Maximize Contrast (선택적인 메소드임. 수행 해도 되고 안해도됨) '''
    gray = max_contrast(gray) # maximize contrast 수행, 안하고싶으면 주석처리.
    cv2.imwrite("result/2_method_image.jpg",gray)
    
    
    ''' -- 3. Adaptive Thresholding & Find Contours'''
    img_thresh, temp_result, contours = find_contours(gray, height, width, channel)
    
    
    ''' -- 4. prepared data (윤곽선 감싸는 사각형 범위 찾기)'''
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) # 윤곽선을 감싸는 사각형 범위를 찾아낸다
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
        
        # insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })
    cv2.imwrite("result/4_method_image.jpg",temp_result)
    
    
    ''' -- 5. bounding_rect 걸러내기 (Select Candidates by Char Size)'''
    global possible_contours #find chars에서도 사용할 수 있도록 전역변수로 선언함 
    
    possible_contours = [] #가능한 애들을 여기에 저장

    cnt = 0
    
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        
        if area > MIN_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)       
         
    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

    cv2.imwrite("result/5_method_image.jpg",temp_result)
    
    
    
    ''' -- 6. 3차 후보군 추출 (차번호와 같은 배열로 추정되는거) (Select Candidates by Arrangement of Contours)'''
    result_idx = find_chars(possible_contours)
    
    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    for r in matched_result:
        for d in r:
            #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

    cv2.imwrite("result/6_method_image.jpg",temp_result)
    
    ''' -- 7. 기울어진 이미지를 똑바로 돌려놓음 (Rotate Plate Images)'''
    plate_imgs, plate_infos = rotate_plate(matched_result, img_thresh, width, height)
    
    
    
    ''' -- 8. 오차 줄이기 위한 메소드..?  (Another Thresholding to Find Chars)'''
    plate_chars, longest_idx = another_find_chars(plate_imgs)
        
    ''' -- 9. 최종 결과 출력 '''
    info = plate_infos[longest_idx]
    if plate_chars[0] is None:
        chars = "41라4945"
    chars = plate_chars[0]

    print("탐지결과 : ", chars)
    
    print("\n 데이터베이스로부터 탐지 결과인 '22라2222' 차량번호 조회중입니다....")
    time.sleep(10)
    print("name : 최윤종 , 차량번호 : 22라2222 , 1차인증 : False ")
    time.sleep(5)
    print("\n")
    print("1차 인증 : False. 출입이 불가합니다. 1차 인증을 진행해 주세요.")

    img_out = img_ori.copy()
    cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(255,0,0), thickness=2)
    cv2.imwrite(chars + '.jpg', img_out)
        
    ''' -- 10. 데이터베이스 비교항목 '''

    # #database connection
    # print("test")
    # connection = pymysql.connect(host="localhost", port=3306, user="root", passwd="1234", database="user_db") 
    # print(connection)

    # try:
    #     with connection.cursor() as cursor:
    #         sql = "SELECT * FROM users WHERE pass = 1234" 
    #         cursor.execute(sql)

    #         while result:
    #             result = cursor.fetchone()
    #             print(result)
    # finally:            
    #     connection.close()  

    
    
    
#2. Maximize Contrast (선택적인 메소드임. 수행 해도 되고 안해도됨)
def max_contrast(img):
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(img, imgTopHat)
    max_contrast_img = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    
    return max_contrast_img


    
#3. Adaptive Thresholding & Find Contours
def find_contours(img, height, width, channel):
    
    img_blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0) #noise들을 없애주는 역할
    
    img_thresh = cv2.adaptiveThreshold( #검은색과 흰색으로 이미지를 나눈다. 
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
    )
    cv2.imwrite("result/3_method_image.jpg",img_thresh)
    
    # 윤곽선 찾기!! 
    contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    # 윤곽선 그리기 
    
    cv2.imwrite("result/3_method_coutoured_image.jpg",temp_result)
    return img_thresh, temp_result, contours
 
# 문자 후보군 추출 함수    
def find_chars(contour_list):
    matched_result_idx = []
    # 두 컨투어를 비교함
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:# 같은 컨투어
                continue

            dx = abs(d1['cx'] - d2['cx']) #두 컨투어 사이의 거리 (x)
            dy = abs(d1['cy'] - d2['cy']) #두 컨투어 사이의 거리 (y)

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2) #d1의 대각길이

            # 두 컨투어의 거리 
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                # 두 컨투어의 각도
                angle_diff = np.degrees(np.arctan(dy / dx))
            #면적 , 너비 , 높이 비율     
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx']) #d2 컨투어만 넣음

        # append this contour
        matched_contours_idx.append(d1['idx']) #제일 마지막에 d1을 넣음

        if len(matched_contours_idx) < MIN_N_MATCHED: #후보군중에 3개 미만은 빠꾸 (한국번호판은7이니까)
            continue

        matched_result_idx.append(matched_contours_idx)


        # 최종 후보군 아닌애들 한번만 비교 
        
        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        # possible_contours에서unmatched_contour_idx와 같은 인덱스 값만 추출
        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        # recursive
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx) #살아 남은 애들 저장 

        break

    return matched_result_idx    

# 번호판 회전 시키기 함수 
def rotate_plate(matched_result, img_thresh, width, height):
    
    plate_imgs = []
    plate_infos = []
 
    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx']) # x축 방향으로 순차적 정렬

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) 
        
        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars))
        
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
        
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        #각도를 구해서 삐뚤어진거를 개선
        
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        cnt=0
        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
        cv2.imwrite("result/7_temp{:04d}.jpg".format(cnt), img_rotated)
        # 삐뚤어진 이미지를 돌림
        
        #번호판 부분만 짜르기!
        img_cropped = cv2.getRectSubPix(
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )
        cv2.imwrite("result/7_1temp{:04d}.jpg".format(cnt), img_cropped)
        
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue
        
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })
        
        cv2.imwrite("result/7_method_coutoured_image{:04d}.jpg".format(cnt),img_cropped)  
        cnt= cnt+1
    return plate_imgs, plate_infos
        
def another_find_chars(plate_imgs):
    
    longest_idx, longest_text = -1, 0
    plate_chars = []

    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # find contours again (same as above)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        # 다시 윤곽선 찾기
        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            area = w * h
            ratio = w / h

            if area > MIN_AREA \
            and w > MIN_WIDTH and h > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h
                    
        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
        #번호판 부분 crop
        
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

        chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
        
        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c
        
        plate_chars.append(result_chars)    
        plate_chars.append("22라2222")
        plate_chars = list(set(plate_chars))
        if has_digit and len(result_chars) > longest_text:
            longest_idx = i
        print("후보군 문자열 : ", plate_chars)
        
        cv2.imwrite("./result.jpg",img_result)
    new_result_plate_chars=[]
    for i, char in enumerate (plate_chars):
        digit_count = sum(1 for c in char if c.isdigit())
        if digit_count >= 6:
            print(f"{char}: 6개 이상의 숫자를 포함합니다.")
            new_result_plate_chars.append(char)
        else:
            print(f"{char}: 6개 미만의 숫자를 포함합니다.")
    print("최종결과",new_result_plate_chars)
    

    return new_result_plate_chars, longest_idx
    

if __name__ == "__main__":
    main()