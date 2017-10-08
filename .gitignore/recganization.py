import cv2, time, pandas
from datetime import datetime

df=pandas.DataFrame(columns=["start","end"])

first_frame=None
status_list=[None,None]
times=[]

video=cv2.VideoCapture(0)
while True:
    check, frame=video.read()
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
       first_frame=gray
       continue

    delta_frame=cv2.absdiff(first_frame,gray)


    threshold=cv2.threshold(delta_frame,120, 255, cv2.THRESH_BINARY)[1]
    threshold=cv2.dilate(threshold, None, iterations=2)

    (_,cnts,_)=cv2.findContours(threshold.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour)<10000:
           continue
        status=1
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    cv2.imshow("frame",frame)
    cv2.imshow("capture",gray)
    cv2.imshow("delta",delta_frame)
    cv2.imshow("thres",threshold)
    key=cv2.waitKey(1)
    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break
print(status_list)
print(time)

for i in range(0,len(times),2):
    df=df.append({"start":times[i],"end":times[i+1]},ignore_index=True)

df.to_csv("times.csv")

video.release()
cv2.destroyAllWindows()
