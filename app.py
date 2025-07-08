import streamlit as st 
import cv2 
import numpy as np 
import tempfile
import time 

def convert_color(img):
    """Convert BGR image to RGB."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb

st.title('🎥 Video Foreground Detection App')
st.markdown('''
Upload a video to see foreground object detection in action.  
- The left panel shows the **original video** with detected objects.  
- The right panel shows the **foreground mask**.
''')

uploaded_file = st.file_uploader('📁 Upload a video', type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()
    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("❌ Error opening video file.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.caption('🎯 Original with Detection')  # ✅ حركت العنوان هنا
            stframe = st.empty()
        with col2:
            st.caption('🧠 Foreground Mask')  # ✅ حركت العنوان هنا
            stframe2 = st.empty()

        background_subtractor = cv2.createBackgroundSubtractorMOG2()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            fg_mask = background_subtractor.apply(frame)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i in contours:
                if cv2.contourArea(i) > 300:
                    x, y, w, h = cv2.boundingRect(i)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            stframe.image(convert_color(frame), channels='RGB')
            stframe2.image(fg_mask, clamp=True)  # ✅ بدلت use_column_width بـ clamp=True لضمان ظهور مناسب

            time.sleep(0.01)  # ✅ زودت الوقت شوية لتقليل التقطيع

        cap.release()
        st.success("✅ Video processing completed.")
