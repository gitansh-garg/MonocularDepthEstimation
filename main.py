import cv2
import torch
import numpy as np

model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)
width = int(cap.get(3))
height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height), isColor=False)  # Set isColor to False

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Normalize the depth values for visualization (adjust as needed)
        normalized_depth = (prediction - prediction.min()) / (prediction.max() - prediction.min())
        depth_colormap = cv2.applyColorMap(np.uint8(255 * normalized_depth), cv2.COLORMAP_JET)

        # Display the input frame
        cv2.imshow('Input Feed', frame)

        # Display the depth map
        cv2.imshow('Depth Map', depth_colormap)

        # Write the depth map to the output video
        out.write(depth_colormap)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
