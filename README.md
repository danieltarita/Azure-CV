# azure-CV
Tests using Azure Custom Vision service

## Project info
requirements.txt for environment setup.

script usage:

python predict.py < Image Path >

Outputs the the input image with the predictions identified with black boxes to ./predictions folder

### Problem Description

I trained a model of Object Detection using Azure Custom Vision and the predictions done using 'quick test' in the portal are different from the ones obtained offline (with the sample code provided).

### Project info
**Domains:** 'General (compact)'

**Export Capabilities:** 'Basic platforms (Tensorflow, CoreML, ONNX, ...)'

Exported for 'TensorFlow'

### Original Image

[![enter image description here][1]][1]
### Quick Test (CV Portal)
I used a dataset of 50 images of cats only. 
 In the portal I got the following result:

[![enter image description here][2]][2]

### Offline Prediction
[![enter image description here][3]][3]

(For box visualization I added the following code to the main method in 'prediction.py' sample code provided in the export .zip file)
```python
print("Predictions[0]", predictions[0]['boundingBox']['left'])
    image_cv = cv2.imread(image_filename)
    HEIGHT, WIDTH, channels = image_cv.shape
    for i in range(len(predictions)):
        print(predictions[i], "\n")
        # TESTING#
        x1, y1 = predictions[i]['boundingBox']['left'], predictions[i]['boundingBox']['top']
        x2, y2 = x1 + predictions[i]['boundingBox']['width'], y1 + predictions[i]['boundingBox']['height']
        x1, y1, x2, y2 = round(x1 * WIDTH), round(y1 * HEIGHT), round(x2 * WIDTH), round(y2 * HEIGHT)

        image_cv = cv2.rectangle(image_cv, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=1)

        # show_names and show_percentage:
        label = "%s : %.3f" % (predictions[i]['tagName'], predictions[i]['probability'])

        b = np.array([x1, y1, x2, y2]).astype(int)
        cv2.putText(image_cv, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (100, 0, 0), 3)
        cv2.putText(image_cv, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imwrite("Predicted-img.jpg", image_cv)
```

  [1]: https://i.stack.imgur.com/2Rk0t.jpg
  [2]: https://i.stack.imgur.com/nHwqT.png
  [3]: https://i.stack.imgur.com/126BX.jpg
