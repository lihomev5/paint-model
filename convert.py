from flask import Flask, request, jsonify
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

portrait_matting = pipeline(Tasks.portrait_matting,model='damo/cv_unet_image-matting')

app = Flask(__name__)


@app.route("/convert", methods=['POST'])
def convert():

    data = request.get_json()

    result = portrait_matting(data['srcFile'])
    cv2.imwrite(data['dstFile'], result[OutputKeys.OUTPUT_IMG])

    return jsonify({
        "errcode": "0"
    })

if __name__ == '__main__':
    app.run(debug=True)
