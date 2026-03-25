# ML-Sharp Browser

This is a heavily-vibe-coded proof of concept to bring [Apple's ML-Sharp model](https://github.com/apple/ml-sharp) for converting images to 3D Gaussian splats into a browser. The model was exported to ONNX then uploaded to HuggingFace ([link](https://huggingface.co/mxtx0123/ml-sharp-onnx)). It's approximately 2.6gb.

A React app is included which allows a user to convert their own images. It can be viewed online at [here](https://miketahani.com/ml-sharp-browser/). No images are uploaded anywhere and inference happens 100% clientside.