# IP-Adapter-Plugin
An IP-Adapter integration plugin to be used with the StableDiffusionAPI

# Installation
## Packages
`pip install -r requirements.txt`

## Face Analysis
In the POST request, we specify which face analysis model we want to use.

By default, we use `buffalo_s`.  These are downloaded when called, but can be manually downloaded [here](https://github.com/deepinsight/insightface/releases/)

## IP Binaries
1. `cd IP-Adapter-Plugin`
2. `git lfs install`
3. `git clone https://huggingface.co/h94/IP-Adapter-FaceID lib/ip-adapter-faceid`