# Google Deep Dream

This is my python package to implement google's deep dream. The use of this package is fairly simple.

It takes the image path or url as input and displays the deep dream implementation of the image.

### For example

Input image to model is this

![Input Image](https://github.com/rajeevratan84/ModernComputerVision/raw/main/castara-tobago.jpeg)

And the model will output following

<img alt="Output" src="https://user-images.githubusercontent.com/77535479/210498014-77ff96e0-dae5-4a76-8258-d452e4e33486.png">

## Following are the steps to implement my GoogleDeepDream model

1. Install the package in your system with following commands

```commandline
python GoogleDeepDream/setup.py install
```

2. Implement the model whether the image is stored remote or local

#### We have two flavours of deep dream model available

2.1 Simple deep dream

Remote image using url of it
```commandline
python GoogleDeepDream/src/Model/DeepDream.py --image_url "https://github.com/rajeevratan84/ModernComputerVision/raw/main/castara-tobago.jpeg"
```

or local image using path of it
```commandline
python GoogleDeepDream/src/Model/DeepDream.py --image_path "castara-tobago.jpeg"
```
Results

<img alt="Deep Dream Output" src="https://user-images.githubusercontent.com/77535479/210497853-5dc4db4c-f99a-4270-85d4-3f4912e21ad6.png">

2.2 Improved deep dream

Remote image using url of it
```commandline
python GoogleDeepDream/src/Model/ImprovedDeepDream.py --image_url "https://github.com/rajeevratan84/ModernComputerVision/raw/main/castara-tobago.jpeg"
```

or local image using path of it
```commandline
python GoogleDeepDream/src/Model/ImprovedDeepDream.py --image_path "castara-tobago.jpeg"
```
Results

<img alt="Improved Deep Dream Output" src="https://user-images.githubusercontent.com/77535479/210498014-77ff96e0-dae5-4a76-8258-d452e4e33486.png">

The code is completely open source you can use it however you like.
If you want to see some improvement in this code, or you did some work on it, please let me. I would love to hear it.

If you're having any problem using it. Please feel free to contact me.

Follow me on

<a href ="https://twitter.com/alihassanrevel">
    <img align="left" alt="Revel Twitter" width="25px" src="https://pbs.twimg.com/profile_images/1488548719062654976/u6qfBBkF_400x400.jpg">
</a>
<a href="https://www.linkedin.com/in/ali-hassan-5b9463217/">
    <img alt="Revel's LinkedIn" width="30px"
    src="https://brand.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg">
</a>
<a href="https://github.com/alihassanrevel">
    <img alt="Revel's github" width="25px" src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png">
</a>
<a href="https://www.youtube.com/channel/UCqRlg2jIdAhlUkU8GSZ0ixg">
    <img alt="Revel's Youtube" width="110" src="https://www.gstatic.com/youtube/img/branding/youtubelogo/svg/youtubelogo.svg">
</a>
