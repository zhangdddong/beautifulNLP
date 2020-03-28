# 使用Wiki语料训练GloVe数据集

### 下载Wiki数据集，下载地址:

>https://dumps.wikimedia.org/zhwiki/
>
>本文使用20191201版本语料，压缩包大小1.9G，解压后7.5G

### 使用GloVe官方代码。下载地址：

> https://github.com/stanfordnlp/GloVe

### 抽取Wiki中文数据

> 使用WikiExtractor，GitHub下载地址：https://github.com/attardi/wikiextractor
>
> 参考：https://www.cnblogs.com/anno-ymy/p/10510791.html
>
> 将下载的Wiki数据.xml.bz2文件（压缩文件）放在当前目录下，在命令行运行以下代码：
>
> ```shell
> python WikiExtractor.py -b 1024M -o ../extracted zhwiki-latest-pages-articles.xml.bz2
> ```
>
> 其中：
>
> ../extracted：是抽取后的文件存储路径
>
> zhwiki-latest-pages-articles.xml.bz2：是我们下载的wiki文件
>
> 抽取后的文件大约1.4G

### 中文繁体转换成简体

> 参参考博客：https://blog.csdn.net/sinat_29957455/article/details/81290356
>
> 百度云下载地址：链接：https://pan.baidu.com/s/10yI1lPRKNOYJ2aSbl4YegA 密码：2kv9
>
> 解压之后在opencc中的share-->opencc中有需要的json文件就是opencc的配置文件，用来制定语言类型的转换。
>
> ![img](https://img-blog.csdn.net/20180730191856182?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI5OTU3NDU1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
>
> 使用
>
> ```shell
> opencc -i 需要转换的文件路径 -o 转换后的文件路径 -c 配置文件路径
> ```
>
> 在linux下安装opencc，可以参考：https://segmentfault.com/a/1190000010122544

### 去掉标点停用词

> 去掉后的文件大小**882**M

### 准备语料

参考链接：https://blog.csdn.net/weixin_37947156/article/details/83145778

> 输入格式为 每行一句话  每句话中的tokens用空格隔开（字向量或者词向量）
>
> ```
> 我 今天 很好
> 是 的
> ```
>
> 并将语料文件放入GloVe的主文件夹下

### 修改GloVe中的bash

> [打开demo.sh](http://xn--demo-9z2h93o.sh/)，修改相应的内容
>
> 因为demo默认是下载网上的语料来训练的，因此如果要训练自己的语料，需要注释掉
> ![在这里插入图片描述](https://img-blog.csdn.net/20181018141101797?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk0NzE1Ng==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
>
> 修改参数设置，将CORPUS设置成语料的名字

### 执行Bash

> 执行bash文件
>
> 进入到主文件夹下
>
> ```shell
> make
> ```
>
> ```shell
> bash demo.sh
> nohup bash demo.sh >output.txt 2>&1 &	# 语料大的时候后台运行
> ```
>
> 坐等训练，最后会得到vectors.txt 以及其他的相应的文件。如果要用gensim的word2ve load进来，那么需要在vectors.txt的第一行加上vacob_size vector_size，第一个数指明一共有多少个向量，第二个数指明每个向量有多少维。