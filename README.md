## 技术选型
### 编程包
- python 3.7
- pytorch 1.10
- pytorch_lightning 1.5
- torchtext   0.11

### 模型选择
textcnn，模型相关介绍可以参见：[【深度学习】textCNN论文与原理](https://piqiandong.blog.csdn.net/article/details/110099713?spm=1001.2014.3001.5502)

## 数据获取
测试的数据来自于开源项目：[bigboNed3/chinese_text_cnn](https://github.com/bigboNed3/chinese_text_cnn)
### 数据背景
从数据来看，应该是对车的正向和负向评论。可以理解为一种情感分析任务。总体来说，数据样本长度不大，选择textcnn模型基本能够完成该文本分类任务。
### 程序介绍
该源码介绍可参见我的博客：[【Pytorch Lightning】基于Pytorch Lighting和TextCNN的中文文本情感分析模型实现](https://blog.csdn.net/meiqi0538/article/details/123466819?spm=1001.2014.3001.5501)

### 模型训练情况
由于使用了tensorboard相关日志记录器，我们可以使用如下命令启动tensorboard服务器：
```shell script
tensorboard --logdir ./TextCNN_Classification_PL
```
其中./texcnn_pl就是训练代码中设置的目录。

![image-20220313212529419](https://img-blog.csdnimg.cn/212d6c7b42c7497c9b2f7a3ecc18bb13.png)

tensorboard各参数情况：

![image-20220313212752873](https://img-blog.csdnimg.cn/d68545f2df704b04b61c619779a2fce3.png)


### 测试集效果效果
```text
precision    recall  f1-score   support

           0       0.94      0.94      0.94      3144
           1       0.94      0.94      0.94      3156

    accuracy                           0.94      6300
   macro avg       0.94      0.94      0.94      6300
weighted avg       0.94      0.94      0.94      6300

--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'f1_score': 0.94349205493927, 'val_loss': 0.2568584084510803}
```
## 联系我

1. 我的github：[https://github.com/Htring](https://github.com/Htring)
2. 我的csdn：[科皮子菊](https://piqiandong.blog.csdn.net/)
3. 我订阅号：AIAS编程有道
   ![AIAS编程有道](https://s2.loli.net/2022/05/05/DS37LjhBQz2xyUJ.png)
4. 知乎：[皮乾东](https://www.zhihu.com/people/piqiandong)