# LogisticRegression
SwiftでLogistic Regression（ロジスティック回帰）をするサンプルプログラム  

![Logistic](https://github.com/koooootake/LogisticRegression/blob/master/ScreenShot/logistic.gif)    

ロジスティック回帰とは  
https://github.com/koooootake/LogisticRegression/blob/master/ScreenShot/%E3%83%AD%E3%82%B8%E3%82%B9%E3%83%86%E3%82%A3%E3%83%83%E3%82%AF%E5%9B%9E%E5%B8%B0.pdf  

## **Usage**
### **Set Data**
![Logistic](https://github.com/koooootake/LogisticRegression/blob/master/ScreenShot/Data1.PNG)
![Logistic](https://github.com/koooootake/LogisticRegression/blob/master/ScreenShot/Data2.PNG)   
Dataボタンを押すことで、サンプルデータをランダムに用意します    
この時の分類直線（正解直線）を紫色の線で示しています  

### **Train**
![Logistic](https://github.com/koooootake/LogisticRegression/blob/master/ScreenShot/Train.PNG)    
Trainボタンを押すことで、計算を開始します    
予測過程を白色の線、予測結果を黄色の線で示しています  

### **Error**
![Logistic](https://github.com/koooootake/LogisticRegression/blob/master/ScreenShot/Error.PNG)     
Errorボタンを押すことで、サンプルデータにエラーデータをランダムに追加します    　　

### **Error & Train**
![Logistic](https://github.com/koooootake/LogisticRegression/blob/master/ScreenShot/Error_Train.PNG)     
エラーデータのあるサンプルデータに対してTrainボタンを押すことで、確率的な推定結果を示します    
透過度が低い（濃い）ほど正解である確率が高く、透過度が高い（薄い）ほど正解である確率が低い結果となっています

## **Requirements**
iOS 8.0+  
Xcode 7.1.1  
