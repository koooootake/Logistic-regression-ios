//
//  ViewController.swift
//  LogisticRegression
//
//  Created by koooootake on 2016/02/12.
//  Copyright © 2016年 koooootake. All rights reserved.
//

import UIKit
import Accelerate

protocol Data{
    var x:Float { get set }
    var y:Float { get set }
    var c:Int { get set }
}

struct TrainData: Data {
    var x:Float
    var y:Float
    var c:Int
}

protocol Vector3{
    var a:Float { get set }
    var b:Float { get set }
    var c:Float { get set }
}

struct WeightVector3: Vector3 {
    var a:Float
    var b:Float
    var c:Float
}

class ViewController: UIViewController {
    
    var TrainDataArray:[TrainData] = Array()//データ配列
    let N:Int = 200//データ数
    var answerLine:WeightVector3 = WeightVector3(a: 0, b: 0, c: 0)//正解パラメータ
    
    var R:[[Float]] = []
    var Z:[[Float]] = []

    @IBOutlet weak var TrainButton: UIButton!
    @IBOutlet weak var DataButton: UIButton!
    @IBOutlet weak var ErrorButton: UIButton!
    
    let defaultColor:UIColor = UIColor(red: 38/255.0, green: 198/255.0, blue: 218/255.0, alpha: 0.5)

    override func viewDidLoad() {
        super.viewDidLoad()
        
        //ボタン設定
        TrainButton.tag = 20000
        DataButton.tag = 30000
        ErrorButton.tag = 40000
        
        TrainButton.backgroundColor = UIColor(red: 255/255.0, green: 167/255.0, blue: 38/255.0, alpha: 1.0)
        DataButton.backgroundColor = UIColor(red: 66/255.0, green: 165/255.0, blue: 245/255.0, alpha: 1.0)
        ErrorButton.backgroundColor = UIColor(red: 239/255.0, green: 83/255.0, blue: 80/255.0, alpha: 1.0)
        
        //(x,y)軸描画
        UIGraphicsBeginImageContextWithOptions(self.view.frame.size, false, 0)
        
        let xyView:UIView = UIView(frame: self.view.frame)
        xyView.tag = 50000
        
        let xpath = UIBezierPath()
        xpath.moveToPoint(CGPointMake(0,self.view.frame.height/2))
        xpath.addLineToPoint(CGPointMake(self.view.frame.width,self.view.frame.height/2))
        UIColor(white: 0.1, alpha: 0.3).setStroke()
        xpath.stroke()
        
        let ypath = UIBezierPath()
        ypath.moveToPoint(CGPointMake(self.view.frame.width/2,0))
        ypath.addLineToPoint(CGPointMake(self.view.frame.width/2,self.view.frame.height))
        UIColor(white: 0.1, alpha: 0.3).setStroke()
        ypath.stroke()

        xyView.layer.contents = UIGraphicsGetImageFromCurrentImageContext().CGImage
        
        UIGraphicsEndImageContext()
        self.view.addSubview(xyView)
        
        //トレーニングデータの取得
        setData()

    }
    
    @IBAction func Data(sender: AnyObject) {
        
        print("Data")
        //線と点を消す
        let subviews = self.view.subviews
        for subview in subviews{
            if (0 < subview.tag && subview.tag <= N) || subview.tag == 10000 || subview.tag == 11000 || subview.tag == 15000{
                subview.removeFromSuperview()
            }
        }
        //トレーニングデータの取得
        setData()

    }
    
    //トレーニングデータを生成
    func setData(){
        
        //データを初期化
        TrainDataArray = []
        
        //正解直線をランダムに決定する
        answerLine = WeightVector3(a: getRandomNumber(Min: -1.0, Max: 1.0), b: getRandomNumber(Min: -1.0, Max: 1.0), c: getRandomNumber(Min: -1.0, Max: 1.0))
        print("AnswerLine :",answerLine.a,"*x + ",answerLine.b,"*y + ",answerLine.c)
        drawAnswerLine(answerLine)

        while TrainDataArray.count < N{
            
            let x:Float = randn()
            let y:Float = randn()
            var c:Int = Int()
            
            if answerLine.a*x + answerLine.b*y + answerLine.c >= 0{
                c = 0
            }else{
                c = 1
            }
            let test = TrainData(x: x, y: y, c: c)
           
            //描画
            let point = UIView(frame:CGRectMake(
                CGFloat(x*(Float(self.view.frame.width)/6) + Float(self.view.frame.width)/2)-12,
                CGFloat(y*(Float(self.view.frame.width)/6) + Float(self.view.frame.height)/2)-12,
                12,12))
            point.layer.masksToBounds = true
            point.layer.cornerRadius = point.frame.size.width/2.0
            point.tag = TrainDataArray.count + 1
            
            if c == 0{
                point.backgroundColor = UIColor(red: 239/255.0, green: 83/255.0, blue: 80/255.0, alpha: 1.0)//赤
                
            }else{
                point.backgroundColor = UIColor(red: 66/255.0, green: 165/255.0, blue: 245/255.0, alpha: 1.0)//青
                
            }
            TrainDataArray.append(test)
            self.view.addSubview(point)
        }
    }
    
    //学習
    @IBAction func Train(sender: AnyObject) {
        print("Train\n | \n v")
        
        var count = 0
        
        //初期パラメータをランダムに設置
        var weightVector:WeightVector3 = WeightVector3(a: getRandomNumber(Min: -1.0, Max: 1.0), b: getRandomNumber(Min: -1.0, Max: 1.0), c: getRandomNumber(Min: -1.0, Max: 1.0))
        
        var oldWeightVector:[WeightVector3] = []//検討したパラメータ
        var beforeWeightVector = weightVector//一つ前のパラメータ
        
        //z=1を持つデータ行列
        var trainMat: [[Float]] = []
        for trainData in TrainDataArray{
            trainMat.append([1.0,trainData.x,trainData.y])
        }
        
        let trainTransposedMat = transposed(trainMat)
        
        //尤度関数が最大になるようにパラメータを決定する
        while count < 100{
            
            R = []
            Z = []
            likelihood(weightVector)//最尤推定
            oldWeightVector.append(weightVector)
            
            //パラメータを計算
            let r1 = product(trainTransposedMat,matB:R)
            let r2 = product(r1, matB: trainMat)
            let r3 = invers(r2)
            let r4 = product(r3, matB: trainTransposedMat)
            let result = product(r4, matB: Z)
            
            //パラメータを更新
            beforeWeightVector = weightVector
            weightVector = WeightVector3(a: beforeWeightVector.a - result[1][0], b: beforeWeightVector.b - result[2][0], c: beforeWeightVector.c - result[0][0])
            print(weightVector)
            
            //変化率が閾値を切った時点で終了
            if ((weightVector.a - beforeWeightVector.a)*(weightVector.a - beforeWeightVector.a) + (weightVector.b - beforeWeightVector.b)*(weightVector.b - beforeWeightVector.b) + (weightVector.c - beforeWeightVector.c)*(weightVector.c - beforeWeightVector.c))/(beforeWeightVector.a * beforeWeightVector.a + beforeWeightVector.b * beforeWeightVector.b + beforeWeightVector.c * beforeWeightVector.c) < 0.00001{
                
                weightVector = beforeWeightVector
                print("END")
                break
            }
 
            count++
        }
        
        drawOldSplitLine(oldWeightVector)//軌跡線を描画
        likelihoodPointColor(weightVector)//確率的な推定を可視化
        drawSplitLine(weightVector)//推定線を描画

        print("PredictLine :",weightVector.a,"*x + ",weightVector.b,"*y + ",weightVector.c)
        
    }
    
    //逆行列
    func invers(mat:[[Float]]) -> [[Float]]{
        var resultMat:[[Float]] = []
        var det:Float = 0.0
        
        var re = 0
        while re < 3{
            resultMat.append(Array(count: 3, repeatedValue: 0))
            re++
        }
        
        for (var i = 0; i < 3; i++){
            
            var right:Float = 1.0
            var left:Float = 1.0
            
            for (var j = 0; j < 3; j++){
                
                right *= mat[(i+j)%3][j%3]
                left *= mat[(i+3-j)%3][j%3]
            
            }
            det = det + right - left
        }
        
        if det == 0{
            return resultMat
            
        }else{
            for (var i = 0; i < 3; i++){
                for (var j = 0; j < 3; j++){
                    
                    let aaa = mat[(i+1)%3][(j+1)%3] * mat[(i+2)%3][(j+2)%3]
                    let bbb = mat[(i+1)%3][(j+2)%3] * mat[(i+2)%3][(j+1)%3]
                    
                    resultMat[j][i] = ( aaa - bbb ) / det
                    
                }
            }
        }
        return resultMat
    }
    
    //転置行列
    func transposed(mat:[[Float]]) -> [[Float]]{
        
        var resultMat:[[Float]] = []
        
        var re = 0
        while re < mat[0].count{
            resultMat.append(Array(count: mat.count, repeatedValue: 0))
            re++
        }
        
        var i = 0
        while i < mat[0].count{//列
            
            var j = 0
            while j < mat.count{//行
                
                resultMat[i][j] = mat[j][i]
            
                j++
            }
            i++
        }

        return resultMat
    }
    
    //行列の積
    func product(matA:[[Float]],matB:[[Float]]) -> [[Float]]{
        
        var resultMat:[[Float]] = []
        
        var re = 0
        while re < matA.count{
            resultMat.append(Array(count: matB[0].count, repeatedValue: 0))
            re++
        }

        var i = 0
        while i < matA.count{//左の行分回す
            
            var j = 0
            while j < matB[0].count{//右の列分回す
                
                var k = 0
                while k < matB.count{//右の行分回す と　左の列
                    resultMat[i][j] += matA[i][k] * matB[k][j]
                    k++
                }
                j++
    
            }
            i++
  
        }
        
        return resultMat
        
    }
    
    
    
    //正規乱数生成
    func randn() -> Float{
        let randn = getRandomNumber(Min: 0.0, Max:1.0) + getRandomNumber(Min: 0.0, Max:1.0)
            + getRandomNumber(Min: 0.0, Max:1.0) + getRandomNumber(Min: 0.0, Max:1.0)
            + getRandomNumber(Min: 0.0, Max:1.0) + getRandomNumber(Min: 0.0, Max:1.0)
            + getRandomNumber(Min: 0.0, Max:1.0) + getRandomNumber(Min: 0.0, Max:1.0)
            + getRandomNumber(Min: 0.0, Max:1.0) + getRandomNumber(Min: 0.0, Max:1.0)
            + getRandomNumber(Min: 0.0, Max:1.0) + getRandomNumber(Min: 0.0, Max:1.0)
        
        return randn - 6.0
    }
    
    //乱数生成
    func getRandomNumber(Min _Min : Float, Max _Max : Float)->Float {
        return ( Float(arc4random_uniform(UINT32_MAX)) / Float(UINT32_MAX) ) * (_Max - _Min) + _Min
    }
    
    //正解線を引く
    func drawAnswerLine(answerVector:WeightVector3){
        
        let lineView:UIView = UIView(frame: self.view.frame)
        lineView.tag = 15000
        
        // CoreGraphicsで描画する
        UIGraphicsBeginImageContextWithOptions(self.view.frame.size, false, 0)
        
        let y_min = (-answerVector.a * -Float(self.view.frame.width)/2 - answerVector.c*Float(self.view.frame.width)/6) / answerVector.b + Float(self.view.frame.size.height)/2
        let y_max = (-answerVector.a * Float(self.view.frame.width)/2 - answerVector.c*Float(self.view.frame.width)/6) / answerVector.b + Float(self.view.frame.size.height)/2
        
        let path = UIBezierPath()
        path.moveToPoint(CGPointMake(0,CGFloat(y_min)))
        path.addLineToPoint(CGPointMake(self.view.frame.width,CGFloat(y_max)))
        path.lineWidth = 2.0
        UIColor(red: 126/255.0, green: 87/255.0, blue: 194/255.0, alpha: 1.0).setStroke()
        path.stroke()
        
        lineView.layer.contents = UIGraphicsGetImageFromCurrentImageContext().CGImage
        
        UIGraphicsEndImageContext()
        
        self.view.addSubview(lineView)
        self.view.bringSubviewToFront(TrainButton)
        self.view.bringSubviewToFront(DataButton)
        self.view.bringSubviewToFront(ErrorButton)

    }
    
    //予測線を引く
    func drawSplitLine(minWeightVector:WeightVector3){
        
        // 一度線を消す
        let subviews = self.view.subviews
        for subview in subviews{
            if subview.tag == 10000{
                subview.removeFromSuperview()
            }
        }
        
        let lineView:UIView = UIView(frame: self.view.frame)
        lineView.tag = 10000
        
        UIGraphicsBeginImageContextWithOptions(self.view.frame.size, false, 0)
        
        // 最小尤度を持つ直線
        let y_min = (-minWeightVector.a * -Float(self.view.frame.width)/2 - minWeightVector.c*Float(self.view.frame.width)/6) / minWeightVector.b + Float(self.view.frame.size.height)/2
        let y_max = (-minWeightVector.a * Float(self.view.frame.width)/2 - minWeightVector.c*Float(self.view.frame.width)/6) / minWeightVector.b + Float(self.view.frame.size.height)/2
        
        let path = UIBezierPath()
        path.moveToPoint(CGPointMake(0,CGFloat(y_min)))
        path.addLineToPoint(CGPointMake(self.view.frame.width,CGFloat(y_max)))
        path.lineWidth = 2.0
        UIColor(red: 255/255.0, green: 167/255.0, blue: 38/255.0, alpha: 1.0).setStroke()
        path.stroke()
        
        // viewのlayerに描画したものをセットする
        lineView.layer.contents = UIGraphicsGetImageFromCurrentImageContext().CGImage
        
        UIGraphicsEndImageContext()
        
        self.view.addSubview(lineView)
        self.view.bringSubviewToFront(TrainButton)
        self.view.bringSubviewToFront(DataButton)
        self.view.bringSubviewToFront(ErrorButton)
        
    }
    
    //検討線を引く関数
    func drawOldSplitLine(oldWeightVector:[WeightVector3]){
        
        // 一度線を消す
        let subviews = self.view.subviews
        for subview in subviews{
            if subview.tag == 11000{
                subview.removeFromSuperview()
            }
        }
        
        let lineView:UIView = UIView(frame: self.view.frame)
        lineView.tag = 11000
        
        UIGraphicsBeginImageContextWithOptions(self.view.frame.size, false, 0)
        
        for weightVector in oldWeightVector{
        
            // 最小尤度を持つ直線
            let y_min = (-weightVector.a * -Float(self.view.frame.width)/2 - weightVector.c*Float(self.view.frame.width)/6) / weightVector.b + Float(self.view.frame.size.height)/2
            let y_max = (-weightVector.a * Float(self.view.frame.width)/2 - weightVector.c*Float(self.view.frame.width)/6) / weightVector.b + Float(self.view.frame.size.height)/2
            
            let path = UIBezierPath()
            path.moveToPoint(CGPointMake(0,CGFloat(y_min)))
            path.addLineToPoint(CGPointMake(self.view.frame.width,CGFloat(y_max)))
            path.lineWidth = 2.0
            UIColor(white: 1.0, alpha: 0.1).setStroke()
            path.stroke()
        }
        
        // viewのlayerに描画したものをセットする
        lineView.layer.contents = UIGraphicsGetImageFromCurrentImageContext().CGImage
        
        UIGraphicsEndImageContext()
        
        self.view.addSubview(lineView)
        self.view.bringSubviewToFront(TrainButton)
        self.view.bringSubviewToFront(DataButton)
        self.view.bringSubviewToFront(ErrorButton)
        
    }
    
    //ロジスティック関数
    func sigmoid(a:Float) -> Float{
        let sig = 1.0/(1.0 + exp(-a))
        return sig
    }
    
    //内積
    func inner(left:TrainData , right:WeightVector3) -> Float{
        return left.x * right.a + left.y * right.b + Float(left.c) * right.c
    }
    
    //(x,y)平面上の任意の点において、得られたデータの属性がp=1である確率
    func getProb(x:Float, y:Float, weightVector:WeightVector3) -> Float{
        
        let feature_vector = TrainData(x: x, y: y, c: 1)
        let a = inner(feature_vector, right: weightVector)
        // print("p(c=1|x,y)",x ,y ,sigmoid(a))
        
        return sigmoid(a)
    }
    

    //トレーニングセットのデータが得られる確率を最尤推定
    func likelihood(weightVector:WeightVector3) -> Float{
 
        var likelihood:Float = 0.0
        
        for (index,trainData)  in TrainDataArray.enumerate(){
            
            let prob = getProb(trainData.x,y: trainData.y,weightVector: weightVector)
            
            //行列生成
            var j = 0
            var r:[Float] = []
            while j < N{
                if j == index{
                    r.append(prob*(1.0-prob))
                }else{
                    r.append(0)
                }
                j++
            }
            
            R.append(r)
            var z:[Float] = []
            z.append(prob-Float(trainData.c))
            Z.append(z)
            
            var iLikelihood:Float  = 0.0
            
            if trainData.c == 1{
                
                if prob == 0.0{
                    iLikelihood = 0
                }else{
                    iLikelihood = log(prob)
                }

            }else if trainData.c == 0{
                
                if prob == 1.0{
                    iLikelihood = 0
                }else{
                    iLikelihood = log(1.0-prob)
                }
            }
            
            likelihood = likelihood - iLikelihood
            
        }
        
        return likelihood
        
    }
    
    
    //Pointの色を変更する
    func likelihoodPointColor(weightVector:WeightVector3){

        for (index,trainData) in TrainDataArray.enumerate(){
            
            let prob = getProb(trainData.x,y: trainData.y,weightVector: weightVector)
            let subviews = self.view.subviews

            if trainData.c == 1{
                
                for subview in subviews {
                    if subview.tag == index + 1{
                        
                        if prob < 0.2{
                            subview.backgroundColor = UIColor(red: 66/255.0, green: 165/255.0, blue: 245/255.0, alpha: CGFloat(0.2))
                        }else{
                            subview.backgroundColor = UIColor(red: 66/255.0, green: 165/255.0, blue: 245/255.0, alpha: CGFloat(prob))
                        }
                    }
                }
                
            }else if trainData.c == 0{
                
                for subview in subviews {
                    if subview.tag == index + 1{
                        
                        if (1.0 - prob) < 0.2{
                            subview.backgroundColor = UIColor(red: 239/255.0, green: 83/255.0, blue: 80/255.0,  alpha: CGFloat(0.2))
                        }else{
                            subview.backgroundColor = UIColor(red: 239/255.0, green: 83/255.0, blue: 80/255.0,  alpha: CGFloat(1.0 - prob))
                        }
                    }
                }
            }
        }
        
        
    }
    
    //エラーデータを追加する
    @IBAction func Error(sender: AnyObject) {
        
        print("Error")
        let subviews = self.view.subviews
        
        for (index,trainData) in TrainDataArray.enumerate(){
            
            if arc4random_uniform(100)%20 == 0{
                
                if trainData.c == 0{
                    TrainDataArray[index] = TrainData(x: trainData.x, y: trainData.y, c: 1)
                    
                    for subview in subviews {
                        if subview.tag == index + 1{
                            subview.backgroundColor = UIColor(red: 66/255.0, green: 165/255.0, blue: 245/255.0, alpha: 1.0)
                        }
                    }
                    
                }else if trainData.c == 1{
                    TrainDataArray[index] = TrainData(x: trainData.x, y: trainData.y, c: 0)
                    
                    for subview in subviews {
                        if subview.tag == index + 1{
                            subview.backgroundColor = UIColor(red: 239/255.0, green: 83/255.0, blue: 80/255.0, alpha: 1.0)
                        }
                    }
                }
                
            }
            
        }
        
        
    }

    
    override func prefersStatusBarHidden() -> Bool {
        return true
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }


}

