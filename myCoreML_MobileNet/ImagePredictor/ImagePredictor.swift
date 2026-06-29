//
//  ImagePredictor.swift
//  myCoreML_MobileNet
//
//  Created by shuai on 2026/6/29.
//

import UIKit
import CoreML
public import Vision
// 预测结果结构体
struct Prediction {
    let label: String
    let confidence: Float
    
}

class ImagePredictor: NSObject {
    // 全局单利，避免重复加载（每次重新加载比较消耗性能）
    static let shared = ImagePredictor()
    
    // 临时存储回调
    private var completionCallback: (([Prediction]) -> Void)?
    
    private override init() {
        super.init()
        setupClassifier()
    }
    
    private var classificationRequest: VNCoreMLRequest!
    
    private func setupClassifier() {
        do {
            // 默认配置
            let configuration = MLModelConfiguration()

            // 导入模型后生成的对象
            let mobileNet = try MobileNetV2(configuration: configuration)
            let mlModel = mobileNet.model

            // 包装成 Vision 模型
            let visionModel = try VNCoreMLModel(for: mlModel)

            // 创建分类请求
            let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
                self?.visionRequestHandler(request, error: error)
            }
            // 图片预处理：中心裁剪缩放适配模型224输入尺寸
            request.imageCropAndScaleOption = .centerCrop
            self.classificationRequest = request
            
            
        } catch {
            fatalError("模型加载失败，\(error.localizedDescription)")
        }
    }
    
    // 执行图像预测入口
    func makePredictions(for image: UIImage, completion: @escaping(([Prediction]) -> Void)){
        // 切换异步线程，防止阻塞UI
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else{return}
            
            // 获取图片方向，修正旋转
            guard let cgImg = image.cgImage,
                  let orientation = CGImagePropertyOrientation(rawValue: UInt32(image.imageOrientation.rawValue))
            else{
                completion([])
                return
            }
            
            //创建Vistion图像处理器
            let handler = VNImageRequestHandler(cgImage: cgImg, orientation: orientation)

            do {
                // 执行识别请求
                try handler.perform([self.classificationRequest])
            } catch{
                print("识别失败\(error)")
                return
            }
        }
    }
    
    
    // 识别结果解析
    private func visionRequestHandler(_ request: VNRequest, error: Error?) {
        if let err = error {
            print("请求结果错误：\(err)")
            
            // 调用回调并传递空数组或者错误信息
            DispatchQueue.main.async {
                self.completionCallback?([])
            }
            return
        }
        
        // 强转分类观测结果
        guard let results = request.results as? [VNClassificationObservation] else {
            print("无法识别图像。")
            
            // 调用回调并传递空数组
            DispatchQueue.main.async {
                self.completionCallback?([])
            }
            return
        }
        
        // 创建 Prediction 数组，并根据置信度排序，取前 3 个结果
        let predictions = results                          // 使用明确类型
            .map { Prediction(label: $0.identifier, confidence: Float($0.confidence)) }  // 第一步：转换
            .sorted { $0.confidence > $1.confidence }                                  // 第二步：排序
            .prefix(3)                                                                // 第三步：截取前 3 个

        DispatchQueue.main.async {
            self.completionCallback?(Array(predictions))   // 转换为 Array，并调用回调
        }
        
        // 处理预测结果（可选，用于调试）
        print(Array(predictions))
    }
}

extension ImagePredictor {
    // 扩展封装调用简化
    func prodictImage(_ img: UIImage, finish:@escaping([Prediction]) -> Void){
        self.completionCallback = finish
        self.makePredictions(for: img, completion: finish)
    }
}
