//
//  SystemImageClassifier.swift
//  myCoreML_MobileNet
//
//  Created by shuai on 2026/6/29.
//

import UIKit
public import Vision
// 统一预测结果结构体，和之前逻辑兼容，上层ViewController不用改
struct SysPrediction {
    let label: String
    let confidence: Float
}

class SystemImageClassifier {
    // 单例
    static let shared = SystemImageClassifier()
    private init() {}
    
    // 系统原生分类请求
    private var classifyRequest: VNClassifyImageRequest!
    
    // 初始化请求
    private func setupRequest() {
        let request = VNClassifyImageRequest { [weak self] request, error in
            guard let self = self else { return }
            if let err = error {
                print("系统分类识别失败：\(err.localizedDescription)")
                DispatchQueue.main.async {
                    self.completion?([])
                }
                return
            }
            // 解析系统分类结果
            guard let observations = request.results as? [VNClassificationObservation] else {
                DispatchQueue.main.async {
                    self.completion?([])
                }
                return
            }
            // 转换为统一结构体，按置信度从高到低排序，取前5
            let results = observations
                .map { SysPrediction(label: $0.identifier, confidence: $0.confidence) }
                .sorted { $0.confidence > $1.confidence }
                .prefix(5)
                .map { $0 }
            
            DispatchQueue.main.async {
                self.completion?(results)
            }
        }
        // 可选：设置识别精度 .accurate / .fast
        request.revision = VNClassifyImageRequestRevision1
        self.classifyRequest = request
    }
    
    // 回调存储
    private var completion: (([SysPrediction]) -> Void)?
    
    // 对外入口方法
    func classifyImage(_ image: UIImage, completion: @escaping ([SysPrediction]) -> Void) {
        self.completion = completion
        // 懒加载请求
        if classifyRequest == nil {
            setupRequest()
        }
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            // 修正图片方向
            guard let cgImg = image.cgImage,
                  let imgOrientation = CGImagePropertyOrientation(rawValue: UInt32(image.imageOrientation.rawValue))
            else {
                DispatchQueue.main.async {
                    completion([])
                }
                return
            }
            let handler = VNImageRequestHandler(cgImage: cgImg, orientation: imgOrientation)
            do {
                try handler.perform([self.classifyRequest])
            } catch {
                print("执行识别请求异常：\(error)")
                DispatchQueue.main.async {
                    completion([])
                }
            }
        }
    }
}
