import UIKit
public import Vision
import CoreML


struct MultiVisionResult {
    var classResult: [Prediction] = []
    var ocrTexts: [String] = []
    @available(iOS 14.0, *)
    var humanPoses: [VNHumanBodyPoseObservation] = []
}

class ImagePredictorV2 {
    static let shared = ImagePredictorV2()
    private init() {}
    
    private var classifyRequest: VNCoreMLRequest?
    
    @available(iOS 14.0, *)
    private let bodyPoseRequest = VNDetectHumanBodyPoseRequest()
    
    private let textRequest: VNRecognizeTextRequest = {
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = true
        return req
    }()
    
    private var waitCallbacks: [() -> Void] = []
    
    // 加载模型（移除load异步闭包，彻底解决throws转换报错）
    private func setupModelRequest() {
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            do {
                let configuration = MLModelConfiguration()
                let mlModel = try AnimalClassifier(configuration: configuration)
                let visionModel = try VNCoreMLModel(for: mlModel.model)
                let visionReq = VNCoreMLRequest(model: visionModel)
                visionReq.imageCropAndScaleOption = .centerCrop
                self.classifyRequest = visionReq
                
                DispatchQueue.main.async {
                    self.runWaitingTasks()
                }
            } catch let err {
                print("模型加载异常：\(err)")
                DispatchQueue.main.async {
                    self.waitCallbacks.forEach { $0() }
                    self.waitCallbacks.removeAll()
                }
            }
        }
    }
    
    private func runWaitingTasks() {
        let list = waitCallbacks
        waitCallbacks.removeAll()
        list.forEach { $0() }
    }
    
    func runMultiTask(image: UIImage, completion: @escaping (MultiVisionResult) -> Void) {
    guard let classifyReq = self.classifyRequest else {
        waitCallbacks.append {
            self.runMultiTask(image: image, completion: completion)
        }
        setupModelRequest()
        return
    }
    
    DispatchQueue.global(qos: .userInitiated).async {
        guard let cgImage = image.cgImage else {
            DispatchQueue.main.async { completion(MultiVisionResult()) }
            return
        }
        
        let imgOrientation: CGImagePropertyOrientation
        switch image.imageOrientation {
        case .up: imgOrientation = .up
        case .upMirrored: imgOrientation = .upMirrored
        case .down: imgOrientation = .down
        case .downMirrored: imgOrientation = .downMirrored
        case .left: imgOrientation = .left
        case .leftMirrored: imgOrientation = .leftMirrored
        case .right: imgOrientation = .right
        case .rightMirrored: imgOrientation = .rightMirrored
        @unknown default: imgOrientation = .up
        }
        
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: imgOrientation)
        do {
            if #available(iOS 14.0, *) {
                try handler.perform([classifyReq, self.bodyPoseRequest, self.textRequest])
            } else {
                try handler.perform([classifyReq, self.textRequest])
            }
        } catch {
            print("多任务推理失败：\(error)")
            DispatchQueue.main.async { completion(MultiVisionResult()) }
            return
        }
        
        var output = MultiVisionResult()
        if let classObs = classifyReq.results as? [VNClassificationObservation] {
            output.classResult = classObs.map { Prediction(label: $0.identifier, confidence: $0.confidence) }
        }
        
        // 修正 OCR 文本提取部分
        let textObs = self.textRequest.results ?? []
        output.ocrTexts = textObs.compactMap { observation in
            observation.topCandidates(1).first?.string // 获取第一个候选文本字符串
        }
        
        if #available(iOS 14.0, *) {
            output.humanPoses = self.bodyPoseRequest.results ?? []
        }
        
        DispatchQueue.main.async { completion(output) }
    }
}
}
