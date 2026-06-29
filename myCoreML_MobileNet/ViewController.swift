//
//  ViewController.swift
//  myCoreML_MobileNet
//
//  Created by shuai on 2026/6/26.
//

import UIKit
import PhotosUI
internal import Vision


class ViewController: UIViewController {
    @IBOutlet weak var resultLabel: UILabel!
    
    @IBOutlet weak var imageView: UIImageView!
    // 相册选取图片

    @IBAction func pickPhotoFromLibrary(_ sender: UIButton) {
        var config = PHPickerConfiguration()
                config.selectionLimit = 1
                config.filter = .images
                let picker = PHPickerViewController(configuration: config)
                picker.delegate = self
                present(picker, animated: true)
    }
    // 相机拍照

    @IBAction func takeCameraPhoto(_ sender: UIButton) {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
                    alert(msg: "当前设备无相机，使用模拟器请选择相册")
                    return
                }
                let camera = UIImagePickerController()
                camera.sourceType = .camera
                camera.delegate = self
                present(camera, animated: true)
    }
    
    let predictor = ImagePredictor.shared
    
    override func viewDidLoad() {
        super.viewDidLoad()
        resultLabel.text = "选择图片开始识别"

    }
    
    func startClassify(image: UIImage) {
        imageView.image = image
        resultLabel.text = "识别中。。。"
        
        // 把 predictor 换成 SystemImageClassifier.shared
//        SystemImageClassifier.shared.classifyImage(image) { [weak self] preds in
//            guard let self = self else { return }
//            var text = "系统原生识别结果：\n"
//            // 过滤极低置信度垃圾数据
//            let validList = preds.filter { $0.confidence > 0.01 }
//            if validList.isEmpty {
//                text += "无高置信匹配物体"
//            } else {
//                validList.forEach { p in
//                    let conf = String(format: "%.2f%%", p.confidence * 100)
//                    text += "\(p.label) 置信度：\(conf)\n"
//                }
//            }
//            self.resultLabel.text = text
//        }
        
        
//        predictor.prodictImage(image) { [weak self] preds in
//            guard let self = self else { return }
//            var text = "识别结果：\n"
//            preds.forEach { p in
//                let conf = String(format: "%.2f%%", p.confidence * 100)
//                text += "\(p.label) 置信度：\(conf)\n"
//            }
//            self.resultLabel.text = text
//        }
        
        
        // 执行多任务并行识别
        ImagePredictorV2.shared.runMultiTask(image: image) {[weak self] result in
            guard let self = self else { return }

            // 1. 打印分类Top1结果
            var text = "识别结果：\n"

            if let topPredict = result.classResult.first {
                text += "识别物体：\(topPredict.label) 置信度：\(String(format: "%.2f%%", topPredict.confidence * 100))\n"
            }

            // 2. 打印识别到的文字
            let allText = result.ocrTexts.joined(separator: "，")
            text += "图片文字：\(allText)\n"

            // 3. 打印人体姿态关键点（iOS14+）
            if #available(iOS 14.0, *) {
                if let pose = result.humanPoses.first {
                    // 获取所有人体关节点
                    let allPoints = try? pose.recognizedPoints(.all)
                    text += "检测到人体，关节点数量：\(allPoints?.count ?? 0)\n"
                }
            } else {
                // Fallback on earlier versions
            }

            self.resultLabel.text = text

        }
        
    }
    

}
// PHPicker 相册代理
extension ViewController: PHPickerViewControllerDelegate {
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        guard let item = results.first else { return }
        item.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] img, err in
            guard let image = img as? UIImage else { return }
            DispatchQueue.main.async {
                self?.startClassify(image: image)
            }
        }
    }
}

// 相机图片回调代理
extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true)
        guard let img = info[.originalImage] as? UIImage else { return }
        startClassify(image: img)
    }
}
extension ViewController {
    func alert(msg: String) {
        let alertController = UIAlertController(title: nil, message: msg, preferredStyle: .alert)
        alertController.addAction(UIAlertAction(title: "确定", style: .default, handler: nil))
        present(alertController, animated: true, completion: nil)
    }
}
