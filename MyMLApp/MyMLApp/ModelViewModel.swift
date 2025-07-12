//
//  ModelViewModel.swift
//  MyMLApp
//
//  Created by Konrad on 30/06/2025.
//

import CoreML
import Foundation
import UIKit

class ModelViewModel: ObservableObject {
    @Published var predictedLabel: String = "No prediction yet"

    private let model: FashionMNISTClassifier

    init?() {
        guard let model = try? FashionMNISTClassifier(configuration: .init()) else {
            return nil
        }
        self.model = model
    }

    func predict(from input: [Double]) {
        let dict = Dictionary(uniqueKeysWithValues:
                                input.enumerated().map { (i, value) in ("pixel_\(i)", value) }
        )

        guard let provider = try? MLDictionaryFeatureProvider(dictionary: dict),
              let result = try? model.model.prediction(from: provider) else {
            predictedLabel = "Prediction failed"
            return
        }

        if let output = result.featureValue(for: "classLabel") {
            if output.type == .string {
                predictedLabel = output.stringValue
            } else if output.type == .int64 {
                let classIndex = Int(output.int64Value)
                let classLabels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
                predictedLabel = classLabels.indices.contains(classIndex) ? classLabels[classIndex] : "Unknown class \(classIndex)"
            } else {
                predictedLabel = "Unsupported label type"
            }
        } else {
            predictedLabel = "classLabel missing"
        }
    }

    func predict(from image: UIImage) {
        guard let input = ImagePreprocessor.preprocess(image) else {
            predictedLabel = "Image conversion failed"
            return
        }
        predict(from: input)
    }
}
