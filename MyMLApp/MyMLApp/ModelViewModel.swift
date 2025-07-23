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
        guard let model = try? FashionMNISTClassifier(
            configuration: .init()) else {
            return nil
        }
        self.model = model
    }

    func predict(from image: UIImage) {
        guard let resized = ImagePreprocessor.preprocess(image) else {
            predictedLabel = "Preprocessing failed"
            return
        }

        let input = FashionMNISTClassifierInput(image: resized)

        guard let result = try? model.prediction(input: input) else {
            predictedLabel = "Prediction failed"
            return
        }

        predictedLabel = result.classLabel
    }
}
