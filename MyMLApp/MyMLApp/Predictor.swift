//
//  Predictor.swift
//  MyMLApp
//
//  Created by Konrad on 30/06/2025.
//

import CoreML

class Predictor {
    let model: FashionMNISTClassifier

    init?() {
        guard let model = try? FashionMNISTClassifier(configuration: .init()) else {
            return nil
        }
        self.model = model
    }

    func predict(input: [Double]) -> String? {
        let inputDict = Dictionary(uniqueKeysWithValues:
                                    input.enumerated().map { (i, value) in ("pixel_\(i)", value) }
        )

        guard let mlInput = try? MLDictionaryFeatureProvider(dictionary: inputDict),
              let result = try? model.model.prediction(from: mlInput),
              let output = result.featureValue(for: "classLabel")?.stringValue else {
            return nil
        }

        return output
    }
}
