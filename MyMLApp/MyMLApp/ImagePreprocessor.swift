//
//  Untitled.swift
//  MyMLApp
//
//  Created by Konrad on 30/06/2025.
//

import UIKit
import CoreImage

struct ImagePreprocessor {
    static func preprocess(_ image: UIImage, size: CGSize = CGSize(width: 28, height: 28)) -> [Double]? {
        guard let cgImage = image.cgImage else { return nil }

        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext()
        let grayscaleFilter = CIFilter(name: "CIPhotoEffectMono")
        grayscaleFilter?.setValue(ciImage, forKey: kCIInputImageKey)

        guard let outputImage = grayscaleFilter?.outputImage,
              let resized = context.createCGImage(outputImage, from: CGRect(origin: .zero, size: size)) else {
            return nil
        }

        let width = Int(size.width)
        let height = Int(size.height)
        let bitmap = CGContext(data: nil,
                               width: width,
                               height: height,
                               bitsPerComponent: 8,
                               bytesPerRow: width,
                               space: CGColorSpaceCreateDeviceGray(),
                               bitmapInfo: CGImageAlphaInfo.none.rawValue)

        bitmap?.draw(resized, in: CGRect(origin: .zero, size: size))

        guard let data = bitmap?.data else { return nil }

        let buffer = UnsafeBufferPointer(start: data.assumingMemoryBound(to: UInt8.self), count: width * height)
        return buffer.map { Double($0) / 255.0 }
    }
}
