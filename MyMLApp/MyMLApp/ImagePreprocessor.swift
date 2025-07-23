//
//  Untitled.swift
//  MyMLApp
//
//  Created by Konrad on 30/06/2025.
//

import UIKit
import CoreImage
import CoreML

struct ImagePreprocessor {
    static func preprocess(_ image: UIImage,
                        size: CGSize = CGSize(width: 28, height: 28)) -> CVPixelBuffer? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_OneComponent8,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            return nil
        }
        
        guard let cgImage = image.cgImage else {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0,
                     width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        
        return buffer
    }
}
