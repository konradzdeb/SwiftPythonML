//
//  ContentView.swift
//  MyMLApp
//
//  Created by Konrad on 16/05/2025.
//

import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var viewModel = ModelViewModel()!
    @State private var showingImagePicker = false
    @State private var inputImage: UIImage?
    @State private var sourceType: UIImagePickerController.SourceType = .camera
    
    var body: some View {
        VStack(spacing: 20) {
            if let image = inputImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 300)
                    .cornerRadius(12)
            } else {
                Rectangle()
                    .fill(Color.gray.opacity(0.2))
                    .frame(height: 300)
                    .overlay(Text("No image selected").foregroundColor(.gray))
            }
            
            Text("Prediction:")
                .font(.title)
            Text(viewModel.predictedLabel)
                .font(.largeTitle)
                .bold()
            
            Picker("Source", selection: $sourceType) {
                Text("Camera").tag(UIImagePickerController.SourceType.camera)
                Text("Library").tag(UIImagePickerController.SourceType.photoLibrary)
            }
            .pickerStyle(.segmented)
            
            Button {
                showingImagePicker = true
            } label: {
                Label("Select Image", systemImage: sourceType == .camera ? "camera" : "photo.on.rectangle")
                    .font(.headline)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
        }
        .padding()
        .sheet(isPresented: $showingImagePicker) {
            ImagePicker(image: $inputImage, sourceType: sourceType, onImagePicked: {
                if let image = inputImage {
                    viewModel.predict(from: image)
                }
            })
        }
        .onChange(of: inputImage) { _, newImage in
            if let image = newImage {
                viewModel.predict(from: image)
            }
        }
    }
}

#Preview {
    ContentView()
}

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    var sourceType: UIImagePickerController.SourceType
    var onImagePicked: () -> Void
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
                parent.onImagePicked()
            }
            picker.dismiss(animated: true)
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = sourceType
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
}
