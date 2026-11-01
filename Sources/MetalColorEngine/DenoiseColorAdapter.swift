import Foundation
import TruongNoiserCore
import MMapCanvas

/// The trained SIDD network sees developed gamma-sRGB, with alpha kept outside the network.
/// Its linear-light residual is restored onto the original P3 pixels, preserving gamut/HDR
/// components that were outside the model's bounded display-domain input.
public struct PreparedDenoiseTile: Sendable {
    public let side: Int
    public let input: [Float16]
    public let originalRGBA: [Float16]
    public let exposure: Float
}

public enum DenoiseColorAdapter {
    public static func prepare(source: MMapBuffer, tile: TileDescriptor, componentPeak: Float) throws -> PreparedDenoiseTile {
        let side = tile.inputSize
        guard side > 0, side <= 1024, componentPeak.isFinite, componentPeak > 0 else {
            throw ScalerError.invalidImageDimensions(width: side, height: side)
        }
        let count = side * side
        var rgba = [Float16](repeating: 0, count: count * 4)
        rgba.withUnsafeMutableBytes { bytes in
            if let base = bytes.baseAddress {
                source.readRegion(x: tile.inputOriginX, y: tile.inputOriginY,
                    regionWidth: side, regionHeight: side, into: base, destinationRowBytes: side * 8)
            }
        }
        let exposure = max(1,componentPeak)
        var input = [Float16](repeating: 0,count:count*3)
        for i in 0..<count {
            let a = Float(rgba[i*4+3])
            guard a.isFinite else { throw ScalerError.modelInferenceFailed(modelName: "Denoise input", tileIndex: tile.tileIndex, reason: "Non-finite source alpha") }
            if a <= 0 { continue }
            let rgb = SIMD3<Float>(Float(rgba[i*4]),Float(rgba[i*4+1]),Float(rgba[i*4+2])) / a
            guard rgb.x.isFinite, rgb.y.isFinite, rgb.z.isFinite else {
                throw ScalerError.modelInferenceFailed(modelName: "Denoise input", tileIndex: tile.tileIndex, reason: "Non-finite source RGB")
            }
            let linear = p3ToSRGB(rgb) / exposure
            for c in 0..<3 { input[c*count+i] = Float16(encode(min(max(linear[c],0),1))) }
        }
        return PreparedDenoiseTile(side: side,input:input,originalRGBA:rgba,exposure:exposure)
    }

    public static func restore(_ prediction: [Float16], prepared: PreparedDenoiseTile,
                               modelName: String, tileIndex: Int, allowed: ClosedRange<Float>,
                               lumaStrength: Float = 0.80, chromaStrength: Float = 1) throws -> [Float16] {
        let count=prepared.side*prepared.side
        guard prepared.input.count == count*3, prepared.originalRGBA.count == count*4,
              prediction.count == count*3,
              lumaStrength.isFinite, chromaStrength.isFinite,
              (0...1).contains(lumaStrength), (0...1).contains(chromaStrength) else {
            throw ScalerError.modelInferenceFailed(modelName:modelName,tileIndex:tileIndex,reason:"Invalid native-resolution RGB prediction shape")
        }
        var output=prepared.originalRGBA
        for i in 0..<count {
            let alpha=Float(prepared.originalRGBA[i*4+3])
            var residual=SIMD3<Float>(repeating:0)
            for c in 0..<3 {
                let predicted=Float(prediction[c*count+i])
                guard predicted.isFinite, allowed.contains(predicted) else {
                    throw ScalerError.modelInferenceFailed(modelName:modelName,tileIndex:tileIndex,reason:"Denoise prediction is non-finite or outside the validated range")
                }
                residual[c]=(decode(min(max(predicted,0),1))-decode(Float(prepared.input[c*count+i])))*prepared.exposure
            }
            if alpha <= 0 { continue }
            let delta=srgbToP3(residual)*alpha
            for c in 0..<3 {
                let value=max(0,Float(prepared.originalRGBA[i*4+c])+delta[c])
                guard value.isFinite, value <= Float(Float16.greatestFiniteMagnitude) else {
                    throw ScalerError.modelInferenceFailed(modelName:modelName,tileIndex:tileIndex,reason:"Denoise reconstruction exceeds finite half-float storage")
                }
                output[i*4+c]=Float16(value)
            }
        }
        let sy=min(max(lumaStrength,0),1), sc=min(max(chromaStrength,0),1)
        guard sy.isFinite, sc.isFinite else {
            throw ScalerError.modelInferenceFailed(modelName:modelName,tileIndex:tileIndex,reason:"Invalid denoise strength")
        }
        if sy == 1 && sc == 1 { return output }
        if sy == 0 && sc == 0 { return prepared.originalRGBA }
        for i in 0..<count {
            let original=SIMD3<Float>(Float(prepared.originalRGBA[i*4]),Float(prepared.originalRGBA[i*4+1]),Float(prepared.originalRGBA[i*4+2]))
            let denoised=SIMD3<Float>(Float(output[i*4]),Float(output[i*4+1]),Float(output[i*4+2]))
            let delta=denoised-original
            let dy=0.22897456*delta.x+0.69173852*delta.y+0.07928691*delta.z
            // Equivalent to independent linear-P3 Y/Cb/Cr interpolation, without a lossy
            // second matrix inversion. Both endpoint strengths remain exact copies.
            let mixed=original+sc*delta+SIMD3<Float>(repeating:(sy-sc)*dy)
            for c in 0..<3 {
                guard mixed[c].isFinite, mixed[c] <= Float(Float16.greatestFiniteMagnitude) else {
                    throw ScalerError.modelInferenceFailed(modelName:modelName,tileIndex:tileIndex,reason:"Denoise mix exceeds finite half-float range")
                }
                output[i*4+c]=Float16(max(0,mixed[c]))
            }
        }
        return output
    }
    private static func encode(_ x:Float)->Float { x <= 0.0031308 ? 12.92*x : 1.055*pow(x,1/2.4)-0.055 }
    private static func decode(_ x:Float)->Float { x <= 0.04045 ? x/12.92 : pow((x+0.055)/1.055,2.4) }
    private static func p3ToSRGB(_ x:SIMD3<Float>)->SIMD3<Float> {
        SIMD3(1.224940176*x.x-0.224940176*x.y,
              -0.042056955*x.x+1.042056955*x.y,
              -0.019637555*x.x-0.078636046*x.y+1.098273601*x.z)
    }
    private static func srgbToP3(_ x:SIMD3<Float>)->SIMD3<Float> {
        SIMD3(0.822461969*x.x+0.177538031*x.y,
              0.033194199*x.x+0.966805801*x.y,
              0.017082631*x.x+0.072397441*x.y+0.910519929*x.z)
    }
}
