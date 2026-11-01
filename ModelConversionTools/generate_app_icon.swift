// Reproducible suite-matched TN artwork. Run from the repository root:
// swift ModelConversionTools/generate_app_icon.swift
// Like Cam and Scaler, one opaque light asset lets iOS provide dark/tinted icons.
import AppKit
import CoreText
import Foundation

let output = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    .appendingPathComponent("Targets/TruongNoiserApp/Resources/Assets.xcassets/AppIcon.appiconset")
let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 1024, pixelsHigh: 1024,
    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
    colorSpaceName: .deviceRGB, bytesPerRow: 4096, bitsPerPixel: 32)!
let graphics = NSGraphicsContext(bitmapImageRep: bitmap)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = graphics

// Canonical Cam/Scaler assets use taupe around RGB 144/125/110 and 145/126/111,
// with warm ivory lettering. Keep the same restrained background variation.
let background = NSColor(srgbRed: 145 / 255, green: 126 / 255, blue: 111 / 255, alpha: 1)
let backgroundShade = NSColor(srgbRed: 142 / 255, green: 125 / 255, blue: 111 / 255, alpha: 1)
let foreground = NSColor(srgbRed: 245 / 255, green: 239 / 255, blue: 232 / 255, alpha: 1)
background.setFill()
NSRect(x: 0, y: 0, width: 1024, height: 1024).fill()
NSGradient(starting: backgroundShade, ending: background)!
    .draw(in: NSRect(x: 0, y: 0, width: 1024, height: 1024), angle: 35)

// Match Scaler's 644×480 centerline frame and 34px stroke. An open path gives
// the clean-image sparkle the same rounded breathing space as its arrow.
foreground.setStroke()
let outline = NSBezierPath()
outline.lineWidth = 34
outline.lineCapStyle = .round
outline.lineJoinStyle = .round
outline.move(to: NSPoint(x: 718, y: 736))
outline.line(to: NSPoint(x: 245, y: 736))
outline.curve(to: NSPoint(x: 183, y: 674), controlPoint1: NSPoint(x: 211, y: 736), controlPoint2: NSPoint(x: 183, y: 708))
outline.line(to: NSPoint(x: 183, y: 318))
outline.curve(to: NSPoint(x: 245, y: 256), controlPoint1: NSPoint(x: 183, y: 284), controlPoint2: NSPoint(x: 211, y: 256))
outline.line(to: NSPoint(x: 765, y: 256))
outline.curve(to: NSPoint(x: 827, y: 318), controlPoint1: NSPoint(x: 799, y: 256), controlPoint2: NSPoint(x: 827, y: 284))
outline.line(to: NSPoint(x: 827, y: 640))
outline.stroke()

// Use actual ink bounds for optical placement. The earlier 322pt artwork had
// a 214px cap height; Scaler's monogram is approximately 273px tall. The same
// Times serif preserves the suite's stroke contrast without synthetic bolding.
let text = NSAttributedString(string: "TN", attributes: [
    .font: NSFont(name: "TimesNewRomanPSMT", size: 410)!,
    .foregroundColor: foreground,
    .kern: -20,
])
let line = CTLineCreateWithAttributedString(text)
let context = graphics.cgContext
let bounds = CTLineGetImageBounds(line, context)
let scale = 273 / bounds.height
context.saveGState()
context.translateBy(x: 512 - bounds.midX * scale, y: 482.5 - bounds.midY * scale)
context.scaleBy(x: scale, y: scale)
context.textPosition = .zero
CTLineDraw(line, context)
context.restoreGState()

// Retain Noiser's four-point clean-image mark at the suite's corner accent.
foreground.setFill()
let sparkle = NSBezierPath()
sparkle.move(to: NSPoint(x: 810, y: 818))
sparkle.line(to: NSPoint(x: 828, y: 773))
sparkle.line(to: NSPoint(x: 873, y: 755))
sparkle.line(to: NSPoint(x: 828, y: 737))
sparkle.line(to: NSPoint(x: 810, y: 692))
sparkle.line(to: NSPoint(x: 792, y: 737))
sparkle.line(to: NSPoint(x: 747, y: 755))
sparkle.line(to: NSPoint(x: 792, y: 773))
sparkle.close()
sparkle.fill()

NSGraphicsContext.restoreGraphicsState()
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
try bitmap.representation(using: .png, properties: [:])!.write(to: output.appendingPathComponent("AppIcon.png"))
let contents = """
{
  "images": [
    {
      "idiom": "universal",
      "platform": "ios",
      "size": "1024x1024",
      "filename": "AppIcon.png"
    }
  ],
  "info": {
    "author": "xcode",
    "version": 1
  }
}

"""
try contents.write(to: output.appendingPathComponent("Contents.json"), atomically: true, encoding: .utf8)
let obsoleteDark = output.appendingPathComponent("AppIcon-dark.png")
if FileManager.default.fileExists(atPath: obsoleteDark.path) {
    try FileManager.default.removeItem(at: obsoleteDark)
}
