struct PositionInSourceFile: Comparable, Hashable {
  var line: Int
  var column: Int

  static func < (l: Self, r: Self) -> Bool {
    (l.line, l.column) < (r.line, r.column)
  }

  static let start = Self(line: 1, column: 1)
}

typealias RangeOfSourceFile = Range<PositionInSourceFile>

struct SourceLocation: Hashable {
  init(fileName: String, _ span: RangeOfSourceFile) {
    self.fileName = fileName
    self.span = span
  }

  let fileName: String
  let span: RangeOfSourceFile

  static var empty
    = SourceLocation(fileName: "", .start ..< .start)
}

extension PositionInSourceFile: CustomStringConvertible {
  var description: String { "\(line):\(column)" }
}

extension SourceLocation: CustomStringConvertible, CustomDebugStringConvertible {
  var description: String {
    "\(fileName):\(span.lowerBound):{\(span.lowerBound)-\(span.upperBound)})"
  }
  
  var debugDescription: String {
    "SourceLocation(fileName: \(String(reflecting: fileName)), \(span))"
  }
}

extension Range {
  func extended(toCover other: Range) -> Self {
    Swift.min(lowerBound, other.lowerBound)
      ..< Swift.max(upperBound, other.upperBound)
  }
}
