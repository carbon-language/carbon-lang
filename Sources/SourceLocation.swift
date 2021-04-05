/// A position relative to the beginning of a source file, in terms understood
/// by text editors.
struct PositionInSourceFile: Comparable, Hashable {
  /// The 1-based line number of the position.
  var line: Int
  /// The 1-based column number of the position.
  var column: Int

  /// Returns `true` iff `l` precedes `r`.
  static func < (l: Self, r: Self) -> Bool {
    (l.line, l.column) < (r.line, r.column)
  }

  /// The first position in any file.
  static let start = Self(line: 1, column: 1)
}

/// A half-open range of positions in a source file.
typealias RangeOfSourceFile = Range<PositionInSourceFile>

/// A contiguous region of text in a particular source file.
struct SourceLocation: Hashable {
  /// Creates an instance that covers `span` in the file named by `f`.
  init(fileName f: String, _ span: RangeOfSourceFile) {
    self.fileName = f
    self.span = span
  }

  /// The name of the file within which this region resides.
  let fileName: String

  /// The range of positions indicated by `self` in the file named by
  /// `fileName`.
  let span: RangeOfSourceFile

  /// An empty location instance that can be used for synthesized AST nodes,
  /// etc.
  static var empty
    = SourceLocation(fileName: "", .start ..< .start)
}

extension PositionInSourceFile: CustomStringConvertible {
  /// A textual representation of `self`.
  var description: String { "\(line):\(column)" }
}

extension SourceLocation: CustomStringConvertible, CustomDebugStringConvertible {
  /// A textual representation of `self` that is commonly recognized by IDEs.
  var description: String {
    "\(fileName):\(span.lowerBound):{\(span.lowerBound)-\(span.upperBound)})"
  }

  /// A textual representation of `self` suitable for debugging.
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
