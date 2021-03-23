public struct PositionInSourceFile: Comparable, Hashable {
  public var line: Int
  public var column: Int

  public static func < (l: Self, r: Self) -> Bool {
      (l.line, l.column) < (r.line, r.column)
  }
}

public typealias RangeOfSourceFile = Range<PositionInSourceFile>

public struct SourceLocation {
  public let filename: String
  public let span: RangeOfSourceFile
}

extension Range {
  func extended(toCover other: Range) -> Self {
    min(lowerBound, other.lowerBound) ..< max(upperBound, other.upperBound)
  }
}
