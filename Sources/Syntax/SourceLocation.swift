public struct PositionInSourceFile: Comparable, Hashable {
  public var line: Int
  public var column: Int

  public static func < (l: Self, r: Self) -> Bool {
      (l.line, l.column) < (r.line, r.column)
  }

  public static let start = Self(line: 1, column: 1)
}

public typealias RangeOfSourceFile = Range<PositionInSourceFile>

public struct SourceLocation: Hashable {
  public let fileName: String
  public let span: RangeOfSourceFile

  public static var empty
    = SourceLocation(fileName: "", span: .start ..< .start)
}

extension Range {
  func extended(toCover other: Range) -> Self {
    Swift.min(lowerBound, other.lowerBound)
      ..< Swift.max(upperBound, other.upperBound)
  }
}
