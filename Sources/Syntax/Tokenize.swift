import Foundation

typealias TokenCode = CarbonParser.CitronTokenCode

public struct TokenKind: Hashable {
  init(_ value: TokenCode) { self.value = value }
  let value: TokenCode
}

typealias KeywordSpec = (literalText: String, token: TokenCode)
typealias PatternSpec = (pattern: String, token: TokenCode?)

let keywords: [KeywordSpec] = [
  ("and", .AND),
  ("->", .ARROW),
  ("auto", .AUTO),
  ("bool", .BOOL),
  ("break", .BREAK),
  ("case", .CASE),
  ("choice", .CHOICE),
  ("continue", .CONTINUE),
  ("=>", .DBLARROW),
  ("default", .DEFAULT),
  ("else", .ELSE),
  ("==", .EQUAL_EQUAL),
  ("false", .FALSE),
  ("fn", .FN),
  ("fnty", .FNTY),
  ("if", .IF),
  ("int", .INT),
  ("match", .MATCH),
  ("not", .NOT),
  ("or", .OR),
  ("return", .RETURN),
  ("struct", .STRUCT),
  ("true", .TRUE),
  ("type", .TYPE),
  ("var", .VAR),
  ("while", .WHILE),
  ("=", .EQUAL),
  ("-", .MINUS),
  ("+", .PLUS),
//  ("*", .STAR),
//  ("/", .SLASH),
  ("(", .LEFT_PARENTHESIS),
  (")", .RIGHT_PARENTHESIS),
  ("{", .LEFT_CURLY_BRACE),
  ("}", .RIGHT_CURLY_BRACE),
  ("[", .LEFT_SQUARE_BRACKET),
  ("]", .RIGHT_SQUARE_BRACKET),
  (".", .PERIOD),
  (",", .COMMA),
  (";", .SEMICOLON),
  (":", .COLON),
]

let patterns: [PatternSpec] = [
  (#"[A-Za-z_][A-Za-z0-9_]*"#, .Identifier),
  (#"[0-9]+"#, .Integer_literal),
  (#"//[^\n]*"#, nil),
  (#"[ \t\r\n]+"#, nil),
]

// A single regex pattern with alternatives for all the keywords
let keywordPattern = keywords.lazy.map { k in k.literalText }
  .sorted { $0.count > $1.count }
  .lazy.map { NSRegularExpression.escapedPattern(for: $0) }
  .joined(separator: "|")

// A mapping from matched keyword content to corresponding TokenCode.
let tokenKindForKeyword = Dictionary(uniqueKeysWithValues: keywords)

let allPatterns = [(keywordPattern, nil)] + patterns
let matchers = allPatterns.map {
  try! (matcher: NSRegularExpression(pattern: $0, options: []), nonterminal: $1)
}

extension String {
  subscript(r: NSRange) -> Substring {
    let start = utf16.index(
      startIndex, offsetBy: r.location, limitedBy: endIndex) ?? endIndex
    let end = utf16.index(start, offsetBy: r.length, limitedBy: endIndex)
      ?? endIndex
    return self[start..<end]
  }
}

public struct Token: Hashable {
  let kind: TokenKind
  let text: String
}

public struct Tokens: Sequence {
  public init(in sourceText: String, from sourceFileName: String) {
    self.sourceText = sourceText
    self.sourceFileName = sourceFileName
  }
  
  public func makeIterator() -> Iterator {
    .init(over: sourceText, from: sourceFileName)
  }

  /// The token streams's iteration state and producer.
  public struct Iterator: IteratorProtocol {
    public init(over sourceText: String, from sourceFileName: String) {
      self.sourceText = sourceText
      textPosition = sourceText.startIndex
      sourceUTF16Length = sourceText.utf16.count
      self.sourceFileName = sourceFileName
    }

    public mutating func next() -> AST<Token>? {
      while true {
        let remainingUTF16 = NSRange(
          location: utf16Offset, length: sourceUTF16Length - utf16Offset)
        
        if remainingUTF16.length == 0 { return nil }
        
        let matchUTF16Lengths = matchers.lazy.map { [sourceText] in
          $0.matcher.firstMatch(
            in: sourceText, options: .anchored, range: remainingUTF16
          )?.range.length ?? 0
        }

        let (bestMatchIndex, bestMatchUTF16Length)
          = matchUTF16Lengths.enumerated().max(by: { $0.1 < $1.1 })!

        let tokenStart = textPosition
        
        textPosition = bestMatchUTF16Length == 0
          ? sourceText[textPosition...].dropFirst().startIndex
          : sourceText[textPosition...].utf16
            .dropFirst(bestMatchUTF16Length).startIndex

        let tokenText = sourceText[tokenStart..<textPosition]
        utf16Offset += tokenText.utf16.count

        let tokenLocationStart = sourceFilePosition
        let tokenLines = tokenText.split(
          omittingEmptySubsequences: false, whereSeparator: \.isNewline)
        let newlineCount = tokenLines.count - 1
        
        sourceFilePosition.line += newlineCount
        sourceFilePosition.column
          = (newlineCount == 0 ? sourceFilePosition.column : 1)
          + tokenLines.last!.count

        if let matchedKind = bestMatchUTF16Length == 0 ? .ILLEGAL_CHARACTER
             : bestMatchIndex == 0 ? tokenKindForKeyword[String(tokenText)]
             : matchers[bestMatchIndex].nonterminal
        {
          return (
            .init(kind: TokenKind(matchedKind), text: String(tokenText)),
            location: SourceLocation(
              fileName: sourceFileName,
              span: tokenLocationStart..<sourceFilePosition))
        }
      }
    }
    private let sourceText: String
    private let sourceFileName: String
    private var sourceFilePosition = PositionInSourceFile(line: 1, column: 1)
    private var textPosition: String.Index
    private var utf16Offset: Int = 0
    private let sourceUTF16Length: Int
  }
  private let sourceText: String
  private let sourceFileName: String
}
