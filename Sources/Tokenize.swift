import Foundation // for regular expressions

/// A token as recognized by the lexical analyzer.
///
/// - Note: TokenID does not include the token's text, except inasmuch as it may
///   be implied (e.g. for keywords, which only match one string.)
typealias TokenID = CarbonParser.CitronTokenCode

/// A mapping from literal strings that can be matched, to the corresponding
/// token ID.
fileprivate let keywords: [String: TokenID] = [
  "and": .AND,
  "->": .ARROW,
  "auto": .AUTO,
  "bool": .BOOL,
  "break": .BREAK,
  "case": .CASE,
  "choice": .CHOICE,
  "continue": .CONTINUE,
  "=>": .DBLARROW,
  "default": .DEFAULT,
  "else": .ELSE,
  "==": .EQUAL_EQUAL,
  "false": .FALSE,
  "fn": .FN,
  "fnty": .FNTY,
  "if": .IF,
  "Int": .INT,
  "match": .MATCH,
  "not": .NOT,
  "or": .OR,
  "return": .RETURN,
  "struct": .STRUCT,
  "true": .TRUE,
  "type": .TYPE,
  "var": .VAR,
  "while": .WHILE,
  "=": .EQUAL,
  "-": .MINUS,
  "+": .PLUS,
//  "*": .STAR,
//  "/": .SLASH,
  "(": .LEFT_PARENTHESIS,
  ")": .RIGHT_PARENTHESIS,
  "{": .LEFT_CURLY_BRACE,
  "}": .RIGHT_CURLY_BRACE,
  "[": .LEFT_SQUARE_BRACKET,
  "]": .RIGHT_SQUARE_BRACKET,
  ".": .PERIOD,
  ",": .COMMA,
  ";": .SEMICOLON,
  ":": .COLON
]

let patterns: [(pattern: String, token: TokenID?)] = [
  (#"[A-Za-z_][A-Za-z0-9_]*"#, .Identifier),
  (#"[0-9]+"#, .Integer_literal),
  // 1-line comment: "//" followed by any number of non-newlines (See
  // https://unicode-org.github.io/icu/userguide/strings/regexp.html
  //   #regular-expression-metacharacters
  // and https://www.unicode.org/reports/tr44/#BC_Values_Table).
  (#"//\P{Bidi_Class=B}*"#, nil),
  (#"\s+"#, nil),
]

// A single regex pattern with alternatives for all the keywords
let keywordPattern = keywords.keys
  .sorted { $0.count > $1.count }
  .lazy.map { NSRegularExpression.escapedPattern(for: $0) }
  .joined(separator: "|")

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

struct Token: Hashable {
  init(_ kind: TokenID, _ text: String) {
    self.kind = kind
    self.text = text
  }
  let kind: TokenID
  let text: String
}

extension Token: CustomStringConvertible {
  var description: String {
    "Token(.\(kind), \(String(reflecting: text)))"
  }
}

struct Tokens: Sequence {
  init(in sourceText: String, from sourceFileName: String) {
    self.sourceText = sourceText
    self.sourceFileName = sourceFileName
  }
  
  func makeIterator() -> Iterator {
    .init(over: sourceText, from: sourceFileName)
  }

  /// The token streams's iteration state and producer.
  struct Iterator: IteratorProtocol {
    init(over sourceText: String, from sourceFileName: String) {
      self.sourceText = sourceText
      textPosition = sourceText.startIndex
      sourceUTF16Length = sourceText.utf16.count
      self.sourceFileName = sourceFileName
    }

    mutating func next() -> AST<Token>? {
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

        let tokenLocationStart = sourcePosition
        let tokenLines = tokenText.split(
          omittingEmptySubsequences: false, whereSeparator: \.isNewline)
        let newlineCount = tokenLines.count - 1
        
        sourcePosition.line += newlineCount
        sourcePosition.column
          = (newlineCount == 0 ? sourcePosition.column : 1)
          + tokenLines.last!.count

        let text = String(tokenText)
        if let matchedKind = bestMatchUTF16Length == 0 ? .ILLEGAL_CHARACTER
             : bestMatchIndex == 0 ? keywords[text]
             : matchers[bestMatchIndex].nonterminal
        {
          return AST(
            .init(matchedKind, text),
            SourceRegion(
              fileName: sourceFileName,
              tokenLocationStart..<sourcePosition))
        }
      }
    }
    private let sourceText: String
    private let sourceFileName: String
    private var sourcePosition = SourcePosition.start
    private var textPosition: String.Index
    private var utf16Offset: Int = 0
    private let sourceUTF16Length: Int
  }
  private let sourceText: String
  private let sourceFileName: String
}
