import Foundation

enum CitronTokenCode: UInt8 {
  case LEFT_CURLY_BRACE               =   1
  case RIGHT_CURLY_BRACE              =   2
  case COLON                          =   3
  case COMMA                          =   4
  case DBLARROW                       =   5
  case OR                             =   6
  case AND                            =   7
  case EQUAL_EQUAL                    =   8
  case NOT                            =   9
  case PLUS                           =  10
  case MINUS                          =  11
  case PERIOD                         =  12
  case ARROW                          =  13
  case LEFT_PARENTHESIS               =  14
  case RIGHT_PARENTHESIS              =  15
  case LEFT_SQUARE_BRACKET            =  16
  case RIGHT_SQUARE_BRACKET           =  17
  case IDENTIFIER                     =  18
  case Identifier                     =  19
  case Integer_literal                =  20
  case TRUE                           =  21
  case FALSE                          =  22
  case INT                            =  23
  case BOOL                           =  24
  case TYPE                           =  25
  case AUTO                           =  26
  case FNTY                           =  27
  case EQUAL                          =  28
  case CASE                           =  29
  case DEFAULT                        =  30
  case SEMICOLON                      =  31
  case VAR                            =  32
  case IF                             =  33
  case WHILE                          =  34
  case BREAK                          =  35
  case CONTINUE                       =  36
  case RETURN                         =  37
  case MATCH                          =  38
  case ELSE                           =  39
  case FN                             =  40
  case STRUCT                         =  41
  case CHOICE                         =  42
  case ILLEGAL_CHARACTER              =  43
}

// typealias TokenKind = CarbonParser.CitronTokenCode
typealias TokenKind = CitronTokenCode
typealias KeywordSpec = (literalText: String, token: TokenKind)
typealias PatternSpec = (pattern: String, token: TokenKind?)
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

// A single regex pattern for all the keywords
let keywordPattern = keywords.lazy.map { k in k.literalText }
  .sorted { $0.count > $1.count }
  .lazy.map { NSRegularExpression.escapedPattern(for: $0) }
  .joined(separator: "|")

// Mapping from matched keyword content to corresponding TokenKind.
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

struct Tokens: Sequence {
  public init(in source: String) {
    self.source = source
  }
  
  public func makeIterator() -> Iterator { .init(over: source) }
  
  public struct Iterator: IteratorProtocol {
    public init(over source: String) {
      self.source = source
      position = source.startIndex
      utf16Offset = 0
      utf16Length = source.utf16.count
    }

    typealias Element = (
      kind: CitronTokenCode, content: Substring)//, location: RangeOfSourceFile)
    
    public mutating func next() -> Element? {
      while true {
        let remainingUTF16 = NSRange(
          location: utf16Offset, length: utf16Length - utf16Offset)
        
        if remainingUTF16.length == 0 { return nil }
        
        let matchUTF16Lengths = matchers.lazy.map { [source] in
          $0.matcher.firstMatch(
            in: source, options: .anchored, range: remainingUTF16
          )?.range.length ?? 0
        }

        let (bestMatchIndex, bestMatchUTF16Length)
          = matchUTF16Lengths.enumerated().max(by: { $0.1 < $1.1 })!

        let nextPosition = bestMatchUTF16Length == 0
          ? source[position...].dropFirst().startIndex
          : source[position...].utf16.dropFirst(bestMatchUTF16Length).startIndex

        let tokenText = source[position..<nextPosition]
        position = nextPosition
        utf16Offset += tokenText.utf16.count

        if let matchedKind = bestMatchUTF16Length == 0 ? .ILLEGAL_CHARACTER
             : bestMatchIndex == 0 ? tokenKindForKeyword[String(tokenText)]
             : matchers[bestMatchIndex].nonterminal
        {
          return (kind: matchedKind, content: tokenText)
        }
      }
    }
    private let source: String
    private var position: String.Index
    private var utf16Offset: Int
    private let utf16Length: Int
  }
  let source: String
}

let program0 =
  """
  an and andey 999 // burp
  ->=-automobile->otto auto bool break case choice continue=>
  default/
  // excuse me
  else==false fn fnty if\t\tint match not or return struct true type var while
  = - +( ){}[]a.b,;:
  """

//let program = "an and"
for t in Tokens(in: program0) {
  print(t)
}
