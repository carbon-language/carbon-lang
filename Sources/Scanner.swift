// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Foundation // for regular expressions

// ===== Lexical Analyzer Specification ======
//
// This is the region of the file to edit when changing the tokens to be
// recognized.

/// A mapping from literal strings to be recognized to the corresponding token
/// ID.
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
  "Type": .TYPE,
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

/// A mapping from regular expression pattern to either a coresponding token ID,
/// or `nil` if the pattern is to be discarded (e.g. for whitespace).
fileprivate let patterns: [String: TokenID?] = [
  #"[A-Za-z_][A-Za-z0-9_]*"#: .Identifier,
  #"[0-9]+"#: .Integer_literal,

  // "//" followed by any number of non-newlines (See
  // https://unicode-org.github.io/icu/userguide/strings/regexp.html
  //   #regular-expression-metacharacters
  // and https://www.unicode.org/reports/tr44/#BC_Values_Table).
  #"//\P{Bidi_Class=B}*"#: nil, // 1-line comment

  #"\s+"#: nil // whitespace
]

// ===== Lexical Analyzer Implementation ======
//
// This is the region of the file to edit if you should happen to find a bug in
// the way tokens are recognized.

/// A single regex pattern with alternatives for all the keywords
fileprivate let keywordPattern = keywords.keys
  // Put the longest ones first because regexps match alternatives eagerly.
  .sorted { $0.count > $1.count }
  .lazy.map { NSRegularExpression.escapedPattern(for: $0) }
  .joined(separator: "|")

/// The keywords pattern, followed by the user-specified patterns.
///
/// The keywords pattern is given the `nil` token ID, but the first element in
/// this list is treated specially.
fileprivate let allPatterns = [(keywordPattern, nil)] + patterns

/// A version of allPatterns with the patterns compiled to regexp matchers.
fileprivate let matchers = allPatterns.map {
  try! (matcher: NSRegularExpression(pattern: $0, options: []), tokenID: $1)
}

extension String {
  /// Accesses the slice of `self` specified by the given range of UTF16
  /// offsets.
  fileprivate subscript(r: NSRange) -> Substring {
    let start = utf16.index(
      startIndex, offsetBy: r.location, limitedBy: endIndex) ?? endIndex
    let end = utf16.index(start, offsetBy: r.length, limitedBy: endIndex)
      ?? endIndex
    return self[start..<end]
  }
}

/// The sequence of tokens in a source file.
///
/// An .ILLEGAL_CHARACTER token is produced for each character that isn't
/// otherwise recognized.
struct Tokens: Sequence {
  /// Creates an instance that extracts the tokens from `sourceText`, labeling
  /// each as having come from the given source file.
  init(in sourceText: String, from sourceFileName: String) {
    self.sourceText = sourceText
    self.sourceFileName = sourceFileName
  }

  /// Returns a new iteration state.
  func makeIterator() -> Iterator {
    .init(over: sourceText, from: sourceFileName)
  }

  /// The token stream's iteration state and element producer.
  struct Iterator: IteratorProtocol {
    /// Creates an instance producing the tokens from `sourceText`, labeling
    /// each as having come from the given source file.
    init(over sourceText: String, from sourceFileName: String) {
      self.sourceText = sourceText
      textPosition = sourceText.startIndex
      sourceUTF16Length = sourceText.utf16.count
      self.sourceFileName = sourceFileName
    }

    /// Returns the next token in the source, or `nil` if the source is
    /// exhausted.
    mutating func next() -> Token? {
      // Repeat until a non-ignored pattern is matched.
      while utf16Offset < sourceUTF16Length {
        // NSRegularExpression matching region
        let remainingUTF16 = NSRange(
          location: utf16Offset, length: sourceUTF16Length - utf16Offset)

        // UTF16 lengths matched by each matcher
        let matchUTF16Lengths = matchers.lazy.map { [sourceText] in
          $0.matcher.firstMatch(
            in: sourceText, options: .anchored, range: remainingUTF16
          )?.range.length ?? 0
        }

        // Choose the longest matcher.
        let (bestMatchIndex, bestMatchUTF16Length)
          = matchUTF16Lengths.enumerated().max(by: { $0.element < $1.element })!

        let tokenStart = textPosition
        let remainingText = sourceText[tokenStart...]

        // Advance past the recognized text, or the first character if nothing
        // matched.
        textPosition = bestMatchUTF16Length == 0
          ? remainingText.dropFirst(1).startIndex
          : remainingText.utf16.dropFirst(bestMatchUTF16Length).startIndex

        let tokenText = remainingText[..<textPosition]
        utf16Offset += tokenText.utf16.count

        let tokenRegionStart = sourcePosition

        // Adjust human-readable source position
        let tokenLines = tokenText.split(
          omittingEmptySubsequences: false, whereSeparator: \.isNewline)
        let newlineCount = tokenLines.count - 1
        sourcePosition.line += newlineCount
        sourcePosition.column
          = (newlineCount == 0 ? sourcePosition.column : 1)
          + tokenLines.last!.count

        let matchedText = String(tokenText)
        if let matchedID = bestMatchUTF16Length == 0 ? .ILLEGAL_CHARACTER
             : bestMatchIndex == 0 ? keywords[matchedText]
             : matchers[bestMatchIndex].tokenID
        {
          return Token(
            matchedID, matchedText,
            ASTSite(
              devaluing: SourceRegion(
                fileName: sourceFileName, tokenRegionStart..<sourcePosition)))
        }
      }
      return nil
    }

    /// The complete text being matched
    private let sourceText: String
    /// The name of the file embedded in each token's source region.
    private let sourceFileName: String
    /// The number of UTF-16 code units in `sourceText`.
    private let sourceUTF16Length: Int

    /// Where scanning for the next token will resume (human-readable form).
    private var sourcePosition = SourcePosition.start
    /// Where scanning for the next token will resume (string form).
    private var textPosition: String.Index
    /// Where scanning for the next token will resume (NSRegularExpression form).
    private var utf16Offset: Int = 0
  }

  /// The complete text from which `self`'s tokens will be derived.
  private let sourceText: String
    /// The name of the file embedded in each token's source region.
  private let sourceFileName: String
}
