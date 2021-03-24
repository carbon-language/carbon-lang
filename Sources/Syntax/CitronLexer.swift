/*
    Copyright (C) 2017 Roopesh Chander <roop@roopc.net>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

import Foundation

typealias CitronLexerPosition = (tokenPosition: String.Index, linePosition: String.Index, lineNumber: Int)

class CitronLexer<TokenData> {
    typealias Action = (TokenData) throws -> Void
    typealias ErrorAction = (CitronLexerError) throws -> Void
    enum LexingRule {
        case string(String, TokenData?)
        case regex(NSRegularExpression, (String) -> TokenData?)
        case regexPattern(String, (String) -> TokenData?)
    }
    let rules: [LexingRule]

    var currentPosition: CitronLexerPosition

    init(rules: [LexingRule]) {
        self.rules = rules.map { rule in
            // Convert .regexPattern values to equivalent .regex values
            switch (rule) {
            case .regexPattern(let pattern, let handler):
                return .regex(try! NSRegularExpression(pattern: pattern), handler)
            default:
                return rule
            }
        }
        currentPosition = (tokenPosition: String.Index(encodedOffset: 0), linePosition: String.Index(encodedOffset: 0), lineNumber: 0)
    }

    func tokenize(_ string: String, onFound: Action) throws {
        try tokenize(string, onFound: onFound, onError: nil)
    }

    func tokenize(_ string: String, onFound: Action, onError: ErrorAction?) throws {
        currentPosition = (tokenPosition: string.startIndex, linePosition: string.startIndex, lineNumber: 1)
        var errorStartPosition: CitronLexerPosition? = nil
        while (currentPosition.tokenPosition < string.endIndex) {
            var matched = false
            for rule in rules {
                switch (rule) {
                case .string(let ruleString, let tokenData):
                    if (string.suffix(from: currentPosition.tokenPosition).hasPrefix(ruleString)) {
                        if let errorStartPosition = errorStartPosition {
                            try onError?(CitronLexerError.noMatchingRuleAt(errorPosition: errorStartPosition))
                        }
                        if let tokenData = tokenData {
                            try onFound(tokenData)
                        }
                        currentPosition = lexerPosition(in: string, advancedFrom: currentPosition, by: ruleString.count)
                        errorStartPosition = nil
                        matched = true
                    }
                case .regex(let ruleRegex, let handler):
                    let result = ruleRegex.firstMatch(in: string, options: .anchored, range:
                        NSRange(
                            location: string.prefix(upTo: currentPosition.tokenPosition).utf16.count,
                            length: string.suffix(from: currentPosition.tokenPosition).utf16.count)
                    )
                    if let matchingRange = result?.range {
                        let start = string.utf16.index(string.utf16.startIndex, offsetBy: matchingRange.lowerBound)
                        let end = string.utf16.index(string.utf16.startIndex, offsetBy: matchingRange.upperBound)
                        if let matchingString = String(string.utf16[start..<end]) {
                            if let errorStartPosition = errorStartPosition {
                                try onError?(CitronLexerError.noMatchingRuleAt(errorPosition: errorStartPosition))
                            }
                            if let tokenData = handler(matchingString) {
                                try onFound(tokenData)
                            }
                            currentPosition = lexerPosition(in: string, advancedFrom: currentPosition, by: matchingString.count)
                            errorStartPosition = nil
                            matched = true

                        }
                    }
                default:
                    fatalError("Internal error")
                }
                if (matched) {
                    break
                }
            }
            if (!matched) {
                if (onError == nil) {
                    throw CitronLexerError.noMatchingRuleAt(errorPosition: currentPosition)
                } else {
                    if (errorStartPosition == nil) {
                        errorStartPosition = currentPosition
                    }
                    currentPosition = lexerPosition(in: string, advancedFrom: currentPosition, by: 1)
                }
            }
        }
        if let errorStartPosition = errorStartPosition {
            try onError?(CitronLexerError.noMatchingRuleAt(errorPosition: errorStartPosition))
        }
    }
}

enum CitronLexerError: Error {
    case noMatchingRuleAt(errorPosition: CitronLexerPosition)
}

private extension CitronLexer {
    func lexerPosition(in str: String, advancedFrom from: CitronLexerPosition, by offset: Int) -> CitronLexerPosition {
         let tokenPosition = str.index(from.tokenPosition, offsetBy: offset)
         var linePosition = from.linePosition
         var lineNumber = from.lineNumber
         var index = from.tokenPosition
         while (index < tokenPosition) {
            if (str[index] == "\n") {
                lineNumber = lineNumber + 1
                linePosition = str.index(after: index)
            }
            index = str.index(after: index)
         }
         return (tokenPosition: tokenPosition, linePosition: linePosition, lineNumber: lineNumber)
    }
}
