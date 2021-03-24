/*

Lemon: LALR(1) parser generator that generates a parser in C

    Author disclaimed copyright

    Public domain code.

Citron: Modifications to Lemon to generate a parser in Swift

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


// This file is part of the Citron parser generator.
//
// This file defines the CitronParser protocol. Citron shall
// auto-generate a class conforming to this protocol based on the input
// grammar.
//
// The CitronParser protocol defined below is compatible with Swift code
// generated using Citron version 2.x.

protocol CitronParser: class {

    // Types

    // Symbol number, state number and rule number are typically
    // mapped to UInt8. However, if there are more than 256 symbols, states
    // or rules respectively, they will get mapped to bigger integer types.
    associatedtype CitronSymbolNumber: BinaryInteger // YYCODETYPE in lemon
    associatedtype CitronStateNumber: BinaryInteger
    associatedtype CitronRuleNumber: BinaryInteger

    // Token code: An enum representing the terminals. The raw value shall
    // be equal to the symbol code representing the terminal.
    associatedtype CitronTokenCode: RawRepresentable, Hashable where CitronTokenCode.RawValue == CitronSymbolNumber

    // Non-terminal code: An enum representing the terminals. The raw value shall
    // be equal to the symbol code representing the terminal.
    associatedtype CitronNonTerminalCode: RawRepresentable, Hashable where CitronNonTerminalCode.RawValue == CitronSymbolNumber

    // Symbol code: An enum representing
    //   - a terminal, or
    //   - a non-terminal, or
    //   - an end-of-input symbol
    associatedtype CitronSymbolCode: RawRepresentable, Equatable where CitronSymbolCode.RawValue == CitronSymbolNumber

    // Token: The type representing a terminal, defined using %token_type in the grammar.
    // ParseTOKENTYPE in lemon.
    associatedtype CitronToken

    // Symbol: An enum type representing any terminal or non-terminal symbol.
    // YYMINORTYPE in lemon.
    associatedtype CitronSymbol

    // Result: The type representing the start symbol of the grammar
    associatedtype CitronResult

    // Counts

    var yyNumberOfSymbols: Int { get } // YYNOCODE in lemon
    var yyNumberOfStates: Int { get } // YYNSTATE in lemon

    // Action tables

    // The action (CitronParsingAction) is applicable only if
    // the look ahead symbol (CitronSymbolNumber) matches
    var yyLookaheadAction: [(CitronSymbolNumber, CitronParsingAction)] { get } // yy_action + yy_lookahead in lemon

    var yyShiftUseDefault: Int { get } // YY_SHIFT_USE_DFLT in lemon
    var yyShiftOffsetMin: Int { get } // YY_SHIFT_MIN in lemon
    var yyShiftOffsetMax: Int { get } // YY_SHIFT_MAX in lemon
    var yyShiftOffset: [Int] { get } // yy_shift_ofst in lemon

    var yyReduceUseDefault: Int { get } // YY_REDUCE_USE_DFLT in lemon
    var yyReduceOffsetMin: Int { get } // YY_REDUCE_MIN in lemon
    var yyReduceOffsetMax: Int { get } // YY_REDUCE_MAX in lemon
    var yyReduceOffset: [Int] { get } // yy_reduce_ofst in lemon

    var yyDefaultAction: [CitronParsingAction] { get } // yy_default in lemon

    // Fallback

    var yyHasFallback: Bool { get } // YYFALLBACK in lemon
    var yyFallback: [CitronSymbolNumber] { get } // yyFallback in lemon

    // Wildcard

    var yyWildcard: CitronSymbolNumber? { get }

    // Rules

    var yyRuleInfo: [(lhs: CitronSymbolNumber, nrhs: UInt)] { get }

    // Stack

    var yyStack: [(stateOrRule: CitronStateOrRule, symbolCode: CitronSymbolNumber,
        symbol: CitronSymbol)] { get set }
    var maxStackSize: Int? { get set }
    var maxAttainedStackSize: Int { get set }

    // Tracing

    var isTracingEnabled: Bool { get set }
    var isTracingPrintsSymbolValues: Bool { get set }
    var isTracingPrintsTokenValues: Bool { get set }
    var yySymbolName: [String] { get } // yyTokenName in lemon
    var yyRuleText: [String] { get } // yyRuleName in lemon

    // Functions that shall be defined in the autogenerated code

    func yyTokenToSymbol(_ token: CitronToken) -> CitronSymbol
    func yyInvokeCodeBlockForRule(ruleNumber: CitronRuleNumber) throws -> CitronSymbol
    func yyUnwrapResultFromSymbol(_ symbol: CitronSymbol) -> CitronResult

    // Error capturing

    associatedtype CitronErrorCaptureDelegate

    var errorCaptureDelegate: CitronErrorCaptureDelegate? { get set }

    var yyErrorCaptureSymbolNumbersForState: [CitronStateNumber:[CitronSymbolNumber]] { get }
    var yyCanErrorCapture: Bool { get }
    var yyErrorCaptureDirectives: [CitronSymbolNumber:(endAfter:[[CitronTokenCode]],endBefore:[CitronTokenCode])] { get }
    var yyErrorCaptureEndBeforeTokens: Set<CitronSymbolNumber> { get }
    var yyErrorCaptureEndAfterSequenceEndingTokens: Set<CitronSymbolNumber> { get }

    var yyStartSymbolNumber: CitronSymbolNumber { get }
    var yyEndStateNumber: CitronStateNumber { get }

    var yyErrorCaptureSavedError: (error: Error, isLexerError: Bool)? { get set }
    var yyErrorCaptureTokensSinceError: [(token: CitronToken, tokenCode: CitronTokenCode)] { get set }
    var yyErrorCaptureStackIndices: [Int] { get set }
    var yyErrorCaptureStartSymbolStackIndex: Int? { get set }

    var numberOfCapturedErrors: Int { get set }

    func yyShouldSaveErrorForCapturing(error: Error) -> Bool
    func yyCaptureError(on: CitronNonTerminalCode, error: Error, state: CitronErrorCaptureState) -> CitronSymbol?
    func yySymbolContent(_ symbol: CitronSymbol) -> Any

    // Error handling

    typealias UnexpectedTokenError = _CitronParserUnexpectedTokenError<CitronToken, CitronTokenCode>
    typealias UnexpectedEndOfInputError = CitronParserUnexpectedEndOfInputError
    typealias StackOverflowError = CitronParserStackOverflowError

    // Type aliases

    typealias CitronParsingAction = _CitronParsingAction<CitronStateNumber, CitronRuleNumber>
    typealias CitronStateOrRule = _CitronStateOrRule<CitronStateNumber, CitronRuleNumber>
    typealias CitronErrorCaptureResult = _CitronErrorCaptureResult<CitronNonTerminalCode, CitronSymbol>
    typealias CitronErrorCaptureState = _CitronErrorCaptureState<CitronToken, CitronTokenCode, CitronSymbolCode>
}

// Error handling

enum _CitronParserError<Token, TokenCode>: Error {
    case syntaxErrorAt(token: Token, tokenCode: TokenCode)
    case unexpectedEndOfInput
    case stackOverflow
}

protocol CitronParserError : Error { }

class _CitronParserUnexpectedTokenError<Token, TokenCode> : CitronParserError {
    let token: Token
    let tokenCode: TokenCode
    init(token: Token, tokenCode: TokenCode) {
        self.token = token
        self.tokenCode = tokenCode
    }
}

class CitronParserUnexpectedEndOfInputError : CitronParserError {
}

class CitronParserStackOverflowError : CitronParserError {
}

// Parser actions and states

enum _CitronParsingAction<StateNumber: BinaryInteger, RuleNumber: BinaryInteger> {
    case SH(StateNumber) // Shift token, then go to state <state>
    case RD(RuleNumber)  // Reduce with rule number <rule>
    case SR(RuleNumber)  // Shift token, then reduce with rule number <rule>
    case ERROR
    case ACCEPT
}

enum _CitronStateOrRule<StateNumber: BinaryInteger, RuleNumber: BinaryInteger> {
    case state(StateNumber)
    case rule(RuleNumber)
}

// Error capturing

enum _CitronErrorCaptureResult<NonTerminalCode, Symbol> {
    case notCaptured
    case capturedOnIntermediateSymbol(symbolCode: NonTerminalCode, didMatchEndBeforeClause: Bool)
    case capturedOnFinalResult(result: Symbol)
}

struct _CitronErrorCaptureState<Token, TokenCode, SymbolCode> {
    let resolvedSymbols: [(symbolCode: SymbolCode, value: Any)]
    let unclaimedTokens: [(token: Token, tokenCode: TokenCode)]
    let nextToken: (token: Token, tokenCode: TokenCode)?

    var lastResolvedSymbol: (symbolCode: SymbolCode, value: Any)? { return resolvedSymbols.last }
    var erroringToken: (token: Token, tokenCode: TokenCode)? { return (unclaimedTokens.first ?? nextToken) }
}

enum CitronErrorCaptureResponse<T> {
    case captureAs(T)
    case dontCapture
}

// Parsing interface

extension CitronParser {
    func consume(token: CitronToken, code tokenCode: CitronTokenCode) throws {
        let symbolCode = tokenCode.rawValue
        tracePrint("Input:", tokenCode: tokenCode, token: token)

        var isErrorCapturedUsingEndBeforeClause: Bool = false

        LOOP: while (!yyStack.isEmpty) {

            if (yyShouldAttemptErrorCapture()) {
                tracePrint("Error capture: Trying to capture saved error")
                let result = try yyAttemptErrorCapture(nextToken: (token: token, tokenCode: tokenCode))
                switch (result) {
                case .notCaptured:
                    tracePrint("Error capture: Failed")
                    yyErrorCaptureTokensSinceError.append((token: token, tokenCode: tokenCode))
                    return
                case .capturedOnIntermediateSymbol(_, let didMatchEndBeforeClause):
                    tracePrint("Error capture: Succeeded")
                    yyErrorCaptureSavedError = nil
                    yyErrorCaptureTokensSinceError = []
                    isErrorCapturedUsingEndBeforeClause = didMatchEndBeforeClause
                case .capturedOnFinalResult(_):
                    fatalError() // Can happen only in endParsing()
                }
            }

            let action = yyFindShiftAction(lookAhead: symbolCode)
            switch (action) {
            case .SH(let s):
                try yyShift(state: s, symbolCode: symbolCode, token: token)
                break LOOP
            case .SR(let r):
                try yyShiftReduce(rule: r, symbolCode: symbolCode, token: token)
                break LOOP
            case .RD(let r):
                let resultSymbol = try yyReduce(rule: r)
                assert(resultSymbol == nil) // Can be non-nil only in endParsing()
                continue LOOP
            case .ERROR:
                if (isErrorCapturedUsingEndBeforeClause) {
                    tracePrint("Error capture: Capture using end_before clause is immediately followed by an error for the same token, indicating that the endBefore clause is inconsistent with the grammar.")
                    // If we save this error and then try to capture it with the same lookAhead,
                    // we'll cause an infinite loop. So, we just ignore this error.
                    yyErrorCaptureTokensSinceError.append((token: token, tokenCode: tokenCode))
                    return
                }
                try throwOrSave(UnexpectedTokenError(token: token, tokenCode: tokenCode))
                continue LOOP // if error is saved, not thrown, we should attempt to capture it right away
            default:
                fatalError("Unexpected action")
            }
        }
        traceStack()
    }

    func endParsing() throws -> CitronResult {
        tracePrint("End of input")

        var errorCaptureExcludeSymbols: Set<CitronNonTerminalCode> = []

        LOOP: while (!yyStack.isEmpty) {

            if (yyShouldAttemptErrorCapture()) {
                tracePrint("Error capture: Trying to capture saved error")
                let result = try yyAttemptErrorCapture(nextToken: nil, excludeSymbols: errorCaptureExcludeSymbols)
                switch (result) {
                case .notCaptured:
                    tracePrint("Error capture: Failed")
                    guard let savedError = yyErrorCaptureSavedError else { fatalError() }
                    tracePrint("Error capture: At end of input, throwing saved uncaptured error")
                    throw savedError.error
                case .capturedOnIntermediateSymbol(let symbolCode, _):
                    tracePrint("Error capture: Succeeded")
                    yyErrorCaptureSavedError = nil
                    yyErrorCaptureTokensSinceError = []
                    // Capturing again on the same symbol will
                    // cause an infinite loop, so we exclude it
                    // in further captures.
                    errorCaptureExcludeSymbols.insert(symbolCode)
                case .capturedOnFinalResult(let resultSymbol):
                    tracePrint("Error capture: Succeeded")
                    yyErrorCaptureSavedError = nil
                    yyErrorCaptureTokensSinceError = []
                    return yyUnwrapResultFromSymbol(resultSymbol)
                }
            }

            let action = yyFindShiftAction(lookAhead: 0)
            switch (action) {
            case .RD(let r):
                let resultSymbol = try yyReduce(rule: r)
                if let resultSymbol = resultSymbol {
                    tracePrint("Parse successful")
                    return yyUnwrapResultFromSymbol(resultSymbol)
                }
                continue LOOP
            case .ERROR:
                try throwOrSave(UnexpectedEndOfInputError())
                continue LOOP // if error is saved, not thrown, we should attempt to capture it right away
            default:
                fatalError("Unexpected action")
            }
        }
        fatalError("Unexpected stack underflow")
    }

    func reset() {
        tracePrint("Resetting the parser")
        while (yyStack.count > 1) {
            yyPop()
        }
    }

    func consume(lexerError: Error) throws {
        tracePrint("Input: Lexer error")
        if (yyErrorCaptureSavedError != nil) {
            tracePrint("Ignoring this lexer error as there is already a saved error")
            // We'll ignore this lexer error, assuming that this is part of
            // the saved error that we're trying to capture.
            return
        }

        while (!yyStack.isEmpty) {
            // In case the top of the stack contains a rule,
            // we should first resolve the rule
            if case .rule(let r) = yyStack.last!.stateOrRule {
                let resultSymbol = try yyReduce(rule: r)
                precondition(resultSymbol == nil)
            } else {
                break
            }
        }

        try throwOrSave(lexerError, isLexerError: true)
    }
}

// Private methods for error capturing

private extension CitronParser {
    func throwOrSave(_ error: Error, isLexerError: Bool = false) throws {
        guard (self.yyCanErrorCapture) else { throw error }
        let saved = saveErrorForCapturingLater(error: error, isLexerError: isLexerError)
        if (!saved) { throw error }
    }

    func saveErrorForCapturingLater(error: Error, isLexerError: Bool) -> Bool {
        // Returns true if saved, false if not saved
        var canCapture: Bool = false
        var stackIndices: [Int] = []
        var startSymbolStackIndex: Int? = nil
        for i in stride(from: yyStack.count - 1, through: 0, by: -1) {
            let stackEntry = yyStack[i]
            switch(stackEntry.stateOrRule) {
            case .state(let s):
                if let symbolCodes = yyErrorCaptureSymbolNumbersForState[s] {
                    canCapture = true
                    stackIndices.append(i)
                    if (startSymbolStackIndex == nil && symbolCodes.contains(yyStartSymbolNumber)) {
                        startSymbolStackIndex = i
                    }
                }
            default:
                break
            }
        }
        if (canCapture && yyShouldSaveErrorForCapturing(error: error)) {
            tracePrint("Error capture: Saved error for later capturing:", "\(error)")
            // Save this error for either capturing or throwing later
            self.yyErrorCaptureSavedError = (error: error, isLexerError: isLexerError)
            self.yyErrorCaptureTokensSinceError = []
            // Save some info for determining when to capture the error
            self.yyErrorCaptureStackIndices = stackIndices
            self.yyErrorCaptureStartSymbolStackIndex = startSymbolStackIndex
            return true
        } else {
            self.yyErrorCaptureSavedError = nil
            self.yyErrorCaptureTokensSinceError = []
            self.yyErrorCaptureStackIndices = []
            self.yyErrorCaptureStartSymbolStackIndex = nil
            return false
        }
    }

    func yyShouldAttemptErrorCapture() -> Bool {
        return (self.yyErrorCaptureSavedError != nil)
    }

    func yyAttemptErrorCapture(nextToken: (token: CitronToken, tokenCode: CitronTokenCode)?, excludeSymbols: Set<CitronNonTerminalCode> = []) throws -> CitronErrorCaptureResult {
        guard let savedError = yyErrorCaptureSavedError else {
            fatalError("No error saved for capturing")
        }

        guard let info = stackUnwindInfoForErrorCapture(lookAhead: nextToken?.tokenCode, excludeSymbols: excludeSymbols) else {
            tracePrint("Error capture: No match in the stack for the current sequence of tokens")
            return .notCaptured
        }

        assert(info.stackIndex < yyStack.count)
        tracePrint("Error capture: Found match at stack index", "\(info.stackIndex)")
        tracePrint("Error capture: Found match on symbol", quoted: symbolNameFor(code: info.symbolCode))

        let stackEntry = yyStack[info.stackIndex]
        guard case .state(_) = stackEntry.stateOrRule else {
            fatalError("Expecting state got rule while attempting error capture")
        }
        let resolvedSymbols: [(symbolCode: CitronSymbolCode, value: Any)] = yyStack[(info.stackIndex + 1) ..< yyStack.count].map {
            (symbolCode: CitronSymbolCode(rawValue: $0.symbolCode)!, value: yySymbolContent($0.symbol))
        }
        let unclaimedTokens = yyErrorCaptureTokensSinceError

        if (resolvedSymbols.isEmpty && unclaimedTokens.isEmpty && savedError.isLexerError == false) {
            tracePrint("Error capture: Cannot capture error on an empty symbol")
            return .notCaptured
        }

        let errorCaptureState = CitronErrorCaptureState(
            resolvedSymbols: resolvedSymbols,
            unclaimedTokens: unclaimedTokens,
            nextToken: nextToken
        )

        guard let errorCapturedSymbol = yyCaptureError(on: info.symbolCode, error: savedError.error, state: errorCaptureState) else {
            return .notCaptured
        }

        yyPop(times: yyStack.count - info.stackIndex - 1)

        var isAccepted: Bool = false
        try yyPerformReduceAction(symbol: errorCapturedSymbol, code: info.symbolCode.rawValue, isAccepted: &isAccepted)
        self.numberOfCapturedErrors = self.numberOfCapturedErrors + 1
        if (isAccepted) {
            return .capturedOnFinalResult(result: errorCapturedSymbol)
        } else {
            return .capturedOnIntermediateSymbol(symbolCode: info.symbolCode, didMatchEndBeforeClause: info.didMatchEndBeforeClause)
        }
    }

    func stackUnwindInfoForErrorCapture(lookAhead: CitronTokenCode?, excludeSymbols: Set<CitronNonTerminalCode>)
        -> (stackIndex: Int, symbolCode: CitronNonTerminalCode, didMatchEndBeforeClause: Bool)? {

        let isAtEndOfInput: Bool = (lookAhead == nil)

        let lastSeenTokenSymbolCode: CitronSymbolNumber? = yyErrorCaptureTokensSinceError.last?.tokenCode.rawValue
        let isEndBeforeMatchPossible = ((lookAhead != nil) &&
            yyErrorCaptureEndBeforeTokens.contains(lookAhead!.rawValue))
        let isEndAfterMatchPossible = ((lastSeenTokenSymbolCode != nil) &&
            yyErrorCaptureEndAfterSequenceEndingTokens.contains(lastSeenTokenSymbolCode!))

        guard (isAtEndOfInput || isEndBeforeMatchPossible || isEndAfterMatchPossible) else { return nil }

        var stackIndices: [Int]
        if case .state(let s) = yyStack.last!.stateOrRule, s == yyEndStateNumber {
            // If we're at the end state, capture only on the start symbol
            if let startSymbolStackIndex = yyErrorCaptureStartSymbolStackIndex {
                stackIndices = [startSymbolStackIndex]
            } else {
                stackIndices = []
            }
        } else {
            stackIndices = yyErrorCaptureStackIndices
        }

        for stackIndex in stackIndices {
            let hasUnclaimedTokensOrLexerError = (!yyErrorCaptureTokensSinceError.isEmpty) || (yyErrorCaptureSavedError?.isLexerError ?? false)
            if (isAtEndOfInput && yyErrorCaptureTokensSinceError.isEmpty && (!hasUnclaimedTokensOrLexerError) && stackIndex == yyStack.count - 1) {
                // Skip matching with the top of the stack, because
                // yyAttemptErrorCapture() would reject it as a
                // capture on an empty symbol and there aren't going to
                // be any more tokens anyway to look for a better match.
                continue
            }
            var symbolNumbers: [CitronSymbolNumber] = []
            let stackEntry = yyStack[stackIndex]
            switch(stackEntry.stateOrRule) {
            case .state(let s):
                if let sc = yyErrorCaptureSymbolNumbersForState[s] {
                    symbolNumbers = sc
                }
            default:
                break
            }
            for s in symbolNumbers {
                let symbolCode = CitronNonTerminalCode(rawValue: s)! // This symbol has to be a non-terminal
                if (excludeSymbols.contains(symbolCode)) {
                    continue
                }
                if (stackIndex + 1 < yyStack.count && yyStack[stackIndex + 1].symbolCode == s) {
                    // If the next symbol on the stack is the same as this one,
                    // it probably means that the error did not happen "inside"
                    // this symbol.
                    // i.e. if the stack has:
                    //     state(0), symbol($)
                    //     ...
                    //     state(2), symbol(B)
                    //     state(3), symbol(C)
                    //     ...
                    // And state(2) looks like:
                    //     base config:    X -> B . C D
                    //     derived config: C -> . P
                    // That derived config might suggest we capture on
                    // state(2) on symbol C, but that would be wrong
                    // because C is already resolved and present on the
                    // stack.
                    continue
                }
                if (isAtEndOfInput) {
                    tracePrint("Error capture: Match at end of input for symbol", quoted: symbolNameFor(code: symbolCode))
                    return (stackIndex: stackIndex, symbolCode: symbolCode, didMatchEndBeforeClause: false)
                }
                guard let directive = yyErrorCaptureDirectives[s] else {
                    continue
                }
                if (lookAhead != nil && directive.endBefore.contains(lookAhead!)) {
                    tracePrint("Error capture: Match for endBefore clause for symbol", quoted: symbolNameFor(code: symbolCode))
                    return (stackIndex: stackIndex, symbolCode: symbolCode, didMatchEndBeforeClause: true)
                }
                for endAfterTokenSequence in directive.endAfter {
                    if (yyErrorCaptureTokensSinceError.map({ $0.tokenCode }).hasSuffix(endAfterTokenSequence)) {
                        tracePrint("Error capture: Match for endAfter clause for symbol", quoted: symbolNameFor(code: symbolCode))
                        return (stackIndex: stackIndex, symbolCode: symbolCode, didMatchEndBeforeClause: false)
                    }
                }
            }
        }
        return nil
    }
}

// Private methods

private extension CitronParser {

    func yyPush(stateOrRule: CitronStateOrRule, symbolCode: CitronSymbolNumber, symbol: CitronSymbol) throws {
        if (maxStackSize != nil && yyStack.count >= maxStackSize!) {
            // Can't grow stack anymore
            throw StackOverflowError()
        }
        yyStack.append((stateOrRule: stateOrRule, symbolCode: symbolCode, symbol: symbol))
        if (maxAttainedStackSize < yyStack.count) {
            maxAttainedStackSize = yyStack.count
        }
    }

    func yyPop() {
        let last = yyStack.popLast()
        if let last = last {
            tracePrint("Popping", quoted: symbolNameFor(symbolNumber: last.symbolCode))
        }
    }

    func yyPopAll() {
        while (!yyStack.isEmpty) {
            yyPop()
        }
    }

    func yyPop(times n: Int) {
        for _ in 0 ..< n { yyPop() }
    }

    func yyFindShiftAction(lookAhead la: CitronSymbolNumber) -> CitronParsingAction {
        guard (!yyStack.isEmpty) else { fatalError("Unexpected empty stack") }

        let state: CitronStateNumber
        switch (yyStack.last!.stateOrRule) {
        case .rule(let r):
            return .RD(r)
        case .state(let s):
            state = s
        }

        var i: Int = 0
        var lookAhead = la
        while (true) {
            assert(Int(state) < yyShiftOffset.count)
            assert(lookAhead < yyNumberOfSymbols)
            i = yyShiftOffset[Int(state)] + Int(lookAhead)

            // Check action table
            if (i >= 0 && i < yyLookaheadAction.count) {
                let (actionLookahead, action) = yyLookaheadAction[i]
                if (actionLookahead == lookAhead) {
                    return action // Pick action from action table
                }
            }

            // Check for fallback
            if let fallback = yyFallback[safe: lookAhead], fallback > 0 {
                tracePrint("Fallback:", quoted: symbolNameFor(symbolNumber: lookAhead))
                tracePrint("       =>", quoted: symbolNameFor(symbolNumber: fallback))
                precondition((yyFallback[safe: fallback] ?? -1) == 0, "Fallback loop detected")
                lookAhead = fallback
                continue
            }

            // Check for wildcard
            if let yyWildcard = yyWildcard {
                let wildcard = yyWildcard
                let j = i - Int(lookAhead) + Int(wildcard)
                let (actionLookahead, action) = yyLookaheadAction[j]
                if ((yyShiftOffsetMin + Int(wildcard) >= 0 || j >= 0) &&
                    (yyShiftOffsetMax + Int(wildcard) < yyLookaheadAction.count || j < yyLookaheadAction.count) &&
                    (actionLookahead == wildcard && lookAhead > 0)) {
                    tracePrint("Wildcard:", quoted: symbolNameFor(symbolNumber: lookAhead))
                    tracePrint("       =>", quoted: symbolNameFor(symbolNumber: wildcard))
                    return action
                }
            }

            // Pick the default action for this state.
            return yyDefaultAction[Int(state)]
        }
    }

    func yyFindReduceAction(state: CitronStateNumber, lookAhead: CitronSymbolNumber) -> CitronParsingAction {
        assert(Int(state) < yyReduceOffset.count)
        var i = yyReduceOffset[Int(state)]

        assert(i != yyReduceUseDefault)
        assert(lookAhead < yyNumberOfSymbols)

        i += Int(lookAhead)
        let (actionLookahead, action) = yyLookaheadAction[i]

        assert(i >= 0 && i < yyLookaheadAction.count)
        assert(actionLookahead == lookAhead)

        return action
    }

    func yyShift(state: CitronStateNumber, symbolCode: CitronSymbolNumber, token: CitronToken) throws {
        let symbol = yyTokenToSymbol(token)
        tracePrint("Shift: Shift", symbolNumber: symbolCode, symbol: symbol)
        tracePrint("       and go to state", "\(state)")
        try yyPush(stateOrRule: .state(state), symbolCode: symbolCode, symbol: symbol)
    }

    func yyShiftReduce(rule: CitronRuleNumber, symbolCode: CitronSymbolNumber, token: CitronToken) throws {
        let symbol = yyTokenToSymbol(token)
        tracePrint("ShiftReduce: Shift", symbolNumber: symbolCode, symbol: symbol)
        tracePrint("       and reduce with rule", "\(rule)")
        try yyPush(stateOrRule: .rule(rule), symbolCode: symbolCode, symbol: symbol)
    }

    // yyReduce: Reduces using the specified rule number.
    // If the parse is accepted, returns the result symbol, else returns nil.
    func yyReduce(rule ruleNumber: CitronRuleNumber) throws -> CitronSymbol? {
        assert(ruleNumber < yyRuleInfo.count)
        guard (!yyStack.isEmpty) else { fatalError("Unexpected empty stack") }
        tracePrint("Reducing with rule", "\(ruleNumber): \(yyRuleText[Int(ruleNumber)])")

        let resultSymbol = try yyInvokeCodeBlockForRule(ruleNumber: ruleNumber)

        let ruleInfo = yyRuleInfo[Int(ruleNumber)]
        let lhsSymbolCode = ruleInfo.lhs
        let numberOfRhsSymbols = ruleInfo.nrhs
        assert(yyStack.count > numberOfRhsSymbols)

        yyPop(times: Int(numberOfRhsSymbols))

        var isAccepted: Bool = false
        try yyPerformReduceAction(symbol: resultSymbol, code: lhsSymbolCode, isAccepted: &isAccepted)

        if (isAccepted) {
            return resultSymbol
        } else {
            return nil
        }
    }

    func yyPerformReduceAction(symbol resultSymbol: CitronSymbol, code lhsSymbolCode: CitronSymbolNumber, isAccepted: inout Bool) throws {

        guard case .state(let stateInStack) = yyStack.last!.stateOrRule else {
            fatalError("Expecting state got rule") // FIXME: Is this correct?
        }
        let action = yyFindReduceAction(state: stateInStack, lookAhead: lhsSymbolCode)

        let stateOrRule: CitronStateOrRule
        switch (action) {
        case .SH(let s): stateOrRule = .state(s)
        case .SR(_): fatalError("Unexpected shift-reduce action after a reduce")
        // There are no SHIFTREDUCE actions on nonterminals because the table
        // generator has simplified them to pure REDUCE actions
        case .RD(let r): stateOrRule = .rule(r)
        case .ERROR: fatalError("Unexpected error action after a reduce")
        // It is not possible for a REDUCE to be followed by an error.
        case .ACCEPT:
            isAccepted = true
            return
        }

        isAccepted = false

        try yyPush(stateOrRule: stateOrRule, symbolCode: lhsSymbolCode, symbol: resultSymbol)
        tracePrint("Shift: Shift", symbolNumber: lhsSymbolCode, symbol: resultSymbol)
        if (isTracingEnabled) {
            switch (stateOrRule) {
            case .state(let s):
                tracePrint("       and go to state", "\(s)")
            case .rule(let r):
                tracePrint("       and reduce with rule", "\(r)")
            }
        }
        traceStack()
    }
}

// Private helpers

private extension CitronParser {
    func tracePrint(_ msg: String) {
        if (isTracingEnabled) {
            print("\(msg)")
        }
    }

    func tracePrint(_ msg: String, tokenCode: CitronTokenCode, token: CitronToken) {
        if (isTracingEnabled) {
            print("\(msg) (tokenCode: \(tokenCode)", terminator: "")
            if (isTracingPrintsTokenValues) {
                print(", token: \"\(token)\")")
            } else {
                print(")")
            }
        }
    }

    func tracePrint(_ msg: String, symbolNumber: CitronSymbolNumber, symbol: CitronSymbol) {
        if (isTracingEnabled) {
            print("\(msg) (symbolCode: \(symbolNameFor(symbolNumber: symbolNumber))", terminator: "")
            if (isTracingPrintsSymbolValues) {
                let symbolString = String(describing: yySymbolContent(symbol))
                print(", symbol: \"\(symbolString)\")")
            } else {
                print(")")
            }
        }
    }

    func tracePrint(_ msg: String, _ closure: @autoclosure () -> String) {
        if (isTracingEnabled) {
            print("\(msg) \(closure())")
        }
    }

    func tracePrint(_ msg: String, quoted closure: @autoclosure () -> String) {
        if (isTracingEnabled) {
            print("\(msg) \"\(closure())\"")
        }
    }

    func symbolNameFor(code: CitronSymbolCode) -> String {
        return symbolNameFor(symbolNumber: code.rawValue)
    }

    func symbolNameFor(code: CitronNonTerminalCode) -> String {
        return symbolNameFor(symbolNumber: code.rawValue)
    }

    func symbolNameFor(symbolNumber i: CitronSymbolNumber) -> String {
        if (i > 0 && i < yySymbolName.count) { return yySymbolName[Int(i)] }
        return "?"
    }

    func traceStack() {
        if (isTracingEnabled) {
            print("STACK contents:")
            for (i, e) in yyStack.enumerated() {
                print("    \(i): (stateOrRule: \(e.stateOrRule)", terminator: "")
                if (e.symbolCode > 0) {
                    print(", symbolCode: \(symbolNameFor(symbolNumber: e.symbolCode))", terminator: "")
                    if (isTracingPrintsSymbolValues) {
                        let symbolString = String(describing: yySymbolContent(e.symbol))
                        print(", symbol: \"\(symbolString)\"", terminator: "")
                    }
                }
                defer {
                    print(")")
                }
            }
        }
    }
}

private extension Array {
    subscript<I: BinaryInteger>(safe i: I) -> Element? {
        get {
            let index = Int(i)
            return index < self.count ? self[index] : nil
        }
    }
}

private extension Array where Element : Equatable {
    func hasSuffix(_ suffix: Array<Element>) -> Bool {
        let start: Int = count - suffix.count
        if (start < 0) { return false }
        for (i, e) in suffix.enumerated() {
            if (self[start + i] != e) { return false }
        }
        return true
    }
}
