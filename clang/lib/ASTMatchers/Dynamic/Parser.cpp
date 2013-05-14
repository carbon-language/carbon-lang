//===--- Parser.cpp - Matcher expression parser -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Recursive parser implementation for the matcher expression grammar.
///
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

#include "clang/ASTMatchers/Dynamic/Parser.h"
#include "clang/ASTMatchers/Dynamic/Registry.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/Twine.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {

/// \brief Simple structure to hold information for one token from the parser.
struct Parser::TokenInfo {
  /// \brief Different possible tokens.
  enum TokenKind {
    TK_Eof = 0,
    TK_OpenParen = 1,
    TK_CloseParen = 2,
    TK_Comma = 3,
    TK_Literal = 4,
    TK_Ident = 5,
    TK_InvalidChar = 6,
    TK_Error = 7
  };

  TokenInfo() : Text(), Kind(TK_Eof), Range(), Value() {}

  StringRef Text;
  TokenKind Kind;
  SourceRange Range;
  VariantValue Value;
};

/// \brief Simple tokenizer for the parser.
class Parser::CodeTokenizer {
public:
  explicit CodeTokenizer(StringRef MatcherCode, Diagnostics *Error)
      : Code(MatcherCode), StartOfLine(MatcherCode), Line(1), Error(Error) {
    NextToken = getNextToken();
  }

  /// \brief Returns but doesn't consume the next token.
  const TokenInfo &peekNextToken() const { return NextToken; }

  /// \brief Consumes and returns the next token.
  TokenInfo consumeNextToken() {
    TokenInfo ThisToken = NextToken;
    NextToken = getNextToken();
    return ThisToken;
  }

  TokenInfo::TokenKind nextTokenKind() const { return NextToken.Kind; }

private:
  TokenInfo getNextToken() {
    consumeWhitespace();
    TokenInfo Result;
    Result.Range.Start = currentLocation();

    if (Code.empty()) {
      Result.Kind = TokenInfo::TK_Eof;
      Result.Text = "";
      return Result;
    }

    switch (Code[0]) {
    case ',':
      Result.Kind = TokenInfo::TK_Comma;
      Result.Text = Code.substr(0, 1);
      Code = Code.drop_front();
      break;
    case '(':
      Result.Kind = TokenInfo::TK_OpenParen;
      Result.Text = Code.substr(0, 1);
      Code = Code.drop_front();
      break;
    case ')':
      Result.Kind = TokenInfo::TK_CloseParen;
      Result.Text = Code.substr(0, 1);
      Code = Code.drop_front();
      break;

    case '"':
    case '\'':
      // Parse a string literal.
      consumeStringLiteral(&Result);
      break;

    default:
      if (isAlphanumeric(Code[0])) {
        // Parse an identifier
        size_t TokenLength = 1;
        while (TokenLength < Code.size() && isAlphanumeric(Code[TokenLength]))
          ++TokenLength;
        Result.Kind = TokenInfo::TK_Ident;
        Result.Text = Code.substr(0, TokenLength);
        Code = Code.drop_front(TokenLength);
      } else {
        Result.Kind = TokenInfo::TK_InvalidChar;
        Result.Text = Code.substr(0, 1);
        Code = Code.drop_front(1);
      }
      break;
    }

    Result.Range.End = currentLocation();
    return Result;
  }

  /// \brief Consume a string literal.
  ///
  /// \c Code must be positioned at the start of the literal (the opening
  /// quote). Consumed until it finds the same closing quote character.
  void consumeStringLiteral(TokenInfo *Result) {
    bool InEscape = false;
    const char Marker = Code[0];
    for (size_t Length = 1, Size = Code.size(); Length != Size; ++Length) {
      if (InEscape) {
        InEscape = false;
        continue;
      }
      if (Code[Length] == '\\') {
        InEscape = true;
        continue;
      }
      if (Code[Length] == Marker) {
        Result->Kind = TokenInfo::TK_Literal;
        Result->Text = Code.substr(0, Length + 1);
        Result->Value = Code.substr(1, Length - 1).str();
        Code = Code.drop_front(Length + 1);
        return;
      }
    }

    StringRef ErrorText = Code;
    Code = Code.drop_front(Code.size());
    SourceRange Range;
    Range.Start = Result->Range.Start;
    Range.End = currentLocation();
    Error->pushErrorFrame(Range, Error->ET_ParserStringError)
        << ErrorText;
    Result->Kind = TokenInfo::TK_Error;
  }

  /// \brief Consume all leading whitespace from \c Code.
  void consumeWhitespace() {
    while (!Code.empty() && isWhitespace(Code[0])) {
      if (Code[0] == '\n') {
        ++Line;
        StartOfLine = Code.drop_front();
      }
      Code = Code.drop_front();
    }
  }

  SourceLocation currentLocation() {
    SourceLocation Location;
    Location.Line = Line;
    Location.Column = Code.data() - StartOfLine.data() + 1;
    return Location;
  }

  StringRef Code;
  StringRef StartOfLine;
  unsigned Line;
  Diagnostics *Error;
  TokenInfo NextToken;
};

Parser::Sema::~Sema() {}

/// \brief Parse and validate a matcher expression.
/// \return \c true on success, in which case \c Value has the matcher parsed.
///   If the input is malformed, or some argument has an error, it
///   returns \c false.
bool Parser::parseMatcherExpressionImpl(VariantValue *Value) {
  const TokenInfo NameToken = Tokenizer->consumeNextToken();
  assert(NameToken.Kind == TokenInfo::TK_Ident);
  const TokenInfo OpenToken = Tokenizer->consumeNextToken();
  if (OpenToken.Kind != TokenInfo::TK_OpenParen) {
    Error->pushErrorFrame(OpenToken.Range, Error->ET_ParserNoOpenParen)
        << OpenToken.Text;
    return false;
  }

  std::vector<ParserValue> Args;
  TokenInfo EndToken;
  while (Tokenizer->nextTokenKind() != TokenInfo::TK_Eof) {
    if (Tokenizer->nextTokenKind() == TokenInfo::TK_CloseParen) {
      // End of args.
      EndToken = Tokenizer->consumeNextToken();
      break;
    }
    if (Args.size() > 0) {
      // We must find a , token to continue.
      const TokenInfo CommaToken = Tokenizer->consumeNextToken();
      if (CommaToken.Kind != TokenInfo::TK_Comma) {
        Error->pushErrorFrame(CommaToken.Range, Error->ET_ParserNoComma)
            << CommaToken.Text;
        return false;
      }
    }

    ParserValue ArgValue;
    ArgValue.Text = Tokenizer->peekNextToken().Text;
    ArgValue.Range = Tokenizer->peekNextToken().Range;
    if (!parseExpressionImpl(&ArgValue.Value)) {
      Error->pushErrorFrame(NameToken.Range,
                            Error->ET_ParserMatcherArgFailure)
          << (Args.size() + 1) << NameToken.Text;
      return false;
    }

    Args.push_back(ArgValue);
  }

  if (EndToken.Kind == TokenInfo::TK_Eof) {
    Error->pushErrorFrame(OpenToken.Range, Error->ET_ParserNoCloseParen);
    return false;
  }

  // Merge the start and end infos.
  SourceRange MatcherRange = NameToken.Range;
  MatcherRange.End = EndToken.Range.End;
  DynTypedMatcher *Result =
      S->actOnMatcherExpression(NameToken.Text, MatcherRange, Args, Error);
  if (Result == NULL) {
    Error->pushErrorFrame(NameToken.Range, Error->ET_ParserMatcherFailure)
        << NameToken.Text;
    return false;
  }

  Value->takeMatcher(Result);
  return true;
}

/// \brief Parse an <Expresssion>
bool Parser::parseExpressionImpl(VariantValue *Value) {
  switch (Tokenizer->nextTokenKind()) {
  case TokenInfo::TK_Literal:
    *Value = Tokenizer->consumeNextToken().Value;
    return true;

  case TokenInfo::TK_Ident:
    return parseMatcherExpressionImpl(Value);

  case TokenInfo::TK_Eof:
    Error->pushErrorFrame(Tokenizer->consumeNextToken().Range,
                          Error->ET_ParserNoCode);
    return false;

  case TokenInfo::TK_Error:
    // This error was already reported by the tokenizer.
    return false;

  case TokenInfo::TK_OpenParen:
  case TokenInfo::TK_CloseParen:
  case TokenInfo::TK_Comma:
  case TokenInfo::TK_InvalidChar:
    const TokenInfo Token = Tokenizer->consumeNextToken();
    Error->pushErrorFrame(Token.Range, Error->ET_ParserInvalidToken)
        << Token.Text;
    return false;
  }

  llvm_unreachable("Unknown token kind.");
}

Parser::Parser(CodeTokenizer *Tokenizer, Sema *S,
               Diagnostics *Error)
    : Tokenizer(Tokenizer), S(S), Error(Error) {}

class RegistrySema : public Parser::Sema {
public:
  virtual ~RegistrySema() {}
  DynTypedMatcher *actOnMatcherExpression(StringRef MatcherName,
                                          const SourceRange &NameRange,
                                          ArrayRef<ParserValue> Args,
                                          Diagnostics *Error) {
    return Registry::constructMatcher(MatcherName, NameRange, Args, Error);
  }
};

bool Parser::parseExpression(StringRef Code, VariantValue *Value,
                             Diagnostics *Error) {
  RegistrySema S;
  return parseExpression(Code, &S, Value, Error);
}

bool Parser::parseExpression(StringRef Code, Sema *S,
                             VariantValue *Value, Diagnostics *Error) {
  CodeTokenizer Tokenizer(Code, Error);
  return Parser(&Tokenizer, S, Error).parseExpressionImpl(Value);
}

DynTypedMatcher *Parser::parseMatcherExpression(StringRef Code,
                                                Diagnostics *Error) {
  RegistrySema S;
  return parseMatcherExpression(Code, &S, Error);
}

DynTypedMatcher *Parser::parseMatcherExpression(StringRef Code,
                                                Parser::Sema *S,
                                                Diagnostics *Error) {
  VariantValue Value;
  if (!parseExpression(Code, S, &Value, Error))
    return NULL;
  if (!Value.isMatcher()) {
    Error->pushErrorFrame(SourceRange(), Error->ET_ParserNotAMatcher);
    return NULL;
  }
  return Value.getMatcher().clone();
}

}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang
