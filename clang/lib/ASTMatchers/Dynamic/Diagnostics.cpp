//===--- Diagnostics.cpp - Helper class for error diagnostics -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/Dynamic/Diagnostics.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {

Diagnostics::ArgStream Diagnostics::pushContextFrame(ContextType Type,
                                                     SourceRange Range) {
  ContextStack.push_back(ContextFrame());
  ContextFrame& data = ContextStack.back();
  data.Type = Type;
  data.Range = Range;
  return ArgStream(&data.Args);
}

Diagnostics::Context::Context(ConstructMatcherEnum, Diagnostics *Error,
                              StringRef MatcherName,
                              const SourceRange &MatcherRange)
    : Error(Error) {
  Error->pushContextFrame(CT_MatcherConstruct, MatcherRange) << MatcherName;
}

Diagnostics::Context::Context(MatcherArgEnum, Diagnostics *Error,
                              StringRef MatcherName,
                              const SourceRange &MatcherRange,
                              unsigned ArgNumber)
    : Error(Error) {
  Error->pushContextFrame(CT_MatcherArg, MatcherRange) << ArgNumber
                                                       << MatcherName;
}

Diagnostics::Context::~Context() { Error->ContextStack.pop_back(); }

Diagnostics::ArgStream &Diagnostics::ArgStream::operator<<(const Twine &Arg) {
  Out->push_back(Arg.str());
  return *this;
}

Diagnostics::ArgStream Diagnostics::addError(const SourceRange &Range,
                                             ErrorType Error) {
  Errors.push_back(ErrorContent());
  ErrorContent &Last = Errors.back();
  Last.ContextStack = ContextStack;
  Last.Range = Range;
  Last.Type = Error;
  return ArgStream(&Last.Args);
}

StringRef ContextTypeToString(Diagnostics::ContextType Type) {
  switch (Type) {
    case Diagnostics::CT_MatcherConstruct:
      return "Error building matcher $0.";
    case Diagnostics::CT_MatcherArg:
      return "Error parsing argument $0 for matcher $1.";
  }
  llvm_unreachable("Unknown ContextType value.");
}

StringRef ErrorTypeToString(Diagnostics::ErrorType Type) {
  switch (Type) {
  case Diagnostics::ET_RegistryNotFound:
    return "Matcher not found: $0";
  case Diagnostics::ET_RegistryWrongArgCount:
    return "Incorrect argument count. (Expected = $0) != (Actual = $1)";
  case Diagnostics::ET_RegistryWrongArgType:
    return "Incorrect type for arg $0. (Expected = $1) != (Actual = $2)";
  case Diagnostics::ET_RegistryNotBindable:
    return "Matcher does not support binding.";

  case Diagnostics::ET_ParserStringError:
    return "Error parsing string token: <$0>";
  case Diagnostics::ET_ParserNoOpenParen:
    return "Error parsing matcher. Found token <$0> while looking for '('.";
  case Diagnostics::ET_ParserNoCloseParen:
    return "Error parsing matcher. Found end-of-code while looking for ')'.";
  case Diagnostics::ET_ParserNoComma:
    return "Error parsing matcher. Found token <$0> while looking for ','.";
  case Diagnostics::ET_ParserNoCode:
    return "End of code found while looking for token.";
  case Diagnostics::ET_ParserNotAMatcher:
    return "Input value is not a matcher expression.";
  case Diagnostics::ET_ParserInvalidToken:
    return "Invalid token <$0> found when looking for a value.";
  case Diagnostics::ET_ParserMalformedBindExpr:
    return "Malformed bind() expression.";
  case Diagnostics::ET_ParserTrailingCode:
    return "Expected end of code.";
  case Diagnostics::ET_ParserUnsignedError:
    return "Error parsing unsigned token: <$0>";
  case Diagnostics::ET_ParserOverloadedType:
    return "Input value has unresolved overloaded type: $0";

  case Diagnostics::ET_None:
    return "<N/A>";
  }
  llvm_unreachable("Unknown ErrorType value.");
}

std::string FormatErrorString(StringRef FormatString,
                              ArrayRef<std::string> Args) {
  std::string Out;
  while (!FormatString.empty()) {
    std::pair<StringRef, StringRef> Pieces = FormatString.split("$");
    Out += Pieces.first.str();
    if (Pieces.second.empty()) break;

    const char Next = Pieces.second.front();
    FormatString = Pieces.second.drop_front();
    if (Next >= '0' && Next <= '9') {
      const unsigned Index = Next - '0';
      if (Index < Args.size()) {
        Out += Args[Index];
      } else {
        Out += "<Argument_Not_Provided>";
      }
    }
  }
  return Out;
}

static std::string MaybeAddLineAndColumn(Twine Input,
                                         const SourceRange &Range) {
  if (Range.Start.Line > 0 && Range.Start.Column > 0)
    return (Twine(Range.Start.Line) + ":" + Twine(Range.Start.Column) + ": " +
            Input).str();
  return Input.str();
}

std::string Diagnostics::ContextFrame::ToString() const {
  return MaybeAddLineAndColumn(
      FormatErrorString(ContextTypeToString(Type), Args), Range);
}

std::string Diagnostics::ErrorContent::ToString() const {
  return MaybeAddLineAndColumn(FormatErrorString(ErrorTypeToString(Type), Args),
                               Range);
}

std::string Diagnostics::ToString() const {
  std::string Result;
  for (size_t i = 0, e = Errors.size(); i != e; ++i) {
    if (i != 0) Result += "\n";
    Result += Errors[i].ToString();
  }
  return Result;
}

std::string Diagnostics::ToStringFull() const {
  std::string Result;
  for (size_t i = 0, e = Errors.size(); i != e; ++i) {
    if (i != 0) Result += "\n";
    const ErrorContent &Error = Errors[i];
    for (size_t i = 0, e = Error.ContextStack.size(); i != e; ++i) {
      Result += Error.ContextStack[i].ToString() + "\n";
    }
    Result += Error.ToString();
  }
  return Result;
}

}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang
