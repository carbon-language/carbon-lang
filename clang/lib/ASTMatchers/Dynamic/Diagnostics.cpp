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

Diagnostics::ArgStream &
Diagnostics::ArgStream::operator<<(const Twine &Arg) {
  Out->push_back(Arg.str());
  return *this;
}

Diagnostics::ArgStream Diagnostics::pushErrorFrame(const SourceRange &Range,
                                                   ErrorType Error) {
  Frames.insert(Frames.begin(), ErrorFrame());
  ErrorFrame &Last = Frames.front();
  Last.Range = Range;
  Last.Type = Error;
  ArgStream Out = { &Last.Args };
  return Out;
}

StringRef ErrorTypeToString(Diagnostics::ErrorType Type) {
  switch (Type) {
  case Diagnostics::ET_RegistryNotFound:
    return "Matcher not found: $0";
  case Diagnostics::ET_RegistryWrongArgCount:
    return "Incorrect argument count. (Expected = $0) != (Actual = $1)";
  case Diagnostics::ET_RegistryWrongArgType:
    return "Incorrect type on function $0 for arg $1.";
  case Diagnostics::ET_RegistryNotBindable:
    return "Matcher does not support binding.";

  case Diagnostics::ET_ParserStringError:
    return "Error parsing string token: <$0>";
  case Diagnostics::ET_ParserMatcherArgFailure:
    return "Error parsing argument $0 for matcher $1.";
  case Diagnostics::ET_ParserMatcherFailure:
    return "Error building matcher $0.";
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

std::string Diagnostics::ErrorFrame::ToString() const {
  StringRef FormatString = ErrorTypeToString(Type);
  std::string ErrorOut = FormatErrorString(FormatString, Args);
  if (Range.Start.Line > 0 && Range.Start.Column > 0)
    return (Twine(Range.Start.Line) + ":" + Twine(Range.Start.Column) + ": " +
            ErrorOut).str();
  return ErrorOut;
}

std::string Diagnostics::ToString() const {
  if (Frames.empty()) return "";
  return Frames[Frames.size() - 1].ToString();
}

std::string Diagnostics::ToStringFull() const {
  std::string Result;
  for (size_t i = 0, end = Frames.size(); i != end; ++i) {
    if (i > 0) Result += "\n";
    Result += Frames[i].ToString();
  }
  return Result;
}

}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang
