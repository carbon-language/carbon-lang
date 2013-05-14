//===--- Diagnostics.h - Helper class for error diagnostics -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Diagnostics class to manage error messages.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_DYNAMIC_DIAGNOSTICS_H
#define LLVM_CLANG_AST_MATCHERS_DYNAMIC_DIAGNOSTICS_H

#include <string>
#include <vector>

#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {

struct SourceLocation {
  SourceLocation() : Line(), Column() {}
  unsigned Line;
  unsigned Column;
};

struct SourceRange {
  SourceLocation Start;
  SourceLocation End;
};

/// \brief A VariantValue instance annotated with its parser context.
struct ParserValue {
  ParserValue() : Text(), Range(), Value() {}
  StringRef Text;
  SourceRange Range;
  VariantValue Value;
};

/// \brief Helper class to manage error messages.
class Diagnostics {
 public:
  /// \brief All errors from the system.
  enum ErrorType {
    ET_None = 0,

    ET_RegistryNotFound = 1,
    ET_RegistryWrongArgCount = 2,
    ET_RegistryWrongArgType = 3,

    ET_ParserStringError = 100,
    ET_ParserMatcherArgFailure = 101,
    ET_ParserMatcherFailure = 102,
    ET_ParserNoOpenParen = 103,
    ET_ParserNoCloseParen = 104,
    ET_ParserNoComma = 105,
    ET_ParserNoCode = 106,
    ET_ParserNotAMatcher = 107,
    ET_ParserInvalidToken = 108
  };

  /// \brief Helper stream class.
  struct ArgStream {
    template <class T> ArgStream &operator<<(const T &Arg) {
      return operator<<(Twine(Arg));
    }
    ArgStream &operator<<(const Twine &Arg);
    std::vector<std::string> *Out;
  };

  /// \brief Push a frame to the beginning of the list
  ///
  /// Returns a helper class to allow the caller to pass the arguments for the
  /// error message, using the << operator.
  ArgStream pushErrorFrame(const SourceRange &Range, ErrorType Error);

  struct ErrorFrame {
    SourceRange Range;
    ErrorType Type;
    std::vector<std::string> Args;

    std::string ToString() const;
  };
  ArrayRef<ErrorFrame> frames() const { return Frames; }

  /// \brief Returns a string representation of the last frame.
  std::string ToString() const;
  /// \brief Returns a string representation of the whole frame stack.
  std::string ToStringFull() const;

 private:
   std::vector<ErrorFrame> Frames;
};

}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang

#endif  // LLVM_CLANG_AST_MATCHERS_DYNAMIC_DIAGNOSTICS_H
