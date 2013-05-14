//===--- Registry.h - Matcher registry -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Registry of all known matchers.
///
/// The registry provides a generic interface to construct any matcher by name.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_DYNAMIC_REGISTRY_H
#define LLVM_CLANG_AST_MATCHERS_DYNAMIC_REGISTRY_H

#include "clang/ASTMatchers/Dynamic/Diagnostics.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {

class Registry {
public:
  /// \brief Construct a matcher from the registry by name.
  ///
  /// Consult the registry of known matchers and construct the appropriate
  /// matcher by name.
  ///
  /// \param MatcherName The name of the matcher to instantiate.
  ///
  /// \param NameRange The location of the name in the matcher source.
  ///   Useful for error reporting.
  ///
  /// \param Args The argument list for the matcher. The number and types of the
  ///   values must be valid for the matcher requested. Otherwise, the function
  ///   will return an error.
  ///
  /// \return The matcher if no error was found. NULL if the matcher is not
  //    found, or if the number of arguments or argument types do not
  ///   match the signature. In that case \c Error will contain the description
  ///   of the error.
  static DynTypedMatcher *constructMatcher(StringRef MatcherName,
                                           const SourceRange &NameRange,
                                           ArrayRef<ParserValue> Args,
                                           Diagnostics *Error);

private:
  Registry() LLVM_DELETED_FUNCTION;
};

}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang

#endif  // LLVM_CLANG_AST_MATCHERS_DYNAMIC_REGISTRY_H
