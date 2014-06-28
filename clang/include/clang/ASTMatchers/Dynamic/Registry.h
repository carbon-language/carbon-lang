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
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {

namespace internal {
class MatcherDescriptor;
}

typedef const internal::MatcherDescriptor *MatcherCtor;

struct MatcherCompletion {
  MatcherCompletion() {}
  MatcherCompletion(StringRef TypedText, StringRef MatcherDecl)
      : TypedText(TypedText), MatcherDecl(MatcherDecl) {}

  /// \brief The text to type to select this matcher.
  std::string TypedText;

  /// \brief The "declaration" of the matcher, with type information.
  std::string MatcherDecl;

  bool operator==(const MatcherCompletion &Other) const {
    return TypedText == Other.TypedText && MatcherDecl == Other.MatcherDecl;
  }
};

class Registry {
public:
  /// \brief Look up a matcher in the registry by name,
  ///
  /// \return An opaque value which may be used to refer to the matcher
  /// constructor, or Optional<MatcherCtor>() if not found.
  static llvm::Optional<MatcherCtor> lookupMatcherCtor(StringRef MatcherName);

  /// \brief Compute the list of completions for \p Context.
  ///
  /// Each element of \p Context represents a matcher invocation, going from
  /// outermost to innermost. Elements are pairs consisting of a reference to the
  /// matcher constructor and the index of the next element in the argument list
  /// of that matcher (or for the last element, the index of the completion
  /// point in the argument list). An empty list requests completion for the
  /// root matcher.
  ///
  /// The completions are ordered first by decreasing relevance, then
  /// alphabetically.  Relevance is determined by how closely the matcher's
  /// type matches that of the context. For example, if the innermost matcher
  /// takes a FunctionDecl matcher, the FunctionDecl matchers are returned
  /// first, followed by the ValueDecl matchers, then NamedDecl, then Decl, then
  /// polymorphic matchers.
  ///
  /// Matchers which are technically convertible to the innermost context but
  /// which would match either all or no nodes are excluded. For example,
  /// namedDecl and varDecl are excluded in a FunctionDecl context, because
  /// those matchers would match respectively all or no nodes in such a context.
  static std::vector<MatcherCompletion>
  getCompletions(ArrayRef<std::pair<MatcherCtor, unsigned> > Context);

  /// \brief Construct a matcher from the registry.
  ///
  /// \param Ctor The matcher constructor to instantiate.
  ///
  /// \param NameRange The location of the name in the matcher source.
  ///   Useful for error reporting.
  ///
  /// \param Args The argument list for the matcher. The number and types of the
  ///   values must be valid for the matcher requested. Otherwise, the function
  ///   will return an error.
  ///
  /// \return The matcher object constructed if no error was found.
  ///   A null matcher if the number of arguments or argument types do not match
  ///   the signature.  In that case \c Error will contain the description of
  ///   the error.
  static VariantMatcher constructMatcher(MatcherCtor Ctor,
                                         const SourceRange &NameRange,
                                         ArrayRef<ParserValue> Args,
                                         Diagnostics *Error);

  /// \brief Construct a matcher from the registry and bind it.
  ///
  /// Similar the \c constructMatcher() above, but it then tries to bind the
  /// matcher to the specified \c BindID.
  /// If the matcher is not bindable, it sets an error in \c Error and returns
  /// a null matcher.
  static VariantMatcher constructBoundMatcher(MatcherCtor Ctor,
                                              const SourceRange &NameRange,
                                              StringRef BindID,
                                              ArrayRef<ParserValue> Args,
                                              Diagnostics *Error);

private:
  Registry() LLVM_DELETED_FUNCTION;
};

}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang

#endif  // LLVM_CLANG_AST_MATCHERS_DYNAMIC_REGISTRY_H
