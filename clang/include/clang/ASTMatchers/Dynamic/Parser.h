//===--- Parser.h - Matcher expression parser -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Simple matcher expression parser.
///
/// The parser understands matcher expressions of the form:
///   MatcherName(Arg0, Arg1, ..., ArgN)
/// as well as simple types like strings.
/// The parser does not know how to process the matchers. It delegates this task
/// to a Sema object received as an argument.
///
/// \code
/// Grammar for the expressions supported:
/// <Expression>        := <Literal> | <NamedValue> | <MatcherExpression>
/// <Literal>           := <StringLiteral> | <Unsigned>
/// <StringLiteral>     := "quoted string"
/// <Unsigned>          := [0-9]+
/// <NamedValue>        := <Identifier>
/// <MatcherExpression> := <Identifier>(<ArgumentList>) |
///                        <Identifier>(<ArgumentList>).bind(<StringLiteral>)
/// <Identifier>        := [a-zA-Z]+
/// <ArgumentList>      := <Expression> | <Expression>,<ArgumentList>
/// \endcode
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_DYNAMIC_PARSER_H
#define LLVM_CLANG_AST_MATCHERS_DYNAMIC_PARSER_H

#include "clang/ASTMatchers/Dynamic/Diagnostics.h"
#include "clang/ASTMatchers/Dynamic/Registry.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {

/// \brief Matcher expression parser.
class Parser {
public:
  /// \brief Interface to connect the parser with the registry and more.
  ///
  /// The parser uses the Sema instance passed into
  /// parseMatcherExpression() to handle all matcher tokens. The simplest
  /// processor implementation would simply call into the registry to create
  /// the matchers.
  /// However, a more complex processor might decide to intercept the matcher
  /// creation and do some extra work. For example, it could apply some
  /// transformation to the matcher by adding some id() nodes, or could detect
  /// specific matcher nodes for more efficient lookup.
  class Sema {
  public:
    virtual ~Sema();

    /// \brief Lookup a value by name.
    ///
    /// This can be used in the Sema layer to declare known constants or to
    /// allow to split an expression in pieces.
    ///
    /// \param Name The name of the value to lookup.
    ///
    /// \return The named value. It could be any type that VariantValue
    ///   supports. An empty value means that the name is not recognized.
    virtual VariantValue getNamedValue(StringRef Name);

    /// \brief Process a matcher expression.
    ///
    /// All the arguments passed here have already been processed.
    ///
    /// \param Ctor A matcher constructor looked up by lookupMatcherCtor.
    ///
    /// \param NameRange The location of the name in the matcher source.
    ///   Useful for error reporting.
    ///
    /// \param BindID The ID to use to bind the matcher, or a null \c StringRef
    ///   if no ID is specified.
    ///
    /// \param Args The argument list for the matcher.
    ///
    /// \return The matcher objects constructed by the processor, or a null
    ///   matcher if an error occurred. In that case, \c Error will contain a
    ///   description of the error.
    virtual VariantMatcher actOnMatcherExpression(MatcherCtor Ctor,
                                                  const SourceRange &NameRange,
                                                  StringRef BindID,
                                                  ArrayRef<ParserValue> Args,
                                                  Diagnostics *Error) = 0;

    /// \brief Look up a matcher by name.
    ///
    /// \param MatcherName The matcher name found by the parser.
    ///
    /// \return The matcher constructor, or Optional<MatcherCtor>() if not
    /// found.
    virtual llvm::Optional<MatcherCtor>
    lookupMatcherCtor(StringRef MatcherName) = 0;
  };

  /// \brief Sema implementation that uses the matcher registry to process the
  ///   tokens.
  class RegistrySema : public Parser::Sema {
   public:
    virtual ~RegistrySema();

    llvm::Optional<MatcherCtor>
    lookupMatcherCtor(StringRef MatcherName) override;

    VariantMatcher actOnMatcherExpression(MatcherCtor Ctor,
                                          const SourceRange &NameRange,
                                          StringRef BindID,
                                          ArrayRef<ParserValue> Args,
                                          Diagnostics *Error) override;
  };

  /// \brief Parse a matcher expression, creating matchers from the registry.
  ///
  /// This overload creates matchers calling directly into the registry. If the
  /// caller needs more control over how the matchers are created, then it can
  /// use the overload below that takes a Sema.
  ///
  /// \param MatcherCode The matcher expression to parse.
  ///
  /// \return The matcher object constructed, or an empty Optional if an error
  ///   occurred.
  ///   In that case, \c Error will contain a description of the error.
  ///   The caller takes ownership of the DynTypedMatcher object returned.
  static llvm::Optional<DynTypedMatcher>
  parseMatcherExpression(StringRef MatcherCode, Diagnostics *Error);

  /// \brief Parse a matcher expression.
  ///
  /// \param MatcherCode The matcher expression to parse.
  ///
  /// \param S The Sema instance that will help the parser
  ///   construct the matchers.
  /// \return The matcher object constructed by the processor, or an empty
  ///   Optional if an error occurred. In that case, \c Error will contain a
  ///   description of the error.
  ///   The caller takes ownership of the DynTypedMatcher object returned.
  static llvm::Optional<DynTypedMatcher>
  parseMatcherExpression(StringRef MatcherCode, Sema *S, Diagnostics *Error);

  /// \brief Parse an expression, creating matchers from the registry.
  ///
  /// Parses any expression supported by this parser. In general, the
  /// \c parseMatcherExpression function is a better approach to get a matcher
  /// object.
  static bool parseExpression(StringRef Code, VariantValue *Value,
                              Diagnostics *Error);

  /// \brief Parse an expression.
  ///
  /// Parses any expression supported by this parser. In general, the
  /// \c parseMatcherExpression function is a better approach to get a matcher
  /// object.
  static bool parseExpression(StringRef Code, Sema *S,
                              VariantValue *Value, Diagnostics *Error);

  /// \brief Complete an expression at the given offset.
  ///
  /// \return The list of completions, which may be empty if there are no
  /// available completions or if an error occurred.
  static std::vector<MatcherCompletion>
  completeExpression(StringRef Code, unsigned CompletionOffset);

private:
  class CodeTokenizer;
  struct ScopedContextEntry;
  struct TokenInfo;

  Parser(CodeTokenizer *Tokenizer, Sema *S,
         Diagnostics *Error);

  bool parseExpressionImpl(VariantValue *Value);
  bool parseMatcherExpressionImpl(const TokenInfo &NameToken,
                                  VariantValue *Value);
  bool parseIdentifierPrefixImpl(VariantValue *Value);

  void addCompletion(const TokenInfo &CompToken, StringRef TypedText,
                     StringRef Decl);
  void addExpressionCompletions();

  CodeTokenizer *const Tokenizer;
  Sema *const S;
  Diagnostics *const Error;

  typedef std::vector<std::pair<MatcherCtor, unsigned> > ContextStackTy;
  ContextStackTy ContextStack;
  std::vector<MatcherCompletion> Completions;
};

}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang

#endif  // LLVM_CLANG_AST_MATCHERS_DYNAMIC_PARSER_H
