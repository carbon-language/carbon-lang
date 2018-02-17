#ifndef FORTRAN_PARSER_PREPROCESSOR_H_
#define FORTRAN_PARSER_PREPROCESSOR_H_

// A Fortran-aware preprocessing module used by the prescanner to implement
// preprocessing directives and macro replacement.  Intended to be efficient
// enough to always run on all source files even when no preprocessing is
// performed, so that special compiler command options &/or source file name
// extensions for preprocessing will not be necessary.

#include "provenance.h"
#include "token-sequence.h"
#include <list>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace Fortran {
namespace parser {

class Prescanner;

// Defines a macro
class Definition {
public:
  Definition(const TokenSequence &, size_t firstToken, size_t tokens);
  Definition(const std::vector<std::string> &argNames, const TokenSequence &,
      size_t firstToken, size_t tokens, bool isVariadic = false);
  Definition(const std::string &predefined, AllSources *);

  bool isFunctionLike() const { return isFunctionLike_; }
  size_t argumentCount() const { return argumentCount_; }
  bool isVariadic() const { return isVariadic_; }
  bool isDisabled() const { return isDisabled_; }
  bool isPredefined() const { return isPredefined_; }
  const TokenSequence &replacement() const { return replacement_; }

  bool set_isDisabled(bool disable);

  TokenSequence Apply(
      const std::vector<TokenSequence> &args, const AllSources &);

private:
  static TokenSequence Tokenize(const std::vector<std::string> &argNames,
      const TokenSequence &token, size_t firstToken, size_t tokens);

  bool isFunctionLike_{false};
  size_t argumentCount_{0};
  bool isVariadic_{false};
  bool isDisabled_{false};
  bool isPredefined_{false};
  TokenSequence replacement_;
};

// Preprocessing state
class Preprocessor {
public:
  explicit Preprocessor(AllSources *);

  // When the input contains macros to be replaced, the new token sequence
  // is appended to the output and the returned value is true.  When
  // no macro replacement is necessary, the output is unmodified and the
  // return value is false.
  bool MacroReplacement(
      const TokenSequence &, const Prescanner &, TokenSequence *);

  // Implements a preprocessor directive; returns true when no fatal error.
  bool Directive(const TokenSequence &, Prescanner *);

private:
  enum class IsElseActive { No, Yes };
  enum class CanDeadElseAppear { No, Yes };

  CharPointerWithLength SaveTokenAsName(const CharPointerWithLength &);
  bool IsNameDefined(const CharPointerWithLength &);
  TokenSequence ReplaceMacros(const TokenSequence &, const Prescanner &);
  bool SkipDisabledConditionalCode(const std::string &, IsElseActive, Prescanner *);
  bool IsIfPredicateTrue(
      const TokenSequence &expr, size_t first, size_t exprTokens, Prescanner *);

  AllSources *allSources_;
  std::list<std::string> names_;
  std::unordered_map<CharPointerWithLength, Definition> definitions_;
  std::stack<CanDeadElseAppear> ifStack_;
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_PREPROCESSOR_H_
