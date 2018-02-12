#ifndef FORTRAN_PREPROCESSOR_H_
#define FORTRAN_PREPROCESSOR_H_

// A Fortran-aware preprocessing module used by the prescanner to implement
// preprocessing directives and macro replacement.  Intended to be efficient
// enough to always run on all source files even when no preprocessing is
// needed, so that special compiler command options &/or source file name
// extensions for preprocessing will not be necessary.

#include "idioms.h"
#include "provenance.h"
#include <cctype>
#include <cstring>
#include <functional>
#include <list>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace Fortran {
namespace parser {

class CookedSource;
class Prescanner;

// Just a const char pointer with an associated length; does not own the
// referenced data.  Used to describe buffered tokens and hash table keys.
class CharPointerWithLength {
public:
  CharPointerWithLength() {}
  CharPointerWithLength(const char *x, size_t n) : data_{x}, bytes_{n} {}
  CharPointerWithLength(const std::string &s)
    : data_{s.data()}, bytes_{s.size()} {}
  CharPointerWithLength(const CharPointerWithLength &that)
    : data_{that.data_}, bytes_{that.bytes_} {}
  CharPointerWithLength &operator=(const CharPointerWithLength &that) {
    data_ = that.data_;
    bytes_ = that.bytes_;
    return *this;
  }

  bool empty() const { return bytes_ == 0; }
  size_t size() const { return bytes_; }
  const char &operator[](size_t j) const { return data_[j]; }

  bool IsBlank() const;
  std::string ToString() const { return std::string{data_, bytes_}; }

private:
  const char *data_{nullptr};
  size_t bytes_{0};
};
}  // namespace parser
}  // namespace Fortran

// Specializations to enable std::unordered_map<CharPointerWithLength, ...>
template<> struct std::hash<Fortran::parser::CharPointerWithLength> {
  size_t operator()(const Fortran::parser::CharPointerWithLength &x) const {
    size_t hash{0}, bytes{x.size()};
    for (size_t j{0}; j < bytes; ++j) {
      hash = (hash * 31) ^ x[j];
    }
    return hash;
  }
};

template<> struct std::equal_to<Fortran::parser::CharPointerWithLength> {
  bool operator()(const Fortran::parser::CharPointerWithLength &x,
      const Fortran::parser::CharPointerWithLength &y) const {
    return x.size() == y.size() &&
        std::memcmp(static_cast<const void *>(&x[0]),
            static_cast<const void *>(&y[0]), x.size()) == 0;
  }
};

namespace Fortran {
namespace parser {

// Buffers a contiguous sequence of characters that has been partitioned into
// a sequence of preprocessing tokens with provenances.
class TokenSequence {
public:
  TokenSequence() {}
  TokenSequence(const TokenSequence &that) { Put(that); }
  TokenSequence(const TokenSequence &that, size_t at, size_t count = 1) {
    Put(that, at, count);
  }
  TokenSequence(TokenSequence &&that)
    : start_{std::move(that.start_)}, nextStart_{that.nextStart_},
      char_{std::move(that.char_)}, provenances_{std::move(that.provenances_)} {
  }
  TokenSequence(const std::string &s, Provenance p) { Put(s, p); }

  TokenSequence &operator=(const TokenSequence &that) {
    clear();
    Put(that);
    return *this;
  }
  TokenSequence &operator=(TokenSequence &&that) {
    start_ = std::move(that.start_);
    nextStart_ = that.nextStart_;
    char_ = std::move(that.char_);
    return *this;
  }

  CharPointerWithLength operator[](size_t token) const {
    return {&char_[start_[token]], TokenBytes(token)};
  }

  bool empty() const { return start_.empty(); }
  size_t size() const { return start_.size(); }
  const char *data() const { return &char_[0]; }
  void clear();
  void pop_back();
  void shrink_to_fit();

  void PutNextTokenChar(char ch, Provenance provenance) {
    char_.emplace_back(ch);
    provenances_.Put({provenance, 1});
  }

  void CloseToken() {
    start_.emplace_back(nextStart_);
    nextStart_ = char_.size();
  }

  void ReopenLastToken() {
    nextStart_ = start_.back();
    start_.pop_back();
  }

  void Put(const TokenSequence &);
  void Put(const TokenSequence &, size_t at, size_t tokens = 1);
  void Put(const char *, size_t, Provenance);
  void Put(const CharPointerWithLength &, Provenance);
  void Put(const std::string &, Provenance);
  void Put(const std::stringstream &, Provenance);
  void EmitWithCaseConversion(CookedSource *) const;
  std::string ToString() const;
  ProvenanceRange GetProvenance(size_t token, size_t offset = 0) const;

private:
  size_t TokenBytes(size_t token) const {
    return (token + 1 >= start_.size() ? char_.size() : start_[token + 1]) -
        start_[token];
  }

  std::vector<size_t> start_;
  size_t nextStart_{0};
  std::vector<char> char_;
  OffsetToProvenanceMappings provenances_;
};

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
      const std::vector<TokenSequence> &args, const Prescanner &);

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
  explicit Preprocessor(Prescanner &);

  // When the input contains macros to be replaced, the new token sequence
  // is appended to the output and the returned value is true.  When
  // no macro replacement is necessary, the output is unmodified and the
  // return value is false.
  bool MacroReplacement(const TokenSequence &, TokenSequence *);

  // Implements a preprocessor directive; returns true when no fatal error.
  bool Directive(const TokenSequence &);

private:
  enum class IsElseActive { No, Yes };
  enum class CanDeadElseAppear { No, Yes };

  void Complain(const std::string &);
  CharPointerWithLength SaveTokenAsName(const CharPointerWithLength &);
  bool IsNameDefined(const CharPointerWithLength &);
  TokenSequence ReplaceMacros(const TokenSequence &);
  bool SkipDisabledConditionalCode(const std::string &dirName, IsElseActive);
  bool IsIfPredicateTrue(
      const TokenSequence &expr, size_t first, size_t exprTokens);

  Prescanner &prescanner_;
  std::list<std::string> names_;
  std::unordered_map<CharPointerWithLength, Definition> definitions_;
  std::stack<CanDeadElseAppear> ifStack_;
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PREPROCESSOR_H_
