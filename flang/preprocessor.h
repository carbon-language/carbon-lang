#ifndef FORTRAN_PREPROCESSOR_H_
#define FORTRAN_PREPROCESSOR_H_

// A Fortran-aware preprocessing module used by the prescanner to implement
// preprocessing directives and macro replacement.  Intended to be efficient
// enough to always run on all source files even when no preprocessing is
// needed, so that special compiler command options &/or source file name
// extensions for preprocessing will not be necessary.

#include "idioms.h"
#include <cctype>
#include <cstring>
#include <functional>
#include <list>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace Fortran {

class CharBuffer;
class Prescanner;

// Just a const char pointer with an associated length; does not own the
// referenced data.  Used to describe buffered tokens and hash table keys.
class CharPointerWithLength {
 public:
  CharPointerWithLength() {}
  CharPointerWithLength(const char *x, size_t n) : data_{x}, bytes_{n} {}
  CharPointerWithLength(const CharPointerWithLength &that)
    : data_{that.data_}, bytes_{that.bytes_} {}
  CharPointerWithLength &operator=(const CharPointerWithLength &that) {
    data_ = that.data_;
    bytes_ = that.bytes_;
    return *this;
  }

  bool empty() const { return bytes_ == 0; }
  size_t size() const { return bytes_; }
  const char *data() const { return data_; }
  const char &operator[](size_t j) const { return data_[j]; }

 private:
  const char *data_{nullptr};
  size_t bytes_{0};
};
}  // namespace Fortran

// Specializations to enable std::unordered_map<CharPointerWithLength, ...>
template<> struct std::hash<Fortran::CharPointerWithLength> {
  size_t operator()(const Fortran::CharPointerWithLength &x) const {
    size_t hash{0};
    const char *p{x.data()}, *limit{p + x.size()};
    for (; p < limit; ++p) {
      hash = (hash * 31) ^ *p;
    }
    return hash;
  }
};

template<> struct std::equal_to<Fortran::CharPointerWithLength> {
  bool operator()(const Fortran::CharPointerWithLength &x,
                  const Fortran::CharPointerWithLength &y) const {
    return x.size() == y.size() &&
           std::memcmp(static_cast<const void *>(x.data()),
                       static_cast<const void *>(y.data()),
                       x.size()) == 0;
  }
};

namespace Fortran {

// Buffers a contiguous sequence of characters that has been partitioned into
// a sequence of preprocessing tokens.
class TokenSequence {
 public:
  TokenSequence() {}
  TokenSequence(TokenSequence &&that)
    : start_{std::move(that.start_)}, nextStart_{that.nextStart_},
      char_{std::move(that.char_)} {}
  TokenSequence &operator=(TokenSequence &&that) {
    start_ = std::move(that.start_);
    nextStart_ = that.nextStart_;
    char_ = std::move(that.char_);
    return *this;
  }

  size_t GetBytes(size_t token) const {
    return (token + 1 >= start_.size() ? char_.size() : start_[token + 1]) -
           start_[token];
  }
  const char *GetText(size_t token) const {
    return &char_[start_[token]];
  }
  std::string GetString(size_t token) const {
    return std::string(GetText(token), GetBytes(token));
  }
  CharPointerWithLength GetToken(size_t token) const {
    return {GetText(token), GetBytes(token)};
  }

  void AddChar(char ch) {
    char_.emplace_back(ch);
  }

  void EndToken() {
    // CHECK(char_.size() > nextStart_);
    start_.emplace_back(nextStart_);
    nextStart_ = char_.size();
  }

  void ReopenLastToken() {
    nextStart_ = start_.back();
    start_.pop_back();
  }

  void Append(const TokenSequence &);

  void EmitWithCaseConversion(CharBuffer *);

  bool empty() const { return start_.empty(); }

  size_t size() const { return start_.size(); }

  const char *data() const { return &char_[0]; }

  void clear() {
    start_.clear();
    nextStart_ = 0;
    char_.clear();
  }

  void pop_back() {
    nextStart_ = start_.back();
    start_.pop_back();
    char_.resize(nextStart_);
  }

  void push_back(const char *s, size_t bytes) {
    for (size_t j{0}; j < bytes; ++j) {
      AddChar(s[j]);
    }
    EndToken();
  }

  void push_back(const CharPointerWithLength &t) {
    size_t bytes{t.size()};
    for (size_t j{0}; j < bytes; ++j) {
      AddChar(t[j]);
    }
    EndToken();
  }

  void push_back(const std::string &s) {
    size_t bytes{s.size()};
    for (size_t j{0}; j < bytes; ++j) {
      AddChar(s[j]);
    }
    EndToken();
  }

  void shrink_to_fit() {
    start_.shrink_to_fit();
    char_.shrink_to_fit();
  }

 private:
  std::vector<int> start_;
  size_t nextStart_{0};
  std::vector<char> char_;
};

// Defines a macro
class Definition {
 public:
  Definition(const TokenSequence &, size_t firstToken, size_t tokens);
  Definition(const std::vector<std::string> &argNames, const TokenSequence &,
             size_t firstToken, size_t tokens);

  bool isFunctionLike() const { return isFunctionLike_; }
  size_t argumentCount() const { return argumentCount_; }
  bool isVariadic() const { return isVariadic_; }
  bool isDisabled() const { return isDisabled_; }
  const TokenSequence &replacement() const { return replacement_; }

  bool set_isDisabled(bool disable);

  TokenSequence Apply(const std::vector<TokenSequence> &args);

 private:
  static TokenSequence Tokenize(const std::vector<std::string> &argNames,
                                const TokenSequence &token, size_t firstToken,
                                size_t tokens);

  bool isFunctionLike_{false};
  size_t argumentCount_{0};
  bool isVariadic_{false};
  bool isDisabled_{false};
  TokenSequence replacement_;
};

// Preprocessing state
class Preprocessor {
 public:
  Preprocessor(Prescanner *ps) : prescanner_{ps} {}

  // When the input contains macros to be replaced, the new token sequence
  // is appended to the output and the returned value is true.  When
  // no macro replacement is necessary, the output is unmodified and the
  // return value is false.
  bool MacroReplacement(const TokenSequence &, TokenSequence *);

  // Implements a preprocessor directive; returns an error message, or an
  // empty string when successful.
  std::string Directive(const TokenSequence &);

 private:
  enum class IsElseActive { No, Yes };
  enum class CanDeadElseAppear { No, Yes };
  bool IsNameDefined(const CharPointerWithLength &);
  std::string SkipDisabledConditionalCode(const std::string &dirName,
                                          IsElseActive);
  bool IsIfPredicateTrue(const TokenSequence &expr, size_t first,
                         size_t exprTokens, std::string *errors);

  std::list<std::string> names_;
  std::unordered_map<CharPointerWithLength, Definition> definitions_;
  std::stack<CanDeadElseAppear> ifStack_;
  Prescanner *prescanner_;
};
}  // namespace Fortran
#endif  // FORTRAN_PREPROCESSOR_H_
