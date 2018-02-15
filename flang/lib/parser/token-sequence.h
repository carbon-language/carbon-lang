#ifndef FORTRAN_TOKEN_SEQUENCE_H_
#define FORTRAN_TOKEN_SEQUENCE_H_

// A buffer class capable of holding a contiguous sequence of characters
// that have been partitioned into preprocessing tokens.

#include "provenance.h"
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace Fortran {
namespace parser {

// Just a const char pointer with an associated length; does not presume
// to own the referenced data.  Used to describe buffered tokens and hash
// table keys.
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
  void Put(const TokenSequence &, ProvenanceRange);
  void Put(const TokenSequence &, size_t at, size_t tokens = 1);
  void Put(const char *, size_t, Provenance);
  void Put(const CharPointerWithLength &, Provenance);
  void Put(const std::string &, Provenance);
  void Put(const std::stringstream &, Provenance);
  void EmitWithCaseConversion(CookedSource *) const;
  std::string ToString() const;
  Provenance GetTokenProvenance(size_t token, size_t offset = 0) const;
  ProvenanceRange GetTokenProvenanceRange(
      size_t token, size_t offset = 0) const;
  ProvenanceRange GetIntervalProvenanceRange(
      size_t token, size_t tokens = 1) const;
  ProvenanceRange GetProvenanceRange() const;

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
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_TOKEN_SEQUENCE_H_
