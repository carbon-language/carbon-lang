#ifndef FORTRAN_PARSER_CHAR_BLOCK_H_
#define FORTRAN_PARSER_CHAR_BLOCK_H_

// Describes a contiguous block of characters; does not own their storage.

#include "interval.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

namespace Fortran {
namespace parser {

class CharBlock {
public:
  CharBlock() {}
  CharBlock(const char *x, std::size_t n = 1) : interval_{x, n} {}
  CharBlock(const char *b, const char *e)
    : interval_{b, static_cast<std::size_t>(e - b)} {}
  CharBlock(const std::string &s) : interval_{s.data(), s.size()} {}
  CharBlock(const CharBlock &) = default;
  CharBlock(CharBlock &&) = default;
  CharBlock &operator=(const CharBlock &) = default;
  CharBlock &operator=(CharBlock &&) = default;

  bool empty() const { return interval_.empty(); }
  std::size_t size() const { return interval_.size(); }
  const char *begin() const { return interval_.start(); }
  const char *end() const { return interval_.start() + interval_.size(); }
  const char &operator[](std::size_t j) const { return interval_.start()[j]; }

  bool IsBlank() const {
    for (char ch : *this) {
      if (ch != ' ' && ch != '\t') {
        return false;
      }
    }
    return true;
  }

  std::string ToString() const {
    return std::string{interval_.start(), interval_.size()};
  }

  bool operator<(const CharBlock &that) const { return Compare(that) < 0; }
  bool operator<=(const CharBlock &that) const { return Compare(that) <= 0; }
  bool operator==(const CharBlock &that) const { return Compare(that) == 0; }
  bool operator!=(const CharBlock &that) const { return Compare(that) != 0; }
  bool operator>=(const CharBlock &that) const { return Compare(that) >= 0; }
  bool operator>(const CharBlock &that) const { return Compare(that) > 0; }

private:
  int Compare(const CharBlock &that) const {
    std::size_t bytes{std::min(size(), that.size())};
    int cmp{std::memcmp(static_cast<const void *>(begin()),
        static_cast<const void *>(that.begin()), bytes)};
    if (cmp != 0) {
      return cmp;
    }
    return size() < that.size() ? -1 : size() > that.size();
  }

  Interval<const char *> interval_{nullptr, 0};
};
}  // namespace parser
}  // namespace Fortran

// Specializations to enable std::unordered_map<CharBlock, ...> &c.
template<> struct std::hash<Fortran::parser::CharBlock> {
  std::size_t operator()(const Fortran::parser::CharBlock &x) const {
    std::size_t hash{0}, bytes{x.size()};
    for (std::size_t j{0}; j < bytes; ++j) {
      hash = (hash * 31) ^ x[j];
    }
    return hash;
  }
};
#endif  // FORTRAN_PARSER_CHAR_BLOCK_H_
