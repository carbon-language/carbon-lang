#ifndef FORTRAN_DEBUG_PARSER_H_
#define FORTRAN_DEBUG_PARSER_H_

// Implements the parser with syntax "(YOUR MESSAGE HERE)"_debug for use
// in temporary modifications to the grammar intended for tracing the
// flow of the parsers.  Not to be used in production.

#include "basic-parsers.h"
#include "parse-state.h"
#include <iostream>
#include <optional>
#include <string>

namespace Fortran {

class DebugParser {
 public:
  using resultType = Success;
  constexpr DebugParser(const DebugParser &) = default;
  constexpr DebugParser(const char *str, size_t n) : str_{str}, length_{n} {}
  std::optional<Success> Parse(ParseState *state) const {
    if (auto context = state->context()) {
      std::cout << *context;
    }
    std::cout << state->position() << ' ' << std::string{str_, length_} << '\n';
    return {Success{}};
  }
 private:
  const char *const str_;
  size_t length_;
};

constexpr DebugParser operator""_debug(const char str[], size_t n) {
  return DebugParser{str, n};
}
}  // namespace Fortran
#endif  // FORTRAN_DEBUG_PARSER_H_
