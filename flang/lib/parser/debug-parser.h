#ifndef FORTRAN_PARSER_DEBUG_PARSER_H_
#define FORTRAN_PARSER_DEBUG_PARSER_H_

// Implements the parser with syntax "(YOUR MESSAGE HERE)"_debug for use
// in temporary modifications to the grammar intended for tracing the
// flow of the parsers.  Not to be used in production.

#include "basic-parsers.h"
#include "parse-state.h"
#include <cstddef>
#include <iostream>
#include <optional>
#include <string>

namespace Fortran {
namespace parser {

class DebugParser {
public:
  using resultType = Success;
  constexpr DebugParser(const DebugParser &) = default;
  constexpr DebugParser(const char *str, std::size_t n)
    : str_{str}, length_{n} {}
  std::optional<Success> Parse(ParseState *state) const {
    const CookedSource &cooked{state->messages().cooked()};
    if (auto context = state->context()) {
      context->Emit(std::cout, cooked);
    }
    Provenance p{cooked.GetProvenance(state->GetLocation()).start()};
    cooked.allSources().Identify(std::cout, p, "", true);
    std::cout << "   parser debug: " << std::string{str_, length_} << "\n\n";
    return {Success{}};
  }

private:
  const char *const str_;
  std::size_t length_;
};

constexpr DebugParser operator""_debug(const char str[], std::size_t n) {
  return DebugParser{str, n};
}
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_DEBUG_PARSER_H_
