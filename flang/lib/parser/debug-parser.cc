#include "debug-parser.h"
#include <iostream>
#include <string>

namespace Fortran {
namespace parser {

std::optional<Success> DebugParser::Parse(ParseState &state) const {
  if (auto ustate = state.userState()) {
    const CookedSource &cooked{ustate->cooked()};
    if (auto context = state.context()) {
      context->Emit(std::cout, cooked);
    }
    Provenance p{cooked.GetProvenance(state.GetLocation()).start()};
    cooked.allSources().Identify(std::cout, p, "", true);
    std::cout << "   parser debug: " << std::string{str_, length_} << "\n\n";
  }
  return {Success{}};
}
}  // namespace parser
}  // namespace Fortran
