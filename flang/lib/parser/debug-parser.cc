#include "debug-parser.h"
#include "user-state.h"
#include <ostream>
#include <string>

namespace Fortran {
namespace parser {

std::optional<Success> DebugParser::Parse(ParseState &state) const {
  if (auto ustate = state.userState()) {
    if (auto out = ustate->debugOutput()) {
      const CookedSource &cooked{ustate->cooked()};
      if (auto context = state.context()) {
        context->Emit(*out, cooked);
      }
      Provenance p{cooked.GetProvenance(state.GetLocation()).start()};
      cooked.allSources().Identify(*out, p, "", true);
      *out << "   parser debug: " << std::string{str_, length_} << "\n\n";
    }
  }
  return {Success{}};
}
}  // namespace parser
}  // namespace Fortran
