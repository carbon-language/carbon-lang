#include "user-state.h"
#include "basic-parsers.h"
#include "grammar.h"
#include "parse-state.h"
#include "stmt-parser.h"
#include "type-parsers.h"
#include <optional>

namespace Fortran {
namespace parser {

std::optional<Success> StartNewSubprogram::Parse(ParseState &state) {
  if (auto ustate = state.userState()) {
    ustate->NewSubprogram();
  }
  return {Success{}};
}

std::optional<CapturedLabelDoStmt::resultType> CapturedLabelDoStmt::Parse(
    ParseState &state) {
  static constexpr auto parser = statement(indirect(Parser<LabelDoStmt>{}));
  auto result = parser.Parse(state);
  if (result) {
    if (auto ustate = state.userState()) {
      ustate->NewDoLabel(std::get<Label>(result->statement->t));
    }
  }
  return result;
}

std::optional<EndDoStmtForCapturedLabelDoStmt::resultType>
EndDoStmtForCapturedLabelDoStmt::Parse(ParseState &state) {
  static constexpr auto parser = statement(indirect(Parser<EndDoStmt>{}));
  if (auto enddo = parser.Parse(state)) {
    if (enddo->label.has_value()) {
      if (auto ustate = state.userState()) {
        if (!ustate->InNonlabelDoConstruct() &&
            ustate->IsDoLabel(enddo->label.value())) {
          return enddo;
        }
      }
    }
  }
  return {};
}

}  // namespace parser
}  // namespace Fortran
