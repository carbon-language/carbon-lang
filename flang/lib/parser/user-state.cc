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

std::optional<Success> EnterNonlabelDoConstruct::Parse(ParseState &state) {
  if (auto ustate = state.userState()) {
    ustate->EnterNonlabelDoConstruct();
  }
  return {Success{}};
}

std::optional<Success> LeaveDoConstruct::Parse(ParseState &state) {
  if (auto ustate = state.userState()) {
    ustate->LeaveDoConstruct();
  }
  return {Success{}};
}

std::optional<Name> OldStructureComponentName::Parse(ParseState &state) {
  if (std::optional<Name> n{name.Parse(state)}) {
    if (const auto *ustate = state.userState()) {
      if (ustate->IsOldStructureComponent(n->source)) {
        return n;
      }
    }
  }
  return {};
}

std::optional<DataComponentDefStmt> StructureComponents::Parse(
    ParseState &state) {
  static constexpr auto stmt = Parser<DataComponentDefStmt>{};
  std::optional<DataComponentDefStmt> defs{stmt.Parse(state)};
  if (defs.has_value()) {
    if (auto ustate = state.userState()) {
      for (const auto &decl : std::get<std::list<ComponentDecl>>(defs->t)) {
        ustate->NoteOldStructureComponent(std::get<Name>(decl.t).source);
      }
    }
  }
  return defs;
}
}  // namespace parser
}  // namespace Fortran
