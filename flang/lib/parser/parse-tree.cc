#include "parse-tree.h"
#include "idioms.h"
#include "indirection.h"
#include "user-state.h"
#include <algorithm>

namespace Fortran {
namespace parser {

// R867
ImportStmt::ImportStmt(Kind &&k, std::list<Name> &&n)
  : kind{k}, names(std::move(n)) {
  CHECK(kind == Kind::Default || kind == Kind::Only || names.empty());
}

// R901 designator
bool Designator::EndsInBareName() const {
  return std::visit(
      visitors{[](const ObjectName &) { return true; },
          [](const DataReference &dr) {
            return std::holds_alternative<Name>(dr.u) ||
                std::holds_alternative<Indirection<StructureComponent>>(dr.u);
          },
          [](const Substring &) { return false; }},
      u);
}

// R911 data-ref -> part-ref [% part-ref]...
DataReference::DataReference(std::list<PartRef> &&prl)
  : u{std::move(prl.front().name)} {
  for (bool first{true}; !prl.empty(); first = false, prl.pop_front()) {
    PartRef &pr{prl.front()};
    if (!first) {
      u = Indirection<StructureComponent>{std::move(*this), std::move(pr.name)};
    }
    if (!pr.subscripts.empty()) {
      u = Indirection<ArrayElement>{std::move(*this), std::move(pr.subscripts)};
    }
    if (pr.imageSelector.has_value()) {
      u = Indirection<CoindexedNamedObject>{
          std::move(*this), std::move(*pr.imageSelector)};
    }
  }
}

// R1001 - R1022 expression
Expr::Expr(Designator &&x) : u{Indirection<Designator>(std::move(x))} {}
Expr::Expr(FunctionReference &&x)
  : u{Indirection<FunctionReference>(std::move(x))} {}
}  // namespace parser
}  // namespace Fortran
