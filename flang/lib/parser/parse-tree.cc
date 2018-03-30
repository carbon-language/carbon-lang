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

ProcedureDesignator Designator::ConvertToProcedureDesignator() {
  return std::visit(
      visitors{
          [](ObjectName &n) -> ProcedureDesignator { return {std::move(n)}; },
          [](DataReference &dr) -> ProcedureDesignator {
            if (Name * n{std::get_if<Name>(&dr.u)}) {
              return {std::move(*n)};
            }
            StructureComponent &sc{
                *std::get<Indirection<StructureComponent>>(dr.u)};
            return {ProcComponentRef{
                Scalar<Variable>{Indirection<Designator>{std::move(sc.base)}},
                std::move(sc.component)}};
          },
          [](Substring &) -> ProcedureDesignator {
            CHECK(!"can't get here");
            return {Name{}};
          }},
      u);
}

std::optional<Call> Designator::ConvertToCall(const UserState *ustate) {
  return std::visit(
      visitors{[](ObjectName &n) -> std::optional<Call> {
                 return {Call{ProcedureDesignator{std::move(n)},
                     std::list<ActualArgSpec>{}}};
               },
          [=](DataReference &dr) -> std::optional<Call> {
            if (std::holds_alternative<Indirection<CoindexedNamedObject>>(
                    dr.u)) {
              return {};
            }
            if (Name * n{std::get_if<Name>(&dr.u)}) {
              return {Call{ProcedureDesignator{std::move(*n)},
                  std::list<ActualArgSpec>{}}};
            }
            if (auto *isc =
                    std::get_if<Indirection<StructureComponent>>(&dr.u)) {
              StructureComponent &sc{**isc};
              if (ustate &&
                  ustate->IsOldStructureComponent(sc.component.source)) {
                return {};
              }
              Variable var{Indirection<Designator>{std::move(sc.base)}};
              ProcComponentRef pcr{
                  Scalar<Variable>{std::move(var)}, std::move(sc.component)};
              return {Call{ProcedureDesignator{std::move(pcr)},
                  std::list<ActualArgSpec>{}}};
            }
            ArrayElement &ae{*std::get<Indirection<ArrayElement>>(dr.u)};
            if (std::any_of(ae.subscripts.begin(), ae.subscripts.end(),
                    [](const SectionSubscript &ss) {
                      return !ss.CanConvertToActualArgument();
                    })) {
              return {};
            }
            std::list<ActualArgSpec> args;
            for (auto &ss : ae.subscripts) {
              args.emplace_back(
                  std::optional<Keyword>{}, ss.ConvertToActualArgument());
            }
            if (Name * n{std::get_if<Name>(&ae.base.u)}) {
              return {
                  Call{ProcedureDesignator{std::move(*n)}, std::move(args)}};
            }
            StructureComponent &bsc{
                *std::get<Indirection<StructureComponent>>(ae.base.u)};
            if (ustate &&
                ustate->IsOldStructureComponent(bsc.component.source)) {
              return {};
            }
            Variable var{Indirection<Designator>{std::move(bsc.base)}};
            ProcComponentRef pcr{
                Scalar<Variable>{std::move(var)}, std::move(bsc.component)};
            return {Call{ProcedureDesignator{std::move(pcr)}, std::move(args)}};
          },
          [](const Substring &) -> std::optional<Call> { return {}; }},
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

// R920 section-subscript
bool SectionSubscript::CanConvertToActualArgument() const {
  return std::visit(visitors{[](const VectorSubscript &) { return true; },
                        [](const ScalarIntExpr &) { return true; },
                        [](const SubscriptTriplet &) { return false; }},
      u);
}

ActualArg SectionSubscript::ConvertToActualArgument() {
  return std::visit(visitors{[](VectorSubscript &vs) -> ActualArg {
                               return vs.thing->ConvertToActualArgument();
                             },
                        [](ScalarIntExpr &vs) -> ActualArg {
                          return vs.thing.thing->ConvertToActualArgument();
                        },
                        [](SubscriptTriplet &) -> ActualArg {
                          CHECK(!"can't happen");
                          return {Name{}};
                        }},
      u);
}

// R1001 - R1022 expression
Expr::Expr(Designator &&x) : u{Indirection<Designator>(std::move(x))} {}
Expr::Expr(FunctionReference &&x)
  : u{Indirection<FunctionReference>(std::move(x))} {}

std::optional<Variable> Expr::ConvertToVariable() {
  if (Indirection<Designator> *id = std::get_if<Indirection<Designator>>(&u)) {
    return {Variable{std::move(*id)}};
  }
  if (Indirection<FunctionReference> *ifr =
          std::get_if<Indirection<FunctionReference>>(&u)) {
    return {Variable{std::move(*ifr)}};
  }
  return {};
}

ActualArg Expr::ConvertToActualArgument() {
  if (std::optional<Variable> var{ConvertToVariable()}) {
    return {std::move(var.value())};
  }
  return {std::move(*this)};
}
}  // namespace parser
}  // namespace Fortran
