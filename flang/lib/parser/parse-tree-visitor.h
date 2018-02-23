#ifndef FORTRAN_PARSE_TREE_VISITOR_H_
#define FORTRAN_PARSE_TREE_VISITOR_H_

#include "format-specification.h"
#include "parse-tree.h"
#include <optional>
#include <tuple>
#include <variant>

/// Parse tree visitor
/// Call visit(x, visitor) to visit each node under x.
///
/// visitor.pre(x) is called before visiting x and its children are not
/// visited if it returns false.
///
/// visitor.post(x) is called after visiting x.

namespace Fortran {
namespace parser {

namespace {

// Apply func to each member of tuple.
template<size_t I = 0, typename Func, typename... Ts>  // clang-format off
typename std::enable_if<I == sizeof...(Ts)>::type
ForEachInTuple(const std::tuple<Ts...> &, Func) {
}

template<size_t I = 0, typename Func, typename... Ts>
typename std::enable_if<I < sizeof...(Ts)>::type
ForEachInTuple(const std::tuple<Ts...> &tuple, Func func) {
  func(std::get<I>(tuple));
  ForEachInTuple<I + 1>(tuple, func);
}  // clang-format on

// Helpers: generic visitor that is called if there is no specific one
// and visitors for std::optional, std::list, and Indirection.

template<typename T, typename V> void visit(const T &x, V &visitor) {
  if (visitor.pre(x)) {
    visitor.post(x);
  }
}

template<typename T, typename V>
void visit(const std::optional<T> &x, V &visitor) {
  if (x) {
    visit(*x, visitor);
  }
}

template<typename T, typename V> void visit(const std::list<T> &x, V &visitor) {
  for (const auto &elem : x) {
    visit(elem, visitor);
  }
}

template<typename T, typename V>
void visit(const Indirection<T> &x, V &visitor) {
  visit(*x, visitor);
}

// Visit the single field 'v' in the class
template<typename T, typename V>
void VisitWrapperClass(const T &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.v, visitor);
    visitor.post(x);
  }
}

// Visit the single tuple field 't' in the class
template<typename T, typename V> void VisitTupleClass(const T &x, V &visitor) {
  if (visitor.pre(x)) {
    ForEachInTuple(x.t, [&](const auto &y) { visit(y, visitor); });
    visitor.post(x);
  }
}

// Visit the single variant field 'u' in the class
template<typename T, typename V> void VisitUnionClass(const T &x, V &visitor) {
  if (visitor.pre(x)) {
    std::visit([&](const auto &y) { visit(y, visitor); }, x.u);
    visitor.post(x);
  }
}

}  // namespace

template<typename T, typename V> void visit(const Scalar<T> &x, V &visitor) {
  visit(x.thing, visitor);
}

template<typename T, typename V> void visit(const Constant<T> &x, V &visitor) {
  visit(x.thing, visitor);
}

template<typename T, typename V> void visit(const Integer<T> &x, V &visitor) {
  visit(x.thing, visitor);
}

template<typename T, typename V> void visit(const Logical<T> &x, V &visitor) {
  visit(x.thing, visitor);
}

template<typename T, typename V>
void visit(const DefaultChar<T> &x, V &visitor) {
  visit(x.thing, visitor);
}

template<typename T, typename V> void visit(const Statement<T> &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.statement, visitor);
    visitor.post(x);
  }
}

template<typename T, typename V>
void visit(const LoopBounds<T> &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.name, visitor);
    visit(x.lower, visitor);
    visit(x.upper, visitor);
    visit(x.step, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const AcImpliedDo &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AcImpliedDoControl &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AcSpec &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.type, visitor);
    visit(x.values, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const AcValue &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const AcValue::Triplet &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AccessId &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const AccessStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ActionStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ActualArg &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ActualArg::PercentRef &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ActualArg::PercentVal &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ActualArgSpec &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AllocOpt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const AllocOpt::Mold &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const AllocOpt::Source &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const AllocatableStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const AllocateCoarraySpec &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AllocateObject &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const AllocateShapeSpec &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AllocateStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Allocation &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AltReturnSpec &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ArithmeticIfStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ArrayConstructor &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ArrayElement &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.base, visitor);
    visit(x.subscripts, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const ArraySpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const AssignStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AssignedGotoStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AssignmentStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AssociateConstruct &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AssociateStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Association &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AssumedImpliedSpec &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const AssumedShapeSpec &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const AssumedSizeSpec &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const AsynchronousStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const AttrSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const BOZLiteralConstant &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const BackspaceStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const BasedPointerStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const BindAttr &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const BindEntity &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const BindStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const BlockConstruct &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const BlockData &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const BlockDataStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const BlockSpecificationPart &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const BlockStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const BoundsRemapping &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const BoundsSpec &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const Call &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const CallStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const CaseConstruct &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const CaseConstruct::Case &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const CaseSelector &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const CaseStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const CaseValueRange &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ChangeTeamConstruct &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ChangeTeamStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const CharLength &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const CharLiteralConstant &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V>
void visit(const CharLiteralConstantSubstring &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const CharSelector &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V>
void visit(const CharSelector::LengthAndKind &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.length, visitor);
    visit(x.kind, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const CharVariable &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const CloseStmt::CloseSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const CoarrayAssociation &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const CoarraySpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const CodimensionDecl &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const CodimensionStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const CoindexedNamedObject &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.base, visitor);
    visit(x.imageSelector, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const CommonBlockObject &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const CommonStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ComplexLiteralConstant &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ComplexPart &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ComponentArraySpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ComponentAttrSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ComponentDataSource &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ComponentDecl &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ComponentDefStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ComponentSpec &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ComputedGotoStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ConcurrentControl &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ConcurrentHeader &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ConnectSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ConnectSpec::CharExpr &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ConnectSpec::Newunit &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ConnectSpec::Recl &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ConstantValue &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ContiguousStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const CriticalConstruct &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const CriticalStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const CycleStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const DataComponentDefStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const DataIDoObject &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const DataImpliedDo &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const DataReference &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const DataStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const DataStmtConstant &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const DataStmtObject &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const DataStmtRepeat &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const DataStmtSet &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const DataStmtValue &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const DeallocateStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const DeclarationConstruct &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const DeclarationTypeSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V>
void visit(const DeclarationTypeSpec::Class &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.derived, visitor);
    visitor.post(x);
  }
}

template<typename V>
void visit(const DeclarationTypeSpec::Record &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V>
void visit(const DeclarationTypeSpec::Type &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.derived, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const DeferredCoshapeSpecList &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const DeferredShapeSpecList &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const DefinedOpName &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const DefinedOperator &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const DerivedTypeDef &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const DerivedTypeSpec &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const DerivedTypeStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Designator &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V>
void visit(const DimensionStmt::Declaration &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const DoConstruct &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const DummyArg &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ElseIfStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ElseStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ElsewhereStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndAssociateStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndBlockDataStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndBlockStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndChangeTeamStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const EndCriticalStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndDoStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndForallStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndFunctionStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndIfStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndInterfaceStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndLabel &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndModuleStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndMpSubprogramStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndProgramStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndSelectStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndSubmoduleStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndSubroutineStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndTypeStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndWhereStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EndfileStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const EntityDecl &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const EntryStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const EnumDef &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Enumerator &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const EnumeratorDefStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EorLabel &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EquivalenceObject &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EquivalenceStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ErrLabel &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const EventPostStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const EventWaitStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V>
void visit(const EventWaitStmt::EventWaitSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ExecutableConstruct &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ExecutionPartConstruct &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ExitStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ExplicitCoshapeSpec &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ExplicitShapeSpec &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ExponentPart &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Expr &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const Expr::AND &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::Add &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::ComplexConstructor &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::Concat &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::DefinedBinary &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Expr::DefinedUnary &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Expr::Divide &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::EQ &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::EQV &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::GE &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::GT &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::IntrinsicBinary &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Expr::IntrinsicUnary &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const Expr::LE &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::LT &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::Multiply &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::NE &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::NEQV &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::NOT &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicUnary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::Negate &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicUnary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::OR &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::Parentheses &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicUnary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::PercentLoc &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const Expr::Power &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::Subtract &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Expr::UnaryPlus &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(static_cast<const Expr::IntrinsicUnary &>(x), visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const ExternalStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const FileUnitNumber &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const FinalProcedureStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const FlushStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ForallAssignmentStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ForallBodyConstruct &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ForallConstruct &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ForallConstructStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ForallStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const FormTeamStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V>
void visit(const FormTeamStmt::FormTeamSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const Format &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const FormatStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const Fortran::ControlEditDesc &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.kind, visitor);
    visitor.post(x);
  }
}

template<typename V>
void visit(const Fortran::DerivedTypeDataEditDesc &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.type, visitor);
    visit(x.parameters, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Fortran::FormatItem &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V>
void visit(const Fortran::IntrinsicTypeDataEditDesc &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.kind, visitor);
    visit(x.width, visitor);
    visit(x.digits, visitor);
    visit(x.exponentWidth, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const FunctionReference &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const FunctionStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const FunctionSubprogram &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const GenericSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const GenericStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const GotoStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const IdExpr &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const IdVariable &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const IfConstruct &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const IfConstruct::ElseBlock &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const IfConstruct::ElseIfBlock &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const IfStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const IfThenStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ImageSelector &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ImageSelectorSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ImageSelectorSpec::Stat &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ImageSelectorSpec::Team &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V>
void visit(const ImageSelectorSpec::Team_Number &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ImplicitPart &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ImplicitPartStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ImplicitSpec &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ImplicitStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ImpliedShapeSpec &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ImportStmt &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.names, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Initialization &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const InputImpliedDo &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const InputItem &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const InquireSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const InquireSpec::CharVar &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const InquireSpec::IntVar &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const InquireSpec::LogVar &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const InquireStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const InquireStmt::Iolength &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const IntLiteralConstant &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const IntegerTypeSpec &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const IntentStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const InterfaceBlock &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const InterfaceBody &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const InterfaceBody::Function &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V>
void visit(const InterfaceBody::Subroutine &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const InterfaceSpecification &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const InterfaceStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const InternalSubprogram &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const InternalSubprogramPart &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const IntrinsicStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const IntrinsicTypeSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V>
void visit(const IntrinsicTypeSpec::Character &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.selector, visitor);
    visitor.post(x);
  }
}

template<typename V>
void visit(const IntrinsicTypeSpec::Complex &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.kind, visitor);
    visitor.post(x);
  }
}

template<typename V>
void visit(const IntrinsicTypeSpec::Logical &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.kind, visitor);
    visitor.post(x);
  }
}

template<typename V>
void visit(const IntrinsicTypeSpec::NCharacter &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const IntrinsicTypeSpec::Real &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.kind, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const IoControlSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V>
void visit(const IoControlSpec::Asynchronous &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const IoControlSpec::CharExpr &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const IoControlSpec::Pos &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const IoControlSpec::Rec &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const IoControlSpec::Size &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const IoUnit &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const KindParam &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const LabelDoStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const LanguageBindingSpec &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const LengthSelector &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const LetterSpec &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const LiteralConstant &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const LocalitySpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const LocalitySpec::Local &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const LocalitySpec::LocalInit &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const LocalitySpec::Shared &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const LockStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const LockStmt::LockStat &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const LogicalLiteralConstant &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const LoopControl &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const LoopControl::Concurrent &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const MainProgram &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Map &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const MaskedElsewhereStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Module &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ModuleStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ModuleSubprogram &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ModuleSubprogramPart &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const MpSubprogramStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const MsgVariable &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const NamedConstant &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const NamedConstantDef &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const NamelistStmt::Group &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const NonLabelDoStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const NullifyStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ObjectDecl &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Only &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const OpenStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const OptionalStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const OtherSpecificationStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const OutputImpliedDo &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const OutputItem &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ParameterStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ParentIdentifier &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const PartRef &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.name, visitor);
    visit(x.subscripts, visitor);
    visit(x.imageSelector, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const Pass &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const PauseStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const PointerAssignmentStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V>
void visit(const PointerAssignmentStmt::Bounds &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const PointerDecl &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const PointerObject &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const PointerStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const PositionOrFlushSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const PrefixSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const PrintStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const PrivateOrSequence &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ProcAttrSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ProcComponentAttrSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ProcComponentDefStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ProcComponentRef &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ProcDecl &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ProcInterface &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ProcPointerInit &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ProcedureDeclarationStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ProcedureDesignator &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ProcedureStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Program &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ProgramUnit &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ProtectedStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const ReadStmt &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.iounit, visitor);
    visit(x.format, visitor);
    visit(x.controls, visitor);
    visit(x.items, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const RealLiteralConstant &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.intPart, visitor);
    visit(x.fraction, visitor);
    visit(x.exponent, visitor);
    visit(x.kind, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const RedimensionStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Rename &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const Rename::Names &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Rename::Operators &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const ReturnStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const RewindStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const SaveStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const SavedEntity &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SectionSubscript &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const SelectCaseStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SelectRankCaseStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SelectRankCaseStmt::Rank &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const SelectRankConstruct &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V>
void visit(const SelectRankConstruct::RankCase &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SelectRankStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SelectTypeConstruct &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V>
void visit(const SelectTypeConstruct::TypeCase &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SelectTypeStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Selector &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const SeparateModuleSubprogram &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V>
void visit(const SignedComplexLiteralConstant &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SignedIntLiteralConstant &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V>
void visit(const SignedRealLiteralConstant &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SpecificationConstruct &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const SpecificationExpr &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const SpecificationPart &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const StatOrErrmsg &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const StatVariable &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const StatusExpr &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const StmtFunctionStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const StopCode &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const StopStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const StructureComponent &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.base, visitor);
    visit(x.component, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const StructureConstructor &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const StructureDef &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const StructureField &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const StructureStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Submodule &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SubmoduleStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SubroutineStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SubroutineSubprogram &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SubscriptTriplet &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Substring &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SubstringRange &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const Suffix &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.binding, visitor);
    visit(x.resultName, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const SyncAllStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const SyncImagesStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const SyncImagesStmt::ImageSet &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const SyncMemoryStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const SyncTeamStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const TargetStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const TypeAttrSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const TypeAttrSpec::Extends &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.name, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const TypeBoundGenericStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const TypeBoundProcBinding &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const TypeBoundProcDecl &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const TypeBoundProcedurePart &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const TypeBoundProcedureStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V>
void visit(const TypeBoundProcedureStmt::WithInterface &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.interfaceName, visitor);
    visit(x.attributes, visitor);
    visit(x.bindingNames, visitor);
    visitor.post(x);
  }
}

template<typename V>
void visit(const TypeBoundProcedureStmt::WithoutInterface &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.attributes, visitor);
    visit(x.declarations, visitor);
    visitor.post(x);
  }
}

template<typename V> void visit(const TypeDeclarationStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const TypeGuardStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const TypeGuardStmt::Guard &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const TypeParamDecl &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const TypeParamDefStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const TypeParamSpec &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const TypeParamValue &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const TypeSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const Union &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const UnlockStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const UseStmt &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const ValueStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const Variable &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const VolatileStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const WaitSpec &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const WaitStmt &x, V &visitor) {
  VisitWrapperClass(x, visitor);
}

template<typename V> void visit(const WhereBodyConstruct &x, V &visitor) {
  VisitUnionClass(x, visitor);
}

template<typename V> void visit(const WhereConstruct &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V>
void visit(const WhereConstruct::Elsewhere &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V>
void visit(const WhereConstruct::MaskedElsewhere &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const WhereConstructStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const WhereStmt &x, V &visitor) {
  VisitTupleClass(x, visitor);
}

template<typename V> void visit(const WriteStmt &x, V &visitor) {
  if (visitor.pre(x)) {
    visit(x.iounit, visitor);
    visit(x.format, visitor);
    visit(x.controls, visitor);
    visit(x.items, visitor);
    visitor.post(x);
  }
}

}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSE_TREE_VISITOR_H_
