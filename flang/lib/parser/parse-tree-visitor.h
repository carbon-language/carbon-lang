#ifndef FORTRAN_PARSER_PARSE_TREE_VISITOR_H_
#define FORTRAN_PARSER_PARSE_TREE_VISITOR_H_

#include "format-specification.h"
#include "parse-tree.h"
#include <optional>
#include <tuple>
#include <variant>

/// Parse tree visitor
/// Call Walk(x, visitor) to visit each node under x.
///
/// visitor.Pre(x) is called before visiting x and its children are not
/// visited if it returns false.
///
/// visitor.Post(x) is called after visiting x.

namespace Fortran {
namespace parser {

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
template<typename T, typename V> void Walk(const T &x, V &visitor) {
  if (visitor.Pre(x)) {
    visitor.Post(x);
  }
}
template<typename T, typename V>
void Walk(const std::optional<T> &x, V &visitor) {
  if (x) {
    Walk(*x, visitor);
  }
}
template<typename T, typename V> void Walk(const std::list<T> &x, V &visitor) {
  for (const auto &elem : x) {
    Walk(elem, visitor);
  }
}
template<typename T, typename V>
void Walk(const Indirection<T> &x, V &visitor) {
  Walk(*x, visitor);
}

// Walk a class with a single field 'v'.
template<typename T, typename V> void WalkWrapperClass(const T &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.v, visitor);
    visitor.Post(x);
  }
}

// Walk a class with a single tuple field 't'.
template<typename T, typename V> void WalkTupleClass(const T &x, V &visitor) {
  if (visitor.Pre(x)) {
    ForEachInTuple(x.t, [&](const auto &y) { Walk(y, visitor); });
    visitor.Post(x);
  }
}

// Walk a class with a single variant field 'u'.
template<typename T, typename V> void WalkUnionClass(const T &x, V &visitor) {
  if (visitor.Pre(x)) {
    std::visit([&](const auto &y) { Walk(y, visitor); }, x.u);
    visitor.Post(x);
  }
}

// Walk a class with a single field 'thing'.
template<typename T, typename V> void Walk(const Scalar<T> &x, V &visitor) {
  Walk(x.thing, visitor);
}
template<typename T, typename V> void Walk(const Constant<T> &x, V &visitor) {
  Walk(x.thing, visitor);
}
template<typename T, typename V> void Walk(const Integer<T> &x, V &visitor) {
  Walk(x.thing, visitor);
}
template<typename T, typename V> void Walk(const Logical<T> &x, V &visitor) {
  Walk(x.thing, visitor);
}
template<typename T, typename V>
void Walk(const DefaultChar<T> &x, V &visitor) {
  Walk(x.thing, visitor);
}

template<typename T, typename V> void Walk(const Statement<T> &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.statement, visitor);
    visitor.Post(x);
  }
}

template<typename T, typename V> void Walk(const LoopBounds<T> &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.name, visitor);
    Walk(x.lower, visitor);
    Walk(x.upper, visitor);
    Walk(x.step, visitor);
    visitor.Post(x);
  }
}

template<typename V> void Walk(const AcSpec &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.type, visitor);
    Walk(x.values, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const ArrayElement &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.base, visitor);
    Walk(x.subscripts, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const CharSelector::LengthAndKind &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.length, visitor);
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const CoindexedNamedObject &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.base, visitor);
    Walk(x.imageSelector, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const DeclarationTypeSpec::Class &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.derived, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const DeclarationTypeSpec::Type &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.derived, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::AND &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::Add &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::ComplexConstructor &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::Concat &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::Divide &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::EQ &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::EQV &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::GE &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::GT &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::LE &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::LT &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::Multiply &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::NE &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::NEQV &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::NOT &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicUnary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::Negate &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicUnary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::OR &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::Parentheses &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicUnary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::Power &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::Subtract &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicBinary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Expr::UnaryPlus &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(static_cast<const Expr::IntrinsicUnary &>(x), visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Fortran::ControlEditDesc &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const Fortran::DerivedTypeDataEditDesc &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.type, visitor);
    Walk(x.parameters, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Fortran::FormatItem &x, V &visitor) {
  Walk(x.repeatCount, visitor);
  WalkUnionClass(x, visitor);
}
template<typename V>
void Walk(const Fortran::IntrinsicTypeDataEditDesc &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    Walk(x.width, visitor);
    Walk(x.digits, visitor);
    Walk(x.exponentWidth, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const ImportStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.names, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const IntrinsicTypeSpec::Character &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.selector, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const IntrinsicTypeSpec::Complex &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const IntrinsicTypeSpec::Logical &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const IntrinsicTypeSpec::Real &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const PartRef &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.name, visitor);
    Walk(x.subscripts, visitor);
    Walk(x.imageSelector, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const ReadStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.iounit, visitor);
    Walk(x.format, visitor);
    Walk(x.controls, visitor);
    Walk(x.items, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const RealLiteralConstant &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.intPart, visitor);
    Walk(x.fraction, visitor);
    Walk(x.exponent, visitor);
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const StructureComponent &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.base, visitor);
    Walk(x.component, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const Suffix &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.binding, visitor);
    Walk(x.resultName, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const TypeAttrSpec::Extends &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.name, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const TypeBoundProcedureStmt::WithInterface &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.interfaceName, visitor);
    Walk(x.attributes, visitor);
    Walk(x.bindingNames, visitor);
    visitor.Post(x);
  }
}
template<typename V>
void Walk(const TypeBoundProcedureStmt::WithoutInterface &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.attributes, visitor);
    Walk(x.declarations, visitor);
    visitor.Post(x);
  }
}
template<typename V> void Walk(const UseStmt &x, V &visitor) {
  Walk(x.nature, visitor);
  Walk(x.moduleName, visitor);
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const WriteStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.iounit, visitor);
    Walk(x.format, visitor);
    Walk(x.controls, visitor);
    Walk(x.items, visitor);
    visitor.Post(x);
  }
}

// tuple classes
template<typename V> void Walk(const AcImpliedDo &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AcImpliedDoControl &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AcValue::Triplet &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AccessStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ActualArgSpec &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AllocateCoarraySpec &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AllocateShapeSpec &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AllocateStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Allocation &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ArithmeticIfStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AssignStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AssignedGotoStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AssignmentStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AssociateConstruct &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AssociateStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Association &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const AssumedSizeSpec &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const BasedPointerStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const BindEntity &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const BindStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const BlockConstruct &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const BlockData &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const BoundsRemapping &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Call &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const CaseConstruct &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const CaseConstruct::Case &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const CaseStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ChangeTeamConstruct &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ChangeTeamStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const CharLiteralConstant &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V>
void Walk(const CharLiteralConstantSubstring &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const CoarrayAssociation &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const CodimensionDecl &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const CommonBlockObject &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const CommonStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ComplexLiteralConstant &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ComponentDecl &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ComponentSpec &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ComputedGotoStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ConcurrentControl &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ConcurrentHeader &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ConnectSpec::CharExpr &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const CriticalConstruct &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const CriticalStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const DataComponentDefStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const DataImpliedDo &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const DataStmtSet &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const DataStmtValue &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const DeallocateStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const DerivedTypeDef &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const DerivedTypeSpec &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const DerivedTypeStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V>
void Walk(const DimensionStmt::Declaration &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const DoConstruct &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ElseIfStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const EndChangeTeamStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const EntityDecl &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const EntryStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const EnumDef &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Enumerator &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const EventPostStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const EventWaitStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ExplicitCoshapeSpec &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ExplicitShapeSpec &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ExponentPart &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Expr::DefinedBinary &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Expr::DefinedUnary &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Expr::IntrinsicBinary &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ForallConstruct &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ForallConstructStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ForallStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const FormTeamStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const FunctionStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const FunctionSubprogram &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const GenericStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const IfConstruct &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const IfConstruct::ElseBlock &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const IfConstruct::ElseIfBlock &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const IfStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const IfThenStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ImageSelector &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ImplicitSpec &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const InputImpliedDo &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const InquireSpec::CharVar &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const InquireSpec::IntVar &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const InquireSpec::LogVar &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const InquireStmt::Iolength &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const IntLiteralConstant &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const IntentStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const InterfaceBlock &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const InterfaceBody::Function &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const InterfaceBody::Subroutine &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const InternalSubprogramPart &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const IoControlSpec::CharExpr &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const LabelDoStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const LetterSpec &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const LockStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const LoopControl::Concurrent &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const MainProgram &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Map &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const MaskedElsewhereStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Module &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ModuleSubprogramPart &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const NamedConstantDef &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const NamelistStmt::Group &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const NonLabelDoStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ObjectDecl &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const OutputImpliedDo &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ParentIdentifier &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const PointerAssignmentStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const PointerDecl &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const PrintStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ProcComponentDefStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ProcComponentRef &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ProcDecl &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ProcedureDeclarationStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const ProcedureStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const RedimensionStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Rename::Names &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Rename::Operators &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SavedEntity &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SelectCaseStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SelectRankCaseStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SelectRankConstruct &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V>
void Walk(const SelectRankConstruct::RankCase &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SelectRankStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SelectTypeConstruct &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V>
void Walk(const SelectTypeConstruct::TypeCase &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SelectTypeStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SeparateModuleSubprogram &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V>
void Walk(const SignedComplexLiteralConstant &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SignedIntLiteralConstant &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SignedRealLiteralConstant &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SpecificationPart &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const StmtFunctionStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const StopStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const StructureConstructor &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const StructureDef &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const StructureStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Submodule &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SubmoduleStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SubroutineStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SubroutineSubprogram &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SubscriptTriplet &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Substring &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SubstringRange &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SyncImagesStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const SyncTeamStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const TypeBoundGenericStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const TypeBoundProcDecl &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const TypeBoundProcedurePart &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const TypeDeclarationStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const TypeGuardStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const TypeParamDecl &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const TypeParamDefStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const TypeParamSpec &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const Union &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const UnlockStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const WhereConstruct &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const WhereConstruct::Elsewhere &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V>
void Walk(const WhereConstruct::MaskedElsewhere &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const WhereConstructStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}
template<typename V> void Walk(const WhereStmt &x, V &visitor) {
  WalkTupleClass(x, visitor);
}

// union classes
template<typename V> void Walk(const AcValue &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const AccessId &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ActionStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ActualArg &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const AllocOpt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const AllocateObject &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ArraySpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const AttrSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const BackspaceStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const BindAttr &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const CaseSelector &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const CaseValueRange &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const CharLength &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const CharSelector &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const CloseStmt::CloseSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const CoarraySpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ComplexPart &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ComponentArraySpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ComponentAttrSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ComponentDefStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ConnectSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ConstantValue &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const DataIDoObject &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const DataReference &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const DataStmtConstant &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const DataStmtObject &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const DataStmtRepeat &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const DeclarationConstruct &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const DeclarationTypeSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const DefinedOperator &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const Designator &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const DummyArg &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const EndfileStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V>
void Walk(const EventWaitStmt::EventWaitSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ExecutableConstruct &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ExecutionPartConstruct &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const Expr &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const FlushStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ForallAssignmentStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ForallBodyConstruct &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V>
void Walk(const FormTeamStmt::FormTeamSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const Format &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const GenericSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ImageSelectorSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ImplicitPartStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ImplicitStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const Initialization &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const InputItem &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const InquireSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const InquireStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const InterfaceBody &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const InterfaceSpecification &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const InterfaceStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const InternalSubprogram &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const IntrinsicTypeSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const IoControlSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const IoUnit &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const KindParam &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const LengthSelector &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const LiteralConstant &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const LocalitySpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const LockStmt::LockStat &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const LoopControl &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ModuleSubprogram &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const Only &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const OtherSpecificationStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const OutputItem &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V>
void Walk(const PointerAssignmentStmt::Bounds &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const PointerObject &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const PositionOrFlushSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const PrefixSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const PrivateOrSequence &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ProcAttrSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ProcComponentAttrSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ProcInterface &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ProcPointerInit &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ProcedureDesignator &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const ProgramUnit &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const Rename &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const RewindStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const SectionSubscript &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const SelectRankCaseStmt::Rank &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const Selector &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const SpecificationConstruct &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const StatOrErrmsg &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const StopCode &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const StructureField &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const SyncImagesStmt::ImageSet &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const TypeAttrSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const TypeBoundProcBinding &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const TypeBoundProcedureStmt &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const TypeGuardStmt::Guard &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const TypeParamValue &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const TypeSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const Variable &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const WaitSpec &x, V &visitor) {
  WalkUnionClass(x, visitor);
}
template<typename V> void Walk(const WhereBodyConstruct &x, V &visitor) {
  WalkUnionClass(x, visitor);
}

// wrapper classes
template<typename V> void Walk(const AccessSpec &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ActualArg::PercentRef &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ActualArg::PercentVal &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const AllocOpt::Mold &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const AllocOpt::Source &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const AllocatableStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const AltReturnSpec &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ArrayConstructor &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ArraySection &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const AssumedImpliedSpec &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const AssumedShapeSpec &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const AsynchronousStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const BOZLiteralConstant &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const BlockDataStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const BlockSpecificationPart &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const BlockStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const BoundsSpec &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const CallStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const CharVariable &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const CloseStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const CodimensionStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ComplexPartDesignator &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ComponentDataSource &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ConnectSpec::Newunit &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ConnectSpec::Recl &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ContiguousStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const CycleStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const DataStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V>
void Walk(const DeclarationTypeSpec::Record &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const DeferredCoshapeSpecList &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const DeferredShapeSpecList &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const DefinedOpName &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const DimensionStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ElseStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ElsewhereStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndAssociateStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndBlockDataStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndBlockStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndCriticalStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndDoStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndForallStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndFunctionStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndIfStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndInterfaceStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndLabel &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndModuleStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndMpSubprogramStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndProgramStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndSelectStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndSubmoduleStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndSubroutineStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndTypeStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EndWhereStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EnumeratorDefStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EorLabel &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EquivalenceObject &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const EquivalenceStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ErrLabel &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ExitStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const Expr::IntrinsicUnary &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const Expr::PercentLoc &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ExternalStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const FileUnitNumber &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const FinalProcedureStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const FormatStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const FunctionReference &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const GotoStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const HollerithLiteralConstant &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const IdExpr &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const IdVariable &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ImageSelectorSpec::Stat &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ImageSelectorSpec::Team &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V>
void Walk(const ImageSelectorSpec::Team_Number &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ImplicitPart &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ImpliedShapeSpec &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const IntegerTypeSpec &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const IntentSpec &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const IntrinsicStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V>
void Walk(const IntrinsicTypeSpec::NCharacter &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V>
void Walk(const IoControlSpec::Asynchronous &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const IoControlSpec::Pos &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const IoControlSpec::Rec &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const IoControlSpec::Size &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const KindSelector &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const LanguageBindingSpec &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const LocalitySpec::Local &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const LocalitySpec::LocalInit &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const LocalitySpec::Shared &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const LogicalLiteralConstant &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ModuleStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const MpSubprogramStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const MsgVariable &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const NamedConstant &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const NamelistStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const NullifyStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const OpenStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const OptionalStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ParameterStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const Pass &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const PauseStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const PointerStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const Program &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ProtectedStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ReturnStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const SaveStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const SpecificationExpr &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const StatVariable &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const StatusExpr &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const SyncAllStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const SyncMemoryStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const TargetStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const TypeParamInquiry &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const ValueStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const VolatileStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}
template<typename V> void Walk(const WaitStmt &x, V &visitor) {
  WalkWrapperClass(x, visitor);
}

}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_PARSE_TREE_VISITOR_H_
