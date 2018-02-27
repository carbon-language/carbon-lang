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
  if (visitor.Pre(x)) {
    Walk(x.repeatCount, visitor);
    std::visit([&](const auto &y) { Walk(y, visitor); }, x.u);
    visitor.Post(x);
  }
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
  if (visitor.Pre(x)) {
    Walk(x.nature, visitor);
    Walk(x.moduleName, visitor);
    std::visit([&](const auto &y) { Walk(y, visitor); }, x.u);
    visitor.Post(x);
  }
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

// Walk a class with a single tuple field 't'.
#define WALK_TUPLE_CLASS(classname) \
  template<typename V> void Walk(const classname &x, V &visitor) { \
    if (visitor.Pre(x)) { \
      ForEachInTuple(x.t, [&](const auto &y) { Walk(y, visitor); }); \
      visitor.Post(x); \
    } \
  }
WALK_TUPLE_CLASS(AcImpliedDo)
WALK_TUPLE_CLASS(AcImpliedDoControl)
WALK_TUPLE_CLASS(AcValue::Triplet)
WALK_TUPLE_CLASS(AccessStmt)
WALK_TUPLE_CLASS(ActualArgSpec)
WALK_TUPLE_CLASS(AllocateCoarraySpec)
WALK_TUPLE_CLASS(AllocateShapeSpec)
WALK_TUPLE_CLASS(AllocateStmt)
WALK_TUPLE_CLASS(Allocation)
WALK_TUPLE_CLASS(ArithmeticIfStmt)
WALK_TUPLE_CLASS(AssignStmt)
WALK_TUPLE_CLASS(AssignedGotoStmt)
WALK_TUPLE_CLASS(AssignmentStmt)
WALK_TUPLE_CLASS(AssociateConstruct)
WALK_TUPLE_CLASS(AssociateStmt)
WALK_TUPLE_CLASS(Association)
WALK_TUPLE_CLASS(AssumedSizeSpec)
WALK_TUPLE_CLASS(BasedPointerStmt)
WALK_TUPLE_CLASS(BindEntity)
WALK_TUPLE_CLASS(BindStmt)
WALK_TUPLE_CLASS(BlockConstruct)
WALK_TUPLE_CLASS(BlockData)
WALK_TUPLE_CLASS(BoundsRemapping)
WALK_TUPLE_CLASS(Call)
WALK_TUPLE_CLASS(CaseConstruct)
WALK_TUPLE_CLASS(CaseConstruct::Case)
WALK_TUPLE_CLASS(CaseStmt)
WALK_TUPLE_CLASS(ChangeTeamConstruct)
WALK_TUPLE_CLASS(ChangeTeamStmt)
WALK_TUPLE_CLASS(CharLiteralConstant)
WALK_TUPLE_CLASS(CharLiteralConstantSubstring)
WALK_TUPLE_CLASS(CoarrayAssociation)
WALK_TUPLE_CLASS(CodimensionDecl)
WALK_TUPLE_CLASS(CommonBlockObject)
WALK_TUPLE_CLASS(CommonStmt)
WALK_TUPLE_CLASS(ComplexLiteralConstant)
WALK_TUPLE_CLASS(ComponentDecl)
WALK_TUPLE_CLASS(ComponentSpec)
WALK_TUPLE_CLASS(ComputedGotoStmt)
WALK_TUPLE_CLASS(ConcurrentControl)
WALK_TUPLE_CLASS(ConcurrentHeader)
WALK_TUPLE_CLASS(ConnectSpec::CharExpr)
WALK_TUPLE_CLASS(CriticalConstruct)
WALK_TUPLE_CLASS(CriticalStmt)
WALK_TUPLE_CLASS(DataComponentDefStmt)
WALK_TUPLE_CLASS(DataImpliedDo)
WALK_TUPLE_CLASS(DataStmtSet)
WALK_TUPLE_CLASS(DataStmtValue)
WALK_TUPLE_CLASS(DeallocateStmt)
WALK_TUPLE_CLASS(DerivedTypeDef)
WALK_TUPLE_CLASS(DerivedTypeSpec)
WALK_TUPLE_CLASS(DerivedTypeStmt)
WALK_TUPLE_CLASS(DimensionStmt::Declaration)
WALK_TUPLE_CLASS(DoConstruct)
WALK_TUPLE_CLASS(ElseIfStmt)
WALK_TUPLE_CLASS(EndChangeTeamStmt)
WALK_TUPLE_CLASS(EntityDecl)
WALK_TUPLE_CLASS(EntryStmt)
WALK_TUPLE_CLASS(EnumDef)
WALK_TUPLE_CLASS(Enumerator)
WALK_TUPLE_CLASS(EventPostStmt)
WALK_TUPLE_CLASS(EventWaitStmt)
WALK_TUPLE_CLASS(ExplicitCoshapeSpec)
WALK_TUPLE_CLASS(ExplicitShapeSpec)
WALK_TUPLE_CLASS(ExponentPart)
WALK_TUPLE_CLASS(Expr::DefinedBinary)
WALK_TUPLE_CLASS(Expr::DefinedUnary)
WALK_TUPLE_CLASS(Expr::IntrinsicBinary)
WALK_TUPLE_CLASS(ForallConstruct)
WALK_TUPLE_CLASS(ForallConstructStmt)
WALK_TUPLE_CLASS(ForallStmt)
WALK_TUPLE_CLASS(FormTeamStmt)
WALK_TUPLE_CLASS(FunctionStmt)
WALK_TUPLE_CLASS(FunctionSubprogram)
WALK_TUPLE_CLASS(GenericStmt)
WALK_TUPLE_CLASS(IfConstruct)
WALK_TUPLE_CLASS(IfConstruct::ElseBlock)
WALK_TUPLE_CLASS(IfConstruct::ElseIfBlock)
WALK_TUPLE_CLASS(IfStmt)
WALK_TUPLE_CLASS(IfThenStmt)
WALK_TUPLE_CLASS(ImageSelector)
WALK_TUPLE_CLASS(ImplicitSpec)
WALK_TUPLE_CLASS(InputImpliedDo)
WALK_TUPLE_CLASS(InquireSpec::CharVar)
WALK_TUPLE_CLASS(InquireSpec::IntVar)
WALK_TUPLE_CLASS(InquireSpec::LogVar)
WALK_TUPLE_CLASS(InquireStmt::Iolength)
WALK_TUPLE_CLASS(IntLiteralConstant)
WALK_TUPLE_CLASS(IntentStmt)
WALK_TUPLE_CLASS(InterfaceBlock)
WALK_TUPLE_CLASS(InterfaceBody::Function)
WALK_TUPLE_CLASS(InterfaceBody::Subroutine)
WALK_TUPLE_CLASS(InternalSubprogramPart)
WALK_TUPLE_CLASS(IoControlSpec::CharExpr)
WALK_TUPLE_CLASS(LabelDoStmt)
WALK_TUPLE_CLASS(LetterSpec)
WALK_TUPLE_CLASS(LockStmt)
WALK_TUPLE_CLASS(LoopControl::Concurrent)
WALK_TUPLE_CLASS(MainProgram)
WALK_TUPLE_CLASS(Map)
WALK_TUPLE_CLASS(MaskedElsewhereStmt)
WALK_TUPLE_CLASS(Module)
WALK_TUPLE_CLASS(ModuleSubprogramPart)
WALK_TUPLE_CLASS(NamedConstantDef)
WALK_TUPLE_CLASS(NamelistStmt::Group)
WALK_TUPLE_CLASS(NonLabelDoStmt)
WALK_TUPLE_CLASS(ObjectDecl)
WALK_TUPLE_CLASS(OutputImpliedDo)
WALK_TUPLE_CLASS(ParentIdentifier)
WALK_TUPLE_CLASS(PointerAssignmentStmt)
WALK_TUPLE_CLASS(PointerDecl)
WALK_TUPLE_CLASS(PrintStmt)
WALK_TUPLE_CLASS(ProcComponentDefStmt)
WALK_TUPLE_CLASS(ProcComponentRef)
WALK_TUPLE_CLASS(ProcDecl)
WALK_TUPLE_CLASS(ProcedureDeclarationStmt)
WALK_TUPLE_CLASS(ProcedureStmt)
WALK_TUPLE_CLASS(RedimensionStmt)
WALK_TUPLE_CLASS(Rename::Names)
WALK_TUPLE_CLASS(Rename::Operators)
WALK_TUPLE_CLASS(SavedEntity)
WALK_TUPLE_CLASS(SelectCaseStmt)
WALK_TUPLE_CLASS(SelectRankCaseStmt)
WALK_TUPLE_CLASS(SelectRankConstruct)
WALK_TUPLE_CLASS(SelectRankConstruct::RankCase)
WALK_TUPLE_CLASS(SelectRankStmt)
WALK_TUPLE_CLASS(SelectTypeConstruct)
WALK_TUPLE_CLASS(SelectTypeConstruct::TypeCase)
WALK_TUPLE_CLASS(SelectTypeStmt)
WALK_TUPLE_CLASS(SeparateModuleSubprogram)
WALK_TUPLE_CLASS(SignedComplexLiteralConstant)
WALK_TUPLE_CLASS(SignedIntLiteralConstant)
WALK_TUPLE_CLASS(SignedRealLiteralConstant)
WALK_TUPLE_CLASS(SpecificationPart)
WALK_TUPLE_CLASS(StmtFunctionStmt)
WALK_TUPLE_CLASS(StopStmt)
WALK_TUPLE_CLASS(StructureConstructor)
WALK_TUPLE_CLASS(StructureDef)
WALK_TUPLE_CLASS(StructureStmt)
WALK_TUPLE_CLASS(Submodule)
WALK_TUPLE_CLASS(SubmoduleStmt)
WALK_TUPLE_CLASS(SubroutineStmt)
WALK_TUPLE_CLASS(SubroutineSubprogram)
WALK_TUPLE_CLASS(SubscriptTriplet)
WALK_TUPLE_CLASS(Substring)
WALK_TUPLE_CLASS(SubstringRange)
WALK_TUPLE_CLASS(SyncImagesStmt)
WALK_TUPLE_CLASS(SyncTeamStmt)
WALK_TUPLE_CLASS(TypeBoundGenericStmt)
WALK_TUPLE_CLASS(TypeBoundProcDecl)
WALK_TUPLE_CLASS(TypeBoundProcedurePart)
WALK_TUPLE_CLASS(TypeDeclarationStmt)
WALK_TUPLE_CLASS(TypeGuardStmt)
WALK_TUPLE_CLASS(TypeParamDecl)
WALK_TUPLE_CLASS(TypeParamDefStmt)
WALK_TUPLE_CLASS(TypeParamSpec)
WALK_TUPLE_CLASS(Union)
WALK_TUPLE_CLASS(UnlockStmt)
WALK_TUPLE_CLASS(WhereConstruct)
WALK_TUPLE_CLASS(WhereConstruct::Elsewhere)
WALK_TUPLE_CLASS(WhereConstruct::MaskedElsewhere)
WALK_TUPLE_CLASS(WhereConstructStmt)
WALK_TUPLE_CLASS(WhereStmt)
#undef WALK_TUPLE_CLASS

// Walk a class with a single variant field 'u'.
#define WALK_UNION_CLASS(classname) \
  template<typename V> void Walk(const classname &x, V &visitor) { \
    if (visitor.Pre(x)) { \
      std::visit([&](const auto &y) { Walk(y, visitor); }, x.u); \
      visitor.Post(x); \
    } \
  }
WALK_UNION_CLASS(AcValue)
WALK_UNION_CLASS(AccessId)
WALK_UNION_CLASS(ActionStmt)
WALK_UNION_CLASS(ActualArg)
WALK_UNION_CLASS(AllocOpt)
WALK_UNION_CLASS(AllocateObject)
WALK_UNION_CLASS(ArraySpec)
WALK_UNION_CLASS(AttrSpec)
WALK_UNION_CLASS(BackspaceStmt)
WALK_UNION_CLASS(BindAttr)
WALK_UNION_CLASS(CaseSelector)
WALK_UNION_CLASS(CaseValueRange)
WALK_UNION_CLASS(CharLength)
WALK_UNION_CLASS(CharSelector)
WALK_UNION_CLASS(CloseStmt::CloseSpec)
WALK_UNION_CLASS(CoarraySpec)
WALK_UNION_CLASS(ComplexPart)
WALK_UNION_CLASS(ComponentArraySpec)
WALK_UNION_CLASS(ComponentAttrSpec)
WALK_UNION_CLASS(ComponentDefStmt)
WALK_UNION_CLASS(ConnectSpec)
WALK_UNION_CLASS(ConstantValue)
WALK_UNION_CLASS(DataIDoObject)
WALK_UNION_CLASS(DataReference)
WALK_UNION_CLASS(DataStmtConstant)
WALK_UNION_CLASS(DataStmtObject)
WALK_UNION_CLASS(DataStmtRepeat)
WALK_UNION_CLASS(DeclarationConstruct)
WALK_UNION_CLASS(DeclarationTypeSpec)
WALK_UNION_CLASS(DefinedOperator)
WALK_UNION_CLASS(Designator)
WALK_UNION_CLASS(DummyArg)
WALK_UNION_CLASS(EndfileStmt)
WALK_UNION_CLASS(EventWaitStmt::EventWaitSpec)
WALK_UNION_CLASS(ExecutableConstruct)
WALK_UNION_CLASS(ExecutionPartConstruct)
WALK_UNION_CLASS(Expr)
WALK_UNION_CLASS(FlushStmt)
WALK_UNION_CLASS(ForallAssignmentStmt)
WALK_UNION_CLASS(ForallBodyConstruct)
WALK_UNION_CLASS(FormTeamStmt::FormTeamSpec)
WALK_UNION_CLASS(Format)
WALK_UNION_CLASS(GenericSpec)
WALK_UNION_CLASS(ImageSelectorSpec)
WALK_UNION_CLASS(ImplicitPartStmt)
WALK_UNION_CLASS(ImplicitStmt)
WALK_UNION_CLASS(Initialization)
WALK_UNION_CLASS(InputItem)
WALK_UNION_CLASS(InquireSpec)
WALK_UNION_CLASS(InquireStmt)
WALK_UNION_CLASS(InterfaceBody)
WALK_UNION_CLASS(InterfaceSpecification)
WALK_UNION_CLASS(InterfaceStmt)
WALK_UNION_CLASS(InternalSubprogram)
WALK_UNION_CLASS(IntrinsicTypeSpec)
WALK_UNION_CLASS(IoControlSpec)
WALK_UNION_CLASS(IoUnit)
WALK_UNION_CLASS(KindParam)
WALK_UNION_CLASS(LengthSelector)
WALK_UNION_CLASS(LiteralConstant)
WALK_UNION_CLASS(LocalitySpec)
WALK_UNION_CLASS(LockStmt::LockStat)
WALK_UNION_CLASS(LoopControl)
WALK_UNION_CLASS(ModuleSubprogram)
WALK_UNION_CLASS(Only)
WALK_UNION_CLASS(OtherSpecificationStmt)
WALK_UNION_CLASS(OutputItem)
WALK_UNION_CLASS(PointerAssignmentStmt::Bounds)
WALK_UNION_CLASS(PointerObject)
WALK_UNION_CLASS(PositionOrFlushSpec)
WALK_UNION_CLASS(PrefixSpec)
WALK_UNION_CLASS(PrivateOrSequence)
WALK_UNION_CLASS(ProcAttrSpec)
WALK_UNION_CLASS(ProcComponentAttrSpec)
WALK_UNION_CLASS(ProcInterface)
WALK_UNION_CLASS(ProcPointerInit)
WALK_UNION_CLASS(ProcedureDesignator)
WALK_UNION_CLASS(ProgramUnit)
WALK_UNION_CLASS(Rename)
WALK_UNION_CLASS(RewindStmt)
WALK_UNION_CLASS(SectionSubscript)
WALK_UNION_CLASS(SelectRankCaseStmt::Rank)
WALK_UNION_CLASS(Selector)
WALK_UNION_CLASS(SpecificationConstruct)
WALK_UNION_CLASS(StatOrErrmsg)
WALK_UNION_CLASS(StopCode)
WALK_UNION_CLASS(StructureField)
WALK_UNION_CLASS(SyncImagesStmt::ImageSet)
WALK_UNION_CLASS(TypeAttrSpec)
WALK_UNION_CLASS(TypeBoundProcBinding)
WALK_UNION_CLASS(TypeBoundProcedureStmt)
WALK_UNION_CLASS(TypeGuardStmt::Guard)
WALK_UNION_CLASS(TypeParamValue)
WALK_UNION_CLASS(TypeSpec)
WALK_UNION_CLASS(Variable)
WALK_UNION_CLASS(WaitSpec)
WALK_UNION_CLASS(WhereBodyConstruct)
#undef WALK_UNION_CLASS

// Walk a class with a single field 'v'.
#define WALK_WRAPPER_CLASS(classname) \
  template<typename V> void Walk(const classname &x, V &visitor) { \
    if (visitor.Pre(x)) { \
      Walk(x.v, visitor); \
      visitor.Post(x); \
    } \
  }
WALK_WRAPPER_CLASS(AccessSpec)
WALK_WRAPPER_CLASS(ActualArg::PercentRef)
WALK_WRAPPER_CLASS(ActualArg::PercentVal)
WALK_WRAPPER_CLASS(AllocOpt::Mold)
WALK_WRAPPER_CLASS(AllocOpt::Source)
WALK_WRAPPER_CLASS(AllocatableStmt)
WALK_WRAPPER_CLASS(AltReturnSpec)
WALK_WRAPPER_CLASS(ArrayConstructor)
WALK_WRAPPER_CLASS(ArraySection)
WALK_WRAPPER_CLASS(AssumedImpliedSpec)
WALK_WRAPPER_CLASS(AssumedShapeSpec)
WALK_WRAPPER_CLASS(AsynchronousStmt)
WALK_WRAPPER_CLASS(BOZLiteralConstant)
WALK_WRAPPER_CLASS(BlockDataStmt)
WALK_WRAPPER_CLASS(BlockSpecificationPart)
WALK_WRAPPER_CLASS(BlockStmt)
WALK_WRAPPER_CLASS(BoundsSpec)
WALK_WRAPPER_CLASS(CallStmt)
WALK_WRAPPER_CLASS(CharVariable)
WALK_WRAPPER_CLASS(CloseStmt)
WALK_WRAPPER_CLASS(CodimensionStmt)
WALK_WRAPPER_CLASS(ComplexPartDesignator)
WALK_WRAPPER_CLASS(ComponentDataSource)
WALK_WRAPPER_CLASS(ConnectSpec::Newunit)
WALK_WRAPPER_CLASS(ConnectSpec::Recl)
WALK_WRAPPER_CLASS(ContiguousStmt)
WALK_WRAPPER_CLASS(CycleStmt)
WALK_WRAPPER_CLASS(DataStmt)
WALK_WRAPPER_CLASS(DeclarationTypeSpec::Record)
WALK_WRAPPER_CLASS(DeferredCoshapeSpecList)
WALK_WRAPPER_CLASS(DeferredShapeSpecList)
WALK_WRAPPER_CLASS(DefinedOpName)
WALK_WRAPPER_CLASS(DimensionStmt)
WALK_WRAPPER_CLASS(ElseStmt)
WALK_WRAPPER_CLASS(ElsewhereStmt)
WALK_WRAPPER_CLASS(EndAssociateStmt)
WALK_WRAPPER_CLASS(EndBlockDataStmt)
WALK_WRAPPER_CLASS(EndBlockStmt)
WALK_WRAPPER_CLASS(EndCriticalStmt)
WALK_WRAPPER_CLASS(EndDoStmt)
WALK_WRAPPER_CLASS(EndForallStmt)
WALK_WRAPPER_CLASS(EndFunctionStmt)
WALK_WRAPPER_CLASS(EndIfStmt)
WALK_WRAPPER_CLASS(EndInterfaceStmt)
WALK_WRAPPER_CLASS(EndLabel)
WALK_WRAPPER_CLASS(EndModuleStmt)
WALK_WRAPPER_CLASS(EndMpSubprogramStmt)
WALK_WRAPPER_CLASS(EndProgramStmt)
WALK_WRAPPER_CLASS(EndSelectStmt)
WALK_WRAPPER_CLASS(EndSubmoduleStmt)
WALK_WRAPPER_CLASS(EndSubroutineStmt)
WALK_WRAPPER_CLASS(EndTypeStmt)
WALK_WRAPPER_CLASS(EndWhereStmt)
WALK_WRAPPER_CLASS(EnumeratorDefStmt)
WALK_WRAPPER_CLASS(EorLabel)
WALK_WRAPPER_CLASS(EquivalenceObject)
WALK_WRAPPER_CLASS(EquivalenceStmt)
WALK_WRAPPER_CLASS(ErrLabel)
WALK_WRAPPER_CLASS(ExitStmt)
WALK_WRAPPER_CLASS(Expr::IntrinsicUnary)
WALK_WRAPPER_CLASS(Expr::PercentLoc)
WALK_WRAPPER_CLASS(ExternalStmt)
WALK_WRAPPER_CLASS(FileUnitNumber)
WALK_WRAPPER_CLASS(FinalProcedureStmt)
WALK_WRAPPER_CLASS(FormatStmt)
WALK_WRAPPER_CLASS(FunctionReference)
WALK_WRAPPER_CLASS(GotoStmt)
WALK_WRAPPER_CLASS(HollerithLiteralConstant)
WALK_WRAPPER_CLASS(IdExpr)
WALK_WRAPPER_CLASS(IdVariable)
WALK_WRAPPER_CLASS(ImageSelectorSpec::Stat)
WALK_WRAPPER_CLASS(ImageSelectorSpec::Team)
WALK_WRAPPER_CLASS(ImageSelectorSpec::Team_Number)
WALK_WRAPPER_CLASS(ImplicitPart)
WALK_WRAPPER_CLASS(ImpliedShapeSpec)
WALK_WRAPPER_CLASS(IntegerTypeSpec)
WALK_WRAPPER_CLASS(IntentSpec)
WALK_WRAPPER_CLASS(IntrinsicStmt)
WALK_WRAPPER_CLASS(IntrinsicTypeSpec::NCharacter)
WALK_WRAPPER_CLASS(IoControlSpec::Asynchronous)
WALK_WRAPPER_CLASS(IoControlSpec::Pos)
WALK_WRAPPER_CLASS(IoControlSpec::Rec)
WALK_WRAPPER_CLASS(IoControlSpec::Size)
WALK_WRAPPER_CLASS(KindSelector)
WALK_WRAPPER_CLASS(LanguageBindingSpec)
WALK_WRAPPER_CLASS(LocalitySpec::Local)
WALK_WRAPPER_CLASS(LocalitySpec::LocalInit)
WALK_WRAPPER_CLASS(LocalitySpec::Shared)
WALK_WRAPPER_CLASS(LogicalLiteralConstant)
WALK_WRAPPER_CLASS(ModuleStmt)
WALK_WRAPPER_CLASS(MpSubprogramStmt)
WALK_WRAPPER_CLASS(MsgVariable)
WALK_WRAPPER_CLASS(NamedConstant)
WALK_WRAPPER_CLASS(NamelistStmt)
WALK_WRAPPER_CLASS(NullifyStmt)
WALK_WRAPPER_CLASS(OpenStmt)
WALK_WRAPPER_CLASS(OptionalStmt)
WALK_WRAPPER_CLASS(ParameterStmt)
WALK_WRAPPER_CLASS(Pass)
WALK_WRAPPER_CLASS(PauseStmt)
WALK_WRAPPER_CLASS(PointerStmt)
WALK_WRAPPER_CLASS(Program)
WALK_WRAPPER_CLASS(ProtectedStmt)
WALK_WRAPPER_CLASS(ReturnStmt)
WALK_WRAPPER_CLASS(SaveStmt)
WALK_WRAPPER_CLASS(SpecificationExpr)
WALK_WRAPPER_CLASS(StatVariable)
WALK_WRAPPER_CLASS(StatusExpr)
WALK_WRAPPER_CLASS(SyncAllStmt)
WALK_WRAPPER_CLASS(SyncMemoryStmt)
WALK_WRAPPER_CLASS(TargetStmt)
WALK_WRAPPER_CLASS(TypeParamInquiry)
WALK_WRAPPER_CLASS(ValueStmt)
WALK_WRAPPER_CLASS(VolatileStmt)
WALK_WRAPPER_CLASS(WaitStmt)
#undef WALK_WRAPPER_CLASS

}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_PARSE_TREE_VISITOR_H_
