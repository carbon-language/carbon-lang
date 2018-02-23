#include <iostream>
#include "../../lib/parser/format-specification.h"
#include "../../lib/parser/grammar.h"
#include "../../lib/parser/idioms.h"
#include "../../lib/parser/indirection.h"
#include "../../lib/parser/message.h"
#include "../../lib/parser/parse-state.h"
#include "../../lib/parser/parse-tree-visitor.h"
#include "../../lib/parser/parse-tree.h"
#include "../../lib/parser/preprocessor.h"
#include "../../lib/parser/prescan.h"
#include "../../lib/parser/provenance.h"
#include "../../lib/parser/source.h"
#include "../../lib/parser/user-state.h"
#include <cstdint>
#include <cstdlib>
#include <list>
#include <optional>
#include <sstream>
#include <stddef.h>
#include <string>

using namespace Fortran::parser;

// A visitor that visits all nodes in the parse tree and prints their
// names, with children indented one space relative to their parent.
class Visitor {
  int indent_{0};
  void out(const char *str) {
    for (int i = 0; i < indent_; ++i) {
      std::cout << ' ';
    }
    std::cout << str << '\n';
    ++indent_;
  }
  void out(const std::string &str) {
    out(str.c_str());
  }

public:

  bool pre(const Abstract &x) {
    out("Abstract");
    return true;
  }
  bool pre(const AcImpliedDo &x) {
    out("AcImpliedDo");
    return true;
  }
  bool pre(const AcImpliedDoControl &x) {
    out("AcImpliedDoControl");
    return true;
  }
  bool pre(const AcSpec &x) {
    out("AcSpec");
    return true;
  }
  bool pre(const AcValue &x) {
    out("AcValue");
    return true;
  }
  bool pre(const AcValue::Triplet &x) {
    out("AcValue::Triplet");
    return true;
  }
  bool pre(const AccessId &x) {
    out("AccessId");
    return true;
  }
  bool pre(const AccessSpec &x) {
    out("AccessSpec");
    return true;
  }
  bool pre(const AccessStmt &x) {
    out("AccessStmt");
    return true;
  }
  bool pre(const ActionStmt &x) {
    out("ActionStmt");
    return true;
  }
  bool pre(const ActualArg &x) {
    out("ActualArg");
    return true;
  }
  bool pre(const ActualArg::PercentRef &x) {
    out("ActualArg::PercentRef");
    return true;
  }
  bool pre(const ActualArg::PercentVal &x) {
    out("ActualArg::PercentVal");
    return true;
  }
  bool pre(const ActualArgSpec &x) {
    out("ActualArgSpec");
    return true;
  }
  bool pre(const AllocOpt &x) {
    out("AllocOpt");
    return true;
  }
  bool pre(const AllocOpt::Mold &x) {
    out("AllocOpt::Mold");
    return true;
  }
  bool pre(const AllocOpt::Source &x) {
    out("AllocOpt::Source");
    return true;
  }
  bool pre(const Allocatable &x) {
    out("Allocatable");
    return true;
  }
  bool pre(const AllocatableStmt &x) {
    out("AllocatableStmt");
    return true;
  }
  bool pre(const AllocateCoarraySpec &x) {
    out("AllocateCoarraySpec");
    return true;
  }
  bool pre(const AllocateObject &x) {
    out("AllocateObject");
    return true;
  }
  bool pre(const AllocateShapeSpec &x) {
    out("AllocateShapeSpec");
    return true;
  }
  bool pre(const AllocateStmt &x) {
    out("AllocateStmt");
    return true;
  }
  bool pre(const Allocation &x) {
    out("Allocation");
    return true;
  }
  bool pre(const AltReturnSpec &x) {
    out("AltReturnSpec");
    return true;
  }
  bool pre(const ArithmeticIfStmt &x) {
    out("ArithmeticIfStmt");
    return true;
  }
  bool pre(const ArrayConstructor &x) {
    out("ArrayConstructor");
    return true;
  }
  bool pre(const ArrayElement &x) {
    out("ArrayElement");
    return true;
  }
  bool pre(const ArraySection &x) {
    out("ArraySection");
    return true;
  }
  bool pre(const ArraySpec &x) {
    out("ArraySpec");
    return true;
  }
  bool pre(const AssignStmt &x) {
    out("AssignStmt");
    return true;
  }
  bool pre(const AssignedGotoStmt &x) {
    out("AssignedGotoStmt");
    return true;
  }
  bool pre(const AssignmentStmt &x) {
    out("AssignmentStmt");
    return true;
  }
  bool pre(const AssociateConstruct &x) {
    out("AssociateConstruct");
    return true;
  }
  bool pre(const AssociateStmt &x) {
    out("AssociateStmt");
    return true;
  }
  bool pre(const Association &x) {
    out("Association");
    return true;
  }
  bool pre(const AssumedImpliedSpec &x) {
    out("AssumedImpliedSpec");
    return true;
  }
  bool pre(const AssumedRankSpec &x) {
    out("AssumedRankSpec");
    return true;
  }
  bool pre(const AssumedShapeSpec &x) {
    out("AssumedShapeSpec");
    return true;
  }
  bool pre(const AssumedSizeSpec &x) {
    out("AssumedSizeSpec");
    return true;
  }
  bool pre(const Asynchronous &x) {
    out("Asynchronous");
    return true;
  }
  bool pre(const AsynchronousStmt &x) {
    out("AsynchronousStmt");
    return true;
  }
  bool pre(const AttrSpec &x) {
    out("AttrSpec");
    return true;
  }
  bool pre(const BOZLiteralConstant &x) {
    out("BOZLiteralConstant");
    return true;
  }
  bool pre(const BackspaceStmt &x) {
    out("BackspaceStmt");
    return true;
  }
  bool pre(const BasedPointerStmt &x) {
    out("BasedPointerStmt");
    return true;
  }
  bool pre(const BindAttr &x) {
    out("BindAttr");
    return true;
  }
  bool pre(const BindAttr::Deferred &x) {
    out("BindAttr::Deferred");
    return true;
  }
  bool pre(const BindAttr::Non_Overridable &x) {
    out("BindAttr::Non_Overridable");
    return true;
  }
  bool pre(const BindEntity &x) {
    out("BindEntity");
    return true;
  }
  bool pre(const BindStmt &x) {
    out("BindStmt");
    return true;
  }
  bool pre(const BlockConstruct &x) {
    out("BlockConstruct");
    return true;
  }
  bool pre(const BlockData &x) {
    out("BlockData");
    return true;
  }
  bool pre(const BlockDataStmt &x) {
    out("BlockDataStmt");
    return true;
  }
  bool pre(const BlockSpecificationPart &x) {
    out("BlockSpecificationPart");
    return true;
  }
  bool pre(const BlockStmt &x) {
    out("BlockStmt");
    return true;
  }
  bool pre(const BoundsRemapping &x) {
    out("BoundsRemapping");
    return true;
  }
  bool pre(const BoundsSpec &x) {
    out("BoundsSpec");
    return true;
  }
  bool pre(const Call &x) {
    out("Call");
    return true;
  }
  bool pre(const CallStmt &x) {
    out("CallStmt");
    return true;
  }
  bool pre(const CaseConstruct &x) {
    out("CaseConstruct");
    return true;
  }
  bool pre(const CaseConstruct::Case &x) {
    out("CaseConstruct::Case");
    return true;
  }
  bool pre(const CaseSelector &x) {
    out("CaseSelector");
    return true;
  }
  bool pre(const CaseStmt &x) {
    out("CaseStmt");
    return true;
  }
  bool pre(const CaseValueRange &x) {
    out("CaseValueRange");
    return true;
  }
  bool pre(const CaseValueRange::Range &x) {
    out("CaseValueRange::Range");
    return true;
  }
  bool pre(const ChangeTeamConstruct &x) {
    out("ChangeTeamConstruct");
    return true;
  }
  bool pre(const ChangeTeamStmt &x) {
    out("ChangeTeamStmt");
    return true;
  }
  bool pre(const CharLength &x) {
    out("CharLength");
    return true;
  }
  bool pre(const CharLiteralConstant &x) {
    out("CharLiteralConstant");
    return true;
  }
  bool pre(const CharLiteralConstantSubstring &x) {
    out("CharLiteralConstantSubstring");
    return true;
  }
  bool pre(const CharSelector &x) {
    out("CharSelector");
    return true;
  }
  bool pre(const CharSelector::LengthAndKind &x) {
    out("CharSelector::LengthAndKind");
    return true;
  }
  bool pre(const CharVariable &x) {
    out("CharVariable");
    return true;
  }
  bool pre(const CloseStmt &x) {
    out("CloseStmt");
    return true;
  }
  bool pre(const CloseStmt::CloseSpec &x) {
    out("CloseStmt::CloseSpec");
    return true;
  }
  bool pre(const CoarrayAssociation &x) {
    out("CoarrayAssociation");
    return true;
  }
  bool pre(const CoarraySpec &x) {
    out("CoarraySpec");
    return true;
  }
  bool pre(const CodimensionDecl &x) {
    out("CodimensionDecl");
    return true;
  }
  bool pre(const CodimensionStmt &x) {
    out("CodimensionStmt");
    return true;
  }
  bool pre(const CoindexedNamedObject &x) {
    out("CoindexedNamedObject");
    return true;
  }
  bool pre(const CommonBlockObject &x) {
    out("CommonBlockObject");
    return true;
  }
  bool pre(const CommonStmt &x) {
    out("CommonStmt");
    return true;
  }
  bool pre(const ComplexLiteralConstant &x) {
    out("ComplexLiteralConstant");
    return true;
  }
  bool pre(const ComplexPart &x) {
    out("ComplexPart");
    return true;
  }
  bool pre(const ComplexPartDesignator &x) {
    out("ComplexPartDesignator");
    return true;
  }
  bool pre(const ComponentArraySpec &x) {
    out("ComponentArraySpec");
    return true;
  }
  bool pre(const ComponentAttrSpec &x) {
    out("ComponentAttrSpec");
    return true;
  }
  bool pre(const ComponentDataSource &x) {
    out("ComponentDataSource");
    return true;
  }
  bool pre(const ComponentDecl &x) {
    out("ComponentDecl");
    return true;
  }
  bool pre(const ComponentDefStmt &x) {
    out("ComponentDefStmt");
    return true;
  }
  bool pre(const ComponentSpec &x) {
    out("ComponentSpec");
    return true;
  }
  bool pre(const ComputedGotoStmt &x) {
    out("ComputedGotoStmt");
    return true;
  }
  bool pre(const ConcurrentControl &x) {
    out("ConcurrentControl");
    return true;
  }
  bool pre(const ConcurrentHeader &x) {
    out("ConcurrentHeader");
    return true;
  }
  bool pre(const ConnectSpec &x) {
    out("ConnectSpec");
    return true;
  }
  bool pre(const ConnectSpec::CharExpr &x) {
    out("ConnectSpec::CharExpr");
    return true;
  }
  bool pre(const ConnectSpec::Newunit &x) {
    out("ConnectSpec::Newunit");
    return true;
  }
  bool pre(const ConnectSpec::Recl &x) {
    out("ConnectSpec::Recl");
    return true;
  }
  bool pre(const ConstantValue &x) {
    out("ConstantValue");
    return true;
  }
  bool pre(const ContainsStmt &x) {
    out("ContainsStmt");
    return true;
  }
  bool pre(const Contiguous &x) {
    out("Contiguous");
    return true;
  }
  bool pre(const ContiguousStmt &x) {
    out("ContiguousStmt");
    return true;
  }
  bool pre(const ContinueStmt &x) {
    out("ContinueStmt");
    return true;
  }
  bool pre(const CriticalConstruct &x) {
    out("CriticalConstruct");
    return true;
  }
  bool pre(const CriticalStmt &x) {
    out("CriticalStmt");
    return true;
  }
  bool pre(const CycleStmt &x) {
    out("CycleStmt");
    return true;
  }
  bool pre(const DataComponentDefStmt &x) {
    out("DataComponentDefStmt");
    return true;
  }
  bool pre(const DataIDoObject &x) {
    out("DataIDoObject");
    return true;
  }
  bool pre(const DataImpliedDo &x) {
    out("DataImpliedDo");
    return true;
  }
  bool pre(const DataReference &x) {
    out("DataReference");
    return true;
  }
  bool pre(const DataStmt &x) {
    out("DataStmt");
    return true;
  }
  bool pre(const DataStmtConstant &x) {
    out("DataStmtConstant");
    return true;
  }
  bool pre(const DataStmtObject &x) {
    out("DataStmtObject");
    return true;
  }
  bool pre(const DataStmtRepeat &x) {
    out("DataStmtRepeat");
    return true;
  }
  bool pre(const DataStmtSet &x) {
    out("DataStmtSet");
    return true;
  }
  bool pre(const DataStmtValue &x) {
    out("DataStmtValue");
    return true;
  }
  bool pre(const DeallocateStmt &x) {
    out("DeallocateStmt");
    return true;
  }
  bool pre(const DeclarationConstruct &x) {
    out("DeclarationConstruct");
    return true;
  }
  bool pre(const DeclarationTypeSpec &x) {
    out("DeclarationTypeSpec");
    return true;
  }
  bool pre(const DeclarationTypeSpec::Class &x) {
    out("DeclarationTypeSpec::Class");
    return true;
  }
  bool pre(const DeclarationTypeSpec::ClassStar &x) {
    out("DeclarationTypeSpec::ClassStar");
    return true;
  }
  bool pre(const DeclarationTypeSpec::Record &x) {
    out("DeclarationTypeSpec::Record");
    return true;
  }
  bool pre(const DeclarationTypeSpec::Type &x) {
    out("DeclarationTypeSpec::Type");
    return true;
  }
  bool pre(const DeclarationTypeSpec::TypeStar &x) {
    out("DeclarationTypeSpec::TypeStar");
    return true;
  }
  bool pre(const Default &x) {
    out("Default");
    return true;
  }
  bool pre(const DeferredCoshapeSpecList &x) {
    out("DeferredCoshapeSpecList");
    return true;
  }
  bool pre(const DeferredShapeSpecList &x) {
    out("DeferredShapeSpecList");
    return true;
  }
  bool pre(const DefinedOpName &x) {
    out("DefinedOpName");
    return true;
  }
  bool pre(const DefinedOperator &x) {
    out("DefinedOperator");
    return true;
  }
  bool pre(const DerivedTypeDef &x) {
    out("DerivedTypeDef");
    return true;
  }
  bool pre(const DerivedTypeSpec &x) {
    out("DerivedTypeSpec");
    return true;
  }
  bool pre(const DerivedTypeStmt &x) {
    out("DerivedTypeStmt");
    return true;
  }
  bool pre(const Designator &x) {
    out("Designator");
    return true;
  }
  bool pre(const DimensionStmt &x) {
    out("DimensionStmt");
    return true;
  }
  bool pre(const DimensionStmt::Declaration &x) {
    out("DimensionStmt::Declaration");
    return true;
  }
  bool pre(const DoConstruct &x) {
    out("DoConstruct");
    return true;
  }
  bool pre(const DummyArg &x) {
    out("DummyArg");
    return true;
  }
  bool pre(const ElseIfStmt &x) {
    out("ElseIfStmt");
    return true;
  }
  bool pre(const ElseStmt &x) {
    out("ElseStmt");
    return true;
  }
  bool pre(const ElsewhereStmt &x) {
    out("ElsewhereStmt");
    return true;
  }
  bool pre(const EndAssociateStmt &x) {
    out("EndAssociateStmt");
    return true;
  }
  bool pre(const EndBlockDataStmt &x) {
    out("EndBlockDataStmt");
    return true;
  }
  bool pre(const EndBlockStmt &x) {
    out("EndBlockStmt");
    return true;
  }
  bool pre(const EndChangeTeamStmt &x) {
    out("EndChangeTeamStmt");
    return true;
  }
  bool pre(const EndCriticalStmt &x) {
    out("EndCriticalStmt");
    return true;
  }
  bool pre(const EndDoStmt &x) {
    out("EndDoStmt");
    return true;
  }
  bool pre(const EndEnumStmt &x) {
    out("EndEnumStmt");
    return true;
  }
  bool pre(const EndForallStmt &x) {
    out("EndForallStmt");
    return true;
  }
  bool pre(const EndFunctionStmt &x) {
    out("EndFunctionStmt");
    return true;
  }
  bool pre(const EndIfStmt &x) {
    out("EndIfStmt");
    return true;
  }
  bool pre(const EndInterfaceStmt &x) {
    out("EndInterfaceStmt");
    return true;
  }
  bool pre(const EndLabel &x) {
    out("EndLabel");
    return true;
  }
  bool pre(const EndModuleStmt &x) {
    out("EndModuleStmt");
    return true;
  }
  bool pre(const EndMpSubprogramStmt &x) {
    out("EndMpSubprogramStmt");
    return true;
  }
  bool pre(const EndProgramStmt &x) {
    out("EndProgramStmt");
    return true;
  }
  bool pre(const EndSelectStmt &x) {
    out("EndSelectStmt");
    return true;
  }
  bool pre(const EndSubmoduleStmt &x) {
    out("EndSubmoduleStmt");
    return true;
  }
  bool pre(const EndSubroutineStmt &x) {
    out("EndSubroutineStmt");
    return true;
  }
  bool pre(const EndTypeStmt &x) {
    out("EndTypeStmt");
    return true;
  }
  bool pre(const EndWhereStmt &x) {
    out("EndWhereStmt");
    return true;
  }
  bool pre(const EndfileStmt &x) {
    out("EndfileStmt");
    return true;
  }
  bool pre(const EntityDecl &x) {
    out("EntityDecl");
    return true;
  }
  bool pre(const EntryStmt &x) {
    out("EntryStmt");
    return true;
  }
  bool pre(const EnumDef &x) {
    out("EnumDef");
    return true;
  }
  bool pre(const EnumDefStmt &x) {
    out("EnumDefStmt");
    return true;
  }
  bool pre(const Enumerator &x) {
    out("Enumerator");
    return true;
  }
  bool pre(const EnumeratorDefStmt &x) {
    out("EnumeratorDefStmt");
    return true;
  }
  bool pre(const EorLabel &x) {
    out("EorLabel");
    return true;
  }
  bool pre(const EquivalenceObject &x) {
    out("EquivalenceObject");
    return true;
  }
  bool pre(const EquivalenceStmt &x) {
    out("EquivalenceStmt");
    return true;
  }
  bool pre(const ErrLabel &x) {
    out("ErrLabel");
    return true;
  }
  bool pre(const ErrorRecovery &x) {
    out("ErrorRecovery");
    return true;
  }
  bool pre(const EventPostStmt &x) {
    out("EventPostStmt");
    return true;
  }
  bool pre(const EventWaitStmt &x) {
    out("EventWaitStmt");
    return true;
  }
  bool pre(const EventWaitStmt::EventWaitSpec &x) {
    out("EventWaitStmt::EventWaitSpec");
    return true;
  }
  bool pre(const ExecutableConstruct &x) {
    out("ExecutableConstruct");
    return true;
  }
  bool pre(const ExecutionPartConstruct &x) {
    out("ExecutionPartConstruct");
    return true;
  }
  bool pre(const ExitStmt &x) {
    out("ExitStmt");
    return true;
  }
  bool pre(const ExplicitCoshapeSpec &x) {
    out("ExplicitCoshapeSpec");
    return true;
  }
  bool pre(const ExplicitShapeSpec &x) {
    out("ExplicitShapeSpec");
    return true;
  }
  bool pre(const ExponentPart &x) {
    out("ExponentPart");
    return true;
  }
  bool pre(const Expr &x) {
    out("Expr");
    return true;
  }
  bool pre(const Expr::AND &x) {
    out("Expr::AND");
    return true;
  }
  bool pre(const Expr::Add &x) {
    out("Expr::Add");
    return true;
  }
  bool pre(const Expr::ComplexConstructor &x) {
    out("Expr::ComplexConstructor");
    return true;
  }
  bool pre(const Expr::Concat &x) {
    out("Expr::Concat");
    return true;
  }
  bool pre(const Expr::DefinedBinary &x) {
    out("Expr::DefinedBinary");
    return true;
  }
  bool pre(const Expr::DefinedUnary &x) {
    out("Expr::DefinedUnary");
    return true;
  }
  bool pre(const Expr::Divide &x) {
    out("Expr::Divide");
    return true;
  }
  bool pre(const Expr::EQ &x) {
    out("Expr::EQ");
    return true;
  }
  bool pre(const Expr::EQV &x) {
    out("Expr::EQV");
    return true;
  }
  bool pre(const Expr::GE &x) {
    out("Expr::GE");
    return true;
  }
  bool pre(const Expr::GT &x) {
    out("Expr::GT");
    return true;
  }
  bool pre(const Expr::IntrinsicBinary &x) {
    out("Expr::IntrinsicBinary");
    return true;
  }
  bool pre(const Expr::IntrinsicUnary &x) {
    out("Expr::IntrinsicUnary");
    return true;
  }
  bool pre(const Expr::LE &x) {
    out("Expr::LE");
    return true;
  }
  bool pre(const Expr::LT &x) {
    out("Expr::LT");
    return true;
  }
  bool pre(const Expr::Multiply &x) {
    out("Expr::Multiply");
    return true;
  }
  bool pre(const Expr::NE &x) {
    out("Expr::NE");
    return true;
  }
  bool pre(const Expr::NEQV &x) {
    out("Expr::NEQV");
    return true;
  }
  bool pre(const Expr::NOT &x) {
    out("Expr::NOT");
    return true;
  }
  bool pre(const Expr::Negate &x) {
    out("Expr::Negate");
    return true;
  }
  bool pre(const Expr::OR &x) {
    out("Expr::OR");
    return true;
  }
  bool pre(const Expr::Parentheses &x) {
    out("Expr::Parentheses");
    return true;
  }
  bool pre(const Expr::PercentLoc &x) {
    out("Expr::PercentLoc");
    return true;
  }
  bool pre(const Expr::Power &x) {
    out("Expr::Power");
    return true;
  }
  bool pre(const Expr::Subtract &x) {
    out("Expr::Subtract");
    return true;
  }
  bool pre(const Expr::UnaryPlus &x) {
    out("Expr::UnaryPlus");
    return true;
  }
  bool pre(const External &x) {
    out("External");
    return true;
  }
  bool pre(const ExternalStmt &x) {
    out("ExternalStmt");
    return true;
  }
  bool pre(const FailImageStmt &x) {
    out("FailImageStmt");
    return true;
  }
  bool pre(const FileUnitNumber &x) {
    out("FileUnitNumber");
    return true;
  }
  bool pre(const FinalProcedureStmt &x) {
    out("FinalProcedureStmt");
    return true;
  }
  bool pre(const FlushStmt &x) {
    out("FlushStmt");
    return true;
  }
  bool pre(const ForallAssignmentStmt &x) {
    out("ForallAssignmentStmt");
    return true;
  }
  bool pre(const ForallBodyConstruct &x) {
    out("ForallBodyConstruct");
    return true;
  }
  bool pre(const ForallConstruct &x) {
    out("ForallConstruct");
    return true;
  }
  bool pre(const ForallConstructStmt &x) {
    out("ForallConstructStmt");
    return true;
  }
  bool pre(const ForallStmt &x) {
    out("ForallStmt");
    return true;
  }
  bool pre(const FormTeamStmt &x) {
    out("FormTeamStmt");
    return true;
  }
  bool pre(const FormTeamStmt::FormTeamSpec &x) {
    out("FormTeamStmt::FormTeamSpec");
    return true;
  }
  bool pre(const Format &x) {
    out("Format");
    return true;
  }
  bool pre(const FormatStmt &x) {
    out("FormatStmt");
    return true;
  }
  bool pre(const Fortran::ControlEditDesc &x) {
    out("Fortran::ControlEditDesc");
    return true;
  }
  bool pre(const Fortran::DerivedTypeDataEditDesc &x) {
    out("Fortran::DerivedTypeDataEditDesc");
    return true;
  }
  bool pre(const Fortran::FormatItem &x) {
    out("Fortran::FormatItem");
    return true;
  }
  bool pre(const Fortran::FormatSpecification &x) {
    out("Fortran::FormatSpecification");
    return true;
  }
  bool pre(const Fortran::IntrinsicTypeDataEditDesc &x) {
    out("Fortran::IntrinsicTypeDataEditDesc");
    return true;
  }
  bool pre(const FunctionReference &x) {
    out("FunctionReference");
    return true;
  }
  bool pre(const FunctionStmt &x) {
    out("FunctionStmt");
    return true;
  }
  bool pre(const FunctionSubprogram &x) {
    out("FunctionSubprogram");
    return true;
  }
  bool pre(const GenericSpec &x) {
    out("GenericSpec");
    return true;
  }
  bool pre(const GenericSpec::Assignment &x) {
    out("GenericSpec::Assignment");
    return true;
  }
  bool pre(const GenericSpec::ReadFormatted &x) {
    out("GenericSpec::ReadFormatted");
    return true;
  }
  bool pre(const GenericSpec::ReadUnformatted &x) {
    out("GenericSpec::ReadUnformatted");
    return true;
  }
  bool pre(const GenericSpec::WriteFormatted &x) {
    out("GenericSpec::WriteFormatted");
    return true;
  }
  bool pre(const GenericSpec::WriteUnformatted &x) {
    out("GenericSpec::WriteUnformatted");
    return true;
  }
  bool pre(const GenericStmt &x) {
    out("GenericStmt");
    return true;
  }
  bool pre(const GotoStmt &x) {
    out("GotoStmt");
    return true;
  }
  bool pre(const HollerithLiteralConstant &x) {
    out("HollerithLiteralConstant");
    return true;
  }
  bool pre(const IdExpr &x) {
    out("IdExpr");
    return true;
  }
  bool pre(const IdVariable &x) {
    out("IdVariable");
    return true;
  }
  bool pre(const IfConstruct &x) {
    out("IfConstruct");
    return true;
  }
  bool pre(const IfConstruct::ElseBlock &x) {
    out("IfConstruct::ElseBlock");
    return true;
  }
  bool pre(const IfConstruct::ElseIfBlock &x) {
    out("IfConstruct::ElseIfBlock");
    return true;
  }
  bool pre(const IfStmt &x) {
    out("IfStmt");
    return true;
  }
  bool pre(const IfThenStmt &x) {
    out("IfThenStmt");
    return true;
  }
  bool pre(const ImageSelector &x) {
    out("ImageSelector");
    return true;
  }
  bool pre(const ImageSelectorSpec &x) {
    out("ImageSelectorSpec");
    return true;
  }
  bool pre(const ImageSelectorSpec::Stat &x) {
    out("ImageSelectorSpec::Stat");
    return true;
  }
  bool pre(const ImageSelectorSpec::Team &x) {
    out("ImageSelectorSpec::Team");
    return true;
  }
  bool pre(const ImageSelectorSpec::Team_Number &x) {
    out("ImageSelectorSpec::Team_Number");
    return true;
  }
  bool pre(const ImplicitPart &x) {
    out("ImplicitPart");
    return true;
  }
  bool pre(const ImplicitPartStmt &x) {
    out("ImplicitPartStmt");
    return true;
  }
  bool pre(const ImplicitSpec &x) {
    out("ImplicitSpec");
    return true;
  }
  bool pre(const ImplicitStmt &x) {
    out("ImplicitStmt");
    return true;
  }
  bool pre(const ImpliedShapeSpec &x) {
    out("ImpliedShapeSpec");
    return true;
  }
  bool pre(const ImportStmt &x) {
    out("ImportStmt");
    return true;
  }
  bool pre(const Initialization &x) {
    out("Initialization");
    return true;
  }
  bool pre(const InputImpliedDo &x) {
    out("InputImpliedDo");
    return true;
  }
  bool pre(const InputItem &x) {
    out("InputItem");
    return true;
  }
  bool pre(const InquireSpec &x) {
    out("InquireSpec");
    return true;
  }
  bool pre(const InquireSpec::CharVar &x) {
    out("InquireSpec::CharVar");
    return true;
  }
  bool pre(const InquireSpec::IntVar &x) {
    out("InquireSpec::IntVar");
    return true;
  }
  bool pre(const InquireSpec::LogVar &x) {
    out("InquireSpec::LogVar");
    return true;
  }
  bool pre(const InquireStmt &x) {
    out("InquireStmt");
    return true;
  }
  bool pre(const InquireStmt::Iolength &x) {
    out("InquireStmt::Iolength");
    return true;
  }
  bool pre(const IntLiteralConstant &x) {
    out("IntLiteralConstant");
    return true;
  }
  bool pre(const IntegerTypeSpec &x) {
    out("IntegerTypeSpec");
    return true;
  }
  bool pre(const IntentSpec &x) {
    out("IntentSpec");
    return true;
  }
  bool pre(const IntentStmt &x) {
    out("IntentStmt");
    return true;
  }
  bool pre(const InterfaceBlock &x) {
    out("InterfaceBlock");
    return true;
  }
  bool pre(const InterfaceBody &x) {
    out("InterfaceBody");
    return true;
  }
  bool pre(const InterfaceBody::Function &x) {
    out("InterfaceBody::Function");
    return true;
  }
  bool pre(const InterfaceBody::Subroutine &x) {
    out("InterfaceBody::Subroutine");
    return true;
  }
  bool pre(const InterfaceSpecification &x) {
    out("InterfaceSpecification");
    return true;
  }
  bool pre(const InterfaceStmt &x) {
    out("InterfaceStmt");
    return true;
  }
  bool pre(const InternalSubprogram &x) {
    out("InternalSubprogram");
    return true;
  }
  bool pre(const InternalSubprogramPart &x) {
    out("InternalSubprogramPart");
    return true;
  }
  bool pre(const Intrinsic &x) {
    out("Intrinsic");
    return true;
  }
  bool pre(const IntrinsicStmt &x) {
    out("IntrinsicStmt");
    return true;
  }
  bool pre(const IntrinsicTypeSpec &x) {
    out("IntrinsicTypeSpec");
    return true;
  }
  bool pre(const IntrinsicTypeSpec::Character &x) {
    out("IntrinsicTypeSpec::Character");
    return true;
  }
  bool pre(const IntrinsicTypeSpec::Complex &x) {
    out("IntrinsicTypeSpec::Complex");
    return true;
  }
  bool pre(const IntrinsicTypeSpec::DoubleComplex &x) {
    out("IntrinsicTypeSpec::DoubleComplex");
    return true;
  }
  bool pre(const IntrinsicTypeSpec::DoublePrecision &x) {
    out("IntrinsicTypeSpec::DoublePrecision");
    return true;
  }
  bool pre(const IntrinsicTypeSpec::Logical &x) {
    out("IntrinsicTypeSpec::Logical");
    return true;
  }
  bool pre(const IntrinsicTypeSpec::NCharacter &x) {
    out("IntrinsicTypeSpec::NCharacter");
    return true;
  }
  bool pre(const IntrinsicTypeSpec::Real &x) {
    out("IntrinsicTypeSpec::Real");
    return true;
  }
  bool pre(const IoControlSpec &x) {
    out("IoControlSpec");
    return true;
  }
  bool pre(const IoControlSpec::Asynchronous &x) {
    out("IoControlSpec::Asynchronous");
    return true;
  }
  bool pre(const IoControlSpec::CharExpr &x) {
    out("IoControlSpec::CharExpr");
    return true;
  }
  bool pre(const IoControlSpec::Pos &x) {
    out("IoControlSpec::Pos");
    return true;
  }
  bool pre(const IoControlSpec::Rec &x) {
    out("IoControlSpec::Rec");
    return true;
  }
  bool pre(const IoControlSpec::Size &x) {
    out("IoControlSpec::Size");
    return true;
  }
  bool pre(const IoUnit &x) {
    out("IoUnit");
    return true;
  }
  bool pre(const KindParam &x) {
    out("KindParam");
    return true;
  }
  bool pre(const KindParam::Kanji &x) {
    out("KindParam::Kanji");
    return true;
  }
  bool pre(const KindSelector &x) {
    out("KindSelector");
    return true;
  }
  bool pre(const LabelDoStmt &x) {
    out("LabelDoStmt");
    return true;
  }
  bool pre(const LanguageBindingSpec &x) {
    out("LanguageBindingSpec");
    return true;
  }
  bool pre(const LengthSelector &x) {
    out("LengthSelector");
    return true;
  }
  bool pre(const LetterSpec &x) {
    out("LetterSpec");
    return true;
  }
  bool pre(const LiteralConstant &x) {
    out("LiteralConstant");
    return true;
  }
  bool pre(const LocalitySpec &x) {
    out("LocalitySpec");
    return true;
  }
  bool pre(const LocalitySpec::DefaultNone &x) {
    out("LocalitySpec::DefaultNone");
    return true;
  }
  bool pre(const LocalitySpec::Local &x) {
    out("LocalitySpec::Local");
    return true;
  }
  bool pre(const LocalitySpec::LocalInit &x) {
    out("LocalitySpec::LocalInit");
    return true;
  }
  bool pre(const LocalitySpec::Shared &x) {
    out("LocalitySpec::Shared");
    return true;
  }
  bool pre(const LockStmt &x) {
    out("LockStmt");
    return true;
  }
  bool pre(const LockStmt::LockStat &x) {
    out("LockStmt::LockStat");
    return true;
  }
  bool pre(const LogicalLiteralConstant &x) {
    out("LogicalLiteralConstant");
    return true;
  }
  bool pre(const LoopControl &x) {
    out("LoopControl");
    return true;
  }
  bool pre(const LoopControl::Concurrent &x) {
    out("LoopControl::Concurrent");
    return true;
  }
  bool pre(const MainProgram &x) {
    out("MainProgram");
    return true;
  }
  bool pre(const Map &x) {
    out("Map");
    return true;
  }
  bool pre(const Map::EndMapStmt &x) {
    out("Map::EndMapStmt");
    return true;
  }
  bool pre(const Map::MapStmt &x) {
    out("Map::MapStmt");
    return true;
  }
  bool pre(const MaskedElsewhereStmt &x) {
    out("MaskedElsewhereStmt");
    return true;
  }
  bool pre(const Module &x) {
    out("Module");
    return true;
  }
  bool pre(const ModuleStmt &x) {
    out("ModuleStmt");
    return true;
  }
  bool pre(const ModuleSubprogram &x) {
    out("ModuleSubprogram");
    return true;
  }
  bool pre(const ModuleSubprogramPart &x) {
    out("ModuleSubprogramPart");
    return true;
  }
  bool pre(const MpSubprogramStmt &x) {
    out("MpSubprogramStmt");
    return true;
  }
  bool pre(const MsgVariable &x) {
    out("MsgVariable");
    return true;
  }
  bool pre(const NamedConstant &x) {
    out("NamedConstant");
    return true;
  }
  bool pre(const NamedConstantDef &x) {
    out("NamedConstantDef");
    return true;
  }
  bool pre(const NamelistStmt &x) {
    out("NamelistStmt");
    return true;
  }
  bool pre(const NamelistStmt::Group &x) {
    out("NamelistStmt::Group");
    return true;
  }
  bool pre(const NoPass &x) {
    out("NoPass");
    return true;
  }
  bool pre(const NonLabelDoStmt &x) {
    out("NonLabelDoStmt");
    return true;
  }
  bool pre(const NullInit &x) {
    out("NullInit");
    return true;
  }
  bool pre(const NullifyStmt &x) {
    out("NullifyStmt");
    return true;
  }
  bool pre(const ObjectDecl &x) {
    out("ObjectDecl");
    return true;
  }
  bool pre(const Only &x) {
    out("Only");
    return true;
  }
  bool pre(const OpenStmt &x) {
    out("OpenStmt");
    return true;
  }
  bool pre(const Optional &x) {
    out("Optional");
    return true;
  }
  bool pre(const OptionalStmt &x) {
    out("OptionalStmt");
    return true;
  }
  bool pre(const OtherSpecificationStmt &x) {
    out("OtherSpecificationStmt");
    return true;
  }
  bool pre(const OutputImpliedDo &x) {
    out("OutputImpliedDo");
    return true;
  }
  bool pre(const OutputItem &x) {
    out("OutputItem");
    return true;
  }
  bool pre(const Parameter &x) {
    out("Parameter");
    return true;
  }
  bool pre(const ParameterStmt &x) {
    out("ParameterStmt");
    return true;
  }
  bool pre(const ParentIdentifier &x) {
    out("ParentIdentifier");
    return true;
  }
  bool pre(const PartRef &x) {
    out("PartRef");
    return true;
  }
  bool pre(const Pass &x) {
    out("Pass");
    return true;
  }
  bool pre(const PauseStmt &x) {
    out("PauseStmt");
    return true;
  }
  bool pre(const Pointer &x) {
    out("Pointer");
    return true;
  }
  bool pre(const PointerAssignmentStmt &x) {
    out("PointerAssignmentStmt");
    return true;
  }
  bool pre(const PointerAssignmentStmt::Bounds &x) {
    out("PointerAssignmentStmt::Bounds");
    return true;
  }
  bool pre(const PointerDecl &x) {
    out("PointerDecl");
    return true;
  }
  bool pre(const PointerObject &x) {
    out("PointerObject");
    return true;
  }
  bool pre(const PointerStmt &x) {
    out("PointerStmt");
    return true;
  }
  bool pre(const PositionOrFlushSpec &x) {
    out("PositionOrFlushSpec");
    return true;
  }
  bool pre(const PrefixSpec &x) {
    out("PrefixSpec");
    return true;
  }
  bool pre(const PrefixSpec::Elemental &x) {
    out("PrefixSpec::Elemental");
    return true;
  }
  bool pre(const PrefixSpec::Impure &x) {
    out("PrefixSpec::Impure");
    return true;
  }
  bool pre(const PrefixSpec::Module &x) {
    out("PrefixSpec::Module");
    return true;
  }
  bool pre(const PrefixSpec::Non_Recursive &x) {
    out("PrefixSpec::Non_Recursive");
    return true;
  }
  bool pre(const PrefixSpec::Pure &x) {
    out("PrefixSpec::Pure");
    return true;
  }
  bool pre(const PrefixSpec::Recursive &x) {
    out("PrefixSpec::Recursive");
    return true;
  }
  bool pre(const PrintStmt &x) {
    out("PrintStmt");
    return true;
  }
  bool pre(const PrivateOrSequence &x) {
    out("PrivateOrSequence");
    return true;
  }
  bool pre(const PrivateStmt &x) {
    out("PrivateStmt");
    return true;
  }
  bool pre(const ProcAttrSpec &x) {
    out("ProcAttrSpec");
    return true;
  }
  bool pre(const ProcComponentAttrSpec &x) {
    out("ProcComponentAttrSpec");
    return true;
  }
  bool pre(const ProcComponentDefStmt &x) {
    out("ProcComponentDefStmt");
    return true;
  }
  bool pre(const ProcComponentRef &x) {
    out("ProcComponentRef");
    return true;
  }
  bool pre(const ProcDecl &x) {
    out("ProcDecl");
    return true;
  }
  bool pre(const ProcInterface &x) {
    out("ProcInterface");
    return true;
  }
  bool pre(const ProcPointerInit &x) {
    out("ProcPointerInit");
    return true;
  }
  bool pre(const ProcedureDeclarationStmt &x) {
    out("ProcedureDeclarationStmt");
    return true;
  }
  bool pre(const ProcedureDesignator &x) {
    out("ProcedureDesignator");
    return true;
  }
  bool pre(const ProcedureStmt &x) {
    out("ProcedureStmt");
    return true;
  }
  bool pre(const Program &x) {
    out("Program");
    return true;
  }
  bool pre(const ProgramUnit &x) {
    out("ProgramUnit");
    return true;
  }
  bool pre(const Protected &x) {
    out("Protected");
    return true;
  }
  bool pre(const ProtectedStmt &x) {
    out("ProtectedStmt");
    return true;
  }
  bool pre(const ReadStmt &x) {
    out("ReadStmt");
    return true;
  }
  bool pre(const RealLiteralConstant &x) {
    out("RealLiteralConstant");
    return true;
  }
  bool pre(const RedimensionStmt &x) {
    out("RedimensionStmt");
    return true;
  }
  bool pre(const Rename &x) {
    out("Rename");
    return true;
  }
  bool pre(const Rename::Names &x) {
    out("Rename::Names");
    return true;
  }
  bool pre(const Rename::Operators &x) {
    out("Rename::Operators");
    return true;
  }
  bool pre(const ReturnStmt &x) {
    out("ReturnStmt");
    return true;
  }
  bool pre(const RewindStmt &x) {
    out("RewindStmt");
    return true;
  }
  bool pre(const Save &x) {
    out("Save");
    return true;
  }
  bool pre(const SaveStmt &x) {
    out("SaveStmt");
    return true;
  }
  bool pre(const SavedEntity &x) {
    out("SavedEntity");
    return true;
  }
  bool pre(const SectionSubscript &x) {
    out("SectionSubscript");
    return true;
  }
  bool pre(const SelectCaseStmt &x) {
    out("SelectCaseStmt");
    return true;
  }
  bool pre(const SelectRankCaseStmt &x) {
    out("SelectRankCaseStmt");
    return true;
  }
  bool pre(const SelectRankCaseStmt::Rank &x) {
    out("SelectRankCaseStmt::Rank");
    return true;
  }
  bool pre(const SelectRankConstruct &x) {
    out("SelectRankConstruct");
    return true;
  }
  bool pre(const SelectRankConstruct::RankCase &x) {
    out("SelectRankConstruct::RankCase");
    return true;
  }
  bool pre(const SelectRankStmt &x) {
    out("SelectRankStmt");
    return true;
  }
  bool pre(const SelectTypeConstruct &x) {
    out("SelectTypeConstruct");
    return true;
  }
  bool pre(const SelectTypeConstruct::TypeCase &x) {
    out("SelectTypeConstruct::TypeCase");
    return true;
  }
  bool pre(const SelectTypeStmt &x) {
    out("SelectTypeStmt");
    return true;
  }
  bool pre(const Selector &x) {
    out("Selector");
    return true;
  }
  bool pre(const SeparateModuleSubprogram &x) {
    out("SeparateModuleSubprogram");
    return true;
  }
  bool pre(const SequenceStmt &x) {
    out("SequenceStmt");
    return true;
  }
  bool pre(const SignedComplexLiteralConstant &x) {
    out("SignedComplexLiteralConstant");
    return true;
  }
  bool pre(const SignedIntLiteralConstant &x) {
    out("SignedIntLiteralConstant");
    return true;
  }
  bool pre(const SignedRealLiteralConstant &x) {
    out("SignedRealLiteralConstant");
    return true;
  }
  bool pre(const SpecificationConstruct &x) {
    out("SpecificationConstruct");
    return true;
  }
  bool pre(const SpecificationExpr &x) {
    out("SpecificationExpr");
    return true;
  }
  bool pre(const SpecificationPart &x) {
    out("SpecificationPart");
    return true;
  }
  bool pre(const Star &x) {
    out("Star");
    return true;
  }
  bool pre(const StatOrErrmsg &x) {
    out("StatOrErrmsg");
    return true;
  }
  bool pre(const StatVariable &x) {
    out("StatVariable");
    return true;
  }
  bool pre(const StatusExpr &x) {
    out("StatusExpr");
    return true;
  }
  bool pre(const StmtFunctionStmt &x) {
    out("StmtFunctionStmt");
    return true;
  }
  bool pre(const StopCode &x) {
    out("StopCode");
    return true;
  }
  bool pre(const StopStmt &x) {
    out("StopStmt");
    return true;
  }
  bool pre(const StructureComponent &x) {
    out("StructureComponent");
    return true;
  }
  bool pre(const StructureConstructor &x) {
    out("StructureConstructor");
    return true;
  }
  bool pre(const StructureDef &x) {
    out("StructureDef");
    return true;
  }
  bool pre(const StructureDef::EndStructureStmt &x) {
    out("StructureDef::EndStructureStmt");
    return true;
  }
  bool pre(const StructureField &x) {
    out("StructureField");
    return true;
  }
  bool pre(const StructureStmt &x) {
    out("StructureStmt");
    return true;
  }
  bool pre(const Submodule &x) {
    out("Submodule");
    return true;
  }
  bool pre(const SubmoduleStmt &x) {
    out("SubmoduleStmt");
    return true;
  }
  bool pre(const SubroutineStmt &x) {
    out("SubroutineStmt");
    return true;
  }
  bool pre(const SubroutineSubprogram &x) {
    out("SubroutineSubprogram");
    return true;
  }
  bool pre(const SubscriptTriplet &x) {
    out("SubscriptTriplet");
    return true;
  }
  bool pre(const Substring &x) {
    out("Substring");
    return true;
  }
  bool pre(const SubstringRange &x) {
    out("SubstringRange");
    return true;
  }
  bool pre(const Suffix &x) {
    out("Suffix");
    return true;
  }
  bool pre(const SyncAllStmt &x) {
    out("SyncAllStmt");
    return true;
  }
  bool pre(const SyncImagesStmt &x) {
    out("SyncImagesStmt");
    return true;
  }
  bool pre(const SyncImagesStmt::ImageSet &x) {
    out("SyncImagesStmt::ImageSet");
    return true;
  }
  bool pre(const SyncMemoryStmt &x) {
    out("SyncMemoryStmt");
    return true;
  }
  bool pre(const SyncTeamStmt &x) {
    out("SyncTeamStmt");
    return true;
  }
  bool pre(const Target &x) {
    out("Target");
    return true;
  }
  bool pre(const TargetStmt &x) {
    out("TargetStmt");
    return true;
  }
  bool pre(const TypeAttrSpec &x) {
    out("TypeAttrSpec");
    return true;
  }
  bool pre(const TypeAttrSpec::BindC &x) {
    out("TypeAttrSpec::BindC");
    return true;
  }
  bool pre(const TypeAttrSpec::Extends &x) {
    out("TypeAttrSpec::Extends");
    return true;
  }
  bool pre(const TypeBoundGenericStmt &x) {
    out("TypeBoundGenericStmt");
    return true;
  }
  bool pre(const TypeBoundProcBinding &x) {
    out("TypeBoundProcBinding");
    return true;
  }
  bool pre(const TypeBoundProcDecl &x) {
    out("TypeBoundProcDecl");
    return true;
  }
  bool pre(const TypeBoundProcedurePart &x) {
    out("TypeBoundProcedurePart");
    return true;
  }
  bool pre(const TypeBoundProcedureStmt &x) {
    out("TypeBoundProcedureStmt");
    return true;
  }
  bool pre(const TypeBoundProcedureStmt::WithInterface &x) {
    out("TypeBoundProcedureStmt::WithInterface");
    return true;
  }
  bool pre(const TypeBoundProcedureStmt::WithoutInterface &x) {
    out("TypeBoundProcedureStmt::WithoutInterface");
    return true;
  }
  bool pre(const TypeDeclarationStmt &x) {
    out("TypeDeclarationStmt");
    return true;
  }
  bool pre(const TypeGuardStmt &x) {
    out("TypeGuardStmt");
    return true;
  }
  bool pre(const TypeGuardStmt::Guard &x) {
    out("TypeGuardStmt::Guard");
    return true;
  }
  bool pre(const TypeParamDecl &x) {
    out("TypeParamDecl");
    return true;
  }
  bool pre(const TypeParamDefStmt &x) {
    out("TypeParamDefStmt");
    return true;
  }
  bool pre(const TypeParamInquiry &x) {
    out("TypeParamInquiry");
    return true;
  }
  bool pre(const TypeParamSpec &x) {
    out("TypeParamSpec");
    return true;
  }
  bool pre(const TypeParamValue &x) {
    out("TypeParamValue");
    return true;
  }
  bool pre(const TypeParamValue::Deferred &x) {
    out("TypeParamValue::Deferred");
    return true;
  }
  bool pre(const TypeSpec &x) {
    out("TypeSpec");
    return true;
  }
  bool pre(const Union &x) {
    out("Union");
    return true;
  }
  bool pre(const Union::EndUnionStmt &x) {
    out("Union::EndUnionStmt");
    return true;
  }
  bool pre(const Union::UnionStmt &x) {
    out("Union::UnionStmt");
    return true;
  }
  bool pre(const UnlockStmt &x) {
    out("UnlockStmt");
    return true;
  }
  bool pre(const UseStmt &x) {
    out("UseStmt");
    return true;
  }
  bool pre(const Value &x) {
    out("Value");
    return true;
  }
  bool pre(const ValueStmt &x) {
    out("ValueStmt");
    return true;
  }
  bool pre(const Variable &x) {
    out("Variable");
    return true;
  }
  bool pre(const Volatile &x) {
    out("Volatile");
    return true;
  }
  bool pre(const VolatileStmt &x) {
    out("VolatileStmt");
    return true;
  }
  bool pre(const WaitSpec &x) {
    out("WaitSpec");
    return true;
  }
  bool pre(const WaitStmt &x) {
    out("WaitStmt");
    return true;
  }
  bool pre(const WhereBodyConstruct &x) {
    out("WhereBodyConstruct");
    return true;
  }
  bool pre(const WhereConstruct &x) {
    out("WhereConstruct");
    return true;
  }
  bool pre(const WhereConstruct::Elsewhere &x) {
    out("WhereConstruct::Elsewhere");
    return true;
  }
  bool pre(const WhereConstruct::MaskedElsewhere &x) {
    out("WhereConstruct::MaskedElsewhere");
    return true;
  }
  bool pre(const WhereConstructStmt &x) {
    out("WhereConstructStmt");
    return true;
  }
  bool pre(const WhereStmt &x) {
    out("WhereStmt");
    return true;
  }
  bool pre(const WriteStmt &x) {
    out("WriteStmt");
    return true;
  }

  template<typename T>
  bool pre(const LoopBounds<T> &x) {
    out("LoopBounds");
    return true;
  }
  template<typename T>
  bool pre(const Statement<T> &x) {
    out("Statement");
    return true;
  }
  bool pre(const int &x) {
    out(std::string{"int: "} + std::to_string(x));
    return true;
  }
  bool pre(const std::uint64_t &x) {
    out(std::string{"std::uint64_t: "} + std::to_string(x));
    return true;
  }
  bool pre(const std::string &x) {
    out(std::string{"std::string: "} + x);
    return true;
  }
  bool pre(const std::int64_t &x) {
    out(std::string{"std::int64_t: "} + std::to_string(x));
    return true;
  }
  bool pre(const char &x) {
    out(std::string{"char: "} + x);
    return true;
  }
  bool pre(const Sign &x) {
    out(std::string{"Sign: "} + (x == Sign::Positive ? "+" : "-"));
    return true;
  }

  template<typename T>
  bool pre(const T &x) {
    out("generic");
    return true;
  }

  template<typename T>
  void post(const T &) {
    --indent_;
  }
};

int main(int argc, char *const argv[]) {
  if (argc != 2) {
    std::cerr << "Expected 1 source file, got " << (argc - 1) << "\n";
    return EXIT_FAILURE;
  }

  std::string path{argv[1]};
  AllSources allSources;
  std::stringstream error;
  const auto *sourceFile = allSources.Open(path, &error);
  if (!sourceFile) {
    std::cerr << error.str() << '\n';
    return 1;
  }

  ProvenanceRange range{allSources.AddIncludedFile(
      *sourceFile, ProvenanceRange{})};
  Messages messages{allSources};
  CookedSource cooked{&allSources};
  Preprocessor preprocessor{&allSources};
  bool prescanOk{Prescanner{&messages, &cooked, &preprocessor}.Prescan(range)};
  messages.Emit(std::cerr);
  if (!prescanOk) {
    return EXIT_FAILURE;
  }
  cooked.Marshal();
  ParseState state{cooked};
  UserState ustate;
  std::optional<Program> result{program.Parse(&state)};
  if (!result.has_value() || state.anyErrorRecovery()) {
    std::cerr << "parse FAILED\n";
    state.messages()->Emit(std::cerr);
    return EXIT_FAILURE;
  }

  Visitor visitor;
  visit(*result, visitor);
  return EXIT_SUCCESS;
}
