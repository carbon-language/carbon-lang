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

  bool Pre(const Abstract &x) {
    out("Abstract");
    return true;
  }
  bool Pre(const AcImpliedDo &x) {
    out("AcImpliedDo");
    return true;
  }
  bool Pre(const AcImpliedDoControl &x) {
    out("AcImpliedDoControl");
    return true;
  }
  bool Pre(const AcSpec &x) {
    out("AcSpec");
    return true;
  }
  bool Pre(const AcValue &x) {
    out("AcValue");
    return true;
  }
  bool Pre(const AcValue::Triplet &x) {
    out("AcValue::Triplet");
    return true;
  }
  bool Pre(const AccessId &x) {
    out("AccessId");
    return true;
  }
  bool Pre(const AccessSpec &x) {
    out("AccessSpec");
    return true;
  }
  bool Pre(const AccessStmt &x) {
    out("AccessStmt");
    return true;
  }
  bool Pre(const ActionStmt &x) {
    out("ActionStmt");
    return true;
  }
  bool Pre(const ActualArg &x) {
    out("ActualArg");
    return true;
  }
  bool Pre(const ActualArg::PercentRef &x) {
    out("ActualArg::PercentRef");
    return true;
  }
  bool Pre(const ActualArg::PercentVal &x) {
    out("ActualArg::PercentVal");
    return true;
  }
  bool Pre(const ActualArgSpec &x) {
    out("ActualArgSpec");
    return true;
  }
  bool Pre(const AllocOpt &x) {
    out("AllocOpt");
    return true;
  }
  bool Pre(const AllocOpt::Mold &x) {
    out("AllocOpt::Mold");
    return true;
  }
  bool Pre(const AllocOpt::Source &x) {
    out("AllocOpt::Source");
    return true;
  }
  bool Pre(const Allocatable &x) {
    out("Allocatable");
    return true;
  }
  bool Pre(const AllocatableStmt &x) {
    out("AllocatableStmt");
    return true;
  }
  bool Pre(const AllocateCoarraySpec &x) {
    out("AllocateCoarraySpec");
    return true;
  }
  bool Pre(const AllocateObject &x) {
    out("AllocateObject");
    return true;
  }
  bool Pre(const AllocateShapeSpec &x) {
    out("AllocateShapeSpec");
    return true;
  }
  bool Pre(const AllocateStmt &x) {
    out("AllocateStmt");
    return true;
  }
  bool Pre(const Allocation &x) {
    out("Allocation");
    return true;
  }
  bool Pre(const AltReturnSpec &x) {
    out("AltReturnSpec");
    return true;
  }
  bool Pre(const ArithmeticIfStmt &x) {
    out("ArithmeticIfStmt");
    return true;
  }
  bool Pre(const ArrayConstructor &x) {
    out("ArrayConstructor");
    return true;
  }
  bool Pre(const ArrayElement &x) {
    out("ArrayElement");
    return true;
  }
  bool Pre(const ArraySection &x) {
    out("ArraySection");
    return true;
  }
  bool Pre(const ArraySpec &x) {
    out("ArraySpec");
    return true;
  }
  bool Pre(const AssignStmt &x) {
    out("AssignStmt");
    return true;
  }
  bool Pre(const AssignedGotoStmt &x) {
    out("AssignedGotoStmt");
    return true;
  }
  bool Pre(const AssignmentStmt &x) {
    out("AssignmentStmt");
    return true;
  }
  bool Pre(const AssociateConstruct &x) {
    out("AssociateConstruct");
    return true;
  }
  bool Pre(const AssociateStmt &x) {
    out("AssociateStmt");
    return true;
  }
  bool Pre(const Association &x) {
    out("Association");
    return true;
  }
  bool Pre(const AssumedImpliedSpec &x) {
    out("AssumedImpliedSpec");
    return true;
  }
  bool Pre(const AssumedRankSpec &x) {
    out("AssumedRankSpec");
    return true;
  }
  bool Pre(const AssumedShapeSpec &x) {
    out("AssumedShapeSpec");
    return true;
  }
  bool Pre(const AssumedSizeSpec &x) {
    out("AssumedSizeSpec");
    return true;
  }
  bool Pre(const Asynchronous &x) {
    out("Asynchronous");
    return true;
  }
  bool Pre(const AsynchronousStmt &x) {
    out("AsynchronousStmt");
    return true;
  }
  bool Pre(const AttrSpec &x) {
    out("AttrSpec");
    return true;
  }
  bool Pre(const BOZLiteralConstant &x) {
    out("BOZLiteralConstant");
    return true;
  }
  bool Pre(const BackspaceStmt &x) {
    out("BackspaceStmt");
    return true;
  }
  bool Pre(const BasedPointerStmt &x) {
    out("BasedPointerStmt");
    return true;
  }
  bool Pre(const BindAttr &x) {
    out("BindAttr");
    return true;
  }
  bool Pre(const BindAttr::Deferred &x) {
    out("BindAttr::Deferred");
    return true;
  }
  bool Pre(const BindAttr::Non_Overridable &x) {
    out("BindAttr::Non_Overridable");
    return true;
  }
  bool Pre(const BindEntity &x) {
    out("BindEntity");
    return true;
  }
  bool Pre(const BindStmt &x) {
    out("BindStmt");
    return true;
  }
  bool Pre(const BlockConstruct &x) {
    out("BlockConstruct");
    return true;
  }
  bool Pre(const BlockData &x) {
    out("BlockData");
    return true;
  }
  bool Pre(const BlockDataStmt &x) {
    out("BlockDataStmt");
    return true;
  }
  bool Pre(const BlockSpecificationPart &x) {
    out("BlockSpecificationPart");
    return true;
  }
  bool Pre(const BlockStmt &x) {
    out("BlockStmt");
    return true;
  }
  bool Pre(const BoundsRemapping &x) {
    out("BoundsRemapping");
    return true;
  }
  bool Pre(const BoundsSpec &x) {
    out("BoundsSpec");
    return true;
  }
  bool Pre(const Call &x) {
    out("Call");
    return true;
  }
  bool Pre(const CallStmt &x) {
    out("CallStmt");
    return true;
  }
  bool Pre(const CaseConstruct &x) {
    out("CaseConstruct");
    return true;
  }
  bool Pre(const CaseConstruct::Case &x) {
    out("CaseConstruct::Case");
    return true;
  }
  bool Pre(const CaseSelector &x) {
    out("CaseSelector");
    return true;
  }
  bool Pre(const CaseStmt &x) {
    out("CaseStmt");
    return true;
  }
  bool Pre(const CaseValueRange &x) {
    out("CaseValueRange");
    return true;
  }
  bool Pre(const CaseValueRange::Range &x) {
    out("CaseValueRange::Range");
    return true;
  }
  bool Pre(const ChangeTeamConstruct &x) {
    out("ChangeTeamConstruct");
    return true;
  }
  bool Pre(const ChangeTeamStmt &x) {
    out("ChangeTeamStmt");
    return true;
  }
  bool Pre(const CharLength &x) {
    out("CharLength");
    return true;
  }
  bool Pre(const CharLiteralConstant &x) {
    out("CharLiteralConstant");
    return true;
  }
  bool Pre(const CharLiteralConstantSubstring &x) {
    out("CharLiteralConstantSubstring");
    return true;
  }
  bool Pre(const CharSelector &x) {
    out("CharSelector");
    return true;
  }
  bool Pre(const CharSelector::LengthAndKind &x) {
    out("CharSelector::LengthAndKind");
    return true;
  }
  bool Pre(const CharVariable &x) {
    out("CharVariable");
    return true;
  }
  bool Pre(const CloseStmt &x) {
    out("CloseStmt");
    return true;
  }
  bool Pre(const CloseStmt::CloseSpec &x) {
    out("CloseStmt::CloseSpec");
    return true;
  }
  bool Pre(const CoarrayAssociation &x) {
    out("CoarrayAssociation");
    return true;
  }
  bool Pre(const CoarraySpec &x) {
    out("CoarraySpec");
    return true;
  }
  bool Pre(const CodimensionDecl &x) {
    out("CodimensionDecl");
    return true;
  }
  bool Pre(const CodimensionStmt &x) {
    out("CodimensionStmt");
    return true;
  }
  bool Pre(const CoindexedNamedObject &x) {
    out("CoindexedNamedObject");
    return true;
  }
  bool Pre(const CommonBlockObject &x) {
    out("CommonBlockObject");
    return true;
  }
  bool Pre(const CommonStmt &x) {
    out("CommonStmt");
    return true;
  }
  bool Pre(const ComplexLiteralConstant &x) {
    out("ComplexLiteralConstant");
    return true;
  }
  bool Pre(const ComplexPart &x) {
    out("ComplexPart");
    return true;
  }
  bool Pre(const ComplexPartDesignator &x) {
    out("ComplexPartDesignator");
    return true;
  }
  bool Pre(const ComponentArraySpec &x) {
    out("ComponentArraySpec");
    return true;
  }
  bool Pre(const ComponentAttrSpec &x) {
    out("ComponentAttrSpec");
    return true;
  }
  bool Pre(const ComponentDataSource &x) {
    out("ComponentDataSource");
    return true;
  }
  bool Pre(const ComponentDecl &x) {
    out("ComponentDecl");
    return true;
  }
  bool Pre(const ComponentDefStmt &x) {
    out("ComponentDefStmt");
    return true;
  }
  bool Pre(const ComponentSpec &x) {
    out("ComponentSpec");
    return true;
  }
  bool Pre(const ComputedGotoStmt &x) {
    out("ComputedGotoStmt");
    return true;
  }
  bool Pre(const ConcurrentControl &x) {
    out("ConcurrentControl");
    return true;
  }
  bool Pre(const ConcurrentHeader &x) {
    out("ConcurrentHeader");
    return true;
  }
  bool Pre(const ConnectSpec &x) {
    out("ConnectSpec");
    return true;
  }
  bool Pre(const ConnectSpec::CharExpr &x) {
    out("ConnectSpec::CharExpr");
    return true;
  }
  bool Pre(const ConnectSpec::Newunit &x) {
    out("ConnectSpec::Newunit");
    return true;
  }
  bool Pre(const ConnectSpec::Recl &x) {
    out("ConnectSpec::Recl");
    return true;
  }
  bool Pre(const ConstantValue &x) {
    out("ConstantValue");
    return true;
  }
  bool Pre(const ContainsStmt &x) {
    out("ContainsStmt");
    return true;
  }
  bool Pre(const Contiguous &x) {
    out("Contiguous");
    return true;
  }
  bool Pre(const ContiguousStmt &x) {
    out("ContiguousStmt");
    return true;
  }
  bool Pre(const ContinueStmt &x) {
    out("ContinueStmt");
    return true;
  }
  bool Pre(const CriticalConstruct &x) {
    out("CriticalConstruct");
    return true;
  }
  bool Pre(const CriticalStmt &x) {
    out("CriticalStmt");
    return true;
  }
  bool Pre(const CycleStmt &x) {
    out("CycleStmt");
    return true;
  }
  bool Pre(const DataComponentDefStmt &x) {
    out("DataComponentDefStmt");
    return true;
  }
  bool Pre(const DataIDoObject &x) {
    out("DataIDoObject");
    return true;
  }
  bool Pre(const DataImpliedDo &x) {
    out("DataImpliedDo");
    return true;
  }
  bool Pre(const DataReference &x) {
    out("DataReference");
    return true;
  }
  bool Pre(const DataStmt &x) {
    out("DataStmt");
    return true;
  }
  bool Pre(const DataStmtConstant &x) {
    out("DataStmtConstant");
    return true;
  }
  bool Pre(const DataStmtObject &x) {
    out("DataStmtObject");
    return true;
  }
  bool Pre(const DataStmtRepeat &x) {
    out("DataStmtRepeat");
    return true;
  }
  bool Pre(const DataStmtSet &x) {
    out("DataStmtSet");
    return true;
  }
  bool Pre(const DataStmtValue &x) {
    out("DataStmtValue");
    return true;
  }
  bool Pre(const DeallocateStmt &x) {
    out("DeallocateStmt");
    return true;
  }
  bool Pre(const DeclarationConstruct &x) {
    out("DeclarationConstruct");
    return true;
  }
  bool Pre(const DeclarationTypeSpec &x) {
    out("DeclarationTypeSpec");
    return true;
  }
  bool Pre(const DeclarationTypeSpec::Class &x) {
    out("DeclarationTypeSpec::Class");
    return true;
  }
  bool Pre(const DeclarationTypeSpec::ClassStar &x) {
    out("DeclarationTypeSpec::ClassStar");
    return true;
  }
  bool Pre(const DeclarationTypeSpec::Record &x) {
    out("DeclarationTypeSpec::Record");
    return true;
  }
  bool Pre(const DeclarationTypeSpec::Type &x) {
    out("DeclarationTypeSpec::Type");
    return true;
  }
  bool Pre(const DeclarationTypeSpec::TypeStar &x) {
    out("DeclarationTypeSpec::TypeStar");
    return true;
  }
  bool Pre(const Default &x) {
    out("Default");
    return true;
  }
  bool Pre(const DeferredCoshapeSpecList &x) {
    out("DeferredCoshapeSpecList");
    return true;
  }
  bool Pre(const DeferredShapeSpecList &x) {
    out("DeferredShapeSpecList");
    return true;
  }
  bool Pre(const DefinedOpName &x) {
    out("DefinedOpName");
    return true;
  }
  bool Pre(const DefinedOperator &x) {
    out("DefinedOperator");
    return true;
  }
  bool Pre(const DerivedTypeDef &x) {
    out("DerivedTypeDef");
    return true;
  }
  bool Pre(const DerivedTypeSpec &x) {
    out("DerivedTypeSpec");
    return true;
  }
  bool Pre(const DerivedTypeStmt &x) {
    out("DerivedTypeStmt");
    return true;
  }
  bool Pre(const Designator &x) {
    out("Designator");
    return true;
  }
  bool Pre(const DimensionStmt &x) {
    out("DimensionStmt");
    return true;
  }
  bool Pre(const DimensionStmt::Declaration &x) {
    out("DimensionStmt::Declaration");
    return true;
  }
  bool Pre(const DoConstruct &x) {
    out("DoConstruct");
    return true;
  }
  bool Pre(const DummyArg &x) {
    out("DummyArg");
    return true;
  }
  bool Pre(const ElseIfStmt &x) {
    out("ElseIfStmt");
    return true;
  }
  bool Pre(const ElseStmt &x) {
    out("ElseStmt");
    return true;
  }
  bool Pre(const ElsewhereStmt &x) {
    out("ElsewhereStmt");
    return true;
  }
  bool Pre(const EndAssociateStmt &x) {
    out("EndAssociateStmt");
    return true;
  }
  bool Pre(const EndBlockDataStmt &x) {
    out("EndBlockDataStmt");
    return true;
  }
  bool Pre(const EndBlockStmt &x) {
    out("EndBlockStmt");
    return true;
  }
  bool Pre(const EndChangeTeamStmt &x) {
    out("EndChangeTeamStmt");
    return true;
  }
  bool Pre(const EndCriticalStmt &x) {
    out("EndCriticalStmt");
    return true;
  }
  bool Pre(const EndDoStmt &x) {
    out("EndDoStmt");
    return true;
  }
  bool Pre(const EndEnumStmt &x) {
    out("EndEnumStmt");
    return true;
  }
  bool Pre(const EndForallStmt &x) {
    out("EndForallStmt");
    return true;
  }
  bool Pre(const EndFunctionStmt &x) {
    out("EndFunctionStmt");
    return true;
  }
  bool Pre(const EndIfStmt &x) {
    out("EndIfStmt");
    return true;
  }
  bool Pre(const EndInterfaceStmt &x) {
    out("EndInterfaceStmt");
    return true;
  }
  bool Pre(const EndLabel &x) {
    out("EndLabel");
    return true;
  }
  bool Pre(const EndModuleStmt &x) {
    out("EndModuleStmt");
    return true;
  }
  bool Pre(const EndMpSubprogramStmt &x) {
    out("EndMpSubprogramStmt");
    return true;
  }
  bool Pre(const EndProgramStmt &x) {
    out("EndProgramStmt");
    return true;
  }
  bool Pre(const EndSelectStmt &x) {
    out("EndSelectStmt");
    return true;
  }
  bool Pre(const EndSubmoduleStmt &x) {
    out("EndSubmoduleStmt");
    return true;
  }
  bool Pre(const EndSubroutineStmt &x) {
    out("EndSubroutineStmt");
    return true;
  }
  bool Pre(const EndTypeStmt &x) {
    out("EndTypeStmt");
    return true;
  }
  bool Pre(const EndWhereStmt &x) {
    out("EndWhereStmt");
    return true;
  }
  bool Pre(const EndfileStmt &x) {
    out("EndfileStmt");
    return true;
  }
  bool Pre(const EntityDecl &x) {
    out("EntityDecl");
    return true;
  }
  bool Pre(const EntryStmt &x) {
    out("EntryStmt");
    return true;
  }
  bool Pre(const EnumDef &x) {
    out("EnumDef");
    return true;
  }
  bool Pre(const EnumDefStmt &x) {
    out("EnumDefStmt");
    return true;
  }
  bool Pre(const Enumerator &x) {
    out("Enumerator");
    return true;
  }
  bool Pre(const EnumeratorDefStmt &x) {
    out("EnumeratorDefStmt");
    return true;
  }
  bool Pre(const EorLabel &x) {
    out("EorLabel");
    return true;
  }
  bool Pre(const EquivalenceObject &x) {
    out("EquivalenceObject");
    return true;
  }
  bool Pre(const EquivalenceStmt &x) {
    out("EquivalenceStmt");
    return true;
  }
  bool Pre(const ErrLabel &x) {
    out("ErrLabel");
    return true;
  }
  bool Pre(const ErrorRecovery &x) {
    out("ErrorRecovery");
    return true;
  }
  bool Pre(const EventPostStmt &x) {
    out("EventPostStmt");
    return true;
  }
  bool Pre(const EventWaitStmt &x) {
    out("EventWaitStmt");
    return true;
  }
  bool Pre(const EventWaitStmt::EventWaitSpec &x) {
    out("EventWaitStmt::EventWaitSpec");
    return true;
  }
  bool Pre(const ExecutableConstruct &x) {
    out("ExecutableConstruct");
    return true;
  }
  bool Pre(const ExecutionPartConstruct &x) {
    out("ExecutionPartConstruct");
    return true;
  }
  bool Pre(const ExitStmt &x) {
    out("ExitStmt");
    return true;
  }
  bool Pre(const ExplicitCoshapeSpec &x) {
    out("ExplicitCoshapeSpec");
    return true;
  }
  bool Pre(const ExplicitShapeSpec &x) {
    out("ExplicitShapeSpec");
    return true;
  }
  bool Pre(const ExponentPart &x) {
    out("ExponentPart");
    return true;
  }
  bool Pre(const Expr &x) {
    out("Expr");
    return true;
  }
  bool Pre(const Expr::AND &x) {
    out("Expr::AND");
    return true;
  }
  bool Pre(const Expr::Add &x) {
    out("Expr::Add");
    return true;
  }
  bool Pre(const Expr::ComplexConstructor &x) {
    out("Expr::ComplexConstructor");
    return true;
  }
  bool Pre(const Expr::Concat &x) {
    out("Expr::Concat");
    return true;
  }
  bool Pre(const Expr::DefinedBinary &x) {
    out("Expr::DefinedBinary");
    return true;
  }
  bool Pre(const Expr::DefinedUnary &x) {
    out("Expr::DefinedUnary");
    return true;
  }
  bool Pre(const Expr::Divide &x) {
    out("Expr::Divide");
    return true;
  }
  bool Pre(const Expr::EQ &x) {
    out("Expr::EQ");
    return true;
  }
  bool Pre(const Expr::EQV &x) {
    out("Expr::EQV");
    return true;
  }
  bool Pre(const Expr::GE &x) {
    out("Expr::GE");
    return true;
  }
  bool Pre(const Expr::GT &x) {
    out("Expr::GT");
    return true;
  }
  bool Pre(const Expr::IntrinsicBinary &x) {
    out("Expr::IntrinsicBinary");
    return true;
  }
  bool Pre(const Expr::IntrinsicUnary &x) {
    out("Expr::IntrinsicUnary");
    return true;
  }
  bool Pre(const Expr::LE &x) {
    out("Expr::LE");
    return true;
  }
  bool Pre(const Expr::LT &x) {
    out("Expr::LT");
    return true;
  }
  bool Pre(const Expr::Multiply &x) {
    out("Expr::Multiply");
    return true;
  }
  bool Pre(const Expr::NE &x) {
    out("Expr::NE");
    return true;
  }
  bool Pre(const Expr::NEQV &x) {
    out("Expr::NEQV");
    return true;
  }
  bool Pre(const Expr::NOT &x) {
    out("Expr::NOT");
    return true;
  }
  bool Pre(const Expr::Negate &x) {
    out("Expr::Negate");
    return true;
  }
  bool Pre(const Expr::OR &x) {
    out("Expr::OR");
    return true;
  }
  bool Pre(const Expr::Parentheses &x) {
    out("Expr::Parentheses");
    return true;
  }
  bool Pre(const Expr::PercentLoc &x) {
    out("Expr::PercentLoc");
    return true;
  }
  bool Pre(const Expr::Power &x) {
    out("Expr::Power");
    return true;
  }
  bool Pre(const Expr::Subtract &x) {
    out("Expr::Subtract");
    return true;
  }
  bool Pre(const Expr::UnaryPlus &x) {
    out("Expr::UnaryPlus");
    return true;
  }
  bool Pre(const External &x) {
    out("External");
    return true;
  }
  bool Pre(const ExternalStmt &x) {
    out("ExternalStmt");
    return true;
  }
  bool Pre(const FailImageStmt &x) {
    out("FailImageStmt");
    return true;
  }
  bool Pre(const FileUnitNumber &x) {
    out("FileUnitNumber");
    return true;
  }
  bool Pre(const FinalProcedureStmt &x) {
    out("FinalProcedureStmt");
    return true;
  }
  bool Pre(const FlushStmt &x) {
    out("FlushStmt");
    return true;
  }
  bool Pre(const ForallAssignmentStmt &x) {
    out("ForallAssignmentStmt");
    return true;
  }
  bool Pre(const ForallBodyConstruct &x) {
    out("ForallBodyConstruct");
    return true;
  }
  bool Pre(const ForallConstruct &x) {
    out("ForallConstruct");
    return true;
  }
  bool Pre(const ForallConstructStmt &x) {
    out("ForallConstructStmt");
    return true;
  }
  bool Pre(const ForallStmt &x) {
    out("ForallStmt");
    return true;
  }
  bool Pre(const FormTeamStmt &x) {
    out("FormTeamStmt");
    return true;
  }
  bool Pre(const FormTeamStmt::FormTeamSpec &x) {
    out("FormTeamStmt::FormTeamSpec");
    return true;
  }
  bool Pre(const Format &x) {
    out("Format");
    return true;
  }
  bool Pre(const FormatStmt &x) {
    out("FormatStmt");
    return true;
  }
  bool Pre(const Fortran::ControlEditDesc &x) {
    out("Fortran::ControlEditDesc");
    return true;
  }
  bool Pre(const Fortran::DerivedTypeDataEditDesc &x) {
    out("Fortran::DerivedTypeDataEditDesc");
    return true;
  }
  bool Pre(const Fortran::FormatItem &x) {
    out("Fortran::FormatItem");
    return true;
  }
  bool Pre(const Fortran::FormatSpecification &x) {
    out("Fortran::FormatSpecification");
    return true;
  }
  bool Pre(const Fortran::IntrinsicTypeDataEditDesc &x) {
    out("Fortran::IntrinsicTypeDataEditDesc");
    return true;
  }
  bool Pre(const FunctionReference &x) {
    out("FunctionReference");
    return true;
  }
  bool Pre(const FunctionStmt &x) {
    out("FunctionStmt");
    return true;
  }
  bool Pre(const FunctionSubprogram &x) {
    out("FunctionSubprogram");
    return true;
  }
  bool Pre(const GenericSpec &x) {
    out("GenericSpec");
    return true;
  }
  bool Pre(const GenericSpec::Assignment &x) {
    out("GenericSpec::Assignment");
    return true;
  }
  bool Pre(const GenericSpec::ReadFormatted &x) {
    out("GenericSpec::ReadFormatted");
    return true;
  }
  bool Pre(const GenericSpec::ReadUnformatted &x) {
    out("GenericSpec::ReadUnformatted");
    return true;
  }
  bool Pre(const GenericSpec::WriteFormatted &x) {
    out("GenericSpec::WriteFormatted");
    return true;
  }
  bool Pre(const GenericSpec::WriteUnformatted &x) {
    out("GenericSpec::WriteUnformatted");
    return true;
  }
  bool Pre(const GenericStmt &x) {
    out("GenericStmt");
    return true;
  }
  bool Pre(const GotoStmt &x) {
    out("GotoStmt");
    return true;
  }
  bool Pre(const HollerithLiteralConstant &x) {
    out("HollerithLiteralConstant");
    return true;
  }
  bool Pre(const IdExpr &x) {
    out("IdExpr");
    return true;
  }
  bool Pre(const IdVariable &x) {
    out("IdVariable");
    return true;
  }
  bool Pre(const IfConstruct &x) {
    out("IfConstruct");
    return true;
  }
  bool Pre(const IfConstruct::ElseBlock &x) {
    out("IfConstruct::ElseBlock");
    return true;
  }
  bool Pre(const IfConstruct::ElseIfBlock &x) {
    out("IfConstruct::ElseIfBlock");
    return true;
  }
  bool Pre(const IfStmt &x) {
    out("IfStmt");
    return true;
  }
  bool Pre(const IfThenStmt &x) {
    out("IfThenStmt");
    return true;
  }
  bool Pre(const ImageSelector &x) {
    out("ImageSelector");
    return true;
  }
  bool Pre(const ImageSelectorSpec &x) {
    out("ImageSelectorSpec");
    return true;
  }
  bool Pre(const ImageSelectorSpec::Stat &x) {
    out("ImageSelectorSpec::Stat");
    return true;
  }
  bool Pre(const ImageSelectorSpec::Team &x) {
    out("ImageSelectorSpec::Team");
    return true;
  }
  bool Pre(const ImageSelectorSpec::Team_Number &x) {
    out("ImageSelectorSpec::Team_Number");
    return true;
  }
  bool Pre(const ImplicitPart &x) {
    out("ImplicitPart");
    return true;
  }
  bool Pre(const ImplicitPartStmt &x) {
    out("ImplicitPartStmt");
    return true;
  }
  bool Pre(const ImplicitSpec &x) {
    out("ImplicitSpec");
    return true;
  }
  bool Pre(const ImplicitStmt &x) {
    out("ImplicitStmt");
    return true;
  }
  bool Pre(const ImpliedShapeSpec &x) {
    out("ImpliedShapeSpec");
    return true;
  }
  bool Pre(const ImportStmt &x) {
    out("ImportStmt");
    return true;
  }
  bool Pre(const Initialization &x) {
    out("Initialization");
    return true;
  }
  bool Pre(const InputImpliedDo &x) {
    out("InputImpliedDo");
    return true;
  }
  bool Pre(const InputItem &x) {
    out("InputItem");
    return true;
  }
  bool Pre(const InquireSpec &x) {
    out("InquireSpec");
    return true;
  }
  bool Pre(const InquireSpec::CharVar &x) {
    out("InquireSpec::CharVar");
    return true;
  }
  bool Pre(const InquireSpec::IntVar &x) {
    out("InquireSpec::IntVar");
    return true;
  }
  bool Pre(const InquireSpec::LogVar &x) {
    out("InquireSpec::LogVar");
    return true;
  }
  bool Pre(const InquireStmt &x) {
    out("InquireStmt");
    return true;
  }
  bool Pre(const InquireStmt::Iolength &x) {
    out("InquireStmt::Iolength");
    return true;
  }
  bool Pre(const IntLiteralConstant &x) {
    out("IntLiteralConstant");
    return true;
  }
  bool Pre(const IntegerTypeSpec &x) {
    out("IntegerTypeSpec");
    return true;
  }
  bool Pre(const IntentSpec &x) {
    out("IntentSpec");
    return true;
  }
  bool Pre(const IntentStmt &x) {
    out("IntentStmt");
    return true;
  }
  bool Pre(const InterfaceBlock &x) {
    out("InterfaceBlock");
    return true;
  }
  bool Pre(const InterfaceBody &x) {
    out("InterfaceBody");
    return true;
  }
  bool Pre(const InterfaceBody::Function &x) {
    out("InterfaceBody::Function");
    return true;
  }
  bool Pre(const InterfaceBody::Subroutine &x) {
    out("InterfaceBody::Subroutine");
    return true;
  }
  bool Pre(const InterfaceSpecification &x) {
    out("InterfaceSpecification");
    return true;
  }
  bool Pre(const InterfaceStmt &x) {
    out("InterfaceStmt");
    return true;
  }
  bool Pre(const InternalSubprogram &x) {
    out("InternalSubprogram");
    return true;
  }
  bool Pre(const InternalSubprogramPart &x) {
    out("InternalSubprogramPart");
    return true;
  }
  bool Pre(const Intrinsic &x) {
    out("Intrinsic");
    return true;
  }
  bool Pre(const IntrinsicStmt &x) {
    out("IntrinsicStmt");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec &x) {
    out("IntrinsicTypeSpec");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Character &x) {
    out("IntrinsicTypeSpec::Character");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Complex &x) {
    out("IntrinsicTypeSpec::Complex");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::DoubleComplex &x) {
    out("IntrinsicTypeSpec::DoubleComplex");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::DoublePrecision &x) {
    out("IntrinsicTypeSpec::DoublePrecision");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Logical &x) {
    out("IntrinsicTypeSpec::Logical");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::NCharacter &x) {
    out("IntrinsicTypeSpec::NCharacter");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Real &x) {
    out("IntrinsicTypeSpec::Real");
    return true;
  }
  bool Pre(const IoControlSpec &x) {
    out("IoControlSpec");
    return true;
  }
  bool Pre(const IoControlSpec::Asynchronous &x) {
    out("IoControlSpec::Asynchronous");
    return true;
  }
  bool Pre(const IoControlSpec::CharExpr &x) {
    out("IoControlSpec::CharExpr");
    return true;
  }
  bool Pre(const IoControlSpec::Pos &x) {
    out("IoControlSpec::Pos");
    return true;
  }
  bool Pre(const IoControlSpec::Rec &x) {
    out("IoControlSpec::Rec");
    return true;
  }
  bool Pre(const IoControlSpec::Size &x) {
    out("IoControlSpec::Size");
    return true;
  }
  bool Pre(const IoUnit &x) {
    out("IoUnit");
    return true;
  }
  bool Pre(const KindParam &x) {
    out("KindParam");
    return true;
  }
  bool Pre(const KindParam::Kanji &x) {
    out("KindParam::Kanji");
    return true;
  }
  bool Pre(const KindSelector &x) {
    out("KindSelector");
    return true;
  }
  bool Pre(const LabelDoStmt &x) {
    out("LabelDoStmt");
    return true;
  }
  bool Pre(const LanguageBindingSpec &x) {
    out("LanguageBindingSpec");
    return true;
  }
  bool Pre(const LengthSelector &x) {
    out("LengthSelector");
    return true;
  }
  bool Pre(const LetterSpec &x) {
    out("LetterSpec");
    return true;
  }
  bool Pre(const LiteralConstant &x) {
    out("LiteralConstant");
    return true;
  }
  bool Pre(const LocalitySpec &x) {
    out("LocalitySpec");
    return true;
  }
  bool Pre(const LocalitySpec::DefaultNone &x) {
    out("LocalitySpec::DefaultNone");
    return true;
  }
  bool Pre(const LocalitySpec::Local &x) {
    out("LocalitySpec::Local");
    return true;
  }
  bool Pre(const LocalitySpec::LocalInit &x) {
    out("LocalitySpec::LocalInit");
    return true;
  }
  bool Pre(const LocalitySpec::Shared &x) {
    out("LocalitySpec::Shared");
    return true;
  }
  bool Pre(const LockStmt &x) {
    out("LockStmt");
    return true;
  }
  bool Pre(const LockStmt::LockStat &x) {
    out("LockStmt::LockStat");
    return true;
  }
  bool Pre(const LogicalLiteralConstant &x) {
    out("LogicalLiteralConstant");
    return true;
  }
  bool Pre(const LoopControl &x) {
    out("LoopControl");
    return true;
  }
  bool Pre(const LoopControl::Concurrent &x) {
    out("LoopControl::Concurrent");
    return true;
  }
  bool Pre(const MainProgram &x) {
    out("MainProgram");
    return true;
  }
  bool Pre(const Map &x) {
    out("Map");
    return true;
  }
  bool Pre(const Map::EndMapStmt &x) {
    out("Map::EndMapStmt");
    return true;
  }
  bool Pre(const Map::MapStmt &x) {
    out("Map::MapStmt");
    return true;
  }
  bool Pre(const MaskedElsewhereStmt &x) {
    out("MaskedElsewhereStmt");
    return true;
  }
  bool Pre(const Module &x) {
    out("Module");
    return true;
  }
  bool Pre(const ModuleStmt &x) {
    out("ModuleStmt");
    return true;
  }
  bool Pre(const ModuleSubprogram &x) {
    out("ModuleSubprogram");
    return true;
  }
  bool Pre(const ModuleSubprogramPart &x) {
    out("ModuleSubprogramPart");
    return true;
  }
  bool Pre(const MpSubprogramStmt &x) {
    out("MpSubprogramStmt");
    return true;
  }
  bool Pre(const MsgVariable &x) {
    out("MsgVariable");
    return true;
  }
  bool Pre(const NamedConstant &x) {
    out("NamedConstant");
    return true;
  }
  bool Pre(const NamedConstantDef &x) {
    out("NamedConstantDef");
    return true;
  }
  bool Pre(const NamelistStmt &x) {
    out("NamelistStmt");
    return true;
  }
  bool Pre(const NamelistStmt::Group &x) {
    out("NamelistStmt::Group");
    return true;
  }
  bool Pre(const NoPass &x) {
    out("NoPass");
    return true;
  }
  bool Pre(const NonLabelDoStmt &x) {
    out("NonLabelDoStmt");
    return true;
  }
  bool Pre(const NullInit &x) {
    out("NullInit");
    return true;
  }
  bool Pre(const NullifyStmt &x) {
    out("NullifyStmt");
    return true;
  }
  bool Pre(const ObjectDecl &x) {
    out("ObjectDecl");
    return true;
  }
  bool Pre(const Only &x) {
    out("Only");
    return true;
  }
  bool Pre(const OpenStmt &x) {
    out("OpenStmt");
    return true;
  }
  bool Pre(const Optional &x) {
    out("Optional");
    return true;
  }
  bool Pre(const OptionalStmt &x) {
    out("OptionalStmt");
    return true;
  }
  bool Pre(const OtherSpecificationStmt &x) {
    out("OtherSpecificationStmt");
    return true;
  }
  bool Pre(const OutputImpliedDo &x) {
    out("OutputImpliedDo");
    return true;
  }
  bool Pre(const OutputItem &x) {
    out("OutputItem");
    return true;
  }
  bool Pre(const Parameter &x) {
    out("Parameter");
    return true;
  }
  bool Pre(const ParameterStmt &x) {
    out("ParameterStmt");
    return true;
  }
  bool Pre(const ParentIdentifier &x) {
    out("ParentIdentifier");
    return true;
  }
  bool Pre(const PartRef &x) {
    out("PartRef");
    return true;
  }
  bool Pre(const Pass &x) {
    out("Pass");
    return true;
  }
  bool Pre(const PauseStmt &x) {
    out("PauseStmt");
    return true;
  }
  bool Pre(const Pointer &x) {
    out("Pointer");
    return true;
  }
  bool Pre(const PointerAssignmentStmt &x) {
    out("PointerAssignmentStmt");
    return true;
  }
  bool Pre(const PointerAssignmentStmt::Bounds &x) {
    out("PointerAssignmentStmt::Bounds");
    return true;
  }
  bool Pre(const PointerDecl &x) {
    out("PointerDecl");
    return true;
  }
  bool Pre(const PointerObject &x) {
    out("PointerObject");
    return true;
  }
  bool Pre(const PointerStmt &x) {
    out("PointerStmt");
    return true;
  }
  bool Pre(const PositionOrFlushSpec &x) {
    out("PositionOrFlushSpec");
    return true;
  }
  bool Pre(const PrefixSpec &x) {
    out("PrefixSpec");
    return true;
  }
  bool Pre(const PrefixSpec::Elemental &x) {
    out("PrefixSpec::Elemental");
    return true;
  }
  bool Pre(const PrefixSpec::Impure &x) {
    out("PrefixSpec::Impure");
    return true;
  }
  bool Pre(const PrefixSpec::Module &x) {
    out("PrefixSpec::Module");
    return true;
  }
  bool Pre(const PrefixSpec::Non_Recursive &x) {
    out("PrefixSpec::Non_Recursive");
    return true;
  }
  bool Pre(const PrefixSpec::Pure &x) {
    out("PrefixSpec::Pure");
    return true;
  }
  bool Pre(const PrefixSpec::Recursive &x) {
    out("PrefixSpec::Recursive");
    return true;
  }
  bool Pre(const PrintStmt &x) {
    out("PrintStmt");
    return true;
  }
  bool Pre(const PrivateOrSequence &x) {
    out("PrivateOrSequence");
    return true;
  }
  bool Pre(const PrivateStmt &x) {
    out("PrivateStmt");
    return true;
  }
  bool Pre(const ProcAttrSpec &x) {
    out("ProcAttrSpec");
    return true;
  }
  bool Pre(const ProcComponentAttrSpec &x) {
    out("ProcComponentAttrSpec");
    return true;
  }
  bool Pre(const ProcComponentDefStmt &x) {
    out("ProcComponentDefStmt");
    return true;
  }
  bool Pre(const ProcComponentRef &x) {
    out("ProcComponentRef");
    return true;
  }
  bool Pre(const ProcDecl &x) {
    out("ProcDecl");
    return true;
  }
  bool Pre(const ProcInterface &x) {
    out("ProcInterface");
    return true;
  }
  bool Pre(const ProcPointerInit &x) {
    out("ProcPointerInit");
    return true;
  }
  bool Pre(const ProcedureDeclarationStmt &x) {
    out("ProcedureDeclarationStmt");
    return true;
  }
  bool Pre(const ProcedureDesignator &x) {
    out("ProcedureDesignator");
    return true;
  }
  bool Pre(const ProcedureStmt &x) {
    out("ProcedureStmt");
    return true;
  }
  bool Pre(const Program &x) {
    out("Program");
    return true;
  }
  bool Pre(const ProgramUnit &x) {
    out("ProgramUnit");
    return true;
  }
  bool Pre(const Protected &x) {
    out("Protected");
    return true;
  }
  bool Pre(const ProtectedStmt &x) {
    out("ProtectedStmt");
    return true;
  }
  bool Pre(const ReadStmt &x) {
    out("ReadStmt");
    return true;
  }
  bool Pre(const RealLiteralConstant &x) {
    out("RealLiteralConstant");
    return true;
  }
  bool Pre(const RedimensionStmt &x) {
    out("RedimensionStmt");
    return true;
  }
  bool Pre(const Rename &x) {
    out("Rename");
    return true;
  }
  bool Pre(const Rename::Names &x) {
    out("Rename::Names");
    return true;
  }
  bool Pre(const Rename::Operators &x) {
    out("Rename::Operators");
    return true;
  }
  bool Pre(const ReturnStmt &x) {
    out("ReturnStmt");
    return true;
  }
  bool Pre(const RewindStmt &x) {
    out("RewindStmt");
    return true;
  }
  bool Pre(const Save &x) {
    out("Save");
    return true;
  }
  bool Pre(const SaveStmt &x) {
    out("SaveStmt");
    return true;
  }
  bool Pre(const SavedEntity &x) {
    out("SavedEntity");
    return true;
  }
  bool Pre(const SectionSubscript &x) {
    out("SectionSubscript");
    return true;
  }
  bool Pre(const SelectCaseStmt &x) {
    out("SelectCaseStmt");
    return true;
  }
  bool Pre(const SelectRankCaseStmt &x) {
    out("SelectRankCaseStmt");
    return true;
  }
  bool Pre(const SelectRankCaseStmt::Rank &x) {
    out("SelectRankCaseStmt::Rank");
    return true;
  }
  bool Pre(const SelectRankConstruct &x) {
    out("SelectRankConstruct");
    return true;
  }
  bool Pre(const SelectRankConstruct::RankCase &x) {
    out("SelectRankConstruct::RankCase");
    return true;
  }
  bool Pre(const SelectRankStmt &x) {
    out("SelectRankStmt");
    return true;
  }
  bool Pre(const SelectTypeConstruct &x) {
    out("SelectTypeConstruct");
    return true;
  }
  bool Pre(const SelectTypeConstruct::TypeCase &x) {
    out("SelectTypeConstruct::TypeCase");
    return true;
  }
  bool Pre(const SelectTypeStmt &x) {
    out("SelectTypeStmt");
    return true;
  }
  bool Pre(const Selector &x) {
    out("Selector");
    return true;
  }
  bool Pre(const SeparateModuleSubprogram &x) {
    out("SeparateModuleSubprogram");
    return true;
  }
  bool Pre(const SequenceStmt &x) {
    out("SequenceStmt");
    return true;
  }
  bool Pre(const SignedComplexLiteralConstant &x) {
    out("SignedComplexLiteralConstant");
    return true;
  }
  bool Pre(const SignedIntLiteralConstant &x) {
    out("SignedIntLiteralConstant");
    return true;
  }
  bool Pre(const SignedRealLiteralConstant &x) {
    out("SignedRealLiteralConstant");
    return true;
  }
  bool Pre(const SpecificationConstruct &x) {
    out("SpecificationConstruct");
    return true;
  }
  bool Pre(const SpecificationExpr &x) {
    out("SpecificationExpr");
    return true;
  }
  bool Pre(const SpecificationPart &x) {
    out("SpecificationPart");
    return true;
  }
  bool Pre(const Star &x) {
    out("Star");
    return true;
  }
  bool Pre(const StatOrErrmsg &x) {
    out("StatOrErrmsg");
    return true;
  }
  bool Pre(const StatVariable &x) {
    out("StatVariable");
    return true;
  }
  bool Pre(const StatusExpr &x) {
    out("StatusExpr");
    return true;
  }
  bool Pre(const StmtFunctionStmt &x) {
    out("StmtFunctionStmt");
    return true;
  }
  bool Pre(const StopCode &x) {
    out("StopCode");
    return true;
  }
  bool Pre(const StopStmt &x) {
    out("StopStmt");
    return true;
  }
  bool Pre(const StructureComponent &x) {
    out("StructureComponent");
    return true;
  }
  bool Pre(const StructureConstructor &x) {
    out("StructureConstructor");
    return true;
  }
  bool Pre(const StructureDef &x) {
    out("StructureDef");
    return true;
  }
  bool Pre(const StructureDef::EndStructureStmt &x) {
    out("StructureDef::EndStructureStmt");
    return true;
  }
  bool Pre(const StructureField &x) {
    out("StructureField");
    return true;
  }
  bool Pre(const StructureStmt &x) {
    out("StructureStmt");
    return true;
  }
  bool Pre(const Submodule &x) {
    out("Submodule");
    return true;
  }
  bool Pre(const SubmoduleStmt &x) {
    out("SubmoduleStmt");
    return true;
  }
  bool Pre(const SubroutineStmt &x) {
    out("SubroutineStmt");
    return true;
  }
  bool Pre(const SubroutineSubprogram &x) {
    out("SubroutineSubprogram");
    return true;
  }
  bool Pre(const SubscriptTriplet &x) {
    out("SubscriptTriplet");
    return true;
  }
  bool Pre(const Substring &x) {
    out("Substring");
    return true;
  }
  bool Pre(const SubstringRange &x) {
    out("SubstringRange");
    return true;
  }
  bool Pre(const Suffix &x) {
    out("Suffix");
    return true;
  }
  bool Pre(const SyncAllStmt &x) {
    out("SyncAllStmt");
    return true;
  }
  bool Pre(const SyncImagesStmt &x) {
    out("SyncImagesStmt");
    return true;
  }
  bool Pre(const SyncImagesStmt::ImageSet &x) {
    out("SyncImagesStmt::ImageSet");
    return true;
  }
  bool Pre(const SyncMemoryStmt &x) {
    out("SyncMemoryStmt");
    return true;
  }
  bool Pre(const SyncTeamStmt &x) {
    out("SyncTeamStmt");
    return true;
  }
  bool Pre(const Target &x) {
    out("Target");
    return true;
  }
  bool Pre(const TargetStmt &x) {
    out("TargetStmt");
    return true;
  }
  bool Pre(const TypeAttrSpec &x) {
    out("TypeAttrSpec");
    return true;
  }
  bool Pre(const TypeAttrSpec::BindC &x) {
    out("TypeAttrSpec::BindC");
    return true;
  }
  bool Pre(const TypeAttrSpec::Extends &x) {
    out("TypeAttrSpec::Extends");
    return true;
  }
  bool Pre(const TypeBoundGenericStmt &x) {
    out("TypeBoundGenericStmt");
    return true;
  }
  bool Pre(const TypeBoundProcBinding &x) {
    out("TypeBoundProcBinding");
    return true;
  }
  bool Pre(const TypeBoundProcDecl &x) {
    out("TypeBoundProcDecl");
    return true;
  }
  bool Pre(const TypeBoundProcedurePart &x) {
    out("TypeBoundProcedurePart");
    return true;
  }
  bool Pre(const TypeBoundProcedureStmt &x) {
    out("TypeBoundProcedureStmt");
    return true;
  }
  bool Pre(const TypeBoundProcedureStmt::WithInterface &x) {
    out("TypeBoundProcedureStmt::WithInterface");
    return true;
  }
  bool Pre(const TypeBoundProcedureStmt::WithoutInterface &x) {
    out("TypeBoundProcedureStmt::WithoutInterface");
    return true;
  }
  bool Pre(const TypeDeclarationStmt &x) {
    out("TypeDeclarationStmt");
    return true;
  }
  bool Pre(const TypeGuardStmt &x) {
    out("TypeGuardStmt");
    return true;
  }
  bool Pre(const TypeGuardStmt::Guard &x) {
    out("TypeGuardStmt::Guard");
    return true;
  }
  bool Pre(const TypeParamDecl &x) {
    out("TypeParamDecl");
    return true;
  }
  bool Pre(const TypeParamDefStmt &x) {
    out("TypeParamDefStmt");
    return true;
  }
  bool Pre(const TypeParamInquiry &x) {
    out("TypeParamInquiry");
    return true;
  }
  bool Pre(const TypeParamSpec &x) {
    out("TypeParamSpec");
    return true;
  }
  bool Pre(const TypeParamValue &x) {
    out("TypeParamValue");
    return true;
  }
  bool Pre(const TypeParamValue::Deferred &x) {
    out("TypeParamValue::Deferred");
    return true;
  }
  bool Pre(const TypeSpec &x) {
    out("TypeSpec");
    return true;
  }
  bool Pre(const Union &x) {
    out("Union");
    return true;
  }
  bool Pre(const Union::EndUnionStmt &x) {
    out("Union::EndUnionStmt");
    return true;
  }
  bool Pre(const Union::UnionStmt &x) {
    out("Union::UnionStmt");
    return true;
  }
  bool Pre(const UnlockStmt &x) {
    out("UnlockStmt");
    return true;
  }
  bool Pre(const UseStmt &x) {
    out("UseStmt");
    return true;
  }
  bool Pre(const Value &x) {
    out("Value");
    return true;
  }
  bool Pre(const ValueStmt &x) {
    out("ValueStmt");
    return true;
  }
  bool Pre(const Variable &x) {
    out("Variable");
    return true;
  }
  bool Pre(const Volatile &x) {
    out("Volatile");
    return true;
  }
  bool Pre(const VolatileStmt &x) {
    out("VolatileStmt");
    return true;
  }
  bool Pre(const WaitSpec &x) {
    out("WaitSpec");
    return true;
  }
  bool Pre(const WaitStmt &x) {
    out("WaitStmt");
    return true;
  }
  bool Pre(const WhereBodyConstruct &x) {
    out("WhereBodyConstruct");
    return true;
  }
  bool Pre(const WhereConstruct &x) {
    out("WhereConstruct");
    return true;
  }
  bool Pre(const WhereConstruct::Elsewhere &x) {
    out("WhereConstruct::Elsewhere");
    return true;
  }
  bool Pre(const WhereConstruct::MaskedElsewhere &x) {
    out("WhereConstruct::MaskedElsewhere");
    return true;
  }
  bool Pre(const WhereConstructStmt &x) {
    out("WhereConstructStmt");
    return true;
  }
  bool Pre(const WhereStmt &x) {
    out("WhereStmt");
    return true;
  }
  bool Pre(const WriteStmt &x) {
    out("WriteStmt");
    return true;
  }

  template<typename T>
  bool Pre(const LoopBounds<T> &x) {
    out("LoopBounds");
    return true;
  }
  template<typename T>
  bool Pre(const Statement<T> &x) {
    out("Statement");
    return true;
  }
  bool Pre(const int &x) {
    out(std::string{"int: "} + std::to_string(x));
    return true;
  }
  bool Pre(const std::uint64_t &x) {
    out(std::string{"std::uint64_t: "} + std::to_string(x));
    return true;
  }
  bool Pre(const std::string &x) {
    out(std::string{"std::string: "} + x);
    return true;
  }
  bool Pre(const std::int64_t &x) {
    out(std::string{"std::int64_t: "} + std::to_string(x));
    return true;
  }
  bool Pre(const char &x) {
    out(std::string{"char: "} + x);
    return true;
  }
  bool Pre(const Sign &x) {
    out(std::string{"Sign: "} + (x == Sign::Positive ? "+" : "-"));
    return true;
  }

  template<typename T>
  bool Pre(const T &x) {
    out("generic");
    return true;
  }

  template<typename T>
  void Post(const T &) {
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
  Walk(*result, visitor);
  return EXIT_SUCCESS;
}
