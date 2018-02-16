#include "parse-tree.h"
#include "idioms.h"
#include "indirection.h"
#include <algorithm>

namespace Fortran {
namespace parser {

#define UNION_FORMATTER(TYPE) \
  std::ostream &operator<<(std::ostream &o, const TYPE &x) { \
    return o << "(" #TYPE " " << x.u << ')'; \
  }

UNION_FORMATTER(ProgramUnit)  // R502
UNION_FORMATTER(ImplicitPartStmt)  // R506
UNION_FORMATTER(DeclarationConstruct)  // R507
UNION_FORMATTER(SpecificationConstruct)  // R508
UNION_FORMATTER(ExecutionPartConstruct)  // R510
UNION_FORMATTER(InternalSubprogram)  // R512
UNION_FORMATTER(OtherSpecificationStmt)  // R513
UNION_FORMATTER(ExecutableConstruct)  // R514
UNION_FORMATTER(ActionStmt)  // R515
UNION_FORMATTER(ConstantValue)  // R604
UNION_FORMATTER(LiteralConstant)  // R605
UNION_FORMATTER(DefinedOperator)  // R609
UNION_FORMATTER(TypeParamValue)  // R701
UNION_FORMATTER(TypeSpec)  // R702
UNION_FORMATTER(DeclarationTypeSpec)  // R703
UNION_FORMATTER(IntrinsicTypeSpec)  // R704
UNION_FORMATTER(KindParam)  // R709
UNION_FORMATTER(CharSelector)  // R721
UNION_FORMATTER(ComplexPart)  // R718 & R719
UNION_FORMATTER(LengthSelector)  // R722
UNION_FORMATTER(CharLength)  // R723
UNION_FORMATTER(TypeAttrSpec)  // R728
UNION_FORMATTER(PrivateOrSequence)  // R729
UNION_FORMATTER(ComponentDefStmt)  // R736
UNION_FORMATTER(ComponentAttrSpec)  // R738
UNION_FORMATTER(ComponentArraySpec)  // R740
UNION_FORMATTER(ProcComponentAttrSpec)  // R742
UNION_FORMATTER(Initialization)  // R743 & R805
UNION_FORMATTER(TypeBoundProcBinding)  // R748
UNION_FORMATTER(TypeBoundProcedureStmt)  // R749
UNION_FORMATTER(BindAttr)  // R752
UNION_FORMATTER(AcValue)  // R773
UNION_FORMATTER(AttrSpec)  // R802
UNION_FORMATTER(CoarraySpec)  // R809
UNION_FORMATTER(ArraySpec)  // R815
UNION_FORMATTER(AccessId)  // R828
UNION_FORMATTER(DataStmtObject)  // R839
UNION_FORMATTER(DataIDoObject)  // R841
UNION_FORMATTER(DataStmtRepeat)  // R844
UNION_FORMATTER(DataStmtConstant)  // R845
UNION_FORMATTER(Designator)  // R901
UNION_FORMATTER(Variable)  // R902
UNION_FORMATTER(DataReference)  // R911
UNION_FORMATTER(SectionSubscript)  // R920
UNION_FORMATTER(ImageSelectorSpec)  // R926
UNION_FORMATTER(StatOrErrmsg)  // R928, R942 & R1165
UNION_FORMATTER(AllocOpt)  // R928
UNION_FORMATTER(AllocateObject)  // R933
UNION_FORMATTER(PointerObject)  // R940
UNION_FORMATTER(Expr)  // R1001
UNION_FORMATTER(PointerAssignmentStmt::Bounds)  // R1033
UNION_FORMATTER(WhereBodyConstruct)  // R1044
UNION_FORMATTER(ForallBodyConstruct)  // R1052
UNION_FORMATTER(ForallAssignmentStmt)  // R1053
UNION_FORMATTER(Selector)  // R1105
UNION_FORMATTER(LoopControl)  // R1123
UNION_FORMATTER(LocalitySpec)  // R1130
UNION_FORMATTER(CaseSelector)  // R1145
UNION_FORMATTER(CaseValueRange)  // R1146
UNION_FORMATTER(SelectRankCaseStmt::Rank)  // R1150
UNION_FORMATTER(TypeGuardStmt::Guard)  // R1154
UNION_FORMATTER(StopCode)  // R1162
UNION_FORMATTER(SyncImagesStmt::ImageSet)  // R1167
UNION_FORMATTER(EventWaitStmt::EventWaitSpec)  // R1173
UNION_FORMATTER(FormTeamStmt::FormTeamSpec)  // R1177
UNION_FORMATTER(LockStmt::LockStat)  // R1179
UNION_FORMATTER(IoUnit)  // R1201, R1203
UNION_FORMATTER(ConnectSpec)  // R1205
UNION_FORMATTER(CloseStmt::CloseSpec)  // R1209
UNION_FORMATTER(IoControlSpec)  // R1213
UNION_FORMATTER(Format)  // R1215
UNION_FORMATTER(InputItem)  // R1216
UNION_FORMATTER(OutputItem)  // R1217
UNION_FORMATTER(WaitSpec)  // R1223
UNION_FORMATTER(BackspaceStmt)  // R1224
UNION_FORMATTER(EndfileStmt)  // R1225
UNION_FORMATTER(RewindStmt)  // R1226
UNION_FORMATTER(PositionOrFlushSpec)  // R1227 & R1229
UNION_FORMATTER(FlushStmt)  // R1228
UNION_FORMATTER(InquireStmt)  // R1230
UNION_FORMATTER(InquireSpec)  // R1231
UNION_FORMATTER(ModuleSubprogram)  // R1408
UNION_FORMATTER(Rename)  // R1411
UNION_FORMATTER(Only)  // R1412
UNION_FORMATTER(InterfaceSpecification)  // R1502
UNION_FORMATTER(InterfaceStmt)  // R1503
UNION_FORMATTER(InterfaceBody)  // R1505
UNION_FORMATTER(GenericSpec)  // R1508
UNION_FORMATTER(ProcInterface)  // R1513
UNION_FORMATTER(ProcAttrSpec)  // R1514
UNION_FORMATTER(ProcPointerInit)  // R1517
UNION_FORMATTER(ProcedureDesignator)  // R1522
UNION_FORMATTER(ActualArg)  // R1524
UNION_FORMATTER(PrefixSpec)  // R1527
UNION_FORMATTER(DummyArg)  // R1536
UNION_FORMATTER(StructureField)  // legacy extension

#undef UNION_FORMATTER

#define TUPLE_FORMATTER(TYPE) \
  std::ostream &operator<<(std::ostream &o, const TYPE &x) { \
    return o << "(" #TYPE " " << x.t << ')'; \
  }

TUPLE_FORMATTER(SpecificationPart)  // R504
TUPLE_FORMATTER(InternalSubprogramPart)  // R511
TUPLE_FORMATTER(SignedIntLiteralConstant)  // R707
TUPLE_FORMATTER(IntLiteralConstant)  // R708
TUPLE_FORMATTER(SignedRealLiteralConstant)  // R713
TUPLE_FORMATTER(ExponentPart)  // R717
TUPLE_FORMATTER(ComplexLiteralConstant)  // R718
TUPLE_FORMATTER(SignedComplexLiteralConstant)  // R718
TUPLE_FORMATTER(CharLiteralConstant)  // R724
TUPLE_FORMATTER(DerivedTypeDef)  // R726, R735
TUPLE_FORMATTER(DerivedTypeStmt)  // R727
TUPLE_FORMATTER(TypeParamDefStmt)  // R732
TUPLE_FORMATTER(TypeParamDecl)  // R733
TUPLE_FORMATTER(DataComponentDefStmt)  // R737
TUPLE_FORMATTER(ComponentDecl)  // R739
TUPLE_FORMATTER(ProcComponentDefStmt)  // R741
TUPLE_FORMATTER(TypeBoundProcedurePart)  // R746
TUPLE_FORMATTER(TypeBoundProcDecl)  // R750
TUPLE_FORMATTER(TypeBoundGenericStmt)  // R751
TUPLE_FORMATTER(DerivedTypeSpec)  // R754
TUPLE_FORMATTER(TypeParamSpec)  // R755
TUPLE_FORMATTER(EnumDef)  // R759
TUPLE_FORMATTER(StructureConstructor)  // R756
TUPLE_FORMATTER(ComponentSpec)  // R757
TUPLE_FORMATTER(Enumerator)  // R762
TUPLE_FORMATTER(AcValue::Triplet)  // R773
TUPLE_FORMATTER(AcImpliedDo)  // R774
TUPLE_FORMATTER(AcImpliedDoControl)  // R775
TUPLE_FORMATTER(TypeDeclarationStmt)  // R801
TUPLE_FORMATTER(EntityDecl)  // R803
TUPLE_FORMATTER(ExplicitCoshapeSpec)  // R811
TUPLE_FORMATTER(ExplicitShapeSpec)  // R816
TUPLE_FORMATTER(AssumedSizeSpec)  // R822
TUPLE_FORMATTER(AccessStmt)  // R827
TUPLE_FORMATTER(ObjectDecl)  // R830 & R860
TUPLE_FORMATTER(BindStmt)  // R832
TUPLE_FORMATTER(BindEntity)  // R833
TUPLE_FORMATTER(CodimensionDecl)  // R835
TUPLE_FORMATTER(DataStmtSet)  // R838
TUPLE_FORMATTER(DataImpliedDo)  // R840
TUPLE_FORMATTER(DataStmtValue)  // R843
TUPLE_FORMATTER(DimensionStmt::Declaration)  // R848
TUPLE_FORMATTER(IntentStmt)  // R849
TUPLE_FORMATTER(NamedConstantDef)  // R852
TUPLE_FORMATTER(PointerDecl)  // R854
TUPLE_FORMATTER(SavedEntity)  // R857, R858
TUPLE_FORMATTER(ImplicitSpec)  // R864
TUPLE_FORMATTER(LetterSpec)  // R865
TUPLE_FORMATTER(NamelistStmt::Group)  // R868, R869
TUPLE_FORMATTER(CommonStmt::Block)  // R873
TUPLE_FORMATTER(CommonStmt)  // R873
TUPLE_FORMATTER(CommonBlockObject)  // R874
TUPLE_FORMATTER(Substring)  // R908, R909
TUPLE_FORMATTER(CharLiteralConstantSubstring)
TUPLE_FORMATTER(SubstringRange)  // R910
TUPLE_FORMATTER(SubscriptTriplet)  // R921
TUPLE_FORMATTER(ImageSelector)  // R924
TUPLE_FORMATTER(AllocateStmt)  // R927
TUPLE_FORMATTER(Allocation)  // R932
TUPLE_FORMATTER(AllocateShapeSpec)  // R934
TUPLE_FORMATTER(AllocateCoarraySpec)  // R937
TUPLE_FORMATTER(DeallocateStmt)  // R941
TUPLE_FORMATTER(Expr::DefinedUnary)  // R1002
TUPLE_FORMATTER(Expr::IntrinsicBinary)
TUPLE_FORMATTER(Expr::Power)
TUPLE_FORMATTER(Expr::Multiply)
TUPLE_FORMATTER(Expr::Divide)
TUPLE_FORMATTER(Expr::Add)
TUPLE_FORMATTER(Expr::Subtract)
TUPLE_FORMATTER(Expr::Concat)
TUPLE_FORMATTER(Expr::LT)
TUPLE_FORMATTER(Expr::LE)
TUPLE_FORMATTER(Expr::EQ)
TUPLE_FORMATTER(Expr::NE)
TUPLE_FORMATTER(Expr::GE)
TUPLE_FORMATTER(Expr::GT)
TUPLE_FORMATTER(Expr::AND)
TUPLE_FORMATTER(Expr::OR)
TUPLE_FORMATTER(Expr::EQV)
TUPLE_FORMATTER(Expr::NEQV)
TUPLE_FORMATTER(Expr::ComplexConstructor)
TUPLE_FORMATTER(Expr::DefinedBinary)  // R1022
TUPLE_FORMATTER(AssignmentStmt)  // R1032
TUPLE_FORMATTER(PointerAssignmentStmt)  // R1033
TUPLE_FORMATTER(BoundsRemapping)  // R1036
TUPLE_FORMATTER(ProcComponentRef)  // R1039
TUPLE_FORMATTER(WhereStmt)  // R1041, R1045, R1046
TUPLE_FORMATTER(WhereConstruct)  // R1042
TUPLE_FORMATTER(WhereConstruct::MaskedElsewhere)  // R1042
TUPLE_FORMATTER(WhereConstruct::Elsewhere)  // R1042
TUPLE_FORMATTER(WhereConstructStmt)  // R1043, R1046
TUPLE_FORMATTER(MaskedElsewhereStmt)  // R1047
TUPLE_FORMATTER(ForallConstruct)  // R1050
TUPLE_FORMATTER(ForallConstructStmt)  // R1051
TUPLE_FORMATTER(ForallStmt)  // R1055
TUPLE_FORMATTER(AssociateConstruct)  // R1102
TUPLE_FORMATTER(AssociateStmt)  // R1103
TUPLE_FORMATTER(Association)  // R1104
TUPLE_FORMATTER(BlockConstruct)  // R1107
TUPLE_FORMATTER(ChangeTeamConstruct)  // R1111
TUPLE_FORMATTER(ChangeTeamStmt)  // R1112
TUPLE_FORMATTER(CoarrayAssociation)  // R1113
TUPLE_FORMATTER(EndChangeTeamStmt)  // R1114
TUPLE_FORMATTER(CriticalConstruct)  // R1116
TUPLE_FORMATTER(CriticalStmt)  // R1117
TUPLE_FORMATTER(DoConstruct)  // R1119
TUPLE_FORMATTER(LabelDoStmt)  // R1121
TUPLE_FORMATTER(NonLabelDoStmt)  // R1122
TUPLE_FORMATTER(LoopControl::Concurrent)  // R1123
TUPLE_FORMATTER(ConcurrentHeader)  // R1125
TUPLE_FORMATTER(ConcurrentControl)  // R1126
TUPLE_FORMATTER(IfConstruct::ElseIfBlock)  // R1134
TUPLE_FORMATTER(IfConstruct::ElseBlock)  // R1134
TUPLE_FORMATTER(IfConstruct)  // R1134
TUPLE_FORMATTER(IfThenStmt)  // R1135
TUPLE_FORMATTER(ElseIfStmt)  // R1136
TUPLE_FORMATTER(IfStmt)  // R1139
TUPLE_FORMATTER(CaseConstruct)  // R1140
TUPLE_FORMATTER(CaseConstruct::Case)  // R1140
TUPLE_FORMATTER(SelectCaseStmt)  // R1141, R1144
TUPLE_FORMATTER(CaseStmt)  // R1142
TUPLE_FORMATTER(SelectRankConstruct)  // R1148
TUPLE_FORMATTER(SelectRankConstruct::RankCase)  // R1148
TUPLE_FORMATTER(SelectRankStmt)  // R1149
TUPLE_FORMATTER(SelectRankCaseStmt)  // R1150
TUPLE_FORMATTER(SelectTypeConstruct)  // R1152
TUPLE_FORMATTER(SelectTypeConstruct::TypeCase)  // R1152
TUPLE_FORMATTER(SelectTypeStmt)  // R1153
TUPLE_FORMATTER(TypeGuardStmt)  // R1154
TUPLE_FORMATTER(ComputedGotoStmt)  // R1158
TUPLE_FORMATTER(StopStmt)  // R1160, R1161
TUPLE_FORMATTER(SyncImagesStmt)  // R1166
TUPLE_FORMATTER(SyncTeamStmt)  // R1169
TUPLE_FORMATTER(EventPostStmt)  // R1170, R1171
TUPLE_FORMATTER(EventWaitStmt)  // R1172
TUPLE_FORMATTER(FormTeamStmt)  // R1175
TUPLE_FORMATTER(LockStmt)  // R1178
TUPLE_FORMATTER(UnlockStmt)  // R1180
TUPLE_FORMATTER(ConnectSpec::CharExpr)  // R1205
TUPLE_FORMATTER(PrintStmt)  // R1212
TUPLE_FORMATTER(IoControlSpec::CharExpr)  // R1213
TUPLE_FORMATTER(InputImpliedDo)  // R1218, R1219
TUPLE_FORMATTER(OutputImpliedDo)  // R1218, R1219
TUPLE_FORMATTER(InquireStmt::Iolength)  // R1230
TUPLE_FORMATTER(InquireSpec::CharVar)  // R1231
TUPLE_FORMATTER(InquireSpec::IntVar)  // R1231
TUPLE_FORMATTER(InquireSpec::LogVar)  // R1231
TUPLE_FORMATTER(MainProgram)  // R1401
TUPLE_FORMATTER(Module)  // R1404
TUPLE_FORMATTER(ModuleSubprogramPart)  // R1407
// TUPLE_FORMATTER(Rename::Names)  // R1411
TUPLE_FORMATTER(Rename::Operators)  // R1414, R1415
TUPLE_FORMATTER(Submodule)  // R1416
TUPLE_FORMATTER(SubmoduleStmt)  // R1417
TUPLE_FORMATTER(ParentIdentifier)  // R1418
TUPLE_FORMATTER(BlockData)  // R1420
TUPLE_FORMATTER(InterfaceBlock)  // R1501
TUPLE_FORMATTER(InterfaceBody::Function)  // R1505
TUPLE_FORMATTER(InterfaceBody::Subroutine)  // R1505
TUPLE_FORMATTER(GenericStmt)  // R1510
TUPLE_FORMATTER(ProcedureDeclarationStmt)  // R1512
TUPLE_FORMATTER(ProcDecl)  // R1515
TUPLE_FORMATTER(Call)  // R1520 & R1521
TUPLE_FORMATTER(ActualArgSpec)  // R1523
TUPLE_FORMATTER(FunctionSubprogram)  // R1529
TUPLE_FORMATTER(FunctionStmt)  // R1530
TUPLE_FORMATTER(SubroutineSubprogram)  // R1534
TUPLE_FORMATTER(SubroutineStmt)  // R1535
TUPLE_FORMATTER(SeparateModuleSubprogram)  // R1538
TUPLE_FORMATTER(EntryStmt)  // R1541
TUPLE_FORMATTER(StmtFunctionStmt)  // R1544

// Extensions and legacies
TUPLE_FORMATTER(BasedPointerStmt)
TUPLE_FORMATTER(RedimensionStmt)
TUPLE_FORMATTER(StructureStmt)
TUPLE_FORMATTER(StructureDef)
TUPLE_FORMATTER(Union)
TUPLE_FORMATTER(Map)
TUPLE_FORMATTER(ArithmeticIfStmt)
TUPLE_FORMATTER(AssignStmt)
TUPLE_FORMATTER(AssignedGotoStmt)

std::ostream &operator<<(std::ostream &o, const Rename::Names &x) {  // R1411
  return o << "(Rename::Names " << std::get<0>(x.t) << ' ' << std::get<1>(x.t)
           << ')';
}

#undef TUPLE_FORMATTER

// R1302 format-specification
std::ostream &operator<<(std::ostream &o, const FormatSpecification &x) {
  return o << "(FormatSpecification " << x.items << ' ' << x.unlimitedItems
           << ')';
}

#define NESTED_ENUM_FORMATTER(T) \
  NESTED_ENUM_TO_STRING(T) \
  std::ostream &operator<<(std::ostream &o, const T &x) { \
    return o << ToString(x); \
  }

NESTED_ENUM_FORMATTER(DefinedOperator::IntrinsicOperator)  // R608
NESTED_ENUM_FORMATTER(TypeParamDefStmt::KindOrLen)  // R734
NESTED_ENUM_FORMATTER(AccessSpec::Kind)  // R807
NESTED_ENUM_FORMATTER(IntentSpec::Intent)  // R826
NESTED_ENUM_FORMATTER(ImplicitStmt::ImplicitNoneNameSpec)  // R866
NESTED_ENUM_FORMATTER(ImportStmt::Kind)  // R867
NESTED_ENUM_FORMATTER(StopStmt::Kind)  // R1160, R1161
NESTED_ENUM_FORMATTER(ConnectSpec::CharExpr::Kind)  // R1205
NESTED_ENUM_FORMATTER(IoControlSpec::CharExpr::Kind)  // R1213
NESTED_ENUM_FORMATTER(InquireSpec::CharVar::Kind)  // R1231
NESTED_ENUM_FORMATTER(InquireSpec::IntVar::Kind)  // R1231
NESTED_ENUM_FORMATTER(InquireSpec::LogVar::Kind)  // R1231
NESTED_ENUM_FORMATTER(UseStmt::ModuleNature)  // R1410
NESTED_ENUM_FORMATTER(ProcedureStmt::Kind)  // R1506

#undef NESTED_ENUM_FORMATTER

// Wrapper class formatting
#define WRAPPER_FORMATTER(TYPE) \
  std::ostream &operator<<(std::ostream &o, const TYPE &x) { \
    return o << "(" #TYPE " " << x.v << ')'; \
  }

WRAPPER_FORMATTER(Program)  // R501
WRAPPER_FORMATTER(ImplicitPart)  // R505
WRAPPER_FORMATTER(NamedConstant)  // R606
WRAPPER_FORMATTER(DefinedOpName)  // R1003, R1023, R1414, R1415
WRAPPER_FORMATTER(DeclarationTypeSpec::Record)  // R703 extension
WRAPPER_FORMATTER(IntrinsicTypeSpec::NCharacter)  // R704 extension
WRAPPER_FORMATTER(IntegerTypeSpec)  // R705
WRAPPER_FORMATTER(KindSelector)  // R706
WRAPPER_FORMATTER(HollerithLiteralConstant)  // extension
WRAPPER_FORMATTER(LogicalLiteralConstant)  // R725
WRAPPER_FORMATTER(TypeAttrSpec::Extends)  // R728
WRAPPER_FORMATTER(EndTypeStmt)  // R730
WRAPPER_FORMATTER(Pass)  // R742 & R752
WRAPPER_FORMATTER(FinalProcedureStmt)  // R753
WRAPPER_FORMATTER(ComponentDataSource)  // R758
WRAPPER_FORMATTER(EnumeratorDefStmt)  // R761
WRAPPER_FORMATTER(BOZLiteralConstant)  // R764, R765, R766, R767
WRAPPER_FORMATTER(ArrayConstructor)  // R769
WRAPPER_FORMATTER(AccessSpec)  // R807
WRAPPER_FORMATTER(LanguageBindingSpec)  // R808 & R1528
WRAPPER_FORMATTER(DeferredCoshapeSpecList)  // R810
WRAPPER_FORMATTER(AssumedShapeSpec)  // R819
WRAPPER_FORMATTER(DeferredShapeSpecList)  // R820
WRAPPER_FORMATTER(AssumedImpliedSpec)  // R821
WRAPPER_FORMATTER(ImpliedShapeSpec)  // R823 & R824
WRAPPER_FORMATTER(IntentSpec)  // R826
WRAPPER_FORMATTER(AllocatableStmt)  // R829
WRAPPER_FORMATTER(AsynchronousStmt)  // R831
WRAPPER_FORMATTER(CodimensionStmt)  // R834
WRAPPER_FORMATTER(ContiguousStmt)  // R836
WRAPPER_FORMATTER(DataStmt)  // R837
WRAPPER_FORMATTER(DimensionStmt)  // R848
WRAPPER_FORMATTER(OptionalStmt)  // R850
WRAPPER_FORMATTER(ParameterStmt)  // R851
WRAPPER_FORMATTER(PointerStmt)  // R853
WRAPPER_FORMATTER(ProtectedStmt)  // R855
WRAPPER_FORMATTER(SaveStmt)  // R856
WRAPPER_FORMATTER(TargetStmt)  // R859
WRAPPER_FORMATTER(ValueStmt)  // R861
WRAPPER_FORMATTER(VolatileStmt)  // R862
WRAPPER_FORMATTER(NamelistStmt)  // R868
WRAPPER_FORMATTER(EquivalenceStmt)  // R870, R871
WRAPPER_FORMATTER(EquivalenceObject)  // R872
WRAPPER_FORMATTER(CharVariable)  // R905
WRAPPER_FORMATTER(ComplexPartDesignator)  // R915
WRAPPER_FORMATTER(TypeParamInquiry)  // R916
WRAPPER_FORMATTER(ArraySection)  // R918
WRAPPER_FORMATTER(ImageSelectorSpec::Stat)  // R926
WRAPPER_FORMATTER(ImageSelectorSpec::Team)  // R926
WRAPPER_FORMATTER(ImageSelectorSpec::Team_Number)  // R926
WRAPPER_FORMATTER(AllocOpt::Mold)  // R928
WRAPPER_FORMATTER(AllocOpt::Source)  // R928
WRAPPER_FORMATTER(StatVariable)  // R929
WRAPPER_FORMATTER(MsgVariable)  // R930 & R1207
WRAPPER_FORMATTER(NullifyStmt)  // R939
WRAPPER_FORMATTER(Expr::Parentheses)  // R1001
WRAPPER_FORMATTER(Expr::UnaryPlus)  // R1006, R1009
WRAPPER_FORMATTER(Expr::Negate)  // R1006, R1009
WRAPPER_FORMATTER(Expr::NOT)  // R1014, R1018
WRAPPER_FORMATTER(Expr::PercentLoc)  // extension
WRAPPER_FORMATTER(SpecificationExpr)  // R1028
WRAPPER_FORMATTER(BoundsSpec)  // R1035
WRAPPER_FORMATTER(ElsewhereStmt)  // R1048
WRAPPER_FORMATTER(EndWhereStmt)  // R1049
WRAPPER_FORMATTER(EndForallStmt)  // R1054
WRAPPER_FORMATTER(EndAssociateStmt)  // R1106
WRAPPER_FORMATTER(BlockStmt)  // R1108
WRAPPER_FORMATTER(BlockSpecificationPart)  // R1109
WRAPPER_FORMATTER(EndBlockStmt)  // R1110
WRAPPER_FORMATTER(EndCriticalStmt)  // R1118
WRAPPER_FORMATTER(LocalitySpec::Local)  // R1130
WRAPPER_FORMATTER(LocalitySpec::LocalInit)  // R1130
WRAPPER_FORMATTER(LocalitySpec::Shared)  // R1130
WRAPPER_FORMATTER(EndDoStmt)  // R1132
WRAPPER_FORMATTER(CycleStmt)  // R1133
WRAPPER_FORMATTER(ElseStmt)  // R1137
WRAPPER_FORMATTER(EndIfStmt)  // R1138
WRAPPER_FORMATTER(EndSelectStmt)  // R1143, R1151, R1155
WRAPPER_FORMATTER(ExitStmt)  // R1156
WRAPPER_FORMATTER(GotoStmt)  // R1157
WRAPPER_FORMATTER(SyncAllStmt)  // R1164
WRAPPER_FORMATTER(SyncMemoryStmt)  // R1168
WRAPPER_FORMATTER(FileUnitNumber)  // R1202
WRAPPER_FORMATTER(OpenStmt)  // R1204
WRAPPER_FORMATTER(StatusExpr)  // R1205 & seq.
WRAPPER_FORMATTER(ErrLabel)  // R1205 & seq.
WRAPPER_FORMATTER(ConnectSpec::Recl)  // R1205
WRAPPER_FORMATTER(ConnectSpec::Newunit)  // R1205
WRAPPER_FORMATTER(CloseStmt)  // R1208
WRAPPER_FORMATTER(IoControlSpec::Asynchronous)  // R1213
WRAPPER_FORMATTER(EndLabel)  // R1213 & R1223
WRAPPER_FORMATTER(EorLabel)  // R1213 & R1223
WRAPPER_FORMATTER(IoControlSpec::Pos)  // R1213
WRAPPER_FORMATTER(IoControlSpec::Rec)  // R1213
WRAPPER_FORMATTER(IoControlSpec::Size)  // R1213
WRAPPER_FORMATTER(IdVariable)  // R1214
WRAPPER_FORMATTER(WaitStmt)  // R1222
WRAPPER_FORMATTER(IdExpr)  // R1223 & R1231
WRAPPER_FORMATTER(FormatStmt)  // R1301
WRAPPER_FORMATTER(ProgramStmt)  // R1402
WRAPPER_FORMATTER(EndProgramStmt)  // R1403
WRAPPER_FORMATTER(ModuleStmt)  // R1405
WRAPPER_FORMATTER(EndModuleStmt)  // R1406
WRAPPER_FORMATTER(EndSubmoduleStmt)  // R1419
WRAPPER_FORMATTER(BlockDataStmt)  // R1420
WRAPPER_FORMATTER(EndBlockDataStmt)  // R1421
WRAPPER_FORMATTER(EndInterfaceStmt)  // R1504
WRAPPER_FORMATTER(ExternalStmt)  // R1511
WRAPPER_FORMATTER(IntrinsicStmt)  // R1519
WRAPPER_FORMATTER(FunctionReference)  // R1520
WRAPPER_FORMATTER(CallStmt)  // R1521
WRAPPER_FORMATTER(ActualArg::PercentRef)  // R1524 extension
WRAPPER_FORMATTER(ActualArg::PercentVal)  // R1524 extension
WRAPPER_FORMATTER(AltReturnSpec)  // R1525
WRAPPER_FORMATTER(EndFunctionStmt)  // R1533
WRAPPER_FORMATTER(EndSubroutineStmt)  // R1537
WRAPPER_FORMATTER(MpSubprogramStmt)  // R1539
WRAPPER_FORMATTER(EndMpSubprogramStmt)  // R1540
WRAPPER_FORMATTER(ReturnStmt)  // R1542
WRAPPER_FORMATTER(PauseStmt)  // legacy

#undef WRAPPER_FORMATTER

#define EMPTY_TYPE_FORMATTER(TYPE) \
  std::ostream &operator<<(std::ostream &o, const TYPE &) { return o << #TYPE; }

EMPTY_TYPE_FORMATTER(ErrorRecovery)
EMPTY_TYPE_FORMATTER(Star)  // R701, R1215, R1536
EMPTY_TYPE_FORMATTER(TypeParamValue::Deferred)  // R701
EMPTY_TYPE_FORMATTER(DeclarationTypeSpec::ClassStar)  // R703
EMPTY_TYPE_FORMATTER(DeclarationTypeSpec::TypeStar)  // R703
EMPTY_TYPE_FORMATTER(IntrinsicTypeSpec::DoublePrecision)  // R704
EMPTY_TYPE_FORMATTER(IntrinsicTypeSpec::DoubleComplex)  // R704 extension
EMPTY_TYPE_FORMATTER(KindParam::Kanji)  // R724 extension
EMPTY_TYPE_FORMATTER(Abstract)  // R728
EMPTY_TYPE_FORMATTER(TypeAttrSpec::BindC)  // R728
EMPTY_TYPE_FORMATTER(Allocatable)  // R738 & R802
EMPTY_TYPE_FORMATTER(Contiguous)  // R738 & R802
EMPTY_TYPE_FORMATTER(SequenceStmt)  // R731
EMPTY_TYPE_FORMATTER(NoPass)  // R742 & R752
EMPTY_TYPE_FORMATTER(Pointer)  // R738, R742, R802, & R1514
EMPTY_TYPE_FORMATTER(PrivateStmt)  // R745, R747
EMPTY_TYPE_FORMATTER(BindAttr::Deferred)  // R752
EMPTY_TYPE_FORMATTER(BindAttr::Non_Overridable)  // R752
EMPTY_TYPE_FORMATTER(EnumDefStmt)  // R760
EMPTY_TYPE_FORMATTER(EndEnumStmt)  // R763
EMPTY_TYPE_FORMATTER(Asynchronous)  // R802
EMPTY_TYPE_FORMATTER(External)  // R802
EMPTY_TYPE_FORMATTER(Intrinsic)  // R802
EMPTY_TYPE_FORMATTER(Optional)  // R802 & R1514
EMPTY_TYPE_FORMATTER(Parameter)  // R802
EMPTY_TYPE_FORMATTER(Protected)  // R802 & R1514
EMPTY_TYPE_FORMATTER(Save)  // R802 & R1514
EMPTY_TYPE_FORMATTER(Target)  // R802
EMPTY_TYPE_FORMATTER(Value)  // R802
EMPTY_TYPE_FORMATTER(Volatile)  // R802
EMPTY_TYPE_FORMATTER(NullInit)  // R806
EMPTY_TYPE_FORMATTER(AssumedRankSpec)  // R825
EMPTY_TYPE_FORMATTER(LocalitySpec::DefaultNone)  // R1130
EMPTY_TYPE_FORMATTER(Default)  // R1145, R1150, R1154
EMPTY_TYPE_FORMATTER(ContinueStmt)  // R1159
EMPTY_TYPE_FORMATTER(FailImageStmt)  // R1163
EMPTY_TYPE_FORMATTER(GenericSpec::Assignment)  // R1508
EMPTY_TYPE_FORMATTER(GenericSpec::ReadFormatted)  // R1509
EMPTY_TYPE_FORMATTER(GenericSpec::ReadUnformatted)  // R1509
EMPTY_TYPE_FORMATTER(GenericSpec::WriteFormatted)  // R1509
EMPTY_TYPE_FORMATTER(GenericSpec::WriteUnformatted)  // R1509
EMPTY_TYPE_FORMATTER(PrefixSpec::Elemental)  // R1527
EMPTY_TYPE_FORMATTER(PrefixSpec::Impure)  // R1527
EMPTY_TYPE_FORMATTER(PrefixSpec::Module)  // R1527
EMPTY_TYPE_FORMATTER(PrefixSpec::Non_Recursive)  // R1527
EMPTY_TYPE_FORMATTER(PrefixSpec::Pure)  // R1527
EMPTY_TYPE_FORMATTER(PrefixSpec::Recursive)  // R1527
EMPTY_TYPE_FORMATTER(ContainsStmt)  // R1543
EMPTY_TYPE_FORMATTER(StructureDef::EndStructureStmt)
EMPTY_TYPE_FORMATTER(Union::UnionStmt)
EMPTY_TYPE_FORMATTER(Union::EndUnionStmt)
EMPTY_TYPE_FORMATTER(Map::MapStmt)
EMPTY_TYPE_FORMATTER(Map::EndMapStmt)

#undef EMPTY_TYPE_FORMATTER

// R703
std::ostream &operator<<(std::ostream &o, const DeclarationTypeSpec::Type &x) {
  return o << "(DeclarationTypeSpec TYPE " << x.derived << ')';
}

std::ostream &operator<<(std::ostream &o, const DeclarationTypeSpec::Class &x) {
  return o << "(DeclarationTypeSpec CLASS " << x.derived << ')';
}

// R704
std::ostream &operator<<(std::ostream &o, const IntrinsicTypeSpec::Real &x) {
  return o << "(Real " << x.kind << ')';
}

std::ostream &operator<<(std::ostream &o, const IntrinsicTypeSpec::Complex &x) {
  return o << "(Complex " << x.kind << ')';
}

std::ostream &operator<<(
    std::ostream &o, const IntrinsicTypeSpec::Character &x) {
  return o << "(Character " << x.selector << ')';
}

std::ostream &operator<<(std::ostream &o, const IntrinsicTypeSpec::Logical &x) {
  return o << "(Logical " << x.kind << ')';
}

// R706
// TODO: Abstract part of this away to utility functions &/or constructors
KindSelector::KindSelector(std::uint64_t &&k)
  : v{IntConstantExpr{ConstantExpr{Indirection<Expr>{
        Expr{LiteralConstant{IntLiteralConstant{std::move(k)}}}}}}} {}

// R712 sign
std::ostream &operator<<(std::ostream &o, Sign x) {
  switch (x) {
  case Sign::Positive: return o << "Positive";
  case Sign::Negative: return o << "Negative";
  default: CRASH_NO_CASE;
  }
  return o;
}

// R714 real-literal-constant
// R715 significand
static std::string charListToString(std::list<char> &&cs) {
  std::string result;
  for (auto ch : cs) {
    result += ch;
  }
  return result;
}

RealLiteralConstant::RealLiteralConstant(std::list<char> &&i,
    std::list<char> &&f, std::optional<ExponentPart> &&expo,
    std::optional<KindParam> &&k)
  : intPart{charListToString(std::move(i))}, fraction{charListToString(
                                                 std::move(f))},
    exponent(std::move(expo)), kind(std::move(k)) {}

RealLiteralConstant::RealLiteralConstant(std::list<char> &&f,
    std::optional<ExponentPart> &&expo, std::optional<KindParam> &&k)
  : fraction{charListToString(std::move(f))}, exponent(std::move(expo)),
    kind(std::move(k)) {}

RealLiteralConstant::RealLiteralConstant(
    std::list<char> &&i, ExponentPart &&expo, std::optional<KindParam> &&k)
  : intPart{charListToString(std::move(i))}, exponent(std::move(expo)),
    kind(std::move(k)) {}

std::ostream &operator<<(std::ostream &o, const RealLiteralConstant &x) {
  return o << "(RealLiteralConstant " << x.intPart << ' ' << x.fraction << ' '
           << x.exponent << ' ' << x.kind << ')';
}

// R721 char-selector
std::ostream &operator<<(
    std::ostream &o, const CharSelector::LengthAndKind &x) {
  return o << "(LengthAndKind " << x.length << ' ' << x.kind << ')';
}

// R749 type-bound-procedure-stmt
std::ostream &operator<<(
    std::ostream &o, const TypeBoundProcedureStmt::WithoutInterface &x) {
  return o << "(TypeBoundProcedureStmt () " << x.attributes << ' '
           << x.declarations << ')';
}

std::ostream &operator<<(
    std::ostream &o, const TypeBoundProcedureStmt::WithInterface &x) {
  return o << "(TypeBoundProcedureStmt " << x.interfaceName << ' '
           << x.attributes << ' ' << x.bindingNames << ')';
}

// R770 ac-spec
std::ostream &operator<<(std::ostream &o, const AcSpec &x) {
  return o << "(AcSpec " << x.type << ' ' << x.values << ')';
}

// R863 implicit-stmt
std::ostream &operator<<(std::ostream &o, const ImplicitStmt &x) {
  o << "(ImplicitStmt ";
  if (std::holds_alternative<std::list<ImplicitStmt::ImplicitNoneNameSpec>>(
          x.u)) {
    o << "NONE ";
  }
  std::visit([&o](const auto &y) { o << y; }, x.u);
  return o << ')';
}

// R867
ImportStmt::ImportStmt(Kind &&k, std::list<Name> &&n)
  : kind{k}, names(std::move(n)) {
  CHECK(kind == Kind::Default || kind == Kind::Only || names.empty());
}

std::ostream &operator<<(std::ostream &o, const ImportStmt &x) {
  o << "(ImportStmt ";
  if (x.kind != ImportStmt::Kind::Default) {
    o << x.kind;
  }
  return o << x.names << ')';
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
            if (Name *n = std::get_if<Name>(&dr.u)) {
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
            return {Name{""}};
          }},
      u);
}

std::optional<Call> Designator::ConvertToCall() {
  return std::visit(
      visitors{[](ObjectName &n) -> std::optional<Call> {
                 return {Call{ProcedureDesignator{std::move(n)},
                     std::list<ActualArgSpec>{}}};
               },
          [this](DataReference &dr) -> std::optional<Call> {
            if (std::holds_alternative<Indirection<CoindexedNamedObject>>(
                    dr.u)) {
              return {};
            }
            if (Name *n = std::get_if<Name>(&dr.u)) {
              return {Call{ProcedureDesignator{std::move(*n)},
                  std::list<ActualArgSpec>{}}};
            }
            if (auto *isc =
                    std::get_if<Indirection<StructureComponent>>(&dr.u)) {
              StructureComponent &sc{**isc};
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

// R913 structure-component -> data-ref
std::ostream &operator<<(std::ostream &o, const StructureComponent &x) {
  return o << "(StructureComponent " << x.base << ' ' << x.component << ')';
}

// R914 coindexed-named-object -> data-ref
std::ostream &operator<<(std::ostream &o, const CoindexedNamedObject &x) {
  return o << "(CoindexedNamedObject " << x.base << ' ' << x.imageSelector
           << ')';
}

// R912 part-ref
std::ostream &operator<<(std::ostream &o, const PartRef &pr) {
  return o << "(PartRef " << pr.name << ' ' << pr.subscripts << ' '
           << pr.imageSelector << ')';
}

// R917 array-element -> data-ref
std::ostream &operator<<(std::ostream &o, const ArrayElement &x) {
  return o << "(ArrayElement " << x.base << ' ' << x.subscripts << ')';
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
                          return {Name{"bad"}};
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

// R1146
std::ostream &operator<<(std::ostream &o, const CaseValueRange::Range &x) {
  return o << "(Range " << x.lower << ' ' << x.upper << ')';
}

// R1307 data-edit-desc (part 1 of 2)
std::ostream &operator<<(std::ostream &o, const IntrinsicTypeDataEditDesc &x) {
  o << "(IntrinsicTypeDataEditDesc ";
  switch (x.kind) {
  case IntrinsicTypeDataEditDesc::Kind::I: o << "I "; break;
  case IntrinsicTypeDataEditDesc::Kind::B: o << "B "; break;
  case IntrinsicTypeDataEditDesc::Kind::O: o << "O "; break;
  case IntrinsicTypeDataEditDesc::Kind::Z: o << "Z "; break;
  case IntrinsicTypeDataEditDesc::Kind::F: o << "F "; break;
  case IntrinsicTypeDataEditDesc::Kind::E: o << "E "; break;
  case IntrinsicTypeDataEditDesc::Kind::EN: o << "EN "; break;
  case IntrinsicTypeDataEditDesc::Kind::ES: o << "ES "; break;
  case IntrinsicTypeDataEditDesc::Kind::EX: o << "EX "; break;
  case IntrinsicTypeDataEditDesc::Kind::G: o << "G "; break;
  case IntrinsicTypeDataEditDesc::Kind::L: o << "L "; break;
  case IntrinsicTypeDataEditDesc::Kind::A: o << "A "; break;
  case IntrinsicTypeDataEditDesc::Kind::D: o << "D "; break;
  default: CRASH_NO_CASE;
  }
  return o << x.width << ' ' << x.digits << ' ' << x.exponentWidth << ')';
}

// R1210 read-stmt
std::ostream &operator<<(std::ostream &o, const ReadStmt &x) {
  return o << "(ReadStmt " << x.iounit << ' ' << x.format << ' ' << x.controls
           << ' ' << x.items << ')';
}

// R1211 write-stmt
std::ostream &operator<<(std::ostream &o, const WriteStmt &x) {
  return o << "(WriteStmt " << x.iounit << ' ' << x.format << ' ' << x.controls
           << ' ' << x.items << ')';
}

// R1307 data-edit-desc (part 2 of 2)
std::ostream &operator<<(std::ostream &o, const DerivedTypeDataEditDesc &x) {
  return o << "(DerivedTypeDataEditDesc " << x.type << ' ' << x.parameters
           << ')';
}

// R1313 control-edit-desc
std::ostream &operator<<(std::ostream &o, const ControlEditDesc &x) {
  o << "(ControlEditDesc ";
  switch (x.kind) {
  case ControlEditDesc::Kind::T: o << "T "; break;
  case ControlEditDesc::Kind::TL: o << "TL "; break;
  case ControlEditDesc::Kind::TR: o << "TR "; break;
  case ControlEditDesc::Kind::X: o << "X "; break;
  case ControlEditDesc::Kind::Slash: o << "/ "; break;
  case ControlEditDesc::Kind::Colon: o << ": "; break;
  case ControlEditDesc::Kind::SS: o << "SS "; break;
  case ControlEditDesc::Kind::SP: o << "SP "; break;
  case ControlEditDesc::Kind::S: o << "S "; break;
  case ControlEditDesc::Kind::P: o << "P "; break;
  case ControlEditDesc::Kind::BN: o << "BN "; break;
  case ControlEditDesc::Kind::BZ: o << "BZ "; break;
  case ControlEditDesc::Kind::RU: o << "RU "; break;
  case ControlEditDesc::Kind::RD: o << "RD "; break;
  case ControlEditDesc::Kind::RN: o << "RN "; break;
  case ControlEditDesc::Kind::RC: o << "RC "; break;
  case ControlEditDesc::Kind::RP: o << "RP "; break;
  case ControlEditDesc::Kind::DC: o << "DC "; break;
  case ControlEditDesc::Kind::DP: o << "DP "; break;
  default: CRASH_NO_CASE;
  }
  return o << x.count << ')';
}

// R1304 format-item
std::ostream &operator<<(std::ostream &o, const FormatItem &x) {
  o << "(FormatItem " << x.repeatCount;
  std::visit([&o](const auto &y) { o << y; }, x.u);
  return o << ')';
}

// R1409
std::ostream &operator<<(std::ostream &o, const UseStmt &x) {
  o << "(UseStmt " << x.nature << ' ' << x.moduleName << ' ';
  std::visit(
      visitors{
          [&o](const std::list<Rename> &y) -> void { o << "RENAME " << y; },
          [&o](const std::list<Only> &y) -> void { o << "ONLY " << y; },
      },
      x.u);
  return o << ')';
}

// R1506
std::ostream &operator<<(std::ostream &o, const ProcedureStmt &x) {
  return o << "(ProcedureStmt " << std::get<0>(x.t) << ' ' << std::get<1>(x.t)
           << ')';
}

// R1532 suffix
std::ostream &operator<<(std::ostream &o, const Suffix &x) {
  return o << "(Suffix " << x.binding << ' ' << x.resultName << ')';
}
}  // namespace parser
}  // namespace Fortran
