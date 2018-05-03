// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef FORTRAN_SEMANTICS_DUMP_PARSE_TREE_H_
#define FORTRAN_SEMANTICS_DUMP_PARSE_TREE_H_

#include "symbol.h"
#include "../parser/format-specification.h"
#include "../parser/idioms.h"
#include "../parser/indirection.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <ostream>
#include <string>

namespace Fortran::semantics {

//
// Dump the Parse Tree hierarchy of any node 'x' of the parse tree.
//

class ParseTreeDumper {
public:
  explicit ParseTreeDumper(std::ostream &out) : out_(out) {}

  constexpr const char *GetNodeName(const char *const &) { return "char *"; }
#define NODE_NAME(T, N) \
  constexpr const char *GetNodeName(const T &) { return N; }
#define NODE(NS, T) NODE_NAME(NS::T, #T)
#define NESTED_NODE(T1, T2) NODE_NAME(T1::T2, #T2)
  NODE_NAME(bool, "bool")
  NODE_NAME(int, "int")
  NODE(format, ControlEditDesc)
  NESTED_NODE(format::ControlEditDesc, Kind)
  NODE(format, DerivedTypeDataEditDesc)
  NODE(format, FormatItem)
  NODE(format, FormatSpecification)
  NODE(format, IntrinsicTypeDataEditDesc)
  NESTED_NODE(format::IntrinsicTypeDataEditDesc, Kind)
  NODE(parser, Abstract)
  NODE(parser, AcImpliedDo)
  NODE(parser, AcImpliedDoControl)
  NODE(parser, AcValue)
  NODE(parser, AccessStmt)
  NODE(parser, AccessId)
  NODE(parser, AccessSpec)
  NESTED_NODE(parser::AccessSpec, Kind)
  NODE(parser, AcSpec)
  NODE(parser, ActionStmt)
  NODE(parser, ActualArg)
  NESTED_NODE(parser::ActualArg, PercentRef)
  NESTED_NODE(parser::ActualArg, PercentVal)
  NODE(parser, ActualArgSpec)
  NESTED_NODE(parser::AcValue, Triplet)
  NODE(parser, AllocOpt)
  NESTED_NODE(parser::AllocOpt, Mold)
  NESTED_NODE(parser::AllocOpt, Source)
  NODE(parser, Allocatable)
  NODE(parser, AllocatableStmt)
  NODE(parser, AllocateCoarraySpec)
  NODE(parser, AllocateObject)
  NODE(parser, AllocateShapeSpec)
  NODE(parser, AllocateStmt)
  NODE(parser, Allocation)
  NODE(parser, AltReturnSpec)
  NODE(parser, ArithmeticIfStmt)
  NODE(parser, ArrayConstructor)
  NODE(parser, ArrayElement)
  NODE(parser, ArraySpec)
  NODE(parser, AssignStmt)
  NODE(parser, AssignedGotoStmt)
  NODE(parser, AssignmentStmt)
  NODE(parser, AssociateConstruct)
  NODE(parser, AssociateStmt)
  NODE(parser, Association)
  NODE(parser, AssumedImpliedSpec)
  NODE(parser, AssumedRankSpec)
  NODE(parser, AssumedShapeSpec)
  NODE(parser, AssumedSizeSpec)
  NODE(parser, Asynchronous)
  NODE(parser, AsynchronousStmt)
  NODE(parser, AttrSpec)
  NODE(parser, BOZLiteralConstant)
  NODE(parser, BackspaceStmt)
  NODE(parser, BasedPointerStmt)
  NODE(parser, BindAttr)
  NESTED_NODE(parser::BindAttr, Deferred)
  NESTED_NODE(parser::BindAttr, Non_Overridable)
  NODE(parser, BindEntity)
  NESTED_NODE(parser::BindEntity, Kind)
  NODE(parser, BindStmt)
  NODE(parser, Block)
  NODE(parser, BlockConstruct)
  NODE(parser, BlockData)
  NODE(parser, BlockDataStmt)
  NODE(parser, BlockSpecificationPart)
  NODE(parser, BlockStmt)
  NODE(parser, BoundsRemapping)
  NODE(parser, BoundsSpec)
  NODE(parser, Call)
  NODE(parser, CallStmt)
  NODE(parser, CaseConstruct)
  NESTED_NODE(parser::CaseConstruct, Case)
  NODE(parser, CaseSelector)
  NODE(parser, CaseStmt)
  NODE(parser, CaseValueRange)
  NESTED_NODE(parser::CaseValueRange, Range)
  NODE(parser, ChangeTeamConstruct)
  NODE(parser, ChangeTeamStmt)
  NODE(parser, CharLength)
  NODE(parser, CharLiteralConstant)
  NODE(parser, CharLiteralConstantSubstring)
  NODE(parser, CharSelector)
  NESTED_NODE(parser::CharSelector, LengthAndKind)
  NODE(parser, CharVariable)
  NODE(parser, CloseStmt)
  NESTED_NODE(parser::CloseStmt, CloseSpec)
  NODE(parser, CoarrayAssociation)
  NODE(parser, CoarraySpec)
  NODE(parser, CodimensionDecl)
  NODE(parser, CodimensionStmt)
  NODE(parser, CoindexedNamedObject)
  NODE(parser, CommonBlockObject)
  NODE(parser, CommonStmt)
  NESTED_NODE(parser::CommonStmt, Block)
  NODE(parser, CompilerDirective)
  NESTED_NODE(parser::CompilerDirective, IVDEP)
  NESTED_NODE(parser::CompilerDirective, IgnoreTKR)
  NODE(parser, ComplexLiteralConstant)
  NODE(parser, ComplexPart)
  NODE(parser, ComponentArraySpec)
  NODE(parser, ComponentAttrSpec)
  NODE(parser, ComponentDataSource)
  NODE(parser, ComponentDecl)
  NODE(parser, ComponentDefStmt)
  NODE(parser, ComponentSpec)
  NODE(parser, ComputedGotoStmt)
  NODE(parser, ConcurrentControl)
  NODE(parser, ConcurrentHeader)
  NODE(parser, ConnectSpec)
  NESTED_NODE(parser::ConnectSpec, CharExpr)
  NESTED_NODE(parser::ConnectSpec::CharExpr, Kind)
  NESTED_NODE(parser::ConnectSpec, Newunit)
  NESTED_NODE(parser::ConnectSpec, Recl)
  NODE(parser, ConstantValue)
  NODE(parser, ContainsStmt)
  NODE(parser, Contiguous)
  NODE(parser, ContiguousStmt)
  NODE(parser, ContinueStmt)
  NODE(parser, CriticalConstruct)
  NODE(parser, CriticalStmt)
  NODE(parser, CycleStmt)
  NODE(parser, DataComponentDefStmt)
  NODE(parser, DataIDoObject)
  NODE(parser, DataImpliedDo)
  NODE(parser, DataRef)
  NODE(parser, DataStmt)
  NODE(parser, DataStmtConstant)
  NODE(parser, DataStmtObject)
  NODE(parser, DataStmtRepeat)
  NODE(parser, DataStmtSet)
  NODE(parser, DataStmtValue)
  NODE(parser, DeallocateStmt)
  NODE(parser, DeclarationConstruct)
  NODE(parser, DeclarationTypeSpec)
  NESTED_NODE(parser::DeclarationTypeSpec, Class)
  NESTED_NODE(parser::DeclarationTypeSpec, ClassStar)
  NESTED_NODE(parser::DeclarationTypeSpec, Record)
  NESTED_NODE(parser::DeclarationTypeSpec, Type)
  NESTED_NODE(parser::DeclarationTypeSpec, TypeStar)
  NODE(parser, Default)
  NODE(parser, DeferredCoshapeSpecList)
  NODE(parser, DeferredShapeSpecList)
  NODE(parser, DefinedOpName)
  NODE(parser, DefinedOperator)
  NESTED_NODE(parser::DefinedOperator, IntrinsicOperator)
  NODE(parser, DerivedTypeDef)
  NODE(parser, DerivedTypeSpec)
  NODE(parser, DerivedTypeStmt)
  NODE(parser, Designator)
  NODE(parser, DimensionStmt)
  NESTED_NODE(parser::DimensionStmt, Declaration)
  NODE(parser, DoConstruct)
  NODE(parser, DummyArg)
  NODE(parser, ElseIfStmt)
  NODE(parser, ElseStmt)
  NODE(parser, ElsewhereStmt)
  NODE(parser, EndAssociateStmt)
  NODE(parser, EndBlockDataStmt)
  NODE(parser, EndBlockStmt)
  NODE(parser, EndChangeTeamStmt)
  NODE(parser, EndCriticalStmt)
  NODE(parser, EndDoStmt)
  NODE(parser, EndEnumStmt)
  NODE(parser, EndForallStmt)
  NODE(parser, EndFunctionStmt)
  NODE(parser, EndIfStmt)
  NODE(parser, EndInterfaceStmt)
  NODE(parser, EndLabel)
  NODE(parser, EndModuleStmt)
  NODE(parser, EndMpSubprogramStmt)
  NODE(parser, EndProgramStmt)
  NODE(parser, EndSelectStmt)
  NODE(parser, EndSubmoduleStmt)
  NODE(parser, EndSubroutineStmt)
  NODE(parser, EndTypeStmt)
  NODE(parser, EndWhereStmt)
  NODE(parser, EndfileStmt)
  NODE(parser, EntityDecl)
  NODE(parser, EntryStmt)
  NODE(parser, EnumDef)
  NODE(parser, EnumDefStmt)
  NODE(parser, Enumerator)
  NODE(parser, EnumeratorDefStmt)
  NODE(parser, EorLabel)
  NODE(parser, EquivalenceObject)
  NODE(parser, EquivalenceStmt)
  NODE(parser, ErrLabel)
  NODE(parser, ErrorRecovery)
  NODE(parser, EventPostStmt)
  NODE(parser, EventWaitStmt)
  NESTED_NODE(parser::EventWaitStmt, EventWaitSpec)
  NODE(parser, ExecutableConstruct)
  NODE(parser, ExecutionPart)
  NODE(parser, ExecutionPartConstruct)
  NODE(parser, ExitStmt)
  NODE(parser, ExplicitCoshapeSpec)
  NODE(parser, ExplicitShapeSpec)
  NODE(parser, Expr)
  NESTED_NODE(parser::Expr, Parentheses)
  NESTED_NODE(parser::Expr, UnaryPlus)
  NESTED_NODE(parser::Expr, Negate)
  NESTED_NODE(parser::Expr, NOT)
  NESTED_NODE(parser::Expr, PercentLoc)
  NESTED_NODE(parser::Expr, DefinedUnary)
  NESTED_NODE(parser::Expr, Power)
  NESTED_NODE(parser::Expr, Multiply)
  NESTED_NODE(parser::Expr, Divide)
  NESTED_NODE(parser::Expr, Add)
  NESTED_NODE(parser::Expr, Subtract)
  NESTED_NODE(parser::Expr, Concat)
  NESTED_NODE(parser::Expr, LT)
  NESTED_NODE(parser::Expr, LE)
  NESTED_NODE(parser::Expr, EQ)
  NESTED_NODE(parser::Expr, NE)
  NESTED_NODE(parser::Expr, GE)
  NESTED_NODE(parser::Expr, GT)
  NESTED_NODE(parser::Expr, AND)
  NESTED_NODE(parser::Expr, OR)
  NESTED_NODE(parser::Expr, EQV)
  NESTED_NODE(parser::Expr, NEQV)
  NESTED_NODE(parser::Expr, XOR)
  NESTED_NODE(parser::Expr, DefinedBinary)
  NESTED_NODE(parser::Expr, ComplexConstructor)
  NODE(parser, External)
  NODE(parser, ExternalStmt)
  NODE(parser, FailImageStmt)
  NODE(parser, FileUnitNumber)
  NODE(parser, FinalProcedureStmt)
  NODE(parser, FlushStmt)
  NODE(parser, ForallAssignmentStmt)
  NODE(parser, ForallBodyConstruct)
  NODE(parser, ForallConstruct)
  NODE(parser, ForallConstructStmt)
  NODE(parser, ForallStmt)
  NODE(parser, FormTeamStmt)
  NESTED_NODE(parser::FormTeamStmt, FormTeamSpec)
  NODE(parser, Format)
  NODE(parser, FormatStmt)
  NODE(parser, FunctionReference)
  NODE(parser, FunctionStmt)
  NODE(parser, FunctionSubprogram)
  NODE(parser, GenericSpec)
  NESTED_NODE(parser::GenericSpec, Assignment)
  NESTED_NODE(parser::GenericSpec, ReadFormatted)
  NESTED_NODE(parser::GenericSpec, ReadUnformatted)
  NESTED_NODE(parser::GenericSpec, WriteFormatted)
  NESTED_NODE(parser::GenericSpec, WriteUnformatted)
  NODE(parser, GenericStmt)
  NODE(parser, GotoStmt)
  NODE(parser, HollerithLiteralConstant)
  NODE(parser, IdExpr)
  NODE(parser, IdVariable)
  NODE(parser, IfConstruct)
  NESTED_NODE(parser::IfConstruct, ElseBlock)
  NESTED_NODE(parser::IfConstruct, ElseIfBlock)
  NODE(parser, IfStmt)
  NODE(parser, IfThenStmt)
  NODE(parser, ImageSelector)
  NODE(parser, ImageSelectorSpec)
  NESTED_NODE(parser::ImageSelectorSpec, Stat)
  NESTED_NODE(parser::ImageSelectorSpec, Team)
  NESTED_NODE(parser::ImageSelectorSpec, Team_Number)
  NODE(parser, ImplicitPart)
  NODE(parser, ImplicitPartStmt)
  NODE(parser, ImplicitSpec)
  NODE(parser, ImplicitStmt)
  NESTED_NODE(parser::ImplicitStmt, ImplicitNoneNameSpec)
  NODE(parser, ImpliedShapeSpec)
  NODE(parser, ImportStmt)
  NODE(parser, Initialization)
  NODE(parser, InputImpliedDo)
  NODE(parser, InputItem)
  NODE(parser, InquireSpec)
  NESTED_NODE(parser::InquireSpec, CharVar)
  NESTED_NODE(parser::InquireSpec::CharVar, Kind)
  NESTED_NODE(parser::InquireSpec, IntVar)
  NESTED_NODE(parser::InquireSpec::IntVar, Kind)
  NESTED_NODE(parser::InquireSpec, LogVar)
  NESTED_NODE(parser::InquireSpec::LogVar, Kind)
  NODE(parser, InquireStmt)
  NESTED_NODE(parser::InquireStmt, Iolength)
  NODE(parser, IntegerTypeSpec)
  NODE(parser, IntentSpec)
  NESTED_NODE(parser::IntentSpec, Intent)
  NODE(parser, IntentStmt)
  NODE(parser, InterfaceBlock)
  NODE(parser, InterfaceBody)
  NESTED_NODE(parser::InterfaceBody, Function)
  NESTED_NODE(parser::InterfaceBody, Subroutine)
  NODE(parser, InterfaceSpecification)
  NODE(parser, InterfaceStmt)
  NODE(parser, InternalSubprogram)
  NODE(parser, InternalSubprogramPart)
  NODE(parser, IntLiteralConstant)
  NODE(parser, Intrinsic)
  NODE(parser, IntrinsicStmt)
  NODE(parser, IntrinsicTypeSpec)
  NESTED_NODE(parser::IntrinsicTypeSpec, Character)
  NESTED_NODE(parser::IntrinsicTypeSpec, Complex)
  NESTED_NODE(parser::IntrinsicTypeSpec, DoubleComplex)
  NESTED_NODE(parser::IntrinsicTypeSpec, DoublePrecision)
  NESTED_NODE(parser::IntrinsicTypeSpec, Logical)
  NESTED_NODE(parser::IntrinsicTypeSpec, NCharacter)
  NESTED_NODE(parser::IntrinsicTypeSpec, Real)
  NODE(parser, IoControlSpec)
  NESTED_NODE(parser::IoControlSpec, Asynchronous)
  NESTED_NODE(parser::IoControlSpec, CharExpr)
  NESTED_NODE(parser::IoControlSpec::CharExpr, Kind)
  NESTED_NODE(parser::IoControlSpec, Pos)
  NESTED_NODE(parser::IoControlSpec, Rec)
  NESTED_NODE(parser::IoControlSpec, Size)
  NODE(parser, IoUnit)
  NODE(parser, Keyword)
  NODE(parser, KindParam)
  NESTED_NODE(parser::KindParam, Kanji)
  NODE(parser, KindSelector)
  NESTED_NODE(parser::KindSelector, StarSize)
  NODE(parser, LabelDoStmt)
  NODE(parser, LanguageBindingSpec)
  NODE(parser, LengthSelector)
  NODE(parser, LetterSpec)
  NODE(parser, LiteralConstant)
  NODE(parser, LocalitySpec)
  NESTED_NODE(parser::LocalitySpec, DefaultNone)
  NESTED_NODE(parser::LocalitySpec, Local)
  NESTED_NODE(parser::LocalitySpec, LocalInit)
  NESTED_NODE(parser::LocalitySpec, Shared)
  NODE(parser, LockStmt)
  NESTED_NODE(parser::LockStmt, LockStat)
  NODE(parser, LogicalLiteralConstant)
  NODE_NAME(parser::LoopBounds<parser::ScalarIntConstantExpr>, "LoopBounds")
  NODE_NAME(parser::LoopBounds<parser::ScalarIntExpr>, "LoopBounds")
  NODE(parser, LoopControl)
  NESTED_NODE(parser::LoopControl, Concurrent)
  NODE(parser, MainProgram)
  NODE(parser, Map)
  NESTED_NODE(parser::Map, EndMapStmt)
  NESTED_NODE(parser::Map, MapStmt)
  NODE(parser, MaskedElsewhereStmt)
  NODE(parser, Module)
  NODE(parser, ModuleStmt)
  NODE(parser, ModuleSubprogram)
  NODE(parser, ModuleSubprogramPart)
  NODE(parser, MpSubprogramStmt)
  NODE(parser, MsgVariable)
  NODE(parser, NamedConstant)
  NODE(parser, NamedConstantDef)
  NODE(parser, NamelistStmt)
  NESTED_NODE(parser::NamelistStmt, Group)
  NODE(parser, NonLabelDoStmt)
  NODE(parser, NoPass)
  NODE(parser, NullifyStmt)
  NODE(parser, NullInit)
  NODE(parser, ObjectDecl)
  NODE(parser, OldParameterStmt)
  NODE(parser, Only)
  NODE(parser, OpenStmt)
  NODE(parser, Optional)
  NODE(parser, OptionalStmt)
  NODE(parser, OtherSpecificationStmt)
  NODE(parser, OutputImpliedDo)
  NODE(parser, OutputItem)
  NODE(parser, Parameter)
  NODE(parser, ParameterStmt)
  NODE(parser, ParentIdentifier)
  NODE(parser, Pass)
  NODE(parser, PauseStmt)
  NODE(parser, Pointer)
  NODE(parser, PointerAssignmentStmt)
  NESTED_NODE(parser::PointerAssignmentStmt, Bounds)
  NODE(parser, PointerDecl)
  NODE(parser, PointerObject)
  NODE(parser, PointerStmt)
  NODE(parser, PositionOrFlushSpec)
  NODE(parser, PrefixSpec)
  NESTED_NODE(parser::PrefixSpec, Elemental)
  NESTED_NODE(parser::PrefixSpec, Impure)
  NESTED_NODE(parser::PrefixSpec, Module)
  NESTED_NODE(parser::PrefixSpec, Non_Recursive)
  NESTED_NODE(parser::PrefixSpec, Pure)
  NESTED_NODE(parser::PrefixSpec, Recursive)
  NODE(parser, PrintStmt)
  NODE(parser, PrivateStmt)
  NODE(parser, PrivateOrSequence)
  NODE(parser, ProcAttrSpec)
  NODE(parser, ProcComponentAttrSpec)
  NODE(parser, ProcComponentDefStmt)
  NODE(parser, ProcComponentRef)
  NODE(parser, ProcDecl)
  NODE(parser, ProcInterface)
  NODE(parser, ProcPointerInit)
  NODE(parser, ProcedureDeclarationStmt)
  NODE(parser, ProcedureDesignator)
  NODE(parser, ProcedureStmt)
  NESTED_NODE(parser::ProcedureStmt, Kind)
  NODE(parser, Program)
  NODE(parser, ProgramStmt)
  NODE(parser, ProgramUnit)
  NODE(parser, Protected)
  NODE(parser, ProtectedStmt)
  NODE(parser, ReadStmt)
  NODE(parser, RealLiteralConstant)
  NESTED_NODE(parser::RealLiteralConstant, Real)
  NODE(parser, Rename)
  NESTED_NODE(parser::Rename, Names)
  NESTED_NODE(parser::Rename, Operators)
  NODE(parser, ReturnStmt)
  NODE(parser, RewindStmt)
  NODE(parser, Save)
  NODE(parser, SaveStmt)
  NODE(parser, SavedEntity)
  NESTED_NODE(parser::SavedEntity, Kind)
  NODE(parser, SectionSubscript)
  NODE(parser, SelectCaseStmt)
  NODE(parser, SelectRankCaseStmt)
  NESTED_NODE(parser::SelectRankCaseStmt, Rank)
  NODE(parser, SelectRankConstruct)
  NESTED_NODE(parser::SelectRankConstruct, RankCase)
  NODE(parser, SelectRankStmt)
  NODE(parser, SelectTypeConstruct)
  NESTED_NODE(parser::SelectTypeConstruct, TypeCase)
  NODE(parser, SelectTypeStmt)
  NODE(parser, Selector)
  NODE(parser, SeparateModuleSubprogram)
  NODE(parser, SequenceStmt)
  NODE(parser, Sign)
  NODE(parser, SignedComplexLiteralConstant)
  NODE(parser, SignedIntLiteralConstant)
  NODE(parser, SignedRealLiteralConstant)
  NODE(parser, SpecificationConstruct)
  NODE(parser, SpecificationExpr)
  NODE(parser, SpecificationPart)
  NODE(parser, Star)
  NODE(parser, StatOrErrmsg)
  NODE(parser, StatVariable)
  NODE(parser, StatusExpr)
  NODE(parser, StmtFunctionStmt)
  NODE(parser, StopCode)
  NODE(parser, StopStmt)
  NESTED_NODE(parser::StopStmt, Kind)
  NODE(parser, StructureComponent)
  NODE(parser, StructureConstructor)
  NODE(parser, StructureDef)
  NESTED_NODE(parser::StructureDef, EndStructureStmt)
  NODE(parser, StructureField)
  NODE(parser, StructureStmt)
  NODE(parser, Submodule)
  NODE(parser, SubmoduleStmt)
  NODE(parser, SubroutineStmt)
  NODE(parser, SubroutineSubprogram)
  NODE(parser, SubscriptTriplet)
  NODE(parser, Substring)
  NODE(parser, SubstringRange)
  NODE(parser, Suffix)
  NODE(parser, SyncAllStmt)
  NODE(parser, SyncImagesStmt)
  NESTED_NODE(parser::SyncImagesStmt, ImageSet)
  NODE(parser, SyncMemoryStmt)
  NODE(parser, SyncTeamStmt)
  NODE(parser, Target)
  NODE(parser, TargetStmt)
  NODE(parser, TypeAttrSpec)
  NESTED_NODE(parser::TypeAttrSpec, BindC)
  NESTED_NODE(parser::TypeAttrSpec, Extends)
  NODE(parser, TypeBoundGenericStmt)
  NODE(parser, TypeBoundProcBinding)
  NODE(parser, TypeBoundProcDecl)
  NODE(parser, TypeBoundProcedurePart)
  NODE(parser, TypeBoundProcedureStmt)
  NESTED_NODE(parser::TypeBoundProcedureStmt, WithInterface)
  NESTED_NODE(parser::TypeBoundProcedureStmt, WithoutInterface)
  NODE(parser, TypeDeclarationStmt)
  NODE(parser, TypeGuardStmt)
  NESTED_NODE(parser::TypeGuardStmt, Guard)
  NODE(parser, TypeParamDecl)
  NODE(parser, TypeParamDefStmt)
  NESTED_NODE(parser::TypeParamDefStmt, KindOrLen)
  NODE(parser, TypeParamInquiry)
  NODE(parser, TypeParamSpec)
  NODE(parser, TypeParamValue)
  NESTED_NODE(parser::TypeParamValue, Deferred)
  NODE(parser, TypeSpec)
  NODE(parser, Union)
  NESTED_NODE(parser::Union, EndUnionStmt)
  NESTED_NODE(parser::Union, UnionStmt)
  NODE(parser, UnlockStmt)
  NODE(parser, UseStmt)
  NESTED_NODE(parser::UseStmt, ModuleNature)
  NODE(parser, Value)
  NODE(parser, ValueStmt)
  NODE(parser, Variable)
  NODE(parser, Volatile)
  NODE(parser, VolatileStmt)
  NODE(parser, WaitSpec)
  NODE(parser, WaitStmt)
  NODE(parser, WhereBodyConstruct)
  NODE(parser, WhereConstruct)
  NESTED_NODE(parser::WhereConstruct, Elsewhere)
  NESTED_NODE(parser::WhereConstruct, MaskedElsewhere)
  NODE(parser, WhereConstructStmt)
  NODE(parser, WhereStmt)
  NODE(parser, WriteStmt)
#undef NESTED_NODE
#undef NODE
#undef NODE_NAME


  template<typename T> bool Pre(const T &x) {
    IndentEmptyLine();
    if (UnionTrait<T> || WrapperTrait<T>) {
      out_ << GetNodeName(x) << " -> ";
      emptyline_ = false;
    } else {
      out_ << GetNodeName(x);
      EndLine();
      ++indent_;
    }
    return true;
  }

  template<typename T> void Post(const T &x) {
    if (UnionTrait<T> || WrapperTrait<T>) {
      if (!emptyline_) {
        EndLine();
      }
    } else {
      --indent_;
    }
  }

  bool PutName(const std::string &name, const semantics::Symbol *symbol) {
    IndentEmptyLine();
    if (symbol != nullptr) {
      out_ << "symbol = " << *symbol;
    } else {
      out_ << "Name = '" << name << '\'';
    }
    ++indent_;
    EndLine();
    return true;
  }

  bool Pre(const parser::Name &x) { return PutName(x.ToString(), x.symbol); }

  void Post(const parser::Name &) { --indent_; }

  bool Pre(const std::string &x) { return PutName(x, nullptr); }

  void Post(const std::string &x) { --indent_; }

  bool Pre(const std::int64_t &x) {
    IndentEmptyLine();
    out_ << "int = '" << x << '\'';
    ++indent_;
    EndLine();
    return true;
  }

  void Post(const std::int64_t &x) { --indent_; }

  bool Pre(const std::uint64_t &x) {
    IndentEmptyLine();
    out_ << "int = '" << x << '\'';
    ++indent_;
    EndLine();
    return true;
  }

  void Post(const std::uint64_t &x) { --indent_; }

  // A few types we want to ignore

  template<typename T> bool Pre(const parser::Statement<T> &) { return true; }

  template<typename T> void Post(const parser::Statement<T> &) {}

  template<typename T> bool Pre(const parser::Indirection<T> &) { return true; }

  template<typename T> void Post(const parser::Indirection<T> &) {}

  template<typename T> bool Pre(const parser::Integer<T> &) { return true; }

  template<typename T> void Post(const parser::Integer<T> &) {}

  template<typename T> bool Pre(const parser::Scalar<T> &) { return true; }

  template<typename T> void Post(const parser::Scalar<T> &) {}

  template<typename... A> bool Pre(const std::tuple<A...> &) { return true; }

  template<typename... A> void Post(const std::tuple<A...> &) {}

  template<typename... A> bool Pre(const std::variant<A...> &) { return true; }

  template<typename... A> void Post(const std::variant<A...> &) {}

protected:
  void IndentEmptyLine() {
    if (emptyline_ && indent_ > 0) {
      for (int i{0}; i < indent_; ++i) {
        out_ << "| ";
      }
      emptyline_ = false;
    }
  }

  void EndLine() {
    out_ << '\n';
    emptyline_ = true;
  }

private:
  int indent_{0};
  std::ostream &out_;
  bool emptyline_{false};
};

template<typename T> void DumpTree(std::ostream &out, const T &x) {
  ParseTreeDumper dumper{out};
  parser::Walk(x, dumper);
}
}  // namespace Fortran::semantics
#endif  // FORTRAN_SEMANTICS_DUMP_PARSE_TREE_H_
