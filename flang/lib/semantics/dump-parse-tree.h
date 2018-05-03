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
#ifndef FORTRAN_SEMANTICS_PARSETREEDUMP_H_
#define FORTRAN_SEMANTICS_PARSETREEDUMP_H_

#include "symbol.h"
#include "../parser/format-specification.h"
#include "../parser/idioms.h"
#include "../parser/indirection.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace Fortran::semantics {

//
// Dump the Parse Tree hiearchy of any node 'x' of the parse tree.
//
// ParseTreeDumper().run(x)
//

class ParseTreeDumper {
private:
  int indent_;
  std::ostream &out;
  bool emptyline;

public:
  ParseTreeDumper(std::ostream &out_ = std::cerr)
    : indent_(0), out(out_), emptyline(false) {}

  // Provide a name to a parse-tree node.
  // TODO: Provide a name for the 400+ classes in the parse-tree.
  template<typename T> const char *GetNodeName(const T &x) {
    if constexpr (std::is_same_v<T, Fortran::parser::AcImpliedDo>) {
      return "AcImpliedDo";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::AcImpliedDoControl>) {
      return "AcImpliedDoControl";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AcValue>) {
      return "AcValue";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AccessStmt>) {
      return "AccessStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AccessId>) {
      return "AccessId";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AccessSpec>) {
      return "AccessSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AccessSpec::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ActionStmt>) {
      return "ActionStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ActualArg>) {
      return "ActualArg";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ActualArgSpec>) {
      return "ActualArgSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AllocOpt>) {
      return "AllocOpt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AllocatableStmt>) {
      return "AllocatableStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::AllocateCoarraySpec>) {
      return "AllocateCoarraySpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AllocateObject>) {
      return "AllocateObject";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::AllocateShapeSpec>) {
      return "AllocateShapeSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AllocateStmt>) {
      return "AllocateStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Allocation>) {
      return "Allocation";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AltReturnSpec>) {
      return "AltReturnSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ArithmeticIfStmt>) {
      return "ArithmeticIfStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ArrayConstructor>) {
      return "ArrayConstructor";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ArrayElement>) {
      return "ArrayElement";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ArraySpec>) {
      return "ArraySpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AssignStmt>) {
      return "AssignStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AssignedGotoStmt>) {
      return "AssignedGotoStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AssignmentStmt>) {
      return "AssignmentStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::AssociateConstruct>) {
      return "AssociateConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AssociateStmt>) {
      return "AssociateStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Association>) {
      return "Association";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::AssumedImpliedSpec>) {
      return "AssumedImpliedSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AssumedShapeSpec>) {
      return "AssumedShapeSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AssumedSizeSpec>) {
      return "AssumedSizeSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AsynchronousStmt>) {
      return "AsynchronousStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AttrSpec>) {
      return "AttrSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::BOZLiteralConstant>) {
      return "BOZLiteralConstant";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BackspaceStmt>) {
      return "BackspaceStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BasedPointerStmt>) {
      return "BasedPointerStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BindAttr>) {
      return "BindAttr";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BindEntity>) {
      return "BindEntity";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BindStmt>) {
      return "BindStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Block>) {
      return "Block";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BlockConstruct>) {
      return "BlockConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BlockData>) {
      return "BlockData";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BlockDataStmt>) {
      return "BlockDataStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::BlockSpecificationPart>) {
      return "BlockSpecificationPart";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BlockStmt>) {
      return "BlockStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BoundsRemapping>) {
      return "BoundsRemapping";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BoundsSpec>) {
      return "BoundsSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Call>) {
      return "Call";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CallStmt>) {
      return "CallStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CaseConstruct>) {
      return "CaseConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CaseSelector>) {
      return "CaseSelector";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CaseStmt>) {
      return "CaseStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CaseValueRange>) {
      return "CaseValueRange";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ChangeTeamConstruct>) {
      return "ChangeTeamConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ChangeTeamStmt>) {
      return "ChangeTeamStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CharLength>) {
      return "CharLength";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CharLiteralConstant>) {
      return "CharLiteralConstant";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CharLiteralConstantSubstring>) {
      return "CharLiteralConstantSubstring";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CharLiteralConstantSubstring>) {
      return "CharLiteralConstantSubstring;";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CharSelector>) {
      return "CharSelector";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CharVariable>) {
      return "CharVariable";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CloseStmt>) {
      return "CloseStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CoarrayAssociation>) {
      return "CoarrayAssociation";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CoarraySpec>) {
      return "CoarraySpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CodimensionDecl>) {
      return "CodimensionDecl";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CodimensionStmt>) {
      return "CodimensionStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CoindexedNamedObject>) {
      return "CoindexedNamedObject";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CommonBlockObject>) {
      return "CommonBlockObject";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CommonStmt>) {
      return "CommonStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CompilerDirective>) {
      return "CompilerDirective";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ComplexLiteralConstant>) {
      return "ComplexLiteralConstant";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ComplexPart>) {
      return "ComplexPart";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ComponentArraySpec>) {
      return "ComponentArraySpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ComponentAttrSpec>) {
      return "ComponentAttrSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ComponentDataSource>) {
      return "ComponentDataSource";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ComponentDecl>) {
      return "ComponentDecl";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ComponentDefStmt>) {
      return "ComponentDefStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ComponentSpec>) {
      return "ComponentSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ComputedGotoStmt>) {
      return "ComputedGotoStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ConcurrentControl>) {
      return "ConcurrentControl";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ConcurrentHeader>) {
      return "ConcurrentHeader";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ConnectSpec>) {
      return "ConnectSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ConstantValue>) {
      return "ConstantValue";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ContiguousStmt>) {
      return "ContiguousStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ContinueStmt>) {
      return "ContinueStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CriticalConstruct>) {
      return "CriticalConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CriticalStmt>) {
      return "CriticalStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::CycleStmt>) {
      return "CycleStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DataComponentDefStmt>) {
      return "DataComponentDefStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DataIDoObject>) {
      return "DataIDoObject";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DataImpliedDo>) {
      return "DataImpliedDo";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DataRef>) {
      return "DataRef";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DataStmt>) {
      return "DataStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DataStmtConstant>) {
      return "DataStmtConstant";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DataStmtObject>) {
      return "DataStmtObject";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DataStmtRepeat>) {
      return "DataStmtRepeat";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DataStmtSet>) {
      return "DataStmtSet";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DataStmtValue>) {
      return "DataStmtValue";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DeallocateStmt>) {
      return "DeallocateStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DeclarationConstruct>) {
      return "DeclarationConstruct";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DeclarationTypeSpec>) {
      return "DeclarationTypeSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DeferredCoshapeSpecList>) {
      return "DeferredCoshapeSpecList";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DeferredShapeSpecList>) {
      return "DeferredShapeSpecList";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DefinedOpName>) {
      return "DefinedOpName";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DefinedOperator>) {
      return "DefinedOperator";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DerivedTypeDef>) {
      return "DerivedTypeDef";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DerivedTypeSpec>) {
      return "DerivedTypeSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DerivedTypeStmt>) {
      return "DerivedTypeStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Designator>) {
      return "Designator";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DimensionStmt>) {
      return "DimensionStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DoConstruct>) {
      return "DoConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::DummyArg>) {
      return "DummyArg";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ElseIfStmt>) {
      return "ElseIfStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ElseStmt>) {
      return "ElseStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ElsewhereStmt>) {
      return "ElsewhereStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndAssociateStmt>) {
      return "EndAssociateStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndBlockDataStmt>) {
      return "EndBlockDataStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndBlockStmt>) {
      return "EndBlockStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::EndChangeTeamStmt>) {
      return "EndChangeTeamStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndCriticalStmt>) {
      return "EndCriticalStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndDoStmt>) {
      return "EndDoStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndForallStmt>) {
      return "EndForallStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndFunctionStmt>) {
      return "EndFunctionStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndIfStmt>) {
      return "EndIfStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndInterfaceStmt>) {
      return "EndInterfaceStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndLabel>) {
      return "EndLabel";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndModuleStmt>) {
      return "EndModuleStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::EndMpSubprogramStmt>) {
      return "EndMpSubprogramStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndProgramStmt>) {
      return "EndProgramStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndSelectStmt>) {
      return "EndSelectStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndSubmoduleStmt>) {
      return "EndSubmoduleStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::EndSubroutineStmt>) {
      return "EndSubroutineStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndTypeStmt>) {
      return "EndTypeStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndWhereStmt>) {
      return "EndWhereStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndfileStmt>) {
      return "EndfileStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EntityDecl>) {
      return "EntityDecl";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EntryStmt>) {
      return "EntryStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EnumDef>) {
      return "EnumDef";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Enumerator>) {
      return "Enumerator";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::EnumeratorDefStmt>) {
      return "EnumeratorDefStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EorLabel>) {
      return "EorLabel";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::EquivalenceObject>) {
      return "EquivalenceObject";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EquivalenceStmt>) {
      return "EquivalenceStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ErrLabel>) {
      return "ErrLabel";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EventPostStmt>) {
      return "EventPostStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EventWaitStmt>) {
      return "EventWaitStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ExecutableConstruct>) {
      return "ExecutableConstruct";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ExecutionPartConstruct>) {
      return "ExecutionPartConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ExitStmt>) {
      return "ExitStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ExplicitCoshapeSpec>) {
      return "ExplicitCoshapeSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ExplicitShapeSpec>) {
      return "ExplicitShapeSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr>) {
      return "Expr";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::Expr::Parentheses>) {
      return "Parentheses";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::UnaryPlus>) {
      return "UnaryPlus";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::Negate>) {
      return "Negate";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::NOT>) {
      return "NOT";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::PercentLoc>) {
      return "PercentLoc";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::Expr::DefinedUnary>) {
      return "DefinedUnary";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::Power>) {
      return "Power";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::Multiply>) {
      return "Multiply";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::Divide>) {
      return "Divide";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::Add>) {
      return "Add";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::Subtract>) {
      return "Subtract";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::Concat>) {
      return "Concat";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::LT>) {
      return "LT";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::LE>) {
      return "LE";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::EQ>) {
      return "EQ";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::NE>) {
      return "NE";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::GE>) {
      return "GE";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::GT>) {
      return "GT";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::AND>) {
      return "AND";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::OR>) {
      return "OR";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::EQV>) {
      return "EQV";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::NEQV>) {
      return "NEQV";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Expr::XOR>) {
      return "XOR";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::Expr::DefinedBinary>) {
      return "DefinedBinary";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::Expr::ComplexConstructor>) {
      return "ComplexConstructor";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::Expr::Parentheses>) {
      return "Parentheses";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AcSpec>) {
      return "AcSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ErrorRecovery>) {
      return "ErrorRecovery";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ExternalStmt>) {
      return "ExternalStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::FailImageStmt>) {
      return "FailImageStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::FileUnitNumber>) {
      return "FileUnitNumber";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::FinalProcedureStmt>) {
      return "FinalProcedureStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::FlushStmt>) {
      return "FlushStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ForallAssignmentStmt>) {
      return "ForallAssignmentStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ForallBodyConstruct>) {
      return "ForallBodyConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ForallConstruct>) {
      return "ForallConstruct";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ForallConstructStmt>) {
      return "ForallConstructStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ForallStmt>) {
      return "ForallStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::FormTeamStmt>) {
      return "FormTeamStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Format>) {
      return "Format";
    } else if constexpr (std::is_same_v<T, Fortran::parser::FormatStmt>) {
      return "FormatStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::FunctionReference>) {
      return "FunctionReference";
    } else if constexpr (std::is_same_v<T, Fortran::parser::FunctionStmt>) {
      return "FunctionStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::FunctionSubprogram>) {
      return "FunctionSubprogram";
    } else if constexpr (std::is_same_v<T, Fortran::parser::GenericSpec>) {
      return "GenericSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::GenericStmt>) {
      return "GenericStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::GotoStmt>) {
      return "GotoStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::IdExpr>) {
      return "IdExpr";
    } else if constexpr (std::is_same_v<T, Fortran::parser::IdVariable>) {
      return "IdVariable";
    } else if constexpr (std::is_same_v<T, Fortran::parser::IfConstruct>) {
      return "IfConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::IfStmt>) {
      return "IfStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::IfThenStmt>) {
      return "IfThenStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ImageSelector>) {
      return "ImageSelector";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ImageSelectorSpec>) {
      return "ImageSelectorSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ImplicitPart>) {
      return "ImplicitPart";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ImplicitPartStmt>) {
      return "ImplicitPartStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ImplicitSpec>) {
      return "ImplicitSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ImplicitStmt>) {
      return "ImplicitStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ImpliedShapeSpec>) {
      return "ImpliedShapeSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ImportStmt>) {
      return "ImportStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Initialization>) {
      return "Initialization";
    } else if constexpr (std::is_same_v<T, Fortran::parser::InputImpliedDo>) {
      return "InputImpliedDo";
    } else if constexpr (std::is_same_v<T, Fortran::parser::InputItem>) {
      return "InputItem";
    } else if constexpr (std::is_same_v<T, Fortran::parser::InquireSpec>) {
      return "InquireSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::InquireStmt>) {
      return "InquireStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IntLiteralConstant>) {
      return "IntLiteralConstant";
    } else if constexpr (std::is_same_v<T, Fortran::parser::IntegerTypeSpec>) {
      return "IntegerTypeSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::IntentStmt>) {
      return "IntentStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::InterfaceBlock>) {
      return "InterfaceBlock";
    } else if constexpr (std::is_same_v<T, Fortran::parser::InterfaceBody>) {
      return "InterfaceBody";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InterfaceSpecification>) {
      return "InterfaceSpecification";
    } else if constexpr (std::is_same_v<T, Fortran::parser::InterfaceStmt>) {
      return "InterfaceStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InternalSubprogram>) {
      return "InternalSubprogram";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InternalSubprogramPart>) {
      return "InternalSubprogramPart";
    } else if constexpr (std::is_same_v<T, Fortran::parser::IntrinsicStmt>) {
      return "IntrinsicStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IntrinsicTypeSpec>) {
      return "IntrinsicTypeSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::IoControlSpec>) {
      return "IoControlSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::IoUnit>) {
      return "IoUnit";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Keyword>) {
      return "Keyword";
    } else if constexpr (std::is_same_v<T, Fortran::parser::KindParam>) {
      return "KindParam";
    } else if constexpr (std::is_same_v<T, Fortran::parser::KindSelector>) {
      return "KindSelector";
    } else if constexpr (std::is_same_v<T, Fortran::parser::LabelDoStmt>) {
      return "LabelDoStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::LengthSelector>) {
      return "LengthSelector";
    } else if constexpr (std::is_same_v<T, Fortran::parser::LetterSpec>) {
      return "LetterSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::LiteralConstant>) {
      return "LiteralConstant";
    } else if constexpr (std::is_same_v<T, Fortran::parser::LocalitySpec>) {
      return "LocalitySpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::LockStmt>) {
      return "LockStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::LogicalLiteralConstant>) {
      return "LogicalLiteralConstant";
    } else if constexpr (std::is_same_v<T, Fortran::parser::LoopControl>) {
      return "LoopControl";
    } else if constexpr (std::is_same_v<T, Fortran::parser::MainProgram>) {
      return "MainProgram";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Map>) {
      return "Map";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::MaskedElsewhereStmt>) {
      return "MaskedElsewhereStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Module>) {
      return "Module";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ModuleStmt>) {
      return "ModuleStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ModuleSubprogram>) {
      return "ModuleSubprogram";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ModuleSubprogramPart>) {
      return "ModuleSubprogramPart";
    } else if constexpr (std::is_same_v<T, Fortran::parser::MpSubprogramStmt>) {
      return "MpSubprogramStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::MsgVariable>) {
      return "MsgVariable";
    } else if constexpr (std::is_same_v<T, Fortran::parser::NamedConstant>) {
      return "NamedConstant";
    } else if constexpr (std::is_same_v<T, Fortran::parser::NamedConstantDef>) {
      return "NamedConstantDef";
    } else if constexpr (std::is_same_v<T, Fortran::parser::NamelistStmt>) {
      return "NamelistStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::NonLabelDoStmt>) {
      return "NonLabelDoStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::NullifyStmt>) {
      return "NullifyStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ObjectDecl>) {
      return "ObjectDecl";
    } else if constexpr (std::is_same_v<T, Fortran::parser::OldParameterStmt>) {
      return "OldParameterStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::OldParameterStmt>) {
      return "OldParameterStmt;";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Only>) {
      return "Only";
    } else if constexpr (std::is_same_v<T, Fortran::parser::OpenStmt>) {
      return "OpenStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::OptionalStmt>) {
      return "OptionalStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::OtherSpecificationStmt>) {
      return "OtherSpecificationStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::OutputImpliedDo>) {
      return "OutputImpliedDo";
    } else if constexpr (std::is_same_v<T, Fortran::parser::OutputItem>) {
      return "OutputItem";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ParameterStmt>) {
      return "ParameterStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ParentIdentifier>) {
      return "ParentIdentifier";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Pass>) {
      return "Pass";
    } else if constexpr (std::is_same_v<T, Fortran::parser::PauseStmt>) {
      return "PauseStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::PointerAssignmentStmt>) {
      return "PointerAssignmentStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::PointerDecl>) {
      return "PointerDecl";
    } else if constexpr (std::is_same_v<T, Fortran::parser::PointerObject>) {
      return "PointerObject";
    } else if constexpr (std::is_same_v<T, Fortran::parser::PointerStmt>) {
      return "PointerStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::PositionOrFlushSpec>) {
      return "PositionOrFlushSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::PrefixSpec>) {
      return "PrefixSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::PrintStmt>) {
      return "PrintStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::PrivateOrSequence>) {
      return "PrivateOrSequence";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ProcAttrSpec>) {
      return "ProcAttrSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ProcComponentAttrSpec>) {
      return "ProcComponentAttrSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ProcComponentDefStmt>) {
      return "ProcComponentDefStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ProcComponentRef>) {
      return "ProcComponentRef";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ProcDecl>) {
      return "ProcDecl";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ProcInterface>) {
      return "ProcInterface";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ProcPointerInit>) {
      return "ProcPointerInit";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ProcedureDeclarationStmt>) {
      return "ProcedureDeclarationStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ProcedureDesignator>) {
      return "ProcedureDesignator";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ProcedureStmt>) {
      return "ProcedureStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Program>) {
      return "Program";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ProgramStmt>) {
      return "ProgramStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ProgramUnit>) {
      return "ProgramUnit";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ProtectedStmt>) {
      return "ProtectedStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ReadStmt>) {
      return "ReadStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Rename>) {
      return "Rename";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ReturnStmt>) {
      return "ReturnStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::RewindStmt>) {
      return "RewindStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SaveStmt>) {
      return "SaveStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SavedEntity>) {
      return "SavedEntity";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SectionSubscript>) {
      return "SectionSubscript";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SelectCaseStmt>) {
      return "SelectCaseStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SelectRankCaseStmt>) {
      return "SelectRankCaseStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SelectRankConstruct>) {
      return "SelectRankConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SelectRankStmt>) {
      return "SelectRankStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SelectTypeConstruct>) {
      return "SelectTypeConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SelectTypeStmt>) {
      return "SelectTypeStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Selector>) {
      return "Selector";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SeparateModuleSubprogram>) {
      return "SeparateModuleSubprogram";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SignedComplexLiteralConstant>) {
      return "SignedComplexLiteralConstant";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SignedIntLiteralConstant>) {
      return "SignedIntLiteralConstant";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SignedRealLiteralConstant>) {
      return "SignedRealLiteralConstant";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SpecificationConstruct>) {
      return "SpecificationConstruct";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SpecificationExpr>) {
      return "SpecificationExpr";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SpecificationPart>) {
      return "SpecificationPart";
    } else if constexpr (std::is_same_v<T, Fortran::parser::StatOrErrmsg>) {
      return "StatOrErrmsg";
    } else if constexpr (std::is_same_v<T, Fortran::parser::StatVariable>) {
      return "StatVariable";
    } else if constexpr (std::is_same_v<T, Fortran::parser::StatusExpr>) {
      return "StatusExpr";
    } else if constexpr (std::is_same_v<T, Fortran::parser::StmtFunctionStmt>) {
      return "StmtFunctionStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::StopCode>) {
      return "StopCode";
    } else if constexpr (std::is_same_v<T, Fortran::parser::StopStmt>) {
      return "StopStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::StructureComponent>) {
      return "StructureComponent";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::StructureConstructor>) {
      return "StructureConstructor";
    } else if constexpr (std::is_same_v<T, Fortran::parser::StructureDef>) {
      return "StructureDef";
    } else if constexpr (std::is_same_v<T, Fortran::parser::StructureField>) {
      return "StructureField";
    } else if constexpr (std::is_same_v<T, Fortran::parser::StructureStmt>) {
      return "StructureStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Submodule>) {
      return "Submodule";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SubmoduleStmt>) {
      return "SubmoduleStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SubroutineStmt>) {
      return "SubroutineStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SubroutineSubprogram>) {
      return "SubroutineSubprogram";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SubscriptTriplet>) {
      return "SubscriptTriplet";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Substring>) {
      return "Substring";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SubstringRange>) {
      return "SubstringRange";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SyncAllStmt>) {
      return "SyncAllStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SyncImagesStmt>) {
      return "SyncImagesStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SyncMemoryStmt>) {
      return "SyncMemoryStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SyncTeamStmt>) {
      return "SyncTeamStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::TargetStmt>) {
      return "TargetStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::TypeAttrSpec>) {
      return "TypeAttrSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeBoundGenericStmt>) {
      return "TypeBoundGenericStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeBoundProcBinding>) {
      return "TypeBoundProcBinding";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeBoundProcDecl>) {
      return "TypeBoundProcDecl";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeBoundProcedurePart>) {
      return "TypeBoundProcedurePart";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeBoundProcedureStmt>) {
      return "TypeBoundProcedureStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeDeclarationStmt>) {
      return "TypeDeclarationStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::TypeGuardStmt>) {
      return "TypeGuardStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::TypeParamDecl>) {
      return "TypeParamDecl";
    } else if constexpr (std::is_same_v<T, Fortran::parser::TypeParamDefStmt>) {
      return "TypeParamDefStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::TypeParamInquiry>) {
      return "TypeParamInquiry";
    } else if constexpr (std::is_same_v<T, Fortran::parser::TypeParamSpec>) {
      return "TypeParamSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::TypeParamValue>) {
      return "TypeParamValue";
    } else if constexpr (std::is_same_v<T, Fortran::parser::TypeSpec>) {
      return "TypeSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Union>) {
      return "Union";
    } else if constexpr (std::is_same_v<T, Fortran::parser::UnlockStmt>) {
      return "UnlockStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::UseStmt>) {
      return "UseStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ValueStmt>) {
      return "ValueStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Variable>) {
      return "Variable";
    } else if constexpr (std::is_same_v<T, Fortran::parser::VolatileStmt>) {
      return "VolatileStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::WaitSpec>) {
      return "WaitSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::WaitStmt>) {
      return "WaitStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::WhereBodyConstruct>) {
      return "WhereBodyConstruct";
    } else if constexpr (std::is_same_v<T, Fortran::parser::WhereConstruct>) {
      return "WhereConstruct";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::WhereConstructStmt>) {
      return "WhereConstructStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::WhereStmt>) {
      return "WhereStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::WriteStmt>) {
      return "WriteStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::HollerithLiteralConstant>) {
      return "HollerithLiteralConstant";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::RealLiteralConstant>) {
      return "RealLiteralConstant";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::RealLiteralConstant::Real>) {
      return "Real";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CloseStmt::CloseSpec>) {
      return "CloseSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InquireStmt::Iolength>) {
      return "Iolength";
    } else if constexpr (std::is_same_v<T, Fortran::format::ControlEditDesc>) {
      return "ControlEditDesc";
    } else if constexpr (std::is_same_v<T,
                             Fortran::format::ControlEditDesc::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T,
                             Fortran::format::DerivedTypeDataEditDesc>) {
      return "DerivedTypeDataEditDesc";
    } else if constexpr (std::is_same_v<T, Fortran::format::FormatItem>) {
      return "FormatItem";
    } else if constexpr (std::is_same_v<T,
                             Fortran::format::FormatSpecification>) {
      return "FormatSpecification";
    } else if constexpr (std::is_same_v<T,
                             Fortran::format::IntrinsicTypeDataEditDesc>) {
      return "IntrinsicTypeDataEditDesc";
    } else if constexpr (
        std::is_same_v<T, Fortran::format::IntrinsicTypeDataEditDesc::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Abstract>) {
      return "Abstract";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AcValue::Triplet>) {
      return "Triplet";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ActualArg::PercentRef>) {
      return "PercentRef";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ActualArg::PercentVal>) {
      return "PercentVal";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AllocOpt::Mold>) {
      return "Mold";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AllocOpt::Source>) {
      return "Source";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Allocatable>) {
      return "Allocatable";
    } else if constexpr (std::is_same_v<T, Fortran::parser::AssumedRankSpec>) {
      return "AssumedRankSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Asynchronous>) {
      return "Asynchronous";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::BindAttr::Deferred>) {
      return "Deferred";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::BindAttr::Non_Overridable>) {
      return "Non_Overridable";
    } else if constexpr (std::is_same_v<T, Fortran::parser::BindEntity::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CaseConstruct::Case>) {
      return "Case";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CaseValueRange::Range>) {
      return "Range";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CharSelector::LengthAndKind>) {
      return "LengthAndKind";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CommonStmt::Block>) {
      return "Block";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CompilerDirective::IVDEP>) {
      return "IVDEP";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::CompilerDirective::IgnoreTKR>) {
      return "IgnoreTKR";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ConnectSpec::CharExpr>) {
      return "CharExpr";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ConnectSpec::CharExpr::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ConnectSpec::Newunit>) {
      return "Newunit";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ConnectSpec::Recl>) {
      return "Recl";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ContainsStmt>) {
      return "ContainsStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Contiguous>) {
      return "Contiguous";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DeclarationTypeSpec::Class>) {
      return "Class";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DeclarationTypeSpec::ClassStar>) {
      return "ClassStar";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DeclarationTypeSpec::Record>) {
      return "Record";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DeclarationTypeSpec::Type>) {
      return "Type";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DeclarationTypeSpec::TypeStar>) {
      return "TypeStar";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Default>) {
      return "Default";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DefinedOperator::
                                 IntrinsicOperator>) {
      return "IntrinsicOperator";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::DimensionStmt::Declaration>) {
      return "Declaration";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EndEnumStmt>) {
      return "EndEnumStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::EnumDefStmt>) {
      return "EnumDefStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::EventWaitStmt::EventWaitSpec>) {
      return "EventWaitSpec";
    } else if constexpr (std::is_same_v<T, Fortran::parser::ExecutionPart>) {
      return "ExecutionPart";
    } else if constexpr (std::is_same_v<T, Fortran::parser::External>) {
      return "External";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::FormTeamStmt::FormTeamSpec>) {
      return "FormTeamSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::GenericSpec::Assignment>) {
      return "Assignment";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::GenericSpec::ReadFormatted>) {
      return "ReadFormatted";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::GenericSpec::ReadUnformatted>) {
      return "ReadUnformatted";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::GenericSpec::WriteFormatted>) {
      return "WriteFormatted";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::GenericSpec::WriteUnformatted>) {
      return "WriteUnformatted";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IfConstruct::ElseBlock>) {
      return "ElseBlock";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IfConstruct::ElseIfBlock>) {
      return "ElseIfBlock";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ImageSelectorSpec::Stat>) {
      return "Stat";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ImageSelectorSpec::Team>) {
      return "Team";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ImageSelectorSpec::Team_Number>) {
      return "Team_Number";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ImplicitStmt::
                                 ImplicitNoneNameSpec>) {
      return "ImplicitNoneNameSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InquireSpec::CharVar>) {
      return "CharVar";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InquireSpec::CharVar::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InquireSpec::IntVar>) {
      return "IntVar";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InquireSpec::IntVar::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InquireSpec::LogVar>) {
      return "LogVar";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InquireSpec::LogVar::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T, Fortran::parser::IntentSpec>) {
      return "IntentSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IntentSpec::Intent>) {
      return "Intent";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InterfaceBody::Function>) {
      return "Function";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::InterfaceBody::Subroutine>) {
      return "Subroutine";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Intrinsic>) {
      return "Intrinsic";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IntrinsicTypeSpec::Character>) {
      return "Character";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IntrinsicTypeSpec::Complex>) {
      return "Complex";
    } else if constexpr (
        std::is_same_v<T, Fortran::parser::IntrinsicTypeSpec::DoubleComplex>) {
      return "DoubleComplex";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IntrinsicTypeSpec::
                                 DoublePrecision>) {
      return "DoublePrecision";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IntrinsicTypeSpec::Logical>) {
      return "Logical";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IntrinsicTypeSpec::NCharacter>) {
      return "NCharacter";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IntrinsicTypeSpec::Real>) {
      return "Real";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IoControlSpec::Asynchronous>) {
      return "Asynchronous";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IoControlSpec::CharExpr>) {
      return "CharExpr";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IoControlSpec::CharExpr::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IoControlSpec::Pos>) {
      return "Pos";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IoControlSpec::Rec>) {
      return "Rec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::IoControlSpec::Size>) {
      return "Size";
    } else if constexpr (std::is_same_v<T, Fortran::parser::KindParam::Kanji>) {
      return "Kanji";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::KindSelector::StarSize>) {
      return "StarSize";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::LanguageBindingSpec>) {
      return "LanguageBindingSpec";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::LocalitySpec::DefaultNone>) {
      return "DefaultNone";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::LocalitySpec::Local>) {
      return "Local";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::LocalitySpec::LocalInit>) {
      return "LocalInit";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::LocalitySpec::Shared>) {
      return "Shared";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::LockStmt::LockStat>) {
      return "LockStat";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::LoopBounds<
                                 Fortran::parser::ScalarIntConstantExpr>>) {
      return "LoopBounds";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::LoopBounds<
                                 Fortran::parser::ScalarIntExpr>>) {
      return "LoopBounds";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::LoopControl::Concurrent>) {
      return "Concurrent";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Map::EndMapStmt>) {
      return "EndMapStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Map::MapStmt>) {
      return "MapStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::NamelistStmt::Group>) {
      return "Group";
    } else if constexpr (std::is_same_v<T, Fortran::parser::NoPass>) {
      return "NoPass";
    } else if constexpr (std::is_same_v<T, Fortran::parser::NullInit>) {
      return "NullInit";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Optional>) {
      return "Optional";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Parameter>) {
      return "Parameter";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Pointer>) {
      return "Pointer";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::PointerAssignmentStmt::Bounds>) {
      return "Bounds";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::PrefixSpec::Elemental>) {
      return "Elemental";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::PrefixSpec::Impure>) {
      return "Impure";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::PrefixSpec::Module>) {
      return "Module";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::PrefixSpec::Non_Recursive>) {
      return "Non_Recursive";
    } else if constexpr (std::is_same_v<T, Fortran::parser::PrefixSpec::Pure>) {
      return "Pure";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::PrefixSpec::Recursive>) {
      return "Recursive";
    } else if constexpr (std::is_same_v<T, Fortran::parser::PrivateStmt>) {
      return "PrivateStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::ProcedureStmt::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Protected>) {
      return "Protected";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Rename::Names>) {
      return "Names";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::Rename::Operators>) {
      return "Operators";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Save>) {
      return "Save";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SavedEntity::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SelectRankCaseStmt::Rank>) {
      return "Rank";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SelectRankConstruct::RankCase>) {
      return "RankCase";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SelectTypeConstruct::TypeCase>) {
      return "TypeCase";
    } else if constexpr (std::is_same_v<T, Fortran::parser::SequenceStmt>) {
      return "SequenceStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Sign>) {
      return "Sign";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Star>) {
      return "Star";
    } else if constexpr (std::is_same_v<T, Fortran::parser::StopStmt::Kind>) {
      return "Kind";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::StructureDef::EndStructureStmt>) {
      return "EndStructureStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Suffix>) {
      return "Suffix";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::SyncImagesStmt::ImageSet>) {
      return "ImageSet";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Target>) {
      return "Target";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeAttrSpec::BindC>) {
      return "BindC";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeAttrSpec::Extends>) {
      return "Extends";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeBoundProcedureStmt::
                                 WithInterface>) {
      return "WithInterface";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeBoundProcedureStmt::
                                 WithoutInterface>) {
      return "WithoutInterface";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeGuardStmt::Guard>) {
      return "Guard";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeParamDefStmt::KindOrLen>) {
      return "KindOrLen";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::TypeParamValue::Deferred>) {
      return "Deferred";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::Union::EndUnionStmt>) {
      return "EndUnionStmt";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Union::UnionStmt>) {
      return "UnionStmt";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::UseStmt::ModuleNature>) {
      return "ModuleNature";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Value>) {
      return "Value";
    } else if constexpr (std::is_same_v<T, Fortran::parser::Volatile>) {
      return "Volatile";
    } else if constexpr (std::is_same_v<T,
                             Fortran::parser::WhereConstruct::Elsewhere>) {
      return "Elsewhere";
    } else if constexpr (
        std::is_same_v<T, Fortran::parser::WhereConstruct::MaskedElsewhere>) {
      return "MaskedElsewhere";
    } else if constexpr (std::is_same_v<T, bool>) {
      return "bool";
    } else if constexpr (std::is_same_v<T, const char *>) {
      return "char*";
    } else if constexpr (std::is_same_v<T, int>) {
      return "int";
    } else {
      // Uncomment the following static_assert to help figure out classes
      // that are not handled here.
      // static_assert(0);
      return "Unknown";
    }
  }

  void out_indent() {
    for (int i = 0; i < indent_; i++) {
      out << "| ";
    }
  }

  template<typename T> bool Pre(const T &x) {
    if (emptyline) {
      out_indent();
      emptyline = false;
    }
    if (UnionTrait<T> || WrapperTrait<T>) {
      out << GetNodeName(x) << " -> ";
      emptyline = false;
    } else {
      out << GetNodeName(x);
      out << "\n";
      indent_++;
      emptyline = true;
    }
    return true;
  }

  template<typename T> void Post(const T &x) {
    if (UnionTrait<T> || WrapperTrait<T>) {
      if (!emptyline) {
        out << "\n";
        emptyline = true;
      }
    } else {
      indent_--;
    }
  }

  bool PutName(const std::string &name, const semantics::Symbol *symbol) {
    if (emptyline) {
      out_indent();
      emptyline = false;
    }
    if (symbol) {
      out << "symbol = " << *symbol;
    } else {
      out << "Name = '" << name << '\'';
    }
    out << '\n';
    indent_++;
    emptyline = true;
    return true;
  }

  bool Pre(const parser::Name &x) { return PutName(x.ToString(), x.symbol); }

  void Post(const parser::Name &) { indent_--; }

  bool Pre(const std::string &x) { return PutName(x, nullptr); }

  void Post(const std::string &x) { indent_--; }

  bool Pre(const std::int64_t &x) {
    if (emptyline) {
      out_indent();
      emptyline = false;
    }
    out << "int = '" << x << "'\n";
    indent_++;
    emptyline = true;
    return true;
  }

  void Post(const std::int64_t &x) { indent_--; }

  bool Pre(const std::uint64_t &x) {
    if (emptyline) {
      out_indent();
      emptyline = false;
    }
    out << "int = '" << x << "'\n";
    indent_++;
    emptyline = true;
    return true;
  }

  void Post(const std::uint64_t &x) { indent_--; }

  // A few types we want to ignore

  template<typename T> bool Pre(const Fortran::parser::Statement<T> &) {
    return true;
  }

  template<typename T> void Post(const Fortran::parser::Statement<T> &) {}

  template<typename T> bool Pre(const Fortran::parser::Indirection<T> &) {
    return true;
  }

  template<typename T> void Post(const Fortran::parser::Indirection<T> &) {}

  template<typename T> bool Pre(const Fortran::parser::Integer<T> &) {
    return true;
  }

  template<typename T> void Post(const Fortran::parser::Integer<T> &) {}

  template<typename T> bool Pre(const Fortran::parser::Scalar<T> &) {
    return true;
  }

  template<typename T> void Post(const Fortran::parser::Scalar<T> &) {}

  template<typename... A> bool Pre(const std::tuple<A...> &) { return true; }

  template<typename... A> void Post(const std::tuple<A...> &) {}

  template<typename... A> bool Pre(const std::variant<A...> &) { return true; }

  template<typename... A> void Post(const std::variant<A...> &) {}
};

template<typename T> void DumpTree(const T &x, std::ostream &out = std::cout) {
  ParseTreeDumper dumper(out);
  Fortran::parser::Walk(x, dumper);
}

}  // namespace Fortran::semantics

// Provide a explicit instantiation for a few selected node types.
// The goal is not to provide the instanciation of all possible
// types but to insure that a call to DumpTree will not cause
// the instanciation of thousands of types.

#define FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE(MODE, TYPE) \
  MODE template void Fortran::parser::Walk( \
      const TYPE &, Fortran::semantics::ParseTreeDumper &);

#define FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE_ALL(MODE) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE(MODE, Fortran::parser::ProgramUnit) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE(MODE, Fortran::parser::SubroutineStmt) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE(MODE, Fortran::parser::ProgramStmt) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE(MODE, Fortran::parser::FunctionStmt) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE(MODE, Fortran::parser::ModuleStmt) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE(MODE, Fortran::parser::Expr) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE(MODE, Fortran::parser::ActionStmt) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE( \
      MODE, Fortran::parser::ExecutableConstruct) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE(MODE, Fortran::parser::Block) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE( \
      MODE, Fortran::parser::DeclarationConstruct) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE( \
      MODE, Fortran::parser::SpecificationPart) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE( \
      MODE, Fortran::parser::OtherSpecificationStmt) \
  FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE( \
      MODE, Fortran::parser::SpecificationConstruct)

FORTRAN_PARSE_TREE_DUMPER_INSTANTIATE_ALL(extern)

#endif  // of FORTRAN_SEMANTICS_PARSETREEDUMP_H_
