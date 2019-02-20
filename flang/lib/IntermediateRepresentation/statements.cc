// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "stmt.h"

namespace Fortran::IntermediateRepresentation {

static std::string dump(const Expression *expression) {
  if (expression) {
    std::stringstream stringStream;
    expression->v.AsFortran(stringStream);
    return stringStream.str();
  }
  return "<null-expr>"s;
}

static std::string dump(const Variable *variable) {
#if 0
  if (auto *var{std::get_if<const semantics::Symbol *>(&variable->u)}) {
    return (*var)->name().ToString();
  }
  return "<var>"s;
#endif
  return (*variable)->name().ToString();
}

static std::string dump(PathVariable *pathVariable) {
  if (pathVariable) {
    return std::visit(
        common::visitors{
            [](const common::Indirection<parser::Designator> &designator) {
              return std::visit(
                  common::visitors{
                      [](const parser::ObjectName &objectName) {
                        return objectName.symbol->name().ToString();
                      },
                      [](const parser::DataRef &dataRef) {
                        return std::visit(
                            common::visitors{
                                [](const parser::Name &name) {
                                  return name.symbol->name().ToString();
                                },
                                [](const common::Indirection<
                                    parser::StructureComponent> &) {
                                  return "<structure-component>"s;
                                },
                                [](const common::Indirection<
                                    parser::ArrayElement> &) {
                                  return "<array-element>"s;
                                },
                                [](const common::Indirection<
                                    parser::CoindexedNamedObject> &) {
                                  return "<coindexed-named-object>"s;
                                },
                            },
                            dataRef.u);
                      },
                      [](const parser::Substring &substring) {
                        return "<substring>"s;
                      },
                  },
                  designator->u);
            },
            [](const common::Indirection<parser::FunctionReference>
                    &functionReference) { return "<function-reference>"s; },
        },
        pathVariable->u);
  }
  return "<emty>"s;
}

std::string Evaluation::dump() const {
  return std::visit(common::visitors{
                        [](Expression *expression) {
                          return IntermediateRepresentation::dump(expression);
                        },
                        [](Variable *variable) {
                          return IntermediateRepresentation::dump(variable);
                        },
                        [](PathVariable *pathVariable) {
                          return IntermediateRepresentation::dump(pathVariable);
                        },
                        [](const semantics::Symbol *symbol) {
                          return symbol->name().ToString();
                        },
                    },
      u);
}

ReturnStmt::ReturnStmt(Expression *expression) : returnValue_{expression} {}

BranchStmt::BranchStmt(
    Expression *condition, BasicBlock *trueBlock, BasicBlock *falseBlock)
  : condition_{condition} {
  succs_[TrueIndex] = trueBlock;
  succs_[FalseIndex] = falseBlock;
}

template<typename L>
static std::list<BasicBlock *> SuccBlocks(const L &valueSuccPairList) {
  std::list<BasicBlock *> result;
  for (auto &p : valueSuccPairList) {
    result.push_back(p.second);
  }
  return result;
}

SwitchStmt::SwitchStmt(const Evaluation &condition, BasicBlock *defaultBlock,
    const ValueSuccPairListType &args)
  : condition_{condition} {
  valueSuccPairs_.push_back({nullptr, defaultBlock});
  valueSuccPairs_.insert(valueSuccPairs_.end(), args.begin(), args.end());
}
std::list<BasicBlock *> SwitchStmt::succ_blocks() const {
  return SuccBlocks(valueSuccPairs_);
}

SwitchCaseStmt::SwitchCaseStmt(const Evaluation &condition,
    BasicBlock *defaultBlock, const ValueSuccPairListType &args)
  : condition_{condition} {
  valueSuccPairs_.push_back({SwitchCaseStmt::Default{}, defaultBlock});
  valueSuccPairs_.insert(valueSuccPairs_.end(), args.begin(), args.end());
}
std::list<BasicBlock *> SwitchCaseStmt::succ_blocks() const {
  return SuccBlocks(valueSuccPairs_);
}

SwitchTypeStmt::SwitchTypeStmt(const Evaluation &condition,
    BasicBlock *defaultBlock, const ValueSuccPairListType &args)
  : condition_{condition} {
  valueSuccPairs_.push_back({SwitchTypeStmt::Default{}, defaultBlock});
  valueSuccPairs_.insert(valueSuccPairs_.end(), args.begin(), args.end());
}
std::list<BasicBlock *> SwitchTypeStmt::succ_blocks() const {
  return SuccBlocks(valueSuccPairs_);
}

SwitchRankStmt ::SwitchRankStmt(const Evaluation &condition,
    BasicBlock *defaultBlock, const ValueSuccPairListType &args)
  : condition_{condition} {
  valueSuccPairs_.push_back({SwitchRankStmt::Default{}, defaultBlock});
  valueSuccPairs_.insert(valueSuccPairs_.end(), args.begin(), args.end());
}
std::list<BasicBlock *> SwitchRankStmt::succ_blocks() const {
  return SuccBlocks(valueSuccPairs_);
}

std::string Statement::dump() const {
  return std::visit(
      common::visitors{
          [](const ReturnStmt &) { return "return"s; },
          [](const BranchStmt &branchStatement) {
            if (branchStatement.hasCondition()) {
              return "branch ("s +
                  IntermediateRepresentation::dump(branchStatement.getCond()) +
                  ") "s +
                  std::to_string(reinterpret_cast<std::intptr_t>(
                      branchStatement.getTrueSucc())) +
                  " "s +
                  std::to_string(reinterpret_cast<std::intptr_t>(
                      branchStatement.getFalseSucc()));
            }
            return "goto "s +
                std::to_string(reinterpret_cast<std::intptr_t>(
                    branchStatement.getTrueSucc()));
          },
          [](const SwitchStmt &switchStatement) {
            return "switch("s + switchStatement.getCond().dump() + ")"s;
          },
          [](const IndirectBrStmt &) { return "ibranch"s; },
          [](const UnreachableStmt &) { return "unreachable"s; },
          [](const AllocateStmt &) { return "alloc"s; },
          [](const DeallocateStmt &) { return "dealloc"s; },
          [](const AssignmentStmt &assignmentStatement) {
            auto computedValue{IntermediateRepresentation::dump(
                assignmentStatement.GetRightHandSide())};
            auto address{IntermediateRepresentation::dump(
                assignmentStatement.GetLeftHandSide())};
            return "assign ("s + computedValue + ") to "s + address;
          },
          [](const PointerAssignStmt &pointerAssignmentStatement) {
            auto computedAddress{IntermediateRepresentation::dump(
                pointerAssignmentStatement.GetRightHandSide())};
            auto address{IntermediateRepresentation::dump(
                pointerAssignmentStatement.GetLeftHandSide())};
            return "assign &("s + computedAddress + ") to "s + address;
          },
          [](const LabelAssignStmt &) { return "lblassn"s; },
          [](const DisassociateStmt &) { return "NULLIFY"s; },
          [](const ExprStmt &expressionStatement) {
            return std::visit(
                common::visitors{
                    [](const parser::AssociateStmt *) {
                      return "<eavl-associate>"s;
                    },
                    [](const parser::ChangeTeamStmt *) {
                      return "<eval-change-team>"s;
                    },
                    [](const parser::NonLabelDoStmt *) { return "<eval-do>"s; },
                    [](const parser::SelectTypeStmt *) {
                      return "<eval-select-type>"s;
                    },
                    [](const parser::ForallConstructStmt *) {
                      return "<eval-forall>"s;
                    },
                    [](const parser::SelectRankStmt *) {
                      return "<eval-select-rank>"s;
                    },
                    [](const evaluate::GenericExprWrapper
                            *genericExpressionWrapper) {
                      return IntermediateRepresentation::dump(
                          genericExpressionWrapper);
                    },
                },
                expressionStatement.u);
          },
          [](const ScopeEnterStmt &) { return "scopeenter"s; },
          [](const ScopeExitStmt &) { return "scopeexit"s; },
          [](const PHIStmt &) { return "PHI"s; },
          [](const CallStmt &) { return "call"s; },
          [](const RuntimeStmt &) { return "runtime-call()"s; },
          [](const IORuntimeStmt &) { return "io-call()"s; },
          [](const SwitchCaseStmt &switchCaseStmt) {
            return "switch-case("s + switchCaseStmt.getCond().dump() + ")"s;
          },
          [](const SwitchTypeStmt &switchTypeStmt) {
            return "switch-type("s + switchTypeStmt.getCond().dump() + ")"s;
          },
          [](const SwitchRankStmt &switchRankStmt) {
            return "switch-rank("s + switchRankStmt.getCond().dump() + ")"s;
          },
          [](const AllocLocalInsn &) { return "alloca"s; },
          [](const LoadInsn &) { return "load"s; },
          [](const StoreInsn &) { return "store"s; },
      },
      u);
}

}
