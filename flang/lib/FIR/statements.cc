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

#include "statements.h"

namespace Fortran::FIR {

static Addressable_impl *GetAddressable(Statement *stmt) {
  return std::visit(
      [](auto &s) -> Addressable_impl * {
        if constexpr (std::is_base_of_v<Addressable_impl,
                          std::decay_t<decltype(s)>>) {
          return &s;
        }
        return nullptr;
      },
      stmt->u);
}

static ApplyExprStmt *GetApplyExpr(Statement *stmt) {
  return std::visit(common::visitors{
                        [](ApplyExprStmt &s) { return &s; },
                        [](auto &) -> ApplyExprStmt * { return nullptr; },
                    },
      stmt->u);
}

static std::string dump(const Expression &e) {
  std::stringstream stringStream;
  e.v.AsFortran(stringStream);
  return stringStream.str();
}

static std::string dump(const Expression *e) {
  if (e) {
    return dump(*e);
  }
  return "<null-expr>"s;
}

static std::string dump(const Variable *var) {
#if 0
  if (auto *var{std::get_if<const semantics::Symbol *>(&var->u)}) {
    return (*var)->name().ToString();
  }
  return "<var>"s;
#endif
  return (*var)->name().ToString();
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
                  designator.value().u);
            },
            [](const common::Indirection<parser::FunctionReference> &) {
              return "<function-reference>"s;
            },
        },
        pathVariable->u);
  }
  return "<emty>"s;
}

std::string Evaluation::dump() const {
  return std::visit(
      common::visitors{
          [](Expression *expression) { return FIR::dump(expression); },
          [](Variable *variable) { return FIR::dump(variable); },
          [](PathVariable *pathVariable) { return FIR::dump(pathVariable); },
          [](const semantics::Symbol *symbol) {
            return symbol->name().ToString();
          },
      },
      u);
}

BranchStmt::BranchStmt(
    Statement *cond, BasicBlock *trueBlock, BasicBlock *falseBlock)
  : condition_{cond}, succs_{trueBlock, falseBlock} {
  CHECK(succs_[TrueIndex]);
  if (cond) {
    CHECK(condition_);
    CHECK(succs_[FalseIndex]);
  }
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

LoadInsn::LoadInsn(Statement *addr) : address_{GetAddressable(addr)} {
  CHECK(address_);
}

StoreInsn::StoreInsn(Statement *addr, Statement *val)
  : address_{GetAddressable(addr)} {
  CHECK(address_);
  if (auto *value{GetAddressable(val)}) {
    value_ = value;
  } else {
    auto *expr{GetApplyExpr(val)};
    CHECK(expr);
    value_ = expr;
  }
}

StoreInsn::StoreInsn(Statement *addr, BasicBlock *val)
  : address_{GetAddressable(addr)}, value_{val} {
  CHECK(address_);
  CHECK(val);
}

IncrementStmt::IncrementStmt(Statement *v1, Statement *v2) : value_{v1, v2} {}

DoConditionStmt::DoConditionStmt(Statement *dir, Statement *v1, Statement *v2)
  : value_{dir, v1, v2} {}

std::string Statement::dump() const {
  return std::visit(
      common::visitors{
          [](const ReturnStmt &) { return "return"s; },
          [](const BranchStmt &branch) {
            if (branch.hasCondition()) {
              std::string cond{"???"};
              if (auto expr{GetApplyExpr(branch.getCond())}) {
                cond = FIR::dump(expr->expression());
              }
              return "branch (" + cond + ") " +
                  std::to_string(
                      reinterpret_cast<std::intptr_t>(branch.getTrueSucc())) +
                  ' ' +
                  std::to_string(
                      reinterpret_cast<std::intptr_t>(branch.getFalseSucc()));
            }
            return "goto " +
                std::to_string(
                    reinterpret_cast<std::intptr_t>(branch.getTrueSucc()));
          },
          [](const SwitchStmt &stmt) {
            return "switch(" + stmt.getCond().dump() + ")";
          },
          [](const SwitchCaseStmt &switchCaseStmt) {
            return "switch-case(" + switchCaseStmt.getCond().dump() + ")";
          },
          [](const SwitchTypeStmt &switchTypeStmt) {
            return "switch-type(" + switchTypeStmt.getCond().dump() + ")";
          },
          [](const SwitchRankStmt &switchRankStmt) {
            return "switch-rank(" + switchRankStmt.getCond().dump() + ")";
          },
          [](const IndirectBranchStmt &) { return "ibranch"s; },
          [](const UnreachableStmt &) { return "unreachable"s; },
          [](const IncrementStmt &) { return "increment"s; },
          [](const DoConditionStmt &) { return "compare"s; },
          [](const ApplyExprStmt &e) { return FIR::dump(e.expression()); },
          [](const LocateExprStmt &e) {
            return "&" + FIR::dump(e.expression());
          },
          [](const AllocateInsn &) { return "alloc"s; },
          [](const DeallocateInsn &) { return "dealloc"s; },
          [](const AllocateLocalInsn &) { return "alloca"s; },
          [](const LoadInsn &) { return "load"s; },
          [](const StoreInsn &) { return "store"s; },
          [](const DisassociateInsn &) { return "NULLIFY"s; },
          [](const CallStmt &) { return "call"s; },
          [](const RuntimeStmt &) { return "runtime-call()"s; },
          [](const IORuntimeStmt &) { return "io-call()"s; },
          [](const ScopeEnterStmt &) { return "scopeenter"s; },
          [](const ScopeExitStmt &) { return "scopeexit"s; },
          [](const PHIStmt &) { return "PHI"s; },
      },
      u);
}
}
