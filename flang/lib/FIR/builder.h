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

#ifndef FORTRAN_FIR_BUILDER_H_
#define FORTRAN_FIR_BUILDER_H_

#include "stmt.h"
#include <initializer_list>

namespace Fortran::FIR {

struct FIRBuilder {
  explicit FIRBuilder(BasicBlock &block)
    : cursorRegion_{block.getParent()}, cursorBlock_{&block} {}
  template<typename A> Statement &Insert(A &&s) {
    CHECK(GetInsertionPoint());
    auto *statement{new Statement(GetInsertionPoint(), s)};
    return *statement;
  }
  template<typename A> Statement &InsertTerminator(A &&s) {
    auto &statement{Insert(s)};
    for (auto *block : s.succ_blocks()) {
      block->addPred(GetInsertionPoint());
    }
    return statement;
  }
  void SetInsertionPoint(BasicBlock *bb) {
    cursorBlock_ = bb;
    cursorRegion_ = bb->getParent();
  }
  void ClearInsertionPoint() { cursorBlock_ = nullptr; }
  BasicBlock *GetInsertionPoint() const { return cursorBlock_; }

  Statement &CreateAlloc(const Expression *object) {
    return Insert(AllocateStmt::Create(object));
  }
  Statement &CreateAssign(const PathVariable *lhs, const Expression *rhs) {
    return Insert(AssignmentStmt::Create(lhs, rhs));
  }
  Statement &CreateAssign(const semantics::Symbol *lhs, BasicBlock *rhs) {
    return Insert(LabelAssignStmt::Create(lhs, rhs));
  }
  Statement &CreateBranch(BasicBlock *block) {
    return InsertTerminator(BranchStmt::Create(block));
  }
  Statement &CreateCall(const FunctionType *type, const Value *callee,
      CallArguments &&arguments) {
    return Insert(CallStmt::Create(type, callee, std::move(arguments)));
  }
  template<typename A>
  Statement &CreateConditionalBranch(
      A *condition, BasicBlock *trueBlock, BasicBlock *falseBlock) {
    return InsertTerminator(
        BranchStmt::Create(condition, trueBlock, falseBlock));
  }
  Statement &CreateDealloc(const Expression *object) {
    return Insert(DeallocateStmt::Create(object));
  }
  template<typename A> Statement &CreateExpr(const A *a) {
    return Insert(ExprStmt::Create(a));
  }
  Statement &CreateIOCall(
      InputOutputCallType call, IOCallArguments &&arguments) {
    return Insert(IORuntimeStmt::Create(call, std::move(arguments)));
  }
  Statement &CreateIndirectBr(
      Variable *var, const std::vector<BasicBlock *> &potentials) {
    return InsertTerminator(IndirectBrStmt::Create(var, potentials));
  }
  Statement &CreateNullify(const parser::NullifyStmt *statement) {
    return Insert(DisassociateStmt::Create(statement));
  }
  Statement &CreatePointerAssign(const Expression *lhs, const Expression *rhs) {
    return Insert(PointerAssignStmt::Create(lhs, rhs));
  }
  Statement &CreateRetVoid() { return InsertTerminator(ReturnStmt::Create()); }
  template<typename A> Statement &CreateReturn(A *expr) {
    return InsertTerminator(ReturnStmt::Create(expr));
  }
  Statement &CreateRuntimeCall(
      RuntimeCallType call, RuntimeCallArguments &&arguments) {
    return Insert(RuntimeStmt::Create(call, std::move(arguments)));
  }
  Statement &CreateSwitch(const Evaluation &condition, BasicBlock *defaultCase,
      const SwitchStmt::ValueSuccPairListType &rest) {
    return InsertTerminator(SwitchStmt::Create(condition, defaultCase, rest));
  }
  Statement &CreateSwitchCase(const Evaluation &condition,
      BasicBlock *defaultCase,
      const SwitchCaseStmt::ValueSuccPairListType &rest) {
    return InsertTerminator(
        SwitchCaseStmt::Create(condition, defaultCase, rest));
  }
  Statement &CreateSwitchType(const Evaluation &condition,
      BasicBlock *defaultCase,
      const SwitchTypeStmt::ValueSuccPairListType &rest) {
    return InsertTerminator(
        SwitchTypeStmt::Create(condition, defaultCase, rest));
  }
  Statement &CreateSwitchRank(const Evaluation &condition,
      BasicBlock *defaultCase,
      const SwitchRankStmt::ValueSuccPairListType &rest) {
    return InsertTerminator(
        SwitchRankStmt::Create(condition, defaultCase, rest));
  }
  Statement &CreateUnreachable() {
    return InsertTerminator(UnreachableStmt::Create());
  }

  void PushBlock(BasicBlock *block) { blockStack_.push_back(block); }
  BasicBlock *PopBlock() {
    auto *block{blockStack_.back()};
    blockStack_.pop_back();
    return block;
  }
  void dump() const;
  void SetCurrentRegion(Region *region) { cursorRegion_ = region; }
  Region *GetCurrentRegion() const { return cursorRegion_; }

private:
  Region *cursorRegion_;
  BasicBlock *cursorBlock_;
  std::vector<BasicBlock *> blockStack_;
};

}

#endif
