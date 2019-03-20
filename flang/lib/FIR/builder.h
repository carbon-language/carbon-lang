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

#include "statements.h"
#include <initializer_list>

namespace Fortran::FIR {

struct FIRBuilder {
  explicit FIRBuilder(BasicBlock &block)
    : cursorRegion_{block.getParent()}, cursorBlock_{&block} {}
  template<typename A> Statement *Insert(A &&s) {
    CHECK(GetInsertionPoint());
    auto *statement{new Statement(GetInsertionPoint(), s)};
    return statement;
  }
  template<typename A> Statement *InsertTerminator(A &&s) {
    auto *stmt{Insert(s)};
    for (auto *block : s.succ_blocks()) {
      block->addPred(GetInsertionPoint());
    }
    return stmt;
  }
  void SetInsertionPoint(BasicBlock *bb) {
    cursorBlock_ = bb;
    cursorRegion_ = bb->getParent();
  }

  void ClearInsertionPoint() { cursorBlock_ = nullptr; }

  BasicBlock *GetInsertionPoint() const { return cursorBlock_; }

  Statement *CreateAddr(const Expression *e) {
    return Insert(LocateExprStmt::Create(e));
  }
  Statement *CreateAddr(Expression &&e) {
    return Insert(LocateExprStmt::Create(std::move(e)));
  }
  Statement *CreateAlloc(Type type) {
    return Insert(AllocateInsn::Create(type));
  }
  Statement *CreateBranch(BasicBlock *block) {
    return InsertTerminator(BranchStmt::Create(block));
  }
  Statement *CreateCall(
      const FunctionType *type, const Value callee, CallArguments &&args) {
    return Insert(CallStmt::Create(type, callee, std::move(args)));
  }
  Statement *CreateConditionalBranch(
      Statement *cond, BasicBlock *trueBlock, BasicBlock *falseBlock) {
    return InsertTerminator(BranchStmt::Create(cond, trueBlock, falseBlock));
  }
  Statement *CreateDealloc(AllocateInsn *alloc) {
    return Insert(DeallocateInsn::Create(alloc));
  }
  Statement *CreateDoCondition(Statement *dir, Statement *v1, Statement *v2) {
    return Insert(DoConditionStmt::Create(dir, v1, v2));
  }
  Statement *CreateExpr(const Expression *e) {
    return Insert(ApplyExprStmt::Create(e));
  }
  Statement *CreateExpr(Expression &&e) {
    return Insert(ApplyExprStmt::Create(std::move(e)));
  }
  ApplyExprStmt *MakeAsExpr(const Expression *e) {
    return GetApplyExpr(CreateExpr(e));
  }
  Statement *CreateIncrement(Statement *v1, Statement *v2) {
    return Insert(IncrementStmt::Create(v1, v2));
  }
  Statement *CreateIndirectBr(Variable *v, const std::vector<BasicBlock *> &p) {
    return InsertTerminator(IndirectBranchStmt::Create(v, p));
  }
  Statement *CreateIOCall(InputOutputCallType c, IOCallArguments &&a) {
    return Insert(IORuntimeStmt::Create(c, std::move(a)));
  }
  Statement *CreateLoad(Statement *addr) {
    return Insert(LoadInsn::Create(addr));
  }
  Statement *CreateLocal(
      Type type, int alignment = 0, Expression *expr = nullptr) {
    return Insert(AllocateLocalInsn::Create(type, alignment, expr));
  }
  Statement *CreateNullify(Statement *s) {
    return Insert(DisassociateInsn::Create(s));
  }
  Statement *CreateReturn(Statement *expr) {
    return InsertTerminator(ReturnStmt::Create(expr));
  }
  Statement *CreateRuntimeCall(
      RuntimeCallType call, RuntimeCallArguments &&arguments) {
    return Insert(RuntimeStmt::Create(call, std::move(arguments)));
  }
  Statement *CreateStore(Statement *addr, Statement *value) {
    return Insert(StoreInsn::Create(addr, value));
  }
  Statement *CreateStore(Statement *addr, BasicBlock *value) {
    return Insert(StoreInsn::Create(addr, value));
  }
  Statement *CreateSwitch(
      Value cond, const SwitchStmt::ValueSuccPairListType &pairs) {
    return InsertTerminator(SwitchStmt::Create(cond, pairs));
  }
  Statement *CreateSwitchCase(
      Value cond, const SwitchCaseStmt::ValueSuccPairListType &pairs) {
    return InsertTerminator(SwitchCaseStmt::Create(cond, pairs));
  }
  Statement *CreateSwitchType(
      Value cond, const SwitchTypeStmt::ValueSuccPairListType &pairs) {
    return InsertTerminator(SwitchTypeStmt::Create(cond, pairs));
  }
  Statement *CreateSwitchRank(
      Value cond, const SwitchRankStmt::ValueSuccPairListType &pairs) {
    return InsertTerminator(SwitchRankStmt::Create(cond, pairs));
  }
  Statement *CreateUnreachable() {
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
