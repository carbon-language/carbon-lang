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

TerminatorStmt_impl::~TerminatorStmt_impl() = default;

Addressable_impl *GetAddressable(Statement *stmt) {
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

static std::string dump(const Expression &e) {
  std::stringstream stringStream;
  stringStream << e;
  return stringStream.str();
}

BranchStmt::BranchStmt(const std::optional<Value> &cond, BasicBlock *trueBlock,
    BasicBlock *falseBlock)
  : condition_{cond}, succs_{trueBlock, falseBlock} {
  CHECK(succs_[TrueIndex]);
  if (cond) {
    CHECK(condition_);
    CHECK(succs_[FalseIndex]);
  }
}

template<typename L>
static std::list<BasicBlock *> SuccBlocks(
    const typename L::ValueSuccPairListType &valueSuccPairList) {
  std::pair<std::list<typename L::ValueType>, std::list<BasicBlock *>> result;
  UnzipSnd(result, valueSuccPairList.begin(), valueSuccPairList.end());
  return result.second;
}

ReturnStmt::ReturnStmt(QualifiedStmt<ApplyExprStmt> exp) : value_{exp} {}
ReturnStmt::ReturnStmt() : value_{QualifiedStmt<ApplyExprStmt>{nullptr}} {}

SwitchStmt::SwitchStmt(const Value &cond, const ValueSuccPairListType &args)
  : condition_{cond} {
  valueSuccPairs_.insert(valueSuccPairs_.end(), args.begin(), args.end());
}
std::list<BasicBlock *> SwitchStmt::succ_blocks() const {
  return SuccBlocks<SwitchStmt>(valueSuccPairs_);
}
BasicBlock *SwitchStmt::defaultSucc() const {
  CHECK(IsNothing(valueSuccPairs_[0].first));
  return valueSuccPairs_[0].second;
}

SwitchCaseStmt::SwitchCaseStmt(Value cond, const ValueSuccPairListType &args)
  : condition_{cond} {
  valueSuccPairs_.insert(valueSuccPairs_.end(), args.begin(), args.end());
}
std::list<BasicBlock *> SwitchCaseStmt::succ_blocks() const {
  return SuccBlocks<SwitchCaseStmt>(valueSuccPairs_);
}
BasicBlock *SwitchCaseStmt::defaultSucc() const {
  CHECK(std::holds_alternative<Default>(valueSuccPairs_[0].first));
  return valueSuccPairs_[0].second;
}

SwitchTypeStmt::SwitchTypeStmt(Value cond, const ValueSuccPairListType &args)
  : condition_{cond} {
  valueSuccPairs_.insert(valueSuccPairs_.end(), args.begin(), args.end());
}
std::list<BasicBlock *> SwitchTypeStmt::succ_blocks() const {
  return SuccBlocks<SwitchTypeStmt>(valueSuccPairs_);
}
BasicBlock *SwitchTypeStmt::defaultSucc() const {
  CHECK(std::holds_alternative<Default>(valueSuccPairs_[0].first));
  return valueSuccPairs_[0].second;
}

SwitchRankStmt ::SwitchRankStmt(Value cond, const ValueSuccPairListType &args)
  : condition_{cond} {
  valueSuccPairs_.insert(valueSuccPairs_.end(), args.begin(), args.end());
}
std::list<BasicBlock *> SwitchRankStmt::succ_blocks() const {
  return SuccBlocks<SwitchRankStmt>(valueSuccPairs_);
}
BasicBlock *SwitchRankStmt::defaultSucc() const {
  CHECK(std::holds_alternative<Default>(valueSuccPairs_[0].first));
  return valueSuccPairs_[0].second;
}

// check LoadInsn constraints
static void CheckLoadInsn(const Value &v) {
  std::visit(
      common::visitors{
          [](DataObject *) { /* ok */ },
          [](Statement *s) { CHECK(GetAddressable(s)); },
          [](auto) { CHECK(!"invalid load input"); },
      },
      v.u);
}
LoadInsn::LoadInsn(const Value &addr) : address_{addr} {
  CheckLoadInsn(address_);
}
LoadInsn::LoadInsn(Value &&addr) : address_{std::move(addr)} {
  CheckLoadInsn(address_);
}
LoadInsn::LoadInsn(Statement *addr) : address_{addr} {
  CHECK(GetAddressable(addr));
}

// Store ctor
StoreInsn::StoreInsn(QualifiedStmt<Addressable_impl> addr, Value val)
  : address_{addr}, value_{val} {
  CHECK(address_);
  CHECK(!IsNothing(value_));
}

// dump is intended for debugging rather than idiomatic FIR output
std::string Statement::dump() const {
  return std::visit(
      common::visitors{
          [](const ReturnStmt &s) { return "return " + ToString(s.value()); },
          [](const BranchStmt &s) {
            if (s.hasCondition()) {
              return "cgoto (" + s.getCond().dump() + ") " +
                  ToString(s.getTrueSucc()) + ", " + ToString(s.getFalseSucc());
            }
            return "goto " + ToString(s.getTrueSucc());
          },
          [](const SwitchStmt &s) {
            return "switch (" + s.getCond().dump() + ")";
          },
          [](const SwitchCaseStmt &s) {
            return "switch-case (" + s.getCond().dump() + ")";
          },
          [](const SwitchTypeStmt &s) {
            return "switch-type (" + s.getCond().dump() + ")";
          },
          [](const SwitchRankStmt &s) {
            return "switch-rank (" + s.getCond().dump() + ")";
          },
          [](const IndirectBranchStmt &s) {
            std::string targets;
            for (auto *b : s.succ_blocks()) {
              targets += " " + ToString(b);
            }
            return "igoto (" + ToString(s.variable()) + ")" + targets;
          },
          [](const UnreachableStmt &) { return "unreachable"s; },
          [&](const ApplyExprStmt &e) {
            return '%' + ToString(&u) + ": eval " + FIR::dump(e.expression());
          },
          [&](const LocateExprStmt &e) {
            return '%' + ToString(&u) + ": addr-of " +
                FIR::dump(e.expression());
          },
          [](const AllocateInsn &) { return "alloc"s; },
          [](const DeallocateInsn &s) {
            return "dealloc (" + ToString(s.alloc()) + ")";
          },
          [&](const AllocateLocalInsn &insn) {
            return '%' + ToString(&u) + ": alloca " +
                FIR::dump(insn.variable());
          },
          [&](const LoadInsn &insn) {
            return '%' + ToString(&u) + ": load " + insn.address().dump();
          },
          [](const StoreInsn &insn) {
            std::string value{insn.value().dump()};
            return "store " + value + " to " +
                FIR::dump(insn.address()->address());
          },
          [](const DisassociateInsn &) { return "NULLIFY"s; },
          [&](const CallStmt &) { return '%' + ToString(&u) + ": call"s; },
          [](const RuntimeStmt &) { return "runtime-call()"s; },
          [](const IORuntimeStmt &) { return "io-call()"s; },
          [](const ScopeEnterStmt &) { return "scopeenter"s; },
          [](const ScopeExitStmt &) { return "scopeexit"s; },
          [](const PHIStmt &) { return "PHI"s; },
      },
      u);
}

std::string Value::dump() const {
  return std::visit(
      common::visitors{
          [](const Nothing &) { return "<none>"s; },
          [](const DataObject *obj) { return "obj_" + ToString(obj); },
          [](const Statement *s) { return "stmt_" + ToString(s); },
          [](const BasicBlock *bb) { return "block_" + ToString(bb); },
          [](const Procedure *p) { return "proc_" + ToString(p); },
      },
      u);
}
}
