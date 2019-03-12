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

#ifndef FORTRAN_FIR_STATEMENTS_H_
#define FORTRAN_FIR_STATEMENTS_H_

#include "basicblock.h"
#include <initializer_list>

namespace Fortran::FIR {

class ReturnStmt;
class BranchStmt;
class SwitchStmt;
class SwitchCaseStmt;
class SwitchTypeStmt;
class SwitchRankStmt;
class IndirectBranchStmt;
class UnreachableStmt;
class IncrementStmt;
class DoConditionStmt;
class ApplyExprStmt;
class LocateExprStmt;
class AllocateInsn;
class DeallocateInsn;
class AllocateLocalInsn;
class LoadInsn;
class StoreInsn;
class DisassociateInsn;
class CallStmt;
class RuntimeStmt;
class IORuntimeStmt;
class ScopeEnterStmt;
class ScopeExitStmt;
class PHIStmt;

class Statement;

CLASS_TRAIT(StatementTrait)
CLASS_TRAIT(TerminatorTrait)
CLASS_TRAIT(ActionTrait)

class Stmt_impl {
public:
  using StatementTrait = std::true_type;
};

// Every basic block must end in a terminator
class TerminatorStmt_impl : public Stmt_impl {
public:
  virtual std::list<BasicBlock *> succ_blocks() const = 0;
  virtual ~TerminatorStmt_impl() = default;
  using TerminatorTrait = std::true_type;
};

// Transfer control out of the current procedure
class ReturnStmt : public TerminatorStmt_impl {
public:
  static ReturnStmt Create(Statement *stmt) { return ReturnStmt{stmt}; }
  std::list<BasicBlock *> succ_blocks() const override { return {}; }
  Statement *returnValue() const;

private:
  ApplyExprStmt *returnValue_;
  explicit ReturnStmt(Statement *exp);
};

// Encodes two-way conditional branch and one-way absolute branch
class BranchStmt : public TerminatorStmt_impl {
public:
  static BranchStmt Create(
      Value condition, BasicBlock *trueBlock, BasicBlock *falseBlock) {
    return BranchStmt{condition, trueBlock, falseBlock};
  }
  static BranchStmt Create(BasicBlock *succ) {
    return BranchStmt{std::nullopt, succ, nullptr};
  }
  bool hasCondition() const { return condition_.has_value(); }
  Value getCond() const { return condition_.value(); }
  std::list<BasicBlock *> succ_blocks() const override {
    if (hasCondition()) {
      return {succs_[TrueIndex], succs_[FalseIndex]};
    }
    return {succs_[TrueIndex]};
  }
  BasicBlock *getTrueSucc() const { return succs_[TrueIndex]; }
  BasicBlock *getFalseSucc() const { return succs_[FalseIndex]; }

private:
  explicit BranchStmt(const std::optional<Value> &condition,
      BasicBlock *trueBlock, BasicBlock *falseBlock);
  static constexpr int TrueIndex{0};
  static constexpr int FalseIndex{1};
  std::optional<Value> condition_;
  BasicBlock *succs_[2];
};

// Switch on an expression into a set of constant values
class SwitchStmt : public TerminatorStmt_impl {
public:
  using ValueType = Value;
  using ValueSuccPairType = std::pair<ValueType, BasicBlock *>;
  using ValueSuccPairListType = std::vector<ValueSuccPairType>;
  static SwitchStmt Create(const Value &switchEval, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args) {
    return SwitchStmt{switchEval, defaultBlock, args};
  }
  BasicBlock *defaultSucc() const { return valueSuccPairs_[0].second; }
  std::list<BasicBlock *> succ_blocks() const override;
  Value getCond() const { return condition_; }

private:
  explicit SwitchStmt(const Value &condition, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args);

  Value condition_;
  ValueSuccPairListType valueSuccPairs_;
};

// Switch on an expression into a set of value (open or closed) ranges
class SwitchCaseStmt : public TerminatorStmt_impl {
public:
  struct Default {};
  struct Exactly {  // selector == v
    ApplyExprStmt *v;
  };
  struct InclusiveAbove {  // v <= selector
    ApplyExprStmt *v;
  };
  struct InclusiveBelow {  // selector <= v
    ApplyExprStmt *v;
  };
  struct InclusiveRange {  // lower <= selector <= upper
    ApplyExprStmt *lower;
    ApplyExprStmt *upper;
  };
  using RangeAlternative =
      std::variant<Exactly, InclusiveAbove, InclusiveBelow, InclusiveRange>;
  using ValueType = std::variant<Default, std::vector<RangeAlternative>>;
  using ValueSuccPairType = std::pair<ValueType, BasicBlock *>;
  using ValueSuccPairListType = std::vector<ValueSuccPairType>;

  static SwitchCaseStmt Create(Value switchEval, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args) {
    return SwitchCaseStmt{switchEval, defaultBlock, args};
  }
  BasicBlock *defaultSucc() const { return valueSuccPairs_[0].second; }
  std::list<BasicBlock *> succ_blocks() const override;
  Value getCond() const { return condition_; }

private:
  explicit SwitchCaseStmt(Value condition, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args);

  Value condition_;
  ValueSuccPairListType valueSuccPairs_;
};

// Switch on the TYPE of the selector into a set of TYPES, etc.
class SwitchTypeStmt : public TerminatorStmt_impl {
public:
  struct Default {};
  struct TypeSpec {
    Type v;
  };
  struct DerivedTypeSpec {
    Type v;
  };
  using ValueType = std::variant<Default, TypeSpec, DerivedTypeSpec>;
  using ValueSuccPairType = std::pair<ValueType, BasicBlock *>;
  using ValueSuccPairListType = std::vector<ValueSuccPairType>;
  static SwitchTypeStmt Create(Value switchEval, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args) {
    return SwitchTypeStmt{switchEval, defaultBlock, args};
  }
  BasicBlock *defaultSucc() const { return valueSuccPairs_[0].second; }
  std::list<BasicBlock *> succ_blocks() const override;
  Value getCond() const { return condition_; }

private:
  explicit SwitchTypeStmt(Value condition, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args);
  Value condition_;
  ValueSuccPairListType valueSuccPairs_;
};

// Switch on the RANK of the selector into a set of constant integers, etc.
class SwitchRankStmt : public TerminatorStmt_impl {
public:
  struct Default {};  // RANK DEFAULT
  struct AssumedSize {};  // RANK(*)
  struct Exactly {  // RANK(n)
    Expression *v;
  };
  using ValueType = std::variant<Exactly, AssumedSize, Default>;
  using ValueSuccPairType = std::pair<ValueType, BasicBlock *>;
  using ValueSuccPairListType = std::vector<ValueSuccPairType>;
  static SwitchRankStmt Create(Value switchEval, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args) {
    return SwitchRankStmt{switchEval, defaultBlock, args};
  }
  BasicBlock *defaultSucc() const { return valueSuccPairs_[0].second; }
  std::list<BasicBlock *> succ_blocks() const override;
  Value getCond() const { return condition_; }

private:
  explicit SwitchRankStmt(Value condition, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args);

  Value condition_;
  ValueSuccPairListType valueSuccPairs_;
};

class IndirectBranchStmt : public TerminatorStmt_impl {
public:
  using TargetListType = std::vector<BasicBlock *>;
  static IndirectBranchStmt Create(
      Variable *variable, const TargetListType &potentialTargets) {
    return IndirectBranchStmt{variable, potentialTargets};
  }

  Variable *variable() const { return variable_; }
  std::list<BasicBlock *> succ_blocks() const override {
    return {potentialTargets_.begin(), potentialTargets_.end()};
  }

private:
  explicit IndirectBranchStmt(
      Variable *variable, const TargetListType &potentialTargets)
    : variable_{variable}, potentialTargets_{potentialTargets} {}
  Variable *variable_;
  TargetListType potentialTargets_;
};

// This statement is not reachable
class UnreachableStmt : public TerminatorStmt_impl {
public:
  static UnreachableStmt Create() { return UnreachableStmt{}; }
  std::list<BasicBlock *> succ_blocks() const override { return {}; }

private:
  explicit UnreachableStmt() = default;
};

class ActionStmt_impl : public Stmt_impl {
public:
  using ActionTrait = std::true_type;

protected:
  ActionStmt_impl() : type{std::nullopt} {}

  // TODO: DynamicType is a placeholder for now
  std::optional<evaluate::DynamicType> type;
};

class IncrementStmt : public ActionStmt_impl {
public:
  static IncrementStmt Create(Value v1, Value v2) {
    return IncrementStmt(v1, v2);
  }
  Value leftValue() const { return value_[0]; }
  Value rightValue() const { return value_[1]; }

private:
  explicit IncrementStmt(Value v1, Value v2);
  Value value_[2];
};

class DoConditionStmt : public ActionStmt_impl {
public:
  static DoConditionStmt Create(Value dir, Value left, Value right) {
    return DoConditionStmt(dir, left, right);
  }
  Value direction() const { return value_[0]; }
  Value leftValue() const { return value_[1]; }
  Value rightValue() const { return value_[2]; }

private:
  explicit DoConditionStmt(Value dir, Value left, Value right);
  Value value_[3];
};

// Compute the value of an expression
class ApplyExprStmt : public ActionStmt_impl {
public:
  static ApplyExprStmt Create(const Expression *e) { return ApplyExprStmt{*e}; }
  static ApplyExprStmt Create(Expression &&e) { return ApplyExprStmt{e}; }

  const Expression &expression() const { return expression_; }

private:
  explicit ApplyExprStmt(const Expression &e) : expression_{e} {}

  Expression expression_;
};

// Base class of all addressable statements
class Addressable_impl : public ActionStmt_impl {
protected:
  Addressable_impl() : addrExpr_{std::nullopt} {}
  explicit Addressable_impl(const Expression &ae) : addrExpr_{ae} {}
  std::optional<Expression> addrExpr_;
};

// Compute the location of an expression
class LocateExprStmt : public Addressable_impl {
public:
  static LocateExprStmt Create(const Expression *e) {
    return LocateExprStmt(*e);
  }
  static LocateExprStmt Create(Expression &&e) { return LocateExprStmt(e); }

  const Expression &expression() const { return *addrExpr_; }

private:
  explicit LocateExprStmt(const Expression &e) : Addressable_impl{e} {}
};

// has memory effect
class MemoryStmt_impl : public ActionStmt_impl {
protected:
  MemoryStmt_impl() {}
};

// Allocate storage (per ALLOCATE)
class AllocateInsn : public Addressable_impl, public MemoryStmt_impl {
public:
  static AllocateInsn Create(Type type, int alignment = 0) {
    return AllocateInsn{type, alignment};
  }

  Type type() const { return type_; }
  int alignment() const { return alignment_; }

private:
  explicit AllocateInsn(Type type, int alignment)
    : type_{type}, alignment_{alignment} {}

  Type type_;
  int alignment_;
};

// Deallocate storage (per DEALLOCATE)
class DeallocateInsn : public MemoryStmt_impl {
public:
  static DeallocateInsn Create(AllocateInsn *alloc) {
    return DeallocateInsn{alloc};
  }

  AllocateInsn *alloc() { return alloc_; }

private:
  explicit DeallocateInsn(AllocateInsn *alloc) : alloc_{alloc} {}
  AllocateInsn *alloc_;
};

// Allocate space for a temporary by its Type. The lifetime of the temporary
// will not exceed that of the containing Procedure.
class AllocateLocalInsn : public Addressable_impl, public MemoryStmt_impl {
public:
  static AllocateLocalInsn Create(
      Type type, int alignment = 0, const Expression *expr = nullptr) {
    if (expr != nullptr) {
      return AllocateLocalInsn{type, alignment, *expr};
    }
    return AllocateLocalInsn{type, alignment};
  }

  Type type() const { return type_; }
  int alignment() const { return alignment_; }

private:
  explicit AllocateLocalInsn(Type type, int alignment, const Expression &expr)
    : Addressable_impl{expr}, type_{type}, alignment_{alignment} {}
  explicit AllocateLocalInsn(Type type, int alignment)
    : type_{type}, alignment_{alignment} {}

  Type type_;
  int alignment_;
};

// Load value(s) from a location
class LoadInsn : public MemoryStmt_impl {
public:
  static LoadInsn Create(Value addr) { return LoadInsn{addr}; }
  static LoadInsn Create(Statement *addr) { return LoadInsn{addr}; }

private:
  explicit LoadInsn(Value addr) : address_{addr} {}
  explicit LoadInsn(Statement *addr);
  std::variant<Addressable_impl *, Value> address_;
};

// Store value(s) from an applied expression to a location
class StoreInsn : public MemoryStmt_impl {
public:
  template<typename T> static StoreInsn Create(T *addr, T *value) {
    return StoreInsn{addr, value};
  }
  template<typename T> static StoreInsn Create(T *addr, BasicBlock *value) {
    return StoreInsn{addr, value};
  }

private:
  explicit StoreInsn(Value addr, Value val);
  explicit StoreInsn(Value addr, BasicBlock *val);
  explicit StoreInsn(Statement *addr, Statement *val);
  explicit StoreInsn(Statement *addr, BasicBlock *val);

  Addressable_impl *address_;
  std::variant<Value, ApplyExprStmt *, Addressable_impl *, BasicBlock *> value_;
};

// NULLIFY - make pointer object disassociated
class DisassociateInsn : public ActionStmt_impl {
public:
  static DisassociateInsn Create(Statement *s) { return DisassociateInsn{s}; }

  Statement *disassociate() { return disassociate_; }

private:
  DisassociateInsn(Statement *s) : disassociate_{s} {}
  Statement *disassociate_;
};

// base class for all call-like IR statements
class CallStmt_impl : public ActionStmt_impl {
public:
  Value Callee() const { return callee_; }
  unsigned NumArgs() const { return arguments_.size(); }

protected:
  CallStmt_impl(
      const FunctionType *functionType, Value callee, CallArguments &&arguments)
    : functionType_{functionType}, callee_{callee}, arguments_{arguments} {}

  const FunctionType *functionType_;
  Value callee_;
  CallArguments arguments_;
};

// CALL statements and function references
// A CallStmt has pass-by-value semantics. Pass-by-reference must be done
// explicitly by passing addresses of objects or temporaries.
class CallStmt : public CallStmt_impl {
public:
  static CallStmt Create(
      const FunctionType *type, Value callee, CallArguments &&arguments) {
    return CallStmt{type, callee, std::move(arguments)};
  }

private:
  explicit CallStmt(
      const FunctionType *functionType, Value callee, CallArguments &&arguments)
    : CallStmt_impl{functionType, callee, std::move(arguments)} {}
};

// Miscellaneous statements that turn into runtime calls
class RuntimeStmt : public CallStmt_impl {
public:
  static RuntimeStmt Create(
      RuntimeCallType call, RuntimeCallArguments &&argument) {
    return RuntimeStmt{call, std::move(argument)};
  }

  RuntimeCallType call() const { return call_; }

private:
  explicit RuntimeStmt(RuntimeCallType call, RuntimeCallArguments &&arguments)
    : CallStmt_impl{nullptr, Procedure::CreateIntrinsicProcedure(call),
          std::move(arguments)},
      call_{call} {}

  RuntimeCallType call_;
};

// The 13 Fortran I/O statements. Will be lowered to whatever becomes of the
// I/O runtime.
class IORuntimeStmt : public CallStmt_impl {
public:
  static IORuntimeStmt Create(
      InputOutputCallType call, IOCallArguments &&arguments) {
    return IORuntimeStmt{call, std::move(arguments)};
  }

  InputOutputCallType call() const { return call_; }

private:
  explicit IORuntimeStmt(InputOutputCallType call, IOCallArguments &&arguments)
    : CallStmt_impl{nullptr, Procedure::CreateIntrinsicProcedure(call),
          std::move(arguments)},
      call_{call} {}

  InputOutputCallType call_;
};

class ScopeStmt_impl : public ActionStmt_impl {
public:
  Scope *GetScope() const { return scope; }

protected:
  ScopeStmt_impl(Scope *scope) : scope{nullptr} {}
  Scope *scope;
};

// From the CFG document
class ScopeEnterStmt : public ScopeStmt_impl {
public:
  static ScopeEnterStmt Create(Scope *scope) { return ScopeEnterStmt{scope}; }

private:
  ScopeEnterStmt(Scope *scope) : ScopeStmt_impl{scope} {}
};

// From the CFG document
class ScopeExitStmt : public ScopeStmt_impl {
public:
  static ScopeExitStmt Create(Scope *scope) { return ScopeExitStmt{scope}; }

private:
  ScopeExitStmt(Scope *scope) : ScopeStmt_impl{scope} {}
};

// From the CFG document to support SSA
class PHIStmt : public ActionStmt_impl {
public:
  static PHIStmt Create(unsigned numReservedValues) {
    return PHIStmt{numReservedValues};
  }

private:
  PHIStmt(unsigned size) : inputs_(size) {}

  std::vector<PHIPair> inputs_;
};

// Sum type over all statement classes
class Statement : public SumTypeMixin<ReturnStmt,  //
                      BranchStmt,  //
                      SwitchStmt,  //
                      SwitchCaseStmt,  //
                      SwitchTypeStmt,  //
                      SwitchRankStmt,  //
                      IndirectBranchStmt,  //
                      UnreachableStmt,  //
                      IncrementStmt,  //
                      DoConditionStmt,  //
                      ApplyExprStmt,  //
                      LocateExprStmt,  //
                      AllocateInsn,  //
                      DeallocateInsn,  //
                      AllocateLocalInsn,  //
                      LoadInsn,  //
                      StoreInsn,  //
                      DisassociateInsn,  //
                      CallStmt,  //
                      RuntimeStmt,  //
                      IORuntimeStmt,  //
                      ScopeEnterStmt,  //
                      ScopeExitStmt,  //
                      PHIStmt  //
                      >,
                  public Value_impl,
                  public ChildMixin<Statement, BasicBlock>,
                  public llvm::ilist_node<Statement> {
public:
  template<typename A>
  Statement(BasicBlock *p, A &&t) : SumTypeMixin{t}, ChildMixin{p} {
    parent->insertBefore(this);
  }
  std::string dump() const;

  static constexpr std::size_t offsetof_impl() {
    Statement *s{nullptr};
    return reinterpret_cast<char *>(&s->u) - reinterpret_cast<char *>(s);
  }
  static Statement *From(Stmt_impl *stmt) {
    return reinterpret_cast<Statement *>(
        reinterpret_cast<char *>(stmt) - Statement::offsetof_impl());
  }
};

inline std::list<BasicBlock *> succ_list(BasicBlock &block) {
  if (auto *terminator{block.terminator()}) {
    return reinterpret_cast<const TerminatorStmt_impl *>(&terminator->u)
        ->succ_blocks();
  }
  // CHECK(false && "block does not have terminator");
  return {};
}

inline Statement *ReturnStmt::returnValue() const {
  return Statement::From(returnValue_);
}

inline ApplyExprStmt *GetApplyExpr(Statement *stmt) {
  return std::visit(
      common::visitors{
          [](ApplyExprStmt &s) { return &s; },
          [](auto &) -> ApplyExprStmt * { return nullptr; },
      },
      stmt->u);
}

Addressable_impl *GetAddressable(Statement *stmt);
}

#endif  // FORTRAN_FIR_STATEMENTS_H_
