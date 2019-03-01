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
#include "common.h"
#include "mixin.h"
#include <initializer_list>
#include <ostream>

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

class Evaluation : public SumTypeCopyMixin<std::variant<Expression *,
                       Variable *, PathVariable *, const semantics::Symbol *>> {
public:
  SUM_TYPE_COPY_MIXIN(Evaluation)
  Evaluation(PathVariable *pv) : SumTypeCopyMixin(pv) {
    if (const auto *designator{
            std::get_if<common::Indirection<parser::Designator>>(&pv->u)}) {
      if (const auto *obj{std::get_if<parser::ObjectName>(&(*designator)->u)}) {
        u = obj->symbol;
      }
    }
  }
  template<typename A> Evaluation(A *a) : SumTypeCopyMixin{a} {}
  std::string dump() const;
};

class Stmt_impl {
public:
  using StatementTrait = std::true_type;
};

// Every basic block must end in a terminator
class TerminatorStmt_impl : public Stmt_impl {
public:
  virtual std::list<BasicBlock *> succ_blocks() const { return {}; }
  using TerminatorTrait = std::true_type;
};

// Transfer control out of the current procedure
class ReturnStmt : public TerminatorStmt_impl {
public:
  static ReturnStmt Create() { return ReturnStmt{}; }
  static ReturnStmt Create(Expression *e) { return ReturnStmt{*e}; }
  static ReturnStmt Create(Expression &&e) { return ReturnStmt{std::move(e)}; }

  bool IsVoid() const { return !returnValue_.has_value(); }

private:
  std::optional<Expression> returnValue_;
  explicit ReturnStmt() : returnValue_{std::nullopt} {}
  explicit ReturnStmt(const Expression &e) : returnValue_{e} {}
  explicit ReturnStmt(Expression &&e) : returnValue_{e} {}
};

// Encodes two-way conditional branch and one-way absolute branch
class BranchStmt : public TerminatorStmt_impl {
public:
  static BranchStmt Create(
      Statement *condition, BasicBlock *trueBlock, BasicBlock *falseBlock) {
    return BranchStmt{condition, trueBlock, falseBlock};
  }
  static BranchStmt Create(BasicBlock *succ) {
    return BranchStmt{nullptr, succ, nullptr};
  }
  bool hasCondition() const { return condition_ != nullptr; }
  Statement *getCond() const { return condition_; }
  std::list<BasicBlock *> succ_blocks() const override {
    if (hasCondition()) {
      return {succs_[TrueIndex], succs_[FalseIndex]};
    }
    return {succs_[TrueIndex]};
  }
  BasicBlock *getTrueSucc() const { return succs_[TrueIndex]; }
  BasicBlock *getFalseSucc() const { return succs_[FalseIndex]; }

private:
  explicit BranchStmt(
      Statement *condition, BasicBlock *trueBlock, BasicBlock *falseBlock);
  static constexpr int TrueIndex{0};
  static constexpr int FalseIndex{1};
  Statement *condition_;
  BasicBlock *succs_[2];
};

// Switch on an expression into a set of constant values
class SwitchStmt : public TerminatorStmt_impl {
public:
  using ValueType = Expression *;
  using ValueSuccPairType = std::pair<ValueType, BasicBlock *>;
  using ValueSuccPairListType = std::vector<ValueSuccPairType>;
  static SwitchStmt Create(const Evaluation &switchEval,
      BasicBlock *defaultBlock, const ValueSuccPairListType &args) {
    return SwitchStmt{switchEval, defaultBlock, args};
  }
  BasicBlock *defaultSucc() const { return valueSuccPairs_[0].second; }
  std::list<BasicBlock *> succ_blocks() const override;
  const Evaluation &getCond() const { return condition_; }

private:
  explicit SwitchStmt(const Evaluation &condition, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args);
  Evaluation condition_;
  ValueSuccPairListType valueSuccPairs_;
};

// Switch on an expression into a set of value (open or closed) ranges
class SwitchCaseStmt : public TerminatorStmt_impl {
public:
  struct Default {};
  struct Exactly {  // selector == v
    Expression *v;
  };
  struct InclusiveAbove {  // v <= selector
    Expression *v;
  };
  struct InclusiveBelow {  // selector <= v
    Expression *v;
  };
  struct InclusiveRange {  // lower <= selector <= upper
    Expression *lower;
    Expression *upper;
  };
  using RangeAlternative =
      std::variant<Exactly, InclusiveAbove, InclusiveBelow, InclusiveRange>;
  using ValueType = std::variant<Default, std::vector<RangeAlternative>>;
  using ValueSuccPairType = std::pair<ValueType, BasicBlock *>;
  using ValueSuccPairListType = std::vector<ValueSuccPairType>;

  static SwitchCaseStmt Create(const Evaluation &switchEval,
      BasicBlock *defaultBlock, const ValueSuccPairListType &args) {
    return SwitchCaseStmt{switchEval, defaultBlock, args};
  }
  BasicBlock *defaultSucc() const { return valueSuccPairs_[0].second; }
  std::list<BasicBlock *> succ_blocks() const override;
  const Evaluation &getCond() const { return condition_; }

private:
  explicit SwitchCaseStmt(const Evaluation &condition, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args);
  Evaluation condition_;
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
  static SwitchTypeStmt Create(const Evaluation &switchEval,
      BasicBlock *defaultBlock, const ValueSuccPairListType &args) {
    return SwitchTypeStmt{switchEval, defaultBlock, args};
  }
  BasicBlock *defaultSucc() const { return valueSuccPairs_[0].second; }
  std::list<BasicBlock *> succ_blocks() const override;
  const Evaluation &getCond() const { return condition_; }

private:
  explicit SwitchTypeStmt(const Evaluation &condition, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args);
  Evaluation condition_;
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
  static SwitchRankStmt Create(const Evaluation &switchEval,
      BasicBlock *defaultBlock, const ValueSuccPairListType &args) {
    return SwitchRankStmt{switchEval, defaultBlock, args};
  }
  BasicBlock *defaultSucc() const { return valueSuccPairs_[0].second; }
  std::list<BasicBlock *> succ_blocks() const override;
  const Evaluation &getCond() const { return condition_; }

private:
  explicit SwitchRankStmt(const Evaluation &condition, BasicBlock *defaultBlock,
      const ValueSuccPairListType &args);
  Evaluation condition_;
  ValueSuccPairListType valueSuccPairs_;
};

class IndirectBranchStmt : public TerminatorStmt_impl {
public:
  using TargetListType = std::vector<BasicBlock *>;
  static IndirectBranchStmt Create(
      Variable *variable, const TargetListType &potentialTargets) {
    return IndirectBranchStmt{variable, potentialTargets};
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
  static IncrementStmt Create(Statement *v1, Statement *v2) {
    return IncrementStmt(v1, v2);
  }
  Statement *leftValue() const { return value_[0]; }
  Statement *rightValue() const { return value_[1]; }

private:
  explicit IncrementStmt(Statement *v1, Statement *v2);
  Statement *value_[2];
};

class DoConditionStmt : public ActionStmt_impl {
public:
  static DoConditionStmt Create(
      Statement *dir, Statement *left, Statement *right) {
    return DoConditionStmt(dir, left, right);
  }
  Statement *direction() const { return value_[0]; }
  Statement *leftValue() const { return value_[1]; }
  Statement *rightValue() const { return value_[2]; }

private:
  explicit DoConditionStmt(Statement *dir, Statement *left, Statement *right);
  Statement *value_[3];
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

private:
  explicit AllocateInsn(Type type, int alignment)
    : type_{type}, alignment_{alignment} {}

  Type type_;
  int alignment_;
};

// Deallocate storage (per DEALLOCATE)
class DeallocateInsn : public MemoryStmt_impl {
public:
  static DeallocateInsn Create(const AllocateInsn *alloc) {
    return DeallocateInsn{alloc};
  }

private:
  explicit DeallocateInsn(const AllocateInsn *alloc) : alloc_{alloc} {}
  const AllocateInsn *alloc_;
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
  static LoadInsn Create(Statement *addr) { return LoadInsn{addr}; }

private:
  explicit LoadInsn(Statement *addr);
  Addressable_impl *address_;
};

// Store value(s) from an applied expression to a location
class StoreInsn : public MemoryStmt_impl {
public:
  static StoreInsn Create(Statement *addr, Statement *value) {
    return StoreInsn{addr, value};
  }
  static StoreInsn Create(Statement *addr, BasicBlock *value) {
    return StoreInsn{addr, value};
  }

private:
  explicit StoreInsn(Statement *addr, Statement *val);
  explicit StoreInsn(Statement *addr, BasicBlock *val);

  Addressable_impl *address_;
  std::variant<ApplyExprStmt *, Addressable_impl *, BasicBlock *> value_;
};

// NULLIFY - make pointer object disassociated
class DisassociateInsn : public ActionStmt_impl {
public:
  static DisassociateInsn Create(const parser::NullifyStmt *n) {
    return DisassociateInsn{n};
  }

private:
  DisassociateInsn(const parser::NullifyStmt *n) : disassociate_{n} {}
  const parser::NullifyStmt *disassociate_;
};

// base class for all call-like IR statements
class CallStmt_impl : public ActionStmt_impl {
public:
  const Value *Callee() const { return callee_; }
  unsigned NumArgs() const { return arguments_.size(); }

protected:
  CallStmt_impl(const FunctionType *functionType, const Value *callee,
      CallArguments &&arguments)
    : functionType_{functionType}, callee_{callee}, arguments_{arguments} {}

  const FunctionType *functionType_;
  const Value *callee_;
  CallArguments arguments_;
};

// CALL statements and function references
// A CallStmt has pass-by-value semantics. Pass-by-reference must be done
// explicitly by passing addresses of objects or temporaries.
class CallStmt : public CallStmt_impl {
public:
  static CallStmt Create(const FunctionType *type, const Value *callee,
      CallArguments &&arguments) {
    return CallStmt{type, callee, std::move(arguments)};
  }

private:
  explicit CallStmt(const FunctionType *functionType, const Value *callee,
      CallArguments &&arguments)
    : CallStmt_impl{functionType, callee, std::move(arguments)} {}
};

// Miscellaneous statements that turn into runtime calls
class RuntimeStmt : public CallStmt_impl {
public:
  static RuntimeStmt Create(
      RuntimeCallType call, RuntimeCallArguments &&argument) {
    return RuntimeStmt{call, std::move(argument)};
  }

private:
  explicit RuntimeStmt(RuntimeCallType call, RuntimeCallArguments &&arguments)
    : CallStmt_impl{nullptr, nullptr, std::move(arguments)}, call_{call} {}

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

private:
  explicit IORuntimeStmt(InputOutputCallType call, IOCallArguments &&arguments)
    : CallStmt_impl{nullptr, nullptr, std::move(arguments)}, call_{call} {}

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
class Statement : public SumTypeMixin<std::variant<ReturnStmt,  //
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
                      >>,
                  public ChildMixin<Statement, BasicBlock>,
                  public llvm::ilist_node<Statement> {
public:
  template<typename A>
  Statement(BasicBlock *p, A &&t) : SumTypeMixin{t}, ChildMixin{p} {
    parent->insertBefore(this);
  }
  std::string dump() const;
};

inline std::list<BasicBlock *> succ_list(BasicBlock &block) {
  if (auto *terminator{block.terminator()}) {
    return reinterpret_cast<const TerminatorStmt_impl *>(&terminator->u)
        ->succ_blocks();
  }
  // CHECK(false && "block does not have terminator");
  return {};
}

}

#endif  // FORTRAN_FIR_STATEMENTS_H_
