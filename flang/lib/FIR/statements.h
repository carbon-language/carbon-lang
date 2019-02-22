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

#include "common.h"
#include "mixin.h"
#include <initializer_list>
#include <ostream>

namespace Fortran::FIR {

#define HANDLE_STMT(num, opcode, name) struct name;
#include "statement.def"

struct Statement;

CLASS_TRAIT(StatementTrait)
CLASS_TRAIT(TerminatorTrait)
CLASS_TRAIT(ActionTrait)

struct Evaluation
  : public SumTypeCopyMixin<std::variant<Expression *, Variable *,
        PathVariable *, const semantics::Symbol *>> {
  SUM_TYPE_COPY_MIXIN(Evaluation)
  Evaluation(PathVariable *pv) : SumTypeCopyMixin{pv} {
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

struct Stmt_impl {
  using StatementTrait = std::true_type;
};

struct TerminatorStmt_impl : public Stmt_impl {
  virtual std::list<BasicBlock *> succ_blocks() const { return {}; }
  using TerminatorTrait = std::true_type;
};

struct ReturnStmt : public TerminatorStmt_impl {
  static ReturnStmt Create() { return ReturnStmt{nullptr}; }
  static ReturnStmt Create(Expression *expression) {
    return ReturnStmt{expression};
  }

private:
  Expression *returnValue_;
  explicit ReturnStmt(Expression *);
};

struct BranchStmt : public TerminatorStmt_impl {
  static BranchStmt Create(
      Expression *condition, BasicBlock *trueBlock, BasicBlock *falseBlock) {
    return BranchStmt{condition, trueBlock, falseBlock};
  }
  static BranchStmt Create(BasicBlock *succ) {
    return BranchStmt{nullptr, succ, nullptr};
  }
  bool hasCondition() const { return condition_ != nullptr; }
  Expression *getCond() const { return condition_; }
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
      Expression *condition, BasicBlock *trueBlock, BasicBlock *falseBlock);
  static constexpr unsigned TrueIndex{0u};
  static constexpr unsigned FalseIndex{1u};
  Expression *condition_;
  BasicBlock *succs_[2];
};

/// Switch on an expression into a set of constant values
struct SwitchStmt : public TerminatorStmt_impl {
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

/// Switch on an expression into a set of value (open or closed) ranges
struct SwitchCaseStmt : public TerminatorStmt_impl {
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

using Type = const semantics::DeclTypeSpec *;  // FIXME
/// Switch on the TYPE of the selector into a set of TYPES, etc.
struct SwitchTypeStmt : public TerminatorStmt_impl {
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

/// Switch on the RANK of the selector into a set of constant integers, etc.
struct SwitchRankStmt : public TerminatorStmt_impl {
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

struct IndirectBrStmt : public TerminatorStmt_impl {
  using TargetListType = std::vector<BasicBlock *>;
  static IndirectBrStmt Create(
      Variable *variable, const TargetListType &potentialTargets) {
    return IndirectBrStmt{variable, potentialTargets};
  }

private:
  explicit IndirectBrStmt(
      Variable *variable, const TargetListType &potentialTargets)
    : variable_{variable}, potentialTargets_{potentialTargets} {}
  Variable *variable_;
  TargetListType potentialTargets_;
};

struct UnreachableStmt : public TerminatorStmt_impl {
  static UnreachableStmt Create() { return UnreachableStmt{}; }

private:
  explicit UnreachableStmt() = default;
};

struct ActionStmt_impl : public Stmt_impl {
  using ActionTrait = std::true_type;

protected:
  ActionStmt_impl() : type{std::nullopt} {}

  // TODO: DynamicType is a placeholder for now
  std::optional<evaluate::DynamicType> type;
};

struct AssignmentStmt : public ActionStmt_impl {
  static AssignmentStmt Create(const PathVariable *lhs, const Expression *rhs) {
    return AssignmentStmt{lhs, rhs};
  }
  const PathVariable *GetLeftHandSide() const { return lhs_; }
  const Expression *GetRightHandSide() const { return rhs_; }

private:
  explicit AssignmentStmt(const PathVariable *lhs, const Expression *rhs)
    : lhs_{lhs}, rhs_{rhs} {}

  const PathVariable *lhs_;
  const Expression *rhs_;
};

struct PointerAssignStmt : public ActionStmt_impl {
  static PointerAssignStmt Create(
      const Expression *lhs, const Expression *rhs) {
    return PointerAssignStmt{lhs, rhs};
  }
  const Expression *GetLeftHandSide() const { return lhs_; }
  const Expression *GetRightHandSide() const { return rhs_; }

private:
  explicit PointerAssignStmt(const Expression *lhs, const Expression *rhs)
    : lhs_{lhs}, rhs_{rhs} {}
  const parser::PointerAssignmentStmt *assign_;
  const Expression *lhs_;
  const Expression *rhs_;
};

struct LabelAssignStmt : public ActionStmt_impl {
  static LabelAssignStmt Create(const semantics::Symbol *lhs, BasicBlock *rhs) {
    return LabelAssignStmt{lhs, rhs};
  }

private:
  explicit LabelAssignStmt(const semantics::Symbol *lhs, BasicBlock *rhs)
    : lhs_{lhs}, rhs_{rhs} {}

  const semantics::Symbol *lhs_;
  BasicBlock *rhs_;
};

struct MemoryStmt_impl : public ActionStmt_impl {
  // FIXME: ought to use a Type, let backend compute size...
protected:
  MemoryStmt_impl() {}
};

/// ALLOCATE allocate space for a pointer target or allocatable and populate the
/// reference
struct AllocateStmt : public MemoryStmt_impl {
  static AllocateStmt Create(const Expression *object) {
    return AllocateStmt{object};
  }

private:
  explicit AllocateStmt(const Expression *object) : object_{object} {}
  const Expression *object_;  // POINTER|ALLOCATABLE to be allocated
  // TODO: maybe add other arguments
};

/// DEALLOCATE deallocate allocatable variables and pointer targets. pointers
/// become disassociated
struct DeallocateStmt : public MemoryStmt_impl {
  static DeallocateStmt Create(const Expression *object) {
    return DeallocateStmt{object};
  }

private:
  explicit DeallocateStmt(const Expression *object) : object_{object} {}
  const Expression *object_;  // POINTER|ALLOCATABLE to be deallocated
  // TODO: maybe add other arguments
};

struct AllocLocalInsn : public MemoryStmt_impl {
  static AllocLocalInsn Create(Type type, unsigned alignment = 0u) {
    return AllocLocalInsn{type, alignment};
  }

private:
  explicit AllocLocalInsn(Type type, unsigned alignment)
    : alignment_{alignment} {}
  unsigned alignment_;
};

struct LoadInsn : public MemoryStmt_impl {
  static LoadInsn Create(const Expression *address) {
    return LoadInsn{address};
  }

private:
  explicit LoadInsn(const Expression *address) : address_{address} {}
  const Expression *address_;
};

struct StoreInsn : public MemoryStmt_impl {
  static StoreInsn Create(const Expression *address, const Expression *value) {
    return StoreInsn{address, value};
  }

private:
  explicit StoreInsn(const Expression *address, const Expression *value)
    : address_{address}, value_{value} {}
  const Expression *address_;
  const Expression *value_;
};

/// NULLIFY make pointer object disassociated
struct DisassociateStmt : public ActionStmt_impl {
  static DisassociateStmt Create(const parser::NullifyStmt *n) {
    return DisassociateStmt{n};
  }

private:
  DisassociateStmt(const parser::NullifyStmt *n) : disassociate_{n} {}
  const parser::NullifyStmt *disassociate_;
};

/// expressions that must be evaluated in various statements
struct ExprStmt
  : public ActionStmt_impl,
    public SumTypeCopyMixin<std::variant<const parser::AssociateStmt *,
        const parser::ChangeTeamStmt *, const parser::NonLabelDoStmt *,
        const parser::ForallConstructStmt *, const Expression *>> {
  template<typename T> static ExprStmt Create(const T *e) {
    return ExprStmt{e};
  }

private:
  template<typename T> explicit ExprStmt(const T *e) : SumTypeCopyMixin{e} {}
  // Evaluation evaluation_;
};

/// base class for all call-like IR statements
struct CallStmt_impl : public ActionStmt_impl {
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

/// CALL statements and function references
struct CallStmt : public CallStmt_impl {
  static CallStmt Create(const FunctionType *type, const Value *callee,
      CallArguments &&arguments) {
    return CallStmt{type, callee, std::move(arguments)};
  }

private:
  explicit CallStmt(const FunctionType *functionType, const Value *callee,
      CallArguments &&arguments)
    : CallStmt_impl{functionType, callee, std::move(arguments)} {}
};

/// Miscellaneous statements that turn into runtime calls
struct RuntimeStmt : public CallStmt_impl {
  static RuntimeStmt Create(
      RuntimeCallType call, RuntimeCallArguments &&argument) {
    return RuntimeStmt{call, std::move(argument)};
  }

private:
  explicit RuntimeStmt(RuntimeCallType call, RuntimeCallArguments &&arguments)
    : CallStmt_impl{nullptr, nullptr, std::move(arguments)}, call_{call} {}

  RuntimeCallType call_;
};

/// The 13 Fortran I/O statements. Will be lowered to whatever becomes of the
/// I/O runtime.
struct IORuntimeStmt : public CallStmt_impl {
  static IORuntimeStmt Create(
      InputOutputCallType call, IOCallArguments &&arguments) {
    return IORuntimeStmt{call, std::move(arguments)};
  }

private:
  explicit IORuntimeStmt(InputOutputCallType call, IOCallArguments &&arguments)
    : CallStmt_impl{nullptr, nullptr, std::move(arguments)}, call_{call} {}

  InputOutputCallType call_;
};

struct ScopeStmt_impl : public ActionStmt_impl {
  Scope *GetScope() const { return scope; }

protected:
  ScopeStmt_impl(Scope *scope) : scope{nullptr} {}
  Scope *scope;
};

/// From the CFG document
struct ScopeEnterStmt : public ScopeStmt_impl {
  static ScopeEnterStmt Create(Scope *scope) { return ScopeEnterStmt{scope}; }

private:
  ScopeEnterStmt(Scope *scope) : ScopeStmt_impl{scope} {}
};

/// From the CFG document
struct ScopeExitStmt : public ScopeStmt_impl {
  static ScopeExitStmt Create(Scope *scope) { return ScopeExitStmt{scope}; }

private:
  ScopeExitStmt(Scope *scope) : ScopeStmt_impl{scope} {}
};

/// From the CFG document to support SSA
struct PHIStmt : public ActionStmt_impl {
  static PHIStmt Create(unsigned numReservedValues) {
    return PHIStmt{numReservedValues};
  }

private:
  PHIStmt(unsigned size) : inputs_(size) {}

  std::vector<PHIPair> inputs_;
};

}

#endif
