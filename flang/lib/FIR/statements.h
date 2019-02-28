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
class AssignmentStmt;
class PointerAssignStmt;
class LabelAssignStmt;
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

class TerminatorStmt_impl : public Stmt_impl {
public:
  virtual std::list<BasicBlock *> succ_blocks() const { return {}; }
  using TerminatorTrait = std::true_type;
};

class ReturnStmt : public TerminatorStmt_impl {
public:
  static ReturnStmt Create() { return ReturnStmt{nullptr}; }
  static ReturnStmt Create(Expression *expression) {
    return ReturnStmt{expression};
  }

private:
  Expression *returnValue_;
  explicit ReturnStmt(Expression *);
};

class BranchStmt : public TerminatorStmt_impl {
public:
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

/// Switch on an expression into a set of value (open or closed) ranges
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

/// Switch on the TYPE of the selector into a set of TYPES, etc.
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

/// Switch on the RANK of the selector into a set of constant integers, etc.
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

/// This statement is not reachable
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

class AssignmentStmt : public ActionStmt_impl {
public:
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

class PointerAssignStmt : public ActionStmt_impl {
public:
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

class LabelAssignStmt : public ActionStmt_impl {
public:
  static LabelAssignStmt Create(const semantics::Symbol *lhs, BasicBlock *rhs) {
    return LabelAssignStmt{lhs, rhs};
  }

private:
  explicit LabelAssignStmt(const semantics::Symbol *lhs, BasicBlock *rhs)
    : lhs_{lhs}, rhs_{rhs} {}

  const semantics::Symbol *lhs_;
  BasicBlock *rhs_;
};

/// Compute the value of an expression
class ApplyExprStmt
  : public ActionStmt_impl,
    public SumTypeCopyMixin<std::variant<const parser::AssociateStmt *,
        const parser::ChangeTeamStmt *, const parser::NonLabelDoStmt *,
        const parser::ForallConstructStmt *, const Expression *>> {
public:
  template<typename T> static ApplyExprStmt Create(const T *e) {
    return ApplyExprStmt{e};
  }

private:
  template<typename T>
  explicit ApplyExprStmt(const T *e) : SumTypeCopyMixin{e} {}
  // Evaluation evaluation_;
};

/// Compute the location of an expression
class LocateExprStmt : public ActionStmt_impl {
public:
  static LocateExprStmt *Create(const Expression *e) {
    return new LocateExprStmt(e);
  }

private:
  explicit LocateExprStmt(const Expression *e) : expression_{e} {}
  const Expression *expression_;
};

class MemoryStmt_impl : public ActionStmt_impl {
public:
  // FIXME: ought to use a Type, let backend compute size...
protected:
  MemoryStmt_impl() {}
};

/// Allocate storage on the heap
class AllocateInsn : public MemoryStmt_impl {
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

/// Deallocate storage on the heap
class DeallocateInsn : public MemoryStmt_impl {
public:
  static DeallocateInsn Create(const AllocateInsn *alloc) {
    return DeallocateInsn{alloc};
  }

private:
  explicit DeallocateInsn(const AllocateInsn *alloc) : alloc_{alloc} {}
  const AllocateInsn *alloc_;
};

/// Allocate space for a varible by its Type. This storage's lifetime will not
/// exceed that of the containing Procedure.
class AllocateLocalInsn : public MemoryStmt_impl {
public:
  static AllocateLocalInsn Create(Type type, int alignment = 0) {
    return AllocateLocalInsn{type, alignment};
  }

private:
  explicit AllocateLocalInsn(Type type, int alignment)
    : type_{type}, alignment_{alignment} {}
  Type type_;
  int alignment_;
};

/// Load value(s) from a location
class LoadInsn : public MemoryStmt_impl {
public:
  static LoadInsn Create(Value *address) { return LoadInsn{address}; }

private:
  explicit LoadInsn(Value *address) : address_{address} {}
  Value *address_;
};

/// Store value(s) from an applied expression to a location
class StoreInsn : public MemoryStmt_impl {
public:
  static StoreInsn Create(Value *address, const Expression *value) {
    return StoreInsn{address, value};
  }

private:
  explicit StoreInsn(Value *address, const Expression *value)
    : address_{address}, value_{value} {}
  Value *address_;
  const Expression *value_;
};

/// NULLIFY - make pointer object disassociated
class DisassociateInsn : public ActionStmt_impl {
public:
  static DisassociateInsn Create(const parser::NullifyStmt *n) {
    return DisassociateInsn{n};
  }

private:
  DisassociateInsn(const parser::NullifyStmt *n) : disassociate_{n} {}
  const parser::NullifyStmt *disassociate_;
};

/// base class for all call-like IR statements
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

/// CALL statements and function references
/// A CallStmt has pass-by-value semantics. Pass-by-reference must be done
/// explicitly by passing addresses of objects or temporaries.
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

/// Miscellaneous statements that turn into runtime calls
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

/// The 13 Fortran I/O statements. Will be lowered to whatever becomes of the
/// I/O runtime.
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

/// From the CFG document
class ScopeEnterStmt : public ScopeStmt_impl {
public:
  static ScopeEnterStmt Create(Scope *scope) { return ScopeEnterStmt{scope}; }

private:
  ScopeEnterStmt(Scope *scope) : ScopeStmt_impl{scope} {}
};

/// From the CFG document
class ScopeExitStmt : public ScopeStmt_impl {
public:
  static ScopeExitStmt Create(Scope *scope) { return ScopeExitStmt{scope}; }

private:
  ScopeExitStmt(Scope *scope) : ScopeStmt_impl{scope} {}
};

/// From the CFG document to support SSA
class PHIStmt : public ActionStmt_impl {
public:
  static PHIStmt Create(unsigned numReservedValues) {
    return PHIStmt{numReservedValues};
  }

private:
  PHIStmt(unsigned size) : inputs_(size) {}

  std::vector<PHIPair> inputs_;
};

/// Sum type over all statement classes
class Statement : public SumTypeMixin<std::variant<ReturnStmt,  //
                      BranchStmt,  //
                      SwitchStmt,  //
                      SwitchCaseStmt,  //
                      SwitchTypeStmt,  //
                      SwitchRankStmt,  //
                      IndirectBranchStmt,  //
                      UnreachableStmt,  //
                      AssignmentStmt,  //
                      PointerAssignStmt,  //
                      LabelAssignStmt,  //
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
  if (auto *terminator{block.getTerminator()}) {
    return reinterpret_cast<const TerminatorStmt_impl *>(&terminator->u)
        ->succ_blocks();
  }
  // CHECK(false && "block does not have terminator");
  return {};
}

}

#endif  // FORTRAN_FIR_STATEMENTS_H_
