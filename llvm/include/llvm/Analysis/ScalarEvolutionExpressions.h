//===- llvm/Analysis/ScalarEvolutionExpressions.h - SCEV Exprs --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the classes used to represent and build scalar expressions.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTION_EXPRESSIONS_H
#define LLVM_ANALYSIS_SCALAREVOLUTION_EXPRESSIONS_H

#include "llvm/Analysis/ScalarEvolution.h"

namespace llvm {
  class ConstantInt;
  class ConstantRange;

  enum SCEVTypes {
    // These should be ordered in terms of increasing complexity to make the
    // folders simpler.
    scConstant, scTruncate, scZeroExtend, scAddExpr, scMulExpr, scUDivExpr,
    scAddRecExpr, scUnknown, scCouldNotCompute
  };

  //===--------------------------------------------------------------------===//
  /// SCEVConstant - This class represents a constant integer value.
  ///
  class SCEVConstant : public SCEV {
    ConstantInt *V;
    SCEVConstant(ConstantInt *v) : SCEV(scConstant), V(v) {}
    
    virtual ~SCEVConstant();
  public:
    /// get method - This just gets and returns a new SCEVConstant object.
    ///
    static SCEVHandle get(ConstantInt *V);

    ConstantInt *getValue() const { return V; }

    /// getValueRange - Return the tightest constant bounds that this value is
    /// known to have.  This method is only valid on integer SCEV objects.
    virtual ConstantRange getValueRange() const;

    virtual bool isLoopInvariant(const Loop *L) const {
      return true;
    }

    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      return false;  // Not loop variant
    }

    virtual const Type *getType() const;

    virtual void print(std::ostream &OS) const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVConstant *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scConstant;
    }
  };

  //===--------------------------------------------------------------------===//
  /// SCEVTruncateExpr - This class represents a truncation of an integer value
  /// to a smaller integer value.
  ///
  class SCEVTruncateExpr : public SCEV {
    SCEVHandle Op;
    const Type *Ty;
    SCEVTruncateExpr(const SCEVHandle &op, const Type *ty);
    virtual ~SCEVTruncateExpr();
  public:
    /// get method - This just gets and returns a new SCEVTruncate object
    ///
    static SCEVHandle get(const SCEVHandle &Op, const Type *Ty);

    const SCEVHandle &getOperand() const { return Op; }
    virtual const Type *getType() const { return Ty; }
    
    virtual bool isLoopInvariant(const Loop *L) const {
      return Op->isLoopInvariant(L);
    }

    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      return Op->hasComputableLoopEvolution(L);
    }

    /// getValueRange - Return the tightest constant bounds that this value is
    /// known to have.  This method is only valid on integer SCEV objects.
    virtual ConstantRange getValueRange() const;

    virtual void print(std::ostream &OS) const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVTruncateExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scTruncate;
    }
  };

  //===--------------------------------------------------------------------===//
  /// SCEVZeroExtendExpr - This class represents a zero extension of a small
  /// integer value to a larger integer value.
  ///
  class SCEVZeroExtendExpr : public SCEV {
    SCEVHandle Op;
    const Type *Ty;
    SCEVZeroExtendExpr(const SCEVHandle &op, const Type *ty);
    virtual ~SCEVZeroExtendExpr();
  public:
    /// get method - This just gets and returns a new SCEVZeroExtend object
    ///
    static SCEVHandle get(const SCEVHandle &Op, const Type *Ty);

    const SCEVHandle &getOperand() const { return Op; }
    virtual const Type *getType() const { return Ty; }
    
    virtual bool isLoopInvariant(const Loop *L) const {
      return Op->isLoopInvariant(L);
    }

    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      return Op->hasComputableLoopEvolution(L);
    }

    /// getValueRange - Return the tightest constant bounds that this value is
    /// known to have.  This method is only valid on integer SCEV objects.
    virtual ConstantRange getValueRange() const;

    virtual void print(std::ostream &OS) const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVZeroExtendExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scZeroExtend;
    }
  };


  //===--------------------------------------------------------------------===//
  /// SCEVCommutativeExpr - This node is the base class for n'ary commutative
  /// operators.
  ///
  class SCEVCommutativeExpr : public SCEV {
    std::vector<SCEVHandle> Operands;

  protected:
    SCEVCommutativeExpr(enum SCEVTypes T, const std::vector<SCEVHandle> &ops)
      : SCEV(T) {
      Operands.reserve(ops.size());
      Operands.insert(Operands.end(), ops.begin(), ops.end());
    }
    ~SCEVCommutativeExpr();

  public:
    unsigned getNumOperands() const { return Operands.size(); }
    const SCEVHandle &getOperand(unsigned i) const {
      assert(i < Operands.size() && "Operand index out of range!");
      return Operands[i];
    }

    const std::vector<SCEVHandle> &getOperands() const { return Operands; }
    typedef std::vector<SCEVHandle>::const_iterator op_iterator;
    op_iterator op_begin() const { return Operands.begin(); }
    op_iterator op_end() const { return Operands.end(); }


    virtual bool isLoopInvariant(const Loop *L) const {
      for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
        if (!getOperand(i)->isLoopInvariant(L)) return false;
      return true;
    }

    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
        if (getOperand(i)->hasComputableLoopEvolution(L)) return true;
      return false;
    }

    virtual const char *getOperationStr() const = 0;

    virtual const Type *getType() const { return getOperand(0)->getType(); }
    virtual void print(std::ostream &OS) const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVCommutativeExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scAddExpr ||
             S->getSCEVType() == scMulExpr;
    }
  };


  //===--------------------------------------------------------------------===//
  /// SCEVAddExpr - This node represents an addition of some number of SCEVs.
  ///
  class SCEVAddExpr : public SCEVCommutativeExpr {
    SCEVAddExpr(const std::vector<SCEVHandle> &ops)
      : SCEVCommutativeExpr(scAddExpr, ops) {
    }

  public:
    static SCEVHandle get(std::vector<SCEVHandle> &Ops);

    static SCEVHandle get(const SCEVHandle &LHS, const SCEVHandle &RHS) {
      std::vector<SCEVHandle> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      return get(Ops);
    }

    static SCEVHandle get(const SCEVHandle &Op0, const SCEVHandle &Op1,
                          const SCEVHandle &Op2) {
      std::vector<SCEVHandle> Ops;
      Ops.push_back(Op0);
      Ops.push_back(Op1);
      Ops.push_back(Op2);
      return get(Ops);
    }

    virtual const char *getOperationStr() const { return " + "; }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVAddExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scAddExpr;
    }
  };

  //===--------------------------------------------------------------------===//
  /// SCEVMulExpr - This node represents multiplication of some number of SCEVs.
  ///
  class SCEVMulExpr : public SCEVCommutativeExpr {
    SCEVMulExpr(const std::vector<SCEVHandle> &ops)
      : SCEVCommutativeExpr(scMulExpr, ops) {
    }

  public:
    static SCEVHandle get(std::vector<SCEVHandle> &Ops);

    static SCEVHandle get(const SCEVHandle &LHS, const SCEVHandle &RHS) {
      std::vector<SCEVHandle> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      return get(Ops);
    }

    virtual const char *getOperationStr() const { return " * "; }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVMulExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scMulExpr;
    }
  };


  //===--------------------------------------------------------------------===//
  /// SCEVUDivExpr - This class represents a binary unsigned division operation.
  ///
  class SCEVUDivExpr : public SCEV {
    SCEVHandle LHS, RHS;
    SCEVUDivExpr(const SCEVHandle &lhs, const SCEVHandle &rhs)
      : SCEV(scUDivExpr), LHS(lhs), RHS(rhs) {}

    virtual ~SCEVUDivExpr();
  public:
    /// get method - This just gets and returns a new SCEVUDiv object.
    ///
    static SCEVHandle get(const SCEVHandle &LHS, const SCEVHandle &RHS);

    const SCEVHandle &getLHS() const { return LHS; }
    const SCEVHandle &getRHS() const { return RHS; }

    virtual bool isLoopInvariant(const Loop *L) const {
      return LHS->isLoopInvariant(L) && RHS->isLoopInvariant(L);
    }

    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      return LHS->hasComputableLoopEvolution(L) &&
             RHS->hasComputableLoopEvolution(L);
    }

    virtual const Type *getType() const;

    void print(std::ostream &OS) const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVUDivExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scUDivExpr;
    }
  };


  //===--------------------------------------------------------------------===//
  /// SCEVAddRecExpr - This node represents a polynomial recurrence on the trip
  /// count of the specified loop.
  ///
  /// All operands of an AddRec are required to be loop invariant.
  ///
  class SCEVAddRecExpr : public SCEV {
    std::vector<SCEVHandle> Operands;
    const Loop *L;

    SCEVAddRecExpr(const std::vector<SCEVHandle> &ops, const Loop *l)
      : SCEV(scAddRecExpr), Operands(ops), L(l) {
      for (unsigned i = 0, e = Operands.size(); i != e; ++i)
        assert(Operands[i]->isLoopInvariant(l) &&
               "Operands of AddRec must be loop-invariant!");
    }
    ~SCEVAddRecExpr();
  public:
    static SCEVHandle get(const SCEVHandle &Start, const SCEVHandle &Step,
                          const Loop *);
    static SCEVHandle get(std::vector<SCEVHandle> &Operands,
                          const Loop *);
    static SCEVHandle get(const std::vector<SCEVHandle> &Operands,
                          const Loop *L) {
      std::vector<SCEVHandle> NewOp(Operands);
      return get(NewOp, L);
    }

    typedef std::vector<SCEVHandle>::const_iterator op_iterator;
    op_iterator op_begin() const { return Operands.begin(); }
    op_iterator op_end() const { return Operands.end(); }

    unsigned getNumOperands() const { return Operands.size(); }
    const SCEVHandle &getOperand(unsigned i) const { return Operands[i]; }
    const SCEVHandle &getStart() const { return Operands[0]; }
    const Loop *getLoop() const { return L; }


    /// getStepRecurrence - This method constructs and returns the recurrence
    /// indicating how much this expression steps by.  If this is a polynomial
    /// of degree N, it returns a chrec of degree N-1.
    SCEVHandle getStepRecurrence() const {
      if (getNumOperands() == 2) return getOperand(1);
      return SCEVAddRecExpr::get(std::vector<SCEVHandle>(op_begin()+1,op_end()),
                                 getLoop());
    }

    virtual bool hasComputableLoopEvolution(const Loop *QL) const {
      if (L == QL) return true;
      /// FIXME: What if the start or step value a recurrence for the specified
      /// loop?
      return false;
    }

    virtual bool isLoopInvariant(const Loop *QueryLoop) const;

    virtual const Type *getType() const { return Operands[0]->getType(); }

    /// isAffine - Return true if this is an affine AddRec (i.e., it represents
    /// an expressions A+B*x where A and B are loop invariant values.
    bool isAffine() const {
      // We know that the start value is invariant.  This expression is thus
      // affine iff the step is also invariant.
      return getNumOperands() == 2;
    }

    /// isQuadratic - Return true if this is an quadratic AddRec (i.e., it
    /// represents an expressions A+B*x+C*x^2 where A, B and C are loop
    /// invariant values.  This corresponds to an addrec of the form {L,+,M,+,N}
    bool isQuadratic() const {
      return getNumOperands() == 3;
    }

    /// evaluateAtIteration - Return the value of this chain of recurrences at
    /// the specified iteration number.
    SCEVHandle evaluateAtIteration(SCEVHandle It) const;

    /// getNumIterationsInRange - Return the number of iterations of this loop
    /// that produce values in the specified constant range.  Another way of
    /// looking at this is that it returns the first iteration number where the
    /// value is not in the condition, thus computing the exit count.  If the
    /// iteration count can't be computed, an instance of SCEVCouldNotCompute is
    /// returned.
    SCEVHandle getNumIterationsInRange(ConstantRange Range) const;


    virtual void print(std::ostream &OS) const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVAddRecExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scAddRecExpr;
    }
  };

  //===--------------------------------------------------------------------===//
  /// SCEVUnknown - This means that we are dealing with an entirely unknown SCEV
  /// value, and only represent it as it's LLVM Value.  This is the "bottom"
  /// value for the analysis.
  ///
  class SCEVUnknown : public SCEV {
    Value *V;
    SCEVUnknown(Value *v) : SCEV(scUnknown), V(v) {}

  protected:
    ~SCEVUnknown();
  public:
    /// get method - For SCEVUnknown, this just gets and returns a new
    /// SCEVUnknown.
    static SCEVHandle get(Value *V);

    /// getIntegerSCEV - Given an integer or FP type, create a constant for the
    /// specified signed integer value and return a SCEV for the constant.
    static SCEVHandle getIntegerSCEV(int Val, const Type *Ty);

    Value *getValue() const { return V; }

    virtual bool isLoopInvariant(const Loop *L) const;
    virtual bool hasComputableLoopEvolution(const Loop *QL) const {
      return false; // not computable
    }

    virtual const Type *getType() const;

    virtual void print(std::ostream &OS) const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVUnknown *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scUnknown;
    }
  };

  /// SCEVVisitor - This class defines a simple visitor class that may be used
  /// for various SCEV analysis purposes.
  template<typename SC, typename RetVal=void>
  struct SCEVVisitor {
    RetVal visit(SCEV *S) {
      switch (S->getSCEVType()) {
      case scConstant:
        return ((SC*)this)->visitConstant((SCEVConstant*)S);
      case scTruncate:
        return ((SC*)this)->visitTruncateExpr((SCEVTruncateExpr*)S);
      case scZeroExtend:
        return ((SC*)this)->visitZeroExtendExpr((SCEVZeroExtendExpr*)S);
      case scAddExpr:
        return ((SC*)this)->visitAddExpr((SCEVAddExpr*)S);
      case scMulExpr:
        return ((SC*)this)->visitMulExpr((SCEVMulExpr*)S);
      case scUDivExpr:
        return ((SC*)this)->visitUDivExpr((SCEVUDivExpr*)S);
      case scAddRecExpr:
        return ((SC*)this)->visitAddRecExpr((SCEVAddRecExpr*)S);
      case scUnknown:
        return ((SC*)this)->visitUnknown((SCEVUnknown*)S);
      case scCouldNotCompute:
        return ((SC*)this)->visitCouldNotCompute((SCEVCouldNotCompute*)S);
      default:
        assert(0 && "Unknown SCEV type!");
      }
    }

    RetVal visitCouldNotCompute(SCEVCouldNotCompute *S) {
      assert(0 && "Invalid use of SCEVCouldNotCompute!");
      abort();
      return RetVal();
    }
  };
}

#endif

