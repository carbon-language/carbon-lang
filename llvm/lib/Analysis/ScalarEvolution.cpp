//===- ScalarEvolution.cpp - Scalar Evolution Analysis ----------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the scalar evolution analysis
// engine, which is used primarily to analyze expressions involving induction
// variables in loops.
//
// There are several aspects to this library.  First is the representation of
// scalar expressions, which are represented as subclasses of the SCEV class.
// These classes are used to represent certain types of subexpressions that we
// can handle.  These classes are reference counted, managed by the SCEVHandle
// class.  We only create one SCEV of a particular shape, so pointer-comparisons
// for equality are legal.
//
// One important aspect of the SCEV objects is that they are never cyclic, even
// if there is a cycle in the dataflow for an expression (ie, a PHI node).  If
// the PHI node is one of the idioms that we can represent (e.g., a polynomial
// recurrence) then we represent it directly as a recurrence node, otherwise we
// represent it as a SCEVUnknown node.
//
// In addition to being able to represent expressions of various types, we also
// have folders that are used to build the *canonical* representation for a
// particular expression.  These folders are capable of using a variety of
// rewrite rules to simplify the expressions.
// 
// Once the folders are defined, we can implement the more interesting
// higher-level code, such as the code that recognizes PHI nodes of various
// types, computes the execution count of a loop, etc.
//
// Orthogonal to the analysis of code above, this file also implements the
// ScalarEvolutionRewriter class, which is used to emit code that represents the
// various recurrences present in a loop, in canonical forms.
//
// TODO: We should use these routines and value representations to implement
// dependence analysis!
//
//===----------------------------------------------------------------------===//
//
// There are several good references for the techniques used in this analysis.
//
//  Chains of recurrences -- a method to expedite the evaluation
//  of closed-form functions
//  Olaf Bachmann, Paul S. Wang, Eugene V. Zima
//
//  On computational properties of chains of recurrences
//  Eugene V. Zima
//
//  Symbolic Evaluation of Chains of Recurrences for Loop Optimization
//  Robert A. van Engelen
//
//  Efficient Symbolic Analysis for Optimizing Compilers
//  Robert A. van Engelen
//
//  Using the chains of recurrences algebra for data dependence testing and
//  induction variable substitution
//  MS Thesis, Johnie Birch
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Value.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/InstIterator.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  RegisterAnalysis<ScalarEvolution>
  R("scalar-evolution", "Scalar Evolution Analysis Printer");

  Statistic<>
  NumBruteForceEvaluations("scalar-evolution",
                           "Number of brute force evaluations needed to calculate high-order polynomial exit values");
  Statistic<>
  NumTripCountsComputed("scalar-evolution",
                        "Number of loops with predictable loop counts");
  Statistic<>
  NumTripCountsNotComputed("scalar-evolution",
                           "Number of loops without predictable loop counts");
}

//===----------------------------------------------------------------------===//
//                           SCEV class definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Implementation of the SCEV class.
//
namespace {
  enum SCEVTypes {
    // These should be ordered in terms of increasing complexity to make the
    // folders simpler.
    scConstant, scTruncate, scZeroExtend, scAddExpr, scMulExpr, scUDivExpr,
    scAddRecExpr, scUnknown, scCouldNotCompute
  };

  /// SCEVComplexityCompare - Return true if the complexity of the LHS is less
  /// than the complexity of the RHS.  If the SCEVs have identical complexity,
  /// order them by their addresses.  This comparator is used to canonicalize
  /// expressions.
  struct SCEVComplexityCompare {
    bool operator()(SCEV *LHS, SCEV *RHS) {
      if (LHS->getSCEVType() < RHS->getSCEVType())
        return true;
      if (LHS->getSCEVType() == RHS->getSCEVType())
        return LHS < RHS;
      return false;
    }
  };
}

SCEV::~SCEV() {}
void SCEV::dump() const {
  print(std::cerr);
}

/// getValueRange - Return the tightest constant bounds that this value is
/// known to have.  This method is only valid on integer SCEV objects.
ConstantRange SCEV::getValueRange() const {
  const Type *Ty = getType();
  assert(Ty->isInteger() && "Can't get range for a non-integer SCEV!");
  Ty = Ty->getUnsignedVersion();
  // Default to a full range if no better information is available.
  return ConstantRange(getType());
}


SCEVCouldNotCompute::SCEVCouldNotCompute() : SCEV(scCouldNotCompute) {}

bool SCEVCouldNotCompute::isLoopInvariant(const Loop *L) const {
  assert(0 && "Attempt to use a SCEVCouldNotCompute object!");
  return false;
}

const Type *SCEVCouldNotCompute::getType() const {
  assert(0 && "Attempt to use a SCEVCouldNotCompute object!");
  return 0;
}

bool SCEVCouldNotCompute::hasComputableLoopEvolution(const Loop *L) const {
  assert(0 && "Attempt to use a SCEVCouldNotCompute object!");
  return false;
}

Value *SCEVCouldNotCompute::expandCodeFor(ScalarEvolutionRewriter &SER,
                                          Instruction *InsertPt) {
  assert(0 && "Attempt to use a SCEVCouldNotCompute object!");
  return 0;
}


void SCEVCouldNotCompute::print(std::ostream &OS) const {
  OS << "***COULDNOTCOMPUTE***";
}

bool SCEVCouldNotCompute::classof(const SCEV *S) {
  return S->getSCEVType() == scCouldNotCompute;
}


//===----------------------------------------------------------------------===//
// SCEVConstant - This class represents a constant integer value.
//
namespace {
  class SCEVConstant;
  // SCEVConstants - Only allow the creation of one SCEVConstant for any
  // particular value.  Don't use a SCEVHandle here, or else the object will
  // never be deleted!
  std::map<ConstantInt*, SCEVConstant*> SCEVConstants;

  class SCEVConstant : public SCEV {
    ConstantInt *V;
    SCEVConstant(ConstantInt *v) : SCEV(scConstant), V(v) {}

    virtual ~SCEVConstant() {
      SCEVConstants.erase(V);
    }
  public:
    /// get method - This just gets and returns a new SCEVConstant object.
    ///
    static SCEVHandle get(ConstantInt *V) {
      // Make sure that SCEVConstant instances are all unsigned.
      if (V->getType()->isSigned()) {
        const Type *NewTy = V->getType()->getUnsignedVersion();
        V = cast<ConstantUInt>(ConstantExpr::getCast(V, NewTy));
      }

      SCEVConstant *&R = SCEVConstants[V];
      if (R == 0) R = new SCEVConstant(V);
      return R;
    }

    ConstantInt *getValue() const { return V; }

    /// getValueRange - Return the tightest constant bounds that this value is
    /// known to have.  This method is only valid on integer SCEV objects.
    virtual ConstantRange getValueRange() const {
      return ConstantRange(V);
    }

    virtual bool isLoopInvariant(const Loop *L) const {
      return true;
    }

    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      return false;  // Not loop variant
    }

    virtual const Type *getType() const { return V->getType(); }

    Value *expandCodeFor(ScalarEvolutionRewriter &SER,
                         Instruction *InsertPt) {
      return getValue();
    }
    
    virtual void print(std::ostream &OS) const {
      WriteAsOperand(OS, V, false);
    }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVConstant *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scConstant;
    }
  };
}


//===----------------------------------------------------------------------===//
// SCEVTruncateExpr - This class represents a truncation of an integer value to
// a smaller integer value.
//
namespace {
  class SCEVTruncateExpr;
  // SCEVTruncates - Only allow the creation of one SCEVTruncateExpr for any
  // particular input.  Don't use a SCEVHandle here, or else the object will
  // never be deleted!
  std::map<std::pair<SCEV*, const Type*>, SCEVTruncateExpr*> SCEVTruncates;

  class SCEVTruncateExpr : public SCEV {
    SCEVHandle Op;
    const Type *Ty;
    SCEVTruncateExpr(const SCEVHandle &op, const Type *ty)
      : SCEV(scTruncate), Op(op), Ty(ty) {
      assert(Op->getType()->isInteger() && Ty->isInteger() &&
             Ty->isUnsigned() &&
             "Cannot truncate non-integer value!");
      assert(Op->getType()->getPrimitiveSize() > Ty->getPrimitiveSize() &&
             "This is not a truncating conversion!");
    }

    virtual ~SCEVTruncateExpr() {
      SCEVTruncates.erase(std::make_pair(Op, Ty));
    }
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
    virtual ConstantRange getValueRange() const {
      return getOperand()->getValueRange().truncate(getType());
    }

    Value *expandCodeFor(ScalarEvolutionRewriter &SER,
                         Instruction *InsertPt);
    
    virtual void print(std::ostream &OS) const {
      OS << "(truncate " << *Op << " to " << *Ty << ")";
    }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVTruncateExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scTruncate;
    }
  };
}


//===----------------------------------------------------------------------===//
// SCEVZeroExtendExpr - This class represents a zero extension of a small
// integer value to a larger integer value.
//
namespace {
  class SCEVZeroExtendExpr;
  // SCEVZeroExtends - Only allow the creation of one SCEVZeroExtendExpr for any
  // particular input.  Don't use a SCEVHandle here, or else the object will
  // never be deleted!
  std::map<std::pair<SCEV*, const Type*>, SCEVZeroExtendExpr*> SCEVZeroExtends;

  class SCEVZeroExtendExpr : public SCEV {
    SCEVHandle Op;
    const Type *Ty;
    SCEVZeroExtendExpr(const SCEVHandle &op, const Type *ty)
      : SCEV(scTruncate), Op(Op), Ty(ty) {
      assert(Op->getType()->isInteger() && Ty->isInteger() &&
             Ty->isUnsigned() &&
             "Cannot zero extend non-integer value!");
      assert(Op->getType()->getPrimitiveSize() < Ty->getPrimitiveSize() &&
             "This is not an extending conversion!");
    }

    virtual ~SCEVZeroExtendExpr() {
      SCEVZeroExtends.erase(std::make_pair(Op, Ty));
    }
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
    virtual ConstantRange getValueRange() const {
      return getOperand()->getValueRange().zeroExtend(getType());
    }

    Value *expandCodeFor(ScalarEvolutionRewriter &SER,
                         Instruction *InsertPt);
    
    virtual void print(std::ostream &OS) const {
      OS << "(zeroextend " << *Op << " to " << *Ty << ")";
    }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVZeroExtendExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scZeroExtend;
    }
  };
}


//===----------------------------------------------------------------------===//
// SCEVCommutativeExpr - This node is the base class for n'ary commutative
// operators.

namespace {
  class SCEVCommutativeExpr;
  // SCEVCommExprs - Only allow the creation of one SCEVCommutativeExpr for any
  // particular input.  Don't use a SCEVHandle here, or else the object will
  // never be deleted!
  std::map<std::pair<unsigned, std::vector<SCEV*> >,
           SCEVCommutativeExpr*> SCEVCommExprs;

  class SCEVCommutativeExpr : public SCEV {
    std::vector<SCEVHandle> Operands;

  protected:
    SCEVCommutativeExpr(enum SCEVTypes T, const std::vector<SCEVHandle> &ops)
      : SCEV(T) {
      Operands.reserve(ops.size());
      Operands.insert(Operands.end(), ops.begin(), ops.end());
    }

    ~SCEVCommutativeExpr() {
      SCEVCommExprs.erase(std::make_pair(getSCEVType(),
                                         std::vector<SCEV*>(Operands.begin(),
                                                            Operands.end())));
    }

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

    virtual const Type *getType() const { return getOperand(0)->getType(); }

    virtual const char *getOperationStr() const = 0;
    
    virtual void print(std::ostream &OS) const {
      assert(Operands.size() > 1 && "This plus expr shouldn't exist!");
      const char *OpStr = getOperationStr();
      OS << "(" << *Operands[0];
      for (unsigned i = 1, e = Operands.size(); i != e; ++i)
        OS << OpStr << *Operands[i];
      OS << ")";
    }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVCommutativeExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scAddExpr ||
             S->getSCEVType() == scMulExpr;
    }
  };
}

//===----------------------------------------------------------------------===//
// SCEVAddExpr - This node represents an addition of some number of SCEV's.
//
namespace {
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

    Value *expandCodeFor(ScalarEvolutionRewriter &SER,
                         Instruction *InsertPt);

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVAddExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scAddExpr;
    }
  };
}

//===----------------------------------------------------------------------===//
// SCEVMulExpr - This node represents multiplication of some number of SCEV's.
//
namespace {
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

    Value *expandCodeFor(ScalarEvolutionRewriter &SER,
                         Instruction *InsertPt);

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVMulExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scMulExpr;
    }
  };
}


//===----------------------------------------------------------------------===//
// SCEVUDivExpr - This class represents a binary unsigned division operation.
//
namespace {
  class SCEVUDivExpr;
  // SCEVUDivs - Only allow the creation of one SCEVUDivExpr for any particular
  // input.  Don't use a SCEVHandle here, or else the object will never be
  // deleted!
  std::map<std::pair<SCEV*, SCEV*>, SCEVUDivExpr*> SCEVUDivs;

  class SCEVUDivExpr : public SCEV {
    SCEVHandle LHS, RHS;
    SCEVUDivExpr(const SCEVHandle &lhs, const SCEVHandle &rhs)
      : SCEV(scUDivExpr), LHS(lhs), RHS(rhs) {}

    virtual ~SCEVUDivExpr() {
      SCEVUDivs.erase(std::make_pair(LHS, RHS));
    }
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

    virtual const Type *getType() const {
      const Type *Ty = LHS->getType();
      if (Ty->isSigned()) Ty = Ty->getUnsignedVersion();
      return Ty;
    }

    Value *expandCodeFor(ScalarEvolutionRewriter &SER,
                         Instruction *InsertPt);
    
    virtual void print(std::ostream &OS) const {
      OS << "(" << *LHS << " /u " << *RHS << ")";
    }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVUDivExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scUDivExpr;
    }
  };
}


//===----------------------------------------------------------------------===//

// SCEVAddRecExpr - This node represents a polynomial recurrence on the trip
// count of the specified loop.
//
// All operands of an AddRec are required to be loop invariant.
//
namespace {
  class SCEVAddRecExpr;
  // SCEVAddRecExprs - Only allow the creation of one SCEVAddRecExpr for any
  // particular input.  Don't use a SCEVHandle here, or else the object will
  // never be deleted!
  std::map<std::pair<const Loop *, std::vector<SCEV*> >,
           SCEVAddRecExpr*> SCEVAddRecExprs;

  class SCEVAddRecExpr : public SCEV {
    std::vector<SCEVHandle> Operands;
    const Loop *L;

    SCEVAddRecExpr(const std::vector<SCEVHandle> &ops, const Loop *l)
      : SCEV(scAddRecExpr), Operands(ops), L(l) {
      for (unsigned i = 0, e = Operands.size(); i != e; ++i)
        assert(Operands[i]->isLoopInvariant(l) &&
               "Operands of AddRec must be loop-invariant!");
    }
    ~SCEVAddRecExpr() {
      SCEVAddRecExprs.erase(std::make_pair(L,
                                           std::vector<SCEV*>(Operands.begin(),
                                                              Operands.end())));
    }
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


    virtual bool isLoopInvariant(const Loop *QueryLoop) const {
      // This recurrence is invariant w.r.t to QueryLoop iff QueryLoop doesn't
      // contain L.
      return !QueryLoop->contains(L->getHeader());
    }

    virtual const Type *getType() const { return Operands[0]->getType(); }

    Value *expandCodeFor(ScalarEvolutionRewriter &SER,
                         Instruction *InsertPt);


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


    virtual void print(std::ostream &OS) const {
      OS << "{" << *Operands[0];
      for (unsigned i = 1, e = Operands.size(); i != e; ++i)
        OS << ",+," << *Operands[i];
      OS << "}<" << L->getHeader()->getName() + ">";
    }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVAddRecExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scAddRecExpr;
    }
  };
}


//===----------------------------------------------------------------------===//
// SCEVUnknown - This means that we are dealing with an entirely unknown SCEV
// value, and only represent it as it's LLVM Value.  This is the "bottom" value
// for the analysis.
//
namespace {
  class SCEVUnknown;
  // SCEVUnknowns - Only allow the creation of one SCEVUnknown for any
  // particular value.  Don't use a SCEVHandle here, or else the object will
  // never be deleted!
  std::map<Value*, SCEVUnknown*> SCEVUnknowns;

  class SCEVUnknown : public SCEV {
    Value *V;
    SCEVUnknown(Value *v) : SCEV(scUnknown), V(v) {}

  protected:
    ~SCEVUnknown() { SCEVUnknowns.erase(V); }
  public:
    /// get method - For SCEVUnknown, this just gets and returns a new
    /// SCEVUnknown.
    static SCEVHandle get(Value *V) {
      if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
        return SCEVConstant::get(CI);
      SCEVUnknown *&Result = SCEVUnknowns[V];
      if (Result == 0) Result = new SCEVUnknown(V);
      return Result;
    }

    Value *getValue() const { return V; }

    Value *expandCodeFor(ScalarEvolutionRewriter &SER,
                         Instruction *InsertPt) {
      return V;
    }

    virtual bool isLoopInvariant(const Loop *L) const {
      // All non-instruction values are loop invariant.  All instructions are
      // loop invariant if they are not contained in the specified loop.
      if (Instruction *I = dyn_cast<Instruction>(V))
        return !L->contains(I->getParent());
      return true;
    }

    virtual bool hasComputableLoopEvolution(const Loop *QL) const {
      return false; // not computable
    }

    virtual const Type *getType() const { return V->getType(); }

    virtual void print(std::ostream &OS) const {
      WriteAsOperand(OS, V, false);
    }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVUnknown *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scUnknown;
    }
  };
}

//===----------------------------------------------------------------------===//
//                      Simple SCEV method implementations
//===----------------------------------------------------------------------===//

/// getIntegerSCEV - Given an integer or FP type, create a constant for the
/// specified signed integer value and return a SCEV for the constant.
static SCEVHandle getIntegerSCEV(int Val, const Type *Ty) {
  Constant *C;
  if (Val == 0) 
    C = Constant::getNullValue(Ty);
  else if (Ty->isFloatingPoint())
    C = ConstantFP::get(Ty, Val);
  else if (Ty->isSigned())
    C = ConstantSInt::get(Ty, Val);
  else {
    C = ConstantSInt::get(Ty->getSignedVersion(), Val);
    C = ConstantExpr::getCast(C, Ty);
  }
  return SCEVUnknown::get(C);
}

/// getTruncateOrZeroExtend - Return a SCEV corresponding to a conversion of the
/// input value to the specified type.  If the type must be extended, it is zero
/// extended.
static SCEVHandle getTruncateOrZeroExtend(const SCEVHandle &V, const Type *Ty) {
  const Type *SrcTy = V->getType();
  assert(SrcTy->isInteger() && Ty->isInteger() &&
         "Cannot truncate or zero extend with non-integer arguments!");
  if (SrcTy->getPrimitiveSize() == Ty->getPrimitiveSize())
    return V;  // No conversion
  if (SrcTy->getPrimitiveSize() > Ty->getPrimitiveSize())
    return SCEVTruncateExpr::get(V, Ty);
  return SCEVZeroExtendExpr::get(V, Ty);
}

/// getNegativeSCEV - Return a SCEV corresponding to -V = -1*V
///
static SCEVHandle getNegativeSCEV(const SCEVHandle &V) {
  if (SCEVConstant *VC = dyn_cast<SCEVConstant>(V))
    return SCEVUnknown::get(ConstantExpr::getNeg(VC->getValue()));
  
  return SCEVMulExpr::get(V, getIntegerSCEV(-1, V->getType()));
}

/// getMinusSCEV - Return a SCEV corresponding to LHS - RHS.
///
static SCEVHandle getMinusSCEV(const SCEVHandle &LHS, const SCEVHandle &RHS) {
  // X - Y --> X + -Y
  return SCEVAddExpr::get(LHS, getNegativeSCEV(RHS));
}


/// Binomial - Evaluate N!/((N-M)!*M!)  .  Note that N is often large and M is
/// often very small, so we try to reduce the number of N! terms we need to
/// evaluate by evaluating this as  (N!/(N-M)!)/M!
static ConstantInt *Binomial(ConstantInt *N, unsigned M) {
  uint64_t NVal = N->getRawValue();
  uint64_t FirstTerm = 1;
  for (unsigned i = 0; i != M; ++i)
    FirstTerm *= NVal-i;

  unsigned MFactorial = 1;
  for (; M; --M)
    MFactorial *= M;

  Constant *Result = ConstantUInt::get(Type::ULongTy, FirstTerm/MFactorial);
  Result = ConstantExpr::getCast(Result, N->getType());
  assert(isa<ConstantInt>(Result) && "Cast of integer not folded??");
  return cast<ConstantInt>(Result);
}

/// PartialFact - Compute V!/(V-NumSteps)!
static SCEVHandle PartialFact(SCEVHandle V, unsigned NumSteps) {
  // Handle this case efficiently, it is common to have constant iteration
  // counts while computing loop exit values.
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(V)) {
    uint64_t Val = SC->getValue()->getRawValue();
    uint64_t Result = 1;
    for (; NumSteps; --NumSteps)
      Result *= Val-(NumSteps-1);
    Constant *Res = ConstantUInt::get(Type::ULongTy, Result);
    return SCEVUnknown::get(ConstantExpr::getCast(Res, V->getType()));
  }

  const Type *Ty = V->getType();
  if (NumSteps == 0)
    return getIntegerSCEV(1, Ty);
  
  SCEVHandle Result = V;
  for (unsigned i = 1; i != NumSteps; ++i)
    Result = SCEVMulExpr::get(Result, getMinusSCEV(V, getIntegerSCEV(i, Ty)));
  return Result;
}


/// evaluateAtIteration - Return the value of this chain of recurrences at
/// the specified iteration number.  We can evaluate this recurrence by
/// multiplying each element in the chain by the binomial coefficient
/// corresponding to it.  In other words, we can evaluate {A,+,B,+,C,+,D} as:
///
///   A*choose(It, 0) + B*choose(It, 1) + C*choose(It, 2) + D*choose(It, 3)
///
/// FIXME/VERIFY: I don't trust that this is correct in the face of overflow.
/// Is the binomial equation safe using modular arithmetic??
///
SCEVHandle SCEVAddRecExpr::evaluateAtIteration(SCEVHandle It) const {
  SCEVHandle Result = getStart();
  int Divisor = 1;
  const Type *Ty = It->getType();
  for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
    SCEVHandle BC = PartialFact(It, i);
    Divisor *= i;
    SCEVHandle Val = SCEVUDivExpr::get(SCEVMulExpr::get(BC, getOperand(i)),
                                       getIntegerSCEV(Divisor, Ty));
    Result = SCEVAddExpr::get(Result, Val);
  }
  return Result;
}


//===----------------------------------------------------------------------===//
//                    SCEV Expression folder implementations
//===----------------------------------------------------------------------===//

SCEVHandle SCEVTruncateExpr::get(const SCEVHandle &Op, const Type *Ty) {
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    return SCEVUnknown::get(ConstantExpr::getCast(SC->getValue(), Ty));

  // If the input value is a chrec scev made out of constants, truncate
  // all of the constants.
  if (SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(Op)) {
    std::vector<SCEVHandle> Operands;
    for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i)
      // FIXME: This should allow truncation of other expression types!
      if (isa<SCEVConstant>(AddRec->getOperand(i)))
        Operands.push_back(get(AddRec->getOperand(i), Ty));
      else
        break;
    if (Operands.size() == AddRec->getNumOperands())
      return SCEVAddRecExpr::get(Operands, AddRec->getLoop());
  }

  SCEVTruncateExpr *&Result = SCEVTruncates[std::make_pair(Op, Ty)];
  if (Result == 0) Result = new SCEVTruncateExpr(Op, Ty);
  return Result;
}

SCEVHandle SCEVZeroExtendExpr::get(const SCEVHandle &Op, const Type *Ty) {
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    return SCEVUnknown::get(ConstantExpr::getCast(SC->getValue(), Ty));

  // FIXME: If the input value is a chrec scev, and we can prove that the value
  // did not overflow the old, smaller, value, we can zero extend all of the
  // operands (often constants).  This would allow analysis of something like
  // this:  for (unsigned char X = 0; X < 100; ++X) { int Y = X; }

  SCEVZeroExtendExpr *&Result = SCEVZeroExtends[std::make_pair(Op, Ty)];
  if (Result == 0) Result = new SCEVZeroExtendExpr(Op, Ty);
  return Result;
}

// get - Get a canonical add expression, or something simpler if possible.
SCEVHandle SCEVAddExpr::get(std::vector<SCEVHandle> &Ops) {
  assert(!Ops.empty() && "Cannot get empty add!");

  // Sort by complexity, this groups all similar expression types together.
  std::sort(Ops.begin(), Ops.end(), SCEVComplexityCompare());

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {
    ++Idx;
    while (SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      Constant *Fold = ConstantExpr::getAdd(LHSC->getValue(), RHSC->getValue());
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Fold)) {
        Ops[0] = SCEVConstant::get(CI);
        Ops.erase(Ops.begin()+1);  // Erase the folded element
        if (Ops.size() == 1) return Ops[0];
      } else {
        // If we couldn't fold the expression, move to the next constant.  Note
        // that this is impossible to happen in practice because we always
        // constant fold constant ints to constant ints.
        ++Idx;
      }
    }

    // If we are left with a constant zero being added, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->isNullValue()) {
      Ops.erase(Ops.begin());
      --Idx;
    }
  }

  if (Ops.size() == 1)
    return Ops[0];
  
  // Okay, check to see if the same value occurs in the operand list twice.  If
  // so, merge them together into an multiply expression.  Since we sorted the
  // list, these values are required to be adjacent.
  const Type *Ty = Ops[0]->getType();
  for (unsigned i = 0, e = Ops.size()-1; i != e; ++i)
    if (Ops[i] == Ops[i+1]) {      //  X + Y + Y  -->  X + Y*2
      // Found a match, merge the two values into a multiply, and add any
      // remaining values to the result.
      SCEVHandle Two = getIntegerSCEV(2, Ty);
      SCEVHandle Mul = SCEVMulExpr::get(Ops[i], Two);
      if (Ops.size() == 2)
        return Mul;
      Ops.erase(Ops.begin()+i, Ops.begin()+i+2);
      Ops.push_back(Mul);
      return SCEVAddExpr::get(Ops);
    }

  // Okay, now we know the first non-constant operand.  If there are add
  // operands they would be next.
  if (Idx < Ops.size()) {
    bool DeletedAdd = false;
    while (SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Ops[Idx])) {
      // If we have an add, expand the add operands onto the end of the operands
      // list.
      Ops.insert(Ops.end(), Add->op_begin(), Add->op_end());
      Ops.erase(Ops.begin()+Idx);
      DeletedAdd = true;
    }

    // If we deleted at least one add, we added operands to the end of the list,
    // and they are not necessarily sorted.  Recurse to resort and resimplify
    // any operands we just aquired.
    if (DeletedAdd)
      return get(Ops);
  }

  // Skip over the add expression until we get to a multiply.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scMulExpr)
    ++Idx;

  // If we are adding something to a multiply expression, make sure the
  // something is not already an operand of the multiply.  If so, merge it into
  // the multiply.
  for (; Idx < Ops.size() && isa<SCEVMulExpr>(Ops[Idx]); ++Idx) {
    SCEVMulExpr *Mul = cast<SCEVMulExpr>(Ops[Idx]);
    for (unsigned MulOp = 0, e = Mul->getNumOperands(); MulOp != e; ++MulOp) {
      SCEV *MulOpSCEV = Mul->getOperand(MulOp);
      for (unsigned AddOp = 0, e = Ops.size(); AddOp != e; ++AddOp)
        if (MulOpSCEV == Ops[AddOp] &&
            (Mul->getNumOperands() != 2 || !isa<SCEVConstant>(MulOpSCEV))) {
          // Fold W + X + (X * Y * Z)  -->  W + (X * ((Y*Z)+1))
          SCEVHandle InnerMul = Mul->getOperand(MulOp == 0);
          if (Mul->getNumOperands() != 2) {
            // If the multiply has more than two operands, we must get the
            // Y*Z term.
            std::vector<SCEVHandle> MulOps(Mul->op_begin(), Mul->op_end());
            MulOps.erase(MulOps.begin()+MulOp);
            InnerMul = SCEVMulExpr::get(MulOps);
          }
          SCEVHandle One = getIntegerSCEV(1, Ty);
          SCEVHandle AddOne = SCEVAddExpr::get(InnerMul, One);
          SCEVHandle OuterMul = SCEVMulExpr::get(AddOne, Ops[AddOp]);
          if (Ops.size() == 2) return OuterMul;
          if (AddOp < Idx) {
            Ops.erase(Ops.begin()+AddOp);
            Ops.erase(Ops.begin()+Idx-1);
          } else {
            Ops.erase(Ops.begin()+Idx);
            Ops.erase(Ops.begin()+AddOp-1);
          }
          Ops.push_back(OuterMul);
          return SCEVAddExpr::get(Ops);
        }
      
      // Check this multiply against other multiplies being added together.
      for (unsigned OtherMulIdx = Idx+1;
           OtherMulIdx < Ops.size() && isa<SCEVMulExpr>(Ops[OtherMulIdx]);
           ++OtherMulIdx) {
        SCEVMulExpr *OtherMul = cast<SCEVMulExpr>(Ops[OtherMulIdx]);
        // If MulOp occurs in OtherMul, we can fold the two multiplies
        // together.
        for (unsigned OMulOp = 0, e = OtherMul->getNumOperands();
             OMulOp != e; ++OMulOp)
          if (OtherMul->getOperand(OMulOp) == MulOpSCEV) {
            // Fold X + (A*B*C) + (A*D*E) --> X + (A*(B*C+D*E))
            SCEVHandle InnerMul1 = Mul->getOperand(MulOp == 0);
            if (Mul->getNumOperands() != 2) {
              std::vector<SCEVHandle> MulOps(Mul->op_begin(), Mul->op_end());
              MulOps.erase(MulOps.begin()+MulOp);
              InnerMul1 = SCEVMulExpr::get(MulOps);
            }
            SCEVHandle InnerMul2 = OtherMul->getOperand(OMulOp == 0);
            if (OtherMul->getNumOperands() != 2) {
              std::vector<SCEVHandle> MulOps(OtherMul->op_begin(),
                                             OtherMul->op_end());
              MulOps.erase(MulOps.begin()+OMulOp);
              InnerMul2 = SCEVMulExpr::get(MulOps);
            }
            SCEVHandle InnerMulSum = SCEVAddExpr::get(InnerMul1,InnerMul2);
            SCEVHandle OuterMul = SCEVMulExpr::get(MulOpSCEV, InnerMulSum);
            if (Ops.size() == 2) return OuterMul;
            Ops.erase(Ops.begin()+Idx);
            Ops.erase(Ops.begin()+OtherMulIdx-1);
            Ops.push_back(OuterMul);
            return SCEVAddExpr::get(Ops);
          }
      }
    }
  }

  // If there are any add recurrences in the operands list, see if any other
  // added values are loop invariant.  If so, we can fold them into the
  // recurrence.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scAddRecExpr)
    ++Idx;

  // Scan over all recurrences, trying to fold loop invariants into them.
  for (; Idx < Ops.size() && isa<SCEVAddRecExpr>(Ops[Idx]); ++Idx) {
    // Scan all of the other operands to this add and add them to the vector if
    // they are loop invariant w.r.t. the recurrence.
    std::vector<SCEVHandle> LIOps;
    SCEVAddRecExpr *AddRec = cast<SCEVAddRecExpr>(Ops[Idx]);
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      if (Ops[i]->isLoopInvariant(AddRec->getLoop())) {
        LIOps.push_back(Ops[i]);
        Ops.erase(Ops.begin()+i);
        --i; --e;
      }

    // If we found some loop invariants, fold them into the recurrence.
    if (!LIOps.empty()) {
      //  NLI + LI + { Start,+,Step}  -->  NLI + { LI+Start,+,Step }
      LIOps.push_back(AddRec->getStart());

      std::vector<SCEVHandle> AddRecOps(AddRec->op_begin(), AddRec->op_end());
      AddRecOps[0] = SCEVAddExpr::get(LIOps);

      SCEVHandle NewRec = SCEVAddRecExpr::get(AddRecOps, AddRec->getLoop());
      // If all of the other operands were loop invariant, we are done.
      if (Ops.size() == 1) return NewRec;

      // Otherwise, add the folded AddRec by the non-liv parts.
      for (unsigned i = 0;; ++i)
        if (Ops[i] == AddRec) {
          Ops[i] = NewRec;
          break;
        }
      return SCEVAddExpr::get(Ops);
    }

    // Okay, if there weren't any loop invariants to be folded, check to see if
    // there are multiple AddRec's with the same loop induction variable being
    // added together.  If so, we can fold them.
    for (unsigned OtherIdx = Idx+1;
         OtherIdx < Ops.size() && isa<SCEVAddRecExpr>(Ops[OtherIdx]);++OtherIdx)
      if (OtherIdx != Idx) {
        SCEVAddRecExpr *OtherAddRec = cast<SCEVAddRecExpr>(Ops[OtherIdx]);
        if (AddRec->getLoop() == OtherAddRec->getLoop()) {
          // Other + {A,+,B} + {C,+,D}  -->  Other + {A+C,+,B+D}
          std::vector<SCEVHandle> NewOps(AddRec->op_begin(), AddRec->op_end());
          for (unsigned i = 0, e = OtherAddRec->getNumOperands(); i != e; ++i) {
            if (i >= NewOps.size()) {
              NewOps.insert(NewOps.end(), OtherAddRec->op_begin()+i,
                            OtherAddRec->op_end());
              break;
            }
            NewOps[i] = SCEVAddExpr::get(NewOps[i], OtherAddRec->getOperand(i));
          }
          SCEVHandle NewAddRec = SCEVAddRecExpr::get(NewOps, AddRec->getLoop());

          if (Ops.size() == 2) return NewAddRec;

          Ops.erase(Ops.begin()+Idx);
          Ops.erase(Ops.begin()+OtherIdx-1);
          Ops.push_back(NewAddRec);
          return SCEVAddExpr::get(Ops);
        }
      }

    // Otherwise couldn't fold anything into this recurrence.  Move onto the
    // next one.
  }

  // Okay, it looks like we really DO need an add expr.  Check to see if we
  // already have one, otherwise create a new one.
  std::vector<SCEV*> SCEVOps(Ops.begin(), Ops.end());
  SCEVCommutativeExpr *&Result = SCEVCommExprs[std::make_pair(scAddExpr,
                                                              SCEVOps)];
  if (Result == 0) Result = new SCEVAddExpr(Ops);
  return Result;
}


SCEVHandle SCEVMulExpr::get(std::vector<SCEVHandle> &Ops) {
  assert(!Ops.empty() && "Cannot get empty mul!");

  // Sort by complexity, this groups all similar expression types together.
  std::sort(Ops.begin(), Ops.end(), SCEVComplexityCompare());

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {

    // C1*(C2+V) -> C1*C2 + C1*V
    if (Ops.size() == 2)
      if (SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Ops[1]))
        if (Add->getNumOperands() == 2 &&
            isa<SCEVConstant>(Add->getOperand(0)))
          return SCEVAddExpr::get(SCEVMulExpr::get(LHSC, Add->getOperand(0)),
                                  SCEVMulExpr::get(LHSC, Add->getOperand(1)));


    ++Idx;
    while (SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      Constant *Fold = ConstantExpr::getMul(LHSC->getValue(), RHSC->getValue());
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Fold)) {
        Ops[0] = SCEVConstant::get(CI);
        Ops.erase(Ops.begin()+1);  // Erase the folded element
        if (Ops.size() == 1) return Ops[0];
      } else {
        // If we couldn't fold the expression, move to the next constant.  Note
        // that this is impossible to happen in practice because we always
        // constant fold constant ints to constant ints.
        ++Idx;
      }
    }

    // If we are left with a constant one being multiplied, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->equalsInt(1)) {
      Ops.erase(Ops.begin());
      --Idx;
    } else if (cast<SCEVConstant>(Ops[0])->getValue()->isNullValue()) {
      // If we have a multiply of zero, it will always be zero.
      return Ops[0];
    }
  }

  // Skip over the add expression until we get to a multiply.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scMulExpr)
    ++Idx;

  if (Ops.size() == 1)
    return Ops[0];
  
  // If there are mul operands inline them all into this expression.
  if (Idx < Ops.size()) {
    bool DeletedMul = false;
    while (SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(Ops[Idx])) {
      // If we have an mul, expand the mul operands onto the end of the operands
      // list.
      Ops.insert(Ops.end(), Mul->op_begin(), Mul->op_end());
      Ops.erase(Ops.begin()+Idx);
      DeletedMul = true;
    }

    // If we deleted at least one mul, we added operands to the end of the list,
    // and they are not necessarily sorted.  Recurse to resort and resimplify
    // any operands we just aquired.
    if (DeletedMul)
      return get(Ops);
  }

  // If there are any add recurrences in the operands list, see if any other
  // added values are loop invariant.  If so, we can fold them into the
  // recurrence.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scAddRecExpr)
    ++Idx;

  // Scan over all recurrences, trying to fold loop invariants into them.
  for (; Idx < Ops.size() && isa<SCEVAddRecExpr>(Ops[Idx]); ++Idx) {
    // Scan all of the other operands to this mul and add them to the vector if
    // they are loop invariant w.r.t. the recurrence.
    std::vector<SCEVHandle> LIOps;
    SCEVAddRecExpr *AddRec = cast<SCEVAddRecExpr>(Ops[Idx]);
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      if (Ops[i]->isLoopInvariant(AddRec->getLoop())) {
        LIOps.push_back(Ops[i]);
        Ops.erase(Ops.begin()+i);
        --i; --e;
      }

    // If we found some loop invariants, fold them into the recurrence.
    if (!LIOps.empty()) {
      //  NLI * LI * { Start,+,Step}  -->  NLI * { LI*Start,+,LI*Step }
      std::vector<SCEVHandle> NewOps;
      NewOps.reserve(AddRec->getNumOperands());
      if (LIOps.size() == 1) {
        SCEV *Scale = LIOps[0];
        for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i)
          NewOps.push_back(SCEVMulExpr::get(Scale, AddRec->getOperand(i)));
      } else {
        for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i) {
          std::vector<SCEVHandle> MulOps(LIOps);
          MulOps.push_back(AddRec->getOperand(i));
          NewOps.push_back(SCEVMulExpr::get(MulOps));
        }
      }

      SCEVHandle NewRec = SCEVAddRecExpr::get(NewOps, AddRec->getLoop());

      // If all of the other operands were loop invariant, we are done.
      if (Ops.size() == 1) return NewRec;

      // Otherwise, multiply the folded AddRec by the non-liv parts.
      for (unsigned i = 0;; ++i)
        if (Ops[i] == AddRec) {
          Ops[i] = NewRec;
          break;
        }
      return SCEVMulExpr::get(Ops);
    }

    // Okay, if there weren't any loop invariants to be folded, check to see if
    // there are multiple AddRec's with the same loop induction variable being
    // multiplied together.  If so, we can fold them.
    for (unsigned OtherIdx = Idx+1;
         OtherIdx < Ops.size() && isa<SCEVAddRecExpr>(Ops[OtherIdx]);++OtherIdx)
      if (OtherIdx != Idx) {
        SCEVAddRecExpr *OtherAddRec = cast<SCEVAddRecExpr>(Ops[OtherIdx]);
        if (AddRec->getLoop() == OtherAddRec->getLoop()) {
          // F * G  -->  {A,+,B} * {C,+,D}  -->  {A*C,+,F*D + G*B + B*D}
          SCEVAddRecExpr *F = AddRec, *G = OtherAddRec;
          SCEVHandle NewStart = SCEVMulExpr::get(F->getStart(),
                                                 G->getStart());
          SCEVHandle B = F->getStepRecurrence();
          SCEVHandle D = G->getStepRecurrence();
          SCEVHandle NewStep = SCEVAddExpr::get(SCEVMulExpr::get(F, D),
                                                SCEVMulExpr::get(G, B),
                                                SCEVMulExpr::get(B, D));
          SCEVHandle NewAddRec = SCEVAddRecExpr::get(NewStart, NewStep,
                                                     F->getLoop());
          if (Ops.size() == 2) return NewAddRec;

          Ops.erase(Ops.begin()+Idx);
          Ops.erase(Ops.begin()+OtherIdx-1);
          Ops.push_back(NewAddRec);
          return SCEVMulExpr::get(Ops);
        }
      }

    // Otherwise couldn't fold anything into this recurrence.  Move onto the
    // next one.
  }

  // Okay, it looks like we really DO need an mul expr.  Check to see if we
  // already have one, otherwise create a new one.
  std::vector<SCEV*> SCEVOps(Ops.begin(), Ops.end());
  SCEVCommutativeExpr *&Result = SCEVCommExprs[std::make_pair(scMulExpr,
                                                              SCEVOps)];
  if (Result == 0) Result = new SCEVMulExpr(Ops);
  return Result;
}

SCEVHandle SCEVUDivExpr::get(const SCEVHandle &LHS, const SCEVHandle &RHS) {
  if (SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS)) {
    if (RHSC->getValue()->equalsInt(1))
      return LHS;                            // X /u 1 --> x
    if (RHSC->getValue()->isAllOnesValue())
      return getNegativeSCEV(LHS);           // X /u -1  -->  -x

    if (SCEVConstant *LHSC = dyn_cast<SCEVConstant>(LHS)) {
      Constant *LHSCV = LHSC->getValue();
      Constant *RHSCV = RHSC->getValue();
      if (LHSCV->getType()->isSigned())
        LHSCV = ConstantExpr::getCast(LHSCV,
                                      LHSCV->getType()->getUnsignedVersion());
      if (RHSCV->getType()->isSigned())
        RHSCV = ConstantExpr::getCast(RHSCV, LHSCV->getType());
      return SCEVUnknown::get(ConstantExpr::getDiv(LHSCV, RHSCV));
    }
  }

  // FIXME: implement folding of (X*4)/4 when we know X*4 doesn't overflow.

  SCEVUDivExpr *&Result = SCEVUDivs[std::make_pair(LHS, RHS)];
  if (Result == 0) Result = new SCEVUDivExpr(LHS, RHS);
  return Result;
}


/// SCEVAddRecExpr::get - Get a add recurrence expression for the
/// specified loop.  Simplify the expression as much as possible.
SCEVHandle SCEVAddRecExpr::get(const SCEVHandle &Start,
                               const SCEVHandle &Step, const Loop *L) {
  std::vector<SCEVHandle> Operands;
  Operands.push_back(Start);
  if (SCEVAddRecExpr *StepChrec = dyn_cast<SCEVAddRecExpr>(Step))
    if (StepChrec->getLoop() == L) {
      Operands.insert(Operands.end(), StepChrec->op_begin(),
                      StepChrec->op_end());
      return get(Operands, L);
    }

  Operands.push_back(Step);
  return get(Operands, L);
}

/// SCEVAddRecExpr::get - Get a add recurrence expression for the
/// specified loop.  Simplify the expression as much as possible.
SCEVHandle SCEVAddRecExpr::get(std::vector<SCEVHandle> &Operands,
                               const Loop *L) {
  if (Operands.size() == 1) return Operands[0];

  if (SCEVConstant *StepC = dyn_cast<SCEVConstant>(Operands.back()))
    if (StepC->getValue()->isNullValue()) {
      Operands.pop_back();
      return get(Operands, L);             // { X,+,0 }  -->  X
    }

  SCEVAddRecExpr *&Result =
    SCEVAddRecExprs[std::make_pair(L, std::vector<SCEV*>(Operands.begin(),
                                                         Operands.end()))];
  if (Result == 0) Result = new SCEVAddRecExpr(Operands, L);
  return Result;
}


//===----------------------------------------------------------------------===//
//                  Non-trivial closed-form SCEV Expanders
//===----------------------------------------------------------------------===//

Value *SCEVTruncateExpr::expandCodeFor(ScalarEvolutionRewriter &SER,
                                       Instruction *InsertPt) {
  Value *V = SER.ExpandCodeFor(getOperand(), InsertPt);
  return new CastInst(V, getType(), "tmp.", InsertPt);
}

Value *SCEVZeroExtendExpr::expandCodeFor(ScalarEvolutionRewriter &SER,
                                         Instruction *InsertPt) {
  Value *V = SER.ExpandCodeFor(getOperand(), InsertPt,
                               getOperand()->getType()->getUnsignedVersion());
  return new CastInst(V, getType(), "tmp.", InsertPt);
}

Value *SCEVAddExpr::expandCodeFor(ScalarEvolutionRewriter &SER,
                                  Instruction *InsertPt) {
  const Type *Ty = getType();
  Value *V = SER.ExpandCodeFor(getOperand(getNumOperands()-1), InsertPt, Ty);

  // Emit a bunch of add instructions
  for (int i = getNumOperands()-2; i >= 0; --i)
    V = BinaryOperator::create(Instruction::Add, V,
                               SER.ExpandCodeFor(getOperand(i), InsertPt, Ty),
                               "tmp.", InsertPt);
  return V;
}

Value *SCEVMulExpr::expandCodeFor(ScalarEvolutionRewriter &SER,
                                  Instruction *InsertPt) {
  const Type *Ty = getType();
  int FirstOp = 0;  // Set if we should emit a subtract.
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(getOperand(0)))
    if (SC->getValue()->isAllOnesValue())
      FirstOp = 1;

  int i = getNumOperands()-2;
  Value *V = SER.ExpandCodeFor(getOperand(i+1), InsertPt, Ty);

  // Emit a bunch of multiply instructions
  for (; i >= FirstOp; --i)
    V = BinaryOperator::create(Instruction::Mul, V,
                               SER.ExpandCodeFor(getOperand(i), InsertPt, Ty),
                               "tmp.", InsertPt);
  // -1 * ...  --->  0 - ...
  if (FirstOp == 1)
    V = BinaryOperator::create(Instruction::Sub, Constant::getNullValue(Ty), V,
                               "tmp.", InsertPt);
  return V;
}

Value *SCEVUDivExpr::expandCodeFor(ScalarEvolutionRewriter &SER,
                                   Instruction *InsertPt) {
  const Type *Ty = getType();
  Value *LHS = SER.ExpandCodeFor(getLHS(), InsertPt, Ty);
  Value *RHS = SER.ExpandCodeFor(getRHS(), InsertPt, Ty);
  return BinaryOperator::create(Instruction::Div, LHS, RHS, "tmp.", InsertPt);
}

Value *SCEVAddRecExpr::expandCodeFor(ScalarEvolutionRewriter &SER,
                                     Instruction *InsertPt) {
  const Type *Ty = getType();
  // We cannot yet do fp recurrences, e.g. the xform of {X,+,F} --> X+{0,+,F}
  assert(Ty->isIntegral() && "Cannot expand fp recurrences yet!");

  // {X,+,F} --> X + {0,+,F}
  if (!isa<SCEVConstant>(getStart()) ||
      !cast<SCEVConstant>(getStart())->getValue()->isNullValue()) {
    Value *Start = SER.ExpandCodeFor(getStart(), InsertPt, Ty);
    std::vector<SCEVHandle> NewOps(op_begin(), op_end());
    NewOps[0] = getIntegerSCEV(0, getType());
    Value *Rest = SER.ExpandCodeFor(SCEVAddRecExpr::get(NewOps, getLoop()),
                                    InsertPt, getType());

    // FIXME: look for an existing add to use.
    return BinaryOperator::create(Instruction::Add, Rest, Start, "tmp.",
                                  InsertPt);
  }

  // {0,+,1} --> Insert a canonical induction variable into the loop!
  if (getNumOperands() == 2 && getOperand(1) == getIntegerSCEV(1, getType())) {
    // Create and insert the PHI node for the induction variable in the
    // specified loop.
    BasicBlock *Header = getLoop()->getHeader();
    PHINode *PN = new PHINode(Ty, "indvar", Header->begin());
    PN->addIncoming(Constant::getNullValue(Ty), L->getLoopPreheader());

    // Insert a unit add instruction after the PHI nodes in the header block.
    BasicBlock::iterator I = PN;
    while (isa<PHINode>(I)) ++I;

    Constant *One = Ty->isFloatingPoint() ?(Constant*)ConstantFP::get(Ty, 1.0)
      :(Constant*)ConstantInt::get(Ty, 1);
    Instruction *Add = BinaryOperator::create(Instruction::Add, PN, One,
                                              "indvar.next", I);

    pred_iterator PI = pred_begin(Header);
    if (*PI == L->getLoopPreheader())
      ++PI;
    PN->addIncoming(Add, *PI);
    return PN;
  }

  // Get the canonical induction variable I for this loop.
  Value *I = SER.GetOrInsertCanonicalInductionVariable(getLoop(), Ty);

  if (getNumOperands() == 2) {   // {0,+,F} --> i*F
    Value *F = SER.ExpandCodeFor(getOperand(1), InsertPt, Ty);
    return BinaryOperator::create(Instruction::Mul, I, F, "tmp.", InsertPt);
  }

  // If this is a chain of recurrences, turn it into a closed form, using the
  // folders, then expandCodeFor the closed form.  This allows the folders to
  // simplify the expression without having to build a bunch of special code
  // into this folder.
  SCEVHandle IH = SCEVUnknown::get(I);   // Get I as a "symbolic" SCEV.

  SCEVHandle V = evaluateAtIteration(IH);
  //std::cerr << "Evaluated: " << *this << "\n     to: " << *V << "\n";

  return SER.ExpandCodeFor(V, InsertPt, Ty);
}


//===----------------------------------------------------------------------===//
//             ScalarEvolutionsImpl Definition and Implementation
//===----------------------------------------------------------------------===//
//
/// ScalarEvolutionsImpl - This class implements the main driver for the scalar
/// evolution code.
///
namespace {
  struct ScalarEvolutionsImpl {
    /// F - The function we are analyzing.
    ///
    Function &F;

    /// LI - The loop information for the function we are currently analyzing.
    ///
    LoopInfo &LI;

    /// UnknownValue - This SCEV is used to represent unknown trip counts and
    /// things.
    SCEVHandle UnknownValue;

    /// Scalars - This is a cache of the scalars we have analyzed so far.
    ///
    std::map<Value*, SCEVHandle> Scalars;

    /// IterationCounts - Cache the iteration count of the loops for this
    /// function as they are computed.
    std::map<const Loop*, SCEVHandle> IterationCounts;

  public:
    ScalarEvolutionsImpl(Function &f, LoopInfo &li)
      : F(f), LI(li), UnknownValue(new SCEVCouldNotCompute()) {}

    /// getSCEV - Return an existing SCEV if it exists, otherwise analyze the
    /// expression and create a new one.
    SCEVHandle getSCEV(Value *V);

    /// getSCEVAtScope - Compute the value of the specified expression within
    /// the indicated loop (which may be null to indicate in no loop).  If the
    /// expression cannot be evaluated, return UnknownValue itself.
    SCEVHandle getSCEVAtScope(SCEV *V, const Loop *L);


    /// hasLoopInvariantIterationCount - Return true if the specified loop has
    /// an analyzable loop-invariant iteration count.
    bool hasLoopInvariantIterationCount(const Loop *L);

    /// getIterationCount - If the specified loop has a predictable iteration
    /// count, return it.  Note that it is not valid to call this method on a
    /// loop without a loop-invariant iteration count.
    SCEVHandle getIterationCount(const Loop *L);

    /// deleteInstructionFromRecords - This method should be called by the
    /// client before it removes an instruction from the program, to make sure
    /// that no dangling references are left around.
    void deleteInstructionFromRecords(Instruction *I);

  private:
    /// createSCEV - We know that there is no SCEV for the specified value.
    /// Analyze the expression.
    SCEVHandle createSCEV(Value *V);
    SCEVHandle createNodeForCast(CastInst *CI);

    /// createNodeForPHI - Provide the special handling we need to analyze PHI
    /// SCEVs.
    SCEVHandle createNodeForPHI(PHINode *PN);
    void UpdatePHIUserScalarEntries(Instruction *I, PHINode *PN,
                                    std::set<Instruction*> &UpdatedInsts);

    /// ComputeIterationCount - Compute the number of times the specified loop
    /// will iterate.
    SCEVHandle ComputeIterationCount(const Loop *L);

    /// HowFarToZero - Return the number of times a backedge comparing the
    /// specified value to zero will execute.  If not computable, return
    /// UnknownValue
    SCEVHandle HowFarToZero(SCEV *V, const Loop *L);

    /// HowFarToNonZero - Return the number of times a backedge checking the
    /// specified value for nonzero will execute.  If not computable, return
    /// UnknownValue
    SCEVHandle HowFarToNonZero(SCEV *V, const Loop *L);
  };
}

//===----------------------------------------------------------------------===//
//            Basic SCEV Analysis and PHI Idiom Recognition Code
//

/// deleteInstructionFromRecords - This method should be called by the
/// client before it removes an instruction from the program, to make sure
/// that no dangling references are left around.
void ScalarEvolutionsImpl::deleteInstructionFromRecords(Instruction *I) {
  Scalars.erase(I);
}


/// getSCEV - Return an existing SCEV if it exists, otherwise analyze the
/// expression and create a new one.
SCEVHandle ScalarEvolutionsImpl::getSCEV(Value *V) {
  assert(V->getType() != Type::VoidTy && "Can't analyze void expressions!");

  std::map<Value*, SCEVHandle>::iterator I = Scalars.find(V);
  if (I != Scalars.end()) return I->second;
  SCEVHandle S = createSCEV(V);
  Scalars.insert(std::make_pair(V, S));
  return S;
}


/// UpdatePHIUserScalarEntries - After PHI node analysis, we have a bunch of
/// entries in the scalar map that refer to the "symbolic" PHI value instead of
/// the recurrence value.  After we resolve the PHI we must loop over all of the
/// using instructions that have scalar map entries and update them.
void ScalarEvolutionsImpl::UpdatePHIUserScalarEntries(Instruction *I,
                                                      PHINode *PN,
                                        std::set<Instruction*> &UpdatedInsts) {
  std::map<Value*, SCEVHandle>::iterator SI = Scalars.find(I);
  if (SI == Scalars.end()) return;   // This scalar wasn't previous processed.
  if (UpdatedInsts.insert(I).second) {
    Scalars.erase(SI);                 // Remove the old entry
    getSCEV(I);                        // Calculate the new entry
    
    for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
         UI != E; ++UI)
      UpdatePHIUserScalarEntries(cast<Instruction>(*UI), PN, UpdatedInsts);
  }
}


/// createNodeForPHI - PHI nodes have two cases.  Either the PHI node exists in
/// a loop header, making it a potential recurrence, or it doesn't.
///
SCEVHandle ScalarEvolutionsImpl::createNodeForPHI(PHINode *PN) {
  if (PN->getNumIncomingValues() == 2)  // The loops have been canonicalized.
    if (const Loop *L = LI.getLoopFor(PN->getParent()))
      if (L->getHeader() == PN->getParent()) {
        // If it lives in the loop header, it has two incoming values, one
        // from outside the loop, and one from inside.
        unsigned IncomingEdge = L->contains(PN->getIncomingBlock(0));
        unsigned BackEdge     = IncomingEdge^1;
        
        // While we are analyzing this PHI node, handle its value symbolically.
        SCEVHandle SymbolicName = SCEVUnknown::get(PN);
        assert(Scalars.find(PN) == Scalars.end() &&
               "PHI node already processed?");
        Scalars.insert(std::make_pair(PN, SymbolicName));

        // Using this symbolic name for the PHI, analyze the value coming around
        // the back-edge.
        SCEVHandle BEValue = getSCEV(PN->getIncomingValue(BackEdge));

        // NOTE: If BEValue is loop invariant, we know that the PHI node just
        // has a special value for the first iteration of the loop.

        // If the value coming around the backedge is an add with the symbolic
        // value we just inserted, then we found a simple induction variable!
        if (SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(BEValue)) {
          // If there is a single occurrence of the symbolic value, replace it
          // with a recurrence.
          unsigned FoundIndex = Add->getNumOperands();
          for (unsigned i = 0, e = Add->getNumOperands(); i != e; ++i)
            if (Add->getOperand(i) == SymbolicName)
              if (FoundIndex == e) {
                FoundIndex = i;
                break;
              }

          if (FoundIndex != Add->getNumOperands()) {
            // Create an add with everything but the specified operand.
            std::vector<SCEVHandle> Ops;
            for (unsigned i = 0, e = Add->getNumOperands(); i != e; ++i)
              if (i != FoundIndex)
                Ops.push_back(Add->getOperand(i));
            SCEVHandle Accum = SCEVAddExpr::get(Ops);

            // This is not a valid addrec if the step amount is varying each
            // loop iteration, but is not itself an addrec in this loop.
            if (Accum->isLoopInvariant(L) ||
                (isa<SCEVAddRecExpr>(Accum) &&
                 cast<SCEVAddRecExpr>(Accum)->getLoop() == L)) {
              SCEVHandle StartVal = getSCEV(PN->getIncomingValue(IncomingEdge));
              SCEVHandle PHISCEV  = SCEVAddRecExpr::get(StartVal, Accum, L);

              // Okay, for the entire analysis of this edge we assumed the PHI
              // to be symbolic.  We now need to go back and update all of the
              // entries for the scalars that use the PHI (except for the PHI
              // itself) to use the new analyzed value instead of the "symbolic"
              // value.
              Scalars.find(PN)->second = PHISCEV;       // Update the PHI value
              std::set<Instruction*> UpdatedInsts;
              UpdatedInsts.insert(PN);
              for (Value::use_iterator UI = PN->use_begin(), E = PN->use_end();
                   UI != E; ++UI)
                UpdatePHIUserScalarEntries(cast<Instruction>(*UI), PN,
                                           UpdatedInsts);
              return PHISCEV;
            }
          }
        }

        return SymbolicName;
      }
  
  // If it's not a loop phi, we can't handle it yet.
  return SCEVUnknown::get(PN);
}

/// createNodeForCast - Handle the various forms of casts that we support.
///
SCEVHandle ScalarEvolutionsImpl::createNodeForCast(CastInst *CI) {
  const Type *SrcTy = CI->getOperand(0)->getType();
  const Type *DestTy = CI->getType();
  
  // If this is a noop cast (ie, conversion from int to uint), ignore it.
  if (SrcTy->isLosslesslyConvertibleTo(DestTy))
    return getSCEV(CI->getOperand(0));
  
  if (SrcTy->isInteger() && DestTy->isInteger()) {
    // Otherwise, if this is a truncating integer cast, we can represent this
    // cast.
    if (SrcTy->getPrimitiveSize() > DestTy->getPrimitiveSize())
      return SCEVTruncateExpr::get(getSCEV(CI->getOperand(0)),
                                   CI->getType()->getUnsignedVersion());
    if (SrcTy->isUnsigned() &&
        SrcTy->getPrimitiveSize() > DestTy->getPrimitiveSize())
      return SCEVZeroExtendExpr::get(getSCEV(CI->getOperand(0)),
                                     CI->getType()->getUnsignedVersion());
  }

  // If this is an sign or zero extending cast and we can prove that the value
  // will never overflow, we could do similar transformations.

  // Otherwise, we can't handle this cast!
  return SCEVUnknown::get(CI);
}


/// createSCEV - We know that there is no SCEV for the specified value.
/// Analyze the expression.
///
SCEVHandle ScalarEvolutionsImpl::createSCEV(Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    switch (I->getOpcode()) {
    case Instruction::Add:
      return SCEVAddExpr::get(getSCEV(I->getOperand(0)),
                              getSCEV(I->getOperand(1)));
    case Instruction::Mul:
      return SCEVMulExpr::get(getSCEV(I->getOperand(0)),
                              getSCEV(I->getOperand(1)));
    case Instruction::Div:
      if (V->getType()->isInteger() && V->getType()->isUnsigned())
        return SCEVUDivExpr::get(getSCEV(I->getOperand(0)),
                                 getSCEV(I->getOperand(1)));
      break;

    case Instruction::Sub:
      return getMinusSCEV(getSCEV(I->getOperand(0)), getSCEV(I->getOperand(1)));

    case Instruction::Shl:
      // Turn shift left of a constant amount into a multiply.
      if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
        Constant *X = ConstantInt::get(V->getType(), 1);
        X = ConstantExpr::getShl(X, SA);
        return SCEVMulExpr::get(getSCEV(I->getOperand(0)), getSCEV(X));
      }
      break;

    case Instruction::Shr:
      if (ConstantUInt *SA = dyn_cast<ConstantUInt>(I->getOperand(1)))
        if (V->getType()->isUnsigned()) {
          Constant *X = ConstantInt::get(V->getType(), 1);
          X = ConstantExpr::getShl(X, SA);
          return SCEVUDivExpr::get(getSCEV(I->getOperand(0)), getSCEV(X));
        }
      break;

    case Instruction::Cast:
      return createNodeForCast(cast<CastInst>(I));

    case Instruction::PHI:
      return createNodeForPHI(cast<PHINode>(I));

    default: // We cannot analyze this expression.
      break;
    }
  }

  return SCEVUnknown::get(V);
}



//===----------------------------------------------------------------------===//
//                   Iteration Count Computation Code
//

/// getIterationCount - If the specified loop has a predictable iteration
/// count, return it.  Note that it is not valid to call this method on a
/// loop without a loop-invariant iteration count.
SCEVHandle ScalarEvolutionsImpl::getIterationCount(const Loop *L) {
  std::map<const Loop*, SCEVHandle>::iterator I = IterationCounts.find(L);
  if (I == IterationCounts.end()) {
    SCEVHandle ItCount = ComputeIterationCount(L);
    I = IterationCounts.insert(std::make_pair(L, ItCount)).first;
    if (ItCount != UnknownValue) {
      assert(ItCount->isLoopInvariant(L) &&
             "Computed trip count isn't loop invariant for loop!");
      ++NumTripCountsComputed;
    } else if (isa<PHINode>(L->getHeader()->begin())) {
      // Only count loops that have phi nodes as not being computable.
      ++NumTripCountsNotComputed;
    }
  }
  return I->second;
}

/// ComputeIterationCount - Compute the number of times the specified loop
/// will iterate.
SCEVHandle ScalarEvolutionsImpl::ComputeIterationCount(const Loop *L) {
  // If the loop has a non-one exit block count, we can't analyze it.
  if (L->getExitBlocks().size() != 1) return UnknownValue;

  // Okay, there is one exit block.  Try to find the condition that causes the
  // loop to be exited.
  BasicBlock *ExitBlock = L->getExitBlocks()[0];

  BasicBlock *ExitingBlock = 0;
  for (pred_iterator PI = pred_begin(ExitBlock), E = pred_end(ExitBlock);
       PI != E; ++PI)
    if (L->contains(*PI)) {
      if (ExitingBlock == 0)
        ExitingBlock = *PI;
      else
        return UnknownValue;   // More than one block exiting!
    }
  assert(ExitingBlock && "No exits from loop, something is broken!");

  // Okay, we've computed the exiting block.  See what condition causes us to
  // exit.
  //
  // FIXME: we should be able to handle switch instructions (with a single exit)
  // FIXME: We should handle cast of int to bool as well
  BranchInst *ExitBr = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
  if (ExitBr == 0) return UnknownValue;
  assert(ExitBr->isConditional() && "If unconditional, it can't be in loop!");
  SetCondInst *ExitCond = dyn_cast<SetCondInst>(ExitBr->getCondition());
  if (ExitCond == 0) return UnknownValue;

  SCEVHandle LHS = getSCEV(ExitCond->getOperand(0));
  SCEVHandle RHS = getSCEV(ExitCond->getOperand(1));

  // Try to evaluate any dependencies out of the loop.
  SCEVHandle Tmp = getSCEVAtScope(LHS, L);
  if (!isa<SCEVCouldNotCompute>(Tmp)) LHS = Tmp;
  Tmp = getSCEVAtScope(RHS, L);
  if (!isa<SCEVCouldNotCompute>(Tmp)) RHS = Tmp;

  // If the condition was exit on true, convert the condition to exit on false.
  Instruction::BinaryOps Cond;
  if (ExitBr->getSuccessor(1) == ExitBlock)
    Cond = ExitCond->getOpcode();
  else
    Cond = ExitCond->getInverseCondition();

  // At this point, we would like to compute how many iterations of the loop the
  // predicate will return true for these inputs.
  if (isa<SCEVConstant>(LHS) && !isa<SCEVConstant>(RHS)) {
    // If there is a constant, force it into the RHS.
    std::swap(LHS, RHS);
    Cond = SetCondInst::getSwappedCondition(Cond);
  }

  // FIXME: think about handling pointer comparisons!  i.e.:
  // while (P != P+100) ++P;

  // If we have a comparison of a chrec against a constant, try to use value
  // ranges to answer this query.
  if (SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS))
    if (SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(LHS))
      if (AddRec->getLoop() == L) {
        // Form the comparison range using the constant of the correct type so
        // that the ConstantRange class knows to do a signed or unsigned
        // comparison.
        ConstantInt *CompVal = RHSC->getValue();
        const Type *RealTy = ExitCond->getOperand(0)->getType();
        CompVal = dyn_cast<ConstantInt>(ConstantExpr::getCast(CompVal, RealTy));
        if (CompVal) {
          // Form the constant range.
          ConstantRange CompRange(Cond, CompVal);
          
          // Now that we have it, if it's signed, convert it to an unsigned
          // range.
          if (CompRange.getLower()->getType()->isSigned()) {
            const Type *NewTy = RHSC->getValue()->getType();
            Constant *NewL = ConstantExpr::getCast(CompRange.getLower(), NewTy);
            Constant *NewU = ConstantExpr::getCast(CompRange.getUpper(), NewTy);
            CompRange = ConstantRange(NewL, NewU);
          }
          
          SCEVHandle Ret = AddRec->getNumIterationsInRange(CompRange);
          if (!isa<SCEVCouldNotCompute>(Ret)) return Ret;
        }
      }
  
  switch (Cond) {
  case Instruction::SetNE:                     // while (X != Y)
    // Convert to: while (X-Y != 0)
    if (LHS->getType()->isInteger())
      return HowFarToZero(getMinusSCEV(LHS, RHS), L);
    break;
  case Instruction::SetEQ:
    // Convert to: while (X-Y == 0)           // while (X == Y)
    if (LHS->getType()->isInteger())
      return HowFarToNonZero(getMinusSCEV(LHS, RHS), L);
    break;
  default:
#if 0
    std::cerr << "ComputeIterationCount ";
    if (ExitCond->getOperand(0)->getType()->isUnsigned())
      std::cerr << "[unsigned] ";
    std::cerr << *LHS << "   "
              << Instruction::getOpcodeName(Cond) << "   " << *RHS << "\n";
#endif
    break;
  }
  return UnknownValue;
}

/// getSCEVAtScope - Compute the value of the specified expression within the
/// indicated loop (which may be null to indicate in no loop).  If the
/// expression cannot be evaluated, return UnknownValue.
SCEVHandle ScalarEvolutionsImpl::getSCEVAtScope(SCEV *V, const Loop *L) {
  // FIXME: this should be turned into a virtual method on SCEV!

  if (isa<SCEVConstant>(V) || isa<SCEVUnknown>(V)) return V;
  if (SCEVCommutativeExpr *Comm = dyn_cast<SCEVCommutativeExpr>(V)) {
    // Avoid performing the look-up in the common case where the specified
    // expression has no loop-variant portions.
    for (unsigned i = 0, e = Comm->getNumOperands(); i != e; ++i) {
      SCEVHandle OpAtScope = getSCEVAtScope(Comm->getOperand(i), L);
      if (OpAtScope != Comm->getOperand(i)) {
        if (OpAtScope == UnknownValue) return UnknownValue;
        // Okay, at least one of these operands is loop variant but might be
        // foldable.  Build a new instance of the folded commutative expression.
        std::vector<SCEVHandle> NewOps(Comm->op_begin(), Comm->op_begin()+i-1);
        NewOps.push_back(OpAtScope);

        for (++i; i != e; ++i) {
          OpAtScope = getSCEVAtScope(Comm->getOperand(i), L);
          if (OpAtScope == UnknownValue) return UnknownValue;
          NewOps.push_back(OpAtScope);
        }
        if (isa<SCEVAddExpr>(Comm))
          return SCEVAddExpr::get(NewOps);
        assert(isa<SCEVMulExpr>(Comm) && "Only know about add and mul!");
        return SCEVMulExpr::get(NewOps);
      }
    }
    // If we got here, all operands are loop invariant.
    return Comm;
  }

  if (SCEVUDivExpr *UDiv = dyn_cast<SCEVUDivExpr>(V)) {
    SCEVHandle LHS = getSCEVAtScope(UDiv->getLHS(), L);
    if (LHS == UnknownValue) return LHS;
    SCEVHandle RHS = getSCEVAtScope(UDiv->getRHS(), L);
    if (RHS == UnknownValue) return RHS;
    if (LHS == UDiv->getLHS() && RHS == UDiv->getRHS())
      return UDiv;   // must be loop invariant
    return SCEVUDivExpr::get(LHS, RHS);
  }

  // If this is a loop recurrence for a loop that does not contain L, then we
  // are dealing with the final value computed by the loop.
  if (SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(V)) {
    if (!L || !AddRec->getLoop()->contains(L->getHeader())) {
      // To evaluate this recurrence, we need to know how many times the AddRec
      // loop iterates.  Compute this now.
      SCEVHandle IterationCount = getIterationCount(AddRec->getLoop());
      if (IterationCount == UnknownValue) return UnknownValue;
      IterationCount = getTruncateOrZeroExtend(IterationCount,
                                               AddRec->getType());
      
      // If the value is affine, simplify the expression evaluation to just
      // Start + Step*IterationCount.
      if (AddRec->isAffine())
        return SCEVAddExpr::get(AddRec->getStart(),
                                SCEVMulExpr::get(IterationCount,
                                                 AddRec->getOperand(1)));

      // Otherwise, evaluate it the hard way.
      return AddRec->evaluateAtIteration(IterationCount);
    }
    return UnknownValue;
  }

  //assert(0 && "Unknown SCEV type!");
  return UnknownValue;
}


/// SolveQuadraticEquation - Find the roots of the quadratic equation for the
/// given quadratic chrec {L,+,M,+,N}.  This returns either the two roots (which
/// might be the same) or two SCEVCouldNotCompute objects.
///
static std::pair<SCEVHandle,SCEVHandle>
SolveQuadraticEquation(const SCEVAddRecExpr *AddRec) {
  assert(AddRec->getNumOperands() == 3 && "This is not a quadratic chrec!");
  SCEVConstant *L = dyn_cast<SCEVConstant>(AddRec->getOperand(0));
  SCEVConstant *M = dyn_cast<SCEVConstant>(AddRec->getOperand(1));
  SCEVConstant *N = dyn_cast<SCEVConstant>(AddRec->getOperand(2));
  
  // We currently can only solve this if the coefficients are constants.
  if (!L || !M || !N) {
    SCEV *CNC = new SCEVCouldNotCompute();
    return std::make_pair(CNC, CNC);
  }

  Constant *Two = ConstantInt::get(L->getValue()->getType(), 2);
  
  // Convert from chrec coefficients to polynomial coefficients AX^2+BX+C
  Constant *C = L->getValue();
  // The B coefficient is M-N/2
  Constant *B = ConstantExpr::getSub(M->getValue(),
                                     ConstantExpr::getDiv(N->getValue(),
                                                          Two));
  // The A coefficient is N/2
  Constant *A = ConstantExpr::getDiv(N->getValue(), Two);
        
  // Compute the B^2-4ac term.
  Constant *SqrtTerm =
    ConstantExpr::getMul(ConstantInt::get(C->getType(), 4),
                         ConstantExpr::getMul(A, C));
  SqrtTerm = ConstantExpr::getSub(ConstantExpr::getMul(B, B), SqrtTerm);

  // Compute floor(sqrt(B^2-4ac))
  ConstantUInt *SqrtVal =
    cast<ConstantUInt>(ConstantExpr::getCast(SqrtTerm,
                                   SqrtTerm->getType()->getUnsignedVersion()));
  uint64_t SqrtValV = SqrtVal->getValue();
  uint64_t SqrtValV2 = (uint64_t)sqrt(SqrtValV);
  // The square root might not be precise for arbitrary 64-bit integer
  // values.  Do some sanity checks to ensure it's correct.
  if (SqrtValV2*SqrtValV2 > SqrtValV ||
      (SqrtValV2+1)*(SqrtValV2+1) <= SqrtValV) {
    SCEV *CNC = new SCEVCouldNotCompute();
    return std::make_pair(CNC, CNC);
  }

  SqrtVal = ConstantUInt::get(Type::ULongTy, SqrtValV2);
  SqrtTerm = ConstantExpr::getCast(SqrtVal, SqrtTerm->getType());
  
  Constant *NegB = ConstantExpr::getNeg(B);
  Constant *TwoA = ConstantExpr::getMul(A, Two);
  
  // The divisions must be performed as signed divisions.
  const Type *SignedTy = NegB->getType()->getSignedVersion();
  NegB = ConstantExpr::getCast(NegB, SignedTy);
  TwoA = ConstantExpr::getCast(TwoA, SignedTy);
  SqrtTerm = ConstantExpr::getCast(SqrtTerm, SignedTy);
  
  Constant *Solution1 =
    ConstantExpr::getDiv(ConstantExpr::getAdd(NegB, SqrtTerm), TwoA);
  Constant *Solution2 =
    ConstantExpr::getDiv(ConstantExpr::getSub(NegB, SqrtTerm), TwoA);
  return std::make_pair(SCEVUnknown::get(Solution1),
                        SCEVUnknown::get(Solution2));
}

/// HowFarToZero - Return the number of times a backedge comparing the specified
/// value to zero will execute.  If not computable, return UnknownValue
SCEVHandle ScalarEvolutionsImpl::HowFarToZero(SCEV *V, const Loop *L) {
  // If the value is a constant
  if (SCEVConstant *C = dyn_cast<SCEVConstant>(V)) {
    // If the value is already zero, the branch will execute zero times.
    if (C->getValue()->isNullValue()) return C;
    return UnknownValue;  // Otherwise it will loop infinitely.
  }

  SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(V);
  if (!AddRec || AddRec->getLoop() != L)
    return UnknownValue;

  if (AddRec->isAffine()) {
    // If this is an affine expression the execution count of this branch is
    // equal to:
    //
    //     (0 - Start/Step)    iff   Start % Step == 0
    //
    // Get the initial value for the loop.
    SCEVHandle Start = getSCEVAtScope(AddRec->getStart(), L->getParentLoop());
    SCEVHandle Step = AddRec->getOperand(1);

    Step = getSCEVAtScope(Step, L->getParentLoop());

    // Figure out if Start % Step == 0.
    // FIXME: We should add DivExpr and RemExpr operations to our AST.
    if (SCEVConstant *StepC = dyn_cast<SCEVConstant>(Step)) {
      if (StepC->getValue()->equalsInt(1))      // N % 1 == 0
        return getNegativeSCEV(Start);  // 0 - Start/1 == -Start
      if (StepC->getValue()->isAllOnesValue())  // N % -1 == 0
        return Start;                   // 0 - Start/-1 == Start

      // Check to see if Start is divisible by SC with no remainder.
      if (SCEVConstant *StartC = dyn_cast<SCEVConstant>(Start)) {
        ConstantInt *StartCC = StartC->getValue();
        Constant *StartNegC = ConstantExpr::getNeg(StartCC);
        Constant *Rem = ConstantExpr::getRem(StartNegC, StepC->getValue());
        if (Rem->isNullValue()) {
          Constant *Result =ConstantExpr::getDiv(StartNegC,StepC->getValue());
          return SCEVUnknown::get(Result);
        }
      }
    }
  } else if (AddRec->isQuadratic() && AddRec->getType()->isInteger()) {
    // If this is a quadratic (3-term) AddRec {L,+,M,+,N}, find the roots of
    // the quadratic equation to solve it.
    std::pair<SCEVHandle,SCEVHandle> Roots = SolveQuadraticEquation(AddRec);
    SCEVConstant *R1 = dyn_cast<SCEVConstant>(Roots.first);
    SCEVConstant *R2 = dyn_cast<SCEVConstant>(Roots.second);
    if (R1) {
#if 0
      std::cerr << "HFTZ: " << *V << " - sol#1: " << *R1
                << "  sol#2: " << *R2 << "\n";
#endif
      // Pick the smallest positive root value.
      assert(R1->getType()->isUnsigned()&&"Didn't canonicalize to unsigned?");
      if (ConstantBool *CB =
          dyn_cast<ConstantBool>(ConstantExpr::getSetLT(R1->getValue(),
                                                        R2->getValue()))) {
        if (CB != ConstantBool::True)
          std::swap(R1, R2);   // R1 is the minimum root now.
          
        // We can only use this value if the chrec ends up with an exact zero
        // value at this index.  When solving for "X*X != 5", for example, we
        // should not accept a root of 2.
        SCEVHandle Val = AddRec->evaluateAtIteration(R1);
        if (SCEVConstant *EvalVal = dyn_cast<SCEVConstant>(Val))
          if (EvalVal->getValue()->isNullValue())
            return R1;  // We found a quadratic root!
      }
    }
  }
  
  return UnknownValue;
}

/// HowFarToNonZero - Return the number of times a backedge checking the
/// specified value for nonzero will execute.  If not computable, return
/// UnknownValue
SCEVHandle ScalarEvolutionsImpl::HowFarToNonZero(SCEV *V, const Loop *L) {
  // Loops that look like: while (X == 0) are very strange indeed.  We don't
  // handle them yet except for the trivial case.  This could be expanded in the
  // future as needed.
 
  // If the value is a constant, check to see if it is known to be non-zero
  // already.  If so, the backedge will execute zero times.
  if (SCEVConstant *C = dyn_cast<SCEVConstant>(V)) {
    Constant *Zero = Constant::getNullValue(C->getValue()->getType());
    Constant *NonZero = ConstantExpr::getSetNE(C->getValue(), Zero);
    if (NonZero == ConstantBool::True)
      return getSCEV(Zero);
    return UnknownValue;  // Otherwise it will loop infinitely.
  }
  
  // We could implement others, but I really doubt anyone writes loops like
  // this, and if they did, they would already be constant folded.
  return UnknownValue;
}

static ConstantInt *
EvaluateConstantChrecAtConstant(const SCEVAddRecExpr *AddRec, Constant *C) {
  SCEVHandle InVal = SCEVConstant::get(cast<ConstantInt>(C));
  SCEVHandle Val = AddRec->evaluateAtIteration(InVal);
  assert(isa<SCEVConstant>(Val) &&
         "Evaluation of SCEV at constant didn't fold correctly?");
  return cast<SCEVConstant>(Val)->getValue();
}


/// getNumIterationsInRange - Return the number of iterations of this loop that
/// produce values in the specified constant range.  Another way of looking at
/// this is that it returns the first iteration number where the value is not in
/// the condition, thus computing the exit count. If the iteration count can't
/// be computed, an instance of SCEVCouldNotCompute is returned.
SCEVHandle SCEVAddRecExpr::getNumIterationsInRange(ConstantRange Range) const {
  if (Range.isFullSet())  // Infinite loop.
    return new SCEVCouldNotCompute();

  // If the start is a non-zero constant, shift the range to simplify things.
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(getStart()))
    if (!SC->getValue()->isNullValue()) {
      std::vector<SCEVHandle> Operands(op_begin(), op_end());
      Operands[0] = getIntegerSCEV(0, SC->getType());
      SCEVHandle Shifted = SCEVAddRecExpr::get(Operands, getLoop());
      if (SCEVAddRecExpr *ShiftedAddRec = dyn_cast<SCEVAddRecExpr>(Shifted))
        return ShiftedAddRec->getNumIterationsInRange(
                                              Range.subtract(SC->getValue()));
      // This is strange and shouldn't happen.
      return new SCEVCouldNotCompute();
    }

  // The only time we can solve this is when we have all constant indices.
  // Otherwise, we cannot determine the overflow conditions.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (!isa<SCEVConstant>(getOperand(i)))
      return new SCEVCouldNotCompute();


  // Okay at this point we know that all elements of the chrec are constants and
  // that the start element is zero.

  // First check to see if the range contains zero.  If not, the first
  // iteration exits.
  ConstantInt *Zero = ConstantInt::get(getType(), 0);
  if (!Range.contains(Zero)) return SCEVConstant::get(Zero);
  
  if (isAffine()) {
    // If this is an affine expression then we have this situation:
    //   Solve {0,+,A} in Range  ===  Ax in Range

    // Since we know that zero is in the range, we know that the upper value of
    // the range must be the first possible exit value.  Also note that we
    // already checked for a full range.
    ConstantInt *Upper = cast<ConstantInt>(Range.getUpper());
    ConstantInt *A     = cast<SCEVConstant>(getOperand(1))->getValue();
    ConstantInt *One   = ConstantInt::get(getType(), 1);

    // The exit value should be (Upper+A-1)/A.
    Constant *ExitValue = Upper;
    if (A != One) {
      ExitValue = ConstantExpr::getSub(ConstantExpr::getAdd(Upper, A), One);
      ExitValue = ConstantExpr::getDiv(ExitValue, A);
    }
    assert(isa<ConstantInt>(ExitValue) &&
           "Constant folding of integers not implemented?");

    // Evaluate at the exit value.  If we really did fall out of the valid
    // range, then we computed our trip count, otherwise wrap around or other
    // things must have happened.
    ConstantInt *Val = EvaluateConstantChrecAtConstant(this, ExitValue);
    if (Range.contains(Val))
      return new SCEVCouldNotCompute();  // Something strange happened

    // Ensure that the previous value is in the range.  This is a sanity check.
    assert(Range.contains(EvaluateConstantChrecAtConstant(this,
                              ConstantExpr::getSub(ExitValue, One))) &&
           "Linear scev computation is off in a bad way!");
    return SCEVConstant::get(cast<ConstantInt>(ExitValue));
  } else if (isQuadratic()) {
    // If this is a quadratic (3-term) AddRec {L,+,M,+,N}, find the roots of the
    // quadratic equation to solve it.  To do this, we must frame our problem in
    // terms of figuring out when zero is crossed, instead of when
    // Range.getUpper() is crossed.
    std::vector<SCEVHandle> NewOps(op_begin(), op_end());
    NewOps[0] = getNegativeSCEV(SCEVUnknown::get(Range.getUpper()));
    SCEVHandle NewAddRec = SCEVAddRecExpr::get(NewOps, getLoop());

    // Next, solve the constructed addrec
    std::pair<SCEVHandle,SCEVHandle> Roots =
      SolveQuadraticEquation(cast<SCEVAddRecExpr>(NewAddRec));
    SCEVConstant *R1 = dyn_cast<SCEVConstant>(Roots.first);
    SCEVConstant *R2 = dyn_cast<SCEVConstant>(Roots.second);
    if (R1) {
      // Pick the smallest positive root value.
      assert(R1->getType()->isUnsigned() && "Didn't canonicalize to unsigned?");
      if (ConstantBool *CB =
          dyn_cast<ConstantBool>(ConstantExpr::getSetLT(R1->getValue(),
                                                        R2->getValue()))) {
        if (CB != ConstantBool::True)
          std::swap(R1, R2);   // R1 is the minimum root now.
          
        // Make sure the root is not off by one.  The returned iteration should
        // not be in the range, but the previous one should be.  When solving
        // for "X*X < 5", for example, we should not return a root of 2.
        ConstantInt *R1Val = EvaluateConstantChrecAtConstant(this,
                                                             R1->getValue());
        if (Range.contains(R1Val)) {
          // The next iteration must be out of the range...
          Constant *NextVal =
            ConstantExpr::getAdd(R1->getValue(),
                                 ConstantInt::get(R1->getType(), 1));
          
          R1Val = EvaluateConstantChrecAtConstant(this, NextVal);
          if (!Range.contains(R1Val))
            return SCEVUnknown::get(NextVal);
          return new SCEVCouldNotCompute();  // Something strange happened
        }
   
        // If R1 was not in the range, then it is a good return value.  Make
        // sure that R1-1 WAS in the range though, just in case.
        Constant *NextVal =
          ConstantExpr::getSub(R1->getValue(),
                               ConstantInt::get(R1->getType(), 1));
        R1Val = EvaluateConstantChrecAtConstant(this, NextVal);
        if (Range.contains(R1Val))
          return R1;
        return new SCEVCouldNotCompute();  // Something strange happened
      }
    }
  }

  // Fallback, if this is a general polynomial, figure out the progression
  // through brute force: evaluate until we find an iteration that fails the
  // test.  This is likely to be slow, but getting an accurate trip count is
  // incredibly important, we will be able to simplify the exit test a lot, and
  // we are almost guaranteed to get a trip count in this case.
  ConstantInt *TestVal = ConstantInt::get(getType(), 0);
  ConstantInt *One     = ConstantInt::get(getType(), 1);
  ConstantInt *EndVal  = TestVal;  // Stop when we wrap around.
  do {
    ++NumBruteForceEvaluations;
    SCEVHandle Val = evaluateAtIteration(SCEVConstant::get(TestVal));
    if (!isa<SCEVConstant>(Val))  // This shouldn't happen.
      return new SCEVCouldNotCompute();

    // Check to see if we found the value!
    if (!Range.contains(cast<SCEVConstant>(Val)->getValue()))
      return SCEVConstant::get(TestVal);

    // Increment to test the next index.
    TestVal = cast<ConstantInt>(ConstantExpr::getAdd(TestVal, One));
  } while (TestVal != EndVal);
  
  return new SCEVCouldNotCompute();
}



//===----------------------------------------------------------------------===//
//                   ScalarEvolution Class Implementation
//===----------------------------------------------------------------------===//

bool ScalarEvolution::runOnFunction(Function &F) {
  Impl = new ScalarEvolutionsImpl(F, getAnalysis<LoopInfo>());
  return false;
}

void ScalarEvolution::releaseMemory() {
  delete (ScalarEvolutionsImpl*)Impl;
  Impl = 0;
}

void ScalarEvolution::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredID(LoopSimplifyID);
  AU.addRequiredTransitive<LoopInfo>();
}

SCEVHandle ScalarEvolution::getSCEV(Value *V) const {
  return ((ScalarEvolutionsImpl*)Impl)->getSCEV(V);
}

SCEVHandle ScalarEvolution::getIterationCount(const Loop *L) const {
  return ((ScalarEvolutionsImpl*)Impl)->getIterationCount(L);
}

bool ScalarEvolution::hasLoopInvariantIterationCount(const Loop *L) const {
  return !isa<SCEVCouldNotCompute>(getIterationCount(L));
}

SCEVHandle ScalarEvolution::getSCEVAtScope(Value *V, const Loop *L) const {
  return ((ScalarEvolutionsImpl*)Impl)->getSCEVAtScope(getSCEV(V), L);
}

void ScalarEvolution::deleteInstructionFromRecords(Instruction *I) const {
  return ((ScalarEvolutionsImpl*)Impl)->deleteInstructionFromRecords(I);
}


/// shouldSubstituteIndVar - Return true if we should perform induction variable
/// substitution for this variable.  This is a hack because we don't have a
/// strength reduction pass yet.  When we do we will promote all vars, because
/// we can strength reduce them later as desired.
bool ScalarEvolution::shouldSubstituteIndVar(const SCEV *S) const {
  // Don't substitute high degree polynomials.
  if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(S))
    if (AddRec->getNumOperands() > 3) return false;
  return true;
}


static void PrintLoopInfo(std::ostream &OS, const ScalarEvolution *SE, 
                          const Loop *L) {
  // Print all inner loops first
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    PrintLoopInfo(OS, SE, *I);
  
  std::cerr << "Loop " << L->getHeader()->getName() << ": ";
  if (L->getExitBlocks().size() != 1)
    std::cerr << "<multiple exits> ";

  if (SE->hasLoopInvariantIterationCount(L)) {
    std::cerr << *SE->getIterationCount(L) << " iterations! ";
  } else {
    std::cerr << "Unpredictable iteration count. ";
  }

  std::cerr << "\n";
}

void ScalarEvolution::print(std::ostream &OS) const {
  Function &F = ((ScalarEvolutionsImpl*)Impl)->F;
  LoopInfo &LI = ((ScalarEvolutionsImpl*)Impl)->LI;

  OS << "Classifying expressions for: " << F.getName() << "\n";
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
    if ((*I)->getType()->isInteger()) {
      OS << **I;
      OS << "  --> ";
      SCEVHandle SV = getSCEV(*I);
      SV->print(OS);
      OS << "\t\t";
      
      if ((*I)->getType()->isIntegral()) {
        ConstantRange Bounds = SV->getValueRange();
        if (!Bounds.isFullSet())
          OS << "Bounds: " << Bounds << " ";
      }

      if (const Loop *L = LI.getLoopFor((*I)->getParent())) {
        OS << "Exits: ";
        SCEVHandle ExitValue = getSCEVAtScope(*I, L->getParentLoop());
        if (isa<SCEVCouldNotCompute>(ExitValue)) {
          OS << "<<Unknown>>";
        } else {
          OS << *ExitValue;
        }
      }


      OS << "\n";
    }

  OS << "Determining loop execution counts for: " << F.getName() << "\n";
  for (LoopInfo::iterator I = LI.begin(), E = LI.end(); I != E; ++I)
    PrintLoopInfo(OS, this, *I);
}

//===----------------------------------------------------------------------===//
//                ScalarEvolutionRewriter Class Implementation
//===----------------------------------------------------------------------===//

Value *ScalarEvolutionRewriter::
GetOrInsertCanonicalInductionVariable(const Loop *L, const Type *Ty) {
  assert((Ty->isInteger() || Ty->isFloatingPoint()) &&
         "Can only insert integer or floating point induction variables!");

  // Check to see if we already inserted one.
  SCEVHandle H = SCEVAddRecExpr::get(getIntegerSCEV(0, Ty),
                                     getIntegerSCEV(1, Ty), L);
  return ExpandCodeFor(H, 0, Ty);
}

/// ExpandCodeFor - Insert code to directly compute the specified SCEV
/// expression into the program.  The inserted code is inserted into the
/// specified block.
Value *ScalarEvolutionRewriter::ExpandCodeFor(SCEVHandle SH,
                                              Instruction *InsertPt,
                                              const Type *Ty) {
  std::map<SCEVHandle, Value*>::iterator ExistVal =InsertedExpressions.find(SH);
  Value *V;
  if (ExistVal != InsertedExpressions.end()) {
    V = ExistVal->second;
  } else {
    // Ask the recurrence object to expand the code for itself.
    V = SH->expandCodeFor(*this, InsertPt);
    // Cache the generated result.
    InsertedExpressions.insert(std::make_pair(SH, V));
  }

  if (Ty == 0 || V->getType() == Ty)
    return V;
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(C, Ty);
  else if (Instruction *I = dyn_cast<Instruction>(V)) {
    // FIXME: check to see if there is already a cast!
    BasicBlock::iterator IP = I; ++IP;
    if (InvokeInst *II = dyn_cast<InvokeInst>(I))
      IP = II->getNormalDest()->begin();
    while (isa<PHINode>(IP)) ++IP;
    return new CastInst(V, Ty, V->getName(), IP);
  } else {
    // FIXME: check to see if there is already a cast!
    return new CastInst(V, Ty, V->getName(), InsertPt);
  }
}
