//===- llvm/Analysis/ScalarEvolution.h - Scalar Evolution -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The ScalarEvolution class is an LLVM pass which can be used to analyze and
// catagorize scalar expressions in loops.  It specializes in recognizing
// general induction variables, representing them with the abstract and opaque
// SCEV class.  Given this analysis, trip counts of loops and other important
// properties can be obtained.
//
// This analysis is primarily useful for induction variable substitution and
// strength reduction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTION_H
#define LLVM_ANALYSIS_SCALAREVOLUTION_H

#include "llvm/Pass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/DataTypes.h"
#include <iosfwd>

namespace llvm {
  class APInt;
  class ConstantInt;
  class Type;
  class SCEVHandle;
  class ScalarEvolution;
  class TargetData;

  /// SCEV - This class represent an analyzed expression in the program.  These
  /// are reference counted opaque objects that the client is not allowed to
  /// do much with directly.
  ///
  class SCEV {
    const unsigned SCEVType;      // The SCEV baseclass this node corresponds to
    mutable unsigned RefCount;

    friend class SCEVHandle;
    void addRef() const { ++RefCount; }
    void dropRef() const {
      if (--RefCount == 0)
        delete this;
    }

    SCEV(const SCEV &);            // DO NOT IMPLEMENT
    void operator=(const SCEV &);  // DO NOT IMPLEMENT
  protected:
    virtual ~SCEV();
  public:
    explicit SCEV(unsigned SCEVTy) : SCEVType(SCEVTy), RefCount(0) {}

    unsigned getSCEVType() const { return SCEVType; }

    /// isLoopInvariant - Return true if the value of this SCEV is unchanging in
    /// the specified loop.
    virtual bool isLoopInvariant(const Loop *L) const = 0;

    /// hasComputableLoopEvolution - Return true if this SCEV changes value in a
    /// known way in the specified loop.  This property being true implies that
    /// the value is variant in the loop AND that we can emit an expression to
    /// compute the value of the expression at any particular loop iteration.
    virtual bool hasComputableLoopEvolution(const Loop *L) const = 0;

    /// getType - Return the LLVM type of this SCEV expression.
    ///
    virtual const Type *getType() const = 0;

    /// isZero - Return true if the expression is a constant zero.
    ///
    bool isZero() const;

    /// replaceSymbolicValuesWithConcrete - If this SCEV internally references
    /// the symbolic value "Sym", construct and return a new SCEV that produces
    /// the same value, but which uses the concrete value Conc instead of the
    /// symbolic value.  If this SCEV does not use the symbolic value, it
    /// returns itself.
    virtual SCEVHandle
    replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                      const SCEVHandle &Conc,
                                      ScalarEvolution &SE) const = 0;

    /// dominates - Return true if elements that makes up this SCEV dominates
    /// the specified basic block.
    virtual bool dominates(BasicBlock *BB, DominatorTree *DT) const = 0;

    /// print - Print out the internal representation of this scalar to the
    /// specified stream.  This should really only be used for debugging
    /// purposes.
    virtual void print(raw_ostream &OS) const = 0;
    void print(std::ostream &OS) const;
    void print(std::ostream *OS) const { if (OS) print(*OS); }

    /// dump - This method is used for debugging.
    ///
    void dump() const;
  };

  inline raw_ostream &operator<<(raw_ostream &OS, const SCEV &S) {
    S.print(OS);
    return OS;
  }

  inline std::ostream &operator<<(std::ostream &OS, const SCEV &S) {
    S.print(OS);
    return OS;
  }

  /// SCEVCouldNotCompute - An object of this class is returned by queries that
  /// could not be answered.  For example, if you ask for the number of
  /// iterations of a linked-list traversal loop, you will get one of these.
  /// None of the standard SCEV operations are valid on this class, it is just a
  /// marker.
  struct SCEVCouldNotCompute : public SCEV {
    SCEVCouldNotCompute();
    ~SCEVCouldNotCompute();

    // None of these methods are valid for this object.
    virtual bool isLoopInvariant(const Loop *L) const;
    virtual const Type *getType() const;
    virtual bool hasComputableLoopEvolution(const Loop *L) const;
    virtual void print(raw_ostream &OS) const;
    virtual SCEVHandle
    replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                      const SCEVHandle &Conc,
                                      ScalarEvolution &SE) const;

    virtual bool dominates(BasicBlock *BB, DominatorTree *DT) const {
      return true;
    }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVCouldNotCompute *S) { return true; }
    static bool classof(const SCEV *S);
  };

  /// SCEVHandle - This class is used to maintain the SCEV object's refcounts,
  /// freeing the objects when the last reference is dropped.
  class SCEVHandle {
    SCEV *S;
    SCEVHandle();  // DO NOT IMPLEMENT
  public:
    SCEVHandle(const SCEV *s) : S(const_cast<SCEV*>(s)) {
      assert(S && "Cannot create a handle to a null SCEV!");
      S->addRef();
    }
    SCEVHandle(const SCEVHandle &RHS) : S(RHS.S) {
      S->addRef();
    }
    ~SCEVHandle() { S->dropRef(); }

    operator SCEV*() const { return S; }

    SCEV &operator*() const { return *S; }
    SCEV *operator->() const { return S; }

    bool operator==(SCEV *RHS) const { return S == RHS; }
    bool operator!=(SCEV *RHS) const { return S != RHS; }

    const SCEVHandle &operator=(SCEV *RHS) {
      if (S != RHS) {
        S->dropRef();
        S = RHS;
        S->addRef();
      }
      return *this;
    }

    const SCEVHandle &operator=(const SCEVHandle &RHS) {
      if (S != RHS.S) {
        S->dropRef();
        S = RHS.S;
        S->addRef();
      }
      return *this;
    }
  };

  template<typename From> struct simplify_type;
  template<> struct simplify_type<const SCEVHandle> {
    typedef SCEV* SimpleType;
    static SimpleType getSimplifiedValue(const SCEVHandle &Node) {
      return Node;
    }
  };
  template<> struct simplify_type<SCEVHandle>
    : public simplify_type<const SCEVHandle> {};

  /// ScalarEvolution - This class is the main scalar evolution driver.  Because
  /// client code (intentionally) can't do much with the SCEV objects directly,
  /// they must ask this class for services.
  ///
  class ScalarEvolution : public FunctionPass {
    /// F - The function we are analyzing.
    ///
    Function *F;

    /// LI - The loop information for the function we are currently analyzing.
    ///
    LoopInfo *LI;

    /// TD - The target data information for the target we are targetting.
    ///
    TargetData *TD;

    /// UnknownValue - This SCEV is used to represent unknown trip counts and
    /// things.
    SCEVHandle UnknownValue;

    /// Scalars - This is a cache of the scalars we have analyzed so far.
    ///
    std::map<Value*, SCEVHandle> Scalars;

    /// BackedgeTakenInfo - Information about the backedge-taken count
    /// of a loop. This currently inclues an exact count and a maximum count.
    ///
    struct BackedgeTakenInfo {
      /// Exact - An expression indicating the exact backedge-taken count of
      /// the loop if it is known, or a SCEVCouldNotCompute otherwise.
      SCEVHandle Exact;

      /// Exact - An expression indicating the least maximum backedge-taken
      /// count of the loop that is known, or a SCEVCouldNotCompute.
      SCEVHandle Max;

      /*implicit*/ BackedgeTakenInfo(SCEVHandle exact) :
        Exact(exact), Max(exact) {}

      /*implicit*/ BackedgeTakenInfo(SCEV *exact) :
        Exact(exact), Max(exact) {}

      BackedgeTakenInfo(SCEVHandle exact, SCEVHandle max) :
        Exact(exact), Max(max) {}

      /// hasAnyInfo - Test whether this BackedgeTakenInfo contains any
      /// computed information, or whether it's all SCEVCouldNotCompute
      /// values.
      bool hasAnyInfo() const {
        return !isa<SCEVCouldNotCompute>(Exact) ||
               !isa<SCEVCouldNotCompute>(Max);
      }
    };

    /// BackedgeTakenCounts - Cache the backedge-taken count of the loops for
    /// this function as they are computed.
    std::map<const Loop*, BackedgeTakenInfo> BackedgeTakenCounts;

    /// ConstantEvolutionLoopExitValue - This map contains entries for all of
    /// the PHI instructions that we attempt to compute constant evolutions for.
    /// This allows us to avoid potentially expensive recomputation of these
    /// properties.  An instruction maps to null if we are unable to compute its
    /// exit value.
    std::map<PHINode*, Constant*> ConstantEvolutionLoopExitValue;

    /// createSCEV - We know that there is no SCEV for the specified value.
    /// Analyze the expression.
    SCEVHandle createSCEV(Value *V);

    /// createNodeForPHI - Provide the special handling we need to analyze PHI
    /// SCEVs.
    SCEVHandle createNodeForPHI(PHINode *PN);

    /// ReplaceSymbolicValueWithConcrete - This looks up the computed SCEV value
    /// for the specified instruction and replaces any references to the
    /// symbolic value SymName with the specified value.  This is used during
    /// PHI resolution.
    void ReplaceSymbolicValueWithConcrete(Instruction *I,
                                          const SCEVHandle &SymName,
                                          const SCEVHandle &NewVal);

    /// getBackedgeTakenInfo - Return the BackedgeTakenInfo for the given
    /// loop, lazily computing new values if the loop hasn't been analyzed
    /// yet.
    const BackedgeTakenInfo &getBackedgeTakenInfo(const Loop *L);

    /// ComputeBackedgeTakenCount - Compute the number of times the specified
    /// loop will iterate.
    BackedgeTakenInfo ComputeBackedgeTakenCount(const Loop *L);

    /// ComputeLoadConstantCompareBackedgeTakenCount - Given an exit condition
    /// of 'icmp op load X, cst', try to see if we can compute the trip count.
    SCEVHandle
      ComputeLoadConstantCompareBackedgeTakenCount(LoadInst *LI,
                                                   Constant *RHS,
                                                   const Loop *L,
                                                   ICmpInst::Predicate p);

    /// ComputeBackedgeTakenCountExhaustively - If the trip is known to execute
    /// a constant number of times (the condition evolves only from constants),
    /// try to evaluate a few iterations of the loop until we get the exit
    /// condition gets a value of ExitWhen (true or false).  If we cannot
    /// evaluate the trip count of the loop, return UnknownValue.
    SCEVHandle ComputeBackedgeTakenCountExhaustively(const Loop *L, Value *Cond,
                                                     bool ExitWhen);

    /// HowFarToZero - Return the number of times a backedge comparing the
    /// specified value to zero will execute.  If not computable, return
    /// UnknownValue.
    SCEVHandle HowFarToZero(SCEV *V, const Loop *L);

    /// HowFarToNonZero - Return the number of times a backedge checking the
    /// specified value for nonzero will execute.  If not computable, return
    /// UnknownValue.
    SCEVHandle HowFarToNonZero(SCEV *V, const Loop *L);

    /// HowManyLessThans - Return the number of times a backedge containing the
    /// specified less-than comparison will execute.  If not computable, return
    /// UnknownValue. isSigned specifies whether the less-than is signed.
    BackedgeTakenInfo HowManyLessThans(SCEV *LHS, SCEV *RHS, const Loop *L,
                                       bool isSigned);

    /// getPredecessorWithUniqueSuccessorForBB - Return a predecessor of BB
    /// (which may not be an immediate predecessor) which has exactly one
    /// successor from which BB is reachable, or null if no such block is
    /// found.
    BasicBlock* getPredecessorWithUniqueSuccessorForBB(BasicBlock *BB);

    /// getConstantEvolutionLoopExitValue - If we know that the specified Phi is
    /// in the header of its containing loop, we know the loop executes a
    /// constant number of times, and the PHI node is just a recurrence
    /// involving constants, fold it.
    Constant *getConstantEvolutionLoopExitValue(PHINode *PN, const APInt& BEs,
                                                const Loop *L);

    /// getSCEVAtScope - Compute the value of the specified expression within
    /// the indicated loop (which may be null to indicate in no loop).  If the
    /// expression cannot be evaluated, return UnknownValue itself.
    SCEVHandle getSCEVAtScope(SCEV *S, const Loop *L);

  public:
    static char ID; // Pass identification, replacement for typeid
    ScalarEvolution();

    /// isSCEVable - Test if values of the given type are analyzable within
    /// the SCEV framework. This primarily includes integer types, and it
    /// can optionally include pointer types if the ScalarEvolution class
    /// has access to target-specific information.
    bool isSCEVable(const Type *Ty) const;

    /// getTypeSizeInBits - Return the size in bits of the specified type,
    /// for which isSCEVable must return true.
    uint64_t getTypeSizeInBits(const Type *Ty) const;

    /// getEffectiveSCEVType - Return a type with the same bitwidth as
    /// the given type and which represents how SCEV will treat the given
    /// type, for which isSCEVable must return true. For pointer types,
    /// this is the pointer-sized integer type.
    const Type *getEffectiveSCEVType(const Type *Ty) const;

    /// getSCEV - Return a SCEV expression handle for the full generality of the
    /// specified expression.
    SCEVHandle getSCEV(Value *V);

    SCEVHandle getConstant(ConstantInt *V);
    SCEVHandle getConstant(const APInt& Val);
    SCEVHandle getTruncateExpr(const SCEVHandle &Op, const Type *Ty);
    SCEVHandle getZeroExtendExpr(const SCEVHandle &Op, const Type *Ty);
    SCEVHandle getSignExtendExpr(const SCEVHandle &Op, const Type *Ty);
    SCEVHandle getAddExpr(std::vector<SCEVHandle> &Ops);
    SCEVHandle getAddExpr(const SCEVHandle &LHS, const SCEVHandle &RHS) {
      std::vector<SCEVHandle> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      return getAddExpr(Ops);
    }
    SCEVHandle getAddExpr(const SCEVHandle &Op0, const SCEVHandle &Op1,
                          const SCEVHandle &Op2) {
      std::vector<SCEVHandle> Ops;
      Ops.push_back(Op0);
      Ops.push_back(Op1);
      Ops.push_back(Op2);
      return getAddExpr(Ops);
    }
    SCEVHandle getMulExpr(std::vector<SCEVHandle> &Ops);
    SCEVHandle getMulExpr(const SCEVHandle &LHS, const SCEVHandle &RHS) {
      std::vector<SCEVHandle> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      return getMulExpr(Ops);
    }
    SCEVHandle getUDivExpr(const SCEVHandle &LHS, const SCEVHandle &RHS);
    SCEVHandle getAddRecExpr(const SCEVHandle &Start, const SCEVHandle &Step,
                             const Loop *L);
    SCEVHandle getAddRecExpr(std::vector<SCEVHandle> &Operands,
                             const Loop *L);
    SCEVHandle getAddRecExpr(const std::vector<SCEVHandle> &Operands,
                             const Loop *L) {
      std::vector<SCEVHandle> NewOp(Operands);
      return getAddRecExpr(NewOp, L);
    }
    SCEVHandle getSMaxExpr(const SCEVHandle &LHS, const SCEVHandle &RHS);
    SCEVHandle getSMaxExpr(std::vector<SCEVHandle> Operands);
    SCEVHandle getUMaxExpr(const SCEVHandle &LHS, const SCEVHandle &RHS);
    SCEVHandle getUMaxExpr(std::vector<SCEVHandle> Operands);
    SCEVHandle getUnknown(Value *V);
    SCEVHandle getCouldNotCompute();

    /// getNegativeSCEV - Return the SCEV object corresponding to -V.
    ///
    SCEVHandle getNegativeSCEV(const SCEVHandle &V);

    /// getNotSCEV - Return the SCEV object corresponding to ~V.
    ///
    SCEVHandle getNotSCEV(const SCEVHandle &V);

    /// getMinusSCEV - Return LHS-RHS.
    ///
    SCEVHandle getMinusSCEV(const SCEVHandle &LHS,
                            const SCEVHandle &RHS);

    /// getTruncateOrZeroExtend - Return a SCEV corresponding to a conversion
    /// of the input value to the specified type.  If the type must be
    /// extended, it is zero extended.
    SCEVHandle getTruncateOrZeroExtend(const SCEVHandle &V, const Type *Ty);

    /// getTruncateOrSignExtend - Return a SCEV corresponding to a conversion
    /// of the input value to the specified type.  If the type must be
    /// extended, it is sign extended.
    SCEVHandle getTruncateOrSignExtend(const SCEVHandle &V, const Type *Ty);

    /// getIntegerSCEV - Given an integer or FP type, create a constant for the
    /// specified signed integer value and return a SCEV for the constant.
    SCEVHandle getIntegerSCEV(int Val, const Type *Ty);

    /// hasSCEV - Return true if the SCEV for this value has already been
    /// computed.
    bool hasSCEV(Value *V) const;

    /// setSCEV - Insert the specified SCEV into the map of current SCEVs for
    /// the specified value.
    void setSCEV(Value *V, const SCEVHandle &H);

    /// getSCEVAtScope - Return a SCEV expression handle for the specified value
    /// at the specified scope in the program.  The L value specifies a loop
    /// nest to evaluate the expression at, where null is the top-level or a
    /// specified loop is immediately inside of the loop.
    ///
    /// This method can be used to compute the exit value for a variable defined
    /// in a loop by querying what the value will hold in the parent loop.
    ///
    /// If this value is not computable at this scope, a SCEVCouldNotCompute
    /// object is returned.
    SCEVHandle getSCEVAtScope(Value *V, const Loop *L);

    /// isLoopGuardedByCond - Test whether entry to the loop is protected by
    /// a conditional between LHS and RHS.  This is used to help avoid max
    /// expressions in loop trip counts.
    bool isLoopGuardedByCond(const Loop *L, ICmpInst::Predicate Pred,
                             SCEV *LHS, SCEV *RHS);

    /// getBackedgeTakenCount - If the specified loop has a predictable
    /// backedge-taken count, return it, otherwise return a SCEVCouldNotCompute
    /// object. The backedge-taken count is the number of times the loop header
    /// will be branched to from within the loop. This is one less than the
    /// trip count of the loop, since it doesn't count the first iteration,
    /// when the header is branched to from outside the loop.
    ///
    /// Note that it is not valid to call this method on a loop without a
    /// loop-invariant backedge-taken count (see
    /// hasLoopInvariantBackedgeTakenCount).
    ///
    SCEVHandle getBackedgeTakenCount(const Loop *L);

    /// getMaxBackedgeTakenCount - Similar to getBackedgeTakenCount, except
    /// return the least SCEV value that is known never to be less than the
    /// actual backedge taken count.
    SCEVHandle getMaxBackedgeTakenCount(const Loop *L);

    /// hasLoopInvariantBackedgeTakenCount - Return true if the specified loop
    /// has an analyzable loop-invariant backedge-taken count.
    bool hasLoopInvariantBackedgeTakenCount(const Loop *L);

    /// forgetLoopBackedgeTakenCount - This method should be called by the
    /// client when it has changed a loop in a way that may effect
    /// ScalarEvolution's ability to compute a trip count, or if the loop
    /// is deleted.
    void forgetLoopBackedgeTakenCount(const Loop *L);

    /// deleteValueFromRecords - This method should be called by the
    /// client before it removes a Value from the program, to make sure
    /// that no dangling references are left around.
    void deleteValueFromRecords(Value *V);

    virtual bool runOnFunction(Function &F);
    virtual void releaseMemory();
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    void print(raw_ostream &OS, const Module* = 0) const;
    virtual void print(std::ostream &OS, const Module* = 0) const;
    void print(std::ostream *OS, const Module* M = 0) const {
      if (OS) print(*OS, M);
    }
  };
}

#endif
