//===- llvm/Analysis/ScalarEvolution.h - Scalar Evolution -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include <set>

namespace llvm {
  class Instruction;
  class Type;
  class ConstantRange;
  class Loop;
  class LoopInfo;
  class SCEVHandle;
  class ScalarEvolutionRewriter;

  /// SCEV - This class represent an analyzed expression in the program.  These
  /// are reference counted opaque objects that the client is not allowed to
  /// do much with directly.
  ///
  class SCEV {
    const unsigned SCEVType;      // The SCEV baseclass this node corresponds to
    unsigned RefCount;

    friend class SCEVHandle;
    void addRef() { ++RefCount; }
    void dropRef() {
      if (--RefCount == 0) {
#if 0
        std::cerr << "DELETING: " << this << ": ";
        print(std::cerr);
        std::cerr << "\n";
#endif
        delete this;
      }
    }

    SCEV(const SCEV &);            // DO NOT IMPLEMENT
    void operator=(const SCEV &);  // DO NOT IMPLEMENT
  protected:
    virtual ~SCEV();
  public:
    SCEV(unsigned SCEVTy) : SCEVType(SCEVTy), RefCount(0) {}

    unsigned getSCEVType() const { return SCEVType; }

    /// getValueRange - Return the tightest constant bounds that this value is
    /// known to have.  This method is only valid on integer SCEV objects.
    virtual ConstantRange getValueRange() const;

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

    /// expandCodeFor - Given a rewriter object, expand this SCEV into a closed
    /// form expression and return a Value corresponding to the expression in
    /// question.
    virtual Value *expandCodeFor(ScalarEvolutionRewriter &SER,
                                 Instruction *InsertPt) = 0;


    /// print - Print out the internal representation of this scalar to the
    /// specified stream.  This should really only be used for debugging
    /// purposes.
    virtual void print(std::ostream &OS) const = 0;

    /// dump - This method is used for debugging.
    ///
    void dump() const;
  };
  
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

    // None of these methods are valid for this object.
    virtual bool isLoopInvariant(const Loop *L) const;
    virtual const Type *getType() const;
    virtual bool hasComputableLoopEvolution(const Loop *L) const;
    virtual Value *expandCodeFor(ScalarEvolutionRewriter &, Instruction *);
    virtual void print(std::ostream &OS) const;


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
    SCEVHandle(SCEV *s) : S(s) {
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
    void *Impl;    // ScalarEvolution uses the pimpl pattern
  public:
    ScalarEvolution() : Impl(0) {}
    
    /// getSCEV - Return a SCEV expression handle for the full generality of the
    /// specified expression.
    SCEVHandle getSCEV(Value *V) const;

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
    SCEVHandle getSCEVAtScope(Value *V, const Loop *L) const;

    /// getIterationCount - If the specified loop has a predictable iteration
    /// count, return it, otherwise return a SCEVCouldNotCompute object.
    SCEVHandle getIterationCount(const Loop *L) const;

    /// hasLoopInvariantIterationCount - Return true if the specified loop has
    /// an analyzable loop-invariant iteration count.
    bool hasLoopInvariantIterationCount(const Loop *L) const;

    /// deleteInstructionFromRecords - This method should be called by the
    /// client before it removes an instruction from the program, to make sure
    /// that no dangling references are left around.
    void deleteInstructionFromRecords(Instruction *I) const;

    /// shouldSubstituteIndVar - Return true if we should perform induction
    /// variable substitution for this variable.  This is a hack because we
    /// don't have a strength reduction pass yet.  When we do we will promote
    /// all vars, because we can strength reduce them later as desired.
    bool shouldSubstituteIndVar(const SCEV *S) const;

    virtual bool runOnFunction(Function &F);
    virtual void releaseMemory();
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void print(std::ostream &OS) const;
  };

  /// ScalarEvolutionRewriter - This class uses information about analyze
  /// scalars to rewrite expressions in canonical form.  This can be used for
  /// induction variable substitution, strength reduction, or loop exit value
  /// replacement.
  ///
  /// Clients should create an instance of this class when rewriting is needed,
  /// and destroying it when finished to allow the release of the associated
  /// memory.
  class ScalarEvolutionRewriter {
    ScalarEvolution &SE;
    LoopInfo &LI;
    std::map<SCEVHandle, Value*> InsertedExpressions;
    std::set<Instruction*> InsertedInstructions;
  public:
    ScalarEvolutionRewriter(ScalarEvolution &se, LoopInfo &li)
      : SE(se), LI(li) {}

    /// isInsertedInstruction - Return true if the specified instruction was
    /// inserted by the code rewriter.  If so, the client should not modify the
    /// instruction.
    bool isInsertedInstruction(Instruction *I) const {
      return InsertedInstructions.count(I);
    }
    
    /// GetOrInsertCanonicalInductionVariable - This method returns the
    /// canonical induction variable of the specified type for the specified
    /// loop (inserts one if there is none).  A canonical induction variable
    /// starts at zero and steps by one on each iteration.
    Value *GetOrInsertCanonicalInductionVariable(const Loop *L, const Type *Ty);

    /// ExpandCodeFor - Insert code to directly compute the specified SCEV
    /// expression into the program.  The inserted code is inserted into the
    /// specified block.
    ///
    /// If a particular value sign is required, a type may be specified for the
    /// result.
    Value *ExpandCodeFor(SCEVHandle SH, Instruction *InsertPt,
                         const Type *Ty = 0);
  };
}

#endif
