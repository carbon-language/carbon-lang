//===---- llvm/Analysis/ScalarEvolutionExpander.h - SCEV Exprs --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the classes used to generate code from scalar expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTION_EXPANDER_H
#define LLVM_ANALYSIS_SCALAREVOLUTION_EXPANDER_H

#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

namespace llvm {
  /// SCEVExpander - This class uses information about analyze scalars to
  /// rewrite expressions in canonical form.
  ///
  /// Clients should create an instance of this class when rewriting is needed,
  /// and destroy it when finished to allow the release of the associated 
  /// memory.
  struct SCEVExpander : public SCEVVisitor<SCEVExpander, Value*> {
    ScalarEvolution &SE;
    LoopInfo &LI;
    std::map<SCEVHandle, Value*> InsertedExpressions;
    std::set<Instruction*> InsertedInstructions;

    BasicBlock::iterator InsertPt;

    friend struct SCEVVisitor<SCEVExpander, Value*>;
  public:
    SCEVExpander(ScalarEvolution &se, LoopInfo &li)
      : SE(se), LI(li) {}

    LoopInfo &getLoopInfo() const { return LI; }

    /// clear - Erase the contents of the InsertedExpressions map so that users
    /// trying to expand the same expression into multiple BasicBlocks or
    /// different places within the same BasicBlock can do so.
    void clear() { InsertedExpressions.clear(); }

    /// isInsertedInstruction - Return true if the specified instruction was
    /// inserted by the code rewriter.  If so, the client should not modify the
    /// instruction.
    bool isInsertedInstruction(Instruction *I) const {
      return InsertedInstructions.count(I);
    }

    /// getOrInsertCanonicalInductionVariable - This method returns the
    /// canonical induction variable of the specified type for the specified
    /// loop (inserting one if there is none).  A canonical induction variable
    /// starts at zero and steps by one on each iteration.
    Value *getOrInsertCanonicalInductionVariable(const Loop *L, const Type *Ty){
      assert(Ty->isInteger() && "Can only insert integer induction variables!");
      SCEVHandle H = SE.getAddRecExpr(SE.getIntegerSCEV(0, Ty),
                                      SE.getIntegerSCEV(1, Ty), L);
      return expand(H);
    }

    /// addInsertedValue - Remember the specified instruction as being the
    /// canonical form for the specified SCEV.
    void addInsertedValue(Instruction *I, const SCEV *S) {
      InsertedExpressions[S] = (Value*)I;
      InsertedInstructions.insert(I);
    }

    void setInsertionPoint(BasicBlock::iterator NewIP) { InsertPt = NewIP; }

    BasicBlock::iterator getInsertionPoint() const { return InsertPt; }

    /// expandCodeFor - Insert code to directly compute the specified SCEV
    /// expression into the program.  The inserted code is inserted into the
    /// SCEVExpander's current insertion point.
    Value *expandCodeFor(SCEVHandle SH, const Type *Ty);

    /// expandCodeFor - Insert code to directly compute the specified SCEV
    /// expression into the program.  The inserted code is inserted into the
    /// specified block.
    Value *expandCodeFor(SCEVHandle SH, const Type *Ty,
                         BasicBlock::iterator IP) {
      setInsertionPoint(IP);
      return expandCodeFor(SH, Ty);
    }

    /// InsertCastOfTo - Insert a cast of V to the specified type, doing what
    /// we can to share the casts.
    Value *InsertCastOfTo(Instruction::CastOps opcode, Value *V,
                          const Type *Ty);

    /// InsertNoopCastOfTo - Insert a cast of V to the specified type,
    /// which must be possible with a noop cast.
    Value *InsertNoopCastOfTo(Value *V, const Type *Ty);

    /// InsertBinop - Insert the specified binary operator, doing a small amount
    /// of work to avoid inserting an obviously redundant operation.
    static Value *InsertBinop(Instruction::BinaryOps Opcode, Value *LHS,
                              Value *RHS, BasicBlock::iterator InsertPt);

  private:
    Value *expand(const SCEV *S);

    Value *visitConstant(const SCEVConstant *S) {
      return S->getValue();
    }

    Value *visitTruncateExpr(const SCEVTruncateExpr *S);

    Value *visitZeroExtendExpr(const SCEVZeroExtendExpr *S);

    Value *visitSignExtendExpr(const SCEVSignExtendExpr *S);

    Value *visitAddExpr(const SCEVAddExpr *S);

    Value *visitMulExpr(const SCEVMulExpr *S);

    Value *visitUDivExpr(const SCEVUDivExpr *S);

    Value *visitAddRecExpr(const SCEVAddRecExpr *S);

    Value *visitSMaxExpr(const SCEVSMaxExpr *S);

    Value *visitUMaxExpr(const SCEVUMaxExpr *S);

    Value *visitUnknown(const SCEVUnknown *S) {
      return S->getValue();
    }
  };
}

#endif

