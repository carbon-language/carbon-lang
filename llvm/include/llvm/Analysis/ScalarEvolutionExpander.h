//===---- llvm/Analysis/ScalarEvolutionExpander.h - SCEV Exprs --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the classes used to generate code from scalar expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTION_EXPANDER_H
#define LLVM_ANALYSIS_SCALAREVOLUTION_EXPANDER_H

#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/CFG.h"

namespace llvm {
  /// SCEVExpander - This class uses information about analyze scalars to
  /// rewrite expressions in canonical form.
  ///
  /// Clients should create an instance of this class when rewriting is needed,
  /// and destroying it when finished to allow the release of the associated
  /// memory.
  struct SCEVExpander : public SCEVVisitor<SCEVExpander, Value*> {
    ScalarEvolution &SE;
    LoopInfo &LI;
    std::map<SCEVHandle, Value*> InsertedExpressions;
    std::set<Instruction*> InsertedInstructions;

    Instruction *InsertPt;

    friend struct SCEVVisitor<SCEVExpander, Value*>;
  public:
    SCEVExpander(ScalarEvolution &se, LoopInfo &li) : SE(se), LI(li) {}

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
      assert((Ty->isInteger() || Ty->isFloatingPoint()) &&
             "Can only insert integer or floating point induction variables!");
      SCEVHandle H = SCEVAddRecExpr::get(SCEVUnknown::getIntegerSCEV(0, Ty),
                                         SCEVUnknown::getIntegerSCEV(1, Ty), L);
      return expand(H);
    }

    /// addInsertedValue - Remember the specified instruction as being the
    /// canonical form for the specified SCEV.
    void addInsertedValue(Instruction *I, SCEV *S) {
      InsertedExpressions[S] = (Value*)I;
      InsertedInstructions.insert(I);
    }

    /// expandCodeFor - Insert code to directly compute the specified SCEV
    /// expression into the program.  The inserted code is inserted into the
    /// specified block.
    ///
    /// If a particular value sign is required, a type may be specified for the
    /// result.
    Value *expandCodeFor(SCEVHandle SH, Instruction *IP, const Type *Ty = 0) {
      // Expand the code for this SCEV.
      this->InsertPt = IP;
      return expandInTy(SH, Ty);
    }

  protected:
    Value *expand(SCEV *S) {
      // Check to see if we already expanded this.
      std::map<SCEVHandle, Value*>::iterator I = InsertedExpressions.find(S);
      if (I != InsertedExpressions.end())
        return I->second;

      Value *V = visit(S);
      InsertedExpressions[S] = V;
      return V;
    }

    Value *expandInTy(SCEV *S, const Type *Ty) {
      Value *V = expand(S);
      if (Ty && V->getType() != Ty) {
        // FIXME: keep track of the cast instruction.
        if (Constant *C = dyn_cast<Constant>(V))
          return ConstantExpr::getCast(C, Ty);
        else if (Instruction *I = dyn_cast<Instruction>(V)) {
          // Check to see if there is already a cast.  If there is, use it.
          for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
               UI != E; ++UI) {
            if ((*UI)->getType() == Ty)
              if (CastInst *CI = dyn_cast<CastInst>(cast<Instruction>(*UI))) {
                BasicBlock::iterator It = I; ++It;
                if (isa<InvokeInst>(I))
                  It = cast<InvokeInst>(I)->getNormalDest()->begin();
                while (isa<PHINode>(It)) ++It;
                if (It != BasicBlock::iterator(CI)) {
                  // Splice the cast immediately after the operand in question.
                  BasicBlock::InstListType &InstList =
                    It->getParent()->getInstList();
                  InstList.splice(It, CI->getParent()->getInstList(), CI);
                }
                return CI;
              }
          }
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
      return V;
    }

    Value *visitConstant(SCEVConstant *S) {
      return S->getValue();
    }

    Value *visitTruncateExpr(SCEVTruncateExpr *S) {
      Value *V = expand(S->getOperand());
      return new CastInst(V, S->getType(), "tmp.", InsertPt);
    }

    Value *visitZeroExtendExpr(SCEVZeroExtendExpr *S) {
      Value *V = expandInTy(S->getOperand(),S->getType()->getUnsignedVersion());
      return new CastInst(V, S->getType(), "tmp.", InsertPt);
    }

    Value *visitAddExpr(SCEVAddExpr *S) {
      const Type *Ty = S->getType();
      Value *V = expandInTy(S->getOperand(S->getNumOperands()-1), Ty);

      // Emit a bunch of add instructions
      for (int i = S->getNumOperands()-2; i >= 0; --i)
        V = BinaryOperator::createAdd(V, expandInTy(S->getOperand(i), Ty),
                                      "tmp.", InsertPt);
      return V;
    }

    Value *visitMulExpr(SCEVMulExpr *S);

    Value *visitUDivExpr(SCEVUDivExpr *S) {
      const Type *Ty = S->getType();
      Value *LHS = expandInTy(S->getLHS(), Ty);
      Value *RHS = expandInTy(S->getRHS(), Ty);
      return BinaryOperator::createDiv(LHS, RHS, "tmp.", InsertPt);
    }

    Value *visitAddRecExpr(SCEVAddRecExpr *S);

    Value *visitUnknown(SCEVUnknown *S) {
      return S->getValue();
    }
  };
}

#endif

