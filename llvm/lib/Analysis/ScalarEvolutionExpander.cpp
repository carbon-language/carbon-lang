//===- ScalarEvolutionExpander.cpp - Scalar Evolution Analysis --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the scalar evolution expander,
// which is used to generate the code corresponding to a given scalar evolution
// expression.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
using namespace llvm;

/// InsertCastOfTo - Insert a cast of V to the specified type, doing what
/// we can to share the casts.
Value *SCEVExpander::InsertCastOfTo(Value *V, const Type *Ty) {
  // FIXME: keep track of the cast instruction.
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(C, Ty);
  
  if (Argument *A = dyn_cast<Argument>(V)) {
    // Check to see if there is already a cast!
    for (Value::use_iterator UI = A->use_begin(), E = A->use_end();
         UI != E; ++UI) {
      if ((*UI)->getType() == Ty)
        if (CastInst *CI = dyn_cast<CastInst>(cast<Instruction>(*UI))) {
          // If the cast isn't in the first instruction of the function,
          // move it.
          if (BasicBlock::iterator(CI) != 
              A->getParent()->getEntryBlock().begin()) {
            CI->moveBefore(A->getParent()->getEntryBlock().begin());
          }
          return CI;
        }
    }
    return new CastInst(V, Ty, V->getName(),
                        A->getParent()->getEntryBlock().begin());
  }
    
  Instruction *I = cast<Instruction>(V);
  
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
          CI->moveBefore(It);
        }
        return CI;
      }
  }
  BasicBlock::iterator IP = I; ++IP;
  if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    IP = II->getNormalDest()->begin();
  while (isa<PHINode>(IP)) ++IP;
  return new CastInst(V, Ty, V->getName(), IP);
}

Value *SCEVExpander::visitMulExpr(SCEVMulExpr *S) {
  const Type *Ty = S->getType();
  int FirstOp = 0;  // Set if we should emit a subtract.
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(S->getOperand(0)))
    if (SC->getValue()->isAllOnesValue())
      FirstOp = 1;

  int i = S->getNumOperands()-2;
  Value *V = expandInTy(S->getOperand(i+1), Ty);

  // Emit a bunch of multiply instructions
  for (; i >= FirstOp; --i)
    V = BinaryOperator::createMul(V, expandInTy(S->getOperand(i), Ty),
                                  "tmp.", InsertPt);
  // -1 * ...  --->  0 - ...
  if (FirstOp == 1)
    V = BinaryOperator::createNeg(V, "tmp.", InsertPt);
  return V;
}

Value *SCEVExpander::visitAddRecExpr(SCEVAddRecExpr *S) {
  const Type *Ty = S->getType();
  const Loop *L = S->getLoop();
  // We cannot yet do fp recurrences, e.g. the xform of {X,+,F} --> X+{0,+,F}
  assert(Ty->isIntegral() && "Cannot expand fp recurrences yet!");

  // {X,+,F} --> X + {0,+,F}
  if (!isa<SCEVConstant>(S->getStart()) ||
      !cast<SCEVConstant>(S->getStart())->getValue()->isNullValue()) {
    Value *Start = expandInTy(S->getStart(), Ty);
    std::vector<SCEVHandle> NewOps(S->op_begin(), S->op_end());
    NewOps[0] = SCEVUnknown::getIntegerSCEV(0, Ty);
    Value *Rest = expandInTy(SCEVAddRecExpr::get(NewOps, L), Ty);

    // FIXME: look for an existing add to use.
    return BinaryOperator::createAdd(Rest, Start, "tmp.", InsertPt);
  }

  // {0,+,1} --> Insert a canonical induction variable into the loop!
  if (S->getNumOperands() == 2 &&
      S->getOperand(1) == SCEVUnknown::getIntegerSCEV(1, Ty)) {
    // Create and insert the PHI node for the induction variable in the
    // specified loop.
    BasicBlock *Header = L->getHeader();
    PHINode *PN = new PHINode(Ty, "indvar", Header->begin());
    PN->addIncoming(Constant::getNullValue(Ty), L->getLoopPreheader());

    pred_iterator HPI = pred_begin(Header);
    assert(HPI != pred_end(Header) && "Loop with zero preds???");
    if (!L->contains(*HPI)) ++HPI;
    assert(HPI != pred_end(Header) && L->contains(*HPI) &&
           "No backedge in loop?");

    // Insert a unit add instruction right before the terminator corresponding
    // to the back-edge.
    Constant *One = Ty->isFloatingPoint() ? (Constant*)ConstantFP::get(Ty, 1.0)
                                          : ConstantInt::get(Ty, 1);
    Instruction *Add = BinaryOperator::createAdd(PN, One, "indvar.next",
                                                 (*HPI)->getTerminator());

    pred_iterator PI = pred_begin(Header);
    if (*PI == L->getLoopPreheader())
      ++PI;
    PN->addIncoming(Add, *PI);
    return PN;
  }

  // Get the canonical induction variable I for this loop.
  Value *I = getOrInsertCanonicalInductionVariable(L, Ty);

  // If this is a simple linear addrec, emit it now as a special case.
  if (S->getNumOperands() == 2) {   // {0,+,F} --> i*F
    Value *F = expandInTy(S->getOperand(1), Ty);
    
    // IF the step is by one, just return the inserted IV.
    if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(F))
      if (CI->getRawValue() == 1)
        return I;
    
    // If the insert point is directly inside of the loop, emit the multiply at
    // the insert point.  Otherwise, L is a loop that is a parent of the insert
    // point loop.  If we can, move the multiply to the outer most loop that it
    // is safe to be in.
    Instruction *MulInsertPt = InsertPt;
    Loop *InsertPtLoop = LI.getLoopFor(MulInsertPt->getParent());
    if (InsertPtLoop != L && InsertPtLoop &&
        L->contains(InsertPtLoop->getHeader())) {
      while (InsertPtLoop != L) {
        // If we cannot hoist the multiply out of this loop, don't.
        if (!InsertPtLoop->isLoopInvariant(F)) break;

        // Otherwise, move the insert point to the preheader of the loop.
        MulInsertPt = InsertPtLoop->getLoopPreheader()->getTerminator();
        InsertPtLoop = InsertPtLoop->getParentLoop();
      }
    }
    
    return BinaryOperator::createMul(I, F, "tmp.", MulInsertPt);
  }

  // If this is a chain of recurrences, turn it into a closed form, using the
  // folders, then expandCodeFor the closed form.  This allows the folders to
  // simplify the expression without having to build a bunch of special code
  // into this folder.
  SCEVHandle IH = SCEVUnknown::get(I);   // Get I as a "symbolic" SCEV.

  SCEVHandle V = S->evaluateAtIteration(IH);
  //std::cerr << "Evaluated: " << *this << "\n     to: " << *V << "\n";

  return expandInTy(V, Ty);
}
