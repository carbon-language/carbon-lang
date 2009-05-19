//===- ScalarEvolutionExpander.cpp - Scalar Evolution Analysis --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the scalar evolution expander,
// which is used to generate the code corresponding to a given scalar evolution
// expression.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;

/// InsertCastOfTo - Insert a cast of V to the specified type, doing what
/// we can to share the casts.
Value *SCEVExpander::InsertCastOfTo(Instruction::CastOps opcode, Value *V, 
                                    const Type *Ty) {
  // Short-circuit unnecessary bitcasts.
  if (opcode == Instruction::BitCast && V->getType() == Ty)
    return V;

  // Short-circuit unnecessary inttoptr<->ptrtoint casts.
  if ((opcode == Instruction::PtrToInt || opcode == Instruction::IntToPtr) &&
      SE.getTypeSizeInBits(Ty) == SE.getTypeSizeInBits(V->getType())) {
    if (CastInst *CI = dyn_cast<CastInst>(V))
      if ((CI->getOpcode() == Instruction::PtrToInt ||
           CI->getOpcode() == Instruction::IntToPtr) &&
          SE.getTypeSizeInBits(CI->getType()) ==
          SE.getTypeSizeInBits(CI->getOperand(0)->getType()))
        return CI->getOperand(0);
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      if ((CE->getOpcode() == Instruction::PtrToInt ||
           CE->getOpcode() == Instruction::IntToPtr) &&
          SE.getTypeSizeInBits(CE->getType()) ==
          SE.getTypeSizeInBits(CE->getOperand(0)->getType()))
        return CE->getOperand(0);
  }

  // FIXME: keep track of the cast instruction.
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(opcode, C, Ty);
  
  if (Argument *A = dyn_cast<Argument>(V)) {
    // Check to see if there is already a cast!
    for (Value::use_iterator UI = A->use_begin(), E = A->use_end();
         UI != E; ++UI) {
      if ((*UI)->getType() == Ty)
        if (CastInst *CI = dyn_cast<CastInst>(cast<Instruction>(*UI)))
          if (CI->getOpcode() == opcode) {
            // If the cast isn't the first instruction of the function, move it.
            if (BasicBlock::iterator(CI) != 
                A->getParent()->getEntryBlock().begin()) {
              // If the CastInst is the insert point, change the insert point.
              if (CI == InsertPt) ++InsertPt;
              // Splice the cast at the beginning of the entry block.
              CI->moveBefore(A->getParent()->getEntryBlock().begin());
            }
            return CI;
          }
    }
    Instruction *I = CastInst::Create(opcode, V, Ty, V->getName(),
                                      A->getParent()->getEntryBlock().begin());
    InsertedValues.insert(I);
    return I;
  }

  Instruction *I = cast<Instruction>(V);

  // Check to see if there is already a cast.  If there is, use it.
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI) {
    if ((*UI)->getType() == Ty)
      if (CastInst *CI = dyn_cast<CastInst>(cast<Instruction>(*UI)))
        if (CI->getOpcode() == opcode) {
          BasicBlock::iterator It = I; ++It;
          if (isa<InvokeInst>(I))
            It = cast<InvokeInst>(I)->getNormalDest()->begin();
          while (isa<PHINode>(It)) ++It;
          if (It != BasicBlock::iterator(CI)) {
            // If the CastInst is the insert point, change the insert point.
            if (CI == InsertPt) ++InsertPt;
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
  Instruction *CI = CastInst::Create(opcode, V, Ty, V->getName(), IP);
  InsertedValues.insert(CI);
  return CI;
}

/// InsertNoopCastOfTo - Insert a cast of V to the specified type,
/// which must be possible with a noop cast.
Value *SCEVExpander::InsertNoopCastOfTo(Value *V, const Type *Ty) {
  Instruction::CastOps Op = CastInst::getCastOpcode(V, false, Ty, false);
  assert((Op == Instruction::BitCast ||
          Op == Instruction::PtrToInt ||
          Op == Instruction::IntToPtr) &&
         "InsertNoopCastOfTo cannot perform non-noop casts!");
  assert(SE.getTypeSizeInBits(V->getType()) == SE.getTypeSizeInBits(Ty) &&
         "InsertNoopCastOfTo cannot change sizes!");
  return InsertCastOfTo(Op, V, Ty);
}

/// InsertBinop - Insert the specified binary operator, doing a small amount
/// of work to avoid inserting an obviously redundant operation.
Value *SCEVExpander::InsertBinop(Instruction::BinaryOps Opcode, Value *LHS,
                                 Value *RHS, BasicBlock::iterator InsertPt) {
  // Fold a binop with constant operands.
  if (Constant *CLHS = dyn_cast<Constant>(LHS))
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantExpr::get(Opcode, CLHS, CRHS);

  // Do a quick scan to see if we have this binop nearby.  If so, reuse it.
  unsigned ScanLimit = 6;
  BasicBlock::iterator BlockBegin = InsertPt->getParent()->begin();
  if (InsertPt != BlockBegin) {
    // Scanning starts from the last instruction before InsertPt.
    BasicBlock::iterator IP = InsertPt;
    --IP;
    for (; ScanLimit; --IP, --ScanLimit) {
      if (IP->getOpcode() == (unsigned)Opcode && IP->getOperand(0) == LHS &&
          IP->getOperand(1) == RHS)
        return IP;
      if (IP == BlockBegin) break;
    }
  }
  
  // If we haven't found this binop, insert it.
  Instruction *BO = BinaryOperator::Create(Opcode, LHS, RHS, "tmp", InsertPt);
  InsertedValues.insert(BO);
  return BO;
}

/// expandAddToGEP - Expand a SCEVAddExpr with a pointer type into a GEP
/// instead of using ptrtoint+arithmetic+inttoptr.
Value *SCEVExpander::expandAddToGEP(const SCEVAddExpr *S,
                                    const PointerType *PTy,
                                    const Type *Ty,
                                    Value *V) {
  const Type *ElTy = PTy->getElementType();
  SmallVector<Value *, 4> GepIndices;
  std::vector<SCEVHandle> Ops = S->getOperands();
  bool AnyNonZeroIndices = false;
  Ops.pop_back();

  // Decend down the pointer's type and attempt to convert the other
  // operands into GEP indices, at each level. The first index in a GEP
  // indexes into the array implied by the pointer operand; the rest of
  // the indices index into the element or field type selected by the
  // preceding index.
  for (;;) {
    APInt ElSize = APInt(SE.getTypeSizeInBits(Ty),
                         ElTy->isSized() ?  SE.TD->getTypeAllocSize(ElTy) : 0);
    std::vector<SCEVHandle> NewOps;
    std::vector<SCEVHandle> ScaledOps;
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      if (ElSize != 0) {
        if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Ops[i]))
          if (!C->getValue()->getValue().srem(ElSize)) {
            ConstantInt *CI =
              ConstantInt::get(C->getValue()->getValue().sdiv(ElSize));
            SCEVHandle Div = SE.getConstant(CI);
            ScaledOps.push_back(Div);
            continue;
          }
        if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(Ops[i]))
          if (const SCEVConstant *C = dyn_cast<SCEVConstant>(M->getOperand(0)))
            if (C->getValue()->getValue() == ElSize) {
              for (unsigned j = 1, f = M->getNumOperands(); j != f; ++j)
                ScaledOps.push_back(M->getOperand(j));
              continue;
            }
        if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(Ops[i]))
          if (BinaryOperator *BO = dyn_cast<BinaryOperator>(U->getValue()))
            if (BO->getOpcode() == Instruction::Mul)
              if (ConstantInt *CI = dyn_cast<ConstantInt>(BO->getOperand(1)))
                if (CI->getValue() == ElSize) {
                  ScaledOps.push_back(SE.getUnknown(BO->getOperand(0)));
                  continue;
                }
        if (ElSize == 1) {
          ScaledOps.push_back(Ops[i]);
          continue;
        }
      }
      NewOps.push_back(Ops[i]);
    }
    Ops = NewOps;
    AnyNonZeroIndices |= !ScaledOps.empty();
    Value *Scaled = ScaledOps.empty() ?
                    Constant::getNullValue(Ty) :
                    expandCodeFor(SE.getAddExpr(ScaledOps), Ty);
    GepIndices.push_back(Scaled);

    // Collect struct field index operands.
    if (!Ops.empty())
      while (const StructType *STy = dyn_cast<StructType>(ElTy)) {
        if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Ops[0]))
          if (SE.getTypeSizeInBits(C->getType()) <= 64) {
            const StructLayout &SL = *SE.TD->getStructLayout(STy);
            uint64_t FullOffset = C->getValue()->getZExtValue();
            if (FullOffset < SL.getSizeInBytes()) {
              unsigned ElIdx = SL.getElementContainingOffset(FullOffset);
              GepIndices.push_back(ConstantInt::get(Type::Int32Ty, ElIdx));
              ElTy = STy->getTypeAtIndex(ElIdx);
              Ops[0] =
                SE.getConstant(ConstantInt::get(Ty,
                                                FullOffset -
                                                  SL.getElementOffset(ElIdx)));
              AnyNonZeroIndices = true;
              continue;
            }
          }
        break;
      }

    if (const ArrayType *ATy = dyn_cast<ArrayType>(ElTy)) {
      ElTy = ATy->getElementType();
      continue;
    }
    break;
  }

  // If none of the operands were convertable to proper GEP indices, cast
  // the base to i8* and do an ugly getelementptr with that. It's still
  // better than ptrtoint+arithmetic+inttoptr at least.
  if (!AnyNonZeroIndices) {
    V = InsertNoopCastOfTo(V,
                           Type::Int8Ty->getPointerTo(PTy->getAddressSpace()));
    Value *Idx = expand(SE.getAddExpr(Ops));
    Idx = InsertNoopCastOfTo(Idx, Ty);

    // Fold a GEP with constant operands.
    if (Constant *CLHS = dyn_cast<Constant>(V))
      if (Constant *CRHS = dyn_cast<Constant>(Idx))
        return ConstantExpr::get(Instruction::GetElementPtr, CLHS, CRHS);

    // Do a quick scan to see if we have this GEP nearby.  If so, reuse it.
    unsigned ScanLimit = 6;
    BasicBlock::iterator BlockBegin = InsertPt->getParent()->begin();
    if (InsertPt != BlockBegin) {
      // Scanning starts from the last instruction before InsertPt.
      BasicBlock::iterator IP = InsertPt;
      --IP;
      for (; ScanLimit; --IP, --ScanLimit) {
        if (IP->getOpcode() == Instruction::GetElementPtr &&
            IP->getOperand(0) == V && IP->getOperand(1) == Idx)
          return IP;
        if (IP == BlockBegin) break;
      }
    }

    Value *GEP = GetElementPtrInst::Create(V, Idx, "scevgep", InsertPt);
    InsertedValues.insert(GEP);
    return GEP;
  }

  // Insert a pretty getelementptr.
  Value *GEP = GetElementPtrInst::Create(V,
                                         GepIndices.begin(),
                                         GepIndices.end(),
                                         "scevgep", InsertPt);
  Ops.push_back(SE.getUnknown(GEP));
  InsertedValues.insert(GEP);
  return expand(SE.getAddExpr(Ops));
}

Value *SCEVExpander::visitAddExpr(const SCEVAddExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *V = expand(S->getOperand(S->getNumOperands()-1));

  // Turn things like ptrtoint+arithmetic+inttoptr into GEP. This helps
  // BasicAliasAnalysis analyze the result. However, it suffers from the
  // underlying bug described in PR2831. Addition in LLVM currently always
  // has two's complement wrapping guaranteed. However, the semantics for
  // getelementptr overflow are ambiguous. In the common case though, this
  // expansion gets used when a GEP in the original code has been converted
  // into integer arithmetic, in which case the resulting code will be no
  // more undefined than it was originally.
  if (SE.TD)
    if (const PointerType *PTy = dyn_cast<PointerType>(V->getType()))
      return expandAddToGEP(S, PTy, Ty, V);

  V = InsertNoopCastOfTo(V, Ty);

  // Emit a bunch of add instructions
  for (int i = S->getNumOperands()-2; i >= 0; --i) {
    Value *W = expand(S->getOperand(i));
    W = InsertNoopCastOfTo(W, Ty);
    V = InsertBinop(Instruction::Add, V, W, InsertPt);
  }
  return V;
}

Value *SCEVExpander::visitMulExpr(const SCEVMulExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  int FirstOp = 0;  // Set if we should emit a subtract.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(S->getOperand(0)))
    if (SC->getValue()->isAllOnesValue())
      FirstOp = 1;

  int i = S->getNumOperands()-2;
  Value *V = expand(S->getOperand(i+1));
  V = InsertNoopCastOfTo(V, Ty);

  // Emit a bunch of multiply instructions
  for (; i >= FirstOp; --i) {
    Value *W = expand(S->getOperand(i));
    W = InsertNoopCastOfTo(W, Ty);
    V = InsertBinop(Instruction::Mul, V, W, InsertPt);
  }

  // -1 * ...  --->  0 - ...
  if (FirstOp == 1)
    V = InsertBinop(Instruction::Sub, Constant::getNullValue(Ty), V, InsertPt);
  return V;
}

Value *SCEVExpander::visitUDivExpr(const SCEVUDivExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());

  Value *LHS = expand(S->getLHS());
  LHS = InsertNoopCastOfTo(LHS, Ty);
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(S->getRHS())) {
    const APInt &RHS = SC->getValue()->getValue();
    if (RHS.isPowerOf2())
      return InsertBinop(Instruction::LShr, LHS,
                         ConstantInt::get(Ty, RHS.logBase2()),
                         InsertPt);
  }

  Value *RHS = expand(S->getRHS());
  RHS = InsertNoopCastOfTo(RHS, Ty);
  return InsertBinop(Instruction::UDiv, LHS, RHS, InsertPt);
}

Value *SCEVExpander::visitAddRecExpr(const SCEVAddRecExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  const Loop *L = S->getLoop();

  // {X,+,F} --> X + {0,+,F}
  if (!S->getStart()->isZero()) {
    std::vector<SCEVHandle> NewOps(S->getOperands());
    NewOps[0] = SE.getIntegerSCEV(0, Ty);
    Value *Rest = expand(SE.getAddRecExpr(NewOps, L));
    return expand(SE.getAddExpr(S->getStart(), SE.getUnknown(Rest)));
  }

  // {0,+,1} --> Insert a canonical induction variable into the loop!
  if (S->isAffine() &&
      S->getOperand(1) == SE.getIntegerSCEV(1, Ty)) {
    // Create and insert the PHI node for the induction variable in the
    // specified loop.
    BasicBlock *Header = L->getHeader();
    PHINode *PN = PHINode::Create(Ty, "indvar", Header->begin());
    InsertedValues.insert(PN);
    PN->addIncoming(Constant::getNullValue(Ty), L->getLoopPreheader());

    pred_iterator HPI = pred_begin(Header);
    assert(HPI != pred_end(Header) && "Loop with zero preds???");
    if (!L->contains(*HPI)) ++HPI;
    assert(HPI != pred_end(Header) && L->contains(*HPI) &&
           "No backedge in loop?");

    // Insert a unit add instruction right before the terminator corresponding
    // to the back-edge.
    Constant *One = ConstantInt::get(Ty, 1);
    Instruction *Add = BinaryOperator::CreateAdd(PN, One, "indvar.next",
                                                 (*HPI)->getTerminator());
    InsertedValues.insert(Add);

    pred_iterator PI = pred_begin(Header);
    if (*PI == L->getLoopPreheader())
      ++PI;
    PN->addIncoming(Add, *PI);
    return PN;
  }

  // Get the canonical induction variable I for this loop.
  Value *I = getOrInsertCanonicalInductionVariable(L, Ty);

  // If this is a simple linear addrec, emit it now as a special case.
  if (S->isAffine()) {   // {0,+,F} --> i*F
    Value *F = expand(S->getOperand(1));
    F = InsertNoopCastOfTo(F, Ty);
    
    // IF the step is by one, just return the inserted IV.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(F))
      if (CI->getValue() == 1)
        return I;
    
    // If the insert point is directly inside of the loop, emit the multiply at
    // the insert point.  Otherwise, L is a loop that is a parent of the insert
    // point loop.  If we can, move the multiply to the outer most loop that it
    // is safe to be in.
    BasicBlock::iterator MulInsertPt = getInsertionPoint();
    Loop *InsertPtLoop = SE.LI->getLoopFor(MulInsertPt->getParent());
    if (InsertPtLoop != L && InsertPtLoop &&
        L->contains(InsertPtLoop->getHeader())) {
      do {
        // If we cannot hoist the multiply out of this loop, don't.
        if (!InsertPtLoop->isLoopInvariant(F)) break;

        BasicBlock *InsertPtLoopPH = InsertPtLoop->getLoopPreheader();

        // If this loop hasn't got a preheader, we aren't able to hoist the
        // multiply.
        if (!InsertPtLoopPH)
          break;

        // Otherwise, move the insert point to the preheader.
        MulInsertPt = InsertPtLoopPH->getTerminator();
        InsertPtLoop = InsertPtLoop->getParentLoop();
      } while (InsertPtLoop != L);
    }
    
    return InsertBinop(Instruction::Mul, I, F, MulInsertPt);
  }

  // If this is a chain of recurrences, turn it into a closed form, using the
  // folders, then expandCodeFor the closed form.  This allows the folders to
  // simplify the expression without having to build a bunch of special code
  // into this folder.
  SCEVHandle IH = SE.getUnknown(I);   // Get I as a "symbolic" SCEV.

  SCEVHandle V = S->evaluateAtIteration(IH, SE);
  //cerr << "Evaluated: " << *this << "\n     to: " << *V << "\n";

  return expand(V);
}

Value *SCEVExpander::visitTruncateExpr(const SCEVTruncateExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *V = expand(S->getOperand());
  V = InsertNoopCastOfTo(V, SE.getEffectiveSCEVType(V->getType()));
  Instruction *I = new TruncInst(V, Ty, "tmp.", InsertPt);
  InsertedValues.insert(I);
  return I;
}

Value *SCEVExpander::visitZeroExtendExpr(const SCEVZeroExtendExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *V = expand(S->getOperand());
  V = InsertNoopCastOfTo(V, SE.getEffectiveSCEVType(V->getType()));
  Instruction *I = new ZExtInst(V, Ty, "tmp.", InsertPt);
  InsertedValues.insert(I);
  return I;
}

Value *SCEVExpander::visitSignExtendExpr(const SCEVSignExtendExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *V = expand(S->getOperand());
  V = InsertNoopCastOfTo(V, SE.getEffectiveSCEVType(V->getType()));
  Instruction *I = new SExtInst(V, Ty, "tmp.", InsertPt);
  InsertedValues.insert(I);
  return I;
}

Value *SCEVExpander::visitSMaxExpr(const SCEVSMaxExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *LHS = expand(S->getOperand(0));
  LHS = InsertNoopCastOfTo(LHS, Ty);
  for (unsigned i = 1; i < S->getNumOperands(); ++i) {
    Value *RHS = expand(S->getOperand(i));
    RHS = InsertNoopCastOfTo(RHS, Ty);
    Instruction *ICmp =
      new ICmpInst(ICmpInst::ICMP_SGT, LHS, RHS, "tmp", InsertPt);
    InsertedValues.insert(ICmp);
    Instruction *Sel = SelectInst::Create(ICmp, LHS, RHS, "smax", InsertPt);
    InsertedValues.insert(Sel);
    LHS = Sel;
  }
  return LHS;
}

Value *SCEVExpander::visitUMaxExpr(const SCEVUMaxExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *LHS = expand(S->getOperand(0));
  LHS = InsertNoopCastOfTo(LHS, Ty);
  for (unsigned i = 1; i < S->getNumOperands(); ++i) {
    Value *RHS = expand(S->getOperand(i));
    RHS = InsertNoopCastOfTo(RHS, Ty);
    Instruction *ICmp =
      new ICmpInst(ICmpInst::ICMP_UGT, LHS, RHS, "tmp", InsertPt);
    InsertedValues.insert(ICmp);
    Instruction *Sel = SelectInst::Create(ICmp, LHS, RHS, "umax", InsertPt);
    InsertedValues.insert(Sel);
    LHS = Sel;
  }
  return LHS;
}

Value *SCEVExpander::expandCodeFor(SCEVHandle SH, const Type *Ty) {
  // Expand the code for this SCEV.
  Value *V = expand(SH);
  if (Ty) {
    assert(SE.getTypeSizeInBits(Ty) == SE.getTypeSizeInBits(SH->getType()) &&
           "non-trivial casts should be done with the SCEVs directly!");
    V = InsertNoopCastOfTo(V, Ty);
  }
  return V;
}

Value *SCEVExpander::expand(const SCEV *S) {
  // Check to see if we already expanded this.
  std::map<SCEVHandle, Value*>::iterator I = InsertedExpressions.find(S);
  if (I != InsertedExpressions.end())
    return I->second;
  
  Value *V = visit(S);
  InsertedExpressions[S] = V;
  return V;
}
