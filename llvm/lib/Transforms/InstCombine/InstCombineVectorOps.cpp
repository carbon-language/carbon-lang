//===- InstCombineVectorOps.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements instcombine for ExtractElement, InsertElement and
// ShuffleVector.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
using namespace llvm;

/// CheapToScalarize - Return true if the value is cheaper to scalarize than it
/// is to leave as a vector operation.
static bool CheapToScalarize(Value *V, bool isConstant) {
  if (isa<ConstantAggregateZero>(V)) 
    return true;
  if (ConstantVector *C = dyn_cast<ConstantVector>(V)) {
    if (isConstant) return true;
    // If all elts are the same, we can extract.
    Constant *Op0 = C->getOperand(0);
    for (unsigned i = 1; i < C->getNumOperands(); ++i)
      if (C->getOperand(i) != Op0)
        return false;
    return true;
  }
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;
  
  // Insert element gets simplified to the inserted element or is deleted if
  // this is constant idx extract element and its a constant idx insertelt.
  if (I->getOpcode() == Instruction::InsertElement && isConstant &&
      isa<ConstantInt>(I->getOperand(2)))
    return true;
  if (I->getOpcode() == Instruction::Load && I->hasOneUse())
    return true;
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I))
    if (BO->hasOneUse() &&
        (CheapToScalarize(BO->getOperand(0), isConstant) ||
         CheapToScalarize(BO->getOperand(1), isConstant)))
      return true;
  if (CmpInst *CI = dyn_cast<CmpInst>(I))
    if (CI->hasOneUse() &&
        (CheapToScalarize(CI->getOperand(0), isConstant) ||
         CheapToScalarize(CI->getOperand(1), isConstant)))
      return true;
  
  return false;
}

/// Read and decode a shufflevector mask.
///
/// It turns undef elements into values that are larger than the number of
/// elements in the input.
static std::vector<unsigned> getShuffleMask(const ShuffleVectorInst *SVI) {
  unsigned NElts = SVI->getType()->getNumElements();
  if (isa<ConstantAggregateZero>(SVI->getOperand(2)))
    return std::vector<unsigned>(NElts, 0);
  if (isa<UndefValue>(SVI->getOperand(2)))
    return std::vector<unsigned>(NElts, 2*NElts);
  
  std::vector<unsigned> Result;
  const ConstantVector *CP = cast<ConstantVector>(SVI->getOperand(2));
  for (User::const_op_iterator i = CP->op_begin(), e = CP->op_end(); i!=e; ++i)
    if (isa<UndefValue>(*i))
      Result.push_back(NElts*2);  // undef -> 8
    else
      Result.push_back(cast<ConstantInt>(*i)->getZExtValue());
  return Result;
}

/// FindScalarElement - Given a vector and an element number, see if the scalar
/// value is already around as a register, for example if it were inserted then
/// extracted from the vector.
static Value *FindScalarElement(Value *V, unsigned EltNo) {
  assert(V->getType()->isVectorTy() && "Not looking at a vector?");
  const VectorType *PTy = cast<VectorType>(V->getType());
  unsigned Width = PTy->getNumElements();
  if (EltNo >= Width)  // Out of range access.
    return UndefValue::get(PTy->getElementType());
  
  if (isa<UndefValue>(V))
    return UndefValue::get(PTy->getElementType());
  if (isa<ConstantAggregateZero>(V))
    return Constant::getNullValue(PTy->getElementType());
  if (ConstantVector *CP = dyn_cast<ConstantVector>(V))
    return CP->getOperand(EltNo);
  
  if (InsertElementInst *III = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert to a variable element, we don't know what it is.
    if (!isa<ConstantInt>(III->getOperand(2))) 
      return 0;
    unsigned IIElt = cast<ConstantInt>(III->getOperand(2))->getZExtValue();
    
    // If this is an insert to the element we are looking for, return the
    // inserted value.
    if (EltNo == IIElt) 
      return III->getOperand(1);
    
    // Otherwise, the insertelement doesn't modify the value, recurse on its
    // vector input.
    return FindScalarElement(III->getOperand(0), EltNo);
  }
  
  if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(V)) {
    unsigned LHSWidth =
    cast<VectorType>(SVI->getOperand(0)->getType())->getNumElements();
    unsigned InEl = getShuffleMask(SVI)[EltNo];
    if (InEl < LHSWidth)
      return FindScalarElement(SVI->getOperand(0), InEl);
    else if (InEl < LHSWidth*2)
      return FindScalarElement(SVI->getOperand(1), InEl - LHSWidth);
    else
      return UndefValue::get(PTy->getElementType());
  }
  
  // Otherwise, we don't know.
  return 0;
}

Instruction *InstCombiner::visitExtractElementInst(ExtractElementInst &EI) {
  // If vector val is undef, replace extract with scalar undef.
  if (isa<UndefValue>(EI.getOperand(0)))
    return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));
  
  // If vector val is constant 0, replace extract with scalar 0.
  if (isa<ConstantAggregateZero>(EI.getOperand(0)))
    return ReplaceInstUsesWith(EI, Constant::getNullValue(EI.getType()));
  
  if (ConstantVector *C = dyn_cast<ConstantVector>(EI.getOperand(0))) {
    // If vector val is constant with all elements the same, replace EI with
    // that element. When the elements are not identical, we cannot replace yet
    // (we do that below, but only when the index is constant).
    Constant *op0 = C->getOperand(0);
    for (unsigned i = 1; i != C->getNumOperands(); ++i)
      if (C->getOperand(i) != op0) {
        op0 = 0; 
        break;
      }
    if (op0)
      return ReplaceInstUsesWith(EI, op0);
  }
  
  // If extracting a specified index from the vector, see if we can recursively
  // find a previously computed scalar that was inserted into the vector.
  if (ConstantInt *IdxC = dyn_cast<ConstantInt>(EI.getOperand(1))) {
    unsigned IndexVal = IdxC->getZExtValue();
    unsigned VectorWidth = EI.getVectorOperandType()->getNumElements();
    
    // If this is extracting an invalid index, turn this into undef, to avoid
    // crashing the code below.
    if (IndexVal >= VectorWidth)
      return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));
    
    // This instruction only demands the single element from the input vector.
    // If the input vector has a single use, simplify it based on this use
    // property.
    if (EI.getOperand(0)->hasOneUse() && VectorWidth != 1) {
      APInt UndefElts(VectorWidth, 0);
      APInt DemandedMask(VectorWidth, 0);
      DemandedMask.set(IndexVal);
      if (Value *V = SimplifyDemandedVectorElts(EI.getOperand(0),
                                                DemandedMask, UndefElts)) {
        EI.setOperand(0, V);
        return &EI;
      }
    }
    
    if (Value *Elt = FindScalarElement(EI.getOperand(0), IndexVal))
      return ReplaceInstUsesWith(EI, Elt);
    
    // If the this extractelement is directly using a bitcast from a vector of
    // the same number of elements, see if we can find the source element from
    // it.  In this case, we will end up needing to bitcast the scalars.
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(EI.getOperand(0))) {
      if (const VectorType *VT = 
          dyn_cast<VectorType>(BCI->getOperand(0)->getType()))
        if (VT->getNumElements() == VectorWidth)
          if (Value *Elt = FindScalarElement(BCI->getOperand(0), IndexVal))
            return new BitCastInst(Elt, EI.getType());
    }
  }
  
  if (Instruction *I = dyn_cast<Instruction>(EI.getOperand(0))) {
    // Push extractelement into predecessor operation if legal and
    // profitable to do so
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
      if (I->hasOneUse() &&
          CheapToScalarize(BO, isa<ConstantInt>(EI.getOperand(1)))) {
        Value *newEI0 =
        Builder->CreateExtractElement(BO->getOperand(0), EI.getOperand(1),
                                      EI.getName()+".lhs");
        Value *newEI1 =
        Builder->CreateExtractElement(BO->getOperand(1), EI.getOperand(1),
                                      EI.getName()+".rhs");
        return BinaryOperator::Create(BO->getOpcode(), newEI0, newEI1);
      }
    } else if (InsertElementInst *IE = dyn_cast<InsertElementInst>(I)) {
      // Extracting the inserted element?
      if (IE->getOperand(2) == EI.getOperand(1))
        return ReplaceInstUsesWith(EI, IE->getOperand(1));
      // If the inserted and extracted elements are constants, they must not
      // be the same value, extract from the pre-inserted value instead.
      if (isa<Constant>(IE->getOperand(2)) && isa<Constant>(EI.getOperand(1))) {
        Worklist.AddValue(EI.getOperand(0));
        EI.setOperand(0, IE->getOperand(0));
        return &EI;
      }
    } else if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(I)) {
      // If this is extracting an element from a shufflevector, figure out where
      // it came from and extract from the appropriate input element instead.
      if (ConstantInt *Elt = dyn_cast<ConstantInt>(EI.getOperand(1))) {
        unsigned SrcIdx = getShuffleMask(SVI)[Elt->getZExtValue()];
        Value *Src;
        unsigned LHSWidth =
        cast<VectorType>(SVI->getOperand(0)->getType())->getNumElements();
        
        if (SrcIdx < LHSWidth)
          Src = SVI->getOperand(0);
        else if (SrcIdx < LHSWidth*2) {
          SrcIdx -= LHSWidth;
          Src = SVI->getOperand(1);
        } else {
          return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));
        }
        return ExtractElementInst::Create(Src,
                                          ConstantInt::get(Type::getInt32Ty(EI.getContext()),
                                                           SrcIdx, false));
      }
    }
    // FIXME: Canonicalize extractelement(bitcast) -> bitcast(extractelement)
  }
  return 0;
}

/// CollectSingleShuffleElements - If V is a shuffle of values that ONLY returns
/// elements from either LHS or RHS, return the shuffle mask and true. 
/// Otherwise, return false.
static bool CollectSingleShuffleElements(Value *V, Value *LHS, Value *RHS,
                                         std::vector<Constant*> &Mask) {
  assert(V->getType() == LHS->getType() && V->getType() == RHS->getType() &&
         "Invalid CollectSingleShuffleElements");
  unsigned NumElts = cast<VectorType>(V->getType())->getNumElements();
  
  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::getInt32Ty(V->getContext())));
    return true;
  }
  
  if (V == LHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()), i));
    return true;
  }
  
  if (V == RHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()),
                                      i+NumElts));
    return true;
  }
  
  if (InsertElementInst *IEI = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert of an extract from some other vector, include it.
    Value *VecOp    = IEI->getOperand(0);
    Value *ScalarOp = IEI->getOperand(1);
    Value *IdxOp    = IEI->getOperand(2);
    
    if (!isa<ConstantInt>(IdxOp))
      return false;
    unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();
    
    if (isa<UndefValue>(ScalarOp)) {  // inserting undef into vector.
      // Okay, we can handle this if the vector we are insertinting into is
      // transitively ok.
      if (CollectSingleShuffleElements(VecOp, LHS, RHS, Mask)) {
        // If so, update the mask to reflect the inserted undef.
        Mask[InsertedIdx] = UndefValue::get(Type::getInt32Ty(V->getContext()));
        return true;
      }      
    } else if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)){
      if (isa<ConstantInt>(EI->getOperand(1)) &&
          EI->getOperand(0)->getType() == V->getType()) {
        unsigned ExtractedIdx =
        cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
        
        // This must be extracting from either LHS or RHS.
        if (EI->getOperand(0) == LHS || EI->getOperand(0) == RHS) {
          // Okay, we can handle this if the vector we are insertinting into is
          // transitively ok.
          if (CollectSingleShuffleElements(VecOp, LHS, RHS, Mask)) {
            // If so, update the mask to reflect the inserted value.
            if (EI->getOperand(0) == LHS) {
              Mask[InsertedIdx % NumElts] = 
              ConstantInt::get(Type::getInt32Ty(V->getContext()),
                               ExtractedIdx);
            } else {
              assert(EI->getOperand(0) == RHS);
              Mask[InsertedIdx % NumElts] = 
              ConstantInt::get(Type::getInt32Ty(V->getContext()),
                               ExtractedIdx+NumElts);
              
            }
            return true;
          }
        }
      }
    }
  }
  // TODO: Handle shufflevector here!
  
  return false;
}

/// CollectShuffleElements - We are building a shuffle of V, using RHS as the
/// RHS of the shuffle instruction, if it is not null.  Return a shuffle mask
/// that computes V and the LHS value of the shuffle.
static Value *CollectShuffleElements(Value *V, std::vector<Constant*> &Mask,
                                     Value *&RHS) {
  assert(V->getType()->isVectorTy() && 
         (RHS == 0 || V->getType() == RHS->getType()) &&
         "Invalid shuffle!");
  unsigned NumElts = cast<VectorType>(V->getType())->getNumElements();
  
  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::getInt32Ty(V->getContext())));
    return V;
  } else if (isa<ConstantAggregateZero>(V)) {
    Mask.assign(NumElts, ConstantInt::get(Type::getInt32Ty(V->getContext()),0));
    return V;
  } else if (InsertElementInst *IEI = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert of an extract from some other vector, include it.
    Value *VecOp    = IEI->getOperand(0);
    Value *ScalarOp = IEI->getOperand(1);
    Value *IdxOp    = IEI->getOperand(2);
    
    if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)) {
      if (isa<ConstantInt>(EI->getOperand(1)) && isa<ConstantInt>(IdxOp) &&
          EI->getOperand(0)->getType() == V->getType()) {
        unsigned ExtractedIdx =
        cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
        unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();
        
        // Either the extracted from or inserted into vector must be RHSVec,
        // otherwise we'd end up with a shuffle of three inputs.
        if (EI->getOperand(0) == RHS || RHS == 0) {
          RHS = EI->getOperand(0);
          Value *V = CollectShuffleElements(VecOp, Mask, RHS);
          Mask[InsertedIdx % NumElts] = 
          ConstantInt::get(Type::getInt32Ty(V->getContext()),
                           NumElts+ExtractedIdx);
          return V;
        }
        
        if (VecOp == RHS) {
          Value *V = CollectShuffleElements(EI->getOperand(0), Mask, RHS);
          // Everything but the extracted element is replaced with the RHS.
          for (unsigned i = 0; i != NumElts; ++i) {
            if (i != InsertedIdx)
              Mask[i] = ConstantInt::get(Type::getInt32Ty(V->getContext()),
                                         NumElts+i);
          }
          return V;
        }
        
        // If this insertelement is a chain that comes from exactly these two
        // vectors, return the vector and the effective shuffle.
        if (CollectSingleShuffleElements(IEI, EI->getOperand(0), RHS, Mask))
          return EI->getOperand(0);
      }
    }
  }
  // TODO: Handle shufflevector here!
  
  // Otherwise, can't do anything fancy.  Return an identity vector.
  for (unsigned i = 0; i != NumElts; ++i)
    Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()), i));
  return V;
}

Instruction *InstCombiner::visitInsertElementInst(InsertElementInst &IE) {
  Value *VecOp    = IE.getOperand(0);
  Value *ScalarOp = IE.getOperand(1);
  Value *IdxOp    = IE.getOperand(2);
  
  // Inserting an undef or into an undefined place, remove this.
  if (isa<UndefValue>(ScalarOp) || isa<UndefValue>(IdxOp))
    ReplaceInstUsesWith(IE, VecOp);
  
  // If the inserted element was extracted from some other vector, and if the 
  // indexes are constant, try to turn this into a shufflevector operation.
  if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)) {
    if (isa<ConstantInt>(EI->getOperand(1)) && isa<ConstantInt>(IdxOp) &&
        EI->getOperand(0)->getType() == IE.getType()) {
      unsigned NumVectorElts = IE.getType()->getNumElements();
      unsigned ExtractedIdx =
      cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
      unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();
      
      if (ExtractedIdx >= NumVectorElts) // Out of range extract.
        return ReplaceInstUsesWith(IE, VecOp);
      
      if (InsertedIdx >= NumVectorElts)  // Out of range insert.
        return ReplaceInstUsesWith(IE, UndefValue::get(IE.getType()));
      
      // If we are extracting a value from a vector, then inserting it right
      // back into the same place, just use the input vector.
      if (EI->getOperand(0) == VecOp && ExtractedIdx == InsertedIdx)
        return ReplaceInstUsesWith(IE, VecOp);      
      
      // If this insertelement isn't used by some other insertelement, turn it
      // (and any insertelements it points to), into one big shuffle.
      if (!IE.hasOneUse() || !isa<InsertElementInst>(IE.use_back())) {
        std::vector<Constant*> Mask;
        Value *RHS = 0;
        Value *LHS = CollectShuffleElements(&IE, Mask, RHS);
        if (RHS == 0) RHS = UndefValue::get(LHS->getType());
        // We now have a shuffle of LHS, RHS, Mask.
        return new ShuffleVectorInst(LHS, RHS,
                                     ConstantVector::get(Mask));
      }
    }
  }
  
  unsigned VWidth = cast<VectorType>(VecOp->getType())->getNumElements();
  APInt UndefElts(VWidth, 0);
  APInt AllOnesEltMask(APInt::getAllOnesValue(VWidth));
  if (SimplifyDemandedVectorElts(&IE, AllOnesEltMask, UndefElts))
    return &IE;
  
  return 0;
}


Instruction *InstCombiner::visitShuffleVectorInst(ShuffleVectorInst &SVI) {
  Value *LHS = SVI.getOperand(0);
  Value *RHS = SVI.getOperand(1);
  std::vector<unsigned> Mask = getShuffleMask(&SVI);
  
  bool MadeChange = false;
  
  // Undefined shuffle mask -> undefined value.
  if (isa<UndefValue>(SVI.getOperand(2)))
    return ReplaceInstUsesWith(SVI, UndefValue::get(SVI.getType()));
  
  unsigned VWidth = cast<VectorType>(SVI.getType())->getNumElements();
  
  if (VWidth != cast<VectorType>(LHS->getType())->getNumElements())
    return 0;
  
  APInt UndefElts(VWidth, 0);
  APInt AllOnesEltMask(APInt::getAllOnesValue(VWidth));
  if (SimplifyDemandedVectorElts(&SVI, AllOnesEltMask, UndefElts)) {
    LHS = SVI.getOperand(0);
    RHS = SVI.getOperand(1);
    MadeChange = true;
  }
  
  // Canonicalize shuffle(x    ,x,mask) -> shuffle(x, undef,mask')
  // Canonicalize shuffle(undef,x,mask) -> shuffle(x, undef,mask').
  if (LHS == RHS || isa<UndefValue>(LHS)) {
    if (isa<UndefValue>(LHS) && LHS == RHS) {
      // shuffle(undef,undef,mask) -> undef.
      return ReplaceInstUsesWith(SVI, LHS);
    }
    
    // Remap any references to RHS to use LHS.
    std::vector<Constant*> Elts;
    for (unsigned i = 0, e = Mask.size(); i != e; ++i) {
      if (Mask[i] >= 2*e)
        Elts.push_back(UndefValue::get(Type::getInt32Ty(SVI.getContext())));
      else {
        if ((Mask[i] >= e && isa<UndefValue>(RHS)) ||
            (Mask[i] <  e && isa<UndefValue>(LHS))) {
          Mask[i] = 2*e;     // Turn into undef.
          Elts.push_back(UndefValue::get(Type::getInt32Ty(SVI.getContext())));
        } else {
          Mask[i] = Mask[i] % e;  // Force to LHS.
          Elts.push_back(ConstantInt::get(Type::getInt32Ty(SVI.getContext()),
                                          Mask[i]));
        }
      }
    }
    SVI.setOperand(0, SVI.getOperand(1));
    SVI.setOperand(1, UndefValue::get(RHS->getType()));
    SVI.setOperand(2, ConstantVector::get(Elts));
    LHS = SVI.getOperand(0);
    RHS = SVI.getOperand(1);
    MadeChange = true;
  }
  
  // Analyze the shuffle, are the LHS or RHS and identity shuffles?
  bool isLHSID = true, isRHSID = true;
  
  for (unsigned i = 0, e = Mask.size(); i != e; ++i) {
    if (Mask[i] >= e*2) continue;  // Ignore undef values.
    // Is this an identity shuffle of the LHS value?
    isLHSID &= (Mask[i] == i);
    
    // Is this an identity shuffle of the RHS value?
    isRHSID &= (Mask[i]-e == i);
  }
  
  // Eliminate identity shuffles.
  if (isLHSID) return ReplaceInstUsesWith(SVI, LHS);
  if (isRHSID) return ReplaceInstUsesWith(SVI, RHS);
  
  // If the LHS is a shufflevector itself, see if we can combine it with this
  // one without producing an unusual shuffle.  Here we are really conservative:
  // we are absolutely afraid of producing a shuffle mask not in the input
  // program, because the code gen may not be smart enough to turn a merged
  // shuffle into two specific shuffles: it may produce worse code.  As such,
  // we only merge two shuffles if the result is either a splat or one of the
  // two input shuffle masks.  In this case, merging the shuffles just removes
  // one instruction, which we know is safe.  This is good for things like
  // turning: (splat(splat)) -> splat.
  if (ShuffleVectorInst *LHSSVI = dyn_cast<ShuffleVectorInst>(LHS)) {
    if (isa<UndefValue>(RHS)) {
      std::vector<unsigned> LHSMask = getShuffleMask(LHSSVI);
      
      if (LHSMask.size() == Mask.size()) {
        std::vector<unsigned> NewMask;
        bool isSplat = true;
        unsigned SplatElt = 2 * Mask.size(); // undef
        for (unsigned i = 0, e = Mask.size(); i != e; ++i) {
          unsigned MaskElt = 2 * e; // undef
          if (Mask[i] < e)
            MaskElt = LHSMask[Mask[i]];
          // Check if this could still be a splat.
          if (MaskElt < 2*e) {
            if (SplatElt < 2*e && SplatElt != MaskElt)
              isSplat = false;
            SplatElt = MaskElt;
          }
          NewMask.push_back(MaskElt);
        }
        
        // If the result mask is equal to the src shuffle or this
        // shuffle mask, do the replacement.
        if (isSplat || NewMask == LHSMask || NewMask == Mask) {
          unsigned LHSInNElts =
          cast<VectorType>(LHSSVI->getOperand(0)->getType())->
          getNumElements();
          std::vector<Constant*> Elts;
          const Type *Int32Ty = Type::getInt32Ty(SVI.getContext());
          for (unsigned i = 0, e = NewMask.size(); i != e; ++i) {
            if (NewMask[i] >= LHSInNElts*2) {
              Elts.push_back(UndefValue::get(Int32Ty));
            } else {
              Elts.push_back(ConstantInt::get(Int32Ty, NewMask[i]));
            }
          }
          return new ShuffleVectorInst(LHSSVI->getOperand(0),
                                       LHSSVI->getOperand(1),
                                       ConstantVector::get(Elts));
        }
      }
    }
  }
  
  return MadeChange ? &SVI : 0;
}

