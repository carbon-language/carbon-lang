//===- Loads.cpp - Local load analysis ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines simple local analyses for load instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Operator.h"
using namespace llvm;

/// AreEquivalentAddressValues - Test if A and B will obviously have the same
/// value. This includes recognizing that %t0 and %t1 will have the same
/// value in code like this:
///   %t0 = getelementptr \@a, 0, 3
///   store i32 0, i32* %t0
///   %t1 = getelementptr \@a, 0, 3
///   %t2 = load i32* %t1
///
static bool AreEquivalentAddressValues(const Value *A, const Value *B) {
  // Test if the values are trivially equivalent.
  if (A == B) return true;

  // Test if the values come from identical arithmetic instructions.
  // Use isIdenticalToWhenDefined instead of isIdenticalTo because
  // this function is only used when one address use dominates the
  // other, which means that they'll always either have the same
  // value or one of them will have an undefined value.
  if (isa<BinaryOperator>(A) || isa<CastInst>(A) ||
      isa<PHINode>(A) || isa<GetElementPtrInst>(A))
    if (const Instruction *BI = dyn_cast<Instruction>(B))
      if (cast<Instruction>(A)->isIdenticalToWhenDefined(BI))
        return true;

  // Otherwise they may not be equivalent.
  return false;
}

/// isSafeToLoadUnconditionally - Return true if we know that executing a load
/// from this value cannot trap.  If it is not obviously safe to load from the
/// specified pointer, we do a quick local scan of the basic block containing
/// ScanFrom, to determine if the address is already accessed.
bool llvm::isSafeToLoadUnconditionally(Value *V, Instruction *ScanFrom,
                                       unsigned Align, const DataLayout *TD) {
  int64_t ByteOffset = 0;
  Value *Base = V;
  Base = GetPointerBaseWithConstantOffset(V, ByteOffset, TD);

  if (ByteOffset < 0) // out of bounds
    return false;

  Type *BaseType = nullptr;
  unsigned BaseAlign = 0;
  if (const AllocaInst *AI = dyn_cast<AllocaInst>(Base)) {
    // An alloca is safe to load from as load as it is suitably aligned.
    BaseType = AI->getAllocatedType();
    BaseAlign = AI->getAlignment();
  } else if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Base)) {
    // Global variables are safe to load from but their size cannot be
    // guaranteed if they are overridden.
    if (!GV->mayBeOverridden()) {
      BaseType = GV->getType()->getElementType();
      BaseAlign = GV->getAlignment();
    }
  }

  if (BaseType && BaseType->isSized()) {
    if (TD && BaseAlign == 0)
      BaseAlign = TD->getPrefTypeAlignment(BaseType);

    if (Align <= BaseAlign) {
      if (!TD)
        return true; // Loading directly from an alloca or global is OK.

      // Check if the load is within the bounds of the underlying object.
      PointerType *AddrTy = cast<PointerType>(V->getType());
      uint64_t LoadSize = TD->getTypeStoreSize(AddrTy->getElementType());
      if (ByteOffset + LoadSize <= TD->getTypeAllocSize(BaseType) &&
          (Align == 0 || (ByteOffset % Align) == 0))
        return true;
    }
  }

  // Otherwise, be a little bit aggressive by scanning the local block where we
  // want to check to see if the pointer is already being loaded or stored
  // from/to.  If so, the previous load or store would have already trapped,
  // so there is no harm doing an extra load (also, CSE will later eliminate
  // the load entirely).
  BasicBlock::iterator BBI = ScanFrom, E = ScanFrom->getParent()->begin();

  while (BBI != E) {
    --BBI;

    // If we see a free or a call which may write to memory (i.e. which might do
    // a free) the pointer could be marked invalid.
    if (isa<CallInst>(BBI) && BBI->mayWriteToMemory() &&
        !isa<DbgInfoIntrinsic>(BBI))
      return false;

    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      if (AreEquivalentAddressValues(LI->getOperand(0), V)) return true;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(BBI)) {
      if (AreEquivalentAddressValues(SI->getOperand(1), V)) return true;
    }
  }
  return false;
}

/// FindAvailableLoadedValue - Scan the ScanBB block backwards (starting at the
/// instruction before ScanFrom) checking to see if we have the value at the
/// memory address *Ptr locally available within a small number of instructions.
/// If the value is available, return it.
///
/// If not, return the iterator for the last validated instruction that the 
/// value would be live through.  If we scanned the entire block and didn't find
/// something that invalidates *Ptr or provides it, ScanFrom would be left at
/// begin() and this returns null.  ScanFrom could also be left 
///
/// MaxInstsToScan specifies the maximum instructions to scan in the block.  If
/// it is set to 0, it will scan the whole block. You can also optionally
/// specify an alias analysis implementation, which makes this more precise.
///
/// If AATags is non-null and a load or store is found, the AA tags from the
/// load or store are recorded there.  If there are no AA tags or if no access
/// is found, it is left unmodified.
Value *llvm::FindAvailableLoadedValue(Value *Ptr, BasicBlock *ScanBB,
                                      BasicBlock::iterator &ScanFrom,
                                      unsigned MaxInstsToScan,
                                      AliasAnalysis *AA,
                                      AAMDNodes *AATags) {
  if (MaxInstsToScan == 0) MaxInstsToScan = ~0U;

  // If we're using alias analysis to disambiguate get the size of *Ptr.
  uint64_t AccessSize = 0;
  if (AA) {
    Type *AccessTy = cast<PointerType>(Ptr->getType())->getElementType();
    AccessSize = AA->getTypeStoreSize(AccessTy);
  }
  
  while (ScanFrom != ScanBB->begin()) {
    // We must ignore debug info directives when counting (otherwise they
    // would affect codegen).
    Instruction *Inst = --ScanFrom;
    if (isa<DbgInfoIntrinsic>(Inst))
      continue;

    // Restore ScanFrom to expected value in case next test succeeds
    ScanFrom++;
   
    // Don't scan huge blocks.
    if (MaxInstsToScan-- == 0) return nullptr;
    
    --ScanFrom;
    // If this is a load of Ptr, the loaded value is available.
    // (This is true even if the load is volatile or atomic, although
    // those cases are unlikely.)
    if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
      if (AreEquivalentAddressValues(LI->getOperand(0), Ptr)) {
        if (AATags) LI->getAAMetadata(*AATags);
        return LI;
      }
    
    if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
      // If this is a store through Ptr, the value is available!
      // (This is true even if the store is volatile or atomic, although
      // those cases are unlikely.)
      if (AreEquivalentAddressValues(SI->getOperand(1), Ptr)) {
        if (AATags) SI->getAAMetadata(*AATags);
        return SI->getOperand(0);
      }
      
      // If Ptr is an alloca and this is a store to a different alloca, ignore
      // the store.  This is a trivial form of alias analysis that is important
      // for reg2mem'd code.
      if ((isa<AllocaInst>(Ptr) || isa<GlobalVariable>(Ptr)) &&
          (isa<AllocaInst>(SI->getOperand(1)) ||
           isa<GlobalVariable>(SI->getOperand(1))))
        continue;
      
      // If we have alias analysis and it says the store won't modify the loaded
      // value, ignore the store.
      if (AA &&
          (AA->getModRefInfo(SI, Ptr, AccessSize) & AliasAnalysis::Mod) == 0)
        continue;
      
      // Otherwise the store that may or may not alias the pointer, bail out.
      ++ScanFrom;
      return nullptr;
    }
    
    // If this is some other instruction that may clobber Ptr, bail out.
    if (Inst->mayWriteToMemory()) {
      // If alias analysis claims that it really won't modify the load,
      // ignore it.
      if (AA &&
          (AA->getModRefInfo(Inst, Ptr, AccessSize) & AliasAnalysis::Mod) == 0)
        continue;
      
      // May modify the pointer, bail out.
      ++ScanFrom;
      return nullptr;
    }
  }
  
  // Got to the start of the block, we didn't find it, but are done for this
  // block.
  return nullptr;
}
