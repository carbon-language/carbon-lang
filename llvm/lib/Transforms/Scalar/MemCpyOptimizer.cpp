//===- MemCpyOptimizer.cpp - Optimize use of memcpy and friends -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs various transformations related to eliminating memcpy
// calls, or transforming sets of stores into memset's.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "memcpyopt"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
#include <list>
using namespace llvm;

STATISTIC(NumMemCpyInstr, "Number of memcpy instructions deleted");
STATISTIC(NumMemSetInfer, "Number of memsets inferred");
STATISTIC(NumMoveToCpy,   "Number of memmoves converted to memcpy");

/// isBytewiseValue - If the specified value can be set by repeating the same
/// byte in memory, return the i8 value that it is represented with.  This is
/// true for all i8 values obviously, but is also true for i32 0, i32 -1,
/// i16 0xF0F0, double 0.0 etc.  If the value can't be handled with a repeated
/// byte store (e.g. i16 0x1234), return null.
static Value *isBytewiseValue(Value *V) {
  LLVMContext &Context = V->getContext();
  
  // All byte-wide stores are splatable, even of arbitrary variables.
  if (V->getType()->isIntegerTy(8)) return V;
  
  // Constant float and double values can be handled as integer values if the
  // corresponding integer value is "byteable".  An important case is 0.0. 
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(V)) {
    if (CFP->getType()->isFloatTy())
      V = ConstantExpr::getBitCast(CFP, Type::getInt32Ty(Context));
    if (CFP->getType()->isDoubleTy())
      V = ConstantExpr::getBitCast(CFP, Type::getInt64Ty(Context));
    // Don't handle long double formats, which have strange constraints.
  }
  
  // We can handle constant integers that are power of two in size and a 
  // multiple of 8 bits.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    unsigned Width = CI->getBitWidth();
    if (isPowerOf2_32(Width) && Width > 8) {
      // We can handle this value if the recursive binary decomposition is the
      // same at all levels.
      APInt Val = CI->getValue();
      APInt Val2;
      while (Val.getBitWidth() != 8) {
        unsigned NextWidth = Val.getBitWidth()/2;
        Val2  = Val.lshr(NextWidth);
        Val2.trunc(Val.getBitWidth()/2);
        Val.trunc(Val.getBitWidth()/2);

        // If the top/bottom halves aren't the same, reject it.
        if (Val != Val2)
          return 0;
      }
      return ConstantInt::get(Context, Val);
    }
  }
  
  // Conceptually, we could handle things like:
  //   %a = zext i8 %X to i16
  //   %b = shl i16 %a, 8
  //   %c = or i16 %a, %b
  // but until there is an example that actually needs this, it doesn't seem
  // worth worrying about.
  return 0;
}

static int64_t GetOffsetFromIndex(const GetElementPtrInst *GEP, unsigned Idx,
                                  bool &VariableIdxFound, TargetData &TD) {
  // Skip over the first indices.
  gep_type_iterator GTI = gep_type_begin(GEP);
  for (unsigned i = 1; i != Idx; ++i, ++GTI)
    /*skip along*/;
  
  // Compute the offset implied by the rest of the indices.
  int64_t Offset = 0;
  for (unsigned i = Idx, e = GEP->getNumOperands(); i != e; ++i, ++GTI) {
    ConstantInt *OpC = dyn_cast<ConstantInt>(GEP->getOperand(i));
    if (OpC == 0)
      return VariableIdxFound = true;
    if (OpC->isZero()) continue;  // No offset.

    // Handle struct indices, which add their field offset to the pointer.
    if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
      Offset += TD.getStructLayout(STy)->getElementOffset(OpC->getZExtValue());
      continue;
    }
    
    // Otherwise, we have a sequential type like an array or vector.  Multiply
    // the index by the ElementSize.
    uint64_t Size = TD.getTypeAllocSize(GTI.getIndexedType());
    Offset += Size*OpC->getSExtValue();
  }

  return Offset;
}

/// IsPointerOffset - Return true if Ptr1 is provably equal to Ptr2 plus a
/// constant offset, and return that constant offset.  For example, Ptr1 might
/// be &A[42], and Ptr2 might be &A[40].  In this case offset would be -8.
static bool IsPointerOffset(Value *Ptr1, Value *Ptr2, int64_t &Offset,
                            TargetData &TD) {
  // Right now we handle the case when Ptr1/Ptr2 are both GEPs with an identical
  // base.  After that base, they may have some number of common (and
  // potentially variable) indices.  After that they handle some constant
  // offset, which determines their offset from each other.  At this point, we
  // handle no other case.
  GetElementPtrInst *GEP1 = dyn_cast<GetElementPtrInst>(Ptr1);
  GetElementPtrInst *GEP2 = dyn_cast<GetElementPtrInst>(Ptr2);
  if (!GEP1 || !GEP2 || GEP1->getOperand(0) != GEP2->getOperand(0))
    return false;
  
  // Skip any common indices and track the GEP types.
  unsigned Idx = 1;
  for (; Idx != GEP1->getNumOperands() && Idx != GEP2->getNumOperands(); ++Idx)
    if (GEP1->getOperand(Idx) != GEP2->getOperand(Idx))
      break;

  bool VariableIdxFound = false;
  int64_t Offset1 = GetOffsetFromIndex(GEP1, Idx, VariableIdxFound, TD);
  int64_t Offset2 = GetOffsetFromIndex(GEP2, Idx, VariableIdxFound, TD);
  if (VariableIdxFound) return false;
  
  Offset = Offset2-Offset1;
  return true;
}


/// MemsetRange - Represents a range of memset'd bytes with the ByteVal value.
/// This allows us to analyze stores like:
///   store 0 -> P+1
///   store 0 -> P+0
///   store 0 -> P+3
///   store 0 -> P+2
/// which sometimes happens with stores to arrays of structs etc.  When we see
/// the first store, we make a range [1, 2).  The second store extends the range
/// to [0, 2).  The third makes a new range [2, 3).  The fourth store joins the
/// two ranges into [0, 3) which is memset'able.
namespace {
struct MemsetRange {
  // Start/End - A semi range that describes the span that this range covers.
  // The range is closed at the start and open at the end: [Start, End).  
  int64_t Start, End;

  /// StartPtr - The getelementptr instruction that points to the start of the
  /// range.
  Value *StartPtr;
  
  /// Alignment - The known alignment of the first store.
  unsigned Alignment;
  
  /// TheStores - The actual stores that make up this range.
  SmallVector<StoreInst*, 16> TheStores;
  
  bool isProfitableToUseMemset(const TargetData &TD) const;

};
} // end anon namespace

bool MemsetRange::isProfitableToUseMemset(const TargetData &TD) const {
  // If we found more than 8 stores to merge or 64 bytes, use memset.
  if (TheStores.size() >= 8 || End-Start >= 64) return true;
  
  // Assume that the code generator is capable of merging pairs of stores
  // together if it wants to.
  if (TheStores.size() <= 2) return false;
  
  // If we have fewer than 8 stores, it can still be worthwhile to do this.
  // For example, merging 4 i8 stores into an i32 store is useful almost always.
  // However, merging 2 32-bit stores isn't useful on a 32-bit architecture (the
  // memset will be split into 2 32-bit stores anyway) and doing so can
  // pessimize the llvm optimizer.
  //
  // Since we don't have perfect knowledge here, make some assumptions: assume
  // the maximum GPR width is the same size as the pointer size and assume that
  // this width can be stored.  If so, check to see whether we will end up
  // actually reducing the number of stores used.
  unsigned Bytes = unsigned(End-Start);
  unsigned NumPointerStores = Bytes/TD.getPointerSize();
  
  // Assume the remaining bytes if any are done a byte at a time.
  unsigned NumByteStores = Bytes - NumPointerStores*TD.getPointerSize();
  
  // If we will reduce the # stores (according to this heuristic), do the
  // transformation.  This encourages merging 4 x i8 -> i32 and 2 x i16 -> i32
  // etc.
  return TheStores.size() > NumPointerStores+NumByteStores;
}    


namespace {
class MemsetRanges {
  /// Ranges - A sorted list of the memset ranges.  We use std::list here
  /// because each element is relatively large and expensive to copy.
  std::list<MemsetRange> Ranges;
  typedef std::list<MemsetRange>::iterator range_iterator;
  TargetData &TD;
public:
  MemsetRanges(TargetData &td) : TD(td) {}
  
  typedef std::list<MemsetRange>::const_iterator const_iterator;
  const_iterator begin() const { return Ranges.begin(); }
  const_iterator end() const { return Ranges.end(); }
  bool empty() const { return Ranges.empty(); }
  
  void addStore(int64_t OffsetFromFirst, StoreInst *SI);
};
  
} // end anon namespace


/// addStore - Add a new store to the MemsetRanges data structure.  This adds a
/// new range for the specified store at the specified offset, merging into
/// existing ranges as appropriate.
void MemsetRanges::addStore(int64_t Start, StoreInst *SI) {
  int64_t End = Start+TD.getTypeStoreSize(SI->getOperand(0)->getType());
  
  // Do a linear search of the ranges to see if this can be joined and/or to
  // find the insertion point in the list.  We keep the ranges sorted for
  // simplicity here.  This is a linear search of a linked list, which is ugly,
  // however the number of ranges is limited, so this won't get crazy slow.
  range_iterator I = Ranges.begin(), E = Ranges.end();
  
  while (I != E && Start > I->End)
    ++I;
  
  // We now know that I == E, in which case we didn't find anything to merge
  // with, or that Start <= I->End.  If End < I->Start or I == E, then we need
  // to insert a new range.  Handle this now.
  if (I == E || End < I->Start) {
    MemsetRange &R = *Ranges.insert(I, MemsetRange());
    R.Start        = Start;
    R.End          = End;
    R.StartPtr     = SI->getPointerOperand();
    R.Alignment    = SI->getAlignment();
    R.TheStores.push_back(SI);
    return;
  }

  // This store overlaps with I, add it.
  I->TheStores.push_back(SI);
  
  // At this point, we may have an interval that completely contains our store.
  // If so, just add it to the interval and return.
  if (I->Start <= Start && I->End >= End)
    return;
  
  // Now we know that Start <= I->End and End >= I->Start so the range overlaps
  // but is not entirely contained within the range.
  
  // See if the range extends the start of the range.  In this case, it couldn't
  // possibly cause it to join the prior range, because otherwise we would have
  // stopped on *it*.
  if (Start < I->Start) {
    I->Start = Start;
    I->StartPtr = SI->getPointerOperand();
    I->Alignment = SI->getAlignment();
  }
    
  // Now we know that Start <= I->End and Start >= I->Start (so the startpoint
  // is in or right at the end of I), and that End >= I->Start.  Extend I out to
  // End.
  if (End > I->End) {
    I->End = End;
    range_iterator NextI = I;
    while (++NextI != E && End >= NextI->Start) {
      // Merge the range in.
      I->TheStores.append(NextI->TheStores.begin(), NextI->TheStores.end());
      if (NextI->End > I->End)
        I->End = NextI->End;
      Ranges.erase(NextI);
      NextI = I;
    }
  }
}

//===----------------------------------------------------------------------===//
//                         MemCpyOpt Pass
//===----------------------------------------------------------------------===//

namespace {
  class MemCpyOpt : public FunctionPass {
    bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    MemCpyOpt() : FunctionPass(ID) {
      initializeMemCpyOptPass(*PassRegistry::getPassRegistry());
    }

  private:
    // This transformation requires dominator postdominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<MemoryDependenceAnalysis>();
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<MemoryDependenceAnalysis>();
    }
  
    // Helper fuctions
    bool processStore(StoreInst *SI, BasicBlock::iterator &BBI);
    bool processMemCpy(MemCpyInst *M);
    bool processMemMove(MemMoveInst *M);
    bool performCallSlotOptzn(Instruction *cpy, Value *cpyDst, Value *cpySrc,
                              uint64_t cpyLen, CallInst *C);
    bool processMemCpyMemCpyDependence(MemCpyInst *M, MemCpyInst *MDep,
                                       uint64_t MSize);
    bool iterateOnFunction(Function &F);
  };
  
  char MemCpyOpt::ID = 0;
}

// createMemCpyOptPass - The public interface to this file...
FunctionPass *llvm::createMemCpyOptPass() { return new MemCpyOpt(); }

INITIALIZE_PASS_BEGIN(MemCpyOpt, "memcpyopt", "MemCpy Optimization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceAnalysis)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(MemCpyOpt, "memcpyopt", "MemCpy Optimization",
                    false, false)

/// processStore - When GVN is scanning forward over instructions, we look for
/// some other patterns to fold away.  In particular, this looks for stores to
/// neighboring locations of memory.  If it sees enough consequtive ones
/// (currently 4) it attempts to merge them together into a memcpy/memset.
bool MemCpyOpt::processStore(StoreInst *SI, BasicBlock::iterator &BBI) {
  if (SI->isVolatile()) return false;
  
  TargetData *TD = getAnalysisIfAvailable<TargetData>();
  if (!TD) return false;

  // Detect cases where we're performing call slot forwarding, but
  // happen to be using a load-store pair to implement it, rather than
  // a memcpy.
  if (LoadInst *LI = dyn_cast<LoadInst>(SI->getOperand(0))) {
    if (!LI->isVolatile() && LI->hasOneUse()) {
      MemoryDependenceAnalysis &MD = getAnalysis<MemoryDependenceAnalysis>();

      MemDepResult dep = MD.getDependency(LI);
      CallInst *C = 0;
      if (dep.isClobber() && !isa<MemCpyInst>(dep.getInst()))
        C = dyn_cast<CallInst>(dep.getInst());
      
      if (C) {
        bool changed = performCallSlotOptzn(LI,
                        SI->getPointerOperand()->stripPointerCasts(), 
                        LI->getPointerOperand()->stripPointerCasts(),
                        TD->getTypeStoreSize(SI->getOperand(0)->getType()), C);
        if (changed) {
          MD.removeInstruction(SI);
          SI->eraseFromParent();
          LI->eraseFromParent();
          ++NumMemCpyInstr;
          return true;
        }
      }
    }
  }
  
  LLVMContext &Context = SI->getContext();

  // There are two cases that are interesting for this code to handle: memcpy
  // and memset.  Right now we only handle memset.
  
  // Ensure that the value being stored is something that can be memset'able a
  // byte at a time like "0" or "-1" or any width, as well as things like
  // 0xA0A0A0A0 and 0.0.
  Value *ByteVal = isBytewiseValue(SI->getOperand(0));
  if (!ByteVal)
    return false;

  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  Module *M = SI->getParent()->getParent()->getParent();

  // Okay, so we now have a single store that can be splatable.  Scan to find
  // all subsequent stores of the same value to offset from the same pointer.
  // Join these together into ranges, so we can decide whether contiguous blocks
  // are stored.
  MemsetRanges Ranges(*TD);
  
  Value *StartPtr = SI->getPointerOperand();
  
  BasicBlock::iterator BI = SI;
  for (++BI; !isa<TerminatorInst>(BI); ++BI) {
    if (isa<CallInst>(BI) || isa<InvokeInst>(BI)) { 
      // If the call is readnone, ignore it, otherwise bail out.  We don't even
      // allow readonly here because we don't want something like:
      // A[1] = 2; strlen(A); A[2] = 2; -> memcpy(A, ...); strlen(A).
      if (AA.getModRefBehavior(CallSite(BI)) ==
            AliasAnalysis::DoesNotAccessMemory)
        continue;
      
      // TODO: If this is a memset, try to join it in.
      
      break;
    } else if (isa<VAArgInst>(BI) || isa<LoadInst>(BI))
      break;

    // If this is a non-store instruction it is fine, ignore it.
    StoreInst *NextStore = dyn_cast<StoreInst>(BI);
    if (NextStore == 0) continue;
    
    // If this is a store, see if we can merge it in.
    if (NextStore->isVolatile()) break;
    
    // Check to see if this stored value is of the same byte-splattable value.
    if (ByteVal != isBytewiseValue(NextStore->getOperand(0)))
      break;

    // Check to see if this store is to a constant offset from the start ptr.
    int64_t Offset;
    if (!IsPointerOffset(StartPtr, NextStore->getPointerOperand(), Offset, *TD))
      break;

    Ranges.addStore(Offset, NextStore);
  }

  // If we have no ranges, then we just had a single store with nothing that
  // could be merged in.  This is a very common case of course.
  if (Ranges.empty())
    return false;
  
  // If we had at least one store that could be merged in, add the starting
  // store as well.  We try to avoid this unless there is at least something
  // interesting as a small compile-time optimization.
  Ranges.addStore(0, SI);
  
  
  // Now that we have full information about ranges, loop over the ranges and
  // emit memset's for anything big enough to be worthwhile.
  bool MadeChange = false;
  for (MemsetRanges::const_iterator I = Ranges.begin(), E = Ranges.end();
       I != E; ++I) {
    const MemsetRange &Range = *I;

    if (Range.TheStores.size() == 1) continue;
    
    // If it is profitable to lower this range to memset, do so now.
    if (!Range.isProfitableToUseMemset(*TD))
      continue;
    
    // Otherwise, we do want to transform this!  Create a new memset.  We put
    // the memset right before the first instruction that isn't part of this
    // memset block.  This ensure that the memset is dominated by any addressing
    // instruction needed by the start of the block.
    BasicBlock::iterator InsertPt = BI;

    // Get the starting pointer of the block.
    StartPtr = Range.StartPtr;

    // Determine alignment
    unsigned Alignment = Range.Alignment;
    if (Alignment == 0) {
      const Type *EltType = 
         cast<PointerType>(StartPtr->getType())->getElementType();
      Alignment = TD->getABITypeAlignment(EltType);
    }

    // Cast the start ptr to be i8* as memset requires.
    const PointerType* StartPTy = cast<PointerType>(StartPtr->getType());
    const PointerType *i8Ptr = Type::getInt8PtrTy(Context,
                                                  StartPTy->getAddressSpace());
    if (StartPTy!= i8Ptr)
      StartPtr = new BitCastInst(StartPtr, i8Ptr, StartPtr->getName(),
                                 InsertPt);

    Value *Ops[] = {
      StartPtr, ByteVal,   // Start, value
      // size
      ConstantInt::get(Type::getInt64Ty(Context), Range.End-Range.Start),
      // align
      ConstantInt::get(Type::getInt32Ty(Context), Alignment),
      // volatile
      ConstantInt::get(Type::getInt1Ty(Context), 0),
    };
    const Type *Tys[] = { Ops[0]->getType(), Ops[2]->getType() };

    Function *MemSetF = Intrinsic::getDeclaration(M, Intrinsic::memset, Tys, 2);

    Value *C = CallInst::Create(MemSetF, Ops, Ops+5, "", InsertPt);
    DEBUG(dbgs() << "Replace stores:\n";
          for (unsigned i = 0, e = Range.TheStores.size(); i != e; ++i)
            dbgs() << *Range.TheStores[i];
          dbgs() << "With: " << *C); C=C;
  
    // Don't invalidate the iterator
    BBI = BI;
  
    // Zap all the stores.
    for (SmallVector<StoreInst*, 16>::const_iterator
         SI = Range.TheStores.begin(),
         SE = Range.TheStores.end(); SI != SE; ++SI)
      (*SI)->eraseFromParent();
    ++NumMemSetInfer;
    MadeChange = true;
  }
  
  return MadeChange;
}


/// performCallSlotOptzn - takes a memcpy and a call that it depends on,
/// and checks for the possibility of a call slot optimization by having
/// the call write its result directly into the destination of the memcpy.
bool MemCpyOpt::performCallSlotOptzn(Instruction *cpy,
                                     Value *cpyDest, Value *cpySrc,
                                     uint64_t cpyLen, CallInst *C) {
  // The general transformation to keep in mind is
  //
  //   call @func(..., src, ...)
  //   memcpy(dest, src, ...)
  //
  // ->
  //
  //   memcpy(dest, src, ...)
  //   call @func(..., dest, ...)
  //
  // Since moving the memcpy is technically awkward, we additionally check that
  // src only holds uninitialized values at the moment of the call, meaning that
  // the memcpy can be discarded rather than moved.

  // Deliberately get the source and destination with bitcasts stripped away,
  // because we'll need to do type comparisons based on the underlying type.
  CallSite CS(C);

  // Require that src be an alloca.  This simplifies the reasoning considerably.
  AllocaInst *srcAlloca = dyn_cast<AllocaInst>(cpySrc);
  if (!srcAlloca)
    return false;

  // Check that all of src is copied to dest.
  TargetData *TD = getAnalysisIfAvailable<TargetData>();
  if (!TD) return false;

  ConstantInt *srcArraySize = dyn_cast<ConstantInt>(srcAlloca->getArraySize());
  if (!srcArraySize)
    return false;

  uint64_t srcSize = TD->getTypeAllocSize(srcAlloca->getAllocatedType()) *
    srcArraySize->getZExtValue();

  if (cpyLen < srcSize)
    return false;

  // Check that accessing the first srcSize bytes of dest will not cause a
  // trap.  Otherwise the transform is invalid since it might cause a trap
  // to occur earlier than it otherwise would.
  if (AllocaInst *A = dyn_cast<AllocaInst>(cpyDest)) {
    // The destination is an alloca.  Check it is larger than srcSize.
    ConstantInt *destArraySize = dyn_cast<ConstantInt>(A->getArraySize());
    if (!destArraySize)
      return false;

    uint64_t destSize = TD->getTypeAllocSize(A->getAllocatedType()) *
      destArraySize->getZExtValue();

    if (destSize < srcSize)
      return false;
  } else if (Argument *A = dyn_cast<Argument>(cpyDest)) {
    // If the destination is an sret parameter then only accesses that are
    // outside of the returned struct type can trap.
    if (!A->hasStructRetAttr())
      return false;

    const Type *StructTy = cast<PointerType>(A->getType())->getElementType();
    uint64_t destSize = TD->getTypeAllocSize(StructTy);

    if (destSize < srcSize)
      return false;
  } else {
    return false;
  }

  // Check that src is not accessed except via the call and the memcpy.  This
  // guarantees that it holds only undefined values when passed in (so the final
  // memcpy can be dropped), that it is not read or written between the call and
  // the memcpy, and that writing beyond the end of it is undefined.
  SmallVector<User*, 8> srcUseList(srcAlloca->use_begin(),
                                   srcAlloca->use_end());
  while (!srcUseList.empty()) {
    User *UI = srcUseList.pop_back_val();

    if (isa<BitCastInst>(UI)) {
      for (User::use_iterator I = UI->use_begin(), E = UI->use_end();
           I != E; ++I)
        srcUseList.push_back(*I);
    } else if (GetElementPtrInst *G = dyn_cast<GetElementPtrInst>(UI)) {
      if (G->hasAllZeroIndices())
        for (User::use_iterator I = UI->use_begin(), E = UI->use_end();
             I != E; ++I)
          srcUseList.push_back(*I);
      else
        return false;
    } else if (UI != C && UI != cpy) {
      return false;
    }
  }

  // Since we're changing the parameter to the callsite, we need to make sure
  // that what would be the new parameter dominates the callsite.
  DominatorTree &DT = getAnalysis<DominatorTree>();
  if (Instruction *cpyDestInst = dyn_cast<Instruction>(cpyDest))
    if (!DT.dominates(cpyDestInst, C))
      return false;

  // In addition to knowing that the call does not access src in some
  // unexpected manner, for example via a global, which we deduce from
  // the use analysis, we also need to know that it does not sneakily
  // access dest.  We rely on AA to figure this out for us.
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  if (AA.getModRefInfo(C, cpyDest, srcSize) !=
      AliasAnalysis::NoModRef)
    return false;

  // All the checks have passed, so do the transformation.
  bool changedArgument = false;
  for (unsigned i = 0; i < CS.arg_size(); ++i)
    if (CS.getArgument(i)->stripPointerCasts() == cpySrc) {
      if (cpySrc->getType() != cpyDest->getType())
        cpyDest = CastInst::CreatePointerCast(cpyDest, cpySrc->getType(),
                                              cpyDest->getName(), C);
      changedArgument = true;
      if (CS.getArgument(i)->getType() == cpyDest->getType())
        CS.setArgument(i, cpyDest);
      else
        CS.setArgument(i, CastInst::CreatePointerCast(cpyDest, 
                          CS.getArgument(i)->getType(), cpyDest->getName(), C));
    }

  if (!changedArgument)
    return false;

  // Drop any cached information about the call, because we may have changed
  // its dependence information by changing its parameter.
  MemoryDependenceAnalysis &MD = getAnalysis<MemoryDependenceAnalysis>();
  MD.removeInstruction(C);

  // Remove the memcpy
  MD.removeInstruction(cpy);
  ++NumMemCpyInstr;

  return true;
}

/// processMemCpyMemCpyDependence - We've found that the (upward scanning)
/// memory dependence of memcpy 'M' is the memcpy 'MDep'.  Try to simplify M to
/// copy from MDep's input if we can.  MSize is the size of M's copy.
/// 
bool MemCpyOpt::processMemCpyMemCpyDependence(MemCpyInst *M, MemCpyInst *MDep,
                                              uint64_t MSize) {
  // We can only transforms memcpy's where the dest of one is the source of the
  // other.
  if (M->getSource() != MDep->getDest())
    return false;
  
  // Second, the length of the memcpy's must be the same, or the preceeding one
  // must be larger than the following one.
  ConstantInt *C1 = dyn_cast<ConstantInt>(MDep->getLength());
  if (!C1) return false;
  
  uint64_t DepSize = C1->getValue().getZExtValue();
  
  if (DepSize < MSize)
    return false;
  
  // Finally, we have to make sure that the dest of the second does not
  // alias the source of the first.
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  if (!AA.isNoAlias(M->getRawDest(), MSize, MDep->getRawSource(), DepSize) ||
      !AA.isNoAlias(M->getRawDest(), MSize, M->getRawSource(), MSize))
    return false;
  
  // If all checks passed, then we can transform these memcpy's
  const Type *ArgTys[3] = {
    M->getRawDest()->getType(),
    MDep->getRawSource()->getType(),
    M->getLength()->getType()
  };
  Function *MemCpyFun =
    Intrinsic::getDeclaration(M->getParent()->getParent()->getParent(),
                              M->getIntrinsicID(), ArgTys, 3);
  
  // Make sure to use the lesser of the alignment of the source and the dest
  // since we're changing where we're reading from, but don't want to increase
  // the alignment past what can be read from or written to.
  // TODO: Is this worth it if we're creating a less aligned memcpy? For
  // example we could be moving from movaps -> movq on x86.
  unsigned Align = std::min(MDep->getAlignmentCst()->getZExtValue(),
                            M->getAlignmentCst()->getZExtValue());
  LLVMContext &Context = M->getContext();
  ConstantInt *AlignCI = ConstantInt::get(Type::getInt32Ty(Context), Align);
  Value *Args[5] = {
    M->getRawDest(), MDep->getRawSource(), M->getLength(),
    AlignCI, M->getVolatileCst()
  };
  CallInst *C = CallInst::Create(MemCpyFun, Args, Args+5, "", M);
  
  
  MemoryDependenceAnalysis &MD = getAnalysis<MemoryDependenceAnalysis>();

  // If C and M don't interfere, then this is a valid transformation.  If they
  // did, this would mean that the two sources overlap, which would be bad.
  MemDepResult dep = MD.getDependency(C);
  if (dep.isClobber() && dep.getInst() == MDep) {
    MD.removeInstruction(M);
    M->eraseFromParent();
    ++NumMemCpyInstr;
    return true;
  }
  
  // Otherwise, there was no point in doing this, so we remove the call we
  // inserted and act like nothing happened.
  MD.removeInstruction(C);
  C->eraseFromParent();
  return false;
}


/// processMemCpy - perform simplification of memcpy's.  If we have memcpy A
/// which copies X to Y, and memcpy B which copies Y to Z, then we can rewrite
/// B to be a memcpy from X to Z (or potentially a memmove, depending on
/// circumstances). This allows later passes to remove the first memcpy
/// altogether.
bool MemCpyOpt::processMemCpy(MemCpyInst *M) {
  MemoryDependenceAnalysis &MD = getAnalysis<MemoryDependenceAnalysis>();

  // We can only optimize statically-sized memcpy's.
  ConstantInt *cpyLen = dyn_cast<ConstantInt>(M->getLength());
  if (!cpyLen) return false;

  // The are two possible optimizations we can do for memcpy:
  //   a) memcpy-memcpy xform which exposes redundance for DSE.
  //   b) call-memcpy xform for return slot optimization.
  MemDepResult dep = MD.getDependency(M);
  if (!dep.isClobber())
    return false;
  
  if (MemCpyInst *MDep = dyn_cast<MemCpyInst>(dep.getInst()))
    return processMemCpyMemCpyDependence(M, MDep, cpyLen->getZExtValue());
    
  if (CallInst *C = dyn_cast<CallInst>(dep.getInst())) {
    bool changed = performCallSlotOptzn(M, M->getDest(), M->getSource(),
                                        cpyLen->getZExtValue(), C);
    if (changed) M->eraseFromParent();
    return changed;
  }
  return false;
}

/// processMemMove - Transforms memmove calls to memcpy calls when the src/dst
/// are guaranteed not to alias.
bool MemCpyOpt::processMemMove(MemMoveInst *M) {
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();

  // If the memmove is a constant size, use it for the alias query, this allows
  // us to optimize things like: memmove(P, P+64, 64);
  uint64_t MemMoveSize = AliasAnalysis::UnknownSize;
  if (ConstantInt *Len = dyn_cast<ConstantInt>(M->getLength()))
    MemMoveSize = Len->getZExtValue();
  
  // See if the pointers alias.
  if (AA.alias(M->getRawDest(), MemMoveSize, M->getRawSource(), MemMoveSize) !=
      AliasAnalysis::NoAlias)
    return false;
  
  DEBUG(dbgs() << "MemCpyOpt: Optimizing memmove -> memcpy: " << *M << "\n");
  
  // If not, then we know we can transform this.
  Module *Mod = M->getParent()->getParent()->getParent();
  const Type *ArgTys[3] = { M->getRawDest()->getType(),
                            M->getRawSource()->getType(),
                            M->getLength()->getType() };
  M->setCalledFunction(Intrinsic::getDeclaration(Mod, Intrinsic::memcpy,
                                                 ArgTys, 3));

  // MemDep may have over conservative information about this instruction, just
  // conservatively flush it from the cache.
  getAnalysis<MemoryDependenceAnalysis>().removeInstruction(M);

  ++NumMoveToCpy;
  return true;
}
  

// MemCpyOpt::iterateOnFunction - Executes one iteration of GVN.
bool MemCpyOpt::iterateOnFunction(Function &F) {
  bool MadeChange = false;

  // Walk all instruction in the function.
  for (Function::iterator BB = F.begin(), BBE = F.end(); BB != BBE; ++BB) {
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
         BI != BE;) {
      // Avoid invalidating the iterator.
      Instruction *I = BI++;
      
      if (StoreInst *SI = dyn_cast<StoreInst>(I))
        MadeChange |= processStore(SI, BI);
      else if (MemCpyInst *M = dyn_cast<MemCpyInst>(I))
        MadeChange |= processMemCpy(M);
      else if (MemMoveInst *M = dyn_cast<MemMoveInst>(I)) {
        if (processMemMove(M)) {
          --BI;         // Reprocess the new memcpy.
          MadeChange = true;
        }
      }
    }
  }
  
  return MadeChange;
}

// MemCpyOpt::runOnFunction - This is the main transformation entry point for a
// function.
//
bool MemCpyOpt::runOnFunction(Function &F) {
  bool MadeChange = false;
  while (1) {
    if (!iterateOnFunction(F))
      break;
    MadeChange = true;
  }
  
  return MadeChange;
}



