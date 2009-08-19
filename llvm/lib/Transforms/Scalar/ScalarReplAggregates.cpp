//===- ScalarReplAggregates.cpp - Scalar Replacement of Aggregates --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation implements the well known scalar replacement of
// aggregates transformation.  This xform breaks up alloca instructions of
// aggregate type (structure or array) into individual alloca instructions for
// each member (if possible).  Then, if possible, it transforms the individual
// alloca instructions into nice clean scalar SSA form.
//
// This combines a simple SRoA algorithm with the Mem2Reg algorithm because
// often interact, especially for C++ programs.  As such, iterating between
// SRoA, then Mem2Reg until we run out of things to promote works well.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "scalarrepl"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumReplaced,  "Number of allocas broken up");
STATISTIC(NumPromoted,  "Number of allocas promoted");
STATISTIC(NumConverted, "Number of aggregates converted to scalar");
STATISTIC(NumGlobals,   "Number of allocas copied from constant global");

namespace {
  struct VISIBILITY_HIDDEN SROA : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    explicit SROA(signed T = -1) : FunctionPass(&ID) {
      if (T == -1)
        SRThreshold = 128;
      else
        SRThreshold = T;
    }

    bool runOnFunction(Function &F);

    bool performScalarRepl(Function &F);
    bool performPromotion(Function &F);

    // getAnalysisUsage - This pass does not require any passes, but we know it
    // will not alter the CFG, so say so.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.addRequired<DominanceFrontier>();
      AU.setPreservesCFG();
    }

  private:
    TargetData *TD;
    
    /// AllocaInfo - When analyzing uses of an alloca instruction, this captures
    /// information about the uses.  All these fields are initialized to false
    /// and set to true when something is learned.
    struct AllocaInfo {
      /// isUnsafe - This is set to true if the alloca cannot be SROA'd.
      bool isUnsafe : 1;
      
      /// needsCleanup - This is set to true if there is some use of the alloca
      /// that requires cleanup.
      bool needsCleanup : 1;
      
      /// isMemCpySrc - This is true if this aggregate is memcpy'd from.
      bool isMemCpySrc : 1;

      /// isMemCpyDst - This is true if this aggregate is memcpy'd into.
      bool isMemCpyDst : 1;

      AllocaInfo()
        : isUnsafe(false), needsCleanup(false), 
          isMemCpySrc(false), isMemCpyDst(false) {}
    };
    
    unsigned SRThreshold;

    void MarkUnsafe(AllocaInfo &I) { I.isUnsafe = true; }

    int isSafeAllocaToScalarRepl(AllocationInst *AI);

    void isSafeUseOfAllocation(Instruction *User, AllocationInst *AI,
                               AllocaInfo &Info);
    void isSafeElementUse(Value *Ptr, bool isFirstElt, AllocationInst *AI,
                         AllocaInfo &Info);
    void isSafeMemIntrinsicOnAllocation(MemIntrinsic *MI, AllocationInst *AI,
                                        unsigned OpNo, AllocaInfo &Info);
    void isSafeUseOfBitCastedAllocation(BitCastInst *User, AllocationInst *AI,
                                        AllocaInfo &Info);
    
    void DoScalarReplacement(AllocationInst *AI, 
                             std::vector<AllocationInst*> &WorkList);
    void CleanupGEP(GetElementPtrInst *GEP);
    void CleanupAllocaUsers(AllocationInst *AI);
    AllocaInst *AddNewAlloca(Function &F, const Type *Ty, AllocationInst *Base);
    
    void RewriteBitCastUserOfAlloca(Instruction *BCInst, AllocationInst *AI,
                                    SmallVector<AllocaInst*, 32> &NewElts);
    
    void RewriteMemIntrinUserOfAlloca(MemIntrinsic *MI, Instruction *BCInst,
                                      AllocationInst *AI,
                                      SmallVector<AllocaInst*, 32> &NewElts);
    void RewriteStoreUserOfWholeAlloca(StoreInst *SI, AllocationInst *AI,
                                       SmallVector<AllocaInst*, 32> &NewElts);
    void RewriteLoadUserOfWholeAlloca(LoadInst *LI, AllocationInst *AI,
                                      SmallVector<AllocaInst*, 32> &NewElts);
    
    bool CanConvertToScalar(Value *V, bool &IsNotTrivial, const Type *&VecTy,
                            bool &SawVec, uint64_t Offset, unsigned AllocaSize);
    void ConvertUsesToScalar(Value *Ptr, AllocaInst *NewAI, uint64_t Offset);
    Value *ConvertScalar_ExtractValue(Value *NV, const Type *ToType,
                                     uint64_t Offset, IRBuilder<> &Builder);
    Value *ConvertScalar_InsertValue(Value *StoredVal, Value *ExistingVal,
                                     uint64_t Offset, IRBuilder<> &Builder);
    static Instruction *isOnlyCopiedFromConstantGlobal(AllocationInst *AI);
  };
}

char SROA::ID = 0;
static RegisterPass<SROA> X("scalarrepl", "Scalar Replacement of Aggregates");

// Public interface to the ScalarReplAggregates pass
FunctionPass *llvm::createScalarReplAggregatesPass(signed int Threshold) { 
  return new SROA(Threshold);
}


bool SROA::runOnFunction(Function &F) {
  TD = getAnalysisIfAvailable<TargetData>();

  bool Changed = performPromotion(F);

  // FIXME: ScalarRepl currently depends on TargetData more than it
  // theoretically needs to. It should be refactored in order to support
  // target-independent IR. Until this is done, just skip the actual
  // scalar-replacement portion of this pass.
  if (!TD) return Changed;

  while (1) {
    bool LocalChange = performScalarRepl(F);
    if (!LocalChange) break;   // No need to repromote if no scalarrepl
    Changed = true;
    LocalChange = performPromotion(F);
    if (!LocalChange) break;   // No need to re-scalarrepl if no promotion
  }

  return Changed;
}


bool SROA::performPromotion(Function &F) {
  std::vector<AllocaInst*> Allocas;
  DominatorTree         &DT = getAnalysis<DominatorTree>();
  DominanceFrontier &DF = getAnalysis<DominanceFrontier>();

  BasicBlock &BB = F.getEntryBlock();  // Get the entry node for the function

  bool Changed = false;

  while (1) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I))       // Is it an alloca?
        if (isAllocaPromotable(AI))
          Allocas.push_back(AI);

    if (Allocas.empty()) break;

    PromoteMemToReg(Allocas, DT, DF, F.getContext());
    NumPromoted += Allocas.size();
    Changed = true;
  }

  return Changed;
}

/// getNumSAElements - Return the number of elements in the specific struct or
/// array.
static uint64_t getNumSAElements(const Type *T) {
  if (const StructType *ST = dyn_cast<StructType>(T))
    return ST->getNumElements();
  return cast<ArrayType>(T)->getNumElements();
}

// performScalarRepl - This algorithm is a simple worklist driven algorithm,
// which runs on all of the malloc/alloca instructions in the function, removing
// them if they are only used by getelementptr instructions.
//
bool SROA::performScalarRepl(Function &F) {
  std::vector<AllocationInst*> WorkList;

  // Scan the entry basic block, adding any alloca's and mallocs to the worklist
  BasicBlock &BB = F.getEntryBlock();
  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
    if (AllocationInst *A = dyn_cast<AllocationInst>(I))
      WorkList.push_back(A);

  // Process the worklist
  bool Changed = false;
  while (!WorkList.empty()) {
    AllocationInst *AI = WorkList.back();
    WorkList.pop_back();
    
    // Handle dead allocas trivially.  These can be formed by SROA'ing arrays
    // with unused elements.
    if (AI->use_empty()) {
      AI->eraseFromParent();
      continue;
    }

    // If this alloca is impossible for us to promote, reject it early.
    if (AI->isArrayAllocation() || !AI->getAllocatedType()->isSized())
      continue;
    
    // Check to see if this allocation is only modified by a memcpy/memmove from
    // a constant global.  If this is the case, we can change all users to use
    // the constant global instead.  This is commonly produced by the CFE by
    // constructs like "void foo() { int A[] = {1,2,3,4,5,6,7,8,9...}; }" if 'A'
    // is only subsequently read.
    if (Instruction *TheCopy = isOnlyCopiedFromConstantGlobal(AI)) {
      DOUT << "Found alloca equal to global: " << *AI;
      DOUT << "  memcpy = " << *TheCopy;
      Constant *TheSrc = cast<Constant>(TheCopy->getOperand(2));
      AI->replaceAllUsesWith(ConstantExpr::getBitCast(TheSrc, AI->getType()));
      TheCopy->eraseFromParent();  // Don't mutate the global.
      AI->eraseFromParent();
      ++NumGlobals;
      Changed = true;
      continue;
    }
    
    // Check to see if we can perform the core SROA transformation.  We cannot
    // transform the allocation instruction if it is an array allocation
    // (allocations OF arrays are ok though), and an allocation of a scalar
    // value cannot be decomposed at all.
    uint64_t AllocaSize = TD->getTypeAllocSize(AI->getAllocatedType());

    // Do not promote [0 x %struct].
    if (AllocaSize == 0) continue;

    // Do not promote any struct whose size is too big.
    if (AllocaSize > SRThreshold) continue;

    if ((isa<StructType>(AI->getAllocatedType()) ||
         isa<ArrayType>(AI->getAllocatedType())) &&
        // Do not promote any struct into more than "32" separate vars.
        getNumSAElements(AI->getAllocatedType()) <= SRThreshold/4) {
      // Check that all of the users of the allocation are capable of being
      // transformed.
      switch (isSafeAllocaToScalarRepl(AI)) {
      default: llvm_unreachable("Unexpected value!");
      case 0:  // Not safe to scalar replace.
        break;
      case 1:  // Safe, but requires cleanup/canonicalizations first
        CleanupAllocaUsers(AI);
        // FALL THROUGH.
      case 3:  // Safe to scalar replace.
        DoScalarReplacement(AI, WorkList);
        Changed = true;
        continue;
      }
    }

    // If we can turn this aggregate value (potentially with casts) into a
    // simple scalar value that can be mem2reg'd into a register value.
    // IsNotTrivial tracks whether this is something that mem2reg could have
    // promoted itself.  If so, we don't want to transform it needlessly.  Note
    // that we can't just check based on the type: the alloca may be of an i32
    // but that has pointer arithmetic to set byte 3 of it or something.
    bool IsNotTrivial = false;
    const Type *VectorTy = 0;
    bool HadAVector = false;
    if (CanConvertToScalar(AI, IsNotTrivial, VectorTy, HadAVector, 
                           0, unsigned(AllocaSize)) && IsNotTrivial) {
      AllocaInst *NewAI;
      // If we were able to find a vector type that can handle this with
      // insert/extract elements, and if there was at least one use that had
      // a vector type, promote this to a vector.  We don't want to promote
      // random stuff that doesn't use vectors (e.g. <9 x double>) because then
      // we just get a lot of insert/extracts.  If at least one vector is
      // involved, then we probably really do have a union of vector/array.
      if (VectorTy && isa<VectorType>(VectorTy) && HadAVector) {
        DOUT << "CONVERT TO VECTOR: " << *AI << "  TYPE = " << *VectorTy <<"\n";
        
        // Create and insert the vector alloca.
        NewAI = new AllocaInst(VectorTy, 0, "",  AI->getParent()->begin());
        ConvertUsesToScalar(AI, NewAI, 0);
      } else {
        DOUT << "CONVERT TO SCALAR INTEGER: " << *AI << "\n";
        
        // Create and insert the integer alloca.
        const Type *NewTy = IntegerType::get(AI->getContext(), AllocaSize*8);
        NewAI = new AllocaInst(NewTy, 0, "", AI->getParent()->begin());
        ConvertUsesToScalar(AI, NewAI, 0);
      }
      NewAI->takeName(AI);
      AI->eraseFromParent();
      ++NumConverted;
      Changed = true;
      continue;
    }
    
    // Otherwise, couldn't process this alloca.
  }

  return Changed;
}

/// DoScalarReplacement - This alloca satisfied the isSafeAllocaToScalarRepl
/// predicate, do SROA now.
void SROA::DoScalarReplacement(AllocationInst *AI, 
                               std::vector<AllocationInst*> &WorkList) {
  DOUT << "Found inst to SROA: " << *AI;
  SmallVector<AllocaInst*, 32> ElementAllocas;
  if (const StructType *ST = dyn_cast<StructType>(AI->getAllocatedType())) {
    ElementAllocas.reserve(ST->getNumContainedTypes());
    for (unsigned i = 0, e = ST->getNumContainedTypes(); i != e; ++i) {
      AllocaInst *NA = new AllocaInst(ST->getContainedType(i), 0, 
                                      AI->getAlignment(),
                                      AI->getName() + "." + Twine(i), AI);
      ElementAllocas.push_back(NA);
      WorkList.push_back(NA);  // Add to worklist for recursive processing
    }
  } else {
    const ArrayType *AT = cast<ArrayType>(AI->getAllocatedType());
    ElementAllocas.reserve(AT->getNumElements());
    const Type *ElTy = AT->getElementType();
    for (unsigned i = 0, e = AT->getNumElements(); i != e; ++i) {
      AllocaInst *NA = new AllocaInst(ElTy, 0, AI->getAlignment(),
                                      AI->getName() + "." + Twine(i), AI);
      ElementAllocas.push_back(NA);
      WorkList.push_back(NA);  // Add to worklist for recursive processing
    }
  }

  // Now that we have created the alloca instructions that we want to use,
  // expand the getelementptr instructions to use them.
  //
  while (!AI->use_empty()) {
    Instruction *User = cast<Instruction>(AI->use_back());
    if (BitCastInst *BCInst = dyn_cast<BitCastInst>(User)) {
      RewriteBitCastUserOfAlloca(BCInst, AI, ElementAllocas);
      BCInst->eraseFromParent();
      continue;
    }
    
    // Replace:
    //   %res = load { i32, i32 }* %alloc
    // with:
    //   %load.0 = load i32* %alloc.0
    //   %insert.0 insertvalue { i32, i32 } zeroinitializer, i32 %load.0, 0 
    //   %load.1 = load i32* %alloc.1
    //   %insert = insertvalue { i32, i32 } %insert.0, i32 %load.1, 1 
    // (Also works for arrays instead of structs)
    if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
      Value *Insert = UndefValue::get(LI->getType());
      for (unsigned i = 0, e = ElementAllocas.size(); i != e; ++i) {
        Value *Load = new LoadInst(ElementAllocas[i], "load", LI);
        Insert = InsertValueInst::Create(Insert, Load, i, "insert", LI);
      }
      LI->replaceAllUsesWith(Insert);
      LI->eraseFromParent();
      continue;
    }

    // Replace:
    //   store { i32, i32 } %val, { i32, i32 }* %alloc
    // with:
    //   %val.0 = extractvalue { i32, i32 } %val, 0 
    //   store i32 %val.0, i32* %alloc.0
    //   %val.1 = extractvalue { i32, i32 } %val, 1 
    //   store i32 %val.1, i32* %alloc.1
    // (Also works for arrays instead of structs)
    if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
      Value *Val = SI->getOperand(0);
      for (unsigned i = 0, e = ElementAllocas.size(); i != e; ++i) {
        Value *Extract = ExtractValueInst::Create(Val, i, Val->getName(), SI);
        new StoreInst(Extract, ElementAllocas[i], SI);
      }
      SI->eraseFromParent();
      continue;
    }
    
    GetElementPtrInst *GEPI = cast<GetElementPtrInst>(User);
    // We now know that the GEP is of the form: GEP <ptr>, 0, <cst>
    unsigned Idx =
       (unsigned)cast<ConstantInt>(GEPI->getOperand(2))->getZExtValue();

    assert(Idx < ElementAllocas.size() && "Index out of range?");
    AllocaInst *AllocaToUse = ElementAllocas[Idx];

    Value *RepValue;
    if (GEPI->getNumOperands() == 3) {
      // Do not insert a new getelementptr instruction with zero indices, only
      // to have it optimized out later.
      RepValue = AllocaToUse;
    } else {
      // We are indexing deeply into the structure, so we still need a
      // getelement ptr instruction to finish the indexing.  This may be
      // expanded itself once the worklist is rerun.
      //
      SmallVector<Value*, 8> NewArgs;
      NewArgs.push_back(Constant::getNullValue(
                                           Type::getInt32Ty(AI->getContext())));
      NewArgs.append(GEPI->op_begin()+3, GEPI->op_end());
      RepValue = GetElementPtrInst::Create(AllocaToUse, NewArgs.begin(),
                                           NewArgs.end(), "", GEPI);
      RepValue->takeName(GEPI);
    }
    
    // If this GEP is to the start of the aggregate, check for memcpys.
    if (Idx == 0 && GEPI->hasAllZeroIndices())
      RewriteBitCastUserOfAlloca(GEPI, AI, ElementAllocas);

    // Move all of the users over to the new GEP.
    GEPI->replaceAllUsesWith(RepValue);
    // Delete the old GEP
    GEPI->eraseFromParent();
  }

  // Finally, delete the Alloca instruction
  AI->eraseFromParent();
  NumReplaced++;
}


/// isSafeElementUse - Check to see if this use is an allowed use for a
/// getelementptr instruction of an array aggregate allocation.  isFirstElt
/// indicates whether Ptr is known to the start of the aggregate.
///
void SROA::isSafeElementUse(Value *Ptr, bool isFirstElt, AllocationInst *AI,
                            AllocaInfo &Info) {
  for (Value::use_iterator I = Ptr->use_begin(), E = Ptr->use_end();
       I != E; ++I) {
    Instruction *User = cast<Instruction>(*I);
    switch (User->getOpcode()) {
    case Instruction::Load:  break;
    case Instruction::Store:
      // Store is ok if storing INTO the pointer, not storing the pointer
      if (User->getOperand(0) == Ptr) return MarkUnsafe(Info);
      break;
    case Instruction::GetElementPtr: {
      GetElementPtrInst *GEP = cast<GetElementPtrInst>(User);
      bool AreAllZeroIndices = isFirstElt;
      if (GEP->getNumOperands() > 1) {
        if (!isa<ConstantInt>(GEP->getOperand(1)) ||
            !cast<ConstantInt>(GEP->getOperand(1))->isZero())
          // Using pointer arithmetic to navigate the array.
          return MarkUnsafe(Info);
       
        if (AreAllZeroIndices)
          AreAllZeroIndices = GEP->hasAllZeroIndices();
      }
      isSafeElementUse(GEP, AreAllZeroIndices, AI, Info);
      if (Info.isUnsafe) return;
      break;
    }
    case Instruction::BitCast:
      if (isFirstElt) {
        isSafeUseOfBitCastedAllocation(cast<BitCastInst>(User), AI, Info);
        if (Info.isUnsafe) return;
        break;
      }
      DOUT << "  Transformation preventing inst: " << *User;
      return MarkUnsafe(Info);
    case Instruction::Call:
      if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(User)) {
        if (isFirstElt) {
          isSafeMemIntrinsicOnAllocation(MI, AI, I.getOperandNo(), Info);
          if (Info.isUnsafe) return;
          break;
        }
      }
      DOUT << "  Transformation preventing inst: " << *User;
      return MarkUnsafe(Info);
    default:
      DOUT << "  Transformation preventing inst: " << *User;
      return MarkUnsafe(Info);
    }
  }
  return;  // All users look ok :)
}

/// AllUsersAreLoads - Return true if all users of this value are loads.
static bool AllUsersAreLoads(Value *Ptr) {
  for (Value::use_iterator I = Ptr->use_begin(), E = Ptr->use_end();
       I != E; ++I)
    if (cast<Instruction>(*I)->getOpcode() != Instruction::Load)
      return false;
  return true;
}

/// isSafeUseOfAllocation - Check to see if this user is an allowed use for an
/// aggregate allocation.
///
void SROA::isSafeUseOfAllocation(Instruction *User, AllocationInst *AI,
                                 AllocaInfo &Info) {
  if (BitCastInst *C = dyn_cast<BitCastInst>(User))
    return isSafeUseOfBitCastedAllocation(C, AI, Info);

  if (LoadInst *LI = dyn_cast<LoadInst>(User))
    if (!LI->isVolatile())
      return;// Loads (returning a first class aggregrate) are always rewritable

  if (StoreInst *SI = dyn_cast<StoreInst>(User))
    if (!SI->isVolatile() && SI->getOperand(0) != AI)
      return;// Store is ok if storing INTO the pointer, not storing the pointer
 
  GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(User);
  if (GEPI == 0)
    return MarkUnsafe(Info);

  gep_type_iterator I = gep_type_begin(GEPI), E = gep_type_end(GEPI);

  // The GEP is not safe to transform if not of the form "GEP <ptr>, 0, <cst>".
  if (I == E ||
      I.getOperand() != Constant::getNullValue(I.getOperand()->getType())) {
    return MarkUnsafe(Info);
  }

  ++I;
  if (I == E) return MarkUnsafe(Info);  // ran out of GEP indices??

  bool IsAllZeroIndices = true;
  
  // If the first index is a non-constant index into an array, see if we can
  // handle it as a special case.
  if (const ArrayType *AT = dyn_cast<ArrayType>(*I)) {
    if (!isa<ConstantInt>(I.getOperand())) {
      IsAllZeroIndices = 0;
      uint64_t NumElements = AT->getNumElements();
      
      // If this is an array index and the index is not constant, we cannot
      // promote... that is unless the array has exactly one or two elements in
      // it, in which case we CAN promote it, but we have to canonicalize this
      // out if this is the only problem.
      if ((NumElements == 1 || NumElements == 2) &&
          AllUsersAreLoads(GEPI)) {
        Info.needsCleanup = true;
        return;  // Canonicalization required!
      }
      return MarkUnsafe(Info);
    }
  }
 
  // Walk through the GEP type indices, checking the types that this indexes
  // into.
  for (; I != E; ++I) {
    // Ignore struct elements, no extra checking needed for these.
    if (isa<StructType>(*I))
      continue;
    
    ConstantInt *IdxVal = dyn_cast<ConstantInt>(I.getOperand());
    if (!IdxVal) return MarkUnsafe(Info);

    // Are all indices still zero?
    IsAllZeroIndices &= IdxVal->isZero();
    
    if (const ArrayType *AT = dyn_cast<ArrayType>(*I)) {
      // This GEP indexes an array.  Verify that this is an in-range constant
      // integer. Specifically, consider A[0][i]. We cannot know that the user
      // isn't doing invalid things like allowing i to index an out-of-range
      // subscript that accesses A[1].  Because of this, we have to reject SROA
      // of any accesses into structs where any of the components are variables. 
      if (IdxVal->getZExtValue() >= AT->getNumElements())
        return MarkUnsafe(Info);
    } else if (const VectorType *VT = dyn_cast<VectorType>(*I)) {
      if (IdxVal->getZExtValue() >= VT->getNumElements())
        return MarkUnsafe(Info);
    }
  }
  
  // If there are any non-simple uses of this getelementptr, make sure to reject
  // them.
  return isSafeElementUse(GEPI, IsAllZeroIndices, AI, Info);
}

/// isSafeMemIntrinsicOnAllocation - Return true if the specified memory
/// intrinsic can be promoted by SROA.  At this point, we know that the operand
/// of the memintrinsic is a pointer to the beginning of the allocation.
void SROA::isSafeMemIntrinsicOnAllocation(MemIntrinsic *MI, AllocationInst *AI,
                                          unsigned OpNo, AllocaInfo &Info) {
  // If not constant length, give up.
  ConstantInt *Length = dyn_cast<ConstantInt>(MI->getLength());
  if (!Length) return MarkUnsafe(Info);
  
  // If not the whole aggregate, give up.
  if (Length->getZExtValue() !=
      TD->getTypeAllocSize(AI->getType()->getElementType()))
    return MarkUnsafe(Info);
  
  // We only know about memcpy/memset/memmove.
  if (!isa<MemIntrinsic>(MI))
    return MarkUnsafe(Info);
  
  // Otherwise, we can transform it.  Determine whether this is a memcpy/set
  // into or out of the aggregate.
  if (OpNo == 1)
    Info.isMemCpyDst = true;
  else {
    assert(OpNo == 2);
    Info.isMemCpySrc = true;
  }
}

/// isSafeUseOfBitCastedAllocation - Return true if all users of this bitcast
/// are 
void SROA::isSafeUseOfBitCastedAllocation(BitCastInst *BC, AllocationInst *AI,
                                          AllocaInfo &Info) {
  for (Value::use_iterator UI = BC->use_begin(), E = BC->use_end();
       UI != E; ++UI) {
    if (BitCastInst *BCU = dyn_cast<BitCastInst>(UI)) {
      isSafeUseOfBitCastedAllocation(BCU, AI, Info);
    } else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(UI)) {
      isSafeMemIntrinsicOnAllocation(MI, AI, UI.getOperandNo(), Info);
    } else if (StoreInst *SI = dyn_cast<StoreInst>(UI)) {
      if (SI->isVolatile())
        return MarkUnsafe(Info);
      
      // If storing the entire alloca in one chunk through a bitcasted pointer
      // to integer, we can transform it.  This happens (for example) when you
      // cast a {i32,i32}* to i64* and store through it.  This is similar to the
      // memcpy case and occurs in various "byval" cases and emulated memcpys.
      if (isa<IntegerType>(SI->getOperand(0)->getType()) &&
          TD->getTypeAllocSize(SI->getOperand(0)->getType()) ==
          TD->getTypeAllocSize(AI->getType()->getElementType())) {
        Info.isMemCpyDst = true;
        continue;
      }
      return MarkUnsafe(Info);
    } else if (LoadInst *LI = dyn_cast<LoadInst>(UI)) {
      if (LI->isVolatile())
        return MarkUnsafe(Info);

      // If loading the entire alloca in one chunk through a bitcasted pointer
      // to integer, we can transform it.  This happens (for example) when you
      // cast a {i32,i32}* to i64* and load through it.  This is similar to the
      // memcpy case and occurs in various "byval" cases and emulated memcpys.
      if (isa<IntegerType>(LI->getType()) &&
          TD->getTypeAllocSize(LI->getType()) ==
          TD->getTypeAllocSize(AI->getType()->getElementType())) {
        Info.isMemCpySrc = true;
        continue;
      }
      return MarkUnsafe(Info);
    } else if (isa<DbgInfoIntrinsic>(UI)) {
      // If one user is DbgInfoIntrinsic then check if all users are
      // DbgInfoIntrinsics.
      if (OnlyUsedByDbgInfoIntrinsics(BC)) {
        Info.needsCleanup = true;
        return;
      }
      else
        MarkUnsafe(Info);
    }
    else {
      return MarkUnsafe(Info);
    }
    if (Info.isUnsafe) return;
  }
}

/// RewriteBitCastUserOfAlloca - BCInst (transitively) bitcasts AI, or indexes
/// to its first element.  Transform users of the cast to use the new values
/// instead.
void SROA::RewriteBitCastUserOfAlloca(Instruction *BCInst, AllocationInst *AI,
                                      SmallVector<AllocaInst*, 32> &NewElts) {
  Value::use_iterator UI = BCInst->use_begin(), UE = BCInst->use_end();
  while (UI != UE) {
    Instruction *User = cast<Instruction>(*UI++);
    if (BitCastInst *BCU = dyn_cast<BitCastInst>(User)) {
      RewriteBitCastUserOfAlloca(BCU, AI, NewElts);
      if (BCU->use_empty()) BCU->eraseFromParent();
      continue;
    }

    if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(User)) {
      // This must be memcpy/memmove/memset of the entire aggregate.
      // Split into one per element.
      RewriteMemIntrinUserOfAlloca(MI, BCInst, AI, NewElts);
      continue;
    }
      
    if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
      // If this is a store of the entire alloca from an integer, rewrite it.
      RewriteStoreUserOfWholeAlloca(SI, AI, NewElts);
      continue;
    }

    if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
      // If this is a load of the entire alloca to an integer, rewrite it.
      RewriteLoadUserOfWholeAlloca(LI, AI, NewElts);
      continue;
    }
    
    // Otherwise it must be some other user of a gep of the first pointer.  Just
    // leave these alone.
    continue;
  }
}

/// RewriteMemIntrinUserOfAlloca - MI is a memcpy/memset/memmove from or to AI.
/// Rewrite it to copy or set the elements of the scalarized memory.
void SROA::RewriteMemIntrinUserOfAlloca(MemIntrinsic *MI, Instruction *BCInst,
                                        AllocationInst *AI,
                                        SmallVector<AllocaInst*, 32> &NewElts) {
  
  // If this is a memcpy/memmove, construct the other pointer as the
  // appropriate type.  The "Other" pointer is the pointer that goes to memory
  // that doesn't have anything to do with the alloca that we are promoting. For
  // memset, this Value* stays null.
  Value *OtherPtr = 0;
  LLVMContext &Context = MI->getContext();
  unsigned MemAlignment = MI->getAlignment();
  if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(MI)) { // memmove/memcopy
    if (BCInst == MTI->getRawDest())
      OtherPtr = MTI->getRawSource();
    else {
      assert(BCInst == MTI->getRawSource());
      OtherPtr = MTI->getRawDest();
    }
  }
  
  // If there is an other pointer, we want to convert it to the same pointer
  // type as AI has, so we can GEP through it safely.
  if (OtherPtr) {
    // It is likely that OtherPtr is a bitcast, if so, remove it.
    if (BitCastInst *BC = dyn_cast<BitCastInst>(OtherPtr))
      OtherPtr = BC->getOperand(0);
    // All zero GEPs are effectively bitcasts.
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(OtherPtr))
      if (GEP->hasAllZeroIndices())
        OtherPtr = GEP->getOperand(0);
    
    if (ConstantExpr *BCE = dyn_cast<ConstantExpr>(OtherPtr))
      if (BCE->getOpcode() == Instruction::BitCast)
        OtherPtr = BCE->getOperand(0);
    
    // If the pointer is not the right type, insert a bitcast to the right
    // type.
    if (OtherPtr->getType() != AI->getType())
      OtherPtr = new BitCastInst(OtherPtr, AI->getType(), OtherPtr->getName(),
                                 MI);
  }
  
  // Process each element of the aggregate.
  Value *TheFn = MI->getOperand(0);
  const Type *BytePtrTy = MI->getRawDest()->getType();
  bool SROADest = MI->getRawDest() == BCInst;
  
  Constant *Zero = Constant::getNullValue(Type::getInt32Ty(MI->getContext()));

  for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
    // If this is a memcpy/memmove, emit a GEP of the other element address.
    Value *OtherElt = 0;
    unsigned OtherEltAlign = MemAlignment;
    
    if (OtherPtr) {
      Value *Idx[2] = { Zero,
                      ConstantInt::get(Type::getInt32Ty(MI->getContext()), i) };
      OtherElt = GetElementPtrInst::Create(OtherPtr, Idx, Idx + 2,
                                           OtherPtr->getNameStr()+"."+Twine(i),
                                           MI);
      uint64_t EltOffset;
      const PointerType *OtherPtrTy = cast<PointerType>(OtherPtr->getType());
      if (const StructType *ST =
            dyn_cast<StructType>(OtherPtrTy->getElementType())) {
        EltOffset = TD->getStructLayout(ST)->getElementOffset(i);
      } else {
        const Type *EltTy =
          cast<SequentialType>(OtherPtr->getType())->getElementType();
        EltOffset = TD->getTypeAllocSize(EltTy)*i;
      }
      
      // The alignment of the other pointer is the guaranteed alignment of the
      // element, which is affected by both the known alignment of the whole
      // mem intrinsic and the alignment of the element.  If the alignment of
      // the memcpy (f.e.) is 32 but the element is at a 4-byte offset, then the
      // known alignment is just 4 bytes.
      OtherEltAlign = (unsigned)MinAlign(OtherEltAlign, EltOffset);
    }
    
    Value *EltPtr = NewElts[i];
    const Type *EltTy = cast<PointerType>(EltPtr->getType())->getElementType();
    
    // If we got down to a scalar, insert a load or store as appropriate.
    if (EltTy->isSingleValueType()) {
      if (isa<MemTransferInst>(MI)) {
        if (SROADest) {
          // From Other to Alloca.
          Value *Elt = new LoadInst(OtherElt, "tmp", false, OtherEltAlign, MI);
          new StoreInst(Elt, EltPtr, MI);
        } else {
          // From Alloca to Other.
          Value *Elt = new LoadInst(EltPtr, "tmp", MI);
          new StoreInst(Elt, OtherElt, false, OtherEltAlign, MI);
        }
        continue;
      }
      assert(isa<MemSetInst>(MI));
      
      // If the stored element is zero (common case), just store a null
      // constant.
      Constant *StoreVal;
      if (ConstantInt *CI = dyn_cast<ConstantInt>(MI->getOperand(2))) {
        if (CI->isZero()) {
          StoreVal = Constant::getNullValue(EltTy);  // 0.0, null, 0, <0,0>
        } else {
          // If EltTy is a vector type, get the element type.
          const Type *ValTy = EltTy->getScalarType();

          // Construct an integer with the right value.
          unsigned EltSize = TD->getTypeSizeInBits(ValTy);
          APInt OneVal(EltSize, CI->getZExtValue());
          APInt TotalVal(OneVal);
          // Set each byte.
          for (unsigned i = 0; 8*i < EltSize; ++i) {
            TotalVal = TotalVal.shl(8);
            TotalVal |= OneVal;
          }
          
          // Convert the integer value to the appropriate type.
          StoreVal = ConstantInt::get(Context, TotalVal);
          if (isa<PointerType>(ValTy))
            StoreVal = ConstantExpr::getIntToPtr(StoreVal, ValTy);
          else if (ValTy->isFloatingPoint())
            StoreVal = ConstantExpr::getBitCast(StoreVal, ValTy);
          assert(StoreVal->getType() == ValTy && "Type mismatch!");
          
          // If the requested value was a vector constant, create it.
          if (EltTy != ValTy) {
            unsigned NumElts = cast<VectorType>(ValTy)->getNumElements();
            SmallVector<Constant*, 16> Elts(NumElts, StoreVal);
            StoreVal = ConstantVector::get(&Elts[0], NumElts);
          }
        }
        new StoreInst(StoreVal, EltPtr, MI);
        continue;
      }
      // Otherwise, if we're storing a byte variable, use a memset call for
      // this element.
    }
    
    // Cast the element pointer to BytePtrTy.
    if (EltPtr->getType() != BytePtrTy)
      EltPtr = new BitCastInst(EltPtr, BytePtrTy, EltPtr->getNameStr(), MI);
    
    // Cast the other pointer (if we have one) to BytePtrTy. 
    if (OtherElt && OtherElt->getType() != BytePtrTy)
      OtherElt = new BitCastInst(OtherElt, BytePtrTy,OtherElt->getNameStr(),
                                 MI);
    
    unsigned EltSize = TD->getTypeAllocSize(EltTy);
    
    // Finally, insert the meminst for this element.
    if (isa<MemTransferInst>(MI)) {
      Value *Ops[] = {
        SROADest ? EltPtr : OtherElt,  // Dest ptr
        SROADest ? OtherElt : EltPtr,  // Src ptr
        ConstantInt::get(MI->getOperand(3)->getType(), EltSize), // Size
        // Align
        ConstantInt::get(Type::getInt32Ty(MI->getContext()), OtherEltAlign)
      };
      CallInst::Create(TheFn, Ops, Ops + 4, "", MI);
    } else {
      assert(isa<MemSetInst>(MI));
      Value *Ops[] = {
        EltPtr, MI->getOperand(2),  // Dest, Value,
        ConstantInt::get(MI->getOperand(3)->getType(), EltSize), // Size
        Zero  // Align
      };
      CallInst::Create(TheFn, Ops, Ops + 4, "", MI);
    }
  }
  MI->eraseFromParent();
}

/// RewriteStoreUserOfWholeAlloca - We found an store of an integer that
/// overwrites the entire allocation.  Extract out the pieces of the stored
/// integer and store them individually.
void SROA::RewriteStoreUserOfWholeAlloca(StoreInst *SI,
                                         AllocationInst *AI,
                                         SmallVector<AllocaInst*, 32> &NewElts){
  // Extract each element out of the integer according to its structure offset
  // and store the element value to the individual alloca.
  Value *SrcVal = SI->getOperand(0);
  const Type *AllocaEltTy = AI->getType()->getElementType();
  uint64_t AllocaSizeBits = TD->getTypeAllocSizeInBits(AllocaEltTy);
  
  // If this isn't a store of an integer to the whole alloca, it may be a store
  // to the first element.  Just ignore the store in this case and normal SROA
  // will handle it.
  if (!isa<IntegerType>(SrcVal->getType()) ||
      TD->getTypeAllocSizeInBits(SrcVal->getType()) != AllocaSizeBits)
    return;
  // Handle tail padding by extending the operand
  if (TD->getTypeSizeInBits(SrcVal->getType()) != AllocaSizeBits)
    SrcVal = new ZExtInst(SrcVal,
                          IntegerType::get(SI->getContext(), AllocaSizeBits), 
                          "", SI);

  DOUT << "PROMOTING STORE TO WHOLE ALLOCA: " << *AI << *SI;

  // There are two forms here: AI could be an array or struct.  Both cases
  // have different ways to compute the element offset.
  if (const StructType *EltSTy = dyn_cast<StructType>(AllocaEltTy)) {
    const StructLayout *Layout = TD->getStructLayout(EltSTy);
    
    for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
      // Get the number of bits to shift SrcVal to get the value.
      const Type *FieldTy = EltSTy->getElementType(i);
      uint64_t Shift = Layout->getElementOffsetInBits(i);
      
      if (TD->isBigEndian())
        Shift = AllocaSizeBits-Shift-TD->getTypeAllocSizeInBits(FieldTy);
      
      Value *EltVal = SrcVal;
      if (Shift) {
        Value *ShiftVal = ConstantInt::get(EltVal->getType(), Shift);
        EltVal = BinaryOperator::CreateLShr(EltVal, ShiftVal,
                                            "sroa.store.elt", SI);
      }
      
      // Truncate down to an integer of the right size.
      uint64_t FieldSizeBits = TD->getTypeSizeInBits(FieldTy);
      
      // Ignore zero sized fields like {}, they obviously contain no data.
      if (FieldSizeBits == 0) continue;
      
      if (FieldSizeBits != AllocaSizeBits)
        EltVal = new TruncInst(EltVal,
                             IntegerType::get(SI->getContext(), FieldSizeBits),
                              "", SI);
      Value *DestField = NewElts[i];
      if (EltVal->getType() == FieldTy) {
        // Storing to an integer field of this size, just do it.
      } else if (FieldTy->isFloatingPoint() || isa<VectorType>(FieldTy)) {
        // Bitcast to the right element type (for fp/vector values).
        EltVal = new BitCastInst(EltVal, FieldTy, "", SI);
      } else {
        // Otherwise, bitcast the dest pointer (for aggregates).
        DestField = new BitCastInst(DestField,
                              PointerType::getUnqual(EltVal->getType()),
                                    "", SI);
      }
      new StoreInst(EltVal, DestField, SI);
    }
    
  } else {
    const ArrayType *ATy = cast<ArrayType>(AllocaEltTy);
    const Type *ArrayEltTy = ATy->getElementType();
    uint64_t ElementOffset = TD->getTypeAllocSizeInBits(ArrayEltTy);
    uint64_t ElementSizeBits = TD->getTypeSizeInBits(ArrayEltTy);

    uint64_t Shift;
    
    if (TD->isBigEndian())
      Shift = AllocaSizeBits-ElementOffset;
    else 
      Shift = 0;
    
    for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
      // Ignore zero sized fields like {}, they obviously contain no data.
      if (ElementSizeBits == 0) continue;
      
      Value *EltVal = SrcVal;
      if (Shift) {
        Value *ShiftVal = ConstantInt::get(EltVal->getType(), Shift);
        EltVal = BinaryOperator::CreateLShr(EltVal, ShiftVal,
                                            "sroa.store.elt", SI);
      }
      
      // Truncate down to an integer of the right size.
      if (ElementSizeBits != AllocaSizeBits)
        EltVal = new TruncInst(EltVal, 
                               IntegerType::get(SI->getContext(), 
                                                ElementSizeBits),"",SI);
      Value *DestField = NewElts[i];
      if (EltVal->getType() == ArrayEltTy) {
        // Storing to an integer field of this size, just do it.
      } else if (ArrayEltTy->isFloatingPoint() || isa<VectorType>(ArrayEltTy)) {
        // Bitcast to the right element type (for fp/vector values).
        EltVal = new BitCastInst(EltVal, ArrayEltTy, "", SI);
      } else {
        // Otherwise, bitcast the dest pointer (for aggregates).
        DestField = new BitCastInst(DestField,
                              PointerType::getUnqual(EltVal->getType()),
                                    "", SI);
      }
      new StoreInst(EltVal, DestField, SI);
      
      if (TD->isBigEndian())
        Shift -= ElementOffset;
      else 
        Shift += ElementOffset;
    }
  }
  
  SI->eraseFromParent();
}

/// RewriteLoadUserOfWholeAlloca - We found an load of the entire allocation to
/// an integer.  Load the individual pieces to form the aggregate value.
void SROA::RewriteLoadUserOfWholeAlloca(LoadInst *LI, AllocationInst *AI,
                                        SmallVector<AllocaInst*, 32> &NewElts) {
  // Extract each element out of the NewElts according to its structure offset
  // and form the result value.
  const Type *AllocaEltTy = AI->getType()->getElementType();
  uint64_t AllocaSizeBits = TD->getTypeAllocSizeInBits(AllocaEltTy);
  
  // If this isn't a load of the whole alloca to an integer, it may be a load
  // of the first element.  Just ignore the load in this case and normal SROA
  // will handle it.
  if (!isa<IntegerType>(LI->getType()) ||
      TD->getTypeAllocSizeInBits(LI->getType()) != AllocaSizeBits)
    return;
  
  DOUT << "PROMOTING LOAD OF WHOLE ALLOCA: " << *AI << *LI;
  
  // There are two forms here: AI could be an array or struct.  Both cases
  // have different ways to compute the element offset.
  const StructLayout *Layout = 0;
  uint64_t ArrayEltBitOffset = 0;
  if (const StructType *EltSTy = dyn_cast<StructType>(AllocaEltTy)) {
    Layout = TD->getStructLayout(EltSTy);
  } else {
    const Type *ArrayEltTy = cast<ArrayType>(AllocaEltTy)->getElementType();
    ArrayEltBitOffset = TD->getTypeAllocSizeInBits(ArrayEltTy);
  }    
  
  Value *ResultVal = 
    Constant::getNullValue(IntegerType::get(LI->getContext(), AllocaSizeBits));
  
  for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
    // Load the value from the alloca.  If the NewElt is an aggregate, cast
    // the pointer to an integer of the same size before doing the load.
    Value *SrcField = NewElts[i];
    const Type *FieldTy =
      cast<PointerType>(SrcField->getType())->getElementType();
    uint64_t FieldSizeBits = TD->getTypeSizeInBits(FieldTy);
    
    // Ignore zero sized fields like {}, they obviously contain no data.
    if (FieldSizeBits == 0) continue;
    
    const IntegerType *FieldIntTy = IntegerType::get(LI->getContext(), 
                                                     FieldSizeBits);
    if (!isa<IntegerType>(FieldTy) && !FieldTy->isFloatingPoint() &&
        !isa<VectorType>(FieldTy))
      SrcField = new BitCastInst(SrcField,
                                 PointerType::getUnqual(FieldIntTy),
                                 "", LI);
    SrcField = new LoadInst(SrcField, "sroa.load.elt", LI);

    // If SrcField is a fp or vector of the right size but that isn't an
    // integer type, bitcast to an integer so we can shift it.
    if (SrcField->getType() != FieldIntTy)
      SrcField = new BitCastInst(SrcField, FieldIntTy, "", LI);

    // Zero extend the field to be the same size as the final alloca so that
    // we can shift and insert it.
    if (SrcField->getType() != ResultVal->getType())
      SrcField = new ZExtInst(SrcField, ResultVal->getType(), "", LI);
    
    // Determine the number of bits to shift SrcField.
    uint64_t Shift;
    if (Layout) // Struct case.
      Shift = Layout->getElementOffsetInBits(i);
    else  // Array case.
      Shift = i*ArrayEltBitOffset;
    
    if (TD->isBigEndian())
      Shift = AllocaSizeBits-Shift-FieldIntTy->getBitWidth();
    
    if (Shift) {
      Value *ShiftVal = ConstantInt::get(SrcField->getType(), Shift);
      SrcField = BinaryOperator::CreateShl(SrcField, ShiftVal, "", LI);
    }

    ResultVal = BinaryOperator::CreateOr(SrcField, ResultVal, "", LI);
  }

  // Handle tail padding by truncating the result
  if (TD->getTypeSizeInBits(LI->getType()) != AllocaSizeBits)
    ResultVal = new TruncInst(ResultVal, LI->getType(), "", LI);

  LI->replaceAllUsesWith(ResultVal);
  LI->eraseFromParent();
}


/// HasPadding - Return true if the specified type has any structure or
/// alignment padding, false otherwise.
static bool HasPadding(const Type *Ty, const TargetData &TD) {
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    const StructLayout *SL = TD.getStructLayout(STy);
    unsigned PrevFieldBitOffset = 0;
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      unsigned FieldBitOffset = SL->getElementOffsetInBits(i);

      // Padding in sub-elements?
      if (HasPadding(STy->getElementType(i), TD))
        return true;

      // Check to see if there is any padding between this element and the
      // previous one.
      if (i) {
        unsigned PrevFieldEnd =
        PrevFieldBitOffset+TD.getTypeSizeInBits(STy->getElementType(i-1));
        if (PrevFieldEnd < FieldBitOffset)
          return true;
      }

      PrevFieldBitOffset = FieldBitOffset;
    }

    //  Check for tail padding.
    if (unsigned EltCount = STy->getNumElements()) {
      unsigned PrevFieldEnd = PrevFieldBitOffset +
                   TD.getTypeSizeInBits(STy->getElementType(EltCount-1));
      if (PrevFieldEnd < SL->getSizeInBits())
        return true;
    }

  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    return HasPadding(ATy->getElementType(), TD);
  } else if (const VectorType *VTy = dyn_cast<VectorType>(Ty)) {
    return HasPadding(VTy->getElementType(), TD);
  }
  return TD.getTypeSizeInBits(Ty) != TD.getTypeAllocSizeInBits(Ty);
}

/// isSafeStructAllocaToScalarRepl - Check to see if the specified allocation of
/// an aggregate can be broken down into elements.  Return 0 if not, 3 if safe,
/// or 1 if safe after canonicalization has been performed.
///
int SROA::isSafeAllocaToScalarRepl(AllocationInst *AI) {
  // Loop over the use list of the alloca.  We can only transform it if all of
  // the users are safe to transform.
  AllocaInfo Info;
  
  for (Value::use_iterator I = AI->use_begin(), E = AI->use_end();
       I != E; ++I) {
    isSafeUseOfAllocation(cast<Instruction>(*I), AI, Info);
    if (Info.isUnsafe) {
      DOUT << "Cannot transform: " << *AI << "  due to user: " << **I;
      return 0;
    }
  }
  
  // Okay, we know all the users are promotable.  If the aggregate is a memcpy
  // source and destination, we have to be careful.  In particular, the memcpy
  // could be moving around elements that live in structure padding of the LLVM
  // types, but may actually be used.  In these cases, we refuse to promote the
  // struct.
  if (Info.isMemCpySrc && Info.isMemCpyDst &&
      HasPadding(AI->getType()->getElementType(), *TD))
    return 0;

  // If we require cleanup, return 1, otherwise return 3.
  return Info.needsCleanup ? 1 : 3;
}

/// CleanupGEP - GEP is used by an Alloca, which can be prompted after the GEP
/// is canonicalized here.
void SROA::CleanupGEP(GetElementPtrInst *GEPI) {
  gep_type_iterator I = gep_type_begin(GEPI);
  ++I;
  
  const ArrayType *AT = dyn_cast<ArrayType>(*I);
  if (!AT) 
    return;

  uint64_t NumElements = AT->getNumElements();
  
  if (isa<ConstantInt>(I.getOperand()))
    return;

  if (NumElements == 1) {
    GEPI->setOperand(2, 
                  Constant::getNullValue(Type::getInt32Ty(GEPI->getContext())));
    return;
  } 
    
  assert(NumElements == 2 && "Unhandled case!");
  // All users of the GEP must be loads.  At each use of the GEP, insert
  // two loads of the appropriate indexed GEP and select between them.
  Value *IsOne = new ICmpInst(GEPI, ICmpInst::ICMP_NE, I.getOperand(), 
                              Constant::getNullValue(I.getOperand()->getType()),
                              "isone");
  // Insert the new GEP instructions, which are properly indexed.
  SmallVector<Value*, 8> Indices(GEPI->op_begin()+1, GEPI->op_end());
  Indices[1] = Constant::getNullValue(Type::getInt32Ty(GEPI->getContext()));
  Value *ZeroIdx = GetElementPtrInst::Create(GEPI->getOperand(0),
                                             Indices.begin(),
                                             Indices.end(),
                                             GEPI->getName()+".0", GEPI);
  Indices[1] = ConstantInt::get(Type::getInt32Ty(GEPI->getContext()), 1);
  Value *OneIdx = GetElementPtrInst::Create(GEPI->getOperand(0),
                                            Indices.begin(),
                                            Indices.end(),
                                            GEPI->getName()+".1", GEPI);
  // Replace all loads of the variable index GEP with loads from both
  // indexes and a select.
  while (!GEPI->use_empty()) {
    LoadInst *LI = cast<LoadInst>(GEPI->use_back());
    Value *Zero = new LoadInst(ZeroIdx, LI->getName()+".0", LI);
    Value *One  = new LoadInst(OneIdx , LI->getName()+".1", LI);
    Value *R = SelectInst::Create(IsOne, One, Zero, LI->getName(), LI);
    LI->replaceAllUsesWith(R);
    LI->eraseFromParent();
  }
  GEPI->eraseFromParent();
}


/// CleanupAllocaUsers - If SROA reported that it can promote the specified
/// allocation, but only if cleaned up, perform the cleanups required.
void SROA::CleanupAllocaUsers(AllocationInst *AI) {
  // At this point, we know that the end result will be SROA'd and promoted, so
  // we can insert ugly code if required so long as sroa+mem2reg will clean it
  // up.
  for (Value::use_iterator UI = AI->use_begin(), E = AI->use_end();
       UI != E; ) {
    User *U = *UI++;
    if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(U))
      CleanupGEP(GEPI);
    else {
      Instruction *I = cast<Instruction>(U);
      SmallVector<DbgInfoIntrinsic *, 2> DbgInUses;
      if (!isa<StoreInst>(I) && OnlyUsedByDbgInfoIntrinsics(I, &DbgInUses)) {
        // Safe to remove debug info uses.
        while (!DbgInUses.empty()) {
          DbgInfoIntrinsic *DI = DbgInUses.back(); DbgInUses.pop_back();
          DI->eraseFromParent();
        }
        I->eraseFromParent();
      }
    }
  }
}

/// MergeInType - Add the 'In' type to the accumulated type (Accum) so far at
/// the offset specified by Offset (which is specified in bytes).
///
/// There are two cases we handle here:
///   1) A union of vector types of the same size and potentially its elements.
///      Here we turn element accesses into insert/extract element operations.
///      This promotes a <4 x float> with a store of float to the third element
///      into a <4 x float> that uses insert element.
///   2) A fully general blob of memory, which we turn into some (potentially
///      large) integer type with extract and insert operations where the loads
///      and stores would mutate the memory.
static void MergeInType(const Type *In, uint64_t Offset, const Type *&VecTy,
                        unsigned AllocaSize, const TargetData &TD,
                        LLVMContext &Context) {
  // If this could be contributing to a vector, analyze it.
  if (VecTy != Type::getVoidTy(Context)) { // either null or a vector type.

    // If the In type is a vector that is the same size as the alloca, see if it
    // matches the existing VecTy.
    if (const VectorType *VInTy = dyn_cast<VectorType>(In)) {
      if (VInTy->getBitWidth()/8 == AllocaSize && Offset == 0) {
        // If we're storing/loading a vector of the right size, allow it as a
        // vector.  If this the first vector we see, remember the type so that
        // we know the element size.
        if (VecTy == 0)
          VecTy = VInTy;
        return;
      }
    } else if (In == Type::getFloatTy(Context) ||
               In == Type::getDoubleTy(Context) ||
               (isa<IntegerType>(In) && In->getPrimitiveSizeInBits() >= 8 &&
                isPowerOf2_32(In->getPrimitiveSizeInBits()))) {
      // If we're accessing something that could be an element of a vector, see
      // if the implied vector agrees with what we already have and if Offset is
      // compatible with it.
      unsigned EltSize = In->getPrimitiveSizeInBits()/8;
      if (Offset % EltSize == 0 &&
          AllocaSize % EltSize == 0 &&
          (VecTy == 0 || 
           cast<VectorType>(VecTy)->getElementType()
                 ->getPrimitiveSizeInBits()/8 == EltSize)) {
        if (VecTy == 0)
          VecTy = VectorType::get(In, AllocaSize/EltSize);
        return;
      }
    }
  }
  
  // Otherwise, we have a case that we can't handle with an optimized vector
  // form.  We can still turn this into a large integer.
  VecTy = Type::getVoidTy(Context);
}

/// CanConvertToScalar - V is a pointer.  If we can convert the pointee and all
/// its accesses to use a to single vector type, return true, and set VecTy to
/// the new type.  If we could convert the alloca into a single promotable
/// integer, return true but set VecTy to VoidTy.  Further, if the use is not a
/// completely trivial use that mem2reg could promote, set IsNotTrivial.  Offset
/// is the current offset from the base of the alloca being analyzed.
///
/// If we see at least one access to the value that is as a vector type, set the
/// SawVec flag.
///
bool SROA::CanConvertToScalar(Value *V, bool &IsNotTrivial, const Type *&VecTy,
                              bool &SawVec, uint64_t Offset,
                              unsigned AllocaSize) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI!=E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    
    if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
      // Don't break volatile loads.
      if (LI->isVolatile())
        return false;
      MergeInType(LI->getType(), Offset, VecTy,
                  AllocaSize, *TD, V->getContext());
      SawVec |= isa<VectorType>(LI->getType());
      continue;
    }
    
    if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
      // Storing the pointer, not into the value?
      if (SI->getOperand(0) == V || SI->isVolatile()) return 0;
      MergeInType(SI->getOperand(0)->getType(), Offset,
                  VecTy, AllocaSize, *TD, V->getContext());
      SawVec |= isa<VectorType>(SI->getOperand(0)->getType());
      continue;
    }
    
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(User)) {
      if (!CanConvertToScalar(BCI, IsNotTrivial, VecTy, SawVec, Offset,
                              AllocaSize))
        return false;
      IsNotTrivial = true;
      continue;
    }

    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User)) {
      // If this is a GEP with a variable indices, we can't handle it.
      if (!GEP->hasAllConstantIndices())
        return false;
      
      // Compute the offset that this GEP adds to the pointer.
      SmallVector<Value*, 8> Indices(GEP->op_begin()+1, GEP->op_end());
      uint64_t GEPOffset = TD->getIndexedOffset(GEP->getOperand(0)->getType(),
                                                &Indices[0], Indices.size());
      // See if all uses can be converted.
      if (!CanConvertToScalar(GEP, IsNotTrivial, VecTy, SawVec,Offset+GEPOffset,
                              AllocaSize))
        return false;
      IsNotTrivial = true;
      continue;
    }

    // If this is a constant sized memset of a constant value (e.g. 0) we can
    // handle it.
    if (MemSetInst *MSI = dyn_cast<MemSetInst>(User)) {
      // Store of constant value and constant size.
      if (isa<ConstantInt>(MSI->getValue()) &&
          isa<ConstantInt>(MSI->getLength())) {
        IsNotTrivial = true;
        continue;
      }
    }

    // If this is a memcpy or memmove into or out of the whole allocation, we
    // can handle it like a load or store of the scalar type.
    if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(User)) {
      if (ConstantInt *Len = dyn_cast<ConstantInt>(MTI->getLength()))
        if (Len->getZExtValue() == AllocaSize && Offset == 0) {
          IsNotTrivial = true;
          continue;
        }
    }
    
    // Ignore dbg intrinsic.
    if (isa<DbgInfoIntrinsic>(User))
      continue;

    // Otherwise, we cannot handle this!
    return false;
  }
  
  return true;
}


/// ConvertUsesToScalar - Convert all of the users of Ptr to use the new alloca
/// directly.  This happens when we are converting an "integer union" to a
/// single integer scalar, or when we are converting a "vector union" to a
/// vector with insert/extractelement instructions.
///
/// Offset is an offset from the original alloca, in bits that need to be
/// shifted to the right.  By the end of this, there should be no uses of Ptr.
void SROA::ConvertUsesToScalar(Value *Ptr, AllocaInst *NewAI, uint64_t Offset) {
  while (!Ptr->use_empty()) {
    Instruction *User = cast<Instruction>(Ptr->use_back());

    if (BitCastInst *CI = dyn_cast<BitCastInst>(User)) {
      ConvertUsesToScalar(CI, NewAI, Offset);
      CI->eraseFromParent();
      continue;
    }

    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User)) {
      // Compute the offset that this GEP adds to the pointer.
      SmallVector<Value*, 8> Indices(GEP->op_begin()+1, GEP->op_end());
      uint64_t GEPOffset = TD->getIndexedOffset(GEP->getOperand(0)->getType(),
                                                &Indices[0], Indices.size());
      ConvertUsesToScalar(GEP, NewAI, Offset+GEPOffset*8);
      GEP->eraseFromParent();
      continue;
    }
    
    IRBuilder<> Builder(User->getParent(), User);
    
    if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
      // The load is a bit extract from NewAI shifted right by Offset bits.
      Value *LoadedVal = Builder.CreateLoad(NewAI, "tmp");
      Value *NewLoadVal
        = ConvertScalar_ExtractValue(LoadedVal, LI->getType(), Offset, Builder);
      LI->replaceAllUsesWith(NewLoadVal);
      LI->eraseFromParent();
      continue;
    }
    
    if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
      assert(SI->getOperand(0) != Ptr && "Consistency error!");
      // FIXME: Remove once builder has Twine API.
      Value *Old = Builder.CreateLoad(NewAI, (NewAI->getName()+".in").str().c_str());
      Value *New = ConvertScalar_InsertValue(SI->getOperand(0), Old, Offset,
                                             Builder);
      Builder.CreateStore(New, NewAI);
      SI->eraseFromParent();
      continue;
    }
    
    // If this is a constant sized memset of a constant value (e.g. 0) we can
    // transform it into a store of the expanded constant value.
    if (MemSetInst *MSI = dyn_cast<MemSetInst>(User)) {
      assert(MSI->getRawDest() == Ptr && "Consistency error!");
      unsigned NumBytes = cast<ConstantInt>(MSI->getLength())->getZExtValue();
      if (NumBytes != 0) {
        unsigned Val = cast<ConstantInt>(MSI->getValue())->getZExtValue();
        
        // Compute the value replicated the right number of times.
        APInt APVal(NumBytes*8, Val);

        // Splat the value if non-zero.
        if (Val)
          for (unsigned i = 1; i != NumBytes; ++i)
            APVal |= APVal << 8;
        
        // FIXME: Remove once builder has Twine API.
        Value *Old = Builder.CreateLoad(NewAI, (NewAI->getName()+".in").str().c_str());
        Value *New = ConvertScalar_InsertValue(
                                    ConstantInt::get(User->getContext(), APVal),
                                               Old, Offset, Builder);
        Builder.CreateStore(New, NewAI);
      }
      MSI->eraseFromParent();
      continue;
    }

    // If this is a memcpy or memmove into or out of the whole allocation, we
    // can handle it like a load or store of the scalar type.
    if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(User)) {
      assert(Offset == 0 && "must be store to start of alloca");
      
      // If the source and destination are both to the same alloca, then this is
      // a noop copy-to-self, just delete it.  Otherwise, emit a load and store
      // as appropriate.
      AllocaInst *OrigAI = cast<AllocaInst>(Ptr->getUnderlyingObject());
      
      if (MTI->getSource()->getUnderlyingObject() != OrigAI) {
        // Dest must be OrigAI, change this to be a load from the original
        // pointer (bitcasted), then a store to our new alloca.
        assert(MTI->getRawDest() == Ptr && "Neither use is of pointer?");
        Value *SrcPtr = MTI->getSource();
        SrcPtr = Builder.CreateBitCast(SrcPtr, NewAI->getType());
        
        LoadInst *SrcVal = Builder.CreateLoad(SrcPtr, "srcval");
        SrcVal->setAlignment(MTI->getAlignment());
        Builder.CreateStore(SrcVal, NewAI);
      } else if (MTI->getDest()->getUnderlyingObject() != OrigAI) {
        // Src must be OrigAI, change this to be a load from NewAI then a store
        // through the original dest pointer (bitcasted).
        assert(MTI->getRawSource() == Ptr && "Neither use is of pointer?");
        LoadInst *SrcVal = Builder.CreateLoad(NewAI, "srcval");

        Value *DstPtr = Builder.CreateBitCast(MTI->getDest(), NewAI->getType());
        StoreInst *NewStore = Builder.CreateStore(SrcVal, DstPtr);
        NewStore->setAlignment(MTI->getAlignment());
      } else {
        // Noop transfer. Src == Dst
      }
          

      MTI->eraseFromParent();
      continue;
    }
    
    // If user is a dbg info intrinsic then it is safe to remove it.
    if (isa<DbgInfoIntrinsic>(User)) {
      User->eraseFromParent();
      continue;
    }

    llvm_unreachable("Unsupported operation!");
  }
}

/// ConvertScalar_ExtractValue - Extract a value of type ToType from an integer
/// or vector value FromVal, extracting the bits from the offset specified by
/// Offset.  This returns the value, which is of type ToType.
///
/// This happens when we are converting an "integer union" to a single
/// integer scalar, or when we are converting a "vector union" to a vector with
/// insert/extractelement instructions.
///
/// Offset is an offset from the original alloca, in bits that need to be
/// shifted to the right.
Value *SROA::ConvertScalar_ExtractValue(Value *FromVal, const Type *ToType,
                                        uint64_t Offset, IRBuilder<> &Builder) {
  // If the load is of the whole new alloca, no conversion is needed.
  if (FromVal->getType() == ToType && Offset == 0)
    return FromVal;

  // If the result alloca is a vector type, this is either an element
  // access or a bitcast to another vector type of the same size.
  if (const VectorType *VTy = dyn_cast<VectorType>(FromVal->getType())) {
    if (isa<VectorType>(ToType))
      return Builder.CreateBitCast(FromVal, ToType, "tmp");

    // Otherwise it must be an element access.
    unsigned Elt = 0;
    if (Offset) {
      unsigned EltSize = TD->getTypeAllocSizeInBits(VTy->getElementType());
      Elt = Offset/EltSize;
      assert(EltSize*Elt == Offset && "Invalid modulus in validity checking");
    }
    // Return the element extracted out of it.
    Value *V = Builder.CreateExtractElement(FromVal, ConstantInt::get(
                    Type::getInt32Ty(FromVal->getContext()), Elt), "tmp");
    if (V->getType() != ToType)
      V = Builder.CreateBitCast(V, ToType, "tmp");
    return V;
  }
  
  // If ToType is a first class aggregate, extract out each of the pieces and
  // use insertvalue's to form the FCA.
  if (const StructType *ST = dyn_cast<StructType>(ToType)) {
    const StructLayout &Layout = *TD->getStructLayout(ST);
    Value *Res = UndefValue::get(ST);
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
      Value *Elt = ConvertScalar_ExtractValue(FromVal, ST->getElementType(i),
                                        Offset+Layout.getElementOffsetInBits(i),
                                              Builder);
      Res = Builder.CreateInsertValue(Res, Elt, i, "tmp");
    }
    return Res;
  }
  
  if (const ArrayType *AT = dyn_cast<ArrayType>(ToType)) {
    uint64_t EltSize = TD->getTypeAllocSizeInBits(AT->getElementType());
    Value *Res = UndefValue::get(AT);
    for (unsigned i = 0, e = AT->getNumElements(); i != e; ++i) {
      Value *Elt = ConvertScalar_ExtractValue(FromVal, AT->getElementType(),
                                              Offset+i*EltSize, Builder);
      Res = Builder.CreateInsertValue(Res, Elt, i, "tmp");
    }
    return Res;
  }

  // Otherwise, this must be a union that was converted to an integer value.
  const IntegerType *NTy = cast<IntegerType>(FromVal->getType());

  // If this is a big-endian system and the load is narrower than the
  // full alloca type, we need to do a shift to get the right bits.
  int ShAmt = 0;
  if (TD->isBigEndian()) {
    // On big-endian machines, the lowest bit is stored at the bit offset
    // from the pointer given by getTypeStoreSizeInBits.  This matters for
    // integers with a bitwidth that is not a multiple of 8.
    ShAmt = TD->getTypeStoreSizeInBits(NTy) -
            TD->getTypeStoreSizeInBits(ToType) - Offset;
  } else {
    ShAmt = Offset;
  }

  // Note: we support negative bitwidths (with shl) which are not defined.
  // We do this to support (f.e.) loads off the end of a structure where
  // only some bits are used.
  if (ShAmt > 0 && (unsigned)ShAmt < NTy->getBitWidth())
    FromVal = Builder.CreateLShr(FromVal,
                                 ConstantInt::get(FromVal->getType(),
                                                           ShAmt), "tmp");
  else if (ShAmt < 0 && (unsigned)-ShAmt < NTy->getBitWidth())
    FromVal = Builder.CreateShl(FromVal, 
                                ConstantInt::get(FromVal->getType(),
                                                          -ShAmt), "tmp");

  // Finally, unconditionally truncate the integer to the right width.
  unsigned LIBitWidth = TD->getTypeSizeInBits(ToType);
  if (LIBitWidth < NTy->getBitWidth())
    FromVal =
      Builder.CreateTrunc(FromVal, IntegerType::get(FromVal->getContext(), 
                                                    LIBitWidth), "tmp");
  else if (LIBitWidth > NTy->getBitWidth())
    FromVal =
       Builder.CreateZExt(FromVal, IntegerType::get(FromVal->getContext(), 
                                                    LIBitWidth), "tmp");

  // If the result is an integer, this is a trunc or bitcast.
  if (isa<IntegerType>(ToType)) {
    // Should be done.
  } else if (ToType->isFloatingPoint() || isa<VectorType>(ToType)) {
    // Just do a bitcast, we know the sizes match up.
    FromVal = Builder.CreateBitCast(FromVal, ToType, "tmp");
  } else {
    // Otherwise must be a pointer.
    FromVal = Builder.CreateIntToPtr(FromVal, ToType, "tmp");
  }
  assert(FromVal->getType() == ToType && "Didn't convert right?");
  return FromVal;
}


/// ConvertScalar_InsertValue - Insert the value "SV" into the existing integer
/// or vector value "Old" at the offset specified by Offset.
///
/// This happens when we are converting an "integer union" to a
/// single integer scalar, or when we are converting a "vector union" to a
/// vector with insert/extractelement instructions.
///
/// Offset is an offset from the original alloca, in bits that need to be
/// shifted to the right.
Value *SROA::ConvertScalar_InsertValue(Value *SV, Value *Old,
                                       uint64_t Offset, IRBuilder<> &Builder) {

  // Convert the stored type to the actual type, shift it left to insert
  // then 'or' into place.
  const Type *AllocaType = Old->getType();
  LLVMContext &Context = Old->getContext();

  if (const VectorType *VTy = dyn_cast<VectorType>(AllocaType)) {
    uint64_t VecSize = TD->getTypeAllocSizeInBits(VTy);
    uint64_t ValSize = TD->getTypeAllocSizeInBits(SV->getType());
    
    // Changing the whole vector with memset or with an access of a different
    // vector type?
    if (ValSize == VecSize)
      return Builder.CreateBitCast(SV, AllocaType, "tmp");

    uint64_t EltSize = TD->getTypeAllocSizeInBits(VTy->getElementType());

    // Must be an element insertion.
    unsigned Elt = Offset/EltSize;
    
    if (SV->getType() != VTy->getElementType())
      SV = Builder.CreateBitCast(SV, VTy->getElementType(), "tmp");
    
    SV = Builder.CreateInsertElement(Old, SV, 
                     ConstantInt::get(Type::getInt32Ty(SV->getContext()), Elt),
                                     "tmp");
    return SV;
  }
  
  // If SV is a first-class aggregate value, insert each value recursively.
  if (const StructType *ST = dyn_cast<StructType>(SV->getType())) {
    const StructLayout &Layout = *TD->getStructLayout(ST);
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
      Value *Elt = Builder.CreateExtractValue(SV, i, "tmp");
      Old = ConvertScalar_InsertValue(Elt, Old, 
                                      Offset+Layout.getElementOffsetInBits(i),
                                      Builder);
    }
    return Old;
  }
  
  if (const ArrayType *AT = dyn_cast<ArrayType>(SV->getType())) {
    uint64_t EltSize = TD->getTypeAllocSizeInBits(AT->getElementType());
    for (unsigned i = 0, e = AT->getNumElements(); i != e; ++i) {
      Value *Elt = Builder.CreateExtractValue(SV, i, "tmp");
      Old = ConvertScalar_InsertValue(Elt, Old, Offset+i*EltSize, Builder);
    }
    return Old;
  }

  // If SV is a float, convert it to the appropriate integer type.
  // If it is a pointer, do the same.
  unsigned SrcWidth = TD->getTypeSizeInBits(SV->getType());
  unsigned DestWidth = TD->getTypeSizeInBits(AllocaType);
  unsigned SrcStoreWidth = TD->getTypeStoreSizeInBits(SV->getType());
  unsigned DestStoreWidth = TD->getTypeStoreSizeInBits(AllocaType);
  if (SV->getType()->isFloatingPoint() || isa<VectorType>(SV->getType()))
    SV = Builder.CreateBitCast(SV,
                            IntegerType::get(SV->getContext(),SrcWidth), "tmp");
  else if (isa<PointerType>(SV->getType()))
    SV = Builder.CreatePtrToInt(SV, TD->getIntPtrType(SV->getContext()), "tmp");

  // Zero extend or truncate the value if needed.
  if (SV->getType() != AllocaType) {
    if (SV->getType()->getPrimitiveSizeInBits() <
             AllocaType->getPrimitiveSizeInBits())
      SV = Builder.CreateZExt(SV, AllocaType, "tmp");
    else {
      // Truncation may be needed if storing more than the alloca can hold
      // (undefined behavior).
      SV = Builder.CreateTrunc(SV, AllocaType, "tmp");
      SrcWidth = DestWidth;
      SrcStoreWidth = DestStoreWidth;
    }
  }

  // If this is a big-endian system and the store is narrower than the
  // full alloca type, we need to do a shift to get the right bits.
  int ShAmt = 0;
  if (TD->isBigEndian()) {
    // On big-endian machines, the lowest bit is stored at the bit offset
    // from the pointer given by getTypeStoreSizeInBits.  This matters for
    // integers with a bitwidth that is not a multiple of 8.
    ShAmt = DestStoreWidth - SrcStoreWidth - Offset;
  } else {
    ShAmt = Offset;
  }

  // Note: we support negative bitwidths (with shr) which are not defined.
  // We do this to support (f.e.) stores off the end of a structure where
  // only some bits in the structure are set.
  APInt Mask(APInt::getLowBitsSet(DestWidth, SrcWidth));
  if (ShAmt > 0 && (unsigned)ShAmt < DestWidth) {
    SV = Builder.CreateShl(SV, ConstantInt::get(SV->getType(),
                           ShAmt), "tmp");
    Mask <<= ShAmt;
  } else if (ShAmt < 0 && (unsigned)-ShAmt < DestWidth) {
    SV = Builder.CreateLShr(SV, ConstantInt::get(SV->getType(),
                            -ShAmt), "tmp");
    Mask = Mask.lshr(-ShAmt);
  }

  // Mask out the bits we are about to insert from the old value, and or
  // in the new bits.
  if (SrcWidth != DestWidth) {
    assert(DestWidth > SrcWidth);
    Old = Builder.CreateAnd(Old, ConstantInt::get(Context, ~Mask), "mask");
    SV = Builder.CreateOr(Old, SV, "ins");
  }
  return SV;
}



/// PointsToConstantGlobal - Return true if V (possibly indirectly) points to
/// some part of a constant global variable.  This intentionally only accepts
/// constant expressions because we don't can't rewrite arbitrary instructions.
static bool PointsToConstantGlobal(Value *V) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return GV->isConstant();
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::BitCast || 
        CE->getOpcode() == Instruction::GetElementPtr)
      return PointsToConstantGlobal(CE->getOperand(0));
  return false;
}

/// isOnlyCopiedFromConstantGlobal - Recursively walk the uses of a (derived)
/// pointer to an alloca.  Ignore any reads of the pointer, return false if we
/// see any stores or other unknown uses.  If we see pointer arithmetic, keep
/// track of whether it moves the pointer (with isOffset) but otherwise traverse
/// the uses.  If we see a memcpy/memmove that targets an unoffseted pointer to
/// the alloca, and if the source pointer is a pointer to a constant  global, we
/// can optimize this.
static bool isOnlyCopiedFromConstantGlobal(Value *V, Instruction *&TheCopy,
                                           bool isOffset) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI!=E; ++UI) {
    if (LoadInst *LI = dyn_cast<LoadInst>(*UI))
      // Ignore non-volatile loads, they are always ok.
      if (!LI->isVolatile())
        continue;
    
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(*UI)) {
      // If uses of the bitcast are ok, we are ok.
      if (!isOnlyCopiedFromConstantGlobal(BCI, TheCopy, isOffset))
        return false;
      continue;
    }
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(*UI)) {
      // If the GEP has all zero indices, it doesn't offset the pointer.  If it
      // doesn't, it does.
      if (!isOnlyCopiedFromConstantGlobal(GEP, TheCopy,
                                         isOffset || !GEP->hasAllZeroIndices()))
        return false;
      continue;
    }
    
    // If this is isn't our memcpy/memmove, reject it as something we can't
    // handle.
    if (!isa<MemTransferInst>(*UI))
      return false;

    // If we already have seen a copy, reject the second one.
    if (TheCopy) return false;
    
    // If the pointer has been offset from the start of the alloca, we can't
    // safely handle this.
    if (isOffset) return false;

    // If the memintrinsic isn't using the alloca as the dest, reject it.
    if (UI.getOperandNo() != 1) return false;
    
    MemIntrinsic *MI = cast<MemIntrinsic>(*UI);
    
    // If the source of the memcpy/move is not a constant global, reject it.
    if (!PointsToConstantGlobal(MI->getOperand(2)))
      return false;
    
    // Otherwise, the transform is safe.  Remember the copy instruction.
    TheCopy = MI;
  }
  return true;
}

/// isOnlyCopiedFromConstantGlobal - Return true if the specified alloca is only
/// modified by a copy from a constant global.  If we can prove this, we can
/// replace any uses of the alloca with uses of the global directly.
Instruction *SROA::isOnlyCopiedFromConstantGlobal(AllocationInst *AI) {
  Instruction *TheCopy = 0;
  if (::isOnlyCopiedFromConstantGlobal(AI, TheCopy, false))
    return TheCopy;
  return 0;
}
