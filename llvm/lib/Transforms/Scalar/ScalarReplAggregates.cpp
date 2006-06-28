//===- ScalarReplAggregates.cpp - Scalar Replacement of Aggregates --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Visibility.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include <iostream>
using namespace llvm;

namespace {
  Statistic<> NumReplaced("scalarrepl", "Number of allocas broken up");
  Statistic<> NumPromoted("scalarrepl", "Number of allocas promoted");
  Statistic<> NumConverted("scalarrepl",
                           "Number of aggregates converted to scalar");

  struct VISIBILITY_HIDDEN SROA : public FunctionPass {
    bool runOnFunction(Function &F);

    bool performScalarRepl(Function &F);
    bool performPromotion(Function &F);

    // getAnalysisUsage - This pass does not require any passes, but we know it
    // will not alter the CFG, so say so.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.addRequired<DominanceFrontier>();
      AU.addRequired<TargetData>();
      AU.setPreservesCFG();
    }

  private:
    int isSafeElementUse(Value *Ptr);
    int isSafeUseOfAllocation(Instruction *User);
    int isSafeAllocaToScalarRepl(AllocationInst *AI);
    void CanonicalizeAllocaUsers(AllocationInst *AI);
    AllocaInst *AddNewAlloca(Function &F, const Type *Ty, AllocationInst *Base);
    
    const Type *CanConvertToScalar(Value *V, bool &IsNotTrivial);
    void ConvertToScalar(AllocationInst *AI, const Type *Ty);
    void ConvertUsesToScalar(Value *Ptr, AllocaInst *NewAI, unsigned Offset);
  };

  RegisterOpt<SROA> X("scalarrepl", "Scalar Replacement of Aggregates");
}

// Public interface to the ScalarReplAggregates pass
FunctionPass *llvm::createScalarReplAggregatesPass() { return new SROA(); }


bool SROA::runOnFunction(Function &F) {
  bool Changed = performPromotion(F);
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
  const TargetData &TD = getAnalysis<TargetData>();
  DominatorTree     &DT = getAnalysis<DominatorTree>();
  DominanceFrontier &DF = getAnalysis<DominanceFrontier>();

  BasicBlock &BB = F.getEntryBlock();  // Get the entry node for the function

  bool Changed = false;

  while (1) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I))       // Is it an alloca?
        if (isAllocaPromotable(AI, TD))
          Allocas.push_back(AI);

    if (Allocas.empty()) break;

    PromoteMemToReg(Allocas, DT, DF, TD);
    NumPromoted += Allocas.size();
    Changed = true;
  }

  return Changed;
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
    
    // If we can turn this aggregate value (potentially with casts) into a
    // simple scalar value that can be mem2reg'd into a register value.
    bool IsNotTrivial = false;
    if (const Type *ActualType = CanConvertToScalar(AI, IsNotTrivial))
      if (IsNotTrivial && ActualType != Type::VoidTy) {
        ConvertToScalar(AI, ActualType);
        Changed = true;
        continue;
      }

    // We cannot transform the allocation instruction if it is an array
    // allocation (allocations OF arrays are ok though), and an allocation of a
    // scalar value cannot be decomposed at all.
    //
    if (AI->isArrayAllocation() ||
        (!isa<StructType>(AI->getAllocatedType()) &&
         !isa<ArrayType>(AI->getAllocatedType()))) continue;

    // Check that all of the users of the allocation are capable of being
    // transformed.
    switch (isSafeAllocaToScalarRepl(AI)) {
    default: assert(0 && "Unexpected value!");
    case 0:  // Not safe to scalar replace.
      continue;
    case 1:  // Safe, but requires cleanup/canonicalizations first
      CanonicalizeAllocaUsers(AI);
    case 3:  // Safe to scalar replace.
      break;
    }

    DEBUG(std::cerr << "Found inst to xform: " << *AI);
    Changed = true;

    std::vector<AllocaInst*> ElementAllocas;
    if (const StructType *ST = dyn_cast<StructType>(AI->getAllocatedType())) {
      ElementAllocas.reserve(ST->getNumContainedTypes());
      for (unsigned i = 0, e = ST->getNumContainedTypes(); i != e; ++i) {
        AllocaInst *NA = new AllocaInst(ST->getContainedType(i), 0, 
                                        AI->getAlignment(),
                                        AI->getName() + "." + utostr(i), AI);
        ElementAllocas.push_back(NA);
        WorkList.push_back(NA);  // Add to worklist for recursive processing
      }
    } else {
      const ArrayType *AT = cast<ArrayType>(AI->getAllocatedType());
      ElementAllocas.reserve(AT->getNumElements());
      const Type *ElTy = AT->getElementType();
      for (unsigned i = 0, e = AT->getNumElements(); i != e; ++i) {
        AllocaInst *NA = new AllocaInst(ElTy, 0, AI->getAlignment(),
                                        AI->getName() + "." + utostr(i), AI);
        ElementAllocas.push_back(NA);
        WorkList.push_back(NA);  // Add to worklist for recursive processing
      }
    }

    // Now that we have created the alloca instructions that we want to use,
    // expand the getelementptr instructions to use them.
    //
    while (!AI->use_empty()) {
      Instruction *User = cast<Instruction>(AI->use_back());
      GetElementPtrInst *GEPI = cast<GetElementPtrInst>(User);
      // We now know that the GEP is of the form: GEP <ptr>, 0, <cst>
      unsigned Idx =
         (unsigned)cast<ConstantInt>(GEPI->getOperand(2))->getRawValue();

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
        std::string OldName = GEPI->getName();  // Steal the old name.
        std::vector<Value*> NewArgs;
        NewArgs.push_back(Constant::getNullValue(Type::IntTy));
        NewArgs.insert(NewArgs.end(), GEPI->op_begin()+3, GEPI->op_end());
        GEPI->setName("");
        RepValue = new GetElementPtrInst(AllocaToUse, NewArgs, OldName, GEPI);
      }

      // Move all of the users over to the new GEP.
      GEPI->replaceAllUsesWith(RepValue);
      // Delete the old GEP
      GEPI->eraseFromParent();
    }

    // Finally, delete the Alloca instruction
    AI->getParent()->getInstList().erase(AI);
    NumReplaced++;
  }

  return Changed;
}


/// isSafeElementUse - Check to see if this use is an allowed use for a
/// getelementptr instruction of an array aggregate allocation.
///
int SROA::isSafeElementUse(Value *Ptr) {
  for (Value::use_iterator I = Ptr->use_begin(), E = Ptr->use_end();
       I != E; ++I) {
    Instruction *User = cast<Instruction>(*I);
    switch (User->getOpcode()) {
    case Instruction::Load:  break;
    case Instruction::Store:
      // Store is ok if storing INTO the pointer, not storing the pointer
      if (User->getOperand(0) == Ptr) return 0;
      break;
    case Instruction::GetElementPtr: {
      GetElementPtrInst *GEP = cast<GetElementPtrInst>(User);
      if (GEP->getNumOperands() > 1) {
        if (!isa<Constant>(GEP->getOperand(1)) ||
            !cast<Constant>(GEP->getOperand(1))->isNullValue())
          return 0;  // Using pointer arithmetic to navigate the array...
      }
      if (!isSafeElementUse(GEP)) return 0;
      break;
    }
    default:
      DEBUG(std::cerr << "  Transformation preventing inst: " << *User);
      return 0;
    }
  }
  return 3;  // All users look ok :)
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
int SROA::isSafeUseOfAllocation(Instruction *User) {
  if (!isa<GetElementPtrInst>(User)) return 0;

  GetElementPtrInst *GEPI = cast<GetElementPtrInst>(User);
  gep_type_iterator I = gep_type_begin(GEPI), E = gep_type_end(GEPI);

  // The GEP is not safe to transform if not of the form "GEP <ptr>, 0, <cst>".
  if (I == E ||
      I.getOperand() != Constant::getNullValue(I.getOperand()->getType()))
    return 0;

  ++I;
  if (I == E) return 0;  // ran out of GEP indices??

  // If this is a use of an array allocation, do a bit more checking for sanity.
  if (const ArrayType *AT = dyn_cast<ArrayType>(*I)) {
    uint64_t NumElements = AT->getNumElements();

    if (ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand())) {
      // Check to make sure that index falls within the array.  If not,
      // something funny is going on, so we won't do the optimization.
      //
      if (cast<ConstantInt>(GEPI->getOperand(2))->getRawValue() >= NumElements)
        return 0;

      // We cannot scalar repl this level of the array unless any array
      // sub-indices are in-range constants.  In particular, consider:
      // A[0][i].  We cannot know that the user isn't doing invalid things like
      // allowing i to index an out-of-range subscript that accesses A[1].
      //
      // Scalar replacing *just* the outer index of the array is probably not
      // going to be a win anyway, so just give up.
      for (++I; I != E && isa<ArrayType>(*I); ++I) {
        const ArrayType *SubArrayTy = cast<ArrayType>(*I);
        uint64_t NumElements = SubArrayTy->getNumElements();
        if (!isa<ConstantInt>(I.getOperand())) return 0;
        if (cast<ConstantInt>(I.getOperand())->getRawValue() >= NumElements)
          return 0;
      }
      
    } else {
      // If this is an array index and the index is not constant, we cannot
      // promote... that is unless the array has exactly one or two elements in
      // it, in which case we CAN promote it, but we have to canonicalize this
      // out if this is the only problem.
      if ((NumElements == 1 || NumElements == 2) &&
          AllUsersAreLoads(GEPI))
        return 1;  // Canonicalization required!
      return 0;
    }
  }

  // If there are any non-simple uses of this getelementptr, make sure to reject
  // them.
  return isSafeElementUse(GEPI);
}

/// isSafeStructAllocaToScalarRepl - Check to see if the specified allocation of
/// an aggregate can be broken down into elements.  Return 0 if not, 3 if safe,
/// or 1 if safe after canonicalization has been performed.
///
int SROA::isSafeAllocaToScalarRepl(AllocationInst *AI) {
  // Loop over the use list of the alloca.  We can only transform it if all of
  // the users are safe to transform.
  //
  int isSafe = 3;
  for (Value::use_iterator I = AI->use_begin(), E = AI->use_end();
       I != E; ++I) {
    isSafe &= isSafeUseOfAllocation(cast<Instruction>(*I));
    if (isSafe == 0) {
      DEBUG(std::cerr << "Cannot transform: " << *AI << "  due to user: "
            << **I);
      return 0;
    }
  }
  // If we require cleanup, isSafe is now 1, otherwise it is 3.
  return isSafe;
}

/// CanonicalizeAllocaUsers - If SROA reported that it can promote the specified
/// allocation, but only if cleaned up, perform the cleanups required.
void SROA::CanonicalizeAllocaUsers(AllocationInst *AI) {
  // At this point, we know that the end result will be SROA'd and promoted, so
  // we can insert ugly code if required so long as sroa+mem2reg will clean it
  // up.
  for (Value::use_iterator UI = AI->use_begin(), E = AI->use_end();
       UI != E; ) {
    GetElementPtrInst *GEPI = cast<GetElementPtrInst>(*UI++);
    gep_type_iterator I = gep_type_begin(GEPI);
    ++I;

    if (const ArrayType *AT = dyn_cast<ArrayType>(*I)) {
      uint64_t NumElements = AT->getNumElements();

      if (!isa<ConstantInt>(I.getOperand())) {
        if (NumElements == 1) {
          GEPI->setOperand(2, Constant::getNullValue(Type::IntTy));
        } else {
          assert(NumElements == 2 && "Unhandled case!");
          // All users of the GEP must be loads.  At each use of the GEP, insert
          // two loads of the appropriate indexed GEP and select between them.
          Value *IsOne = BinaryOperator::createSetNE(I.getOperand(),
                              Constant::getNullValue(I.getOperand()->getType()),
                                                     "isone", GEPI);
          // Insert the new GEP instructions, which are properly indexed.
          std::vector<Value*> Indices(GEPI->op_begin()+1, GEPI->op_end());
          Indices[1] = Constant::getNullValue(Type::IntTy);
          Value *ZeroIdx = new GetElementPtrInst(GEPI->getOperand(0), Indices,
                                                 GEPI->getName()+".0", GEPI);
          Indices[1] = ConstantInt::get(Type::IntTy, 1);
          Value *OneIdx = new GetElementPtrInst(GEPI->getOperand(0), Indices,
                                                GEPI->getName()+".1", GEPI);
          // Replace all loads of the variable index GEP with loads from both
          // indexes and a select.
          while (!GEPI->use_empty()) {
            LoadInst *LI = cast<LoadInst>(GEPI->use_back());
            Value *Zero = new LoadInst(ZeroIdx, LI->getName()+".0", LI);
            Value *One  = new LoadInst(OneIdx , LI->getName()+".1", LI);
            Value *R = new SelectInst(IsOne, One, Zero, LI->getName(), LI);
            LI->replaceAllUsesWith(R);
            LI->eraseFromParent();
          }
          GEPI->eraseFromParent();
        }
      }
    }
  }
}

/// MergeInType - Add the 'In' type to the accumulated type so far.  If the
/// types are incompatible, return true, otherwise update Accum and return
/// false.
///
/// There are two cases we handle here:
///   1) An effectively integer union, where the pieces are stored into as
///      smaller integers (common with byte swap and other idioms).
///   2) A union of a vector and its elements.  Here we turn element accesses
///      into insert/extract element operations.
static bool MergeInType(const Type *In, const Type *&Accum) {
  // If this is our first type, just use it.
  const PackedType *PTy;
  if (Accum == Type::VoidTy || In == Accum) {
    Accum = In;
  } else if (In->isIntegral() && Accum->isIntegral()) {   // integer union.
    // Otherwise pick whichever type is larger.
    if (In->getTypeID() > Accum->getTypeID())
      Accum = In;
  } else if ((PTy = dyn_cast<PackedType>(Accum)) && 
             PTy->getElementType() == In) {
    // Accum is a vector, and we are accessing an element: ok.
  } else if ((PTy = dyn_cast<PackedType>(In)) && 
             PTy->getElementType() == Accum) {
    // In is a vector, and accum is an element: ok, remember In.
    Accum = In;
  } else {
    return true;
  }
  return false;
}

/// getUIntAtLeastAsBitAs - Return an unsigned integer type that is at least
/// as big as the specified type.  If there is no suitable type, this returns
/// null.
const Type *getUIntAtLeastAsBitAs(unsigned NumBits) {
  if (NumBits > 64) return 0;
  if (NumBits > 32) return Type::ULongTy;
  if (NumBits > 16) return Type::UIntTy;
  if (NumBits > 8) return Type::UShortTy;
  return Type::UByteTy;    
}

/// CanConvertToScalar - V is a pointer.  If we can convert the pointee to a
/// single scalar integer type, return that type.  Further, if the use is not
/// a completely trivial use that mem2reg could promote, set IsNotTrivial.  If
/// there are no uses of this pointer, return Type::VoidTy to differentiate from
/// failure.
///
const Type *SROA::CanConvertToScalar(Value *V, bool &IsNotTrivial) {
  const Type *UsedType = Type::VoidTy; // No uses, no forced type.
  const TargetData &TD = getAnalysis<TargetData>();
  const PointerType *PTy = cast<PointerType>(V->getType());

  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI!=E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    
    if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
      if (MergeInType(LI->getType(), UsedType))
        return 0;
      
    } else if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
      // Storing the pointer, not the into the value?
      if (SI->getOperand(0) == V) return 0;
      
      // NOTE: We could handle storing of FP imms into integers here!
      
      if (MergeInType(SI->getOperand(0)->getType(), UsedType))
        return 0;
    } else if (CastInst *CI = dyn_cast<CastInst>(User)) {
      if (!isa<PointerType>(CI->getType())) return 0;
      IsNotTrivial = true;
      const Type *SubTy = CanConvertToScalar(CI, IsNotTrivial);
      if (!SubTy || MergeInType(SubTy, UsedType)) return 0;
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User)) {
      // Check to see if this is stepping over an element: GEP Ptr, int C
      if (GEP->getNumOperands() == 2 && isa<ConstantInt>(GEP->getOperand(1))) {
        unsigned Idx = cast<ConstantInt>(GEP->getOperand(1))->getRawValue();
        unsigned ElSize = TD.getTypeSize(PTy->getElementType());
        unsigned BitOffset = Idx*ElSize*8;
        if (BitOffset > 64 || !isPowerOf2_32(ElSize)) return 0;
        
        IsNotTrivial = true;
        const Type *SubElt = CanConvertToScalar(GEP, IsNotTrivial);
        if (SubElt == 0) return 0;
        if (SubElt != Type::VoidTy && SubElt->isInteger()) {
          const Type *NewTy = 
            getUIntAtLeastAsBitAs(SubElt->getPrimitiveSizeInBits()+BitOffset);
          if (NewTy == 0 || MergeInType(NewTy, UsedType)) return 0;
          continue;
        }
      } else if (GEP->getNumOperands() == 3 && 
                 isa<ConstantInt>(GEP->getOperand(1)) &&
                 isa<ConstantInt>(GEP->getOperand(2)) &&
                 cast<Constant>(GEP->getOperand(1))->isNullValue()) {
        // We are stepping into an element, e.g. a structure or an array:
        // GEP Ptr, int 0, uint C
        const Type *AggTy = PTy->getElementType();
        unsigned Idx = cast<ConstantInt>(GEP->getOperand(2))->getRawValue();
        
        if (const ArrayType *ATy = dyn_cast<ArrayType>(AggTy)) {
          if (Idx >= ATy->getNumElements()) return 0;  // Out of range.
        } else if (const PackedType *PackedTy = dyn_cast<PackedType>(AggTy)) {
          // Getting an element of the packed vector.
          if (Idx >= PackedTy->getNumElements()) return 0;  // Out of range.

          // Merge in the packed type.
          if (MergeInType(PackedTy, UsedType)) return 0;
          
          const Type *SubTy = CanConvertToScalar(GEP, IsNotTrivial);
          if (SubTy == 0) return 0;
          
          if (SubTy != Type::VoidTy && MergeInType(SubTy, UsedType))
            return 0;

          // We'll need to change this to an insert/extract element operation.
          IsNotTrivial = true;
          continue;    // Everything looks ok
          
        } else if (isa<StructType>(AggTy)) {
          // Structs are always ok.
        } else {
          return 0;
        }
        const Type *NTy = getUIntAtLeastAsBitAs(TD.getTypeSize(AggTy)*8);
        if (NTy == 0 || MergeInType(NTy, UsedType)) return 0;
        const Type *SubTy = CanConvertToScalar(GEP, IsNotTrivial);
        if (SubTy == 0) return 0;
        if (SubTy != Type::VoidTy && MergeInType(SubTy, UsedType))
          return 0;
        continue;    // Everything looks ok
      }
      return 0;
    } else {
      // Cannot handle this!
      return 0;
    }
  }
  
  return UsedType;
}

/// ConvertToScalar - The specified alloca passes the CanConvertToScalar
/// predicate and is non-trivial.  Convert it to something that can be trivially
/// promoted into a register by mem2reg.
void SROA::ConvertToScalar(AllocationInst *AI, const Type *ActualTy) {
  DEBUG(std::cerr << "CONVERT TO SCALAR: " << *AI << "  TYPE = "
                  << *ActualTy << "\n");
  ++NumConverted;
  
  BasicBlock *EntryBlock = AI->getParent();
  assert(EntryBlock == &EntryBlock->getParent()->front() &&
         "Not in the entry block!");
  EntryBlock->getInstList().remove(AI);  // Take the alloca out of the program.
  
  if (ActualTy->isInteger())
    ActualTy = ActualTy->getUnsignedVersion();
  
  // Create and insert the alloca.
  AllocaInst *NewAI = new AllocaInst(ActualTy, 0, AI->getName(),
                                     EntryBlock->begin());
  ConvertUsesToScalar(AI, NewAI, 0);
  delete AI;
}


/// ConvertUsesToScalar - Convert all of the users of Ptr to use the new alloca
/// directly.  This happens when we are converting an "integer union" to a
/// single integer scalar, or when we are converting a "vector union" to a
/// vector with insert/extractelement instructions.
///
/// Offset is an offset from the original alloca, in bits that need to be
/// shifted to the right.  By the end of this, there should be no uses of Ptr.
void SROA::ConvertUsesToScalar(Value *Ptr, AllocaInst *NewAI, unsigned Offset) {
  bool isVectorInsert = isa<PackedType>(NewAI->getType()->getElementType());
  while (!Ptr->use_empty()) {
    Instruction *User = cast<Instruction>(Ptr->use_back());
    
    if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
      // The load is a bit extract from NewAI shifted right by Offset bits.
      Value *NV = new LoadInst(NewAI, LI->getName(), LI);
      if (NV->getType() != LI->getType()) {
        if (const PackedType *PTy = dyn_cast<PackedType>(NV->getType())) {
          // Must be an element access.
          unsigned Elt = Offset/PTy->getElementType()->getPrimitiveSizeInBits();
          NV = new ExtractElementInst(NV, ConstantUInt::get(Type::UIntTy, Elt),
                                      "tmp", LI);
        } else {
          assert(NV->getType()->isInteger() && "Unknown promotion!");
          if (Offset && Offset < NV->getType()->getPrimitiveSizeInBits())
            NV = new ShiftInst(Instruction::Shr, NV,
                               ConstantUInt::get(Type::UByteTy, Offset),
                               LI->getName(), LI);
          NV = new CastInst(NV, LI->getType(), LI->getName(), LI);
        }
      }
      LI->replaceAllUsesWith(NV);
      LI->eraseFromParent();
    } else if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
      assert(SI->getOperand(0) != Ptr && "Consistency error!");

      // Convert the stored type to the actual type, shift it left to insert
      // then 'or' into place.
      Value *SV = SI->getOperand(0);
      const Type *AllocaType = NewAI->getType()->getElementType();
      if (SV->getType() != AllocaType) {
        Value *Old = new LoadInst(NewAI, NewAI->getName()+".in", SI);
        
        if (const PackedType *PTy = dyn_cast<PackedType>(AllocaType)) {
          // Must be an element insertion.
          unsigned Elt = Offset/PTy->getElementType()->getPrimitiveSizeInBits();
          SV = new InsertElementInst(Old, SV,
                                     ConstantUInt::get(Type::UIntTy, Elt),
                                     "tmp", SI);
        } else {
          // If SV is signed, convert it to unsigned, so that the next cast zero
          // extends the value.
          if (SV->getType()->isSigned())
            SV = new CastInst(SV, SV->getType()->getUnsignedVersion(),
                              SV->getName(), SI);
          SV = new CastInst(SV, Old->getType(), SV->getName(), SI);
          if (Offset && Offset < SV->getType()->getPrimitiveSizeInBits())
            SV = new ShiftInst(Instruction::Shl, SV,
                               ConstantUInt::get(Type::UByteTy, Offset),
                               SV->getName()+".adj", SI);
          // Mask out the bits we are about to insert from the old value.
          unsigned TotalBits = SV->getType()->getPrimitiveSizeInBits();
          unsigned InsertBits =
            SI->getOperand(0)->getType()->getPrimitiveSizeInBits();
          if (TotalBits != InsertBits) {
            assert(TotalBits > InsertBits);
            uint64_t Mask = ~(((1ULL << InsertBits)-1) << Offset);
            if (TotalBits != 64)
              Mask = Mask & ((1ULL << TotalBits)-1);
            Old = BinaryOperator::createAnd(Old,
                                        ConstantUInt::get(Old->getType(), Mask),
                                            Old->getName()+".mask", SI);
            SV = BinaryOperator::createOr(Old, SV, SV->getName()+".ins", SI);
          }
        }
      }
      new StoreInst(SV, NewAI, SI);
      SI->eraseFromParent();
      
    } else if (CastInst *CI = dyn_cast<CastInst>(User)) {
      unsigned NewOff = Offset;
      const TargetData &TD = getAnalysis<TargetData>();
      if (TD.isBigEndian() && !isVectorInsert) {
        // Adjust the pointer.  For example, storing 16-bits into a 32-bit
        // alloca with just a cast makes it modify the top 16-bits.
        const Type *SrcTy = cast<PointerType>(Ptr->getType())->getElementType();
        const Type *DstTy = cast<PointerType>(CI->getType())->getElementType();
        int PtrDiffBits = TD.getTypeSize(SrcTy)*8-TD.getTypeSize(DstTy)*8;
        NewOff += PtrDiffBits;
      }
      ConvertUsesToScalar(CI, NewAI, NewOff);
      CI->eraseFromParent();
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User)) {
      const PointerType *AggPtrTy = 
        cast<PointerType>(GEP->getOperand(0)->getType());
      const TargetData &TD = getAnalysis<TargetData>();
      unsigned AggSizeInBits = TD.getTypeSize(AggPtrTy->getElementType())*8;
      
      // Check to see if this is stepping over an element: GEP Ptr, int C
      unsigned NewOffset = Offset;
      if (GEP->getNumOperands() == 2) {
        unsigned Idx = cast<ConstantInt>(GEP->getOperand(1))->getRawValue();
        unsigned BitOffset = Idx*AggSizeInBits;
        
        if (TD.isLittleEndian() || isVectorInsert)
          NewOffset += BitOffset;
        else
          NewOffset -= BitOffset;
        
      } else if (GEP->getNumOperands() == 3) {
        // We know that operand #2 is zero.
        unsigned Idx = cast<ConstantInt>(GEP->getOperand(2))->getRawValue();
        const Type *AggTy = AggPtrTy->getElementType();
        if (const SequentialType *SeqTy = dyn_cast<SequentialType>(AggTy)) {
          unsigned ElSizeBits = TD.getTypeSize(SeqTy->getElementType())*8;

          if (TD.isLittleEndian() || isVectorInsert)
            NewOffset += ElSizeBits*Idx;
          else
            NewOffset += AggSizeInBits-ElSizeBits*(Idx+1);
        } else if (const StructType *STy = dyn_cast<StructType>(AggTy)) {
          unsigned EltBitOffset = TD.getStructLayout(STy)->MemberOffsets[Idx]*8;
          
          if (TD.isLittleEndian() || isVectorInsert)
            NewOffset += EltBitOffset;
          else {
            const PointerType *ElPtrTy = cast<PointerType>(GEP->getType());
            unsigned ElSizeBits = TD.getTypeSize(ElPtrTy->getElementType())*8;
            NewOffset += AggSizeInBits-(EltBitOffset+ElSizeBits);
          }
          
        } else {
          assert(0 && "Unsupported operation!");
          abort();
        }
      } else {
        assert(0 && "Unsupported operation!");
        abort();
      }
      ConvertUsesToScalar(GEP, NewAI, NewOffset);
      GEP->eraseFromParent();
    } else {
      assert(0 && "Unsupported operation!");
      abort();
    }
  }
}
