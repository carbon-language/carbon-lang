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
#include "llvm/iMemory.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/StringExtras.h"
using namespace llvm;

namespace {
  Statistic<> NumReplaced("scalarrepl", "Number of allocas broken up");
  Statistic<> NumPromoted("scalarrepl", "Number of allocas promoted");

  struct SROA : public FunctionPass {
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
    bool isSafeElementUse(Value *Ptr);
    bool isSafeUseOfAllocation(Instruction *User);
    bool isSafeAllocaToPromote(AllocationInst *AI);
    AllocaInst *AddNewAlloca(Function &F, const Type *Ty, AllocationInst *Base);
  };

  RegisterOpt<SROA> X("scalarrepl", "Scalar Replacement of Aggregates");
}

// Public interface to the ScalarReplAggregates pass
Pass *llvm::createScalarReplAggregatesPass() { return new SROA(); }


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

    // We cannot transform the allocation instruction if it is an array
    // allocation (allocations OF arrays are ok though), and an allocation of a
    // scalar value cannot be decomposed at all.
    //
    if (AI->isArrayAllocation() ||
        (!isa<StructType>(AI->getAllocatedType()) &&
         !isa<ArrayType>(AI->getAllocatedType()))) continue;

    // Check that all of the users of the allocation are capable of being
    // transformed.
    if (!isSafeAllocaToPromote(AI))
      continue;

    DEBUG(std::cerr << "Found inst to xform: " << *AI);
    Changed = true;
    
    std::vector<AllocaInst*> ElementAllocas;
    if (const StructType *ST = dyn_cast<StructType>(AI->getAllocatedType())) {
      ElementAllocas.reserve(ST->getNumContainedTypes());
      for (unsigned i = 0, e = ST->getNumContainedTypes(); i != e; ++i) {
        AllocaInst *NA = new AllocaInst(ST->getContainedType(i), 0,
                                        AI->getName() + "." + utostr(i), AI);
        ElementAllocas.push_back(NA);
        WorkList.push_back(NA);  // Add to worklist for recursive processing
      }
    } else {
      const ArrayType *AT = cast<ArrayType>(AI->getAllocatedType());
      ElementAllocas.reserve(AT->getNumElements());
      const Type *ElTy = AT->getElementType();
      for (unsigned i = 0, e = AT->getNumElements(); i != e; ++i) {
        AllocaInst *NA = new AllocaInst(ElTy, 0,
                                        AI->getName() + "." + utostr(i), AI);
        ElementAllocas.push_back(NA);
        WorkList.push_back(NA);  // Add to worklist for recursive processing
      }
    }
    
    // Now that we have created the alloca instructions that we want to use,
    // expand the getelementptr instructions to use them.
    //
    for (Value::use_iterator I = AI->use_begin(), E = AI->use_end();
         I != E; ++I) {
      Instruction *User = cast<Instruction>(*I);
      if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(User)) {
        // We now know that the GEP is of the form: GEP <ptr>, 0, <cst>
        uint64_t Idx = cast<ConstantInt>(GEPI->getOperand(2))->getRawValue();
        
        assert(Idx < ElementAllocas.size() && "Index out of range?");
        AllocaInst *AllocaToUse = ElementAllocas[Idx];

        Value *RepValue;
        if (GEPI->getNumOperands() == 3) {
          // Do not insert a new getelementptr instruction with zero indices,
          // only to have it optimized out later.
          RepValue = AllocaToUse;
        } else {
          // We are indexing deeply into the structure, so we still need a
          // getelement ptr instruction to finish the indexing.  This may be
          // expanded itself once the worklist is rerun.
          //
          std::string OldName = GEPI->getName();  // Steal the old name...
          std::vector<Value*> NewArgs;
          NewArgs.push_back(Constant::getNullValue(Type::LongTy));
          NewArgs.insert(NewArgs.end(), GEPI->op_begin()+3, GEPI->op_end());
          GEPI->setName("");
          RepValue =
            new GetElementPtrInst(AllocaToUse, NewArgs, OldName, GEPI);
        }

        // Move all of the users over to the new GEP.
        GEPI->replaceAllUsesWith(RepValue);
        // Delete the old GEP
        GEPI->getParent()->getInstList().erase(GEPI);
      } else {
        assert(0 && "Unexpected instruction type!");
      }
    }

    // Finally, delete the Alloca instruction
    AI->getParent()->getInstList().erase(AI);
    NumReplaced++;
  }

  return Changed;
}


/// isSafeUseOfAllocation - Check to see if this user is an allowed use for an
/// aggregate allocation.
///
bool SROA::isSafeUseOfAllocation(Instruction *User) {
  if (!isa<GetElementPtrInst>(User)) return false;

  GetElementPtrInst *GEPI = cast<GetElementPtrInst>(User);
  gep_type_iterator I = gep_type_begin(GEPI), E = gep_type_end(GEPI);

  // The GEP is safe to transform if it is of the form GEP <ptr>, 0, <cst>
  if (I == E ||
      I.getOperand() != Constant::getNullValue(I.getOperand()->getType()))
    return false;

  ++I;
  if (I == E || !isa<ConstantInt>(I.getOperand()))
    return false;

  // If this is a use of an array allocation, do a bit more checking for sanity.
  if (const ArrayType *AT = dyn_cast<ArrayType>(*I)) {
    uint64_t NumElements = AT->getNumElements();
    
    // Check to make sure that index falls within the array.  If not,
    // something funny is going on, so we won't do the optimization.
    //
    if (cast<ConstantInt>(GEPI->getOperand(2))->getRawValue() >= NumElements)
      return false;
  }

  // If there are any non-simple uses of this getelementptr, make sure to reject
  // them.
  return isSafeElementUse(GEPI);
}

/// isSafeElementUse - Check to see if this use is an allowed use for a
/// getelementptr instruction of an array aggregate allocation.
///
bool SROA::isSafeElementUse(Value *Ptr) {
  for (Value::use_iterator I = Ptr->use_begin(), E = Ptr->use_end();
       I != E; ++I) {
    Instruction *User = cast<Instruction>(*I);
    switch (User->getOpcode()) {
    case Instruction::Load:  break;
    case Instruction::Store:
      // Store is ok if storing INTO the pointer, not storing the pointer
      if (User->getOperand(0) == Ptr) return false;
      break;
    case Instruction::GetElementPtr: {
      GetElementPtrInst *GEP = cast<GetElementPtrInst>(User);
      if (GEP->getNumOperands() > 1) {
        if (!isa<Constant>(GEP->getOperand(1)) ||
            !cast<Constant>(GEP->getOperand(1))->isNullValue())
          return false;  // Using pointer arithmetic to navigate the array...
      }
      if (!isSafeElementUse(GEP)) return false;
      break;
    }
    default:
      DEBUG(std::cerr << "  Transformation preventing inst: " << *User);
      return false;
    }
  }
  return true;  // All users look ok :)
}


/// isSafeStructAllocaToPromote - Check to see if the specified allocation of a
/// structure can be broken down into elements.
///
bool SROA::isSafeAllocaToPromote(AllocationInst *AI) {
  // Loop over the use list of the alloca.  We can only transform it if all of
  // the users are safe to transform.
  //
  for (Value::use_iterator I = AI->use_begin(), E = AI->use_end();
       I != E; ++I)
    if (!isSafeUseOfAllocation(cast<Instruction>(*I))) {
      DEBUG(std::cerr << "Cannot transform: " << *AI << "  due to user: "
                      << *I);
      return false;
    }
  return true;
}
