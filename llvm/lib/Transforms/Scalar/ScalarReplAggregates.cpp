//===- ScalarReplAggregates.cpp - Scalar Replacement of Aggregates --------===//
//
// This transformation implements the well known scalar replacement of
// aggregates transformation.  This xform breaks up alloca instructions of
// aggregate type (structure or array) into individual alloca instructions for
// each member (if possible).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/iMemory.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/StringExtras.h"

namespace {
  Statistic<> NumReplaced("scalarrepl", "Number of alloca's broken up");

  struct SROA : public FunctionPass {
    bool runOnFunction(Function &F);

  private:
    bool isSafeElementUse(Value *Ptr);
    bool isSafeUseOfAllocation(Instruction *User);
    bool isSafeStructAllocaToPromote(AllocationInst *AI);
    bool isSafeArrayAllocaToPromote(AllocationInst *AI);
    AllocaInst *AddNewAlloca(Function &F, const Type *Ty, AllocationInst *Base);
  };

  RegisterOpt<SROA> X("scalarrepl", "Scalar Replacement of Aggregates");
}

Pass *createScalarReplAggregatesPass() { return new SROA(); }


// runOnFunction - This algorithm is a simple worklist driven algorithm, which
// runs on all of the malloc/alloca instructions in the function, removing them
// if they are only used by getelementptr instructions.
//
bool SROA::runOnFunction(Function &F) {
  std::vector<AllocationInst*> WorkList;

  // Scan the entry basic block, adding any alloca's and mallocs to the worklist
  BasicBlock &BB = F.getEntryNode();
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
    if (isa<StructType>(AI->getAllocatedType())) {
      if (!isSafeStructAllocaToPromote(AI))
        continue;
    } else if (!isSafeArrayAllocaToPromote(AI))
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
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(User)) {
    // The GEP is safe to transform if it is of the form GEP <ptr>, 0, <cst>
    if (GEPI->getNumOperands() <= 2 ||
        GEPI->getOperand(1) != Constant::getNullValue(Type::LongTy) ||
        !isa<Constant>(GEPI->getOperand(2)) ||
        isa<ConstantExpr>(GEPI->getOperand(2)))
      return false;
  } else {
    return false;
  }
  return true;
}

/// isSafeElementUse - Check to see if this use is an allowed use for a
/// getelementptr instruction of an array aggregate allocation.
///
bool SROA::isSafeElementUse(Value *Ptr) {
  for (Value::use_iterator I = Ptr->use_begin(), E = Ptr->use_end();
       I != E; ++I) {
    Instruction *User = cast<Instruction>(*I);
    switch (User->getOpcode()) {
    case Instruction::Load:  return true;
    case Instruction::Store: return User->getOperand(0) != Ptr;
    case Instruction::GetElementPtr: {
      GetElementPtrInst *GEP = cast<GetElementPtrInst>(User);
      if (GEP->getNumOperands() > 1) {
        if (!isa<Constant>(GEP->getOperand(1)) ||
            !cast<Constant>(GEP->getOperand(1))->isNullValue())
          return false;  // Using pointer arithmetic to navigate the array...
      }
      return isSafeElementUse(GEP);
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
bool SROA::isSafeStructAllocaToPromote(AllocationInst *AI) {
  // Loop over the use list of the alloca.  We can only transform it if all of
  // the users are safe to transform.
  //
  for (Value::use_iterator I = AI->use_begin(), E = AI->use_end();
       I != E; ++I) {
    if (!isSafeUseOfAllocation(cast<Instruction>(*I))) {
      DEBUG(std::cerr << "Cannot transform: " << *AI << "  due to user: "
                      << *I);
      return false;
    }

    // Pedantic check to avoid breaking broken programs...
    if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(*I))
      if (GEPI->getNumOperands() == 3 && !isSafeElementUse(GEPI))
        return false;
  }
  return true;
}


/// isSafeArrayAllocaToPromote - Check to see if the specified allocation of a
/// structure can be broken down into elements.
///
bool SROA::isSafeArrayAllocaToPromote(AllocationInst *AI) {
  const ArrayType *AT = cast<ArrayType>(AI->getAllocatedType());
  int64_t NumElements = AT->getNumElements();

  // Loop over the use list of the alloca.  We can only transform it if all of
  // the users are safe to transform.  Array allocas have extra constraints to
  // meet though.
  //
  for (Value::use_iterator I = AI->use_begin(), E = AI->use_end();
       I != E; ++I) {
    Instruction *User = cast<Instruction>(*I);
    if (!isSafeUseOfAllocation(User)) {
      DEBUG(std::cerr << "Cannot transform: " << *AI << "  due to user: "
                      << User);
      return false;
    }

    // Check to make sure that getelementptr follow the extra rules for arrays:
    if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(User)) {
      // Check to make sure that index falls within the array.  If not,
      // something funny is going on, so we won't do the optimization.
      //
      if (cast<ConstantSInt>(GEPI->getOperand(2))->getValue() >= NumElements)
        return false;

      // Check to make sure that the only thing that uses the resultant pointer
      // is safe for an array access.  For example, code that looks like:
      //   P = &A[0];  P = P + 1
      // is legal, and should prevent promotion.
      //
      if (!isSafeElementUse(GEPI)) {
        DEBUG(std::cerr << "Cannot transform: " << *AI
                        << "  due to uses of user: " << *GEPI);
        return false;
      }
    }
  }
  return true;
}

