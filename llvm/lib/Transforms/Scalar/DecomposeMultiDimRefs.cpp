//===- llvm/Transforms/DecomposeMultiDimRefs.cpp - Lower array refs to 1D -===//
//
// DecomposeMultiDimRefs - Convert multi-dimensional references consisting of
// any combination of 2 or more array and structure indices into a sequence of
// instructions (using getelementpr and cast) so that each instruction has at
// most one index (except structure references, which need an extra leading
// index of [0]).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constant.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Pass.h"
#include "Support/StatisticReporter.h"

static Statistic<> NumAdded("lowerrefs\t\t- New instructions added");

namespace {
  struct DecomposePass : public BasicBlockPass {
    const char *getPassName() const { return "Decompose Subscripting Exps"; }

    virtual bool runOnBasicBlock(BasicBlock &BB);

  private:
    static void decomposeArrayRef(BasicBlock::iterator &BBI);
  };
}

Pass *createDecomposeMultiDimRefsPass() {
  return new DecomposePass();
}


// runOnBasicBlock - Entry point for array or structure references with multiple
// indices.
//
bool DecomposePass::runOnBasicBlock(BasicBlock &BB) {
  bool Changed = false;
  for (BasicBlock::iterator II = BB.begin(); II != BB.end(); ) {
    if (MemAccessInst *MAI = dyn_cast<MemAccessInst>(&*II)) {
      if (MAI->getNumOperands() > MAI->getFirstIndexOperandNumber()+1) {
        decomposeArrayRef(II);
        Changed = true;
      } else {
        ++II;
      }
    } else {
      ++II;
    }
  }
  
  return Changed;
}

// 
// For any combination of 2 or more array and structure indices,
// this function repeats the foll. until we have a one-dim. reference: {
//      ptr1 = getElementPtr [CompositeType-N] * lastPtr, uint firstIndex
//      ptr2 = cast [CompositeType-N] * ptr1 to [CompositeType-N] *
// }
// Then it replaces the original instruction with an equivalent one that
// uses the last ptr2 generated in the loop and a single index.
// If any index is (uint) 0, we omit the getElementPtr instruction.
// 

void DecomposePass::decomposeArrayRef(BasicBlock::iterator &BBI) {
  MemAccessInst &MAI = cast<MemAccessInst>(*BBI);
  BasicBlock *BB = MAI.getParent();
  Value *LastPtr = MAI.getPointerOperand();

  // Remove the instruction from the stream
  BB->getInstList().remove(BBI);

  std::vector<Instruction*> NewInsts;
  
  // Process each index except the last one.
  // 

  User::const_op_iterator OI = MAI.idx_begin(), OE = MAI.idx_end();
  for (; OI+1 != OE; ++OI) {
    assert(isa<PointerType>(LastPtr->getType()));
      
    // Check for a zero index.  This will need a cast instead of
    // a getElementPtr, or it may need neither.
    bool indexIsZero = isa<Constant>(*OI) && 
                       cast<Constant>(OI->get())->isNullValue() &&
                       OI->get()->getType() == Type::UIntTy;
      
    // Extract the first index.  If the ptr is a pointer to a structure
    // and the next index is a structure offset (i.e., not an array offset), 
    // we need to include an initial [0] to index into the pointer.
    //

    std::vector<Value*> Indices;
    const PointerType *PtrTy = cast<PointerType>(LastPtr->getType());

    if (isa<StructType>(PtrTy->getElementType())
        && !PtrTy->indexValid(*OI))
      Indices.push_back(Constant::getNullValue(Type::UIntTy));
    Indices.push_back(*OI);

    // Get the type obtained by applying the first index.
    // It must be a structure or array.
    const Type *NextTy = MemAccessInst::getIndexedType(LastPtr->getType(),
                                                       Indices, true);
    assert(isa<CompositeType>(NextTy));
    
    // Get a pointer to the structure or to the elements of the array.
    const Type *NextPtrTy =
      PointerType::get(isa<StructType>(NextTy) ? NextTy
                       : cast<ArrayType>(NextTy)->getElementType());
      
    // Instruction 1: nextPtr1 = GetElementPtr LastPtr, Indices
    // This is not needed if the index is zero.
    if (!indexIsZero) {
      LastPtr = new GetElementPtrInst(LastPtr, Indices, "ptr1");
      NewInsts.push_back(cast<Instruction>(LastPtr));
      ++NumAdded;
    }

      
    // Instruction 2: nextPtr2 = cast nextPtr1 to NextPtrTy
    // This is not needed if the two types are identical.
    //
    if (LastPtr->getType() != NextPtrTy) {
      LastPtr = new CastInst(LastPtr, NextPtrTy, "ptr2");
      NewInsts.push_back(cast<Instruction>(LastPtr));
      ++NumAdded;
    }
  }
  
  // 
  // Now create a new instruction to replace the original one
  //
  const PointerType *PtrTy = cast<PointerType>(LastPtr->getType());

  // First, get the final index vector.  As above, we may need an initial [0].

  std::vector<Value*> Indices;
  if (isa<StructType>(PtrTy->getElementType())
      && !PtrTy->indexValid(*OI))
    Indices.push_back(Constant::getNullValue(Type::UIntTy));

  Indices.push_back(*OI);

  Instruction *NewI = 0;
  switch(MAI.getOpcode()) {
  case Instruction::Load:
    NewI = new LoadInst(LastPtr, Indices, MAI.getName());
    break;
  case Instruction::Store:
    NewI = new StoreInst(MAI.getOperand(0), LastPtr, Indices);
    break;
  case Instruction::GetElementPtr:
    NewI = new GetElementPtrInst(LastPtr, Indices, MAI.getName());
    break;
  default:
    assert(0 && "Unrecognized memory access instruction");
  }
  NewInsts.push_back(NewI);

  
  // Replace all uses of the old instruction with the new
  MAI.replaceAllUsesWith(NewI);

  // Now delete the old instruction...
  delete &MAI;

  // Insert all of the new instructions...
  BB->getInstList().insert(BBI, NewInsts.begin(), NewInsts.end());
  
  // Advance the iterator to the instruction following the one just inserted...
  BBI = NewInsts.back();
  ++BBI;
}
