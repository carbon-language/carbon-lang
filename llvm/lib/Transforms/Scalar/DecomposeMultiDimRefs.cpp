//===- llvm/Transforms/DecomposeMultiDimRefs.cpp - Lower array refs to 1D -===//
//
// DecomposeMultiDimRefs - 
// Convert multi-dimensional references consisting of any combination
// of 2 or more array and structure indices into a sequence of
// instructions (using getelementpr and cast) so that each instruction
// has at most one index (except structure references,
// which need an extra leading index of [0]).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/DecomposeMultiDimRefs.h"
#include "llvm/Constants.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"

namespace {
  struct DecomposePass : public BasicBlockPass {
    virtual bool runOnBasicBlock(BasicBlock *BB);

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
bool DecomposePass::runOnBasicBlock(BasicBlock *BB) {
  bool Changed = false;
  
  for (BasicBlock::iterator II = BB->begin(); II != BB->end(); ) {
    if (MemAccessInst *MAI = dyn_cast<MemAccessInst>(*II)) {
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
void DecomposePass::decomposeArrayRef(BasicBlock::iterator &BBI){
  MemAccessInst *memI = cast<MemAccessInst>(*BBI);
  BasicBlock* BB = memI->getParent();
  Value* lastPtr = memI->getPointerOperand();

  // Remove the instruction from the stream
  BB->getInstList().remove(BBI);

  vector<Instruction*> newIvec;
  
  // Process each index except the last one.
  // 
  User::const_op_iterator OI = memI->idx_begin(), OE = memI->idx_end();
  for (; OI != OE && OI+1 != OE; ++OI) {
    assert(isa<PointerType>(lastPtr->getType()));
      
    // Check for a zero index.  This will need a cast instead of
    // a getElementPtr, or it may need neither.
    bool indexIsZero = isa<ConstantUInt>(*OI) && 
                       cast<Constant>(*OI)->isNullValue();
      
    // Extract the first index.  If the ptr is a pointer to a structure
    // and the next index is a structure offset (i.e., not an array offset), 
    // we need to include an initial [0] to index into the pointer.
    vector<Value*> idxVec(1, *OI);
    PointerType* ptrType = cast<PointerType>(lastPtr->getType());
    if (isa<StructType>(ptrType->getElementType())
        && ! ptrType->indexValid(*OI))
      idxVec.insert(idxVec.begin(), ConstantUInt::get(Type::UIntTy, 0));
    
    // Get the type obtained by applying the first index.
    // It must be a structure or array.
    const Type* nextType = MemAccessInst::getIndexedType(lastPtr->getType(),
                                                         idxVec, true);
    assert(isa<StructType>(nextType) || isa<ArrayType>(nextType));
    
    // Get a pointer to the structure or to the elements of the array.
    const Type* nextPtrType =
      PointerType::get(isa<StructType>(nextType) ? nextType
                       : cast<ArrayType>(nextType)->getElementType());
      
    // Instruction 1: nextPtr1 = GetElementPtr lastPtr, idxVec
    // This is not needed if the index is zero.
    Value *gepValue;
    if (indexIsZero)
      gepValue = lastPtr;
    else {
      gepValue = new GetElementPtrInst(lastPtr, idxVec,"ptr1");
      newIvec.push_back(cast<Instruction>(gepValue));
    }
      
    // Instruction 2: nextPtr2 = cast nextPtr1 to nextPtrType
    // This is not needed if the two types are identical.
    Value *castInst;
    if (gepValue->getType() == nextPtrType)
      castInst = gepValue;
    else {
      castInst = new CastInst(gepValue, nextPtrType, "ptr2");
      newIvec.push_back(cast<Instruction>(castInst));
    }
      
    lastPtr = castInst;
  }
  
  // 
  // Now create a new instruction to replace the original one
  //
  PointerType *ptrType = cast<PointerType>(lastPtr->getType());

  // First, get the final index vector.  As above, we may need an initial [0].
  vector<Value*> idxVec(1, *OI);
  if (isa<StructType>(ptrType->getElementType())
      && !ptrType->indexValid(*OI))
    idxVec.insert(idxVec.begin(), Constant::getNullValue(Type::UIntTy));
  
  Instruction* newInst = NULL;
  switch(memI->getOpcode()) {
  case Instruction::Load:
    newInst = new LoadInst(lastPtr, idxVec, memI->getName());
    break;
  case Instruction::Store:
    newInst = new StoreInst(memI->getOperand(0), lastPtr, idxVec);
    break;
  case Instruction::GetElementPtr:
    newInst = new GetElementPtrInst(lastPtr, idxVec, memI->getName());
    break;
  default:
    assert(0 && "Unrecognized memory access instruction");
  }
  newIvec.push_back(newInst);
  
  // Replace all uses of the old instruction with the new
  memI->replaceAllUsesWith(newInst);

  // Now delete the old instruction...
  delete memI;

  // Convert our iterator into an index... that cannot get invalidated
  unsigned ItOffs = BBI-BB->begin();

  // Insert all of the new instructions...
  BB->getInstList().insert(BBI, newIvec.begin(), newIvec.end());
  
  // Advance the iterator to the instruction following the one just inserted...
  BBI = BB->begin() + (ItOffs+newIvec.size());
}
