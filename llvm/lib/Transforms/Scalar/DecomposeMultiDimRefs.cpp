//===- llvm/Transforms/DecomposeMultiDimRefs.cpp - Lower array refs to 1D ---=//
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
static BasicBlock::iterator
decomposeArrayRef(BasicBlock::iterator& BBI)
{
  MemAccessInst *memI = cast<MemAccessInst>(*BBI);
  BasicBlock* BB = memI->getParent();
  Value* lastPtr = memI->getPointerOperand();
  vector<Instruction*> newIvec;
  
  // Process each index except the last one.
  // 
  MemAccessInst::const_op_iterator OI = memI->idx_begin();
  MemAccessInst::const_op_iterator OE = memI->idx_end();
  for ( ; OI != OE; ++OI)
    {
      assert(isa<PointerType>(lastPtr->getType()));
      
      if (OI+1 == OE)                   // stop before the last operand
        break;
      
      // Check for a zero index.  This will need a cast instead of
      // a getElementPtr, or it may need neither.
      bool indexIsZero = bool(isa<ConstantUInt>(*OI) && 
                              cast<ConstantUInt>(*OI)->getValue() == 0);
      
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
        PointerType::get(isa<StructType>(nextType)? nextType
                         : cast<ArrayType>(nextType)->getElementType());
      
      // Instruction 1: nextPtr1 = GetElementPtr lastPtr, idxVec
      // This is not needed if the index is zero.
      Value* gepValue;
      if (indexIsZero)
        gepValue = lastPtr;
      else
        {
          gepValue = new GetElementPtrInst(lastPtr, idxVec,"ptr1");
          newIvec.push_back(cast<Instruction>(gepValue));
        }
      
      // Instruction 2: nextPtr2 = cast nextPtr1 to nextPtrType
      // This is not needed if the two types are identical.
      Value* castInst;
      if (gepValue->getType() == nextPtrType)
        castInst = gepValue;
      else
        {
          castInst = new CastInst(gepValue, nextPtrType, "ptr2");
          newIvec.push_back(cast<Instruction>(castInst));
        }
      
      lastPtr = castInst;
    }
  
  // 
  // Now create a new instruction to replace the original one
  //
  PointerType* ptrType = cast<PointerType>(lastPtr->getType());
  assert(ptrType);

  // First, get the final index vector.  As above, we may need an initial [0].
  vector<Value*> idxVec(1, *OI);
  if (isa<StructType>(ptrType->getElementType())
      && ! ptrType->indexValid(*OI))
    idxVec.insert(idxVec.begin(), ConstantUInt::get(Type::UIntTy, 0));
  
  const std::string newInstName = memI->hasName()? memI->getName()
                                                 : string("finalRef");
  Instruction* newInst = NULL;
  
  switch(memI->getOpcode())
    {
    case Instruction::Load:
      newInst = new LoadInst(lastPtr, idxVec /*, newInstName */); break;
    case Instruction::Store:
      newInst = new StoreInst(memI->getOperand(0),
                              lastPtr, idxVec /*, newInstName */); break;
      break;
    case Instruction::GetElementPtr:
      newInst = new GetElementPtrInst(lastPtr, idxVec /*, newInstName */); break;
    default:
      assert(0 && "Unrecognized memory access instruction"); break;
    }
  
  newIvec.push_back(newInst);
  
  // Replace all uses of the old instruction with the new
  memI->replaceAllUsesWith(newInst);
  
  BasicBlock::iterator newI = BBI;;
  for (int i = newIvec.size()-1; i >= 0; i--)
    newI = BB->getInstList().insert(newI, newIvec[i]);
  
  // Now delete the old instruction and return a pointer to the last new one
  BB->getInstList().remove(memI);
  delete memI;
  
  return newI + newIvec.size() - 1;           // pointer to last new instr
}


//---------------------------------------------------------------------------
// Entry point for array or  structure references with multiple indices.
//---------------------------------------------------------------------------

static bool
doDecomposeMultiDimRefs(Function *F)
{
  bool changed = false;
  
  for (Function::iterator BI = F->begin(), BE = F->end(); BI != BE; ++BI)
    for (BasicBlock::iterator newI, II = (*BI)->begin();
         II != (*BI)->end(); II = ++newI)
      {
        newI = II;
        if (MemAccessInst *memI = dyn_cast<MemAccessInst>(*II))
          if (memI->getNumOperands() > 1 + memI->getFirstIndexOperandNumber())
            {
              newI = decomposeArrayRef(II);
              changed = true;
            }
      }
  
  return changed;
}


namespace {
  struct DecomposeMultiDimRefsPass : public FunctionPass {
    virtual bool runOnFunction(Function *F) {
      return doDecomposeMultiDimRefs(F);
    }
  };
}

Pass *createDecomposeMultiDimRefsPass() {
  return new DecomposeMultiDimRefsPass();
}
