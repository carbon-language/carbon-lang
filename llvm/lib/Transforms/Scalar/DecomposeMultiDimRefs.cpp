//===- llvm/Transforms/DecomposeArrayRefs.cpp - Lower array refs to 1D -----=//
//
// DecomposeArrayRefs - 
// Convert multi-dimensional array references into a sequence of
// instructions (using getelementpr and cast) so that each instruction
// has at most one array offset.
//
//===---------------------------------------------------------------------===//

#include "llvm/Transforms/DecomposeArrayRefs.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/Pass.h"


// 
// This function repeats until we have a one-dim. reference: {
//      // For an N-dim array ref, where N > 1, insert:
//      aptr1 = getElementPtr [N-dim array] * lastPtr, uint firstIndex
//      aptr2 = cast [N-dim-arry] * aptr to [<N-1>-dim-array] *
// }
// Then it replaces the original instruction with an equivalent one that
// uses the last aptr2 generated in the loop and a single index.
// 
static BasicBlock::reverse_iterator
decomposeArrayRef(BasicBlock::reverse_iterator& BBI)
{
  MemAccessInst *memI = cast<MemAccessInst>(*BBI);
  BasicBlock* BB = memI->getParent();
  Value* lastPtr = memI->getPointerOperand();
  vector<Instruction*> newIvec;
  
  MemAccessInst::const_op_iterator OI = memI->idx_begin();
  for (MemAccessInst::const_op_iterator OE = memI->idx_end(); OI != OE; ++OI)
    {
      if (OI+1 == OE)                     // skip the last operand
        break;
      
      assert(isa<PointerType>(lastPtr->getType()));
      vector<Value*> idxVec(1, *OI);

      // The first index does not change the type of the pointer
      // since all pointers are treated as potential arrays (i.e.,
      // int *X is either a scalar X[0] or an array at X[i]).
      // 
      const Type* nextPtrType;
      // if (OI == memI->idx_begin())
      //   nextPtrType = lastPtr->getType();
      // else
      //   {
             const Type* nextArrayType =  
               MemAccessInst::getIndexedType(lastPtr->getType(), idxVec,
                                             /*allowCompositeLeaf*/ true);
             nextPtrType = PointerType::get(cast<SequentialType>(nextArrayType)
                                            ->getElementType());
      //   }
      
      Instruction* gepInst  = new GetElementPtrInst(lastPtr, idxVec, "aptr1");
      Instruction* castInst = new CastInst(gepInst, nextPtrType, "aptr2");
      lastPtr  = castInst;
      
      newIvec.push_back(gepInst);
      newIvec.push_back(castInst);
    }
  
  // Now create a new instruction to replace the original one
  assert(lastPtr != memI->getPointerOperand() && "the above loop did not execute?");
  assert(isa<PointerType>(lastPtr->getType()));
  vector<Value*> idxVec(1, *OI);
  const std::string newInstName = memI->hasName()? memI->getName()
                                                 : string("oneDimRef");
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
  
  // Insert the instructions created in reverse order.  insert is destructive
  // so we always have to use the new pointer returned by insert.
  BasicBlock::iterator newI = BBI.base(); // gives ptr to instr. after memI
  --newI;                                 // step back to memI
  for (int i = newIvec.size()-1; i >= 0; i--)
    newI = BB->getInstList().insert(newI, newIvec[i]);
  
  // Now delete the old instruction and return a pointer to the first new one
  BB->getInstList().remove(memI);
  delete memI;
  
  BasicBlock::reverse_iterator retI(newI); // reverse ptr to instr before newI
  return --retI;                           // reverse pointer to newI
}


//---------------------------------------------------------------------------
// Entry point for decomposing multi-dimensional array references
//---------------------------------------------------------------------------

static bool
doDecomposeArrayRefs(Method *M)
{
  bool changed = false;
  
  for (Method::iterator BI = M->begin(), BE = M->end(); BI != BE; ++BI)
    for (BasicBlock::reverse_iterator newI, II=(*BI)->rbegin();
         II != (*BI)->rend(); II = ++newI)
      {
        newI = II;
        if (MemAccessInst *memI = dyn_cast<MemAccessInst>(*II))
          { // Check for a multi-dimensional array access
            const PointerType* ptrType =
              cast<PointerType>(memI->getPointerOperand()->getType()); 
            if (isa<ArrayType>(ptrType->getElementType()) &&
                memI->getNumOperands() > 1+ memI->getFirstIndexOperandNumber())
              {
                newI = decomposeArrayRef(II);
                changed = true;
              }
          }
      }
  
  return changed;
}


namespace {
  struct DecomposeArrayRefsPass : public MethodPass {
    virtual bool runOnMethod(Method *M) { return doDecomposeArrayRefs(M); }
  };
}

Pass *createDecomposeArrayRefsPass() { return new DecomposeArrayRefsPass(); }
