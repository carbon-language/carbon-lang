//===- InstructionCombining.cpp - Combine multiple instructions -------------=//
//
// InstructionCombining - Combine instructions to form fewer, simple
//   instructions.  This pass does not modify the CFG, and has a tendancy to
//   make instructions dead, so a subsequent DCE pass is useful.
//
// This pass combines things like:
//    %Y = add int 1, %X
//    %Z = add int 1, %Y
// into:
//    %Z = add int 2, %X
//
// This is a simple worklist driven algorithm.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/InstructionCombining.h"
#include "../TransformInternals.h"
#include "llvm/Optimizations/ConstantHandling.h"
#include "llvm/Method.h"
#include "llvm/iMemory.h"

using namespace opt;

static Instruction *CombineBinOp(BinaryOperator *I) {
  bool Changed = false;

  // First thing we do is make sure that this instruction has a constant on the
  // right hand side if it has any constant arguments.
  //
  if (isa<Constant>(I->getOperand(0)) && !isa<Constant>(I->getOperand(1)))
    if (!I->swapOperands())
      Changed = true;

  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;
    Value *Op1 = I->getOperand(0);
    if (Constant *Op2 = dyn_cast<Constant>(I->getOperand(1))) {
      if (I->getOpcode() == Instruction::Add) {
        if (Instruction *IOp1 = dyn_cast<Instruction>(Op1)) {
          if (IOp1->getOpcode() == Instruction::Add &&
              isa<Constant>(IOp1->getOperand(1))) {
            // Fold:
            //    %Y = add int %X, 1
            //    %Z = add int %Y, 1
            // into:
            //    %Z = add int %X, 2
            //   
            // Constant fold both constants...
            Constant *Val = *Op2 + *cast<Constant>(IOp1->getOperand(1));
            
            if (Val) {
              I->setOperand(0, IOp1->getOperand(0));
              I->setOperand(1, Val);
              LocalChange = true;
            }
          }
          
        }
      }
    }
    Changed |= LocalChange;
  }

  if (!Changed) return 0;
  return I;
}

// Combine Indices - If the source pointer to this mem access instruction is a
// getelementptr instruction, combine the indices of the GEP into this
// instruction
//
static Instruction *CombineIndicies(MemAccessInst *MAI) {
  GetElementPtrInst *Src =
    dyn_cast<GetElementPtrInst>(MAI->getPointerOperand());
  if (!Src) return 0;

  vector<Value *> Indices;
  
  // Only special case we have to watch out for is pointer arithmetic on the
  // 0th index of MAI. 
  unsigned FirstIdx = MAI->getFirstIndexOperandNumber();
  if (FirstIdx == MAI->getNumOperands() || 
      (FirstIdx == MAI->getNumOperands()-1 &&
       MAI->getOperand(FirstIdx) == ConstantUInt::get(Type::UIntTy, 0))) { 
    // Replace the index list on this MAI with the index on the getelementptr
    Indices.insert(Indices.end(), Src->idx_begin(), Src->idx_end());
  } else if (*MAI->idx_begin() == ConstantUInt::get(Type::UIntTy, 0)) { 
    // Otherwise we can do the fold if the first index of the GEP is a zero
    Indices.insert(Indices.end(), Src->idx_begin(), Src->idx_end());
    Indices.insert(Indices.end(), MAI->idx_begin()+1, MAI->idx_end());
  }

  if (Indices.empty()) return 0;  // Can't do the fold?

  switch (MAI->getOpcode()) {
  case Instruction::GetElementPtr:
    return new GetElementPtrInst(Src->getOperand(0), Indices, MAI->getName());
  case Instruction::Load:
    return new LoadInst(Src->getOperand(0), Indices, MAI->getName());
  case Instruction::Store:
    return new StoreInst(MAI->getOperand(0), Src->getOperand(0),
                         Indices, MAI->getName());
  default:
    assert(0 && "Unknown memaccessinst!");
    break;
  }
  abort();
  return 0;
}

bool InstructionCombining::CombineInstruction(Instruction *I) {
  Instruction *Result = 0;
  if (BinaryOperator *BOP = dyn_cast<BinaryOperator>(I))
    Result = CombineBinOp(BOP);
  else if (MemAccessInst *MAI = dyn_cast<MemAccessInst>(I))
    Result = CombineIndicies(MAI);

  if (!Result) return false;
  if (Result == I) return true;

  // If we get to here, we are to replace I with Result.
  ReplaceInstWithInst(I, Result);
  return true;
}


bool InstructionCombining::doit(Method *M) {
  // Start the worklist out with all of the instructions in the method in it.
  vector<Instruction*> WorkList(M->inst_begin(), M->inst_end());

  while (!WorkList.empty()) {
    Instruction *I = WorkList.back();  // Get an instruction from the worklist
    WorkList.pop_back();

    // Now that we have an instruction, try combining it to simplify it...
    if (CombineInstruction(I)) {
      // The instruction was simplified, add all users of the instruction to
      // the work lists because they might get more simplified now...
      //
      for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
           UI != UE; ++UI)
        if (Instruction *User = dyn_cast<Instruction>(*UI))
          WorkList.push_back(User);
    }
  }

  return false;
}
