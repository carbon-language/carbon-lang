


// FIXME: document

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/iTerminators.h"
#include "llvm/Function.h"
#include <map>

// FIXME: This should be merged with FunctionInlining

// RemapInstruction - Convert the instruction operands from referencing the 
// current values into those specified by ValueMap.
//
static inline void RemapInstruction(Instruction *I, 
                                    std::map<const Value *, Value*> &ValueMap) {
  for (unsigned op = 0, E = I->getNumOperands(); op != E; ++op) {
    const Value *Op = I->getOperand(op);
    Value *V = ValueMap[Op];
    if (!V && (isa<GlobalValue>(Op) || isa<Constant>(Op)))
      continue;  // Globals and constants don't get relocated

#ifndef NDEBUG
    if (!V) {
      std::cerr << "Val = \n" << Op << "Addr = " << (void*)Op;
      std::cerr << "\nInst = " << I;
    }
#endif
    assert(V && "Referenced value not in value map!");
    I->setOperand(op, V);
  }
}

// Clone OldFunc into NewFunc, transforming the old arguments into references to
// ArgMap values.
//
void CloneFunctionInto(Function *NewFunc, const Function *OldFunc,
                       const std::vector<Value*> &ArgMap,
                       std::vector<ReturnInst*> &Returns,
                       const char *NameSuffix) {
  assert(NameSuffix && "NameSuffix cannot be null!");
  assert(OldFunc->asize() == ArgMap.size() &&
         "Improper number of argument values to map specified!");
  
  // Keep a mapping between the original function's values and the new
  // duplicated code's values.  This includes all of: Function arguments,
  // instruction values, constant pool entries, and basic blocks.
  //
  std::map<const Value *, Value*> ValueMap;

  // Add all of the function arguments to the mapping...
  unsigned i = 0;
  for (Function::const_aiterator I = OldFunc->abegin(), E = OldFunc->aend();
       I != E; ++I, ++i)
    ValueMap[I] = ArgMap[i];


  // Loop over all of the basic blocks in the function, cloning them as
  // appropriate.  Note that we save BE this way in order to handle cloning of
  // recursive functions into themselves.
  //
  for (Function::const_iterator BI = OldFunc->begin(), BE = OldFunc->end();
       BI != BE; ++BI) {
    const BasicBlock &BB = *BI;
    
    // Create a new basic block to copy instructions into!
    BasicBlock *CBB = new BasicBlock("", NewFunc);
    if (BB.hasName()) CBB->setName(BB.getName()+NameSuffix);
    ValueMap[&BB] = CBB;                       // Add basic block mapping.

    // Loop over all instructions copying them over...
    for (BasicBlock::const_iterator II = BB.begin(), IE = BB.end();
         II != IE; ++II) {
      Instruction *NewInst = II->clone();
      if (II->hasName())
        NewInst->setName(II->getName()+NameSuffix);     // Name is not cloned...
      CBB->getInstList().push_back(NewInst);
      ValueMap[II] = NewInst;                // Add instruction map to value.
    }

    if (ReturnInst *RI = dyn_cast<ReturnInst>(CBB->getTerminator()))
      Returns.push_back(RI);
  }

  // Loop over all of the instructions in the function, fixing up operand 
  // references as we go.  This uses ValueMap to do all the hard work.
  //
  for (Function::const_iterator BB = OldFunc->begin(), BE = OldFunc->end();
       BB != BE; ++BB) {
    BasicBlock *NBB = cast<BasicBlock>(ValueMap[BB]);
    
    // Loop over all instructions, fixing each one as we find it...
    for (BasicBlock::iterator II = NBB->begin(); II != NBB->end(); ++II)
      RemapInstruction(II, ValueMap);
  }
}
