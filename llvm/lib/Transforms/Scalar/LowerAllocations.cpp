//===- llvm/Transforms/LowerAllocations.h - Remove Malloc & Free Insts ------=//
//
// This file implements a pass that lowers malloc and free instructions to
// calls to %malloc & %free functions.  This transformation is a target
// dependant tranformation because we depend on the size of data types and
// alignment constraints.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/LowerAllocations.h"
#include "llvm/Target/TargetData.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/SymbolTable.h"

// doPassInitialization - For the lower allocations pass, this ensures that a
// module contains a declaration for a malloc and a free function.
//
// This function is always successful.
//
bool LowerAllocations::doPassInitializationVirt(Module *M) {
  const MethodType *MallocType = 
    MethodType::get(PointerType::get(Type::UByteTy),
                    vector<const Type*>(1, Type::UIntTy), false);

  SymbolTable *SymTab = M->getSymbolTableSure();
  
  // Check for a definition of malloc
  if (Value *V = SymTab->lookup(PointerType::get(MallocType), "malloc")) {
    MallocMeth = cast<Method>(V);      // Yup, got it
  } else {                             // Nope, add one
    M->getMethodList().push_back(MallocMeth = new Method(MallocType, "malloc"));
  }

  const MethodType *FreeType = 
    MethodType::get(Type::VoidTy,
                    vector<const Type*>(1, PointerType::get(Type::UByteTy)),
		    false);

  // Check for a definition of free
  if (Value *V = SymTab->lookup(PointerType::get(FreeType), "free")) {
    FreeMeth = cast<Method>(V);      // Yup, got it
  } else {                             // Nope, add one
    M->getMethodList().push_back(FreeMeth = new Method(FreeType, "free"));
  }

  return false;  // Always successful
}

// doPerMethodWork - This method does the actual work of converting
// instructions over, assuming that the pass has already been initialized.
//
bool LowerAllocations::doPerMethodWorkVirt(Method *M) {
  assert(MallocMeth && FreeMeth && M && "Pass not initialized!");

  // Loop over all of the instructions, looking for malloc or free instructions
  for (Method::iterator BBI = M->begin(), BBE = M->end(); BBI != BBE; ++BBI) {
    BasicBlock *BB = *BBI;
    for (unsigned i = 0; i < BB->size(); ++i) {
      BasicBlock::InstListType &BBIL = BB->getInstList();
      if (MallocInst *MI = dyn_cast<MallocInst>(*(BBIL.begin()+i))) {
        BBIL.remove(BBIL.begin()+i);   // remove the malloc instr...
        
        const Type *AllocTy = cast<PointerType>(MI->getType())->getValueType();

        // If the user is allocating an unsized array with a dynamic size arg,
        // start by getting the size of one element.
        //
        if (const ArrayType *ATy = dyn_cast<ArrayType>(AllocTy)) 
          if (ATy->isUnsized()) AllocTy = ATy->getElementType();

        // Get the number of bytes to be allocated for one element of the
        // requested type...
        unsigned Size = DataLayout.getTypeSize(AllocTy);

        // malloc(type) becomes sbyte *malloc(constint)
        Value *MallocArg = ConstPoolUInt::get(Type::UIntTy, Size);
        if (MI->getNumOperands() && Size == 1) {
          MallocArg = MI->getOperand(0);         // Operand * 1 = Operand
        } else if (MI->getNumOperands()) {
          // Multiply it by the array size if neccesary...
          MallocArg = BinaryOperator::create(Instruction::Mul,MI->getOperand(0),
                                             MallocArg);
          BBIL.insert(BBIL.begin()+i++, cast<Instruction>(MallocArg));
        }

        // Create the call to Malloc...
        CallInst *MCall = new CallInst(MallocMeth,
                                       vector<Value*>(1, MallocArg));
        BBIL.insert(BBIL.begin()+i, MCall);

        // Create a cast instruction to convert to the right type...
        CastInst *MCast = new CastInst(MCall, MI->getType());
        BBIL.insert(BBIL.begin()+i+1, MCast);

        // Replace all uses of the old malloc inst with the cast inst
        MI->replaceAllUsesWith(MCast);
        delete MI;                          // Delete the malloc inst
      } else if (FreeInst *FI = dyn_cast<FreeInst>(*(BBIL.begin()+i))) {
        BBIL.remove(BB->getInstList().begin()+i);

        // Cast the argument to free into a ubyte*...
        CastInst *MCast = new CastInst(FI->getOperand(0), 
                                       PointerType::get(Type::UByteTy));
        BBIL.insert(BBIL.begin()+i, MCast);

        // Insert a call to the free function...
        CallInst *FCall = new CallInst(FreeMeth,
                                       vector<Value*>(1, MCast));
        BBIL.insert(BBIL.begin()+i+1, FCall);

        // Delete the old free instruction
        delete FI;
      }
    }
  }

  return false;   // Always successful
}

