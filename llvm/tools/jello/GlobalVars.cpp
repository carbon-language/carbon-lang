//===-- GlobalVars.cpp - Code to emit global variables to memory ----------===//
//
// This file contains the code to generate global variables to memory.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Target/TargetMachine.h"
#include "VM.h"

/// EmitGlobals - Emit all of the global variables to memory, storing their
/// addresses into GlobalAddress.  This must make sure to copy the contents of
/// their initializers into the memory.
///
void VM::emitGlobals() {
  const TargetData &TD = TM.getTargetData();
  
  // Loop over all of the global variables in the program, allocating the memory
  // to hold them.
  for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
    if (!I->isExternal()) {
      // Get the type of the global...
      const Type *Ty = I->getType()->getElementType();
      
      // Allocate some memory for it!
      GlobalAddress[I] = new char[TD.getTypeSize(Ty)];
      
      std::cerr << "Allocated global '" << I->getName()
                << "' to addr 0x" << std::hex << GlobalAddress[I] << std::dec
                << "\n";
    } else {
      assert(0 && "References to external globals not handled yet!");
    }
  
  // Now that all of the globals are set up in memory, loop through them all and
  // initialize their contents.
  for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
    if (!I->isExternal())
      emitConstantToMemory(I->getInitializer(), GlobalAddress[I]);
}

/// emitConstantToMemory - Use the specified LLVM constant to initialize the
/// specified region of memory.
///
void VM::emitConstantToMemory(Constant *Init, void *Addr) {
  const TargetData &TD = TM.getTargetData();
  if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(Init)) {
    switch (CI->getType()->getPrimitiveID()) {
    case Type::BoolTyID:
      *(char*)Addr = cast<ConstantBool>(CI)->getValue();
      return;
    case Type::UByteTyID:
      *(unsigned char*)Addr = cast<ConstantUInt>(CI)->getValue();
      return;
    case Type::SByteTyID:
      *(  signed char*)Addr = cast<ConstantSInt>(CI)->getValue();
      return;
    case Type::UShortTyID:
      *(unsigned short*)Addr = cast<ConstantUInt>(CI)->getValue();
      return;
    case Type::ShortTyID:
      *(  signed short*)Addr = cast<ConstantSInt>(CI)->getValue();
      return;
    case Type::UIntTyID:
      *(unsigned int*)Addr = cast<ConstantUInt>(CI)->getValue();
      return;
    case Type::IntTyID:
      *(  signed int*)Addr = cast<ConstantSInt>(CI)->getValue();
      return;
    case Type::ULongTyID:
      *(uint64_t*)Addr = cast<ConstantUInt>(CI)->getValue();
      return;
    case Type::LongTyID:
      *(int64_t*)Addr = cast<ConstantSInt>(CI)->getValue();
      return;
    default: break;
    }
  } else if (ConstantArray *CA = dyn_cast<ConstantArray>(Init)) {
    unsigned ElementSize = TD.getTypeSize(CA->getType()->getElementType());
    for (unsigned i = 0, e = CA->getType()->getNumElements(); i != e; ++i) {
      emitConstantToMemory(cast<Constant>(CA->getOperand(i)), Addr);
      Addr = (char*)Addr+ElementSize;
    }
    return;
  }

  assert(0 && "Don't know how to emit this constant to memory!");
}
