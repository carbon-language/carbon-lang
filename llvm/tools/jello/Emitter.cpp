//===-- Emitter.cpp - Write machine code to executable memory -------------===//
//
// This file defines a MachineCodeEmitter object that is used by Jello to write
// machine code to memory and remember where relocatable values lie.
//
//===----------------------------------------------------------------------===//

#include "VM.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace {
  class Emitter : public MachineCodeEmitter {
    VM &TheVM;

    unsigned char *CurBlock;
    unsigned char *CurByte;
  public:
    Emitter(VM &vm) : TheVM(vm) {}

    virtual void startFunction(MachineFunction &F);
    virtual void finishFunction(MachineFunction &F);
    virtual void startBasicBlock(MachineBasicBlock &BB) {}
    virtual void emitByte(unsigned char B);
    virtual void emitPCRelativeDisp(Value *V);
  };
}

MachineCodeEmitter *VM::createEmitter(VM &V) {
  return new Emitter(V);
}


#define _POSIX_MAPPED_FILES
#include <unistd.h>
#include <sys/mman.h>

static void *getMemory() {
  return mmap(0, 4096*2, PROT_READ|PROT_WRITE|PROT_EXEC,
              MAP_PRIVATE|MAP_ANONYMOUS, 0, 0);
}


void Emitter::startFunction(MachineFunction &F) {
  CurBlock = (unsigned char *)getMemory();
  CurByte = CurBlock;  // Start writing at the beginning of the fn.
}

#include <iostream>
#include "llvm/Function.h"

void Emitter::finishFunction(MachineFunction &F) {
  std::cerr << "Finished Code Generation of Function: "
            << F.getFunction()->getName() << ": " << CurByte-CurBlock
            << " bytes of text\n";
  TheVM.addGlobalMapping(F.getFunction(), CurBlock);
}



void Emitter::emitByte(unsigned char B) {
  *CurByte++ = B;   // Write the byte to memory
}


// emitPCRelativeDisp - Just output a displacement that will cause a reference
// to the zero page, which will cause a seg-fault, causing things to get
// resolved on demand.  Keep track of these markers.
//
void Emitter::emitPCRelativeDisp(Value *V) {
  unsigned ZeroAddr = -(unsigned)CurByte;  // Calculate displacement to null
  *(unsigned*)CurByte = ZeroAddr;   // 4 byte offset
  CurByte += 4;
}
