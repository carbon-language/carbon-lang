//===-- Emitter.cpp - Write machine code to executable memory -------------===//
//
// This file defines a MachineCodeEmitter object that is used by Jello to write
// machine code to memory and remember where relocatable values lie.
//
//===----------------------------------------------------------------------===//

#include "VM.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Function.h"
#include "Support/Statistic.h"

namespace {
  class Emitter : public MachineCodeEmitter {
    VM &TheVM;

    unsigned char *CurBlock;
    unsigned char *CurByte;
    
    std::vector<std::pair<BasicBlock*, unsigned *> > BBRefs;
    std::map<BasicBlock*, unsigned> BBLocations;
  public:
    Emitter(VM &vm) : TheVM(vm) {}

    virtual void startFunction(MachineFunction &F);
    virtual void finishFunction(MachineFunction &F);
    virtual void startBasicBlock(MachineBasicBlock &BB);
    virtual void emitByte(unsigned char B);
    virtual void emitPCRelativeDisp(Value *V);
    virtual void emitGlobalAddress(GlobalValue *V);
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
  TheVM.addGlobalMapping(F.getFunction(), CurBlock);
}

void Emitter::finishFunction(MachineFunction &F) {
  for (unsigned i = 0, e = BBRefs.size(); i != e; ++i) {
    unsigned Location = BBLocations[BBRefs[i].first];
    unsigned *Ref = BBRefs[i].second;
    *Ref = Location-(unsigned)Ref-4;
  }
  BBRefs.clear();
  BBLocations.clear();

  DEBUG(std::cerr << "Finished Code Generation of Function: "
                  << F.getFunction()->getName() << ": " << CurByte-CurBlock
                  << " bytes of text\n");
}

void Emitter::startBasicBlock(MachineBasicBlock &BB) {
  BBLocations[BB.getBasicBlock()] = (unsigned)CurByte;
}


void Emitter::emitByte(unsigned char B) {
  *CurByte++ = B;   // Write the byte to memory
}


// emitPCRelativeDisp - For functions, just output a displacement that will
// cause a reference to the zero page, which will cause a seg-fault, causing
// things to get resolved on demand.  Keep track of these markers.
//
// For basic block references, keep track of where the references are so they
// may be patched up when the basic block is defined.
//
void Emitter::emitPCRelativeDisp(Value *V) {
  if (Function *F = dyn_cast<Function>(V)) {
    TheVM.addFunctionRef(CurByte, F);
    unsigned ZeroAddr = -(unsigned)CurByte-4; // Calculate displacement to null
    *(unsigned*)CurByte = ZeroAddr;           // 4 byte offset
    CurByte += 4;
  } else {
    BasicBlock *BB = cast<BasicBlock>(V);     // Keep track of reference...
    BBRefs.push_back(std::make_pair(BB, (unsigned*)CurByte));
    CurByte += 4;
  }
}

void Emitter::emitGlobalAddress(GlobalValue *V) {
  *(void**)CurByte = TheVM.getPointerToGlobal(V);
  CurByte += 4;
}
