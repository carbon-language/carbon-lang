//===-- Emitter.cpp - Write machine code to executable memory -------------===//
//
// This file defines a MachineCodeEmitter object that is used by Jello to write
// machine code to memory and remember where relocatable values lie.
//
//===----------------------------------------------------------------------===//

#include "VM.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Function.h"
#include "Support/Statistic.h"

namespace {
  Statistic<> NumBytes("jello", "Number of bytes of machine code compiled");

  class Emitter : public MachineCodeEmitter {
    VM &TheVM;

    unsigned char *CurBlock;
    unsigned char *CurByte;
    
    std::vector<std::pair<BasicBlock*, unsigned *> > BBRefs;
    std::map<BasicBlock*, unsigned> BBLocations;
    std::vector<void*> ConstantPoolAddresses;
  public:
    Emitter(VM &vm) : TheVM(vm) {}

    virtual void startFunction(MachineFunction &F);
    virtual void finishFunction(MachineFunction &F);
    virtual void emitConstantPool(MachineConstantPool *MCP);
    virtual void startBasicBlock(MachineBasicBlock &BB);
    virtual void emitByte(unsigned char B);
    virtual void emitPCRelativeDisp(Value *V);
    virtual void emitGlobalAddress(GlobalValue *V, bool isPCRelative);
    virtual void emitGlobalAddress(const std::string &Name, bool isPCRelative);
    virtual void emitFunctionConstantValueAddress(unsigned ConstantNum,
						  int Offset);
  private:
    void emitAddress(void *Addr, bool isPCRelative);
  };
}

MachineCodeEmitter *VM::createEmitter(VM &V) {
  return new Emitter(V);
}


#define _POSIX_MAPPED_FILES
#include <unistd.h>
#include <sys/mman.h>

static void *getMemory() {
  return mmap(0, 4096*8, PROT_READ|PROT_WRITE|PROT_EXEC,
              MAP_PRIVATE|MAP_ANONYMOUS, 0, 0);
}


void Emitter::startFunction(MachineFunction &F) {
  CurBlock = (unsigned char *)getMemory();
  CurByte = CurBlock;  // Start writing at the beginning of the fn.
  TheVM.addGlobalMapping(F.getFunction(), CurBlock);
}

void Emitter::finishFunction(MachineFunction &F) {
  ConstantPoolAddresses.clear();
  for (unsigned i = 0, e = BBRefs.size(); i != e; ++i) {
    unsigned Location = BBLocations[BBRefs[i].first];
    unsigned *Ref = BBRefs[i].second;
    *Ref = Location-(unsigned)(intptr_t)Ref-4;
  }
  BBRefs.clear();
  BBLocations.clear();

  NumBytes += CurByte-CurBlock;

  DEBUG(std::cerr << "Finished CodeGen of [0x" << std::hex
                  << (unsigned)(intptr_t)CurBlock
                  << std::dec << "] Function: " << F.getFunction()->getName()
                  << ": " << CurByte-CurBlock << " bytes of text\n");
}

void Emitter::emitConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &Constants = MCP->getConstants();
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    // For now we just allocate some memory on the heap, this can be
    // dramatically improved.
    const Type *Ty = ((Value*)Constants[i])->getType();
    void *Addr = malloc(TheVM.getTargetData().getTypeSize(Ty));
    TheVM.InitializeMemory(Constants[i], Addr);
    ConstantPoolAddresses.push_back(Addr);
  }
}


void Emitter::startBasicBlock(MachineBasicBlock &BB) {
  BBLocations[BB.getBasicBlock()] = (unsigned)(intptr_t)CurByte;
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
  BasicBlock *BB = cast<BasicBlock>(V);     // Keep track of reference...
  BBRefs.push_back(std::make_pair(BB, (unsigned*)CurByte));
  CurByte += 4;
}

// emitAddress - Emit an address in either direct or PCRelative form...
//
void Emitter::emitAddress(void *Addr, bool isPCRelative) {
  if (isPCRelative) {
    *(intptr_t*)CurByte = (intptr_t)Addr - (intptr_t)CurByte-4;
  } else {
    *(void**)CurByte = Addr;
  }
  CurByte += 4;
}

void Emitter::emitGlobalAddress(GlobalValue *V, bool isPCRelative) {
  if (isPCRelative) { // must be a call, this is a major hack!
    // Try looking up the function to see if it is already compiled!
    if (void *Addr = TheVM.getPointerToGlobalIfAvailable(V)) {
      emitAddress(Addr, isPCRelative);
    } else {  // Function has not yet been code generated!
      TheVM.addFunctionRef(CurByte, cast<Function>(V));

      // Delayed resolution...
      emitAddress((void*)VM::CompilationCallback, isPCRelative);
    }
  } else {
    emitAddress(TheVM.getPointerToGlobal(V), isPCRelative);
  }
}

void Emitter::emitGlobalAddress(const std::string &Name, bool isPCRelative) {
  emitAddress(TheVM.getPointerToNamedFunction(Name), isPCRelative);
}

void Emitter::emitFunctionConstantValueAddress(unsigned ConstantNum,
					       int Offset) {
  assert(ConstantNum < ConstantPoolAddresses.size() &&
	 "Invalid ConstantPoolIndex!");
  *(void**)CurByte = (char*)ConstantPoolAddresses[ConstantNum]+Offset;
  CurByte += 4;
}
