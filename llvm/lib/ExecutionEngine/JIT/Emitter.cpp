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

static VM *TheVM = 0;

namespace {
  Statistic<> NumBytes("jello", "Number of bytes of machine code compiled");

  class Emitter : public MachineCodeEmitter {
    // CurBlock - The start of the current block of memory.  CurByte - The
    // current byte being emitted to.
    unsigned char *CurBlock, *CurByte;

    // When outputting a function stub in the context of some other function, we
    // save CurBlock and CurByte here.
    unsigned char *SavedCurBlock, *SavedCurByte;

    // ConstantPoolAddresses - Contains the location for each entry in the
    // constant pool.
    std::vector<void*> ConstantPoolAddresses;
  public:
    Emitter(VM &vm) { TheVM = &vm; }

    virtual void startFunction(MachineFunction &F);
    virtual void finishFunction(MachineFunction &F);
    virtual void emitConstantPool(MachineConstantPool *MCP);
    virtual void startFunctionStub(const Function &F, unsigned StubSize);
    virtual void* finishFunctionStub(const Function &F);
    virtual void emitByte(unsigned char B);
    virtual void emitWord(unsigned W);

    virtual uint64_t getGlobalValueAddress(GlobalValue *V);
    virtual uint64_t getGlobalValueAddress(const std::string &Name);
    virtual uint64_t getConstantPoolEntryAddress(unsigned Entry);
    virtual uint64_t getCurrentPCValue();

    // forceCompilationOf - Force the compilation of the specified function, and
    // return its address, because we REALLY need the address now.
    //
    // FIXME: This is JIT specific!
    //
    virtual uint64_t forceCompilationOf(Function *F);
  };
}

MachineCodeEmitter *VM::createX86Emitter(VM &V) {
  return new Emitter(V);
}


#define _POSIX_MAPPED_FILES
#include <unistd.h>
#include <sys/mman.h>

// FIXME: This should be rewritten to support a real memory manager for
// executable memory pages!
static void *getMemory(unsigned NumPages) {
  return mmap(0, 4096*NumPages, PROT_READ|PROT_WRITE|PROT_EXEC,
              MAP_PRIVATE|MAP_ANONYMOUS, 0, 0);
}


void Emitter::startFunction(MachineFunction &F) {
  CurBlock = (unsigned char *)getMemory(16);
  CurByte = CurBlock;  // Start writing at the beginning of the fn.
  TheVM->addGlobalMapping(F.getFunction(), CurBlock);
}

void Emitter::finishFunction(MachineFunction &F) {
  ConstantPoolAddresses.clear();
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
    void *Addr = malloc(TheVM->getTargetData().getTypeSize(Ty));
    TheVM->InitializeMemory(Constants[i], Addr);
    ConstantPoolAddresses.push_back(Addr);
  }
}

void Emitter::startFunctionStub(const Function &F, unsigned StubSize) {
  SavedCurBlock = CurBlock;  SavedCurByte = CurByte;
  // FIXME: this is a huge waste of memory.
  CurBlock = (unsigned char *)getMemory((StubSize+4095)/4096);
  CurByte = CurBlock;  // Start writing at the beginning of the fn.
}

void *Emitter::finishFunctionStub(const Function &F) {
  NumBytes += CurByte-CurBlock;
  DEBUG(std::cerr << "Finished CodeGen of [0x" << std::hex
                  << (unsigned)(intptr_t)CurBlock
                  << std::dec << "] Function stub for: " << F.getName()
                  << ": " << CurByte-CurBlock << " bytes of text\n");
  std::swap(CurBlock, SavedCurBlock);
  CurByte = SavedCurByte;
  return SavedCurBlock;
}

void Emitter::emitByte(unsigned char B) {
  *CurByte++ = B;   // Write the byte to memory
}

void Emitter::emitWord(unsigned W) {
  // FIXME: This won't work if the endianness of the host and target don't
  // agree!  (For a JIT this can't happen though.  :)
  *(unsigned*)CurByte = W;
  CurByte += sizeof(unsigned);
}


uint64_t Emitter::getGlobalValueAddress(GlobalValue *V) {
  // Try looking up the function to see if it is already compiled, if not return
  // 0.
  return (intptr_t)TheVM->getPointerToGlobalIfAvailable(V);
}
uint64_t Emitter::getGlobalValueAddress(const std::string &Name) {
  return (intptr_t)TheVM->getPointerToNamedFunction(Name);
}

// getConstantPoolEntryAddress - Return the address of the 'ConstantNum' entry
// in the constant pool that was last emitted with the 'emitConstantPool'
// method.
//
uint64_t Emitter::getConstantPoolEntryAddress(unsigned ConstantNum) {
  assert(ConstantNum < ConstantPoolAddresses.size() &&
	 "Invalid ConstantPoolIndex!");
  return (intptr_t)ConstantPoolAddresses[ConstantNum];
}

// getCurrentPCValue - This returns the address that the next emitted byte
// will be output to.
//
uint64_t Emitter::getCurrentPCValue() {
  return (intptr_t)CurByte;
}

uint64_t Emitter::forceCompilationOf(Function *F) {
  return (intptr_t)TheVM->getPointerToFunction(F);
}

#if 0


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
    if (void *Addr = TheVM->getPointerToGlobalIfAvailable(V)) {
      emitAddress(Addr, isPCRelative);
    } else {  // Function has not yet been code generated!
      TheVM->addFunctionRef(CurByte, cast<Function>(V));

      // Delayed resolution...
      emitAddress((void*)VM::CompilationCallback, isPCRelative);
    }
  } else {
    emitAddress(TheVM->getPointerToGlobal(V), isPCRelative);
  }
}

void Emitter::emitFunctionConstantValueAddress(unsigned ConstantNum,
					       int Offset) {
  assert(ConstantNum < ConstantPoolAddresses.size() &&
	 "Invalid ConstantPoolIndex!");
  *(void**)CurByte = (char*)ConstantPoolAddresses[ConstantNum]+Offset;
  CurByte += 4;
}
#endif
