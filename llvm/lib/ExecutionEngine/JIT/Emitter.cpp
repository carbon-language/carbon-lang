//===-- Emitter.cpp - Write machine code to executable memory -------------===//
//
// This file defines a MachineCodeEmitter object that is used by Jello to write
// machine code to memory and remember where relocatable values lie.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#ifndef _POSIX_MAPPED_FILES
#define _POSIX_MAPPED_FILES
#endif
#include "VM.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Module.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Config/unistd.h"
#include "Config/sys/mman.h"

namespace {
  Statistic<> NumBytes("jello", "Number of bytes of machine code compiled");
  VM *TheVM = 0;

  /// JITMemoryManager - Manage memory for the JIT code generation in a logical,
  /// sane way.  This splits a large block of MAP_NORESERVE'd memory into two
  /// sections, one for function stubs, one for the functions themselves.  We
  /// have to do this because we may need to emit a function stub while in the
  /// middle of emitting a function, and we don't know how large the function we
  /// are emitting is.  This never bothers to release the memory, because when
  /// we are ready to destroy the JIT, the program exits.
  class JITMemoryManager {
    unsigned char *MemBase;      // Base of block of memory, start of stub mem
    unsigned char *FunctionBase; // Start of the function body area
    unsigned char *CurStubPtr, *CurFunctionPtr;
  public:
    JITMemoryManager();
    
    inline unsigned char *allocateStub(unsigned StubSize);
    inline unsigned char *startFunctionBody();
    inline void endFunctionBody(unsigned char *FunctionEnd);    
  };
}

// getMemory - Return a pointer to the specified number of bytes, which is
// mapped as executable readable and writable.
static void *getMemory(unsigned NumBytes) {
  if (NumBytes == 0) return 0;
  static const long pageSize = sysconf(_SC_PAGESIZE);
  unsigned NumPages = (NumBytes+pageSize-1)/pageSize;

#if defined(i386) || defined(__i386__) || defined(__x86__)
  /* Linux and *BSD tend to have these flags named differently. */
#if defined(MAP_ANON) && !defined(MAP_ANONYMOUS)
# define MAP_ANONYMOUS MAP_ANON
#endif /* defined(MAP_ANON) && !defined(MAP_ANONYMOUS) */
#define fd  0
#elif defined(sparc) || defined(__sparc__) || defined(__sparcv9)
#define fd -1
#else
  std::cerr << "This architecture is not supported by the JIT!\n";
  abort();
#endif
  
  unsigned mmapFlags = MAP_PRIVATE|MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
  mmapFlags |= MAP_NORESERVE;
#endif

  void *pa = mmap(0, pageSize*NumPages, PROT_READ|PROT_WRITE|PROT_EXEC,
                  mmapFlags, fd, 0);
  if (pa == MAP_FAILED) {
    perror("mmap");
    abort();
  }
  return pa;
}

JITMemoryManager::JITMemoryManager() {
  // Allocate a 16M block of memory...
  MemBase = (unsigned char*)getMemory(16 << 20);
  FunctionBase = MemBase + 512*1024; // Use 512k for stubs

  // Allocate stubs backwards from the function base, allocate functions forward
  // from the function base.
  CurStubPtr = CurFunctionPtr = FunctionBase;
}

unsigned char *JITMemoryManager::allocateStub(unsigned StubSize) {
  CurStubPtr -= StubSize;
  if (CurStubPtr < MemBase) {
    std::cerr << "JIT ran out of memory for function stubs!\n";
    abort();
  }
  return CurStubPtr;
}

unsigned char *JITMemoryManager::startFunctionBody() {
  // Round up to an even multiple of 4 bytes, this should eventually be target
  // specific.
  return (unsigned char*)(((intptr_t)CurFunctionPtr + 3) & ~3);
}

void JITMemoryManager::endFunctionBody(unsigned char *FunctionEnd) {
  assert(FunctionEnd > CurFunctionPtr);
  CurFunctionPtr = FunctionEnd;
}



namespace {
  /// Emitter - The JIT implementation of the MachineCodeEmiter, which is used
  /// to output functions to memory for execution.
  class Emitter : public MachineCodeEmitter {
    JITMemoryManager MemMgr;

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

MachineCodeEmitter *VM::createEmitter(VM &V) {
  return new Emitter(V);
}

void Emitter::startFunction(MachineFunction &F) {
  CurByte = CurBlock = MemMgr.startFunctionBody();
  TheVM->addGlobalMapping(F.getFunction(), CurBlock);
}

void Emitter::finishFunction(MachineFunction &F) {
  MemMgr.endFunctionBody(CurByte);
  ConstantPoolAddresses.clear();
  NumBytes += CurByte-CurBlock;

  DEBUG(std::cerr << "Finished CodeGen of [" << (void*)CurBlock
                  << "] Function: " << F.getFunction()->getName()
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
  CurByte = CurBlock = MemMgr.allocateStub(StubSize);
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
  // This won't work if the endianness of the host and target don't agree!  (For
  // a JIT this can't happen though.  :)
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

// getPointerToNamedFunction - This function is used as a global wrapper to
// VM::getPointerToNamedFunction for the purpose of resolving symbols when
// bugpoint is debugging the JIT. In that scenario, we are loading an .so and
// need to resolve function(s) that are being mis-codegenerated, so we need to
// resolve their addresses at runtime, and this is the way to do it.
extern "C" {
  void *getPointerToNamedFunction(const char *Name) {
    Module &M = TheVM->getModule();
    if (Function *F = M.getNamedFunction(Name))
      return TheVM->getPointerToFunction(F);
    return TheVM->getPointerToNamedFunction(Name);
  }
}
