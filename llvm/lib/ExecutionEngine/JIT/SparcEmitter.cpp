//===-- SparcEmitter.cpp - Write machine code to executable memory --------===//
//
// This file defines a MachineCodeEmitter object that is used by Jello to write
// machine code to memory and remember where relocatable values lie.
//
//===----------------------------------------------------------------------===//

#include "VM.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Function.h"
#include "Support/Statistic.h"
// FIXME
#include "../../../lib/Target/Sparc/SparcV9CodeEmitter.h"

namespace {
  Statistic<> NumBytes("jello", "Number of bytes of machine code compiled");

  class SparcEmitter : public MachineCodeEmitter {
    VM &TheVM;

    unsigned char *CurBlock, *CurByte;

    // When outputting a function stub in the context of some other function, we
    // save CurBlock and CurByte here.
    unsigned char *SavedCurBlock, *SavedCurByte;
    
    std::vector<std::pair<BasicBlock*,
                          std::pair<unsigned*,MachineInstr*> > > BBRefs;
    std::map<BasicBlock*, unsigned> BBLocations;
    std::vector<void*> ConstantPoolAddresses;
    std::vector<void*> funcMemory;
  public:
    SparcEmitter(VM &vm) : TheVM(vm) {}
    ~SparcEmitter() {
      while (! funcMemory.empty()) {
        void* addr = funcMemory.back();
        free(addr);
        funcMemory.pop_back();
      }
    }

    virtual void startFunction(MachineFunction &F);
    virtual void finishFunction(MachineFunction &F);
    virtual void emitConstantPool(MachineConstantPool *MCP);
    virtual void startBasicBlock(MachineBasicBlock &BB);
    virtual void startFunctionStub(const Function &F, unsigned StubSize);
    virtual void* finishFunctionStub(const Function &F);
    virtual void emitByte(unsigned char B);
    virtual void emitPCRelativeDisp(Value *V);
    virtual void emitGlobalAddress(GlobalValue *V, bool isPCRelative);
    virtual void emitGlobalAddress(const std::string &Name, bool isPCRelative);
    virtual void emitFunctionConstantValueAddress(unsigned ConstantNum,
						  int Offset);

    virtual void saveBBreference(BasicBlock *BB, MachineInstr &MI);
    

  private:
    void emitAddress(void *Addr, bool isPCRelative);
    void* getMemory(unsigned NumPages);
  };
}

MachineCodeEmitter *VM::createSparcEmitter(VM &V) {
  return new SparcEmitter(V);
}


#define _POSIX_MAPPED_FILES
#include <unistd.h>
#include <sys/mman.h>

// FIXME: This should be rewritten to support a real memory manager for
// executable memory pages!
void * SparcEmitter::getMemory(unsigned NumPages) {
  void *pa;
  if (NumPages == 0) return 0;
  static const long pageSize = sysconf (_SC_PAGESIZE);
  pa = mmap(0, pageSize*NumPages, PROT_READ|PROT_WRITE|PROT_EXEC,
                  MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if (pa == MAP_FAILED) {
    perror("mmap");
    abort();
  }
  return pa;
}


void SparcEmitter::startFunction(MachineFunction &F) {
  CurBlock = (unsigned char *)getMemory(8);
  std::cerr << "Starting function " << F.getFunction()->getName() << "\n";
  CurByte = CurBlock;  // Start writing at the beginning of the fn.
  TheVM.addGlobalMapping(F.getFunction(), CurBlock);
}

void SparcEmitter::finishFunction(MachineFunction &F) {
  ConstantPoolAddresses.clear();
  // Re-write branches to BasicBlocks for the entire function
  for (unsigned i = 0, e = BBRefs.size(); i != e; ++i) {
    unsigned Location = BBLocations[BBRefs[i].first];
    unsigned *Ref = BBRefs[i].second.first;
    MachineInstr *MI = BBRefs[i].second.second;
    for (unsigned i=0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &op = MI->getOperand(i);
      if (op.isImmediate()) {
        MI->SetMachineOperandConst(i, op.getType(), Location);
        break;
      }
    }
    unsigned fixedInstr = SparcV9CodeEmitter::getBinaryCodeForInstr(*MI);
    *Ref = fixedInstr;
  }
  BBRefs.clear();
  BBLocations.clear();

  NumBytes += CurByte-CurBlock;

  DEBUG(std::cerr << "Finished CodeGen of [0x" << std::hex
                  << (unsigned)(intptr_t)CurBlock
                  << std::dec << "] Function: " << F.getFunction()->getName()
                  << ": " << CurByte-CurBlock << " bytes of text\n");
}

void SparcEmitter::emitConstantPool(MachineConstantPool *MCP) {
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


void SparcEmitter::startBasicBlock(MachineBasicBlock &BB) {
  BBLocations[BB.getBasicBlock()] = (unsigned)(intptr_t)CurByte;
}


void SparcEmitter::startFunctionStub(const Function &F, unsigned StubSize) {
  SavedCurBlock = CurBlock;  SavedCurByte = CurByte;
  // FIXME: this is a huge waste of memory.
  CurBlock = (unsigned char *)getMemory((StubSize+4095)/4096);
  CurByte = CurBlock;  // Start writing at the beginning of the fn.
}

void *SparcEmitter::finishFunctionStub(const Function &F) {
  NumBytes += CurByte-CurBlock;
  DEBUG(std::cerr << "Finished CodeGen of [0x" << std::hex
                  << (unsigned)(intptr_t)CurBlock
                  << std::dec << "] Function stub for: " << F.getName()
                  << ": " << CurByte-CurBlock << " bytes of text\n");
  std::swap(CurBlock, SavedCurBlock);
  CurByte = SavedCurByte;
  return SavedCurBlock;
}

void SparcEmitter::emitByte(unsigned char B) {
  *CurByte++ = B;   // Write the byte to memory
}

// BasicBlock -> pair<memloc, MachineInstr>
// when the BB is emitted, machineinstr is modified with then-currbyte, 
// processed with MCE, and written out at memloc.
// Should be called by the emitter if its outputting a PCRelative disp
void SparcEmitter::saveBBreference(BasicBlock *BB, MachineInstr &MI) {
  BBRefs.push_back(std::make_pair(BB, std::make_pair((unsigned*)CurByte, &MI)));
}


// emitPCRelativeDisp - For functions, just output a displacement that will
// cause a reference to the zero page, which will cause a seg-fault, causing
// things to get resolved on demand.  Keep track of these markers.
//
// For basic block references, keep track of where the references are so they
// may be patched up when the basic block is defined.
//
// BasicBlock -> pair<memloc, MachineInstr>
// when the BB is emitted, machineinstr is modified with then-currbyte, 
// processed with MCE, and written out at memloc.

void SparcEmitter::emitPCRelativeDisp(Value *V) {
#if 0
  BasicBlock *BB = cast<BasicBlock>(V);     // Keep track of reference...
  BBRefs.push_back(std::make_pair(BB, (unsigned*)CurByte));
  CurByte += 4;
#endif
}

// emitAddress - Emit an address in either direct or PCRelative form...
//
void SparcEmitter::emitAddress(void *Addr, bool isPCRelative) {
#if 0
  if (isPCRelative) {
    *(intptr_t*)CurByte = (intptr_t)Addr - (intptr_t)CurByte-4;
  } else {
    *(void**)CurByte = Addr;
  }
  CurByte += 4;
#endif
}

void SparcEmitter::emitGlobalAddress(GlobalValue *V, bool isPCRelative) {
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

void SparcEmitter::emitGlobalAddress(const std::string &Name, bool isPCRelative)
{
#if 0
  emitAddress(TheVM.getPointerToNamedFunction(Name), isPCRelative);
#endif
}

void SparcEmitter::emitFunctionConstantValueAddress(unsigned ConstantNum,
					       int Offset) {
  assert(ConstantNum < ConstantPoolAddresses.size() &&
	 "Invalid ConstantPoolIndex!");
  *(void**)CurByte = (char*)ConstantPoolAddresses[ConstantNum]+Offset;
  CurByte += 4;
}
