//===-- JITEmitter.cpp - Write machine code to executable memory ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a MachineCodeEmitter object that is used by the JIT to
// write machine code to memory and remember where relocatable values are.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "JIT.h"
#include "llvm/Constant.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineRelocation.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetJITInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/System/Memory.h"
#include <algorithm>
#include <iostream>
#include <list>
using namespace llvm;

namespace {
  Statistic<> NumBytes("jit", "Number of bytes of machine code compiled");
  Statistic<> NumRelos("jit", "Number of relocations applied");
  JIT *TheJIT = 0;
}


//===----------------------------------------------------------------------===//
// JITMemoryManager code.
//
namespace {
  /// JITMemoryManager - Manage memory for the JIT code generation in a logical,
  /// sane way.  This splits a large block of MAP_NORESERVE'd memory into two
  /// sections, one for function stubs, one for the functions themselves.  We
  /// have to do this because we may need to emit a function stub while in the
  /// middle of emitting a function, and we don't know how large the function we
  /// are emitting is.  This never bothers to release the memory, because when
  /// we are ready to destroy the JIT, the program exits.
  class JITMemoryManager {
    std::list<sys::MemoryBlock> Blocks; // List of blocks allocated by the JIT
    unsigned char *FunctionBase; // Start of the function body area
    unsigned char *GlobalBase; // Start of the Global area
    unsigned char *ConstantBase; // Memory allocated for constant pools
    unsigned char *CurStubPtr, *CurFunctionPtr, *CurConstantPtr, *CurGlobalPtr;
    unsigned char *GOTBase; //Target Specific reserved memory

    // centralize memory block allocation
    sys::MemoryBlock getNewMemoryBlock(unsigned size);
  public:
    JITMemoryManager(bool useGOT);
    ~JITMemoryManager();

    inline unsigned char *allocateStub(unsigned StubSize);
    inline unsigned char *allocateConstant(unsigned ConstantSize,
                                           unsigned Alignment);
    inline unsigned char* allocateGlobal(unsigned Size,
                                         unsigned Alignment);
    inline unsigned char *startFunctionBody();
    inline void endFunctionBody(unsigned char *FunctionEnd);
    inline unsigned char* getGOTBase() const;

    inline bool isManagingGOT() const;
  };
}

JITMemoryManager::JITMemoryManager(bool useGOT) {
  // Allocate a 16M block of memory for functions
  sys::MemoryBlock FunBlock = getNewMemoryBlock(16 << 20);
  // Allocate a 1M block of memory for Constants
  sys::MemoryBlock ConstBlock = getNewMemoryBlock(1 << 20);
  // Allocate a 1M Block of memory for Globals
  sys::MemoryBlock GVBlock = getNewMemoryBlock(1 << 20);

  Blocks.push_front(FunBlock);
  Blocks.push_front(ConstBlock);
  Blocks.push_front(GVBlock);

  FunctionBase = reinterpret_cast<unsigned char*>(FunBlock.base());
  ConstantBase = reinterpret_cast<unsigned char*>(ConstBlock.base());
  GlobalBase = reinterpret_cast<unsigned char*>(GVBlock.base());

  // Allocate stubs backwards from the base, allocate functions forward
  // from the base.
  CurStubPtr = CurFunctionPtr = FunctionBase + 512*1024;// Use 512k for stubs

  CurConstantPtr = ConstantBase + ConstBlock.size();
  CurGlobalPtr = GlobalBase + GVBlock.size();

  //Allocate the GOT just like a global array
  GOTBase = NULL;
  if (useGOT)
    GOTBase = allocateGlobal(sizeof(void*) * 8192, 8);
}

JITMemoryManager::~JITMemoryManager() {
  for (std::list<sys::MemoryBlock>::iterator ib = Blocks.begin(),
       ie = Blocks.end(); ib != ie; ++ib)
    sys::Memory::ReleaseRWX(*ib);
  Blocks.clear();
}

unsigned char *JITMemoryManager::allocateStub(unsigned StubSize) {
  CurStubPtr -= StubSize;
  if (CurStubPtr < FunctionBase) {
    //FIXME: allocate a new block
    std::cerr << "JIT ran out of memory for function stubs!\n";
    abort();
  }
  return CurStubPtr;
}

unsigned char *JITMemoryManager::allocateConstant(unsigned ConstantSize,
                                                  unsigned Alignment) {
  // Reserve space and align pointer.
  CurConstantPtr -= ConstantSize;
  CurConstantPtr =
    (unsigned char *)((intptr_t)CurConstantPtr & ~((intptr_t)Alignment - 1));

  if (CurConstantPtr < ConstantBase) {
    //Either allocate another MB or 2xConstantSize
    sys::MemoryBlock ConstBlock = getNewMemoryBlock(2 * ConstantSize);
    ConstantBase = reinterpret_cast<unsigned char*>(ConstBlock.base());
    CurConstantPtr = ConstantBase + ConstBlock.size();
    return allocateConstant(ConstantSize, Alignment);
  }
  return CurConstantPtr;
}

unsigned char *JITMemoryManager::allocateGlobal(unsigned Size,
                                                unsigned Alignment) {
 // Reserve space and align pointer.
  CurGlobalPtr -= Size;
  CurGlobalPtr =
    (unsigned char *)((intptr_t)CurGlobalPtr & ~((intptr_t)Alignment - 1));

  if (CurGlobalPtr < GlobalBase) {
    //Either allocate another MB or 2xSize
    sys::MemoryBlock GVBlock =  getNewMemoryBlock(2 * Size);
    GlobalBase = reinterpret_cast<unsigned char*>(GVBlock.base());
    CurGlobalPtr = GlobalBase + GVBlock.size();
    return allocateGlobal(Size, Alignment);
  }
  return CurGlobalPtr;
}

unsigned char *JITMemoryManager::startFunctionBody() {
  // Round up to an even multiple of 8 bytes, this should eventually be target
  // specific.
  return (unsigned char*)(((intptr_t)CurFunctionPtr + 7) & ~7);
}

void JITMemoryManager::endFunctionBody(unsigned char *FunctionEnd) {
  assert(FunctionEnd > CurFunctionPtr);
  CurFunctionPtr = FunctionEnd;
}

unsigned char* JITMemoryManager::getGOTBase() const {
  return GOTBase;
}

bool JITMemoryManager::isManagingGOT() const {
  return GOTBase != NULL;
}

sys::MemoryBlock JITMemoryManager::getNewMemoryBlock(unsigned size) {
  const sys::MemoryBlock* BOld = 0;
  if (Blocks.size())
    BOld = &Blocks.front();
  //never allocate less than 1 MB
  sys::MemoryBlock B;
  try {
    B = sys::Memory::AllocateRWX(std::max(((unsigned)1 << 20), size), BOld);
  } catch (std::string& err) {
    std::cerr << "Allocation failed when allocating new memory in the JIT\n";
    std::cerr << err << "\n";
    abort();
  }
  Blocks.push_front(B);
  return B;
}

//===----------------------------------------------------------------------===//
// JIT lazy compilation code.
//
namespace {
  class JITResolverState {
  private:
    /// FunctionToStubMap - Keep track of the stub created for a particular
    /// function so that we can reuse them if necessary.
    std::map<Function*, void*> FunctionToStubMap;

    /// StubToFunctionMap - Keep track of the function that each stub
    /// corresponds to.
    std::map<void*, Function*> StubToFunctionMap;

  public:
    std::map<Function*, void*>& getFunctionToStubMap(const MutexGuard& locked) {
      assert(locked.holds(TheJIT->lock));
      return FunctionToStubMap;
    }

    std::map<void*, Function*>& getStubToFunctionMap(const MutexGuard& locked) {
      assert(locked.holds(TheJIT->lock));
      return StubToFunctionMap;
    }
  };

  /// JITResolver - Keep track of, and resolve, call sites for functions that
  /// have not yet been compiled.
  class JITResolver {
    /// MCE - The MachineCodeEmitter to use to emit stubs with.
    MachineCodeEmitter &MCE;

    /// LazyResolverFn - The target lazy resolver function that we actually
    /// rewrite instructions to use.
    TargetJITInfo::LazyResolverFn LazyResolverFn;

    JITResolverState state;

    /// ExternalFnToStubMap - This is the equivalent of FunctionToStubMap for
    /// external functions.
    std::map<void*, void*> ExternalFnToStubMap;

    //map addresses to indexes in the GOT
    std::map<void*, unsigned> revGOTMap;
    unsigned nextGOTIndex;

  public:
    JITResolver(MachineCodeEmitter &mce) : MCE(mce), nextGOTIndex(0) {
      LazyResolverFn =
        TheJIT->getJITInfo().getLazyResolverFunction(JITCompilerFn);
    }

    /// getFunctionStub - This returns a pointer to a function stub, creating
    /// one on demand as needed.
    void *getFunctionStub(Function *F);

    /// getExternalFunctionStub - Return a stub for the function at the
    /// specified address, created lazily on demand.
    void *getExternalFunctionStub(void *FnAddr);

    /// AddCallbackAtLocation - If the target is capable of rewriting an
    /// instruction without the use of a stub, record the location of the use so
    /// we know which function is being used at the location.
    void *AddCallbackAtLocation(Function *F, void *Location) {
      MutexGuard locked(TheJIT->lock);
      /// Get the target-specific JIT resolver function.
      state.getStubToFunctionMap(locked)[Location] = F;
      return (void*)LazyResolverFn;
    }

    /// getGOTIndexForAddress - Return a new or existing index in the GOT for
    /// and address.  This function only manages slots, it does not manage the
    /// contents of the slots or the memory associated with the GOT.
    unsigned getGOTIndexForAddr(void* addr);

    /// JITCompilerFn - This function is called to resolve a stub to a compiled
    /// address.  If the LLVM Function corresponding to the stub has not yet
    /// been compiled, this function compiles it first.
    static void *JITCompilerFn(void *Stub);
  };
}

/// getJITResolver - This function returns the one instance of the JIT resolver.
///
static JITResolver &getJITResolver(MachineCodeEmitter *MCE = 0) {
  static JITResolver TheJITResolver(*MCE);
  return TheJITResolver;
}

/// getFunctionStub - This returns a pointer to a function stub, creating
/// one on demand as needed.
void *JITResolver::getFunctionStub(Function *F) {
  MutexGuard locked(TheJIT->lock);

  // If we already have a stub for this function, recycle it.
  void *&Stub = state.getFunctionToStubMap(locked)[F];
  if (Stub) return Stub;

  // Call the lazy resolver function unless we already KNOW it is an external
  // function, in which case we just skip the lazy resolution step.
  void *Actual = (void*)LazyResolverFn;
  if (F->isExternal() && F->hasExternalLinkage())
    Actual = TheJIT->getPointerToFunction(F);

  // Otherwise, codegen a new stub.  For now, the stub will call the lazy
  // resolver function.
  Stub = TheJIT->getJITInfo().emitFunctionStub(Actual, MCE);

  if (Actual != (void*)LazyResolverFn) {
    // If we are getting the stub for an external function, we really want the
    // address of the stub in the GlobalAddressMap for the JIT, not the address
    // of the external function.
    TheJIT->updateGlobalMapping(F, Stub);
  }

  DEBUG(std::cerr << "JIT: Stub emitted at [" << Stub << "] for function '"
                  << F->getName() << "'\n");

  // Finally, keep track of the stub-to-Function mapping so that the
  // JITCompilerFn knows which function to compile!
  state.getStubToFunctionMap(locked)[Stub] = F;
  return Stub;
}

/// getExternalFunctionStub - Return a stub for the function at the
/// specified address, created lazily on demand.
void *JITResolver::getExternalFunctionStub(void *FnAddr) {
  // If we already have a stub for this function, recycle it.
  void *&Stub = ExternalFnToStubMap[FnAddr];
  if (Stub) return Stub;

  Stub = TheJIT->getJITInfo().emitFunctionStub(FnAddr, MCE);
  DEBUG(std::cerr << "JIT: Stub emitted at [" << Stub
        << "] for external function at '" << FnAddr << "'\n");
  return Stub;
}

unsigned JITResolver::getGOTIndexForAddr(void* addr) {
  unsigned idx = revGOTMap[addr];
  if (!idx) {
    idx = ++nextGOTIndex;
    revGOTMap[addr] = idx;
    DEBUG(std::cerr << "Adding GOT entry " << idx
          << " for addr " << addr << "\n");
    //    ((void**)MemMgr.getGOTBase())[idx] = addr;
  }
  return idx;
}

/// JITCompilerFn - This function is called when a lazy compilation stub has
/// been entered.  It looks up which function this stub corresponds to, compiles
/// it if necessary, then returns the resultant function pointer.
void *JITResolver::JITCompilerFn(void *Stub) {
  JITResolver &JR = getJITResolver();

  MutexGuard locked(TheJIT->lock);

  // The address given to us for the stub may not be exactly right, it might be
  // a little bit after the stub.  As such, use upper_bound to find it.
  std::map<void*, Function*>::iterator I =
    JR.state.getStubToFunctionMap(locked).upper_bound(Stub);
  assert(I != JR.state.getStubToFunctionMap(locked).begin() &&
         "This is not a known stub!");
  Function *F = (--I)->second;

  // We might like to remove the stub from the StubToFunction map.
  // We can't do that! Multiple threads could be stuck, waiting to acquire the
  // lock above. As soon as the 1st function finishes compiling the function,
  // the next one will be released, and needs to be able to find the function it
  // needs to call.
  //JR.state.getStubToFunctionMap(locked).erase(I);

  DEBUG(std::cerr << "JIT: Lazily resolving function '" << F->getName()
                  << "' In stub ptr = " << Stub << " actual ptr = "
                  << I->first << "\n");

  void *Result = TheJIT->getPointerToFunction(F);

  // We don't need to reuse this stub in the future, as F is now compiled.
  JR.state.getFunctionToStubMap(locked).erase(F);

  // FIXME: We could rewrite all references to this stub if we knew them.

  // What we will do is set the compiled function address to map to the
  // same GOT entry as the stub so that later clients may update the GOT
  // if they see it still using the stub address.
  // Note: this is done so the Resolver doesn't have to manage GOT memory
  // Do this without allocating map space if the target isn't using a GOT
  if(JR.revGOTMap.find(Stub) != JR.revGOTMap.end())
    JR.revGOTMap[Result] = JR.revGOTMap[Stub];

  return Result;
}


// getPointerToFunctionOrStub - If the specified function has been
// code-gen'd, return a pointer to the function.  If not, compile it, or use
// a stub to implement lazy compilation if available.
//
void *JIT::getPointerToFunctionOrStub(Function *F) {
  // If we have already code generated the function, just return the address.
  if (void *Addr = getPointerToGlobalIfAvailable(F))
    return Addr;

  // Get a stub if the target supports it
  return getJITResolver(MCE).getFunctionStub(F);
}



//===----------------------------------------------------------------------===//
// JITEmitter code.
//
namespace {
  /// JITEmitter - The JIT implementation of the MachineCodeEmitter, which is
  /// used to output functions to memory for execution.
  class JITEmitter : public MachineCodeEmitter {
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

    /// Relocations - These are the relocations that the function needs, as
    /// emitted.
    std::vector<MachineRelocation> Relocations;

  public:
    JITEmitter(JIT &jit)
      :MemMgr(jit.getJITInfo().needsGOT())
    {
      TheJIT = &jit;
      DEBUG(std::cerr <<
            (MemMgr.isManagingGOT() ? "JIT is managing GOT\n"
             : "JIT is not managing GOT\n"));
    }

    virtual void startFunction(MachineFunction &F);
    virtual void finishFunction(MachineFunction &F);
    virtual void emitConstantPool(MachineConstantPool *MCP);
    virtual void startFunctionStub(unsigned StubSize);
    virtual void* finishFunctionStub(const Function *F);
    virtual void emitByte(unsigned char B);
    virtual void emitWord(unsigned W);
    virtual void emitWordAt(unsigned W, unsigned *Ptr);

    virtual void addRelocation(const MachineRelocation &MR) {
      Relocations.push_back(MR);
    }

    virtual uint64_t getCurrentPCValue();
    virtual uint64_t getCurrentPCOffset();
    virtual uint64_t getConstantPoolEntryAddress(unsigned Entry);
    virtual unsigned char* allocateGlobal(unsigned size, unsigned alignment);

  private:
    void *getPointerToGlobal(GlobalValue *GV, void *Reference, bool NoNeedStub);
  };
}

MachineCodeEmitter *JIT::createEmitter(JIT &jit) {
  return new JITEmitter(jit);
}

void *JITEmitter::getPointerToGlobal(GlobalValue *V, void *Reference,
                                     bool DoesntNeedStub) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    /// FIXME: If we straightened things out, this could actually emit the
    /// global immediately instead of queuing it for codegen later!
    return TheJIT->getOrEmitGlobalVariable(GV);
  }

  // If we have already compiled the function, return a pointer to its body.
  Function *F = cast<Function>(V);
  void *ResultPtr = TheJIT->getPointerToGlobalIfAvailable(F);
  if (ResultPtr) return ResultPtr;

  if (F->hasExternalLinkage() && F->isExternal()) {
    // If this is an external function pointer, we can force the JIT to
    // 'compile' it, which really just adds it to the map.
    if (DoesntNeedStub)
      return TheJIT->getPointerToFunction(F);

    return getJITResolver(this).getFunctionStub(F);
  }

  // Okay, the function has not been compiled yet, if the target callback
  // mechanism is capable of rewriting the instruction directly, prefer to do
  // that instead of emitting a stub.
  if (DoesntNeedStub)
    return getJITResolver(this).AddCallbackAtLocation(F, Reference);

  // Otherwise, we have to emit a lazy resolving stub.
  return getJITResolver(this).getFunctionStub(F);
}

void JITEmitter::startFunction(MachineFunction &F) {
  CurByte = CurBlock = MemMgr.startFunctionBody();
  TheJIT->addGlobalMapping(F.getFunction(), CurBlock);
}

void JITEmitter::finishFunction(MachineFunction &F) {
  MemMgr.endFunctionBody(CurByte);
  NumBytes += CurByte-CurBlock;

  if (!Relocations.empty()) {
    NumRelos += Relocations.size();

    // Resolve the relocations to concrete pointers.
    for (unsigned i = 0, e = Relocations.size(); i != e; ++i) {
      MachineRelocation &MR = Relocations[i];
      void *ResultPtr;
      if (MR.isString()) {
        ResultPtr = TheJIT->getPointerToNamedFunction(MR.getString());

        // If the target REALLY wants a stub for this function, emit it now.
        if (!MR.doesntNeedFunctionStub())
          ResultPtr = getJITResolver(this).getExternalFunctionStub(ResultPtr);
      } else if (MR.isGlobalValue())
        ResultPtr = getPointerToGlobal(MR.getGlobalValue(),
                                       CurBlock+MR.getMachineCodeOffset(),
                                       MR.doesntNeedFunctionStub());
      else //ConstantPoolIndex
        ResultPtr =
       (void*)(intptr_t)getConstantPoolEntryAddress(MR.getConstantPoolIndex());

      MR.setResultPointer(ResultPtr);

      // if we are managing the GOT and the relocation wants an index,
      // give it one
      if (MemMgr.isManagingGOT() && !MR.isConstantPoolIndex() &&
          MR.isGOTRelative()) {
        unsigned idx = getJITResolver(this).getGOTIndexForAddr(ResultPtr);
        MR.setGOTIndex(idx);
        if (((void**)MemMgr.getGOTBase())[idx] != ResultPtr) {
          DEBUG(std::cerr << "GOT was out of date for " << ResultPtr
                << " pointing at " << ((void**)MemMgr.getGOTBase())[idx]
                << "\n");
          ((void**)MemMgr.getGOTBase())[idx] = ResultPtr;
        }
      }
    }

    TheJIT->getJITInfo().relocate(CurBlock, &Relocations[0],
                                  Relocations.size(), MemMgr.getGOTBase());
  }

  //Update the GOT entry for F to point to the new code.
  if(MemMgr.isManagingGOT()) {
    unsigned idx = getJITResolver(this).getGOTIndexForAddr((void*)CurBlock);
    if (((void**)MemMgr.getGOTBase())[idx] != (void*)CurBlock) {
      DEBUG(std::cerr << "GOT was out of date for " << (void*)CurBlock
            << " pointing at " << ((void**)MemMgr.getGOTBase())[idx] << "\n");
      ((void**)MemMgr.getGOTBase())[idx] = (void*)CurBlock;
    }
  }

  DEBUG(std::cerr << "JIT: Finished CodeGen of [" << (void*)CurBlock
                  << "] Function: " << F.getFunction()->getName()
                  << ": " << CurByte-CurBlock << " bytes of text, "
                  << Relocations.size() << " relocations\n");
  Relocations.clear();
  ConstantPoolAddresses.clear();
}

void JITEmitter::emitConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &Constants = MCP->getConstants();
  if (Constants.empty()) return;

  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    const Type *Ty = Constants[i]->getType();
    unsigned Size      = (unsigned)TheJIT->getTargetData().getTypeSize(Ty);
    unsigned Alignment = TheJIT->getTargetData().getTypeAlignment(Ty);

    void *Addr = MemMgr.allocateConstant(Size, Alignment);
    TheJIT->InitializeMemory(Constants[i], Addr);
    ConstantPoolAddresses.push_back(Addr);
  }
}

void JITEmitter::startFunctionStub(unsigned StubSize) {
  SavedCurBlock = CurBlock;  SavedCurByte = CurByte;
  CurByte = CurBlock = MemMgr.allocateStub(StubSize);
}

void *JITEmitter::finishFunctionStub(const Function *F) {
  NumBytes += CurByte-CurBlock;
  std::swap(CurBlock, SavedCurBlock);
  CurByte = SavedCurByte;
  return SavedCurBlock;
}

void JITEmitter::emitByte(unsigned char B) {
  *CurByte++ = B;   // Write the byte to memory
}

void JITEmitter::emitWord(unsigned W) {
  // This won't work if the endianness of the host and target don't agree!  (For
  // a JIT this can't happen though.  :)
  *(unsigned*)CurByte = W;
  CurByte += sizeof(unsigned);
}

void JITEmitter::emitWordAt(unsigned W, unsigned *Ptr) {
  *Ptr = W;
}

// getConstantPoolEntryAddress - Return the address of the 'ConstantNum' entry
// in the constant pool that was last emitted with the 'emitConstantPool'
// method.
//
uint64_t JITEmitter::getConstantPoolEntryAddress(unsigned ConstantNum) {
  assert(ConstantNum < ConstantPoolAddresses.size() &&
         "Invalid ConstantPoolIndex!");
  return (intptr_t)ConstantPoolAddresses[ConstantNum];
}

unsigned char* JITEmitter::allocateGlobal(unsigned size, unsigned alignment)
{
  return MemMgr.allocateGlobal(size, alignment);
}

// getCurrentPCValue - This returns the address that the next emitted byte
// will be output to.
//
uint64_t JITEmitter::getCurrentPCValue() {
  return (intptr_t)CurByte;
}

uint64_t JITEmitter::getCurrentPCOffset() {
  return (intptr_t)CurByte-(intptr_t)CurBlock;
}

// getPointerToNamedFunction - This function is used as a global wrapper to
// JIT::getPointerToNamedFunction for the purpose of resolving symbols when
// bugpoint is debugging the JIT. In that scenario, we are loading an .so and
// need to resolve function(s) that are being mis-codegenerated, so we need to
// resolve their addresses at runtime, and this is the way to do it.
extern "C" {
  void *getPointerToNamedFunction(const char *Name) {
    Module &M = TheJIT->getModule();
    if (Function *F = M.getNamedFunction(Name))
      return TheJIT->getPointerToFunction(F);
    return TheJIT->getPointerToNamedFunction(Name);
  }
}
