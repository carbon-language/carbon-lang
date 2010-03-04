//===-- JITEmitter.cpp - Write machine code to executable memory ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a MachineCodeEmitter object that is used by the JIT to
// write machine code to memory and remember where relocatable values are.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "JIT.h"
#include "JITDebugRegisterer.h"
#include "JITDwarfEmitter.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRelocation.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/CodeGen/MachineCodeInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetJITInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Disassembler.h"
#include "llvm/System/Memory.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/ValueMap.h"
#include <algorithm>
#ifndef NDEBUG
#include <iomanip>
#endif
using namespace llvm;

STATISTIC(NumBytes, "Number of bytes of machine code compiled");
STATISTIC(NumRelos, "Number of relocations applied");
STATISTIC(NumRetries, "Number of retries with more memory");


// A declaration may stop being a declaration once it's fully read from bitcode.
// This function returns true if F is fully read and is still a declaration.
static bool isNonGhostDeclaration(const Function *F) {
  return F->isDeclaration() && !F->isMaterializable();
}

//===----------------------------------------------------------------------===//
// JIT lazy compilation code.
//
namespace {
  class JITEmitter;
  class JITResolverState;

  template<typename ValueTy>
  struct NoRAUWValueMapConfig : public ValueMapConfig<ValueTy> {
    typedef JITResolverState *ExtraData;
    static void onRAUW(JITResolverState *, Value *Old, Value *New) {
      assert(false && "The JIT doesn't know how to handle a"
             " RAUW on a value it has emitted.");
    }
  };

  struct CallSiteValueMapConfig : public NoRAUWValueMapConfig<Function*> {
    typedef JITResolverState *ExtraData;
    static void onDelete(JITResolverState *JRS, Function *F);
  };

  class JITResolverState {
  public:
    typedef ValueMap<Function*, void*, NoRAUWValueMapConfig<Function*> >
      FunctionToLazyStubMapTy;
    typedef std::map<void*, AssertingVH<Function> > CallSiteToFunctionMapTy;
    typedef ValueMap<Function *, SmallPtrSet<void*, 1>,
                     CallSiteValueMapConfig> FunctionToCallSitesMapTy;
    typedef std::map<AssertingVH<GlobalValue>, void*> GlobalToIndirectSymMapTy;
  private:
    /// FunctionToLazyStubMap - Keep track of the lazy stub created for a
    /// particular function so that we can reuse them if necessary.
    FunctionToLazyStubMapTy FunctionToLazyStubMap;

    /// CallSiteToFunctionMap - Keep track of the function that each lazy call
    /// site corresponds to, and vice versa.
    CallSiteToFunctionMapTy CallSiteToFunctionMap;
    FunctionToCallSitesMapTy FunctionToCallSitesMap;

    /// GlobalToIndirectSymMap - Keep track of the indirect symbol created for a
    /// particular GlobalVariable so that we can reuse them if necessary.
    GlobalToIndirectSymMapTy GlobalToIndirectSymMap;

    /// Instance of the JIT this ResolverState serves.
    JIT *TheJIT;

  public:
    JITResolverState(JIT *jit) : FunctionToLazyStubMap(this),
                                 FunctionToCallSitesMap(this),
                                 TheJIT(jit) {}

    FunctionToLazyStubMapTy& getFunctionToLazyStubMap(
      const MutexGuard& locked) {
      assert(locked.holds(TheJIT->lock));
      return FunctionToLazyStubMap;
    }

    GlobalToIndirectSymMapTy& getGlobalToIndirectSymMap(const MutexGuard& locked) {
      assert(locked.holds(TheJIT->lock));
      return GlobalToIndirectSymMap;
    }

    pair<void *, Function *> LookupFunctionFromCallSite(
        const MutexGuard &locked, void *CallSite) const {
      assert(locked.holds(TheJIT->lock));

      // The address given to us for the stub may not be exactly right, it might be
      // a little bit after the stub.  As such, use upper_bound to find it.
      CallSiteToFunctionMapTy::const_iterator I =
        CallSiteToFunctionMap.upper_bound(CallSite);
      assert(I != CallSiteToFunctionMap.begin() &&
             "This is not a known call site!");
      --I;
      return *I;
    }

    void AddCallSite(const MutexGuard &locked, void *CallSite, Function *F) {
      assert(locked.holds(TheJIT->lock));

      bool Inserted = CallSiteToFunctionMap.insert(
          std::make_pair(CallSite, F)).second;
      (void)Inserted;
      assert(Inserted && "Pair was already in CallSiteToFunctionMap");
      FunctionToCallSitesMap[F].insert(CallSite);
    }

    // Returns the Function of the stub if a stub was erased, or NULL if there
    // was no stub.  This function uses the call-site->function map to find a
    // relevant function, but asserts that only stubs and not other call sites
    // will be passed in.
    Function *EraseStub(const MutexGuard &locked, void *Stub);

    void EraseAllCallSitesFor(const MutexGuard &locked, Function *F) {
      assert(locked.holds(TheJIT->lock));
      EraseAllCallSitesForPrelocked(F);
    }
    void EraseAllCallSitesForPrelocked(Function *F);

    // Erases _all_ call sites regardless of their function.  This is used to
    // unregister the stub addresses from the StubToResolverMap in
    // ~JITResolver().
    void EraseAllCallSitesPrelocked();
  };

  /// JITResolver - Keep track of, and resolve, call sites for functions that
  /// have not yet been compiled.
  class JITResolver {
    typedef JITResolverState::FunctionToLazyStubMapTy FunctionToLazyStubMapTy;
    typedef JITResolverState::CallSiteToFunctionMapTy CallSiteToFunctionMapTy;
    typedef JITResolverState::GlobalToIndirectSymMapTy GlobalToIndirectSymMapTy;

    /// LazyResolverFn - The target lazy resolver function that we actually
    /// rewrite instructions to use.
    TargetJITInfo::LazyResolverFn LazyResolverFn;

    JITResolverState state;

    /// ExternalFnToStubMap - This is the equivalent of FunctionToLazyStubMap
    /// for external functions.  TODO: Of course, external functions don't need
    /// a lazy stub.  It's actually here to make it more likely that far calls
    /// succeed, but no single stub can guarantee that.  I'll remove this in a
    /// subsequent checkin when I actually fix far calls.
    std::map<void*, void*> ExternalFnToStubMap;

    /// revGOTMap - map addresses to indexes in the GOT
    std::map<void*, unsigned> revGOTMap;
    unsigned nextGOTIndex;

    JITEmitter &JE;

    /// Instance of JIT corresponding to this Resolver.
    JIT *TheJIT;

  public:
    explicit JITResolver(JIT &jit, JITEmitter &je)
      : state(&jit), nextGOTIndex(0), JE(je), TheJIT(&jit) {
      LazyResolverFn = jit.getJITInfo().getLazyResolverFunction(JITCompilerFn);
    }

    ~JITResolver();

    /// getLazyFunctionStubIfAvailable - This returns a pointer to a function's
    /// lazy-compilation stub if it has already been created.
    void *getLazyFunctionStubIfAvailable(Function *F);

    /// getLazyFunctionStub - This returns a pointer to a function's
    /// lazy-compilation stub, creating one on demand as needed.
    void *getLazyFunctionStub(Function *F);

    /// getExternalFunctionStub - Return a stub for the function at the
    /// specified address, created lazily on demand.
    void *getExternalFunctionStub(void *FnAddr);

    /// getGlobalValueIndirectSym - Return an indirect symbol containing the
    /// specified GV address.
    void *getGlobalValueIndirectSym(GlobalValue *V, void *GVAddress);

    void getRelocatableGVs(SmallVectorImpl<GlobalValue*> &GVs,
                           SmallVectorImpl<void*> &Ptrs);

    GlobalValue *invalidateStub(void *Stub);

    /// getGOTIndexForAddress - Return a new or existing index in the GOT for
    /// an address.  This function only manages slots, it does not manage the
    /// contents of the slots or the memory associated with the GOT.
    unsigned getGOTIndexForAddr(void *addr);

    /// JITCompilerFn - This function is called to resolve a stub to a compiled
    /// address.  If the LLVM Function corresponding to the stub has not yet
    /// been compiled, this function compiles it first.
    static void *JITCompilerFn(void *Stub);
  };

  class StubToResolverMapTy {
    /// Map a stub address to a specific instance of a JITResolver so that
    /// lazily-compiled functions can find the right resolver to use.
    ///
    /// Guarded by Lock.
    std::map<void*, JITResolver*> Map;

    /// Guards Map from concurrent accesses.
    mutable sys::Mutex Lock;

  public:
    /// Registers a Stub to be resolved by Resolver.
    void RegisterStubResolver(void *Stub, JITResolver *Resolver) {
      MutexGuard guard(Lock);
      Map.insert(std::make_pair(Stub, Resolver));
    }
    /// Unregisters the Stub when it's invalidated.
    void UnregisterStubResolver(void *Stub) {
      MutexGuard guard(Lock);
      Map.erase(Stub);
    }
    /// Returns the JITResolver instance that owns the Stub.
    JITResolver *getResolverFromStub(void *Stub) const {
      MutexGuard guard(Lock);
      // The address given to us for the stub may not be exactly right, it might
      // be a little bit after the stub.  As such, use upper_bound to find it.
      // This is the same trick as in LookupFunctionFromCallSite from
      // JITResolverState.
      std::map<void*, JITResolver*>::const_iterator I = Map.upper_bound(Stub);
      assert(I != Map.begin() && "This is not a known stub!");
      --I;
      return I->second;
    }
    /// True if any stubs refer to the given resolver. Only used in an assert().
    /// O(N)
    bool ResolverHasStubs(JITResolver* Resolver) const {
      MutexGuard guard(Lock);
      for (std::map<void*, JITResolver*>::const_iterator I = Map.begin(),
             E = Map.end(); I != E; ++I) {
        if (I->second == Resolver)
          return true;
      }
      return false;
    }
  };
  /// This needs to be static so that a lazy call stub can access it with no
  /// context except the address of the stub.
  ManagedStatic<StubToResolverMapTy> StubToResolverMap;

  /// JITEmitter - The JIT implementation of the MachineCodeEmitter, which is
  /// used to output functions to memory for execution.
  class JITEmitter : public JITCodeEmitter {
    JITMemoryManager *MemMgr;

    // When outputting a function stub in the context of some other function, we
    // save BufferBegin/BufferEnd/CurBufferPtr here.
    uint8_t *SavedBufferBegin, *SavedBufferEnd, *SavedCurBufferPtr;

    // When reattempting to JIT a function after running out of space, we store
    // the estimated size of the function we're trying to JIT here, so we can
    // ask the memory manager for at least this much space.  When we
    // successfully emit the function, we reset this back to zero.
    uintptr_t SizeEstimate;

    /// Relocations - These are the relocations that the function needs, as
    /// emitted.
    std::vector<MachineRelocation> Relocations;

    /// MBBLocations - This vector is a mapping from MBB ID's to their address.
    /// It is filled in by the StartMachineBasicBlock callback and queried by
    /// the getMachineBasicBlockAddress callback.
    std::vector<uintptr_t> MBBLocations;

    /// ConstantPool - The constant pool for the current function.
    ///
    MachineConstantPool *ConstantPool;

    /// ConstantPoolBase - A pointer to the first entry in the constant pool.
    ///
    void *ConstantPoolBase;

    /// ConstPoolAddresses - Addresses of individual constant pool entries.
    ///
    SmallVector<uintptr_t, 8> ConstPoolAddresses;

    /// JumpTable - The jump tables for the current function.
    ///
    MachineJumpTableInfo *JumpTable;

    /// JumpTableBase - A pointer to the first entry in the jump table.
    ///
    void *JumpTableBase;

    /// Resolver - This contains info about the currently resolved functions.
    JITResolver Resolver;

    /// DE - The dwarf emitter for the jit.
    OwningPtr<JITDwarfEmitter> DE;

    /// DR - The debug registerer for the jit.
    OwningPtr<JITDebugRegisterer> DR;

    /// LabelLocations - This vector is a mapping from Label ID's to their
    /// address.
    std::vector<uintptr_t> LabelLocations;

    /// MMI - Machine module info for exception informations
    MachineModuleInfo* MMI;

    // GVSet - a set to keep track of which globals have been seen
    SmallPtrSet<const GlobalVariable*, 8> GVSet;

    // CurFn - The llvm function being emitted.  Only valid during
    // finishFunction().
    const Function *CurFn;

    /// Information about emitted code, which is passed to the
    /// JITEventListeners.  This is reset in startFunction and used in
    /// finishFunction.
    JITEvent_EmittedFunctionDetails EmissionDetails;

    struct EmittedCode {
      void *FunctionBody;  // Beginning of the function's allocation.
      void *Code;  // The address the function's code actually starts at.
      void *ExceptionTable;
      EmittedCode() : FunctionBody(0), Code(0), ExceptionTable(0) {}
    };
    struct EmittedFunctionConfig : public ValueMapConfig<const Function*> {
      typedef JITEmitter *ExtraData;
      static void onDelete(JITEmitter *, const Function*);
      static void onRAUW(JITEmitter *, const Function*, const Function*);
    };
    ValueMap<const Function *, EmittedCode,
             EmittedFunctionConfig> EmittedFunctions;

    // CurFnStubUses - For a given Function, a vector of stubs that it
    // references.  This facilitates the JIT detecting that a stub is no
    // longer used, so that it may be deallocated.
    DenseMap<AssertingVH<const Function>, SmallVector<void*, 1> > CurFnStubUses;

    // StubFnRefs - For a given pointer to a stub, a set of Functions which
    // reference the stub.  When the count of a stub's references drops to zero,
    // the stub is unused.
    DenseMap<void *, SmallPtrSet<const Function*, 1> > StubFnRefs;

    DILocation PrevDLT;

    /// Instance of the JIT
    JIT *TheJIT;

  public:
    JITEmitter(JIT &jit, JITMemoryManager *JMM, TargetMachine &TM)
      : SizeEstimate(0), Resolver(jit, *this), MMI(0), CurFn(0),
        EmittedFunctions(this), PrevDLT(NULL), TheJIT(&jit) {
      MemMgr = JMM ? JMM : JITMemoryManager::CreateDefaultMemManager();
      if (jit.getJITInfo().needsGOT()) {
        MemMgr->AllocateGOT();
        DEBUG(dbgs() << "JIT is managing a GOT\n");
      }

      if (DwarfExceptionHandling || JITEmitDebugInfo) {
        DE.reset(new JITDwarfEmitter(jit));
      }
      if (JITEmitDebugInfo) {
        DR.reset(new JITDebugRegisterer(TM));
      }
    }
    ~JITEmitter() {
      delete MemMgr;
    }

    /// classof - Methods for support type inquiry through isa, cast, and
    /// dyn_cast:
    ///
    static inline bool classof(const JITEmitter*) { return true; }
    static inline bool classof(const MachineCodeEmitter*) { return true; }

    JITResolver &getJITResolver() { return Resolver; }

    virtual void startFunction(MachineFunction &F);
    virtual bool finishFunction(MachineFunction &F);

    void emitConstantPool(MachineConstantPool *MCP);
    void initJumpTableInfo(MachineJumpTableInfo *MJTI);
    void emitJumpTableInfo(MachineJumpTableInfo *MJTI);

    void startGVStub(const GlobalValue* GV,
                     unsigned StubSize, unsigned Alignment = 1);
    void startGVStub(void *Buffer, unsigned StubSize);
    void finishGVStub();
    virtual void *allocIndirectGV(const GlobalValue *GV,
                                  const uint8_t *Buffer, size_t Size,
                                  unsigned Alignment);

    /// allocateSpace - Reserves space in the current block if any, or
    /// allocate a new one of the given size.
    virtual void *allocateSpace(uintptr_t Size, unsigned Alignment);

    /// allocateGlobal - Allocate memory for a global.  Unlike allocateSpace,
    /// this method does not allocate memory in the current output buffer,
    /// because a global may live longer than the current function.
    virtual void *allocateGlobal(uintptr_t Size, unsigned Alignment);

    virtual void addRelocation(const MachineRelocation &MR) {
      Relocations.push_back(MR);
    }

    virtual void StartMachineBasicBlock(MachineBasicBlock *MBB) {
      if (MBBLocations.size() <= (unsigned)MBB->getNumber())
        MBBLocations.resize((MBB->getNumber()+1)*2);
      MBBLocations[MBB->getNumber()] = getCurrentPCValue();
      DEBUG(dbgs() << "JIT: Emitting BB" << MBB->getNumber() << " at ["
                   << (void*) getCurrentPCValue() << "]\n");
    }

    virtual uintptr_t getConstantPoolEntryAddress(unsigned Entry) const;
    virtual uintptr_t getJumpTableEntryAddress(unsigned Entry) const;

    virtual uintptr_t getMachineBasicBlockAddress(MachineBasicBlock *MBB) const {
      assert(MBBLocations.size() > (unsigned)MBB->getNumber() &&
             MBBLocations[MBB->getNumber()] && "MBB not emitted!");
      return MBBLocations[MBB->getNumber()];
    }

    /// retryWithMoreMemory - Log a retry and deallocate all memory for the
    /// given function.  Increase the minimum allocation size so that we get
    /// more memory next time.
    void retryWithMoreMemory(MachineFunction &F);

    /// deallocateMemForFunction - Deallocate all memory for the specified
    /// function body.
    void deallocateMemForFunction(const Function *F);

    /// AddStubToCurrentFunction - Mark the current function being JIT'd as
    /// using the stub at the specified address. Allows
    /// deallocateMemForFunction to also remove stubs no longer referenced.
    void AddStubToCurrentFunction(void *Stub);

    virtual void processDebugLoc(DebugLoc DL, bool BeforePrintingInsn);

    virtual void emitLabel(uint64_t LabelID) {
      if (LabelLocations.size() <= LabelID)
        LabelLocations.resize((LabelID+1)*2);
      LabelLocations[LabelID] = getCurrentPCValue();
    }

    virtual uintptr_t getLabelAddress(uint64_t LabelID) const {
      assert(LabelLocations.size() > (unsigned)LabelID &&
             LabelLocations[LabelID] && "Label not emitted!");
      return LabelLocations[LabelID];
    }

    virtual void setModuleInfo(MachineModuleInfo* Info) {
      MMI = Info;
      if (DE.get()) DE->setModuleInfo(Info);
    }

    void setMemoryExecutable() {
      MemMgr->setMemoryExecutable();
    }

    JITMemoryManager *getMemMgr() const { return MemMgr; }

  private:
    void *getPointerToGlobal(GlobalValue *GV, void *Reference,
                             bool MayNeedFarStub);
    void *getPointerToGVIndirectSym(GlobalValue *V, void *Reference);
    unsigned addSizeOfGlobal(const GlobalVariable *GV, unsigned Size);
    unsigned addSizeOfGlobalsInConstantVal(const Constant *C, unsigned Size);
    unsigned addSizeOfGlobalsInInitializer(const Constant *Init, unsigned Size);
    unsigned GetSizeOfGlobalsInBytes(MachineFunction &MF);
  };
}

void CallSiteValueMapConfig::onDelete(JITResolverState *JRS, Function *F) {
  JRS->EraseAllCallSitesForPrelocked(F);
}

Function *JITResolverState::EraseStub(const MutexGuard &locked, void *Stub) {
  CallSiteToFunctionMapTy::iterator C2F_I =
    CallSiteToFunctionMap.find(Stub);
  if (C2F_I == CallSiteToFunctionMap.end()) {
    // Not a stub.
    return NULL;
  }

  StubToResolverMap->UnregisterStubResolver(Stub);

  Function *const F = C2F_I->second;
#ifndef NDEBUG
  void *RealStub = FunctionToLazyStubMap.lookup(F);
  assert(RealStub == Stub &&
         "Call-site that wasn't a stub passed in to EraseStub");
#endif
  FunctionToLazyStubMap.erase(F);
  CallSiteToFunctionMap.erase(C2F_I);

  // Remove the stub from the function->call-sites map, and remove the whole
  // entry from the map if that was the last call site.
  FunctionToCallSitesMapTy::iterator F2C_I = FunctionToCallSitesMap.find(F);
  assert(F2C_I != FunctionToCallSitesMap.end() &&
         "FunctionToCallSitesMap broken");
  bool Erased = F2C_I->second.erase(Stub);
  (void)Erased;
  assert(Erased && "FunctionToCallSitesMap broken");
  if (F2C_I->second.empty())
    FunctionToCallSitesMap.erase(F2C_I);

  return F;
}

void JITResolverState::EraseAllCallSitesForPrelocked(Function *F) {
  FunctionToCallSitesMapTy::iterator F2C = FunctionToCallSitesMap.find(F);
  if (F2C == FunctionToCallSitesMap.end())
    return;
  StubToResolverMapTy &S2RMap = *StubToResolverMap;
  for (SmallPtrSet<void*, 1>::const_iterator I = F2C->second.begin(),
         E = F2C->second.end(); I != E; ++I) {
    S2RMap.UnregisterStubResolver(*I);
    bool Erased = CallSiteToFunctionMap.erase(*I);
    (void)Erased;
    assert(Erased && "Missing call site->function mapping");
  }
  FunctionToCallSitesMap.erase(F2C);
}

void JITResolverState::EraseAllCallSitesPrelocked() {
  StubToResolverMapTy &S2RMap = *StubToResolverMap;
  for (CallSiteToFunctionMapTy::const_iterator
         I = CallSiteToFunctionMap.begin(),
         E = CallSiteToFunctionMap.end(); I != E; ++I) {
    S2RMap.UnregisterStubResolver(I->first);
  }
  CallSiteToFunctionMap.clear();
  FunctionToCallSitesMap.clear();
}

JITResolver::~JITResolver() {
  // No need to lock because we're in the destructor, and state isn't shared.
  state.EraseAllCallSitesPrelocked();
  assert(!StubToResolverMap->ResolverHasStubs(this) &&
         "Resolver destroyed with stubs still alive.");
}

/// getLazyFunctionStubIfAvailable - This returns a pointer to a function stub
/// if it has already been created.
void *JITResolver::getLazyFunctionStubIfAvailable(Function *F) {
  MutexGuard locked(TheJIT->lock);

  // If we already have a stub for this function, recycle it.
  return state.getFunctionToLazyStubMap(locked).lookup(F);
}

/// getFunctionStub - This returns a pointer to a function stub, creating
/// one on demand as needed.
void *JITResolver::getLazyFunctionStub(Function *F) {
  MutexGuard locked(TheJIT->lock);

  // If we already have a lazy stub for this function, recycle it.
  void *&Stub = state.getFunctionToLazyStubMap(locked)[F];
  if (Stub) return Stub;

  // Call the lazy resolver function if we are JIT'ing lazily.  Otherwise we
  // must resolve the symbol now.
  void *Actual = TheJIT->isCompilingLazily()
    ? (void *)(intptr_t)LazyResolverFn : (void *)0;

  // If this is an external declaration, attempt to resolve the address now
  // to place in the stub.
  if (isNonGhostDeclaration(F) || F->hasAvailableExternallyLinkage()) {
    Actual = TheJIT->getPointerToFunction(F);

    // If we resolved the symbol to a null address (eg. a weak external)
    // don't emit a stub. Return a null pointer to the application.
    if (!Actual) return 0;
  }

  TargetJITInfo::StubLayout SL = TheJIT->getJITInfo().getStubLayout();
  JE.startGVStub(F, SL.Size, SL.Alignment);
  // Codegen a new stub, calling the lazy resolver or the actual address of the
  // external function, if it was resolved.
  Stub = TheJIT->getJITInfo().emitFunctionStub(F, Actual, JE);
  JE.finishGVStub();

  if (Actual != (void*)(intptr_t)LazyResolverFn) {
    // If we are getting the stub for an external function, we really want the
    // address of the stub in the GlobalAddressMap for the JIT, not the address
    // of the external function.
    TheJIT->updateGlobalMapping(F, Stub);
  }

  DEBUG(dbgs() << "JIT: Lazy stub emitted at [" << Stub << "] for function '"
        << F->getName() << "'\n");

  if (TheJIT->isCompilingLazily()) {
    // Register this JITResolver as the one corresponding to this call site so
    // JITCompilerFn will be able to find it.
    StubToResolverMap->RegisterStubResolver(Stub, this);

    // Finally, keep track of the stub-to-Function mapping so that the
    // JITCompilerFn knows which function to compile!
    state.AddCallSite(locked, Stub, F);
  } else if (!Actual) {
    // If we are JIT'ing non-lazily but need to call a function that does not
    // exist yet, add it to the JIT's work list so that we can fill in the
    // stub address later.
    assert(!isNonGhostDeclaration(F) && !F->hasAvailableExternallyLinkage() &&
           "'Actual' should have been set above.");
    TheJIT->addPendingFunction(F);
  }

  return Stub;
}

/// getGlobalValueIndirectSym - Return a lazy pointer containing the specified
/// GV address.
void *JITResolver::getGlobalValueIndirectSym(GlobalValue *GV, void *GVAddress) {
  MutexGuard locked(TheJIT->lock);

  // If we already have a stub for this global variable, recycle it.
  void *&IndirectSym = state.getGlobalToIndirectSymMap(locked)[GV];
  if (IndirectSym) return IndirectSym;

  // Otherwise, codegen a new indirect symbol.
  IndirectSym = TheJIT->getJITInfo().emitGlobalValueIndirectSym(GV, GVAddress,
                                                                JE);

  DEBUG(dbgs() << "JIT: Indirect symbol emitted at [" << IndirectSym
        << "] for GV '" << GV->getName() << "'\n");

  return IndirectSym;
}

/// getExternalFunctionStub - Return a stub for the function at the
/// specified address, created lazily on demand.
void *JITResolver::getExternalFunctionStub(void *FnAddr) {
  // If we already have a stub for this function, recycle it.
  void *&Stub = ExternalFnToStubMap[FnAddr];
  if (Stub) return Stub;

  TargetJITInfo::StubLayout SL = TheJIT->getJITInfo().getStubLayout();
  JE.startGVStub(0, SL.Size, SL.Alignment);
  Stub = TheJIT->getJITInfo().emitFunctionStub(0, FnAddr, JE);
  JE.finishGVStub();

  DEBUG(dbgs() << "JIT: Stub emitted at [" << Stub
               << "] for external function at '" << FnAddr << "'\n");
  return Stub;
}

unsigned JITResolver::getGOTIndexForAddr(void* addr) {
  unsigned idx = revGOTMap[addr];
  if (!idx) {
    idx = ++nextGOTIndex;
    revGOTMap[addr] = idx;
    DEBUG(dbgs() << "JIT: Adding GOT entry " << idx << " for addr ["
                 << addr << "]\n");
  }
  return idx;
}

void JITResolver::getRelocatableGVs(SmallVectorImpl<GlobalValue*> &GVs,
                                    SmallVectorImpl<void*> &Ptrs) {
  MutexGuard locked(TheJIT->lock);

  const FunctionToLazyStubMapTy &FM = state.getFunctionToLazyStubMap(locked);
  GlobalToIndirectSymMapTy &GM = state.getGlobalToIndirectSymMap(locked);

  for (FunctionToLazyStubMapTy::const_iterator i = FM.begin(), e = FM.end();
       i != e; ++i){
    Function *F = i->first;
    if (F->isDeclaration() && F->hasExternalLinkage()) {
      GVs.push_back(i->first);
      Ptrs.push_back(i->second);
    }
  }
  for (GlobalToIndirectSymMapTy::iterator i = GM.begin(), e = GM.end();
       i != e; ++i) {
    GVs.push_back(i->first);
    Ptrs.push_back(i->second);
  }
}

GlobalValue *JITResolver::invalidateStub(void *Stub) {
  MutexGuard locked(TheJIT->lock);

  // Remove the stub from the StubToResolverMap.
  StubToResolverMap->UnregisterStubResolver(Stub);

  GlobalToIndirectSymMapTy &GM = state.getGlobalToIndirectSymMap(locked);

  // Look up the cheap way first, to see if it's a function stub we are
  // invalidating.  If so, remove it from both the forward and reverse maps.
  if (Function *F = state.EraseStub(locked, Stub)) {
    return F;
  }

  // Otherwise, it might be an indirect symbol stub.  Find it and remove it.
  for (GlobalToIndirectSymMapTy::iterator i = GM.begin(), e = GM.end();
       i != e; ++i) {
    if (i->second != Stub)
      continue;
    GlobalValue *GV = i->first;
    GM.erase(i);
    return GV;
  }

  // Lastly, check to see if it's in the ExternalFnToStubMap.
  for (std::map<void *, void *>::iterator i = ExternalFnToStubMap.begin(),
       e = ExternalFnToStubMap.end(); i != e; ++i) {
    if (i->second != Stub)
      continue;
    ExternalFnToStubMap.erase(i);
    break;
  }

  return 0;
}

/// JITCompilerFn - This function is called when a lazy compilation stub has
/// been entered.  It looks up which function this stub corresponds to, compiles
/// it if necessary, then returns the resultant function pointer.
void *JITResolver::JITCompilerFn(void *Stub) {
  JITResolver *JR = StubToResolverMap->getResolverFromStub(Stub);
  assert(JR && "Unable to find the corresponding JITResolver to the call site");

  Function* F = 0;
  void* ActualPtr = 0;

  {
    // Only lock for getting the Function. The call getPointerToFunction made
    // in this function might trigger function materializing, which requires
    // JIT lock to be unlocked.
    MutexGuard locked(JR->TheJIT->lock);

    // The address given to us for the stub may not be exactly right, it might
    // be a little bit after the stub.  As such, use upper_bound to find it.
    pair<void*, Function*> I =
      JR->state.LookupFunctionFromCallSite(locked, Stub);
    F = I.second;
    ActualPtr = I.first;
  }

  // If we have already code generated the function, just return the address.
  void *Result = JR->TheJIT->getPointerToGlobalIfAvailable(F);

  if (!Result) {
    // Otherwise we don't have it, do lazy compilation now.

    // If lazy compilation is disabled, emit a useful error message and abort.
    if (!JR->TheJIT->isCompilingLazily()) {
      llvm_report_error("LLVM JIT requested to do lazy compilation of function '"
                        + F->getName() + "' when lazy compiles are disabled!");
    }

    DEBUG(dbgs() << "JIT: Lazily resolving function '" << F->getName()
          << "' In stub ptr = " << Stub << " actual ptr = "
          << ActualPtr << "\n");

    Result = JR->TheJIT->getPointerToFunction(F);
  }

  // Reacquire the lock to update the GOT map.
  MutexGuard locked(JR->TheJIT->lock);

  // We might like to remove the call site from the CallSiteToFunction map, but
  // we can't do that! Multiple threads could be stuck, waiting to acquire the
  // lock above. As soon as the 1st function finishes compiling the function,
  // the next one will be released, and needs to be able to find the function it
  // needs to call.

  // FIXME: We could rewrite all references to this stub if we knew them.

  // What we will do is set the compiled function address to map to the
  // same GOT entry as the stub so that later clients may update the GOT
  // if they see it still using the stub address.
  // Note: this is done so the Resolver doesn't have to manage GOT memory
  // Do this without allocating map space if the target isn't using a GOT
  if(JR->revGOTMap.find(Stub) != JR->revGOTMap.end())
    JR->revGOTMap[Result] = JR->revGOTMap[Stub];

  return Result;
}

//===----------------------------------------------------------------------===//
// JITEmitter code.
//
void *JITEmitter::getPointerToGlobal(GlobalValue *V, void *Reference,
                                     bool MayNeedFarStub) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return TheJIT->getOrEmitGlobalVariable(GV);

  if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V))
    return TheJIT->getPointerToGlobal(GA->resolveAliasedGlobal(false));

  // If we have already compiled the function, return a pointer to its body.
  Function *F = cast<Function>(V);

  void *FnStub = Resolver.getLazyFunctionStubIfAvailable(F);
  if (FnStub) {
    // Return the function stub if it's already created.  We do this first so
    // that we're returning the same address for the function as any previous
    // call.  TODO: Yes, this is wrong. The lazy stub isn't guaranteed to be
    // close enough to call.
    AddStubToCurrentFunction(FnStub);
    return FnStub;
  }

  // If we know the target can handle arbitrary-distance calls, try to
  // return a direct pointer.
  if (!MayNeedFarStub) {
    // If we have code, go ahead and return that.
    void *ResultPtr = TheJIT->getPointerToGlobalIfAvailable(F);
    if (ResultPtr) return ResultPtr;

    // If this is an external function pointer, we can force the JIT to
    // 'compile' it, which really just adds it to the map.
    if (isNonGhostDeclaration(F) || F->hasAvailableExternallyLinkage())
      return TheJIT->getPointerToFunction(F);
  }

  // Otherwise, we may need a to emit a stub, and, conservatively, we
  // always do so.
  void *StubAddr = Resolver.getLazyFunctionStub(F);

  // Add the stub to the current function's list of referenced stubs, so we can
  // deallocate them if the current function is ever freed.  It's possible to
  // return null from getLazyFunctionStub in the case of a weak extern that
  // fails to resolve.
  if (StubAddr)
    AddStubToCurrentFunction(StubAddr);

  return StubAddr;
}

void *JITEmitter::getPointerToGVIndirectSym(GlobalValue *V, void *Reference) {
  // Make sure GV is emitted first, and create a stub containing the fully
  // resolved address.
  void *GVAddress = getPointerToGlobal(V, Reference, false);
  void *StubAddr = Resolver.getGlobalValueIndirectSym(V, GVAddress);

  // Add the stub to the current function's list of referenced stubs, so we can
  // deallocate them if the current function is ever freed.
  AddStubToCurrentFunction(StubAddr);

  return StubAddr;
}

void JITEmitter::AddStubToCurrentFunction(void *StubAddr) {
  assert(CurFn && "Stub added to current function, but current function is 0!");

  SmallVectorImpl<void*> &StubsUsed = CurFnStubUses[CurFn];
  StubsUsed.push_back(StubAddr);

  SmallPtrSet<const Function *, 1> &FnRefs = StubFnRefs[StubAddr];
  FnRefs.insert(CurFn);
}

void JITEmitter::processDebugLoc(DebugLoc DL, bool BeforePrintingInsn) {
  if (!DL.isUnknown()) {
    DILocation CurDLT = EmissionDetails.MF->getDILocation(DL);

    if (BeforePrintingInsn) {
      if (CurDLT.getScope().getNode() != 0 
          && PrevDLT.getNode() != CurDLT.getNode()) {
        JITEvent_EmittedFunctionDetails::LineStart NextLine;
        NextLine.Address = getCurrentPCValue();
        NextLine.Loc = DL;
        EmissionDetails.LineStarts.push_back(NextLine);
      }

      PrevDLT = CurDLT;
    }
  }
}

static unsigned GetConstantPoolSizeInBytes(MachineConstantPool *MCP,
                                           const TargetData *TD) {
  const std::vector<MachineConstantPoolEntry> &Constants = MCP->getConstants();
  if (Constants.empty()) return 0;

  unsigned Size = 0;
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    MachineConstantPoolEntry CPE = Constants[i];
    unsigned AlignMask = CPE.getAlignment() - 1;
    Size = (Size + AlignMask) & ~AlignMask;
    const Type *Ty = CPE.getType();
    Size += TD->getTypeAllocSize(Ty);
  }
  return Size;
}

static unsigned GetJumpTableSizeInBytes(MachineJumpTableInfo *MJTI, JIT *jit) {
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return 0;

  unsigned NumEntries = 0;
  for (unsigned i = 0, e = JT.size(); i != e; ++i)
    NumEntries += JT[i].MBBs.size();

  return NumEntries * MJTI->getEntrySize(*jit->getTargetData());
}

static uintptr_t RoundUpToAlign(uintptr_t Size, unsigned Alignment) {
  if (Alignment == 0) Alignment = 1;
  // Since we do not know where the buffer will be allocated, be pessimistic.
  return Size + Alignment;
}

/// addSizeOfGlobal - add the size of the global (plus any alignment padding)
/// into the running total Size.

unsigned JITEmitter::addSizeOfGlobal(const GlobalVariable *GV, unsigned Size) {
  const Type *ElTy = GV->getType()->getElementType();
  size_t GVSize = (size_t)TheJIT->getTargetData()->getTypeAllocSize(ElTy);
  size_t GVAlign =
      (size_t)TheJIT->getTargetData()->getPreferredAlignment(GV);
  DEBUG(dbgs() << "JIT: Adding in size " << GVSize << " alignment " << GVAlign);
  DEBUG(GV->dump());
  // Assume code section ends with worst possible alignment, so first
  // variable needs maximal padding.
  if (Size==0)
    Size = 1;
  Size = ((Size+GVAlign-1)/GVAlign)*GVAlign;
  Size += GVSize;
  return Size;
}

/// addSizeOfGlobalsInConstantVal - find any globals that we haven't seen yet
/// but are referenced from the constant; put them in GVSet and add their
/// size into the running total Size.

unsigned JITEmitter::addSizeOfGlobalsInConstantVal(const Constant *C,
                                              unsigned Size) {
  // If its undefined, return the garbage.
  if (isa<UndefValue>(C))
    return Size;

  // If the value is a ConstantExpr
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    Constant *Op0 = CE->getOperand(0);
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::BitCast: {
      Size = addSizeOfGlobalsInConstantVal(Op0, Size);
      break;
    }
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      Size = addSizeOfGlobalsInConstantVal(Op0, Size);
      Size = addSizeOfGlobalsInConstantVal(CE->getOperand(1), Size);
      break;
    }
    default: {
       std::string msg;
       raw_string_ostream Msg(msg);
       Msg << "ConstantExpr not handled: " << *CE;
       llvm_report_error(Msg.str());
    }
    }
  }

  if (C->getType()->getTypeID() == Type::PointerTyID)
    if (const GlobalVariable* GV = dyn_cast<GlobalVariable>(C))
      if (GVSet.insert(GV))
        Size = addSizeOfGlobal(GV, Size);

  return Size;
}

/// addSizeOfGLobalsInInitializer - handle any globals that we haven't seen yet
/// but are referenced from the given initializer.

unsigned JITEmitter::addSizeOfGlobalsInInitializer(const Constant *Init,
                                              unsigned Size) {
  if (!isa<UndefValue>(Init) &&
      !isa<ConstantVector>(Init) &&
      !isa<ConstantAggregateZero>(Init) &&
      !isa<ConstantArray>(Init) &&
      !isa<ConstantStruct>(Init) &&
      Init->getType()->isFirstClassType())
    Size = addSizeOfGlobalsInConstantVal(Init, Size);
  return Size;
}

/// GetSizeOfGlobalsInBytes - walk the code for the function, looking for
/// globals; then walk the initializers of those globals looking for more.
/// If their size has not been considered yet, add it into the running total
/// Size.

unsigned JITEmitter::GetSizeOfGlobalsInBytes(MachineFunction &MF) {
  unsigned Size = 0;
  GVSet.clear();

  for (MachineFunction::iterator MBB = MF.begin(), E = MF.end();
       MBB != E; ++MBB) {
    for (MachineBasicBlock::const_iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      const TargetInstrDesc &Desc = I->getDesc();
      const MachineInstr &MI = *I;
      unsigned NumOps = Desc.getNumOperands();
      for (unsigned CurOp = 0; CurOp < NumOps; CurOp++) {
        const MachineOperand &MO = MI.getOperand(CurOp);
        if (MO.isGlobal()) {
          GlobalValue* V = MO.getGlobal();
          const GlobalVariable *GV = dyn_cast<const GlobalVariable>(V);
          if (!GV)
            continue;
          // If seen in previous function, it will have an entry here.
          if (TheJIT->getPointerToGlobalIfAvailable(GV))
            continue;
          // If seen earlier in this function, it will have an entry here.
          // FIXME: it should be possible to combine these tables, by
          // assuming the addresses of the new globals in this module
          // start at 0 (or something) and adjusting them after codegen
          // complete.  Another possibility is to grab a marker bit in GV.
          if (GVSet.insert(GV))
            // A variable as yet unseen.  Add in its size.
            Size = addSizeOfGlobal(GV, Size);
        }
      }
    }
  }
  DEBUG(dbgs() << "JIT: About to look through initializers\n");
  // Look for more globals that are referenced only from initializers.
  // GVSet.end is computed each time because the set can grow as we go.
  for (SmallPtrSet<const GlobalVariable *, 8>::iterator I = GVSet.begin();
       I != GVSet.end(); I++) {
    const GlobalVariable* GV = *I;
    if (GV->hasInitializer())
      Size = addSizeOfGlobalsInInitializer(GV->getInitializer(), Size);
  }

  return Size;
}

void JITEmitter::startFunction(MachineFunction &F) {
  DEBUG(dbgs() << "JIT: Starting CodeGen of Function "
        << F.getFunction()->getName() << "\n");

  uintptr_t ActualSize = 0;
  // Set the memory writable, if it's not already
  MemMgr->setMemoryWritable();
  if (MemMgr->NeedsExactSize()) {
    DEBUG(dbgs() << "JIT: ExactSize\n");
    const TargetInstrInfo* TII = F.getTarget().getInstrInfo();
    MachineConstantPool *MCP = F.getConstantPool();

    // Ensure the constant pool/jump table info is at least 4-byte aligned.
    ActualSize = RoundUpToAlign(ActualSize, 16);

    // Add the alignment of the constant pool
    ActualSize = RoundUpToAlign(ActualSize, MCP->getConstantPoolAlignment());

    // Add the constant pool size
    ActualSize += GetConstantPoolSizeInBytes(MCP, TheJIT->getTargetData());

    if (MachineJumpTableInfo *MJTI = F.getJumpTableInfo()) {
      // Add the aligment of the jump table info
      ActualSize = RoundUpToAlign(ActualSize,
                             MJTI->getEntryAlignment(*TheJIT->getTargetData()));

      // Add the jump table size
      ActualSize += GetJumpTableSizeInBytes(MJTI, TheJIT);
    }

    // Add the alignment for the function
    ActualSize = RoundUpToAlign(ActualSize,
                                std::max(F.getFunction()->getAlignment(), 8U));

    // Add the function size
    ActualSize += TII->GetFunctionSizeInBytes(F);

    DEBUG(dbgs() << "JIT: ActualSize before globals " << ActualSize << "\n");
    // Add the size of the globals that will be allocated after this function.
    // These are all the ones referenced from this function that were not
    // previously allocated.
    ActualSize += GetSizeOfGlobalsInBytes(F);
    DEBUG(dbgs() << "JIT: ActualSize after globals " << ActualSize << "\n");
  } else if (SizeEstimate > 0) {
    // SizeEstimate will be non-zero on reallocation attempts.
    ActualSize = SizeEstimate;
  }

  BufferBegin = CurBufferPtr = MemMgr->startFunctionBody(F.getFunction(),
                                                         ActualSize);
  BufferEnd = BufferBegin+ActualSize;
  EmittedFunctions[F.getFunction()].FunctionBody = BufferBegin;

  // Ensure the constant pool/jump table info is at least 4-byte aligned.
  emitAlignment(16);

  emitConstantPool(F.getConstantPool());
  if (MachineJumpTableInfo *MJTI = F.getJumpTableInfo())
    initJumpTableInfo(MJTI);

  // About to start emitting the machine code for the function.
  emitAlignment(std::max(F.getFunction()->getAlignment(), 8U));
  TheJIT->updateGlobalMapping(F.getFunction(), CurBufferPtr);
  EmittedFunctions[F.getFunction()].Code = CurBufferPtr;

  MBBLocations.clear();

  EmissionDetails.MF = &F;
  EmissionDetails.LineStarts.clear();
}

bool JITEmitter::finishFunction(MachineFunction &F) {
  if (CurBufferPtr == BufferEnd) {
    // We must call endFunctionBody before retrying, because
    // deallocateMemForFunction requires it.
    MemMgr->endFunctionBody(F.getFunction(), BufferBegin, CurBufferPtr);
    retryWithMoreMemory(F);
    return true;
  }

  if (MachineJumpTableInfo *MJTI = F.getJumpTableInfo())
    emitJumpTableInfo(MJTI);

  // FnStart is the start of the text, not the start of the constant pool and
  // other per-function data.
  uint8_t *FnStart =
    (uint8_t *)TheJIT->getPointerToGlobalIfAvailable(F.getFunction());

  // FnEnd is the end of the function's machine code.
  uint8_t *FnEnd = CurBufferPtr;

  if (!Relocations.empty()) {
    CurFn = F.getFunction();
    NumRelos += Relocations.size();

    // Resolve the relocations to concrete pointers.
    for (unsigned i = 0, e = Relocations.size(); i != e; ++i) {
      MachineRelocation &MR = Relocations[i];
      void *ResultPtr = 0;
      if (!MR.letTargetResolve()) {
        if (MR.isExternalSymbol()) {
          ResultPtr = TheJIT->getPointerToNamedFunction(MR.getExternalSymbol(),
                                                        false);
          DEBUG(dbgs() << "JIT: Map \'" << MR.getExternalSymbol() << "\' to ["
                       << ResultPtr << "]\n");

          // If the target REALLY wants a stub for this function, emit it now.
          if (MR.mayNeedFarStub()) {
            ResultPtr = Resolver.getExternalFunctionStub(ResultPtr);
          }
        } else if (MR.isGlobalValue()) {
          ResultPtr = getPointerToGlobal(MR.getGlobalValue(),
                                         BufferBegin+MR.getMachineCodeOffset(),
                                         MR.mayNeedFarStub());
        } else if (MR.isIndirectSymbol()) {
          ResultPtr = getPointerToGVIndirectSym(
              MR.getGlobalValue(), BufferBegin+MR.getMachineCodeOffset());
        } else if (MR.isBasicBlock()) {
          ResultPtr = (void*)getMachineBasicBlockAddress(MR.getBasicBlock());
        } else if (MR.isConstantPoolIndex()) {
          ResultPtr = (void*)getConstantPoolEntryAddress(MR.getConstantPoolIndex());
        } else {
          assert(MR.isJumpTableIndex());
          ResultPtr=(void*)getJumpTableEntryAddress(MR.getJumpTableIndex());
        }

        MR.setResultPointer(ResultPtr);
      }

      // if we are managing the GOT and the relocation wants an index,
      // give it one
      if (MR.isGOTRelative() && MemMgr->isManagingGOT()) {
        unsigned idx = Resolver.getGOTIndexForAddr(ResultPtr);
        MR.setGOTIndex(idx);
        if (((void**)MemMgr->getGOTBase())[idx] != ResultPtr) {
          DEBUG(dbgs() << "JIT: GOT was out of date for " << ResultPtr
                       << " pointing at " << ((void**)MemMgr->getGOTBase())[idx]
                       << "\n");
          ((void**)MemMgr->getGOTBase())[idx] = ResultPtr;
        }
      }
    }

    CurFn = 0;
    TheJIT->getJITInfo().relocate(BufferBegin, &Relocations[0],
                                  Relocations.size(), MemMgr->getGOTBase());
  }

  // Update the GOT entry for F to point to the new code.
  if (MemMgr->isManagingGOT()) {
    unsigned idx = Resolver.getGOTIndexForAddr((void*)BufferBegin);
    if (((void**)MemMgr->getGOTBase())[idx] != (void*)BufferBegin) {
      DEBUG(dbgs() << "JIT: GOT was out of date for " << (void*)BufferBegin
                   << " pointing at " << ((void**)MemMgr->getGOTBase())[idx]
                   << "\n");
      ((void**)MemMgr->getGOTBase())[idx] = (void*)BufferBegin;
    }
  }

  // CurBufferPtr may have moved beyond FnEnd, due to memory allocation for
  // global variables that were referenced in the relocations.
  MemMgr->endFunctionBody(F.getFunction(), BufferBegin, CurBufferPtr);

  if (CurBufferPtr == BufferEnd) {
    retryWithMoreMemory(F);
    return true;
  } else {
    // Now that we've succeeded in emitting the function, reset the
    // SizeEstimate back down to zero.
    SizeEstimate = 0;
  }

  BufferBegin = CurBufferPtr = 0;
  NumBytes += FnEnd-FnStart;

  // Invalidate the icache if necessary.
  sys::Memory::InvalidateInstructionCache(FnStart, FnEnd-FnStart);

  TheJIT->NotifyFunctionEmitted(*F.getFunction(), FnStart, FnEnd-FnStart,
                                EmissionDetails);

  DEBUG(dbgs() << "JIT: Finished CodeGen of [" << (void*)FnStart
        << "] Function: " << F.getFunction()->getName()
        << ": " << (FnEnd-FnStart) << " bytes of text, "
        << Relocations.size() << " relocations\n");

  Relocations.clear();
  ConstPoolAddresses.clear();

  // Mark code region readable and executable if it's not so already.
  MemMgr->setMemoryExecutable();

  DEBUG(
    if (sys::hasDisassembler()) {
      dbgs() << "JIT: Disassembled code:\n";
      dbgs() << sys::disassembleBuffer(FnStart, FnEnd-FnStart,
                                       (uintptr_t)FnStart);
    } else {
      dbgs() << "JIT: Binary code:\n";
      uint8_t* q = FnStart;
      for (int i = 0; q < FnEnd; q += 4, ++i) {
        if (i == 4)
          i = 0;
        if (i == 0)
          dbgs() << "JIT: " << (long)(q - FnStart) << ": ";
        bool Done = false;
        for (int j = 3; j >= 0; --j) {
          if (q + j >= FnEnd)
            Done = true;
          else
            dbgs() << (unsigned short)q[j];
        }
        if (Done)
          break;
        dbgs() << ' ';
        if (i == 3)
          dbgs() << '\n';
      }
      dbgs()<< '\n';
    }
        );

  if (DwarfExceptionHandling || JITEmitDebugInfo) {
    uintptr_t ActualSize = 0;
    SavedBufferBegin = BufferBegin;
    SavedBufferEnd = BufferEnd;
    SavedCurBufferPtr = CurBufferPtr;

    if (MemMgr->NeedsExactSize()) {
      ActualSize = DE->GetDwarfTableSizeInBytes(F, *this, FnStart, FnEnd);
    }

    BufferBegin = CurBufferPtr = MemMgr->startExceptionTable(F.getFunction(),
                                                             ActualSize);
    BufferEnd = BufferBegin+ActualSize;
    EmittedFunctions[F.getFunction()].ExceptionTable = BufferBegin;
    uint8_t *EhStart;
    uint8_t *FrameRegister = DE->EmitDwarfTable(F, *this, FnStart, FnEnd,
                                                EhStart);
    MemMgr->endExceptionTable(F.getFunction(), BufferBegin, CurBufferPtr,
                              FrameRegister);
    uint8_t *EhEnd = CurBufferPtr;
    BufferBegin = SavedBufferBegin;
    BufferEnd = SavedBufferEnd;
    CurBufferPtr = SavedCurBufferPtr;

    if (DwarfExceptionHandling) {
      TheJIT->RegisterTable(FrameRegister);
    }

    if (JITEmitDebugInfo) {
      DebugInfo I;
      I.FnStart = FnStart;
      I.FnEnd = FnEnd;
      I.EhStart = EhStart;
      I.EhEnd = EhEnd;
      DR->RegisterFunction(F.getFunction(), I);
    }
  }

  if (MMI)
    MMI->EndFunction();

  return false;
}

void JITEmitter::retryWithMoreMemory(MachineFunction &F) {
  DEBUG(dbgs() << "JIT: Ran out of space for native code.  Reattempting.\n");
  Relocations.clear();  // Clear the old relocations or we'll reapply them.
  ConstPoolAddresses.clear();
  ++NumRetries;
  deallocateMemForFunction(F.getFunction());
  // Try again with at least twice as much free space.
  SizeEstimate = (uintptr_t)(2 * (BufferEnd - BufferBegin));
}

/// deallocateMemForFunction - Deallocate all memory for the specified
/// function body.  Also drop any references the function has to stubs.
/// May be called while the Function is being destroyed inside ~Value().
void JITEmitter::deallocateMemForFunction(const Function *F) {
  ValueMap<const Function *, EmittedCode, EmittedFunctionConfig>::iterator
    Emitted = EmittedFunctions.find(F);
  if (Emitted != EmittedFunctions.end()) {
    MemMgr->deallocateFunctionBody(Emitted->second.FunctionBody);
    MemMgr->deallocateExceptionTable(Emitted->second.ExceptionTable);
    TheJIT->NotifyFreeingMachineCode(Emitted->second.Code);

    EmittedFunctions.erase(Emitted);
  }

  // TODO: Do we need to unregister exception handling information from libgcc
  // here?

  if (JITEmitDebugInfo) {
    DR->UnregisterFunction(F);
  }

  // If the function did not reference any stubs, return.
  if (CurFnStubUses.find(F) == CurFnStubUses.end())
    return;

  // For each referenced stub, erase the reference to this function, and then
  // erase the list of referenced stubs.
  SmallVectorImpl<void *> &StubList = CurFnStubUses[F];
  for (unsigned i = 0, e = StubList.size(); i != e; ++i) {
    void *Stub = StubList[i];

    // If we already invalidated this stub for this function, continue.
    if (StubFnRefs.count(Stub) == 0)
      continue;

    SmallPtrSet<const Function *, 1> &FnRefs = StubFnRefs[Stub];
    FnRefs.erase(F);

    // If this function was the last reference to the stub, invalidate the stub
    // in the JITResolver.  Were there a memory manager deallocateStub routine,
    // we could call that at this point too.
    if (FnRefs.empty()) {
      DEBUG(dbgs() << "\nJIT: Invalidated Stub at [" << Stub << "]\n");
      StubFnRefs.erase(Stub);

      // Invalidate the stub.  If it is a GV stub, update the JIT's global
      // mapping for that GV to zero.
      GlobalValue *GV = Resolver.invalidateStub(Stub);
      if (GV) {
        TheJIT->updateGlobalMapping(GV, 0);
      }
    }
  }
  CurFnStubUses.erase(F);
}


void* JITEmitter::allocateSpace(uintptr_t Size, unsigned Alignment) {
  if (BufferBegin)
    return JITCodeEmitter::allocateSpace(Size, Alignment);

  // create a new memory block if there is no active one.
  // care must be taken so that BufferBegin is invalidated when a
  // block is trimmed
  BufferBegin = CurBufferPtr = MemMgr->allocateSpace(Size, Alignment);
  BufferEnd = BufferBegin+Size;
  return CurBufferPtr;
}

void* JITEmitter::allocateGlobal(uintptr_t Size, unsigned Alignment) {
  // Delegate this call through the memory manager.
  return MemMgr->allocateGlobal(Size, Alignment);
}

void JITEmitter::emitConstantPool(MachineConstantPool *MCP) {
  if (TheJIT->getJITInfo().hasCustomConstantPool())
    return;

  const std::vector<MachineConstantPoolEntry> &Constants = MCP->getConstants();
  if (Constants.empty()) return;

  unsigned Size = GetConstantPoolSizeInBytes(MCP, TheJIT->getTargetData());
  unsigned Align = MCP->getConstantPoolAlignment();
  ConstantPoolBase = allocateSpace(Size, Align);
  ConstantPool = MCP;

  if (ConstantPoolBase == 0) return;  // Buffer overflow.

  DEBUG(dbgs() << "JIT: Emitted constant pool at [" << ConstantPoolBase
               << "] (size: " << Size << ", alignment: " << Align << ")\n");

  // Initialize the memory for all of the constant pool entries.
  unsigned Offset = 0;
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    MachineConstantPoolEntry CPE = Constants[i];
    unsigned AlignMask = CPE.getAlignment() - 1;
    Offset = (Offset + AlignMask) & ~AlignMask;

    uintptr_t CAddr = (uintptr_t)ConstantPoolBase + Offset;
    ConstPoolAddresses.push_back(CAddr);
    if (CPE.isMachineConstantPoolEntry()) {
      // FIXME: add support to lower machine constant pool values into bytes!
      llvm_report_error("Initialize memory with machine specific constant pool"
                        "entry has not been implemented!");
    }
    TheJIT->InitializeMemory(CPE.Val.ConstVal, (void*)CAddr);
    DEBUG(dbgs() << "JIT:   CP" << i << " at [0x";
          dbgs().write_hex(CAddr) << "]\n");

    const Type *Ty = CPE.Val.ConstVal->getType();
    Offset += TheJIT->getTargetData()->getTypeAllocSize(Ty);
  }
}

void JITEmitter::initJumpTableInfo(MachineJumpTableInfo *MJTI) {
  if (TheJIT->getJITInfo().hasCustomJumpTables())
    return;

  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return;

  unsigned NumEntries = 0;
  for (unsigned i = 0, e = JT.size(); i != e; ++i)
    NumEntries += JT[i].MBBs.size();

  unsigned EntrySize = MJTI->getEntrySize(*TheJIT->getTargetData());

  // Just allocate space for all the jump tables now.  We will fix up the actual
  // MBB entries in the tables after we emit the code for each block, since then
  // we will know the final locations of the MBBs in memory.
  JumpTable = MJTI;
  JumpTableBase = allocateSpace(NumEntries * EntrySize,
                             MJTI->getEntryAlignment(*TheJIT->getTargetData()));
}

void JITEmitter::emitJumpTableInfo(MachineJumpTableInfo *MJTI) {
  if (TheJIT->getJITInfo().hasCustomJumpTables())
    return;

  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty() || JumpTableBase == 0) return;

  
  switch (MJTI->getEntryKind()) {
  case MachineJumpTableInfo::EK_BlockAddress: {
    // EK_BlockAddress - Each entry is a plain address of block, e.g.:
    //     .word LBB123
    assert(MJTI->getEntrySize(*TheJIT->getTargetData()) == sizeof(void*) &&
           "Cross JIT'ing?");
    
    // For each jump table, map each target in the jump table to the address of
    // an emitted MachineBasicBlock.
    intptr_t *SlotPtr = (intptr_t*)JumpTableBase;
    
    for (unsigned i = 0, e = JT.size(); i != e; ++i) {
      const std::vector<MachineBasicBlock*> &MBBs = JT[i].MBBs;
      // Store the address of the basic block for this jump table slot in the
      // memory we allocated for the jump table in 'initJumpTableInfo'
      for (unsigned mi = 0, me = MBBs.size(); mi != me; ++mi)
        *SlotPtr++ = getMachineBasicBlockAddress(MBBs[mi]);
    }
    break;
  }
      
  case MachineJumpTableInfo::EK_Custom32:
  case MachineJumpTableInfo::EK_GPRel32BlockAddress:
  case MachineJumpTableInfo::EK_LabelDifference32: {
    assert(MJTI->getEntrySize(*TheJIT->getTargetData()) == 4&&"Cross JIT'ing?");
    // For each jump table, place the offset from the beginning of the table
    // to the target address.
    int *SlotPtr = (int*)JumpTableBase;

    for (unsigned i = 0, e = JT.size(); i != e; ++i) {
      const std::vector<MachineBasicBlock*> &MBBs = JT[i].MBBs;
      // Store the offset of the basic block for this jump table slot in the
      // memory we allocated for the jump table in 'initJumpTableInfo'
      uintptr_t Base = (uintptr_t)SlotPtr;
      for (unsigned mi = 0, me = MBBs.size(); mi != me; ++mi) {
        uintptr_t MBBAddr = getMachineBasicBlockAddress(MBBs[mi]);
        /// FIXME: USe EntryKind instead of magic "getPICJumpTableEntry" hook.
        *SlotPtr++ = TheJIT->getJITInfo().getPICJumpTableEntry(MBBAddr, Base);
      }
    }
    break;
  }
  }
}

void JITEmitter::startGVStub(const GlobalValue* GV,
                             unsigned StubSize, unsigned Alignment) {
  SavedBufferBegin = BufferBegin;
  SavedBufferEnd = BufferEnd;
  SavedCurBufferPtr = CurBufferPtr;

  BufferBegin = CurBufferPtr = MemMgr->allocateStub(GV, StubSize, Alignment);
  BufferEnd = BufferBegin+StubSize+1;
}

void JITEmitter::startGVStub(void *Buffer, unsigned StubSize) {
  SavedBufferBegin = BufferBegin;
  SavedBufferEnd = BufferEnd;
  SavedCurBufferPtr = CurBufferPtr;

  BufferBegin = CurBufferPtr = (uint8_t *)Buffer;
  BufferEnd = BufferBegin+StubSize+1;
}

void JITEmitter::finishGVStub() {
  assert(CurBufferPtr != BufferEnd && "Stub overflowed allocated space.");
  NumBytes += getCurrentPCOffset();
  BufferBegin = SavedBufferBegin;
  BufferEnd = SavedBufferEnd;
  CurBufferPtr = SavedCurBufferPtr;
}

void *JITEmitter::allocIndirectGV(const GlobalValue *GV,
                                  const uint8_t *Buffer, size_t Size,
                                  unsigned Alignment) {
  uint8_t *IndGV = MemMgr->allocateStub(GV, Size, Alignment);
  memcpy(IndGV, Buffer, Size);
  return IndGV;
}

// getConstantPoolEntryAddress - Return the address of the 'ConstantNum' entry
// in the constant pool that was last emitted with the 'emitConstantPool'
// method.
//
uintptr_t JITEmitter::getConstantPoolEntryAddress(unsigned ConstantNum) const {
  assert(ConstantNum < ConstantPool->getConstants().size() &&
         "Invalid ConstantPoolIndex!");
  return ConstPoolAddresses[ConstantNum];
}

// getJumpTableEntryAddress - Return the address of the JumpTable with index
// 'Index' in the jumpp table that was last initialized with 'initJumpTableInfo'
//
uintptr_t JITEmitter::getJumpTableEntryAddress(unsigned Index) const {
  const std::vector<MachineJumpTableEntry> &JT = JumpTable->getJumpTables();
  assert(Index < JT.size() && "Invalid jump table index!");

  unsigned EntrySize = JumpTable->getEntrySize(*TheJIT->getTargetData());

  unsigned Offset = 0;
  for (unsigned i = 0; i < Index; ++i)
    Offset += JT[i].MBBs.size();

   Offset *= EntrySize;

  return (uintptr_t)((char *)JumpTableBase + Offset);
}

void JITEmitter::EmittedFunctionConfig::onDelete(
  JITEmitter *Emitter, const Function *F) {
  Emitter->deallocateMemForFunction(F);
}
void JITEmitter::EmittedFunctionConfig::onRAUW(
  JITEmitter *, const Function*, const Function*) {
  llvm_unreachable("The JIT doesn't know how to handle a"
                   " RAUW on a value it has emitted.");
}


//===----------------------------------------------------------------------===//
//  Public interface to this file
//===----------------------------------------------------------------------===//

JITCodeEmitter *JIT::createEmitter(JIT &jit, JITMemoryManager *JMM,
                                   TargetMachine &tm) {
  return new JITEmitter(jit, JMM, tm);
}

// getPointerToFunctionOrStub - If the specified function has been
// code-gen'd, return a pointer to the function.  If not, compile it, or use
// a stub to implement lazy compilation if available.
//
void *JIT::getPointerToFunctionOrStub(Function *F) {
  // If we have already code generated the function, just return the address.
  if (void *Addr = getPointerToGlobalIfAvailable(F))
    return Addr;

  // Get a stub if the target supports it.
  assert(isa<JITEmitter>(JCE) && "Unexpected MCE?");
  JITEmitter *JE = cast<JITEmitter>(getCodeEmitter());
  return JE->getJITResolver().getLazyFunctionStub(F);
}

void JIT::updateFunctionStub(Function *F) {
  // Get the empty stub we generated earlier.
  assert(isa<JITEmitter>(JCE) && "Unexpected MCE?");
  JITEmitter *JE = cast<JITEmitter>(getCodeEmitter());
  void *Stub = JE->getJITResolver().getLazyFunctionStub(F);
  void *Addr = getPointerToGlobalIfAvailable(F);
  assert(Addr != Stub && "Function must have non-stub address to be updated.");

  // Tell the target jit info to rewrite the stub at the specified address,
  // rather than creating a new one.
  TargetJITInfo::StubLayout layout = getJITInfo().getStubLayout();
  JE->startGVStub(Stub, layout.Size);
  getJITInfo().emitFunctionStub(F, Addr, *getCodeEmitter());
  JE->finishGVStub();
}

/// freeMachineCodeForFunction - release machine code memory for given Function.
///
void JIT::freeMachineCodeForFunction(Function *F) {
  // Delete translation for this from the ExecutionEngine, so it will get
  // retranslated next time it is used.
  updateGlobalMapping(F, 0);

  // Free the actual memory for the function body and related stuff.
  assert(isa<JITEmitter>(JCE) && "Unexpected MCE?");
  cast<JITEmitter>(JCE)->deallocateMemForFunction(F);
}
