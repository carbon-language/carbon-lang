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
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRelocation.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetJITInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/System/Memory.h"
#include <algorithm>
#include <iostream>
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
  /// MemoryRangeHeader - For a range of memory, this is the header that we put
  /// on the block of memory.  It is carefully crafted to be one word of memory.
  /// Allocated blocks have just this header, free'd blocks have FreeRangeHeader
  /// which starts with this.
  struct FreeRangeHeader;
  struct MemoryRangeHeader {
    /// ThisAllocated - This is true if this block is currently allocated.  If
    /// not, this can be converted to a FreeRangeHeader.
    intptr_t ThisAllocated : 1;
    
    /// PrevAllocated - Keep track of whether the block immediately before us is
    /// allocated.  If not, the word immediately before this header is the size
    /// of the previous block.
    intptr_t PrevAllocated : 1;
    
    /// BlockSize - This is the size in bytes of this memory block,
    /// including this header.
    uintptr_t BlockSize : (sizeof(intptr_t)*8 - 2);
    

    /// getBlockAfter - Return the memory block immediately after this one.
    ///
    MemoryRangeHeader &getBlockAfter() const {
      return *(MemoryRangeHeader*)((char*)this+BlockSize);
    }
    
    /// getFreeBlockBefore - If the block before this one is free, return it,
    /// otherwise return null.
    FreeRangeHeader *getFreeBlockBefore() const {
      if (PrevAllocated) return 0;
      intptr_t PrevSize = ((intptr_t *)this)[-1];
      return (FreeRangeHeader*)((char*)this-PrevSize);
    }
    
    /// FreeBlock - Turn an allocated block into a free block, adjusting
    /// bits in the object headers, and adding an end of region memory block.
    FreeRangeHeader *FreeBlock(FreeRangeHeader *FreeList);
    
    /// TrimAllocationToSize - If this allocated block is significantly larger
    /// than NewSize, split it into two pieces (where the former is NewSize
    /// bytes, including the header), and add the new block to the free list.
    FreeRangeHeader *TrimAllocationToSize(FreeRangeHeader *FreeList, 
                                          uint64_t NewSize);
  };

  /// FreeRangeHeader - For a memory block that isn't already allocated, this
  /// keeps track of the current block and has a pointer to the next free block.
  /// Free blocks are kept on a circularly linked list.
  struct FreeRangeHeader : public MemoryRangeHeader {
    FreeRangeHeader *Prev;
    FreeRangeHeader *Next;
    
    /// getMinBlockSize - Get the minimum size for a memory block.  Blocks
    /// smaller than this size cannot be created.
    static unsigned getMinBlockSize() {
      return sizeof(FreeRangeHeader)+sizeof(intptr_t);
    }
    
    /// SetEndOfBlockSizeMarker - The word at the end of every free block is
    /// known to be the size of the free block.  Set it for this block.
    void SetEndOfBlockSizeMarker() {
      void *EndOfBlock = (char*)this + BlockSize;
      ((intptr_t *)EndOfBlock)[-1] = BlockSize;
    }

    FreeRangeHeader *RemoveFromFreeList() {
      assert(Next->Prev == this && Prev->Next == this && "Freelist broken!");
      Next->Prev = Prev;
      return Prev->Next = Next;
    }
    
    void AddToFreeList(FreeRangeHeader *FreeList) {
      Next = FreeList;
      Prev = FreeList->Prev;
      Prev->Next = this;
      Next->Prev = this;
    }

    /// GrowBlock - The block after this block just got deallocated.  Merge it
    /// into the current block.
    void GrowBlock(uintptr_t NewSize);
    
    /// AllocateBlock - Mark this entire block allocated, updating freelists
    /// etc.  This returns a pointer to the circular free-list.
    FreeRangeHeader *AllocateBlock();
  };
}


/// AllocateBlock - Mark this entire block allocated, updating freelists
/// etc.  This returns a pointer to the circular free-list.
FreeRangeHeader *FreeRangeHeader::AllocateBlock() {
  assert(!ThisAllocated && !getBlockAfter().PrevAllocated &&
         "Cannot allocate an allocated block!");
  // Mark this block allocated.
  ThisAllocated = 1;
  getBlockAfter().PrevAllocated = 1;
 
  // Remove it from the free list.
  return RemoveFromFreeList();
}

/// FreeBlock - Turn an allocated block into a free block, adjusting
/// bits in the object headers, and adding an end of region memory block.
/// If possible, coallesce this block with neighboring blocks.  Return the
/// FreeRangeHeader to allocate from.
FreeRangeHeader *MemoryRangeHeader::FreeBlock(FreeRangeHeader *FreeList) {
  MemoryRangeHeader *FollowingBlock = &getBlockAfter();
  assert(ThisAllocated && "This block is already allocated!");
  assert(FollowingBlock->PrevAllocated && "Flags out of sync!");
  
  FreeRangeHeader *FreeListToReturn = FreeList;
  
  // If the block after this one is free, merge it into this block.
  if (!FollowingBlock->ThisAllocated) {
    FreeRangeHeader &FollowingFreeBlock = *(FreeRangeHeader *)FollowingBlock;
    // "FreeList" always needs to be a valid free block.  If we're about to
    // coallesce with it, update our notion of what the free list is.
    if (&FollowingFreeBlock == FreeList) {
      FreeList = FollowingFreeBlock.Next;
      FreeListToReturn = 0;
      assert(&FollowingFreeBlock != FreeList && "No tombstone block?");
    }
    FollowingFreeBlock.RemoveFromFreeList();
    
    // Include the following block into this one.
    BlockSize += FollowingFreeBlock.BlockSize;
    FollowingBlock = &FollowingFreeBlock.getBlockAfter();
    
    // Tell the block after the block we are coallescing that this block is
    // allocated.
    FollowingBlock->PrevAllocated = 1;
  }
  
  assert(FollowingBlock->ThisAllocated && "Missed coallescing?");
  
  if (FreeRangeHeader *PrevFreeBlock = getFreeBlockBefore()) {
    PrevFreeBlock->GrowBlock(PrevFreeBlock->BlockSize + BlockSize);
    return FreeListToReturn ? FreeListToReturn : PrevFreeBlock;
  }

  // Otherwise, mark this block free.
  FreeRangeHeader &FreeBlock = *(FreeRangeHeader*)this;
  FollowingBlock->PrevAllocated = 0;
  FreeBlock.ThisAllocated = 0;

  // Link this into the linked list of free blocks.
  FreeBlock.AddToFreeList(FreeList);

  // Add a marker at the end of the block, indicating the size of this free
  // block.
  FreeBlock.SetEndOfBlockSizeMarker();
  return FreeListToReturn ? FreeListToReturn : &FreeBlock;
}

/// GrowBlock - The block after this block just got deallocated.  Merge it
/// into the current block.
void FreeRangeHeader::GrowBlock(uintptr_t NewSize) {
  assert(NewSize > BlockSize && "Not growing block?");
  BlockSize = NewSize;
  SetEndOfBlockSizeMarker();
  getBlockAfter().PrevAllocated = 0;
}

/// TrimAllocationToSize - If this allocated block is significantly larger
/// than NewSize, split it into two pieces (where the former is NewSize
/// bytes, including the header), and add the new block to the free list.
FreeRangeHeader *MemoryRangeHeader::
TrimAllocationToSize(FreeRangeHeader *FreeList, uint64_t NewSize) {
  assert(ThisAllocated && getBlockAfter().PrevAllocated &&
         "Cannot deallocate part of an allocated block!");

  // Round up size for alignment of header.
  unsigned HeaderAlign = __alignof(FreeRangeHeader);
  NewSize = (NewSize+ (HeaderAlign-1)) & ~(HeaderAlign-1);
  
  // Size is now the size of the block we will remove from the start of the
  // current block.
  assert(NewSize <= BlockSize &&
         "Allocating more space from this block than exists!");
  
  // If splitting this block will cause the remainder to be too small, do not
  // split the block.
  if (BlockSize <= NewSize+FreeRangeHeader::getMinBlockSize())
    return FreeList;
  
  // Otherwise, we splice the required number of bytes out of this block, form
  // a new block immediately after it, then mark this block allocated.
  MemoryRangeHeader &FormerNextBlock = getBlockAfter();
  
  // Change the size of this block.
  BlockSize = NewSize;
  
  // Get the new block we just sliced out and turn it into a free block.
  FreeRangeHeader &NewNextBlock = (FreeRangeHeader &)getBlockAfter();
  NewNextBlock.BlockSize = (char*)&FormerNextBlock - (char*)&NewNextBlock;
  NewNextBlock.ThisAllocated = 0;
  NewNextBlock.PrevAllocated = 1;
  NewNextBlock.SetEndOfBlockSizeMarker();
  FormerNextBlock.PrevAllocated = 0;
  NewNextBlock.AddToFreeList(FreeList);
  return &NewNextBlock;
}

 
namespace {  
  /// JITMemoryManager - Manage memory for the JIT code generation in a logical,
  /// sane way.  This splits a large block of MAP_NORESERVE'd memory into two
  /// sections, one for function stubs, one for the functions themselves.  We
  /// have to do this because we may need to emit a function stub while in the
  /// middle of emitting a function, and we don't know how large the function we
  /// are emitting is.  This never bothers to release the memory, because when
  /// we are ready to destroy the JIT, the program exits.
  class JITMemoryManager {
    std::vector<sys::MemoryBlock> Blocks; // Memory blocks allocated by the JIT
    FreeRangeHeader *FreeMemoryList;      // Circular list of free blocks.
    
    // When emitting code into a memory block, this is the block.
    MemoryRangeHeader *CurBlock;
    
    unsigned char *CurStubPtr, *StubBase;
    unsigned char *GOTBase;      // Target Specific reserved memory

    // Centralize memory block allocation.
    sys::MemoryBlock getNewMemoryBlock(unsigned size);
    
    std::map<const Function*, MemoryRangeHeader*> FunctionBlocks;
  public:
    JITMemoryManager(bool useGOT);
    ~JITMemoryManager();

    inline unsigned char *allocateStub(unsigned StubSize);
    
    /// startFunctionBody - When a function starts, allocate a block of free
    /// executable memory, returning a pointer to it and its actual size.
    unsigned char *startFunctionBody(uintptr_t &ActualSize) {
      CurBlock = FreeMemoryList;
      
      // Allocate the entire memory block.
      FreeMemoryList = FreeMemoryList->AllocateBlock();
      ActualSize = CurBlock->BlockSize-sizeof(MemoryRangeHeader);
      return (unsigned char *)(CurBlock+1);
    }
    
    /// endFunctionBody - The function F is now allocated, and takes the memory
    /// in the range [FunctionStart,FunctionEnd).
    void endFunctionBody(const Function *F, unsigned char *FunctionStart,
                         unsigned char *FunctionEnd) {
      assert(FunctionEnd > FunctionStart);
      assert(FunctionStart == (unsigned char *)(CurBlock+1) &&
             "Mismatched function start/end!");
      
      uintptr_t BlockSize = FunctionEnd - (unsigned char *)CurBlock;
      FunctionBlocks[F] = CurBlock;

      // Release the memory at the end of this block that isn't needed.
      FreeMemoryList =CurBlock->TrimAllocationToSize(FreeMemoryList, BlockSize);
    }
    
    unsigned char *getGOTBase() const {
      return GOTBase;
    }
    bool isManagingGOT() const {
      return GOTBase != NULL;
    }
    
    /// deallocateMemForFunction - Deallocate all memory for the specified
    /// function body.
    void deallocateMemForFunction(const Function *F) {
      std::map<const Function*, MemoryRangeHeader*>::iterator
        I = FunctionBlocks.find(F);
      if (I == FunctionBlocks.end()) return;
      
      // Find the block that is allocated for this function.
      MemoryRangeHeader *MemRange = I->second;
      assert(MemRange->ThisAllocated && "Block isn't allocated!");
      
      // Fill the buffer with garbage!
      DEBUG(memset(MemRange+1, 0xCD, MemRange->BlockSize-sizeof(*MemRange)));
      
      // Free the memory.
      FreeMemoryList = MemRange->FreeBlock(FreeMemoryList);
      
      // Finally, remove this entry from FunctionBlocks.
      FunctionBlocks.erase(I);
    }
  };
}

JITMemoryManager::JITMemoryManager(bool useGOT) {
  // Allocate a 16M block of memory for functions.
  sys::MemoryBlock MemBlock = getNewMemoryBlock(16 << 20);

  unsigned char *MemBase = reinterpret_cast<unsigned char*>(MemBlock.base());

  // Allocate stubs backwards from the base, allocate functions forward
  // from the base.
  StubBase   = MemBase;
  CurStubPtr = MemBase + 512*1024; // Use 512k for stubs, working backwards.
  
  // We set up the memory chunk with 4 mem regions, like this:
  //  [ START
  //    [ Free      #0 ] -> Large space to allocate functions from.
  //    [ Allocated #1 ] -> Tiny space to separate regions.
  //    [ Free      #2 ] -> Tiny space so there is always at least 1 free block.
  //    [ Allocated #3 ] -> Tiny space to prevent looking past end of block.
  //  END ]
  //
  // The last three blocks are never deallocated or touched.
  
  // Add MemoryRangeHeader to the end of the memory region, indicating that
  // the space after the block of memory is allocated.  This is block #3.
  MemoryRangeHeader *Mem3 = (MemoryRangeHeader*)(MemBase+MemBlock.size())-1;
  Mem3->ThisAllocated = 1;
  Mem3->PrevAllocated = 0;
  Mem3->BlockSize     = 0;
  
  /// Add a tiny free region so that the free list always has one entry.
  FreeRangeHeader *Mem2 = 
    (FreeRangeHeader *)(((char*)Mem3)-FreeRangeHeader::getMinBlockSize());
  Mem2->ThisAllocated = 0;
  Mem2->PrevAllocated = 1;
  Mem2->BlockSize     = FreeRangeHeader::getMinBlockSize();
  Mem2->SetEndOfBlockSizeMarker();
  Mem2->Prev = Mem2;   // Mem2 *is* the free list for now.
  Mem2->Next = Mem2;

  /// Add a tiny allocated region so that Mem2 is never coallesced away.
  MemoryRangeHeader *Mem1 = (MemoryRangeHeader*)Mem2-1;
  Mem1->ThisAllocated = 1;
  Mem1->PrevAllocated = 0;
  Mem1->BlockSize     = (char*)Mem2 - (char*)Mem1;
  
  // Add a FreeRangeHeader to the start of the function body region, indicating
  // that the space is free.  Mark the previous block allocated so we never look
  // at it.
  FreeRangeHeader *Mem0 = (FreeRangeHeader*)CurStubPtr;
  Mem0->ThisAllocated = 0;
  Mem0->PrevAllocated = 1;
  Mem0->BlockSize = (char*)Mem1-(char*)Mem0;
  Mem0->SetEndOfBlockSizeMarker();
  Mem0->AddToFreeList(Mem2);
  
  // Start out with the freelist pointing to Mem0.
  FreeMemoryList = Mem0;

  // Allocate the GOT.
  GOTBase = NULL;
  if (useGOT) GOTBase = new unsigned char[sizeof(void*) * 8192];
}

JITMemoryManager::~JITMemoryManager() {
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i)
    sys::Memory::ReleaseRWX(Blocks[i]);
  
  delete[] GOTBase;
  Blocks.clear();
}

unsigned char *JITMemoryManager::allocateStub(unsigned StubSize) {
  CurStubPtr -= StubSize;
  if (CurStubPtr < StubBase) {
    // FIXME: allocate a new block
    std::cerr << "JIT ran out of memory for function stubs!\n";
    abort();
  }
  return CurStubPtr;
}

sys::MemoryBlock JITMemoryManager::getNewMemoryBlock(unsigned size) {
  // Allocate a new block close to the last one.
  const sys::MemoryBlock *BOld = Blocks.empty() ? 0 : &Blocks.front();
  std::string ErrMsg;
  sys::MemoryBlock B = sys::Memory::AllocateRWX(size, BOld, &ErrMsg);
  if (B.base() == 0) {
    std::cerr << "Allocation failed when allocating new memory in the JIT\n";
    std::cerr << ErrMsg << "\n";
    abort();
  }
  Blocks.push_back(B);
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
      return (void*)(intptr_t)LazyResolverFn;
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

#if (defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)) && \
    defined(__APPLE__)
extern "C" void sys_icache_invalidate(const void *Addr, size_t len);
#endif

/// synchronizeICache - On some targets, the JIT emitted code must be
/// explicitly refetched to ensure correct execution.
static void synchronizeICache(const void *Addr, size_t len) {
#if (defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)) && \
    defined(__APPLE__)
  sys_icache_invalidate(Addr, len);
#endif
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
  void *Actual = (void*)(intptr_t)LazyResolverFn;
  if (F->isExternal() && F->hasExternalLinkage())
    Actual = TheJIT->getPointerToFunction(F);

  // Otherwise, codegen a new stub.  For now, the stub will call the lazy
  // resolver function.
  Stub = TheJIT->getJITInfo().emitFunctionStub(Actual, MCE);

  if (Actual != (void*)(intptr_t)LazyResolverFn) {
    // If we are getting the stub for an external function, we really want the
    // address of the stub in the GlobalAddressMap for the JIT, not the address
    // of the external function.
    TheJIT->updateGlobalMapping(F, Stub);
  }

  // Invalidate the icache if necessary.
  synchronizeICache(Stub, MCE.getCurrentPCValue()-(intptr_t)Stub);

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

  // Invalidate the icache if necessary.
  synchronizeICache(Stub, MCE.getCurrentPCValue()-(intptr_t)Stub);

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


//===----------------------------------------------------------------------===//
// JITEmitter code.
//
namespace {
  /// JITEmitter - The JIT implementation of the MachineCodeEmitter, which is
  /// used to output functions to memory for execution.
  class JITEmitter : public MachineCodeEmitter {
    JITMemoryManager MemMgr;

    // When outputting a function stub in the context of some other function, we
    // save BufferBegin/BufferEnd/CurBufferPtr here.
    unsigned char *SavedBufferBegin, *SavedBufferEnd, *SavedCurBufferPtr;

    /// Relocations - These are the relocations that the function needs, as
    /// emitted.
    std::vector<MachineRelocation> Relocations;
    
    /// MBBLocations - This vector is a mapping from MBB ID's to their address.
    /// It is filled in by the StartMachineBasicBlock callback and queried by
    /// the getMachineBasicBlockAddress callback.
    std::vector<intptr_t> MBBLocations;

    /// ConstantPool - The constant pool for the current function.
    ///
    MachineConstantPool *ConstantPool;

    /// ConstantPoolBase - A pointer to the first entry in the constant pool.
    ///
    void *ConstantPoolBase;

    /// ConstantPool - The constant pool for the current function.
    ///
    MachineJumpTableInfo *JumpTable;
    
    /// JumpTableBase - A pointer to the first entry in the jump table.
    ///
    void *JumpTableBase;
public:
    JITEmitter(JIT &jit) : MemMgr(jit.getJITInfo().needsGOT()) {
      TheJIT = &jit;
      DEBUG(if (MemMgr.isManagingGOT()) std::cerr << "JIT is managing a GOT\n");
    }

    virtual void startFunction(MachineFunction &F);
    virtual bool finishFunction(MachineFunction &F);
    
    void emitConstantPool(MachineConstantPool *MCP);
    void initJumpTableInfo(MachineJumpTableInfo *MJTI);
    void emitJumpTableInfo(MachineJumpTableInfo *MJTI);
    
    virtual void startFunctionStub(unsigned StubSize);
    virtual void* finishFunctionStub(const Function *F);

    virtual void addRelocation(const MachineRelocation &MR) {
      Relocations.push_back(MR);
    }
    
    virtual void StartMachineBasicBlock(MachineBasicBlock *MBB) {
      if (MBBLocations.size() <= (unsigned)MBB->getNumber())
        MBBLocations.resize((MBB->getNumber()+1)*2);
      MBBLocations[MBB->getNumber()] = getCurrentPCValue();
    }

    virtual intptr_t getConstantPoolEntryAddress(unsigned Entry) const;
    virtual intptr_t getJumpTableEntryAddress(unsigned Entry) const;
    
    virtual intptr_t getMachineBasicBlockAddress(MachineBasicBlock *MBB) const {
      assert(MBBLocations.size() > (unsigned)MBB->getNumber() && 
             MBBLocations[MBB->getNumber()] && "MBB not emitted!");
      return MBBLocations[MBB->getNumber()];
    }

    /// deallocateMemForFunction - Deallocate all memory for the specified
    /// function body.
    void deallocateMemForFunction(Function *F) {
      MemMgr.deallocateMemForFunction(F);
    }
  private:
    void *getPointerToGlobal(GlobalValue *GV, void *Reference, bool NoNeedStub);
  };
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
  uintptr_t ActualSize;
  BufferBegin = CurBufferPtr = MemMgr.startFunctionBody(ActualSize);
  BufferEnd = BufferBegin+ActualSize;
  
  emitConstantPool(F.getConstantPool());
  initJumpTableInfo(F.getJumpTableInfo());

  // About to start emitting the machine code for the function.
  emitAlignment(std::max(F.getFunction()->getAlignment(), 8U));
  TheJIT->updateGlobalMapping(F.getFunction(), CurBufferPtr);

  MBBLocations.clear();
}

bool JITEmitter::finishFunction(MachineFunction &F) {
  if (CurBufferPtr == BufferEnd) {
    // FIXME: Allocate more space, then try again.
    std::cerr << "JIT: Ran out of space for generated machine code!\n";
    abort();
  }
  
  emitJumpTableInfo(F.getJumpTableInfo());
  
  // FnStart is the start of the text, not the start of the constant pool and
  // other per-function data.
  unsigned char *FnStart =
    (unsigned char *)TheJIT->getPointerToGlobalIfAvailable(F.getFunction());
  unsigned char *FnEnd   = CurBufferPtr;
  
  MemMgr.endFunctionBody(F.getFunction(), BufferBegin, FnEnd);
  NumBytes += FnEnd-FnStart;

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
      } else if (MR.isGlobalValue()) {
        ResultPtr = getPointerToGlobal(MR.getGlobalValue(),
                                       BufferBegin+MR.getMachineCodeOffset(),
                                       MR.doesntNeedFunctionStub());
      } else if (MR.isConstantPoolIndex()){
        assert(MR.isConstantPoolIndex());
        ResultPtr=(void*)getConstantPoolEntryAddress(MR.getConstantPoolIndex());
      } else {
        assert(MR.isJumpTableIndex());
        ResultPtr=(void*)getJumpTableEntryAddress(MR.getJumpTableIndex());

      }

      MR.setResultPointer(ResultPtr);

      // if we are managing the GOT and the relocation wants an index,
      // give it one
      if (MemMgr.isManagingGOT() && MR.isGOTRelative()) {
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

    TheJIT->getJITInfo().relocate(BufferBegin, &Relocations[0],
                                  Relocations.size(), MemMgr.getGOTBase());
  }

  // Update the GOT entry for F to point to the new code.
  if(MemMgr.isManagingGOT()) {
    unsigned idx = getJITResolver(this).getGOTIndexForAddr((void*)BufferBegin);
    if (((void**)MemMgr.getGOTBase())[idx] != (void*)BufferBegin) {
      DEBUG(std::cerr << "GOT was out of date for " << (void*)BufferBegin
            << " pointing at " << ((void**)MemMgr.getGOTBase())[idx] << "\n");
      ((void**)MemMgr.getGOTBase())[idx] = (void*)BufferBegin;
    }
  }

  // Resolve BasicaBlock references.
  TheJIT->getJITInfo().resolveBBRefs(*this);

  // Invalidate the icache if necessary.
  synchronizeICache(FnStart, FnEnd-FnStart);

  DEBUG(std::cerr << "JIT: Finished CodeGen of [" << (void*)FnStart
                  << "] Function: " << F.getFunction()->getName()
                  << ": " << (FnEnd-FnStart) << " bytes of text, "
                  << Relocations.size() << " relocations\n");
  Relocations.clear();
  return false;
}

void JITEmitter::emitConstantPool(MachineConstantPool *MCP) {
  const std::vector<MachineConstantPoolEntry> &Constants = MCP->getConstants();
  if (Constants.empty()) return;

  unsigned Size = Constants.back().Offset;
  Size += TheJIT->getTargetData()->getTypeSize(Constants.back().Val->getType());

  ConstantPoolBase = allocateSpace(Size, 1 << MCP->getConstantPoolAlignment());
  ConstantPool = MCP;

  if (ConstantPoolBase == 0) return;  // Buffer overflow.

  // Initialize the memory for all of the constant pool entries.
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    void *CAddr = (char*)ConstantPoolBase+Constants[i].Offset;
    TheJIT->InitializeMemory(Constants[i].Val, CAddr);
  }
}

void JITEmitter::initJumpTableInfo(MachineJumpTableInfo *MJTI) {
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return;
  
  unsigned NumEntries = 0;
  for (unsigned i = 0, e = JT.size(); i != e; ++i)
    NumEntries += JT[i].MBBs.size();

  unsigned EntrySize = MJTI->getEntrySize();

  // Just allocate space for all the jump tables now.  We will fix up the actual
  // MBB entries in the tables after we emit the code for each block, since then
  // we will know the final locations of the MBBs in memory.
  JumpTable = MJTI;
  JumpTableBase = allocateSpace(NumEntries * EntrySize, MJTI->getAlignment());
}

void JITEmitter::emitJumpTableInfo(MachineJumpTableInfo *MJTI) {
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty() || JumpTableBase == 0) return;

  unsigned Offset = 0;
  assert(MJTI->getEntrySize() == sizeof(void*) && "Cross JIT'ing?");
  
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
}

void JITEmitter::startFunctionStub(unsigned StubSize) {
  SavedBufferBegin = BufferBegin;
  SavedBufferEnd = BufferEnd;
  SavedCurBufferPtr = CurBufferPtr;
  
  BufferBegin = CurBufferPtr = MemMgr.allocateStub(StubSize);
  BufferEnd = BufferBegin+StubSize+1;
}

void *JITEmitter::finishFunctionStub(const Function *F) {
  NumBytes += getCurrentPCOffset();
  std::swap(SavedBufferBegin, BufferBegin);
  BufferEnd = SavedBufferEnd;
  CurBufferPtr = SavedCurBufferPtr;
  return SavedBufferBegin;
}

// getConstantPoolEntryAddress - Return the address of the 'ConstantNum' entry
// in the constant pool that was last emitted with the 'emitConstantPool'
// method.
//
intptr_t JITEmitter::getConstantPoolEntryAddress(unsigned ConstantNum) const {
  assert(ConstantNum < ConstantPool->getConstants().size() &&
         "Invalid ConstantPoolIndex!");
  return (intptr_t)ConstantPoolBase +
         ConstantPool->getConstants()[ConstantNum].Offset;
}

// getJumpTableEntryAddress - Return the address of the JumpTable with index
// 'Index' in the jumpp table that was last initialized with 'initJumpTableInfo'
//
intptr_t JITEmitter::getJumpTableEntryAddress(unsigned Index) const {
  const std::vector<MachineJumpTableEntry> &JT = JumpTable->getJumpTables();
  assert(Index < JT.size() && "Invalid jump table index!");
  
  unsigned Offset = 0;
  unsigned EntrySize = JumpTable->getEntrySize();
  
  for (unsigned i = 0; i < Index; ++i)
    Offset += JT[i].MBBs.size() * EntrySize;
  
  return (intptr_t)((char *)JumpTableBase + Offset);
}

//===----------------------------------------------------------------------===//
//  Public interface to this file
//===----------------------------------------------------------------------===//

MachineCodeEmitter *JIT::createEmitter(JIT &jit) {
  return new JITEmitter(jit);
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

/// freeMachineCodeForFunction - release machine code memory for given Function.
///
void JIT::freeMachineCodeForFunction(Function *F) {
  // Delete translation for this from the ExecutionEngine, so it will get
  // retranslated next time it is used.
  updateGlobalMapping(F, 0);

  // Free the actual memory for the function body and related stuff.
  assert(dynamic_cast<JITEmitter*>(MCE) && "Unexpected MCE?");
  dynamic_cast<JITEmitter*>(MCE)->deallocateMemForFunction(F);
}

