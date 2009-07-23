//===-- JITMemoryManager.cpp - Memory Allocator for JIT'd code ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DefaultJITMemoryManager class.
//
//===----------------------------------------------------------------------===//

#include "llvm/GlobalValue.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/System/Memory.h"
#include <map>
#include <vector>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace llvm;


JITMemoryManager::~JITMemoryManager() {}

//===----------------------------------------------------------------------===//
// Memory Block Implementation.
//===----------------------------------------------------------------------===//

namespace {
  /// MemoryRangeHeader - For a range of memory, this is the header that we put
  /// on the block of memory.  It is carefully crafted to be one word of memory.
  /// Allocated blocks have just this header, free'd blocks have FreeRangeHeader
  /// which starts with this.
  struct FreeRangeHeader;
  struct MemoryRangeHeader {
    /// ThisAllocated - This is true if this block is currently allocated.  If
    /// not, this can be converted to a FreeRangeHeader.
    unsigned ThisAllocated : 1;
    
    /// PrevAllocated - Keep track of whether the block immediately before us is
    /// allocated.  If not, the word immediately before this header is the size
    /// of the previous block.
    unsigned PrevAllocated : 1;
    
    /// BlockSize - This is the size in bytes of this memory block,
    /// including this header.
    uintptr_t BlockSize : (sizeof(intptr_t)*CHAR_BIT - 2);
    

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
/// If possible, coalesce this block with neighboring blocks.  Return the
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
    // coalesce with it, update our notion of what the free list is.
    if (&FollowingFreeBlock == FreeList) {
      FreeList = FollowingFreeBlock.Next;
      FreeListToReturn = 0;
      assert(&FollowingFreeBlock != FreeList && "No tombstone block?");
    }
    FollowingFreeBlock.RemoveFromFreeList();
    
    // Include the following block into this one.
    BlockSize += FollowingFreeBlock.BlockSize;
    FollowingBlock = &FollowingFreeBlock.getBlockAfter();
    
    // Tell the block after the block we are coalescing that this block is
    // allocated.
    FollowingBlock->PrevAllocated = 1;
  }
  
  assert(FollowingBlock->ThisAllocated && "Missed coalescing?");
  
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

  // Don't allow blocks to be trimmed below minimum required size
  NewSize = std::max<uint64_t>(FreeRangeHeader::getMinBlockSize(), NewSize);

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

//===----------------------------------------------------------------------===//
// Memory Block Implementation.
//===----------------------------------------------------------------------===//

namespace {  
  /// DefaultJITMemoryManager - Manage memory for the JIT code generation.
  /// This splits a large block of MAP_NORESERVE'd memory into two
  /// sections, one for function stubs, one for the functions themselves.  We
  /// have to do this because we may need to emit a function stub while in the
  /// middle of emitting a function, and we don't know how large the function we
  /// are emitting is.
  class VISIBILITY_HIDDEN DefaultJITMemoryManager : public JITMemoryManager {
    bool PoisonMemory;  // Whether to poison freed memory.

    std::vector<sys::MemoryBlock> Blocks; // Memory blocks allocated by the JIT
    FreeRangeHeader *FreeMemoryList;      // Circular list of free blocks.
    
    // When emitting code into a memory block, this is the block.
    MemoryRangeHeader *CurBlock;
    
    uint8_t *CurStubPtr, *StubBase;
    uint8_t *CurGlobalPtr, *GlobalEnd;
    uint8_t *GOTBase;     // Target Specific reserved memory
    void *DlsymTable;     // Stub external symbol information

    // Centralize memory block allocation.
    sys::MemoryBlock getNewMemoryBlock(unsigned size);
    
    std::map<const Function*, MemoryRangeHeader*> FunctionBlocks;
    std::map<const Function*, MemoryRangeHeader*> TableBlocks;
  public:
    DefaultJITMemoryManager();
    ~DefaultJITMemoryManager();

    void AllocateGOT();
    void SetDlsymTable(void *);
    
    uint8_t *allocateStub(const GlobalValue* F, unsigned StubSize,
                          unsigned Alignment);
    
    /// startFunctionBody - When a function starts, allocate a block of free
    /// executable memory, returning a pointer to it and its actual size.
    uint8_t *startFunctionBody(const Function *F, uintptr_t &ActualSize) {
      
      FreeRangeHeader* candidateBlock = FreeMemoryList;
      FreeRangeHeader* head = FreeMemoryList;
      FreeRangeHeader* iter = head->Next;

      uintptr_t largest = candidateBlock->BlockSize;
      
      // Search for the largest free block
      while (iter != head) {
          if (iter->BlockSize > largest) {
              largest = iter->BlockSize;
              candidateBlock = iter;
          }
          iter = iter->Next;
      }
      
      // Select this candidate block for allocation
      CurBlock = candidateBlock;

      // Allocate the entire memory block.
      FreeMemoryList = candidateBlock->AllocateBlock();
      ActualSize = CurBlock->BlockSize-sizeof(MemoryRangeHeader);
      return (uint8_t *)(CurBlock+1);
    }
    
    /// endFunctionBody - The function F is now allocated, and takes the memory
    /// in the range [FunctionStart,FunctionEnd).
    void endFunctionBody(const Function *F, uint8_t *FunctionStart,
                         uint8_t *FunctionEnd) {
      assert(FunctionEnd > FunctionStart);
      assert(FunctionStart == (uint8_t *)(CurBlock+1) &&
             "Mismatched function start/end!");

      uintptr_t BlockSize = FunctionEnd - (uint8_t *)CurBlock;
      FunctionBlocks[F] = CurBlock;

      // Release the memory at the end of this block that isn't needed.
      FreeMemoryList =CurBlock->TrimAllocationToSize(FreeMemoryList, BlockSize);
    }

    /// allocateSpace - Allocate a memory block of the given size.
    uint8_t *allocateSpace(intptr_t Size, unsigned Alignment) {
      CurBlock = FreeMemoryList;
      FreeMemoryList = FreeMemoryList->AllocateBlock();

      uint8_t *result = (uint8_t *)(CurBlock + 1);

      if (Alignment == 0) Alignment = 1;
      result = (uint8_t*)(((intptr_t)result+Alignment-1) &
               ~(intptr_t)(Alignment-1));

      uintptr_t BlockSize = result + Size - (uint8_t *)CurBlock;
      FreeMemoryList =CurBlock->TrimAllocationToSize(FreeMemoryList, BlockSize);

      return result;
    }

    /// allocateGlobal - Allocate memory for a global.  Unlike allocateSpace,
    /// this method does not touch the current block and can be called at any
    /// time.
    uint8_t *allocateGlobal(uintptr_t Size, unsigned Alignment) {
      uint8_t *Result = CurGlobalPtr;

      // Align the pointer.
      if (Alignment == 0) Alignment = 1;
      Result = (uint8_t*)(((uintptr_t)Result + Alignment-1) &
                          ~(uintptr_t)(Alignment-1));

      // Move the current global pointer forward.
      CurGlobalPtr += Result - CurGlobalPtr + Size;

      // Check for overflow.
      if (CurGlobalPtr > GlobalEnd) {
        // FIXME: Allocate more memory.
        llvm_report_error("JIT ran out of memory for globals!");
      }

      return Result;
    }

    /// startExceptionTable - Use startFunctionBody to allocate memory for the 
    /// function's exception table.
    uint8_t* startExceptionTable(const Function* F, uintptr_t &ActualSize) {
      return startFunctionBody(F, ActualSize);
    }

    /// endExceptionTable - The exception table of F is now allocated, 
    /// and takes the memory in the range [TableStart,TableEnd).
    void endExceptionTable(const Function *F, uint8_t *TableStart,
                           uint8_t *TableEnd, uint8_t* FrameRegister) {
      assert(TableEnd > TableStart);
      assert(TableStart == (uint8_t *)(CurBlock+1) &&
             "Mismatched table start/end!");
      
      uintptr_t BlockSize = TableEnd - (uint8_t *)CurBlock;
      TableBlocks[F] = CurBlock;

      // Release the memory at the end of this block that isn't needed.
      FreeMemoryList =CurBlock->TrimAllocationToSize(FreeMemoryList, BlockSize);
    }
    
    uint8_t *getGOTBase() const {
      return GOTBase;
    }
    
    void *getDlsymTable() const {
      return DlsymTable;
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
      if (PoisonMemory) {
        memset(MemRange+1, 0xCD, MemRange->BlockSize-sizeof(*MemRange));
      }

      // Free the memory.
      FreeMemoryList = MemRange->FreeBlock(FreeMemoryList);
      
      // Finally, remove this entry from FunctionBlocks.
      FunctionBlocks.erase(I);
      
      I = TableBlocks.find(F);
      if (I == TableBlocks.end()) return;
      
      // Find the block that is allocated for this function.
      MemRange = I->second;
      assert(MemRange->ThisAllocated && "Block isn't allocated!");

      // Fill the buffer with garbage!
      if (PoisonMemory) {
        memset(MemRange+1, 0xCD, MemRange->BlockSize-sizeof(*MemRange));
      }

      // Free the memory.
      FreeMemoryList = MemRange->FreeBlock(FreeMemoryList);
      
      // Finally, remove this entry from TableBlocks.
      TableBlocks.erase(I);
    }

    /// setMemoryWritable - When code generation is in progress,
    /// the code pages may need permissions changed.
    void setMemoryWritable(void)
    {
      for (unsigned i = 0, e = Blocks.size(); i != e; ++i)
        sys::Memory::setWritable(Blocks[i]);
    }
    /// setMemoryExecutable - When code generation is done and we're ready to
    /// start execution, the code pages may need permissions changed.
    void setMemoryExecutable(void)
    {
      for (unsigned i = 0, e = Blocks.size(); i != e; ++i)
        sys::Memory::setExecutable(Blocks[i]);
    }

    /// setPoisonMemory - Controls whether we write garbage over freed memory.
    ///
    void setPoisonMemory(bool poison) {
      PoisonMemory = poison;
    }
  };
}

DefaultJITMemoryManager::DefaultJITMemoryManager() {
#ifdef NDEBUG
  PoisonMemory = true;
#else
  PoisonMemory = false;
#endif

  // Allocate a 16M block of memory for functions.
#if defined(__APPLE__) && defined(__arm__)
  sys::MemoryBlock MemBlock = getNewMemoryBlock(4 << 20);
#else
  sys::MemoryBlock MemBlock = getNewMemoryBlock(16 << 20);
#endif

  uint8_t *MemBase = static_cast<uint8_t*>(MemBlock.base());

  // Allocate stubs backwards to the base, globals forward from the stubs, and
  // functions forward after globals.
  StubBase   = MemBase;
  CurStubPtr = MemBase + 512*1024; // Use 512k for stubs, working backwards.
  CurGlobalPtr = CurStubPtr;       // Use 2M for globals, working forwards.
  GlobalEnd = CurGlobalPtr + 2*1024*1024;

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

  /// Add a tiny allocated region so that Mem2 is never coalesced away.
  MemoryRangeHeader *Mem1 = (MemoryRangeHeader*)Mem2-1;
  Mem1->ThisAllocated = 1;
  Mem1->PrevAllocated = 0;
  Mem1->BlockSize     = (char*)Mem2 - (char*)Mem1;
  
  // Add a FreeRangeHeader to the start of the function body region, indicating
  // that the space is free.  Mark the previous block allocated so we never look
  // at it.
  FreeRangeHeader *Mem0 = (FreeRangeHeader*)GlobalEnd;
  Mem0->ThisAllocated = 0;
  Mem0->PrevAllocated = 1;
  Mem0->BlockSize = (char*)Mem1-(char*)Mem0;
  Mem0->SetEndOfBlockSizeMarker();
  Mem0->AddToFreeList(Mem2);
  
  // Start out with the freelist pointing to Mem0.
  FreeMemoryList = Mem0;

  GOTBase = NULL;
  DlsymTable = NULL;
}

void DefaultJITMemoryManager::AllocateGOT() {
  assert(GOTBase == 0 && "Cannot allocate the got multiple times");
  GOTBase = new uint8_t[sizeof(void*) * 8192];
  HasGOT = true;
}

void DefaultJITMemoryManager::SetDlsymTable(void *ptr) {
  DlsymTable = ptr;
}

DefaultJITMemoryManager::~DefaultJITMemoryManager() {
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i)
    sys::Memory::ReleaseRWX(Blocks[i]);
  
  delete[] GOTBase;
  Blocks.clear();
}

uint8_t *DefaultJITMemoryManager::allocateStub(const GlobalValue* F,
                                                     unsigned StubSize,
                                                     unsigned Alignment) {
  CurStubPtr -= StubSize;
  CurStubPtr = (uint8_t*)(((intptr_t)CurStubPtr) &
                          ~(intptr_t)(Alignment-1));
  if (CurStubPtr < StubBase) {
    // FIXME: allocate a new block
    llvm_report_error("JIT ran out of memory for function stubs!");
  }
  return CurStubPtr;
}

sys::MemoryBlock DefaultJITMemoryManager::getNewMemoryBlock(unsigned size) {
  // Allocate a new block close to the last one.
  const sys::MemoryBlock *BOld = Blocks.empty() ? 0 : &Blocks.back();
  std::string ErrMsg;
  sys::MemoryBlock B = sys::Memory::AllocateRWX(size, BOld, &ErrMsg);
  if (B.base() == 0) {
    llvm_report_error("Allocation failed when allocating new memory in the"
                      " JIT\n" + ErrMsg);
  }
  Blocks.push_back(B);
  return B;
}


JITMemoryManager *JITMemoryManager::CreateDefaultMemManager() {
  return new DefaultJITMemoryManager();
}
