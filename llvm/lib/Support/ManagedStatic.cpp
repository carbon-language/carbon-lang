//===-- ManagedStatic.cpp - Static Global wrapper -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ManagedStatic class and llvm_shutdown().
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ManagedStatic.h"
#include "llvm/Config/config.h"
#include "llvm/System/Atomic.h"
#include "llvm/System/Mutex.h"
#include <cassert>
using namespace llvm;

static const ManagedStaticBase *StaticList = 0;

static sys::Mutex* ManagedStaticMutex = 0;

void ManagedStaticBase::RegisterManagedStatic(void *(*Creator)(),
                                              void (*Deleter)(void*)) const {
  if (ManagedStaticMutex) {
    ManagedStaticMutex->acquire();

    if (Ptr == 0) {
      void* tmp = Creator ? Creator() : 0;

      sys::MemoryFence();
      Ptr = tmp;
      DeleterFn = Deleter;
      
      // Add to list of managed statics.
      Next = StaticList;
      StaticList = this;
    }

    ManagedStaticMutex->release();
  } else {
    assert(Ptr == 0 && DeleterFn == 0 && Next == 0 &&
	   "Partially initialized ManagedStatic!?");
    Ptr = Creator ? Creator() : 0;
    DeleterFn = Deleter;
  
    // Add to list of managed statics.
    Next = StaticList;
    StaticList = this;
  }
}

void ManagedStaticBase::destroy() const {
  assert(DeleterFn && "ManagedStatic not initialized correctly!");
  assert(StaticList == this &&
         "Not destroyed in reverse order of construction?");
  // Unlink from list.
  StaticList = Next;
  Next = 0;

  // Destroy memory.
  DeleterFn(Ptr);
  
  // Cleanup.
  Ptr = 0;
  DeleterFn = 0;
}

void llvm::llvm_start_multithreaded() {
#if LLVM_MULTITHREADED
  assert(ManagedStaticMutex == 0 && "Multithreaded LLVM already initialized!");
  ManagedStaticMutex = new sys::Mutex(true);
#else
  assert(0 && "LLVM built without multithreading support!");
#endif
}

/// llvm_shutdown - Deallocate and destroy all ManagedStatic variables.
void llvm::llvm_shutdown() {
  while (StaticList)
    StaticList->destroy();

  if (ManagedStaticMutex) {
    delete ManagedStaticMutex;
    ManagedStaticMutex = 0;
  }
}

