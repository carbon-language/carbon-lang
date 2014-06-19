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
#include "llvm/Support/Atomic.h"
#include <cassert>
#include <mutex>
using namespace llvm;

static const ManagedStaticBase *StaticList = nullptr;

// ManagedStatics can get created during execution of static constructors.  As a
// result, we cannot use a global static std::mutex object for the lock since it
// may not have been constructed.  Instead, we do a call-once initialization of
// a pointer to a mutex.
static std::once_flag MutexInitializationFlag;
static std::recursive_mutex* ManagedStaticMutex = nullptr;

// Not all supported platforms (in particular VS2012) have thread-safe function
// static initialization, so roll our own.
static std::recursive_mutex& GetManagedStaticMutex() {
  std::call_once(MutexInitializationFlag,
      []() { ManagedStaticMutex = new std::recursive_mutex(); } );

  return *ManagedStaticMutex;
}

void ManagedStaticBase::RegisterManagedStatic(void *(*Creator)(),
                                              void (*Deleter)(void*)) const {
  assert(Creator);
  if (llvm_is_multithreaded()) {
    std::lock_guard<std::recursive_mutex> Lock(GetManagedStaticMutex());

    if (!Ptr) {
      void* tmp = Creator();

      TsanHappensBefore(this);
      sys::MemoryFence();

      // This write is racy against the first read in the ManagedStatic
      // accessors. The race is benign because it does a second read after a
      // memory fence, at which point it isn't possible to get a partial value.
      TsanIgnoreWritesBegin();
      Ptr = tmp;
      TsanIgnoreWritesEnd();
      DeleterFn = Deleter;
      
      // Add to list of managed statics.
      Next = StaticList;
      StaticList = this;
    }
  } else {
    assert(!Ptr && !DeleterFn && !Next &&
           "Partially initialized ManagedStatic!?");
    Ptr = Creator();
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
  Next = nullptr;

  // Destroy memory.
  DeleterFn(Ptr);
  
  // Cleanup.
  Ptr = nullptr;
  DeleterFn = nullptr;
}

/// llvm_shutdown - Deallocate and destroy all ManagedStatic variables.
void llvm::llvm_shutdown() {
  std::lock_guard<std::recursive_mutex> Lock(GetManagedStaticMutex());

  while (StaticList)
    StaticList->destroy();

  if (llvm_is_multithreaded()) llvm_stop_multithreaded();
}
