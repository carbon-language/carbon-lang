// RUN: %clangxx_tsan -O1 --std=c++11 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "custom_mutex.h"

#include <type_traits>

// Test that the destruction events of a mutex are ignored when the
// annotations request this.
//
// Use after destruction is UB, but __tsan_mutex_linker_init and
// __tsan_mutex_not_static exist to support global variables of mutex type,
// which might be accessed during program shutdown after the class's destructor
// has run.

int main() {
  std::aligned_storage<sizeof(Mutex), alignof(Mutex)>::type mu1_store;
  Mutex* mu1 = reinterpret_cast<Mutex*>(&mu1_store);
  new(&mu1_store) Mutex(false, __tsan_mutex_linker_init);
  mu1->Lock();
  mu1->~Mutex();
  mu1->Unlock();

  std::aligned_storage<sizeof(Mutex), alignof(Mutex)>::type mu2_store;
  Mutex* mu2 = reinterpret_cast<Mutex*>(&mu2_store);
  new(&mu2_store) Mutex(false, 0, __tsan_mutex_not_static);
  mu2->Lock();
  mu2->~Mutex();
  mu2->Unlock();

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
