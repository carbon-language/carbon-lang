//===- llvm/unittest/Support/ManagedStatic.cpp - ManagedStatic tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Threading.h"
#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#include "gtest/gtest.h"

using namespace llvm;

namespace {

#if defined(HAVE_PTHREAD_H) && !__has_feature(memory_sanitizer)
namespace test1 {
  llvm::ManagedStatic<int> ms;
  void *helper(void*) {
    *ms;
    return NULL;
  }

  // Valgrind's leak checker complains glibc's stack allocation.
  // To appease valgrind, we provide our own stack for each thread.
  void *allocate_stack(pthread_attr_t &a, size_t n = 65536) {
    void *stack = malloc(n);
    pthread_attr_init(&a);
#if defined(__linux__)
    pthread_attr_setstack(&a, stack, n);
#endif
    return stack;
  }
}

TEST(Initialize, MultipleThreads) {
  // Run this test under tsan: http://code.google.com/p/data-race-test/

  pthread_attr_t a1, a2;
  void *p1 = test1::allocate_stack(a1);
  void *p2 = test1::allocate_stack(a2);

  llvm_start_multithreaded();
  pthread_t t1, t2;
  pthread_create(&t1, &a1, test1::helper, NULL);
  pthread_create(&t2, &a2, test1::helper, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  free(p1);
  free(p2);
  llvm_stop_multithreaded();
}
#endif

} // anonymous namespace
