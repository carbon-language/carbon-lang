//===- ThreadLocal.cpp - Thread Local Data ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the llvm::sys::ThreadLocal class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/System/ThreadLocal.h"

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

#if !defined(ENABLE_THREADS) || ENABLE_THREADS == 0
// Define all methods as no-ops if threading is explicitly disabled
namespace llvm {
using namespace sys;
ThreadLocalImpl::ThreadLocalImpl() { }
ThreadLocalImpl::~ThreadLocalImpl() { }
void ThreadLocalImpl::setInstance(const void* d) { data = const_cast<void*>(d);}
const void* ThreadLocalImpl::getInstance() { return data; }
void ThreadLocalImpl::removeInstance() { data = 0; }
}
#else

#if defined(HAVE_PTHREAD_H) && defined(HAVE_PTHREAD_GETSPECIFIC)

#include <cassert>
#include <pthread.h>
#include <stdlib.h>

namespace llvm {
using namespace sys;

ThreadLocalImpl::ThreadLocalImpl() : data(0) {
  pthread_key_t* key = new pthread_key_t;
  int errorcode = pthread_key_create(key, NULL);
  assert(errorcode == 0);
  (void) errorcode;
  data = (void*)key;
}

ThreadLocalImpl::~ThreadLocalImpl() {
  pthread_key_t* key = static_cast<pthread_key_t*>(data);
  int errorcode = pthread_key_delete(*key);
  assert(errorcode == 0);
  (void) errorcode;
  delete key;
}

void ThreadLocalImpl::setInstance(const void* d) {
  pthread_key_t* key = static_cast<pthread_key_t*>(data);
  int errorcode = pthread_setspecific(*key, d);
  assert(errorcode == 0);
  (void) errorcode;
}

const void* ThreadLocalImpl::getInstance() {
  pthread_key_t* key = static_cast<pthread_key_t*>(data);
  return pthread_getspecific(*key);
}

void ThreadLocalImpl::removeInstance() {
  setInstance(0);
}

}

#elif defined(LLVM_ON_UNIX)
#include "Unix/ThreadLocal.inc"
#elif defined( LLVM_ON_WIN32)
#include "Win32/ThreadLocal.inc"
#else
#warning Neither LLVM_ON_UNIX nor LLVM_ON_WIN32 was set in System/ThreadLocal.cpp
#endif
#endif

