//===-- Support/ThreadSupport-PThreads.h - PThreads support -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines pthreads implementations of the generic threading
// mechanisms.  Users should never #include this file directly!
//
//===----------------------------------------------------------------------===//

// Users should never #include this file directly!  As such, no include guards
// are needed.

#ifndef SUPPORT_THREADSUPPORT_H
#error "Code should not #include Support/ThreadSupport/PThreads.h directly!"
#endif

#include <pthread.h>

namespace llvm {

  /// Mutex - This class allows user code to protect variables shared between
  /// threads.  It implements a "recursive" mutex, to simplify user code.
  ///
  class Mutex {
    pthread_mutex_t mutex;
    Mutex(const Mutex &);           // DO NOT IMPLEMENT
    void operator=(const Mutex &);  // DO NOT IMPLEMENT
  public:
    Mutex() {
      pthread_mutexattr_t Attr;
      pthread_mutex_init(&mutex, &Attr);
    }
    ~Mutex() { pthread_mutex_destroy(&mutex); }
    void acquire () { pthread_mutex_lock (&mutex); }
    void release () { pthread_mutex_unlock (&mutex); }
  };
} // end namespace llvm
