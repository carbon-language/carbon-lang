//===-- llvm/Support/ThreadSupport-PThreads.h - PThreads support *- C++ -*-===//
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

#ifndef LLVM_SUPPORT_THREADSUPPORT_H
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
      // Initialize the mutex as a recursive mutex
      pthread_mutexattr_t Attr;
      int errorcode = pthread_mutexattr_init(&Attr);
      assert(errorcode == 0);

      errorcode = pthread_mutexattr_settype(&Attr, PTHREAD_MUTEX_RECURSIVE);
      assert(errorcode == 0);

      errorcode = pthread_mutex_init(&mutex, &Attr);
      assert(errorcode == 0);

      errorcode = pthread_mutexattr_destroy(&Attr);
      assert(errorcode == 0);
    }

    ~Mutex() {
      int errorcode = pthread_mutex_destroy(&mutex);
      assert(errorcode == 0);
    }

    void acquire () {
      int errorcode = pthread_mutex_lock(&mutex);
      assert(errorcode == 0);
    }

    void release () {
      int errorcode = pthread_mutex_unlock(&mutex);
      assert(errorcode == 0);
    }
  };
} // end namespace llvm
