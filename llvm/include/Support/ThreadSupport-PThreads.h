//===-- Support/Lock.h - Platform-agnostic mutual exclusion -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains classes that implement locks (mutual exclusion
// variables) in a platform-agnostic way. Basically the user should
// just call Lock::create() to get a Lock object of the correct sort
// for the current platform, and use its acquire() and release()
// methods, or a LockHolder, to protect critical sections of code for
// thread-safety.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_LOCK_H
#define SUPPORT_LOCK_H

#include <pthread.h>
#include <cstdlib>

namespace llvm {

/// Lock - Abstract class that provides mutual exclusion (also known
/// as a mutex.)
///
class Lock {
protected:
  virtual ~Lock() {}                        // Derive from me
public:
  virtual void acquire () { abort (); }
  virtual void release () { abort (); }

  /// create - Static method that returns a Lock of the correct class
  /// for the current host OS.
  ///
  static Lock create ();
};

/// POSIXLock - Specialization of Lock class implemented using
/// pthread_mutex_t objects.
///
class POSIXLock : public Lock {
  pthread_mutex_t mutex;
public:
  POSIXLock () { pthread_mutex_init (&mutex, 0); }
  virtual ~POSIXLock () { pthread_mutex_destroy (&mutex); }
  virtual void acquire () { pthread_mutex_lock (&mutex); }
  virtual void release () { pthread_mutex_unlock (&mutex); }
};

/// LockHolder - Instances of this class acquire a given Lock when
/// constructed and hold that lock until destruction.  Uncle Bjarne
/// says, "Resource acquisition is allocation." Or is it the other way
/// around? I never can remember.
///
class LockHolder {
  Lock &L;
public:
  LockHolder (Lock &_L) : L (_L) { L.acquire (); }
  ~LockHolder () { L.release (); }
};

} // end namespace llvm

#endif // SUPPORT_LOCK_H
