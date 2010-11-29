//===- RWMutex.cpp - Reader/Writer Mutual Exclusion Lock --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the llvm::sys::RWMutex class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/Support/RWMutex.h"
#include <cstring>

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

#if !defined(ENABLE_THREADS) || ENABLE_THREADS == 0
// Define all methods as no-ops if threading is explicitly disabled
namespace llvm {
using namespace sys;
RWMutexImpl::RWMutexImpl() { }
RWMutexImpl::~RWMutexImpl() { }
bool RWMutexImpl::reader_acquire() { return true; }
bool RWMutexImpl::reader_release() { return true; }
bool RWMutexImpl::writer_acquire() { return true; }
bool RWMutexImpl::writer_release() { return true; }
}
#else

#if defined(HAVE_PTHREAD_H) && defined(HAVE_PTHREAD_RWLOCK_INIT)

#include <cassert>
#include <pthread.h>
#include <stdlib.h>

namespace llvm {
using namespace sys;


// This variable is useful for situations where the pthread library has been
// compiled with weak linkage for its interface symbols. This allows the
// threading support to be turned off by simply not linking against -lpthread.
// In that situation, the value of pthread_mutex_init will be 0 and
// consequently pthread_enabled will be false. In such situations, all the
// pthread operations become no-ops and the functions all return false. If
// pthread_rwlock_init does have an address, then rwlock support is enabled.
// Note: all LLVM tools will link against -lpthread if its available since it
//       is configured into the LIBS variable.
// Note: this line of code generates a warning if pthread_rwlock_init is not
//       declared with weak linkage. It's safe to ignore the warning.
static const bool pthread_enabled = true;

// Construct a RWMutex using pthread calls
RWMutexImpl::RWMutexImpl()
  : data_(0)
{
  if (pthread_enabled)
  {
    // Declare the pthread_rwlock data structures
    pthread_rwlock_t* rwlock =
      static_cast<pthread_rwlock_t*>(malloc(sizeof(pthread_rwlock_t)));

#ifdef __APPLE__
    // Workaround a bug/mis-feature in Darwin's pthread_rwlock_init.
    bzero(rwlock, sizeof(pthread_rwlock_t));
#endif

    // Initialize the rwlock
    int errorcode = pthread_rwlock_init(rwlock, NULL);
    (void)errorcode;
    assert(errorcode == 0);

    // Assign the data member
    data_ = rwlock;
  }
}

// Destruct a RWMutex
RWMutexImpl::~RWMutexImpl()
{
  if (pthread_enabled)
  {
    pthread_rwlock_t* rwlock = static_cast<pthread_rwlock_t*>(data_);
    assert(rwlock != 0);
    pthread_rwlock_destroy(rwlock);
    free(rwlock);
  }
}

bool
RWMutexImpl::reader_acquire()
{
  if (pthread_enabled)
  {
    pthread_rwlock_t* rwlock = static_cast<pthread_rwlock_t*>(data_);
    assert(rwlock != 0);

    int errorcode = pthread_rwlock_rdlock(rwlock);
    return errorcode == 0;
  } else return false;
}

bool
RWMutexImpl::reader_release()
{
  if (pthread_enabled)
  {
    pthread_rwlock_t* rwlock = static_cast<pthread_rwlock_t*>(data_);
    assert(rwlock != 0);

    int errorcode = pthread_rwlock_unlock(rwlock);
    return errorcode == 0;
  } else return false;
}

bool
RWMutexImpl::writer_acquire()
{
  if (pthread_enabled)
  {
    pthread_rwlock_t* rwlock = static_cast<pthread_rwlock_t*>(data_);
    assert(rwlock != 0);

    int errorcode = pthread_rwlock_wrlock(rwlock);
    return errorcode == 0;
  } else return false;
}

bool
RWMutexImpl::writer_release()
{
  if (pthread_enabled)
  {
    pthread_rwlock_t* rwlock = static_cast<pthread_rwlock_t*>(data_);
    assert(rwlock != 0);

    int errorcode = pthread_rwlock_unlock(rwlock);
    return errorcode == 0;
  } else return false;
}

}

#elif defined(LLVM_ON_UNIX)
#include "Unix/RWMutex.inc"
#elif defined( LLVM_ON_WIN32)
#include "Windows/RWMutex.inc"
#else
#warning Neither LLVM_ON_UNIX nor LLVM_ON_WIN32 was set in System/Mutex.cpp
#endif
#endif
