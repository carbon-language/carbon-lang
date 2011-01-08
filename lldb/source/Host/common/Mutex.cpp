//===-- Mutex.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Mutex.h"

#if 0
// This logging is way too verbose to enable even for a log channel. 
// This logging can be enabled by changing the "#if 0", but should be
// reverted prior to checking in.
#include <cstdio>
#define DEBUG_LOG(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

using namespace lldb_private;

//----------------------------------------------------------------------
// Default constructor.
//
// This will create a scoped mutex locking object that doesn't have
// a mutex to lock. One will need to be provided using the Reset()
// method.
//----------------------------------------------------------------------
Mutex::Locker::Locker () :
    m_mutex_ptr(NULL)
{
}

//----------------------------------------------------------------------
// Constructor with a Mutex object.
//
// This will create a scoped mutex locking object that extracts the
// mutex owned by "m" and locks it.
//----------------------------------------------------------------------
Mutex::Locker::Locker (Mutex& m) :
    m_mutex_ptr(m.GetMutex())
{
    if (m_mutex_ptr)
        Mutex::Lock (m_mutex_ptr);
}

//----------------------------------------------------------------------
// Constructor with a Mutex object pointer.
//
// This will create a scoped mutex locking object that extracts the
// mutex owned by "m" and locks it.
//----------------------------------------------------------------------
Mutex::Locker::Locker (Mutex* m) :
    m_mutex_ptr(m ? m->GetMutex() : NULL)
{
    if (m_mutex_ptr)
        Mutex::Lock (m_mutex_ptr);
}

//----------------------------------------------------------------------
// Constructor with a raw pthread mutex object pointer.
//
// This will create a scoped mutex locking object that locks "mutex"
//----------------------------------------------------------------------
Mutex::Locker::Locker (pthread_mutex_t *mutex_ptr) :
    m_mutex_ptr(mutex_ptr)
{
    if (m_mutex_ptr)
        Mutex::Lock (m_mutex_ptr);
}

//----------------------------------------------------------------------
// Destructor
//
// Unlocks any owned mutex object (if it is valid).
//----------------------------------------------------------------------
Mutex::Locker::~Locker ()
{
    Reset();
}

//----------------------------------------------------------------------
// Unlock the current mutex in this object (if this owns a valid
// mutex) and lock the new "mutex" object if it is non-NULL.
//----------------------------------------------------------------------
void
Mutex::Locker::Reset (pthread_mutex_t *mutex_ptr)
{
    // We already have this mutex locked or both are NULL...
    if (m_mutex_ptr == mutex_ptr)
        return;

    if (m_mutex_ptr)
        Mutex::Unlock (m_mutex_ptr);

    m_mutex_ptr = mutex_ptr;
    if (m_mutex_ptr)
        Mutex::Lock (m_mutex_ptr);
}

bool
Mutex::Locker::TryLock (pthread_mutex_t *mutex_ptr)
{
    // We already have this mutex locked!
    if (m_mutex_ptr == mutex_ptr)
        return true;

    Reset ();

    if (mutex_ptr)
    {
        if (Mutex::TryLock (mutex_ptr) == 0)
            m_mutex_ptr = mutex_ptr;
    }
    return m_mutex_ptr != NULL;
}

//----------------------------------------------------------------------
// Default constructor.
//
// Creates a pthread mutex with no attributes.
//----------------------------------------------------------------------
Mutex::Mutex () :
    m_mutex()
{
    int err;
    err = ::pthread_mutex_init (&m_mutex, NULL);
    assert(err == 0);
}

//----------------------------------------------------------------------
// Default constructor.
//
// Creates a pthread mutex with "type" as the mutex type.
//----------------------------------------------------------------------
Mutex::Mutex (Mutex::Type type) :
    m_mutex()
{
    int err;
    ::pthread_mutexattr_t attr;
    err = ::pthread_mutexattr_init (&attr);
    assert(err == 0);
    switch (type)
    {
    case eMutexTypeNormal:
        err = ::pthread_mutexattr_settype (&attr, PTHREAD_MUTEX_NORMAL);
        break;

    case eMutexTypeRecursive:
        err = ::pthread_mutexattr_settype (&attr, PTHREAD_MUTEX_RECURSIVE);
        break;

    default:
        err = -1;
        break;
    }
    assert(err == 0);
    err = ::pthread_mutex_init (&m_mutex, &attr);
    assert(err == 0);
    err = ::pthread_mutexattr_destroy (&attr);
    assert(err == 0);
}

//----------------------------------------------------------------------
// Destructor.
//
// Destroys the mutex owned by this object.
//----------------------------------------------------------------------
Mutex::~Mutex()
{
    int err;
    err = ::pthread_mutex_destroy (&m_mutex);
}

//----------------------------------------------------------------------
// Mutex get accessor.
//----------------------------------------------------------------------
pthread_mutex_t *
Mutex::GetMutex()
{
    return &m_mutex;
}

int
Mutex::Lock (pthread_mutex_t *mutex_ptr)
{
    DEBUG_LOG ("[%4.4x/%4.4x] pthread_mutex_lock (%p)...\n", Host::GetCurrentProcessID(), Host::GetCurrentThreadID(), mutex_ptr);
    int err = ::pthread_mutex_lock (mutex_ptr);
    DEBUG_LOG ("[%4.4x/%4.4x] pthread_mutex_lock (%p) => %i\n", Host::GetCurrentProcessID(), Host::GetCurrentThreadID(), mutex_ptr, err);
    return err;
}

int
Mutex::TryLock (pthread_mutex_t *mutex_ptr)
{
    int err = ::pthread_mutex_trylock (mutex_ptr);
    DEBUG_LOG ("[%4.4x/%4.4x] pthread_mutex_trylock (%p) => %i\n", Host::GetCurrentProcessID(), Host::GetCurrentThreadID(), mutex_ptr, err);
    return err;
}

int
Mutex::Unlock (pthread_mutex_t *mutex_ptr)
{
    int err = ::pthread_mutex_unlock (mutex_ptr);
    DEBUG_LOG ("[%4.4x/%4.4x] pthread_mutex_unlock (%p) => %i\n", Host::GetCurrentProcessID(), Host::GetCurrentThreadID(), mutex_ptr, err);
    return err;
}

//----------------------------------------------------------------------
// Locks the mutex owned by this object, if the mutex is already
// locked, the calling thread will block until the mutex becomes
// available.
//
// RETURNS
//  The error code from the pthread_mutex_lock() function call.
//----------------------------------------------------------------------
int
Mutex::Lock()
{
    return Mutex::Lock (&m_mutex);
}

//----------------------------------------------------------------------
// Attempts to lock the mutex owned by this object without blocking.
// If the mutex is already locked, TryLock() will not block waiting
// for the mutex, but will return an error condition.
//
// RETURNS
//  The error code from the pthread_mutex_trylock() function call.
//----------------------------------------------------------------------
int
Mutex::TryLock()
{
    return Mutex::TryLock (&m_mutex);
}

//----------------------------------------------------------------------
// If the current thread holds the lock on the owned mutex, then
// Unlock() will unlock the mutex. Calling Unlock() on this object
// that the calling thread does not hold will result in undefined
// behavior.
//
// RETURNS
//  The error code from the pthread_mutex_unlock() function call.
//----------------------------------------------------------------------
int
Mutex::Unlock()
{
    return Mutex::Unlock (&m_mutex);
}
