//===-- ProcessRunLock.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessRunLock_h_
#define liblldb_ProcessRunLock_h_
#if defined(__cplusplus)

#include "lldb/Host/Mutex.h"
#include "lldb/Host/Condition.h"
#include <pthread.h>
#include <stdint.h>
#include <time.h>

//----------------------------------------------------------------------
/// Enumerations for broadcasting.
//----------------------------------------------------------------------
namespace lldb_private {

//----------------------------------------------------------------------
/// @class ProcessRunLock ProcessRunLock.h "lldb/Host/ProcessRunLock.h"
/// @brief A class used to prevent the process from starting while other
/// threads are accessing its data, and prevent access to its data while
/// it is running.
//----------------------------------------------------------------------
    
class ProcessRunLock
{
public:
    ProcessRunLock () :
        m_rwlock(),
        m_running(false)
    {
        int err = ::pthread_rwlock_init(&m_rwlock, NULL); (void)err;
//#if LLDB_CONFIGURATION_DEBUG
//        assert(err == 0);
//#endif
    }

    ~ProcessRunLock ()
    {
        int err = ::pthread_rwlock_destroy (&m_rwlock); (void)err;
//#if LLDB_CONFIGURATION_DEBUG
//        assert(err == 0);
//#endif
    }

    bool
    ReadTryLock ()
    {
        ::pthread_rwlock_rdlock (&m_rwlock);
        if (m_running == false)
        {
            return true;
        }
        ::pthread_rwlock_unlock (&m_rwlock);
        return false;
    }

    bool
    ReadUnlock ()
    {
        return ::pthread_rwlock_unlock (&m_rwlock) == 0;
    }
    
    bool
    SetRunning()
    {
        ::pthread_rwlock_wrlock (&m_rwlock);
        m_running = true;
        ::pthread_rwlock_unlock (&m_rwlock);
        return true;
    }
    
    bool
    TrySetRunning()
    {
        bool r;

        if (::pthread_rwlock_trywrlock (&m_rwlock) == 0)
        {
            r = !m_running;
            m_running = true;
            ::pthread_rwlock_unlock (&m_rwlock);
            return r;
        }
        return false;
    }
    
    bool
    SetStopped ()
    {
        ::pthread_rwlock_wrlock (&m_rwlock);
        m_running = false;
        ::pthread_rwlock_unlock (&m_rwlock);
        return true;
    }

    class ProcessRunLocker
    {
    public:
        ProcessRunLocker () :
            m_lock (NULL)
        {
        }

        ~ProcessRunLocker()
        {
            Unlock();
        }

        // Try to lock the read lock, but only do so if there are no writers.
        bool
        TryLock (ProcessRunLock *lock)
        {
            if (m_lock)
            {
                if (m_lock == lock)
                    return true; // We already have this lock locked
                else
                    Unlock();
            }
            if (lock)
            {
                if (lock->ReadTryLock())
                {
                    m_lock = lock;
                    return true;
                }
            }
            return false;
        }

    protected:
        void
        Unlock ()
        {
            if (m_lock)
            {
                m_lock->ReadUnlock();
                m_lock = NULL;
            }
        }
        
        ProcessRunLock *m_lock;
    private:
        DISALLOW_COPY_AND_ASSIGN(ProcessRunLocker);
    };

protected:
    pthread_rwlock_t m_rwlock;
    bool m_running;
private:
    DISALLOW_COPY_AND_ASSIGN(ProcessRunLock);
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif // #ifndef liblldb_ProcessRunLock_h_
