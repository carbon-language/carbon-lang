//===-- MIUtilThreadBaseStd.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Copyright:   None.
//--

#pragma once

// Third party headers:
#ifdef _MSC_VER
#include <eh.h>
#endif // _MSC_VER
#include <thread>
#include <mutex>

// In-house headers:
#include "MIDataTypes.h"
#include "MIUtilString.h"

//++ ============================================================================
// Details: MI common code utility class. Handle thread mutual exclusion.
//          Embed Mutexes in your Active Object and then use them through Locks.
// Gotchas: None.
// Authors: Aidan Dodds 10/03/2014.
// Changes: None.
//--
class CMIUtilThreadMutex
{
    // Methods:
  public:
    /* ctor */ CMIUtilThreadMutex(void){};
    //
    void
    Lock(void); // Wait until mutex can be obtained
    void
    Unlock(void); // Release the mutex
    bool
    TryLock(void); // Gain the lock if available

    // Overrideable:
  public:
    // From CMICmnBase
    /* dtor */ virtual ~CMIUtilThreadMutex(void){};

    // Attributes:
  private:
    std::recursive_mutex m_mutex;
};

//++ ============================================================================
// Details: MI common code utility class. Thread object.
// Gotchas: None.
// Authors: Aidan Dodds 10/03/2014.
// Changes: None.
//--
class CMIUtilThread
{
    // Typedef:
  public:
    typedef MIuint (*FnThreadProc)(void *vpThisClass);

    // Methods:
  public:
    /* ctor */ CMIUtilThread(void);
    //
    bool
    Start(FnThreadProc vpFn, void *vpArg); // Start execution of this thread
    bool
    Join(void); // Wait for this thread to stop
    bool
    IsActive(void); // Returns true if this thread is running
    void
    Finish(void); // Finish this thread

    // Overrideable:
  public:
    /* dtor */ virtual ~CMIUtilThread(void);

    // Methods:
  private:
    CMIUtilThreadMutex m_mutex;
    std::thread *m_pThread;
    bool m_bIsActive;
};

//++ ============================================================================
// Details: MI common code utility class. Base class for a worker thread active
//          object. Runs an 'captive thread'.
// Gotchas: None.
// Authors: Aidan Dodds 10/03/2014..
// Changes: None.
//--
class CMIUtilThreadActiveObjBase
{
    // Methods:
  public:
    /* ctor */ CMIUtilThreadActiveObjBase(void);
    //
    bool
    Acquire(void); // Obtain a reference to this object
    bool
    Release(void); // Release a reference to this object
    bool
    ThreadIsActive(void); // Return true if this object is running
    bool
    ThreadJoin(void); // Wait for this thread to stop running
    bool
    ThreadKill(void); // Force this thread to stop, regardless of references
    bool
    ThreadExecute(void); // Start this objects execution in another thread
    void ThreadManage(void);

    // Overrideable:
  public:
    /* dtor */ virtual ~CMIUtilThreadActiveObjBase(void);
    //
    // Each thread object must supple a unique name that can be used to locate it
    virtual const CMIUtilString &ThreadGetName(void) const = 0;

    // Statics:
  protected:
    static MIuint
    ThreadEntry(void *vpThisClass); // Thread entry point

    // Overrideable:
  protected:
    virtual bool
    ThreadRun(bool &vrIsAlive) = 0; // Call the main worker method
    virtual bool
    ThreadFinish(void) = 0; // Finish of what you were doing

    // Attributes:
  protected:
    volatile MIuint m_references;   // Stores the current lifetime state of this thread, 0 = running, > 0 = shutting down
    volatile bool m_bHasBeenKilled; // Set to true when this thread has been killed
    CMIUtilThread m_thread;         // The execution thread
    CMIUtilThreadMutex
        m_mutex; // This mutex allows us to safely communicate with this thread object across the interface from multiple threads
};

//++ ============================================================================
// Details: MI common code utility class. Handle thread resource locking.
//          Put Locks inside all the methods of your Active Object that access
//          data shared with the captive thread.
// Gotchas: None.
// Authors: Aidan Dodds 10/03/2014.
// Changes: None.
//--
class CMIUtilThreadLock
{
    // Methods:
  public:
    /* ctor */
    CMIUtilThreadLock(CMIUtilThreadMutex &vMutex)
        : m_rMutex(vMutex)
    {
        m_rMutex.Lock();
    }

    // Overrideable:
  public:
    /* dtor */
    virtual ~CMIUtilThreadLock(void) { m_rMutex.Unlock(); }

    // Attributes:
  private:
    CMIUtilThreadMutex &m_rMutex;
};
