//===-- ProcessMacOSXRemote.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//----------------------------------------------------------------------
//
//  ProcessMacOSXRemote.h
//  liblldb
//
//  Created by Greg Clayton on 4/21/09.
//
//
//----------------------------------------------------------------------

#ifndef liblldb_ProcessMacOSXRemote_H_
#define liblldb_ProcessMacOSXRemote_H_

// C Includes

// C++ Includes
#include <list>

// Other libraries and framework includes
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"

class ThreadMacOSXRemote;

class ProcessMacOSXRemote :
    public Process
{
public:
    friend class ThreadMacOSX;

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ProcessMacOSXRemote(Target& target);
    virtual ~DCProcessMacOSXRemote();

    static Process* CreateInstance (Target& target);

    //------------------------------------------------------------------
    // Check if a given Process
    //------------------------------------------------------------------
    virtual bool        CanDebug(Target &target);

    //------------------------------------------------------------------
    // Creating a new process, or attaching to an existing one
    //------------------------------------------------------------------
    virtual lldb::pid_t DoLaunch (Module* module,
                                char const *argv[],         // Can be NULL
                                char const *envp[],         // Can be NULL
                                const char *stdin_path,     // Can be NULL
                                const char *stdout_path,    // Can be NULL
                                const char *stderr_path);   // Can be NULL
    virtual void        DidLaunch ();
    virtual lldb::pid_t DoAttach (lldb::pid_t pid);
    virtual void        DidAttach ();

    //------------------------------------------------------------------
    // Process Control
    //------------------------------------------------------------------
//  virtual bool        WillResume ();
    virtual bool        DoResume ();
//  virtual void        DidResume ();

    virtual bool        DoHalt ();
    virtual bool        WillDetach ();
    virtual bool        DoDetach ();
    virtual bool        DoKill (int signal);

    virtual bool        ShouldStop ();

    //------------------------------------------------------------------
    // Process Queries
    //------------------------------------------------------------------
    virtual bool        IsAlive ();
    virtual bool        IsRunning ();
    virtual lldb::addr_t    GetImageInfoAddress();

    //------------------------------------------------------------------
    // Process Memory
    //------------------------------------------------------------------
    virtual size_t      DoReadMemory (lldb::addr_t addr, void *buf, size_t size);
    virtual size_t      DoWriteMemory (lldb::addr_t addr, const void *buf, size_t size);

    //------------------------------------------------------------------
    // Process STDIO
    //------------------------------------------------------------------
    virtual size_t      GetSTDOUT (char *buf, size_t buf_size);
    virtual size_t      GetSTDERR (char *buf, size_t buf_size);

    //----------------------------------------------------------------------
    // Process Breakpoints
    //----------------------------------------------------------------------
    virtual size_t
    GetSoftwareBreakpointTrapOpcode (lldb::BreakpointSite *bp_site);

    //----------------------------------------------------------------------
    // Process Breakpoints
    //----------------------------------------------------------------------
    virtual bool
    EnableBreakpoint (lldb::BreakpointSite *bp_site);

    virtual bool
    DisableBreakpoint (lldb::BreakpointSite *bp_site);

    //----------------------------------------------------------------------
    // Process Watchpoints
    //----------------------------------------------------------------------
    virtual bool        EnableWatchpoint (WatchpointLocation *wp_loc);
    virtual bool        DisableWatchpoint (WatchpointLocation *wp_loc);

    //------------------------------------------------------------------
    // Thread Queries
    //------------------------------------------------------------------
    virtual Thread *    GetCurrentThread ();
    virtual bool        SetCurrentThread (lldb::tid_t tid);
    virtual Thread *    GetThreadAtIndex (uint32_t idx);
    virtual Thread *    GetThreadByID (lldb::tid_t tid);
    virtual size_t      GetNumThreads ();

    virtual ByteOrder   GetByteOrder () const;

    virtual DynamicLoader *
    GetDynamicLoader ();

protected:
    Flags m_flags; // Process specific flags (see eFlags enums)
    ArchSpec m_arch_spec;
    std::auto_ptr<DynamicLoader> m_dynamic_loader_ap;
    ByteOrder m_byte_order;

    //----------------------------------------------------------------------
    // Accessors
    //----------------------------------------------------------------------
    bool
    ProcessIDIsValid ( ) const;

    bool
    IsRunning ( State state )
    {
        return state == eStateRunning || IsStepping(state);
    }

    bool
    IsStepping ( State state)
    {
        return state == eStateStepping;
    }
    bool
    CanResume ( State state)
    {
        return state == eStateStopped;
    }

    ArchSpec&
    GetArchSpec()
    {
        return m_arch_spec;
    }
    const ArchSpec&
    GetArchSpec() const
    {
        return m_arch_spec;
    }

    enum
    {
        eFlagsNone = 0,
        eFlagsAttached = (1 << 0),
        eFlagsUsingSBS = (1 << 1)
    };

    void
    Clear ( );

    Flags &
    GetFlags ()
    {
        return m_flags;
    }

    const Flags &
    GetFlags () const
    {
        return m_flags;
    }

    uint32_t
    UpdateThreadListIfNeeded ();

private:
    //------------------------------------------------------------------
    // For ProcessMacOSXRemote only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ProcessMacOSXRemote);

};

#endif  // liblldb_ProcessMacOSXRemote_H_
