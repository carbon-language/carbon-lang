//===-- SBProcess.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBProcess_h_
#define LLDB_SBProcess_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBTarget.h"
#include <stdio.h>

namespace lldb {

class SBEvent;

#ifdef SWIG
%feature("docstring",
         "Represents the process associated with the target program."
         ) SBProcess;
#endif
class SBProcess
{
#ifdef SWIG
    %feature("autodoc", "1");
#endif

public:
    //------------------------------------------------------------------
    /// Broadcaster event bits definitions.
    //------------------------------------------------------------------
    enum
    {
        eBroadcastBitStateChanged   = (1 << 0),
        eBroadcastBitInterrupt      = (1 << 1),
        eBroadcastBitSTDOUT         = (1 << 2),
        eBroadcastBitSTDERR         = (1 << 3)
    };

    SBProcess ();

    SBProcess (const lldb::SBProcess& rhs);

#ifndef SWIG
    const lldb::SBProcess&
    operator = (const lldb::SBProcess& rhs);
#endif

    ~SBProcess();

    void
    Clear ();

    bool
    IsValid() const;

    lldb::SBTarget
    GetTarget() const;

    lldb::ByteOrder
    GetByteOrder() const;

    size_t
    PutSTDIN (const char *src, size_t src_len);

    size_t
    GetSTDOUT (char *dst, size_t dst_len) const;

    size_t
    GetSTDERR (char *dst, size_t dst_len) const;

    void
    ReportEventState (const lldb::SBEvent &event, FILE *out) const;

    void
    AppendEventStateReport (const lldb::SBEvent &event, lldb::SBCommandReturnObject &result);

    //------------------------------------------------------------------
    // Remote connection related functions. These will fail if the
    // process is not in eStateConnected. They are intended for use
    // when connecting to an externally managed debugserver instance.
    //------------------------------------------------------------------
    bool
    RemoteAttachToProcessWithID (lldb::pid_t pid,
                                 lldb::SBError& error);
    
#ifdef SWIG
    %feature("docstring",
             "See SBTarget.Launch for argument description and usage."
             ) RemoteLaunch;
#endif
    bool
    RemoteLaunch (char const **argv,
                  char const **envp,
                  const char *stdin_path,
                  const char *stdout_path,
                  const char *stderr_path,
                  const char *working_directory,
                  uint32_t launch_flags,
                  bool stop_at_entry,
                  lldb::SBError& error);
    
    //------------------------------------------------------------------
    // Thread related functions
    //------------------------------------------------------------------
    uint32_t
    GetNumThreads ();

    lldb::SBThread
    GetThreadAtIndex (size_t index);

    lldb::SBThread
    GetThreadByID (lldb::tid_t sb_thread_id);

    lldb::SBThread
    GetSelectedThread () const;

    bool
    SetSelectedThread (const lldb::SBThread &thread);

    bool
    SetSelectedThreadByID (uint32_t tid);

    //------------------------------------------------------------------
    // Stepping related functions
    //------------------------------------------------------------------

    lldb::StateType
    GetState ();

    int
    GetExitStatus ();

    const char *
    GetExitDescription ();

    lldb::pid_t
    GetProcessID ();

    uint32_t
    GetAddressByteSize() const;

#ifdef SWIG
    %feature("docstring",
             "Kills the process and shuts down all threads that were spawned to"
             " track and monitor process."
             ) Destroy;
#endif
    lldb::SBError
    Destroy ();

    lldb::SBError
    Continue ();

    lldb::SBError
    Stop ();

#ifdef SWIG
    %feature("docstring", "Same as Destroy(self).") Destroy;
#endif
    lldb::SBError
    Kill ();

    lldb::SBError
    Detach ();

#ifdef SWIG
    %feature("docstring", "Sends the process a unix signal.") Signal;
#endif
    lldb::SBError
    Signal (int signal);

#ifdef SWIG
    %feature("autodoc",
"Reads memory from the current process's address space and removes any
traps that may have been inserted into the memory. It returns the byte
buffer in a Python string. Example:

# Read 4 bytes from address 'addr' and assume error.Success() is True.
content = process.ReadMemory(addr, 4, error)
# Use 'ascii' encoding as each byte of 'content' is within [0..255].
new_bytes = bytearray(content, 'ascii')"
             ) ReadMemory;
#endif
    size_t
    ReadMemory (addr_t addr, void *buf, size_t size, lldb::SBError &error);

#ifdef SWIG
    %feature("autodoc",
"Writes memory to the current process's address space and maintains any
traps that might be present due to software breakpoints. Example:

# Create a Python string from the byte array.
new_value = str(bytes)
result = process.WriteMemory(addr, new_value, error)
if not error.Success() or result != len(bytes):
    print 'SBProcess.WriteMemory() failed!'"
             ) WriteMemory;
#endif
    size_t
    WriteMemory (addr_t addr, const void *buf, size_t size, lldb::SBError &error);

    // Events
    static lldb::StateType
    GetStateFromEvent (const lldb::SBEvent &event);

    static bool
    GetRestartedFromEvent (const lldb::SBEvent &event);

    static lldb::SBProcess
    GetProcessFromEvent (const lldb::SBEvent &event);

    lldb::SBBroadcaster
    GetBroadcaster () const;

    bool
    GetDescription (lldb::SBStream &description);

    uint32_t
    LoadImage (lldb::SBFileSpec &image_spec, lldb::SBError &error);
    
    lldb::SBError
    UnloadImage (uint32_t image_token);

protected:
    friend class SBAddress;
    friend class SBBreakpoint;
    friend class SBBreakpointLocation;
    friend class SBCommandInterpreter;
    friend class SBDebugger;
    friend class SBFunction;
    friend class SBTarget;
    friend class SBThread;
    friend class SBValue;

#ifndef SWIG

    lldb_private::Process *
    operator->() const;

    // Mimic shared pointer...
    lldb_private::Process *
    get() const;

#endif


    SBProcess (const lldb::ProcessSP &process_sp);

    void
    SetProcess (const lldb::ProcessSP &process_sp);

    lldb::ProcessSP m_opaque_sp;
};

}  // namespace lldb

#endif  // LLDB_SBProcess_h_
