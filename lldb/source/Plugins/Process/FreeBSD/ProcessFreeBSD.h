//===-- ProcessFreeBSD.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessFreeBSD_H_
#define liblldb_ProcessFreeBSD_H_

// C Includes

// C++ Includes
#include <queue>

// Other libraries and framework includes
#include "lldb/Target/Process.h"
#include "lldb/Target/ThreadList.h"
#include "ProcessMessage.h"
#include "ProcessPOSIX.h"

class ProcessMonitor;

class ProcessFreeBSD :
    public ProcessPOSIX
{

public:
    //------------------------------------------------------------------
    // Static functions.
    //------------------------------------------------------------------
    static lldb::ProcessSP
    CreateInstance(lldb_private::Target& target,
                   lldb_private::Listener &listener,
                   const lldb_private::FileSpec *crash_file_path);

    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    //------------------------------------------------------------------
    // Constructors and destructors
    //------------------------------------------------------------------
    ProcessFreeBSD(lldb_private::Target& target,
                   lldb_private::Listener &listener);

    virtual lldb_private::Error
    DoDetach(bool keep_stopped);

    virtual bool
    UpdateThreadList(lldb_private::ThreadList &old_thread_list, lldb_private::ThreadList &new_thread_list);

    virtual lldb_private::Error
    DoResume();

    virtual lldb_private::Error
    WillResume();

    virtual void
    SendMessage(const ProcessMessage &message);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

    virtual void
    GetPluginCommandHelp(const char *command, lldb_private::Stream *strm);

    virtual lldb_private::Error
    ExecutePluginCommand(lldb_private::Args &command,
                         lldb_private::Stream *strm);

    virtual lldb_private::Log *
    EnablePluginLogging(lldb_private::Stream *strm,
                        lldb_private::Args &command);

protected:
    friend class FreeBSDThread;

    typedef std::vector<lldb::tid_t> tid_collection;
    tid_collection m_suspend_tids;
    tid_collection m_run_tids;
    tid_collection m_step_tids;

    int m_resume_signo;

};

#endif  // liblldb_ProcessFreeBSD_H_
