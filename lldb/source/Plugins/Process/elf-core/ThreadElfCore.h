//===-- ThreadElfCore.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadElfCore_h_
#define liblldb_ThreadElfCore_h_

#include <string>

#include "lldb/Target/Thread.h"
#include "lldb/Core/DataExtractor.h"

struct compat_timeval
{
    int64_t tv_sec;
    int32_t tv_usec;
};

// PRSTATUS structure's size differs based on architecture.
// Currently parsing done only for x86-64 architecture by
// simply reading data from the buffer.
// The following macros are used to specify the size.
// Calculating size using sizeof() wont work because of padding.
#define ELFPRSTATUS64_SIZE (112)
#define ELFPRPSINFO64_SIZE (132)

struct ELFPrStatus
{
    int32_t         si_signo;
    int32_t         si_code;
    int32_t         si_errno;

    int16_t         pr_cursig;
    
    uint64_t        pr_sigpend;
    uint64_t        pr_sighold;

    uint32_t        pr_pid;
    uint32_t        pr_ppid;
    uint32_t        pr_pgrp;
    uint32_t        pr_sid;

    compat_timeval  pr_utime;
    compat_timeval  pr_stime;
    compat_timeval  pr_cutime;
    compat_timeval  pr_cstime;

    ELFPrStatus();

    bool
    Parse(lldb_private::DataExtractor &data, lldb_private::ArchSpec &arch);

    static size_t
    GetSize(lldb_private::ArchSpec &arch)
    {
        switch(arch.GetCore())
        {
            case lldb_private::ArchSpec::eCore_x86_64_x86_64:
                return ELFPRSTATUS64_SIZE;
            default:
                return 0;
        }
    }
};

struct ELFPrPsInfo
{
    char        pr_state;
    char        pr_sname;
    char        pr_zomb;
    char        pr_nice;
    uint64_t    pr_flag;
    uint32_t    pr_uid;
    uint32_t    pr_gid;
    int32_t     pr_pid;
    int32_t     pr_ppid;
    int32_t     pr_pgrp;
    int32_t     pr_sid;
    char        pr_fname[16];
    char        pr_psargs[80];

    ELFPrPsInfo();

    bool
    Parse(lldb_private::DataExtractor &data, lldb_private::ArchSpec &arch);

    static size_t
    GetSize(lldb_private::ArchSpec &arch)
    {
        switch(arch.GetCore())
        {
            case lldb_private::ArchSpec::eCore_x86_64_x86_64:
                return ELFPRPSINFO64_SIZE;
            default:
                return 0;
        }
    }

};

class ThreadElfCore : public lldb_private::Thread
{
public:
    ThreadElfCore (lldb_private::Process &process, lldb::tid_t tid,
                   lldb_private::DataExtractor prstatus,
                   lldb_private::DataExtractor prpsinfo,
                   lldb_private::DataExtractor fpregset);

    virtual
    ~ThreadElfCore ();

    virtual void
    RefreshStateAfterStop();

    virtual lldb::RegisterContextSP
    GetRegisterContext ();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    virtual void
    ClearStackFrames ();

    static bool
    ThreadIDIsValid (lldb::tid_t thread)
    {
        return thread != 0;
    }

    virtual const char *
    GetName ()
    {
        if (m_thread_name.empty())
            return NULL;
        return m_thread_name.c_str();
    }

    void
    SetName (const char *name)
    {
        if (name && name[0])
            m_thread_name.assign (name);
        else
            m_thread_name.clear();
    }

protected:
    
    friend class ProcessElfCore;

    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    std::string m_thread_name;
    lldb::RegisterContextSP m_thread_reg_ctx_sp;

    ELFPrStatus m_prstatus;
    ELFPrPsInfo m_prpsinfo;
    lldb_private::DataExtractor m_prstatus_data;
    lldb_private::DataExtractor m_fpregset_data;

    virtual bool CalculateStopInfo();

};

#endif  // liblldb_ThreadElfCore_h_
