//===-- ThreadElfCore.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DataExtractor.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Unwind.h"
#include "ProcessPOSIXLog.h"

#include "ThreadElfCore.h"
#include "ProcessElfCore.h"
#include "RegisterContextLinux_x86_64.h"
#include "RegisterContextFreeBSD_mips64.h"
#include "RegisterContextFreeBSD_x86_64.h"
#include "RegisterContextPOSIXCore_mips64.h"
#include "RegisterContextPOSIXCore_x86_64.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Construct a Thread object with given data
//----------------------------------------------------------------------
ThreadElfCore::ThreadElfCore (Process &process, tid_t tid,
                              const ThreadData &td) :
    Thread(process, tid),
    m_thread_name(td.name),
    m_thread_reg_ctx_sp (),
    m_signo(td.signo),
    m_gpregset_data(td.gpregset),
    m_fpregset_data(td.fpregset)
{
}

ThreadElfCore::~ThreadElfCore ()
{
    DestroyThread();
}

void
ThreadElfCore::RefreshStateAfterStop()
{
    GetRegisterContext()->InvalidateIfNeeded (false);
}

void
ThreadElfCore::ClearStackFrames ()
{
    Unwind *unwinder = GetUnwinder ();
    if (unwinder)
        unwinder->Clear();
    Thread::ClearStackFrames();
}

RegisterContextSP
ThreadElfCore::GetRegisterContext ()
{
    if (m_reg_context_sp.get() == NULL) {
        m_reg_context_sp = CreateRegisterContextForFrame (NULL);
    }
    return m_reg_context_sp;
}

RegisterContextSP
ThreadElfCore::CreateRegisterContextForFrame (StackFrame *frame)
{
    RegisterContextSP reg_ctx_sp;
    uint32_t concrete_frame_idx = 0;
    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_THREAD));

    if (frame)
        concrete_frame_idx = frame->GetConcreteFrameIndex ();

    if (concrete_frame_idx == 0)
    {
        if (m_thread_reg_ctx_sp)
            return m_thread_reg_ctx_sp;

        ProcessElfCore *process = static_cast<ProcessElfCore *>(GetProcess().get());
        ArchSpec arch = process->GetArchitecture();
        switch (arch.GetMachine())
        {
            case llvm::Triple::mips64:
                switch (arch.GetTriple().getOS())
                {
                    case llvm::Triple::FreeBSD:
                        m_thread_reg_ctx_sp.reset(new RegisterContextCorePOSIX_mips64 (*this, new RegisterContextFreeBSD_mips64(arch), m_gpregset_data, m_fpregset_data));
                        break;
                    default:
                        if (log)
                            log->Printf ("elf-core::%s:: OS(%d) not supported",
                                         __FUNCTION__, arch.GetTriple().getOS());
                        assert (false && "OS not supported");
                        break;
                }
                break;
            case llvm::Triple::x86_64:
                switch (arch.GetTriple().getOS())
                {
                    case llvm::Triple::FreeBSD:
                        m_thread_reg_ctx_sp.reset(new RegisterContextCorePOSIX_x86_64 (*this, new RegisterContextFreeBSD_x86_64(arch), m_gpregset_data, m_fpregset_data));
                        break;
                    case llvm::Triple::Linux:
                        m_thread_reg_ctx_sp.reset(new RegisterContextCorePOSIX_x86_64 (*this, new RegisterContextLinux_x86_64(arch), m_gpregset_data, m_fpregset_data));
                        break;
                    default:
                        if (log)
                            log->Printf ("elf-core::%s:: OS(%d) not supported",
                                         __FUNCTION__, arch.GetTriple().getOS());
                        assert (false && "OS not supported");
                        break;
                }
                break;
            default:
                if (log)
                    log->Printf ("elf-core::%s:: Architecture(%d) not supported",
                                 __FUNCTION__, arch.GetMachine());
                assert (false && "Architecture not supported");
        }
        reg_ctx_sp = m_thread_reg_ctx_sp;
    }
    else if (m_unwinder_ap.get())
    {
        reg_ctx_sp = m_unwinder_ap->CreateRegisterContextForFrame (frame);
    }
    return reg_ctx_sp;
}

bool
ThreadElfCore::CalculateStopInfo ()
{
    ProcessSP process_sp (GetProcess());
    if (process_sp)
    {
        SetStopInfo(StopInfo::CreateStopReasonWithSignal (*this, m_signo));
        return true;
    }
    return false;
}

//----------------------------------------------------------------
// Parse PRSTATUS from NOTE entry
//----------------------------------------------------------------
ELFLinuxPrStatus::ELFLinuxPrStatus()
{
    memset(this, 0, sizeof(ELFLinuxPrStatus));
}

bool
ELFLinuxPrStatus::Parse(DataExtractor &data, ArchSpec &arch)
{
    ByteOrder byteorder = data.GetByteOrder();
    size_t len;
    switch(arch.GetCore())
    {
        case ArchSpec::eCore_x86_64_x86_64:
            len = data.ExtractBytes(0, ELFLINUXPRSTATUS64_SIZE, byteorder, this);
            return len == ELFLINUXPRSTATUS64_SIZE;
        default:
            return false;
    }
}

//----------------------------------------------------------------
// Parse PRPSINFO from NOTE entry
//----------------------------------------------------------------
ELFLinuxPrPsInfo::ELFLinuxPrPsInfo()
{
    memset(this, 0, sizeof(ELFLinuxPrPsInfo));
}

bool
ELFLinuxPrPsInfo::Parse(DataExtractor &data, ArchSpec &arch)
{
    ByteOrder byteorder = data.GetByteOrder();
    size_t len;
    switch(arch.GetCore())
    {
        case ArchSpec::eCore_x86_64_x86_64:
            len = data.ExtractBytes(0, ELFLINUXPRPSINFO64_SIZE, byteorder, this);
            return len == ELFLINUXPRPSINFO64_SIZE;
        default:
            return false;
    }
}

