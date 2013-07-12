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
#include "RegisterContextCoreLinux_x86_64.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Construct a Thread object with given PRSTATUS, PRPSINFO and FPREGSET
//----------------------------------------------------------------------
ThreadElfCore::ThreadElfCore (Process &process, tid_t tid, DataExtractor prstatus,
                              DataExtractor prpsinfo, DataExtractor fpregset) :
    Thread(process, tid),
    m_thread_reg_ctx_sp ()
{
    ProcessElfCore *pr = static_cast<ProcessElfCore *>(GetProcess().get());
    ArchSpec arch = pr->GetArchitecture();

    /* Parse the datastructures from the file */
    m_prstatus.Parse(prstatus, arch);
    m_prpsinfo.Parse(prpsinfo, arch);

    m_prstatus_data = prstatus;
    m_fpregset_data = fpregset;

    m_thread_name = std::string(m_prpsinfo.pr_fname);
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
        size_t header_size = ELFPrStatus::GetSize(arch);
        size_t len = m_prstatus_data.GetByteSize() - header_size;
        DataExtractor gpregset_data = DataExtractor(m_prstatus_data, header_size, len);
        switch (arch.GetMachine())
        {
            case llvm::Triple::x86_64:
                m_thread_reg_ctx_sp.reset(new RegisterContextCoreLinux_x86_64 (*this, gpregset_data, m_fpregset_data));
                break;
            default:
                if (log)
                    log->Printf ("elf-core::%s:: Architecture(%d) not supported",
                                 __FUNCTION__, arch.GetMachine());
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
        SetStopInfo(StopInfo::CreateStopReasonWithSignal (*this, m_prstatus.pr_cursig));
        return true;
    }
    return false;
}

//----------------------------------------------------------------
// Parse PRSTATUS from NOTE entry
//----------------------------------------------------------------
ELFPrStatus::ELFPrStatus()
{
    memset(this, 0, sizeof(ELFPrStatus));
}

bool
ELFPrStatus::Parse(DataExtractor &data, ArchSpec &arch)
{
    ByteOrder byteorder = data.GetByteOrder();
    size_t len;
    switch(arch.GetCore())
    {
        case ArchSpec::eCore_x86_64_x86_64:
            len = data.ExtractBytes(0, ELFPRSTATUS64_SIZE, byteorder, this);
            return len == ELFPRSTATUS64_SIZE;
        default:
            return false;
    }
}

//----------------------------------------------------------------
// Parse PRPSINFO from NOTE entry
//----------------------------------------------------------------
ELFPrPsInfo::ELFPrPsInfo()
{
    memset(this, 0, sizeof(ELFPrPsInfo));
}

bool
ELFPrPsInfo::Parse(DataExtractor &data, ArchSpec &arch)
{
    ByteOrder byteorder = data.GetByteOrder();
    size_t len;
    switch(arch.GetCore())
    {
        case ArchSpec::eCore_x86_64_x86_64:
            len = data.ExtractBytes(0, ELFPRPSINFO64_SIZE, byteorder, this);
            return len == ELFPRPSINFO64_SIZE;
        default:
            return false;
    }
}

