//===-- NativeThreadLinux.cpp --------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeThreadLinux.h"

#include <signal.h>

#include "NativeProcessLinux.h"
#include "NativeRegisterContextLinux_x86_64.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostNativeThread.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-log.h"

#include "llvm/ADT/SmallString.h"

#include "Plugins/Process/Utility/RegisterContextLinux_arm64.h"
#include "Plugins/Process/Utility/RegisterContextLinux_i386.h"
#include "Plugins/Process/Utility/RegisterContextLinux_x86_64.h"
#include "Plugins/Process/Utility/RegisterInfoInterface.h"

using namespace lldb;
using namespace lldb_private;

namespace
{
    void LogThreadStopInfo (Log &log, const ThreadStopInfo &stop_info, const char *const header)
    {
        switch (stop_info.reason)
        {
            case eStopReasonSignal:
                log.Printf ("%s: %s signal 0x%" PRIx32, __FUNCTION__, header, stop_info.details.signal.signo);
                return;
            case eStopReasonException:
                log.Printf ("%s: %s exception type 0x%" PRIx64, __FUNCTION__, header, stop_info.details.exception.type);
                return;
            case eStopReasonExec:
                log.Printf ("%s: %s exec, stopping signal 0x%" PRIx32, __FUNCTION__, header, stop_info.details.signal.signo);
                return;
            default:
                log.Printf ("%s: %s invalid stop reason %" PRIu32, __FUNCTION__, header, static_cast<uint32_t> (stop_info.reason));
        }
    }
}

NativeThreadLinux::NativeThreadLinux (NativeProcessLinux *process, lldb::tid_t tid) :
    NativeThreadProtocol (process, tid),
    m_state (StateType::eStateInvalid),
    m_stop_info (),
    m_reg_context_sp ()
{
}

const char *
NativeThreadLinux::GetName()
{
    NativeProcessProtocolSP process_sp = m_process_wp.lock ();
    if (!process_sp)
        return "<unknown: no process>";

    // const NativeProcessLinux *const process = reinterpret_cast<NativeProcessLinux*> (process_sp->get ());
    llvm::SmallString<32> thread_name;
    HostNativeThread::GetName(GetID(), thread_name);
    return thread_name.c_str();
}

lldb::StateType
NativeThreadLinux::GetState ()
{
    return m_state;
}


bool
NativeThreadLinux::GetStopReason (ThreadStopInfo &stop_info)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD));
    switch (m_state)
    {
    case eStateStopped:
    case eStateCrashed:
    case eStateExited:
    case eStateSuspended:
    case eStateUnloaded:
        if (log)
            LogThreadStopInfo (*log, m_stop_info, "m_stop_info in thread:");
        stop_info = m_stop_info;
        if (log)
            LogThreadStopInfo (*log, stop_info, "returned stop_info:");
        return true;

    case eStateInvalid:
    case eStateConnected:
    case eStateAttaching:
    case eStateLaunching:
    case eStateRunning:
    case eStateStepping:
    case eStateDetached:
        if (log)
        {
            log->Printf ("NativeThreadLinux::%s tid %" PRIu64 " in state %s cannot answer stop reason",
                    __FUNCTION__, GetID (), StateAsCString (m_state));
        }
        return false;
    }
}

lldb_private::NativeRegisterContextSP
NativeThreadLinux::GetRegisterContext ()
{
    // Return the register context if we already created it.
    if (m_reg_context_sp)
        return m_reg_context_sp;

    // First select the appropriate RegisterInfoInterface.
    RegisterInfoInterface *reg_interface = nullptr;
    NativeProcessProtocolSP m_process_sp = m_process_wp.lock ();
    if (!m_process_sp)
        return NativeRegisterContextSP ();

    ArchSpec target_arch;
    if (!m_process_sp->GetArchitecture (target_arch))
        return NativeRegisterContextSP ();

    switch (target_arch.GetTriple().getOS())
    {
        case llvm::Triple::Linux:
            switch (target_arch.GetMachine())
            {
            case llvm::Triple::aarch64:
                assert((HostInfo::GetArchitecture ().GetAddressByteSize() == 8) && "Register setting path assumes this is a 64-bit host");
                reg_interface = static_cast<RegisterInfoInterface*>(new RegisterContextLinux_arm64(target_arch));
                break;
            case llvm::Triple::x86:
            case llvm::Triple::x86_64:
                if (HostInfo::GetArchitecture().GetAddressByteSize() == 4)
                {
                    // 32-bit hosts run with a RegisterContextLinux_i386 context.
                    reg_interface = static_cast<RegisterInfoInterface*>(new RegisterContextLinux_i386(target_arch));
                }
                else
                {
                    assert((HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
                           "Register setting path assumes this is a 64-bit host");
                    // X86_64 hosts know how to work with 64-bit and 32-bit EXEs using the x86_64 register context.
                    reg_interface = static_cast<RegisterInfoInterface*> (new RegisterContextLinux_x86_64 (target_arch));
                }
                break;
            default:
                break;
            }
            break;
        default:
            break;
    }

    assert(reg_interface && "OS or CPU not supported!");
    if (!reg_interface)
        return NativeRegisterContextSP ();

    // Now create the register context.
    switch (target_arch.GetMachine())
    {
#if 0
        case llvm::Triple::mips64:
        {
            RegisterContextPOSIXProcessMonitor_mips64 *reg_ctx = new RegisterContextPOSIXProcessMonitor_mips64(*this, 0, reg_interface);
            m_posix_thread = reg_ctx;
            m_reg_context_sp.reset(reg_ctx);
            break;
        }
#endif
#if 0
        case llvm::Triple::x86:
#endif
        case llvm::Triple::x86_64:
        {
            const uint32_t concrete_frame_idx = 0;
            m_reg_context_sp.reset (new NativeRegisterContextLinux_x86_64(*this, concrete_frame_idx, reg_interface));
            break;
        }
        default:
            break;
    }

    return m_reg_context_sp;
}

Error
NativeThreadLinux::SetWatchpoint (lldb::addr_t addr, size_t size, uint32_t watch_flags, bool hardware)
{
    // TODO implement
    return Error ("not implemented");
}

Error
NativeThreadLinux::RemoveWatchpoint (lldb::addr_t addr)
{
    // TODO implement
    return Error ("not implemented");
}

void
NativeThreadLinux::SetLaunching ()
{
    const StateType new_state = StateType::eStateLaunching;
    MaybeLogStateChange (new_state);
    m_state = new_state;

    // Also mark it as stopped since launching temporarily stops the newly created thread
    // in the ptrace machinery.
    m_stop_info.reason = StopReason::eStopReasonSignal;
    m_stop_info.details.signal.signo = SIGSTOP;
}


void
NativeThreadLinux::SetRunning ()
{
    const StateType new_state = StateType::eStateRunning;
    MaybeLogStateChange (new_state);
    m_state = new_state;

    m_stop_info.reason = StopReason::eStopReasonNone;
}

void
NativeThreadLinux::SetStepping ()
{
    const StateType new_state = StateType::eStateStepping;
    MaybeLogStateChange (new_state);
    m_state = new_state;

    m_stop_info.reason = StopReason::eStopReasonNone;
}

void
NativeThreadLinux::SetStoppedBySignal (uint32_t signo)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf ("NativeThreadLinux::%s called with signal 0x%" PRIx32, __FUNCTION__, signo);

    const StateType new_state = StateType::eStateStopped;
    MaybeLogStateChange (new_state);
    m_state = new_state;

    m_stop_info.reason = StopReason::eStopReasonSignal;
    m_stop_info.details.signal.signo = signo;
}

void
NativeThreadLinux::SetStoppedByExec ()
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf ("NativeThreadLinux::%s()", __FUNCTION__);

    const StateType new_state = StateType::eStateStopped;
    MaybeLogStateChange (new_state);
    m_state = new_state;

    m_stop_info.reason = StopReason::eStopReasonExec;
    m_stop_info.details.signal.signo = SIGSTOP;
}

void
NativeThreadLinux::SetStoppedByBreakpoint ()
{
    const StateType new_state = StateType::eStateStopped;
    MaybeLogStateChange (new_state);
    m_state = new_state;

    m_stop_info.reason = StopReason::eStopReasonSignal;
    m_stop_info.details.signal.signo = SIGTRAP;
}

bool
NativeThreadLinux::IsStoppedAtBreakpoint ()
{
    // Are we stopped? If not, this can't be a breakpoint.
    if (GetState () != StateType::eStateStopped)
        return false;

    // Was the stop reason a signal with signal number SIGTRAP? If not, not a breakpoint.
    return (m_stop_info.reason == StopReason::eStopReasonSignal) &&
            (m_stop_info.details.signal.signo == SIGTRAP);
}

void
NativeThreadLinux::SetCrashedWithException (uint64_t exception_type, lldb::addr_t exception_addr)
{
    const StateType new_state = StateType::eStateCrashed;
    MaybeLogStateChange (new_state);
    m_state = new_state;

    m_stop_info.reason = StopReason::eStopReasonException;
    m_stop_info.details.exception.type = exception_type;
    m_stop_info.details.exception.data_count = 1;
    m_stop_info.details.exception.data[0] = exception_addr;
}


void
NativeThreadLinux::SetSuspended ()
{
    const StateType new_state = StateType::eStateSuspended;
    MaybeLogStateChange (new_state);
    m_state = new_state;

    // FIXME what makes sense here? Do we need a suspended StopReason?
    m_stop_info.reason = StopReason::eStopReasonNone;
}

void
NativeThreadLinux::SetExited ()
{
    const StateType new_state = StateType::eStateExited;
    MaybeLogStateChange (new_state);
    m_state = new_state;

    m_stop_info.reason = StopReason::eStopReasonThreadExiting;
}

void
NativeThreadLinux::MaybeLogStateChange (lldb::StateType new_state)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD));
    // If we're not logging, we're done.
    if (!log)
        return;

    // If this is a state change to the same state, we're done.
    lldb::StateType old_state = m_state;
    if (new_state == old_state)
        return;

    NativeProcessProtocolSP m_process_sp = m_process_wp.lock ();
    lldb::pid_t pid = m_process_sp ? m_process_sp->GetID () : LLDB_INVALID_PROCESS_ID;

    // Log it.
    log->Printf ("NativeThreadLinux: thread (pid=%" PRIu64 ", tid=%" PRIu64 ") changing from state %s to %s", pid, GetID (), StateAsCString (old_state), StateAsCString (new_state));
}

uint32_t
NativeThreadLinux::TranslateStopInfoToGdbSignal (const ThreadStopInfo &stop_info) const
{
    switch (stop_info.reason)
    {
        case eStopReasonSignal:
            // No translation.
            return stop_info.details.signal.signo;

        case eStopReasonException:
            {
                Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD));
                // FIXME I think the eStopReasonException is a xnu/Mach exception, which we
                // shouldn't see on Linux.
                // No translation.
                if (log)
                    log->Printf ("NativeThreadLinux::%s saw an exception stop type (signo %"
                                 PRIu64 "), not expecting to see exceptions on Linux",
                                 __FUNCTION__,
                                 stop_info.details.exception.type);
                return static_cast<uint32_t> (stop_info.details.exception.type);
            }

        default:
            assert (0 && "unexpected stop_info.reason found");
            return 0;
    }
}

