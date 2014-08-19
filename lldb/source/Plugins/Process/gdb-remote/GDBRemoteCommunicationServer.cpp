//===-- GDBRemoteCommunicationServer.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <errno.h>

#include "lldb/Host/Config.h"

#include "GDBRemoteCommunicationServer.h"
#include "lldb/Core/StreamGDBRemote.h"

// C Includes
// C++ Includes
#include <cstring>
#include <chrono>
#include <thread>

// Other libraries and framework includes
#include "llvm/ADT/Triple.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Debug.h"
#include "lldb/Host/Endian.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Target/FileAction.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/NativeRegisterContext.h"
#include "Host/common/NativeProcessProtocol.h"
#include "Host/common/NativeThreadProtocol.h"

// Project includes
#include "Utility/StringExtractorGDBRemote.h"
#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// GDBRemote Errors
//----------------------------------------------------------------------

namespace
{
    enum GDBRemoteServerError
    {
        // Set to the first unused error number in literal form below
        eErrorFirst = 29,
        eErrorNoProcess = eErrorFirst,
        eErrorResume,
        eErrorExitStatus
    };
}

//----------------------------------------------------------------------
// GDBRemoteCommunicationServer constructor
//----------------------------------------------------------------------
GDBRemoteCommunicationServer::GDBRemoteCommunicationServer(bool is_platform) :
    GDBRemoteCommunication ("gdb-remote.server", "gdb-remote.server.rx_packet", is_platform),
    m_platform_sp (Platform::GetDefaultPlatform ()),
    m_async_thread (LLDB_INVALID_HOST_THREAD),
    m_process_launch_info (),
    m_process_launch_error (),
    m_spawned_pids (),
    m_spawned_pids_mutex (Mutex::eMutexTypeRecursive),
    m_proc_infos (),
    m_proc_infos_index (0),
    m_port_map (),
    m_port_offset(0),
    m_current_tid (LLDB_INVALID_THREAD_ID),
    m_continue_tid (LLDB_INVALID_THREAD_ID),
    m_debugged_process_mutex (Mutex::eMutexTypeRecursive),
    m_debugged_process_sp (),
    m_debugger_sp (),
    m_stdio_communication ("process.stdio"),
    m_exit_now (false),
    m_inferior_prev_state (StateType::eStateInvalid),
    m_thread_suffix_supported (false),
    m_list_threads_in_stop_reply (false),
    m_active_auxv_buffer_sp (),
    m_saved_registers_mutex (),
    m_saved_registers_map (),
    m_next_saved_registers_id (1)
{
    assert(is_platform && "must be lldb-platform if debugger is not specified");
}

GDBRemoteCommunicationServer::GDBRemoteCommunicationServer(bool is_platform,
                                                           const lldb::PlatformSP& platform_sp,
                                                           lldb::DebuggerSP &debugger_sp) :
    GDBRemoteCommunication ("gdb-remote.server", "gdb-remote.server.rx_packet", is_platform),
    m_platform_sp (platform_sp),
    m_async_thread (LLDB_INVALID_HOST_THREAD),
    m_process_launch_info (),
    m_process_launch_error (),
    m_spawned_pids (),
    m_spawned_pids_mutex (Mutex::eMutexTypeRecursive),
    m_proc_infos (),
    m_proc_infos_index (0),
    m_port_map (),
    m_port_offset(0),
    m_current_tid (LLDB_INVALID_THREAD_ID),
    m_continue_tid (LLDB_INVALID_THREAD_ID),
    m_debugged_process_mutex (Mutex::eMutexTypeRecursive),
    m_debugged_process_sp (),
    m_debugger_sp (debugger_sp),
    m_stdio_communication ("process.stdio"),
    m_exit_now (false),
    m_inferior_prev_state (StateType::eStateInvalid),
    m_thread_suffix_supported (false),
    m_list_threads_in_stop_reply (false),
    m_active_auxv_buffer_sp (),
    m_saved_registers_mutex (),
    m_saved_registers_map (),
    m_next_saved_registers_id (1)
{
    assert(platform_sp);
    assert((is_platform || debugger_sp) && "must specify non-NULL debugger_sp when lldb-gdbserver");
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
GDBRemoteCommunicationServer::~GDBRemoteCommunicationServer()
{
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::GetPacketAndSendResponse (uint32_t timeout_usec,
                                                        Error &error,
                                                        bool &interrupt,
                                                        bool &quit)
{
    StringExtractorGDBRemote packet;

    PacketResult packet_result = WaitForPacketWithTimeoutMicroSecondsNoLock (packet, timeout_usec);
    if (packet_result == PacketResult::Success)
    {
        const StringExtractorGDBRemote::ServerPacketType packet_type = packet.GetServerPacketType ();
        switch (packet_type)
        {
        case StringExtractorGDBRemote::eServerPacketType_nack:
        case StringExtractorGDBRemote::eServerPacketType_ack:
            break;

        case StringExtractorGDBRemote::eServerPacketType_invalid:
            error.SetErrorString("invalid packet");
            quit = true;
            break;

        default:
        case StringExtractorGDBRemote::eServerPacketType_unimplemented:
            packet_result = SendUnimplementedResponse (packet.GetStringRef().c_str());
            break;

        case StringExtractorGDBRemote::eServerPacketType_A:
            packet_result = Handle_A (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qfProcessInfo:
            packet_result = Handle_qfProcessInfo (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qsProcessInfo:
            packet_result = Handle_qsProcessInfo (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qC:
            packet_result = Handle_qC (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qHostInfo:
            packet_result = Handle_qHostInfo (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qLaunchGDBServer:
            packet_result = Handle_qLaunchGDBServer (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qKillSpawnedProcess:
            packet_result = Handle_qKillSpawnedProcess (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_k:
            packet_result = Handle_k (packet);
            quit = true;
            break;

        case StringExtractorGDBRemote::eServerPacketType_qLaunchSuccess:
            packet_result = Handle_qLaunchSuccess (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qGroupName:
            packet_result = Handle_qGroupName (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qProcessInfo:
            packet_result = Handle_qProcessInfo (packet);
            break;
                
        case StringExtractorGDBRemote::eServerPacketType_qProcessInfoPID:
            packet_result = Handle_qProcessInfoPID (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qSpeedTest:
            packet_result = Handle_qSpeedTest (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qUserName:
            packet_result = Handle_qUserName (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qGetWorkingDir:
            packet_result = Handle_qGetWorkingDir(packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QEnvironment:
            packet_result = Handle_QEnvironment (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QLaunchArch:
            packet_result = Handle_QLaunchArch (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QSetDisableASLR:
            packet_result = Handle_QSetDisableASLR (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QSetDetachOnError:
            packet_result = Handle_QSetDetachOnError (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QSetSTDIN:
            packet_result = Handle_QSetSTDIN (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QSetSTDOUT:
            packet_result = Handle_QSetSTDOUT (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QSetSTDERR:
            packet_result = Handle_QSetSTDERR (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QSetWorkingDir:
            packet_result = Handle_QSetWorkingDir (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QStartNoAckMode:
            packet_result = Handle_QStartNoAckMode (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qPlatform_mkdir:
            packet_result = Handle_qPlatform_mkdir (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qPlatform_chmod:
            packet_result = Handle_qPlatform_chmod (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qPlatform_shell:
            packet_result = Handle_qPlatform_shell (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_C:
            packet_result = Handle_C (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_c:
            packet_result = Handle_c (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vCont:
            packet_result = Handle_vCont (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vCont_actions:
            packet_result = Handle_vCont_actions (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_stop_reason: // ?
            packet_result = Handle_stop_reason (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vFile_open:
            packet_result = Handle_vFile_Open (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vFile_close:
            packet_result = Handle_vFile_Close (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vFile_pread:
            packet_result = Handle_vFile_pRead (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vFile_pwrite:
            packet_result = Handle_vFile_pWrite (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vFile_size:
            packet_result = Handle_vFile_Size (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vFile_mode:
            packet_result = Handle_vFile_Mode (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vFile_exists:
            packet_result = Handle_vFile_Exists (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vFile_stat:
            packet_result = Handle_vFile_Stat (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vFile_md5:
            packet_result = Handle_vFile_MD5 (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vFile_symlink:
            packet_result = Handle_vFile_symlink (packet);
            break;
            
        case StringExtractorGDBRemote::eServerPacketType_vFile_unlink:
            packet_result = Handle_vFile_unlink (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qRegisterInfo:
            packet_result = Handle_qRegisterInfo (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qfThreadInfo:
            packet_result = Handle_qfThreadInfo (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qsThreadInfo:
            packet_result = Handle_qsThreadInfo (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_p:
            packet_result = Handle_p (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_P:
            packet_result = Handle_P (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_H:
            packet_result = Handle_H (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_m:
            packet_result = Handle_m (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_M:
            packet_result = Handle_M (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qMemoryRegionInfoSupported:
            packet_result = Handle_qMemoryRegionInfoSupported (packet);
            break;
                
        case StringExtractorGDBRemote::eServerPacketType_qMemoryRegionInfo:
            packet_result = Handle_qMemoryRegionInfo (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_interrupt:
            if (IsGdbServer ())
                packet_result = Handle_interrupt (packet);
            else
            {
                error.SetErrorString("interrupt received");
                interrupt = true;
            }
            break;

        case StringExtractorGDBRemote::eServerPacketType_Z:
            packet_result = Handle_Z (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_z:
            packet_result = Handle_z (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_s:
            packet_result = Handle_s (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qSupported:
            packet_result = Handle_qSupported (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QThreadSuffixSupported:
            packet_result = Handle_QThreadSuffixSupported (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QListThreadsInStopReply:
            packet_result = Handle_QListThreadsInStopReply (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_qXfer_auxv_read:
            packet_result = Handle_qXfer_auxv_read (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QSaveRegisterState:
            packet_result = Handle_QSaveRegisterState (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_QRestoreRegisterState:
            packet_result = Handle_QRestoreRegisterState (packet);
            break;

        case StringExtractorGDBRemote::eServerPacketType_vAttach:
            packet_result = Handle_vAttach (packet);
            break;
        }
    }
    else
    {
        if (!IsConnected())
        {
            error.SetErrorString("lost connection");
            quit = true;
        }
        else
        {
            error.SetErrorString("timeout");
        }
    }

    // Check if anything occurred that would force us to want to exit.
    if (m_exit_now)
        quit = true;

    return packet_result;
}

lldb_private::Error
GDBRemoteCommunicationServer::SetLaunchArguments (const char *const args[], int argc)
{
    if ((argc < 1) || !args || !args[0] || !args[0][0])
        return lldb_private::Error ("%s: no process command line specified to launch", __FUNCTION__);

    m_process_launch_info.SetArguments (const_cast<const char**> (args), true);
    return lldb_private::Error ();
}

lldb_private::Error
GDBRemoteCommunicationServer::SetLaunchFlags (unsigned int launch_flags)
{
    m_process_launch_info.GetFlags ().Set (launch_flags);
    return lldb_private::Error ();
}

lldb_private::Error
GDBRemoteCommunicationServer::LaunchProcess ()
{
    // FIXME This looks an awful lot like we could override this in
    // derived classes, one for lldb-platform, the other for lldb-gdbserver.
    if (IsGdbServer ())
        return LaunchDebugServerProcess ();
    else
        return LaunchPlatformProcess ();
}

lldb_private::Error
GDBRemoteCommunicationServer::LaunchDebugServerProcess ()
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

    if (!m_process_launch_info.GetArguments ().GetArgumentCount ())
        return lldb_private::Error ("%s: no process command line specified to launch", __FUNCTION__);

    lldb_private::Error error;
    {
        Mutex::Locker locker (m_debugged_process_mutex);
        assert (!m_debugged_process_sp && "lldb-gdbserver creating debugged process but one already exists");
        error = m_platform_sp->LaunchNativeProcess (
            m_process_launch_info,
            *this,
            m_debugged_process_sp);
    }

    if (!error.Success ())
    {
        fprintf (stderr, "%s: failed to launch executable %s", __FUNCTION__, m_process_launch_info.GetArguments ().GetArgumentAtIndex (0));
        return error;
    }

    // Setup stdout/stderr mapping from inferior.
    auto terminal_fd = m_debugged_process_sp->GetTerminalFileDescriptor ();
    if (terminal_fd >= 0)
    {
        if (log)
            log->Printf ("ProcessGDBRemoteCommunicationServer::%s setting inferior STDIO fd to %d", __FUNCTION__, terminal_fd);
        error = SetSTDIOFileDescriptor (terminal_fd);
        if (error.Fail ())
            return error;
    }
    else
    {
        if (log)
            log->Printf ("ProcessGDBRemoteCommunicationServer::%s ignoring inferior STDIO since terminal fd reported as %d", __FUNCTION__, terminal_fd);
    }

    printf ("Launched '%s' as process %" PRIu64 "...\n", m_process_launch_info.GetArguments ().GetArgumentAtIndex (0), m_process_launch_info.GetProcessID ());

    // Add to list of spawned processes.
    lldb::pid_t pid;
    if ((pid = m_process_launch_info.GetProcessID ()) != LLDB_INVALID_PROCESS_ID)
    {
        // add to spawned pids
        {
            Mutex::Locker locker (m_spawned_pids_mutex);
            // On an lldb-gdbserver, we would expect there to be only one.
            assert (m_spawned_pids.empty () && "lldb-gdbserver adding tracked process but one already existed");
            m_spawned_pids.insert (pid);
        }
    }

    if (error.Success ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s beginning check to wait for launched application to hit first stop", __FUNCTION__);

        int iteration = 0;
        // Wait for the process to hit its first stop state.
        while (!StateIsStoppedState (m_debugged_process_sp->GetState (), false))
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s waiting for launched process to hit first stop (%d)...", __FUNCTION__, iteration++);

            // FIXME use a finer granularity.
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s launched application has hit first stop", __FUNCTION__);

    }

    return error;
}

lldb_private::Error
GDBRemoteCommunicationServer::LaunchPlatformProcess ()
{
    if (!m_process_launch_info.GetArguments ().GetArgumentCount ())
        return lldb_private::Error ("%s: no process command line specified to launch", __FUNCTION__);

    // specify the process monitor if not already set.  This should
    // generally be what happens since we need to reap started
    // processes.
    if (!m_process_launch_info.GetMonitorProcessCallback ())
        m_process_launch_info.SetMonitorProcessCallback(ReapDebuggedProcess, this, false);

    lldb_private::Error error = m_platform_sp->LaunchProcess (m_process_launch_info);
    if (!error.Success ())
    {
        fprintf (stderr, "%s: failed to launch executable %s", __FUNCTION__, m_process_launch_info.GetArguments ().GetArgumentAtIndex (0));
        return error;
    }

    printf ("Launched '%s' as process %" PRIu64 "...\n", m_process_launch_info.GetArguments ().GetArgumentAtIndex (0), m_process_launch_info.GetProcessID());

    // add to list of spawned processes.  On an lldb-gdbserver, we
    // would expect there to be only one.
    lldb::pid_t pid;
    if ( (pid = m_process_launch_info.GetProcessID()) != LLDB_INVALID_PROCESS_ID )
    {
        // add to spawned pids
        {
            Mutex::Locker locker (m_spawned_pids_mutex);
            m_spawned_pids.insert(pid);
        }
    }

    return error;
}

lldb_private::Error
GDBRemoteCommunicationServer::AttachToProcess (lldb::pid_t pid)
{
    Error error;

    if (!IsGdbServer ())
    {
        error.SetErrorString("cannot AttachToProcess () unless process is lldb-gdbserver");
        return error;
    }

    Log *log (GetLogIfAnyCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64, __FUNCTION__, pid);

    // Scope for mutex locker.
    {
        // Before we try to attach, make sure we aren't already monitoring something else.
        Mutex::Locker locker (m_spawned_pids_mutex);
        if (!m_spawned_pids.empty ())
        {
            error.SetErrorStringWithFormat ("cannot attach to a process %" PRIu64 " when another process with pid %" PRIu64 " is being debugged.", pid, *m_spawned_pids.begin());
            return error;
        }

        // Try to attach.
        error = m_platform_sp->AttachNativeProcess (pid, *this, m_debugged_process_sp);
        if (!error.Success ())
        {
            fprintf (stderr, "%s: failed to attach to process %" PRIu64 ": %s", __FUNCTION__, pid, error.AsCString ());
            return error;
        }

        // Setup stdout/stderr mapping from inferior.
        auto terminal_fd = m_debugged_process_sp->GetTerminalFileDescriptor ();
        if (terminal_fd >= 0)
        {
            if (log)
                log->Printf ("ProcessGDBRemoteCommunicationServer::%s setting inferior STDIO fd to %d", __FUNCTION__, terminal_fd);
            error = SetSTDIOFileDescriptor (terminal_fd);
            if (error.Fail ())
                return error;
        }
        else
        {
            if (log)
                log->Printf ("ProcessGDBRemoteCommunicationServer::%s ignoring inferior STDIO since terminal fd reported as %d", __FUNCTION__, terminal_fd);
        }

        printf ("Attached to process %" PRIu64 "...\n", pid);

        // Add to list of spawned processes.
        assert (m_spawned_pids.empty () && "lldb-gdbserver adding tracked process but one already existed");
        m_spawned_pids.insert (pid);

        return error;
    }
}

void
GDBRemoteCommunicationServer::InitializeDelegate (lldb_private::NativeProcessProtocol *process)
{
    assert (process && "process cannot be NULL");
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
    if (log)
    {
        log->Printf ("GDBRemoteCommunicationServer::%s called with NativeProcessProtocol pid %" PRIu64 ", current state: %s",
                __FUNCTION__,
                process->GetID (),
                StateAsCString (process->GetState ()));
    }
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendWResponse (lldb_private::NativeProcessProtocol *process)
{
    assert (process && "process cannot be NULL");
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

    // send W notification
    ExitType exit_type = ExitType::eExitTypeInvalid;
    int return_code = 0;
    std::string exit_description;

    const bool got_exit_info = process->GetExitStatus (&exit_type, &return_code, exit_description);
    if (!got_exit_info)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 ", failed to retrieve process exit status", __FUNCTION__, process->GetID ());

        StreamGDBRemote response;
        response.PutChar ('E');
        response.PutHex8 (GDBRemoteServerError::eErrorExitStatus);
        return SendPacketNoLock(response.GetData(), response.GetSize());
    }
    else
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 ", returning exit type %d, return code %d [%s]", __FUNCTION__, process->GetID (), exit_type, return_code, exit_description.c_str ());

        StreamGDBRemote response;

        char return_type_code;
        switch (exit_type)
        {
            case ExitType::eExitTypeExit:   return_type_code = 'W'; break;
            case ExitType::eExitTypeSignal: return_type_code = 'X'; break;
            case ExitType::eExitTypeStop:   return_type_code = 'S'; break;

            case ExitType::eExitTypeInvalid:
            default:                        return_type_code = 'E'; break;
        }
        response.PutChar (return_type_code);

        // POSIX exit status limited to unsigned 8 bits.
        response.PutHex8 (return_code);

        return SendPacketNoLock(response.GetData(), response.GetSize());
    }
}

static void
AppendHexValue (StreamString &response, const uint8_t* buf, uint32_t buf_size, bool swap)
{
    int64_t i;
    if (swap)
    {
        for (i = buf_size-1; i >= 0; i--)
            response.PutHex8 (buf[i]);
    }
    else
    {
        for (i = 0; i < buf_size; i++)
            response.PutHex8 (buf[i]);
    }
}

static void
WriteRegisterValueInHexFixedWidth (StreamString &response,
                                   NativeRegisterContextSP &reg_ctx_sp,
                                   const RegisterInfo &reg_info,
                                   const RegisterValue *reg_value_p)
{
    RegisterValue reg_value;
    if (!reg_value_p)
    {
        Error error = reg_ctx_sp->ReadRegister (&reg_info, reg_value);
        if (error.Success ())
            reg_value_p = &reg_value;
        // else log.
    }

    if (reg_value_p)
    {
        AppendHexValue (response, (const uint8_t*) reg_value_p->GetBytes (), reg_value_p->GetByteSize (), false);
    }
    else
    {
        // Zero-out any unreadable values.
        if (reg_info.byte_size > 0)
        {
            std::basic_string<uint8_t> zeros(reg_info.byte_size, '\0');
            AppendHexValue (response, zeros.data(), zeros.size(), false);
        }
    }
}

// WriteGdbRegnumWithFixedWidthHexRegisterValue (response, reg_ctx_sp, *reg_info_p, reg_value);


static void
WriteGdbRegnumWithFixedWidthHexRegisterValue (StreamString &response,
                                              NativeRegisterContextSP &reg_ctx_sp,
                                              const RegisterInfo &reg_info,
                                              const RegisterValue &reg_value)
{
    // Output the register number as 'NN:VVVVVVVV;' where NN is a 2 bytes HEX
    // gdb register number, and VVVVVVVV is the correct number of hex bytes
    // as ASCII for the register value.
    if (reg_info.kinds[eRegisterKindGDB] == LLDB_INVALID_REGNUM)
        return;

    response.Printf ("%.02x:", reg_info.kinds[eRegisterKindGDB]);
    WriteRegisterValueInHexFixedWidth (response, reg_ctx_sp, reg_info, &reg_value);
    response.PutChar (';');
}


GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendStopReplyPacketForThread (lldb::tid_t tid)
{
    Log *log (GetLogIfAnyCategoriesSet (LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

    // Ensure we're llgs.
    if (!IsGdbServer ())
    {
        // Only supported on llgs
        return SendUnimplementedResponse ("");
    }

    // Ensure we have a debugged process.
    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
        return SendErrorResponse (50);

    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s preparing packet for pid %" PRIu64 " tid %" PRIu64,
                __FUNCTION__, m_debugged_process_sp->GetID (), tid);

    // Ensure we can get info on the given thread.
    NativeThreadProtocolSP thread_sp (m_debugged_process_sp->GetThreadByID (tid));
    if (!thread_sp)
        return SendErrorResponse (51);

    // Grab the reason this thread stopped.
    struct ThreadStopInfo tid_stop_info;
    if (!thread_sp->GetStopReason (tid_stop_info))
        return SendErrorResponse (52);

    const bool did_exec = tid_stop_info.reason == eStopReasonExec;
    // FIXME implement register handling for exec'd inferiors.
    // if (did_exec)
    // {
    //     const bool force = true;
    //     InitializeRegisters(force);
    // }

    StreamString response;
    // Output the T packet with the thread
    response.PutChar ('T');
    int signum = tid_stop_info.details.signal.signo;
    if (log)
    {
        log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " tid %" PRIu64 " got signal signo = %d, reason = %d, exc_type = %" PRIu64, 
                __FUNCTION__,
                m_debugged_process_sp->GetID (),
                tid,
                signum,
                tid_stop_info.reason,
                tid_stop_info.details.exception.type);
    }

    switch (tid_stop_info.reason)
    {
    case eStopReasonSignal:
    case eStopReasonException:
        signum = thread_sp->TranslateStopInfoToGdbSignal (tid_stop_info);
        break;
    default:
        signum = 0;
        if (log)
        {
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " tid %" PRIu64 " has stop reason %d, using signo = 0 in stop reply response",
                __FUNCTION__,
                m_debugged_process_sp->GetID (),
                tid,
                tid_stop_info.reason);
        }
        break;
    }

    // Print the signal number.
    response.PutHex8 (signum & 0xff);

    // Include the tid.
    response.Printf ("thread:%" PRIx64 ";", tid);

    // Include the thread name if there is one.
    const char *thread_name = thread_sp->GetName ();
    if (thread_name && thread_name[0])
    {
        size_t thread_name_len = strlen(thread_name);

        if (::strcspn (thread_name, "$#+-;:") == thread_name_len)
        {
            response.PutCString ("name:");
            response.PutCString (thread_name);
        }
        else
        {
            // The thread name contains special chars, send as hex bytes.
            response.PutCString ("hexname:");
            response.PutCStringAsRawHex8 (thread_name);
        }
        response.PutChar (';');
    }

    // FIXME look for analog
    // thread_identifier_info_data_t thread_ident_info;
    // if (DNBThreadGetIdentifierInfo (pid, tid, &thread_ident_info))
    // {
    //     if (thread_ident_info.dispatch_qaddr != 0)
    //         ostrm << std::hex << "qaddr:" << thread_ident_info.dispatch_qaddr << ';';
    // }

    // If a 'QListThreadsInStopReply' was sent to enable this feature, we
    // will send all thread IDs back in the "threads" key whose value is
    // a list of hex thread IDs separated by commas:
    //  "threads:10a,10b,10c;"
    // This will save the debugger from having to send a pair of qfThreadInfo
    // and qsThreadInfo packets, but it also might take a lot of room in the
    // stop reply packet, so it must be enabled only on systems where there
    // are no limits on packet lengths.
    if (m_list_threads_in_stop_reply)
    {
        response.PutCString ("threads:");

        uint32_t thread_index = 0;
        NativeThreadProtocolSP listed_thread_sp;
        for (listed_thread_sp = m_debugged_process_sp->GetThreadAtIndex (thread_index); listed_thread_sp; ++thread_index, listed_thread_sp = m_debugged_process_sp->GetThreadAtIndex (thread_index))
        {
            if (thread_index > 0)
                response.PutChar (',');
            response.Printf ("%" PRIx64, listed_thread_sp->GetID ());
        }
        response.PutChar (';');
    }

    //
    // Expedite registers.
    //

    // Grab the register context.
    NativeRegisterContextSP reg_ctx_sp = thread_sp->GetRegisterContext ();
    if (reg_ctx_sp)
    {
        // Expedite all registers in the first register set (i.e. should be GPRs) that are not contained in other registers.
        const RegisterSet *reg_set_p;
        if (reg_ctx_sp->GetRegisterSetCount () > 0 && ((reg_set_p = reg_ctx_sp->GetRegisterSet (0)) != nullptr))
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s expediting registers from set '%s' (registers set count: %zu)", __FUNCTION__, reg_set_p->name ? reg_set_p->name : "<unnamed-set>", reg_set_p->num_registers);

            for (const uint32_t *reg_num_p = reg_set_p->registers; *reg_num_p != LLDB_INVALID_REGNUM; ++reg_num_p)
            {
                const RegisterInfo *const reg_info_p = reg_ctx_sp->GetRegisterInfoAtIndex (*reg_num_p);
                if (reg_info_p == nullptr)
                {
                    if (log)
                        log->Printf ("GDBRemoteCommunicationServer::%s failed to get register info for register set '%s', register index %" PRIu32, __FUNCTION__, reg_set_p->name ? reg_set_p->name : "<unnamed-set>", *reg_num_p);
                }
                else if (reg_info_p->value_regs == nullptr)
                {
                    // Only expediate registers that are not contained in other registers.
                    RegisterValue reg_value;
                    Error error = reg_ctx_sp->ReadRegister (reg_info_p, reg_value);
                    if (error.Success ())
                        WriteGdbRegnumWithFixedWidthHexRegisterValue (response, reg_ctx_sp, *reg_info_p, reg_value);
                    else
                    {
                        if (log)
                            log->Printf ("GDBRemoteCommunicationServer::%s failed to read register '%s' index %" PRIu32 ": %s", __FUNCTION__, reg_info_p->name ? reg_info_p->name : "<unnamed-register>", *reg_num_p, error.AsCString ());

                    }
                }
            }
        }
    }

    if (did_exec)
    {
        response.PutCString ("reason:exec;");
    }
    else if ((tid_stop_info.reason == eStopReasonException) && tid_stop_info.details.exception.type)
    {
        response.PutCString ("metype:");
        response.PutHex64 (tid_stop_info.details.exception.type);
        response.PutCString (";mecount:");
        response.PutHex32 (tid_stop_info.details.exception.data_count);
        response.PutChar (';');

        for (uint32_t i = 0; i < tid_stop_info.details.exception.data_count; ++i)
        {
            response.PutCString ("medata:");
            response.PutHex64 (tid_stop_info.details.exception.data[i]);
            response.PutChar (';');
        }
    }

    return SendPacketNoLock (response.GetData(), response.GetSize());
}

void
GDBRemoteCommunicationServer::HandleInferiorState_Exited (lldb_private::NativeProcessProtocol *process)
{
    assert (process && "process cannot be NULL");

    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s called", __FUNCTION__);

    // Send the exit result, and don't flush output.
    // Note: flushing output here would join the inferior stdio reflection thread, which
    // would gunk up the waitpid monitor thread that is calling this.
    PacketResult result = SendStopReasonForState (StateType::eStateExited, false);
    if (result != PacketResult::Success)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed to send stop notification for PID %" PRIu64 ", state: eStateExited", __FUNCTION__, process->GetID ());
    }

    // Remove the process from the list of spawned pids.
    {
        Mutex::Locker locker (m_spawned_pids_mutex);
        if (m_spawned_pids.erase (process->GetID ()) < 1)
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s failed to remove PID %" PRIu64 " from the spawned pids list", __FUNCTION__, process->GetID ());

        }
    }

    // FIXME can't do this yet - since process state propagation is currently
    // synchronous, it is running off the NativeProcessProtocol's innards and
    // will tear down the NPP while it still has code to execute.
#if 0
    // Clear the NativeProcessProtocol pointer.
    {
        Mutex::Locker locker (m_debugged_process_mutex);
        m_debugged_process_sp.reset();
    }
#endif

    // Close the pipe to the inferior terminal i/o if we launched it
    // and set one up.  Otherwise, 'k' and its flush of stdio could
    // end up waiting on a thread join that will never end.  Consider
    // adding a timeout to the connection thread join call so we
    // can avoid that scenario altogether.
    MaybeCloseInferiorTerminalConnection ();

    // We are ready to exit the debug monitor.
    m_exit_now = true;
}

void
GDBRemoteCommunicationServer::HandleInferiorState_Stopped (lldb_private::NativeProcessProtocol *process)
{
    assert (process && "process cannot be NULL");

    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s called", __FUNCTION__);

    // Send the stop reason unless this is the stop after the
    // launch or attach.
    switch (m_inferior_prev_state)
    {
        case eStateLaunching:
        case eStateAttaching:
            // Don't send anything per debugserver behavior.
            break;
        default:
            // In all other cases, send the stop reason.
            PacketResult result = SendStopReasonForState (StateType::eStateStopped, false);
            if (result != PacketResult::Success)
            {
                if (log)
                    log->Printf ("GDBRemoteCommunicationServer::%s failed to send stop notification for PID %" PRIu64 ", state: eStateExited", __FUNCTION__, process->GetID ());
            }
            break;
    }
}

void
GDBRemoteCommunicationServer::ProcessStateChanged (lldb_private::NativeProcessProtocol *process, lldb::StateType state)
{
    assert (process && "process cannot be NULL");
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
    if (log)
    {
        log->Printf ("GDBRemoteCommunicationServer::%s called with NativeProcessProtocol pid %" PRIu64 ", state: %s",
                __FUNCTION__,
                process->GetID (),
                StateAsCString (state));
    }

    switch (state)
    {
    case StateType::eStateExited:
        HandleInferiorState_Exited (process);
        break;

    case StateType::eStateStopped:
        HandleInferiorState_Stopped (process);
        break;

    default:
        if (log)
        {
            log->Printf ("GDBRemoteCommunicationServer::%s didn't handle state change for pid %" PRIu64 ", new state: %s",
                    __FUNCTION__,
                    process->GetID (),
                    StateAsCString (state));
        }
        break;
    }

    // Remember the previous state reported to us.
    m_inferior_prev_state = state;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendONotification (const char *buffer, uint32_t len)
{
    if ((buffer == nullptr) || (len == 0))
    {
        // Nothing to send.
        return PacketResult::Success;
    }

    StreamString response;
    response.PutChar ('O');
    response.PutBytesAsRawHex8 (buffer, len);

    return SendPacketNoLock (response.GetData (), response.GetSize ());
}

lldb_private::Error
GDBRemoteCommunicationServer::SetSTDIOFileDescriptor (int fd)
{
    Error error;

    // Set up the Read Thread for reading/handling process I/O
    std::unique_ptr<ConnectionFileDescriptor> conn_up (new ConnectionFileDescriptor (fd, true));
    if (!conn_up)
    {
        error.SetErrorString ("failed to create ConnectionFileDescriptor");
        return error;
    }

    m_stdio_communication.SetConnection (conn_up.release());
    if (!m_stdio_communication.IsConnected ())
    {
        error.SetErrorString ("failed to set connection for inferior I/O communication");
        return error;
    }

    m_stdio_communication.SetReadThreadBytesReceivedCallback (STDIOReadThreadBytesReceived, this);
    m_stdio_communication.StartReadThread();

    return error;
}

void
GDBRemoteCommunicationServer::STDIOReadThreadBytesReceived (void *baton, const void *src, size_t src_len)
{
    GDBRemoteCommunicationServer *server = reinterpret_cast<GDBRemoteCommunicationServer*> (baton);
    static_cast<void> (server->SendONotification (static_cast<const char *>(src), src_len));
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendUnimplementedResponse (const char *)
{
    // TODO: Log the packet we aren't handling...
    return SendPacketNoLock ("", 0);
}


GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendErrorResponse (uint8_t err)
{
    char packet[16];
    int packet_len = ::snprintf (packet, sizeof(packet), "E%2.2x", err);
    assert (packet_len < (int)sizeof(packet));
    return SendPacketNoLock (packet, packet_len);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendIllFormedResponse (const StringExtractorGDBRemote &failed_packet, const char *message)
{
    Log *log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_PACKETS));
    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s: ILLFORMED: '%s' (%s)", __FUNCTION__, failed_packet.GetStringRef ().c_str (), message ? message : "");
    return SendErrorResponse (0x03);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendOKResponse ()
{
    return SendPacketNoLock ("OK", 2);
}

bool
GDBRemoteCommunicationServer::HandshakeWithClient(Error *error_ptr)
{
    return GetAck() == PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qHostInfo (StringExtractorGDBRemote &packet)
{
    StreamString response;

    // $cputype:16777223;cpusubtype:3;ostype:Darwin;vendor:apple;endian:little;ptrsize:8;#00

    ArchSpec host_arch (Host::GetArchitecture ());
    const llvm::Triple &host_triple = host_arch.GetTriple();
    response.PutCString("triple:");
    response.PutCString(host_triple.getTriple().c_str());
    response.Printf (";ptrsize:%u;",host_arch.GetAddressByteSize());

    const char* distribution_id = host_arch.GetDistributionId ().AsCString ();
    if (distribution_id)
    {
        response.PutCString("distribution_id:");
        response.PutCStringAsRawHex8(distribution_id);
        response.PutCString(";");
    }

    // Only send out MachO info when lldb-platform/llgs is running on a MachO host.
#if defined(__APPLE__)
    uint32_t cpu = host_arch.GetMachOCPUType();
    uint32_t sub = host_arch.GetMachOCPUSubType();
    if (cpu != LLDB_INVALID_CPUTYPE)
        response.Printf ("cputype:%u;", cpu);
    if (sub != LLDB_INVALID_CPUTYPE)
        response.Printf ("cpusubtype:%u;", sub);

    if (cpu == ArchSpec::kCore_arm_any)
        response.Printf("watchpoint_exceptions_received:before;");   // On armv7 we use "synchronous" watchpoints which means the exception is delivered before the instruction executes.
    else
        response.Printf("watchpoint_exceptions_received:after;");
#else
    response.Printf("watchpoint_exceptions_received:after;");
#endif

    switch (lldb::endian::InlHostByteOrder())
    {
    case eByteOrderBig:     response.PutCString ("endian:big;"); break;
    case eByteOrderLittle:  response.PutCString ("endian:little;"); break;
    case eByteOrderPDP:     response.PutCString ("endian:pdp;"); break;
    default:                response.PutCString ("endian:unknown;"); break;
    }

    uint32_t major = UINT32_MAX;
    uint32_t minor = UINT32_MAX;
    uint32_t update = UINT32_MAX;
    if (HostInfo::GetOSVersion(major, minor, update))
    {
        if (major != UINT32_MAX)
        {
            response.Printf("os_version:%u", major);
            if (minor != UINT32_MAX)
            {
                response.Printf(".%u", minor);
                if (update != UINT32_MAX)
                    response.Printf(".%u", update);
            }
            response.PutChar(';');
        }
    }

    std::string s;
#if !defined(__linux__)
    if (HostInfo::GetOSBuildString(s))
    {
        response.PutCString ("os_build:");
        response.PutCStringAsRawHex8(s.c_str());
        response.PutChar(';');
    }
    if (HostInfo::GetOSKernelDescription(s))
    {
        response.PutCString ("os_kernel:");
        response.PutCStringAsRawHex8(s.c_str());
        response.PutChar(';');
    }
#endif

#if defined(__APPLE__)

#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
    // For iOS devices, we are connected through a USB Mux so we never pretend
    // to actually have a hostname as far as the remote lldb that is connecting
    // to this lldb-platform is concerned
    response.PutCString ("hostname:");
    response.PutCStringAsRawHex8("127.0.0.1");
    response.PutChar(';');
#else   // #if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
    if (HostInfo::GetHostname(s))
    {
        response.PutCString ("hostname:");
        response.PutCStringAsRawHex8(s.c_str());
        response.PutChar(';');
    }
#endif  // #if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)

#else   // #if defined(__APPLE__)
    if (HostInfo::GetHostname(s))
    {
        response.PutCString ("hostname:");
        response.PutCStringAsRawHex8(s.c_str());
        response.PutChar(';');
    }
#endif  // #if defined(__APPLE__)

    return SendPacketNoLock (response.GetData(), response.GetSize());
}

static void
CreateProcessInfoResponse (const ProcessInstanceInfo &proc_info, StreamString &response)
{
    response.Printf ("pid:%" PRIu64 ";ppid:%" PRIu64 ";uid:%i;gid:%i;euid:%i;egid:%i;",
                     proc_info.GetProcessID(),
                     proc_info.GetParentProcessID(),
                     proc_info.GetUserID(),
                     proc_info.GetGroupID(),
                     proc_info.GetEffectiveUserID(),
                     proc_info.GetEffectiveGroupID());
    response.PutCString ("name:");
    response.PutCStringAsRawHex8(proc_info.GetName());
    response.PutChar(';');
    const ArchSpec &proc_arch = proc_info.GetArchitecture();
    if (proc_arch.IsValid())
    {
        const llvm::Triple &proc_triple = proc_arch.GetTriple();
        response.PutCString("triple:");
        response.PutCString(proc_triple.getTriple().c_str());
        response.PutChar(';');
    }
}

static void
CreateProcessInfoResponse_DebugServerStyle (const ProcessInstanceInfo &proc_info, StreamString &response)
{
    response.Printf ("pid:%" PRIx64 ";parent-pid:%" PRIx64 ";real-uid:%x;real-gid:%x;effective-uid:%x;effective-gid:%x;",
                     proc_info.GetProcessID(),
                     proc_info.GetParentProcessID(),
                     proc_info.GetUserID(),
                     proc_info.GetGroupID(),
                     proc_info.GetEffectiveUserID(),
                     proc_info.GetEffectiveGroupID());

    const ArchSpec &proc_arch = proc_info.GetArchitecture();
    if (proc_arch.IsValid())
    {
        const uint32_t cpu_type = proc_arch.GetMachOCPUType();
        if (cpu_type != 0)
            response.Printf ("cputype:%" PRIx32 ";", cpu_type);
        
        const uint32_t cpu_subtype = proc_arch.GetMachOCPUSubType();
        if (cpu_subtype != 0)
            response.Printf ("cpusubtype:%" PRIx32 ";", cpu_subtype);
        
        const llvm::Triple &proc_triple = proc_arch.GetTriple();
        const std::string vendor = proc_triple.getVendorName ();
        if (!vendor.empty ())
            response.Printf ("vendor:%s;", vendor.c_str ());

        std::string ostype = proc_triple.getOSName ();
        // Adjust so ostype reports ios for Apple/ARM and Apple/ARM64.
        if (proc_triple.getVendor () == llvm::Triple::Apple)
        {
            switch (proc_triple.getArch ())
            {
                case llvm::Triple::arm:
                case llvm::Triple::aarch64:
                    ostype = "ios";
                    break;
                default:
                    // No change.
                    break;
            }
        }
        response.Printf ("ostype:%s;", ostype.c_str ());


        switch (proc_arch.GetByteOrder ())
        {
            case lldb::eByteOrderLittle: response.PutCString ("endian:little;"); break;
            case lldb::eByteOrderBig:    response.PutCString ("endian:big;");    break;
            case lldb::eByteOrderPDP:    response.PutCString ("endian:pdp;");    break;
            default:
                // Nothing.
                break;
        }

        if (proc_triple.isArch64Bit ())
            response.PutCString ("ptrsize:8;");
        else if (proc_triple.isArch32Bit ())
            response.PutCString ("ptrsize:4;");
        else if (proc_triple.isArch16Bit ())
            response.PutCString ("ptrsize:2;");
    }

}


GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qProcessInfo (StringExtractorGDBRemote &packet)
{
    // Only the gdb server handles this.
    if (!IsGdbServer ())
        return SendUnimplementedResponse (packet.GetStringRef ().c_str ());
    
    // Fail if we don't have a current process.
    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
        return SendErrorResponse (68);
    
    ProcessInstanceInfo proc_info;
    if (Host::GetProcessInfo (m_debugged_process_sp->GetID (), proc_info))
    {
        StreamString response;
        CreateProcessInfoResponse_DebugServerStyle(proc_info, response);
        return SendPacketNoLock (response.GetData (), response.GetSize ());
    }
    
    return SendErrorResponse (1);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qProcessInfoPID (StringExtractorGDBRemote &packet)
{
    // Packet format: "qProcessInfoPID:%i" where %i is the pid
    packet.SetFilePos(::strlen ("qProcessInfoPID:"));
    lldb::pid_t pid = packet.GetU32 (LLDB_INVALID_PROCESS_ID);
    if (pid != LLDB_INVALID_PROCESS_ID)
    {
        ProcessInstanceInfo proc_info;
        if (Host::GetProcessInfo(pid, proc_info))
        {
            StreamString response;
            CreateProcessInfoResponse (proc_info, response);
            return SendPacketNoLock (response.GetData(), response.GetSize());
        }
    }
    return SendErrorResponse (1);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qfProcessInfo (StringExtractorGDBRemote &packet)
{
    m_proc_infos_index = 0;
    m_proc_infos.Clear();

    ProcessInstanceInfoMatch match_info;
    packet.SetFilePos(::strlen ("qfProcessInfo"));
    if (packet.GetChar() == ':')
    {

        std::string key;
        std::string value;
        while (packet.GetNameColonValue(key, value))
        {
            bool success = true;
            if (key.compare("name") == 0)
            {
                StringExtractor extractor;
                extractor.GetStringRef().swap(value);
                extractor.GetHexByteString (value);
                match_info.GetProcessInfo().GetExecutableFile().SetFile(value.c_str(), false);
            }
            else if (key.compare("name_match") == 0)
            {
                if (value.compare("equals") == 0)
                {
                    match_info.SetNameMatchType (eNameMatchEquals);
                }
                else if (value.compare("starts_with") == 0)
                {
                    match_info.SetNameMatchType (eNameMatchStartsWith);
                }
                else if (value.compare("ends_with") == 0)
                {
                    match_info.SetNameMatchType (eNameMatchEndsWith);
                }
                else if (value.compare("contains") == 0)
                {
                    match_info.SetNameMatchType (eNameMatchContains);
                }
                else if (value.compare("regex") == 0)
                {
                    match_info.SetNameMatchType (eNameMatchRegularExpression);
                }
                else
                {
                    success = false;
                }
            }
            else if (key.compare("pid") == 0)
            {
                match_info.GetProcessInfo().SetProcessID (Args::StringToUInt32(value.c_str(), LLDB_INVALID_PROCESS_ID, 0, &success));
            }
            else if (key.compare("parent_pid") == 0)
            {
                match_info.GetProcessInfo().SetParentProcessID (Args::StringToUInt32(value.c_str(), LLDB_INVALID_PROCESS_ID, 0, &success));
            }
            else if (key.compare("uid") == 0)
            {
                match_info.GetProcessInfo().SetUserID (Args::StringToUInt32(value.c_str(), UINT32_MAX, 0, &success));
            }
            else if (key.compare("gid") == 0)
            {
                match_info.GetProcessInfo().SetGroupID (Args::StringToUInt32(value.c_str(), UINT32_MAX, 0, &success));
            }
            else if (key.compare("euid") == 0)
            {
                match_info.GetProcessInfo().SetEffectiveUserID (Args::StringToUInt32(value.c_str(), UINT32_MAX, 0, &success));
            }
            else if (key.compare("egid") == 0)
            {
                match_info.GetProcessInfo().SetEffectiveGroupID (Args::StringToUInt32(value.c_str(), UINT32_MAX, 0, &success));
            }
            else if (key.compare("all_users") == 0)
            {
                match_info.SetMatchAllUsers(Args::StringToBoolean(value.c_str(), false, &success));
            }
            else if (key.compare("triple") == 0)
            {
                match_info.GetProcessInfo().GetArchitecture().SetTriple (value.c_str(), NULL);
            }
            else
            {
                success = false;
            }

            if (!success)
                return SendErrorResponse (2);
        }
    }

    if (Host::FindProcesses (match_info, m_proc_infos))
    {
        // We found something, return the first item by calling the get
        // subsequent process info packet handler...
        return Handle_qsProcessInfo (packet);
    }
    return SendErrorResponse (3);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qsProcessInfo (StringExtractorGDBRemote &packet)
{
    if (m_proc_infos_index < m_proc_infos.GetSize())
    {
        StreamString response;
        CreateProcessInfoResponse (m_proc_infos.GetProcessInfoAtIndex(m_proc_infos_index), response);
        ++m_proc_infos_index;
        return SendPacketNoLock (response.GetData(), response.GetSize());
    }
    return SendErrorResponse (4);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qUserName (StringExtractorGDBRemote &packet)
{
    // Packet format: "qUserName:%i" where %i is the uid
    packet.SetFilePos(::strlen ("qUserName:"));
    uint32_t uid = packet.GetU32 (UINT32_MAX);
    if (uid != UINT32_MAX)
    {
        std::string name;
        if (Host::GetUserName (uid, name))
        {
            StreamString response;
            response.PutCStringAsRawHex8 (name.c_str());
            return SendPacketNoLock (response.GetData(), response.GetSize());
        }
    }
    return SendErrorResponse (5);

}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qGroupName (StringExtractorGDBRemote &packet)
{
    // Packet format: "qGroupName:%i" where %i is the gid
    packet.SetFilePos(::strlen ("qGroupName:"));
    uint32_t gid = packet.GetU32 (UINT32_MAX);
    if (gid != UINT32_MAX)
    {
        std::string name;
        if (Host::GetGroupName (gid, name))
        {
            StreamString response;
            response.PutCStringAsRawHex8 (name.c_str());
            return SendPacketNoLock (response.GetData(), response.GetSize());
        }
    }
    return SendErrorResponse (6);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qSpeedTest (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("qSpeedTest:"));

    std::string key;
    std::string value;
    bool success = packet.GetNameColonValue(key, value);
    if (success && key.compare("response_size") == 0)
    {
        uint32_t response_size = Args::StringToUInt32(value.c_str(), 0, 0, &success);
        if (success)
        {
            if (response_size == 0)
                return SendOKResponse();
            StreamString response;
            uint32_t bytes_left = response_size;
            response.PutCString("data:");
            while (bytes_left > 0)
            {
                if (bytes_left >= 26)
                {
                    response.PutCString("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
                    bytes_left -= 26;
                }
                else
                {
                    response.Printf ("%*.*s;", bytes_left, bytes_left, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
                    bytes_left = 0;
                }
            }
            return SendPacketNoLock (response.GetData(), response.GetSize());
        }
    }
    return SendErrorResponse (7);
}

//
//static bool
//WaitForProcessToSIGSTOP (const lldb::pid_t pid, const int timeout_in_seconds)
//{
//    const int time_delta_usecs = 100000;
//    const int num_retries = timeout_in_seconds/time_delta_usecs;
//    for (int i=0; i<num_retries; i++)
//    {
//        struct proc_bsdinfo bsd_info;
//        int error = ::proc_pidinfo (pid, PROC_PIDTBSDINFO,
//                                    (uint64_t) 0,
//                                    &bsd_info,
//                                    PROC_PIDTBSDINFO_SIZE);
//
//        switch (error)
//        {
//            case EINVAL:
//            case ENOTSUP:
//            case ESRCH:
//            case EPERM:
//                return false;
//
//            default:
//                break;
//
//            case 0:
//                if (bsd_info.pbi_status == SSTOP)
//                    return true;
//        }
//        ::usleep (time_delta_usecs);
//    }
//    return false;
//}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_A (StringExtractorGDBRemote &packet)
{
    // The 'A' packet is the most over designed packet ever here with
    // redundant argument indexes, redundant argument lengths and needed hex
    // encoded argument string values. Really all that is needed is a comma
    // separated hex encoded argument value list, but we will stay true to the
    // documented version of the 'A' packet here...

    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
    int actual_arg_index = 0;

    packet.SetFilePos(1); // Skip the 'A'
    bool success = true;
    while (success && packet.GetBytesLeft() > 0)
    {
        // Decode the decimal argument string length. This length is the
        // number of hex nibbles in the argument string value.
        const uint32_t arg_len = packet.GetU32(UINT32_MAX);
        if (arg_len == UINT32_MAX)
            success = false;
        else
        {
            // Make sure the argument hex string length is followed by a comma
            if (packet.GetChar() != ',')
                success = false;
            else
            {
                // Decode the argument index. We ignore this really because
                // who would really send down the arguments in a random order???
                const uint32_t arg_idx = packet.GetU32(UINT32_MAX);
                if (arg_idx == UINT32_MAX)
                    success = false;
                else
                {
                    // Make sure the argument index is followed by a comma
                    if (packet.GetChar() != ',')
                        success = false;
                    else
                    {
                        // Decode the argument string value from hex bytes
                        // back into a UTF8 string and make sure the length
                        // matches the one supplied in the packet
                        std::string arg;
                        if (packet.GetHexByteStringFixedLength(arg, arg_len) != (arg_len / 2))
                            success = false;
                        else
                        {
                            // If there are any bytes left
                            if (packet.GetBytesLeft())
                            {
                                if (packet.GetChar() != ',')
                                    success = false;
                            }

                            if (success)
                            {
                                if (arg_idx == 0)
                                    m_process_launch_info.GetExecutableFile().SetFile(arg.c_str(), false);
                                m_process_launch_info.GetArguments().AppendArgument(arg.c_str());
                                if (log)
                                    log->Printf ("GDBRemoteCommunicationServer::%s added arg %d: \"%s\"", __FUNCTION__, actual_arg_index, arg.c_str ());
                                ++actual_arg_index;
                            }
                        }
                    }
                }
            }
        }
    }

    if (success)
    {
        m_process_launch_error = LaunchProcess ();
        if (m_process_launch_info.GetProcessID() != LLDB_INVALID_PROCESS_ID)
        {
            return SendOKResponse ();
        }
        else
        {
            Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
            if (log)
                log->Printf("GDBRemoteCommunicationServer::%s failed to launch exe: %s",
                        __FUNCTION__,
                        m_process_launch_error.AsCString());

        }
    }
    return SendErrorResponse (8);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qC (StringExtractorGDBRemote &packet)
{
    StreamString response;

    if (IsGdbServer ())
    {
        // Fail if we don't have a current process.
        if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
            return SendErrorResponse (68);

        // Make sure we set the current thread so g and p packets return
        // the data the gdb will expect.
        lldb::tid_t tid = m_debugged_process_sp->GetCurrentThreadID ();
        SetCurrentThreadID (tid);

        NativeThreadProtocolSP thread_sp = m_debugged_process_sp->GetCurrentThread ();
        if (!thread_sp)
            return SendErrorResponse (69);

        response.Printf ("QC%" PRIx64, thread_sp->GetID ());
    }
    else
    {
        // NOTE: lldb should now be using qProcessInfo for process IDs.  This path here
        // should not be used.  It is reporting process id instead of thread id.  The
        // correct answer doesn't seem to make much sense for lldb-platform.
        // CONSIDER: flip to "unsupported".
        lldb::pid_t pid = m_process_launch_info.GetProcessID();
        response.Printf("QC%" PRIx64, pid);

        // this should always be platform here
        assert (m_is_platform && "this code path should only be traversed for lldb-platform");

        if (m_is_platform)
        {
            // If we launch a process and this GDB server is acting as a platform,
            // then we need to clear the process launch state so we can start
            // launching another process. In order to launch a process a bunch or
            // packets need to be sent: environment packets, working directory,
            // disable ASLR, and many more settings. When we launch a process we
            // then need to know when to clear this information. Currently we are
            // selecting the 'qC' packet as that packet which seems to make the most
            // sense.
            if (pid != LLDB_INVALID_PROCESS_ID)
            {
                m_process_launch_info.Clear();
            }
        }
    }
    return SendPacketNoLock (response.GetData(), response.GetSize());
}

bool
GDBRemoteCommunicationServer::DebugserverProcessReaped (lldb::pid_t pid)
{
    Mutex::Locker locker (m_spawned_pids_mutex);
    FreePortForProcess(pid);
    return m_spawned_pids.erase(pid) > 0;
}
bool
GDBRemoteCommunicationServer::ReapDebugserverProcess (void *callback_baton,
                                                      lldb::pid_t pid,
                                                      bool exited,
                                                      int signal,    // Zero for no signal
                                                      int status)    // Exit value of process if signal is zero
{
    GDBRemoteCommunicationServer *server = (GDBRemoteCommunicationServer *)callback_baton;
    server->DebugserverProcessReaped (pid);
    return true;
}

bool
GDBRemoteCommunicationServer::DebuggedProcessReaped (lldb::pid_t pid)
{
    // reap a process that we were debugging (but not debugserver)
    Mutex::Locker locker (m_spawned_pids_mutex);
    return m_spawned_pids.erase(pid) > 0;
}

bool
GDBRemoteCommunicationServer::ReapDebuggedProcess (void *callback_baton,
                                                   lldb::pid_t pid,
                                                   bool exited,
                                                   int signal,    // Zero for no signal
                                                   int status)    // Exit value of process if signal is zero
{
    GDBRemoteCommunicationServer *server = (GDBRemoteCommunicationServer *)callback_baton;
    server->DebuggedProcessReaped (pid);
    return true;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qLaunchGDBServer (StringExtractorGDBRemote &packet)
{
#ifdef _WIN32
    return SendErrorResponse(9);
#else
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));

    // Spawn a local debugserver as a platform so we can then attach or launch
    // a process...

    if (m_is_platform)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s() called", __FUNCTION__);

        // Sleep and wait a bit for debugserver to start to listen...
        ConnectionFileDescriptor file_conn;
        std::string hostname;
        // TODO: /tmp/ should not be hardcoded. User might want to override /tmp
        // with the TMPDIR environment variable
        packet.SetFilePos(::strlen ("qLaunchGDBServer;"));
        std::string name;
        std::string value;
        uint16_t port = UINT16_MAX;
        while (packet.GetNameColonValue(name, value))
        {
            if (name.compare ("host") == 0)
                hostname.swap(value);
            else if (name.compare ("port") == 0)
                port = Args::StringToUInt32(value.c_str(), 0, 0);
        }
        if (port == UINT16_MAX)
            port = GetNextAvailablePort();

        // Spawn a new thread to accept the port that gets bound after
        // binding to port 0 (zero).

        // Spawn a debugserver and try to get the port it listens to.
        ProcessLaunchInfo debugserver_launch_info;
        if (hostname.empty())
            hostname = "127.0.0.1";
        if (log)
            log->Printf("Launching debugserver with: %s:%u...\n", hostname.c_str(), port);

        debugserver_launch_info.SetMonitorProcessCallback(ReapDebugserverProcess, this, false);

        Error error = StartDebugserverProcess (hostname.empty() ? NULL : hostname.c_str(),
                                         port,
                                         debugserver_launch_info,
                                         port);

        lldb::pid_t debugserver_pid = debugserver_launch_info.GetProcessID();


        if (debugserver_pid != LLDB_INVALID_PROCESS_ID)
        {
            Mutex::Locker locker (m_spawned_pids_mutex);
            m_spawned_pids.insert(debugserver_pid);
            if (port > 0)
                AssociatePortWithProcess(port, debugserver_pid);
        }
        else
        {
            if (port > 0)
                FreePort (port);
        }

        if (error.Success())
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s() debugserver launched successfully as pid %" PRIu64, __FUNCTION__, debugserver_pid);

            char response[256];
            const int response_len = ::snprintf (response, sizeof(response), "pid:%" PRIu64 ";port:%u;", debugserver_pid, port + m_port_offset);
            assert (response_len < (int)sizeof(response));
            PacketResult packet_result = SendPacketNoLock (response, response_len);

            if (packet_result != PacketResult::Success)
            {
                if (debugserver_pid != LLDB_INVALID_PROCESS_ID)
                    ::kill (debugserver_pid, SIGINT);
            }
            return packet_result;
        }
        else
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s() debugserver launch failed: %s", __FUNCTION__, error.AsCString ());
        }
    }
    return SendErrorResponse (9);
#endif
}

bool
GDBRemoteCommunicationServer::KillSpawnedProcess (lldb::pid_t pid)
{
    // make sure we know about this process
    {
        Mutex::Locker locker (m_spawned_pids_mutex);
        if (m_spawned_pids.find(pid) == m_spawned_pids.end())
            return false;
    }

    // first try a SIGTERM (standard kill)
    Host::Kill (pid, SIGTERM);

    // check if that worked
    for (size_t i=0; i<10; ++i)
    {
        {
            Mutex::Locker locker (m_spawned_pids_mutex);
            if (m_spawned_pids.find(pid) == m_spawned_pids.end())
            {
                // it is now killed
                return true;
            }
        }
        usleep (10000);
    }

    // check one more time after the final usleep
    {
        Mutex::Locker locker (m_spawned_pids_mutex);
        if (m_spawned_pids.find(pid) == m_spawned_pids.end())
            return true;
    }

    // the launched process still lives.  Now try killing it again,
    // this time with an unblockable signal.
    Host::Kill (pid, SIGKILL);

    for (size_t i=0; i<10; ++i)
    {
        {
            Mutex::Locker locker (m_spawned_pids_mutex);
            if (m_spawned_pids.find(pid) == m_spawned_pids.end())
            {
                // it is now killed
                return true;
            }
        }
        usleep (10000);
    }

    // check one more time after the final usleep
    // Scope for locker
    {
        Mutex::Locker locker (m_spawned_pids_mutex);
        if (m_spawned_pids.find(pid) == m_spawned_pids.end())
            return true;
    }

    // no luck - the process still lives
    return false;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qKillSpawnedProcess (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("qKillSpawnedProcess:"));

    lldb::pid_t pid = packet.GetU64(LLDB_INVALID_PROCESS_ID);

    // verify that we know anything about this pid.
    // Scope for locker
    {
        Mutex::Locker locker (m_spawned_pids_mutex);
        if (m_spawned_pids.find(pid) == m_spawned_pids.end())
        {
            // not a pid we know about
            return SendErrorResponse (10);
        }
    }

    // go ahead and attempt to kill the spawned process
    if (KillSpawnedProcess (pid))
        return SendOKResponse ();
    else
        return SendErrorResponse (11);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_k (StringExtractorGDBRemote &packet)
{
    // ignore for now if we're lldb_platform
    if (m_is_platform)
        return SendUnimplementedResponse (packet.GetStringRef().c_str());

    // shutdown all spawned processes
    std::set<lldb::pid_t> spawned_pids_copy;

    // copy pids
    {
        Mutex::Locker locker (m_spawned_pids_mutex);
        spawned_pids_copy.insert (m_spawned_pids.begin (), m_spawned_pids.end ());
    }

    // nuke the spawned processes
    for (auto it = spawned_pids_copy.begin (); it != spawned_pids_copy.end (); ++it)
    {
        lldb::pid_t spawned_pid = *it;
        if (!KillSpawnedProcess (spawned_pid))
        {
            fprintf (stderr, "%s: failed to kill spawned pid %" PRIu64 ", ignoring.\n", __FUNCTION__, spawned_pid);
        }
    }

    FlushInferiorOutput ();

    // No OK response for kill packet.
    // return SendOKResponse ();
    return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qLaunchSuccess (StringExtractorGDBRemote &packet)
{
    if (m_process_launch_error.Success())
        return SendOKResponse();
    StreamString response;
    response.PutChar('E');
    response.PutCString(m_process_launch_error.AsCString("<unknown error>"));
    return SendPacketNoLock (response.GetData(), response.GetSize());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_QEnvironment  (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QEnvironment:"));
    const uint32_t bytes_left = packet.GetBytesLeft();
    if (bytes_left > 0)
    {
        m_process_launch_info.GetEnvironmentEntries ().AppendArgument (packet.Peek());
        return SendOKResponse ();
    }
    return SendErrorResponse (12);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_QLaunchArch (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QLaunchArch:"));
    const uint32_t bytes_left = packet.GetBytesLeft();
    if (bytes_left > 0)
    {
        const char* arch_triple = packet.Peek();
        ArchSpec arch_spec(arch_triple,NULL);
        m_process_launch_info.SetArchitecture(arch_spec);
        return SendOKResponse();
    }
    return SendErrorResponse(13);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_QSetDisableASLR (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QSetDisableASLR:"));
    if (packet.GetU32(0))
        m_process_launch_info.GetFlags().Set (eLaunchFlagDisableASLR);
    else
        m_process_launch_info.GetFlags().Clear (eLaunchFlagDisableASLR);
    return SendOKResponse ();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_QSetWorkingDir (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QSetWorkingDir:"));
    std::string path;
    packet.GetHexByteString(path);
    if (m_is_platform)
    {
#ifdef _WIN32
        // Not implemented on Windows
        return SendUnimplementedResponse("GDBRemoteCommunicationServer::Handle_QSetWorkingDir unimplemented");
#else
        // If this packet is sent to a platform, then change the current working directory
        if (::chdir(path.c_str()) != 0)
            return SendErrorResponse(errno);
#endif
    }
    else
    {
        m_process_launch_info.SwapWorkingDirectory (path);
    }
    return SendOKResponse ();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qGetWorkingDir (StringExtractorGDBRemote &packet)
{
    StreamString response;

    if (m_is_platform)
    {
        // If this packet is sent to a platform, then change the current working directory
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd)) == NULL)
        {
            return SendErrorResponse(errno);
        }
        else
        {
            response.PutBytesAsRawHex8(cwd, strlen(cwd));
            return SendPacketNoLock(response.GetData(), response.GetSize());
        }
    }
    else
    {
        const char *working_dir = m_process_launch_info.GetWorkingDirectory();
        if (working_dir && working_dir[0])
        {
            response.PutBytesAsRawHex8(working_dir, strlen(working_dir));
            return SendPacketNoLock(response.GetData(), response.GetSize());
        }
        else
        {
            return SendErrorResponse(14);
        }
    }
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_QSetSTDIN (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QSetSTDIN:"));
    FileAction file_action;
    std::string path;
    packet.GetHexByteString(path);
    const bool read = false;
    const bool write = true;
    if (file_action.Open(STDIN_FILENO, path.c_str(), read, write))
    {
        m_process_launch_info.AppendFileAction(file_action);
        return SendOKResponse ();
    }
    return SendErrorResponse (15);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_QSetSTDOUT (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QSetSTDOUT:"));
    FileAction file_action;
    std::string path;
    packet.GetHexByteString(path);
    const bool read = true;
    const bool write = false;
    if (file_action.Open(STDOUT_FILENO, path.c_str(), read, write))
    {
        m_process_launch_info.AppendFileAction(file_action);
        return SendOKResponse ();
    }
    return SendErrorResponse (16);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_QSetSTDERR (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QSetSTDERR:"));
    FileAction file_action;
    std::string path;
    packet.GetHexByteString(path);
    const bool read = true;
    const bool write = false;
    if (file_action.Open(STDERR_FILENO, path.c_str(), read, write))
    {
        m_process_launch_info.AppendFileAction(file_action);
        return SendOKResponse ();
    }
    return SendErrorResponse (17);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_C (StringExtractorGDBRemote &packet)
{
    if (!IsGdbServer ())
        return SendUnimplementedResponse (packet.GetStringRef().c_str());

    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS|LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s called", __FUNCTION__);

    // Ensure we have a native process.
    if (!m_debugged_process_sp)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s no debugged process shared pointer", __FUNCTION__);
        return SendErrorResponse (0x36);
    }

    // Pull out the signal number.
    packet.SetFilePos (::strlen ("C"));
    if (packet.GetBytesLeft () < 1)
    {
        // Shouldn't be using a C without a signal.
        return SendIllFormedResponse (packet, "C packet specified without signal.");
    }
    const uint32_t signo = packet.GetHexMaxU32 (false, std::numeric_limits<uint32_t>::max ());
    if (signo == std::numeric_limits<uint32_t>::max ())
        return SendIllFormedResponse (packet, "failed to parse signal number");

    // Handle optional continue address.
    if (packet.GetBytesLeft () > 0)
    {
        // FIXME add continue at address support for $C{signo}[;{continue-address}].
        if (*packet.Peek () == ';')
            return SendUnimplementedResponse (packet.GetStringRef().c_str());
        else
            return SendIllFormedResponse (packet, "unexpected content after $C{signal-number}");
    }

    lldb_private::ResumeActionList resume_actions (StateType::eStateRunning, 0);
    Error error;

    // We have two branches: what to do if a continue thread is specified (in which case we target
    // sending the signal to that thread), or when we don't have a continue thread set (in which
    // case we send a signal to the process).

    // TODO discuss with Greg Clayton, make sure this makes sense.

    lldb::tid_t signal_tid = GetContinueThreadID ();
    if (signal_tid != LLDB_INVALID_THREAD_ID)
    {
        // The resume action for the continue thread (or all threads if a continue thread is not set).
        lldb_private::ResumeAction action = { GetContinueThreadID (), StateType::eStateRunning, static_cast<int> (signo) };

        // Add the action for the continue thread (or all threads when the continue thread isn't present).
        resume_actions.Append (action);
    }
    else
    {
        // Send the signal to the process since we weren't targeting a specific continue thread with the signal.
        error = m_debugged_process_sp->Signal (signo);
        if (error.Fail ())
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s failed to send signal for process %" PRIu64 ": %s",
                             __FUNCTION__,
                             m_debugged_process_sp->GetID (),
                             error.AsCString ());

            return SendErrorResponse (0x52);
        }
    }

    // Resume the threads.
    error = m_debugged_process_sp->Resume (resume_actions);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed to resume threads for process %" PRIu64 ": %s",
                         __FUNCTION__,
                         m_debugged_process_sp->GetID (),
                         error.AsCString ());

        return SendErrorResponse (0x38);
    }

    // Don't send an "OK" packet; response is the stopped/exited message.
    return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_c (StringExtractorGDBRemote &packet, bool skip_file_pos_adjustment)
{
    if (!IsGdbServer ())
        return SendUnimplementedResponse (packet.GetStringRef().c_str());

    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS|LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s called", __FUNCTION__);

    // We reuse this method in vCont - don't double adjust the file position.
    if (!skip_file_pos_adjustment)
        packet.SetFilePos (::strlen ("c"));

    // For now just support all continue.
    const bool has_continue_address = (packet.GetBytesLeft () > 0);
    if (has_continue_address)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s not implemented for c{address} variant [%s remains]", __FUNCTION__, packet.Peek ());
        return SendUnimplementedResponse (packet.GetStringRef().c_str());
    }

    // Ensure we have a native process.
    if (!m_debugged_process_sp)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s no debugged process shared pointer", __FUNCTION__);
        return SendErrorResponse (0x36);
    }

    // Build the ResumeActionList
    lldb_private::ResumeActionList actions (StateType::eStateRunning, 0);

    Error error = m_debugged_process_sp->Resume (actions);
    if (error.Fail ())
    {
        if (log)
        {
            log->Printf ("GDBRemoteCommunicationServer::%s c failed for process %" PRIu64 ": %s",
                         __FUNCTION__,
                         m_debugged_process_sp->GetID (),
                         error.AsCString ());
        }
        return SendErrorResponse (GDBRemoteServerError::eErrorResume);
    }

    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s continued process %" PRIu64, __FUNCTION__, m_debugged_process_sp->GetID ());

    // No response required from continue.
    return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vCont_actions (StringExtractorGDBRemote &packet)
{
    if (!IsGdbServer ())
    {
        // only llgs supports $vCont.
        return SendUnimplementedResponse (packet.GetStringRef().c_str());
    }

    // We handle $vCont messages for c.
    // TODO add C, s and S.
    StreamString response;
    response.Printf("vCont;c;C;s;S");

    return SendPacketNoLock(response.GetData(), response.GetSize());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vCont (StringExtractorGDBRemote &packet)
{
    if (!IsGdbServer ())
    {
        // only llgs supports $vCont
        return SendUnimplementedResponse (packet.GetStringRef().c_str());
    }

    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s handling vCont packet", __FUNCTION__);

    packet.SetFilePos (::strlen ("vCont"));

    // Check if this is all continue (no options or ";c").
    if (!packet.GetBytesLeft () || (::strcmp (packet.Peek (), ";c") == 0))
    {
        // Move the packet past the ";c".
        if (packet.GetBytesLeft ())
            packet.SetFilePos (packet.GetFilePos () + ::strlen (";c"));

        const bool skip_file_pos_adjustment = true;
        return Handle_c (packet, skip_file_pos_adjustment);
    }
    else if (::strcmp (packet.Peek (), ";s") == 0)
    {
        // Move past the ';', then do a simple 's'.
        packet.SetFilePos (packet.GetFilePos () + 1);
        return Handle_s (packet);
    }

    // Ensure we have a native process.
    if (!m_debugged_process_sp)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s no debugged process shared pointer", __FUNCTION__);
        return SendErrorResponse (0x36);
    }

    ResumeActionList thread_actions;

    while (packet.GetBytesLeft () && *packet.Peek () == ';')
    {
        // Skip the semi-colon.
        packet.GetChar ();

        // Build up the thread action.
        ResumeAction thread_action;
        thread_action.tid = LLDB_INVALID_THREAD_ID;
        thread_action.state = eStateInvalid;
        thread_action.signal = 0;

        const char action = packet.GetChar ();
        switch (action)
        {
            case 'C':
                thread_action.signal = packet.GetHexMaxU32 (false, 0);
                if (thread_action.signal == 0)
                    return SendIllFormedResponse (packet, "Could not parse signal in vCont packet C action");
                // Fall through to next case...

            case 'c':
                // Continue
                thread_action.state = eStateRunning;
                break;

            case 'S':
                thread_action.signal = packet.GetHexMaxU32 (false, 0);
                if (thread_action.signal == 0)
                    return SendIllFormedResponse (packet, "Could not parse signal in vCont packet S action");
                // Fall through to next case...

            case 's':
                // Step
                thread_action.state = eStateStepping;
                break;

            default:
                return SendIllFormedResponse (packet, "Unsupported vCont action");
                break;
        }

        // Parse out optional :{thread-id} value.
        if (packet.GetBytesLeft () && (*packet.Peek () == ':'))
        {
            // Consume the separator.
            packet.GetChar ();

            thread_action.tid = packet.GetHexMaxU32 (false, LLDB_INVALID_THREAD_ID);
            if (thread_action.tid == LLDB_INVALID_THREAD_ID)
                return SendIllFormedResponse (packet, "Could not parse thread number in vCont packet");
        }

        thread_actions.Append (thread_action);
    }

    // If a default action for all other threads wasn't mentioned
    // then we should stop the threads.
    thread_actions.SetDefaultThreadActionIfNeeded (eStateStopped, 0);

    Error error = m_debugged_process_sp->Resume (thread_actions);
    if (error.Fail ())
    {
        if (log)
        {
            log->Printf ("GDBRemoteCommunicationServer::%s vCont failed for process %" PRIu64 ": %s",
                         __FUNCTION__,
                         m_debugged_process_sp->GetID (),
                         error.AsCString ());
        }
        return SendErrorResponse (GDBRemoteServerError::eErrorResume);
    }

    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s continued process %" PRIu64, __FUNCTION__, m_debugged_process_sp->GetID ());

    // No response required from vCont.
    return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_QStartNoAckMode (StringExtractorGDBRemote &packet)
{
    // Send response first before changing m_send_acks to we ack this packet
    PacketResult packet_result = SendOKResponse ();
    m_send_acks = false;
    return packet_result;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qPlatform_mkdir (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("qPlatform_mkdir:"));
    mode_t mode = packet.GetHexMaxU32(false, UINT32_MAX);
    if (packet.GetChar() == ',')
    {
        std::string path;
        packet.GetHexByteString(path);
        Error error = FileSystem::MakeDirectory(path.c_str(), mode);
        if (error.Success())
            return SendPacketNoLock ("OK", 2);
        else
            return SendErrorResponse(error.GetError());
    }
    return SendErrorResponse(20);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qPlatform_chmod (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("qPlatform_chmod:"));
    
    mode_t mode = packet.GetHexMaxU32(false, UINT32_MAX);
    if (packet.GetChar() == ',')
    {
        std::string path;
        packet.GetHexByteString(path);
        Error error = FileSystem::SetFilePermissions(path.c_str(), mode);
        if (error.Success())
            return SendPacketNoLock ("OK", 2);
        else
            return SendErrorResponse(error.GetError());
    }
    return SendErrorResponse(19);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vFile_Open (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:open:"));
    std::string path;
    packet.GetHexByteStringTerminatedBy(path,',');
    if (!path.empty())
    {
        if (packet.GetChar() == ',')
        {
            uint32_t flags = packet.GetHexMaxU32(false, 0);
            if (packet.GetChar() == ',')
            {
                mode_t mode = packet.GetHexMaxU32(false, 0600);
                Error error;
                int fd = ::open (path.c_str(), flags, mode);
                const int save_errno = fd == -1 ? errno : 0;
                StreamString response;
                response.PutChar('F');
                response.Printf("%i", fd);
                if (save_errno)
                    response.Printf(",%i", save_errno);
                return SendPacketNoLock(response.GetData(), response.GetSize());
            }
        }
    }
    return SendErrorResponse(18);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vFile_Close (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:close:"));
    int fd = packet.GetS32(-1);
    Error error;
    int err = -1;
    int save_errno = 0;
    if (fd >= 0)
    {
        err = close(fd);
        save_errno = err == -1 ? errno : 0;
    }
    else
    {
        save_errno = EINVAL;
    }
    StreamString response;
    response.PutChar('F');
    response.Printf("%i", err);
    if (save_errno)
        response.Printf(",%i", save_errno);
    return SendPacketNoLock(response.GetData(), response.GetSize());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vFile_pRead (StringExtractorGDBRemote &packet)
{
#ifdef _WIN32
    // Not implemented on Windows
    return SendUnimplementedResponse("GDBRemoteCommunicationServer::Handle_vFile_pRead() unimplemented");
#else
    StreamGDBRemote response;
    packet.SetFilePos(::strlen("vFile:pread:"));
    int fd = packet.GetS32(-1);
    if (packet.GetChar() == ',')
    {
        uint64_t count = packet.GetU64(UINT64_MAX);
        if (packet.GetChar() == ',')
        {
            uint64_t offset = packet.GetU64(UINT32_MAX);
            if (count == UINT64_MAX)
            {
                response.Printf("F-1:%i", EINVAL);
                return SendPacketNoLock(response.GetData(), response.GetSize());
            }
            
            std::string buffer(count, 0);
            const ssize_t bytes_read = ::pread (fd, &buffer[0], buffer.size(), offset);
            const int save_errno = bytes_read == -1 ? errno : 0;
            response.PutChar('F');
            response.Printf("%zi", bytes_read);
            if (save_errno)
                response.Printf(",%i", save_errno);
            else
            {
                response.PutChar(';');
                response.PutEscapedBytes(&buffer[0], bytes_read);
            }
            return SendPacketNoLock(response.GetData(), response.GetSize());
        }
    }
    return SendErrorResponse(21);

#endif
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vFile_pWrite (StringExtractorGDBRemote &packet)
{
#ifdef _WIN32
    return SendUnimplementedResponse("GDBRemoteCommunicationServer::Handle_vFile_pWrite() unimplemented");
#else
    packet.SetFilePos(::strlen("vFile:pwrite:"));

    StreamGDBRemote response;
    response.PutChar('F');

    int fd = packet.GetU32(UINT32_MAX);
    if (packet.GetChar() == ',')
    {
        off_t offset = packet.GetU64(UINT32_MAX);
        if (packet.GetChar() == ',')
        {
            std::string buffer;
            if (packet.GetEscapedBinaryData(buffer))
            {
                const ssize_t bytes_written = ::pwrite (fd, buffer.data(), buffer.size(), offset);
                const int save_errno = bytes_written == -1 ? errno : 0;
                response.Printf("%zi", bytes_written);
                if (save_errno)
                    response.Printf(",%i", save_errno);
            }
            else
            {
                response.Printf ("-1,%i", EINVAL);
            }
            return SendPacketNoLock(response.GetData(), response.GetSize());
        }
    }
    return SendErrorResponse(27);
#endif
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vFile_Size (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:size:"));
    std::string path;
    packet.GetHexByteString(path);
    if (!path.empty())
    {
        lldb::user_id_t retcode = FileSystem::GetFileSize(FileSpec(path.c_str(), false));
        StreamString response;
        response.PutChar('F');
        response.PutHex64(retcode);
        if (retcode == UINT64_MAX)
        {
            response.PutChar(',');
            response.PutHex64(retcode); // TODO: replace with Host::GetSyswideErrorCode()
        }
        return SendPacketNoLock(response.GetData(), response.GetSize());
    }
    return SendErrorResponse(22);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vFile_Mode (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:mode:"));
    std::string path;
    packet.GetHexByteString(path);
    if (!path.empty())
    {
        Error error;
        const uint32_t mode = File::GetPermissions(path.c_str(), error);
        StreamString response;
        response.Printf("F%u", mode);
        if (mode == 0 || error.Fail())
            response.Printf(",%i", (int)error.GetError());
        return SendPacketNoLock(response.GetData(), response.GetSize());
    }
    return SendErrorResponse(23);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vFile_Exists (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:exists:"));
    std::string path;
    packet.GetHexByteString(path);
    if (!path.empty())
    {
        bool retcode = FileSystem::GetFileExists(FileSpec(path.c_str(), false));
        StreamString response;
        response.PutChar('F');
        response.PutChar(',');
        if (retcode)
            response.PutChar('1');
        else
            response.PutChar('0');
        return SendPacketNoLock(response.GetData(), response.GetSize());
    }
    return SendErrorResponse(24);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vFile_symlink (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:symlink:"));
    std::string dst, src;
    packet.GetHexByteStringTerminatedBy(dst, ',');
    packet.GetChar(); // Skip ',' char
    packet.GetHexByteString(src);
    Error error = FileSystem::Symlink(src.c_str(), dst.c_str());
    StreamString response;
    response.Printf("F%u,%u", error.GetError(), error.GetError());
    return SendPacketNoLock(response.GetData(), response.GetSize());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vFile_unlink (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:unlink:"));
    std::string path;
    packet.GetHexByteString(path);
    Error error = FileSystem::Unlink(path.c_str());
    StreamString response;
    response.Printf("F%u,%u", error.GetError(), error.GetError());
    return SendPacketNoLock(response.GetData(), response.GetSize());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qPlatform_shell (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("qPlatform_shell:"));
    std::string path;
    std::string working_dir;
    packet.GetHexByteStringTerminatedBy(path,',');
    if (!path.empty())
    {
        if (packet.GetChar() == ',')
        {
            // FIXME: add timeout to qPlatform_shell packet
            // uint32_t timeout = packet.GetHexMaxU32(false, 32);
            uint32_t timeout = 10;
            if (packet.GetChar() == ',')
                packet.GetHexByteString(working_dir);
            int status, signo;
            std::string output;
            Error err = Host::RunShellCommand(path.c_str(),
                                              working_dir.empty() ? NULL : working_dir.c_str(),
                                              &status, &signo, &output, timeout);
            StreamGDBRemote response;
            if (err.Fail())
            {
                response.PutCString("F,");
                response.PutHex32(UINT32_MAX);
            }
            else
            {
                response.PutCString("F,");
                response.PutHex32(status);
                response.PutChar(',');
                response.PutHex32(signo);
                response.PutChar(',');
                response.PutEscapedBytes(output.c_str(), output.size());
            }
            return SendPacketNoLock(response.GetData(), response.GetSize());
        }
    }
    return SendErrorResponse(24);
}

void
GDBRemoteCommunicationServer::SetCurrentThreadID (lldb::tid_t tid)
{
    assert (IsGdbServer () && "SetCurrentThreadID() called when not GdbServer code");

    Log *log (GetLogIfAnyCategoriesSet (LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s setting current thread id to %" PRIu64, __FUNCTION__, tid);

    m_current_tid = tid;
    if (m_debugged_process_sp)
        m_debugged_process_sp->SetCurrentThreadID (m_current_tid);
}

void
GDBRemoteCommunicationServer::SetContinueThreadID (lldb::tid_t tid)
{
    assert (IsGdbServer () && "SetContinueThreadID() called when not GdbServer code");

    Log *log (GetLogIfAnyCategoriesSet (LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s setting continue thread id to %" PRIu64, __FUNCTION__, tid);

    m_continue_tid = tid;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_stop_reason (StringExtractorGDBRemote &packet)
{
    // Handle the $? gdbremote command.
    if (!IsGdbServer ())
        return SendUnimplementedResponse("GDBRemoteCommunicationServer::Handle_stop_reason() unimplemented");

    // If no process, indicate error
    if (!m_debugged_process_sp)
        return SendErrorResponse (02);

    return SendStopReasonForState (m_debugged_process_sp->GetState (), true);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendStopReasonForState (lldb::StateType process_state, bool flush_on_exit)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

    switch (process_state)
    {
        case eStateAttaching:
        case eStateLaunching:
        case eStateRunning:
        case eStateStepping:
        case eStateDetached:
            // NOTE: gdb protocol doc looks like it should return $OK
            // when everything is running (i.e. no stopped result).
            return PacketResult::Success;  // Ignore

        case eStateSuspended:
        case eStateStopped:
        case eStateCrashed:
        {
            lldb::tid_t tid = m_debugged_process_sp->GetCurrentThreadID ();
            // Make sure we set the current thread so g and p packets return
            // the data the gdb will expect.
            SetCurrentThreadID (tid);
            return SendStopReplyPacketForThread (tid);
        }

        case eStateInvalid:
        case eStateUnloaded:
        case eStateExited:
            if (flush_on_exit)
                FlushInferiorOutput ();
            return SendWResponse(m_debugged_process_sp.get());

        default:
            if (log)
            {
                log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 ", current state reporting not handled: %s",
                             __FUNCTION__,
                             m_debugged_process_sp->GetID (),
                             StateAsCString (process_state));
            }
            break;
    }
    
    return SendErrorResponse (0);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vFile_Stat (StringExtractorGDBRemote &packet)
{
    return SendUnimplementedResponse("GDBRemoteCommunicationServer::Handle_vFile_Stat() unimplemented");
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_vFile_MD5 (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:MD5:"));
    std::string path;
    packet.GetHexByteString(path);
    if (!path.empty())
    {
        uint64_t a,b;
        StreamGDBRemote response;
        if (FileSystem::CalculateMD5(FileSpec(path.c_str(), false), a, b) == false)
        {
            response.PutCString("F,");
            response.PutCString("x");
        }
        else
        {
            response.PutCString("F,");
            response.PutHex64(a);
            response.PutHex64(b);
        }
        return SendPacketNoLock(response.GetData(), response.GetSize());
    }
    return SendErrorResponse(25);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qRegisterInfo (StringExtractorGDBRemote &packet)
{
    // Ensure we're llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse("GDBRemoteCommunicationServer::Handle_qRegisterInfo() unimplemented");

    // Fail if we don't have a current process.
    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
        return SendErrorResponse (68);

    // Ensure we have a thread.
    NativeThreadProtocolSP thread_sp (m_debugged_process_sp->GetThreadAtIndex (0));
    if (!thread_sp)
        return SendErrorResponse (69);

    // Get the register context for the first thread.
    NativeRegisterContextSP reg_context_sp (thread_sp->GetRegisterContext ());
    if (!reg_context_sp)
        return SendErrorResponse (69);

    // Parse out the register number from the request.
    packet.SetFilePos (strlen("qRegisterInfo"));
    const uint32_t reg_index = packet.GetHexMaxU32 (false, std::numeric_limits<uint32_t>::max ());
    if (reg_index == std::numeric_limits<uint32_t>::max ())
        return SendErrorResponse (69);

    // Return the end of registers response if we've iterated one past the end of the register set.
    if (reg_index >= reg_context_sp->GetRegisterCount ())
        return SendErrorResponse (69);

    const RegisterInfo *reg_info = reg_context_sp->GetRegisterInfoAtIndex(reg_index);
    if (!reg_info)
        return SendErrorResponse (69);

    // Build the reginfos response.
    StreamGDBRemote response;

    response.PutCString ("name:");
    response.PutCString (reg_info->name);
    response.PutChar (';');

    if (reg_info->alt_name && reg_info->alt_name[0])
    {
        response.PutCString ("alt-name:");
        response.PutCString (reg_info->alt_name);
        response.PutChar (';');
    }

    response.Printf ("bitsize:%" PRIu32 ";offset:%" PRIu32 ";", reg_info->byte_size * 8, reg_info->byte_offset);

    switch (reg_info->encoding)
    {
        case eEncodingUint:    response.PutCString ("encoding:uint;"); break;
        case eEncodingSint:    response.PutCString ("encoding:sint;"); break;
        case eEncodingIEEE754: response.PutCString ("encoding:ieee754;"); break;
        case eEncodingVector:  response.PutCString ("encoding:vector;"); break;
        default: break;
    }

    switch (reg_info->format)
    {
        case eFormatBinary:          response.PutCString ("format:binary;"); break;
        case eFormatDecimal:         response.PutCString ("format:decimal;"); break;
        case eFormatHex:             response.PutCString ("format:hex;"); break;
        case eFormatFloat:           response.PutCString ("format:float;"); break;
        case eFormatVectorOfSInt8:   response.PutCString ("format:vector-sint8;"); break;
        case eFormatVectorOfUInt8:   response.PutCString ("format:vector-uint8;"); break;
        case eFormatVectorOfSInt16:  response.PutCString ("format:vector-sint16;"); break;
        case eFormatVectorOfUInt16:  response.PutCString ("format:vector-uint16;"); break;
        case eFormatVectorOfSInt32:  response.PutCString ("format:vector-sint32;"); break;
        case eFormatVectorOfUInt32:  response.PutCString ("format:vector-uint32;"); break;
        case eFormatVectorOfFloat32: response.PutCString ("format:vector-float32;"); break;
        case eFormatVectorOfUInt128: response.PutCString ("format:vector-uint128;"); break;
        default: break;
    };

    const char *const register_set_name = reg_context_sp->GetRegisterSetNameForRegisterAtIndex(reg_index);
    if (register_set_name)
    {
        response.PutCString ("set:");
        response.PutCString (register_set_name);
        response.PutChar (';');
    }

    if (reg_info->kinds[RegisterKind::eRegisterKindGCC] != LLDB_INVALID_REGNUM)
        response.Printf ("gcc:%" PRIu32 ";", reg_info->kinds[RegisterKind::eRegisterKindGCC]);

    if (reg_info->kinds[RegisterKind::eRegisterKindDWARF] != LLDB_INVALID_REGNUM)
        response.Printf ("dwarf:%" PRIu32 ";", reg_info->kinds[RegisterKind::eRegisterKindDWARF]);

    switch (reg_info->kinds[RegisterKind::eRegisterKindGeneric])
    {
        case LLDB_REGNUM_GENERIC_PC:     response.PutCString("generic:pc;"); break;
        case LLDB_REGNUM_GENERIC_SP:     response.PutCString("generic:sp;"); break;
        case LLDB_REGNUM_GENERIC_FP:     response.PutCString("generic:fp;"); break;
        case LLDB_REGNUM_GENERIC_RA:     response.PutCString("generic:ra;"); break;
        case LLDB_REGNUM_GENERIC_FLAGS:  response.PutCString("generic:flags;"); break;
        case LLDB_REGNUM_GENERIC_ARG1:   response.PutCString("generic:arg1;"); break;
        case LLDB_REGNUM_GENERIC_ARG2:   response.PutCString("generic:arg2;"); break;
        case LLDB_REGNUM_GENERIC_ARG3:   response.PutCString("generic:arg3;"); break;
        case LLDB_REGNUM_GENERIC_ARG4:   response.PutCString("generic:arg4;"); break;
        case LLDB_REGNUM_GENERIC_ARG5:   response.PutCString("generic:arg5;"); break;
        case LLDB_REGNUM_GENERIC_ARG6:   response.PutCString("generic:arg6;"); break;
        case LLDB_REGNUM_GENERIC_ARG7:   response.PutCString("generic:arg7;"); break;
        case LLDB_REGNUM_GENERIC_ARG8:   response.PutCString("generic:arg8;"); break;
        default: break;
    }

    if (reg_info->value_regs && reg_info->value_regs[0] != LLDB_INVALID_REGNUM)
    {
        response.PutCString ("container-regs:");
        int i = 0;
        for (const uint32_t *reg_num = reg_info->value_regs; *reg_num != LLDB_INVALID_REGNUM; ++reg_num, ++i)
        {
            if (i > 0)
                response.PutChar (',');
            response.Printf ("%" PRIx32, *reg_num);
        }
        response.PutChar (';');
    }

    if (reg_info->invalidate_regs && reg_info->invalidate_regs[0])
    {
        response.PutCString ("invalidate-regs:");
        int i = 0;
        for (const uint32_t *reg_num = reg_info->invalidate_regs; *reg_num != LLDB_INVALID_REGNUM; ++reg_num, ++i)
        {
            if (i > 0)
                response.PutChar (',');
            response.Printf ("%" PRIx32, *reg_num);
        }
        response.PutChar (';');
    }

    return SendPacketNoLock(response.GetData(), response.GetSize());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qfThreadInfo (StringExtractorGDBRemote &packet)
{
    // Ensure we're llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse("GDBRemoteCommunicationServer::Handle_qfThreadInfo() unimplemented");

    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

    // Fail if we don't have a current process.
    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s() no process (%s), returning OK", __FUNCTION__, m_debugged_process_sp ? "invalid process id" : "null m_debugged_process_sp");
        return SendOKResponse ();
    }

    StreamGDBRemote response;
    response.PutChar ('m');

    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s() starting thread iteration", __FUNCTION__);

    NativeThreadProtocolSP thread_sp;
    uint32_t thread_index;
    for (thread_index = 0, thread_sp = m_debugged_process_sp->GetThreadAtIndex (thread_index);
         thread_sp;
         ++thread_index, thread_sp = m_debugged_process_sp->GetThreadAtIndex (thread_index))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s() iterated thread %" PRIu32 "(%s, tid=0x%" PRIx64 ")", __FUNCTION__, thread_index, thread_sp ? "is not null" : "null", thread_sp ? thread_sp->GetID () : LLDB_INVALID_THREAD_ID);
        if (thread_index > 0)
            response.PutChar(',');
        response.Printf ("%" PRIx64, thread_sp->GetID ());
    }

    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s() finished thread iteration", __FUNCTION__);

    return SendPacketNoLock(response.GetData(), response.GetSize());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_qsThreadInfo (StringExtractorGDBRemote &packet)
{
    // Ensure we're llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("GDBRemoteCommunicationServer::Handle_qsThreadInfo() unimplemented");

    // FIXME for now we return the full thread list in the initial packet and always do nothing here.
    return SendPacketNoLock ("l", 1);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_p (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

    // Ensure we're llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("GDBRemoteCommunicationServer::Handle_p() unimplemented");

    // Parse out the register number from the request.
    packet.SetFilePos (strlen("p"));
    const uint32_t reg_index = packet.GetHexMaxU32 (false, std::numeric_limits<uint32_t>::max ());
    if (reg_index == std::numeric_limits<uint32_t>::max ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, could not parse register number from request \"%s\"", __FUNCTION__, packet.GetStringRef ().c_str ());
        return SendErrorResponse (0x15);
    }

    // Get the thread to use.
    NativeThreadProtocolSP thread_sp = GetThreadFromSuffix (packet);
    if (!thread_sp)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, no thread available", __FUNCTION__);
        return SendErrorResponse (0x15);
    }

    // Get the thread's register context.
    NativeRegisterContextSP reg_context_sp (thread_sp->GetRegisterContext ());
    if (!reg_context_sp)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " tid %" PRIu64 " failed, no register context available for the thread", __FUNCTION__, m_debugged_process_sp->GetID (), thread_sp->GetID ());
        return SendErrorResponse (0x15);
    }

    // Return the end of registers response if we've iterated one past the end of the register set.
    if (reg_index >= reg_context_sp->GetRegisterCount ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, requested register %" PRIu32 " beyond register count %" PRIu32, __FUNCTION__, reg_index, reg_context_sp->GetRegisterCount ());
        return SendErrorResponse (0x15);
    }

    const RegisterInfo *reg_info = reg_context_sp->GetRegisterInfoAtIndex(reg_index);
    if (!reg_info)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, requested register %" PRIu32 " returned NULL", __FUNCTION__, reg_index);
        return SendErrorResponse (0x15);
    }

    // Build the reginfos response.
    StreamGDBRemote response;

    // Retrieve the value
    RegisterValue reg_value;
    Error error = reg_context_sp->ReadRegister (reg_info, reg_value);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, read of requested register %" PRIu32 " (%s) failed: %s", __FUNCTION__, reg_index, reg_info->name, error.AsCString ());
        return SendErrorResponse (0x15);
    }

    const uint8_t *const data = reinterpret_cast<const uint8_t*> (reg_value.GetBytes ());
    if (!data)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed to get data bytes from requested register %" PRIu32, __FUNCTION__, reg_index);
        return SendErrorResponse (0x15);
    }

    // FIXME flip as needed to get data in big/little endian format for this host.
    for (uint32_t i = 0; i < reg_value.GetByteSize (); ++i)
        response.PutHex8 (data[i]);

    return SendPacketNoLock (response.GetData (), response.GetSize ());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_P (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

    // Ensure we're llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("GDBRemoteCommunicationServer::Handle_P() unimplemented");

    // Ensure there is more content.
    if (packet.GetBytesLeft () < 1)
        return SendIllFormedResponse (packet, "Empty P packet");

    // Parse out the register number from the request.
    packet.SetFilePos (strlen("P"));
    const uint32_t reg_index = packet.GetHexMaxU32 (false, std::numeric_limits<uint32_t>::max ());
    if (reg_index == std::numeric_limits<uint32_t>::max ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, could not parse register number from request \"%s\"", __FUNCTION__, packet.GetStringRef ().c_str ());
        return SendErrorResponse (0x29);
    }

    // Note debugserver would send an E30 here.
    if ((packet.GetBytesLeft () < 1) || (packet.GetChar () != '='))
        return SendIllFormedResponse (packet, "P packet missing '=' char after register number");

    // Get process architecture.
    ArchSpec process_arch;
    if (!m_debugged_process_sp || !m_debugged_process_sp->GetArchitecture (process_arch))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed to retrieve inferior architecture", __FUNCTION__);
        return SendErrorResponse (0x49);
    }

    // Parse out the value.
    const uint64_t raw_value = packet.GetHexMaxU64 (process_arch.GetByteOrder () == lldb::eByteOrderLittle, std::numeric_limits<uint64_t>::max ());

    // Get the thread to use.
    NativeThreadProtocolSP thread_sp = GetThreadFromSuffix (packet);
    if (!thread_sp)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, no thread available (thread index 0)", __FUNCTION__);
        return SendErrorResponse (0x28);
    }

    // Get the thread's register context.
    NativeRegisterContextSP reg_context_sp (thread_sp->GetRegisterContext ());
    if (!reg_context_sp)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " tid %" PRIu64 " failed, no register context available for the thread", __FUNCTION__, m_debugged_process_sp->GetID (), thread_sp->GetID ());
        return SendErrorResponse (0x15);
    }

    const RegisterInfo *reg_info = reg_context_sp->GetRegisterInfoAtIndex(reg_index);
    if (!reg_info)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, requested register %" PRIu32 " returned NULL", __FUNCTION__, reg_index);
        return SendErrorResponse (0x48);
    }

    // Return the end of registers response if we've iterated one past the end of the register set.
    if (reg_index >= reg_context_sp->GetRegisterCount ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, requested register %" PRIu32 " beyond register count %" PRIu32, __FUNCTION__, reg_index, reg_context_sp->GetRegisterCount ());
        return SendErrorResponse (0x47);
    }


    // Build the reginfos response.
    StreamGDBRemote response;

    // FIXME Could be suffixed with a thread: parameter.
    // That thread then needs to be fed back into the reg context retrieval above.
    Error error = reg_context_sp->WriteRegisterFromUnsigned (reg_info, raw_value);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, write of requested register %" PRIu32 " (%s) failed: %s", __FUNCTION__, reg_index, reg_info->name, error.AsCString ());
        return SendErrorResponse (0x32);
    }

    return SendOKResponse();
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_H (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

    // Ensure we're llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse("GDBRemoteCommunicationServer::Handle_H() unimplemented");

    // Fail if we don't have a current process.
    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, no process available", __FUNCTION__);
        return SendErrorResponse (0x15);
    }

    // Parse out which variant of $H is requested.
    packet.SetFilePos (strlen("H"));
    if (packet.GetBytesLeft () < 1)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, H command missing {g,c} variant", __FUNCTION__);
        return SendIllFormedResponse (packet, "H command missing {g,c} variant");
    }

    const char h_variant = packet.GetChar ();
    switch (h_variant)
    {
        case 'g':
            break;

        case 'c':
            break;

        default:
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s failed, invalid $H variant %c", __FUNCTION__, h_variant);
            return SendIllFormedResponse (packet, "H variant unsupported, should be c or g");
    }

    // Parse out the thread number.
    // FIXME return a parse success/fail value.  All values are valid here.
    const lldb::tid_t tid = packet.GetHexMaxU64 (false, std::numeric_limits<lldb::tid_t>::max ());

    // Ensure we have the given thread when not specifying -1 (all threads) or 0 (any thread).
    if (tid != LLDB_INVALID_THREAD_ID && tid != 0)
    {
        NativeThreadProtocolSP thread_sp (m_debugged_process_sp->GetThreadByID (tid));
        if (!thread_sp)
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s failed, tid %" PRIu64 " not found", __FUNCTION__, tid);
            return SendErrorResponse (0x15);
        }
    }

    // Now switch the given thread type.
    switch (h_variant)
    {
        case 'g':
            SetCurrentThreadID (tid);
            break;

        case 'c':
            SetContinueThreadID (tid);
            break;

        default:
            assert (false && "unsupported $H variant - shouldn't get here");
            return SendIllFormedResponse (packet, "H variant unsupported, should be c or g");
    }

    return SendOKResponse();
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_interrupt (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

    // Ensure we're llgs.
    if (!IsGdbServer())
    {
        // Only supported on llgs
        return SendUnimplementedResponse ("");
    }

    // Fail if we don't have a current process.
    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, no process available", __FUNCTION__);
        return SendErrorResponse (0x15);
    }

    // Build the ResumeActionList - stop everything.
    lldb_private::ResumeActionList actions (StateType::eStateStopped, 0);

    Error error = m_debugged_process_sp->Resume (actions);
    if (error.Fail ())
    {
        if (log)
        {
            log->Printf ("GDBRemoteCommunicationServer::%s failed for process %" PRIu64 ": %s",
                         __FUNCTION__,
                         m_debugged_process_sp->GetID (),
                         error.AsCString ());
        }
        return SendErrorResponse (GDBRemoteServerError::eErrorResume);
    }

    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s stopped process %" PRIu64, __FUNCTION__, m_debugged_process_sp->GetID ());

    // No response required from stop all.
    return PacketResult::Success;
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_m (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

    // Ensure we're llgs.
    if (!IsGdbServer())
    {
        // Only supported on llgs
        return SendUnimplementedResponse ("");
    }

    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, no process available", __FUNCTION__);
        return SendErrorResponse (0x15);
    }

    // Parse out the memory address.
    packet.SetFilePos (strlen("m"));
    if (packet.GetBytesLeft() < 1)
        return SendIllFormedResponse(packet, "Too short m packet");

    // Read the address.  Punting on validation.
    // FIXME replace with Hex U64 read with no default value that fails on failed read.
    const lldb::addr_t read_addr = packet.GetHexMaxU64(false, 0);

    // Validate comma.
    if ((packet.GetBytesLeft() < 1) || (packet.GetChar() != ','))
        return SendIllFormedResponse(packet, "Comma sep missing in m packet");

    // Get # bytes to read.
    if (packet.GetBytesLeft() < 1)
        return SendIllFormedResponse(packet, "Length missing in m packet");

    const uint64_t byte_count = packet.GetHexMaxU64(false, 0);
    if (byte_count == 0)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s nothing to read: zero-length packet", __FUNCTION__);
        return PacketResult::Success;
    }

    // Allocate the response buffer.
    std::string buf(byte_count, '\0');
    if (buf.empty())
        return SendErrorResponse (0x78);


    // Retrieve the process memory.
    lldb::addr_t bytes_read = 0;
    lldb_private::Error error = m_debugged_process_sp->ReadMemory (read_addr, &buf[0], byte_count, bytes_read);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " mem 0x%" PRIx64 ": failed to read. Error: %s", __FUNCTION__, m_debugged_process_sp->GetID (), read_addr, error.AsCString ());
        return SendErrorResponse (0x08);
    }

    if (bytes_read == 0)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " mem 0x%" PRIx64 ": read %" PRIu64 " of %" PRIu64 " requested bytes", __FUNCTION__, m_debugged_process_sp->GetID (), read_addr, bytes_read, byte_count);
        return SendErrorResponse (0x08);
    }

    StreamGDBRemote response;
    for (lldb::addr_t i = 0; i < bytes_read; ++i)
        response.PutHex8(buf[i]);

    return SendPacketNoLock(response.GetData(), response.GetSize());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::Handle_QSetDetachOnError (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QSetDetachOnError:"));
    if (packet.GetU32(0))
        m_process_launch_info.GetFlags().Set (eLaunchFlagDetachOnError);
    else
        m_process_launch_info.GetFlags().Clear (eLaunchFlagDetachOnError);
    return SendOKResponse ();
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_M (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

    // Ensure we're llgs.
    if (!IsGdbServer())
    {
        // Only supported on llgs
        return SendUnimplementedResponse ("");
    }

    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, no process available", __FUNCTION__);
        return SendErrorResponse (0x15);
    }

    // Parse out the memory address.
    packet.SetFilePos (strlen("M"));
    if (packet.GetBytesLeft() < 1)
        return SendIllFormedResponse(packet, "Too short M packet");

    // Read the address.  Punting on validation.
    // FIXME replace with Hex U64 read with no default value that fails on failed read.
    const lldb::addr_t write_addr = packet.GetHexMaxU64(false, 0);

    // Validate comma.
    if ((packet.GetBytesLeft() < 1) || (packet.GetChar() != ','))
        return SendIllFormedResponse(packet, "Comma sep missing in M packet");

    // Get # bytes to read.
    if (packet.GetBytesLeft() < 1)
        return SendIllFormedResponse(packet, "Length missing in M packet");

    const uint64_t byte_count = packet.GetHexMaxU64(false, 0);
    if (byte_count == 0)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s nothing to write: zero-length packet", __FUNCTION__);
        return PacketResult::Success;
    }

    // Validate colon.
    if ((packet.GetBytesLeft() < 1) || (packet.GetChar() != ':'))
        return SendIllFormedResponse(packet, "Comma sep missing in M packet after byte length");

    // Allocate the conversion buffer.
    std::vector<uint8_t> buf(byte_count, 0);
    if (buf.empty())
        return SendErrorResponse (0x78);

    // Convert the hex memory write contents to bytes.
    StreamGDBRemote response;
    const uint64_t convert_count = static_cast<uint64_t> (packet.GetHexBytes (&buf[0], byte_count, 0));
    if (convert_count != byte_count)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " mem 0x%" PRIx64 ": asked to write %" PRIu64 " bytes, but only found %" PRIu64 " to convert.", __FUNCTION__, m_debugged_process_sp->GetID (), write_addr, byte_count, convert_count);
        return SendIllFormedResponse (packet, "M content byte length specified did not match hex-encoded content length");
    }

    // Write the process memory.
    lldb::addr_t bytes_written = 0;
    lldb_private::Error error = m_debugged_process_sp->WriteMemory (write_addr, &buf[0], byte_count, bytes_written);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " mem 0x%" PRIx64 ": failed to write. Error: %s", __FUNCTION__, m_debugged_process_sp->GetID (), write_addr, error.AsCString ());
        return SendErrorResponse (0x09);
    }

    if (bytes_written == 0)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " mem 0x%" PRIx64 ": wrote %" PRIu64 " of %" PRIu64 " requested bytes", __FUNCTION__, m_debugged_process_sp->GetID (), write_addr, bytes_written, byte_count);
        return SendErrorResponse (0x09);
    }

    return SendOKResponse ();
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_qMemoryRegionInfoSupported (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

    // We don't support if we're not llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("");

    // Currently only the NativeProcessProtocol knows if it can handle a qMemoryRegionInfoSupported
    // request, but we're not guaranteed to be attached to a process.  For now we'll assume the
    // client only asks this when a process is being debugged.

    // Ensure we have a process running; otherwise, we can't figure this out
    // since we won't have a NativeProcessProtocol.
    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, no process available", __FUNCTION__);
        return SendErrorResponse (0x15);
    }

    // Test if we can get any region back when asking for the region around NULL.
    MemoryRegionInfo region_info;
    const Error error = m_debugged_process_sp->GetMemoryRegionInfo (0, region_info);
    if (error.Fail ())
    {
        // We don't support memory region info collection for this NativeProcessProtocol.
        return SendUnimplementedResponse ("");
    }

    return SendOKResponse();
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_qMemoryRegionInfo (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

    // We don't support if we're not llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("");

    // Ensure we have a process.
    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, no process available", __FUNCTION__);
        return SendErrorResponse (0x15);
    }

    // Parse out the memory address.
    packet.SetFilePos (strlen("qMemoryRegionInfo:"));
    if (packet.GetBytesLeft() < 1)
        return SendIllFormedResponse(packet, "Too short qMemoryRegionInfo: packet");

    // Read the address.  Punting on validation.
    const lldb::addr_t read_addr = packet.GetHexMaxU64(false, 0);

    StreamGDBRemote response;

    // Get the memory region info for the target address.
    MemoryRegionInfo region_info;
    const Error error = m_debugged_process_sp->GetMemoryRegionInfo (read_addr, region_info);
    if (error.Fail ())
    {
        // Return the error message.

        response.PutCString ("error:");
        response.PutCStringAsRawHex8 (error.AsCString ());
        response.PutChar (';');
    }
    else
    {
        // Range start and size.
        response.Printf ("start:%" PRIx64 ";size:%" PRIx64 ";", region_info.GetRange ().GetRangeBase (), region_info.GetRange ().GetByteSize ());

        // Permissions.
        if (region_info.GetReadable () ||
            region_info.GetWritable () ||
            region_info.GetExecutable ())
        {
            // Write permissions info.
            response.PutCString ("permissions:");

            if (region_info.GetReadable ())
                response.PutChar ('r');
            if (region_info.GetWritable ())
                response.PutChar('w');
            if (region_info.GetExecutable())
                response.PutChar ('x');

            response.PutChar (';');
        }
    }

    return SendPacketNoLock(response.GetData(), response.GetSize());
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_Z (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));

    // We don't support if we're not llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("");

    // Ensure we have a process.
    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, no process available", __FUNCTION__);
        return SendErrorResponse (0x15);
    }

    // Parse out software or hardware breakpoint requested.
    packet.SetFilePos (strlen("Z"));
    if (packet.GetBytesLeft() < 1)
        return SendIllFormedResponse(packet, "Too short Z packet, missing software/hardware specifier");

    bool want_breakpoint = true;
    bool want_hardware = false;

    const char breakpoint_type_char = packet.GetChar ();
    switch (breakpoint_type_char)
    {
        case '0': want_hardware = false; want_breakpoint = true;  break;
        case '1': want_hardware = true;  want_breakpoint = true;  break;
        case '2': want_breakpoint = false; break;
        case '3': want_breakpoint = false; break;
        default:
            return SendIllFormedResponse(packet, "Z packet had invalid software/hardware specifier");

    }

    if ((packet.GetBytesLeft() < 1) || packet.GetChar () != ',')
        return SendIllFormedResponse(packet, "Malformed Z packet, expecting comma after breakpoint type");

    // FIXME implement watchpoint support.
    if (!want_breakpoint)
        return SendUnimplementedResponse ("watchpoint support not yet implemented");

    // Parse out the breakpoint address.
    if (packet.GetBytesLeft() < 1)
        return SendIllFormedResponse(packet, "Too short Z packet, missing address");
    const lldb::addr_t breakpoint_addr = packet.GetHexMaxU64(false, 0);

    if ((packet.GetBytesLeft() < 1) || packet.GetChar () != ',')
        return SendIllFormedResponse(packet, "Malformed Z packet, expecting comma after address");

    // Parse out the breakpoint kind (i.e. size hint for opcode size).
    const uint32_t kind = packet.GetHexMaxU32 (false, std::numeric_limits<uint32_t>::max ());
    if (kind == std::numeric_limits<uint32_t>::max ())
        return SendIllFormedResponse(packet, "Malformed Z packet, failed to parse kind argument");

    if (want_breakpoint)
    {
        // Try to set the breakpoint.
        const Error error = m_debugged_process_sp->SetBreakpoint (breakpoint_addr, kind, want_hardware);
        if (error.Success ())
            return SendOKResponse ();
        else
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " failed to set breakpoint: %s", __FUNCTION__, m_debugged_process_sp->GetID (), error.AsCString ());
            return SendErrorResponse (0x09);
        }
    }

    // FIXME fix up after watchpoints are handled.
    return SendUnimplementedResponse ("");
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_z (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));

    // We don't support if we're not llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("");

    // Ensure we have a process.
    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, no process available", __FUNCTION__);
        return SendErrorResponse (0x15);
    }

    // Parse out software or hardware breakpoint requested.
    packet.SetFilePos (strlen("Z"));
    if (packet.GetBytesLeft() < 1)
        return SendIllFormedResponse(packet, "Too short z packet, missing software/hardware specifier");

    bool want_breakpoint = true;

    const char breakpoint_type_char = packet.GetChar ();
    switch (breakpoint_type_char)
    {
        case '0': want_breakpoint = true;  break;
        case '1': want_breakpoint = true;  break;
        case '2': want_breakpoint = false; break;
        case '3': want_breakpoint = false; break;
        default:
            return SendIllFormedResponse(packet, "z packet had invalid software/hardware specifier");

    }

    if ((packet.GetBytesLeft() < 1) || packet.GetChar () != ',')
        return SendIllFormedResponse(packet, "Malformed z packet, expecting comma after breakpoint type");

    // FIXME implement watchpoint support.
    if (!want_breakpoint)
        return SendUnimplementedResponse ("watchpoint support not yet implemented");

    // Parse out the breakpoint address.
    if (packet.GetBytesLeft() < 1)
        return SendIllFormedResponse(packet, "Too short z packet, missing address");
    const lldb::addr_t breakpoint_addr = packet.GetHexMaxU64(false, 0);

    if ((packet.GetBytesLeft() < 1) || packet.GetChar () != ',')
        return SendIllFormedResponse(packet, "Malformed z packet, expecting comma after address");

    // Parse out the breakpoint kind (i.e. size hint for opcode size).
    const uint32_t kind = packet.GetHexMaxU32 (false, std::numeric_limits<uint32_t>::max ());
    if (kind == std::numeric_limits<uint32_t>::max ())
        return SendIllFormedResponse(packet, "Malformed z packet, failed to parse kind argument");

    if (want_breakpoint)
    {
        // Try to set the breakpoint.
        const Error error = m_debugged_process_sp->RemoveBreakpoint (breakpoint_addr);
        if (error.Success ())
            return SendOKResponse ();
        else
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " failed to remove breakpoint: %s", __FUNCTION__, m_debugged_process_sp->GetID (), error.AsCString ());
            return SendErrorResponse (0x09);
        }
    }

    // FIXME fix up after watchpoints are handled.
    return SendUnimplementedResponse ("");
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_s (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS|LIBLLDB_LOG_THREAD));

    // We don't support if we're not llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("");

    // Ensure we have a process.
    if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed, no process available", __FUNCTION__);
        return SendErrorResponse (0x32);
    }

    // We first try to use a continue thread id.  If any one or any all set, use the current thread.
    // Bail out if we don't have a thread id.
    lldb::tid_t tid = GetContinueThreadID ();
    if (tid == 0 || tid == LLDB_INVALID_THREAD_ID)
        tid = GetCurrentThreadID ();
    if (tid == LLDB_INVALID_THREAD_ID)
        return SendErrorResponse (0x33);

    // Double check that we have such a thread.
    // TODO investigate: on MacOSX we might need to do an UpdateThreads () here.
    NativeThreadProtocolSP thread_sp = m_debugged_process_sp->GetThreadByID (tid);
    if (!thread_sp || thread_sp->GetID () != tid)
        return SendErrorResponse (0x33);

    // Create the step action for the given thread.
    lldb_private::ResumeAction action = { tid, eStateStepping, 0 };

    // Setup the actions list.
    lldb_private::ResumeActionList actions;
    actions.Append (action);

    // All other threads stop while we're single stepping a thread.
    actions.SetDefaultThreadActionIfNeeded(eStateStopped, 0);
    Error error = m_debugged_process_sp->Resume (actions);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " tid %" PRIu64 " Resume() failed with error: %s", __FUNCTION__, m_debugged_process_sp->GetID (), tid, error.AsCString ());
        return SendErrorResponse(0x49);
    }

    // No response here - the stop or exit will come from the resulting action.
    return PacketResult::Success;
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_qSupported (StringExtractorGDBRemote &packet)
{
    StreamGDBRemote response;

    // Features common to lldb-platform and llgs.
    uint32_t max_packet_size = 128 * 1024;  // 128KBytes is a reasonable max packet size--debugger can always use less
    response.Printf ("PacketSize=%x", max_packet_size);

    response.PutCString (";QStartNoAckMode+");
    response.PutCString (";QThreadSuffixSupported+");
    response.PutCString (";QListThreadsInStopReply+");
#if defined(__linux__)
    response.PutCString (";qXfer:auxv:read+");
#endif

    return SendPacketNoLock(response.GetData(), response.GetSize());
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_QThreadSuffixSupported (StringExtractorGDBRemote &packet)
{
    m_thread_suffix_supported = true;
    return SendOKResponse();
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_QListThreadsInStopReply (StringExtractorGDBRemote &packet)
{
    m_list_threads_in_stop_reply = true;
    return SendOKResponse();
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_qXfer_auxv_read (StringExtractorGDBRemote &packet)
{
    // We don't support if we're not llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("only supported for lldb-gdbserver");

    // *BSD impls should be able to do this too.
#if defined(__linux__)
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

    // Parse out the offset.
    packet.SetFilePos (strlen("qXfer:auxv:read::"));
    if (packet.GetBytesLeft () < 1)
        return SendIllFormedResponse (packet, "qXfer:auxv:read:: packet missing offset");

    const uint64_t auxv_offset = packet.GetHexMaxU64 (false, std::numeric_limits<uint64_t>::max ());
    if (auxv_offset == std::numeric_limits<uint64_t>::max ())
        return SendIllFormedResponse (packet, "qXfer:auxv:read:: packet missing offset");

    // Parse out comma.
    if (packet.GetBytesLeft () < 1 || packet.GetChar () != ',')
        return SendIllFormedResponse (packet, "qXfer:auxv:read:: packet missing comma after offset");

    // Parse out the length.
    const uint64_t auxv_length = packet.GetHexMaxU64 (false, std::numeric_limits<uint64_t>::max ());
    if (auxv_length == std::numeric_limits<uint64_t>::max ())
        return SendIllFormedResponse (packet, "qXfer:auxv:read:: packet missing length");

    // Grab the auxv data if we need it.
    if (!m_active_auxv_buffer_sp)
    {
        // Make sure we have a valid process.
        if (!m_debugged_process_sp || (m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID))
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s failed, no process available", __FUNCTION__);
            return SendErrorResponse (0x10);
        }

        // Grab the auxv data.
        m_active_auxv_buffer_sp = Host::GetAuxvData (m_debugged_process_sp->GetID ());
        if (!m_active_auxv_buffer_sp || m_active_auxv_buffer_sp->GetByteSize () ==  0)
        {
            // Hmm, no auxv data, call that an error.
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s failed, no auxv data retrieved", __FUNCTION__);
            m_active_auxv_buffer_sp.reset ();
            return SendErrorResponse (0x11);
        }
    }

    // FIXME find out if/how I lock the stream here.

    StreamGDBRemote response;
    bool done_with_buffer = false;

    if (auxv_offset >= m_active_auxv_buffer_sp->GetByteSize ())
    {
        // We have nothing left to send.  Mark the buffer as complete.
        response.PutChar ('l');
        done_with_buffer = true;
    }
    else
    {
        // Figure out how many bytes are available starting at the given offset.
        const uint64_t bytes_remaining = m_active_auxv_buffer_sp->GetByteSize () - auxv_offset;

        // Figure out how many bytes we're going to read.
        const uint64_t bytes_to_read = (auxv_length > bytes_remaining) ? bytes_remaining : auxv_length;

        // Mark the response type according to whether we're reading the remainder of the auxv data.
        if (bytes_to_read >= bytes_remaining)
        {
            // There will be nothing left to read after this
            response.PutChar ('l');
            done_with_buffer = true;
        }
        else
        {
            // There will still be bytes to read after this request.
            response.PutChar ('m');
        }

        // Now write the data in encoded binary form.
        response.PutEscapedBytes (m_active_auxv_buffer_sp->GetBytes () + auxv_offset, bytes_to_read);
    }

    if (done_with_buffer)
        m_active_auxv_buffer_sp.reset ();

    return SendPacketNoLock(response.GetData(), response.GetSize());
#else
    return SendUnimplementedResponse ("not implemented on this platform");
#endif
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_QSaveRegisterState (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

    // We don't support if we're not llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("only supported for lldb-gdbserver");

    // Move past packet name.
    packet.SetFilePos (strlen ("QSaveRegisterState"));

    // Get the thread to use.
    NativeThreadProtocolSP thread_sp = GetThreadFromSuffix (packet);
    if (!thread_sp)
    {
        if (m_thread_suffix_supported)
            return SendIllFormedResponse (packet, "No thread specified in QSaveRegisterState packet");
        else
            return SendIllFormedResponse (packet, "No thread was is set with the Hg packet");
    }

    // Grab the register context for the thread.
    NativeRegisterContextSP reg_context_sp (thread_sp->GetRegisterContext ());
    if (!reg_context_sp)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " tid %" PRIu64 " failed, no register context available for the thread", __FUNCTION__, m_debugged_process_sp->GetID (), thread_sp->GetID ());
        return SendErrorResponse (0x15);
    }

    // Save registers to a buffer.
    DataBufferSP register_data_sp;
    Error error = reg_context_sp->ReadAllRegisterValues (register_data_sp);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " failed to save all register values: %s", __FUNCTION__, m_debugged_process_sp->GetID (), error.AsCString ());
        return SendErrorResponse (0x75);
    }

    // Allocate a new save id.
    const uint32_t save_id = GetNextSavedRegistersID ();
    assert ((m_saved_registers_map.find (save_id) == m_saved_registers_map.end ()) && "GetNextRegisterSaveID() returned an existing register save id");

    // Save the register data buffer under the save id.
    {
        Mutex::Locker locker (m_saved_registers_mutex);
        m_saved_registers_map[save_id] = register_data_sp;
    }

    // Write the response.
    StreamGDBRemote response;
    response.Printf ("%" PRIu32, save_id);
    return SendPacketNoLock(response.GetData(), response.GetSize());
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_QRestoreRegisterState (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

    // We don't support if we're not llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("only supported for lldb-gdbserver");

    // Parse out save id.
    packet.SetFilePos (strlen ("QRestoreRegisterState:"));
    if (packet.GetBytesLeft () < 1)
        return SendIllFormedResponse (packet, "QRestoreRegisterState packet missing register save id");

    const uint32_t save_id = packet.GetU32 (0);
    if (save_id == 0)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s QRestoreRegisterState packet has malformed save id, expecting decimal uint32_t", __FUNCTION__);
        return SendErrorResponse (0x76);
    }

    // Get the thread to use.
    NativeThreadProtocolSP thread_sp = GetThreadFromSuffix (packet);
    if (!thread_sp)
    {
        if (m_thread_suffix_supported)
            return SendIllFormedResponse (packet, "No thread specified in QRestoreRegisterState packet");
        else
            return SendIllFormedResponse (packet, "No thread was is set with the Hg packet");
    }

    // Grab the register context for the thread.
    NativeRegisterContextSP reg_context_sp (thread_sp->GetRegisterContext ());
    if (!reg_context_sp)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " tid %" PRIu64 " failed, no register context available for the thread", __FUNCTION__, m_debugged_process_sp->GetID (), thread_sp->GetID ());
        return SendErrorResponse (0x15);
    }

    // Retrieve register state buffer, then remove from the list.
    DataBufferSP register_data_sp;
    {
        Mutex::Locker locker (m_saved_registers_mutex);

        // Find the register set buffer for the given save id.
        auto it = m_saved_registers_map.find (save_id);
        if (it == m_saved_registers_map.end ())
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " does not have a register set save buffer for id %" PRIu32, __FUNCTION__, m_debugged_process_sp->GetID (), save_id);
            return SendErrorResponse (0x77);
        }
        register_data_sp = it->second;

        // Remove it from the map.
        m_saved_registers_map.erase (it);
    }

    Error error = reg_context_sp->WriteAllRegisterValues (register_data_sp);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s pid %" PRIu64 " failed to restore all register values: %s", __FUNCTION__, m_debugged_process_sp->GetID (), error.AsCString ());
        return SendErrorResponse (0x77);
    }

    return SendOKResponse();
}

GDBRemoteCommunicationServer::PacketResult
GDBRemoteCommunicationServer::Handle_vAttach (StringExtractorGDBRemote &packet)
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

    // We don't support if we're not llgs.
    if (!IsGdbServer())
        return SendUnimplementedResponse ("only supported for lldb-gdbserver");

    // Consume the ';' after vAttach.
    packet.SetFilePos (strlen ("vAttach"));
    if (!packet.GetBytesLeft () || packet.GetChar () != ';')
        return SendIllFormedResponse (packet, "vAttach missing expected ';'");

    // Grab the PID to which we will attach (assume hex encoding).
    lldb::pid_t pid = packet.GetU32 (LLDB_INVALID_PROCESS_ID, 16);
    if (pid == LLDB_INVALID_PROCESS_ID)
        return SendIllFormedResponse (packet, "vAttach failed to parse the process id");

    // Attempt to attach.
    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s attempting to attach to pid %" PRIu64, __FUNCTION__, pid);

    Error error = AttachToProcess (pid);

    if (error.Fail ())
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s failed to attach to pid %" PRIu64 ": %s\n", __FUNCTION__, pid, error.AsCString());
        return SendErrorResponse (0x01);
    }

    // Notify we attached by sending a stop packet.
    return SendStopReasonForState (m_debugged_process_sp->GetState (), true);

    return PacketResult::Success;
}

void
GDBRemoteCommunicationServer::FlushInferiorOutput ()
{
    // If we're not monitoring an inferior's terminal, ignore this.
    if (!m_stdio_communication.IsConnected())
        return;

    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf ("GDBRemoteCommunicationServer::%s() called", __FUNCTION__);

    // FIXME implement a timeout on the join.
    m_stdio_communication.JoinReadThread();
}

void
GDBRemoteCommunicationServer::MaybeCloseInferiorTerminalConnection ()
{
    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

    // Tell the stdio connection to shut down.
    if (m_stdio_communication.IsConnected())
    {
        auto connection = m_stdio_communication.GetConnection();
        if (connection)
        {
            Error error;
            connection->Disconnect (&error);

            if (error.Success ())
            {
                if (log)
                    log->Printf ("GDBRemoteCommunicationServer::%s disconnect process terminal stdio - SUCCESS", __FUNCTION__);
            }
            else
            {
                if (log)
                    log->Printf ("GDBRemoteCommunicationServer::%s disconnect process terminal stdio - FAIL: %s", __FUNCTION__, error.AsCString ());
            }
        }
    }
}


lldb_private::NativeThreadProtocolSP
GDBRemoteCommunicationServer::GetThreadFromSuffix (StringExtractorGDBRemote &packet)
{
    NativeThreadProtocolSP thread_sp;

    // We have no thread if we don't have a process.
    if (!m_debugged_process_sp || m_debugged_process_sp->GetID () == LLDB_INVALID_PROCESS_ID)
        return thread_sp;

    // If the client hasn't asked for thread suffix support, there will not be a thread suffix.
    // Use the current thread in that case.
    if (!m_thread_suffix_supported)
    {
        const lldb::tid_t current_tid = GetCurrentThreadID ();
        if (current_tid == LLDB_INVALID_THREAD_ID)
            return thread_sp;
        else if (current_tid == 0)
        {
            // Pick a thread.
            return m_debugged_process_sp->GetThreadAtIndex (0);
        }
        else
            return m_debugged_process_sp->GetThreadByID (current_tid);
    }

    Log *log (GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

    // Parse out the ';'.
    if (packet.GetBytesLeft () < 1 || packet.GetChar () != ';')
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s gdb-remote parse error: expected ';' prior to start of thread suffix: packet contents = '%s'", __FUNCTION__, packet.GetStringRef ().c_str ());
        return thread_sp;
    }

    if (!packet.GetBytesLeft ())
        return thread_sp;

    // Parse out thread: portion.
    if (strncmp (packet.Peek (), "thread:", strlen("thread:")) != 0)
    {
        if (log)
            log->Printf ("GDBRemoteCommunicationServer::%s gdb-remote parse error: expected 'thread:' but not found, packet contents = '%s'", __FUNCTION__, packet.GetStringRef ().c_str ());
        return thread_sp;
    }
    packet.SetFilePos (packet.GetFilePos () + strlen("thread:"));
    const lldb::tid_t tid = packet.GetHexMaxU64(false, 0);
    if (tid != 0)
        return m_debugged_process_sp->GetThreadByID (tid);

    return thread_sp;
}

lldb::tid_t
GDBRemoteCommunicationServer::GetCurrentThreadID () const
{
    if (m_current_tid == 0 || m_current_tid == LLDB_INVALID_THREAD_ID)
    {
        // Use whatever the debug process says is the current thread id
        // since the protocol either didn't specify or specified we want
        // any/all threads marked as the current thread.
        if (!m_debugged_process_sp)
            return LLDB_INVALID_THREAD_ID;
        return m_debugged_process_sp->GetCurrentThreadID ();
    }
    // Use the specific current thread id set by the gdb remote protocol.
    return m_current_tid;
}

uint32_t
GDBRemoteCommunicationServer::GetNextSavedRegistersID ()
{
    Mutex::Locker locker (m_saved_registers_mutex);
    return m_next_saved_registers_id++;
}

