//===-- GDBRemoteCommunicationServer.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <errno.h>

#include "GDBRemoteCommunicationServer.h"
#include "lldb/Core/StreamGDBRemote.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/ADT/Triple.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Endian.h"
#include "lldb/Host/File.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Target/Process.h"

// Project includes
#include "Utility/StringExtractorGDBRemote.h"
#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// GDBRemoteCommunicationServer constructor
//----------------------------------------------------------------------
GDBRemoteCommunicationServer::GDBRemoteCommunicationServer(bool is_platform) :
    GDBRemoteCommunication ("gdb-remote.server", "gdb-remote.server.rx_packet", is_platform),
    m_async_thread (LLDB_INVALID_HOST_THREAD),
    m_process_launch_info (),
    m_process_launch_error (),
    m_spawned_pids (),
    m_spawned_pids_mutex (Mutex::eMutexTypeRecursive),
    m_proc_infos (),
    m_proc_infos_index (0),
    m_port_map (),
    m_port_offset(0)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
GDBRemoteCommunicationServer::~GDBRemoteCommunicationServer()
{
}


//void *
//GDBRemoteCommunicationServer::AsyncThread (void *arg)
//{
//    GDBRemoteCommunicationServer *server = (GDBRemoteCommunicationServer*) arg;
//
//    Log *log;// (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));
//    if (log)
//        log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %i) thread starting...", __FUNCTION__, arg, process->GetID());
//
//    StringExtractorGDBRemote packet;
//
//    while ()
//    {
//        if (packet.
//    }
//
//    if (log)
//        log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %i) thread exiting...", __FUNCTION__, arg, process->GetID());
//
//    process->m_async_thread = LLDB_INVALID_HOST_THREAD;
//    return NULL;
//}
//
bool
GDBRemoteCommunicationServer::GetPacketAndSendResponse (uint32_t timeout_usec,
                                                        Error &error,
                                                        bool &interrupt,
                                                        bool &quit)
{
    StringExtractorGDBRemote packet;
    if (WaitForPacketWithTimeoutMicroSecondsNoLock (packet, timeout_usec))
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

            case StringExtractorGDBRemote::eServerPacketType_interrupt:
                error.SetErrorString("interrupt received");
                interrupt = true;
                break;

            case StringExtractorGDBRemote::eServerPacketType_unimplemented:
                return SendUnimplementedResponse (packet.GetStringRef().c_str()) > 0;

            case StringExtractorGDBRemote::eServerPacketType_A:
                return Handle_A (packet);

            case StringExtractorGDBRemote::eServerPacketType_qfProcessInfo:
                return Handle_qfProcessInfo (packet);

            case StringExtractorGDBRemote::eServerPacketType_qsProcessInfo:
                return Handle_qsProcessInfo (packet);

            case StringExtractorGDBRemote::eServerPacketType_qC:
                return Handle_qC (packet);

            case StringExtractorGDBRemote::eServerPacketType_qHostInfo:
                return Handle_qHostInfo (packet);

            case StringExtractorGDBRemote::eServerPacketType_qLaunchGDBServer:
                return Handle_qLaunchGDBServer (packet);

            case StringExtractorGDBRemote::eServerPacketType_qKillSpawnedProcess:
                return Handle_qKillSpawnedProcess (packet);

            case StringExtractorGDBRemote::eServerPacketType_qLaunchSuccess:
                return Handle_qLaunchSuccess (packet);

            case StringExtractorGDBRemote::eServerPacketType_qGroupName:
                return Handle_qGroupName (packet);

            case StringExtractorGDBRemote::eServerPacketType_qProcessInfoPID:
                return Handle_qProcessInfoPID (packet);

            case StringExtractorGDBRemote::eServerPacketType_qSpeedTest:
                return Handle_qSpeedTest (packet);

            case StringExtractorGDBRemote::eServerPacketType_qUserName:
                return Handle_qUserName (packet);

            case StringExtractorGDBRemote::eServerPacketType_qGetWorkingDir:
                return Handle_qGetWorkingDir(packet);
    
            case StringExtractorGDBRemote::eServerPacketType_QEnvironment:
                return Handle_QEnvironment (packet);

            case StringExtractorGDBRemote::eServerPacketType_QLaunchArch:
                return Handle_QLaunchArch (packet);

            case StringExtractorGDBRemote::eServerPacketType_QSetDisableASLR:
                return Handle_QSetDisableASLR (packet);

            case StringExtractorGDBRemote::eServerPacketType_QSetSTDIN:
                return Handle_QSetSTDIN (packet);

            case StringExtractorGDBRemote::eServerPacketType_QSetSTDOUT:
                return Handle_QSetSTDOUT (packet);

            case StringExtractorGDBRemote::eServerPacketType_QSetSTDERR:
                return Handle_QSetSTDERR (packet);

            case StringExtractorGDBRemote::eServerPacketType_QSetWorkingDir:
                return Handle_QSetWorkingDir (packet);

            case StringExtractorGDBRemote::eServerPacketType_QStartNoAckMode:
                return Handle_QStartNoAckMode (packet);

            case StringExtractorGDBRemote::eServerPacketType_qPlatform_mkdir:
                return Handle_qPlatform_mkdir (packet);

            case StringExtractorGDBRemote::eServerPacketType_qPlatform_chmod:
                return Handle_qPlatform_chmod (packet);

            case StringExtractorGDBRemote::eServerPacketType_qPlatform_shell:
                return Handle_qPlatform_shell (packet);

            case StringExtractorGDBRemote::eServerPacketType_vFile_open:
                return Handle_vFile_Open (packet);

            case StringExtractorGDBRemote::eServerPacketType_vFile_close:
                return Handle_vFile_Close (packet);

            case StringExtractorGDBRemote::eServerPacketType_vFile_pread:
                return Handle_vFile_pRead (packet);

            case StringExtractorGDBRemote::eServerPacketType_vFile_pwrite:
                return Handle_vFile_pWrite (packet);

            case StringExtractorGDBRemote::eServerPacketType_vFile_size:
                return Handle_vFile_Size (packet);

            case StringExtractorGDBRemote::eServerPacketType_vFile_mode:
                return Handle_vFile_Mode (packet);

            case StringExtractorGDBRemote::eServerPacketType_vFile_exists:
                return Handle_vFile_Exists (packet);

            case StringExtractorGDBRemote::eServerPacketType_vFile_stat:
                return Handle_vFile_Stat (packet);

            case StringExtractorGDBRemote::eServerPacketType_vFile_md5:
                return Handle_vFile_MD5 (packet);

            case StringExtractorGDBRemote::eServerPacketType_vFile_symlink:
                return Handle_vFile_symlink (packet);
                
            case StringExtractorGDBRemote::eServerPacketType_vFile_unlink:
                return Handle_vFile_unlink (packet);
        }
        return true;
    }
    else
    {
        if (!IsConnected())
            error.SetErrorString("lost connection");
        else
            error.SetErrorString("timeout");
    }

    return false;
}

size_t
GDBRemoteCommunicationServer::SendUnimplementedResponse (const char *)
{
    // TODO: Log the packet we aren't handling...
    return SendPacketNoLock ("", 0);
}

size_t
GDBRemoteCommunicationServer::SendErrorResponse (uint8_t err)
{
    char packet[16];
    int packet_len = ::snprintf (packet, sizeof(packet), "E%2.2x", err);
    assert (packet_len < (int)sizeof(packet));
    return SendPacketNoLock (packet, packet_len);
}


size_t
GDBRemoteCommunicationServer::SendOKResponse ()
{
    return SendPacketNoLock ("OK", 2);
}

bool
GDBRemoteCommunicationServer::HandshakeWithClient(Error *error_ptr)
{
    return GetAck();
}

bool
GDBRemoteCommunicationServer::Handle_qHostInfo (StringExtractorGDBRemote &packet)
{
    StreamString response;

    // $cputype:16777223;cpusubtype:3;ostype:Darwin;vendor:apple;endian:little;ptrsize:8;#00

    ArchSpec host_arch (Host::GetArchitecture ());
    const llvm::Triple &host_triple = host_arch.GetTriple();
    response.PutCString("triple:");
    response.PutCStringAsRawHex8(host_triple.getTriple().c_str());
    response.Printf (";ptrsize:%u;",host_arch.GetAddressByteSize());

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
    if (Host::GetOSVersion (major, minor, update))
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
    if (Host::GetOSBuildString (s))
    {
        response.PutCString ("os_build:");
        response.PutCStringAsRawHex8(s.c_str());
        response.PutChar(';');
    }
    if (Host::GetOSKernelDescription (s))
    {
        response.PutCString ("os_kernel:");
        response.PutCStringAsRawHex8(s.c_str());
        response.PutChar(';');
    }
#if defined(__APPLE__)

#if defined(__arm__)
    // For iOS devices, we are connected through a USB Mux so we never pretend
    // to actually have a hostname as far as the remote lldb that is connecting
    // to this lldb-platform is concerned
    response.PutCString ("hostname:");
    response.PutCStringAsRawHex8("localhost");
    response.PutChar(';');
#else   // #if defined(__arm__)
    if (Host::GetHostname (s))
    {
        response.PutCString ("hostname:");
        response.PutCStringAsRawHex8(s.c_str());
        response.PutChar(';');
    }

#endif  // #if defined(__arm__)

#else   // #if defined(__APPLE__)
    if (Host::GetHostname (s))
    {
        response.PutCString ("hostname:");
        response.PutCStringAsRawHex8(s.c_str());
        response.PutChar(';');
    }
#endif  // #if defined(__APPLE__)

    return SendPacketNoLock (response.GetData(), response.GetSize()) > 0;
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
        response.PutCStringAsRawHex8(proc_triple.getTriple().c_str());
        response.PutChar(';');
    }
}

bool
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

bool
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

bool
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

bool
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

bool
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

bool
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


static void *
AcceptPortFromInferior (void *arg)
{
    const char *connect_url = (const char *)arg;
    ConnectionFileDescriptor file_conn;
    Error error;
    if (file_conn.Connect (connect_url, &error) == eConnectionStatusSuccess)
    {
        char pid_str[256];
        ::memset (pid_str, 0, sizeof(pid_str));
        ConnectionStatus status;
        const size_t pid_str_len = file_conn.Read (pid_str, sizeof(pid_str), 0, status, NULL);
        if (pid_str_len > 0)
        {
            int pid = atoi (pid_str);
            return (void *)(intptr_t)pid;
        }
    }
    return NULL;
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

bool
GDBRemoteCommunicationServer::Handle_A (StringExtractorGDBRemote &packet)
{
    // The 'A' packet is the most over designed packet ever here with
    // redundant argument indexes, redundant argument lengths and needed hex
    // encoded argument string values. Really all that is needed is a comma
    // separated hex encoded argument value list, but we will stay true to the
    // documented version of the 'A' packet here...

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
                // Decode the argument index. We ignore this really becuase
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
                        if (packet.GetHexByteString(arg) != (arg_len / 2))
                            success = false;
                        else
                        {
                            // If there are any bytes lft
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
                            }
                        }
                    }
                }
            }
        }
    }

    if (success)
    {
        m_process_launch_info.GetFlags().Set (eLaunchFlagDebug);
        m_process_launch_error = Host::LaunchProcess (m_process_launch_info);
        if (m_process_launch_info.GetProcessID() != LLDB_INVALID_PROCESS_ID)
        {
            return SendOKResponse ();
        }
    }
    return SendErrorResponse (8);
}

bool
GDBRemoteCommunicationServer::Handle_qC (StringExtractorGDBRemote &packet)
{
    lldb::pid_t pid = m_process_launch_info.GetProcessID();
    StreamString response;
    response.Printf("QC%" PRIx64, pid);
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
GDBRemoteCommunicationServer::Handle_qLaunchGDBServer (StringExtractorGDBRemote &packet)
{
#ifdef _WIN32
    // No unix sockets on windows
    return false;
#else
    // Spawn a local debugserver as a platform so we can then attach or launch
    // a process...

    if (m_is_platform)
    {
        // Sleep and wait a bit for debugserver to start to listen...
        ConnectionFileDescriptor file_conn;
        char connect_url[PATH_MAX];
        Error error;
        std::string hostname;
        // TODO: /tmp/ should not be hardcoded. User might want to override /tmp
        // with the TMPDIR environnement variable
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
        lldb::thread_t accept_thread = LLDB_INVALID_HOST_THREAD;
        const char *unix_socket_name = NULL;
        char unix_socket_name_buf[PATH_MAX] = "/tmp/XXXXXXXXX";

        if (port == 0)
        {
            if (::mkstemp (unix_socket_name_buf) == 0)
            {
                unix_socket_name = unix_socket_name_buf;
                ::snprintf (connect_url, sizeof(connect_url), "unix-accept://%s", unix_socket_name);
                accept_thread = Host::ThreadCreate (unix_socket_name,
                                                    AcceptPortFromInferior,
                                                    connect_url,
                                                    &error);
            }
            else
            {
                error.SetErrorStringWithFormat("failed to make temporary path for a unix socket: %s", strerror(errno));
            }
        }

        if (error.Success())
        {
            // Spawn a debugserver and try to get the port it listens to.
            ProcessLaunchInfo debugserver_launch_info;
            StreamString host_and_port;
            if (hostname.empty())
                hostname = "localhost";
            host_and_port.Printf("%s:%u", hostname.c_str(), port);
            const char *host_and_port_cstr = host_and_port.GetString().c_str();
            Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));
            if (log)
                log->Printf("Launching debugserver with: %s...\n", host_and_port_cstr);

            debugserver_launch_info.SetMonitorProcessCallback(ReapDebugserverProcess, this, false);
            
            error = StartDebugserverProcess (host_and_port_cstr,
                                             unix_socket_name,
                                             debugserver_launch_info);

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
                bool success = false;

                if (IS_VALID_LLDB_HOST_THREAD(accept_thread))
                {
                    thread_result_t accept_thread_result = NULL;
                    if (Host::ThreadJoin (accept_thread, &accept_thread_result, &error))
                    {
                        if (accept_thread_result)
                        {
                            port = (intptr_t)accept_thread_result;
                            char response[256];
                            const int response_len = ::snprintf (response, sizeof(response), "pid:%" PRIu64 ";port:%u;", debugserver_pid, port + m_port_offset);
                            assert (response_len < sizeof(response));
                            //m_port_to_pid_map[port] = debugserver_launch_info.GetProcessID();
                            success = SendPacketNoLock (response, response_len) > 0;
                        }
                    }
                }
                else
                {
                    char response[256];
                    const int response_len = ::snprintf (response, sizeof(response), "pid:%" PRIu64 ";port:%u;", debugserver_pid, port + m_port_offset);
                    assert (response_len < sizeof(response));
                    //m_port_to_pid_map[port] = debugserver_launch_info.GetProcessID();
                    success = SendPacketNoLock (response, response_len) > 0;

                }
                Host::Unlink (unix_socket_name);

                if (!success)
                {
                    if (debugserver_pid != LLDB_INVALID_PROCESS_ID)
                        ::kill (debugserver_pid, SIGINT);
                }
                return success;
            }
            else if (accept_thread)
            {
                Host::Unlink (unix_socket_name);
            }
        }
    }
    return SendErrorResponse (9);
#endif
}

bool
GDBRemoteCommunicationServer::Handle_qKillSpawnedProcess (StringExtractorGDBRemote &packet)
{
    // Spawn a local debugserver as a platform so we can then attach or launch
    // a process...

    if (m_is_platform)
    {
        packet.SetFilePos(::strlen ("qKillSpawnedProcess:"));

        lldb::pid_t pid = packet.GetU64(LLDB_INVALID_PROCESS_ID);

        // Scope for locker
        {
            Mutex::Locker locker (m_spawned_pids_mutex);
            if (m_spawned_pids.find(pid) == m_spawned_pids.end())
                return SendErrorResponse (10);
        }
        Host::Kill (pid, SIGTERM);

        for (size_t i=0; i<10; ++i)
        {
            // Scope for locker
            {
                Mutex::Locker locker (m_spawned_pids_mutex);
                if (m_spawned_pids.find(pid) == m_spawned_pids.end())
                    return SendOKResponse();
            }
            usleep (10000);
        }

        // Scope for locker
        {
            Mutex::Locker locker (m_spawned_pids_mutex);
            if (m_spawned_pids.find(pid) == m_spawned_pids.end())
                return SendOKResponse();
        }
        Host::Kill (pid, SIGKILL);

        for (size_t i=0; i<10; ++i)
        {
            // Scope for locker
            {
                Mutex::Locker locker (m_spawned_pids_mutex);
                if (m_spawned_pids.find(pid) == m_spawned_pids.end())
                    return SendOKResponse();
            }
            usleep (10000);
        }
    }
    return SendErrorResponse (11);
}

bool
GDBRemoteCommunicationServer::Handle_qLaunchSuccess (StringExtractorGDBRemote &packet)
{
    if (m_process_launch_error.Success())
        return SendOKResponse();
    StreamString response;
    response.PutChar('E');
    response.PutCString(m_process_launch_error.AsCString("<unknown error>"));
    return SendPacketNoLock (response.GetData(), response.GetSize());
}

bool
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

bool
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

bool
GDBRemoteCommunicationServer::Handle_QSetDisableASLR (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QSetDisableASLR:"));
    if (packet.GetU32(0))
        m_process_launch_info.GetFlags().Set (eLaunchFlagDisableASLR);
    else
        m_process_launch_info.GetFlags().Clear (eLaunchFlagDisableASLR);
    return SendOKResponse ();
}

bool
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

bool
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
            SendPacketNoLock(response.GetData(), response.GetSize());
            return true;
        }
    }
    else
    {
        const char *working_dir = m_process_launch_info.GetWorkingDirectory();
        if (working_dir && working_dir[0])
        {
            response.PutBytesAsRawHex8(working_dir, strlen(working_dir));
            SendPacketNoLock(response.GetData(), response.GetSize());
            return true;
        }
        else
        {
            return SendErrorResponse(14);
        }
    }
}

bool
GDBRemoteCommunicationServer::Handle_QSetSTDIN (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QSetSTDIN:"));
    ProcessLaunchInfo::FileAction file_action;
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

bool
GDBRemoteCommunicationServer::Handle_QSetSTDOUT (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QSetSTDOUT:"));
    ProcessLaunchInfo::FileAction file_action;
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

bool
GDBRemoteCommunicationServer::Handle_QSetSTDERR (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen ("QSetSTDERR:"));
    ProcessLaunchInfo::FileAction file_action;
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

bool
GDBRemoteCommunicationServer::Handle_QStartNoAckMode (StringExtractorGDBRemote &packet)
{
    // Send response first before changing m_send_acks to we ack this packet
    SendOKResponse ();
    m_send_acks = false;
    return true;
}

bool
GDBRemoteCommunicationServer::Handle_qPlatform_mkdir (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("qPlatform_mkdir:"));
    mode_t mode = packet.GetHexMaxU32(false, UINT32_MAX);
    if (packet.GetChar() == ',')
    {
        std::string path;
        packet.GetHexByteString(path);
        Error error = Host::MakeDirectory(path.c_str(),mode);
        if (error.Success())
            return SendPacketNoLock ("OK", 2);
        else
            return SendErrorResponse(error.GetError());
    }
    return SendErrorResponse(20);
}

bool
GDBRemoteCommunicationServer::Handle_qPlatform_chmod (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("qPlatform_chmod:"));
    
    mode_t mode = packet.GetHexMaxU32(false, UINT32_MAX);
    if (packet.GetChar() == ',')
    {
        std::string path;
        packet.GetHexByteString(path);
        Error error = Host::SetFilePermissions (path.c_str(), mode);
        if (error.Success())
            return SendPacketNoLock ("OK", 2);
        else
            return SendErrorResponse(error.GetError());
    }
    return SendErrorResponse(19);
}

bool
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
                printf ("open('%s', flags=0x%x, mode=%o) fd = %i (%s)\n", path.c_str(), flags, mode, fd, fd == -1 ? strerror(errno) : "<success>");
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

bool
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

bool
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

bool
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

bool
GDBRemoteCommunicationServer::Handle_vFile_Size (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:size:"));
    std::string path;
    packet.GetHexByteString(path);
    if (!path.empty())
    {
        lldb::user_id_t retcode = Host::GetFileSize(FileSpec(path.c_str(), false));
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

bool
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

bool
GDBRemoteCommunicationServer::Handle_vFile_Exists (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:exists:"));
    std::string path;
    packet.GetHexByteString(path);
    if (!path.empty())
    {
        bool retcode = Host::GetFileExists(FileSpec(path.c_str(), false));
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

bool
GDBRemoteCommunicationServer::Handle_vFile_symlink (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:symlink:"));
    std::string dst, src;
    packet.GetHexByteStringTerminatedBy(dst, ',');
    packet.GetChar(); // Skip ',' char
    packet.GetHexByteString(src);
    Error error = Host::Symlink(src.c_str(), dst.c_str());
    StreamString response;
    response.Printf("F%u,%u", error.GetError(), error.GetError());
    return SendPacketNoLock(response.GetData(), response.GetSize());
}

bool
GDBRemoteCommunicationServer::Handle_vFile_unlink (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:unlink:"));
    std::string path;
    packet.GetHexByteString(path);
    Error error = Host::Unlink(path.c_str());
    StreamString response;
    response.Printf("F%u,%u", error.GetError(), error.GetError());
    return SendPacketNoLock(response.GetData(), response.GetSize());
}

bool
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

bool
GDBRemoteCommunicationServer::Handle_vFile_Stat (StringExtractorGDBRemote &packet)
{
    return SendUnimplementedResponse("GDBRemoteCommunicationServer::Handle_vFile_Stat() unimplemented");
}

bool
GDBRemoteCommunicationServer::Handle_vFile_MD5 (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(::strlen("vFile:MD5:"));
    std::string path;
    packet.GetHexByteString(path);
    if (!path.empty())
    {
        uint64_t a,b;
        StreamGDBRemote response;
        if (Host::CalculateMD5(FileSpec(path.c_str(),false),a,b) == false)
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

