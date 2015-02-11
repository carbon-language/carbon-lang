//===-- GDBRemoteCommunicationServerCommon.h --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GDBRemoteCommunicationServerCommon_h_
#define liblldb_GDBRemoteCommunicationServerCommon_h_

// C Includes
// C++ Includes
#include <set>

// Other libraries and framework includes
#include "lldb/lldb-private-forward.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"

// Project includes
#include "GDBRemoteCommunicationServer.h"
#include "GDBRemoteCommunicationServerCommon.h"

class ProcessGDBRemote;
class StringExtractorGDBRemote;

class GDBRemoteCommunicationServerCommon :
    public GDBRemoteCommunicationServer
{
public:
    GDBRemoteCommunicationServerCommon(const char *comm_name, const char *listener_name);

    virtual
    ~GDBRemoteCommunicationServerCommon();

    virtual bool
    GetThreadSuffixSupported () override
    {
        return true;
    }

    //------------------------------------------------------------------
    /// Launch a process with the current launch settings.
    ///
    /// This method supports running an lldb-gdbserver or similar
    /// server in a situation where the startup code has been provided
    /// with all the information for a child process to be launched.
    ///
    /// @return
    ///     An Error object indicating the success or failure of the
    ///     launch.
    //------------------------------------------------------------------
    virtual lldb_private::Error
    LaunchProcess () = 0;

protected:
    std::set<lldb::pid_t> m_spawned_pids;
    lldb_private::Mutex m_spawned_pids_mutex;
    lldb_private::ProcessLaunchInfo m_process_launch_info;
    lldb_private::Error m_process_launch_error;
    lldb_private::ProcessInstanceInfoList m_proc_infos;
    uint32_t m_proc_infos_index;
    bool m_thread_suffix_supported;
    bool m_list_threads_in_stop_reply;

    PacketResult
    Handle_A (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qHostInfo (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qProcessInfoPID (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qfProcessInfo (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qsProcessInfo (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qUserName (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qGroupName (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qSpeedTest (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qKillSpawnedProcess (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vFile_Open (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vFile_Close (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vFile_pRead (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vFile_pWrite (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vFile_Size (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vFile_Mode (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vFile_Exists (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vFile_symlink (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vFile_unlink (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vFile_Stat (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vFile_MD5 (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qPlatform_shell (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qPlatform_mkdir (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qPlatform_chmod (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qSupported (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QThreadSuffixSupported (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QListThreadsInStopReply (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QSetDetachOnError (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QStartNoAckMode (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QSetSTDIN (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QSetSTDOUT (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QSetSTDERR (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qLaunchSuccess (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QEnvironment  (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QLaunchArch (StringExtractorGDBRemote &packet);

    bool
    KillSpawnedProcess (lldb::pid_t pid);

    static void
    CreateProcessInfoResponse (const lldb_private::ProcessInstanceInfo &proc_info,
                               lldb_private::StreamString &response);

    static void
    CreateProcessInfoResponse_DebugServerStyle (const lldb_private::ProcessInstanceInfo &proc_info,
                                                lldb_private::StreamString &response);

    template <typename T>
    using MemberFunctionPacketHandler = PacketResult (T::*) (StringExtractorGDBRemote& packet);

    template <typename T>
    void
    RegisterMemberFunctionHandler(StringExtractorGDBRemote::ServerPacketType packet_type,
                                  MemberFunctionPacketHandler<T> handler)
    {
        RegisterPacketHandler(packet_type,
                              [this, handler] (StringExtractorGDBRemote packet,
                                               lldb_private::Error &error,
                                               bool &interrupt,
                                               bool &quit)
                              {
                                  return (static_cast<T*>(this)->*handler) (packet);
                              });
    }
};

#endif  // liblldb_GDBRemoteCommunicationServerCommon_h_
