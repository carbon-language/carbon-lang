//===-- GDBRemoteCommunicationServer.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GDBRemoteCommunicationServer_h_
#define liblldb_GDBRemoteCommunicationServer_h_

// C Includes
// C++ Includes
#include <vector>
#include <set>
#include <unordered_map>
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private-forward.h"
#include "lldb/Core/Communication.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"
#include "GDBRemoteCommunication.h"

#include "../../../Host/common/NativeProcessProtocol.h"

class ProcessGDBRemote;
class StringExtractorGDBRemote;

class GDBRemoteCommunicationServer :
    public GDBRemoteCommunication,
    public lldb_private::NativeProcessProtocol::NativeDelegate
{
public:
    typedef std::map<uint16_t, lldb::pid_t> PortMap;

    enum
    {
        eBroadcastBitRunPacketSent = kLoUserBroadcastBit
    };
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    GDBRemoteCommunicationServer(bool is_platform);

    GDBRemoteCommunicationServer(bool is_platform,
                                 const lldb::PlatformSP& platform_sp,
                                 lldb::DebuggerSP& debugger_sp);

    virtual
    ~GDBRemoteCommunicationServer();

    PacketResult
    GetPacketAndSendResponse (uint32_t timeout_usec,
                              lldb_private::Error &error,
                              bool &interrupt, 
                              bool &quit);

    virtual bool
    GetThreadSuffixSupported ()
    {
        return true;
    }

    // After connecting, do a little handshake with the client to make sure
    // we are at least communicating
    bool
    HandshakeWithClient (lldb_private::Error *error_ptr);

    // Set both ports to zero to let the platform automatically bind to 
    // a port chosen by the OS.
    void
    SetPortMap (PortMap &&port_map)
    {
        m_port_map = port_map;
    }

    //----------------------------------------------------------------------
    // If we are using a port map where we can only use certain ports,
    // get the next available port.
    //
    // If we are using a port map and we are out of ports, return UINT16_MAX
    //
    // If we aren't using a port map, return 0 to indicate we should bind to
    // port 0 and then figure out which port we used.
    //----------------------------------------------------------------------
    uint16_t
    GetNextAvailablePort ()
    {
        if (m_port_map.empty())
            return 0; // Bind to port zero and get a port, we didn't have any limitations
        
        for (auto &pair : m_port_map)
        {
            if (pair.second == LLDB_INVALID_PROCESS_ID)
            {
                pair.second = ~(lldb::pid_t)LLDB_INVALID_PROCESS_ID;
                return pair.first;
            }
        }
        return UINT16_MAX;
    }

    bool
    AssociatePortWithProcess (uint16_t port, lldb::pid_t pid)
    {
        PortMap::iterator pos = m_port_map.find(port);
        if (pos != m_port_map.end())
        {
            pos->second = pid;
            return true;
        }
        return false;
    }

    bool
    FreePort (uint16_t port)
    {
        PortMap::iterator pos = m_port_map.find(port);
        if (pos != m_port_map.end())
        {
            pos->second = LLDB_INVALID_PROCESS_ID;
            return true;
        }
        return false;
    }

    bool
    FreePortForProcess (lldb::pid_t pid)
    {
        if (!m_port_map.empty())
        {
            for (auto &pair : m_port_map)
            {
                if (pair.second == pid)
                {
                    pair.second = LLDB_INVALID_PROCESS_ID;
                    return true;
                }
            }
        }
        return false;
    }

    void
    SetPortOffset (uint16_t port_offset)
    {
        m_port_offset = port_offset;
    }

    //------------------------------------------------------------------
    /// Specify the program to launch and its arguments.
    ///
    /// @param[in] args
    ///     The command line to launch.
    ///
    /// @param[in] argc
    ///     The number of elements in the args array of cstring pointers.
    ///
    /// @return
    ///     An Error object indicating the success or failure of making
    ///     the setting.
    //------------------------------------------------------------------
    lldb_private::Error
    SetLaunchArguments (const char *const args[], int argc);

    //------------------------------------------------------------------
    /// Specify the launch flags for the process.
    ///
    /// @param[in] launch_flags
    ///     The launch flags to use when launching this process.
    ///
    /// @return
    ///     An Error object indicating the success or failure of making
    ///     the setting.
    //------------------------------------------------------------------
    lldb_private::Error
    SetLaunchFlags (unsigned int launch_flags);

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
    lldb_private::Error
    LaunchProcess ();

    //------------------------------------------------------------------
    /// Attach to a process.
    ///
    /// This method supports attaching llgs to a process accessible via the
    /// configured Platform.
    ///
    /// @return
    ///     An Error object indicating the success or failure of the
    ///     attach operation.
    //------------------------------------------------------------------
    lldb_private::Error
    AttachToProcess (lldb::pid_t pid);

    //------------------------------------------------------------------
    // NativeProcessProtocol::NativeDelegate overrides
    //------------------------------------------------------------------
    void
    InitializeDelegate (lldb_private::NativeProcessProtocol *process) override;

    void
    ProcessStateChanged (lldb_private::NativeProcessProtocol *process, lldb::StateType state) override;

    void
    DidExec (lldb_private::NativeProcessProtocol *process) override;

protected:
    lldb::PlatformSP m_platform_sp;
    lldb::thread_t m_async_thread;
    lldb_private::ProcessLaunchInfo m_process_launch_info;
    lldb_private::Error m_process_launch_error;
    std::set<lldb::pid_t> m_spawned_pids;
    lldb_private::Mutex m_spawned_pids_mutex;
    lldb_private::ProcessInstanceInfoList m_proc_infos;
    uint32_t m_proc_infos_index;
    PortMap m_port_map;
    uint16_t m_port_offset;
    lldb::tid_t m_current_tid;
    lldb::tid_t m_continue_tid;
    lldb_private::Mutex m_debugged_process_mutex;
    lldb_private::NativeProcessProtocolSP m_debugged_process_sp;
    lldb::DebuggerSP m_debugger_sp;
    Communication m_stdio_communication;
    bool m_exit_now; // use in asynchronous handling to indicate process should exit.
    lldb::StateType m_inferior_prev_state;
    bool m_thread_suffix_supported;
    bool m_list_threads_in_stop_reply;
    lldb::DataBufferSP m_active_auxv_buffer_sp;
    lldb_private::Mutex m_saved_registers_mutex;
    std::unordered_map<uint32_t, lldb::DataBufferSP> m_saved_registers_map;
    uint32_t m_next_saved_registers_id;

    PacketResult
    SendUnimplementedResponse (const char *packet);

    PacketResult
    SendErrorResponse (uint8_t error);

    PacketResult
    SendIllFormedResponse (const StringExtractorGDBRemote &packet, const char *error_message);

    PacketResult
    SendOKResponse ();

    PacketResult
    SendONotification (const char *buffer, uint32_t len);

    PacketResult
    SendWResponse (lldb_private::NativeProcessProtocol *process);

    PacketResult
    SendStopReplyPacketForThread (lldb::tid_t tid);

    PacketResult
    SendStopReasonForState (lldb::StateType process_state, bool flush_on_exit);

    PacketResult
    Handle_A (StringExtractorGDBRemote &packet);
    
    PacketResult
    Handle_qLaunchSuccess (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qHostInfo (StringExtractorGDBRemote &packet);
    
    PacketResult
    Handle_qLaunchGDBServer (StringExtractorGDBRemote &packet);
    
    PacketResult
    Handle_qKillSpawnedProcess (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_k (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qPlatform_mkdir (StringExtractorGDBRemote &packet);
    
    PacketResult
    Handle_qPlatform_chmod (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qProcessInfo (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qProcessInfoPID (StringExtractorGDBRemote &packet);
    
    PacketResult
    Handle_qfProcessInfo (StringExtractorGDBRemote &packet);
    
    PacketResult
    Handle_qsProcessInfo (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qC (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qUserName (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qGroupName (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qSpeedTest (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QEnvironment  (StringExtractorGDBRemote &packet);
    
    PacketResult
    Handle_QLaunchArch (StringExtractorGDBRemote &packet);
    
    PacketResult
    Handle_QSetDisableASLR (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QSetDetachOnError (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QSetWorkingDir (StringExtractorGDBRemote &packet);
    
    PacketResult
    Handle_qGetWorkingDir (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QStartNoAckMode (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QSetSTDIN (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QSetSTDOUT (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QSetSTDERR (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_C (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_c (StringExtractorGDBRemote &packet, bool skip_file_pos_adjustment = false);

    PacketResult
    Handle_vCont (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vCont_actions (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_stop_reason (StringExtractorGDBRemote &packet);

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
    Handle_qRegisterInfo (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qfThreadInfo (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qsThreadInfo (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_p (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_P (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_H (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_interrupt (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_m (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_M (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qMemoryRegionInfoSupported (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qMemoryRegionInfo (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_Z (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_z (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_s (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qSupported (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QThreadSuffixSupported (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QListThreadsInStopReply (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qXfer_auxv_read (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QSaveRegisterState (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_QRestoreRegisterState (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_vAttach (StringExtractorGDBRemote &packet);

    PacketResult
    Handle_qThreadStopInfo (StringExtractorGDBRemote &packet);

    void
    SetCurrentThreadID (lldb::tid_t tid);

    lldb::tid_t
    GetCurrentThreadID () const;

    void
    SetContinueThreadID (lldb::tid_t tid);

    lldb::tid_t
    GetContinueThreadID () const { return m_continue_tid; }

    lldb_private::Error
    SetSTDIOFileDescriptor (int fd);

    static void
    STDIOReadThreadBytesReceived (void *baton, const void *src, size_t src_len);

private:
    bool
    DebugserverProcessReaped (lldb::pid_t pid);
    
    static bool
    ReapDebugserverProcess (void *callback_baton,
                            lldb::pid_t pid,
                            bool exited,
                            int signal,
                            int status);

    bool
    DebuggedProcessReaped (lldb::pid_t pid);

    static bool
    ReapDebuggedProcess (void *callback_baton,
                         lldb::pid_t pid,
                         bool exited,
                         int signal,
                         int status);

    bool
    KillSpawnedProcess (lldb::pid_t pid);

    bool
    IsGdbServer ()
    {
        return !m_is_platform;
    }

    /// Launch an inferior process from lldb-gdbserver
    lldb_private::Error
    LaunchProcessForDebugging ();

    /// Launch a process from lldb-platform
    lldb_private::Error
    LaunchPlatformProcess ();

    void
    HandleInferiorState_Exited (lldb_private::NativeProcessProtocol *process);

    void
    HandleInferiorState_Stopped (lldb_private::NativeProcessProtocol *process);

    void
    FlushInferiorOutput ();

    lldb_private::NativeThreadProtocolSP
    GetThreadFromSuffix (StringExtractorGDBRemote &packet);

    uint32_t
    GetNextSavedRegistersID ();

    void
    MaybeCloseInferiorTerminalConnection ();

    void
    ClearProcessSpecificData ();

    bool
    ShouldRedirectInferiorOutputOverGdbRemote (const lldb_private::ProcessLaunchInfo &launch_info) const;

    //------------------------------------------------------------------
    // For GDBRemoteCommunicationServer only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunicationServer);
};

#endif  // liblldb_GDBRemoteCommunicationServer_h_
