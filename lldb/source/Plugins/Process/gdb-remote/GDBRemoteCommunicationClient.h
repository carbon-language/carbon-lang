//===-- GDBRemoteCommunicationClient.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GDBRemoteCommunicationClient_h_
#define liblldb_GDBRemoteCommunicationClient_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Target/Process.h"

#include "GDBRemoteCommunication.h"

typedef enum 
{
    eBreakpointSoftware = 0,
    eBreakpointHardware,
    eWatchpointWrite,
    eWatchpointRead,
    eWatchpointReadWrite
} GDBStoppointType;

class GDBRemoteCommunicationClient : public GDBRemoteCommunication
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    GDBRemoteCommunicationClient(bool is_platform);

    ~GDBRemoteCommunicationClient();

    //------------------------------------------------------------------
    // After connecting, send the handshake to the server to make sure
    // we are communicating with it.
    //------------------------------------------------------------------
    bool
    HandshakeWithServer (lldb_private::Error *error_ptr);

    PacketResult
    SendPacketAndWaitForResponse (const char *send_payload,
                                  StringExtractorGDBRemote &response,
                                  bool send_async);

    PacketResult
    SendPacketAndWaitForResponse (const char *send_payload,
                                  size_t send_length,
                                  StringExtractorGDBRemote &response,
                                  bool send_async);

    // For packets which specify a range of output to be returned,
    // return all of the output via a series of request packets of the form
    // <prefix>0,<size>
    // <prefix><size>,<size>
    // <prefix><size>*2,<size>
    // <prefix><size>*3,<size>
    // ...
    // until a "$l..." packet is received, indicating the end.
    // (size is in hex; this format is used by a standard gdbserver to
    // return the given portion of the output specified by <prefix>;
    // for example, "qXfer:libraries-svr4:read::fff,1000" means
    // "return a chunk of the xml description file for shared
    // library load addresses, where the chunk starts at offset 0xfff
    // and continues for 0x1000 bytes").
    // Concatenate the resulting server response packets together and
    // return in response_string.  If any packet fails, the return value
    // indicates that failure and the returned string value is undefined.
    PacketResult
    SendPacketsAndConcatenateResponses (const char *send_payload_prefix,
                                        std::string &response_string);

    lldb::StateType
    SendContinuePacketAndWaitForResponse (ProcessGDBRemote *process,
                                          const char *packet_payload,
                                          size_t packet_length,
                                          StringExtractorGDBRemote &response);

    bool
    GetThreadSuffixSupported ();

    // This packet is usually sent first and the boolean return value
    // indicates if the packet was send and any response was received
    // even in the response is UNIMPLEMENTED. If the packet failed to
    // get a response, then false is returned. This quickly tells us
    // if we were able to connect and communicte with the remote GDB
    // server
    bool
    QueryNoAckModeSupported ();

    void
    GetListThreadsInStopReplySupported ();

    bool
    SendAsyncSignal (int signo);

    bool
    SendInterrupt (lldb_private::Mutex::Locker &locker, 
                   uint32_t seconds_to_wait_for_stop, 
                   bool &timed_out);

    lldb::pid_t
    GetCurrentProcessID ();

    bool
    GetLaunchSuccess (std::string &error_str);

    uint16_t
    LaunchGDBserverAndGetPort (lldb::pid_t &pid, const char *remote_accept_hostname);
    
    bool
    KillSpawnedProcess (lldb::pid_t pid);

    //------------------------------------------------------------------
    /// Sends a GDB remote protocol 'A' packet that delivers program
    /// arguments to the remote server.
    ///
    /// @param[in] argv
    ///     A NULL terminated array of const C strings to use as the
    ///     arguments.
    ///
    /// @return
    ///     Zero if the response was "OK", a positive value if the
    ///     the response was "Exx" where xx are two hex digits, or
    ///     -1 if the call is unsupported or any other unexpected
    ///     response was received.
    //------------------------------------------------------------------
    int
    SendArgumentsPacket (const lldb_private::ProcessLaunchInfo &launch_info);

    //------------------------------------------------------------------
    /// Sends a "QEnvironment:NAME=VALUE" packet that will build up the
    /// environment that will get used when launching an application
    /// in conjunction with the 'A' packet. This function can be called
    /// multiple times in a row in order to pass on the desired
    /// environment that the inferior should be launched with.
    ///
    /// @param[in] name_equal_value
    ///     A NULL terminated C string that contains a single environment
    ///     in the format "NAME=VALUE".
    ///
    /// @return
    ///     Zero if the response was "OK", a positive value if the
    ///     the response was "Exx" where xx are two hex digits, or
    ///     -1 if the call is unsupported or any other unexpected
    ///     response was received.
    //------------------------------------------------------------------
    int
    SendEnvironmentPacket (char const *name_equal_value);

    int
    SendLaunchArchPacket (const char *arch);
    //------------------------------------------------------------------
    /// Sends a "vAttach:PID" where PID is in hex. 
    ///
    /// @param[in] pid
    ///     A process ID for the remote gdb server to attach to.
    ///
    /// @param[out] response
    ///     The response received from the gdb server. If the return
    ///     value is zero, \a response will contain a stop reply 
    ///     packet.
    ///
    /// @return
    ///     Zero if the attach was successful, or an error indicating
    ///     an error code.
    //------------------------------------------------------------------
    int
    SendAttach (lldb::pid_t pid, 
                StringExtractorGDBRemote& response);


    //------------------------------------------------------------------
    /// Sets the path to use for stdin/out/err for a process
    /// that will be launched with the 'A' packet.
    ///
    /// @param[in] path
    ///     The path to use for stdin/out/err
    ///
    /// @return
    ///     Zero if the for success, or an error code for failure.
    //------------------------------------------------------------------
    int
    SetSTDIN (char const *path);
    int
    SetSTDOUT (char const *path);
    int
    SetSTDERR (char const *path);

    //------------------------------------------------------------------
    /// Sets the disable ASLR flag to \a enable for a process that will 
    /// be launched with the 'A' packet.
    ///
    /// @param[in] enable
    ///     A boolean value indicating wether to disable ASLR or not.
    ///
    /// @return
    ///     Zero if the for success, or an error code for failure.
    //------------------------------------------------------------------
    int
    SetDisableASLR (bool enable);

    //------------------------------------------------------------------
    /// Sets the working directory to \a path for a process that will 
    /// be launched with the 'A' packet for non platform based
    /// connections. If this packet is sent to a GDB server that
    /// implements the platform, it will change the current working
    /// directory for the platform process.
    ///
    /// @param[in] path
    ///     The path to a directory to use when launching our processs
    ///
    /// @return
    ///     Zero if the for success, or an error code for failure.
    //------------------------------------------------------------------
    int
    SetWorkingDir (char const *path);

    //------------------------------------------------------------------
    /// Gets the current working directory of a remote platform GDB
    /// server.
    ///
    /// @param[out] cwd
    ///     The current working directory on the remote platform.
    ///
    /// @return
    ///     Boolean for success
    //------------------------------------------------------------------
    bool
    GetWorkingDir (std::string &cwd);

    lldb::addr_t
    AllocateMemory (size_t size, uint32_t permissions);

    bool
    DeallocateMemory (lldb::addr_t addr);

    lldb_private::Error
    Detach (bool keep_stopped);

    lldb_private::Error
    GetMemoryRegionInfo (lldb::addr_t addr, 
                        lldb_private::MemoryRegionInfo &range_info); 

    lldb_private::Error
    GetWatchpointSupportInfo (uint32_t &num); 

    lldb_private::Error
    GetWatchpointSupportInfo (uint32_t &num, bool& after);
    
    lldb_private::Error
    GetWatchpointsTriggerAfterInstruction (bool &after);

    const lldb_private::ArchSpec &
    GetHostArchitecture ();
    
    uint32_t
    GetHostDefaultPacketTimeout();

    const lldb_private::ArchSpec &
    GetProcessArchitecture ();

    void
    GetRemoteQSupported();

    bool
    GetVContSupported (char flavor);

    bool
    GetpPacketSupported (lldb::tid_t tid);

    bool
    GetVAttachOrWaitSupported ();
    
    bool
    GetSyncThreadStateSupported();
    
    void
    ResetDiscoverableSettings();

    bool
    GetHostInfo (bool force = false);
    
    bool
    GetOSVersion (uint32_t &major, 
                  uint32_t &minor, 
                  uint32_t &update);

    bool
    GetOSBuildString (std::string &s);
    
    bool
    GetOSKernelDescription (std::string &s);

    lldb_private::ArchSpec
    GetSystemArchitecture ();

    bool
    GetHostname (std::string &s);

    lldb::addr_t
    GetShlibInfoAddr();

    bool
    GetSupportsThreadSuffix ();

    bool
    GetProcessInfo (lldb::pid_t pid, 
                    lldb_private::ProcessInstanceInfo &process_info);

    uint32_t
    FindProcesses (const lldb_private::ProcessInstanceInfoMatch &process_match_info,
                   lldb_private::ProcessInstanceInfoList &process_infos);

    bool
    GetUserName (uint32_t uid, std::string &name);
    
    bool
    GetGroupName (uint32_t gid, std::string &name);

    bool
    HasFullVContSupport ()
    {
        return GetVContSupported ('A');
    }

    bool
    HasAnyVContSupport ()
    {
        return GetVContSupported ('a');
    }
    
    bool
    GetStopReply (StringExtractorGDBRemote &response);

    bool
    GetThreadStopInfo (lldb::tid_t tid, 
                       StringExtractorGDBRemote &response);

    bool
    SupportsGDBStoppointPacket (GDBStoppointType type)
    {
        switch (type)
        {
        case eBreakpointSoftware:   return m_supports_z0;
        case eBreakpointHardware:   return m_supports_z1;
        case eWatchpointWrite:      return m_supports_z2;
        case eWatchpointRead:       return m_supports_z3;
        case eWatchpointReadWrite:  return m_supports_z4;
        }
        return false;
    }
    uint8_t
    SendGDBStoppointTypePacket (GDBStoppointType type,   // Type of breakpoint or watchpoint
                                bool insert,              // Insert or remove?
                                lldb::addr_t addr,        // Address of breakpoint or watchpoint
                                uint32_t length);         // Byte Size of breakpoint or watchpoint

    void
    TestPacketSpeed (const uint32_t num_packets);

    // This packet is for testing the speed of the interface only. Both
    // the client and server need to support it, but this allows us to
    // measure the packet speed without any other work being done on the
    // other end and avoids any of that work affecting the packet send
    // and response times.
    bool
    SendSpeedTestPacket (uint32_t send_size, 
                         uint32_t recv_size);
    
    bool
    SetCurrentThread (uint64_t tid);
    
    bool
    SetCurrentThreadForRun (uint64_t tid);

    bool
    GetQXferLibrariesReadSupported ();

    bool
    GetQXferLibrariesSVR4ReadSupported ();

    uint64_t
    GetRemoteMaxPacketSize();

    bool
    GetAugmentedLibrariesSVR4ReadSupported ();

    lldb_private::LazyBool
    SupportsAllocDeallocMemory () // const
    {
        // Uncomment this to have lldb pretend the debug server doesn't respond to alloc/dealloc memory packets.
        // m_supports_alloc_dealloc_memory = lldb_private::eLazyBoolNo;
        return m_supports_alloc_dealloc_memory;
    }

    size_t
    GetCurrentThreadIDs (std::vector<lldb::tid_t> &thread_ids,
                         bool &sequence_mutex_unavailable);
    
    bool
    GetInterruptWasSent () const
    {
        return m_interrupt_sent;
    }
    
    lldb::user_id_t
    OpenFile (const lldb_private::FileSpec& file_spec,
              uint32_t flags,
              mode_t mode,
              lldb_private::Error &error);
    
    bool
    CloseFile (lldb::user_id_t fd,
               lldb_private::Error &error);
    
    lldb::user_id_t
    GetFileSize (const lldb_private::FileSpec& file_spec);
    
    lldb_private::Error
    GetFilePermissions(const char *path, uint32_t &file_permissions);

    lldb_private::Error
    SetFilePermissions(const char *path, uint32_t file_permissions);

    uint64_t
    ReadFile (lldb::user_id_t fd,
              uint64_t offset,
              void *dst,
              uint64_t dst_len,
              lldb_private::Error &error);
    
    uint64_t
    WriteFile (lldb::user_id_t fd,
               uint64_t offset,
               const void* src,
               uint64_t src_len,
               lldb_private::Error &error);
    
    lldb_private::Error
    CreateSymlink (const char *src,
                   const char *dst);
    
    lldb_private::Error
    Unlink (const char *path);

    lldb_private::Error
    MakeDirectory (const char *path,
                   uint32_t mode);
    
    bool
    GetFileExists (const lldb_private::FileSpec& file_spec);
    
    lldb_private::Error
    RunShellCommand (const char *command,           // Shouldn't be NULL
                     const char *working_dir,       // Pass NULL to use the current working directory
                     int *status_ptr,               // Pass NULL if you don't want the process exit status
                     int *signo_ptr,                // Pass NULL if you don't want the signal that caused the process to exit
                     std::string *command_output,   // Pass NULL if you don't want the command output
                     uint32_t timeout_sec);         // Timeout in seconds to wait for shell program to finish
    
    bool
    CalculateMD5 (const lldb_private::FileSpec& file_spec,
                  uint64_t &high,
                  uint64_t &low);
    
    std::string
    HarmonizeThreadIdsForProfileData (ProcessGDBRemote *process,
                                      StringExtractorGDBRemote &inputStringExtractor);

    bool
    ReadRegister(lldb::tid_t tid,
                 uint32_t reg_num,
                 StringExtractorGDBRemote &response);

    bool
    ReadAllRegisters (lldb::tid_t tid,
                      StringExtractorGDBRemote &response);

    bool
    SaveRegisterState (lldb::tid_t tid, uint32_t &save_id);
    
    bool
    RestoreRegisterState (lldb::tid_t tid, uint32_t save_id);
    
protected:

    PacketResult
    SendPacketAndWaitForResponseNoLock (const char *payload,
                                        size_t payload_length,
                                        StringExtractorGDBRemote &response);

    bool
    GetCurrentProcessInfo ();

    //------------------------------------------------------------------
    // Classes that inherit from GDBRemoteCommunicationClient can see and modify these
    //------------------------------------------------------------------
    lldb_private::LazyBool m_supports_not_sending_acks;
    lldb_private::LazyBool m_supports_thread_suffix;
    lldb_private::LazyBool m_supports_threads_in_stop_reply;
    lldb_private::LazyBool m_supports_vCont_all;
    lldb_private::LazyBool m_supports_vCont_any;
    lldb_private::LazyBool m_supports_vCont_c;
    lldb_private::LazyBool m_supports_vCont_C;
    lldb_private::LazyBool m_supports_vCont_s;
    lldb_private::LazyBool m_supports_vCont_S;
    lldb_private::LazyBool m_qHostInfo_is_valid;
    lldb_private::LazyBool m_qProcessInfo_is_valid;
    lldb_private::LazyBool m_supports_alloc_dealloc_memory;
    lldb_private::LazyBool m_supports_memory_region_info;
    lldb_private::LazyBool m_supports_watchpoint_support_info;
    lldb_private::LazyBool m_supports_detach_stay_stopped;
    lldb_private::LazyBool m_watchpoints_trigger_after_instruction;
    lldb_private::LazyBool m_attach_or_wait_reply;
    lldb_private::LazyBool m_prepare_for_reg_writing_reply;
    lldb_private::LazyBool m_supports_p;
    lldb_private::LazyBool m_supports_QSaveRegisterState;
    lldb_private::LazyBool m_supports_qXfer_libraries_read;
    lldb_private::LazyBool m_supports_qXfer_libraries_svr4_read;
    lldb_private::LazyBool m_supports_augmented_libraries_svr4_read;

    bool
        m_supports_qProcessInfoPID:1,
        m_supports_qfProcessInfo:1,
        m_supports_qUserName:1,
        m_supports_qGroupName:1,
        m_supports_qThreadStopInfo:1,
        m_supports_z0:1,
        m_supports_z1:1,
        m_supports_z2:1,
        m_supports_z3:1,
        m_supports_z4:1,
        m_supports_QEnvironment:1,
        m_supports_QEnvironmentHexEncoded:1;
    

    lldb::tid_t m_curr_tid;         // Current gdb remote protocol thread index for all other operations
    lldb::tid_t m_curr_tid_run;     // Current gdb remote protocol thread index for continue, step, etc


    uint32_t m_num_supported_hardware_watchpoints;

    // If we need to send a packet while the target is running, the m_async_XXX
    // member variables take care of making this happen.
    lldb_private::Mutex m_async_mutex;
    lldb_private::Predicate<bool> m_async_packet_predicate;
    std::string m_async_packet;
    PacketResult m_async_result;
    StringExtractorGDBRemote m_async_response;
    int m_async_signal; // We were asked to deliver a signal to the inferior process.
    bool m_interrupt_sent;
    std::string m_partial_profile_data;
    std::map<uint64_t, uint32_t> m_thread_id_to_used_usec_map;
    
    lldb_private::ArchSpec m_host_arch;
    lldb_private::ArchSpec m_process_arch;
    uint32_t m_os_version_major;
    uint32_t m_os_version_minor;
    uint32_t m_os_version_update;
    std::string m_os_build;
    std::string m_os_kernel;
    std::string m_hostname;
    uint32_t m_default_packet_timeout;
    uint64_t m_max_packet_size;  // as returned by qSupported
    
    bool
    DecodeProcessInfoResponse (StringExtractorGDBRemote &response, 
                               lldb_private::ProcessInstanceInfo &process_info);
private:
    //------------------------------------------------------------------
    // For GDBRemoteCommunicationClient only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunicationClient);
};

#endif  // liblldb_GDBRemoteCommunicationClient_h_
