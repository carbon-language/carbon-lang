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
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ArchSpec.h"

#include "GDBRemoteCommunication.h"

class GDBRemoteCommunicationClient : public GDBRemoteCommunication
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    GDBRemoteCommunicationClient();

    virtual
    ~GDBRemoteCommunicationClient();

    //------------------------------------------------------------------
    // After connecting, send the handshake to the server to make sure
    // we are communicating with it.
    //------------------------------------------------------------------
    bool
    HandshakeWithServer (lldb_private::Error *error_ptr);

    size_t
    SendPacketAndWaitForResponse (const char *send_payload,
                                  StringExtractorGDBRemote &response,
                                  bool send_async);

    size_t
    SendPacketAndWaitForResponse (const char *send_payload,
                                  size_t send_length,
                                  StringExtractorGDBRemote &response,
                                  bool send_async);

    lldb::StateType
    SendContinuePacketAndWaitForResponse (ProcessGDBRemote *process,
                                          const char *packet_payload,
                                          size_t packet_length,
                                          StringExtractorGDBRemote &response);

    virtual bool
    GetThreadSuffixSupported ();

    void
    QueryNoAckModeSupported ();

    bool
    SendAsyncSignal (int signo);

    bool
    SendInterrupt (lldb_private::Mutex::Locker &locker, 
                   uint32_t seconds_to_wait_for_stop, 
                   bool &sent_interrupt, 
                   bool &timed_out);

    lldb::pid_t
    GetCurrentProcessID ();

    bool
    GetLaunchSuccess (std::string &error_str);

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
    SendArgumentsPacket (char const *argv[]);

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
    /// be launched with the 'A' packet.
    ///
    /// @param[in] path
    ///     The path to a directory to use when launching our processs
    ///
    /// @return
    ///     Zero if the for success, or an error code for failure.
    //------------------------------------------------------------------
    int
    SetWorkingDir (char const *path);

    lldb::addr_t
    AllocateMemory (size_t size, uint32_t permissions);

    bool
    DeallocateMemory (lldb::addr_t addr);

    const lldb_private::ArchSpec &
    GetHostArchitecture ();
    
    bool
    GetVContSupported (char flavor);

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

    bool
    GetSupportsThreadSuffix ();

    bool
    GetProcessInfo (lldb::pid_t pid, 
                    lldb_private::ProcessInfo &process_info);

    uint32_t
    FindProcesses (const lldb_private::ProcessInfoMatch &process_match_info,
                   lldb_private::ProcessInfoList &process_infos);

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
    
    uint32_t 
    SetPacketTimeout (uint32_t packet_timeout)
    {
        const uint32_t old_packet_timeout = m_packet_timeout;
        m_packet_timeout = packet_timeout;
        return old_packet_timeout;
    }

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
protected:

    //------------------------------------------------------------------
    // Classes that inherit from GDBRemoteCommunicationClient can see and modify these
    //------------------------------------------------------------------
    lldb_private::LazyBool m_supports_not_sending_acks;
    lldb_private::LazyBool m_supports_thread_suffix;
    lldb_private::LazyBool m_supports_vCont_all;
    lldb_private::LazyBool m_supports_vCont_any;
    lldb_private::LazyBool m_supports_vCont_c;
    lldb_private::LazyBool m_supports_vCont_C;
    lldb_private::LazyBool m_supports_vCont_s;
    lldb_private::LazyBool m_supports_vCont_S;
    lldb_private::LazyBool m_qHostInfo_is_valid;
    bool m_supports_qProcessInfoPID;
    bool m_supports_qfProcessInfo;
    bool m_supports_qUserName;
    bool m_supports_qGroupName;

    // If we need to send a packet while the target is running, the m_async_XXX
    // member variables take care of making this happen.
    lldb_private::Mutex m_async_mutex;
    lldb_private::Predicate<bool> m_async_packet_predicate;
    std::string m_async_packet;
    StringExtractorGDBRemote m_async_response;
    int m_async_signal; // We were asked to deliver a signal to the inferior process.
    
    lldb_private::ArchSpec m_host_arch;
    uint32_t m_os_version_major;
    uint32_t m_os_version_minor;
    uint32_t m_os_version_update;
    std::string m_os_build;
    std::string m_os_kernel;
    std::string m_hostname;
    
    bool
    DecodeProcessInfoResponse (StringExtractorGDBRemote &response, 
                               lldb_private::ProcessInfo &process_info);
private:
    //------------------------------------------------------------------
    // For GDBRemoteCommunicationClient only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunicationClient);
};

#endif  // liblldb_GDBRemoteCommunicationClient_h_
