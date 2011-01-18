//===-- GDBRemoteCommunication.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GDBRemoteCommunication_h_
#define liblldb_GDBRemoteCommunication_h_

// C Includes
// C++ Includes
#include <list>
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Communication.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Listener.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Host/Predicate.h"

#include "Utility/StringExtractorGDBRemote.h"

class ProcessGDBRemote;

class GDBRemoteCommunication :
    public lldb_private::Communication
{
public:
    enum
    {
        eBroadcastBitRunPacketSent = kLoUserBroadcastBit
    };
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    GDBRemoteCommunication();

    virtual
    ~GDBRemoteCommunication();

    size_t
    SendPacket (const char *payload);

    size_t
    SendPacket (const char *payload,
                size_t payload_length);

    size_t
    SendPacketAndWaitForResponse (const char *send_payload,
                                  StringExtractorGDBRemote &response,
                                  uint32_t timeout_seconds,
                                  bool send_async);

    size_t
    SendPacketAndWaitForResponse (const char *send_payload,
                                  size_t send_length,
                                  StringExtractorGDBRemote &response,
                                  uint32_t timeout_seconds,
                                  bool send_async);

    lldb::StateType
    SendContinuePacketAndWaitForResponse (ProcessGDBRemote *process,
                                          const char *packet_payload,
                                          size_t packet_length,
                                          StringExtractorGDBRemote &response);

    // Wait for a packet within 'nsec' seconds
    size_t
    WaitForPacket (StringExtractorGDBRemote &response,
                   uint32_t nsec);

    // Wait for a packet with an absolute timeout time. If 'timeout' is NULL
    // wait indefinitely.
    size_t
    WaitForPacket (StringExtractorGDBRemote &response,
                   lldb_private::TimeValue* timeout);

    char
    GetAck (uint32_t timeout_seconds);

    size_t
    SendAck (char ack_char);

    char
    CalculcateChecksum (const char *payload,
                        size_t payload_length);

    void
    SetAckMode (bool enabled)
    {
        m_send_acks = enabled;
    }

    bool
    GetThreadSuffixSupported () const
    {
        return m_thread_suffix_supported;
    }
    
    void
    SetThreadSuffixSupported (bool enabled)
    {
        m_thread_suffix_supported = enabled;
    }

    bool
    SendAsyncSignal (int signo);

    bool
    SendInterrupt (lldb_private::Mutex::Locker &locker, 
                   uint32_t seconds_to_wait_for_stop, 
                   bool *timed_out = NULL);

    bool
    GetSequenceMutex(lldb_private::Mutex::Locker& locker);

    //------------------------------------------------------------------
    // Communication overrides
    //------------------------------------------------------------------
    virtual void
    AppendBytesToCache (const uint8_t *src, size_t src_len, bool broadcast, lldb::ConnectionStatus status);


    lldb::pid_t
    GetCurrentProcessID (uint32_t timeout_seconds);

    bool
    GetLaunchSuccess (uint32_t timeout_seconds, std::string &error_str);

    //------------------------------------------------------------------
    /// Sends a GDB remote protocol 'A' packet that delivers program
    /// arguments to the remote server.
    ///
    /// @param[in] argv
    ///     A NULL terminated array of const C strings to use as the
    ///     arguments.
    ///
    /// @param[in] timeout_seconds
    ///     The number of seconds to wait for a response from the remote
    ///     server.
    ///
    /// @return
    ///     Zero if the response was "OK", a positive value if the
    ///     the response was "Exx" where xx are two hex digits, or
    ///     -1 if the call is unsupported or any other unexpected
    ///     response was received.
    //------------------------------------------------------------------
    int
    SendArgumentsPacket (char const *argv[], uint32_t timeout_seconds);

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
    /// @param[in] timeout_seconds
    ///     The number of seconds to wait for a response from the remote
    ///     server.
    ///
    /// @return
    ///     Zero if the response was "OK", a positive value if the
    ///     the response was "Exx" where xx are two hex digits, or
    ///     -1 if the call is unsupported or any other unexpected
    ///     response was received.
    //------------------------------------------------------------------
    int
    SendEnvironmentPacket (char const *name_equal_value,
                           uint32_t timeout_seconds);

    //------------------------------------------------------------------
    /// Sends a "vAttach:PID" where PID is in hex. 
    ///
    /// @param[in] pid
    ///     A process ID for the remote gdb server to attach to.
    ///
    /// @param[in] timeout_seconds
    ///     The number of seconds to wait for a response from the remote
    ///     server.
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
                uint32_t timeout_seconds, 
                StringExtractorGDBRemote& response);


    lldb::addr_t
    AllocateMemory (size_t size, uint32_t permissions, uint32_t timeout_seconds);

    bool
    DeallocateMemory (lldb::addr_t addr, uint32_t timeout_seconds);

    bool
    IsRunning() const
    {
        return m_is_running.GetValue();
    }
    
    bool
    GetHostInfo (uint32_t timeout_seconds);

    bool 
    HostInfoIsValid () const
    {
        return m_pointer_byte_size != 0;
    }

    const lldb_private::ArchSpec &
    GetHostArchitecture ();
    
    const lldb_private::ConstString &
    GetOSString ();
    
    const lldb_private::ConstString &
    GetVendorString();
    
    lldb::ByteOrder
    GetByteOrder ();

    uint32_t
    GetAddressByteSize ();

protected:
    typedef std::list<std::string> packet_collection;

    size_t
    SendPacketNoLock (const char *payload, 
                      size_t payload_length);

    size_t
    WaitForPacketNoLock (StringExtractorGDBRemote &response, 
                         lldb_private::TimeValue* timeout_time_ptr);

    //------------------------------------------------------------------
    // Classes that inherit from GDBRemoteCommunication can see and modify these
    //------------------------------------------------------------------
    bool m_send_acks:1,
         m_thread_suffix_supported:1;
    lldb_private::Listener m_rx_packet_listener;
    lldb_private::Mutex m_sequence_mutex;    // Restrict access to sending/receiving packets to a single thread at a time
    lldb_private::Predicate<bool> m_is_running;

    // If we need to send a packet while the target is running, the m_async_XXX
    // member variables take care of making this happen.
    lldb_private::Mutex m_async_mutex;
    lldb_private::Predicate<bool> m_async_packet_predicate;
    std::string m_async_packet;
    StringExtractorGDBRemote m_async_response;
    uint32_t m_async_timeout;
    int m_async_signal; // We were asked to deliver a signal to the inferior process.
    
    lldb_private::ArchSpec m_arch;      // Results from the qHostInfo call
    uint32_t m_cpusubtype;              // Results from the qHostInfo call
    lldb_private::ConstString m_os;     // Results from the qHostInfo call
    lldb_private::ConstString m_vendor; // Results from the qHostInfo call
    lldb::ByteOrder m_byte_order;       // Results from the qHostInfo call
    uint32_t m_pointer_byte_size;       // Results from the qHostInfo call
    
    
private:
    //------------------------------------------------------------------
    // For GDBRemoteCommunication only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunication);
};

#endif  // liblldb_GDBRemoteCommunication_h_
