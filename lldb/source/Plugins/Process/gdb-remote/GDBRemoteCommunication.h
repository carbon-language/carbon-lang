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
#include "lldb/Core/Communication.h"
#include "lldb/Core/Listener.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Host/Predicate.h"

#include "Utility/StringExtractorGDBRemote.h"

class ProcessGDBRemote;

class GDBRemoteCommunication : public lldb_private::Communication
{
public:
    enum
    {
        eBroadcastBitRunPacketSent = kLoUserBroadcastBit
    };
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    GDBRemoteCommunication(const char *comm_name, 
                           const char *listener_name,
                           bool is_platform);

    virtual
    ~GDBRemoteCommunication();

    size_t
    SendPacket (const char *payload);

    size_t
    SendPacket (const char *payload,
                size_t payload_length);

    size_t
    SendPacket (lldb_private::StreamString &response);

    // Wait for a packet within 'nsec' seconds
    size_t
    WaitForPacket (StringExtractorGDBRemote &response,
                   uint32_t sec);

    // Wait for a packet with an absolute timeout time. If 'timeout' is NULL
    // wait indefinitely.
    size_t
    WaitForPacket (StringExtractorGDBRemote &response,
                   const lldb_private::TimeValue* timeout);

    char
    GetAck ();

    size_t
    SendAck ();

    size_t
    SendNack ();

    char
    CalculcateChecksum (const char *payload,
                        size_t payload_length);

    bool
    GetSequenceMutex(lldb_private::Mutex::Locker& locker);

    //------------------------------------------------------------------
    // Communication overrides
    //------------------------------------------------------------------
    virtual void
    AppendBytesToCache (const uint8_t *src, size_t src_len, bool broadcast, lldb::ConnectionStatus status);

    bool
    IsRunning() const
    {
        return m_public_is_running.GetValue();
    }

    bool
    GetSendAcks ()
    {
        return m_send_acks;
    }

    //------------------------------------------------------------------
    // Client and server must implement these pure virtual functions
    //------------------------------------------------------------------
    virtual bool
    GetThreadSuffixSupported () = 0;

    //------------------------------------------------------------------
    // Set the global packet timeout.
    //
    // For clients, this is the timeout that gets used when sending
    // packets and waiting for responses. For servers, this might not
    // get used, and if it doesn't this should be moved to the
    // GDBRemoteCommunicationClient.
    //------------------------------------------------------------------
    uint32_t 
    SetPacketTimeout (uint32_t packet_timeout)
    {
        const uint32_t old_packet_timeout = m_packet_timeout;
        m_packet_timeout = packet_timeout;
        return old_packet_timeout;
    }

    //------------------------------------------------------------------
    // Start a debugserver instance on the current host using the
    // supplied connection URL.
    //------------------------------------------------------------------
    lldb_private::Error
    StartDebugserverProcess (const char *connect_url,
                             const char *unix_socket_name,
                             lldb_private::ProcessLaunchInfo &launch_info); 

protected:
    typedef std::list<std::string> packet_collection;

    size_t
    SendPacketNoLock (const char *payload, 
                      size_t payload_length);

    size_t
    WaitForPacketNoLock (StringExtractorGDBRemote &response, 
                         const lldb_private::TimeValue* timeout_ptr);

    bool
    WaitForNotRunningPrivate (const lldb_private::TimeValue *timeout_ptr);

    //------------------------------------------------------------------
    // Classes that inherit from GDBRemoteCommunication can see and modify these
    //------------------------------------------------------------------
    uint32_t m_packet_timeout;
    lldb_private::Listener m_rx_packet_listener;
    lldb_private::Mutex m_sequence_mutex;    // Restrict access to sending/receiving packets to a single thread at a time
    lldb_private::Predicate<bool> m_public_is_running;
    lldb_private::Predicate<bool> m_private_is_running;
    bool m_send_acks;
    bool m_is_platform; // Set to true if this class represents a platform,
                        // false if this class represents a debug session for
                        // a single process
    



private:
    //------------------------------------------------------------------
    // For GDBRemoteCommunication only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunication);
};

#endif  // liblldb_GDBRemoteCommunication_h_
