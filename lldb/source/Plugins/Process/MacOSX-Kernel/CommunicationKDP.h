//===-- CommunicationKDP.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommunicationKDP_h_
#define liblldb_CommunicationKDP_h_

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
#include "lldb/Host/TimeValue.h"

class StringExtractor;

class CommunicationKDP : public lldb_private::Communication
{
public:
    enum
    {
        eBroadcastBitRunPacketSent = kLoUserBroadcastBit
    };
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommunicationKDP (const char *comm_name);

    virtual
    ~CommunicationKDP();

    size_t
    SendPacket (const char *payload);

    size_t
    SendPacket (const char *payload,
                size_t payload_length);

    size_t
    SendPacket (lldb_private::StreamString &response);

    // Wait for a packet within 'nsec' seconds
    size_t
    WaitForPacketWithTimeoutMicroSeconds (StringExtractor &response,
                                          uint32_t usec);

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

    bool
    CheckForPacket (const uint8_t *src, 
                    size_t src_len, 
                    StringExtractor &packet);
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
    // Set the global packet timeout.
    //
    // For clients, this is the timeout that gets used when sending
    // packets and waiting for responses. For servers, this might not
    // get used, and if it doesn't this should be moved to the
    // CommunicationKDPClient.
    //------------------------------------------------------------------
    uint32_t 
    SetPacketTimeout (uint32_t packet_timeout)
    {
        const uint32_t old_packet_timeout = m_packet_timeout;
        m_packet_timeout = packet_timeout;
        return old_packet_timeout;
    }

    uint32_t
    GetPacketTimeoutInMicroSeconds () const
    {
        return m_packet_timeout * lldb_private::TimeValue::MicroSecPerSec;
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
    WaitForPacketWithTimeoutMicroSecondsNoLock (StringExtractor &response, 
                                                uint32_t timeout_usec);

    bool
    WaitForNotRunningPrivate (const lldb_private::TimeValue *timeout_ptr);

    //------------------------------------------------------------------
    // Classes that inherit from CommunicationKDP can see and modify these
    //------------------------------------------------------------------
    uint32_t m_packet_timeout;
    lldb_private::Mutex m_sequence_mutex;    // Restrict access to sending/receiving packets to a single thread at a time
    lldb_private::Predicate<bool> m_public_is_running;
    lldb_private::Predicate<bool> m_private_is_running;
    bool m_send_acks;

private:
    //------------------------------------------------------------------
    // For CommunicationKDP only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (CommunicationKDP);
};

#endif  // liblldb_CommunicationKDP_h_
