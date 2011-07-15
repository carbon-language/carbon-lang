//===-- CommunicationKDP.h --------------------------------------*- C++ -*-===//
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
    
    const static uint32_t kMaxPacketSize = 1200;
    const static uint32_t kMaxDataSize = 1024;
    
    typedef enum 
    {
        eRequestTypeConnect = 0u,
        eRequestTypeDisconnect,
        eRequestTypeHostInfo,
        eRequestTypeVersion,
        eRequestTypeMaxBytes,
        eRequestTypeReadMemory,
        eRequestTypeWriteMemory,
        eRequestTypeReadRegisters,
        eRequestTypeWriteRegisters,
        eRequestTypeLoad,
        eRequestTypeImagePath,
        eRequestTypeSuspend,
        eRequestTypeResume,
        eRequestTypeException,
        eRequestTypeTermination,
        eRequestTypeBreakpointSet,
        eRequestTypeBreakpointRemove,
        eRequestTypeRegions,
        eRequestTypeReattach,
        eRequestTypeHostReboot,
        eRequestTypeReadMemory64,
        eRequestTypeWriteMemory64,
        eRequestTypeBreakpointSet64,
        eRequestTypeBreakpointRemove64,
        eRequestTypeKernelVersion
    } RequestType;

    typedef enum 
    {
        eErrorSuccess = 0,
        eErrorAlreadyConnected,
        eErrorPacketToBig,
        eErrorInvalidRegisterFlavor,
        eErrorUnimplemented
    } ErrorType;
    
    typedef enum
    {
        ePacketTypeRequest  = 0u,
        ePacketTypeReply    = 1u
    } PacketType;
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommunicationKDP (const char *comm_name);

    virtual
    ~CommunicationKDP();

    bool
    SendRequestPacket (const lldb_private::StreamString &request_packet);

    // Wait for a packet within 'nsec' seconds
    size_t
    WaitForPacketWithTimeoutMicroSeconds (StringExtractor &response,
                                          uint32_t usec);

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

    
    ErrorType
    Connect (uint16_t reply_port, 
             uint16_t exc_port, 
             const char *greeting);

    ErrorType
    Disconnect ();

protected:
    typedef std::list<std::string> packet_collection;

    bool
    SendRequestPacketNoLock (const lldb_private::StreamString &request_packet);

    size_t
    WaitForPacketWithTimeoutMicroSecondsNoLock (StringExtractor &response, 
                                                uint32_t timeout_usec);

    bool
    WaitForNotRunningPrivate (const lldb_private::TimeValue *timeout_ptr);

    void
    MakeRequestPacketHeader (RequestType request_type, 
                             lldb_private::StreamString &request_packet);

    //------------------------------------------------------------------
    // Classes that inherit from CommunicationKDP can see and modify these
    //------------------------------------------------------------------
    uint32_t m_packet_timeout;
    lldb_private::Mutex m_sequence_mutex;    // Restrict access to sending/receiving packets to a single thread at a time
    lldb_private::Predicate<bool> m_public_is_running;
    lldb_private::Predicate<bool> m_private_is_running;
    uint32_t m_session_key;
    uint8_t m_request_sequence_id;
    uint8_t m_exception_sequence_id;
private:
    //------------------------------------------------------------------
    // For CommunicationKDP only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (CommunicationKDP);
};

#endif  // liblldb_CommunicationKDP_h_
