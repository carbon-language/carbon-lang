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
#include "lldb/Core/StreamBuffer.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Host/TimeValue.h"

class CommunicationKDP : public lldb_private::Communication
{
public:
    enum
    {
        eBroadcastBitRunPacketSent = kLoUserBroadcastBit
    };
    
    const static uint32_t kMaxPacketSize = 1200;
    const static uint32_t kMaxDataSize = 1024;
    typedef lldb_private::StreamBuffer<1024> PacketStreamType;
    typedef enum 
    {
        eCommandTypeConnect = 0u,
        eCommandTypeDisconnect,
        eCommandTypeHostInfo,
        eCommandTypeVersion,
        eCommandTypeMaxBytes,
        eCommandTypeReadMemory,
        eCommandTypeWriteMemory,
        eCommandTypeReadRegisters,
        eCommandTypeWriteRegisters,
        eCommandTypeLoad,
        eCommandTypeImagePath,
        eCommandTypeSuspend,
        eCommandTypeResume,
        eCommandTypeException,
        eCommandTypeTermination,
        eCommandTypeBreakpointSet,
        eCommandTypeBreakpointRemove,
        eCommandTypeRegions,
        eCommandTypeReattach,
        eCommandTypeHostReboot,
        eCommandTypeReadMemory64,
        eCommandTypeWriteMemory64,
        eCommandTypeBreakpointSet64,
        eCommandTypeBreakpointRemove64,
        eCommandTypeKernelVersion
    } CommandType;

    enum 
    {
        eFeatureLocalBreakpointsSupported = (1u << 0),
    };

    typedef enum
    {
        KDP_PROTERR_SUCCESS = 0,
        KDP_PROTERR_ALREADY_CONNECTED,
        KDP_PROTERR_BAD_NBYTES,
        KDP_PROTERR_BADFLAVOR
    } KDPError;
    
    typedef enum
    {
        ePacketTypeRequest  = 0x00u,
        ePacketTypeReply    = 0x80u,
        ePacketTypeMask     = 0x80u,
        eCommandTypeMask    = 0x7fu
    } PacketType;
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommunicationKDP (const char *comm_name);

    virtual
    ~CommunicationKDP();

    bool
    SendRequestPacket (const PacketStreamType &request_packet);

    // Wait for a packet within 'nsec' seconds
    size_t
    WaitForPacketWithTimeoutMicroSeconds (lldb_private::DataExtractor &response,
                                          uint32_t usec);

    bool
    GetSequenceMutex(lldb_private::Mutex::Locker& locker);

    bool
    CheckForPacket (const uint8_t *src, 
                    size_t src_len, 
                    lldb_private::DataExtractor &packet);
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

    
    //------------------------------------------------------------------
    // Public Request Packets
    //------------------------------------------------------------------
    bool
    SendRequestConnect (uint16_t reply_port, 
                        uint16_t exc_port, 
                        const char *greeting);

    bool
    SendRequestReattach (uint16_t reply_port);

    bool
    SendRequestDisconnect ();
    
    uint32_t
    SendRequestReadMemory (lldb::addr_t addr, 
                           void *dst, 
                           uint32_t dst_size,
                           lldb_private::Error &error);

    uint32_t
    SendRequestWriteMemory (lldb::addr_t addr, 
                            const void *src, 
                            uint32_t src_len,
                            lldb_private::Error &error);

    uint32_t
    SendRequestReadRegisters (uint32_t cpu,
                              uint32_t flavor,
                              void *dst, 
                              uint32_t dst_size,
                              lldb_private::Error &error);
    
    const char *
    GetKernelVersion ();
    
    uint32_t
    GetVersion ();

    uint32_t
    GetFeatureFlags ();

    bool
    LocalBreakpointsAreSupported ()
    {
        return (GetFeatureFlags() & eFeatureLocalBreakpointsSupported) != 0;
    }

    uint32_t
    GetCPUMask ();
    
    uint32_t
    GetCPUType ();
    
    uint32_t
    GetCPUSubtype ();

    // If cpu_mask is zero, then we will resume all CPUs
    bool
    SendRequestResume (uint32_t cpu_mask = 0);

    bool
    SendRequestSuspend ();

    bool
    SendRequestBreakpoint (bool set, lldb::addr_t addr);

protected:
    typedef std::list<std::string> packet_collection;

    bool
    SendRequestPacketNoLock (const PacketStreamType &request_packet);

    size_t
    WaitForPacketWithTimeoutMicroSecondsNoLock (lldb_private::DataExtractor &response, 
                                                uint32_t timeout_usec);

    bool
    WaitForNotRunningPrivate (const lldb_private::TimeValue *timeout_ptr);

    void
    MakeRequestPacketHeader (CommandType request_type, 
                             PacketStreamType &request_packet,
                             uint16_t request_length);

    //------------------------------------------------------------------
    // Protected Request Packets (use public accessors which will cache
    // results.
    //------------------------------------------------------------------
    bool
    SendRequestVersion ();
    
    bool
    SendRequestHostInfo ();

    bool
    SendRequestKernelVersion ();
    
    void
    DumpPacket (lldb_private::Stream &s, 
                const void *data, 
                uint32_t data_len);

    void
    DumpPacket (lldb_private::Stream &s, 
                const lldb_private::DataExtractor& extractor);

    bool
    VersionIsValid() const
    {
        return m_kdp_version_version != 0;
    }

    bool
    HostInfoIsValid() const
    {
        return m_kdp_hostinfo_cpu_type != 0;
    }

    bool
    ExtractIsReply (uint8_t first_packet_byte) const
    {
        // TODO: handle big endian...
        return (first_packet_byte & ePacketTypeMask) != 0;
    }

    CommandType
    ExtractCommand (uint8_t first_packet_byte) const
    {
        // TODO: handle big endian...
        return (CommandType)(first_packet_byte & eCommandTypeMask);
    }
    
    static const char *
    GetCommandAsCString (uint8_t command);

    void
    ClearKDPSettings ();
    
    bool
    SendRequestAndGetReply (const CommandType command,
                            const uint8_t request_sequence_id,
                            const PacketStreamType &request_packet, 
                            lldb_private::DataExtractor &reply_packet);
    //------------------------------------------------------------------
    // Classes that inherit from CommunicationKDP can see and modify these
    //------------------------------------------------------------------
    uint32_t m_addr_byte_size;
    lldb::ByteOrder m_byte_order;
    uint32_t m_packet_timeout;
    lldb_private::Mutex m_sequence_mutex;    // Restrict access to sending/receiving packets to a single thread at a time
    lldb_private::Predicate<bool> m_public_is_running;
    lldb_private::Predicate<bool> m_private_is_running;
    uint32_t m_session_key;
    uint8_t m_request_sequence_id;
    uint8_t m_exception_sequence_id;
    uint32_t m_kdp_version_version;
    uint32_t m_kdp_version_feature;
    uint32_t m_kdp_hostinfo_cpu_mask;
    uint32_t m_kdp_hostinfo_cpu_type;
    uint32_t m_kdp_hostinfo_cpu_subtype;
    std::string m_kernel_version;
    lldb::addr_t m_last_read_memory_addr; // Last memory read address for logging
private:
    //------------------------------------------------------------------
    // For CommunicationKDP only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (CommunicationKDP);
};

#endif  // liblldb_CommunicationKDP_h_
