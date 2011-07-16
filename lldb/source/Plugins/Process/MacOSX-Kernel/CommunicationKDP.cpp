//===-- CommunicationKDP.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "CommunicationKDP.h"

// C Includes
#include <limits.h>
#include <string.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Target/Process.h"

// Project includes
#include "ProcessKDPLog.h"

#define DEBUGSERVER_BASENAME    "debugserver"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// CommunicationKDP constructor
//----------------------------------------------------------------------
CommunicationKDP::CommunicationKDP (const char *comm_name) :
    Communication(comm_name),
    m_byte_order (eByteOrderLittle),
    m_packet_timeout (1),
    m_sequence_mutex (Mutex::eMutexTypeRecursive),
    m_public_is_running (false),
    m_private_is_running (false),
    m_session_key (0u),
    m_request_sequence_id (0u),
    m_exception_sequence_id (0u),
    m_kdp_version_version (0u),
    m_kdp_version_feature (0u),
    m_kdp_hostinfo_cpu_mask (0u),
    m_kdp_hostinfo_cpu_type (0u),
    m_kdp_hostinfo_cpu_subtype (0u)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CommunicationKDP::~CommunicationKDP()
{
    if (IsConnected())
    {
        Disconnect();
    }
}

bool
CommunicationKDP::SendRequestPacket (const PacketStreamType &request_packet)
{
    Mutex::Locker locker(m_sequence_mutex);
    return SendRequestPacketNoLock (request_packet);
}

#if 0
typedef struct {
	uint8_t     request;	// Either: CommandType | ePacketTypeRequest, or CommandType | ePacketTypeReply
	uint8_t     sequence;
	uint16_t    length;		// Length of entire packet including this header
	uint32_t	key;		// Session key
} kdp_hdr_t;
#endif

void
CommunicationKDP::MakeRequestPacketHeader (CommandType request_type, 
                                           PacketStreamType &request_packet,
                                           uint16_t request_length)
{
    request_packet.Clear();
    request_packet.PutHex8 (request_type | ePacketTypeRequest); // Set the request type
    request_packet.PutHex8 (m_request_sequence_id++);           // Sequence number
    request_packet.PutHex16 (request_length);                   // Length of the packet including this header
    request_packet.PutHex32 (m_session_key);                    // Session key
}

bool
CommunicationKDP::SendRequestAndGetReply (const CommandType command,
                                          const uint8_t request_sequence_id,
                                          const PacketStreamType &request_packet, 
                                          DataExtractor &reply_packet)
{

    Mutex::Locker locker(m_sequence_mutex);    
    if (SendRequestPacketNoLock(request_packet))
    {
        if (WaitForPacketWithTimeoutMicroSecondsNoLock (reply_packet, m_packet_timeout))
        {
            uint32_t offset = 0;
            const uint8_t reply_command = reply_packet.GetU8 (&offset);
            const uint8_t reply_sequence_id = reply_packet.GetU8 (&offset);
            if ((reply_command & eCommandTypeMask) == command)
            {
                if (request_sequence_id == reply_sequence_id)
                    return true;
            }
        }
    }
    reply_packet.Clear();
    return false;
}

bool
CommunicationKDP::SendRequestPacketNoLock (const PacketStreamType &request_packet)
{
    if (IsConnected())
    {
        const char *packet_data = request_packet.GetData();
        const size_t packet_size = request_packet.GetSize();

        LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PACKETS));
        if (log)
        {
            PacketStreamType log_strm;
            
            DataExtractor::DumpHexBytes (&log_strm, packet_data, packet_size, 0);
            
            log->Printf("request packet: <%u>\n%s", packet_size, log_strm.GetData());
        }
        ConnectionStatus status = eConnectionStatusSuccess;

        size_t bytes_written = Write (packet_data, 
                                      packet_size, 
                                      status, 
                                      NULL);

        if (bytes_written == packet_size)
            return true;
        
        if (log)
            log->Printf ("error: failed to send packet entire packet %zu of %zu bytes sent", bytes_written, packet_size);
    }
    return false;
}

bool
CommunicationKDP::GetSequenceMutex (Mutex::Locker& locker)
{
    return locker.TryLock (m_sequence_mutex.GetMutex());
}


bool
CommunicationKDP::WaitForNotRunningPrivate (const TimeValue *timeout_ptr)
{
    return m_private_is_running.WaitForValueEqualTo (false, timeout_ptr, NULL);
}

size_t
CommunicationKDP::WaitForPacketWithTimeoutMicroSeconds (DataExtractor &packet, uint32_t timeout_usec)
{
    Mutex::Locker locker(m_sequence_mutex);
    return WaitForPacketWithTimeoutMicroSecondsNoLock (packet, timeout_usec);
}

size_t
CommunicationKDP::WaitForPacketWithTimeoutMicroSecondsNoLock (DataExtractor &packet, uint32_t timeout_usec)
{
    uint8_t buffer[8192];
    Error error;

    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PACKETS | KDP_LOG_VERBOSE));

    // Check for a packet from our cache first without trying any reading...
    if (CheckForPacket (NULL, 0, packet))
        return packet.GetByteSize();

    bool timed_out = false;
    while (IsConnected() && !timed_out)
    {
        lldb::ConnectionStatus status;
        size_t bytes_read = Read (buffer, sizeof(buffer), timeout_usec, status, &error);
        
        if (log)
            log->Printf ("%s: Read (buffer, (sizeof(buffer), timeout_usec = 0x%x, status = %s, error = %s) => bytes_read = %zu",
                         __PRETTY_FUNCTION__,
                         timeout_usec, 
                         Communication::ConnectionStatusAsCString (status),
                         error.AsCString(), 
                         bytes_read);

        if (bytes_read > 0)
        {
            if (CheckForPacket (buffer, bytes_read, packet))
                return packet.GetByteSize();
        }
        else
        {
            switch (status)
            {
            case eConnectionStatusTimedOut:
                timed_out = true;
                break;
            case eConnectionStatusSuccess:
                //printf ("status = success but error = %s\n", error.AsCString("<invalid>"));
                break;
                
            case eConnectionStatusEndOfFile:
            case eConnectionStatusNoConnection:
            case eConnectionStatusLostConnection:
            case eConnectionStatusError:
                Disconnect();
                break;
            }
        }
    }
    packet.Clear ();    
    return 0;
}

bool
CommunicationKDP::CheckForPacket (const uint8_t *src, size_t src_len, DataExtractor &packet)
{
    // Put the packet data into the buffer in a thread safe fashion
    Mutex::Locker locker(m_bytes_mutex);
    
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PACKETS));

    if (src && src_len > 0)
    {
        if (log && log->GetVerbose())
        {
            PacketStreamType log_strm;
            DataExtractor::DumpHexBytes (&log_strm, src, src_len, 0);
            log->Printf ("CommunicationKDP::%s adding %u bytes: %s",
                         __FUNCTION__, 
                         (uint32_t)src_len, 
                         log_strm.GetData());
        }
        m_bytes.append ((const char *)src, src_len);
    }

    // Make sure we at least have enough bytes for a packet header
    const size_t bytes_available = m_bytes.size();
    if (bytes_available >= 8)
    {
        packet.SetData (&m_bytes[0], bytes_available, m_byte_order);
        uint32_t offset = 0;
        uint8_t reply_command = packet.GetU8(&offset);
        switch (reply_command)
        {
        case ePacketTypeReply | eCommandTypeConnect:
        case ePacketTypeReply | eCommandTypeDisconnect:
        case ePacketTypeReply | eCommandTypeHostInfo:
        case ePacketTypeReply | eCommandTypeVersion:
        case ePacketTypeReply | eCommandTypeMaxBytes:
        case ePacketTypeReply | eCommandTypeReadMemory:
        case ePacketTypeReply | eCommandTypeWriteMemory:
        case ePacketTypeReply | eCommandTypeReadRegisters:
        case ePacketTypeReply | eCommandTypeWriteRegisters:
        case ePacketTypeReply | eCommandTypeLoad:
        case ePacketTypeReply | eCommandTypeImagePath:
        case ePacketTypeReply | eCommandTypeSuspend:
        case ePacketTypeReply | eCommandTypeResume:
        case ePacketTypeReply | eCommandTypeException:
        case ePacketTypeReply | eCommandTypeTermination:
        case ePacketTypeReply | eCommandTypeBreakpointSet:
        case ePacketTypeReply | eCommandTypeBreakpointRemove:
        case ePacketTypeReply | eCommandTypeRegions:
        case ePacketTypeReply | eCommandTypeReattach:
        case ePacketTypeReply | eCommandTypeHostReboot:
        case ePacketTypeReply | eCommandTypeReadMemory64:
        case ePacketTypeReply | eCommandTypeWriteMemory64:
        case ePacketTypeReply | eCommandTypeBreakpointSet64:
        case ePacketTypeReply | eCommandTypeBreakpointRemove64:
        case ePacketTypeReply | eCommandTypeKernelVersion:
            {
                offset = 2;
                const uint16_t length = packet.GetU16 (&offset);
                if (length <= bytes_available)
                {
                    // We have an entire packet ready, we need to copy the data
                    // bytes into a buffer that will be owned by the packet and
                    // erase the bytes from our communcation buffer "m_bytes"
                    packet.SetData (DataBufferSP (new DataBufferHeap (&m_bytes[0], length)));
                    m_bytes.erase (0, length);
                    return true;
                }
            }
            break;

        default:
            // Unrecognized reply command byte, erase this byte and try to get back on track
            if (log)
                log->Printf ("CommunicationKDP::%s: tossing junk byte: 0x%2.2x", 
                             __FUNCTION__, 
                             (uint8_t)m_bytes[0]);
            m_bytes.erase(0, 1);
            break;
        }
    }
    packet.Clear();
    return false;
}


bool
CommunicationKDP::Connect (uint16_t reply_port, 
                           uint16_t exc_port, 
                           const char *greeting)
{
    PacketStreamType request_packet (Stream::eBinary, 4, m_byte_order);
    if (greeting == NULL)
        greeting = "";

    const CommandType command = eCommandTypeConnect;
    // Length is 82 uint16_t and the length of the greeting C string
    const uint32_t command_length = 8 + 2 + 2 + ::strlen(greeting);
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    request_packet.PutHex16(reply_port);
    request_packet.PutHex16(exc_port);
    request_packet.PutCString(greeting);
    DataExtractor reply_packet;
    return SendRequestAndGetReply (command, request_sequence_id, request_packet, reply_packet);
}

void
CommunicationKDP::ClearKDPSettings ()
{
    m_request_sequence_id = 0;
    m_kdp_version_version = 0;
    m_kdp_version_feature = 0;
    m_kdp_hostinfo_cpu_mask = 0;
    m_kdp_hostinfo_cpu_type = 0;
    m_kdp_hostinfo_cpu_subtype = 0;
}

bool
CommunicationKDP::Reattach (uint16_t reply_port)
{
    PacketStreamType request_packet (Stream::eBinary, 4, m_byte_order);
    const CommandType command = eCommandTypeReattach;
    // Length is 8 bytes for the header plus 2 bytes for the reply UDP port
    const uint32_t command_length = 8 + 2;
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    request_packet.PutHex16(reply_port);
    DataExtractor reply_packet;
    if (SendRequestAndGetReply (command, request_sequence_id, request_packet, reply_packet))
    {
        // Reset the sequence ID to zero for reattach
        ClearKDPSettings ();
        uint32_t offset = 4;
        m_session_key = reply_packet.GetU32 (&offset);
        return true;
    }
    return false;
}

uint32_t
CommunicationKDP::GetVersion ()
{
    if (!VersionIsValid())
        SendRequestVersion();
    return m_kdp_version_version;
}

uint32_t
CommunicationKDP::GetFeatureFlags ()
{
    if (!VersionIsValid())
        SendRequestVersion();
    return m_kdp_version_feature;
}

bool
CommunicationKDP::SendRequestVersion ()
{
    PacketStreamType request_packet (Stream::eBinary, 4, m_byte_order);
    const CommandType command = eCommandTypeVersion;
    const uint32_t command_length = 8;
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    DataExtractor reply_packet;
    if (SendRequestAndGetReply (command, request_sequence_id, request_packet, reply_packet))
    {
        // Reset the sequence ID to zero for reattach
        uint32_t offset = 8;
        m_kdp_version_version = reply_packet.GetU32 (&offset);
        m_kdp_version_feature = reply_packet.GetU32 (&offset);
        return true;
    }
    return false;
}

uint32_t
CommunicationKDP::GetCPUMask ()
{
    if (!HostInfoIsValid())
        SendRequestHostInfo();
    return m_kdp_hostinfo_cpu_mask;
}

uint32_t
CommunicationKDP::GetCPUType ()
{
    if (!HostInfoIsValid())
        SendRequestHostInfo();
    return m_kdp_hostinfo_cpu_type;
}

uint32_t
CommunicationKDP::GetCPUSubtype ()
{
    if (!HostInfoIsValid())
        SendRequestHostInfo();
    return m_kdp_hostinfo_cpu_subtype;
}

bool
CommunicationKDP::SendRequestHostInfo ()
{
    PacketStreamType request_packet (Stream::eBinary, 4, m_byte_order);
    const CommandType command = eCommandTypeHostInfo;
    const uint32_t command_length = 8;
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    DataExtractor reply_packet;
    if (SendRequestAndGetReply (command, request_sequence_id, request_packet, reply_packet))
    {
        // Reset the sequence ID to zero for reattach
        uint32_t offset = 8;
        m_kdp_hostinfo_cpu_mask = reply_packet.GetU32 (&offset);
        m_kdp_hostinfo_cpu_type = reply_packet.GetU32 (&offset);
        m_kdp_hostinfo_cpu_subtype = reply_packet.GetU32 (&offset);
        return true;
    }
    return false;
}

bool
CommunicationKDP::Disconnect ()
{
    PacketStreamType request_packet (Stream::eBinary, 4, m_byte_order);
    const CommandType command = eCommandTypeDisconnect;
    const uint32_t command_length = 8;
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    DataExtractor reply_packet;
    if (SendRequestAndGetReply (command, request_sequence_id, request_packet, reply_packet))
    {
        // Are we supposed to get a reply for disconnect?
    }
    ClearKDPSettings ();
    return true;
}

