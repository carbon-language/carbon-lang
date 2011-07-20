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
#include "llvm/Support/MachO.h"

// Other libraries and framework includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/State.h"
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
    m_addr_byte_size (4),
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
        if (WaitForPacketWithTimeoutMicroSecondsNoLock (reply_packet, GetPacketTimeoutInMicroSeconds ()))
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
            DumpPacket (log_strm, packet_data, packet_size);
            log->Printf("%.*s", (uint32_t)log_strm.GetSize(), log_strm.GetData());
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
            DataExtractor::DumpHexBytes (&log_strm, src, src_len, UINT32_MAX, LLDB_INVALID_ADDRESS);
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
                    
                    if (log)
                    {
                        PacketStreamType log_strm;
                        DumpPacket (log_strm, packet);
                        
                        log->Printf("%.*s", (uint32_t)log_strm.GetSize(), log_strm.GetData());
                    }
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
CommunicationKDP::SendRequestConnect (uint16_t reply_port, 
                                      uint16_t exc_port, 
                                      const char *greeting)
{
    PacketStreamType request_packet (Stream::eBinary, m_addr_byte_size, m_byte_order);
    if (greeting == NULL)
        greeting = "";

    const CommandType command = eCommandTypeConnect;
    // Length is 82 uint16_t and the length of the greeting C string with the terminating NULL
    const uint32_t command_length = 8 + 2 + 2 + ::strlen(greeting) + 1;
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    // Always send connect ports as little endian
    request_packet.SetByteOrder (eByteOrderLittle);
    request_packet.PutHex16 (reply_port);
    request_packet.PutHex16 (exc_port);
    request_packet.SetByteOrder (m_byte_order);
    request_packet.PutCString (greeting);
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
CommunicationKDP::SendRequestReattach (uint16_t reply_port)
{
    PacketStreamType request_packet (Stream::eBinary, m_addr_byte_size, m_byte_order);
    const CommandType command = eCommandTypeReattach;
    // Length is 8 bytes for the header plus 2 bytes for the reply UDP port
    const uint32_t command_length = 8 + 2;
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    // Always send connect ports as little endian
    request_packet.SetByteOrder (eByteOrderLittle);
    request_packet.PutHex16(reply_port);
    request_packet.SetByteOrder (m_byte_order);
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
    PacketStreamType request_packet (Stream::eBinary, m_addr_byte_size, m_byte_order);
    const CommandType command = eCommandTypeVersion;
    const uint32_t command_length = 8;
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    DataExtractor reply_packet;
    if (SendRequestAndGetReply (command, request_sequence_id, request_packet, reply_packet))
    {
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
    PacketStreamType request_packet (Stream::eBinary, m_addr_byte_size, m_byte_order);
    const CommandType command = eCommandTypeHostInfo;
    const uint32_t command_length = 8;
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    DataExtractor reply_packet;
    if (SendRequestAndGetReply (command, request_sequence_id, request_packet, reply_packet))
    {
        uint32_t offset = 8;
        m_kdp_hostinfo_cpu_mask     = reply_packet.GetU32 (&offset);
        m_kdp_hostinfo_cpu_type     = reply_packet.GetU32 (&offset);
        m_kdp_hostinfo_cpu_subtype  = reply_packet.GetU32 (&offset);
        
        ArchSpec kernel_arch;
        kernel_arch.SetArchitecture (eArchTypeMachO, 
                                     m_kdp_hostinfo_cpu_type, 
                                     m_kdp_hostinfo_cpu_subtype);
    
        m_addr_byte_size = kernel_arch.GetAddressByteSize();
        m_byte_order = kernel_arch.GetByteOrder();
        return true;
    }
    return false;
}

bool
CommunicationKDP::SendRequestDisconnect ()
{
    PacketStreamType request_packet (Stream::eBinary, m_addr_byte_size, m_byte_order);
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

uint32_t
CommunicationKDP::SendRequestReadMemory (lldb::addr_t addr, 
                                         void *dst, 
                                         uint32_t dst_len,
                                         Error &error)
{
    PacketStreamType request_packet (Stream::eBinary, m_addr_byte_size, m_byte_order);
    bool use_64 = (GetVersion() >= 11);
    uint32_t command_addr_byte_size = use_64 ? 8 : 4;
    const CommandType command = use_64 ? eCommandTypeReadMemory64 : eCommandTypeReadMemory;
    // Size is header + address size + uint32_t length
    const uint32_t command_length = 8 + command_addr_byte_size + 4;
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    request_packet.PutMaxHex64 (addr, command_addr_byte_size);
    request_packet.PutHex32 (dst_len);
    DataExtractor reply_packet;
    if (SendRequestAndGetReply (command, request_sequence_id, request_packet, reply_packet))
    {
        uint32_t offset = 8;
        uint32_t kdp_error = reply_packet.GetU32 (&offset);
        uint32_t src_len = reply_packet.GetByteSize() - 12;
        
        if (src_len > 0)
        {
            const void *src = reply_packet.GetData(&offset, src_len);
            if (src)
            {
                ::memcpy (dst, src, src_len);
                error.Clear();
                return src_len;
            }
        }
        if (kdp_error)
            error.SetErrorStringWithFormat ("kdp read memory failed (error %u)", kdp_error);
        else
            error.SetErrorString ("kdp read memory failed");
    }
    return 0;
}


uint32_t
CommunicationKDP::SendRequestWriteMemory (lldb::addr_t addr, 
                                          const void *src, 
                                          uint32_t src_len,
                                          Error &error)
{
    PacketStreamType request_packet (Stream::eBinary, m_addr_byte_size, m_byte_order);
    bool use_64 = (GetVersion() >= 11);
    uint32_t command_addr_byte_size = use_64 ? 8 : 4;
    const CommandType command = use_64 ? eCommandTypeWriteMemory64 : eCommandTypeWriteMemory;
    // Size is header + address size + uint32_t length
    const uint32_t command_length = 8 + command_addr_byte_size + 4;
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    request_packet.PutMaxHex64 (addr, command_addr_byte_size);
    request_packet.PutHex32 (src_len);
    request_packet.PutBytesAsRawHex8(src, src_len);

    DataExtractor reply_packet;
    if (SendRequestAndGetReply (command, request_sequence_id, request_packet, reply_packet))
    {
        uint32_t offset = 8;
        uint32_t kdp_error = reply_packet.GetU32 (&offset);
        if (kdp_error)
            error.SetErrorStringWithFormat ("kdp write memory failed (error %u)", kdp_error);
        else
        {
            error.Clear();
            return src_len;
        }
    }
    return 0;
}

const char *
CommunicationKDP::GetCommandAsCString (uint8_t command)
{
    switch (command)
    {
    case eCommandTypeConnect:               return "KDP_CONNECT";
    case eCommandTypeDisconnect:            return "KDP_DISCONNECT";
    case eCommandTypeHostInfo:              return "KDP_HOSTINFO";
    case eCommandTypeVersion:               return "KDP_VERSION";
    case eCommandTypeMaxBytes:              return "KDP_MAXBYTES";
    case eCommandTypeReadMemory:            return "KDP_READMEM";
    case eCommandTypeWriteMemory:           return "KDP_WRITEMEM";
    case eCommandTypeReadRegisters:         return "KDP_READREGS";
    case eCommandTypeWriteRegisters:        return "KDP_WRITEREGS";
    case eCommandTypeLoad:                  return "KDP_LOAD";
    case eCommandTypeImagePath:             return "KDP_IMAGEPATH";
    case eCommandTypeSuspend:               return "KDP_SUSPEND";
    case eCommandTypeResume:                return "KDP_RESUMECPUS";
    case eCommandTypeException:             return "KDP_EXCEPTION";
    case eCommandTypeTermination:           return "KDP_TERMINATION";
    case eCommandTypeBreakpointSet:         return "KDP_BREAKPOINT_SET";
    case eCommandTypeBreakpointRemove:      return "KDP_BREAKPOINT_REMOVE";
    case eCommandTypeRegions:               return "KDP_REGIONS";
    case eCommandTypeReattach:              return "KDP_REATTACH";
    case eCommandTypeHostReboot:            return "KDP_HOSTREBOOT";
    case eCommandTypeReadMemory64:          return "KDP_READMEM64";
    case eCommandTypeWriteMemory64:         return "KDP_WRITEMEM64";
    case eCommandTypeBreakpointSet64:       return "KDP_BREAKPOINT64_SET";
    case eCommandTypeBreakpointRemove64:    return "KDP_BREAKPOINT64_REMOVE";
    case eCommandTypeKernelVersion:         return "KDP_KERNELVERSION";
    }
    return NULL;
}

void
CommunicationKDP::DumpPacket (Stream &s, const void *data, uint32_t data_len)
{
    DataExtractor extractor (data, data_len, m_byte_order, m_addr_byte_size);
    DumpPacket (s, extractor);
}

void
CommunicationKDP::DumpPacket (Stream &s, const DataExtractor& packet)
{
    const char *error_desc = NULL;
    if (packet.GetByteSize() < 8)
    {
        error_desc = "error: invalid packet (too short): ";
    }
    else
    {
        uint32_t offset = 0;
        const uint8_t first_packet_byte = packet.GetU8 (&offset);
        const uint8_t sequence_id = packet.GetU8 (&offset);
        const uint16_t length = packet.GetU16 (&offset);
        const uint32_t key = packet.GetU32 (&offset);
        const CommandType command = ExtractCommand (first_packet_byte);
        const char *command_name = GetCommandAsCString (command);
        if (command_name)
        {
            const bool is_reply = ExtractIsReply(first_packet_byte);
            s.Printf ("%s {%u:%u} <0x%4.4x> %s", 
                      is_reply ? "<--" : "-->", 
                      key,
                      sequence_id,
                      length,
                      command_name);
            
            if (is_reply)
            {
                // Dump request reply packets
                switch (command)
                {
                    // Commands that return a single 32 bit error
                    case eCommandTypeConnect:
                    case eCommandTypeWriteMemory:
                    case eCommandTypeWriteMemory64:
                    case eCommandTypeBreakpointSet:
                    case eCommandTypeBreakpointRemove:
                    case eCommandTypeBreakpointSet64:
                    case eCommandTypeBreakpointRemove64:
                    case eCommandTypeWriteRegisters:
                    case eCommandTypeLoad:
                        {
                            const uint32_t error = packet.GetU32 (&offset);
                            s.Printf(" (error=0x%8.8x)", error);
                        }
                        break;
                    
                    case eCommandTypeDisconnect:
                    case eCommandTypeReattach:
                    case eCommandTypeHostReboot:
                    case eCommandTypeSuspend:
                    case eCommandTypeResume:
                    case eCommandTypeException:
                    case eCommandTypeTermination:
                        // No return value for the reply, just the header to ack
                        s.PutCString(" ()");
                        break;

                    case eCommandTypeHostInfo:
                        {
                            const uint32_t cpu_mask = packet.GetU32 (&offset);
                            const uint32_t cpu_type = packet.GetU32 (&offset);
                            const uint32_t cpu_subtype = packet.GetU32 (&offset);
                            s.Printf(" (cpu_mask=0x%8.8x, cpu_type=0x%8.8x, cpu_subtype=0x%8.8x)", cpu_mask, cpu_type, cpu_subtype);
                        }
                        break;
                        
                    case eCommandTypeVersion:
                        {
                            const uint32_t version = packet.GetU32 (&offset);
                            const uint32_t feature = packet.GetU32 (&offset);
                            s.Printf(" (version=0x%8.8x, feature=0x%8.8x)", version, feature);
                        }
                        break;
                        
                    case eCommandTypeRegions:
                        {
                            const uint32_t region_count = packet.GetU32 (&offset);
                            s.Printf(" (count = %u", region_count); 
                            for (uint32_t i=0; i<region_count; ++i)
                            {
                                const addr_t region_addr = packet.GetPointer (&offset);
                                const uint32_t region_size = packet.GetU32 (&offset);
                                const uint32_t region_prot = packet.GetU32 (&offset);
                                s.Printf("\n\tregion[%i] = { range = [0x%16.16llx - 0x%16.16llx), size = 0x%8.8x, prot = %s }", region_addr, region_addr + region_size, region_size, GetPermissionsAsCString (region_prot)); 
                            }
                        }
                        break;

                    case eCommandTypeReadMemory:
                    case eCommandTypeReadMemory64:
                        {
                            const uint32_t error = packet.GetU32 (&offset);
                            const uint32_t count = packet.GetByteSize() - offset;
                            s.Printf(" (error = 0x%8.8x <0x%x>:\n", error, count); 
                            if (count > 0)
                                DataExtractor::DumpHexBytes(&s, packet.GetData(&offset, count), count, 32, LLDB_INVALID_ADDRESS);
                        }
                        break;

                    case eCommandTypeReadRegisters:
                        {
                            const uint32_t error = packet.GetU32 (&offset);
                            const uint32_t count = packet.GetByteSize() - offset;
                            s.Printf(" (error = 0x%8.8x <0x%x> regs:\n", error, count); 
                            if (count > 0)
                                packet.Dump (&s,                        // Stream to dump to
                                             offset,                    // Offset within "packet"
                                             eFormatHex,                // Format to use
                                             m_addr_byte_size,          // Size of each item in bytes
                                             count / m_addr_byte_size,  // Number of items
                                             16 / m_addr_byte_size,     // Number per line
                                             LLDB_INVALID_ADDRESS,      // Don't show addresses before each line
                                             0, 0);                     // No bitfields
                        }
                        break;

                    case eCommandTypeMaxBytes:
                    case eCommandTypeImagePath:
                    case eCommandTypeKernelVersion:
                        s.Printf(" (add support for dumping this packet reply!!!"); 
                        break;
                    
                } 
            }
            else
            {
                // Dump request packets
                switch (command)
                {
                    case eCommandTypeConnect:               
                        {
                            const uint16_t reply_port = packet.GetU16 (&offset);
                            const uint16_t exc_port = packet.GetU16 (&offset);
                            s.Printf(" (reply_port=%u, exc_port=%u, greeting=\"%s\")", reply_port, exc_port, packet.GetCStr(&offset));
                        }
                        break;
                                 
                    case eCommandTypeDisconnect:
                    case eCommandTypeHostReboot:
                    case eCommandTypeHostInfo:
                    case eCommandTypeVersion:
                    case eCommandTypeRegions:
                        // No args, just the header in the request...
                        break;

                    case eCommandTypeReadMemory:
                        {
                            const uint32_t addr = packet.GetU32 (&offset);
                            const uint32_t size = packet.GetU32 (&offset);
                            s.Printf(" (addr = 0x%8.8x, size=%u)", addr, size);
                        }
                        break;

                    case eCommandTypeWriteMemory:
                        {
                            const uint32_t addr = packet.GetU32 (&offset);
                            const uint32_t size = packet.GetU32 (&offset);
                            s.Printf(" (addr = 0x%8.8x, size=%u, bytes:\n", addr, size);
                            if (size > 0)
                                DataExtractor::DumpHexBytes(&s, packet.GetData(&offset, size), size, 32, addr);
                        }
                        break;

                    case eCommandTypeReadMemory64:
                        {
                            const uint64_t addr = packet.GetU64 (&offset);
                            const uint32_t size = packet.GetU32 (&offset);
                            s.Printf(" (addr = 0x%16.16llx, size=%u)", addr, size);
                        }
                        break;

                    case eCommandTypeWriteMemory64:
                        {
                            const uint64_t addr = packet.GetU64 (&offset);
                            const uint32_t size = packet.GetU32 (&offset);
                            s.Printf(" (addr = 0x%16.16llx, size=%u, bytes:", addr, size);
                            if (size > 0)
                                DataExtractor::DumpHexBytes(&s, packet.GetData(&offset, size), size, 32, addr);
                        }
                        break;

                    case eCommandTypeReadRegisters:
                        {
                            const uint32_t cpu = packet.GetU32 (&offset);
                            const uint32_t flavor = packet.GetU32 (&offset);
                            s.Printf(" (cpu = %u, flavor=%u)", cpu, flavor);
                        }
                        break;

                    case eCommandTypeWriteRegisters:
                        {
                            const uint32_t cpu = packet.GetU32 (&offset);
                            const uint32_t flavor = packet.GetU32 (&offset);
                            const uint32_t nbytes = packet.GetByteSize() - offset;
                            s.Printf(" (cpu = %u, flavor=%u, regs:\n", cpu, flavor);
                            if (nbytes > 0)
                                packet.Dump (&s,                        // Stream to dump to
                                             offset,                    // Offset within "packet"
                                             eFormatHex,                // Format to use
                                             m_addr_byte_size,          // Size of each item in bytes
                                             nbytes / m_addr_byte_size, // Number of items
                                             16 / m_addr_byte_size,     // Number per line
                                             LLDB_INVALID_ADDRESS,      // Don't show addresses before each line
                                             0, 0);                     // No bitfields
                        }
                        break;

                    case eCommandTypeMaxBytes:

                    case eCommandTypeLoad:
                    case eCommandTypeImagePath:
                    case eCommandTypeSuspend:
                    case eCommandTypeResume:
                    case eCommandTypeException:
                    case eCommandTypeTermination:
                    case eCommandTypeBreakpointSet:
                    case eCommandTypeBreakpointRemove:
                        break;

                    case eCommandTypeReattach:
                        {
                            const uint16_t reply_port = packet.GetU16 (&offset);
                            s.Printf(" (reply_port=%u)", reply_port);
                        }
                        break;

                    case eCommandTypeBreakpointSet64:
                    case eCommandTypeBreakpointRemove64:
                    case eCommandTypeKernelVersion:
                        
                        break;
                }
            }
        }
        else
        {
            error_desc = "error: invalid packet command: ";
        }
    }

    if (error_desc)
    {
        s.PutCString (error_desc);

        packet.Dump (&s,                    // Stream to dump to
                     0,                     // Offset into "packet"
                     eFormatBytes,          // Dump as hex bytes
                     1,                     // Size of each item is 1 for single bytes
                     packet.GetByteSize(),  // Number of bytes
                     UINT32_MAX,            // Num bytes per line
                     LLDB_INVALID_ADDRESS,  // Base address
                     0, 0);                 // Bitfield info set to not do anything bitfield related
    }
}

uint32_t
CommunicationKDP::SendRequestReadRegisters (uint32_t cpu,
                                            uint32_t flavor,
                                            void *dst, 
                                            uint32_t dst_len,
                                            Error &error)
{
    PacketStreamType request_packet (Stream::eBinary, m_addr_byte_size, m_byte_order);
    const CommandType command = eCommandTypeReadRegisters;
    // Size is header + 4 byte cpu and 4 byte flavor
    const uint32_t command_length = 8 + 4 + 4;
    const uint32_t request_sequence_id = m_request_sequence_id;
    MakeRequestPacketHeader (command, request_packet, command_length);
    request_packet.PutHex32 (cpu);
    request_packet.PutHex32 (flavor);
    DataExtractor reply_packet;
    if (SendRequestAndGetReply (command, request_sequence_id, request_packet, reply_packet))
    {
        uint32_t offset = 8;
        uint32_t kdp_error = reply_packet.GetU32 (&offset);
        uint32_t src_len = reply_packet.GetByteSize() - 12;
        
        if (src_len > 0)
        {
            const uint32_t bytes_to_copy = std::min<uint32_t>(src_len, dst_len);
            const void *src = reply_packet.GetData(&offset, bytes_to_copy);
            if (src)
            {
                ::memcpy (dst, src, bytes_to_copy);
                error.Clear();
                // Return the number of bytes we could have returned regardless if
                // we copied them or not, just so we know when things don't match up
                return src_len; 
            }
        }
        if (kdp_error)
            error.SetErrorStringWithFormat("failed to read kdp registers for cpu %u flavor %u (error %u)", cpu, flavor, kdp_error);
        else
            error.SetErrorStringWithFormat("failed to read kdp registers for cpu %u flavor %u", cpu, flavor);
    }
    return 0;
}

