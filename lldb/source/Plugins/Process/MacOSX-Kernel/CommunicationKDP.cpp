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
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Target/Process.h"
#include "Utility/StringExtractor.h"

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
    m_packet_timeout (1),
    m_sequence_mutex (Mutex::eMutexTypeRecursive),
    m_public_is_running (false),
    m_private_is_running (false),
    m_session_key (0),
    m_request_sequence_id (0),
    m_exception_sequence_id (0)
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
CommunicationKDP::SendRequestPacket (const StreamString &request_packet)
{
    Mutex::Locker locker(m_sequence_mutex);
    return SendRequestPacketNoLock (request_packet);
}

void
CommunicationKDP::MakeRequestPacketHeader (RequestType request_type, 
                                           StreamString &request_packet)
{
    request_packet.Clear();
    request_packet.PutHex32 (request_type);             // Set the request type
    request_packet.PutHex8  (ePacketTypeRequest);       // Set the packet type
    request_packet.PutHex8  (++m_request_sequence_id);  // Sequence number
    request_packet.PutHex16 (0);                        // Pad1 and Pad2 bytes
    request_packet.PutHex32 (m_session_key);            // Session key
}


bool
CommunicationKDP::SendRequestPacketNoLock (const StreamString &request_packet)
{
    if (IsConnected())
    {
        const char *packet_data = request_packet.GetData();
        const size_t packet_size = request_packet.GetSize();

        LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PACKETS));
        if (log)
        {
            StreamString log_strm;
            DataExtractor data (packet_data, 
                                packet_size,
                                request_packet.GetByteOrder(),
                                request_packet.GetAddressByteSize());
            data.Dump (&log_strm, 
                       0, 
                       eFormatBytes,
                       1, 
                       packet_size,
                       32,  // Num bytes per line
                       0,   // Base address
                       0, 
                       0);
            
            log->Printf("request packet: <%u>\n%s", packet_size, log_strm.GetString().c_str());
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
CommunicationKDP::WaitForPacketWithTimeoutMicroSeconds (StringExtractor &packet, uint32_t timeout_usec)
{
    Mutex::Locker locker(m_sequence_mutex);
    return WaitForPacketWithTimeoutMicroSecondsNoLock (packet, timeout_usec);
}

size_t
CommunicationKDP::WaitForPacketWithTimeoutMicroSecondsNoLock (StringExtractor &packet, uint32_t timeout_usec)
{
    uint8_t buffer[8192];
    Error error;

    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PACKETS | KDP_LOG_VERBOSE));

    // Check for a packet from our cache first without trying any reading...
    if (CheckForPacket (NULL, 0, packet))
        return packet.GetStringRef().size();

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
                return packet.GetStringRef().size();
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
CommunicationKDP::CheckForPacket (const uint8_t *src, size_t src_len, StringExtractor &packet)
{
    // Put the packet data into the buffer in a thread safe fashion
    Mutex::Locker locker(m_bytes_mutex);
    
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PACKETS));

    if (src && src_len > 0)
    {
        if (log && log->GetVerbose())
        {
            StreamString s;
            log->Printf ("CommunicationKDP::%s adding %u bytes: %.*s",
                         __FUNCTION__, 
                         (uint32_t)src_len, 
                         (uint32_t)src_len, 
                         src);
        }
        m_bytes.append ((const char *)src, src_len);
    }

    // Parse up the packets into gdb remote packets
    if (!m_bytes.empty())
    {
        // TODO: Figure out if we have a full packet reply
    }
    packet.Clear();
    return false;
}


CommunicationKDP::ErrorType
CommunicationKDP::Connect (uint16_t reply_port, 
                           uint16_t exc_port, 
                           const char *greeting)
{
    StreamString request_packet (Stream::eBinary, 4, eByteOrderLittle);
    MakeRequestPacketHeader (eRequestTypeConnect, request_packet);
    request_packet.PutHex16(reply_port);
    request_packet.PutHex16(exc_port);
    request_packet.PutCString(greeting);
    
    return eErrorUnimplemented;
}

CommunicationKDP::ErrorType
CommunicationKDP::Disconnect ()
{
    return eErrorUnimplemented;
}

