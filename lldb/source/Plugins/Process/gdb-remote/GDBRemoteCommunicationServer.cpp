//===-- GDBRemoteCommunicationServer.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "GDBRemoteCommunicationServer.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/ADT/Triple.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/TimeValue.h"

// Project includes
#include "Utility/StringExtractorGDBRemote.h"
#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// GDBRemoteCommunicationServer constructor
//----------------------------------------------------------------------
GDBRemoteCommunicationServer::GDBRemoteCommunicationServer() :
    GDBRemoteCommunication ("gdb-remote.server", "gdb-remote.server.rx_packet"),
    m_async_thread (LLDB_INVALID_HOST_THREAD)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
GDBRemoteCommunicationServer::~GDBRemoteCommunicationServer()
{
}


//void *
//GDBRemoteCommunicationServer::AsyncThread (void *arg)
//{
//    GDBRemoteCommunicationServer *server = (GDBRemoteCommunicationServer*) arg;
//
//    LogSP log;// (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));
//    if (log)
//        log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %i) thread starting...", __FUNCTION__, arg, process->GetID());
//
//    StringExtractorGDBRemote packet;
//    
//    while ()
//    {
//        if (packet.
//    }
//
//    if (log)
//        log->Printf ("ProcessGDBRemote::%s (arg = %p, pid = %i) thread exiting...", __FUNCTION__, arg, process->GetID());
//
//    process->m_async_thread = LLDB_INVALID_HOST_THREAD;
//    return NULL;
//}
//
bool
GDBRemoteCommunicationServer::GetPacketAndSendResponse (const TimeValue* timeout_ptr, 
                                                        Error &error,
                                                        bool &interrupt, 
                                                        bool &quit)
{
    StringExtractorGDBRemote packet;
    if (WaitForPacketNoLock (packet, timeout_ptr))
    {
        const StringExtractorGDBRemote::ServerPacketType packet_type = packet.GetServerPacketType ();
        switch (packet_type)
        {
        case StringExtractorGDBRemote::eServerPacketType_nack:
        case StringExtractorGDBRemote::eServerPacketType_ack:
            break;

        case StringExtractorGDBRemote::eServerPacketType_invalid:
            error.SetErrorString("invalid packet");
            quit = true;
            break;

        case StringExtractorGDBRemote::eServerPacketType_interrupt:
            error.SetErrorString("interrupt received");
            interrupt = true;
            break;
        
        case StringExtractorGDBRemote::eServerPacketType_unimplemented:
            return SendUnimplementedResponse (packet.GetStringRef().c_str()) > 0;

        case StringExtractorGDBRemote::eServerPacketType_qHostInfo:
            return Handle_qHostInfo (packet);

        case StringExtractorGDBRemote::eServerPacketType_qProcessInfoPID:
            return Handle_qProcessInfoPID (packet);

        case StringExtractorGDBRemote::eServerPacketType_qfProcessInfo:
            return Handle_qfProcessInfo (packet);

        case StringExtractorGDBRemote::eServerPacketType_qsProcessInfo:
            return Handle_qsProcessInfo (packet);
        
        case StringExtractorGDBRemote::eServerPacketType_qUserName:
            return Handle_qUserName (packet);

        case StringExtractorGDBRemote::eServerPacketType_qGroupName:
            return Handle_qGroupName (packet);

        case StringExtractorGDBRemote::eServerPacketType_qSpeedTest:
            return Handle_qSpeedTest (packet);
        case StringExtractorGDBRemote::eServerPacketType_QStartNoAckMode:
            return Handle_QStartNoAckMode (packet);
        }
        return true;
    }
    else
    {
        if (!IsConnected())
            error.SetErrorString("lost connection");
        else
            error.SetErrorString("timeout");
    }

    return false;
}

size_t
GDBRemoteCommunicationServer::SendUnimplementedResponse (const char *)
{
    // TODO: Log the packet we aren't handling...
    return SendPacket ("");
}

size_t
GDBRemoteCommunicationServer::SendErrorResponse (uint8_t err)
{
    char packet[16];
    int packet_len = ::snprintf (packet, sizeof(packet), "E%2.2x", err);
    assert (packet_len < sizeof(packet));
    return SendPacket (packet, packet_len);
}


size_t
GDBRemoteCommunicationServer::SendOKResponse ()
{
    return SendPacket ("OK");
}

bool
GDBRemoteCommunicationServer::HandshakeWithClient(Error *error_ptr)
{
    if (StartReadThread(error_ptr))
        return GetAck();
    return false;
}

bool
GDBRemoteCommunicationServer::Handle_qHostInfo (StringExtractorGDBRemote &packet)
{
    StreamString response;
    
    // $cputype:16777223;cpusubtype:3;ostype:Darwin;vendor:apple;endian:little;ptrsize:8;#00

    ArchSpec host_arch (Host::GetArchitecture ());
    const llvm::Triple &host_triple = host_arch.GetTriple();
    response.PutCString("triple:");
    response.PutCStringAsRawHex8(host_triple.getTriple().c_str());
    response.Printf (";ptrsize:%u;",host_arch.GetAddressByteSize());

    uint32_t cpu = host_arch.GetMachOCPUType();
    uint32_t sub = host_arch.GetMachOCPUSubType();
    if (cpu != LLDB_INVALID_CPUTYPE)
        response.Printf ("cputype:%u;", cpu);
    if (sub != LLDB_INVALID_CPUTYPE)
        response.Printf ("cpusubtype:%u;", sub);
    
    switch (lldb::endian::InlHostByteOrder())
    {
    case eByteOrderBig:     response.PutCString ("endian:big;"); break;
    case eByteOrderLittle:  response.PutCString ("endian:little;"); break;
    case eByteOrderPDP:     response.PutCString ("endian:pdp;"); break;
    default:                response.PutCString ("endian:unknown;"); break;
    }
    
    uint32_t major = UINT32_MAX;
    uint32_t minor = UINT32_MAX;
    uint32_t update = UINT32_MAX;
    if (Host::GetOSVersion (major, minor, update))
    {
        if (major != UINT32_MAX)
        {
            response.Printf("os_version:%u", major);
            if (minor != UINT32_MAX)
            {
                response.Printf(".%u", minor);
                if (update != UINT32_MAX)
                    response.Printf(".%u", update);
            }
            response.PutChar(';');
        }
    }

    std::string s;
    if (Host::GetOSBuildString (s))
    {
        response.PutCString ("os_build:");
        response.PutCStringAsRawHex8(s.c_str());
        response.PutChar(';');
    }
    if (Host::GetOSKernelDescription (s))
    {
        response.PutCString ("os_kernel:");
        response.PutCStringAsRawHex8(s.c_str());
        response.PutChar(';');
    }
    if (Host::GetHostname (s))
    {
        response.PutCString ("hostname:");
        response.PutCStringAsRawHex8(s.c_str());
        response.PutChar(';');
    }
    
    return SendPacket (response) > 0;
}

static void
CreateProcessInfoResponse (const ProcessInfo &proc_info, StreamString &response)
{
    response.Printf ("pid:%i;ppid:%i;uid:%i;gid:%i;euid:%i;egid:%i;", 
                     proc_info.GetProcessID(),
                     proc_info.GetParentProcessID(),
                     proc_info.GetRealUserID(),
                     proc_info.GetRealGroupID(),
                     proc_info.GetEffectiveUserID(),
                     proc_info.GetEffectiveGroupID());
    response.PutCString ("name:");
    response.PutCStringAsRawHex8(proc_info.GetName());
    response.PutChar(';');
    const ArchSpec &proc_arch = proc_info.GetArchitecture();
    if (proc_arch.IsValid())
    {
        const llvm::Triple &proc_triple = proc_arch.GetTriple();
        response.PutCString("triple:");
        response.PutCStringAsRawHex8(proc_triple.getTriple().c_str());
        response.PutChar(';');
    }
}

bool
GDBRemoteCommunicationServer::Handle_qProcessInfoPID (StringExtractorGDBRemote &packet)
{
    // Packet format: "qProcessInfoPID:%i" where %i is the pid
    packet.SetFilePos(strlen ("qProcessInfoPID:"));
    lldb::pid_t pid = packet.GetU32 (LLDB_INVALID_PROCESS_ID);
    if (pid != LLDB_INVALID_PROCESS_ID)
    {
        ProcessInfo proc_info;
        if (Host::GetProcessInfo(pid, proc_info))
        {
            StreamString response;
            CreateProcessInfoResponse (proc_info, response);
            return SendPacket (response);
        }
    }
    return SendErrorResponse (1);
}

bool 
GDBRemoteCommunicationServer::Handle_qfProcessInfo (StringExtractorGDBRemote &packet)
{
    m_proc_infos_index = 0;
    m_proc_infos.Clear();

    ProcessInfoMatch match_info;
    packet.SetFilePos(strlen ("qfProcessInfo"));
    if (packet.GetChar() == ':')
    {
    
        std::string key;
        std::string value;
        while (packet.GetNameColonValue(key, value))
        {
            bool success = true;
            if (key.compare("name") == 0)
            {
                StringExtractor extractor;
                extractor.GetStringRef().swap(value);
                extractor.GetHexByteString (value);
                match_info.GetProcessInfo().SetName (value.c_str());
            }
            else if (key.compare("name_match") == 0)
            {
                if (value.compare("equals") == 0)
                {
                    match_info.SetNameMatchType (eNameMatchEquals);
                }
                else if (value.compare("starts_with") == 0)
                {
                    match_info.SetNameMatchType (eNameMatchStartsWith);
                }
                else if (value.compare("ends_with") == 0)
                {
                    match_info.SetNameMatchType (eNameMatchEndsWith);
                }
                else if (value.compare("contains") == 0)
                {
                    match_info.SetNameMatchType (eNameMatchContains);
                }
                else if (value.compare("regex") == 0)       
                {
                    match_info.SetNameMatchType (eNameMatchRegularExpression);
                }
                else 
                {
                    success = false;
                }
            }
            else if (key.compare("pid") == 0)
            {
                match_info.GetProcessInfo().SetProcessID (Args::StringToUInt32(value.c_str(), LLDB_INVALID_PROCESS_ID, 0, &success));
            }
            else if (key.compare("parent_pid") == 0)
            {
                match_info.GetProcessInfo().SetParentProcessID (Args::StringToUInt32(value.c_str(), LLDB_INVALID_PROCESS_ID, 0, &success));
            }
            else if (key.compare("uid") == 0)
            {
                match_info.GetProcessInfo().SetRealUserID (Args::StringToUInt32(value.c_str(), UINT32_MAX, 0, &success));
            }
            else if (key.compare("gid") == 0)
            {
                match_info.GetProcessInfo().SetRealGroupID (Args::StringToUInt32(value.c_str(), UINT32_MAX, 0, &success));
            }
            else if (key.compare("euid") == 0)
            {
                match_info.GetProcessInfo().SetEffectiveUserID (Args::StringToUInt32(value.c_str(), UINT32_MAX, 0, &success));
            }
            else if (key.compare("egid") == 0)
            {
                match_info.GetProcessInfo().SetEffectiveGroupID (Args::StringToUInt32(value.c_str(), UINT32_MAX, 0, &success));
            }
            else if (key.compare("all_users") == 0)
            {
                match_info.SetMatchAllUsers(Args::StringToBoolean(value.c_str(), false, &success));
            }
            else if (key.compare("triple") == 0)
            {
                match_info.GetProcessInfo().GetArchitecture().SetTriple (value.c_str(), NULL);
            }
            else
            {
                success = false;
            }
            
            if (!success)
                return SendErrorResponse (2);
        }
    }

    if (Host::FindProcesses (match_info, m_proc_infos))
    {
        // We found something, return the first item by calling the get
        // subsequent process info packet handler...
        return Handle_qsProcessInfo (packet);
    }
    return SendErrorResponse (3);
}

bool 
GDBRemoteCommunicationServer::Handle_qsProcessInfo (StringExtractorGDBRemote &packet)
{
    if (m_proc_infos_index < m_proc_infos.GetSize())
    {
        StreamString response;
        CreateProcessInfoResponse (m_proc_infos.GetProcessInfoAtIndex(m_proc_infos_index), response);
        ++m_proc_infos_index;
        return SendPacket (response);
    }
    return SendErrorResponse (4);
}

bool 
GDBRemoteCommunicationServer::Handle_qUserName (StringExtractorGDBRemote &packet)
{
    // Packet format: "qUserName:%i" where %i is the uid
    packet.SetFilePos(strlen ("qUserName:"));
    uint32_t uid = packet.GetU32 (UINT32_MAX);
    if (uid != UINT32_MAX)
    {
        std::string name;
        if (Host::GetUserName (uid, name))
        {
            StreamString response;
            response.PutCStringAsRawHex8 (name.c_str());
            return SendPacket (response);
        }
    }
    return SendErrorResponse (5);
    
}

bool 
GDBRemoteCommunicationServer::Handle_qGroupName (StringExtractorGDBRemote &packet)
{
    // Packet format: "qGroupName:%i" where %i is the gid
    packet.SetFilePos(strlen ("qGroupName:"));
    uint32_t gid = packet.GetU32 (UINT32_MAX);
    if (gid != UINT32_MAX)
    {
        std::string name;
        if (Host::GetGroupName (gid, name))
        {
            StreamString response;
            response.PutCStringAsRawHex8 (name.c_str());
            return SendPacket (response);
        }
    }
    return SendErrorResponse (6);
}

bool
GDBRemoteCommunicationServer::Handle_qSpeedTest (StringExtractorGDBRemote &packet)
{
    packet.SetFilePos(strlen ("qSpeedTest:"));

    std::string key;
    std::string value;
    bool success = packet.GetNameColonValue(key, value);
    if (success && key.compare("response_size") == 0)
    {
        uint32_t response_size = Args::StringToUInt32(value.c_str(), 0, 0, &success);
        if (success)
        {
            if (response_size == 0)
                return SendOKResponse();
            StreamString response;
            uint32_t bytes_left = response_size;
            response.PutCString("data:");
            while (bytes_left > 0)
            {
                if (bytes_left >= 26)
                {
                    response.PutCString("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
                    bytes_left -= 26;
                }
                else
                {
                    response.Printf ("%*.*s;", bytes_left, bytes_left, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
                    bytes_left = 0;
                }
            }
            return SendPacket (response);
        }
    }
    return SendErrorResponse (7);
}
bool
GDBRemoteCommunicationServer::Handle_QStartNoAckMode (StringExtractorGDBRemote &packet)
{
    // Send response first before changing m_send_acks to we ack this packet
    SendOKResponse ();
    m_send_acks = false;
    return true;
}
