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
            return SendUnimplementedResponse () > 0;

        case StringExtractorGDBRemote::eServerPacketType_qHostInfo:
            return Handle_qHostInfo ();
            
        case StringExtractorGDBRemote::eServerPacketType_QStartNoAckMode:
            return Handle_QStartNoAckMode ();
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
GDBRemoteCommunicationServer::SendUnimplementedResponse ()
{
    return SendPacket ("");
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
GDBRemoteCommunicationServer::Handle_qHostInfo ()
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
    
    return SendPacket (response.GetString().c_str(),response.GetString().size()) > 0;
}


bool
GDBRemoteCommunicationServer::Handle_QStartNoAckMode ()
{
    SendOKResponse ();
    m_send_acks = false;
    return true;
}
