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
    m_async_thread (LLDB_INVALID_HOST_THREAD),
    m_send_acks (true)
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
GDBRemoteCommunicationServer::GetPacketAndSendResponse (const TimeValue* timeout_time_ptr)
{
    StringExtractorGDBRemote packet;
    if (WaitForPacketNoLock (packet, timeout_time_ptr))
    {
        const StringExtractorGDBRemote::ServerPacketType packet_type = packet.GetServerPacketType ();
        switch (packet_type)
        {
        case StringExtractorGDBRemote::eServerPacketType_nack:
        case StringExtractorGDBRemote::eServerPacketType_ack:
            break;

        case StringExtractorGDBRemote::eServerPacketType_invalid:
        case StringExtractorGDBRemote::eServerPacketType_unimplemented:
            return SendUnimplementedResponse () > 0;

        case StringExtractorGDBRemote::eServerPacketType_qHostInfo:
            return Handle_qHostInfo ();
        }
        return true;
    }
    return false;
}

size_t
GDBRemoteCommunicationServer::SendUnimplementedResponse ()
{
    return SendPacket ("");
}


bool
GDBRemoteCommunicationServer::Handle_qHostInfo ()
{
    StreamString response;
    
    // $cputype:16777223;cpusubtype:3;ostype:Darwin;vendor:apple;endian:little;ptrsize:8;#00

    ArchSpec host_arch (Host::GetArchitecture ());
    
    const llvm::Triple &host_triple = host_arch.GetTriple();
    const llvm::StringRef arch_name (host_triple.getArchName());
    const llvm::StringRef vendor_name (host_triple.getOSName());
    const llvm::StringRef os_name (host_triple.getVendorName());
    response.Printf ("arch:%.*s;ostype:%.*s;vendor:%.*s;ptrsize:%u", 
                     (int)arch_name.size(), arch_name.data(),
                     (int)os_name.size(), os_name.data(),
                     (int)vendor_name.size(), vendor_name.data(),
                     host_arch.GetAddressByteSize());

    switch (lldb::endian::InlHostByteOrder())
    {
    case eByteOrderBig:     response.PutCString ("endian:big;"); break;
    case eByteOrderLittle:  response.PutCString ("endian:little;"); break;
    case eByteOrderPDP:     response.PutCString ("endian:pdp;"); break;
    default:                response.PutCString ("endian:unknown;"); break;
    }
    
    return SendPacket (response.GetString().c_str(),response.GetString().size()) > 0;
}
