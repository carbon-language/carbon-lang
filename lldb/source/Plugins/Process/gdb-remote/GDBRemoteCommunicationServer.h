//===-- GDBRemoteCommunicationServer.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GDBRemoteCommunicationServer_h_
#define liblldb_GDBRemoteCommunicationServer_h_

// C Includes
// C++ Includes
#include <functional>
#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private-forward.h"
#include "GDBRemoteCommunication.h"

class ProcessGDBRemote;
class StringExtractorGDBRemote;

class GDBRemoteCommunicationServer :
    public GDBRemoteCommunication
{
public:
    using PortMap = std::map<uint16_t, lldb::pid_t>;
    using PacketHandler = std::function<PacketResult(StringExtractorGDBRemote &packet,
                                                     lldb_private::Error &error,
                                                     bool &interrupt,
                                                     bool &quit)>;

    GDBRemoteCommunicationServer(const char *comm_name,
                                 const char *listener_name);

    virtual
    ~GDBRemoteCommunicationServer();

    void RegisterPacketHandler(StringExtractorGDBRemote::ServerPacketType packet_type,
                               PacketHandler handler);

    PacketResult
    GetPacketAndSendResponse (uint32_t timeout_usec,
                              lldb_private::Error &error,
                              bool &interrupt, 
                              bool &quit);

    // After connecting, do a little handshake with the client to make sure
    // we are at least communicating
    bool
    HandshakeWithClient (lldb_private::Error *error_ptr);

protected:
    std::map<StringExtractorGDBRemote::ServerPacketType, PacketHandler> m_packet_handlers;
    bool m_exit_now; // use in asynchronous handling to indicate process should exit.

    PacketResult
    SendUnimplementedResponse (const char *packet);

    PacketResult
    SendErrorResponse (uint8_t error);

    PacketResult
    SendIllFormedResponse (const StringExtractorGDBRemote &packet, const char *error_message);

    PacketResult
    SendOKResponse ();

private:
    //------------------------------------------------------------------
    // For GDBRemoteCommunicationServer only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunicationServer);
};

#endif  // liblldb_GDBRemoteCommunicationServer_h_
