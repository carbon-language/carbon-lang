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
// Other libraries and framework includes
// Project includes
#include "GDBRemoteCommunication.h"

class ProcessGDBRemote;

class GDBRemoteCommunicationServer : public GDBRemoteCommunication
{
public:
    enum
    {
        eBroadcastBitRunPacketSent = kLoUserBroadcastBit
    };
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    GDBRemoteCommunicationServer();

    virtual
    ~GDBRemoteCommunicationServer();

    bool
    GetPacketAndSendResponse (const lldb_private::TimeValue* timeout_ptr,
                              lldb_private::Error &error,
                              bool &interrupt, 
                              bool &quit);

    virtual bool
    GetThreadSuffixSupported ()
    {
        return true;
    }

    // After connecting, do a little handshake with the client to make sure
    // we are at least communicating
    bool
    HandshakeWithClient (lldb_private::Error *error_ptr);

protected:
    lldb::thread_t m_async_thread;

    size_t
    SendUnimplementedResponse ();

    size_t
    SendOKResponse ();

    bool
    Handle_qHostInfo ();

    bool
    Handle_QStartNoAckMode ();

private:
    //------------------------------------------------------------------
    // For GDBRemoteCommunicationServer only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunicationServer);
};

#endif  // liblldb_GDBRemoteCommunicationServer_h_
