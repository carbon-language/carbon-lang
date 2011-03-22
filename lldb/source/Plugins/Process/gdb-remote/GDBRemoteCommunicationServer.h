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
    GetPacketAndSendResponse (const lldb_private::TimeValue* timeout_time_ptr);

    virtual bool
    GetThreadSuffixSupported ()
    {
        return true;
    }

    virtual bool
    GetSendAcks ()
    {
        return m_send_acks;
    }

protected:
    lldb::thread_t m_async_thread;
    bool m_send_acks;

    size_t
    SendUnimplementedResponse ();


    bool
    Handle_qHostInfo ();

private:
    //------------------------------------------------------------------
    // For GDBRemoteCommunicationServer only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunicationServer);
};

#endif  // liblldb_GDBRemoteCommunicationServer_h_
