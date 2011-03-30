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
#include "lldb/Target/Process.h"

#include "GDBRemoteCommunication.h"

class ProcessGDBRemote;
class StringExtractorGDBRemote;

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
    lldb_private::ProcessInfoList m_proc_infos;
    uint32_t m_proc_infos_index;

    size_t
    SendUnimplementedResponse (const char *packet);

    size_t
    SendErrorResponse (uint8_t error);

    size_t
    SendOKResponse ();

    bool
    Handle_qHostInfo (StringExtractorGDBRemote &packet);
    
    bool
    Handle_qProcessInfoPID (StringExtractorGDBRemote &packet);
    
    bool
    Handle_qfProcessInfo (StringExtractorGDBRemote &packet);
    
    bool 
    Handle_qsProcessInfo (StringExtractorGDBRemote &packet);

    bool 
    Handle_qUserName (StringExtractorGDBRemote &packet);

    bool 
    Handle_qGroupName (StringExtractorGDBRemote &packet);

    bool
    Handle_QStartNoAckMode (StringExtractorGDBRemote &packet);

private:
    //------------------------------------------------------------------
    // For GDBRemoteCommunicationServer only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunicationServer);
};

#endif  // liblldb_GDBRemoteCommunicationServer_h_
