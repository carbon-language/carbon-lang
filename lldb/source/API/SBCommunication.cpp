//===-- SBCommunication.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBCommunication.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/Core/Communication.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Log.h"

using namespace lldb;
using namespace lldb_private;



SBCommunication::SBCommunication() :
    m_opaque (NULL),
    m_opaque_owned (false)
{
}

SBCommunication::SBCommunication(const char * broadcaster_name) :
    m_opaque (new Communication (broadcaster_name)),
    m_opaque_owned (true)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBCommunication::SBCommunication (broadcaster_name='%s') => "
                     "this.obj = %p", broadcaster_name, m_opaque);
}

SBCommunication::~SBCommunication()
{
    if (m_opaque && m_opaque_owned)
        delete m_opaque;
    m_opaque = NULL;
    m_opaque_owned = false;
}

ConnectionStatus
SBCommunication::CheckIfBytesAvailable ()
{
    if (m_opaque)
        return m_opaque->BytesAvailable (0, NULL);
    return eConnectionStatusNoConnection;
}

ConnectionStatus
SBCommunication::WaitForBytesAvailableInfinite ()
{
    if (m_opaque)
        return m_opaque->BytesAvailable (UINT32_MAX, NULL);
    return eConnectionStatusNoConnection;
}

ConnectionStatus
SBCommunication::WaitForBytesAvailableWithTimeout (uint32_t timeout_usec)
{
    if (m_opaque)
        return m_opaque->BytesAvailable (timeout_usec, NULL);
    return eConnectionStatusNoConnection;
}

ConnectionStatus
SBCommunication::Connect (const char *url)
{
    if (m_opaque)
    {
        if (!m_opaque->HasConnection ())
            m_opaque->SetConnection (new ConnectionFileDescriptor());
        return m_opaque->Connect (url, NULL);
    }
    return eConnectionStatusNoConnection;
}

ConnectionStatus
SBCommunication::AdoptFileDesriptor (int fd, bool owns_fd)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBCommunication::AdoptFileDescriptor (this=%p, fd='%d', owns_fd='%s')", this, fd, 
    //                 (owns_fd ? "true" : "false"));

    if (m_opaque)
    {
        if (m_opaque->HasConnection ())
        {
            if (m_opaque->IsConnected())
                m_opaque->Disconnect();
        }
        m_opaque->SetConnection (new ConnectionFileDescriptor (fd, owns_fd));
        if (m_opaque->IsConnected())
        {
            if (log)
                log->Printf ("SBCommunication::AdoptFileDescriptor (this.obj=%p, fd=%d, ownd_fd='%s') "
                             "=> eConnectionStatusSuccess", m_opaque, fd, (owns_fd ? "true" : "false"));
            return eConnectionStatusSuccess;
        }
        else
        { 
            if (log)
                log->Printf ("SBCommunication::AdoptFileDescriptor (this.obj=%p, fd=%d, ownd_fd='%s') "
                             "=> eConnectionStatusLostConnection", m_opaque, fd, (owns_fd ? "true" : "false"));
            return eConnectionStatusLostConnection;
        }
    }

    if (log)
        log->Printf ("SBCommunication::AdoptFileDescriptor (this,obj=%p, fd=%d, ownd_fd='%s') "
                     "=> eConnectionStatusNoConnection", m_opaque, fd, (owns_fd ? "true" : "false"));

    return eConnectionStatusNoConnection;
}


ConnectionStatus
SBCommunication::Disconnect ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBCommunication::Disconnect ()");

    ConnectionStatus status= eConnectionStatusNoConnection;
    if (m_opaque)
        status = m_opaque->Disconnect ();

    if (log)
        log->Printf ("SBCommunication::Disconnect (this.obj=%p) => '%s'", m_opaque,
                     Communication::ConnectionStatusAsCString (status));

    return status;
}

bool
SBCommunication::IsConnected () const
{
    if (m_opaque)
        return m_opaque->IsConnected ();
    return false;
}

size_t
SBCommunication::Read (void *dst, size_t dst_len, uint32_t timeout_usec, ConnectionStatus &status)
{
    if (m_opaque)
        return m_opaque->Read (dst, dst_len, timeout_usec, status, NULL);
    status = eConnectionStatusNoConnection;
    return 0;
}


size_t
SBCommunication::Write (const void *src, size_t src_len, ConnectionStatus &status)
{
    if (m_opaque)
        return m_opaque->Write (src, src_len, status, NULL);
    status = eConnectionStatusNoConnection;
    return 0;
}

bool
SBCommunication::ReadThreadStart ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    //if (log)
    //    log->Printf ("SBCommunication::ReadThreadStart ()");

    bool success = false;
    if (m_opaque)
        success = m_opaque->StartReadThread ();

    log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    if (log)
        log->Printf ("SBCommunication::ReadThreadStart (this.obj=%p) => '%s'", m_opaque, (success ? "true" : "false"));

    return success;
}


bool
SBCommunication::ReadThreadStop ()
{

    //if (log)
    //    log->Printf ("SBCommunication::ReadThreadStop ()");

    bool success = false;
    if (m_opaque)
        success = m_opaque->StopReadThread ();

    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    if (log)
        log->Printf ("SBCommunication::ReadThreadStop (this.obj=%p) => '%s'", m_opaque, (success ? "true" : "false"));

    return success;
}

bool
SBCommunication::ReadThreadIsRunning ()
{
    if (m_opaque)
        return m_opaque->ReadThreadIsRunning ();
    return false;
}

bool
SBCommunication::SetReadThreadBytesReceivedCallback
(
    ReadThreadBytesReceived callback,
    void *callback_baton
)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBCommunication::SetReadThreadBytesReceivedCallback (this.obj=%p, callback=%p, baton=%p)",
                     m_opaque, callback, callback_baton);

    if (m_opaque)
    {
        m_opaque->SetReadThreadBytesReceivedCallback (callback, callback_baton);
        if (log)
            log->Printf ("SBCommunication::SetReaDThreadBytesReceivedCallback (this.obj=%p...) => true", m_opaque);
        return true;
    }

    if (log)
        log->Printf ("SBCommunication::SetReaDThreadBytesReceivedCallback (this.obj=%p...) => false", m_opaque);

    return false;
}

SBBroadcaster
SBCommunication::GetBroadcaster ()
{
    SBBroadcaster broadcaster (m_opaque, false);
    return broadcaster;
}


//
//void
//SBCommunication::CreateIfNeeded ()
//{
//    if (m_opaque == NULL)
//    {
//        static uint32_t g_broadcaster_num;
//        char broadcaster_name[256];
//        ::snprintf (name, broadcaster_name, "%p SBCommunication", this);
//        m_opaque = new Communication (broadcaster_name);
//        m_opaque_owned = true;
//    }
//    assert (m_opaque);
//}
//
//
