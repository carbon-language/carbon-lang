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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBCommunication::SBCommunication () ==> this = %p", this);
}

SBCommunication::SBCommunication(const char * broadcaster_name) :
    m_opaque (new Communication (broadcaster_name)),
    m_opaque_owned (true)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBCommunication::SBCommunication (const char *broadcaster_name) broadcaster_name = '%s' ==> "
                     "this = %p (m_opaque = %p)", broadcaster_name, this, m_opaque);
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

    if (log)
        log->Printf ("SBCommunication::AdoptFileDescriptor (%d, %s)", fd, (owns_fd ? "true" : "false"));

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
                log->Printf ("SBCommunication::AdoptFileDescriptor ==> eConnectionStatusSuccess");
            return eConnectionStatusSuccess;
        }
        else
        {
            if (log)
                log->Printf ("SBCommunication::AdoptFileDescriptor ==> eConnectionStatusLostConnection");
            return eConnectionStatusLostConnection;
        }
    }

    if (log)
        log->Printf ("SBCommunication::AdoptFileDescriptor ==> eConnectionStatusNoConnection");

    return eConnectionStatusNoConnection;
}


ConnectionStatus
SBCommunication::Disconnect ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBCommunication::Disconnect ()");

    ConnectionStatus status= eConnectionStatusNoConnection;
    if (m_opaque)
        status = m_opaque->Disconnect ();

    if (log)
        log->Printf ("SBCommunication::Disconnect ==> %s", Communication::ConnectionStatusAsCString (status));

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

    if (log)
        log->Printf ("SBCommunication::ReadThreadStart ()");

    bool success = false;
    if (m_opaque)
        success = m_opaque->StartReadThread ();

    if (log)
        log->Printf ("SBCommunication::ReadThreadStart ==> %s", (success ? "true" : "false"));

    return success;
}


bool
SBCommunication::ReadThreadStop ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBCommunication::ReadThreadStop ()");

    bool success = false;
    if (m_opaque)
        success = m_opaque->StopReadThread ();

    if (log)
        log->Printf ("SBCommunication::ReadThreadStop ==> %s", (success ? "true" : "false"));

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
        log->Printf ("SBCommunication::SetReadThreadBytesReceivedCallback (callback, baton)");
    // CAROLINE: Fixme: Fix the arguments printed out in the log message above

    if (m_opaque)
    {
        m_opaque->SetReadThreadBytesReceivedCallback (callback, callback_baton);
        if (log)
            log->Printf ("SBCommunication::SetReaDThreadBytesReceivedCallback ==> true");
        return true;
    }

    if (log)
        log->Printf ("SBCommunication::SetReaDThreadBytesReceivedCallback ==> false");

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
