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

using namespace lldb;
using namespace lldb_private;



SBCommunication::SBCommunication() :
    m_lldb_object (NULL),
    m_lldb_object_owned (false)
{
}

SBCommunication::SBCommunication(const char * broadcaster_name) :
    m_lldb_object (new Communication (broadcaster_name)),
    m_lldb_object_owned (true)
{
}

SBCommunication::~SBCommunication()
{
    if (m_lldb_object && m_lldb_object_owned)
        delete m_lldb_object;
    m_lldb_object = NULL;
    m_lldb_object_owned = false;
}

ConnectionStatus
SBCommunication::CheckIfBytesAvailable ()
{
    if (m_lldb_object)
        return m_lldb_object->BytesAvailable (0, NULL);
    return eConnectionStatusNoConnection;
}

ConnectionStatus
SBCommunication::WaitForBytesAvailableInfinite ()
{
    if (m_lldb_object)
        return m_lldb_object->BytesAvailable (UINT32_MAX, NULL);
    return eConnectionStatusNoConnection;
}

ConnectionStatus
SBCommunication::WaitForBytesAvailableWithTimeout (uint32_t timeout_usec)
{
    if (m_lldb_object)
        return m_lldb_object->BytesAvailable (timeout_usec, NULL);
    return eConnectionStatusNoConnection;
}

ConnectionStatus
SBCommunication::Connect (const char *url)
{
    if (m_lldb_object)
    {
        if (!m_lldb_object->HasConnection ())
            m_lldb_object->SetConnection (new ConnectionFileDescriptor());
        return m_lldb_object->Connect (url, NULL);
    }
    return eConnectionStatusNoConnection;
}

ConnectionStatus
SBCommunication::AdoptFileDesriptor (int fd, bool owns_fd)
{
    if (m_lldb_object)
    {
        if (m_lldb_object->HasConnection ())
        {
            if (m_lldb_object->IsConnected())
                m_lldb_object->Disconnect ();
        }
        m_lldb_object->SetConnection (new ConnectionFileDescriptor (fd, owns_fd));
        if (m_lldb_object->IsConnected())
            return eConnectionStatusSuccess;
        else
            return eConnectionStatusLostConnection;
    }
    return eConnectionStatusNoConnection;
}


ConnectionStatus
SBCommunication::Disconnect ()
{
    if (m_lldb_object)
        return m_lldb_object->Disconnect ();
    return eConnectionStatusNoConnection;
}

bool
SBCommunication::IsConnected () const
{
    if (m_lldb_object)
        return m_lldb_object->IsConnected ();
    return false;
}

size_t
SBCommunication::Read (void *dst, size_t dst_len, uint32_t timeout_usec, ConnectionStatus &status)
{
    if (m_lldb_object)
        return m_lldb_object->Read (dst, dst_len, timeout_usec, status, NULL);
    status = eConnectionStatusNoConnection;
    return 0;
}


size_t
SBCommunication::Write (const void *src, size_t src_len, ConnectionStatus &status)
{
    if (m_lldb_object)
        return m_lldb_object->Write (src, src_len, status, NULL);
    status = eConnectionStatusNoConnection;
    return 0;
}

bool
SBCommunication::ReadThreadStart ()
{
    if (m_lldb_object)
        return m_lldb_object->StartReadThread ();
    return false;
}


bool
SBCommunication::ReadThreadStop ()
{
    if (m_lldb_object)
        return m_lldb_object->StopReadThread ();
    return false;
}

bool
SBCommunication::ReadThreadIsRunning ()
{
    if (m_lldb_object)
        return m_lldb_object->ReadThreadIsRunning ();
    return false;
}

bool
SBCommunication::SetReadThreadBytesReceivedCallback
(
    ReadThreadBytesReceived callback,
    void *callback_baton
)
{
    if (m_lldb_object)
    {
        m_lldb_object->SetReadThreadBytesReceivedCallback (callback, callback_baton);
        return true;
    }
    return false;
}

SBBroadcaster
SBCommunication::GetBroadcaster ()
{
    SBBroadcaster broadcaster (m_lldb_object, false);
    return broadcaster;
}


//
//void
//SBCommunication::CreateIfNeeded ()
//{
//    if (m_lldb_object == NULL)
//    {
//        static uint32_t g_broadcaster_num;
//        char broadcaster_name[256];
//        ::snprintf (name, broadcaster_name, "%p SBCommunication", this);
//        m_lldb_object = new Communication (broadcaster_name);
//        m_lldb_object_owned = true;
//    }
//    assert (m_lldb_object);
//}
//
//
