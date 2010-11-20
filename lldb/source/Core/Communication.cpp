//===-- Communication.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private-log.h"
#include "lldb/Core/Communication.h"
#include "lldb/Core/Connection.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/Event.h"
#include <string.h>

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
Communication::Communication(const char *name) :
    Broadcaster (name),
    m_connection_ap (),
    m_read_thread (LLDB_INVALID_HOST_THREAD),
    m_read_thread_enabled (false),
    m_bytes(),
    m_bytes_mutex (Mutex::eMutexTypeRecursive),
    m_callback (NULL),
    m_callback_baton (NULL)

{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_OBJECT | LIBLLDB_LOG_COMMUNICATION,
                                 "%p Communication::Communication (name = %s)",
                                 this, name);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Communication::~Communication()
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_OBJECT | LIBLLDB_LOG_COMMUNICATION,
                                 "%p Communication::~Communication (name = %s)",
                                 this, m_broadcaster_name.AsCString(""));
    Clear();
}

void
Communication::Clear()
{
    StopReadThread (NULL);
    Disconnect (NULL);
}

ConnectionStatus
Communication::BytesAvailable (uint32_t timeout_usec, Error *error_ptr)
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_COMMUNICATION, "%p Communication::BytesAvailable (timeout_usec = %u)", this, timeout_usec);

    if (m_connection_ap.get())
        return m_connection_ap->BytesAvailable (timeout_usec, error_ptr);
    if (error_ptr)
        error_ptr->SetErrorString("Invalid connection.");
    return eConnectionStatusNoConnection;
}

ConnectionStatus
Communication::Connect (const char *url, Error *error_ptr)
{
    Clear();

    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_COMMUNICATION, "%p Communication::Connect (url = %s)", this, url);

    if (m_connection_ap.get())
        return m_connection_ap->Connect (url, error_ptr);
    if (error_ptr)
        error_ptr->SetErrorString("Invalid connection.");
    return eConnectionStatusNoConnection;
}

ConnectionStatus
Communication::Disconnect (Error *error_ptr)
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_COMMUNICATION, "%p Communication::Disconnect ()", this);

    if (m_connection_ap.get())
    {
        ConnectionStatus status = m_connection_ap->Disconnect (error_ptr);
        m_connection_ap.reset();
        return status;
    }
    return eConnectionStatusNoConnection;
}

bool
Communication::IsConnected () const
{
    if (m_connection_ap.get())
        return m_connection_ap->IsConnected ();
    return false;
}

bool
Communication::HasConnection () const
{
    return m_connection_ap.get() != NULL;
}

size_t
Communication::Read (void *dst, size_t dst_len, uint32_t timeout_usec, ConnectionStatus &status, Error *error_ptr)
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_COMMUNICATION,
                                 "%p Communication::Write (dst = %p, dst_len = %zu, timeout_usec = %u) connection = %p",
                                 this, dst, dst_len, timeout_usec, m_connection_ap.get());

    if (m_read_thread != LLDB_INVALID_HOST_THREAD)
    {
        // We have a dedicated read thread that is getting data for us
        size_t cached_bytes = GetCachedBytes (dst, dst_len);
        if (cached_bytes > 0 || timeout_usec == 0)
        {
            status = eConnectionStatusSuccess;
            return cached_bytes;
        }

        if (m_connection_ap.get() == NULL)
        {
            if (error_ptr)
                error_ptr->SetErrorString("Invalid connection.");
            status = eConnectionStatusNoConnection;
            return 0;
        }
        // Set the timeout appropriately
        TimeValue timeout_time;
        if (timeout_usec != UINT32_MAX)
        {
            timeout_time = TimeValue::Now();
            timeout_time.OffsetWithMicroSeconds (timeout_usec);
        }

        Listener listener ("Communication::Read");
        listener.StartListeningForEvents (this, eBroadcastBitReadThreadGotBytes | eBroadcastBitReadThreadDidExit);
        EventSP event_sp;
        while (listener.WaitForEvent (timeout_time.IsValid() ? &timeout_time : NULL, event_sp))
        {
            const uint32_t event_type = event_sp->GetType();
            if (event_type & eBroadcastBitReadThreadGotBytes)
            {
                return GetCachedBytes (dst, dst_len);
            }

            if (event_type & eBroadcastBitReadThreadDidExit)
            {
                Disconnect (NULL);
                break;
            }
        }
        return 0;
    }

    // We aren't using a read thread, just read the data synchronously in this
    // thread.
    if (m_connection_ap.get())
    {
        status = m_connection_ap->BytesAvailable (timeout_usec, error_ptr);
        if (status == eConnectionStatusSuccess)
            return m_connection_ap->Read (dst, dst_len, status, error_ptr);
    }

    if (error_ptr)
        error_ptr->SetErrorString("Invalid connection.");
    status = eConnectionStatusNoConnection;
    return 0;
}


size_t
Communication::Write (const void *src, size_t src_len, ConnectionStatus &status, Error *error_ptr)
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_COMMUNICATION,
                                 "%p Communication::Write (src = %p, src_len = %zu) connection = %p",
                                 this, src, src_len, m_connection_ap.get());

    if (m_connection_ap.get())
        return m_connection_ap->Write (src, src_len, status, error_ptr);

    if (error_ptr)
        error_ptr->SetErrorString("Invalid connection.");
    status = eConnectionStatusNoConnection;
    return 0;
}


bool
Communication::StartReadThread (Error *error_ptr)
{
    if (m_read_thread != LLDB_INVALID_HOST_THREAD)
        return true;

    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_COMMUNICATION,
                                 "%p Communication::StartReadThread ()", this);


    char thread_name[1024];
    snprintf(thread_name, sizeof(thread_name), "<lldb.comm.%s>", m_broadcaster_name.AsCString());

    m_read_thread_enabled = true;
    m_read_thread = Host::ThreadCreate (thread_name, Communication::ReadThread, this, error_ptr);
    if (m_read_thread == LLDB_INVALID_HOST_THREAD)
        m_read_thread_enabled = false;
    return m_read_thread_enabled;
}

bool
Communication::StopReadThread (Error *error_ptr)
{
    if (m_read_thread == LLDB_INVALID_HOST_THREAD)
        return true;

    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_COMMUNICATION,
                                 "%p Communication::StopReadThread ()", this);

    m_read_thread_enabled = false;

    BroadcastEvent (eBroadcastBitReadThreadShouldExit, NULL);

    Host::ThreadCancel (m_read_thread, error_ptr);

    return Host::ThreadJoin (m_read_thread, NULL, error_ptr);
    m_read_thread = LLDB_INVALID_HOST_THREAD;
}


size_t
Communication::GetCachedBytes (void *dst, size_t dst_len)
{
    Mutex::Locker locker(m_bytes_mutex);
    if (m_bytes.size() > 0)
    {
        // If DST is NULL and we have a thread, then return the number
        // of bytes that are available so the caller can call again
        if (dst == NULL)
            return m_bytes.size();

        const size_t len = std::min<size_t>(dst_len, m_bytes.size());

        ::memcpy (dst, m_bytes.c_str(), len);
        m_bytes.erase(m_bytes.begin(), m_bytes.begin() + len);

        return len;
    }
    return 0;
}

void
Communication::AppendBytesToCache (const uint8_t * bytes, size_t len, bool broadcast, ConnectionStatus status)
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_COMMUNICATION,
                                 "%p Communication::AppendBytesToCache (src = %p, src_len = %zu, broadcast = %i)",
                                 this, bytes, len, broadcast);
    if (bytes == NULL || len == 0)
        return;
    if (m_callback)
    {
        // If the user registered a callback, then call it and do not broadcast
        m_callback (m_callback_baton, bytes, len);
    }
    else
    {
        Mutex::Locker locker(m_bytes_mutex);
        m_bytes.append ((const char *)bytes, len);
        if (broadcast)
            BroadcastEventIfUnique (eBroadcastBitReadThreadGotBytes);
    }
}

size_t
Communication::ReadFromConnection (void *dst, size_t dst_len, ConnectionStatus &status, Error *error_ptr)
{
    if (m_connection_ap.get())
        return m_connection_ap->Read (dst, dst_len, status, error_ptr);
    return 0;
}


bool
Communication::ReadThreadIsRunning ()
{
    return m_read_thread != LLDB_INVALID_HOST_THREAD;
}

void *
Communication::ReadThread (void *p)
{
    Communication *comm = (Communication *)p;

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_COMMUNICATION));

    if (log)
        log->Printf ("%p Communication::ReadThread () thread starting...", p);

    uint8_t buf[1024];

    Error error;
    ConnectionStatus status = eConnectionStatusSuccess;
    bool done = false;
    while (!done && comm->m_read_thread_enabled)
    {
        status = comm->BytesAvailable (UINT32_MAX, &error);

        if (status == eConnectionStatusSuccess)
        {
            size_t bytes_read = comm->ReadFromConnection (buf, sizeof(buf), status, &error);
            if (bytes_read > 0)
                comm->AppendBytesToCache (buf, bytes_read, true, status);
        }

        switch (status)
        {
        case eConnectionStatusSuccess:
            break;

        case eConnectionStatusEndOfFile:
        case eConnectionStatusNoConnection:     // No connection
        case eConnectionStatusLostConnection:   // Lost connection while connected to a valid connection
            done = true;
            // Fall through...
        default:
        case eConnectionStatusError:            // Check GetError() for details
        case eConnectionStatusTimedOut:         // Request timed out
            if (log)
                error.LogIfError(log.get(), "%p Communication::BytesAvailable () => status = %i", p, status);
            break;
        }
    }
    log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_COMMUNICATION);
    if (log)
        log->Printf ("%p Communication::ReadThread () thread exiting...", p);

    // Let clients know that this thread is exiting
    comm->BroadcastEvent (eBroadcastBitReadThreadDidExit);
    return NULL;
}

void
Communication::SetReadThreadBytesReceivedCallback
(
    ReadThreadBytesReceived callback,
    void *callback_baton
)
{
    m_callback = callback;
    m_callback_baton = callback_baton;
}

void
Communication::SetConnection (Connection *connection)
{
    StopReadThread(NULL);
    Disconnect (NULL);
    m_connection_ap.reset(connection);
}

const char *
Communication::ConnectionStatusAsCString (lldb::ConnectionStatus status)
{
    switch (status)
    {
    case eConnectionStatusSuccess:        return "success";
    case eConnectionStatusError:          return "error";
    case eConnectionStatusTimedOut:       return "timed out";
    case eConnectionStatusNoConnection:   return "no connection";
    case eConnectionStatusLostConnection: return "lost connection";
    }

    static char unknown_state_string[64];
    snprintf(unknown_state_string, sizeof (unknown_state_string), "ConnectionStatus = %i", status);
    return unknown_state_string;
}
