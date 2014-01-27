//===-- StreamBroadcast.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include "lldb/lldb-private.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/StreamAsynchronousIO.h"

using namespace lldb;
using namespace lldb_private;


StreamAsynchronousIO::StreamAsynchronousIO (Broadcaster &broadcaster, uint32_t broadcast_event_type) :
    Stream (0, 4, eByteOrderBig),
    m_broadcaster (broadcaster),
    m_broadcast_event_type (broadcast_event_type),
    m_accumulated_data ()
{
}

StreamAsynchronousIO::~StreamAsynchronousIO ()
{
    // Flush when we destroy to make sure we display the data
    Flush();
}

void
StreamAsynchronousIO::Flush ()
{
    if (!m_accumulated_data.empty())
    {
        std::unique_ptr<EventDataBytes> data_bytes_ap (new EventDataBytes);
        // Let's swap the bytes to avoid LARGE string copies.
        data_bytes_ap->SwapBytes (m_accumulated_data);
        EventSP new_event_sp (new Event (m_broadcast_event_type, data_bytes_ap.release()));
        m_broadcaster.BroadcastEvent (new_event_sp);
    }
}

size_t
StreamAsynchronousIO::Write (const void *s, size_t length)
{
    m_accumulated_data.append ((const char *)s, length);
    return length;
}
