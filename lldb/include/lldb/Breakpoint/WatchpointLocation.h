//===-- WatchpointLocation.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_WatchpointLocation_h_
#define liblldb_WatchpointLocation_h_

// C Includes

// C++ Includes
#include <list>

// Other libraries and framework includes

// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/UserID.h"
#include "lldb/Breakpoint/StoppointLocation.h"

namespace lldb_private {

class WatchpointLocation :
    public StoppointLocation
{
public:

    WatchpointLocation (lldb::addr_t m_addr, size_t size, bool hardware);

    ~WatchpointLocation ();

    bool
    IsEnabled () const;

    void
    SetEnabled (bool enabled);

    bool        WatchpointRead () const;
    bool        WatchpointWrite () const;
    uint32_t    GetIgnoreCount () const;
    void        SetIgnoreCount (uint32_t n);
    void        SetWatchpointType (uint32_t type);
    bool        BreakpointWasHit (StoppointCallbackContext *context);
    bool        SetCallback (WatchpointHitCallback callback, void *callback_baton);
    void        Dump (Stream *s) const;

private:
    bool        m_enabled;          // Is this breakpoint enabled
    uint32_t    m_watch_read:1,     // 1 if we stop when the watched data is read from
                m_watch_write:1,    // 1 if we stop when the watched data is written to
                m_watch_was_read:1, // Set to 1 when watchpoint is hit for a read access
                m_watch_was_written:1;  // Set to 1 when watchpoint is hit for a write access
    uint32_t    m_ignore_count;     // Number of times to ignore this breakpoint
    WatchpointHitCallback m_callback;
    void *      m_callback_baton;   // Callback user data to pass to callback

    static lldb::break_id_t
    GetNextID();

    DISALLOW_COPY_AND_ASSIGN (WatchpointLocation);
};

} // namespace lldb_private

#endif  // liblldb_WatchpointLocation_h_
