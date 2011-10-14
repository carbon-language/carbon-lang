//===-- Watchpoint.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Watchpoint_h_
#define liblldb_Watchpoint_h_

// C Includes

// C++ Includes
#include <list>
#include <string>

// Other libraries and framework includes

// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/Target.h"
#include "lldb/Core/UserID.h"
#include "lldb/Breakpoint/StoppointLocation.h"

namespace lldb_private {

class Watchpoint :
    public StoppointLocation
{
public:

    Watchpoint (lldb::addr_t addr, size_t size, bool hardware = true);

    ~Watchpoint ();

    bool
    IsEnabled () const;

    void
    SetEnabled (bool enabled);

    virtual bool
    IsHardware () const;

    virtual bool
    ShouldStop (StoppointCallbackContext *context);

    bool        WatchpointRead () const;
    bool        WatchpointWrite () const;
    uint32_t    GetIgnoreCount () const;
    void        SetIgnoreCount (uint32_t n);
    void        SetWatchpointType (uint32_t type);
    bool        SetCallback (WatchpointHitCallback callback, void *callback_baton);
    void        SetDeclInfo (std::string &str);
    void        GetDescription (Stream *s, lldb::DescriptionLevel level);
    void        Dump (Stream *s) const;
    void        DumpWithLevel (Stream *s, lldb::DescriptionLevel description_level) const;
    Target      &GetTarget() { return *m_target; }
    const Error &GetError() { return m_error; }

private:
    friend class Target;

    void        SetTarget(Target *target_ptr) { m_target = target_ptr; }

    Target      *m_target;
    bool        m_enabled;             // Is this watchpoint enabled
    bool        m_is_hardware;         // Is this a hardware watchpoint
    uint32_t    m_watch_read:1,        // 1 if we stop when the watched data is read from
                m_watch_write:1,       // 1 if we stop when the watched data is written to
                m_watch_was_read:1,    // Set to 1 when watchpoint is hit for a read access
                m_watch_was_written:1; // Set to 1 when watchpoint is hit for a write access
    uint32_t    m_ignore_count;        // Number of times to ignore this breakpoint
    WatchpointHitCallback m_callback;
    void *      m_callback_baton;      // Callback user data to pass to callback
    std::string m_decl_str;            // Declaration information, if any.
    Error       m_error;               // An error object describing errors creating watchpoint.


    static lldb::break_id_t
    GetNextID();

    DISALLOW_COPY_AND_ASSIGN (Watchpoint);
};

} // namespace lldb_private

#endif  // liblldb_Watchpoint_h_
