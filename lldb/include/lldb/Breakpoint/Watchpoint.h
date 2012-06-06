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
    void        ClearCallback();
    void        SetDeclInfo (std::string &str);
    void        SetWatchSpec (std::string &str);
    void        GetDescription (Stream *s, lldb::DescriptionLevel level);
    void        Dump (Stream *s) const;
    void        DumpWithLevel (Stream *s, lldb::DescriptionLevel description_level) const;
    Target      &GetTarget() { return *m_target; }
    const Error &GetError() { return m_error; }

    //------------------------------------------------------------------
    /// Invoke the callback action when the watchpoint is hit.
    ///
    /// @param[in] context
    ///     Described the watchpoint event.
    ///
    /// @return
    ///     \b true if the target should stop at this watchpoint and \b false not.
    //------------------------------------------------------------------
    bool
    InvokeCallback (StoppointCallbackContext *context);

    //------------------------------------------------------------------
    // Condition
    //------------------------------------------------------------------
    //------------------------------------------------------------------
    /// Set the breakpoint's condition.
    ///
    /// @param[in] condition
    ///    The condition expression to evaluate when the breakpoint is hit.
    ///    Pass in NULL to clear the condition.
    //------------------------------------------------------------------
    void SetCondition (const char *condition);
    
    //------------------------------------------------------------------
    /// Return a pointer to the text of the condition expression.
    ///
    /// @return
    ///    A pointer to the condition expression text, or NULL if no
    //     condition has been set.
    //------------------------------------------------------------------
    const char *GetConditionText () const;

private:
    friend class Target;
    friend class WatchpointList;

    void        SetTarget(Target *target_ptr) { m_target = target_ptr; }
    std::string GetWatchSpec() { return m_watch_spec_str; }
    void        ResetHitCount() { m_hit_count = 0; }

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
    std::string m_watch_spec_str;      // Spec for the watchpoint (for future use).
    Error       m_error;               // An error object describing errors associated with this watchpoint.

    std::auto_ptr<ClangUserExpression> m_condition_ap;  // The condition to test.

    void SetID(lldb::watch_id_t id) { m_loc_id = id; }

    DISALLOW_COPY_AND_ASSIGN (Watchpoint);
};

} // namespace lldb_private

#endif  // liblldb_Watchpoint_h_
