//===-- BreakpointOptions.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BreakpointOptions_h_
#define liblldb_BreakpointOptions_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Baton.h"
#include "lldb/Core/StringList.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class BreakpointOptions BreakpointOptions.h "lldb/Breakpoint/BreakpointOptions.h"
/// @brief Class that manages the options on a breakpoint or breakpoint location.
//----------------------------------------------------------------------

class BreakpointOptions
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    //------------------------------------------------------------------
    /// Default constructor.  The breakpoint is enabled, and has no condition,
    /// callback, ignore count, etc...
    //------------------------------------------------------------------
    BreakpointOptions();
    BreakpointOptions(const BreakpointOptions& rhs);

    
    //------------------------------------------------------------------
    /// This constructor allows you to specify all the breakpoint options.
    ///
    /// @param[in] condition
    ///    The expression which if it evaluates to \b true if we are to stop
    ///
    /// @param[in] callback
    ///    This is the plugin for some code that gets run, returns \b true if we are to stop.
    ///
    /// @param[in] baton
    ///    Client data that will get passed to the callback.
    ///
    /// @param[in] enabled
    ///    Is this breakpoint enabled.
    ///
    /// @param[in] ignore
    ///    How many breakpoint hits we should ignore before stopping.
    ///
    /// @param[in] thread_id
    ///    Only stop if \a thread_id hits the breakpoint.
    //------------------------------------------------------------------
    BreakpointOptions(void *condition,
                      BreakpointHitCallback callback,
                      void *baton,
                      bool enabled = true,
                      int32_t ignore = 0,
                      lldb::tid_t thread_id = LLDB_INVALID_THREAD_ID);

    virtual ~BreakpointOptions();

    //------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------
    const BreakpointOptions&
    operator=(const BreakpointOptions& rhs);

    //------------------------------------------------------------------
    // Callbacks
    //------------------------------------------------------------------
    void SetCallback (BreakpointHitCallback callback, const lldb::BatonSP &baton_sp, bool synchronous = false);
    bool InvokeCallback (StoppointCallbackContext *context, lldb::user_id_t break_id, lldb::user_id_t break_loc_id);
    bool IsCallbackSynchronous () {
        return m_callback_is_synchronous;
    };
    Baton *GetBaton ();
    void ClearCallback ();

    //------------------------------------------------------------------
    // Enabled/Ignore Count
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Check the Enable/Disable state.
    /// @return
    ///     \b true if the breakpoint is enabled, \b false if disabled.
    //------------------------------------------------------------------
    bool         
    IsEnabled () const;

    //------------------------------------------------------------------
    /// If \a enable is \b true, enable the breakpoint, if \b false disable it.
    //------------------------------------------------------------------
    void
    SetEnabled (bool enabled);

    void
    SetIgnoreCount (int32_t n);

    //------------------------------------------------------------------
    /// Return the current Ignore Count.
    /// @return
    ///     The number of breakpoint hits to be ignored.
    //------------------------------------------------------------------
    int32_t
    GetIgnoreCount () const;

    //------------------------------------------------------------------
    /// Set the breakpoint to ignore the next \a count breakpoint hits.
    /// @param[in] count
    ///    The number of breakpoint hits to ignore.
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Return the current stop thread value.
    /// @return
    ///     The thread id for which the breakpoint hit will stop, 
    ///     LLDB_INVALID_THREAD_ID for all threads.
    //------------------------------------------------------------------
    lldb::tid_t
    GetThreadID () const;

    //------------------------------------------------------------------
    /// Set the valid thread to be checked when the breakpoint is hit.
    /// @param[in] thread_id
    ///    If this thread hits the breakpoint, we stop, otherwise not.
    //------------------------------------------------------------------
    void
    SetThreadID (lldb::tid_t thread_id);

    //------------------------------------------------------------------
    /// This is the default empty callback.
    /// @return
    ///     The thread id for which the breakpoint hit will stop, 
    ///     LLDB_INVALID_THREAD_ID for all threads.
    //------------------------------------------------------------------
    static bool 
    NullCallback (void *baton, 
                  StoppointCallbackContext *context, 
                  lldb::user_id_t break_id,
                  lldb::user_id_t break_loc_id);
    
    
    struct CommandData
    {
        CommandData () :
            user_source(),
            script_source()
        {
        }

        ~CommandData ()
        {
        }
        
        StringList user_source;
        StringList script_source;
    };

    class CommandBaton : public Baton
    {
    public:
        CommandBaton (CommandData *data) :
            Baton (data)
        {
        }

        virtual
        ~CommandBaton ()
        {
            delete ((CommandData *)m_data);
            m_data = NULL;
        }
        
        virtual void
        GetDescription (Stream *s, lldb::DescriptionLevel level) const;

    };

protected:
    //------------------------------------------------------------------
    // Classes that inherit from BreakpointOptions can see and modify these
    //------------------------------------------------------------------

private:
    //------------------------------------------------------------------
    // For BreakpointOptions only
    //------------------------------------------------------------------
    BreakpointHitCallback m_callback; // This is the callback function pointer
    lldb::BatonSP m_callback_baton_sp; // This is the client data for the callback
    bool m_callback_is_synchronous;
    bool m_enabled;
    int32_t m_ignore_count; // Number of times to ignore this breakpoint
    lldb::tid_t m_thread_id; // Thread for which this breakpoint will take

};

} // namespace lldb_private

#endif  // liblldb_BreakpointOptions_h_
