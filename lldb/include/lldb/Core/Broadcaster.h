//===-- Broadcaster.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Broadcaster_h_
#define liblldb_Broadcaster_h_

// C Includes
// C++ Includes
#include <map>
#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
//#include "lldb/Core/Flags.h"
#include "lldb/Core/Listener.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Broadcaster Broadcaster.h "lldb/Core/Broadcaster.h"
/// @brief An event broadcasting class.
///
/// The Broadcaster class is designed to be subclassed by objects that
/// wish to vend events in a multi-threaded environment. Broadcaster
/// objects can each vend 32 events. Each event is represented by a bit
/// in a 32 bit value and these bits can be set:
///     @see Broadcaster::SetEventBits(uint32_t)
/// or cleared:
///     @see Broadcaster::ResetEventBits(uint32_t)
/// When an event gets set the Broadcaster object will notify the
/// Listener object that is listening for the event (if there is one).
///
/// Subclasses should provide broadcast bit definitions for any events
/// they vend, typically using an enumeration:
///     \code
///         class Foo : public Broadcaster
///         {
///         public:
///         //----------------------------------------------------------
///         // Broadcaster event bits definitions.
///         //----------------------------------------------------------
///         enum
///         {
///             eBroadcastBitStateChanged   = (1 << 0),
///             eBroadcastBitInterrupt      = (1 << 1),
///             eBroadcastBitSTDOUT         = (1 << 2),
///             eBroadcastBitSTDERR         = (1 << 3)
///         };
///     \endcode
//----------------------------------------------------------------------
class Broadcaster
{
public:
    //------------------------------------------------------------------
    /// Construct with a broadcaster with a name.
    ///
    /// @param[in] name
    ///     A NULL terminated C string that contains the name of the
    ///     broadcaster object.
    //------------------------------------------------------------------
    Broadcaster (const char *name);

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual since this class gets subclassed.
    //------------------------------------------------------------------
    virtual
    ~Broadcaster();

    //------------------------------------------------------------------
    /// Broadcast an event which has no associated data.
    ///
    /// @param[in] event_type
    ///     The element from the enum defining this broadcaster's events
    ///     that is being broadcast.
    ///
    /// @param[in] event_data
    ///     User event data that will be owned by the lldb::Event that
    ///     is created internally.
    ///
    /// @param[in] unique
    ///     If true, then only add an event of this type if there isn't
    ///     one already in the queue.
    ///
    //------------------------------------------------------------------
    void
    BroadcastEvent (lldb::EventSP &event_sp);

    void
    BroadcastEventIfUnique (lldb::EventSP &event_sp);

    void
    BroadcastEvent (uint32_t event_type, EventData *event_data = NULL);

    void
    BroadcastEventIfUnique (uint32_t event_type, EventData *event_data = NULL);

    virtual void
    AddInitialEventsToListener (Listener *listener, uint32_t requested_events);

    //------------------------------------------------------------------
    /// Listen for any events specified by \a event_mask.
    ///
    /// Only one listener can listen to each event bit in a given
    /// Broadcaster. Once a listener has acquired an event bit, no
    /// other broadcaster will have access to it until it is
    /// relinquished by the first listener that gets it. The actual
    /// event bits that get acquired by \a listener may be different
    /// from what is requested in \a event_mask, and to track this the
    /// actual event bits that are acquired get returned.
    ///
    /// @param[in] listener
    ///     The Listener object that wants to monitor the events that
    ///     get broadcast by this object.
    ///
    /// @param[in] event_mask
    ///     A bit mask that indicates which events the listener is
    ///     asking to monitor.
    ///
    /// @return
    ///     The actual event bits that were acquired by \a listener.
    //------------------------------------------------------------------
    uint32_t
    AddListener (Listener* listener, uint32_t event_mask);

    //------------------------------------------------------------------
    /// Get the NULL terminated C string name of this Broadcaster
    /// object.
    ///
    /// @return
    ///     The NULL terminated C string name of this Broadcaster.
    //------------------------------------------------------------------
    const ConstString &
    GetBroadcasterName ();


    //------------------------------------------------------------------
    /// Get the event name(s) for one or more event bits.
    ///
    /// @param[in] event_mask
    ///     A bit mask that indicates which events to get names for.
    ///
    /// @return
    ///     The NULL terminated C string name of this Broadcaster.
    //------------------------------------------------------------------
    bool
    GetEventNames (Stream &s, const uint32_t event_mask, bool prefix_with_broadcaster_name) const;

    //------------------------------------------------------------------
    /// Set the name for an event bit.
    ///
    /// @param[in] event_mask
    ///     A bit mask that indicates which events the listener is
    ///     asking to monitor.
    ///
    /// @return
    ///     The NULL terminated C string name of this Broadcaster.
    //------------------------------------------------------------------
    void
    SetEventName (uint32_t event_mask, const char *name)
    {
        m_event_names[event_mask] = name;
    }
    
    const char *
    GetEventName (uint32_t event_mask) const
    {
        event_names_map::const_iterator pos = m_event_names.find (event_mask);
        if (pos != m_event_names.end())
            return pos->second.c_str();
        return NULL;
    }

    bool
    EventTypeHasListeners (uint32_t event_type);

    //------------------------------------------------------------------
    /// Removes a Listener from this broadcasters list and frees the
    /// event bits specified by \a event_mask that were previously
    /// acquired by \a listener (assuming \a listener was listening to
    /// this object) for other listener objects to use.
    ///
    /// @param[in] listener
    ///     A Listener object that previously called AddListener.
    ///
    /// @param[in] event_mask
    ///     The event bits \a listener wishes to relinquish.
    ///
    /// @return
    ///     \b True if the listener was listening to this broadcaster
    ///     and was removed, \b false otherwise.
    ///
    /// @see uint32_t Broadcaster::AddListener (Listener*, uint32_t)
    //------------------------------------------------------------------
    bool
    RemoveListener (Listener* listener, uint32_t event_mask = UINT32_MAX);
    
    //------------------------------------------------------------------
    /// Provides a simple mechanism to temporarily redirect events from 
    /// broadcaster.  When you call this function passing in a listener and
    /// event type mask, all events from the broadcaster matching the mask
    /// will now go to the hijacking listener.
    /// Only one hijack can occur at a time.  If we need more than this we
    /// will have to implement a Listener stack.
    ///
    /// @param[in] listener
    ///     A Listener object.  You do not need to call StartListeningForEvents
    ///     for this broadcaster (that would fail anyway since the event bits
    ///     would most likely be taken by the listener(s) you are usurping.
    ///
    /// @param[in] event_mask
    ///     The event bits \a listener wishes to hijack.
    ///
    /// @return
    ///     \b True if the event mask could be hijacked, \b false otherwise.
    ///
    /// @see uint32_t Broadcaster::AddListener (Listener*, uint32_t)
    //------------------------------------------------------------------
    bool
    HijackBroadcaster (Listener *listener, uint32_t event_mask = UINT32_MAX);
    
    bool
    IsHijackedForEvent (uint32_t event_mask)
    {
        if (m_hijacking_listeners.size() > 0)
            return (event_mask & m_hijacking_masks.back()) != 0;
        return false;
    }

    //------------------------------------------------------------------
    /// Restore the state of the Broadcaster from a previous hijack attempt.
    ///
    //------------------------------------------------------------------
    void
    RestoreBroadcaster ();

protected:

    
    void
    PrivateBroadcastEvent (lldb::EventSP &event_sp, bool unique);

    //------------------------------------------------------------------
    // Classes that inherit from Broadcaster can see and modify these
    //------------------------------------------------------------------
    typedef std::vector< std::pair<Listener*,uint32_t> > collection;
    typedef std::map<uint32_t, std::string> event_names_map;
    // Prefix the name of our member variables with "m_broadcaster_"
    // since this is a class that gets subclassed.
    const ConstString m_broadcaster_name;   ///< The name of this broadcaster object.
    event_names_map m_event_names;  ///< Optionally define event names for readability and logging for each event bit
    collection m_listeners;     ///< A list of Listener / event_mask pairs that are listening to this broadcaster.
    Mutex m_listeners_mutex;    ///< A mutex that protects \a m_listeners.
    std::vector<Listener *> m_hijacking_listeners;  // A simple mechanism to intercept events from a broadcaster 
    std::vector<uint32_t> m_hijacking_masks;        // At some point we may want to have a stack or Listener
                                                    // collections, but for now this is just for private hijacking.
    
private:
    //------------------------------------------------------------------
    // For Broadcaster only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (Broadcaster);
};

} // namespace lldb_private

#endif  // liblldb_Broadcaster_h_
