//===-- WatchpointLocationList.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_WatchpointLocationList_h_
#define liblldb_WatchpointLocationList_h_

// C Includes
// C++ Includes
#include <vector>
#include <map>
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Breakpoint/WatchpointLocation.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class WatchpointLocationList WatchpointLocationList.h "lldb/Breakpoint/WatchpointLocationList.h"
/// @brief This class is used by Watchpoint to manage a list of watchpoint locations,
//  each watchpoint location in the list
/// has a unique ID, and is unique by Address as well.
//----------------------------------------------------------------------

class WatchpointLocationList
{
// Only Target can make the location list, or add elements to it.
// This is not just some random collection of locations.  Rather, the act of adding the location
// to this list sets its ID.
friend class WatchpointLocation;

public:
    //------------------------------------------------------------------
    /// Default constructor makes an empty list.
    //------------------------------------------------------------------
    WatchpointLocationList();

    //------------------------------------------------------------------
    /// Destructor, currently does nothing.
    //------------------------------------------------------------------
    ~WatchpointLocationList();

    //------------------------------------------------------------------
    /// Add a WatchpointLocation to the list.
    ///
    /// @param[in] wp_loc_sp
    ///    A shared pointer to a watchpoint location being added to the list.
    ///
    /// @return
    ///    The ID of the WatchpointLocation in the list.
    //------------------------------------------------------------------
    lldb::watch_id_t
    Add (const lldb::WatchpointLocationSP& wp_loc_sp);

    //------------------------------------------------------------------
    /// Standard "Dump" method.
    //------------------------------------------------------------------
    void
    Dump (Stream *s) const;

    //------------------------------------------------------------------
    /// Returns a shared pointer to the watchpoint location at address
    /// \a addr - const version.
    ///
    /// @param[in] addr
    ///     The address to look for.
    ///
    /// @result
    ///     A shared pointer to the watchpoint.  May contain a NULL
    ///     pointer if the watchpoint doesn't exist.
    //------------------------------------------------------------------
    const lldb::WatchpointLocationSP
    FindByAddress (lldb::addr_t addr) const;

    //------------------------------------------------------------------
    /// Returns a shared pointer to the watchpoint location with id
    /// \a breakID, const version.
    ///
    /// @param[in] breakID
    ///     The watchpoint location ID to seek for.
    ///
    /// @result
    ///     A shared pointer to the watchpoint.  May contain a NULL
    ///     pointer if the watchpoint doesn't exist.
    //------------------------------------------------------------------
    lldb::WatchpointLocationSP
    FindByID (lldb::watch_id_t watchID) const;

    //------------------------------------------------------------------
    /// Returns the watchpoint location id to the watchpoint location
    /// at address \a addr.
    ///
    /// @param[in] addr
    ///     The address to match.
    ///
    /// @result
    ///     The ID of the watchpoint location, or LLDB_INVALID_WATCH_ID.
    //------------------------------------------------------------------
    lldb::watch_id_t
    FindIDByAddress (lldb::addr_t addr);

    //------------------------------------------------------------------
    /// Removes the watchpoint location given by \b watchID from this list.
    ///
    /// @param[in] watchID
    ///   The watchpoint location ID to remove.
    ///
    /// @result
    ///   \b true if the watchpoint location \a watchID was in the list.
    //------------------------------------------------------------------
    bool
    Remove (lldb::watch_id_t watchID);

    //------------------------------------------------------------------
    /// Returns the number hit count of all locations in this list.
    ///
    /// @result
    ///     Hit count of all locations in this list.
    //------------------------------------------------------------------
    uint32_t
    GetHitCount () const;

    //------------------------------------------------------------------
    /// Enquires of the watchpoint location in this list with ID \a
    /// watchID whether we should stop.
    ///
    /// @param[in] context
    ///     This contains the information about this stop.
    ///
    /// @param[in] watchID
    ///     This watch ID that we hit.
    ///
    /// @return
    ///     \b true if we should stop, \b false otherwise.
    //------------------------------------------------------------------
    bool
    ShouldStop (StoppointCallbackContext *context,
                lldb::watch_id_t watchID);

    //------------------------------------------------------------------
    /// Returns the number of elements in this watchpoint location list.
    ///
    /// @result
    ///     The number of elements.
    //------------------------------------------------------------------
    size_t
    GetSize() const
    {
        return m_address_to_location.size();
    }

    //------------------------------------------------------------------
    /// Print a description of the watchpoint locations in this list to
    /// the stream \a s.
    ///
    /// @param[in] s
    ///     The stream to which to print the description.
    ///
    /// @param[in] level
    ///     The description level that indicates the detail level to
    ///     provide.
    ///
    /// @see lldb::DescriptionLevel
    //------------------------------------------------------------------
    void
    GetDescription (Stream *s,
                    lldb::DescriptionLevel level);

    //------------------------------------------------------------------
    /// Sets the passed in Locker to hold the Watchpoint Location List mutex.
    ///
    /// @param[in] locker
    ///   The locker object that is set.
    //------------------------------------------------------------------
    void
    GetListMutex (lldb_private::Mutex::Locker &locker);

protected:
    typedef std::map<lldb::addr_t, lldb::WatchpointLocationSP> addr_map;

    addr_map::iterator
    GetIDIterator(lldb::watch_id_t watchID);

    addr_map::const_iterator
    GetIDConstIterator(lldb::watch_id_t watchID) const;

    addr_map m_address_to_location;
    mutable Mutex m_mutex;
};

} // namespace lldb_private

#endif  // liblldb_WatchpointLocationList_h_
