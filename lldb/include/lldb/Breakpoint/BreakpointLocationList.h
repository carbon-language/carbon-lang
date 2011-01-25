//===-- BreakpointLocationList.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BreakpointLocationList_h_
#define liblldb_BreakpointLocationList_h_

// C Includes
// C++ Includes
#include <vector>
#include <map>
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Host/Mutex.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class BreakpointLocationList BreakpointLocationList.h "lldb/Breakpoint/BreakpointLocationList.h"
/// @brief This class is used by Breakpoint to manage a list of breakpoint locations,
//  each breakpoint location in the list
/// has a unique ID, and is unique by Address as well.
//----------------------------------------------------------------------

class BreakpointLocationList
{
// Only Breakpoints can make the location list, or add elements to it.
// This is not just some random collection of locations.  Rather, the act of adding the location
// to this list sets its ID, and implicitly all the locations have the same breakpoint ID as
// well.  If you need a generic container for breakpoint locations, use BreakpointLocationCollection.
friend class Breakpoint;

public:
    virtual 
    ~BreakpointLocationList();

    //------------------------------------------------------------------
    /// Standard "Dump" method.  At present it does nothing.
    //------------------------------------------------------------------
    void
    Dump (Stream *s) const;

    //------------------------------------------------------------------
    /// Returns a shared pointer to the breakpoint location at address
    /// \a addr - const version.
    ///
    /// @param[in] addr
    ///     The address to look for.
    ///
    /// @result
    ///     A shared pointer to the breakpoint.  May contain a NULL
    ///     pointer if the breakpoint doesn't exist.
    //------------------------------------------------------------------
    const lldb::BreakpointLocationSP
    FindByAddress (Address &addr) const;

    //------------------------------------------------------------------
    /// Returns a shared pointer to the breakpoint location with id \a
    /// breakID.
    ///
    /// @param[in] breakID
    ///     The breakpoint location ID to seek for.
    ///
    /// @result
    ///     A shared pointer to the breakpoint.  May contain a NULL
    ///     pointer if the breakpoint doesn't exist.
    //------------------------------------------------------------------
    lldb::BreakpointLocationSP
    FindByID (lldb::break_id_t breakID);

    //------------------------------------------------------------------
    /// Returns a shared pointer to the breakpoint location with id
    /// \a breakID, const version.
    ///
    /// @param[in] breakID
    ///     The breakpoint location ID to seek for.
    ///
    /// @result
    ///     A shared pointer to the breakpoint.  May contain a NULL
    ///     pointer if the breakpoint doesn't exist.
    //------------------------------------------------------------------
    const lldb::BreakpointLocationSP
    FindByID (lldb::break_id_t breakID) const;

    //------------------------------------------------------------------
    /// Returns the breakpoint location id to the breakpoint location
    /// at address \a addr.
    ///
    /// @param[in] addr
    ///     The address to match.
    ///
    /// @result
    ///     The ID of the breakpoint location, or LLDB_INVALID_BREAK_ID.
    //------------------------------------------------------------------
    lldb::break_id_t
    FindIDByAddress (Address &addr);

    //------------------------------------------------------------------
    /// Returns a breakpoint location list of the breakpoint locations
    /// in the module \a module.  This list is allocated, and owned by
    /// the caller.
    ///
    /// @param[in] module
    ///     The module to seek in.
    ///
    /// @param[in]
    ///     A breakpoint collection that gets any breakpoint locations
    ///     that match \a module appended to.
    ///
    /// @result
    ///     The number of matches
    //------------------------------------------------------------------
    size_t
    FindInModule (Module *module,
                  BreakpointLocationCollection& bp_loc_list);

    //------------------------------------------------------------------
    /// Returns a shared pointer to the breakpoint location with
    /// index \a i.
    ///
    /// @param[in] i
    ///     The breakpoint location index to seek for.
    ///
    /// @result
    ///     A shared pointer to the breakpoint.  May contain a NULL
    ///     pointer if the breakpoint doesn't exist.
    //------------------------------------------------------------------
    lldb::BreakpointLocationSP
    GetByIndex (uint32_t i);

    //------------------------------------------------------------------
    /// Returns a shared pointer to the breakpoint location with index
    /// \a i, const version.
    ///
    /// @param[in] i
    ///     The breakpoint location index to seek for.
    ///
    /// @result
    ///     A shared pointer to the breakpoint.  May contain a NULL
    ///     pointer if the breakpoint doesn't exist.
    //------------------------------------------------------------------
    const lldb::BreakpointLocationSP
    GetByIndex (uint32_t i) const;

    //------------------------------------------------------------------
    /// Removes all the locations in this list from their breakpoint site
    /// owners list.
    //------------------------------------------------------------------
    void
    ClearAllBreakpointSites ();

    //------------------------------------------------------------------
    /// Tells all the breakopint locations in this list to attempt to
    /// resolve any possible breakpoint sites.
    //------------------------------------------------------------------
    void
    ResolveAllBreakpointSites ();

    //------------------------------------------------------------------
    /// Returns the number of breakpoint locations in this list with
    /// resolved breakpoints.
    ///
    /// @result
    ///     Number of qualifying breakpoint locations.
    //------------------------------------------------------------------
    size_t
    GetNumResolvedLocations() const;

    //------------------------------------------------------------------
    /// Returns the number hit count of all locations in this list.
    ///
    /// @result
    ///     Hit count of all locations in this list.
    //------------------------------------------------------------------
    uint32_t
    GetHitCount () const;

    //------------------------------------------------------------------
    /// Removes the breakpoint location given by \b breakID from this
    /// list.
    ///
    /// @param[in] breakID
    ///     The breakpoint location index to remove.
    ///
    /// @result
    ///     \b true if the breakpoint \a breakID was in the list.
    //------------------------------------------------------------------
    bool
    Remove (lldb::break_id_t breakID);

    //------------------------------------------------------------------
    /// Enquires of the breakpoint location in this list with ID \a
    /// breakID whether we should stop.
    ///
    /// @param[in] context
    ///     This contains the information about this stop.
    ///
    /// @param[in] breakID
    ///     This break ID that we hit.
    ///
    /// @return
    ///     \b true if we should stop, \b false otherwise.
    //------------------------------------------------------------------
    bool
    ShouldStop (StoppointCallbackContext *context,
                lldb::break_id_t breakID);

    //------------------------------------------------------------------
    /// Returns the number of elements in this breakpoint location list.
    ///
    /// @result
    ///     The number of elements.
    //------------------------------------------------------------------
    size_t
    GetSize() const
    {
        return m_locations.size();
    }

    //------------------------------------------------------------------
    /// Print a description of the breakpoint locations in this list to
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

protected:

    //------------------------------------------------------------------
    /// This is the standard constructor.
    ///
    /// It creates an empty breakpoint location list. It is protected
    /// here because only Breakpoints are allowed to create the
    /// breakpoint location list.
    //------------------------------------------------------------------
    BreakpointLocationList();

    //------------------------------------------------------------------
    /// Add the breakpoint \a bp_loc_sp to the list.
    ///
    /// @param[in] bp_sp
    ///     Shared pointer to the breakpoint location that will get
    ///     added to the list.
    ///
    /// @result
    ///     Returns breakpoint location id.
    //------------------------------------------------------------------
    virtual lldb::break_id_t
    Add (lldb::BreakpointLocationSP& bp_loc_sp);

    typedef std::vector<lldb::BreakpointLocationSP> collection;
    typedef std::map<lldb_private::Address,
                     lldb::BreakpointLocationSP,
                     Address::ModulePointerAndOffsetLessThanFunctionObject> addr_map;

    // The breakpoint locations are stored in their Parent Breakpoint's location list by an
    // index that is unique to this list, and not across all breakpoint location lists.
    // This is only set in the Breakpoint's AddLocation method.
    // There is another breakpoint location list, the owner's list in the BreakpointSite,
    // but that should not reset the ID.  Unfortunately UserID's SetID method is public.
    lldb::break_id_t
    GetNextID();

    collection::iterator
    GetIDIterator(lldb::break_id_t breakID);

    collection::const_iterator
    GetIDConstIterator(lldb::break_id_t breakID) const;

    collection m_locations;
    addr_map m_address_to_location;
    mutable Mutex m_mutex;
    lldb::break_id_t m_next_id;
};

} // namespace lldb_private

#endif  // liblldb_BreakpointLocationList_h_
