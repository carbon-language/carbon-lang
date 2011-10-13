//===-- SBWatchpoint.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBWatchpoint_h_
#define LLDB_SBWatchpoint_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBWatchpoint
{
public:

    SBWatchpoint ();

    SBWatchpoint (const lldb::SBWatchpoint &rhs);

    ~SBWatchpoint ();

#ifndef SWIG
    const lldb::SBWatchpoint &
    operator = (const lldb::SBWatchpoint &rhs);
#endif

    lldb::SBError
    GetError ();

    watch_id_t
    GetID ();

    bool
    IsValid() const;

    /// With -1 representing an invalid hardware index.
    int32_t
    GetHardwareIndex ();

    lldb::addr_t
    GetWatchAddress ();

    size_t
    GetWatchSize();

    void
    SetEnabled(bool enabled);

    bool
    IsEnabled ();

    uint32_t
    GetHitCount ();

    uint32_t
    GetIgnoreCount ();

    void
    SetIgnoreCount (uint32_t n);

    bool
    GetDescription (lldb::SBStream &description, DescriptionLevel level);

#ifndef SWIG
    SBWatchpoint (const lldb::WatchpointLocationSP &watch_loc_sp);
#endif

private:
    friend class SBTarget;

#ifndef SWIG

    lldb_private::WatchpointLocation *
    operator->();

    lldb_private::WatchpointLocation *
    get();

    lldb::WatchpointLocationSP &
    operator *();

#endif

    lldb::WatchpointLocationSP m_opaque_sp;

};

} // namespace lldb

#endif  // LLDB_SBWatchpoint_h_
