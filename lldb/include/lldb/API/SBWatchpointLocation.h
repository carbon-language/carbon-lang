//===-- SBWatchpointLocation.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBWatchpointLocation_h_
#define LLDB_SBWatchpointLocation_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBWatchpointLocation
{
public:

    SBWatchpointLocation ();

    SBWatchpointLocation (const lldb::SBWatchpointLocation &rhs);

    ~SBWatchpointLocation ();

#ifndef SWIG
    const lldb::SBWatchpointLocation &
    operator = (const lldb::SBWatchpointLocation &rhs);
#endif

    bool
    IsValid() const;

    lldb::addr_t
    GetWatchAddress () const;

    size_t
    GetWatchSize() const;

    void
    SetEnabled(bool enabled);

    bool
    IsEnabled ();

    uint32_t
    GetIgnoreCount ();

    void
    SetIgnoreCount (uint32_t n);

    bool
    GetDescription (lldb::SBStream &description, DescriptionLevel level);

#ifndef SWIG
    SBWatchpointLocation (const lldb::WatchpointLocationSP &watch_loc_sp);
#endif

private:
    friend class SBTarget;

#ifndef SWIG

    lldb_private::WatchpointLocation *
    operator->() const;

    lldb_private::WatchpointLocation *
    get() const;

    lldb::WatchpointLocationSP &
    operator *();

    const lldb::WatchpointLocationSP &
    operator *() const;

#endif

    lldb::WatchpointLocationSP m_opaque_sp;

};

} // namespace lldb

#endif  // LLDB_SBWatchpointLocation_h_
