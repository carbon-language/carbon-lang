//===-- BreakpointIDList.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BreakpointIDList_h_
#define liblldb_BreakpointIDList_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private.h"
#include "lldb/Breakpoint/BreakpointID.h"

namespace lldb_private {

//----------------------------------------------------------------------
// class BreakpointIDList
//----------------------------------------------------------------------


class BreakpointIDList
{
public:
    typedef std::vector<BreakpointID> BreakpointIDArray;

    BreakpointIDList ();

    virtual
    ~BreakpointIDList ();

    int
    Size();

    BreakpointID &
    GetBreakpointIDAtIndex (int index);

    bool
    RemoveBreakpointIDAtIndex (int index);

    void
    Clear();

    bool
    AddBreakpointID (BreakpointID bp_id);

    bool
    AddBreakpointID (const char *bp_id);

    bool
    FindBreakpointID (BreakpointID &bp_id, int *position);

    bool
    FindBreakpointID (const char *bp_id, int *position);

    void
    InsertStringArray (const char **string_array, int array_size, CommandReturnObject &result);

    static bool
    StringContainsIDRangeExpression (const char *in_string, int *range_start_len, int *range_end_pos);

    static void
    FindAndReplaceIDRanges (Args &old_args, Target *target, CommandReturnObject &result, Args &new_args);

private:
    BreakpointIDArray m_breakpoint_ids;
    BreakpointID m_invalid_id;

    DISALLOW_COPY_AND_ASSIGN(BreakpointIDList);
};

} // namespace lldb_private

#endif  // liblldb_BreakpointIDList_h_
