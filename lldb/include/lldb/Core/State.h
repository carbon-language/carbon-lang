//===-- State.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_State_h_
#define liblldb_State_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"

namespace lldb_private {

//------------------------------------------------------------------
/// Converts a StateType to a C string.
///
/// @param[in] state
///     The StateType object to convert.
///
/// @return
///     A NULL terminated C string that describes \a state. The
///     returned string comes from constant string buffers and does
///     not need to be freed.
//------------------------------------------------------------------
const char *
StateAsCString (lldb::StateType state);

bool
StateIsRunningState (lldb::StateType state);

bool
StateIsStoppedState (lldb::StateType state);
    
const char *
GetPermissionsAsCString (uint32_t permissions);
    
} // namespace lldb_private

#endif  // liblldb_State_h_
