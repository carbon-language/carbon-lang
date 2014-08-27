//===-- HostProcess.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_HostProcess_h_
#define lldb_Host_HostProcess_h_

//----------------------------------------------------------------------
/// @class HostInfo HostInfo.h "lldb/Host/HostProcess.h"
/// @brief A class that represents a running process on the host machine.
///
/// HostProcess allows querying and manipulation of processes running on the
/// host machine.  It is not intended to be represent a process which is
/// being debugged, although the native debug engine of a platform may likely
/// back inferior processes by a HostProcess.
///
/// HostProcess is implemented using static polymorphism so that on any given
/// platform, an instance of HostProcess will always be able to bind statically
/// to the concrete Process implementation for that platform.  See HostInfo
/// for more details.
///
//----------------------------------------------------------------------

#if defined(_WIN32)
#include "lldb/Host/windows/HostProcessWindows.h"
#define HOST_PROCESS_TYPE HostProcessWindows
#else
#include "lldb/Host/posix/HostProcessPosix.h"
#define HOST_PROCESS_TYPE HostProcessPosix
#endif

namespace lldb_private
{
  typedef HOST_PROCESS_TYPE HostProcess;
}

#undef HOST_PROCESS_TYPE

#endif
