//===-- GDBServerLog.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//----------------------------------------------------------------------
//
//  GDBServerLog.h
//  liblldb
//
//  Created by Greg Clayton on 6/19/09.
//
//
//----------------------------------------------------------------------

#ifndef liblldb_GDBServerLog_h_
#define liblldb_GDBServerLog_h_

// C Includes
// C++ Includes
// Other libraries and framework includes

#include "lldb/Core/Log.h"

// Project includes
#define GS_LOG_VERBOSE  (1u << 0)
#define GS_LOG_DEBUG        (1u << 1)
#define GS_LOG_PACKETS  (1u << 2)
#define GS_LOG_EVENTS       (1u << 3)
#define GS_LOG_MINIMAL  (1u << 4)
#define GS_LOG_ALL      (UINT32_MAX)
#define GS_LOG_DEFAULT  (GS_LOG_VERBOSE     |\
                             GS_LOG_PACKETS)

namespace lldb {

class GDBServerLog
{
public:
    static Log *
    GetLog (uint32_t mask = 0);

    static void
    SetLog (Log *log);

    static void
    LogIf (uint32_t mask, const char *format, ...);
};

} // namespace lldb

#endif  // liblldb_GDBServerLog_h_
