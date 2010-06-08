//===-- DynamicLoaderMacOSXDYLDLog.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DynamicLoaderMacOSXDYLDLog_h_
#define liblldb_DynamicLoaderMacOSXDYLDLog_h_

// C Includes
// C++ Includes
// Other libraries and framework includes

#include "lldb/lldb-private.h"

// Project includes

class DynamicLoaderMacOSXDYLDLog
{
public:
    static lldb_private::Log *
    GetLogIfAllCategoriesSet (uint32_t mask);

    static void
    SetLog (lldb_private::Log *log);

    static void
    LogIf (uint32_t mask, const char *format, ...);
};

#endif  // liblldb_DynamicLoaderMacOSXDYLDLog_h_
