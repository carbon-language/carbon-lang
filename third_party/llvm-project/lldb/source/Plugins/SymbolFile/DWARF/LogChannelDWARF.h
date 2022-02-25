//===-- LogChannelDWARF.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_LOGCHANNELDWARF_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_LOGCHANNELDWARF_H

#include "lldb/Utility/Log.h"

#define DWARF_LOG_DEBUG_INFO (1u << 1)
#define DWARF_LOG_DEBUG_LINE (1u << 2)
#define DWARF_LOG_LOOKUPS (1u << 3)
#define DWARF_LOG_TYPE_COMPLETION (1u << 4)
#define DWARF_LOG_DEBUG_MAP (1u << 5)
#define DWARF_LOG_ALL (UINT32_MAX)
#define DWARF_LOG_DEFAULT (DWARF_LOG_DEBUG_INFO)

namespace lldb_private {
class LogChannelDWARF {
  static Log::Channel g_channel;

public:
  static void Initialize();
  static void Terminate();

  static Log *GetLogIfAll(uint32_t mask) { return g_channel.GetLogIfAll(mask); }
  static Log *GetLogIfAny(uint32_t mask) { return g_channel.GetLogIfAny(mask); }
};
}

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_LOGCHANNELDWARF_H
