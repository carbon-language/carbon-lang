// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_VLOG_H_
#define CARBON_COMMON_VLOG_H_

#include "common/vlog_internal.h"

namespace Carbon {

// Logs when verbose logging is enabled (vlog_stream_ is non-null).
//
// For example:
//   CARBON_VLOG() << "Verbose message";
#define CARBON_VLOG()                 \
  (vlog_stream_ == nullptr) ? (void)0 \
                            : CARBON_VLOG_INTERNAL_STREAM(vlog_stream_)

}  // namespace Carbon

#endif  // CARBON_COMMON_VLOG_H_
