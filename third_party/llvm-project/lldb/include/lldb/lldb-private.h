//===-- lldb-private.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_LLDB_PRIVATE_H
#define LLDB_LLDB_PRIVATE_H

#if defined(__cplusplus)

#include "lldb/lldb-private-defines.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-private-interfaces.h"
#include "lldb/lldb-private-types.h"
#include "lldb/lldb-public.h"

namespace lldb_private {

const char *GetVersion();

} // namespace lldb_private

#endif // defined(__cplusplus)

#endif // LLDB_LLDB_PRIVATE_H
