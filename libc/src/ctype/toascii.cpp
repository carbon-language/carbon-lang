//===-- Implementation of toascii------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/toascii.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, toascii, (int c)) { return (c & 0x7f); }

} // namespace __llvm_libc
