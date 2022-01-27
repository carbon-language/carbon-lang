//===-- SDCOMP-26094 specific items -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SDCOMP26094_H
#define LLVM_LIBC_TEST_SRC_MATH_SDCOMP26094_H

#include "src/__support/CPP/Array.h"

#include <stdint.h>

namespace __llvm_libc {
namespace testing {

static constexpr __llvm_libc::cpp::Array<uint32_t, 10> SDCOMP26094_VALUES{
    0x46427f1b, 0x4647e568, 0x46428bac, 0x4647f1f9, 0x4647fe8a,
    0x45d8d7f1, 0x45d371a4, 0x45ce0b57, 0x45d35882, 0x45cdf235,
};

} // namespace testing
} // namespace __llvm_libc

#endif // LLVM_LIBC_TEST_SRC_MATH_SDCOMP26094_H
