//===-- Implementation header for isascii -------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_CTYPE_ISASCII_H
#define LLVM_LIBC_SRC_CTYPE_ISASCII_H

#undef isascii

namespace __llvm_libc {

int isascii(int c);

} // namespace __llvm_libc

#endif //  LLVM_LIBC_SRC_CTYPE_ISASCII_H
