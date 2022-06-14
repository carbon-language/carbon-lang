//===-- FILE Writer implementation for printf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf_core/file_writer.h"
#include "src/__support/File/file.h"
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

void write_to_file(void *raw_pointer, const char *__restrict to_write,
                   size_t len) {
  __llvm_libc::File *file = reinterpret_cast<__llvm_libc::File *>(raw_pointer);
  file->write(to_write, len);
}

} // namespace printf_core
} // namespace __llvm_libc
