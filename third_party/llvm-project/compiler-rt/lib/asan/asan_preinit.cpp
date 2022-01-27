//===-- asan_preinit.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Call __asan_init at the very early stage of process startup.
//===----------------------------------------------------------------------===//
#include "asan_internal.h"

using namespace __asan;

#if SANITIZER_CAN_USE_PREINIT_ARRAY
  // The symbol is called __local_asan_preinit, because it's not intended to be
  // exported.
  // This code linked into the main executable when -fsanitize=address is in
  // the link flags. It can only use exported interface functions.
  __attribute__((section(".preinit_array"), used))
  void (*__local_asan_preinit)(void) = __asan_init;
#endif
