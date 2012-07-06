//===-- asan_mac.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Mac-specific ASan definitions.
//===----------------------------------------------------------------------===//
#ifndef ASAN_MAC_H
#define ASAN_MAC_H

enum {
  MACOS_VERSION_UNKNOWN = 0,
  MACOS_VERSION_LEOPARD,
  MACOS_VERSION_SNOW_LEOPARD,
  MACOS_VERSION_LION
};

namespace __asan {

int GetMacosVersion();

}  // namespace __asan

#endif  // ASAN_MAC_H
