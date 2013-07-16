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

// CF_RC_BITS, the layout of CFRuntimeBase and __CFStrIsConstant are internal
// and subject to change in further CoreFoundation versions. Apple does not
// guarantee any binary compatibility from release to release.

// See http://opensource.apple.com/source/CF/CF-635.15/CFInternal.h
#if defined(__BIG_ENDIAN__)
#define CF_RC_BITS 0
#endif

#if defined(__LITTLE_ENDIAN__)
#define CF_RC_BITS 3
#endif

// See http://opensource.apple.com/source/CF/CF-635.15/CFRuntime.h
typedef struct __CFRuntimeBase {
  uptr _cfisa;
  u8 _cfinfo[4];
#if __LP64__
  u32 _rc;
#endif
} CFRuntimeBase;

enum {
  MACOS_VERSION_UNKNOWN = 0,
  MACOS_VERSION_LEOPARD,
  MACOS_VERSION_SNOW_LEOPARD,
  MACOS_VERSION_LION,
  MACOS_VERSION_MOUNTAIN_LION,
  MACOS_VERSION_MAVERICKS
};

// Used by asan_malloc_mac.cc and asan_mac.cc
extern "C" void __CFInitialize();

namespace __asan {

int GetMacosVersion();
void MaybeReplaceCFAllocator();

}  // namespace __asan

#endif  // ASAN_MAC_H
