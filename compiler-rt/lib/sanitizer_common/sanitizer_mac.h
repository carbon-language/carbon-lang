//===-- sanitizer_mac.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between various sanitizers' runtime libraries and
// provides definitions for OSX-specific functions.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_MAC_H
#define SANITIZER_MAC_H

#include "sanitizer_platform.h"
#if SANITIZER_MAC
#include "sanitizer_posix.h"

namespace __sanitizer {

enum MacosVersion {
  MACOS_VERSION_UNINITIALIZED = 0,
  MACOS_VERSION_UNKNOWN,
  MACOS_VERSION_LEOPARD,
  MACOS_VERSION_SNOW_LEOPARD,
  MACOS_VERSION_LION,
  MACOS_VERSION_MOUNTAIN_LION,
  MACOS_VERSION_MAVERICKS,
  MACOS_VERSION_YOSEMITE,
  MACOS_VERSION_UNKNOWN_NEWER
};

MacosVersion GetMacosVersion();

char **GetEnviron();

}  // namespace __sanitizer

#endif  // SANITIZER_MAC
#endif  // SANITIZER_MAC_H
