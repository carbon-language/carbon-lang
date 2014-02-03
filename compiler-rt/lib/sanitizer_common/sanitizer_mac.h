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

namespace __sanitizer {

enum MacosVersion {
  MACOS_VERSION_UNINITIALIZED = 0,
  MACOS_VERSION_UNKNOWN,
  MACOS_VERSION_LEOPARD,
  MACOS_VERSION_SNOW_LEOPARD,
  MACOS_VERSION_LION,
  MACOS_VERSION_MOUNTAIN_LION,
  MACOS_VERSION_MAVERICKS
};

MacosVersion GetMacosVersion();

}  // namespace __sanitizer

#endif  // SANITIZER_MAC_H
