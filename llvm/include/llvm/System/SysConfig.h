//===- llvm/System/SysConfig.h - System Configuration  ----------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares the SysConfig utilities for platform independent system
// configuration (both globally and at the process level).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_SYSCONFIG_H
#define LLVM_SYSTEM_SYSCONFIG_H

namespace llvm {
namespace sys {

  /// This function makes the necessary calls to the operating system to prevent
  /// core files or any other kind of large memory dumps that can occur when a
  /// program fails.
  /// @brief Prevent core file generation.
  void PreventCoreFiles();

} // End sys namespace
} // End llvm namespace

#endif
