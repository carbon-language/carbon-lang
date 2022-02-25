//===-- PlatformRemoteAppleTV.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEAPPLETV_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEAPPLETV_H

#include <string>

#include "lldb/Utility/FileSpec.h"

#include "llvm/Support/FileSystem.h"

#include "PlatformRemoteDarwinDevice.h"

class PlatformRemoteAppleTV : public PlatformRemoteDarwinDevice {
public:
  PlatformRemoteAppleTV();

  // Class Functions
  static lldb::PlatformSP CreateInstance(bool force,
                                         const lldb_private::ArchSpec *arch);

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "remote-tvos"; }

  static llvm::StringRef GetDescriptionStatic();

  // lldb_private::PluginInterface functions
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  // lldb_private::Platform functions

  llvm::StringRef GetDescription() override { return GetDescriptionStatic(); }

  std::vector<lldb_private::ArchSpec> GetSupportedArchitectures() override;

protected:
  llvm::StringRef GetDeviceSupportDirectoryName() override;
  llvm::StringRef GetPlatformName() override;
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEAPPLETV_H
