//===-- PlatformRemoteMacOSX.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEMACOSX_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEMACOSX_H

#include <string>

#include "lldb/Utility/FileSpec.h"

#include "llvm/Support/FileSystem.h"

#include "PlatformMacOSX.h"
#include "PlatformRemoteDarwinDevice.h"

class PlatformRemoteMacOSX : public virtual PlatformRemoteDarwinDevice {
public:
  PlatformRemoteMacOSX();

  static lldb::PlatformSP CreateInstance(bool force,
                                         const lldb_private::ArchSpec *arch);

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "remote-macosx"; }

  static llvm::StringRef GetDescriptionStatic();

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  llvm::StringRef GetDescription() override { return GetDescriptionStatic(); }

  lldb_private::Status
  GetFileWithUUID(const lldb_private::FileSpec &platform_file,
                  const lldb_private::UUID *uuid_ptr,
                  lldb_private::FileSpec &local_file) override;

  std::vector<lldb_private::ArchSpec> GetSupportedArchitectures() override;

protected:
  llvm::StringRef GetDeviceSupportDirectoryName() override;
  llvm::StringRef GetPlatformName() override;
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMREMOTEMACOSX_H
