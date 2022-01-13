//===-- PlatformWindows.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_WINDOWS_PLATFORMWINDOWS_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_WINDOWS_PLATFORMWINDOWS_H

#include "lldb/Target/RemoteAwarePlatform.h"

namespace lldb_private {

class PlatformWindows : public RemoteAwarePlatform {
public:
  PlatformWindows(bool is_host);

  static void Initialize();

  static void Terminate();

  // lldb_private::PluginInterface functions
  static lldb::PlatformSP CreateInstance(bool force,
                                         const lldb_private::ArchSpec *arch);

  static lldb_private::ConstString GetPluginNameStatic(bool is_host);

  static const char *GetPluginDescriptionStatic(bool is_host);

  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override { return 1; }

  // lldb_private::Platform functions
  const char *GetDescription() override {
    return GetPluginDescriptionStatic(IsHost());
  }

  lldb_private::Status ConnectRemote(lldb_private::Args &args) override;

  lldb_private::Status DisconnectRemote() override;

  lldb::ProcessSP DebugProcess(lldb_private::ProcessLaunchInfo &launch_info,
                               lldb_private::Debugger &debugger,
                               lldb_private::Target *target,
                               lldb_private::Status &error) override;

  lldb::ProcessSP Attach(lldb_private::ProcessAttachInfo &attach_info,
                         lldb_private::Debugger &debugger,
                         lldb_private::Target *target,
                         lldb_private::Status &error) override;

  bool GetSupportedArchitectureAtIndex(uint32_t idx,
                                       lldb_private::ArchSpec &arch) override;

  void GetStatus(lldb_private::Stream &strm) override;

  bool CanDebugProcess() override;

  // FIXME not sure what the _sigtramp equivalent would be on this platform
  void CalculateTrapHandlerSymbolNames() override {}

  ConstString GetFullNameForDylib(ConstString basename) override;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_WINDOWS_PLATFORMWINDOWS_H
