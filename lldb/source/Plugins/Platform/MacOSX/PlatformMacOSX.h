//===-- PlatformMacOSX.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMMACOSX_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMMACOSX_H

#include "PlatformDarwin.h"

class PlatformMacOSX : public PlatformDarwin {
public:
  PlatformMacOSX();

  // Class functions
  static lldb::PlatformSP CreateInstance(bool force,
                                         const lldb_private::ArchSpec *arch);

  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetDescriptionStatic();

  // lldb_private::PluginInterface functions
  lldb_private::ConstString GetPluginName() override {
    return GetPluginNameStatic();
  }

  uint32_t GetPluginVersion() override { return 1; }

  lldb_private::Status
  GetSharedModule(const lldb_private::ModuleSpec &module_spec,
                  lldb_private::Process *process, lldb::ModuleSP &module_sp,
                  const lldb_private::FileSpecList *module_search_paths_ptr,
                  llvm::SmallVectorImpl<lldb::ModuleSP> *old_modules,
                  bool *did_create_ptr) override;

  const char *GetDescription() override { return GetDescriptionStatic(); }

  lldb_private::Status
  GetFile(const lldb_private::FileSpec &source,
          const lldb_private::FileSpec &destination) override {
    return PlatformDarwin::GetFile(source, destination);
  }

  bool GetSupportedArchitectureAtIndex(uint32_t idx,
                                       lldb_private::ArchSpec &arch) override;

  lldb_private::ConstString
  GetSDKDirectory(lldb_private::Target &target) override;

  void
  AddClangModuleCompilationOptions(lldb_private::Target *target,
                                   std::vector<std::string> &options) override {
    return PlatformDarwin::AddClangModuleCompilationOptionsForSDKType(
        target, options, lldb_private::XcodeSDK::Type::MacOSX);
  }

private:
#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
  uint32_t m_num_arm_arches = 0;
#endif
};

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_MACOSX_PLATFORMMACOSX_H
