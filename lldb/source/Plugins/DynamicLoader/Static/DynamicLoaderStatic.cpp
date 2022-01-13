//===-- DynamicLoaderStatic.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"

#include "DynamicLoaderStatic.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(DynamicLoaderStatic)

// Create an instance of this class. This function is filled into the plugin
// info class that gets handed out by the plugin factory and allows the lldb to
// instantiate an instance of this class.
DynamicLoader *DynamicLoaderStatic::CreateInstance(Process *process,
                                                   bool force) {
  bool create = force;
  if (!create) {
    const llvm::Triple &triple_ref =
        process->GetTarget().GetArchitecture().GetTriple();
    const llvm::Triple::OSType os_type = triple_ref.getOS();
    const llvm::Triple::ArchType arch_type = triple_ref.getArch();
    if (os_type == llvm::Triple::UnknownOS) {
      // The WASM and Hexagon plugin check the ArchType rather than the OSType,
      // so explicitly reject those here.
      switch(arch_type) {
        case llvm::Triple::hexagon:
        case llvm::Triple::wasm32:
        case llvm::Triple::wasm64:
          break;
        default:
          create = true;
      }
    }
  }

  if (!create) {
    Module *exe_module = process->GetTarget().GetExecutableModulePointer();
    if (exe_module) {
      ObjectFile *object_file = exe_module->GetObjectFile();
      if (object_file) {
        create = (object_file->GetStrata() == ObjectFile::eStrataRawImage);
      }
    }
  }

  if (create)
    return new DynamicLoaderStatic(process);
  return nullptr;
}

// Constructor
DynamicLoaderStatic::DynamicLoaderStatic(Process *process)
    : DynamicLoader(process) {}

/// Called after attaching a process.
///
/// Allow DynamicLoader plug-ins to execute some code after
/// attaching to a process.
void DynamicLoaderStatic::DidAttach() { LoadAllImagesAtFileAddresses(); }

/// Called after attaching a process.
///
/// Allow DynamicLoader plug-ins to execute some code after
/// attaching to a process.
void DynamicLoaderStatic::DidLaunch() { LoadAllImagesAtFileAddresses(); }

void DynamicLoaderStatic::LoadAllImagesAtFileAddresses() {
  const ModuleList &module_list = m_process->GetTarget().GetImages();

  ModuleList loaded_module_list;

  // Disable JIT for static dynamic loader targets
  m_process->SetCanJIT(false);

  for (ModuleSP module_sp : module_list.Modules()) {
    if (module_sp) {
      bool changed = false;
      ObjectFile *image_object_file = module_sp->GetObjectFile();
      if (image_object_file) {
        SectionList *section_list = image_object_file->GetSectionList();
        if (section_list) {
          // All sections listed in the dyld image info structure will all
          // either be fixed up already, or they will all be off by a single
          // slide amount that is determined by finding the first segment that
          // is at file offset zero which also has bytes (a file size that is
          // greater than zero) in the object file.

          // Determine the slide amount (if any)
          const size_t num_sections = section_list->GetSize();
          size_t sect_idx = 0;
          for (sect_idx = 0; sect_idx < num_sections; ++sect_idx) {
            // Iterate through the object file sections to find the first
            // section that starts of file offset zero and that has bytes in
            // the file...
            SectionSP section_sp(section_list->GetSectionAtIndex(sect_idx));
            if (section_sp) {
              // If this section already has a load address set in the target,
              // don't re-set it to the file address.  Something may have
              // set it to a more correct value already.
              if (m_process->GetTarget()
                      .GetSectionLoadList()
                      .GetSectionLoadAddress(section_sp) !=
                  LLDB_INVALID_ADDRESS) {
                continue;
              }
              if (m_process->GetTarget().SetSectionLoadAddress(
                      section_sp, section_sp->GetFileAddress()))
                changed = true;
            }
          }
        }
      }

      if (changed)
        loaded_module_list.AppendIfNeeded(module_sp);
    }
  }

  m_process->GetTarget().ModulesDidLoad(loaded_module_list);
}

ThreadPlanSP
DynamicLoaderStatic::GetStepThroughTrampolinePlan(Thread &thread,
                                                  bool stop_others) {
  return ThreadPlanSP();
}

Status DynamicLoaderStatic::CanLoadImage() {
  Status error;
  error.SetErrorString("can't load images on with a static debug session");
  return error;
}

void DynamicLoaderStatic::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void DynamicLoaderStatic::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString DynamicLoaderStatic::GetPluginNameStatic() {
  static ConstString g_name("static");
  return g_name;
}

const char *DynamicLoaderStatic::GetPluginDescriptionStatic() {
  return "Dynamic loader plug-in that will load any images at the static "
         "addresses contained in each image.";
}

// PluginInterface protocol
lldb_private::ConstString DynamicLoaderStatic::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t DynamicLoaderStatic::GetPluginVersion() { return 1; }
