//===-- DynamicLoaderMacOS.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/State.h"

#include "DynamicLoaderDarwin.h"
#include "DynamicLoaderMacOS.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

using namespace lldb;
using namespace lldb_private;

// Create an instance of this class. This function is filled into the plugin
// info class that gets handed out by the plugin factory and allows the lldb to
// instantiate an instance of this class.
DynamicLoader *DynamicLoaderMacOS::CreateInstance(Process *process,
                                                  bool force) {
  bool create = force;
  if (!create) {
    create = true;
    Module *exe_module = process->GetTarget().GetExecutableModulePointer();
    if (exe_module) {
      ObjectFile *object_file = exe_module->GetObjectFile();
      if (object_file) {
        create = (object_file->GetStrata() == ObjectFile::eStrataUser);
      }
    }

    if (create) {
      const llvm::Triple &triple_ref =
          process->GetTarget().GetArchitecture().GetTriple();
      switch (triple_ref.getOS()) {
      case llvm::Triple::Darwin:
      case llvm::Triple::MacOSX:
      case llvm::Triple::IOS:
      case llvm::Triple::TvOS:
      case llvm::Triple::WatchOS:
      // NEED_BRIDGEOS_TRIPLE case llvm::Triple::BridgeOS:
        create = triple_ref.getVendor() == llvm::Triple::Apple;
        break;
      default:
        create = false;
        break;
      }
    }
  }

  if (!UseDYLDSPI(process)) {
    create = false;
  }

  if (create)
    return new DynamicLoaderMacOS(process);
  return nullptr;
}

// Constructor
DynamicLoaderMacOS::DynamicLoaderMacOS(Process *process)
    : DynamicLoaderDarwin(process), m_image_infos_stop_id(UINT32_MAX),
      m_break_id(LLDB_INVALID_BREAK_ID), m_mutex(),
      m_maybe_image_infos_address(LLDB_INVALID_ADDRESS) {}

// Destructor
DynamicLoaderMacOS::~DynamicLoaderMacOS() {
  if (LLDB_BREAK_ID_IS_VALID(m_break_id))
    m_process->GetTarget().RemoveBreakpointByID(m_break_id);
}

bool DynamicLoaderMacOS::ProcessDidExec() {
  std::lock_guard<std::recursive_mutex> baseclass_guard(GetMutex());
  bool did_exec = false;
  if (m_process) {
    // If we are stopped after an exec, we will have only one thread...
    if (m_process->GetThreadList().GetSize() == 1) {
      // Maybe we still have an image infos address around?  If so see
      // if that has changed, and if so we have exec'ed.
      if (m_maybe_image_infos_address != LLDB_INVALID_ADDRESS) {
        lldb::addr_t image_infos_address = m_process->GetImageInfoAddress();
        if (image_infos_address != m_maybe_image_infos_address) {
          // We don't really have to reset this here, since we are going to
          // call DoInitialImageFetch right away to handle the exec.  But in
          // case anybody looks at it in the meantime, it can't hurt.
          m_maybe_image_infos_address = image_infos_address;
          did_exec = true;
        }
      }

      if (!did_exec) {
        // See if we are stopped at '_dyld_start'
        ThreadSP thread_sp(m_process->GetThreadList().GetThreadAtIndex(0));
        if (thread_sp) {
          lldb::StackFrameSP frame_sp(thread_sp->GetStackFrameAtIndex(0));
          if (frame_sp) {
            const Symbol *symbol =
                frame_sp->GetSymbolContext(eSymbolContextSymbol).symbol;
            if (symbol) {
              if (symbol->GetName() == "_dyld_start")
                did_exec = true;
            }
          }
        }
      }
    }
  }

  if (did_exec) {
    m_libpthread_module_wp.reset();
    m_pthread_getspecific_addr.Clear();
  }
  return did_exec;
}

// Clear out the state of this class.
void DynamicLoaderMacOS::DoClear() {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  if (LLDB_BREAK_ID_IS_VALID(m_break_id))
    m_process->GetTarget().RemoveBreakpointByID(m_break_id);

  m_break_id = LLDB_INVALID_BREAK_ID;
}

// Check if we have found DYLD yet
bool DynamicLoaderMacOS::DidSetNotificationBreakpoint() {
  return LLDB_BREAK_ID_IS_VALID(m_break_id);
}

void DynamicLoaderMacOS::ClearNotificationBreakpoint() {
  if (LLDB_BREAK_ID_IS_VALID(m_break_id)) {
    m_process->GetTarget().RemoveBreakpointByID(m_break_id);
    m_break_id = LLDB_INVALID_BREAK_ID;
  }
}

// Try and figure out where dyld is by first asking the Process if it knows
// (which currently calls down in the lldb::Process to get the DYLD info
// (available on SnowLeopard only). If that fails, then check in the default
// addresses.
void DynamicLoaderMacOS::DoInitialImageFetch() {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));

  // Remove any binaries we pre-loaded in the Target before
  // launching/attaching. If the same binaries are present in the process,
  // we'll get them from the shared module cache, we won't need to re-load them
  // from disk.
  UnloadAllImages();

  StructuredData::ObjectSP all_image_info_json_sp(
      m_process->GetLoadedDynamicLibrariesInfos());
  ImageInfo::collection image_infos;
  if (all_image_info_json_sp.get() &&
      all_image_info_json_sp->GetAsDictionary() &&
      all_image_info_json_sp->GetAsDictionary()->HasKey("images") &&
      all_image_info_json_sp->GetAsDictionary()
          ->GetValueForKey("images")
          ->GetAsArray()) {
    if (JSONImageInformationIntoImageInfo(all_image_info_json_sp,
                                          image_infos)) {
      LLDB_LOGF(log, "Initial module fetch:  Adding %" PRId64 " modules.\n",
                (uint64_t)image_infos.size());

      UpdateSpecialBinariesFromNewImageInfos(image_infos);
      AddModulesUsingImageInfos(image_infos);
    }
  }

  m_dyld_image_infos_stop_id = m_process->GetStopID();
  m_maybe_image_infos_address = m_process->GetImageInfoAddress();
}

bool DynamicLoaderMacOS::NeedToDoInitialImageFetch() { return true; }

// Static callback function that gets called when our DYLD notification
// breakpoint gets hit. We update all of our image infos and then let our super
// class DynamicLoader class decide if we should stop or not (based on global
// preference).
bool DynamicLoaderMacOS::NotifyBreakpointHit(void *baton,
                                             StoppointCallbackContext *context,
                                             lldb::user_id_t break_id,
                                             lldb::user_id_t break_loc_id) {
  // Let the event know that the images have changed
  // DYLD passes three arguments to the notification breakpoint.
  // Arg1: enum dyld_notify_mode mode - 0 = adding, 1 = removing, 2 = remove
  // all Arg2: unsigned long icount        - Number of shared libraries
  // added/removed Arg3: uint64_t mach_headers[]     - Array of load addresses
  // of binaries added/removed

  DynamicLoaderMacOS *dyld_instance = (DynamicLoaderMacOS *)baton;

  ExecutionContext exe_ctx(context->exe_ctx_ref);
  Process *process = exe_ctx.GetProcessPtr();

  // This is a sanity check just in case this dyld_instance is an old dyld
  // plugin's breakpoint still lying around.
  if (process != dyld_instance->m_process)
    return false;

  if (dyld_instance->m_image_infos_stop_id != UINT32_MAX &&
      process->GetStopID() < dyld_instance->m_image_infos_stop_id) {
    return false;
  }

  const lldb::ABISP &abi = process->GetABI();
  if (abi) {
    // Build up the value array to store the three arguments given above, then
    // get the values from the ABI:

    TypeSystemClang *clang_ast_context =
        ScratchTypeSystemClang::GetForTarget(process->GetTarget());
    if (!clang_ast_context)
      return false;

    ValueList argument_values;

    Value mode_value;    // enum dyld_notify_mode { dyld_notify_adding=0,
                         // dyld_notify_removing=1, dyld_notify_remove_all=2 };
    Value count_value;   // unsigned long count
    Value headers_value; // uint64_t machHeaders[] (aka void*)

    CompilerType clang_void_ptr_type =
        clang_ast_context->GetBasicType(eBasicTypeVoid).GetPointerType();
    CompilerType clang_uint32_type =
        clang_ast_context->GetBuiltinTypeForEncodingAndBitSize(
            lldb::eEncodingUint, 32);
    CompilerType clang_uint64_type =
        clang_ast_context->GetBuiltinTypeForEncodingAndBitSize(
            lldb::eEncodingUint, 32);

    mode_value.SetValueType(Value::ValueType::Scalar);
    mode_value.SetCompilerType(clang_uint32_type);

    if (process->GetTarget().GetArchitecture().GetAddressByteSize() == 4) {
      count_value.SetValueType(Value::ValueType::Scalar);
      count_value.SetCompilerType(clang_uint32_type);
    } else {
      count_value.SetValueType(Value::ValueType::Scalar);
      count_value.SetCompilerType(clang_uint64_type);
    }

    headers_value.SetValueType(Value::ValueType::Scalar);
    headers_value.SetCompilerType(clang_void_ptr_type);

    argument_values.PushValue(mode_value);
    argument_values.PushValue(count_value);
    argument_values.PushValue(headers_value);

    if (abi->GetArgumentValues(exe_ctx.GetThreadRef(), argument_values)) {
      uint32_t dyld_mode =
          argument_values.GetValueAtIndex(0)->GetScalar().UInt(-1);
      if (dyld_mode != static_cast<uint32_t>(-1)) {
        // Okay the mode was right, now get the number of elements, and the
        // array of new elements...
        uint32_t image_infos_count =
            argument_values.GetValueAtIndex(1)->GetScalar().UInt(-1);
        if (image_infos_count != static_cast<uint32_t>(-1)) {
          addr_t header_array =
              argument_values.GetValueAtIndex(2)->GetScalar().ULongLong(-1);
          if (header_array != static_cast<uint64_t>(-1)) {
            std::vector<addr_t> image_load_addresses;
            for (uint64_t i = 0; i < image_infos_count; i++) {
              Status error;
              addr_t addr = process->ReadUnsignedIntegerFromMemory(
                  header_array + (8 * i), 8, LLDB_INVALID_ADDRESS, error);
              if (addr != LLDB_INVALID_ADDRESS) {
                image_load_addresses.push_back(addr);
              }
            }
            if (dyld_mode == 0) {
              // dyld_notify_adding
              dyld_instance->AddBinaries(image_load_addresses);
            } else if (dyld_mode == 1) {
              // dyld_notify_removing
              dyld_instance->UnloadImages(image_load_addresses);
            } else if (dyld_mode == 2) {
              // dyld_notify_remove_all
              dyld_instance->UnloadAllImages();
            }
          }
        }
      }
    }
  } else {
    process->GetTarget().GetDebugger().GetAsyncErrorStream()->Printf(
        "No ABI plugin located for triple %s -- shared libraries will not be "
        "registered!\n",
        process->GetTarget().GetArchitecture().GetTriple().getTriple().c_str());
  }

  // Return true to stop the target, false to just let the target run
  return dyld_instance->GetStopWhenImagesChange();
}

void DynamicLoaderMacOS::AddBinaries(
    const std::vector<lldb::addr_t> &load_addresses) {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  ImageInfo::collection image_infos;

  LLDB_LOGF(log, "Adding %" PRId64 " modules.",
            (uint64_t)load_addresses.size());
  StructuredData::ObjectSP binaries_info_sp =
      m_process->GetLoadedDynamicLibrariesInfos(load_addresses);
  if (binaries_info_sp.get() && binaries_info_sp->GetAsDictionary() &&
      binaries_info_sp->GetAsDictionary()->HasKey("images") &&
      binaries_info_sp->GetAsDictionary()
          ->GetValueForKey("images")
          ->GetAsArray() &&
      binaries_info_sp->GetAsDictionary()
              ->GetValueForKey("images")
              ->GetAsArray()
              ->GetSize() == load_addresses.size()) {
    if (JSONImageInformationIntoImageInfo(binaries_info_sp, image_infos)) {
      UpdateSpecialBinariesFromNewImageInfos(image_infos);
      AddModulesUsingImageInfos(image_infos);
    }
    m_dyld_image_infos_stop_id = m_process->GetStopID();
  }
}

// Dump the _dyld_all_image_infos members and all current image infos that we
// have parsed to the file handle provided.
void DynamicLoaderMacOS::PutToLog(Log *log) const {
  if (log == nullptr)
    return;
}

bool DynamicLoaderMacOS::SetNotificationBreakpoint() {
  if (m_break_id == LLDB_INVALID_BREAK_ID) {
    ModuleSP dyld_sp(GetDYLDModule());
    if (dyld_sp) {
      bool internal = true;
      bool hardware = false;
      LazyBool skip_prologue = eLazyBoolNo;
      FileSpecList *source_files = nullptr;
      FileSpecList dyld_filelist;
      dyld_filelist.Append(dyld_sp->GetFileSpec());

      Breakpoint *breakpoint =
          m_process->GetTarget()
              .CreateBreakpoint(&dyld_filelist, source_files,
                                "_dyld_debugger_notification",
                                eFunctionNameTypeFull, eLanguageTypeC, 0,
                                skip_prologue, internal, hardware)
              .get();
      breakpoint->SetCallback(DynamicLoaderMacOS::NotifyBreakpointHit, this,
                              true);
      breakpoint->SetBreakpointKind("shared-library-event");
      m_break_id = breakpoint->GetID();
    }
  }
  return m_break_id != LLDB_INVALID_BREAK_ID;
}

addr_t
DynamicLoaderMacOS::GetDyldLockVariableAddressFromModule(Module *module) {
  SymbolContext sc;
  Target &target = m_process->GetTarget();
  if (Symtab *symtab = module->GetSymtab()) {
    std::vector<uint32_t> match_indexes;
    ConstString g_symbol_name("_dyld_global_lock_held");
    uint32_t num_matches = 0;
    num_matches =
        symtab->AppendSymbolIndexesWithName(g_symbol_name, match_indexes);
    if (num_matches == 1) {
      Symbol *symbol = symtab->SymbolAtIndex(match_indexes[0]);
      if (symbol &&
          (symbol->ValueIsAddress() || symbol->GetAddressRef().IsValid())) {
        return symbol->GetAddressRef().GetOpcodeLoadAddress(&target);
      }
    }
  }
  return LLDB_INVALID_ADDRESS;
}

//  Look for this symbol:
//
//  int __attribute__((visibility("hidden")))           _dyld_global_lock_held =
//  0;
//
//  in libdyld.dylib.
Status DynamicLoaderMacOS::CanLoadImage() {
  Status error;
  addr_t symbol_address = LLDB_INVALID_ADDRESS;
  ConstString g_libdyld_name("libdyld.dylib");
  Target &target = m_process->GetTarget();
  const ModuleList &target_modules = target.GetImages();
  std::lock_guard<std::recursive_mutex> guard(target_modules.GetMutex());

  // Find any modules named "libdyld.dylib" and look for the symbol there first
  for (ModuleSP module_sp : target.GetImages().ModulesNoLocking()) {
    if (module_sp) {
      if (module_sp->GetFileSpec().GetFilename() == g_libdyld_name) {
        symbol_address = GetDyldLockVariableAddressFromModule(module_sp.get());
        if (symbol_address != LLDB_INVALID_ADDRESS)
          break;
      }
    }
  }

  // Search through all modules looking for the symbol in them
  if (symbol_address == LLDB_INVALID_ADDRESS) {
    for (ModuleSP module_sp : target.GetImages().Modules()) {
      if (module_sp) {
        addr_t symbol_address =
            GetDyldLockVariableAddressFromModule(module_sp.get());
        if (symbol_address != LLDB_INVALID_ADDRESS)
          break;
      }
    }
  }

  // Default assumption is that it is OK to load images. Only say that we
  // cannot load images if we find the symbol in libdyld and it indicates that
  // we cannot.

  if (symbol_address != LLDB_INVALID_ADDRESS) {
    {
      int lock_held =
          m_process->ReadUnsignedIntegerFromMemory(symbol_address, 4, 0, error);
      if (lock_held != 0) {
        error.SetErrorString("dyld lock held - unsafe to load images.");
      }
    }
  } else {
    // If we were unable to find _dyld_global_lock_held in any modules, or it
    // is not loaded into memory yet, we may be at process startup (sitting  at
    // _dyld_start) - so we should not allow dlopen calls. But if we found more
    // than one module then we are clearly past _dyld_start so in that case
    // we'll default to "it's safe".
    if (target.GetImages().GetSize() <= 1)
      error.SetErrorString("could not find the dyld library or "
                           "the dyld lock symbol");
  }
  return error;
}

bool DynamicLoaderMacOS::GetSharedCacheInformation(
    lldb::addr_t &base_address, UUID &uuid, LazyBool &using_shared_cache,
    LazyBool &private_shared_cache) {
  base_address = LLDB_INVALID_ADDRESS;
  uuid.Clear();
  using_shared_cache = eLazyBoolCalculate;
  private_shared_cache = eLazyBoolCalculate;

  if (m_process) {
    StructuredData::ObjectSP info = m_process->GetSharedCacheInfo();
    StructuredData::Dictionary *info_dict = nullptr;
    if (info.get() && info->GetAsDictionary()) {
      info_dict = info->GetAsDictionary();
    }

    // {"shared_cache_base_address":140735683125248,"shared_cache_uuid
    // ":"DDB8D70C-
    // C9A2-3561-B2C8-BE48A4F33F96","no_shared_cache":false,"shared_cache_private_cache":false}

    if (info_dict && info_dict->HasKey("shared_cache_uuid") &&
        info_dict->HasKey("no_shared_cache") &&
        info_dict->HasKey("shared_cache_base_address")) {
      base_address = info_dict->GetValueForKey("shared_cache_base_address")
                         ->GetIntegerValue(LLDB_INVALID_ADDRESS);
      std::string uuid_str = std::string(
          info_dict->GetValueForKey("shared_cache_uuid")->GetStringValue());
      if (!uuid_str.empty())
        uuid.SetFromStringRef(uuid_str);
      if (!info_dict->GetValueForKey("no_shared_cache")->GetBooleanValue())
        using_shared_cache = eLazyBoolYes;
      else
        using_shared_cache = eLazyBoolNo;
      if (info_dict->GetValueForKey("shared_cache_private_cache")
              ->GetBooleanValue())
        private_shared_cache = eLazyBoolYes;
      else
        private_shared_cache = eLazyBoolNo;

      return true;
    }
  }
  return false;
}

void DynamicLoaderMacOS::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void DynamicLoaderMacOS::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef DynamicLoaderMacOS::GetPluginDescriptionStatic() {
  return "Dynamic loader plug-in that watches for shared library loads/unloads "
         "in MacOSX user processes.";
}
