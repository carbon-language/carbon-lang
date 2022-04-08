//===-- DYLDRendezvous.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"

#include "llvm/Support/Path.h"

#include "DYLDRendezvous.h"

using namespace lldb;
using namespace lldb_private;

DYLDRendezvous::DYLDRendezvous(Process *process)
    : m_process(process), m_rendezvous_addr(LLDB_INVALID_ADDRESS),
      m_executable_interpreter(false), m_current(), m_previous(),
      m_loaded_modules(), m_soentries(), m_added_soentries(),
      m_removed_soentries() {
  m_thread_info.valid = false;
  UpdateExecutablePath();
}

addr_t DYLDRendezvous::ResolveRendezvousAddress() {
  Log *log = GetLog(LLDBLog::DynamicLoader);
  addr_t info_location;
  addr_t info_addr;
  Status error;

  if (!m_process) {
    LLDB_LOGF(log, "%s null process provided", __FUNCTION__);
    return LLDB_INVALID_ADDRESS;
  }

  // Try to get it from our process.  This might be a remote process and might
  // grab it via some remote-specific mechanism.
  info_location = m_process->GetImageInfoAddress();
  LLDB_LOGF(log, "%s info_location = 0x%" PRIx64, __FUNCTION__, info_location);

  // If the process fails to return an address, fall back to seeing if the
  // local object file can help us find it.
  if (info_location == LLDB_INVALID_ADDRESS) {
    Target *target = &m_process->GetTarget();
    if (target) {
      ObjectFile *obj_file = target->GetExecutableModule()->GetObjectFile();
      Address addr = obj_file->GetImageInfoAddress(target);

      if (addr.IsValid()) {
        info_location = addr.GetLoadAddress(target);
        LLDB_LOGF(log,
                  "%s resolved via direct object file approach to 0x%" PRIx64,
                  __FUNCTION__, info_location);
      } else {
        const Symbol *_r_debug =
            target->GetExecutableModule()->FindFirstSymbolWithNameAndType(
                ConstString("_r_debug"));
        if (_r_debug) {
          info_addr = _r_debug->GetAddress().GetLoadAddress(target);
          if (info_addr != LLDB_INVALID_ADDRESS) {
            LLDB_LOGF(log,
                      "%s resolved by finding symbol '_r_debug' whose value is "
                      "0x%" PRIx64,
                      __FUNCTION__, info_addr);
            m_executable_interpreter = true;
            return info_addr;
          }
        }
        LLDB_LOGF(log,
                  "%s FAILED - direct object file approach did not yield a "
                  "valid address",
                  __FUNCTION__);
      }
    }
  }

  if (info_location == LLDB_INVALID_ADDRESS) {
    LLDB_LOGF(log, "%s FAILED - invalid info address", __FUNCTION__);
    return LLDB_INVALID_ADDRESS;
  }

  LLDB_LOGF(log, "%s reading pointer (%" PRIu32 " bytes) from 0x%" PRIx64,
            __FUNCTION__, m_process->GetAddressByteSize(), info_location);

  info_addr = m_process->ReadPointerFromMemory(info_location, error);
  if (error.Fail()) {
    LLDB_LOGF(log, "%s FAILED - could not read from the info location: %s",
              __FUNCTION__, error.AsCString());
    return LLDB_INVALID_ADDRESS;
  }

  if (info_addr == 0) {
    LLDB_LOGF(log,
              "%s FAILED - the rendezvous address contained at 0x%" PRIx64
              " returned a null value",
              __FUNCTION__, info_location);
    return LLDB_INVALID_ADDRESS;
  }

  return info_addr;
}

void DYLDRendezvous::UpdateExecutablePath() {
  if (m_process) {
    Log *log = GetLog(LLDBLog::DynamicLoader);
    Module *exe_mod = m_process->GetTarget().GetExecutableModulePointer();
    if (exe_mod) {
      m_exe_file_spec = exe_mod->GetPlatformFileSpec();
      LLDB_LOGF(log, "DYLDRendezvous::%s exe module executable path set: '%s'",
                __FUNCTION__, m_exe_file_spec.GetCString());
    } else {
      LLDB_LOGF(log,
                "DYLDRendezvous::%s cannot cache exe module path: null "
                "executable module pointer",
                __FUNCTION__);
    }
  }
}

bool DYLDRendezvous::Resolve() {
  Log *log = GetLog(LLDBLog::DynamicLoader);

  const size_t word_size = 4;
  Rendezvous info;
  size_t address_size;
  size_t padding;
  addr_t info_addr;
  addr_t cursor;

  address_size = m_process->GetAddressByteSize();
  padding = address_size - word_size;
  LLDB_LOGF(log,
            "DYLDRendezvous::%s address size: %" PRIu64 ", padding %" PRIu64,
            __FUNCTION__, uint64_t(address_size), uint64_t(padding));

  if (m_rendezvous_addr == LLDB_INVALID_ADDRESS)
    cursor = info_addr =
        ResolveRendezvousAddress();
  else
    cursor = info_addr = m_rendezvous_addr;
  LLDB_LOGF(log, "DYLDRendezvous::%s cursor = 0x%" PRIx64, __FUNCTION__,
            cursor);

  if (cursor == LLDB_INVALID_ADDRESS)
    return false;

  if (!(cursor = ReadWord(cursor, &info.version, word_size)))
    return false;

  if (!(cursor = ReadPointer(cursor + padding, &info.map_addr)))
    return false;

  if (!(cursor = ReadPointer(cursor, &info.brk)))
    return false;

  if (!(cursor = ReadWord(cursor, &info.state, word_size)))
    return false;

  if (!(cursor = ReadPointer(cursor + padding, &info.ldbase)))
    return false;

  // The rendezvous was successfully read.  Update our internal state.
  m_rendezvous_addr = info_addr;
  m_previous = m_current;
  m_current = info;

  if (m_current.map_addr == 0)
    return false;

  if (UpdateSOEntriesFromRemote())
    return true;

  return UpdateSOEntries();
}

bool DYLDRendezvous::IsValid() {
  return m_rendezvous_addr != LLDB_INVALID_ADDRESS;
}

DYLDRendezvous::RendezvousAction DYLDRendezvous::GetAction() const {
  switch (m_current.state) {

  case eConsistent:
    switch (m_previous.state) {
    // When the previous and current states are consistent this is the first
    // time we have been asked to update.  Just take a snapshot of the
    // currently loaded modules.
    case eConsistent:
      return eTakeSnapshot;
    // If we are about to add or remove a shared object clear out the current
    // state and take a snapshot of the currently loaded images.
    case eAdd:
      return eAddModules;
    case eDelete:
      return eRemoveModules;
    }
    break;

  case eAdd:
  case eDelete:
    // Some versions of the android dynamic linker might send two
    // notifications with state == eAdd back to back. Ignore them until we
    // get an eConsistent notification.
    if (!(m_previous.state == eConsistent ||
          (m_previous.state == eAdd && m_current.state == eDelete)))
      return eNoAction;

    return eTakeSnapshot;
  }

  return eNoAction;
}

bool DYLDRendezvous::UpdateSOEntriesFromRemote() {
  auto action = GetAction();

  if (action == eNoAction)
    return false;

  if (action == eTakeSnapshot) {
    m_added_soentries.clear();
    m_removed_soentries.clear();
    // We already have the loaded list from the previous update so no need to
    // find all the modules again.
    if (!m_loaded_modules.m_list.empty())
      return true;
  }

  llvm::Expected<LoadedModuleInfoList> module_list =
      m_process->GetLoadedModuleList();
  if (!module_list) {
    llvm::consumeError(module_list.takeError());
    return false;
  }

  switch (action) {
  case eTakeSnapshot:
    m_soentries.clear();
    return SaveSOEntriesFromRemote(*module_list);
  case eAddModules:
    return AddSOEntriesFromRemote(*module_list);
  case eRemoveModules:
    return RemoveSOEntriesFromRemote(*module_list);
  case eNoAction:
    return false;
  }
  llvm_unreachable("Fully covered switch above!");
}

bool DYLDRendezvous::UpdateSOEntries() {
  switch (GetAction()) {
  case eTakeSnapshot:
    m_soentries.clear();
    m_added_soentries.clear();
    m_removed_soentries.clear();
    return TakeSnapshot(m_soentries);
  case eAddModules:
    return AddSOEntries();
  case eRemoveModules:
    return RemoveSOEntries();
  case eNoAction:
    return false;
  }
  llvm_unreachable("Fully covered switch above!");
}

bool DYLDRendezvous::FillSOEntryFromModuleInfo(
    LoadedModuleInfoList::LoadedModuleInfo const &modInfo, SOEntry &entry) {
  addr_t link_map_addr;
  addr_t base_addr;
  addr_t dyn_addr;
  std::string name;

  if (!modInfo.get_link_map(link_map_addr) || !modInfo.get_base(base_addr) ||
      !modInfo.get_dynamic(dyn_addr) || !modInfo.get_name(name))
    return false;

  entry.link_addr = link_map_addr;
  entry.base_addr = base_addr;
  entry.dyn_addr = dyn_addr;

  entry.file_spec.SetFile(name, FileSpec::Style::native);

  UpdateBaseAddrIfNecessary(entry, name);

  // not needed if we're using ModuleInfos
  entry.next = 0;
  entry.prev = 0;
  entry.path_addr = 0;

  return true;
}

bool DYLDRendezvous::SaveSOEntriesFromRemote(
    const LoadedModuleInfoList &module_list) {
  for (auto const &modInfo : module_list.m_list) {
    SOEntry entry;
    if (!FillSOEntryFromModuleInfo(modInfo, entry))
      return false;

    // Only add shared libraries and not the executable.
    if (!SOEntryIsMainExecutable(entry)) {
      UpdateFileSpecIfNecessary(entry);
      m_soentries.push_back(entry);
    }
  }

  m_loaded_modules = module_list;
  return true;
}

bool DYLDRendezvous::AddSOEntriesFromRemote(
    const LoadedModuleInfoList &module_list) {
  for (auto const &modInfo : module_list.m_list) {
    bool found = false;
    for (auto const &existing : m_loaded_modules.m_list) {
      if (modInfo == existing) {
        found = true;
        break;
      }
    }

    if (found)
      continue;

    SOEntry entry;
    if (!FillSOEntryFromModuleInfo(modInfo, entry))
      return false;

    // Only add shared libraries and not the executable.
    if (!SOEntryIsMainExecutable(entry)) {
      UpdateFileSpecIfNecessary(entry);
      m_soentries.push_back(entry);
      m_added_soentries.push_back(entry);
    }
  }

  m_loaded_modules = module_list;
  return true;
}

bool DYLDRendezvous::RemoveSOEntriesFromRemote(
    const LoadedModuleInfoList &module_list) {
  for (auto const &existing : m_loaded_modules.m_list) {
    bool found = false;
    for (auto const &modInfo : module_list.m_list) {
      if (modInfo == existing) {
        found = true;
        break;
      }
    }

    if (found)
      continue;

    SOEntry entry;
    if (!FillSOEntryFromModuleInfo(existing, entry))
      return false;

    // Only add shared libraries and not the executable.
    if (!SOEntryIsMainExecutable(entry)) {
      auto pos = std::find(m_soentries.begin(), m_soentries.end(), entry);
      if (pos == m_soentries.end())
        return false;

      m_soentries.erase(pos);
      m_removed_soentries.push_back(entry);
    }
  }

  m_loaded_modules = module_list;
  return true;
}

bool DYLDRendezvous::AddSOEntries() {
  SOEntry entry;
  iterator pos;

  assert(m_previous.state == eAdd);

  if (m_current.map_addr == 0)
    return false;

  for (addr_t cursor = m_current.map_addr; cursor != 0; cursor = entry.next) {
    if (!ReadSOEntryFromMemory(cursor, entry))
      return false;

    // Only add shared libraries and not the executable.
    if (SOEntryIsMainExecutable(entry))
      continue;

    UpdateFileSpecIfNecessary(entry);

    pos = std::find(m_soentries.begin(), m_soentries.end(), entry);
    if (pos == m_soentries.end()) {
      m_soentries.push_back(entry);
      m_added_soentries.push_back(entry);
    }
  }

  return true;
}

bool DYLDRendezvous::RemoveSOEntries() {
  SOEntryList entry_list;
  iterator pos;

  assert(m_previous.state == eDelete);

  if (!TakeSnapshot(entry_list))
    return false;

  for (iterator I = begin(); I != end(); ++I) {
    pos = std::find(entry_list.begin(), entry_list.end(), *I);
    if (pos == entry_list.end())
      m_removed_soentries.push_back(*I);
  }

  m_soentries = entry_list;
  return true;
}

bool DYLDRendezvous::SOEntryIsMainExecutable(const SOEntry &entry) {
  // On some systes the executable is indicated by an empty path in the entry.
  // On others it is the full path to the executable.

  auto triple = m_process->GetTarget().GetArchitecture().GetTriple();
  switch (triple.getOS()) {
  case llvm::Triple::FreeBSD:
  case llvm::Triple::NetBSD:
    return entry.file_spec == m_exe_file_spec;
  case llvm::Triple::Linux:
    if (triple.isAndroid())
      return entry.file_spec == m_exe_file_spec;
    // If we are debugging ld.so, then all SOEntries should be treated as
    // libraries, including the "main" one (denoted by an empty string).
    if (!entry.file_spec && m_executable_interpreter)
      return false;
    return !entry.file_spec;
  default:
    return false;
  }
}

bool DYLDRendezvous::TakeSnapshot(SOEntryList &entry_list) {
  SOEntry entry;

  if (m_current.map_addr == 0)
    return false;

  // Clear previous entries since we are about to obtain an up to date list.
  entry_list.clear();

  for (addr_t cursor = m_current.map_addr; cursor != 0; cursor = entry.next) {
    if (!ReadSOEntryFromMemory(cursor, entry))
      return false;

    // Only add shared libraries and not the executable.
    if (SOEntryIsMainExecutable(entry))
      continue;

    UpdateFileSpecIfNecessary(entry);

    entry_list.push_back(entry);
  }

  return true;
}

addr_t DYLDRendezvous::ReadWord(addr_t addr, uint64_t *dst, size_t size) {
  Status error;

  *dst = m_process->ReadUnsignedIntegerFromMemory(addr, size, 0, error);
  if (error.Fail())
    return 0;

  return addr + size;
}

addr_t DYLDRendezvous::ReadPointer(addr_t addr, addr_t *dst) {
  Status error;

  *dst = m_process->ReadPointerFromMemory(addr, error);
  if (error.Fail())
    return 0;

  return addr + m_process->GetAddressByteSize();
}

std::string DYLDRendezvous::ReadStringFromMemory(addr_t addr) {
  std::string str;
  Status error;

  if (addr == LLDB_INVALID_ADDRESS)
    return std::string();

  m_process->ReadCStringFromMemory(addr, str, error);

  return str;
}

// Returns true if the load bias reported by the linker is incorrect for the
// given entry. This function is used to handle cases where we want to work
// around a bug in the system linker.
static bool isLoadBiasIncorrect(Target &target, const std::string &file_path) {
  // On Android L (API 21, 22) the load address of the "/system/bin/linker"
  // isn't filled in correctly.
  unsigned os_major = target.GetPlatform()->GetOSVersion().getMajor();
  return target.GetArchitecture().GetTriple().isAndroid() &&
         (os_major == 21 || os_major == 22) &&
         (file_path == "/system/bin/linker" ||
          file_path == "/system/bin/linker64");
}

void DYLDRendezvous::UpdateBaseAddrIfNecessary(SOEntry &entry,
                                               std::string const &file_path) {
  // If the load bias reported by the linker is incorrect then fetch the load
  // address of the file from the proc file system.
  if (isLoadBiasIncorrect(m_process->GetTarget(), file_path)) {
    lldb::addr_t load_addr = LLDB_INVALID_ADDRESS;
    bool is_loaded = false;
    Status error =
        m_process->GetFileLoadAddress(entry.file_spec, is_loaded, load_addr);
    if (error.Success() && is_loaded)
      entry.base_addr = load_addr;
  }
}

void DYLDRendezvous::UpdateFileSpecIfNecessary(SOEntry &entry) {
  // Updates filename if empty. It is useful while debugging ld.so,
  // when the link map returns empty string for the main executable.
  if (!entry.file_spec) {
    MemoryRegionInfo region;
    Status region_status =
        m_process->GetMemoryRegionInfo(entry.dyn_addr, region);
    if (!region.GetName().IsEmpty())
      entry.file_spec.SetFile(region.GetName().AsCString(),
                              FileSpec::Style::native);
  }
}

bool DYLDRendezvous::ReadSOEntryFromMemory(lldb::addr_t addr, SOEntry &entry) {
  entry.clear();

  entry.link_addr = addr;

  if (!(addr = ReadPointer(addr, &entry.base_addr)))
    return false;

  // mips adds an extra load offset field to the link map struct on FreeBSD and
  // NetBSD (need to validate other OSes).
  // http://svnweb.freebsd.org/base/head/sys/sys/link_elf.h?revision=217153&view=markup#l57
  const ArchSpec &arch = m_process->GetTarget().GetArchitecture();
  if ((arch.GetTriple().getOS() == llvm::Triple::FreeBSD ||
       arch.GetTriple().getOS() == llvm::Triple::NetBSD) &&
      arch.IsMIPS()) {
    addr_t mips_l_offs;
    if (!(addr = ReadPointer(addr, &mips_l_offs)))
      return false;
    if (mips_l_offs != 0 && mips_l_offs != entry.base_addr)
      return false;
  }

  if (!(addr = ReadPointer(addr, &entry.path_addr)))
    return false;

  if (!(addr = ReadPointer(addr, &entry.dyn_addr)))
    return false;

  if (!(addr = ReadPointer(addr, &entry.next)))
    return false;

  if (!(addr = ReadPointer(addr, &entry.prev)))
    return false;

  std::string file_path = ReadStringFromMemory(entry.path_addr);
  entry.file_spec.SetFile(file_path, FileSpec::Style::native);

  UpdateBaseAddrIfNecessary(entry, file_path);

  return true;
}

bool DYLDRendezvous::FindMetadata(const char *name, PThreadField field,
                                  uint32_t &value) {
  Target &target = m_process->GetTarget();

  SymbolContextList list;
  target.GetImages().FindSymbolsWithNameAndType(ConstString(name),
                                                eSymbolTypeAny, list);
  if (list.IsEmpty())
  return false;

  Address address = list[0].symbol->GetAddress();
  addr_t addr = address.GetLoadAddress(&target);
  if (addr == LLDB_INVALID_ADDRESS)
    return false;

  Status error;
  value = (uint32_t)m_process->ReadUnsignedIntegerFromMemory(
      addr + field * sizeof(uint32_t), sizeof(uint32_t), 0, error);
  if (error.Fail())
    return false;

  if (field == eSize)
    value /= 8; // convert bits to bytes

  return true;
}

const DYLDRendezvous::ThreadInfo &DYLDRendezvous::GetThreadInfo() {
  if (!m_thread_info.valid) {
    bool ok = true;

    ok &= FindMetadata("_thread_db_pthread_dtvp", eOffset,
                       m_thread_info.dtv_offset);
    ok &=
        FindMetadata("_thread_db_dtv_dtv", eSize, m_thread_info.dtv_slot_size);
    ok &= FindMetadata("_thread_db_link_map_l_tls_modid", eOffset,
                       m_thread_info.modid_offset);
    ok &= FindMetadata("_thread_db_dtv_t_pointer_val", eOffset,
                       m_thread_info.tls_offset);

    if (ok)
      m_thread_info.valid = true;
  }

  return m_thread_info;
}

void DYLDRendezvous::DumpToLog(Log *log) const {
  int state = GetState();

  if (!log)
    return;

  log->PutCString("DYLDRendezvous:");
  LLDB_LOGF(log, "   Address: %" PRIx64, GetRendezvousAddress());
  LLDB_LOGF(log, "   Version: %" PRIu64, GetVersion());
  LLDB_LOGF(log, "   Link   : %" PRIx64, GetLinkMapAddress());
  LLDB_LOGF(log, "   Break  : %" PRIx64, GetBreakAddress());
  LLDB_LOGF(log, "   LDBase : %" PRIx64, GetLDBase());
  LLDB_LOGF(log, "   State  : %s",
            (state == eConsistent)
                ? "consistent"
                : (state == eAdd) ? "add"
                                  : (state == eDelete) ? "delete" : "unknown");

  iterator I = begin();
  iterator E = end();

  if (I != E)
    log->PutCString("DYLDRendezvous SOEntries:");

  for (int i = 1; I != E; ++I, ++i) {
    LLDB_LOGF(log, "\n   SOEntry [%d] %s", i, I->file_spec.GetCString());
    LLDB_LOGF(log, "      Base : %" PRIx64, I->base_addr);
    LLDB_LOGF(log, "      Path : %" PRIx64, I->path_addr);
    LLDB_LOGF(log, "      Dyn  : %" PRIx64, I->dyn_addr);
    LLDB_LOGF(log, "      Next : %" PRIx64, I->next);
    LLDB_LOGF(log, "      Prev : %" PRIx64, I->prev);
  }
}
