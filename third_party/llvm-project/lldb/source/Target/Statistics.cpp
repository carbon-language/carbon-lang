//===-- Statistics.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Statistics.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnixSignals.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

static void EmplaceSafeString(llvm::json::Object &obj, llvm::StringRef key,
                              const std::string &str) {
  if (str.empty())
    return;
  if (LLVM_LIKELY(llvm::json::isUTF8(str)))
    obj.try_emplace(key, str);
  else
    obj.try_emplace(key, llvm::json::fixUTF8(str));
}

json::Value StatsSuccessFail::ToJSON() const {
  return json::Object{{"successes", successes}, {"failures", failures}};
}

static double elapsed(const StatsTimepoint &start, const StatsTimepoint &end) {
  StatsDuration elapsed = end.time_since_epoch() - start.time_since_epoch();
  return elapsed.count();
}

void TargetStats::CollectStats(Target &target) {
  m_module_identifiers.clear();
  for (ModuleSP module_sp : target.GetImages().Modules())
    m_module_identifiers.emplace_back((intptr_t)module_sp.get());
}

json::Value ModuleStats::ToJSON() const {
  json::Object module;
  EmplaceSafeString(module, "path", path);
  EmplaceSafeString(module, "uuid", uuid);
  EmplaceSafeString(module, "triple", triple);
  module.try_emplace("identifier", identifier);
  module.try_emplace("symbolTableParseTime", symtab_parse_time);
  module.try_emplace("symbolTableIndexTime", symtab_index_time);
  module.try_emplace("symbolTableLoadedFromCache", symtab_loaded_from_cache);
  module.try_emplace("symbolTableSavedToCache", symtab_saved_to_cache);
  module.try_emplace("debugInfoParseTime", debug_parse_time);
  module.try_emplace("debugInfoIndexTime", debug_index_time);
  module.try_emplace("debugInfoByteSize", (int64_t)debug_info_size);
  module.try_emplace("debugInfoIndexLoadedFromCache",
                     debug_info_index_loaded_from_cache);
  module.try_emplace("debugInfoIndexSavedToCache",
                     debug_info_index_saved_to_cache);
  return module;
}

json::Value TargetStats::ToJSON(Target &target) {
  CollectStats(target);

  json::Array json_module_uuid_array;
  for (auto module_identifier : m_module_identifiers)
    json_module_uuid_array.emplace_back(module_identifier);

  json::Object target_metrics_json{
      {m_expr_eval.name, m_expr_eval.ToJSON()},
      {m_frame_var.name, m_frame_var.ToJSON()},
      {"moduleIdentifiers", std::move(json_module_uuid_array)}};

  if (m_launch_or_attach_time && m_first_private_stop_time) {
    double elapsed_time =
        elapsed(*m_launch_or_attach_time, *m_first_private_stop_time);
    target_metrics_json.try_emplace("launchOrAttachTime", elapsed_time);
  }
  if (m_launch_or_attach_time && m_first_public_stop_time) {
    double elapsed_time =
        elapsed(*m_launch_or_attach_time, *m_first_public_stop_time);
    target_metrics_json.try_emplace("firstStopTime", elapsed_time);
  }
  target_metrics_json.try_emplace("targetCreateTime", m_create_time.count());

  json::Array breakpoints_array;
  double totalBreakpointResolveTime = 0.0;
  // Rport both the normal breakpoint list and the internal breakpoint list.
  for (int i = 0; i < 2; ++i) {
    BreakpointList &breakpoints = target.GetBreakpointList(i == 1);
    std::unique_lock<std::recursive_mutex> lock;
    breakpoints.GetListMutex(lock);
    size_t num_breakpoints = breakpoints.GetSize();
    for (size_t i = 0; i < num_breakpoints; i++) {
      Breakpoint *bp = breakpoints.GetBreakpointAtIndex(i).get();
      breakpoints_array.push_back(bp->GetStatistics());
      totalBreakpointResolveTime += bp->GetResolveTime().count();
    }
  }

  ProcessSP process_sp = target.GetProcessSP();
  if (process_sp) {
    UnixSignalsSP unix_signals_sp = process_sp->GetUnixSignals();
    if (unix_signals_sp)
      target_metrics_json.try_emplace("signals",
                                      unix_signals_sp->GetHitCountStatistics());
    uint32_t stop_id = process_sp->GetStopID();
    target_metrics_json.try_emplace("stopCount", stop_id);
  }
  target_metrics_json.try_emplace("breakpoints", std::move(breakpoints_array));
  target_metrics_json.try_emplace("totalBreakpointResolveTime",
                                  totalBreakpointResolveTime);

  return target_metrics_json;
}

void TargetStats::SetLaunchOrAttachTime() {
  m_launch_or_attach_time = StatsClock::now();
  m_first_private_stop_time = llvm::None;
}

void TargetStats::SetFirstPrivateStopTime() {
  // Launching and attaching has many paths depending on if synchronous mode
  // was used or if we are stopping at the entry point or not. Only set the
  // first stop time if it hasn't already been set.
  if (!m_first_private_stop_time)
    m_first_private_stop_time = StatsClock::now();
}

void TargetStats::SetFirstPublicStopTime() {
  // Launching and attaching has many paths depending on if synchronous mode
  // was used or if we are stopping at the entry point or not. Only set the
  // first stop time if it hasn't already been set.
  if (!m_first_public_stop_time)
    m_first_public_stop_time = StatsClock::now();
}

bool DebuggerStats::g_collecting_stats = false;

llvm::json::Value DebuggerStats::ReportStatistics(Debugger &debugger,
                                                  Target *target) {
  json::Array json_targets;
  json::Array json_modules;
  double symtab_parse_time = 0.0;
  double symtab_index_time = 0.0;
  double debug_parse_time = 0.0;
  double debug_index_time = 0.0;
  uint32_t symtabs_loaded = 0;
  uint32_t symtabs_saved = 0;
  uint32_t debug_index_loaded = 0;
  uint32_t debug_index_saved = 0;
  uint64_t debug_info_size = 0;
  if (target) {
    json_targets.emplace_back(target->ReportStatistics());
  } else {
    for (const auto &target : debugger.GetTargetList().Targets())
      json_targets.emplace_back(target->ReportStatistics());
  }
  std::vector<ModuleStats> modules;
  std::lock_guard<std::recursive_mutex> guard(
      Module::GetAllocationModuleCollectionMutex());
  const size_t num_modules = Module::GetNumberAllocatedModules();
  for (size_t image_idx = 0; image_idx < num_modules; ++image_idx) {
    Module *module = Module::GetAllocatedModuleAtIndex(image_idx);
    ModuleStats module_stat;
    module_stat.identifier = (intptr_t)module;
    module_stat.path = module->GetFileSpec().GetPath();
    if (ConstString object_name = module->GetObjectName()) {
      module_stat.path.append(1, '(');
      module_stat.path.append(object_name.GetStringRef().str());
      module_stat.path.append(1, ')');
    }
    module_stat.uuid = module->GetUUID().GetAsString();
    module_stat.triple = module->GetArchitecture().GetTriple().str();
    module_stat.symtab_parse_time = module->GetSymtabParseTime().count();
    module_stat.symtab_index_time = module->GetSymtabIndexTime().count();
    Symtab *symtab = module->GetSymtab();
    if (symtab) {
      module_stat.symtab_loaded_from_cache = symtab->GetWasLoadedFromCache();
      if (module_stat.symtab_loaded_from_cache)
        ++symtabs_loaded;
      module_stat.symtab_saved_to_cache = symtab->GetWasSavedToCache();
      if (module_stat.symtab_saved_to_cache)
        ++symtabs_saved;
    }
    SymbolFile *sym_file = module->GetSymbolFile();
    if (sym_file) {
      module_stat.debug_index_time = sym_file->GetDebugInfoIndexTime().count();
      module_stat.debug_parse_time = sym_file->GetDebugInfoParseTime().count();
      module_stat.debug_info_size = sym_file->GetDebugInfoSize();
      module_stat.debug_info_index_loaded_from_cache =
          sym_file->GetDebugInfoIndexWasLoadedFromCache();
      if (module_stat.debug_info_index_loaded_from_cache)
        ++debug_index_loaded;
      module_stat.debug_info_index_saved_to_cache =
          sym_file->GetDebugInfoIndexWasSavedToCache();
      if (module_stat.debug_info_index_saved_to_cache)
        ++debug_index_saved;
    }
    symtab_parse_time += module_stat.symtab_parse_time;
    symtab_index_time += module_stat.symtab_index_time;
    debug_parse_time += module_stat.debug_parse_time;
    debug_index_time += module_stat.debug_index_time;
    debug_info_size += module_stat.debug_info_size;
    json_modules.emplace_back(module_stat.ToJSON());
  }

  json::Object global_stats{
      {"targets", std::move(json_targets)},
      {"modules", std::move(json_modules)},
      {"totalSymbolTableParseTime", symtab_parse_time},
      {"totalSymbolTableIndexTime", symtab_index_time},
      {"totalSymbolTablesLoadedFromCache", symtabs_loaded},
      {"totalSymbolTablesSavedToCache", symtabs_saved},
      {"totalDebugInfoParseTime", debug_parse_time},
      {"totalDebugInfoIndexTime", debug_index_time},
      {"totalDebugInfoIndexLoadedFromCache", debug_index_loaded},
      {"totalDebugInfoIndexSavedToCache", debug_index_saved},
      {"totalDebugInfoByteSize", debug_info_size},
  };
  return std::move(global_stats);
}
