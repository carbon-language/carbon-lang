//===-- ProcessGDBRemote.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_PROCESSGDBREMOTE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_PROCESSGDBREMOTE_H

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "lldb/Core/LoadedModuleInfoList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/ThreadSafeValue.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/Broadcaster.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/GDBRemote.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StringExtractor.h"
#include "lldb/Utility/StringList.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/lldb-private-forward.h"

#include "GDBRemoteCommunicationClient.h"
#include "GDBRemoteCommunicationReplayServer.h"
#include "GDBRemoteRegisterContext.h"

#include "llvm/ADT/DenseMap.h"

namespace lldb_private {
namespace repro {
class Loader;
}
namespace process_gdb_remote {

class ThreadGDBRemote;

class ProcessGDBRemote : public Process,
                         private GDBRemoteClientBase::ContinueDelegate {
public:
  ProcessGDBRemote(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp);

  ~ProcessGDBRemote() override;

  static lldb::ProcessSP CreateInstance(lldb::TargetSP target_sp,
                                        lldb::ListenerSP listener_sp,
                                        const FileSpec *crash_file_path,
                                        bool can_connect);

  static void Initialize();

  static void DebuggerInitialize(Debugger &debugger);

  static void Terminate();

  static ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  static std::chrono::seconds GetPacketTimeout();

  // Check if a given Process
  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

  CommandObject *GetPluginCommandObject() override;

  // Creating a new process, or attaching to an existing one
  Status WillLaunch(Module *module) override;

  Status DoLaunch(Module *exe_module, ProcessLaunchInfo &launch_info) override;

  void DidLaunch() override;

  Status WillAttachToProcessWithID(lldb::pid_t pid) override;

  Status WillAttachToProcessWithName(const char *process_name,
                                     bool wait_for_launch) override;

  Status DoConnectRemote(llvm::StringRef remote_url) override;

  Status WillLaunchOrAttach();

  Status DoAttachToProcessWithID(lldb::pid_t pid,
                                 const ProcessAttachInfo &attach_info) override;

  Status
  DoAttachToProcessWithName(const char *process_name,
                            const ProcessAttachInfo &attach_info) override;

  void DidAttach(ArchSpec &process_arch) override;

  // PluginInterface protocol
  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  // Process Control
  Status WillResume() override;

  Status DoResume() override;

  Status DoHalt(bool &caused_stop) override;

  Status DoDetach(bool keep_stopped) override;

  bool DetachRequiresHalt() override { return true; }

  Status DoSignal(int signal) override;

  Status DoDestroy() override;

  void RefreshStateAfterStop() override;

  void SetUnixSignals(const lldb::UnixSignalsSP &signals_sp);

  // Process Queries
  bool IsAlive() override;

  lldb::addr_t GetImageInfoAddress() override;

  void WillPublicStop() override;

  // Process Memory
  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                      Status &error) override;

  Status
  WriteObjectFile(std::vector<ObjectFile::LoadableData> entries) override;

  size_t DoWriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                       Status &error) override;

  lldb::addr_t DoAllocateMemory(size_t size, uint32_t permissions,
                                Status &error) override;

  Status GetMemoryRegionInfo(lldb::addr_t load_addr,
                             MemoryRegionInfo &region_info) override;

  Status DoDeallocateMemory(lldb::addr_t ptr) override;

  // Process STDIO
  size_t PutSTDIN(const char *buf, size_t buf_size, Status &error) override;

  // Process Breakpoints
  Status EnableBreakpointSite(BreakpointSite *bp_site) override;

  Status DisableBreakpointSite(BreakpointSite *bp_site) override;

  // Process Watchpoints
  Status EnableWatchpoint(Watchpoint *wp, bool notify = true) override;

  Status DisableWatchpoint(Watchpoint *wp, bool notify = true) override;

  Status GetWatchpointSupportInfo(uint32_t &num) override;

  llvm::Expected<TraceSupportedResponse> TraceSupported() override;

  llvm::Error TraceStop(const TraceStopRequest &request) override;

  llvm::Error TraceStart(const llvm::json::Value &request) override;

  llvm::Expected<std::string> TraceGetState(llvm::StringRef type) override;

  llvm::Expected<std::vector<uint8_t>>
  TraceGetBinaryData(const TraceGetBinaryDataRequest &request) override;

  Status GetWatchpointSupportInfo(uint32_t &num, bool &after) override;

  bool StartNoticingNewThreads() override;

  bool StopNoticingNewThreads() override;

  GDBRemoteCommunicationClient &GetGDBRemote() { return m_gdb_comm; }

  Status SendEventData(const char *data) override;

  // Override DidExit so we can disconnect from the remote GDB server
  void DidExit() override;

  void SetUserSpecifiedMaxMemoryTransferSize(uint64_t user_specified_max);

  bool GetModuleSpec(const FileSpec &module_file_spec, const ArchSpec &arch,
                     ModuleSpec &module_spec) override;

  void PrefetchModuleSpecs(llvm::ArrayRef<FileSpec> module_file_specs,
                           const llvm::Triple &triple) override;

  llvm::VersionTuple GetHostOSVersion() override;
  llvm::VersionTuple GetHostMacCatalystVersion() override;

  llvm::Error LoadModules() override;

  llvm::Expected<LoadedModuleInfoList> GetLoadedModuleList() override;

  Status GetFileLoadAddress(const FileSpec &file, bool &is_loaded,
                            lldb::addr_t &load_addr) override;

  void ModulesDidLoad(ModuleList &module_list) override;

  StructuredData::ObjectSP
  GetLoadedDynamicLibrariesInfos(lldb::addr_t image_list_address,
                                 lldb::addr_t image_count) override;

  Status
  ConfigureStructuredData(ConstString type_name,
                          const StructuredData::ObjectSP &config_sp) override;

  StructuredData::ObjectSP GetLoadedDynamicLibrariesInfos() override;

  StructuredData::ObjectSP GetLoadedDynamicLibrariesInfos(
      const std::vector<lldb::addr_t> &load_addresses) override;

  StructuredData::ObjectSP
  GetLoadedDynamicLibrariesInfos_sender(StructuredData::ObjectSP args);

  StructuredData::ObjectSP GetSharedCacheInfo() override;

  std::string HarmonizeThreadIdsForProfileData(
      StringExtractorGDBRemote &inputStringExtractor);

protected:
  friend class ThreadGDBRemote;
  friend class GDBRemoteCommunicationClient;
  friend class GDBRemoteRegisterContext;

  bool SupportsMemoryTagging() override;

  /// Broadcaster event bits definitions.
  enum {
    eBroadcastBitAsyncContinue = (1 << 0),
    eBroadcastBitAsyncThreadShouldExit = (1 << 1),
    eBroadcastBitAsyncThreadDidExit = (1 << 2)
  };

  GDBRemoteCommunicationClient m_gdb_comm;
  GDBRemoteCommunicationReplayServer m_gdb_replay_server;
  std::atomic<lldb::pid_t> m_debugserver_pid;
  std::vector<StringExtractorGDBRemote> m_stop_packet_stack; // The stop packet
                                                             // stack replaces
                                                             // the last stop
                                                             // packet variable
  std::recursive_mutex m_last_stop_packet_mutex;
  GDBRemoteDynamicRegisterInfoSP m_register_info_sp;
  Broadcaster m_async_broadcaster;
  lldb::ListenerSP m_async_listener_sp;
  HostThread m_async_thread;
  std::recursive_mutex m_async_thread_state_mutex;
  typedef std::vector<lldb::tid_t> tid_collection;
  typedef std::vector<std::pair<lldb::tid_t, int>> tid_sig_collection;
  typedef std::map<lldb::addr_t, lldb::addr_t> MMapMap;
  typedef std::map<uint32_t, std::string> ExpeditedRegisterMap;
  tid_collection m_thread_ids; // Thread IDs for all threads. This list gets
                               // updated after stopping
  std::vector<lldb::addr_t> m_thread_pcs;     // PC values for all the threads.
  StructuredData::ObjectSP m_jstopinfo_sp;    // Stop info only for any threads
                                              // that have valid stop infos
  StructuredData::ObjectSP m_jthreadsinfo_sp; // Full stop info, expedited
                                              // registers and memory for all
                                              // threads if "jThreadsInfo"
                                              // packet is supported
  tid_collection m_continue_c_tids;           // 'c' for continue
  tid_sig_collection m_continue_C_tids;       // 'C' for continue with signal
  tid_collection m_continue_s_tids;           // 's' for step
  tid_sig_collection m_continue_S_tids;       // 'S' for step with signal
  uint64_t m_max_memory_size; // The maximum number of bytes to read/write when
                              // reading and writing memory
  uint64_t m_remote_stub_max_memory_size; // The maximum memory size the remote
                                          // gdb stub can handle
  MMapMap m_addr_to_mmap_size;
  lldb::BreakpointSP m_thread_create_bp_sp;
  bool m_waiting_for_attach;
  bool m_destroy_tried_resuming;
  lldb::CommandObjectSP m_command_sp;
  int64_t m_breakpoint_pc_offset;
  lldb::tid_t m_initial_tid; // The initial thread ID, given by stub on attach
  bool m_use_g_packet_for_reading;

  bool m_replay_mode;
  bool m_allow_flash_writes;
  using FlashRangeVector = lldb_private::RangeVector<lldb::addr_t, size_t>;
  using FlashRange = FlashRangeVector::Entry;
  FlashRangeVector m_erased_flash_ranges;

  // Accessors
  bool IsRunning(lldb::StateType state) {
    return state == lldb::eStateRunning || IsStepping(state);
  }

  bool IsStepping(lldb::StateType state) {
    return state == lldb::eStateStepping;
  }

  bool CanResume(lldb::StateType state) { return state == lldb::eStateStopped; }

  bool HasExited(lldb::StateType state) { return state == lldb::eStateExited; }

  bool ProcessIDIsValid() const;

  void Clear();

  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override;

  Status ConnectToReplayServer();

  Status EstablishConnectionIfNeeded(const ProcessInfo &process_info);

  Status LaunchAndConnectToDebugserver(const ProcessInfo &process_info);

  void KillDebugserverProcess();

  void BuildDynamicRegisterInfo(bool force);

  void SetLastStopPacket(const StringExtractorGDBRemote &response);

  bool ParsePythonTargetDefinition(const FileSpec &target_definition_fspec);

  DataExtractor GetAuxvData() override;

  StructuredData::ObjectSP GetExtendedInfoForThread(lldb::tid_t tid);

  void GetMaxMemorySize();

  bool CalculateThreadStopInfo(ThreadGDBRemote *thread);

  size_t UpdateThreadPCsFromStopReplyThreadsValue(std::string &value);

  size_t UpdateThreadIDsFromStopReplyThreadsValue(llvm::StringRef value);

  bool HandleNotifyPacket(StringExtractorGDBRemote &packet);

  bool StartAsyncThread();

  void StopAsyncThread();

  static lldb::thread_result_t AsyncThread(void *arg);

  static bool
  MonitorDebugserverProcess(std::weak_ptr<ProcessGDBRemote> process_wp,
                            lldb::pid_t pid, bool exited, int signo,
                            int exit_status);

  lldb::StateType SetThreadStopInfo(StringExtractor &stop_packet);

  bool
  GetThreadStopInfoFromJSON(ThreadGDBRemote *thread,
                            const StructuredData::ObjectSP &thread_infos_sp);

  lldb::ThreadSP SetThreadStopInfo(StructuredData::Dictionary *thread_dict);

  lldb::ThreadSP
  SetThreadStopInfo(lldb::tid_t tid,
                    ExpeditedRegisterMap &expedited_register_map, uint8_t signo,
                    const std::string &thread_name, const std::string &reason,
                    const std::string &description, uint32_t exc_type,
                    const std::vector<lldb::addr_t> &exc_data,
                    lldb::addr_t thread_dispatch_qaddr, bool queue_vars_valid,
                    lldb_private::LazyBool associated_with_libdispatch_queue,
                    lldb::addr_t dispatch_queue_t, std::string &queue_name,
                    lldb::QueueKind queue_kind, uint64_t queue_serial);

  void HandleStopReplySequence();

  void ClearThreadIDList();

  bool UpdateThreadIDList();

  void DidLaunchOrAttach(ArchSpec &process_arch);
  void MaybeLoadExecutableModule();

  Status ConnectToDebugserver(llvm::StringRef host_port);

  const char *GetDispatchQueueNameForThread(lldb::addr_t thread_dispatch_qaddr,
                                            std::string &dispatch_queue_name);

  DynamicLoader *GetDynamicLoader() override;

  bool GetGDBServerRegisterInfoXMLAndProcess(ArchSpec &arch_to_use,
                                             std::string xml_filename,
                                             uint32_t &cur_reg_remote,
                                             uint32_t &cur_reg_local);

  // Query remote GDBServer for register information
  bool GetGDBServerRegisterInfo(ArchSpec &arch);

  lldb::ModuleSP LoadModuleAtAddress(const FileSpec &file,
                                     lldb::addr_t link_map,
                                     lldb::addr_t base_addr,
                                     bool value_is_offset);

  Status UpdateAutomaticSignalFiltering() override;

  Status FlashErase(lldb::addr_t addr, size_t size);

  Status FlashDone();

  bool HasErased(FlashRange range);

  llvm::Expected<std::vector<uint8_t>>
  DoReadMemoryTags(lldb::addr_t addr, size_t len, int32_t type) override;

  Status DoWriteMemoryTags(lldb::addr_t addr, size_t len, int32_t type,
                           const std::vector<uint8_t> &tags) override;

private:
  // For ProcessGDBRemote only
  std::string m_partial_profile_data;
  std::map<uint64_t, uint32_t> m_thread_id_to_used_usec_map;
  uint64_t m_last_signals_version = 0;

  static bool NewThreadNotifyBreakpointHit(void *baton,
                                           StoppointCallbackContext *context,
                                           lldb::user_id_t break_id,
                                           lldb::user_id_t break_loc_id);

  // ContinueDelegate interface
  void HandleAsyncStdout(llvm::StringRef out) override;
  void HandleAsyncMisc(llvm::StringRef data) override;
  void HandleStopReply() override;
  void HandleAsyncStructuredDataPacket(llvm::StringRef data) override;

  void SetThreadPc(const lldb::ThreadSP &thread_sp, uint64_t index);
  using ModuleCacheKey = std::pair<std::string, std::string>;
  // KeyInfo for the cached module spec DenseMap.
  // The invariant is that all real keys will have the file and architecture
  // set.
  // The empty key has an empty file and an empty arch.
  // The tombstone key has an invalid arch and an empty file.
  // The comparison and hash functions take the file name and architecture
  // triple into account.
  struct ModuleCacheInfo {
    static ModuleCacheKey getEmptyKey() { return ModuleCacheKey(); }

    static ModuleCacheKey getTombstoneKey() { return ModuleCacheKey("", "T"); }

    static unsigned getHashValue(const ModuleCacheKey &key) {
      return llvm::hash_combine(key.first, key.second);
    }

    static bool isEqual(const ModuleCacheKey &LHS, const ModuleCacheKey &RHS) {
      return LHS == RHS;
    }
  };

  llvm::DenseMap<ModuleCacheKey, ModuleSpec, ModuleCacheInfo>
      m_cached_module_specs;

  ProcessGDBRemote(const ProcessGDBRemote &) = delete;
  const ProcessGDBRemote &operator=(const ProcessGDBRemote &) = delete;
};

} // namespace process_gdb_remote
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_PROCESSGDBREMOTE_H
