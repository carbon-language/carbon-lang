//===-- Debugger.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_DEBUGGER_H
#define LLDB_CORE_DEBUGGER_H

#include <cstdint>

#include <memory>
#include <vector>

#include "lldb/Core/DebuggerEvents.h"
#include "lldb/Core/FormatEntity.h"
#include "lldb/Core/IOHandler.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/Terminal.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Utility/Broadcaster.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UserID.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-private-types.h"
#include "lldb/lldb-types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Threading.h"

#include <cassert>
#include <cstddef>
#include <cstdio>

namespace llvm {
class raw_ostream;
class ThreadPool;
}

namespace lldb_private {
class Address;
class CommandInterpreter;
class Process;
class Stream;
class SymbolContext;
class Target;

namespace repro {
class DataRecorder;
}

/// \class Debugger Debugger.h "lldb/Core/Debugger.h"
/// A class to manage flag bits.
///
/// Provides a global root objects for the debugger core.

class Debugger : public std::enable_shared_from_this<Debugger>,
                 public UserID,
                 public Properties {
  friend class SourceManager; // For GetSourceFileCache.

public:
  /// Broadcaster event bits definitions.
  enum {
    eBroadcastBitProgress = (1 << 0),
    eBroadcastBitWarning = (1 << 1),
    eBroadcastBitError = (1 << 2),
  };

  static ConstString GetStaticBroadcasterClass();

  /// Get the public broadcaster for this debugger.
  Broadcaster &GetBroadcaster() { return m_broadcaster; }
  const Broadcaster &GetBroadcaster() const { return m_broadcaster; }

  ~Debugger() override;

  static lldb::DebuggerSP
  CreateInstance(lldb::LogOutputCallback log_callback = nullptr,
                 void *baton = nullptr);

  static lldb::TargetSP FindTargetWithProcessID(lldb::pid_t pid);

  static lldb::TargetSP FindTargetWithProcess(Process *process);

  static void Initialize(LoadPluginCallbackType load_plugin_callback);

  static void Terminate();

  static void SettingsInitialize();

  static void SettingsTerminate();

  static void Destroy(lldb::DebuggerSP &debugger_sp);

  static lldb::DebuggerSP FindDebuggerWithID(lldb::user_id_t id);

  static lldb::DebuggerSP
  FindDebuggerWithInstanceName(ConstString instance_name);

  static size_t GetNumDebuggers();

  static lldb::DebuggerSP GetDebuggerAtIndex(size_t index);

  static bool FormatDisassemblerAddress(const FormatEntity::Entry *format,
                                        const SymbolContext *sc,
                                        const SymbolContext *prev_sc,
                                        const ExecutionContext *exe_ctx,
                                        const Address *addr, Stream &s);

  void Clear();

  bool GetAsyncExecution();

  void SetAsyncExecution(bool async);

  lldb::FileSP GetInputFileSP() { return m_input_file_sp; }

  lldb::StreamFileSP GetOutputStreamSP() { return m_output_stream_sp; }

  lldb::StreamFileSP GetErrorStreamSP() { return m_error_stream_sp; }

  File &GetInputFile() { return *m_input_file_sp; }

  File &GetOutputFile() { return m_output_stream_sp->GetFile(); }

  File &GetErrorFile() { return m_error_stream_sp->GetFile(); }

  StreamFile &GetOutputStream() { return *m_output_stream_sp; }

  StreamFile &GetErrorStream() { return *m_error_stream_sp; }

  repro::DataRecorder *GetInputRecorder();

  Status SetInputString(const char *data);

  // This method will setup data recorder if reproducer enabled.
  // On reply mode this method should take instructions from reproducer file.
  Status SetInputFile(lldb::FileSP file);

  void SetInputFile(lldb::FileSP file, repro::DataRecorder *recorder);

  void SetOutputFile(lldb::FileSP file);

  void SetErrorFile(lldb::FileSP file);

  void SaveInputTerminalState();

  void RestoreInputTerminalState();

  lldb::StreamSP GetAsyncOutputStream();

  lldb::StreamSP GetAsyncErrorStream();

  CommandInterpreter &GetCommandInterpreter() {
    assert(m_command_interpreter_up.get());
    return *m_command_interpreter_up;
  }

  ScriptInterpreter *
  GetScriptInterpreter(bool can_create = true,
                       llvm::Optional<lldb::ScriptLanguage> language = {});

  lldb::ListenerSP GetListener() { return m_listener_sp; }

  // This returns the Debugger's scratch source manager.  It won't be able to
  // look up files in debug information, but it can look up files by absolute
  // path and display them to you. To get the target's source manager, call
  // GetSourceManager on the target instead.
  SourceManager &GetSourceManager();

  lldb::TargetSP GetSelectedTarget() {
    return m_target_list.GetSelectedTarget();
  }

  ExecutionContext GetSelectedExecutionContext();
  /// Get accessor for the target list.
  ///
  /// The target list is part of the global debugger object. This the single
  /// debugger shared instance to control where targets get created and to
  /// allow for tracking and searching for targets based on certain criteria.
  ///
  /// \return
  ///     A global shared target list.
  TargetList &GetTargetList() { return m_target_list; }

  PlatformList &GetPlatformList() { return m_platform_list; }

  void DispatchInputInterrupt();

  void DispatchInputEndOfFile();

  // If any of the streams are not set, set them to the in/out/err stream of
  // the top most input reader to ensure they at least have something
  void AdoptTopIOHandlerFilesIfInvalid(lldb::FileSP &in,
                                       lldb::StreamFileSP &out,
                                       lldb::StreamFileSP &err);

  /// Run the given IO handler and return immediately.
  void RunIOHandlerAsync(const lldb::IOHandlerSP &reader_sp,
                         bool cancel_top_handler = true);

  /// Run the given IO handler and block until it's complete.
  void RunIOHandlerSync(const lldb::IOHandlerSP &reader_sp);

  ///  Remove the given IO handler if it's currently active.
  bool RemoveIOHandler(const lldb::IOHandlerSP &reader_sp);

  bool IsTopIOHandler(const lldb::IOHandlerSP &reader_sp);

  bool CheckTopIOHandlerTypes(IOHandler::Type top_type,
                              IOHandler::Type second_top_type);

  void PrintAsync(const char *s, size_t len, bool is_stdout);

  ConstString GetTopIOHandlerControlSequence(char ch);

  const char *GetIOHandlerCommandPrefix();

  const char *GetIOHandlerHelpPrologue();

  void ClearIOHandlers();

  bool GetCloseInputOnEOF() const;

  void SetCloseInputOnEOF(bool b);

  bool EnableLog(llvm::StringRef channel,
                 llvm::ArrayRef<const char *> categories,
                 llvm::StringRef log_file, uint32_t log_options,
                 llvm::raw_ostream &error_stream);

  void SetLoggingCallback(lldb::LogOutputCallback log_callback, void *baton);

  // Properties Functions
  enum StopDisassemblyType {
    eStopDisassemblyTypeNever = 0,
    eStopDisassemblyTypeNoDebugInfo,
    eStopDisassemblyTypeNoSource,
    eStopDisassemblyTypeAlways
  };

  Status SetPropertyValue(const ExecutionContext *exe_ctx,
                          VarSetOperationType op, llvm::StringRef property_path,
                          llvm::StringRef value) override;

  bool GetAutoConfirm() const;

  const FormatEntity::Entry *GetDisassemblyFormat() const;

  const FormatEntity::Entry *GetFrameFormat() const;

  const FormatEntity::Entry *GetFrameFormatUnique() const;

  uint32_t GetStopDisassemblyMaxSize() const;

  const FormatEntity::Entry *GetThreadFormat() const;

  const FormatEntity::Entry *GetThreadStopFormat() const;

  lldb::ScriptLanguage GetScriptLanguage() const;

  bool SetScriptLanguage(lldb::ScriptLanguage script_lang);

  lldb::LanguageType GetREPLLanguage() const;

  bool SetREPLLanguage(lldb::LanguageType repl_lang);

  uint32_t GetTerminalWidth() const;

  bool SetTerminalWidth(uint32_t term_width);

  llvm::StringRef GetPrompt() const;

  void SetPrompt(llvm::StringRef p);
  void SetPrompt(const char *) = delete;

  llvm::StringRef GetReproducerPath() const;

  bool GetUseExternalEditor() const;

  bool SetUseExternalEditor(bool use_external_editor_p);

  bool GetUseColor() const;

  bool SetUseColor(bool use_color);

  bool GetShowProgress() const;

  bool SetShowProgress(bool show_progress);

  llvm::StringRef GetShowProgressAnsiPrefix() const;

  llvm::StringRef GetShowProgressAnsiSuffix() const;

  bool GetUseAutosuggestion() const;

  llvm::StringRef GetAutosuggestionAnsiPrefix() const;

  llvm::StringRef GetAutosuggestionAnsiSuffix() const;

  bool GetUseSourceCache() const;

  bool SetUseSourceCache(bool use_source_cache);

  bool GetHighlightSource() const;

  lldb::StopShowColumn GetStopShowColumn() const;

  llvm::StringRef GetStopShowColumnAnsiPrefix() const;

  llvm::StringRef GetStopShowColumnAnsiSuffix() const;

  uint32_t GetStopSourceLineCount(bool before) const;

  StopDisassemblyType GetStopDisassemblyDisplay() const;

  uint32_t GetDisassemblyLineCount() const;

  llvm::StringRef GetStopShowLineMarkerAnsiPrefix() const;

  llvm::StringRef GetStopShowLineMarkerAnsiSuffix() const;

  bool GetAutoOneLineSummaries() const;

  bool GetAutoIndent() const;

  bool SetAutoIndent(bool b);

  bool GetPrintDecls() const;

  bool SetPrintDecls(bool b);

  uint32_t GetTabSize() const;

  bool SetTabSize(uint32_t tab_size);

  bool GetEscapeNonPrintables() const;

  bool GetNotifyVoid() const;

  ConstString GetInstanceName() { return m_instance_name; }

  bool LoadPlugin(const FileSpec &spec, Status &error);

  void RunIOHandlers();

  bool IsForwardingEvents();

  void EnableForwardEvents(const lldb::ListenerSP &listener_sp);

  void CancelForwardEvents(const lldb::ListenerSP &listener_sp);

  bool IsHandlingEvents() const { return m_event_handler_thread.IsJoinable(); }

  Status RunREPL(lldb::LanguageType language, const char *repl_options);

  // This is for use in the command interpreter, when you either want the
  // selected target, or if no target is present you want to prime the dummy
  // target with entities that will be copied over to new targets.
  Target &GetSelectedOrDummyTarget(bool prefer_dummy = false);
  Target &GetDummyTarget() { return *m_dummy_target_sp; }

  lldb::BroadcasterManagerSP GetBroadcasterManager() {
    return m_broadcaster_manager_sp;
  }

  /// Shared thread poll. Use only with ThreadPoolTaskGroup.
  static llvm::ThreadPool &GetThreadPool();

  /// Report warning events.
  ///
  /// Progress events will be delivered to any debuggers that have listeners
  /// for the eBroadcastBitError.
  ///
  /// \param[in] message
  ///   The warning message to be reported.
  ///
  /// \param [in] debugger_id
  ///   If this optional parameter has a value, it indicates the unique
  ///   debugger identifier that this progress should be delivered to. If this
  ///   optional parameter does not have a value, the progress will be
  ///   delivered to all debuggers.
  ///
  /// \param [in] once
  ///   If a pointer is passed to a std::once_flag, then it will be used to
  ///   ensure the given warning is only broadcast once.
  static void
  ReportWarning(std::string messsage,
                llvm::Optional<lldb::user_id_t> debugger_id = llvm::None,
                std::once_flag *once = nullptr);

  /// Report error events.
  ///
  /// Progress events will be delivered to any debuggers that have listeners
  /// for the eBroadcastBitError.
  ///
  /// \param[in] message
  ///   The error message to be reported.
  ///
  /// \param [in] debugger_id
  ///   If this optional parameter has a value, it indicates the unique
  ///   debugger identifier that this progress should be delivered to. If this
  ///   optional parameter does not have a value, the progress will be
  ///   delivered to all debuggers.
  ///
  /// \param [in] once
  ///   If a pointer is passed to a std::once_flag, then it will be used to
  ///   ensure the given error is only broadcast once.
  static void
  ReportError(std::string messsage,
              llvm::Optional<lldb::user_id_t> debugger_id = llvm::None,
              std::once_flag *once = nullptr);

protected:
  friend class CommandInterpreter;
  friend class REPL;
  friend class Progress;

  /// Report progress events.
  ///
  /// Progress events will be delivered to any debuggers that have listeners
  /// for the eBroadcastBitProgress. This function is called by the
  /// lldb_private::Progress class to deliver the events to any debuggers that
  /// qualify.
  ///
  /// \param [in] progress_id
  ///   The unique integer identifier for the progress to report.
  ///
  /// \param[in] message
  ///   The title of the progress dialog to display in the UI.
  ///
  /// \param [in] completed
  ///   The amount of work completed. If \a completed is zero, then this event
  ///   is a progress started event. If \a completed is equal to \a total, then
  ///   this event is a progress end event. Otherwise completed indicates the
  ///   current progress compare to the total value.
  ///
  /// \param [in] total
  ///   The total amount of work units that need to be completed. If this value
  ///   is UINT64_MAX, then an indeterminate progress indicator should be
  ///   displayed.
  ///
  /// \param [in] debugger_id
  ///   If this optional parameter has a value, it indicates the unique
  ///   debugger identifier that this progress should be delivered to. If this
  ///   optional parameter does not have a value, the progress will be
  ///   delivered to all debuggers.
  static void ReportProgress(uint64_t progress_id, const std::string &message,
                             uint64_t completed, uint64_t total,
                             llvm::Optional<lldb::user_id_t> debugger_id);

  static void ReportDiagnosticImpl(DiagnosticEventData::Type type,
                                   std::string message,
                                   llvm::Optional<lldb::user_id_t> debugger_id,
                                   std::once_flag *once);

  void PrintProgress(const ProgressEventData &data);

  bool StartEventHandlerThread();

  void StopEventHandlerThread();

  void PushIOHandler(const lldb::IOHandlerSP &reader_sp,
                     bool cancel_top_handler = true);

  bool PopIOHandler(const lldb::IOHandlerSP &reader_sp);

  bool HasIOHandlerThread();

  bool StartIOHandlerThread();

  void StopIOHandlerThread();

  void JoinIOHandlerThread();

  lldb::thread_result_t IOHandlerThread();

  lldb::thread_result_t DefaultEventHandler();

  void HandleBreakpointEvent(const lldb::EventSP &event_sp);

  void HandleProcessEvent(const lldb::EventSP &event_sp);

  void HandleThreadEvent(const lldb::EventSP &event_sp);

  void HandleProgressEvent(const lldb::EventSP &event_sp);

  void HandleDiagnosticEvent(const lldb::EventSP &event_sp);

  // Ensures two threads don't attempt to flush process output in parallel.
  std::mutex m_output_flush_mutex;
  void FlushProcessOutput(Process &process, bool flush_stdout,
                          bool flush_stderr);

  SourceManager::SourceFileCache &GetSourceFileCache() {
    return m_source_file_cache;
  }

  void InstanceInitialize();

  // these should never be NULL
  lldb::FileSP m_input_file_sp;
  lldb::StreamFileSP m_output_stream_sp;
  lldb::StreamFileSP m_error_stream_sp;

  /// Used for shadowing the input file when capturing a reproducer.
  repro::DataRecorder *m_input_recorder;

  lldb::BroadcasterManagerSP m_broadcaster_manager_sp; // The debugger acts as a
                                                       // broadcaster manager of
                                                       // last resort.
  // It needs to get constructed before the target_list or any other member
  // that might want to broadcast through the debugger.

  TerminalState m_terminal_state;
  TargetList m_target_list;

  PlatformList m_platform_list;
  lldb::ListenerSP m_listener_sp;
  std::unique_ptr<SourceManager> m_source_manager_up; // This is a scratch
                                                      // source manager that we
                                                      // return if we have no
                                                      // targets.
  SourceManager::SourceFileCache m_source_file_cache; // All the source managers
                                                      // for targets created in
                                                      // this debugger used this
                                                      // shared
                                                      // source file cache.
  std::unique_ptr<CommandInterpreter> m_command_interpreter_up;

  std::recursive_mutex m_script_interpreter_mutex;
  std::array<lldb::ScriptInterpreterSP, lldb::eScriptLanguageUnknown>
      m_script_interpreters;

  IOHandlerStack m_io_handler_stack;
  std::recursive_mutex m_io_handler_synchronous_mutex;

  llvm::Optional<uint64_t> m_current_event_id;

  llvm::StringMap<std::weak_ptr<llvm::raw_ostream>> m_log_streams;
  std::shared_ptr<llvm::raw_ostream> m_log_callback_stream_sp;
  ConstString m_instance_name;
  static LoadPluginCallbackType g_load_plugin_callback;
  typedef std::vector<llvm::sys::DynamicLibrary> LoadedPluginsList;
  LoadedPluginsList m_loaded_plugins;
  HostThread m_event_handler_thread;
  HostThread m_io_handler_thread;
  Broadcaster m_sync_broadcaster; ///< Private debugger synchronization.
  Broadcaster m_broadcaster;      ///< Public Debugger event broadcaster.
  lldb::ListenerSP m_forward_listener_sp;
  llvm::once_flag m_clear_once;
  lldb::TargetSP m_dummy_target_sp;

  // Events for m_sync_broadcaster
  enum {
    eBroadcastBitEventThreadIsListening = (1 << 0),
  };

private:
  // Use Debugger::CreateInstance() to get a shared pointer to a new debugger
  // object
  Debugger(lldb::LogOutputCallback m_log_callback, void *baton);

  Debugger(const Debugger &) = delete;
  const Debugger &operator=(const Debugger &) = delete;
};

} // namespace lldb_private

#endif // LLDB_CORE_DEBUGGER_H
