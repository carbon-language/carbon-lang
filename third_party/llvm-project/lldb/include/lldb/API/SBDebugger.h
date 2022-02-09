//===-- SBDebugger.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBDEBUGGER_H
#define LLDB_API_SBDEBUGGER_H

#include <cstdio>

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBPlatform.h"

namespace lldb {

class LLDB_API SBInputReader {
public:
  SBInputReader() = default;
  ~SBInputReader() = default;

  SBError Initialize(lldb::SBDebugger &sb_debugger,
                     unsigned long (*callback)(void *, lldb::SBInputReader *,
                                               lldb::InputReaderAction,
                                               char const *, unsigned long),
                     void *a, lldb::InputReaderGranularity b, char const *c,
                     char const *d, bool e);
  void SetIsDone(bool);
  bool IsActive() const;
};

class LLDB_API SBDebugger {
public:
  FLAGS_ANONYMOUS_ENUM(){eBroadcastBitProgress = (1 << 0)};

  SBDebugger();

  SBDebugger(const lldb::SBDebugger &rhs);

  SBDebugger(const lldb::DebuggerSP &debugger_sp);

  ~SBDebugger();

  static const char *GetBroadcasterClass();

  lldb::SBBroadcaster GetBroadcaster();

  /// Get progress data from a SBEvent whose type is eBroadcastBitProgress.
  ///
  /// \param [in] event
  ///   The event to extract the progress information from.
  ///
  /// \param [out] progress_id
  ///   The unique integer identifier for the progress to report.
  ///
  /// \param [out] completed
  ///   The amount of work completed. If \a completed is zero, then this event
  ///   is a progress started event. If \a completed is equal to \a total, then
  ///   this event is a progress end event. Otherwise completed indicates the
  ///   current progress update.
  ///
  /// \param [out] total
  ///   The total amount of work units that need to be completed. If this value
  ///   is UINT64_MAX, then an indeterminate progress indicator should be
  ///   displayed.
  ///
  /// \param [out] is_debugger_specific
  ///   Set to true if this progress is specific to this debugger only. Many
  ///   progress events are not specific to a debugger instance, like any
  ///   progress events for loading information in modules since LLDB has a
  ///   global module cache that all debuggers use.
  ///
  /// \return The message for the progress. If the returned value is NULL, then
  ///   \a event was not a eBroadcastBitProgress event.
  static const char *GetProgressFromEvent(const lldb::SBEvent &event,
                                          uint64_t &progress_id,
                                          uint64_t &completed, uint64_t &total,
                                          bool &is_debugger_specific);

  lldb::SBDebugger &operator=(const lldb::SBDebugger &rhs);

  static void Initialize();

  static lldb::SBError InitializeWithErrorHandling();

  static void Terminate();

  // Deprecated, use the one that takes a source_init_files bool.
  static lldb::SBDebugger Create();

  static lldb::SBDebugger Create(bool source_init_files);

  static lldb::SBDebugger Create(bool source_init_files,
                                 lldb::LogOutputCallback log_callback,
                                 void *baton);

  static void Destroy(lldb::SBDebugger &debugger);

  static void MemoryPressureDetected();

  explicit operator bool() const;

  bool IsValid() const;

  void Clear();

  void SetAsync(bool b);

  bool GetAsync();

  void SkipLLDBInitFiles(bool b);

  void SkipAppInitFiles(bool b);

  void SetInputFileHandle(FILE *f, bool transfer_ownership);

  void SetOutputFileHandle(FILE *f, bool transfer_ownership);

  void SetErrorFileHandle(FILE *f, bool transfer_ownership);

  FILE *GetInputFileHandle();

  FILE *GetOutputFileHandle();

  FILE *GetErrorFileHandle();

  SBError SetInputString(const char *data);

  SBError SetInputFile(SBFile file);

  SBError SetOutputFile(SBFile file);

  SBError SetErrorFile(SBFile file);

  SBError SetInputFile(FileSP file);

  SBError SetOutputFile(FileSP file);

  SBError SetErrorFile(FileSP file);

  SBFile GetInputFile();

  SBFile GetOutputFile();

  SBFile GetErrorFile();

  void SaveInputTerminalState();

  void RestoreInputTerminalState();

  lldb::SBCommandInterpreter GetCommandInterpreter();

  void HandleCommand(const char *command);

  lldb::SBListener GetListener();

  void HandleProcessEvent(const lldb::SBProcess &process,
                          const lldb::SBEvent &event, FILE *out,
                          FILE *err); // DEPRECATED

  void HandleProcessEvent(const lldb::SBProcess &process,
                          const lldb::SBEvent &event, SBFile out, SBFile err);

  void HandleProcessEvent(const lldb::SBProcess &process,
                          const lldb::SBEvent &event, FileSP out, FileSP err);

  lldb::SBTarget CreateTarget(const char *filename, const char *target_triple,
                              const char *platform_name,
                              bool add_dependent_modules, lldb::SBError &error);

  lldb::SBTarget CreateTargetWithFileAndTargetTriple(const char *filename,
                                                     const char *target_triple);

  lldb::SBTarget CreateTargetWithFileAndArch(const char *filename,
                                             const char *archname);

  lldb::SBTarget CreateTarget(const char *filename);

  lldb::SBTarget GetDummyTarget();

  // Return true if target is deleted from the target list of the debugger.
  bool DeleteTarget(lldb::SBTarget &target);

  lldb::SBTarget GetTargetAtIndex(uint32_t idx);

  uint32_t GetIndexOfTarget(lldb::SBTarget target);

  lldb::SBTarget FindTargetWithProcessID(pid_t pid);

  lldb::SBTarget FindTargetWithFileAndArch(const char *filename,
                                           const char *arch);

  uint32_t GetNumTargets();

  lldb::SBTarget GetSelectedTarget();

  void SetSelectedTarget(SBTarget &target);

  lldb::SBPlatform GetSelectedPlatform();

  void SetSelectedPlatform(lldb::SBPlatform &platform);

  /// Get the number of currently active platforms.
  uint32_t GetNumPlatforms();

  /// Get one of the currently active platforms.
  lldb::SBPlatform GetPlatformAtIndex(uint32_t idx);

  /// Get the number of available platforms.
  ///
  /// The return value should match the number of entries output by the
  /// "platform list" command.
  uint32_t GetNumAvailablePlatforms();

  /// Get the name and description of one of the available platforms.
  ///
  /// \param[in] idx
  ///     Zero-based index of the platform for which info should be retrieved,
  ///     must be less than the value returned by GetNumAvailablePlatforms().
  lldb::SBStructuredData GetAvailablePlatformInfoAtIndex(uint32_t idx);

  lldb::SBSourceManager GetSourceManager();

  // REMOVE: just for a quick fix, need to expose platforms through
  // SBPlatform from this class.
  lldb::SBError SetCurrentPlatform(const char *platform_name);

  bool SetCurrentPlatformSDKRoot(const char *sysroot);

  // FIXME: Once we get the set show stuff in place, the driver won't need
  // an interface to the Set/Get UseExternalEditor.
  bool SetUseExternalEditor(bool input);

  bool GetUseExternalEditor();

  bool SetUseColor(bool use_color);

  bool GetUseColor() const;

  bool SetUseSourceCache(bool use_source_cache);

  bool GetUseSourceCache() const;

  static bool GetDefaultArchitecture(char *arch_name, size_t arch_name_len);

  static bool SetDefaultArchitecture(const char *arch_name);

  lldb::ScriptLanguage GetScriptingLanguage(const char *script_language_name);

  SBStructuredData GetScriptInterpreterInfo(ScriptLanguage);

  static const char *GetVersionString();

  static const char *StateAsCString(lldb::StateType state);

  static SBStructuredData GetBuildConfiguration();

  static bool StateIsRunningState(lldb::StateType state);

  static bool StateIsStoppedState(lldb::StateType state);

  bool EnableLog(const char *channel, const char **categories);

  void SetLoggingCallback(lldb::LogOutputCallback log_callback, void *baton);

  // DEPRECATED
  void DispatchInput(void *baton, const void *data, size_t data_len);

  void DispatchInput(const void *data, size_t data_len);

  void DispatchInputInterrupt();

  void DispatchInputEndOfFile();

  void PushInputReader(lldb::SBInputReader &reader);

  const char *GetInstanceName();

  static SBDebugger FindDebuggerWithID(int id);

  static lldb::SBError SetInternalVariable(const char *var_name,
                                           const char *value,
                                           const char *debugger_instance_name);

  static lldb::SBStringList
  GetInternalVariableValue(const char *var_name,
                           const char *debugger_instance_name);

  bool GetDescription(lldb::SBStream &description);

  uint32_t GetTerminalWidth() const;

  void SetTerminalWidth(uint32_t term_width);

  lldb::user_id_t GetID();

  const char *GetPrompt() const;

  void SetPrompt(const char *prompt);

  const char *GetReproducerPath() const;

  lldb::ScriptLanguage GetScriptLanguage() const;

  void SetScriptLanguage(lldb::ScriptLanguage script_lang);

  lldb::LanguageType GetREPLLanguage() const;

  void SetREPLLanguage(lldb::LanguageType repl_lang);

  bool GetCloseInputOnEOF() const;

  void SetCloseInputOnEOF(bool b);

  SBTypeCategory GetCategory(const char *category_name);

  SBTypeCategory GetCategory(lldb::LanguageType lang_type);

  SBTypeCategory CreateCategory(const char *category_name);

  bool DeleteCategory(const char *category_name);

  uint32_t GetNumCategories();

  SBTypeCategory GetCategoryAtIndex(uint32_t);

  SBTypeCategory GetDefaultCategory();

  SBTypeFormat GetFormatForType(SBTypeNameSpecifier);

  SBTypeSummary GetSummaryForType(SBTypeNameSpecifier);

  SBTypeFilter GetFilterForType(SBTypeNameSpecifier);

  SBTypeSynthetic GetSyntheticForType(SBTypeNameSpecifier);

  /// Run the command interpreter.
  ///
  /// \param[in] auto_handle_events
  ///     If true, automatically handle resulting events. This takes precedence
  ///     and overrides the corresponding option in
  ///     SBCommandInterpreterRunOptions.
  ///
  /// \param[in] spawn_thread
  ///     If true, start a new thread for IO handling. This takes precedence
  ///     and overrides the corresponding option in
  ///     SBCommandInterpreterRunOptions.
  void RunCommandInterpreter(bool auto_handle_events, bool spawn_thread);

  /// Run the command interpreter.
  ///
  /// \param[in] auto_handle_events
  ///     If true, automatically handle resulting events. This takes precedence
  ///     and overrides the corresponding option in
  ///     SBCommandInterpreterRunOptions.
  ///
  /// \param[in] spawn_thread
  ///     If true, start a new thread for IO handling. This takes precedence
  ///     and overrides the corresponding option in
  ///     SBCommandInterpreterRunOptions.
  ///
  /// \param[in] options
  ///     Parameter collection of type SBCommandInterpreterRunOptions.
  ///
  /// \param[out] num_errors
  ///     The number of errors.
  ///
  /// \param[out] quit_requested
  ///     Whether a quit was requested.
  ///
  /// \param[out] stopped_for_crash
  ///     Whether the interpreter stopped for a crash.
  void RunCommandInterpreter(bool auto_handle_events, bool spawn_thread,
                             SBCommandInterpreterRunOptions &options,
                             int &num_errors, bool &quit_requested,
                             bool &stopped_for_crash);

  SBCommandInterpreterRunResult
  RunCommandInterpreter(const SBCommandInterpreterRunOptions &options);

  SBError RunREPL(lldb::LanguageType language, const char *repl_options);

private:
  friend class SBCommandInterpreter;
  friend class SBInputReader;
  friend class SBListener;
  friend class SBProcess;
  friend class SBSourceManager;
  friend class SBTarget;

  lldb::SBTarget FindTargetWithLLDBProcess(const lldb::ProcessSP &processSP);

  void reset(const lldb::DebuggerSP &debugger_sp);

  lldb_private::Debugger *get() const;

  lldb_private::Debugger &ref() const;

  const lldb::DebuggerSP &get_sp() const;

  lldb::DebuggerSP m_opaque_sp;

}; // class SBDebugger

} // namespace lldb

#endif // LLDB_API_SBDEBUGGER_H
