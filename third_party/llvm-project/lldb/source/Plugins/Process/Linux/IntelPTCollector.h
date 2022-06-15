//===-- IntelPTCollector.h ------------------------------------ -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IntelPTCollector_H_
#define liblldb_IntelPTCollector_H_

#include "Perf.h"

#include "IntelPTMultiCoreTrace.h"
#include "IntelPTPerThreadProcessTrace.h"
#include "IntelPTSingleBufferTrace.h"

#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"
#include "lldb/lldb-types.h"

#include <linux/perf_event.h>
#include <sys/mman.h>
#include <unistd.h>

namespace lldb_private {

namespace process_linux {

/// Main class that manages intel-pt process and thread tracing.
class IntelPTCollector {
public:
  /// \param[in] process
  ///     Process to be traced.
  IntelPTCollector(NativeProcessProtocol &process);

  static bool IsSupported();

  /// To be invoked whenever the state of the target process has changed.
  void OnProcessStateChanged(lldb::StateType state);

  /// If "process tracing" is enabled, then trace the given thread.
  llvm::Error OnThreadCreated(lldb::tid_t tid);

  /// Stops tracing a tracing upon a destroy event.
  llvm::Error OnThreadDestroyed(lldb::tid_t tid);

  /// Implementation of the jLLDBTraceStop packet
  llvm::Error TraceStop(const TraceStopRequest &request);

  /// Implementation of the jLLDBTraceStart packet
  llvm::Error TraceStart(const TraceIntelPTStartRequest &request);

  /// Implementation of the jLLDBTraceGetState packet
  llvm::Expected<llvm::json::Value> GetState();

  /// Implementation of the jLLDBTraceGetBinaryData packet
  llvm::Expected<std::vector<uint8_t>>
  GetBinaryData(const TraceGetBinaryDataRequest &request);

  /// Dispose of all traces
  void Clear();

private:
  llvm::Error TraceStop(lldb::tid_t tid);

  /// Start tracing a specific thread.
  llvm::Error TraceStart(lldb::tid_t tid,
                         const TraceIntelPTStartRequest &request);

  llvm::Expected<IntelPTSingleBufferTrace &> GetTracedThread(lldb::tid_t tid);

  bool IsProcessTracingEnabled() const;

  void ClearProcessTracing();

  NativeProcessProtocol &m_process;
  /// Threads traced due to "thread tracing"
  IntelPTThreadTraceCollection m_thread_traces;

  /// Only one of the following "process tracing" handlers can be active at a
  /// given time.
  ///
  /// \{
  /// Threads traced due to per-thread "process tracing".  This might be \b
  /// nullptr.
  IntelPTPerThreadProcessTraceUP m_per_thread_process_trace_up;
  /// Cores traced due to per-core "process tracing".  This might be \b nullptr.
  IntelPTMultiCoreTraceUP m_per_core_process_trace_up;
  /// \}

  /// TSC to wall time conversion.
  TraceTscConversionUP m_tsc_conversion;
};

} // namespace process_linux
} // namespace lldb_private

#endif // liblldb_IntelPTCollector_H_
