//===-- IntelPTPerThreadProcessTrace.h ------------------------ -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IntelPTPerThreadProcessTrace_H_
#define liblldb_IntelPTPerThreadProcessTrace_H_

#include "IntelPTSingleBufferTrace.h"
#include "IntelPTThreadTraceCollection.h"

namespace lldb_private {
namespace process_linux {

class IntelPTPerThreadProcessTrace;
using IntelPTPerThreadProcessTraceUP =
    std::unique_ptr<IntelPTPerThreadProcessTrace>;

/// Manages a "process trace" instance by tracing each thread individually.
class IntelPTPerThreadProcessTrace {
public:
  /// Start tracing the current process by tracing each of its tids
  /// individually.
  ///
  /// \param[in] request
  ///   Intel PT configuration parameters.
  ///
  /// \param[in] current_tids
  ///   List of tids currently alive. In the future, whenever a new thread is
  ///   spawned, they should be traced by calling the \a TraceStart(tid) method.
  ///
  /// \return
  ///   An \a IntelPTMultiCoreTrace instance if tracing was successful, or
  ///   an \a llvm::Error otherwise.
  static llvm::Expected<IntelPTPerThreadProcessTraceUP>
  Start(const TraceIntelPTStartRequest &request,
        llvm::ArrayRef<lldb::tid_t> current_tids);

  bool TracesThread(lldb::tid_t tid) const;

  IntelPTThreadTraceCollection &GetThreadTraces();

  /// \copydoc IntelPTThreadTraceCollection::TraceStart()
  llvm::Error TraceStart(lldb::tid_t tid);

  /// \copydoc IntelPTThreadTraceCollection::TraceStop()
  llvm::Error TraceStop(lldb::tid_t tid);

private:
  IntelPTPerThreadProcessTrace(const TraceIntelPTStartRequest &request)
      : m_tracing_params(request) {}

  IntelPTThreadTraceCollection m_thread_traces;
  /// Params used to trace threads when the user started "process tracing".
  TraceIntelPTStartRequest m_tracing_params;
};

} // namespace process_linux
} // namespace lldb_private

#endif // liblldb_IntelPTPerThreadProcessTrace_H_
