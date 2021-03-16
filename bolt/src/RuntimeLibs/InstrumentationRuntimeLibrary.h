//===--- InstrumentationRuntimeLibrary.h - The Instrument Runtime Library -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_INSTRUMENTATION_RUNTIME_LIBRARY_H
#define LLVM_TOOLS_LLVM_BOLT_INSTRUMENTATION_RUNTIME_LIBRARY_H

#include "Passes/InstrumentationSummary.h"
#include "RuntimeLibs/RuntimeLibrary.h"

namespace llvm {
namespace bolt {

class InstrumentationRuntimeLibrary : public RuntimeLibrary {
public:
  InstrumentationRuntimeLibrary(std::unique_ptr<InstrumentationSummary> Summary)
      : Summary(std::move(Summary)) {}

  void addRuntimeLibSections(std::vector<std::string> &SecNames) const final {
    SecNames.push_back(".bolt.instr.counters");
  }

  void adjustCommandLineOptions(const BinaryContext &BC) const final;

  void emitBinary(BinaryContext &BC, MCStreamer &Streamer) final;

  void link(BinaryContext &BC, StringRef ToolPath, RuntimeDyld &RTDyld,
            std::function<void(RuntimeDyld &)> OnLoad) final;

private:
  std::string buildTables(BinaryContext &BC);

  /// Create a non-allocatable ELF section with read-only tables necessary for
  /// writing the instrumented data profile during program finish. The runtime
  /// library needs to open the program executable file and read this data from
  /// disk, this is not loaded by the system.
  void emitTablesAsELFNote(BinaryContext &BC);

  std::unique_ptr<InstrumentationSummary> Summary;
};

} // namespace bolt
} // namespace llvm

#endif
