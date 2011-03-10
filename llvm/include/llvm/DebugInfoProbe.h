//===-- DebugInfoProbe.h - DebugInfo Probe ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a probe, DebugInfoProbe, that can be used by pass
// manager to analyze how optimizer is treating debugging information.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_DEBUGINFOPROBE_H
#define LLVM_TRANSFORMS_UTILS_DEBUGINFOPROBE_H

#include "llvm/ADT/StringMap.h"

namespace llvm {
  class Function;
  class Pass;
  class DebugInfoProbeImpl;

  /// DebugInfoProbe - This class provides a interface to monitor
  /// how an optimization pass is preserving debugging information.
  class DebugInfoProbe {
    public:
    DebugInfoProbe();
    ~DebugInfoProbe();

    /// initialize - Collect information before running an optimization pass.
    void initialize(StringRef PName, Function &F);

    /// finalize - Collect information after running an optimization pass. This
    /// must be used after initialization.
    void finalize(Function &F);

    /// report - Report findings. This should be invoked after finalize.
    void report();

    private:
    DebugInfoProbeImpl *pImpl;
  };

  /// DebugInfoProbeInfo - This class provides an interface that a pass manager
  /// can use to manage debug info probes.
  class DebugInfoProbeInfo {
    StringMap<DebugInfoProbe *> Probes;
  public:
    DebugInfoProbeInfo() {}

    /// ~DebugInfoProbeInfo - Report data collected by all probes before deleting
    /// them.
    ~DebugInfoProbeInfo();

    /// initialize - Collect information before running an optimization pass.
    void initialize(Pass *P, Function &F);

    /// finalize - Collect information after running an optimization pass. This
    /// must be used after initialization.
    void finalize(Pass *P, Function &F);
  };

} // End llvm namespace

#endif
