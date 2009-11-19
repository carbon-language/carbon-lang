//===-- llvm/CodeGen/Spiller.h - Spiller -*- C++ -*------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SPILLER_H
#define LLVM_CODEGEN_SPILLER_H

#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace llvm {

  class LiveInterval;
  class LiveIntervals;
  class LiveStacks;
  class MachineFunction;
  class MachineInstr;
  class MachineLoopInfo;
  class VirtRegMap;
  class VNInfo;

  /// Spiller interface.
  ///
  /// Implementations are utility classes which insert spill or remat code on
  /// demand.
  class Spiller {
  public:
    virtual ~Spiller() = 0;

    /// Spill the given live range. The method used will depend on the Spiller
    /// implementation selected.
    virtual std::vector<LiveInterval*> spill(LiveInterval *li,
                                   SmallVectorImpl<LiveInterval*> &spillIs) = 0;

  };

  /// Create and return a spiller object, as specified on the command line.
  Spiller* createSpiller(MachineFunction *mf, LiveIntervals *li,
                         LiveStacks *ls, const MachineLoopInfo *loopInfo,
                         VirtRegMap *vrm);
}

#endif
