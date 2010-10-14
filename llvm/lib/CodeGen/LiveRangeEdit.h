//===---- LiveRangeEdit.h - Basic tools for split and spill -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The LiveRangeEdit class represents changes done to a virtual register when it
// is spilled or split.
//
// The parent register is never changed. Instead, a number of new virtual
// registers are created and added to the newRegs vector.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVERANGEEDIT_H
#define LLVM_CODEGEN_LIVERANGEEDIT_H

#include "llvm/CodeGen/LiveInterval.h"

namespace llvm {

class LiveIntervals;
class MachineRegisterInfo;
class VirtRegMap;

class LiveRangeEdit {
  LiveInterval &parent_;
  SmallVectorImpl<LiveInterval*> &newRegs_;
  const SmallVectorImpl<LiveInterval*> &uselessRegs_;

  /// firstNew_ - Index of the first register added to newRegs_.
  const unsigned firstNew_;

public:
  /// Create a LiveRangeEdit for breaking down parent into smaller pieces.
  /// @param parent The register being spilled or split.
  /// @param newRegs List to receive any new registers created. This needn't be
  ///                empty initially, any existing registers are ignored.
  /// @param uselessRegs List of registers that can't be used when
  ///        rematerializing values because they are about to be removed.
  LiveRangeEdit(LiveInterval &parent,
                SmallVectorImpl<LiveInterval*> &newRegs,
                const SmallVectorImpl<LiveInterval*> &uselessRegs)
    : parent_(parent), newRegs_(newRegs), uselessRegs_(uselessRegs),
      firstNew_(newRegs.size()) {}

  LiveInterval &getParent() const { return parent_; }
  unsigned getReg() const { return parent_.reg; }

  /// Iterator for accessing the new registers added by this edit.
  typedef SmallVectorImpl<LiveInterval*>::const_iterator iterator;
  iterator begin() const { return newRegs_.begin()+firstNew_; }
  iterator end() const { return newRegs_.end(); }

  /// create - Create a new register with the same class as parentReg_.
  LiveInterval &create(MachineRegisterInfo&, LiveIntervals&, VirtRegMap&);

  /// allUsesAvailableAt - Return true if all registers used by OrigMI at
  /// OrigIdx are also available with the same value at UseIdx.
  bool allUsesAvailableAt(const MachineInstr *OrigMI, SlotIndex OrigIdx,
                          SlotIndex UseIdx, LiveIntervals &lis);

};

}

#endif
