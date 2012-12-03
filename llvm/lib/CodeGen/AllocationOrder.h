//===-- llvm/CodeGen/AllocationOrder.h - Allocation Order -*- C++ -*-------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an allocation order for virtual registers.
//
// The preferred allocation order for a virtual register depends on allocation
// hints and target hooks. The AllocationOrder class encapsulates all of that.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ALLOCATIONORDER_H
#define LLVM_CODEGEN_ALLOCATIONORDER_H

#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {

class RegisterClassInfo;
class VirtRegMap;

class AllocationOrder {
  SmallVector<MCPhysReg, 16> Hints;
  ArrayRef<MCPhysReg> Order;
  unsigned Pos;

public:
  /// Create a new AllocationOrder for VirtReg.
  /// @param VirtReg      Virtual register to allocate for.
  /// @param VRM          Virtual register map for function.
  /// @param RegClassInfo Information about reserved and allocatable registers.
  AllocationOrder(unsigned VirtReg,
                  const VirtRegMap &VRM,
                  const RegisterClassInfo &RegClassInfo);

  /// Return the next physical register in the allocation order, or 0.
  /// It is safe to call next() again after it returned 0, it will keep
  /// returning 0 until rewind() is called.
  unsigned next();

  /// Start over from the beginning.
  void rewind() { Pos = 0; }

  /// Return true if the last register returned from next() was a preferred register.
  bool isHint() const { return Pos <= Hints.size(); }

  /// Return true if PhysReg is a preferred register.
  bool isHint(unsigned PhysReg) const;
};

} // end namespace llvm

#endif
