//===- ARMConstantPoolValue.h - ARM constantpool value ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ARM_CONSTANTPOOLVALUE_H
#define LLVM_TARGET_ARM_CONSTANTPOOLVALUE_H

#include "llvm/CodeGen/MachineConstantPool.h"

namespace llvm {

/// ARMConstantPoolValue - ARM specific constantpool value. This is used to
/// represent PC relative displacement between the address of the load
/// instruction and the global value being loaded, i.e. (&GV-(LPIC+8)).
class ARMConstantPoolValue : public MachineConstantPoolValue {
  GlobalValue *GV;         // GlobalValue being loaded.
  unsigned LabelId;        // Label id of the load.
  bool isNonLazyPtr;       // True if loading a Mac OS X non_lazy_ptr stub.
  unsigned char PCAdjust;  // Extra adjustment if constantpool is pc relative.
                           // 8 for ARM, 4 for Thumb.

public:
  ARMConstantPoolValue(GlobalValue *gv, unsigned id, bool isNonLazy = false,
                       unsigned char PCAdj = 0);

  GlobalValue *getGV() const { return GV; }
  unsigned getLabelId() const { return LabelId; }
  bool isNonLazyPointer() const { return isNonLazyPtr; }
  unsigned char getPCAdjustment() const { return PCAdjust; }

  virtual int getExistingMachineCPValue(MachineConstantPool *CP,
                                        unsigned Alignment);

  virtual void AddSelectionDAGCSEId(FoldingSetNodeID &ID);

  virtual void print(std::ostream &O) const;
};
  
}

#endif
