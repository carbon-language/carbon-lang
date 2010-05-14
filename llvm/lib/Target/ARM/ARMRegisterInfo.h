//===- ARMRegisterInfo.h - ARM Register Information Impl --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMREGISTERINFO_H
#define ARMREGISTERINFO_H

#include "ARM.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "ARMBaseRegisterInfo.h"

namespace llvm {
  class ARMSubtarget;
  class ARMBaseInstrInfo;
  class Type;

namespace ARM {
  /// SubregIndex - The index of various subregister classes. Note that 
  /// these indices must be kept in sync with the class indices in the 
  /// ARMRegisterInfo.td file.
  enum SubregIndex {
    SSUBREG_0 = 1,  SSUBREG_1 = 2,  SSUBREG_2 = 3,  SSUBREG_3 = 4,
    DSUBREG_0 = 5,  DSUBREG_1 = 6,  DSUBREG_2 = 7,  DSUBREG_3 = 8,
    DSUBREG_4 = 9,  DSUBREG_5 = 10, DSUBREG_6 = 11, DSUBREG_7 = 12,
    QSUBREG_0 = 13, QSUBREG_1 = 14, QSUBREG_2 = 15, QSUBREG_3 = 16,
    QQSUBREG_0= 17, QQSUBREG_1= 18
  };
}

struct ARMRegisterInfo : public ARMBaseRegisterInfo {
public:
  ARMRegisterInfo(const ARMBaseInstrInfo &tii, const ARMSubtarget &STI);
};

} // end namespace llvm

#endif
