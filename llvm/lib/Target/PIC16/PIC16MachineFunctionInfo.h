//====- PIC16MachineFuctionInfo.h - PIC16 machine function info -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares PIC16-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16MACHINEFUNCTIONINFO_H
#define PIC16MACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// PIC16MachineFunctionInfo - This class is derived from MachineFunction
/// private PIC16 target-specific information for each MachineFunction.
class PIC16MachineFunctionInfo : public MachineFunctionInfo {
  // The frameindexes generated for spill/reload are stack based.
  // This maps maintain zero based indexes for these FIs.
  std::map<unsigned, unsigned> FiTmpOffsetMap;
  unsigned TmpSize;

  // These are the frames for return value and argument passing 
  // These FrameIndices will be expanded to foo.frame external symbol
  // and all others will be expanded to foo.tmp external symbol.
  unsigned ReservedFrameCount;

public:
  PIC16MachineFunctionInfo()
    : TmpSize(0), ReservedFrameCount(0) {}

  explicit PIC16MachineFunctionInfo(MachineFunction &MF)
    : TmpSize(0), ReservedFrameCount(0) {}

  std::map<unsigned, unsigned> &getFiTmpOffsetMap() { return FiTmpOffsetMap; }

  unsigned getTmpSize() const { return TmpSize; }
  void setTmpSize(unsigned Size) { TmpSize = Size; }

  unsigned getReservedFrameCount() const { return ReservedFrameCount; }
  void setReservedFrameCount(unsigned Count) { ReservedFrameCount = Count; }
};

} // End llvm namespace

#endif
