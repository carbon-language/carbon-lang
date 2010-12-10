//===-- llvm/Target/TargetAsmInfo.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to provide the information necessary for producing assembly files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETASMINFO_H
#define LLVM_TARGET_TARGETASMINFO_H

#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class MCSection;
  class MCContext;
  class TargetMachine;
  class TargetLoweringObjectFile;

class TargetAsmInfo {
  unsigned PointerSize;
  bool IsLittleEndian;
  TargetFrameInfo::StackDirection StackDir;
  const TargetRegisterInfo *TRI;
  std::vector<MachineMove> InitialFrameState;
  const TargetLoweringObjectFile *TLOF;

public:
  explicit TargetAsmInfo(const TargetMachine &TM);

  /// getPointerSize - Get the pointer size in bytes.
  unsigned getPointerSize() const {
    return PointerSize;
  }

  /// islittleendian - True if the target is little endian.
  bool isLittleEndian() const {
    return IsLittleEndian;
  }

  TargetFrameInfo::StackDirection getStackGrowthDirection() const {
    return StackDir;
  }

  const MCSection *getDwarfLineSection() const {
    return TLOF->getDwarfLineSection();
  }

  const MCSection *getEHFrameSection() const {
    return TLOF->getEHFrameSection();
  }

  unsigned getDwarfRARegNum(bool isEH) const {
    return TRI->getDwarfRegNum(TRI->getRARegister(), isEH);
  }

  const std::vector<MachineMove> &getInitialFrameState() const {
    return InitialFrameState;
  }

  int getDwarfRegNum(unsigned RegNum, bool isEH) const {
    return TRI->getDwarfRegNum(RegNum, isEH);
  }
};

}
#endif
