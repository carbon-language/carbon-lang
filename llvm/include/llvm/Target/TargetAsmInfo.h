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
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class MCSection;
  class MCContext;
  class MachineFunction;
  class TargetMachine;
  class TargetLoweringObjectFile;

class TargetAsmInfo {
  unsigned PointerSize;
  bool IsLittleEndian;
  TargetFrameLowering::StackDirection StackDir;
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

  TargetFrameLowering::StackDirection getStackGrowthDirection() const {
    return StackDir;
  }

  const MCSection *getDwarfLineSection() const {
    return TLOF->getDwarfLineSection();
  }

  const MCSection *getEHFrameSection() const {
    return TLOF->getEHFrameSection();
  }

  const MCSection *getDwarfFrameSection() const {
    return TLOF->getDwarfFrameSection();
  }

  const MCSection *getWin64EHFuncTableSection() const {
    return TLOF->getWin64EHFuncTableSection();
  }

  const MCSection *getWin64EHTableSection() const {
    return TLOF->getWin64EHTableSection();
  }

  unsigned getFDEEncoding(bool CFI) const {
    return TLOF->getFDEEncoding(CFI);
  }

  bool isFunctionEHFrameSymbolPrivate() const {
    return TLOF->isFunctionEHFrameSymbolPrivate();
  }

  const unsigned *getCalleeSavedRegs(MachineFunction *MF = 0) const {
    return TRI->getCalleeSavedRegs(MF);
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

  int getSEHRegNum(unsigned RegNum) const {
    return TRI->getSEHRegNum(RegNum);
  }
};

}
#endif
