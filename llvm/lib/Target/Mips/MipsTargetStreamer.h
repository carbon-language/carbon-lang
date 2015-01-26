//===-- MipsTargetStreamer.h - Mips Target Streamer ------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MIPSTARGETSTREAMER_H
#define LLVM_LIB_TARGET_MIPS_MIPSTARGETSTREAMER_H

#include "MCTargetDesc/MipsABIFlagsSection.h"
#include "MCTargetDesc/MipsABIInfo.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {

struct MipsABIFlagsSection;

class MipsTargetStreamer : public MCTargetStreamer {
public:
  MipsTargetStreamer(MCStreamer &S);
  virtual void emitDirectiveSetMicroMips();
  virtual void emitDirectiveSetNoMicroMips();
  virtual void emitDirectiveSetMips16();
  virtual void emitDirectiveSetNoMips16();

  virtual void emitDirectiveSetReorder();
  virtual void emitDirectiveSetNoReorder();
  virtual void emitDirectiveSetMacro();
  virtual void emitDirectiveSetNoMacro();
  virtual void emitDirectiveSetMsa();
  virtual void emitDirectiveSetNoMsa();
  virtual void emitDirectiveSetAt();
  virtual void emitDirectiveSetNoAt();
  virtual void emitDirectiveEnd(StringRef Name);

  virtual void emitDirectiveEnt(const MCSymbol &Symbol);
  virtual void emitDirectiveAbiCalls();
  virtual void emitDirectiveNaN2008();
  virtual void emitDirectiveNaNLegacy();
  virtual void emitDirectiveOptionPic0();
  virtual void emitDirectiveOptionPic2();
  virtual void emitFrame(unsigned StackReg, unsigned StackSize,
                         unsigned ReturnReg);
  virtual void emitMask(unsigned CPUBitmask, int CPUTopSavedRegOff);
  virtual void emitFMask(unsigned FPUBitmask, int FPUTopSavedRegOff);

  virtual void emitDirectiveSetArch(StringRef Arch);
  virtual void emitDirectiveSetMips0();
  virtual void emitDirectiveSetMips1();
  virtual void emitDirectiveSetMips2();
  virtual void emitDirectiveSetMips3();
  virtual void emitDirectiveSetMips4();
  virtual void emitDirectiveSetMips5();
  virtual void emitDirectiveSetMips32();
  virtual void emitDirectiveSetMips32R2();
  virtual void emitDirectiveSetMips32R6();
  virtual void emitDirectiveSetMips64();
  virtual void emitDirectiveSetMips64R2();
  virtual void emitDirectiveSetMips64R6();
  virtual void emitDirectiveSetDsp();
  virtual void emitDirectiveSetNoDsp();
  virtual void emitDirectiveSetPop();
  virtual void emitDirectiveSetPush();

  // PIC support
  virtual void emitDirectiveCpLoad(unsigned RegNo);
  virtual void emitDirectiveCpsetup(unsigned RegNo, int RegOrOffset,
                                    const MCSymbol &Sym, bool IsReg);

  /// Emit a '.module fp=value' directive using the given values.
  /// Updates the .MIPS.abiflags section
  virtual void emitDirectiveModuleFP(MipsABIFlagsSection::FpABIKind Value,
                                     bool Is32BitABI) {
    ABIFlagsSection.setFpABI(Value, Is32BitABI);
  }

  /// Emit a '.module fp=value' directive using the current values of the
  /// .MIPS.abiflags section.
  void emitDirectiveModuleFP() {
    emitDirectiveModuleFP(ABIFlagsSection.getFpABI(),
                          ABIFlagsSection.Is32BitABI);
  }

  virtual void emitDirectiveModuleOddSPReg(bool Enabled, bool IsO32ABI);
  virtual void emitDirectiveSetFp(MipsABIFlagsSection::FpABIKind Value){};
  virtual void emitMipsAbiFlags(){};
  void forbidModuleDirective() { ModuleDirectiveAllowed = false; }
  bool isModuleDirectiveAllowed() { return ModuleDirectiveAllowed; }

  // This method enables template classes to set internal abi flags
  // structure values.
  template <class PredicateLibrary>
  void updateABIInfo(const PredicateLibrary &P) {
    ABI = &P.getABI();
    ABIFlagsSection.setAllFromPredicates(P);
  }

  MipsABIFlagsSection &getABIFlagsSection() { return ABIFlagsSection; }
  const MipsABIInfo &getABI() const {
    assert(ABI && "ABI hasn't been set!");
    return *ABI;
  }

protected:
  const MipsABIInfo *ABI;
  MipsABIFlagsSection ABIFlagsSection;

  bool GPRInfoSet;
  unsigned GPRBitMask;
  int GPROffset;

  bool FPRInfoSet;
  unsigned FPRBitMask;
  int FPROffset;

  bool FrameInfoSet;
  int FrameOffset;
  unsigned FrameReg;
  unsigned ReturnReg;

private:
  bool ModuleDirectiveAllowed;
};

// This part is for ascii assembly output
class MipsTargetAsmStreamer : public MipsTargetStreamer {
  formatted_raw_ostream &OS;

public:
  MipsTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);
  void emitDirectiveSetMicroMips() override;
  void emitDirectiveSetNoMicroMips() override;
  void emitDirectiveSetMips16() override;
  void emitDirectiveSetNoMips16() override;

  void emitDirectiveSetReorder() override;
  void emitDirectiveSetNoReorder() override;
  void emitDirectiveSetMacro() override;
  void emitDirectiveSetNoMacro() override;
  void emitDirectiveSetMsa() override;
  void emitDirectiveSetNoMsa() override;
  void emitDirectiveSetAt() override;
  void emitDirectiveSetNoAt() override;
  void emitDirectiveEnd(StringRef Name) override;

  void emitDirectiveEnt(const MCSymbol &Symbol) override;
  void emitDirectiveAbiCalls() override;
  void emitDirectiveNaN2008() override;
  void emitDirectiveNaNLegacy() override;
  void emitDirectiveOptionPic0() override;
  void emitDirectiveOptionPic2() override;
  void emitFrame(unsigned StackReg, unsigned StackSize,
                 unsigned ReturnReg) override;
  void emitMask(unsigned CPUBitmask, int CPUTopSavedRegOff) override;
  void emitFMask(unsigned FPUBitmask, int FPUTopSavedRegOff) override;

  void emitDirectiveSetArch(StringRef Arch) override;
  void emitDirectiveSetMips0() override;
  void emitDirectiveSetMips1() override;
  void emitDirectiveSetMips2() override;
  void emitDirectiveSetMips3() override;
  void emitDirectiveSetMips4() override;
  void emitDirectiveSetMips5() override;
  void emitDirectiveSetMips32() override;
  void emitDirectiveSetMips32R2() override;
  void emitDirectiveSetMips32R6() override;
  void emitDirectiveSetMips64() override;
  void emitDirectiveSetMips64R2() override;
  void emitDirectiveSetMips64R6() override;
  void emitDirectiveSetDsp() override;
  void emitDirectiveSetNoDsp() override;
  void emitDirectiveSetPop() override;
  void emitDirectiveSetPush() override;

  // PIC support
  void emitDirectiveCpLoad(unsigned RegNo) override;
  void emitDirectiveCpsetup(unsigned RegNo, int RegOrOffset,
                            const MCSymbol &Sym, bool IsReg) override;

  // ABI Flags
  void emitDirectiveModuleFP(MipsABIFlagsSection::FpABIKind Value,
                             bool Is32BitABI) override;
  void emitDirectiveModuleOddSPReg(bool Enabled, bool IsO32ABI) override;
  void emitDirectiveSetFp(MipsABIFlagsSection::FpABIKind Value) override;
  void emitMipsAbiFlags() override;
};

// This part is for ELF object output
class MipsTargetELFStreamer : public MipsTargetStreamer {
  bool MicroMipsEnabled;
  const MCSubtargetInfo &STI;
  bool Pic;

public:
  bool isMicroMipsEnabled() const { return MicroMipsEnabled; }
  MCELFStreamer &getStreamer();
  MipsTargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);

  void emitLabel(MCSymbol *Symbol) override;
  void emitAssignment(MCSymbol *Symbol, const MCExpr *Value) override;
  void finish() override;

  void emitDirectiveSetMicroMips() override;
  void emitDirectiveSetNoMicroMips() override;
  void emitDirectiveSetMips16() override;

  void emitDirectiveSetNoReorder() override;
  void emitDirectiveEnd(StringRef Name) override;

  void emitDirectiveEnt(const MCSymbol &Symbol) override;
  void emitDirectiveAbiCalls() override;
  void emitDirectiveNaN2008() override;
  void emitDirectiveNaNLegacy() override;
  void emitDirectiveOptionPic0() override;
  void emitDirectiveOptionPic2() override;
  void emitFrame(unsigned StackReg, unsigned StackSize,
                 unsigned ReturnReg) override;
  void emitMask(unsigned CPUBitmask, int CPUTopSavedRegOff) override;
  void emitFMask(unsigned FPUBitmask, int FPUTopSavedRegOff) override;

  // PIC support
  void emitDirectiveCpLoad(unsigned RegNo) override;
  void emitDirectiveCpsetup(unsigned RegNo, int RegOrOffset,
                            const MCSymbol &Sym, bool IsReg) override;

  // ABI Flags
  void emitDirectiveModuleOddSPReg(bool Enabled, bool IsO32ABI) override;
  void emitMipsAbiFlags() override;
};
}
#endif
