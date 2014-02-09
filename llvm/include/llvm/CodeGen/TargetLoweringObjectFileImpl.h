//==-- llvm/CodeGen/TargetLoweringObjectFileImpl.h - Object Info -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements classes used to handle lowerings specific to common
// object file formats.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_TARGETLOWERINGOBJECTFILEIMPL_H
#define LLVM_CODEGEN_TARGETLOWERINGOBJECTFILEIMPL_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {
  class MachineModuleInfo;
  class Mangler;
  class MCAsmInfo;
  class MCExpr;
  class MCSection;
  class MCSectionMachO;
  class MCSymbol;
  class MCContext;
  class GlobalValue;
  class TargetMachine;


class TargetLoweringObjectFileELF : public TargetLoweringObjectFile {
  bool UseInitArray;

public:
  virtual ~TargetLoweringObjectFileELF() {}

  void emitPersonalityValue(MCStreamer &Streamer, const TargetMachine &TM,
                            const MCSymbol *Sym) const LLVM_OVERRIDE;

  /// Given a constant with the SectionKind, return a section that it should be
  /// placed in.
  const MCSection *getSectionForConstant(SectionKind Kind) const LLVM_OVERRIDE;

  const MCSection *getExplicitSectionGlobal(const GlobalValue *GV,
                                            SectionKind Kind, Mangler &Mang,
                                            const TargetMachine &TM) const
      LLVM_OVERRIDE;

  const MCSection *SelectSectionForGlobal(const GlobalValue *GV,
                                          SectionKind Kind, Mangler &Mang,
                                          const TargetMachine &TM) const
      LLVM_OVERRIDE;

  /// Return an MCExpr to use for a reference to the specified type info global
  /// variable from exception handling information.
  const MCExpr *getTTypeGlobalReference(const GlobalValue *GV,
                                        unsigned Encoding, Mangler &Mang,
                                        MachineModuleInfo *MMI,
                                        MCStreamer &Streamer) const
      LLVM_OVERRIDE;

  // The symbol that gets passed to .cfi_personality.
  MCSymbol *getCFIPersonalitySymbol(const GlobalValue *GV, Mangler &Mang,
                                    MachineModuleInfo *MMI) const LLVM_OVERRIDE;

  void InitializeELF(bool UseInitArray_);
  const MCSection *getStaticCtorSection(unsigned Priority = 65535) const
      LLVM_OVERRIDE;
  const MCSection *getStaticDtorSection(unsigned Priority = 65535) const
      LLVM_OVERRIDE;
};



class TargetLoweringObjectFileMachO : public TargetLoweringObjectFile {
public:
  virtual ~TargetLoweringObjectFileMachO() {}

  /// Extract the dependent library name from a linker option string. Returns
  /// StringRef() if the option does not specify a library.
  StringRef getDepLibFromLinkerOpt(StringRef LinkerOption) const LLVM_OVERRIDE;

  /// Emit the module flags that specify the garbage collection information.
  void emitModuleFlags(MCStreamer &Streamer,
                       ArrayRef<Module::ModuleFlagEntry> ModuleFlags,
                       Mangler &Mang, const TargetMachine &TM) const
      LLVM_OVERRIDE;

  const MCSection *SelectSectionForGlobal(const GlobalValue *GV,
                                          SectionKind Kind, Mangler &Mang,
                                          const TargetMachine &TM) const
      LLVM_OVERRIDE;

  const MCSection *getExplicitSectionGlobal(const GlobalValue *GV,
                                            SectionKind Kind, Mangler &Mang,
                                            const TargetMachine &TM) const
      LLVM_OVERRIDE;

  const MCSection *getSectionForConstant(SectionKind Kind) const LLVM_OVERRIDE;

  /// This hook allows targets to selectively decide not to emit the
  /// UsedDirective for some symbols in llvm.used.
  /// FIXME: REMOVE this (rdar://7071300)
  bool shouldEmitUsedDirectiveFor(const GlobalValue *GV, Mangler &Mang) const
      LLVM_OVERRIDE;

  /// The mach-o version of this method defaults to returning a stub reference.
  const MCExpr *getTTypeGlobalReference(const GlobalValue *GV,
                                        unsigned Encoding, Mangler &Mang,
                                        MachineModuleInfo *MMI,
                                        MCStreamer &Streamer) const
      LLVM_OVERRIDE;

  // The symbol that gets passed to .cfi_personality.
  MCSymbol *getCFIPersonalitySymbol(const GlobalValue *GV, Mangler &Mang,
                                    MachineModuleInfo *MMI) const LLVM_OVERRIDE;
};



class TargetLoweringObjectFileCOFF : public TargetLoweringObjectFile {
public:
  virtual ~TargetLoweringObjectFileCOFF() {}

  const MCSection *getExplicitSectionGlobal(const GlobalValue *GV,
                                            SectionKind Kind, Mangler &Mang,
                                            const TargetMachine &TM) const
      LLVM_OVERRIDE;

  const MCSection *SelectSectionForGlobal(const GlobalValue *GV,
                                          SectionKind Kind, Mangler &Mang,
                                          const TargetMachine &TM) const
      LLVM_OVERRIDE;

  /// Extract the dependent library name from a linker option string. Returns
  /// StringRef() if the option does not specify a library.
  StringRef getDepLibFromLinkerOpt(StringRef LinkerOption) const LLVM_OVERRIDE;

  /// Emit Obj-C garbage collection and linker options. Only linker option
  /// emission is implemented for COFF.
  void emitModuleFlags(MCStreamer &Streamer,
                       ArrayRef<Module::ModuleFlagEntry> ModuleFlags,
                       Mangler &Mang, const TargetMachine &TM) const
      LLVM_OVERRIDE;
};

} // end namespace llvm

#endif
