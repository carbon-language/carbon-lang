//===-- llvm/Target/TargetLoweringObjectFile.h - Object Info ----*- C++ -*-===//
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

#ifndef LLVM_TARGET_TARGETLOWERINGOBJECTFILE_H
#define LLVM_TARGET_TARGETLOWERINGOBJECTFILE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/SectionKind.h"

namespace llvm {
  class MachineModuleInfo;
  class Mangler;
  class MCContext;
  class MCExpr;
  class MCSection;
  class MCSymbol;
  class MCSymbolRefExpr;
  class MCStreamer;
  class ConstantExpr;
  class GlobalValue;
  class TargetMachine;

class TargetLoweringObjectFile : public MCObjectFileInfo {
  MCContext *Ctx;
  const DataLayout *DL;

  TargetLoweringObjectFile(
    const TargetLoweringObjectFile&) LLVM_DELETED_FUNCTION;
  void operator=(const TargetLoweringObjectFile&) LLVM_DELETED_FUNCTION;

public:
  MCContext &getContext() const { return *Ctx; }

  TargetLoweringObjectFile() : MCObjectFileInfo(), Ctx(0), DL(0) {}

  virtual ~TargetLoweringObjectFile();

  /// Initialize - this method must be called before any actual lowering is
  /// done.  This specifies the current context for codegen, and gives the
  /// lowering implementations a chance to set up their default sections.
  virtual void Initialize(MCContext &ctx, const TargetMachine &TM);

  virtual void emitPersonalityValue(MCStreamer &Streamer,
                                    const TargetMachine &TM,
                                    const MCSymbol *Sym) const;

  /// getDepLibFromLinkerOpt - Extract the dependent library name from a linker
  /// option string. Returns StringRef() if the option does not specify a library.
  virtual StringRef getDepLibFromLinkerOpt(StringRef LinkerOption) const {
    return StringRef();
  }

  /// emitModuleFlags - Emit the module flags that the platform cares about.
  virtual void emitModuleFlags(MCStreamer &,
                               ArrayRef<Module::ModuleFlagEntry>,
                               Mangler *, const TargetMachine &) const {
  }

  /// shouldEmitUsedDirectiveFor - This hook allows targets to selectively
  /// decide not to emit the UsedDirective for some symbols in llvm.used.
  /// FIXME: REMOVE this (rdar://7071300)
  virtual bool shouldEmitUsedDirectiveFor(const GlobalValue *GV,
                                          Mangler *) const {
    return GV != 0;
  }

  /// getSectionForConstant - Given a constant with the SectionKind, return a
  /// section that it should be placed in.
  virtual const MCSection *getSectionForConstant(SectionKind Kind) const;

  /// getKindForGlobal - Classify the specified global variable into a set of
  /// target independent categories embodied in SectionKind.
  static SectionKind getKindForGlobal(const GlobalValue *GV,
                                      const TargetMachine &TM);

  /// SectionForGlobal - This method computes the appropriate section to emit
  /// the specified global variable or function definition.  This should not
  /// be passed external (or available externally) globals.
  const MCSection *SectionForGlobal(const GlobalValue *GV,
                                    SectionKind Kind, Mangler *Mang,
                                    const TargetMachine &TM) const;

  /// SectionForGlobal - This method computes the appropriate section to emit
  /// the specified global variable or function definition.  This should not
  /// be passed external (or available externally) globals.
  const MCSection *SectionForGlobal(const GlobalValue *GV,
                                    Mangler *Mang,
                                    const TargetMachine &TM) const {
    return SectionForGlobal(GV, getKindForGlobal(GV, TM), Mang, TM);
  }

  /// getExplicitSectionGlobal - Targets should implement this method to assign
  /// a section to globals with an explicit section specfied.  The
  /// implementation of this method can assume that GV->hasSection() is true.
  virtual const MCSection *
  getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                           Mangler *Mang, const TargetMachine &TM) const = 0;

  /// getSpecialCasedSectionGlobals - Allow the target to completely override
  /// section assignment of a global.
  virtual const MCSection *
  getSpecialCasedSectionGlobals(const GlobalValue *GV, Mangler *Mang,
                                SectionKind Kind) const {
    return 0;
  }

  /// getTTypeGlobalReference - Return an MCExpr to use for a reference
  /// to the specified global variable from exception handling information.
  ///
  virtual const MCExpr *
  getTTypeGlobalReference(const GlobalValue *GV, Mangler *Mang,
                          MachineModuleInfo *MMI, unsigned Encoding,
                          MCStreamer &Streamer) const;

  /// Return the MCSymbol for the specified global value.  This symbol is the
  /// main label that is the address of the global
  MCSymbol *getSymbol(Mangler &M, const GlobalValue *GV) const;

  /// Return the MCSymbol for a private symbol with global value name as its
  /// base, with the specified suffix.
  MCSymbol *getSymbolWithGlobalValueBase(Mangler &M, const GlobalValue *GV,
                                         StringRef Suffix) const;

  // getCFIPersonalitySymbol - The symbol that gets passed to .cfi_personality.
  virtual MCSymbol *
  getCFIPersonalitySymbol(const GlobalValue *GV, Mangler *Mang,
                          MachineModuleInfo *MMI) const;

  ///
  const MCExpr *
  getTTypeReference(const MCSymbolRefExpr *Sym, unsigned Encoding,
                    MCStreamer &Streamer) const;

  virtual const MCSection *
  getStaticCtorSection(unsigned Priority = 65535) const {
    (void)Priority;
    return StaticCtorSection;
  }
  virtual const MCSection *
  getStaticDtorSection(unsigned Priority = 65535) const {
    (void)Priority;
    return StaticDtorSection;
  }

  /// \brief Create a symbol reference to describe the given TLS variable when
  /// emitting the address in debug info.
  virtual const MCExpr *getDebugThreadLocalSymbol(const MCSymbol *Sym) const;

  virtual const MCExpr *
  getExecutableRelativeSymbol(const ConstantExpr *CE, Mangler *Mang) const {
    return 0;
  }

protected:
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};

} // end namespace llvm

#endif
