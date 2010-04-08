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
protected:
  /// TLSDataSection - Section directive for Thread Local data.
  ///
  const MCSection *TLSDataSection;        // Defaults to ".tdata".

  /// TLSBSSSection - Section directive for Thread Local uninitialized data.
  /// Null if this target doesn't support a BSS section.
  ///
  const MCSection *TLSBSSSection;         // Defaults to ".tbss".

  const MCSection *DataRelSection;
  const MCSection *DataRelLocalSection;
  const MCSection *DataRelROSection;
  const MCSection *DataRelROLocalSection;

  const MCSection *MergeableConst4Section;
  const MCSection *MergeableConst8Section;
  const MCSection *MergeableConst16Section;
public:
  TargetLoweringObjectFileELF() {}
  ~TargetLoweringObjectFileELF() {}

  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);

  const MCSection *getDataRelSection() const { return DataRelSection; }

  /// getSectionForConstant - Given a constant with the SectionKind, return a
  /// section that it should be placed in.
  virtual const MCSection *getSectionForConstant(SectionKind Kind) const;


  virtual const MCSection *
  getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                           Mangler *Mang, const TargetMachine &TM) const;

  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;

  /// getExprForDwarfGlobalReference - Return an MCExpr to use for a reference
  /// to the specified global variable from exception handling information.
  ///
  virtual const MCExpr *
  getExprForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                                 MachineModuleInfo *MMI, unsigned Encoding,
                                 MCStreamer &Streamer) const;
};



class TargetLoweringObjectFileMachO : public TargetLoweringObjectFile {
  const MCSection *CStringSection;
  const MCSection *UStringSection;
  const MCSection *TextCoalSection;
  const MCSection *ConstTextCoalSection;
  const MCSection *ConstDataCoalSection;
  const MCSection *ConstDataSection;
  const MCSection *DataCoalSection;
  const MCSection *DataCommonSection;
  const MCSection *DataBSSSection;
  const MCSection *FourByteConstantSection;
  const MCSection *EightByteConstantSection;
  const MCSection *SixteenByteConstantSection;

  const MCSection *LazySymbolPointerSection;
  const MCSection *NonLazySymbolPointerSection;
public:
  TargetLoweringObjectFileMachO() {}
  ~TargetLoweringObjectFileMachO() {}

  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);

  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;

  virtual const MCSection *
  getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                           Mangler *Mang, const TargetMachine &TM) const;

  virtual const MCSection *getSectionForConstant(SectionKind Kind) const;

  /// shouldEmitUsedDirectiveFor - This hook allows targets to selectively
  /// decide not to emit the UsedDirective for some symbols in llvm.used.
  /// FIXME: REMOVE this (rdar://7071300)
  virtual bool shouldEmitUsedDirectiveFor(const GlobalValue *GV,
                                          Mangler *) const;

  /// getTextCoalSection - Return the "__TEXT,__textcoal_nt" section we put weak
  /// text symbols into.
  const MCSection *getTextCoalSection() const {
    return TextCoalSection;
  }

  /// getConstTextCoalSection - Return the "__TEXT,__const_coal" section
  /// we put weak read-only symbols into.
  const MCSection *getConstTextCoalSection() const {
    return ConstTextCoalSection;
  }

  /// getLazySymbolPointerSection - Return the section corresponding to
  /// the .lazy_symbol_pointer directive.
  const MCSection *getLazySymbolPointerSection() const {
    return LazySymbolPointerSection;
  }

  /// getNonLazySymbolPointerSection - Return the section corresponding to
  /// the .non_lazy_symbol_pointer directive.
  const MCSection *getNonLazySymbolPointerSection() const {
    return NonLazySymbolPointerSection;
  }

  /// getExprForDwarfGlobalReference - The mach-o version of this method
  /// defaults to returning a stub reference.
  virtual const MCExpr *
  getExprForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                                 MachineModuleInfo *MMI, unsigned Encoding,
                                 MCStreamer &Streamer) const;

  virtual unsigned getPersonalityEncoding() const;
  virtual unsigned getLSDAEncoding() const;
  virtual unsigned getFDEEncoding() const;
  virtual unsigned getTTypeEncoding() const;
};



class TargetLoweringObjectFileCOFF : public TargetLoweringObjectFile {
  mutable void *UniquingMap;
public:
  TargetLoweringObjectFileCOFF() : UniquingMap(0) {}
  ~TargetLoweringObjectFileCOFF();

  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);

  virtual const MCSection *
  getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                           Mangler *Mang, const TargetMachine &TM) const;

  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;

  /// getCOFFSection - Return the MCSection for the specified COFF section.
  /// FIXME: Switch this to a semantic view eventually.
  const MCSection *getCOFFSection(StringRef Name, bool isDirective,
                                  SectionKind K) const;
};

} // end namespace llvm

#endif
