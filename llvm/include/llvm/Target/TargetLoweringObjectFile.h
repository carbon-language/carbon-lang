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

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/SectionKind.h"

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
  
class TargetLoweringObjectFile {
  MCContext *Ctx;
  
  TargetLoweringObjectFile(const TargetLoweringObjectFile&); // DO NOT IMPLEMENT
  void operator=(const TargetLoweringObjectFile&);           // DO NOT IMPLEMENT
protected:
  
  TargetLoweringObjectFile();
  
  /// TextSection - Section directive for standard text.
  ///
  const MCSection *TextSection;
  
  /// DataSection - Section directive for standard data.
  ///
  const MCSection *DataSection;
  
  /// BSSSection - Section that is default initialized to zero.
  const MCSection *BSSSection;
  
  /// ReadOnlySection - Section that is readonly and can contain arbitrary
  /// initialized data.  Targets are not required to have a readonly section.
  /// If they don't, various bits of code will fall back to using the data
  /// section for constants.
  const MCSection *ReadOnlySection;
  
  /// StaticCtorSection - This section contains the static constructor pointer
  /// list.
  const MCSection *StaticCtorSection;

  /// StaticDtorSection - This section contains the static destructor pointer
  /// list.
  const MCSection *StaticDtorSection;
  
  /// LSDASection - If exception handling is supported by the target, this is
  /// the section the Language Specific Data Area information is emitted to.
  const MCSection *LSDASection;
  
  /// EHFrameSection - If exception handling is supported by the target, this is
  /// the section the EH Frame is emitted to.
  const MCSection *EHFrameSection;
  
  // Dwarf sections for debug info.  If a target supports debug info, these must
  // be set.
  const MCSection *DwarfAbbrevSection;
  const MCSection *DwarfInfoSection;
  const MCSection *DwarfLineSection;
  const MCSection *DwarfFrameSection;
  const MCSection *DwarfPubNamesSection;
  const MCSection *DwarfPubTypesSection;
  const MCSection *DwarfDebugInlineSection;
  const MCSection *DwarfStrSection;
  const MCSection *DwarfLocSection;
  const MCSection *DwarfARangesSection;
  const MCSection *DwarfRangesSection;
  const MCSection *DwarfMacroInfoSection;
  
public:
  
  MCContext &getContext() const { return *Ctx; }
  

  virtual ~TargetLoweringObjectFile();
  
  /// Initialize - this method must be called before any actual lowering is
  /// done.  This specifies the current context for codegen, and gives the
  /// lowering implementations a chance to set up their default sections.
  virtual void Initialize(MCContext &ctx, const TargetMachine &TM) {
    Ctx = &ctx;
  }
  
  
  const MCSection *getTextSection() const { return TextSection; }
  const MCSection *getDataSection() const { return DataSection; }
  const MCSection *getBSSSection() const { return BSSSection; }
  const MCSection *getStaticCtorSection() const { return StaticCtorSection; }
  const MCSection *getStaticDtorSection() const { return StaticDtorSection; }
  const MCSection *getLSDASection() const { return LSDASection; }
  const MCSection *getEHFrameSection() const { return EHFrameSection; }
  const MCSection *getDwarfAbbrevSection() const { return DwarfAbbrevSection; }
  const MCSection *getDwarfInfoSection() const { return DwarfInfoSection; }
  const MCSection *getDwarfLineSection() const { return DwarfLineSection; }
  const MCSection *getDwarfFrameSection() const { return DwarfFrameSection; }
  const MCSection *getDwarfPubNamesSection() const{return DwarfPubNamesSection;}
  const MCSection *getDwarfPubTypesSection() const{return DwarfPubTypesSection;}
  const MCSection *getDwarfDebugInlineSection() const {
    return DwarfDebugInlineSection;
  }
  const MCSection *getDwarfStrSection() const { return DwarfStrSection; }
  const MCSection *getDwarfLocSection() const { return DwarfLocSection; }
  const MCSection *getDwarfARangesSection() const { return DwarfARangesSection;}
  const MCSection *getDwarfRangesSection() const { return DwarfRangesSection; }
  const MCSection *getDwarfMacroInfoSection() const {
    return DwarfMacroInfoSection;
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
  
  /// getSymbolForDwarfGlobalReference - Return an MCExpr to use for a reference
  /// to the specified global variable from exception handling information.
  ///
  virtual const MCExpr *
  getSymbolForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                              MachineModuleInfo *MMI, unsigned Encoding) const;

  virtual const MCExpr *
  getSymbolForDwarfReference(const MCSymbol *Sym, MachineModuleInfo *MMI,
                             unsigned Encoding) const;

  virtual unsigned getPersonalityEncoding() const;
  virtual unsigned getLSDAEncoding() const;
  virtual unsigned getFDEEncoding() const;
  virtual unsigned getTTypeEncoding() const;

protected:
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};

} // end namespace llvm

#endif
