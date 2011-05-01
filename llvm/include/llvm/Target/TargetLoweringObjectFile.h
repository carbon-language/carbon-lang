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
  class MCContext;
  class MCExpr;
  class MCSection;
  class MCSectionMachO;
  class MCSymbol;
  class MCStreamer;
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
  
  // Extra TLS Variable Data section.  If the target needs to put additional
  // information for a TLS variable, it'll go here.
  const MCSection *TLSExtraDataSection;
  
  /// CommDirectiveSupportsAlignment - True if .comm supports alignment.  This
  /// is a hack for as long as we support 10.4 Tiger, whose assembler doesn't
  /// support alignment on comm.
  bool CommDirectiveSupportsAlignment;
  
  /// SupportsWeakEmptyEHFrame - True if target object file supports a
  /// weak_definition of constant 0 for an omitted EH frame.
  bool SupportsWeakOmittedEHFrame;
  
  /// IsFunctionEHSymbolGlobal - This flag is set to true if the ".eh" symbol
  /// for a function should be marked .globl.
  bool IsFunctionEHSymbolGlobal;
  
  /// IsFunctionEHFrameSymbolPrivate - This flag is set to true if the
  /// "EH_frame" symbol for EH information should be an assembler temporary (aka
  /// private linkage, aka an L or .L label) or false if it should be a normal
  /// non-.globl label.  This defaults to true.
  bool IsFunctionEHFrameSymbolPrivate;
public:
  
  MCContext &getContext() const { return *Ctx; }
  
  virtual ~TargetLoweringObjectFile();
  
  /// Initialize - this method must be called before any actual lowering is
  /// done.  This specifies the current context for codegen, and gives the
  /// lowering implementations a chance to set up their default sections.
  virtual void Initialize(MCContext &ctx, const TargetMachine &TM) {
    Ctx = &ctx;
  }
  
  bool isFunctionEHSymbolGlobal() const {
    return IsFunctionEHSymbolGlobal;
  }
  bool isFunctionEHFrameSymbolPrivate() const {
    return IsFunctionEHFrameSymbolPrivate;
  }
  bool getSupportsWeakOmittedEHFrame() const {
    return SupportsWeakOmittedEHFrame;
  }
  
  bool getCommDirectiveSupportsAlignment() const {
    return CommDirectiveSupportsAlignment;
  }

  const MCSection *getTextSection() const { return TextSection; }
  const MCSection *getDataSection() const { return DataSection; }
  const MCSection *getBSSSection() const { return BSSSection; }
  const MCSection *getStaticCtorSection() const { return StaticCtorSection; }
  const MCSection *getStaticDtorSection() const { return StaticDtorSection; }
  const MCSection *getLSDASection() const { return LSDASection; }
  virtual const MCSection *getEHFrameSection() const = 0;
  virtual void emitPersonalityValue(MCStreamer &Streamer,
                                    const TargetMachine &TM,
                                    const MCSymbol *Sym) const;
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
  const MCSection *getTLSExtraDataSection() const {
    return TLSExtraDataSection;
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
  
  /// getExprForDwarfGlobalReference - Return an MCExpr to use for a reference
  /// to the specified global variable from exception handling information.
  ///
  virtual const MCExpr *
  getExprForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                                 MachineModuleInfo *MMI, unsigned Encoding,
                                 MCStreamer &Streamer) const;

  // getCFIPersonalitySymbol - The symbol that gets passed to .cfi_personality.
  virtual MCSymbol *
  getCFIPersonalitySymbol(const GlobalValue *GV, Mangler *Mang,
                          MachineModuleInfo *MMI) const;

  /// 
  const MCExpr *
  getExprForDwarfReference(const MCSymbol *Sym, unsigned Encoding,
                           MCStreamer &Streamer) const;
  
  virtual unsigned getPersonalityEncoding() const;
  virtual unsigned getLSDAEncoding() const;
  virtual unsigned getFDEEncoding(bool CFI) const;
  virtual unsigned getTTypeEncoding() const;

protected:
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};

} // end namespace llvm

#endif
