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

#include "llvm/MC/SectionKind.h"

namespace llvm {
  class MachineModuleInfo;
  class Mangler;
  class MCAsmInfo;
  class MCExpr;
  class MCSection;
  class MCSectionMachO;
  class MCContext;
  class GlobalValue;
  class StringRef;
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
  
  /// getSymbolForDwarfGlobalReference - Return an MCExpr to use for a
  /// pc-relative reference to the specified global variable from exception
  /// handling information.  In addition to the symbol, this returns
  /// by-reference:
  ///
  /// IsIndirect - True if the returned symbol is actually a stub that contains
  ///    the address of the symbol, false if the symbol is the global itself.
  ///
  /// IsPCRel - True if the symbol reference is already pc-relative, false if
  ///    the caller needs to subtract off the address of the reference from the
  ///    symbol.
  ///
  virtual const MCExpr *
  getSymbolForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                                   MachineModuleInfo *MMI,
                                   bool &IsIndirect, bool &IsPCRel) const;
  
protected:
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};
  
  
  

class TargetLoweringObjectFileELF : public TargetLoweringObjectFile {
  mutable void *UniquingMap;
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
  
protected:
  const MCSection *getELFSection(StringRef Section, unsigned Type, 
                                 unsigned Flags, SectionKind Kind,
                                 bool IsExplicit = false) const;
public:
  TargetLoweringObjectFileELF() : UniquingMap(0) {}
  ~TargetLoweringObjectFileELF();
  
  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);
  
  /// getSectionForConstant - Given a constant with the SectionKind, return a
  /// section that it should be placed in.
  virtual const MCSection *getSectionForConstant(SectionKind Kind) const;
  
  
  virtual const MCSection *
  getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind, 
                           Mangler *Mang, const TargetMachine &TM) const;
  
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};

  
  
class TargetLoweringObjectFileMachO : public TargetLoweringObjectFile {
  mutable void *UniquingMap;
  
  const MCSection *CStringSection;
  const MCSection *UStringSection;
  const MCSection *TextCoalSection;
  const MCSection *ConstTextCoalSection;
  const MCSection *ConstDataCoalSection;
  const MCSection *ConstDataSection;
  const MCSection *DataCoalSection;
  const MCSection *FourByteConstantSection;
  const MCSection *EightByteConstantSection;
  const MCSection *SixteenByteConstantSection;
  
  const MCSection *LazySymbolPointerSection;
  const MCSection *NonLazySymbolPointerSection;
public:
  TargetLoweringObjectFileMachO() : UniquingMap(0) {}
  ~TargetLoweringObjectFileMachO();
  
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

  /// getMachOSection - Return the MCSection for the specified mach-o section.
  /// This requires the operands to be valid.
  const MCSectionMachO *getMachOSection(const StringRef &Segment,
                                        const StringRef &Section,
                                        unsigned TypeAndAttributes,
                                        SectionKind K) const {
    return getMachOSection(Segment, Section, TypeAndAttributes, 0, K);
  }
  const MCSectionMachO *getMachOSection(const StringRef &Segment,
                                        const StringRef &Section,
                                        unsigned TypeAndAttributes,
                                        unsigned Reserved2,
                                        SectionKind K) const;

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
  
  /// getSymbolForDwarfGlobalReference - The mach-o version of this method
  /// defaults to returning a stub reference.
  virtual const MCExpr *
  getSymbolForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                                   MachineModuleInfo *MMI,
                                   bool &IsIndirect, bool &IsPCRel) const;
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
  const MCSection *getCOFFSection(const char *Name, bool isDirective,
                                  SectionKind K) const;
};

} // end namespace llvm

#endif
