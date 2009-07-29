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

// FIXME: Switch to MC.
#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {
  
class TargetLoweringObjectFile {
private:
  mutable StringMap<Section> Sections;
protected:
  
  TargetLoweringObjectFile();
  
  /// TextSection - Section directive for standard text.
  ///
  const Section *TextSection;           // Defaults to ".text".
  
  /// DataSection - Section directive for standard data.
  ///
  const Section *DataSection;           // Defaults to ".data".
  
  
  
  // FIXME: SINK THESE.
  const Section *BSSSection_;

  /// ReadOnlySection - This is the directive that is emitted to switch to a
  /// read-only section for constant data (e.g. data declared const,
  /// jump tables).
  const Section *ReadOnlySection;       // Defaults to NULL
  
  /// TLSDataSection - Section directive for Thread Local data.
  ///
  const Section *TLSDataSection;        // Defaults to ".tdata".
  
  /// TLSBSSSection - Section directive for Thread Local uninitialized data.
  /// Null if this target doesn't support a BSS section.
  ///
  const Section *TLSBSSSection;         // Defaults to ".tbss".
  
  const Section *CStringSection_;
  
public:
  // FIXME: NONPUB.
  const Section *getOrCreateSection(const char *Name,
                                    bool isDirective,
                                    SectionKind::Kind K) const;
public:
  
  virtual ~TargetLoweringObjectFile();
  
  const Section *getTextSection() const { return TextSection; }
  const Section *getDataSection() const { return DataSection; }
  
  
  /// getSectionForMergeableConstant - Given a mergeable constant with the
  /// specified size and relocation information, return a section that it
  /// should be placed in.
  virtual const Section *
  getSectionForMergeableConstant(SectionKind Kind) const;
  
  /// getKindForNamedSection - If this target wants to be able to override
  /// section flags based on the name of the section specified for a global
  /// variable, it can implement this.  This is used on ELF systems so that
  /// ".tbss" gets the TLS bit set etc.
  virtual SectionKind::Kind getKindForNamedSection(const char *Section,
                                                   SectionKind::Kind K) const{
    return K;
  }
  
  /// SectionForGlobal - This method computes the appropriate section to emit
  /// the specified global variable or function definition.  This should not
  /// be passed external (or available externally) globals.
  const Section *SectionForGlobal(const GlobalValue *GV,
                                  Mangler *Mang,
                                  const TargetMachine &TM) const;
  
  /// getSpecialCasedSectionGlobals - Allow the target to completely override
  /// section assignment of a global.
  /// FIXME: ELIMINATE this by making PIC16 implement ADDRESS with
  /// getFlagsForNamedSection.
  virtual const Section *
  getSpecialCasedSectionGlobals(const GlobalValue *GV, Mangler *Mang,
                                SectionKind Kind) const {
    return 0;
  }
  
  /// getSectionFlagsAsString - Turn the flags in the specified SectionKind
  /// into a string that can be printed to the assembly file after the
  /// ".section foo" part of a section directive.
  virtual void getSectionFlagsAsString(SectionKind Kind,
                                       SmallVectorImpl<char> &Str) const {
  }
  
protected:
  virtual const Section *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};
  
  
  

class TargetLoweringObjectFileELF : public TargetLoweringObjectFile {
  bool AtIsCommentChar;  // True if @ is the comment character on this target.
public:
  /// ELF Constructor - AtIsCommentChar is true if the CommentCharacter from TAI
  /// is "@".
  TargetLoweringObjectFileELF(bool AtIsCommentChar = false,
                              // FIXME: REMOVE AFTER UNIQUING IS FIXED.
                              bool HasCrazyBSS = false);
  
  /// getSectionForMergeableConstant - Given a mergeable constant with the
  /// specified size and relocation information, return a section that it
  /// should be placed in.
  virtual const Section *
  getSectionForMergeableConstant(SectionKind Kind) const;
  
  virtual SectionKind::Kind getKindForNamedSection(const char *Section,
                                                   SectionKind::Kind K) const;
  void getSectionFlagsAsString(SectionKind Kind,
                               SmallVectorImpl<char> &Str) const;
  
  virtual const Section *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
protected:
  const Section *DataRelSection;
  const Section *DataRelLocalSection;
  const Section *DataRelROSection;
  const Section *DataRelROLocalSection;
  
  const Section *MergeableConst4Section;
  const Section *MergeableConst8Section;
  const Section *MergeableConst16Section;
};

class TargetLoweringObjectFileMachO : public TargetLoweringObjectFile {
  const Section *TextCoalSection;
  const Section *ConstTextCoalSection;
  const Section *ConstDataCoalSection;
  const Section *ConstDataSection;
  const Section *DataCoalSection;
  const Section *FourByteConstantSection;
  const Section *EightByteConstantSection;
  const Section *SixteenByteConstantSection;
public:
  TargetLoweringObjectFileMachO(const TargetMachine &TM);
  virtual const Section *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
  
  virtual const Section *
  getSectionForMergeableConstant(SectionKind Kind) const;
};



class TargetLoweringObjectFileCOFF : public TargetLoweringObjectFile {
public:
  TargetLoweringObjectFileCOFF();
  virtual void getSectionFlagsAsString(SectionKind Kind,
                                       SmallVectorImpl<char> &Str) const;
  
  virtual const Section *
  SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};

} // end namespace llvm

#endif
