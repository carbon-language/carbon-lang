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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
  class MCSection;
  class MCContext;
  class GlobalValue;
  class Mangler;
  class TargetMachine;
  
/// SectionKind - This is a simple POD value that classifies the properties of
/// a section.  A global variable is classified into the deepest possible
/// classification, and then the target maps them onto their sections based on
/// what capabilities they have.
///
/// The comments below describe these as if they were an inheritance hierarchy
/// in order to explain the predicates below.
class SectionKind {
public:
  enum Kind {
    /// Metadata - Debug info sections or other metadata.
    Metadata,
    
    /// Text - Text section, used for functions and other executable code.
    Text,
    
    /// ReadOnly - Data that is never written to at program runtime by the
    /// program or the dynamic linker.  Things in the top-level readonly
    /// SectionKind are not mergeable.
    ReadOnly,

        /// MergeableCString - This is a special section for nul-terminated
        /// strings.  The linker can unique the C strings, knowing their
        /// semantics.  Because it uniques based on the nul terminators, the
        /// compiler can't put strings in this section that have embeded nuls
        /// in them.
        MergeableCString,
    
        /// MergeableConst - These are sections for merging fixed-length
        /// constants together.  For example, this can be used to unique
        /// constant pool entries etc.
        MergeableConst,
    
            /// MergeableConst4 - This is a section used by 4-byte constants,
            /// for example, floats.
            MergeableConst4,
    
            /// MergeableConst8 - This is a section used by 8-byte constants,
            /// for example, doubles.
            MergeableConst8,

            /// MergeableConst16 - This is a section used by 16-byte constants,
            /// for example, vectors.
            MergeableConst16,
    
    /// Writeable - This is the base of all segments that need to be written
    /// to during program runtime.
    
       /// ThreadLocal - This is the base of all TLS segments.  All TLS
       /// objects must be writeable, otherwise there is no reason for them to
       /// be thread local!
    
           /// ThreadBSS - Zero-initialized TLS data objects.
           ThreadBSS,
    
           /// ThreadData - Initialized TLS data objects.
           ThreadData,
    
       /// GlobalWriteableData - Writeable data that is global (not thread
       /// local).
    
           /// BSS - Zero initialized writeable data.
           BSS,

           /// DataRel - This is the most general form of data that is written
           /// to by the program, it can have random relocations to arbitrary
           /// globals.
           DataRel,

               /// DataRelLocal - This is writeable data that has a non-zero
               /// initializer and has relocations in it, but all of the
               /// relocations are known to be within the final linked image
               /// the global is linked into.
               DataRelLocal,

                   /// DataNoRel - This is writeable data that has a non-zero
                   /// initializer, but whose initializer is known to have no
                   /// relocations.
                   DataNoRel,

           /// ReadOnlyWithRel - These are global variables that are never
           /// written to by the program, but that have relocations, so they
           /// must be stuck in a writeable section so that the dynamic linker
           /// can write to them.  If it chooses to, the dynamic linker can
           /// mark the pages these globals end up on as read-only after it is
           /// done with its relocation phase.
           ReadOnlyWithRel,
    
               /// ReadOnlyWithRelLocal - This is data that is readonly by the
               /// program, but must be writeable so that the dynamic linker
               /// can perform relocations in it.  This is used when we know
               /// that all the relocations are to globals in this final
               /// linked image.
               ReadOnlyWithRelLocal
    
  };
  
protected:
  Kind K : 6;
  
public:
  
  bool isMetadata() const { return K == Metadata; }
  bool isText() const { return K == Text; }
  
  bool isReadOnly() const {
    return K == ReadOnly || K == MergeableCString || isMergeableConst();
  }

  bool isMergeableCString() const { return K == MergeableCString; }
  bool isMergeableConst() const {
    return K == MergeableConst || K == MergeableConst4 ||
           K == MergeableConst8 || K == MergeableConst16;
  }
  
  bool isMergeableConst4() const { return K == MergeableConst4; }
  bool isMergeableConst8() const { return K == MergeableConst8; }
  bool isMergeableConst16() const { return K == MergeableConst16; }
  
  bool isWriteable() const {
    return isThreadLocal() || isGlobalWriteableData();
  }
  
  bool isThreadLocal() const {
    return K == ThreadData || K == ThreadBSS;
  }
  
  bool isThreadBSS() const { return K == ThreadBSS; } 
  bool isThreadData() const { return K == ThreadData; } 

  bool isGlobalWriteableData() const {
    return isBSS() || isDataRel() || isReadOnlyWithRel();
  }
  
  bool isBSS() const { return K == BSS; }
  
  bool isDataRel() const {
    return K == DataRel || K == DataRelLocal || K == DataNoRel;
  }
  
  bool isDataRelLocal() const {
    return K == DataRelLocal || K == DataNoRel;
  }

  bool isDataNoRel() const { return K == DataNoRel; }
  
  bool isReadOnlyWithRel() const {
    return K == ReadOnlyWithRel || K == ReadOnlyWithRelLocal;
  }

  bool isReadOnlyWithRelLocal() const {
    return K == ReadOnlyWithRelLocal;
  }
  
  static SectionKind get(Kind K) {
    SectionKind Res;
    Res.K = K;
    return Res;
  }
};

  
/// SectionInfo - This class is a target-independent classification of a global
/// which is used to simplify target-specific code by exposing common
/// predicates.
class SectionInfo : public SectionKind {
  /// Weak - This is true if the referenced symbol is weak (i.e. linkonce,
  /// weak, weak_odr, etc).  This is orthogonal from the categorization.
  bool Weak : 1;
  
public:
  
  /// Weak - This is true if the referenced symbol is weak (i.e. linkonce,
  /// weak, weak_odr, etc).  This is orthogonal from the categorization.
  bool isWeak() const { return Weak; }
  
  static SectionInfo get(Kind K, bool isWeak = false) {
    SectionInfo Res;
    Res.K = K;
    Res.Weak = isWeak;
    return Res;
  }
  static SectionInfo get(SectionKind K, bool isWeak = false) {
    SectionInfo Res;
    *(SectionKind*)&Res = K;
    Res.Weak = isWeak;
    return Res;
  }
};
  
class TargetLoweringObjectFile {
  MCContext *Ctx;
protected:
  
  TargetLoweringObjectFile();
  
  /// TextSection - Section directive for standard text.
  ///
  const MCSection *TextSection;           // Defaults to ".text".
  
  /// DataSection - Section directive for standard data.
  ///
  const MCSection *DataSection;           // Defaults to ".data".
  
  
  
  // FIXME: SINK THESE.
  const MCSection *BSSSection_;

  /// ReadOnlySection - This is the directive that is emitted to switch to a
  /// read-only section for constant data (e.g. data declared const,
  /// jump tables).
  const MCSection *ReadOnlySection;       // Defaults to NULL
  
  /// TLSDataSection - Section directive for Thread Local data.
  ///
  const MCSection *TLSDataSection;        // Defaults to ".tdata".
  
  /// TLSBSSSection - Section directive for Thread Local uninitialized data.
  /// Null if this target doesn't support a BSS section.
  ///
  const MCSection *TLSBSSSection;         // Defaults to ".tbss".
  
  const MCSection *CStringSection_;
  
public:
  // FIXME: NONPUB.
  const MCSection *getOrCreateSection(const char *Name,
                                      bool isDirective,
                                      SectionKind K) const;
public:
  
  virtual ~TargetLoweringObjectFile();
  
  /// Initialize - this method must be called before any actual lowering is
  /// done.  This specifies the current context for codegen, and gives the
  /// lowering implementations a chance to set up their default sections.
  virtual void Initialize(MCContext &ctx, const TargetMachine &TM) {
    Ctx = &ctx;
  }
  
  
  const MCSection *getTextSection() const { return TextSection; }
  const MCSection *getDataSection() const { return DataSection; }
  
  /// shouldEmitUsedDirectiveFor - This hook allows targets to selectively
  /// decide not to emit the UsedDirective for some symbols in llvm.used.
  /// FIXME: REMOVE this (rdar://7071300)
  virtual bool shouldEmitUsedDirectiveFor(const GlobalValue *GV,
                                          Mangler *) const {
    return (GV!=0);
  }
  
  /// getSectionForMergeableConstant - Given a mergeable constant with the
  /// specified size and relocation information, return a section that it
  /// should be placed in.
  virtual const MCSection *
  getSectionForMergeableConstant(SectionKind Kind) const;
  
  /// getKindForNamedSection - If this target wants to be able to override
  /// section flags based on the name of the section specified for a global
  /// variable, it can implement this.  This is used on ELF systems so that
  /// ".tbss" gets the TLS bit set etc.
  virtual SectionKind getKindForNamedSection(const char *Section,
                                             SectionKind K) const {
    return K;
  }
  
  /// SectionForGlobal - This method computes the appropriate section to emit
  /// the specified global variable or function definition.  This should not
  /// be passed external (or available externally) globals.
  const MCSection *SectionForGlobal(const GlobalValue *GV,
                                    Mangler *Mang,
                                    const TargetMachine &TM) const;
  
  /// getSpecialCasedSectionGlobals - Allow the target to completely override
  /// section assignment of a global.
  /// FIXME: ELIMINATE this by making PIC16 implement ADDRESS with
  /// getFlagsForNamedSection.
  virtual const MCSection *
  getSpecialCasedSectionGlobals(const GlobalValue *GV, Mangler *Mang,
                                SectionInfo Kind) const {
    return 0;
  }
  
  /// getSectionFlagsAsString - Turn the flags in the specified SectionKind
  /// into a string that can be printed to the assembly file after the
  /// ".section foo" part of a section directive.
  virtual void getSectionFlagsAsString(SectionKind Kind,
                                       SmallVectorImpl<char> &Str) const {
  }
  
protected:
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionInfo Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};
  
  
  

class TargetLoweringObjectFileELF : public TargetLoweringObjectFile {
  bool AtIsCommentChar;  // True if @ is the comment character on this target.
  bool HasCrazyBSS;
public:
  /// ELF Constructor - AtIsCommentChar is true if the CommentCharacter from TAI
  /// is "@".
  TargetLoweringObjectFileELF(bool atIsCommentChar = false,
                              // FIXME: REMOVE AFTER UNIQUING IS FIXED.
                              bool hasCrazyBSS = false)
    : AtIsCommentChar(atIsCommentChar), HasCrazyBSS(hasCrazyBSS) {}
    
  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);
  
  
  /// getSectionForMergeableConstant - Given a mergeable constant with the
  /// specified size and relocation information, return a section that it
  /// should be placed in.
  virtual const MCSection *
  getSectionForMergeableConstant(SectionKind Kind) const;
  
  virtual SectionKind getKindForNamedSection(const char *Section,
                                             SectionKind K) const;
  void getSectionFlagsAsString(SectionKind Kind,
                               SmallVectorImpl<char> &Str) const;
  
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionInfo Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
protected:
  const MCSection *DataRelSection;
  const MCSection *DataRelLocalSection;
  const MCSection *DataRelROSection;
  const MCSection *DataRelROLocalSection;
  
  const MCSection *MergeableConst4Section;
  const MCSection *MergeableConst8Section;
  const MCSection *MergeableConst16Section;
};

  
  
class TargetLoweringObjectFileMachO : public TargetLoweringObjectFile {
  const MCSection *TextCoalSection;
  const MCSection *ConstTextCoalSection;
  const MCSection *ConstDataCoalSection;
  const MCSection *ConstDataSection;
  const MCSection *DataCoalSection;
  const MCSection *FourByteConstantSection;
  const MCSection *EightByteConstantSection;
  const MCSection *SixteenByteConstantSection;
public:
  
  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);

  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionInfo Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
  
  virtual const MCSection *
  getSectionForMergeableConstant(SectionKind Kind) const;
  
  /// shouldEmitUsedDirectiveFor - This hook allows targets to selectively
  /// decide not to emit the UsedDirective for some symbols in llvm.used.
  /// FIXME: REMOVE this (rdar://7071300)
  virtual bool shouldEmitUsedDirectiveFor(const GlobalValue *GV,
                                          Mangler *) const;
};



class TargetLoweringObjectFileCOFF : public TargetLoweringObjectFile {
public:
  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);
  
  virtual void getSectionFlagsAsString(SectionKind Kind,
                                       SmallVectorImpl<char> &Str) const;
  
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionInfo Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};

} // end namespace llvm

#endif
