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
  
private:
  Kind K : 6;
  
  /// Weak - This is true if the referenced symbol is weak (i.e. linkonce,
  /// weak, weak_odr, etc).  This is orthogonal from the categorization.
  bool Weak : 1;
  
  /// ExplicitSection - This is true if the global had a section explicitly
  /// specified on it.
  bool ExplicitSection : 1;
public:
  
  // FIXME: REMOVE.
  Kind getKind() const { return K; }
  
  bool isWeak() const { return Weak; }
  bool hasExplicitSection() const { return ExplicitSection; }
  
  
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
  
  static SectionKind get(Kind K, bool isWeak = false,
                         bool hasExplicitSection = false) {
    SectionKind Res;
    Res.K = K;
    Res.Weak = isWeak;
    Res.ExplicitSection = hasExplicitSection;
    return Res;
  }
};

class Section {
public:

  std::string Name;
  SectionKind Kind;

  explicit Section() { }
  Section(const std::string &N, SectionKind K) : Name(N), Kind(K) {}
  const std::string &getName() const { return Name; }
  SectionKind getKind() const { return Kind; }
};
  
  
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
