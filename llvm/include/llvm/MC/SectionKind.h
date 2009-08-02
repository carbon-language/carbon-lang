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

#ifndef LLVM_MC_SECTIONKIND_H
#define LLVM_MC_SECTIONKIND_H

namespace llvm {

/// SectionKind - This is a simple POD value that classifies the properties of
/// a section.  A section is classified into the deepest possible
/// classification, and then the target maps them onto their sections based on
/// what capabilities they have.
///
/// The comments below describe these as if they were an inheritance hierarchy
/// in order to explain the predicates below.
///
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
  Kind K : 8;
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
private: 
  static SectionKind get(Kind K) {
    SectionKind Res;
    Res.K = K;
    return Res;
  }
public:
  
  static SectionKind getMetadata() { return get(Metadata); }
  static SectionKind getText() { return get(Text); }
  static SectionKind getReadOnly() { return get(ReadOnly); }
  static SectionKind getMergeableCString() { return get(MergeableCString); }
  static SectionKind getMergeableConst() { return get(MergeableConst); }
  static SectionKind getMergeableConst4() { return get(MergeableConst4); }
  static SectionKind getMergeableConst8() { return get(MergeableConst8); }
  static SectionKind getMergeableConst16() { return get(MergeableConst16); }
  static SectionKind getThreadBSS() { return get(ThreadBSS); }
  static SectionKind getThreadData() { return get(ThreadData); }
  static SectionKind getBSS() { return get(BSS); }
  static SectionKind getDataRel() { return get(DataRel); }
  static SectionKind getDataRelLocal() { return get(DataRelLocal); }
  static SectionKind getDataNoRel() { return get(DataNoRel); }
  static SectionKind getReadOnlyWithRel() { return get(ReadOnlyWithRel); }
  static SectionKind getReadOnlyWithRelLocal(){
    return get(ReadOnlyWithRelLocal);
  }
};
  
} // end namespace llvm

#endif
