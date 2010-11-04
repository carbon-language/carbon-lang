//===--- llvm/Analysis/DIBuilder.h - Debug Information Builder --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a DIBuilder that is useful for creating debugging 
// information entries in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DIBUILDER_H
#define LLVM_ANALYSIS_DIBUILDER_H

#include "llvm/System/DataTypes.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
  class Module;
  class LLVMContext;
  class MDNode;
  class StringRef;
  class DIDescriptor;
  class DIFile;
  class DIEnumerator;
  class DIType;

  class DIBuilder {
    private:
    Module &M;
    LLVMContext & VMContext;
    MDNode *TheCU;

    DIBuilder(const DIBuilder &);       // DO NOT IMPLEMENT
    void operator=(const DIBuilder &);  // DO NOT IMPLEMENT

    public:
    explicit DIBuilder(Module &M);
    const MDNode *getCU() { return TheCU; }

    /// CreateCompileUnit - A CompileUnit provides an anchor for all debugging
    /// information generated during this instance of compilation.
    void CreateCompileUnit(unsigned Lang, StringRef F, StringRef D, StringRef P,
                           bool isOptimized, StringRef Flags, unsigned RV);

    /// CreateFile - Create a file descriptor to hold debugging information
    /// for a file.
    DIFile CreateFile(StringRef Filename, StringRef Directory);
                           
    /// CreateEnumerator - Create a single enumerator value.
    DIEnumerator CreateEnumerator(StringRef Name, uint64_t Val);

    /// CreateBasicType - Create debugging information entry for a basic 
    /// type, e.g 'char'.
    DIType CreateBasicType(StringRef Name, uint64_t SizeInBits, 
                           uint64_t AlignInBits, unsigned Encoding);

    /// CreateQaulifiedType - Create debugging information entry for a qualified
    /// type, e.g. 'const int'.
    DIType CreateQualifiedType(unsigned Tag, DIType FromTy);

    /// CreatePointerType - Create debugging information entry for a pointer.
    DIType CreatePointerType(DIType PointeeTy, uint64_t SizeInBits,
                             uint64_t AlignInBits = 0, StringRef Name = StringRef());

    /// CreateReferenceType - Create debugging information entry for a reference.
    DIType CreateReferenceType(DIType RTy);

    /// CreateTypedef - Create debugging information entry for a typedef.
    DIType CreateTypedef(DIType Ty, StringRef Name, DIFile F, unsigned LineNo);

    /// CreateFriend - Create debugging information entry for a 'friend'.
    DIType CreateFriend(DIType Ty, DIType FriendTy);

    /// CreateInheritance - Create debugging information entry to establish
    /// inheritnace relationship between two types.
    DIType CreateInheritance(DIType Ty, DIType BaseTy, uint64_t BaseOffset,
                             unsigned Flags);

    /// CreateMemberType - Create debugging information entry for a member.
    DIType CreateMemberType(DIDescriptor Context, StringRef Name, DIFile F,
                            unsigned LineNumber, uint64_t SizeInBits, 
                            uint64_t AlignInBits, uint64_t OffsetInBits, 
                            unsigned Flags, DIType Ty);

    /// CreateArtificialType - Create a new DIType with "artificial" flag set.
    DIType CreateArtificialType(DIType Ty);
  };
} // end namespace llvm

#endif
