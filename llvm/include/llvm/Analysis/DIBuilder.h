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

#include "llvm/Support/DataTypes.h"
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
    /// @param Lang     Source programming language, eg. dwarf::DW_LANG_C99
    /// @param File     File name
    /// @param Dir      Directory
    /// @param Producer String identify producer of debugging information. 
    ///                 Usuall this is a compiler version string.
    /// @param isOptimized A boolean flag which indicates whether optimization
    ///                    is ON or not.
    /// @param Flags    This string lists command line options. This string is 
    ///                 directly embedded in debug info output which may be used
    ///                 by a tool analyzing generated debugging information.
    /// @param RV       This indicates runtime version for languages like 
    ///                 Objective-C.
    void CreateCompileUnit(unsigned Lang, StringRef File, StringRef Dir, 
                           StringRef Producer,
                           bool isOptimized, StringRef Flags, unsigned RV);

    /// CreateFile - Create a file descriptor to hold debugging information
    /// for a file.
    DIFile CreateFile(StringRef Filename, StringRef Directory);
                           
    /// CreateEnumerator - Create a single enumerator value.
    DIEnumerator CreateEnumerator(StringRef Name, uint64_t Val);

    /// CreateBasicType - Create debugging information entry for a basic 
    /// type.
    /// @param Name        Type name.
    /// @param SizeInBits  Size of the type.
    /// @param AlignInBits Type alignment.
    /// @param Encoding    DWARF encoding code, e.g. dwarf::DW_ATE_float.
    DIType CreateBasicType(StringRef Name, uint64_t SizeInBits, 
                           uint64_t AlignInBits, unsigned Encoding);

    /// CreateQualifiedType - Create debugging information entry for a qualified
    /// type, e.g. 'const int'.
    /// @param Tag         Tag identifing type, e.g. dwarf::TAG_volatile_type
    /// @param FromTy      Base Type.
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
    /// inheritance relationship between two types.
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
