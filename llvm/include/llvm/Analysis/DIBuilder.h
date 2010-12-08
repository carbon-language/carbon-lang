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
  class BasicBlock;
  class Instruction;
  class Function;
  class Module;
  class Value;
  class LLVMContext;
  class MDNode;
  class StringRef;
  class DIDescriptor;
  class DIFile;
  class DIEnumerator;
  class DIType;
  class DIArray;
  class DIGlobalVariable;
  class DINameSpace;
  class DIVariable;

  class DIBuilder {
    private:
    Module &M;
    LLVMContext & VMContext;
    MDNode *TheCU;

    Function *DeclareFn;     // llvm.dbg.declare
    Function *ValueFn;       // llvm.dbg.value

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
    /// @param PointeeTy   Type pointed by this pointer.
    /// @param SizeInBits  Size.
    /// @param AlignInBits Alignment. (optional)
    /// @param Name        Pointer type name. (optional)
    DIType CreatePointerType(DIType PointeeTy, uint64_t SizeInBits,
                             uint64_t AlignInBits = 0, 
                             StringRef Name = StringRef());

    /// CreateReferenceType - Create debugging information entry for a c++
    /// style reference.
    DIType CreateReferenceType(DIType RTy);

    /// CreateTypedef - Create debugging information entry for a typedef.
    /// @param Ty          Original type.
    /// @param Name        Typedef name.
    /// @param File        File where this type is defined.
    /// @param LineNo      Line number.
    DIType CreateTypedef(DIType Ty, StringRef Name, DIFile File, 
                         unsigned LineNo);

    /// CreateFriend - Create debugging information entry for a 'friend'.
    DIType CreateFriend(DIType Ty, DIType FriendTy);

    /// CreateInheritance - Create debugging information entry to establish
    /// inheritance relationship between two types.
    /// @param Ty           Original type.
    /// @param BaseTy       Base type. Ty is inherits from base.
    /// @param BaseOffset   Base offset.
    /// @param Flags        Flags to describe inheritance attribute, 
    ///                     e.g. private
    DIType CreateInheritance(DIType Ty, DIType BaseTy, uint64_t BaseOffset,
                             unsigned Flags);

    /// CreateMemberType - Create debugging information entry for a member.
    /// @param Name         Member name.
    /// @param File         File where this member is defined.
    /// @param LineNo       Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param OffsetInBits Member offset.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Ty           Parent type.
    DIType CreateMemberType(StringRef Name, DIFile File,
                            unsigned LineNo, uint64_t SizeInBits, 
                            uint64_t AlignInBits, uint64_t OffsetInBits, 
                            unsigned Flags, DIType Ty);

    /// CreateStructType - Create debugging information entry for a struct.
    DIType CreateStructType(DIDescriptor Context, StringRef Name, DIFile F,
                            unsigned LineNumber, uint64_t SizeInBits,
                            uint64_t AlignInBits, unsigned Flags,
                            DIArray Elements, unsigned RunTimeLang = 0);

    /// CreateArtificialType - Create a new DIType with "artificial" flag set.
    DIType CreateArtificialType(DIType Ty);

    /// CreateTemporaryType - Create a temporary forward-declared type.
    DIType CreateTemporaryType();
    DIType CreateTemporaryType(DIFile F);

    /// GetOrCreateArray - Get a DIArray, create one if required.
    DIArray GetOrCreateArray(Value *const *Elements, unsigned NumElements);

    /// CreateGlobalVariable - Create a new descriptor for the specified global.
    /// @param Name        Name of the variable.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type.
    /// @param isLocalToUnit Boolean flag indicate whether this variable is
    ///                      externally visible or not.
    /// @param Val         llvm::Value of the variable.
    DIGlobalVariable
    CreateGlobalVariable(StringRef Name, DIFile File, unsigned LineNo,
                         DIType Ty, bool isLocalToUnit, llvm::Value *Val);


    /// CreateStaticVariable - Create a new descriptor for the specified 
    /// variable.
    /// @param Conext      Variable scope. 
    /// @param Name        Name of the variable.
    /// @param LinakgeName Mangled  name of the variable.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type.
    /// @param isLocalToUnit Boolean flag indicate whether this variable is
    ///                      externally visible or not.
    /// @param Val         llvm::Value of the variable.
    DIGlobalVariable
    CreateStaticVariable(DIDescriptor Context, StringRef Name, 
                         StringRef LinkageName, DIFile File, unsigned LineNo, 
                         DIType Ty, bool isLocalToUnit, llvm::Value *Val);


    /// CreateLocalVariable - Create a new descriptor for the specified 
    /// local variable.
    /// @param Tag         Dwarf TAG. Usually DW_TAG_auto_variable or
    ///                    DW_TAG_arg_variable.
    /// @param Scope       Variable scope.
    /// @param Name        Variable name.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type
    /// @param AlwaysPreserve Boolean. Set to true if debug info for this
    ///                       variable should be preserved in optimized build.
    /// @param Flags          Flags, e.g. artificial variable.
    DIVariable CreateLocalVariable(unsigned Tag, DIDescriptor Scope,
                                   StringRef Name,
                                   DIFile File, unsigned LineNo,
                                   DIType Ty, bool AlwaysPreserve = false,
                                   unsigned Flags = 0);


    /// CreateComplexVariable - Create a new descriptor for the specified
    /// variable which has a complex address expression for its address.
    /// @param Tag         Dwarf TAG. Usually DW_TAG_auto_variable or
    ///                    DW_TAG_arg_variable.
    /// @param Scope       Variable scope.
    /// @param Name        Variable name.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type
    /// @param Addr        A pointer to a vector of complex address operations.
    /// @param NumAddr     Num of address operations in the vector.
    DIVariable CreateComplexVariable(unsigned Tag, DIDescriptor Scope,
                                     StringRef Name, DIFile F, unsigned LineNo,
                                     DIType Ty, Value *const *Addr,
                                     unsigned NumAddr);


    /// CreateNameSpace - This creates new descriptor for a namespace
    /// with the specified parent scope.
    /// @param Scope       Namespace scope
    /// @param Name        Name of this namespace
    /// @param File        Source file
    /// @param LineNo      Line number
    DINameSpace CreateNameSpace(DIDescriptor Scope, StringRef Name,
                                DIFile File, unsigned LineNo);


    /// InsertDeclare - Insert a new llvm.dbg.declare intrinsic call.
    /// @param Storage     llvm::Value of the variable
    /// @param VarInfo     Variable's debug info descriptor.
    /// @param InsertAtEnd Location for the new intrinsic.
    Instruction *InsertDeclare(llvm::Value *Storage, DIVariable VarInfo,
                               BasicBlock *InsertAtEnd);

    /// InsertDeclare - Insert a new llvm.dbg.declare intrinsic call.
    /// @param Storage      llvm::Value of the variable
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param InsertBefore Location for the new intrinsic.
    Instruction *InsertDeclare(llvm::Value *Storage, DIVariable VarInfo,
                               Instruction *InsertBefore);


    /// InsertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
    /// @param Val          llvm::Value of the variable
    /// @param Offset       Offset
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param InsertAtEnd Location for the new intrinsic.
    Instruction *InsertDbgValueIntrinsic(llvm::Value *Val, uint64_t Offset,
                                         DIVariable VarInfo, 
                                         BasicBlock *InsertAtEnd);
    
    /// InsertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
    /// @param Val          llvm::Value of the variable
    /// @param Offset       Offset
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param InsertBefore Location for the new intrinsic.
    Instruction *InsertDbgValueIntrinsic(llvm::Value *Val, uint64_t Offset,
                                         DIVariable VarInfo, 
                                         Instruction *InsertBefore);

  };
} // end namespace llvm

#endif
