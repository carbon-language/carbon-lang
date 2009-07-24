//===--- CGRecordLayoutBuilder.h - Record builder helper --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a helper class used to build CGRecordLayout objects and LLVM types.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGRECORDLAYOUTBUILDER_H
#define CLANG_CODEGEN_CGRECORDLAYOUTBUILDER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {
  class Type;
}

namespace clang {
  class FieldDecl;
  class RecordDecl;
  
namespace CodeGen {
  class CGRecordLayout;
  class CodeGenTypes;

class CGRecordLayoutBuilder {  
  CodeGenTypes &Types;
  
  /// Packed - Whether the resulting LLVM struct will be packed or not.
  bool Packed;

  /// AlignmentAsLLVMStruct - Will contain the maximum alignment of all the
  /// LLVM types.
  unsigned AlignmentAsLLVMStruct;
  
  /// BitsAvailableInLastField - If a bit field spans only part of a LLVM field,
  /// this will have the number of bits still available in the field.
  char BitsAvailableInLastField;
  
  /// FieldTypes - Holds the LLVM types that the struct is created from.
  std::vector<const llvm::Type *> FieldTypes;
  
  /// FieldInfo - Holds size and offset information about a field.
  /// FIXME: I think we can get rid of this.
  struct FieldInfo {
    FieldInfo(uint64_t OffsetInBytes, uint64_t SizeInBytes)
      : OffsetInBytes(OffsetInBytes), SizeInBytes(SizeInBytes) { }
    
    const uint64_t OffsetInBytes;
    const uint64_t SizeInBytes;
  };
  llvm::SmallVector<FieldInfo, 16> FieldInfos;

  /// LLVMFieldInfo - Holds a field and its corresponding LLVM field number.
  typedef std::pair<const FieldDecl *, unsigned> LLVMFieldInfo;
  llvm::SmallVector<LLVMFieldInfo, 16> LLVMFields;

  /// LLVMBitFieldInfo - Holds location and size information about a bit field.
  struct LLVMBitFieldInfo {
    LLVMBitFieldInfo(const FieldDecl *FD, unsigned FieldNo, unsigned Start, 
                     unsigned Size)
      : FD(FD), FieldNo(FieldNo), Start(Start), Size(Size) { }
    
    const FieldDecl *FD;
    
    unsigned FieldNo;
    unsigned Start;
    unsigned Size;
  };
  llvm::SmallVector<LLVMBitFieldInfo, 16> LLVMBitFields;
  
  CGRecordLayoutBuilder(CodeGenTypes &Types) 
    : Types(Types), Packed(false), AlignmentAsLLVMStruct(1)
    , BitsAvailableInLastField(0) { }

  /// Layout - Will layout a RecordDecl.
  void Layout(const RecordDecl *D);

  /// LayoutUnion - Will layout a union RecordDecl.
  void LayoutUnion(const RecordDecl *D);
  
  /// LayoutField - try to layout all fields in the record decl.
  /// Returns false if the operation failed because the struct is not packed.
  bool LayoutFields(const RecordDecl *D);
  
  /// LayoutField - layout a single field. Returns false if the operation failed
  /// because the current struct is not packed.
  bool LayoutField(const FieldDecl *D, uint64_t FieldOffset);
  
  /// LayoutBitField - layout a single bit field.
  void LayoutBitField(const FieldDecl *D, uint64_t FieldOffset);

  /// AppendField - Appends a field with the given offset and type.
  void AppendField(uint64_t FieldOffsetInBytes, const llvm::Type *FieldTy);

  /// AppendPadding - Appends enough padding bytes so that the total struct
  /// size matches the alignment of the passed in type.
  void AppendPadding(uint64_t FieldOffsetInBytes, const llvm::Type *FieldTy);

  /// AppendPadding - Appends enough padding bytes so that the total
  /// struct size is a multiple of the field alignment.
  void AppendPadding(uint64_t FieldOffsetInBytes, unsigned FieldAlignment);
  
  /// AppendBytes - Append a given number of bytes to the record.
  void AppendBytes(uint64_t NumBytes);

  /// getNextFieldOffsetInBytes - returns where the next field offset is.
  uint64_t getNextFieldOffsetInBytes() const;
  
  unsigned getTypeAlignment(const llvm::Type *Ty) const;
  uint64_t getTypeSizeInBytes(const llvm::Type *Ty) const;

public:
  /// ComputeLayout - Return the right record layout for a given record decl.
  static CGRecordLayout *ComputeLayout(CodeGenTypes &Types, 
                                       const RecordDecl *D);
};
  
} // end namespace CodeGen
} // end namespace clang
                                             

#endif 
