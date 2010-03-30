//===--- CGRecordLayout.h - LLVM Record Layout Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGRECORDLAYOUT_H
#define CLANG_CODEGEN_CGRECORDLAYOUT_H

namespace clang {
namespace CodeGen {

/// CGRecordLayout - This class handles struct and union layout info while
/// lowering AST types to LLVM types.
class CGRecordLayout {
  CGRecordLayout(const CGRecordLayout&); // DO NOT IMPLEMENT
  void operator=(const CGRecordLayout&); // DO NOT IMPLEMENT

  /// The LLVMType corresponding to this record layout.
  const llvm::Type *LLVMType;

  /// Whether one of the fields in this record layout is a pointer to data
  /// member, or a struct that contains pointer to data member.
  bool ContainsPointerToDataMember;

public:
  CGRecordLayout(const llvm::Type *T, bool ContainsPointerToDataMember)
    : LLVMType(T), ContainsPointerToDataMember(ContainsPointerToDataMember) {}

  /// getLLVMType - Return llvm type associated with this record.
  const llvm::Type *getLLVMType() const {
    return LLVMType;
  }

  /// containsPointerToDataMember - Whether this struct contains pointers to
  /// data members.
  bool containsPointerToDataMember() const {
    return ContainsPointerToDataMember;
  }
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
