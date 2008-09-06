//===--- CodeGenTypes.h - Type translation for LLVM CodeGen -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the code that handles AST -> LLVM type lowering. 
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CODEGENTYPES_H
#define CLANG_CODEGEN_CODEGENTYPES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include <vector>

namespace llvm {
  class Module;
  class Type;
  class OpaqueType;
  class PATypeHolder;
  class TargetData;
}

namespace clang {
  class ASTContext;
  class TagDecl;
  class TargetInfo;
  class QualType;
  class PointerType;
  class PointerLikeType;
  class Type;
  class FunctionTypeProto;
  class FieldDecl;
  class RecordDecl;
  class ObjCInterfaceDecl;
  class ObjCIvarDecl;

namespace CodeGen {
  class CodeGenTypes;

  /// CGRecordLayout - This class handles struct and union layout info while 
  /// lowering AST types to LLVM types.
  class CGRecordLayout {
    CGRecordLayout(); // DO NOT IMPLEMENT
  public:
    CGRecordLayout(llvm::Type *T, llvm::SmallSet<unsigned, 8> &PF) 
      : STy(T), PaddingFields(PF) {
      // FIXME : Collect info about fields that requires adjustments 
      // (i.e. fields that do not directly map to llvm struct fields.)
    }

    /// getLLVMType - Return llvm type associated with this record.
    llvm::Type *getLLVMType() const {
      return STy;
    }

    bool isPaddingField(unsigned No) const {
      return PaddingFields.count(No) != 0;
    }

    unsigned getNumPaddingFields() {
      return PaddingFields.size();
    }

  private:
    llvm::Type *STy;
    llvm::SmallSet<unsigned, 8> PaddingFields;
  };
  
/// CodeGenTypes - This class organizes the cross-module state that is used
/// while lowering AST types to LLVM types.
class CodeGenTypes {
  ASTContext &Context;
  TargetInfo &Target;
  llvm::Module& TheModule;
  const llvm::TargetData& TheTargetData;
  

  llvm::SmallVector<std::pair<const PointerLikeType *,
                              llvm::OpaqueType *>, 8>  PointersToResolve;

  llvm::DenseMap<const Type*, llvm::PATypeHolder> TagDeclTypes;

  /// CGRecordLayouts - This maps llvm struct type with corresponding 
  /// record layout info. 
  /// FIXME : If CGRecordLayout is less than 16 bytes then use 
  /// inline it in the map.
  llvm::DenseMap<const Type*, CGRecordLayout *> CGRecordLayouts;

  /// FieldInfo - This maps struct field with corresponding llvm struct type
  /// field no. This info is populated by record organizer.
  llvm::DenseMap<const FieldDecl *, unsigned> FieldInfo;
  llvm::DenseMap<const ObjCIvarDecl *, unsigned> ObjCIvarInfo;

public:
  class BitFieldInfo {
  public:
    explicit BitFieldInfo(unsigned short B, unsigned short S)
      : Begin(B), Size(S) {}

    unsigned short Begin;
    unsigned short Size;
  };

private:
  llvm::DenseMap<const FieldDecl *, BitFieldInfo> BitFields;

  /// TypeCache - This map keeps cache of llvm::Types (through PATypeHolder)
  /// and maps llvm::Types to corresponding clang::Type. llvm::PATypeHolder is
  /// used instead of llvm::Type because it allows us to bypass potential 
  /// dangling type pointers due to type refinement on llvm side.
  llvm::DenseMap<Type *, llvm::PATypeHolder> TypeCache;

  /// ConvertNewType - Convert type T into a llvm::Type. Do not use this
  /// method directly because it does not do any type caching. This method
  /// is available only for ConvertType(). CovertType() is preferred
  /// interface to convert type T into a llvm::Type.
  const llvm::Type *ConvertNewType(QualType T);
public:
  CodeGenTypes(ASTContext &Ctx, llvm::Module &M, const llvm::TargetData &TD);
  ~CodeGenTypes();
  
  const llvm::TargetData &getTargetData() const { return TheTargetData; }
  TargetInfo &getTarget() const { return Target; }
  ASTContext &getContext() const { return Context; }

  /// ConvertType - Convert type T into a llvm::Type.  
  const llvm::Type *ConvertType(QualType T);
  const llvm::Type *ConvertTypeRecursive(QualType T);
  /// ConvertReturnType - Convert T into an llvm::Type assuming that it will be
  /// used as a function return type.
  const llvm::Type *ConvertReturnType(QualType T);
  
  /// ConvertTypeForMem - Convert type T into a llvm::Type.  This differs from
  /// ConvertType in that it is used to convert to the memory representation for
  /// a type.  For example, the scalar representation for _Bool is i1, but the
  /// memory representation is usually i8 or i32, depending on the target.
  const llvm::Type *ConvertTypeForMem(QualType T);
  
  void CollectObjCIvarTypes(ObjCInterfaceDecl *ObjCClass,
                            std::vector<const llvm::Type*> &IvarTypes);
  
  const CGRecordLayout *getCGRecordLayout(const TagDecl*) const;
  /// Returns a StructType representing an Objective-C object
  const llvm::Type *ConvertObjCInterfaceToStruct(const ObjCInterfaceDecl *OID);
  
  /// getLLVMFieldNo - Return llvm::StructType element number
  /// that corresponds to the field FD.
  unsigned getLLVMFieldNo(const FieldDecl *FD);
  unsigned getLLVMFieldNo(const ObjCIvarDecl *OID);
    
  
  /// UpdateCompletedType - When we find the full definition for a TagDecl,
  /// replace the 'opaque' type we previously made for it if applicable.
  void UpdateCompletedType(const TagDecl *TD);
  
public:  // These are internal details of CGT that shouldn't be used externally.
  void DecodeArgumentTypes(const FunctionTypeProto &FTP, 
                           std::vector<const llvm::Type*> &ArgTys);

  /// addFieldInfo - Assign field number to field FD.
  void addFieldInfo(const FieldDecl *FD, unsigned No);

  /// addBitFieldInfo - Assign a start bit and a size to field FD.
  void addBitFieldInfo(const FieldDecl *FD, unsigned Begin, unsigned Size);

  /// getBitFieldInfo - Return the BitFieldInfo  that corresponds to the field
  /// FD.
  BitFieldInfo getBitFieldInfo(const FieldDecl *FD);

  /// ConvertTagDeclType - Lay out a tagged decl type like struct or union or
  /// enum.
  const llvm::Type *ConvertTagDeclType(const TagDecl *TD);
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
