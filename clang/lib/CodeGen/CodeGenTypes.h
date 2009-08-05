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

#include "llvm/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include <vector>

#include "CGCall.h"

namespace llvm {
  class FunctionType;
  class Module;
  class OpaqueType;
  class PATypeHolder;
  class TargetData;
  class Type;
  struct LLVMContext;
}

namespace clang {
  class ABIInfo;
  class ASTContext;
  class CXXMethodDecl;
  class FieldDecl;
  class FunctionProtoType;
  class ObjCInterfaceDecl;
  class ObjCIvarDecl;
  class PointerType;
  class QualType;
  class RecordDecl;
  class TagDecl;
  class TargetInfo;
  class Type;

namespace CodeGen {
  class CodeGenTypes;

  /// CGRecordLayout - This class handles struct and union layout info while 
  /// lowering AST types to LLVM types.
  class CGRecordLayout {
    CGRecordLayout(); // DO NOT IMPLEMENT
  public:
    CGRecordLayout(const llvm::Type *T, 
                   const llvm::SmallSet<unsigned, 8> &PF) 
      : STy(T), PaddingFields(PF) {
      // FIXME : Collect info about fields that requires adjustments 
      // (i.e. fields that do not directly map to llvm struct fields.)
    }

    /// getLLVMType - Return llvm type associated with this record.
    const llvm::Type *getLLVMType() const {
      return STy;
    }

    bool isPaddingField(unsigned No) const {
      return PaddingFields.count(No) != 0;
    }

    unsigned getNumPaddingFields() {
      return PaddingFields.size();
    }

  private:
    const llvm::Type *STy;
    llvm::SmallSet<unsigned, 8> PaddingFields;
  };
  
/// CodeGenTypes - This class organizes the cross-module state that is used
/// while lowering AST types to LLVM types.
class CodeGenTypes {
  ASTContext &Context;
  TargetInfo &Target;
  llvm::Module& TheModule;
  const llvm::TargetData& TheTargetData;
  mutable const ABIInfo* TheABIInfo;
  
  llvm::SmallVector<std::pair<QualType,
                              llvm::OpaqueType *>, 8>  PointersToResolve;

  llvm::DenseMap<const Type*, llvm::PATypeHolder> TagDeclTypes;

  llvm::DenseMap<const Type*, llvm::PATypeHolder> FunctionTypes;

  /// The opaque type map for Objective-C interfaces. All direct
  /// manipulation is done by the runtime interfaces, which are
  /// responsible for coercing to the appropriate type; these opaque
  /// types are never refined.
  llvm::DenseMap<const ObjCInterfaceType*, const llvm::Type *> InterfaceTypes;

  /// CGRecordLayouts - This maps llvm struct type with corresponding 
  /// record layout info. 
  /// FIXME : If CGRecordLayout is less than 16 bytes then use 
  /// inline it in the map.
  llvm::DenseMap<const Type*, CGRecordLayout *> CGRecordLayouts;

  /// FieldInfo - This maps struct field with corresponding llvm struct type
  /// field no. This info is populated by record organizer.
  llvm::DenseMap<const FieldDecl *, unsigned> FieldInfo;

  /// FunctionInfos - Hold memoized CGFunctionInfo results.
  llvm::FoldingSet<CGFunctionInfo> FunctionInfos;

public:
  struct BitFieldInfo {
    BitFieldInfo(unsigned FieldNo, 
                 unsigned Start, 
                 unsigned Size)
      : FieldNo(FieldNo), Start(Start), Size(Size) {}

    unsigned FieldNo;
    unsigned Start;
    unsigned Size;
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
  const ABIInfo &getABIInfo() const;
  llvm::LLVMContext &getLLVMContext() { return TheModule.getContext(); }

  /// ConvertType - Convert type T into a llvm::Type.  
  const llvm::Type *ConvertType(QualType T);
  const llvm::Type *ConvertTypeRecursive(QualType T);
  
  /// ConvertTypeForMem - Convert type T into a llvm::Type.  This differs from
  /// ConvertType in that it is used to convert to the memory representation for
  /// a type.  For example, the scalar representation for _Bool is i1, but the
  /// memory representation is usually i8 or i32, depending on the target.
  const llvm::Type *ConvertTypeForMem(QualType T);
  const llvm::Type *ConvertTypeForMemRecursive(QualType T);

  /// GetFunctionType - Get the LLVM function type for \arg Info.
  const llvm::FunctionType *GetFunctionType(const CGFunctionInfo &Info,
                                            bool IsVariadic);
  
  const CGRecordLayout *getCGRecordLayout(const TagDecl*) const;
  
  /// getLLVMFieldNo - Return llvm::StructType element number
  /// that corresponds to the field FD.
  unsigned getLLVMFieldNo(const FieldDecl *FD);
  
  /// UpdateCompletedType - When we find the full definition for a TagDecl,
  /// replace the 'opaque' type we previously made for it if applicable.
  void UpdateCompletedType(const TagDecl *TD);

  /// getFunctionInfo - Get the CGFunctionInfo for this function signature.
  const CGFunctionInfo &getFunctionInfo(QualType RetTy, 
                                        const llvm::SmallVector<QualType,16> 
                                        &ArgTys);

  const CGFunctionInfo &getFunctionInfo(const FunctionNoProtoType *FTNP);
  const CGFunctionInfo &getFunctionInfo(const FunctionProtoType *FTP);
  const CGFunctionInfo &getFunctionInfo(const FunctionDecl *FD);
  const CGFunctionInfo &getFunctionInfo(const CXXMethodDecl *MD);
  const CGFunctionInfo &getFunctionInfo(const ObjCMethodDecl *MD);
  const CGFunctionInfo &getFunctionInfo(QualType ResTy, 
                                        const CallArgList &Args);
public:
  const CGFunctionInfo &getFunctionInfo(QualType ResTy, 
                                        const FunctionArgList &Args);
  
public:  // These are internal details of CGT that shouldn't be used externally.
  /// addFieldInfo - Assign field number to field FD.
  void addFieldInfo(const FieldDecl *FD, unsigned FieldNo);

  /// addBitFieldInfo - Assign a start bit and a size to field FD.
  void addBitFieldInfo(const FieldDecl *FD, unsigned FieldNo,
                       unsigned Start, unsigned Size);

  /// getBitFieldInfo - Return the BitFieldInfo  that corresponds to the field
  /// FD.
  BitFieldInfo getBitFieldInfo(const FieldDecl *FD);

  /// ConvertTagDeclType - Lay out a tagged decl type like struct or union or
  /// enum.
  const llvm::Type *ConvertTagDeclType(const TagDecl *TD);

  /// GetExpandedTypes - Expand the type \arg Ty into the LLVM
  /// argument types it would be passed as on the provided vector \arg
  /// ArgTys. See ABIArgInfo::Expand.
  void GetExpandedTypes(QualType Ty, std::vector<const llvm::Type*> &ArgTys);
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
