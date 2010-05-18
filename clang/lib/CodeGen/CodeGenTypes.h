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
#include <vector>

#include "CGCall.h"
#include "GlobalDecl.h"

namespace llvm {
  class FunctionType;
  class Module;
  class OpaqueType;
  class PATypeHolder;
  class TargetData;
  class Type;
  class LLVMContext;
}

namespace clang {
  class ABIInfo;
  class ASTContext;
  template <typename> class CanQual;
  class CXXConstructorDecl;
  class CXXDestructorDecl;
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
  typedef CanQual<Type> CanQualType;

namespace CodeGen {
  class CGRecordLayout;

/// CodeGenTypes - This class organizes the cross-module state that is used
/// while lowering AST types to LLVM types.
class CodeGenTypes {
  ASTContext &Context;
  const TargetInfo &Target;
  llvm::Module& TheModule;
  const llvm::TargetData& TheTargetData;
  const ABIInfo& TheABIInfo;

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
  llvm::DenseMap<const Type*, CGRecordLayout *> CGRecordLayouts;

  /// FunctionInfos - Hold memoized CGFunctionInfo results.
  llvm::FoldingSet<CGFunctionInfo> FunctionInfos;

private:
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
  CodeGenTypes(ASTContext &Ctx, llvm::Module &M, const llvm::TargetData &TD,
               const ABIInfo &Info);
  ~CodeGenTypes();

  const llvm::TargetData &getTargetData() const { return TheTargetData; }
  const TargetInfo &getTarget() const { return Target; }
  ASTContext &getContext() const { return Context; }
  const ABIInfo &getABIInfo() const { return TheABIInfo; }
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

  const llvm::FunctionType *GetFunctionType(GlobalDecl GD);


  /// GetFunctionTypeForVTable - Get the LLVM function type for use in a vtable,
  /// given a CXXMethodDecl. If the method to has an incomplete return type, 
  /// and/or incomplete argument types, this will return the opaque type.
  const llvm::Type *GetFunctionTypeForVTable(const CXXMethodDecl *MD);
                                                     
  const CGRecordLayout &getCGRecordLayout(const RecordDecl*) const;

  /// UpdateCompletedType - When we find the full definition for a TagDecl,
  /// replace the 'opaque' type we previously made for it if applicable.
  void UpdateCompletedType(const TagDecl *TD);

  /// getFunctionInfo - Get the function info for the specified function decl.
  const CGFunctionInfo &getFunctionInfo(GlobalDecl GD);
  
  const CGFunctionInfo &getFunctionInfo(const FunctionDecl *FD);
  const CGFunctionInfo &getFunctionInfo(const CXXMethodDecl *MD);
  const CGFunctionInfo &getFunctionInfo(const ObjCMethodDecl *MD);
  const CGFunctionInfo &getFunctionInfo(const CXXConstructorDecl *D,
                                        CXXCtorType Type);
  const CGFunctionInfo &getFunctionInfo(const CXXDestructorDecl *D,
                                        CXXDtorType Type);

  const CGFunctionInfo &getFunctionInfo(const CallArgList &Args,
                                        const FunctionType *Ty) {
    return getFunctionInfo(Ty->getResultType(), Args,
                           Ty->getExtInfo());
  }
  const CGFunctionInfo &getFunctionInfo(CanQual<FunctionProtoType> Ty);
  const CGFunctionInfo &getFunctionInfo(CanQual<FunctionNoProtoType> Ty);

  // getFunctionInfo - Get the function info for a member function.
  const CGFunctionInfo &getFunctionInfo(const CXXRecordDecl *RD,
                                        const FunctionProtoType *FTP);
  
  /// getFunctionInfo - Get the function info for a function described by a
  /// return type and argument types. If the calling convention is not
  /// specified, the "C" calling convention will be used.
  const CGFunctionInfo &getFunctionInfo(QualType ResTy,
                                        const CallArgList &Args,
                                        const FunctionType::ExtInfo &Info);
  const CGFunctionInfo &getFunctionInfo(QualType ResTy,
                                        const FunctionArgList &Args,
                                        const FunctionType::ExtInfo &Info);

  /// Retrieves the ABI information for the given function signature.
  /// 
  /// \param ArgTys - must all actually be canonical as params
  const CGFunctionInfo &getFunctionInfo(CanQualType RetTy,
                               const llvm::SmallVectorImpl<CanQualType> &ArgTys,
                                        const FunctionType::ExtInfo &Info);

  /// \brief Compute a new LLVM record layout object for the given record.
  CGRecordLayout *ComputeRecordLayout(const RecordDecl *D);

public:  // These are internal details of CGT that shouldn't be used externally.
  /// ConvertTagDeclType - Lay out a tagged decl type like struct or union or
  /// enum.
  const llvm::Type *ConvertTagDeclType(const TagDecl *TD);

  /// GetExpandedTypes - Expand the type \arg Ty into the LLVM
  /// argument types it would be passed as on the provided vector \arg
  /// ArgTys. See ABIArgInfo::Expand.
  void GetExpandedTypes(QualType Ty, std::vector<const llvm::Type*> &ArgTys);
  
  /// ContainsPointerToDataMember - Return whether the given type contains a
  /// pointer to a data member.
  bool ContainsPointerToDataMember(QualType T);
  
  /// ContainsPointerToDataMember - Return whether the record decl contains a
  /// pointer to a data member.
  bool ContainsPointerToDataMember(const CXXRecordDecl *RD);
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
