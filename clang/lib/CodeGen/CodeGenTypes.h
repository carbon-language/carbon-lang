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

#include "CGCall.h"
#include "clang/AST/GlobalDecl.h"
#include "llvm/Module.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>

namespace llvm {
  class FunctionType;
  class Module;
  class TargetData;
  class Type;
  class LLVMContext;
  class StructType;
}

namespace clang {
  class ABIInfo;
  class ASTContext;
  template <typename> class CanQual;
  class CXXConstructorDecl;
  class CXXDestructorDecl;
  class CXXMethodDecl;
  class CodeGenOptions;
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
  class CGCXXABI;
  class CGRecordLayout;

/// CodeGenTypes - This class organizes the cross-module state that is used
/// while lowering AST types to LLVM types.
class CodeGenTypes {
  ASTContext &Context;
  const TargetInfo &Target;
  llvm::Module &TheModule;
  const llvm::TargetData &TheTargetData;
  const ABIInfo &TheABIInfo;
  CGCXXABI &TheCXXABI;
  const CodeGenOptions &CodeGenOpts;

  /// The opaque type map for Objective-C interfaces. All direct
  /// manipulation is done by the runtime interfaces, which are
  /// responsible for coercing to the appropriate type; these opaque
  /// types are never refined.
  llvm::DenseMap<const ObjCInterfaceType*, llvm::Type *> InterfaceTypes;

  /// CGRecordLayouts - This maps llvm struct type with corresponding
  /// record layout info.
  llvm::DenseMap<const Type*, CGRecordLayout *> CGRecordLayouts;

  /// RecordDeclTypes - This contains the LLVM IR type for any converted
  /// RecordDecl.
  llvm::DenseMap<const Type*, llvm::StructType *> RecordDeclTypes;
  
  /// FunctionInfos - Hold memoized CGFunctionInfo results.
  llvm::FoldingSet<CGFunctionInfo> FunctionInfos;

  /// RecordsBeingLaidOut - This set keeps track of records that we're currently
  /// converting to an IR type.  For example, when converting:
  /// struct A { struct B { int x; } } when processing 'x', the 'A' and 'B'
  /// types will be in this set.
  llvm::SmallPtrSet<const Type*, 4> RecordsBeingLaidOut;
  
  llvm::SmallPtrSet<const CGFunctionInfo*, 4> FunctionsBeingProcessed;
  
  /// SkippedLayout - True if we didn't layout a function due to a being inside
  /// a recursive struct conversion, set this to true.
  bool SkippedLayout;

  llvm::SmallVector<const RecordDecl *, 8> DeferredRecords;
  
private:
  /// TypeCache - This map keeps cache of llvm::Types
  /// and maps llvm::Types to corresponding clang::Type.
  llvm::DenseMap<const Type *, llvm::Type *> TypeCache;

public:
  CodeGenTypes(ASTContext &Ctx, llvm::Module &M, const llvm::TargetData &TD,
               const ABIInfo &Info, CGCXXABI &CXXABI,
               const CodeGenOptions &Opts);
  ~CodeGenTypes();

  const llvm::TargetData &getTargetData() const { return TheTargetData; }
  const TargetInfo &getTarget() const { return Target; }
  ASTContext &getContext() const { return Context; }
  const ABIInfo &getABIInfo() const { return TheABIInfo; }
  const CodeGenOptions &getCodeGenOpts() const { return CodeGenOpts; }
  CGCXXABI &getCXXABI() const { return TheCXXABI; }
  llvm::LLVMContext &getLLVMContext() { return TheModule.getContext(); }

  /// ConvertType - Convert type T into a llvm::Type.
  llvm::Type *ConvertType(QualType T);

  /// ConvertTypeForMem - Convert type T into a llvm::Type.  This differs from
  /// ConvertType in that it is used to convert to the memory representation for
  /// a type.  For example, the scalar representation for _Bool is i1, but the
  /// memory representation is usually i8 or i32, depending on the target.
  llvm::Type *ConvertTypeForMem(QualType T);

  /// GetFunctionType - Get the LLVM function type for \arg Info.
  llvm::FunctionType *GetFunctionType(const CGFunctionInfo &Info,
                                      bool IsVariadic);

  llvm::FunctionType *GetFunctionType(GlobalDecl GD);

  /// isFuncTypeConvertible - Utility to check whether a function type can
  /// be converted to an LLVM type (i.e. doesn't depend on an incomplete tag
  /// type).
  bool isFuncTypeConvertible(const FunctionType *FT);
  bool isFuncTypeArgumentConvertible(QualType Ty);
  
  /// GetFunctionTypeForVTable - Get the LLVM function type for use in a vtable,
  /// given a CXXMethodDecl. If the method to has an incomplete return type,
  /// and/or incomplete argument types, this will return the opaque type.
  llvm::Type *GetFunctionTypeForVTable(GlobalDecl GD);

  const CGRecordLayout &getCGRecordLayout(const RecordDecl*);

  /// UpdateCompletedType - When we find the full definition for a TagDecl,
  /// replace the 'opaque' type we previously made for it if applicable.
  void UpdateCompletedType(const TagDecl *TD);

  /// getNullaryFunctionInfo - Get the function info for a void()
  /// function with standard CC.
  const CGFunctionInfo &getNullaryFunctionInfo();

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

  /// getFunctionInfo - Get the function info for a member function of
  /// the given type.  This is used for calls through member function
  /// pointers.
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
  CGRecordLayout *ComputeRecordLayout(const RecordDecl *D,
                                      llvm::StructType *Ty);

  /// addRecordTypeName - Compute a name from the given record decl with an
  /// optional suffix and name the given LLVM type using it.
  void addRecordTypeName(const RecordDecl *RD, llvm::StructType *Ty,
                         llvm::StringRef suffix);
  

public:  // These are internal details of CGT that shouldn't be used externally.
  /// ConvertRecordDeclType - Lay out a tagged decl type like struct or union.
  llvm::StructType *ConvertRecordDeclType(const RecordDecl *TD);

  /// GetExpandedTypes - Expand the type \arg Ty into the LLVM
  /// argument types it would be passed as on the provided vector \arg
  /// ArgTys. See ABIArgInfo::Expand.
  void GetExpandedTypes(QualType type,
                        llvm::SmallVectorImpl<llvm::Type*> &expanded);

  /// IsZeroInitializable - Return whether a type can be
  /// zero-initialized (in the C++ sense) with an LLVM zeroinitializer.
  bool isZeroInitializable(QualType T);

  /// IsZeroInitializable - Return whether a record type can be
  /// zero-initialized (in the C++ sense) with an LLVM zeroinitializer.
  bool isZeroInitializable(const CXXRecordDecl *RD);
  
  bool isRecordLayoutComplete(const Type *Ty) const;
  bool noRecordsBeingLaidOut() const {
    return RecordsBeingLaidOut.empty();
  }
  bool isRecordBeingLaidOut(const Type *Ty) const {
    return RecordsBeingLaidOut.count(Ty);
  }
                            
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
