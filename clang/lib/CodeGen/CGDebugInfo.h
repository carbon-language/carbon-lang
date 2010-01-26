//===--- CGDebugInfo.h - DebugInfo for LLVM CodeGen -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the source level debug info generator for llvm translation.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGDEBUGINFO_H
#define CLANG_CODEGEN_CGDEBUGINFO_H

#include "clang/AST/Type.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/Allocator.h"
#include <map>

#include "CGBuilder.h"

namespace llvm {
  class MDNode;
}

namespace clang {
  class VarDecl;
  class ObjCInterfaceDecl;

namespace CodeGen {
  class CodeGenModule;
  class CodeGenFunction;
  class GlobalDecl;

/// CGDebugInfo - This class gathers all debug information during compilation
/// and is responsible for emitting to llvm globals or pass directly to
/// the backend.
class CGDebugInfo {
  CodeGenModule &CGM;
  bool isMainCompileUnitCreated;
  llvm::DIFactory DebugFactory;

  SourceLocation CurLoc, PrevLoc;

  /// CompileUnitCache - Cache of previously constructed CompileUnits.
  llvm::DenseMap<unsigned, llvm::DICompileUnit> CompileUnitCache;

  /// TypeCache - Cache of previously constructed Types.
  // FIXME: Eliminate this map.  Be careful of iterator invalidation.
  std::map<void *, llvm::WeakVH> TypeCache;

  bool BlockLiteralGenericSet;
  llvm::DIType BlockLiteralGeneric;

  std::vector<llvm::TrackingVH<llvm::MDNode> > RegionStack;

  /// FunctionNames - This is a storage for function names that are
  /// constructed on demand. For example, C++ destructors, C++ operators etc..
  llvm::BumpPtrAllocator FunctionNames;

  llvm::DenseMap<const FunctionDecl *, llvm::WeakVH> SPCache;

  /// Helper functions for getOrCreateType.
  llvm::DIType CreateType(const BuiltinType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const ComplexType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateQualifiedType(QualType Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const TypedefType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const ObjCObjectPointerType *Ty,
                          llvm::DICompileUnit Unit);
  llvm::DIType CreateType(const PointerType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const BlockPointerType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const FunctionType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const TagType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const RecordType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const ObjCInterfaceType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const EnumType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const ArrayType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const LValueReferenceType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const MemberPointerType *Ty, llvm::DICompileUnit U);
  
  llvm::DIType CreatePointerLikeType(unsigned Tag,
                                     const Type *Ty, QualType PointeeTy,
                                     llvm::DICompileUnit U);
  
  llvm::DISubprogram CreateCXXMemberFunction(const CXXMethodDecl *Method,
                                             llvm::DICompileUnit Unit,
                                             llvm::DICompositeType &RecordTy);
  
  void CollectCXXMemberFunctions(const CXXRecordDecl *Decl,
                                 llvm::DICompileUnit U,
                                 llvm::SmallVectorImpl<llvm::DIDescriptor> &E,
                                 llvm::DICompositeType &T);
  void CollectCXXBases(const CXXRecordDecl *Decl,
                       llvm::DICompileUnit Unit,
                       llvm::SmallVectorImpl<llvm::DIDescriptor> &EltTys,
                       llvm::DICompositeType &RecordTy);


  void CollectRecordFields(const RecordDecl *Decl, llvm::DICompileUnit U,
                           llvm::SmallVectorImpl<llvm::DIDescriptor> &E);
public:
  CGDebugInfo(CodeGenModule &CGM);
  ~CGDebugInfo();

  /// setLocation - Update the current source location. If \arg loc is
  /// invalid it is ignored.
  void setLocation(SourceLocation Loc);

  /// EmitStopPoint - Emit a call to llvm.dbg.stoppoint to indicate a change of
  /// source line.
  void EmitStopPoint(llvm::Function *Fn, CGBuilderTy &Builder);

  /// EmitFunctionStart - Emit a call to llvm.dbg.function.start to indicate
  /// start of a new function.
  void EmitFunctionStart(GlobalDecl GD, QualType FnType,
                         llvm::Function *Fn, CGBuilderTy &Builder);

  /// EmitRegionStart - Emit a call to llvm.dbg.region.start to indicate start
  /// of a new block.
  void EmitRegionStart(llvm::Function *Fn, CGBuilderTy &Builder);

  /// EmitRegionEnd - Emit call to llvm.dbg.region.end to indicate end of a
  /// block.
  void EmitRegionEnd(llvm::Function *Fn, CGBuilderTy &Builder);

  /// EmitDeclareOfAutoVariable - Emit call to llvm.dbg.declare for an automatic
  /// variable declaration.
  void EmitDeclareOfAutoVariable(const VarDecl *Decl, llvm::Value *AI,
                                 CGBuilderTy &Builder);

  /// EmitDeclareOfBlockDeclRefVariable - Emit call to llvm.dbg.declare for an
  /// imported variable declaration in a block.
  void EmitDeclareOfBlockDeclRefVariable(const BlockDeclRefExpr *BDRE,
                                         llvm::Value *AI,
                                         CGBuilderTy &Builder,
                                         CodeGenFunction *CGF);

  /// EmitDeclareOfArgVariable - Emit call to llvm.dbg.declare for an argument
  /// variable declaration.
  void EmitDeclareOfArgVariable(const VarDecl *Decl, llvm::Value *AI,
                                CGBuilderTy &Builder);

  /// EmitGlobalVariable - Emit information about a global variable.
  void EmitGlobalVariable(llvm::GlobalVariable *GV, const VarDecl *Decl);

  /// EmitGlobalVariable - Emit information about an objective-c interface.
  void EmitGlobalVariable(llvm::GlobalVariable *GV, ObjCInterfaceDecl *Decl);

private:
  /// EmitDeclare - Emit call to llvm.dbg.declare for a variable declaration.
  void EmitDeclare(const VarDecl *decl, unsigned Tag, llvm::Value *AI,
                   CGBuilderTy &Builder);

  /// EmitDeclare - Emit call to llvm.dbg.declare for a variable declaration.
  void EmitDeclare(const BlockDeclRefExpr *BDRE, unsigned Tag, llvm::Value *AI,
                   CGBuilderTy &Builder, CodeGenFunction *CGF);

  /// getContext - Get context info for the decl.
  llvm::DIDescriptor getContext(const VarDecl *Decl,llvm::DIDescriptor &CU);

  /// getOrCreateCompileUnit - Get the compile unit from the cache or create a
  /// new one if necessary.
  llvm::DICompileUnit getOrCreateCompileUnit(SourceLocation Loc);

  /// getOrCreateType - Get the type from the cache or create a new type if
  /// necessary.
  llvm::DIType getOrCreateType(QualType Ty, llvm::DICompileUnit Unit);

  /// CreateTypeNode - Create type metadata for a source language type.
  llvm::DIType CreateTypeNode(QualType Ty, llvm::DICompileUnit Unit);

  /// getFunctionName - Get function name for the given FunctionDecl. If the
  /// name is constructred on demand (e.g. C++ destructor) then the name
  /// is stored on the side.
  llvm::StringRef getFunctionName(const FunctionDecl *FD);
};
} // namespace CodeGen
} // namespace clang


#endif
