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
#include "llvm/Analysis/DIBuilder.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/Allocator.h"

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
  llvm::DIBuilder DBuilder;
  llvm::DICompileUnit TheCU;
  SourceLocation CurLoc, PrevLoc;
  llvm::DIType VTablePtrType;
  
  /// TypeCache - Cache of previously constructed Types.
  llvm::DenseMap<void *, llvm::WeakVH> TypeCache;

  bool BlockLiteralGenericSet;
  llvm::DIType BlockLiteralGeneric;

  std::vector<llvm::TrackingVH<llvm::MDNode> > RegionStack;
  llvm::DenseMap<const Decl *, llvm::WeakVH> RegionMap;
  // FnBeginRegionCount - Keep track of RegionStack counter at the beginning
  // of a function. This is used to pop unbalanced regions at the end of a
  // function.
  std::vector<unsigned> FnBeginRegionCount;

  /// LineDirectiveFiles - This stack is used to keep track of 
  /// scopes introduced by #line directives.
  std::vector<const char *> LineDirectiveFiles;

  /// DebugInfoNames - This is a storage for names that are
  /// constructed on demand. For example, C++ destructors, C++ operators etc..
  llvm::BumpPtrAllocator DebugInfoNames;
  llvm::StringRef CWDName;

  llvm::DenseMap<const char *, llvm::WeakVH> DIFileCache;
  llvm::DenseMap<const FunctionDecl *, llvm::WeakVH> SPCache;
  llvm::DenseMap<const NamespaceDecl *, llvm::WeakVH> NameSpaceCache;

  /// Helper functions for getOrCreateType.
  llvm::DIType CreateType(const BuiltinType *Ty);
  llvm::DIType CreateType(const ComplexType *Ty);
  llvm::DIType CreateQualifiedType(QualType Ty, llvm::DIFile F);
  llvm::DIType CreateType(const TypedefType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const ObjCObjectPointerType *Ty,
                          llvm::DIFile F);
  llvm::DIType CreateType(const PointerType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const BlockPointerType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const FunctionType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const TagType *Ty);
  llvm::DIType CreateType(const RecordType *Ty);
  llvm::DIType CreateType(const ObjCInterfaceType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const ObjCObjectType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const VectorType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const ArrayType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const LValueReferenceType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const RValueReferenceType *Ty, llvm::DIFile Unit);
  llvm::DIType CreateType(const MemberPointerType *Ty, llvm::DIFile F);
  llvm::DIType CreateEnumType(const EnumDecl *ED);
  llvm::DIType getOrCreateMethodType(const CXXMethodDecl *Method,
                                     llvm::DIFile F);
  llvm::DIType getOrCreateVTablePtrType(llvm::DIFile F);
  llvm::DINameSpace getOrCreateNameSpace(const NamespaceDecl *N);
  llvm::DIType CreatePointeeType(QualType PointeeTy, llvm::DIFile F);
  llvm::DIType CreatePointerLikeType(unsigned Tag,
                                     const Type *Ty, QualType PointeeTy,
                                     llvm::DIFile F);
  
  llvm::DISubprogram CreateCXXMemberFunction(const CXXMethodDecl *Method,
                                             llvm::DIFile F,
                                             llvm::DIType RecordTy);
  
  void CollectCXXMemberFunctions(const CXXRecordDecl *Decl,
                                 llvm::DIFile F,
                                 llvm::SmallVectorImpl<llvm::Value *> &E,
                                 llvm::DIType T);

  void CollectCXXFriends(const CXXRecordDecl *Decl,
                       llvm::DIFile F,
                       llvm::SmallVectorImpl<llvm::Value *> &EltTys,
                       llvm::DIType RecordTy);

  void CollectCXXBases(const CXXRecordDecl *Decl,
                       llvm::DIFile F,
                       llvm::SmallVectorImpl<llvm::Value *> &EltTys,
                       llvm::DIType RecordTy);


  void CollectRecordFields(const RecordDecl *Decl, llvm::DIFile F,
                           llvm::SmallVectorImpl<llvm::Value *> &E);

  void CollectVTableInfo(const CXXRecordDecl *Decl,
                         llvm::DIFile F,
                         llvm::SmallVectorImpl<llvm::Value *> &EltTys);

public:
  CGDebugInfo(CodeGenModule &CGM);
  ~CGDebugInfo();

  /// setLocation - Update the current source location. If \arg loc is
  /// invalid it is ignored.
  void setLocation(SourceLocation Loc);

  /// EmitStopPoint - Emit a call to llvm.dbg.stoppoint to indicate a change of
  /// source line.
  void EmitStopPoint(CGBuilderTy &Builder);

  /// EmitFunctionStart - Emit a call to llvm.dbg.function.start to indicate
  /// start of a new function.
  void EmitFunctionStart(GlobalDecl GD, QualType FnType,
                         llvm::Function *Fn, CGBuilderTy &Builder);

  /// EmitFunctionEnd - Constructs the debug code for exiting a function.
  void EmitFunctionEnd(CGBuilderTy &Builder);

  /// UpdateLineDirectiveRegion - Update region stack only if #line directive
  /// has introduced scope change.
  void UpdateLineDirectiveRegion(CGBuilderTy &Builder);

  /// EmitRegionStart - Emit a call to llvm.dbg.region.start to indicate start
  /// of a new block.
  void EmitRegionStart(CGBuilderTy &Builder);

  /// EmitRegionEnd - Emit call to llvm.dbg.region.end to indicate end of a
  /// block.
  void EmitRegionEnd(CGBuilderTy &Builder);

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

  /// EmitGlobalVariable - Emit global variable's debug info.
  void EmitGlobalVariable(const ValueDecl *VD, llvm::Constant *Init);

  /// getOrCreateRecordType - Emit record type's standalone debug info. 
  llvm::DIType getOrCreateRecordType(QualType Ty, SourceLocation L);
private:
  /// EmitDeclare - Emit call to llvm.dbg.declare for a variable declaration.
  void EmitDeclare(const VarDecl *decl, unsigned Tag, llvm::Value *AI,
                   CGBuilderTy &Builder);

  /// EmitDeclare - Emit call to llvm.dbg.declare for a variable declaration.
  void EmitDeclare(const BlockDeclRefExpr *BDRE, unsigned Tag, llvm::Value *AI,
                   CGBuilderTy &Builder, CodeGenFunction *CGF);

  // EmitTypeForVarWithBlocksAttr - Build up structure info for the byref.  
  // See BuildByRefType.
  llvm::DIType EmitTypeForVarWithBlocksAttr(const ValueDecl *VD, 
                                            uint64_t *OffSet);

  /// getContextDescriptor - Get context info for the decl.
  llvm::DIDescriptor getContextDescriptor(const Decl *Decl);

  /// getCurrentDirname - Return current directory name.
  llvm::StringRef getCurrentDirname();

  /// CreateCompileUnit - Create new compile unit.
  void CreateCompileUnit();

  /// getOrCreateFile - Get the file debug info descriptor for the input 
  /// location.
  llvm::DIFile getOrCreateFile(SourceLocation Loc);

  /// getOrCreateMainFile - Get the file info for main compile unit.
  llvm::DIFile getOrCreateMainFile();

  /// getOrCreateType - Get the type from the cache or create a new type if
  /// necessary.
  llvm::DIType getOrCreateType(QualType Ty, llvm::DIFile F);

  /// CreateTypeNode - Create type metadata for a source language type.
  llvm::DIType CreateTypeNode(QualType Ty, llvm::DIFile F);

  /// CreateMemberType - Create new member and increase Offset by FType's size.
  llvm::DIType CreateMemberType(llvm::DIFile Unit, QualType FType,
                                llvm::StringRef Name, uint64_t *Offset);

  /// getFunctionName - Get function name for the given FunctionDecl. If the
  /// name is constructred on demand (e.g. C++ destructor) then the name
  /// is stored on the side.
  llvm::StringRef getFunctionName(const FunctionDecl *FD);

  /// getObjCMethodName - Returns the unmangled name of an Objective-C method.
  /// This is the display name for the debugging info.  
  llvm::StringRef getObjCMethodName(const ObjCMethodDecl *FD);

  /// getClassName - Get class name including template argument list.
  llvm::StringRef getClassName(RecordDecl *RD);

  /// getVTableName - Get vtable name for the given Class.
  llvm::StringRef getVTableName(const CXXRecordDecl *Decl);

  /// getLineNumber - Get line number for the location. If location is invalid
  /// then use current location.
  unsigned getLineNumber(SourceLocation Loc);

  /// getColumnNumber - Get column number for the location. If location is 
  /// invalid then use current location.
  unsigned getColumnNumber(SourceLocation Loc);
};
} // namespace CodeGen
} // namespace clang


#endif
