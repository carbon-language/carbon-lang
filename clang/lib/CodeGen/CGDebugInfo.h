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

#include "CGBuilder.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/DIBuilder.h"
#include "llvm/DebugInfo.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ValueHandle.h"

namespace llvm {
  class MDNode;
}

namespace clang {
  class CXXMethodDecl;
  class VarDecl;
  class ObjCInterfaceDecl;
  class ObjCIvarDecl;
  class ClassTemplateSpecializationDecl;
  class GlobalDecl;
  class UsingDecl;

namespace CodeGen {
  class CodeGenModule;
  class CodeGenFunction;
  class CGBlockInfo;

/// CGDebugInfo - This class gathers all debug information during compilation
/// and is responsible for emitting to llvm globals or pass directly to
/// the backend.
class CGDebugInfo {
  CodeGenModule &CGM;
  CodeGenOptions::DebugInfoKind DebugKind;
  llvm::DIBuilder DBuilder;
  llvm::DICompileUnit TheCU;
  SourceLocation CurLoc, PrevLoc;
  llvm::DIType VTablePtrType;
  llvm::DIType ClassTy;
  llvm::DICompositeType ObjTy;
  llvm::DIType SelTy;
  llvm::DIType OCLImage1dDITy, OCLImage1dArrayDITy, OCLImage1dBufferDITy;
  llvm::DIType OCLImage2dDITy, OCLImage2dArrayDITy;
  llvm::DIType OCLImage3dDITy;
  llvm::DIType OCLEventDITy;

  /// TypeCache - Cache of previously constructed Types.
  llvm::DenseMap<void *, llvm::WeakVH> TypeCache;

  /// ObjCInterfaceCache - Cache of previously constructed interfaces
  /// which may change. Storing a pair of DIType and checksum.
  llvm::DenseMap<void *, std::pair<llvm::WeakVH, unsigned > >
    ObjCInterfaceCache;

  /// RetainedTypes - list of interfaces we want to keep even if orphaned.
  std::vector<void *> RetainedTypes;

  /// CompleteTypeCache - Cache of previously constructed complete RecordTypes.
  llvm::DenseMap<void *, llvm::WeakVH> CompletedTypeCache;

  /// ReplaceMap - Cache of forward declared types to RAUW at the end of
  /// compilation.
  std::vector<std::pair<void *, llvm::WeakVH> >ReplaceMap;

  bool BlockLiteralGenericSet;
  llvm::DIType BlockLiteralGeneric;

  // LexicalBlockStack - Keep track of our current nested lexical block.
  std::vector<llvm::TrackingVH<llvm::MDNode> > LexicalBlockStack;
  llvm::DenseMap<const Decl *, llvm::WeakVH> RegionMap;
  // FnBeginRegionCount - Keep track of LexicalBlockStack counter at the
  // beginning of a function. This is used to pop unbalanced regions at
  // the end of a function.
  std::vector<unsigned> FnBeginRegionCount;

  /// DebugInfoNames - This is a storage for names that are
  /// constructed on demand. For example, C++ destructors, C++ operators etc..
  llvm::BumpPtrAllocator DebugInfoNames;
  StringRef CWDName;

  llvm::DenseMap<const char *, llvm::WeakVH> DIFileCache;
  llvm::DenseMap<const FunctionDecl *, llvm::WeakVH> SPCache;
  /// \brief Cache declarations relevant to DW_TAG_imported_declarations (C++
  /// using declarations) that aren't covered by other more specific caches.
  llvm::DenseMap<const Decl *, llvm::WeakVH> DeclCache;
  llvm::DenseMap<const NamespaceDecl *, llvm::WeakVH> NameSpaceCache;
  llvm::DenseMap<const Decl *, llvm::WeakVH> StaticDataMemberCache;

  /// Helper functions for getOrCreateType.
  unsigned Checksum(const ObjCInterfaceDecl *InterfaceDecl);
  llvm::DIType CreateType(const BuiltinType *Ty);
  llvm::DIType CreateType(const ComplexType *Ty);
  llvm::DIType CreateQualifiedType(QualType Ty, llvm::DIFile F);
  llvm::DIType CreateType(const TypedefType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const ObjCObjectPointerType *Ty,
                          llvm::DIFile F);
  llvm::DIType CreateType(const PointerType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const BlockPointerType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const FunctionType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const RecordType *Ty);
  llvm::DIType CreateLimitedType(const RecordType *Ty);
  llvm::DIType CreateType(const ObjCInterfaceType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const ObjCObjectType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const VectorType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const ArrayType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const LValueReferenceType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const RValueReferenceType *Ty, llvm::DIFile Unit);
  llvm::DIType CreateType(const MemberPointerType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const AtomicType *Ty, llvm::DIFile F);
  llvm::DIType CreateEnumType(const EnumDecl *ED);
  llvm::DIType CreateSelfType(const QualType &QualTy, llvm::DIType Ty);
  llvm::DIType getTypeOrNull(const QualType);
  llvm::DIType getCompletedTypeOrNull(const QualType);
  llvm::DIType getOrCreateMethodType(const CXXMethodDecl *Method,
                                     llvm::DIFile F);
  llvm::DIType getOrCreateInstanceMethodType(
      QualType ThisPtr, const FunctionProtoType *Func, llvm::DIFile Unit);
  llvm::DIType getOrCreateFunctionType(const Decl *D, QualType FnType,
                                       llvm::DIFile F);
  llvm::DIType getOrCreateVTablePtrType(llvm::DIFile F);
  llvm::DINameSpace getOrCreateNameSpace(const NamespaceDecl *N);
  llvm::DIType CreatePointeeType(QualType PointeeTy, llvm::DIFile F);
  llvm::DIType CreatePointerLikeType(unsigned Tag,
                                     const Type *Ty, QualType PointeeTy,
                                     llvm::DIFile F);

  llvm::Value *getCachedInterfaceTypeOrNull(const QualType Ty);
  llvm::DIType getOrCreateStructPtrType(StringRef Name, llvm::DIType &Cache);

  llvm::DISubprogram CreateCXXMemberFunction(const CXXMethodDecl *Method,
                                             llvm::DIFile F,
                                             llvm::DIType RecordTy);

  void CollectCXXMemberFunctions(const CXXRecordDecl *Decl,
                                 llvm::DIFile F,
                                 SmallVectorImpl<llvm::Value *> &E,
                                 llvm::DIType T);

  void CollectCXXFriends(const CXXRecordDecl *Decl,
                       llvm::DIFile F,
                       SmallVectorImpl<llvm::Value *> &EltTys,
                       llvm::DIType RecordTy);

  void CollectCXXBases(const CXXRecordDecl *Decl,
                       llvm::DIFile F,
                       SmallVectorImpl<llvm::Value *> &EltTys,
                       llvm::DIType RecordTy);

  llvm::DIArray
  CollectTemplateParams(const TemplateParameterList *TPList,
                        const TemplateArgumentList &TAList,
                        llvm::DIFile Unit);
  llvm::DIArray
  CollectFunctionTemplateParams(const FunctionDecl *FD, llvm::DIFile Unit);
  llvm::DIArray
  CollectCXXTemplateParams(const ClassTemplateSpecializationDecl *TS,
                           llvm::DIFile F);

  llvm::DIType createFieldType(StringRef name, QualType type,
                               uint64_t sizeInBitsOverride, SourceLocation loc,
                               AccessSpecifier AS, uint64_t offsetInBits,
                               llvm::DIFile tunit,
                               llvm::DIDescriptor scope);

  // Helpers for collecting fields of a record.
  void CollectRecordLambdaFields(const CXXRecordDecl *CXXDecl,
                                 SmallVectorImpl<llvm::Value *> &E,
                                 llvm::DIType RecordTy);
  void CollectRecordStaticField(const VarDecl *Var,
                                SmallVectorImpl<llvm::Value *> &E,
                                llvm::DIType RecordTy);
  void CollectRecordNormalField(const FieldDecl *Field, uint64_t OffsetInBits,
                                llvm::DIFile F,
                                SmallVectorImpl<llvm::Value *> &E,
                                llvm::DIType RecordTy);
  void CollectRecordFields(const RecordDecl *Decl, llvm::DIFile F,
                           SmallVectorImpl<llvm::Value *> &E,
                           llvm::DIType RecordTy);

  void CollectVTableInfo(const CXXRecordDecl *Decl,
                         llvm::DIFile F,
                         SmallVectorImpl<llvm::Value *> &EltTys);

  // CreateLexicalBlock - Create a new lexical block node and push it on
  // the stack.
  void CreateLexicalBlock(SourceLocation Loc);

public:
  CGDebugInfo(CodeGenModule &CGM);
  ~CGDebugInfo();

  void finalize();

  /// setLocation - Update the current source location. If \arg loc is
  /// invalid it is ignored.
  void setLocation(SourceLocation Loc);

  /// getLocation - Return the current source location.
  SourceLocation getLocation() const { return CurLoc; }

  /// EmitLocation - Emit metadata to indicate a change in line/column
  /// information in the source file.
  /// \param ForceColumnInfo  Assume DebugColumnInfo option is true.
  void EmitLocation(CGBuilderTy &Builder, SourceLocation Loc,
                    bool ForceColumnInfo = false);

  /// EmitFunctionStart - Emit a call to llvm.dbg.function.start to indicate
  /// start of a new function.
  void EmitFunctionStart(GlobalDecl GD, QualType FnType,
                         llvm::Function *Fn, CGBuilderTy &Builder);

  /// EmitFunctionEnd - Constructs the debug code for exiting a function.
  void EmitFunctionEnd(CGBuilderTy &Builder);

  /// EmitLexicalBlockStart - Emit metadata to indicate the beginning of a
  /// new lexical block and push the block onto the stack.
  void EmitLexicalBlockStart(CGBuilderTy &Builder, SourceLocation Loc);

  /// EmitLexicalBlockEnd - Emit metadata to indicate the end of a new lexical
  /// block and pop the current block.
  void EmitLexicalBlockEnd(CGBuilderTy &Builder, SourceLocation Loc);

  /// EmitDeclareOfAutoVariable - Emit call to llvm.dbg.declare for an automatic
  /// variable declaration.
  void EmitDeclareOfAutoVariable(const VarDecl *Decl, llvm::Value *AI,
                                 CGBuilderTy &Builder);

  /// EmitDeclareOfBlockDeclRefVariable - Emit call to llvm.dbg.declare for an
  /// imported variable declaration in a block.
  void EmitDeclareOfBlockDeclRefVariable(const VarDecl *variable,
                                         llvm::Value *storage,
                                         CGBuilderTy &Builder,
                                         const CGBlockInfo &blockInfo);

  /// EmitDeclareOfArgVariable - Emit call to llvm.dbg.declare for an argument
  /// variable declaration.
  void EmitDeclareOfArgVariable(const VarDecl *Decl, llvm::Value *AI,
                                unsigned ArgNo, CGBuilderTy &Builder);

  /// EmitDeclareOfBlockLiteralArgVariable - Emit call to
  /// llvm.dbg.declare for the block-literal argument to a block
  /// invocation function.
  void EmitDeclareOfBlockLiteralArgVariable(const CGBlockInfo &block,
                                            llvm::Value *Arg,
                                            llvm::Value *LocalAddr,
                                            CGBuilderTy &Builder);

  /// EmitGlobalVariable - Emit information about a global variable.
  void EmitGlobalVariable(llvm::GlobalVariable *GV, const VarDecl *Decl);

  /// EmitGlobalVariable - Emit information about an objective-c interface.
  void EmitGlobalVariable(llvm::GlobalVariable *GV, ObjCInterfaceDecl *Decl);

  /// EmitGlobalVariable - Emit global variable's debug info.
  void EmitGlobalVariable(const ValueDecl *VD, llvm::Constant *Init);

  /// \brief - Emit C++ using directive.
  void EmitUsingDirective(const UsingDirectiveDecl &UD);

  /// \brief - Emit C++ using declaration.
  void EmitUsingDecl(const UsingDecl &UD);

  /// getOrCreateRecordType - Emit record type's standalone debug info.
  llvm::DIType getOrCreateRecordType(QualType Ty, SourceLocation L);

  /// getOrCreateInterfaceType - Emit an objective c interface type standalone
  /// debug info.
  llvm::DIType getOrCreateInterfaceType(QualType Ty,
                                        SourceLocation Loc);

private:
  /// EmitDeclare - Emit call to llvm.dbg.declare for a variable declaration.
  void EmitDeclare(const VarDecl *decl, unsigned Tag, llvm::Value *AI,
                   unsigned ArgNo, CGBuilderTy &Builder);

  // EmitTypeForVarWithBlocksAttr - Build up structure info for the byref.
  // See BuildByRefType.
  llvm::DIType EmitTypeForVarWithBlocksAttr(const VarDecl *VD,
                                            uint64_t *OffSet);

  /// getContextDescriptor - Get context info for the decl.
  llvm::DIScope getContextDescriptor(const Decl *Decl);

  llvm::DIScope getCurrentContextDescriptor(const Decl *Decl);

  /// createRecordFwdDecl - Create a forward decl for a RecordType in a given
  /// context.
  llvm::DIType createRecordFwdDecl(const RecordDecl *, llvm::DIDescriptor);

  /// createContextChain - Create a set of decls for the context chain.
  llvm::DIDescriptor createContextChain(const Decl *Decl);

  /// getCurrentDirname - Return current directory name.
  StringRef getCurrentDirname();

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

  /// getOrCreateLimitedType - Get the type from the cache or create a new
  /// partial type if necessary.
  llvm::DIType getOrCreateLimitedType(QualType Ty, llvm::DIFile F);

  /// CreateTypeNode - Create type metadata for a source language type.
  llvm::DIType CreateTypeNode(QualType Ty, llvm::DIFile F);

  /// getObjCInterfaceDecl - return the underlying ObjCInterfaceDecl
  /// if Ty is an ObjCInterface or a pointer to one.
  ObjCInterfaceDecl* getObjCInterfaceDecl(QualType Ty);

  /// CreateLimitedTypeNode - Create type metadata for a source language
  /// type, but only partial types for records.
  llvm::DIType CreateLimitedTypeNode(QualType Ty, llvm::DIFile F);

  /// CreateMemberType - Create new member and increase Offset by FType's size.
  llvm::DIType CreateMemberType(llvm::DIFile Unit, QualType FType,
                                StringRef Name, uint64_t *Offset);

  /// \brief Retrieve the DIDescriptor, if any, for the canonical form of this
  /// declaration.
  llvm::DIDescriptor getDeclarationOrDefinition(const Decl *D);

  /// getFunctionDeclaration - Return debug info descriptor to describe method
  /// declaration for the given method definition.
  llvm::DISubprogram getFunctionDeclaration(const Decl *D);

  /// getStaticDataMemberDeclaration - Return debug info descriptor to
  /// describe in-class static data member declaration for the given
  /// out-of-class definition.
  llvm::DIDerivedType getStaticDataMemberDeclaration(const Decl *D);

  /// getFunctionName - Get function name for the given FunctionDecl. If the
  /// name is constructred on demand (e.g. C++ destructor) then the name
  /// is stored on the side.
  StringRef getFunctionName(const FunctionDecl *FD);

  /// getObjCMethodName - Returns the unmangled name of an Objective-C method.
  /// This is the display name for the debugging info.
  StringRef getObjCMethodName(const ObjCMethodDecl *FD);

  /// getSelectorName - Return selector name. This is used for debugging
  /// info.
  StringRef getSelectorName(Selector S);

  /// getClassName - Get class name including template argument list.
  StringRef getClassName(const RecordDecl *RD);

  /// getVTableName - Get vtable name for the given Class.
  StringRef getVTableName(const CXXRecordDecl *Decl);

  /// getLineNumber - Get line number for the location. If location is invalid
  /// then use current location.
  unsigned getLineNumber(SourceLocation Loc);

  /// getColumnNumber - Get column number for the location. If location is
  /// invalid then use current location.
  /// \param Force  Assume DebugColumnInfo option is true.
  unsigned getColumnNumber(SourceLocation Loc, bool Force=false);
};
} // namespace CodeGen
} // namespace clang


#endif
