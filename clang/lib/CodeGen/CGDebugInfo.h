//===--- CGDebugInfo.h - DebugInfo for LLVM CodeGen -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the source-level debug info generator for llvm translation.
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
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Allocator.h"

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
  friend class ArtificialLocation;
  friend class SaveAndRestoreLocation;
  CodeGenModule &CGM;
  const CodeGenOptions::DebugInfoKind DebugKind;
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
  llvm::DIType BlockLiteralGeneric;

  /// TypeCache - Cache of previously constructed Types.
  llvm::DenseMap<const void *, llvm::WeakVH> TypeCache;

  struct ObjCInterfaceCacheEntry {
    const ObjCInterfaceType *Type;
    llvm::DIType Decl;
    llvm::DIFile Unit;
    ObjCInterfaceCacheEntry(const ObjCInterfaceType *Type, llvm::DIType Decl,
                            llvm::DIFile Unit)
        : Type(Type), Decl(Decl), Unit(Unit) {}
  };

  /// ObjCInterfaceCache - Cache of previously constructed interfaces
  /// which may change.
  llvm::SmallVector<ObjCInterfaceCacheEntry, 32> ObjCInterfaceCache;

  /// RetainedTypes - list of interfaces we want to keep even if orphaned.
  std::vector<void *> RetainedTypes;

  /// ReplaceMap - Cache of forward declared types to RAUW at the end of
  /// compilation.
  std::vector<std::pair<const TagType *, llvm::WeakVH>> ReplaceMap;

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
  llvm::DenseMap<const NamespaceAliasDecl *, llvm::WeakVH> NamespaceAliasCache;
  llvm::DenseMap<const Decl *, llvm::WeakVH> StaticDataMemberCache;

  /// Helper functions for getOrCreateType.
  unsigned Checksum(const ObjCInterfaceDecl *InterfaceDecl);
  llvm::DIType CreateType(const BuiltinType *Ty);
  llvm::DIType CreateType(const ComplexType *Ty);
  llvm::DIType CreateQualifiedType(QualType Ty, llvm::DIFile Fg);
  llvm::DIType CreateType(const TypedefType *Ty, llvm::DIFile Fg);
  llvm::DIType CreateType(const TemplateSpecializationType *Ty, llvm::DIFile Fg);
  llvm::DIType CreateType(const ObjCObjectPointerType *Ty,
                          llvm::DIFile F);
  llvm::DIType CreateType(const PointerType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const BlockPointerType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const FunctionType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const RecordType *Tyg);
  llvm::DIType CreateTypeDefinition(const RecordType *Ty);
  llvm::DICompositeType CreateLimitedType(const RecordType *Ty);
  void CollectContainingType(const CXXRecordDecl *RD, llvm::DICompositeType CT);
  llvm::DIType CreateType(const ObjCInterfaceType *Ty, llvm::DIFile F);
  llvm::DIType CreateTypeDefinition(const ObjCInterfaceType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const ObjCObjectType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const VectorType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const ArrayType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const LValueReferenceType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const RValueReferenceType *Ty, llvm::DIFile Unit);
  llvm::DIType CreateType(const MemberPointerType *Ty, llvm::DIFile F);
  llvm::DIType CreateType(const AtomicType *Ty, llvm::DIFile F);
  llvm::DIType CreateEnumType(const EnumType *Ty);
  llvm::DIType CreateTypeDefinition(const EnumType *Ty);
  llvm::DIType CreateSelfType(const QualType &QualTy, llvm::DIType Ty);
  llvm::DIType getTypeOrNull(const QualType);
  llvm::DICompositeType getOrCreateMethodType(const CXXMethodDecl *Method,
                                              llvm::DIFile F);
  llvm::DICompositeType getOrCreateInstanceMethodType(
      QualType ThisPtr, const FunctionProtoType *Func, llvm::DIFile Unit);
  llvm::DICompositeType getOrCreateFunctionType(const Decl *D, QualType FnType,
                                                llvm::DIFile F);
  llvm::DIType getOrCreateVTablePtrType(llvm::DIFile F);
  llvm::DINameSpace getOrCreateNameSpace(const NamespaceDecl *N);
  llvm::DIType getOrCreateTypeDeclaration(QualType PointeeTy, llvm::DIFile F);
  llvm::DIType CreatePointerLikeType(llvm::dwarf::Tag Tag,
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

  void CollectCXXBases(const CXXRecordDecl *Decl,
                       llvm::DIFile F,
                       SmallVectorImpl<llvm::Value *> &EltTys,
                       llvm::DIType RecordTy);

  llvm::DIArray
  CollectTemplateParams(const TemplateParameterList *TPList,
                        ArrayRef<TemplateArgument> TAList,
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
                               llvm::DIScope scope);

  // Helpers for collecting fields of a record.
  void CollectRecordLambdaFields(const CXXRecordDecl *CXXDecl,
                                 SmallVectorImpl<llvm::Value *> &E,
                                 llvm::DIType RecordTy);
  llvm::DIDerivedType CreateRecordStaticField(const VarDecl *Var,
                                              llvm::DIType RecordTy);
  void CollectRecordNormalField(const FieldDecl *Field, uint64_t OffsetInBits,
                                llvm::DIFile F,
                                SmallVectorImpl<llvm::Value *> &E,
                                llvm::DIType RecordTy);
  void CollectRecordFields(const RecordDecl *Decl, llvm::DIFile F,
                           SmallVectorImpl<llvm::Value *> &E,
                           llvm::DICompositeType RecordTy);

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
  /// \param Loc       The location of the function header.
  /// \param ScopeLoc  The location of the function body.
  void EmitFunctionStart(GlobalDecl GD,
                         SourceLocation Loc, SourceLocation ScopeLoc,
                         QualType FnType, llvm::Function *Fn,
                         CGBuilderTy &Builder);

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

  /// EmitGlobalVariable - Emit global variable's debug info.
  void EmitGlobalVariable(const ValueDecl *VD, llvm::Constant *Init);

  /// \brief - Emit C++ using directive.
  void EmitUsingDirective(const UsingDirectiveDecl &UD);

  /// \brief - Emit C++ using declaration.
  void EmitUsingDecl(const UsingDecl &UD);

  /// \brief - Emit C++ namespace alias.
  llvm::DIImportedEntity EmitNamespaceAlias(const NamespaceAliasDecl &NA);

  /// getOrCreateRecordType - Emit record type's standalone debug info.
  llvm::DIType getOrCreateRecordType(QualType Ty, SourceLocation L);

  /// getOrCreateInterfaceType - Emit an objective c interface type standalone
  /// debug info.
  llvm::DIType getOrCreateInterfaceType(QualType Ty,
                                        SourceLocation Loc);

  void completeType(const EnumDecl *ED);
  void completeType(const RecordDecl *RD);
  void completeRequiredType(const RecordDecl *RD);
  void completeClassData(const RecordDecl *RD);

  void completeTemplateDefinition(const ClassTemplateSpecializationDecl &SD);

private:
  /// EmitDeclare - Emit call to llvm.dbg.declare for a variable declaration.
  /// Tag accepts custom types DW_TAG_arg_variable and DW_TAG_auto_variable,
  /// otherwise would be of type llvm::dwarf::Tag.
  void EmitDeclare(const VarDecl *decl, llvm::dwarf::LLVMConstants Tag,
                   llvm::Value *AI, unsigned ArgNo, CGBuilderTy &Builder);

  // EmitTypeForVarWithBlocksAttr - Build up structure info for the byref.
  // See BuildByRefType.
  llvm::DIType EmitTypeForVarWithBlocksAttr(const VarDecl *VD,
                                            uint64_t *OffSet);

  /// getContextDescriptor - Get context info for the decl.
  llvm::DIScope getContextDescriptor(const Decl *Decl);

  llvm::DIScope getCurrentContextDescriptor(const Decl *Decl);

  /// \brief Create a forward decl for a RecordType in a given context.
  llvm::DICompositeType getOrCreateRecordFwdDecl(const RecordType *,
                                                 llvm::DIDescriptor);

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
  llvm::DIType getOrCreateType(QualType Ty, llvm::DIFile Fg);

  /// getOrCreateLimitedType - Get the type from the cache or create a new
  /// partial type if necessary.
  llvm::DIType getOrCreateLimitedType(const RecordType *Ty, llvm::DIFile F);

  /// CreateTypeNode - Create type metadata for a source language type.
  llvm::DIType CreateTypeNode(QualType Ty, llvm::DIFile Fg);

  /// getObjCInterfaceDecl - return the underlying ObjCInterfaceDecl
  /// if Ty is an ObjCInterface or a pointer to one.
  ObjCInterfaceDecl* getObjCInterfaceDecl(QualType Ty);

  /// CreateMemberType - Create new member and increase Offset by FType's size.
  llvm::DIType CreateMemberType(llvm::DIFile Unit, QualType FType,
                                StringRef Name, uint64_t *Offset);

  /// \brief Retrieve the DIScope, if any, for the canonical form of this
  /// declaration.
  llvm::DIScope getDeclarationOrDefinition(const Decl *D);

  /// getFunctionDeclaration - Return debug info descriptor to describe method
  /// declaration for the given method definition.
  llvm::DISubprogram getFunctionDeclaration(const Decl *D);

  /// Return debug info descriptor to describe in-class static data member
  /// declaration for the given out-of-class definition.
  llvm::DIDerivedType
  getOrCreateStaticDataMemberDeclarationOrNull(const VarDecl *D);

  /// Return a global variable that represents one of the collection of
  /// global variables created for an anonmyous union.
  llvm::DIGlobalVariable
  CollectAnonRecordDecls(const RecordDecl *RD, llvm::DIFile Unit, unsigned LineNo,
                         StringRef LinkageName, llvm::GlobalVariable *Var,
                         llvm::DIDescriptor DContext);

  /// getFunctionName - Get function name for the given FunctionDecl. If the
  /// name is constructed on demand (e.g. C++ destructor) then the name
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

  /// internString - Allocate a copy of \p A using the DebugInfoNames allocator
  /// and return a reference to it. If multiple arguments are given the strings
  /// are concatenated.
  StringRef internString(StringRef A, StringRef B = StringRef()) {
    char *Data = DebugInfoNames.Allocate<char>(A.size() + B.size());
    std::memcpy(Data, A.data(), A.size());
    std::memcpy(Data + A.size(), B.data(), B.size());
    return StringRef(Data, A.size() + B.size());
  }
};

/// SaveAndRestoreLocation - An RAII object saves the current location
/// and automatically restores it to the original value.
class SaveAndRestoreLocation {
protected:
  SourceLocation SavedLoc;
  CGDebugInfo *DI;
  CGBuilderTy &Builder;
public:
  SaveAndRestoreLocation(CodeGenFunction &CGF, CGBuilderTy &B);
  /// Autorestore everything back to normal.
  ~SaveAndRestoreLocation();
};

/// NoLocation - An RAII object that temporarily disables debug
/// locations. This is useful for emitting instructions that should be
/// counted towards the function prologue.
class NoLocation : public SaveAndRestoreLocation {
public:
  NoLocation(CodeGenFunction &CGF, CGBuilderTy &B);
  /// Autorestore everything back to normal.
  ~NoLocation();
};

/// ArtificialLocation - An RAII object that temporarily switches to
/// an artificial debug location that has a valid scope, but no line
/// information. This is useful when emitting compiler-generated
/// helper functions that have no source location associated with
/// them. The DWARF specification allows the compiler to use the
/// special line number 0 to indicate code that can not be attributed
/// to any source location.
///
/// This is necessary because passing an empty SourceLocation to
/// CGDebugInfo::setLocation() will result in the last valid location
/// being reused.
class ArtificialLocation : public SaveAndRestoreLocation {
public:
  ArtificialLocation(CodeGenFunction &CGF, CGBuilderTy &B);

  /// Set the current location to line 0, but within the current scope
  /// (= the top of the LexicalBlockStack).
  void Emit();

  /// Autorestore everything back to normal.
  ~ArtificialLocation();
};


} // namespace CodeGen
} // namespace clang


#endif
