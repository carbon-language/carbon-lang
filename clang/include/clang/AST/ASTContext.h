//===--- ASTContext.h - Context to hold long-lived AST nodes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTContext interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTCONTEXT_H
#define LLVM_CLANG_AST_ASTCONTEXT_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/CanonicalType.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/Allocator.h"
#include <vector>

namespace llvm {
  struct fltSemantics;
}

namespace clang {
  class FileManager;
  class ASTRecordLayout;
  class Expr;
  class ExternalASTSource;
  class IdentifierTable;
  class SelectorTable;
  class SourceManager;
  class TargetInfo;
  // Decls
  class Decl;
  class FieldDecl;
  class ObjCIvarDecl;
  class ObjCIvarRefExpr;
  class ObjCPropertyDecl;
  class RecordDecl;
  class TagDecl;
  class TemplateTypeParmDecl;
  class TranslationUnitDecl;
  class TypeDecl;
  class TypedefDecl;
  class UnresolvedUsingDecl;
  class UsingDecl;

  namespace Builtin { class Context; }

/// ASTContext - This class holds long-lived AST nodes (such as types and
/// decls) that can be referred to throughout the semantic analysis of a file.
class ASTContext {
  std::vector<Type*> Types;
  llvm::FoldingSet<ExtQuals> ExtQualNodes;
  llvm::FoldingSet<ComplexType> ComplexTypes;
  llvm::FoldingSet<PointerType> PointerTypes;
  llvm::FoldingSet<BlockPointerType> BlockPointerTypes;
  llvm::FoldingSet<LValueReferenceType> LValueReferenceTypes;
  llvm::FoldingSet<RValueReferenceType> RValueReferenceTypes;
  llvm::FoldingSet<MemberPointerType> MemberPointerTypes;
  llvm::FoldingSet<ConstantArrayType> ConstantArrayTypes;
  llvm::FoldingSet<IncompleteArrayType> IncompleteArrayTypes;
  std::vector<VariableArrayType*> VariableArrayTypes;
  llvm::FoldingSet<DependentSizedArrayType> DependentSizedArrayTypes;
  llvm::FoldingSet<DependentSizedExtVectorType> DependentSizedExtVectorTypes;
  llvm::FoldingSet<VectorType> VectorTypes;
  llvm::FoldingSet<FunctionNoProtoType> FunctionNoProtoTypes;
  llvm::FoldingSet<FunctionProtoType> FunctionProtoTypes;
  llvm::FoldingSet<DependentTypeOfExprType> DependentTypeOfExprTypes;
  llvm::FoldingSet<DependentDecltypeType> DependentDecltypeTypes;
  llvm::FoldingSet<TemplateTypeParmType> TemplateTypeParmTypes;
  llvm::FoldingSet<SubstTemplateTypeParmType> SubstTemplateTypeParmTypes;
  llvm::FoldingSet<TemplateSpecializationType> TemplateSpecializationTypes;
  llvm::FoldingSet<QualifiedNameType> QualifiedNameTypes;
  llvm::FoldingSet<TypenameType> TypenameTypes;
  llvm::FoldingSet<ObjCInterfaceType> ObjCInterfaceTypes;
  llvm::FoldingSet<ObjCObjectPointerType> ObjCObjectPointerTypes;
  llvm::FoldingSet<ElaboratedType> ElaboratedTypes;

  llvm::FoldingSet<QualifiedTemplateName> QualifiedTemplateNames;
  llvm::FoldingSet<DependentTemplateName> DependentTemplateNames;

  /// \brief The set of nested name specifiers.
  ///
  /// This set is managed by the NestedNameSpecifier class.
  llvm::FoldingSet<NestedNameSpecifier> NestedNameSpecifiers;
  NestedNameSpecifier *GlobalNestedNameSpecifier;
  friend class NestedNameSpecifier;

  /// ASTRecordLayouts - A cache mapping from RecordDecls to ASTRecordLayouts.
  ///  This is lazily created.  This is intentionally not serialized.
  llvm::DenseMap<const RecordDecl*, const ASTRecordLayout*> ASTRecordLayouts;
  llvm::DenseMap<const ObjCContainerDecl*, const ASTRecordLayout*> ObjCLayouts;

  /// \brief Mapping from ObjCContainers to their ObjCImplementations.
  llvm::DenseMap<ObjCContainerDecl*, ObjCImplDecl*> ObjCImpls;

  llvm::DenseMap<unsigned, FixedWidthIntType*> SignedFixedWidthIntTypes;
  llvm::DenseMap<unsigned, FixedWidthIntType*> UnsignedFixedWidthIntTypes;

  /// BuiltinVaListType - built-in va list type.
  /// This is initially null and set by Sema::LazilyCreateBuiltin when
  /// a builtin that takes a valist is encountered.
  QualType BuiltinVaListType;

  /// ObjCIdType - a pseudo built-in typedef type (set by Sema).
  QualType ObjCIdTypedefType;

  /// ObjCSelType - another pseudo built-in typedef type (set by Sema).
  QualType ObjCSelType;
  const RecordType *SelStructType;

  /// ObjCProtoType - another pseudo built-in typedef type (set by Sema).
  QualType ObjCProtoType;
  const RecordType *ProtoStructType;

  /// ObjCClassType - another pseudo built-in typedef type (set by Sema).
  QualType ObjCClassTypedefType;

  QualType ObjCConstantStringType;
  RecordDecl *CFConstantStringTypeDecl;

  RecordDecl *ObjCFastEnumerationStateTypeDecl;

  /// \brief The type for the C FILE type.
  TypeDecl *FILEDecl;

  /// \brief The type for the C jmp_buf type.
  TypeDecl *jmp_bufDecl;

  /// \brief The type for the C sigjmp_buf type.
  TypeDecl *sigjmp_bufDecl;

  /// \brief Type for the Block descriptor for Blocks CodeGen.
  RecordDecl *BlockDescriptorType;

  /// \brief Type for the Block descriptor for Blocks CodeGen.
  RecordDecl *BlockDescriptorExtendedType;

  /// \brief Keeps track of all declaration attributes.
  ///
  /// Since so few decls have attrs, we keep them in a hash map instead of
  /// wasting space in the Decl class.
  llvm::DenseMap<const Decl*, Attr*> DeclAttrs;

  /// \brief Keeps track of the static data member templates from which
  /// static data members of class template specializations were instantiated.
  ///
  /// This data structure stores the mapping from instantiations of static
  /// data members to the static data member representations within the
  /// class template from which they were instantiated along with the kind
  /// of instantiation or specialization (a TemplateSpecializationKind - 1).
  ///
  /// Given the following example:
  ///
  /// \code
  /// template<typename T>
  /// struct X {
  ///   static T value;
  /// };
  ///
  /// template<typename T>
  ///   T X<T>::value = T(17);
  ///
  /// int *x = &X<int>::value;
  /// \endcode
  ///
  /// This mapping will contain an entry that maps from the VarDecl for
  /// X<int>::value to the corresponding VarDecl for X<T>::value (within the
  /// class template X) and will be marked TSK_ImplicitInstantiation.
  llvm::DenseMap<const VarDecl *, MemberSpecializationInfo *> 
    InstantiatedFromStaticDataMember;

  /// \brief Keeps track of the UnresolvedUsingDecls from which UsingDecls
  /// where created during instantiation.
  ///
  /// For example:
  /// \code
  /// template<typename T>
  /// struct A {
  ///   void f();
  /// };
  ///
  /// template<typename T>
  /// struct B : A<T> {
  ///   using A<T>::f;
  /// };
  ///
  /// template struct B<int>;
  /// \endcode
  ///
  /// This mapping will contain an entry that maps from the UsingDecl in
  /// B<int> to the UnresolvedUsingDecl in B<T>.
  llvm::DenseMap<UsingDecl *, UnresolvedUsingDecl *>
    InstantiatedFromUnresolvedUsingDecl;

  llvm::DenseMap<FieldDecl *, FieldDecl *> InstantiatedFromUnnamedFieldDecl;

  TranslationUnitDecl *TUDecl;

  /// SourceMgr - The associated SourceManager object.
  SourceManager &SourceMgr;

  /// LangOpts - The language options used to create the AST associated with
  ///  this ASTContext object.
  LangOptions LangOpts;

  /// \brief Whether we have already loaded comment source ranges from an
  /// external source.
  bool LoadedExternalComments;

  /// MallocAlloc/BumpAlloc - The allocator objects used to create AST objects.
  bool FreeMemory;
  llvm::MallocAllocator MallocAlloc;
  llvm::BumpPtrAllocator BumpAlloc;

  /// \brief Mapping from declarations to their comments, once we have
  /// already looked up the comment associated with a given declaration.
  llvm::DenseMap<const Decl *, std::string> DeclComments;

public:
  TargetInfo &Target;
  IdentifierTable &Idents;
  SelectorTable &Selectors;
  Builtin::Context &BuiltinInfo;
  DeclarationNameTable DeclarationNames;
  llvm::OwningPtr<ExternalASTSource> ExternalSource;
  clang::PrintingPolicy PrintingPolicy;

  // Typedefs which may be provided defining the structure of Objective-C
  // pseudo-builtins
  QualType ObjCIdRedefinitionType;
  QualType ObjCClassRedefinitionType;

  /// \brief Source ranges for all of the comments in the source file,
  /// sorted in order of appearance in the translation unit.
  std::vector<SourceRange> Comments;

  SourceManager& getSourceManager() { return SourceMgr; }
  const SourceManager& getSourceManager() const { return SourceMgr; }
  void *Allocate(unsigned Size, unsigned Align = 8) {
    return FreeMemory ? MallocAlloc.Allocate(Size, Align) :
                        BumpAlloc.Allocate(Size, Align);
  }
  void Deallocate(void *Ptr) {
    if (FreeMemory)
      MallocAlloc.Deallocate(Ptr);
  }
  const LangOptions& getLangOptions() const { return LangOpts; }

  FullSourceLoc getFullLoc(SourceLocation Loc) const {
    return FullSourceLoc(Loc,SourceMgr);
  }

  /// \brief Retrieve the attributes for the given declaration.
  Attr*& getDeclAttrs(const Decl *D) { return DeclAttrs[D]; }

  /// \brief Erase the attributes corresponding to the given declaration.
  void eraseDeclAttrs(const Decl *D) { DeclAttrs.erase(D); }

  /// \brief If this variable is an instantiated static data member of a
  /// class template specialization, returns the templated static data member
  /// from which it was instantiated.
  MemberSpecializationInfo *getInstantiatedFromStaticDataMember(
                                                           const VarDecl *Var);

  /// \brief Note that the static data member \p Inst is an instantiation of
  /// the static data member template \p Tmpl of a class template.
  void setInstantiatedFromStaticDataMember(VarDecl *Inst, VarDecl *Tmpl,
                                           TemplateSpecializationKind TSK);

  /// \brief If this using decl is instantiated from an unresolved using decl,
  /// return it.
  UnresolvedUsingDecl *getInstantiatedFromUnresolvedUsingDecl(UsingDecl *UUD);

  /// \brief Note that the using decl \p Inst is an instantiation of
  /// the unresolved using decl \p Tmpl of a class template.
  void setInstantiatedFromUnresolvedUsingDecl(UsingDecl *Inst,
                                              UnresolvedUsingDecl *Tmpl);


  FieldDecl *getInstantiatedFromUnnamedFieldDecl(FieldDecl *Field);

  void setInstantiatedFromUnnamedFieldDecl(FieldDecl *Inst, FieldDecl *Tmpl);

  TranslationUnitDecl *getTranslationUnitDecl() const { return TUDecl; }


  const char *getCommentForDecl(const Decl *D);

  // Builtin Types.
  CanQualType VoidTy;
  CanQualType BoolTy;
  CanQualType CharTy;
  CanQualType WCharTy;  // [C++ 3.9.1p5], integer type in C99.
  CanQualType Char16Ty; // [C++0x 3.9.1p5], integer type in C99.
  CanQualType Char32Ty; // [C++0x 3.9.1p5], integer type in C99.
  CanQualType SignedCharTy, ShortTy, IntTy, LongTy, LongLongTy, Int128Ty;
  CanQualType UnsignedCharTy, UnsignedShortTy, UnsignedIntTy, UnsignedLongTy;
  CanQualType UnsignedLongLongTy, UnsignedInt128Ty;
  CanQualType FloatTy, DoubleTy, LongDoubleTy;
  CanQualType FloatComplexTy, DoubleComplexTy, LongDoubleComplexTy;
  CanQualType VoidPtrTy, NullPtrTy;
  CanQualType OverloadTy;
  CanQualType DependentTy;
  CanQualType UndeducedAutoTy;
  CanQualType ObjCBuiltinIdTy, ObjCBuiltinClassTy;

  ASTContext(const LangOptions& LOpts, SourceManager &SM, TargetInfo &t,
             IdentifierTable &idents, SelectorTable &sels,
             Builtin::Context &builtins,
             bool FreeMemory = true, unsigned size_reserve=0);

  ~ASTContext();

  /// \brief Attach an external AST source to the AST context.
  ///
  /// The external AST source provides the ability to load parts of
  /// the abstract syntax tree as needed from some external storage,
  /// e.g., a precompiled header.
  void setExternalSource(llvm::OwningPtr<ExternalASTSource> &Source);

  /// \brief Retrieve a pointer to the external AST source associated
  /// with this AST context, if any.
  ExternalASTSource *getExternalSource() const { return ExternalSource.get(); }

  void PrintStats() const;
  const std::vector<Type*>& getTypes() const { return Types; }

  //===--------------------------------------------------------------------===//
  //                           Type Constructors
  //===--------------------------------------------------------------------===//

private:
  /// getExtQualType - Return a type with extended qualifiers.
  QualType getExtQualType(const Type *Base, Qualifiers Quals);

public:
  /// getAddSpaceQualType - Return the uniqued reference to the type for an
  /// address space qualified type with the specified type and address space.
  /// The resulting type has a union of the qualifiers from T and the address
  /// space. If T already has an address space specifier, it is silently
  /// replaced.
  QualType getAddrSpaceQualType(QualType T, unsigned AddressSpace);

  /// getObjCGCQualType - Returns the uniqued reference to the type for an
  /// objc gc qualified type. The retulting type has a union of the qualifiers
  /// from T and the gc attribute.
  QualType getObjCGCQualType(QualType T, Qualifiers::GC gcAttr);

  /// getRestrictType - Returns the uniqued reference to the type for a
  /// 'restrict' qualified type.  The resulting type has a union of the
  /// qualifiers from T and 'restrict'.
  QualType getRestrictType(QualType T) {
    return T.withFastQualifiers(Qualifiers::Restrict);
  }

  /// getVolatileType - Returns the uniqued reference to the type for a
  /// 'volatile' qualified type.  The resulting type has a union of the
  /// qualifiers from T and 'volatile'.
  QualType getVolatileType(QualType T);

  /// getConstType - Returns the uniqued reference to the type for a
  /// 'const' qualified type.  The resulting type has a union of the
  /// qualifiers from T and 'const'.
  ///
  /// It can be reasonably expected that this will always be
  /// equivalent to calling T.withConst().
  QualType getConstType(QualType T) { return T.withConst(); }

  /// getNoReturnType - Add the noreturn attribute to the given type which must
  /// be a FunctionType or a pointer to an allowable type or a BlockPointer.
  QualType getNoReturnType(QualType T);

  /// getComplexType - Return the uniqued reference to the type for a complex
  /// number with the specified element type.
  QualType getComplexType(QualType T);
  CanQualType getComplexType(CanQualType T) {
    return CanQualType::CreateUnsafe(getComplexType((QualType) T));
  }

  /// getPointerType - Return the uniqued reference to the type for a pointer to
  /// the specified type.
  QualType getPointerType(QualType T);
  CanQualType getPointerType(CanQualType T) {
    return CanQualType::CreateUnsafe(getPointerType((QualType) T));
  }

  /// getBlockPointerType - Return the uniqued reference to the type for a block
  /// of the specified type.
  QualType getBlockPointerType(QualType T);

  /// This gets the struct used to keep track of the descriptor for pointer to
  /// blocks.
  QualType getBlockDescriptorType();

  // Set the type for a Block descriptor type.
  void setBlockDescriptorType(QualType T);
  /// Get the BlockDescriptorType type, or NULL if it hasn't yet been built.
  QualType getRawBlockdescriptorType() {
    if (BlockDescriptorType)
      return getTagDeclType(BlockDescriptorType);
    return QualType();
  }

  /// This gets the struct used to keep track of the extended descriptor for
  /// pointer to blocks.
  QualType getBlockDescriptorExtendedType();

  // Set the type for a Block descriptor extended type.
  void setBlockDescriptorExtendedType(QualType T);
  /// Get the BlockDescriptorExtendedType type, or NULL if it hasn't yet been
  /// built.
  QualType getRawBlockdescriptorExtendedType() {
    if (BlockDescriptorExtendedType)
      return getTagDeclType(BlockDescriptorExtendedType);
    return QualType();
  }

  /// This gets the struct used to keep track of pointer to blocks, complete
  /// with captured variables.
  QualType getBlockParmType(bool BlockHasCopyDispose,
                            llvm::SmallVector<const Expr *, 8> &BDRDs);

  /// This builds the struct used for __block variables.
  QualType BuildByRefType(const char *DeclName, QualType Ty);

  /// Returns true iff we need copy/dispose helpers for the given type.
  bool BlockRequiresCopying(QualType Ty);

  /// getLValueReferenceType - Return the uniqued reference to the type for an
  /// lvalue reference to the specified type.
  QualType getLValueReferenceType(QualType T, bool SpelledAsLValue = true);

  /// getRValueReferenceType - Return the uniqued reference to the type for an
  /// rvalue reference to the specified type.
  QualType getRValueReferenceType(QualType T);

  /// getMemberPointerType - Return the uniqued reference to the type for a
  /// member pointer to the specified type in the specified class. The class
  /// is a Type because it could be a dependent name.
  QualType getMemberPointerType(QualType T, const Type *Cls);

  /// getVariableArrayType - Returns a non-unique reference to the type for a
  /// variable array of the specified element type.
  QualType getVariableArrayType(QualType EltTy, Expr *NumElts,
                                ArrayType::ArraySizeModifier ASM,
                                unsigned EltTypeQuals,
                                SourceRange Brackets);

  /// getDependentSizedArrayType - Returns a non-unique reference to
  /// the type for a dependently-sized array of the specified element
  /// type. FIXME: We will need these to be uniqued, or at least
  /// comparable, at some point.
  QualType getDependentSizedArrayType(QualType EltTy, Expr *NumElts,
                                      ArrayType::ArraySizeModifier ASM,
                                      unsigned EltTypeQuals,
                                      SourceRange Brackets);

  /// getIncompleteArrayType - Returns a unique reference to the type for a
  /// incomplete array of the specified element type.
  QualType getIncompleteArrayType(QualType EltTy,
                                  ArrayType::ArraySizeModifier ASM,
                                  unsigned EltTypeQuals);

  /// getConstantArrayType - Return the unique reference to the type for a
  /// constant array of the specified element type.
  QualType getConstantArrayType(QualType EltTy, const llvm::APInt &ArySize,
                                ArrayType::ArraySizeModifier ASM,
                                unsigned EltTypeQuals);

  /// getVectorType - Return the unique reference to a vector type of
  /// the specified element type and size. VectorType must be a built-in type.
  QualType getVectorType(QualType VectorType, unsigned NumElts);

  /// getExtVectorType - Return the unique reference to an extended vector type
  /// of the specified element type and size.  VectorType must be a built-in
  /// type.
  QualType getExtVectorType(QualType VectorType, unsigned NumElts);

  /// getDependentSizedExtVectorType - Returns a non-unique reference to
  /// the type for a dependently-sized vector of the specified element
  /// type. FIXME: We will need these to be uniqued, or at least
  /// comparable, at some point.
  QualType getDependentSizedExtVectorType(QualType VectorType,
                                          Expr *SizeExpr,
                                          SourceLocation AttrLoc);

  /// getFunctionNoProtoType - Return a K&R style C function type like 'int()'.
  ///
  QualType getFunctionNoProtoType(QualType ResultTy, bool NoReturn = false);

  /// getFunctionType - Return a normal function type with a typed argument
  /// list.  isVariadic indicates whether the argument list includes '...'.
  QualType getFunctionType(QualType ResultTy, const QualType *ArgArray,
                           unsigned NumArgs, bool isVariadic,
                           unsigned TypeQuals, bool hasExceptionSpec = false,
                           bool hasAnyExceptionSpec = false,
                           unsigned NumExs = 0, const QualType *ExArray = 0,
                           bool NoReturn = false);

  /// getTypeDeclType - Return the unique reference to the type for
  /// the specified type declaration.
  QualType getTypeDeclType(TypeDecl *Decl, TypeDecl* PrevDecl=0);

  /// getTypedefType - Return the unique reference to the type for the
  /// specified typename decl.
  QualType getTypedefType(TypedefDecl *Decl);

  QualType getSubstTemplateTypeParmType(const TemplateTypeParmType *Replaced,
                                        QualType Replacement);

  QualType getTemplateTypeParmType(unsigned Depth, unsigned Index,
                                   bool ParameterPack,
                                   IdentifierInfo *Name = 0);

  QualType getTemplateSpecializationType(TemplateName T,
                                         const TemplateArgument *Args,
                                         unsigned NumArgs,
                                         QualType Canon = QualType());

  QualType getTemplateSpecializationType(TemplateName T,
                                         const TemplateArgumentLoc *Args,
                                         unsigned NumArgs,
                                         QualType Canon = QualType());

  QualType getQualifiedNameType(NestedNameSpecifier *NNS,
                                QualType NamedType);
  QualType getTypenameType(NestedNameSpecifier *NNS,
                           const IdentifierInfo *Name,
                           QualType Canon = QualType());
  QualType getTypenameType(NestedNameSpecifier *NNS,
                           const TemplateSpecializationType *TemplateId,
                           QualType Canon = QualType());
  QualType getElaboratedType(QualType UnderlyingType,
                             ElaboratedType::TagKind Tag);

  QualType getObjCInterfaceType(const ObjCInterfaceDecl *Decl,
                                ObjCProtocolDecl **Protocols = 0,
                                unsigned NumProtocols = 0);

  /// getObjCObjectPointerType - Return a ObjCObjectPointerType type for the
  /// given interface decl and the conforming protocol list.
  QualType getObjCObjectPointerType(QualType OIT,
                                    ObjCProtocolDecl **ProtocolList = 0,
                                    unsigned NumProtocols = 0);

  /// getTypeOfType - GCC extension.
  QualType getTypeOfExprType(Expr *e);
  QualType getTypeOfType(QualType t);

  /// getDecltypeType - C++0x decltype.
  QualType getDecltypeType(Expr *e);

  /// getTagDeclType - Return the unique reference to the type for the
  /// specified TagDecl (struct/union/class/enum) decl.
  QualType getTagDeclType(const TagDecl *Decl);

  /// getSizeType - Return the unique type for "size_t" (C99 7.17), defined
  /// in <stddef.h>. The sizeof operator requires this (C99 6.5.3.4p4).
  QualType getSizeType() const;

  /// getWCharType - In C++, this returns the unique wchar_t type.  In C99, this
  /// returns a type compatible with the type defined in <stddef.h> as defined
  /// by the target.
  QualType getWCharType() const { return WCharTy; }

  /// getSignedWCharType - Return the type of "signed wchar_t".
  /// Used when in C++, as a GCC extension.
  QualType getSignedWCharType() const;

  /// getUnsignedWCharType - Return the type of "unsigned wchar_t".
  /// Used when in C++, as a GCC extension.
  QualType getUnsignedWCharType() const;

  /// getPointerDiffType - Return the unique type for "ptrdiff_t" (ref?)
  /// defined in <stddef.h>. Pointer - pointer requires this (C99 6.5.6p9).
  QualType getPointerDiffType() const;

  // getCFConstantStringType - Return the C structure type used to represent
  // constant CFStrings.
  QualType getCFConstantStringType();

  /// Get the structure type used to representation CFStrings, or NULL
  /// if it hasn't yet been built.
  QualType getRawCFConstantStringType() {
    if (CFConstantStringTypeDecl)
      return getTagDeclType(CFConstantStringTypeDecl);
    return QualType();
  }
  void setCFConstantStringType(QualType T);

  // This setter/getter represents the ObjC type for an NSConstantString.
  void setObjCConstantStringInterface(ObjCInterfaceDecl *Decl);
  QualType getObjCConstantStringInterface() const {
    return ObjCConstantStringType;
  }

  //// This gets the struct used to keep track of fast enumerations.
  QualType getObjCFastEnumerationStateType();

  /// Get the ObjCFastEnumerationState type, or NULL if it hasn't yet
  /// been built.
  QualType getRawObjCFastEnumerationStateType() {
    if (ObjCFastEnumerationStateTypeDecl)
      return getTagDeclType(ObjCFastEnumerationStateTypeDecl);
    return QualType();
  }

  void setObjCFastEnumerationStateType(QualType T);

  /// \brief Set the type for the C FILE type.
  void setFILEDecl(TypeDecl *FILEDecl) { this->FILEDecl = FILEDecl; }

  /// \brief Retrieve the C FILE type.
  QualType getFILEType() {
    if (FILEDecl)
      return getTypeDeclType(FILEDecl);
    return QualType();
  }

  /// \brief Set the type for the C jmp_buf type.
  void setjmp_bufDecl(TypeDecl *jmp_bufDecl) {
    this->jmp_bufDecl = jmp_bufDecl;
  }

  /// \brief Retrieve the C jmp_buf type.
  QualType getjmp_bufType() {
    if (jmp_bufDecl)
      return getTypeDeclType(jmp_bufDecl);
    return QualType();
  }

  /// \brief Set the type for the C sigjmp_buf type.
  void setsigjmp_bufDecl(TypeDecl *sigjmp_bufDecl) {
    this->sigjmp_bufDecl = sigjmp_bufDecl;
  }

  /// \brief Retrieve the C sigjmp_buf type.
  QualType getsigjmp_bufType() {
    if (sigjmp_bufDecl)
      return getTypeDeclType(sigjmp_bufDecl);
    return QualType();
  }

  /// getObjCEncodingForType - Emit the ObjC type encoding for the
  /// given type into \arg S. If \arg NameFields is specified then
  /// record field names are also encoded.
  void getObjCEncodingForType(QualType t, std::string &S,
                              const FieldDecl *Field=0);

  void getLegacyIntegralTypeEncoding(QualType &t) const;

  // Put the string version of type qualifiers into S.
  void getObjCEncodingForTypeQualifier(Decl::ObjCDeclQualifier QT,
                                       std::string &S) const;

  /// getObjCEncodingForMethodDecl - Return the encoded type for this method
  /// declaration.
  void getObjCEncodingForMethodDecl(const ObjCMethodDecl *Decl, std::string &S);

  /// getObjCEncodingForPropertyDecl - Return the encoded type for
  /// this method declaration. If non-NULL, Container must be either
  /// an ObjCCategoryImplDecl or ObjCImplementationDecl; it should
  /// only be NULL when getting encodings for protocol properties.
  void getObjCEncodingForPropertyDecl(const ObjCPropertyDecl *PD,
                                      const Decl *Container,
                                      std::string &S);

  bool ProtocolCompatibleWithProtocol(ObjCProtocolDecl *lProto,
                                      ObjCProtocolDecl *rProto);

  /// getObjCEncodingTypeSize returns size of type for objective-c encoding
  /// purpose.
  int getObjCEncodingTypeSize(QualType t);

  /// This setter/getter represents the ObjC 'id' type. It is setup lazily, by
  /// Sema.  id is always a (typedef for a) pointer type, a pointer to a struct.
  QualType getObjCIdType() const { return ObjCIdTypedefType; }
  void setObjCIdType(QualType T);

  void setObjCSelType(QualType T);
  QualType getObjCSelType() const { return ObjCSelType; }

  void setObjCProtoType(QualType QT);
  QualType getObjCProtoType() const { return ObjCProtoType; }

  /// This setter/getter repreents the ObjC 'Class' type. It is setup lazily, by
  /// Sema.  'Class' is always a (typedef for a) pointer type, a pointer to a
  /// struct.
  QualType getObjCClassType() const { return ObjCClassTypedefType; }
  void setObjCClassType(QualType T);

  void setBuiltinVaListType(QualType T);
  QualType getBuiltinVaListType() const { return BuiltinVaListType; }

  QualType getFixedWidthIntType(unsigned Width, bool Signed);

  /// getCVRQualifiedType - Returns a type with additional const,
  /// volatile, or restrict qualifiers.
  QualType getCVRQualifiedType(QualType T, unsigned CVR) {
    return getQualifiedType(T, Qualifiers::fromCVRMask(CVR));
  }

  /// getQualifiedType - Returns a type with additional qualifiers.
  QualType getQualifiedType(QualType T, Qualifiers Qs) {
    if (!Qs.hasNonFastQualifiers())
      return T.withFastQualifiers(Qs.getFastQualifiers());
    QualifierCollector Qc(Qs);
    const Type *Ptr = Qc.strip(T);
    return getExtQualType(Ptr, Qc);
  }

  /// getQualifiedType - Returns a type with additional qualifiers.
  QualType getQualifiedType(const Type *T, Qualifiers Qs) {
    if (!Qs.hasNonFastQualifiers())
      return QualType(T, Qs.getFastQualifiers());
    return getExtQualType(T, Qs);
  }

  TemplateName getQualifiedTemplateName(NestedNameSpecifier *NNS,
                                        bool TemplateKeyword,
                                        TemplateDecl *Template);
  TemplateName getQualifiedTemplateName(NestedNameSpecifier *NNS,
                                        bool TemplateKeyword,
                                        OverloadedFunctionDecl *Template);

  TemplateName getDependentTemplateName(NestedNameSpecifier *NNS,
                                        const IdentifierInfo *Name);
  TemplateName getDependentTemplateName(NestedNameSpecifier *NNS,
                                        OverloadedOperatorKind Operator);

  enum GetBuiltinTypeError {
    GE_None,              //< No error
    GE_Missing_stdio,     //< Missing a type from <stdio.h>
    GE_Missing_setjmp     //< Missing a type from <setjmp.h>
  };

  /// GetBuiltinType - Return the type for the specified builtin.
  QualType GetBuiltinType(unsigned ID, GetBuiltinTypeError &Error);

private:
  CanQualType getFromTargetType(unsigned Type) const;

  //===--------------------------------------------------------------------===//
  //                         Type Predicates.
  //===--------------------------------------------------------------------===//

public:
  /// getObjCGCAttr - Returns one of GCNone, Weak or Strong objc's
  /// garbage collection attribute.
  ///
  Qualifiers::GC getObjCGCAttrKind(const QualType &Ty) const;

  /// isObjCNSObjectType - Return true if this is an NSObject object with
  /// its NSObject attribute set.
  bool isObjCNSObjectType(QualType Ty) const;

  //===--------------------------------------------------------------------===//
  //                         Type Sizing and Analysis
  //===--------------------------------------------------------------------===//

  /// getFloatTypeSemantics - Return the APFloat 'semantics' for the specified
  /// scalar floating point type.
  const llvm::fltSemantics &getFloatTypeSemantics(QualType T) const;

  /// getTypeInfo - Get the size and alignment of the specified complete type in
  /// bits.
  std::pair<uint64_t, unsigned> getTypeInfo(const Type *T);
  std::pair<uint64_t, unsigned> getTypeInfo(QualType T) {
    return getTypeInfo(T.getTypePtr());
  }

  /// getTypeSize - Return the size of the specified type, in bits.  This method
  /// does not work on incomplete types.
  uint64_t getTypeSize(QualType T) {
    return getTypeInfo(T).first;
  }
  uint64_t getTypeSize(const Type *T) {
    return getTypeInfo(T).first;
  }

  /// getTypeAlign - Return the ABI-specified alignment of a type, in bits.
  /// This method does not work on incomplete types.
  unsigned getTypeAlign(QualType T) {
    return getTypeInfo(T).second;
  }
  unsigned getTypeAlign(const Type *T) {
    return getTypeInfo(T).second;
  }

  /// getPreferredTypeAlign - Return the "preferred" alignment of the specified
  /// type for the current target in bits.  This can be different than the ABI
  /// alignment in cases where it is beneficial for performance to overalign
  /// a data type.
  unsigned getPreferredTypeAlign(const Type *T);

  /// getDeclAlignInBytes - Return the alignment of the specified decl
  /// that should be returned by __alignof().  Note that bitfields do
  /// not have a valid alignment, so this method will assert on them.
  unsigned getDeclAlignInBytes(const Decl *D);

  /// getASTRecordLayout - Get or compute information about the layout of the
  /// specified record (struct/union/class), which indicates its size and field
  /// position information.
  const ASTRecordLayout &getASTRecordLayout(const RecordDecl *D);

  /// getASTObjCInterfaceLayout - Get or compute information about the
  /// layout of the specified Objective-C interface.
  const ASTRecordLayout &getASTObjCInterfaceLayout(const ObjCInterfaceDecl *D);

  /// getASTObjCImplementationLayout - Get or compute information about
  /// the layout of the specified Objective-C implementation. This may
  /// differ from the interface if synthesized ivars are present.
  const ASTRecordLayout &
  getASTObjCImplementationLayout(const ObjCImplementationDecl *D);

  void CollectObjCIvars(const ObjCInterfaceDecl *OI,
                        llvm::SmallVectorImpl<FieldDecl*> &Fields);

  void ShallowCollectObjCIvars(const ObjCInterfaceDecl *OI,
                               llvm::SmallVectorImpl<ObjCIvarDecl*> &Ivars,
                               bool CollectSynthesized = true);
  void CollectSynthesizedIvars(const ObjCInterfaceDecl *OI,
                               llvm::SmallVectorImpl<ObjCIvarDecl*> &Ivars);
  void CollectProtocolSynthesizedIvars(const ObjCProtocolDecl *PD,
                               llvm::SmallVectorImpl<ObjCIvarDecl*> &Ivars);
  unsigned CountSynthesizedIvars(const ObjCInterfaceDecl *OI);
  unsigned CountProtocolSynthesizedIvars(const ObjCProtocolDecl *PD);
  void CollectInheritedProtocols(const Decl *CDecl,
                          llvm::SmallVectorImpl<ObjCProtocolDecl*> &Protocols);

  //===--------------------------------------------------------------------===//
  //                            Type Operators
  //===--------------------------------------------------------------------===//

  /// getCanonicalType - Return the canonical (structural) type corresponding to
  /// the specified potentially non-canonical type.  The non-canonical version
  /// of a type may have many "decorated" versions of types.  Decorators can
  /// include typedefs, 'typeof' operators, etc. The returned type is guaranteed
  /// to be free of any of these, allowing two canonical types to be compared
  /// for exact equality with a simple pointer comparison.
  CanQualType getCanonicalType(QualType T);
  const Type *getCanonicalType(const Type *T) {
    return T->getCanonicalTypeInternal().getTypePtr();
  }

  /// getCanonicalParamType - Return the canonical parameter type
  /// corresponding to the specific potentially non-canonical one.
  /// Qualifiers are stripped off, functions are turned into function
  /// pointers, and arrays decay one level into pointers.
  CanQualType getCanonicalParamType(QualType T);

  /// \brief Determine whether the given types are equivalent.
  bool hasSameType(QualType T1, QualType T2) {
    return getCanonicalType(T1) == getCanonicalType(T2);
  }

  /// \brief Determine whether the given types are equivalent after
  /// cvr-qualifiers have been removed.
  bool hasSameUnqualifiedType(QualType T1, QualType T2) {
    T1 = getCanonicalType(T1);
    T2 = getCanonicalType(T2);
    return T1.getUnqualifiedType() == T2.getUnqualifiedType();
  }

  /// \brief Retrieves the "canonical" declaration of

  /// \brief Retrieves the "canonical" nested name specifier for a
  /// given nested name specifier.
  ///
  /// The canonical nested name specifier is a nested name specifier
  /// that uniquely identifies a type or namespace within the type
  /// system. For example, given:
  ///
  /// \code
  /// namespace N {
  ///   struct S {
  ///     template<typename T> struct X { typename T* type; };
  ///   };
  /// }
  ///
  /// template<typename T> struct Y {
  ///   typename N::S::X<T>::type member;
  /// };
  /// \endcode
  ///
  /// Here, the nested-name-specifier for N::S::X<T>:: will be
  /// S::X<template-param-0-0>, since 'S' and 'X' are uniquely defined
  /// by declarations in the type system and the canonical type for
  /// the template type parameter 'T' is template-param-0-0.
  NestedNameSpecifier *
  getCanonicalNestedNameSpecifier(NestedNameSpecifier *NNS);

  /// \brief Retrieves the "canonical" template name that refers to a
  /// given template.
  ///
  /// The canonical template name is the simplest expression that can
  /// be used to refer to a given template. For most templates, this
  /// expression is just the template declaration itself. For example,
  /// the template std::vector can be referred to via a variety of
  /// names---std::vector, ::std::vector, vector (if vector is in
  /// scope), etc.---but all of these names map down to the same
  /// TemplateDecl, which is used to form the canonical template name.
  ///
  /// Dependent template names are more interesting. Here, the
  /// template name could be something like T::template apply or
  /// std::allocator<T>::template rebind, where the nested name
  /// specifier itself is dependent. In this case, the canonical
  /// template name uses the shortest form of the dependent
  /// nested-name-specifier, which itself contains all canonical
  /// types, values, and templates.
  TemplateName getCanonicalTemplateName(TemplateName Name);

  /// \brief Determine whether the given template names refer to the same
  /// template.
  bool hasSameTemplateName(TemplateName X, TemplateName Y);
  
  /// \brief Retrieve the "canonical" template argument.
  ///
  /// The canonical template argument is the simplest template argument
  /// (which may be a type, value, expression, or declaration) that
  /// expresses the value of the argument.
  TemplateArgument getCanonicalTemplateArgument(const TemplateArgument &Arg);

  /// Type Query functions.  If the type is an instance of the specified class,
  /// return the Type pointer for the underlying maximally pretty type.  This
  /// is a member of ASTContext because this may need to do some amount of
  /// canonicalization, e.g. to move type qualifiers into the element type.
  const ArrayType *getAsArrayType(QualType T);
  const ConstantArrayType *getAsConstantArrayType(QualType T) {
    return dyn_cast_or_null<ConstantArrayType>(getAsArrayType(T));
  }
  const VariableArrayType *getAsVariableArrayType(QualType T) {
    return dyn_cast_or_null<VariableArrayType>(getAsArrayType(T));
  }
  const IncompleteArrayType *getAsIncompleteArrayType(QualType T) {
    return dyn_cast_or_null<IncompleteArrayType>(getAsArrayType(T));
  }

  /// getBaseElementType - Returns the innermost element type of an array type.
  /// For example, will return "int" for int[m][n]
  QualType getBaseElementType(const ArrayType *VAT);

  /// getBaseElementType - Returns the innermost element type of a type
  /// (which needn't actually be an array type).
  QualType getBaseElementType(QualType QT);

  /// getConstantArrayElementCount - Returns number of constant array elements.
  uint64_t getConstantArrayElementCount(const ConstantArrayType *CA) const;

  /// getArrayDecayedType - Return the properly qualified result of decaying the
  /// specified array type to a pointer.  This operation is non-trivial when
  /// handling typedefs etc.  The canonical type of "T" must be an array type,
  /// this returns a pointer to a properly qualified element of the array.
  ///
  /// See C99 6.7.5.3p7 and C99 6.3.2.1p3.
  QualType getArrayDecayedType(QualType T);

  /// getPromotedIntegerType - Returns the type that Promotable will
  /// promote to: C99 6.3.1.1p2, assuming that Promotable is a promotable
  /// integer type.
  QualType getPromotedIntegerType(QualType PromotableType);

  /// \brief Whether this is a promotable bitfield reference according
  /// to C99 6.3.1.1p2, bullet 2 (and GCC extensions).
  ///
  /// \returns the type this bit-field will promote to, or NULL if no
  /// promotion occurs.
  QualType isPromotableBitField(Expr *E);

  /// getIntegerTypeOrder - Returns the highest ranked integer type:
  /// C99 6.3.1.8p1.  If LHS > RHS, return 1.  If LHS == RHS, return 0. If
  /// LHS < RHS, return -1.
  int getIntegerTypeOrder(QualType LHS, QualType RHS);

  /// getFloatingTypeOrder - Compare the rank of the two specified floating
  /// point types, ignoring the domain of the type (i.e. 'double' ==
  /// '_Complex double').  If LHS > RHS, return 1.  If LHS == RHS, return 0. If
  /// LHS < RHS, return -1.
  int getFloatingTypeOrder(QualType LHS, QualType RHS);

  /// getFloatingTypeOfSizeWithinDomain - Returns a real floating
  /// point or a complex type (based on typeDomain/typeSize).
  /// 'typeDomain' is a real floating point or complex type.
  /// 'typeSize' is a real floating point or complex type.
  QualType getFloatingTypeOfSizeWithinDomain(QualType typeSize,
                                             QualType typeDomain) const;

private:
  // Helper for integer ordering
  unsigned getIntegerRank(Type* T);

public:

  //===--------------------------------------------------------------------===//
  //                    Type Compatibility Predicates
  //===--------------------------------------------------------------------===//

  /// Compatibility predicates used to check assignment expressions.
  bool typesAreCompatible(QualType, QualType); // C99 6.2.7p1

  bool isObjCIdType(QualType T) const {
    return T == ObjCIdTypedefType;
  }
  bool isObjCClassType(QualType T) const {
    return T == ObjCClassTypedefType;
  }
  bool isObjCSelType(QualType T) const {
    assert(SelStructType && "isObjCSelType used before 'SEL' type is built");
    return T->getAsStructureType() == SelStructType;
  }
  bool QualifiedIdConformsQualifiedId(QualType LHS, QualType RHS);
  bool ObjCQualifiedIdTypesAreCompatible(QualType LHS, QualType RHS,
                                         bool ForCompare);

  // Check the safety of assignment from LHS to RHS
  bool canAssignObjCInterfaces(const ObjCObjectPointerType *LHSOPT,
                               const ObjCObjectPointerType *RHSOPT);
  bool canAssignObjCInterfaces(const ObjCInterfaceType *LHS,
                               const ObjCInterfaceType *RHS);
  bool areComparableObjCPointerTypes(QualType LHS, QualType RHS);
  QualType areCommonBaseCompatible(const ObjCObjectPointerType *LHSOPT,
                                   const ObjCObjectPointerType *RHSOPT);
  
  // Functions for calculating composite types
  QualType mergeTypes(QualType, QualType);
  QualType mergeFunctionTypes(QualType, QualType);

  /// UsualArithmeticConversionsType - handles the various conversions
  /// that are common to binary operators (C99 6.3.1.8, C++ [expr]p9)
  /// and returns the result type of that conversion.
  QualType UsualArithmeticConversionsType(QualType lhs, QualType rhs);

  //===--------------------------------------------------------------------===//
  //                    Integer Predicates
  //===--------------------------------------------------------------------===//

  // The width of an integer, as defined in C99 6.2.6.2. This is the number
  // of bits in an integer type excluding any padding bits.
  unsigned getIntWidth(QualType T);

  // Per C99 6.2.5p6, for every signed integer type, there is a corresponding
  // unsigned integer type.  This method takes a signed type, and returns the
  // corresponding unsigned integer type.
  QualType getCorrespondingUnsignedType(QualType T);

  //===--------------------------------------------------------------------===//
  //                    Type Iterators.
  //===--------------------------------------------------------------------===//

  typedef std::vector<Type*>::iterator       type_iterator;
  typedef std::vector<Type*>::const_iterator const_type_iterator;

  type_iterator types_begin() { return Types.begin(); }
  type_iterator types_end() { return Types.end(); }
  const_type_iterator types_begin() const { return Types.begin(); }
  const_type_iterator types_end() const { return Types.end(); }

  //===--------------------------------------------------------------------===//
  //                    Integer Values
  //===--------------------------------------------------------------------===//

  /// MakeIntValue - Make an APSInt of the appropriate width and
  /// signedness for the given \arg Value and integer \arg Type.
  llvm::APSInt MakeIntValue(uint64_t Value, QualType Type) {
    llvm::APSInt Res(getIntWidth(Type), !Type->isSignedIntegerType());
    Res = Value;
    return Res;
  }

  /// \brief Get the implementation of ObjCInterfaceDecl,or NULL if none exists.
  ObjCImplementationDecl *getObjCImplementation(ObjCInterfaceDecl *D);
  /// \brief Get the implementation of ObjCCategoryDecl, or NULL if none exists.
  ObjCCategoryImplDecl   *getObjCImplementation(ObjCCategoryDecl *D);

  /// \brief Set the implementation of ObjCInterfaceDecl.
  void setObjCImplementation(ObjCInterfaceDecl *IFaceD,
                             ObjCImplementationDecl *ImplD);
  /// \brief Set the implementation of ObjCCategoryDecl.
  void setObjCImplementation(ObjCCategoryDecl *CatD,
                             ObjCCategoryImplDecl *ImplD);

  /// \brief Allocate an uninitialized DeclaratorInfo.
  ///
  /// The caller should initialize the memory held by DeclaratorInfo using
  /// the TypeLoc wrappers.
  ///
  /// \param T the type that will be the basis for type source info. This type
  /// should refer to how the declarator was written in source code, not to
  /// what type semantic analysis resolved the declarator to.
  ///
  /// \param Size the size of the type info to create, or 0 if the size
  /// should be calculated based on the type.
  DeclaratorInfo *CreateDeclaratorInfo(QualType T, unsigned Size = 0);

  /// \brief Allocate a DeclaratorInfo where all locations have been
  /// initialized to a given location, which defaults to the empty
  /// location.
  DeclaratorInfo *
  getTrivialDeclaratorInfo(QualType T, SourceLocation Loc = SourceLocation());

private:
  ASTContext(const ASTContext&); // DO NOT IMPLEMENT
  void operator=(const ASTContext&); // DO NOT IMPLEMENT

  void InitBuiltinTypes();
  void InitBuiltinType(CanQualType &R, BuiltinType::Kind K);

  // Return the ObjC type encoding for a given type.
  void getObjCEncodingForTypeImpl(QualType t, std::string &S,
                                  bool ExpandPointedToStructures,
                                  bool ExpandStructures,
                                  const FieldDecl *Field,
                                  bool OutermostType = false,
                                  bool EncodingProperty = false);

  const ASTRecordLayout &getObjCLayout(const ObjCInterfaceDecl *D,
                                       const ObjCImplementationDecl *Impl);
};
  
/// @brief Utility function for constructing a nullary selector.
static inline Selector GetNullarySelector(const char* name, ASTContext& Ctx) {
  IdentifierInfo* II = &Ctx.Idents.get(name);
  return Ctx.Selectors.getSelector(0, &II);
}

/// @brief Utility function for constructing an unary selector.
static inline Selector GetUnarySelector(const char* name, ASTContext& Ctx) {
  IdentifierInfo* II = &Ctx.Idents.get(name);
  return Ctx.Selectors.getSelector(1, &II);
}

}  // end namespace clang

// operator new and delete aren't allowed inside namespaces.
// The throw specifications are mandated by the standard.
/// @brief Placement new for using the ASTContext's allocator.
///
/// This placement form of operator new uses the ASTContext's allocator for
/// obtaining memory. It is a non-throwing new, which means that it returns
/// null on error. (If that is what the allocator does. The current does, so if
/// this ever changes, this operator will have to be changed, too.)
/// Usage looks like this (assuming there's an ASTContext 'Context' in scope):
/// @code
/// // Default alignment (16)
/// IntegerLiteral *Ex = new (Context) IntegerLiteral(arguments);
/// // Specific alignment
/// IntegerLiteral *Ex2 = new (Context, 8) IntegerLiteral(arguments);
/// @endcode
/// Please note that you cannot use delete on the pointer; it must be
/// deallocated using an explicit destructor call followed by
/// @c Context.Deallocate(Ptr).
///
/// @param Bytes The number of bytes to allocate. Calculated by the compiler.
/// @param C The ASTContext that provides the allocator.
/// @param Alignment The alignment of the allocated memory (if the underlying
///                  allocator supports it).
/// @return The allocated memory. Could be NULL.
inline void *operator new(size_t Bytes, clang::ASTContext &C,
                          size_t Alignment) throw () {
  return C.Allocate(Bytes, Alignment);
}
/// @brief Placement delete companion to the new above.
///
/// This operator is just a companion to the new above. There is no way of
/// invoking it directly; see the new operator for more details. This operator
/// is called implicitly by the compiler if a placement new expression using
/// the ASTContext throws in the object constructor.
inline void operator delete(void *Ptr, clang::ASTContext &C, size_t)
              throw () {
  C.Deallocate(Ptr);
}

/// This placement form of operator new[] uses the ASTContext's allocator for
/// obtaining memory. It is a non-throwing new[], which means that it returns
/// null on error.
/// Usage looks like this (assuming there's an ASTContext 'Context' in scope):
/// @code
/// // Default alignment (16)
/// char *data = new (Context) char[10];
/// // Specific alignment
/// char *data = new (Context, 8) char[10];
/// @endcode
/// Please note that you cannot use delete on the pointer; it must be
/// deallocated using an explicit destructor call followed by
/// @c Context.Deallocate(Ptr).
///
/// @param Bytes The number of bytes to allocate. Calculated by the compiler.
/// @param C The ASTContext that provides the allocator.
/// @param Alignment The alignment of the allocated memory (if the underlying
///                  allocator supports it).
/// @return The allocated memory. Could be NULL.
inline void *operator new[](size_t Bytes, clang::ASTContext& C,
                            size_t Alignment = 16) throw () {
  return C.Allocate(Bytes, Alignment);
}

/// @brief Placement delete[] companion to the new[] above.
///
/// This operator is just a companion to the new[] above. There is no way of
/// invoking it directly; see the new[] operator for more details. This operator
/// is called implicitly by the compiler if a placement new[] expression using
/// the ASTContext throws in the object constructor.
inline void operator delete[](void *Ptr, clang::ASTContext &C) throw () {
  C.Deallocate(Ptr);
}

#endif
