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
#include "clang/AST/Builtins.h"
#include "clang/AST/Decl.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Bitcode/SerializationFwd.h"
#include "llvm/Support/Allocator.h"
#include <vector>

namespace llvm {
  struct fltSemantics;
}

namespace clang {
  class FileManager;
  class ASTRecordLayout;
  class Expr;
  class IdentifierTable;
  class SelectorTable;
  class SourceManager;
  class TargetInfo;
  // Decls
  class Decl;
  class ObjCPropertyDecl;
  class RecordDecl;
  class TagDecl;
  class TranslationUnitDecl;
  class TypeDecl;
  class TypedefDecl;
  class TemplateTypeParmDecl;
  class FieldDecl;
  class ObjCIvarRefExpr;
  class ObjCIvarDecl;
  
/// ASTContext - This class holds long-lived AST nodes (such as types and
/// decls) that can be referred to throughout the semantic analysis of a file.
class ASTContext {  
  std::vector<Type*> Types;
  llvm::FoldingSet<ExtQualType> ExtQualTypes;
  llvm::FoldingSet<ComplexType> ComplexTypes;
  llvm::FoldingSet<PointerType> PointerTypes;
  llvm::FoldingSet<BlockPointerType> BlockPointerTypes;
  llvm::FoldingSet<LValueReferenceType> LValueReferenceTypes;
  llvm::FoldingSet<RValueReferenceType> RValueReferenceTypes;
  llvm::FoldingSet<MemberPointerType> MemberPointerTypes;
  llvm::FoldingSet<ConstantArrayType> ConstantArrayTypes;
  llvm::FoldingSet<IncompleteArrayType> IncompleteArrayTypes;
  std::vector<VariableArrayType*> VariableArrayTypes;
  std::vector<DependentSizedArrayType*> DependentSizedArrayTypes;
  llvm::FoldingSet<VectorType> VectorTypes;
  llvm::FoldingSet<FunctionNoProtoType> FunctionNoProtoTypes;
  llvm::FoldingSet<FunctionProtoType> FunctionProtoTypes;
  llvm::FoldingSet<TemplateTypeParmType> TemplateTypeParmTypes;
  llvm::FoldingSet<TemplateSpecializationType> TemplateSpecializationTypes;
  llvm::FoldingSet<QualifiedNameType> QualifiedNameTypes;
  llvm::FoldingSet<TypenameType> TypenameTypes;
  llvm::FoldingSet<ObjCQualifiedInterfaceType> ObjCQualifiedInterfaceTypes;
  llvm::FoldingSet<ObjCQualifiedIdType> ObjCQualifiedIdTypes;

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
  llvm::DenseMap<const ObjCInterfaceDecl*, 
                 const ASTRecordLayout*> ASTObjCInterfaces;

  llvm::DenseMap<unsigned, FixedWidthIntType*> SignedFixedWidthIntTypes;
  llvm::DenseMap<unsigned, FixedWidthIntType*> UnsignedFixedWidthIntTypes;

  // FIXME: ASTRecordForInterface/ASTFieldForIvarRef and addRecordToClass and
  // getFieldDecl be part of the backend (i.e. CodeGenTypes)?
  llvm::DenseMap<const ObjCInterfaceDecl*, RecordDecl*> ASTRecordForInterface;
  llvm::DenseMap<const ObjCIvarRefExpr*, const FieldDecl*> ASTFieldForIvarRef;
  
  /// BuiltinVaListType - built-in va list type.
  /// This is initially null and set by Sema::LazilyCreateBuiltin when
  /// a builtin that takes a valist is encountered.
  QualType BuiltinVaListType;
  
  /// ObjCIdType - a pseudo built-in typedef type (set by Sema).
  QualType ObjCIdType;
  const RecordType *IdStructType;
  
  /// ObjCSelType - another pseudo built-in typedef type (set by Sema).
  QualType ObjCSelType;
  const RecordType *SelStructType;
  
  /// ObjCProtoType - another pseudo built-in typedef type (set by Sema).
  QualType ObjCProtoType;
  const RecordType *ProtoStructType;

  /// ObjCClassType - another pseudo built-in typedef type (set by Sema).
  QualType ObjCClassType;
  const RecordType *ClassStructType;
  
  QualType ObjCConstantStringType;
  RecordDecl *CFConstantStringTypeDecl;

  RecordDecl *ObjCFastEnumerationStateTypeDecl;
  
  TranslationUnitDecl *TUDecl;

  /// SourceMgr - The associated SourceManager object.
  SourceManager &SourceMgr;
  
  /// LangOpts - The language options used to create the AST associated with
  ///  this ASTContext object.
  LangOptions LangOpts;

  /// MallocAlloc/BumpAlloc - The allocator objects used to create AST objects.
  bool FreeMemory;
  llvm::MallocAllocator MallocAlloc;
  llvm::BumpPtrAllocator BumpAlloc;
public:
  TargetInfo &Target;
  IdentifierTable &Idents;
  SelectorTable &Selectors;
  DeclarationNameTable DeclarationNames;

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

  TranslationUnitDecl *getTranslationUnitDecl() const { return TUDecl; }

  /// This is intentionally not serialized.  It is populated by the
  /// ASTContext ctor, and there are no external pointers/references to
  /// internal variables of BuiltinInfo.
  Builtin::Context BuiltinInfo;

  // Builtin Types.
  QualType VoidTy;
  QualType BoolTy;
  QualType CharTy;
  QualType WCharTy; // [C++ 3.9.1p5], integer type in C99.
  QualType SignedCharTy, ShortTy, IntTy, LongTy, LongLongTy;
  QualType UnsignedCharTy, UnsignedShortTy, UnsignedIntTy, UnsignedLongTy;
  QualType UnsignedLongLongTy;
  QualType FloatTy, DoubleTy, LongDoubleTy;
  QualType FloatComplexTy, DoubleComplexTy, LongDoubleComplexTy;
  QualType VoidPtrTy;
  QualType OverloadTy;
  QualType DependentTy;

  ASTContext(const LangOptions& LOpts, SourceManager &SM, TargetInfo &t,
             IdentifierTable &idents, SelectorTable &sels, 
             bool FreeMemory = true, unsigned size_reserve=0);

  ~ASTContext();
  
  void PrintStats() const;
  const std::vector<Type*>& getTypes() const { return Types; }
  
  //===--------------------------------------------------------------------===//
  //                           Type Constructors
  //===--------------------------------------------------------------------===//
  
  /// getAddSpaceQualType - Return the uniqued reference to the type for an 
  /// address space qualified type with the specified type and address space.  
  /// The resulting type has a union of the qualifiers from T and the address 
  /// space. If T already has an address space specifier, it is silently 
  /// replaced.
  QualType getAddrSpaceQualType(QualType T, unsigned AddressSpace);
  
  /// getObjCGCQualType - Returns the uniqued reference to the type for an
  /// objc gc qualified type. The retulting type has a union of the qualifiers
  /// from T and the gc attribute.
  QualType getObjCGCQualType(QualType T, QualType::GCAttrTypes gcAttr);
  
  /// getComplexType - Return the uniqued reference to the type for a complex
  /// number with the specified element type.
  QualType getComplexType(QualType T);
  
  /// getPointerType - Return the uniqued reference to the type for a pointer to
  /// the specified type.
  QualType getPointerType(QualType T);

  /// getBlockPointerType - Return the uniqued reference to the type for a block
  /// of the specified type.
  QualType getBlockPointerType(QualType T);

  /// getLValueReferenceType - Return the uniqued reference to the type for an
  /// lvalue reference to the specified type.
  QualType getLValueReferenceType(QualType T);

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
                                unsigned EltTypeQuals);
  
  /// getDependentSizedArrayType - Returns a non-unique reference to
  /// the type for a dependently-sized array of the specified element
  /// type. FIXME: We will need these to be uniqued, or at least
  /// comparable, at some point.
  QualType getDependentSizedArrayType(QualType EltTy, Expr *NumElts,
                                      ArrayType::ArraySizeModifier ASM,
                                      unsigned EltTypeQuals);

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

  /// getFunctionNoProtoType - Return a K&R style C function type like 'int()'.
  ///
  QualType getFunctionNoProtoType(QualType ResultTy);
  
  /// getFunctionType - Return a normal function type with a typed argument
  /// list.  isVariadic indicates whether the argument list includes '...'.
  QualType getFunctionType(QualType ResultTy, const QualType *ArgArray,
                           unsigned NumArgs, bool isVariadic,
                           unsigned TypeQuals);

  /// getTypeDeclType - Return the unique reference to the type for
  /// the specified type declaration.
  QualType getTypeDeclType(TypeDecl *Decl, TypeDecl* PrevDecl=0);

  /// getTypedefType - Return the unique reference to the type for the
  /// specified typename decl.
  QualType getTypedefType(TypedefDecl *Decl);
  QualType getObjCInterfaceType(ObjCInterfaceDecl *Decl);

  QualType getTemplateTypeParmType(unsigned Depth, unsigned Index, 
                                   IdentifierInfo *Name = 0);

  QualType getTemplateSpecializationType(TemplateName T,
                                         const TemplateArgument *Args,
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

  /// getObjCQualifiedInterfaceType - Return a 
  /// ObjCQualifiedInterfaceType type for the given interface decl and
  /// the conforming protocol list.
  QualType getObjCQualifiedInterfaceType(ObjCInterfaceDecl *Decl,
                                         ObjCProtocolDecl **ProtocolList,
                                         unsigned NumProtocols);
  
  /// getObjCQualifiedIdType - Return an ObjCQualifiedIdType for a 
  /// given 'id' and conforming protocol list.
  QualType getObjCQualifiedIdType(ObjCProtocolDecl **ProtocolList, 
                                  unsigned NumProtocols);
                                  

  /// getTypeOfType - GCC extension.
  QualType getTypeOfExprType(Expr *e);
  QualType getTypeOfType(QualType t);
  
  /// getTagDeclType - Return the unique reference to the type for the
  /// specified TagDecl (struct/union/class/enum) decl.
  QualType getTagDeclType(TagDecl *Decl);
  
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
  
  // This setter/getter represents the ObjC type for an NSConstantString.
  void setObjCConstantStringInterface(ObjCInterfaceDecl *Decl);
  QualType getObjCConstantStringInterface() const { 
    return ObjCConstantStringType; 
  }

  //// This gets the struct used to keep track of fast enumerations.
  QualType getObjCFastEnumerationStateType();
  
  /// getObjCEncodingForType - Emit the ObjC type encoding for the
  /// given type into \arg S. If \arg NameFields is specified then
  /// record field names are also encoded.
  void getObjCEncodingForType(QualType t, std::string &S, 
                              FieldDecl *Field=NULL) const;

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
  
  /// getObjCEncodingTypeSize returns size of type for objective-c encoding
  /// purpose.
  int getObjCEncodingTypeSize(QualType t);
    
  /// This setter/getter represents the ObjC 'id' type. It is setup lazily, by
  /// Sema.  id is always a (typedef for a) pointer type, a pointer to a struct.
  QualType getObjCIdType() const { return ObjCIdType; }
  void setObjCIdType(TypedefDecl *Decl);
  
  void setObjCSelType(TypedefDecl *Decl);
  QualType getObjCSelType() const { return ObjCSelType; }
  
  void setObjCProtoType(QualType QT);
  QualType getObjCProtoType() const { return ObjCProtoType; }
  
  /// This setter/getter repreents the ObjC 'Class' type. It is setup lazily, by
  /// Sema.  'Class' is always a (typedef for a) pointer type, a pointer to a
  /// struct.
  QualType getObjCClassType() const { return ObjCClassType; }
  void setObjCClassType(TypedefDecl *Decl);
  
  void setBuiltinVaListType(QualType T);
  QualType getBuiltinVaListType() const { return BuiltinVaListType; }

  QualType getFixedWidthIntType(unsigned Width, bool Signed);

  TemplateName getQualifiedTemplateName(NestedNameSpecifier *NNS, 
                                        bool TemplateKeyword,
                                        TemplateDecl *Template);

  TemplateName getDependentTemplateName(NestedNameSpecifier *NNS, 
                                        const IdentifierInfo *Name);

private:
  QualType getFromTargetType(unsigned Type) const;

  //===--------------------------------------------------------------------===//
  //                         Type Predicates.
  //===--------------------------------------------------------------------===//
 
public:
  /// isObjCObjectPointerType - Returns true if type is an Objective-C pointer
  /// to an object type.  This includes "id" and "Class" (two 'special' pointers
  /// to struct), Interface* (pointer to ObjCInterfaceType) and id<P> (qualified
  /// ID type).
  bool isObjCObjectPointerType(QualType Ty) const;

  /// getObjCGCAttr - Returns one of GCNone, Weak or Strong objc's
  /// garbage collection attribute.
  ///
  QualType::GCAttrTypes getObjCGCAttrKind(const QualType &Ty) const;
  
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
  
  const ASTRecordLayout &getASTObjCInterfaceLayout(const ObjCInterfaceDecl *D);
  const RecordDecl *addRecordToClass(const ObjCInterfaceDecl *D);
  void CollectObjCIvars(const ObjCInterfaceDecl *OI,
                        llvm::SmallVectorImpl<FieldDecl*> &Fields) const;
  const FieldDecl *getFieldDecl(const ObjCIvarRefExpr *MRef) {
    llvm::DenseMap<const ObjCIvarRefExpr *, const FieldDecl*>::iterator I 
      = ASTFieldForIvarRef.find(MRef);
    assert (I != ASTFieldForIvarRef.end()  && "Unable to find field_decl");
    return I->second;
  }
  void setFieldDecl(const ObjCInterfaceDecl *OI,
                    const ObjCIvarDecl *Ivar,
                    const ObjCIvarRefExpr *MRef);
  //===--------------------------------------------------------------------===//
  //                            Type Operators
  //===--------------------------------------------------------------------===//
  
  /// getCanonicalType - Return the canonical (structural) type corresponding to
  /// the specified potentially non-canonical type.  The non-canonical version
  /// of a type may have many "decorated" versions of types.  Decorators can
  /// include typedefs, 'typeof' operators, etc. The returned type is guaranteed
  /// to be free of any of these, allowing two canonical types to be compared
  /// for exact equality with a simple pointer comparison.
  QualType getCanonicalType(QualType T);
  const Type *getCanonicalType(const Type *T) {
    return T->getCanonicalTypeInternal().getTypePtr();
  }

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

  /// \brief Retrieves the "canonical" declaration of the given tag
  /// declaration.
  ///
  /// The canonical declaration for the given tag declaration is
  /// either the definition of the tag (if it is a complete type) or
  /// the first declaration of that tag.
  TagDecl *getCanonicalDecl(TagDecl *Tag) {
    QualType T = getTagDeclType(Tag);
    return cast<TagDecl>(cast<TagType>(T.getTypePtr()->CanonicalType)
                           ->getDecl());
  }

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

  /// getBaseElementType - Returns the innermost element type of a variable
  /// length array type. For example, will return "int" for int[m][n]
  QualType getBaseElementType(const VariableArrayType *VAT);
  
  /// getArrayDecayedType - Return the properly qualified result of decaying the
  /// specified array type to a pointer.  This operation is non-trivial when
  /// handling typedefs etc.  The canonical type of "T" must be an array type,
  /// this returns a pointer to a properly qualified element of the array.
  ///
  /// See C99 6.7.5.3p7 and C99 6.3.2.1p3.
  QualType getArrayDecayedType(QualType T);
  
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
  bool typesAreBlockCompatible(QualType lhs, QualType rhs);
  
  bool isObjCIdType(QualType T) const {
    return T == ObjCIdType;
  }
  bool isObjCIdStructType(QualType T) const {
    if (!IdStructType) // ObjC isn't enabled
      return false;
    return T->getAsStructureType() == IdStructType;
  }
  bool isObjCClassType(QualType T) const {
    return T == ObjCClassType;
  }
  bool isObjCClassStructType(QualType T) const {
    if (!ClassStructType) // ObjC isn't enabled
      return false;
    return T->getAsStructureType() == ClassStructType;
  }
  bool isObjCSelType(QualType T) const {
    assert(SelStructType && "isObjCSelType used before 'SEL' type is built");
    return T->getAsStructureType() == SelStructType;
  }

  // Check the safety of assignment from LHS to RHS
  bool canAssignObjCInterfaces(const ObjCInterfaceType *LHS, 
                               const ObjCInterfaceType *RHS);
  bool areComparableObjCPointerTypes(QualType LHS, QualType RHS);

  // Functions for calculating composite types
  QualType mergeTypes(QualType, QualType);
  QualType mergeFunctionTypes(QualType, QualType);

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
  //                    Serialization
  //===--------------------------------------------------------------------===//

  void EmitASTBitcodeBuffer(std::vector<unsigned char> &Buffer) const;
  static ASTContext *ReadASTBitcodeBuffer(llvm::MemoryBuffer &MBuffer,
                                          FileManager &FMgr);  

  void Emit(llvm::Serializer& S) const;
  static ASTContext *Create(llvm::Deserializer& D);  
  
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

private:
  ASTContext(const ASTContext&); // DO NOT IMPLEMENT
  void operator=(const ASTContext&); // DO NOT IMPLEMENT
  
  void InitBuiltinTypes();
  void InitBuiltinType(QualType &R, BuiltinType::Kind K);
  
  // Return the ObjC type encoding for a given type.
  void getObjCEncodingForTypeImpl(QualType t, std::string &S, 
                                  bool ExpandPointedToStructures,
                                  bool ExpandStructures,
                                  FieldDecl *Field,
                                  bool OutermostType = false,
                                  bool EncodingProperty = false) const;
                                  
};

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
                          size_t Alignment = 16) throw () {
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
