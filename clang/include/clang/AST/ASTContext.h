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

#include "clang/Basic/LangOptions.h"
#include "clang/AST/Builtins.h"
#include "clang/AST/DeclBase.h"
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
  class ASTRecordLayout;
  class Expr;
  class IdentifierTable;
  class TargetInfo;
  class SelectorTable;
  class SourceManager;
  // Decls
  class Decl;
  class ObjCPropertyDecl;
  class RecordDecl;
  class TagDecl;
  class TranslationUnitDecl;
  class TypeDecl;
  class TypedefDecl;

/// ASTContext - This class holds long-lived AST nodes (such as types and
/// decls) that can be referred to throughout the semantic analysis of a file.
class ASTContext {  
  std::vector<Type*> Types;
  llvm::FoldingSet<ASQualType> ASQualTypes;
  llvm::FoldingSet<ComplexType> ComplexTypes;
  llvm::FoldingSet<PointerType> PointerTypes;
  llvm::FoldingSet<BlockPointerType> BlockPointerTypes;
  llvm::FoldingSet<ReferenceType> ReferenceTypes;
  llvm::FoldingSet<ConstantArrayType> ConstantArrayTypes;
  llvm::FoldingSet<IncompleteArrayType> IncompleteArrayTypes;
  std::vector<VariableArrayType*> VariableArrayTypes;
  llvm::FoldingSet<VectorType> VectorTypes;
  llvm::FoldingSet<FunctionTypeNoProto> FunctionTypeNoProtos;
  llvm::FoldingSet<FunctionTypeProto> FunctionTypeProtos;
  llvm::FoldingSet<ObjCQualifiedInterfaceType> ObjCQualifiedInterfaceTypes;
  llvm::FoldingSet<ObjCQualifiedIdType> ObjCQualifiedIdTypes;
  /// ASTRecordLayouts - A cache mapping from RecordDecls to ASTRecordLayouts.
  ///  This is lazily created.  This is intentionally not serialized.
  llvm::DenseMap<const RecordDecl*, const ASTRecordLayout*> ASTRecordLayouts;
  llvm::DenseMap<const ObjCInterfaceDecl*, 
                 const ASTRecordLayout*> ASTObjCInterfaces;
  
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

  /// Allocator - The allocator object used to create AST objects.
  llvm::MallocAllocator Allocator;

public:
  TargetInfo &Target;
  IdentifierTable &Idents;
  SelectorTable &Selectors;
  
  SourceManager& getSourceManager() { return SourceMgr; }
  llvm::MallocAllocator &getAllocator() { return Allocator; }  
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
  QualType WCharTy; // [C++ 3.9.1p5]
  QualType SignedCharTy, ShortTy, IntTy, LongTy, LongLongTy;
  QualType UnsignedCharTy, UnsignedShortTy, UnsignedIntTy, UnsignedLongTy;
  QualType UnsignedLongLongTy;
  QualType FloatTy, DoubleTy, LongDoubleTy;
  QualType FloatComplexTy, DoubleComplexTy, LongDoubleComplexTy;
  QualType VoidPtrTy;
  QualType OverloadTy;

  ASTContext(const LangOptions& LOpts, SourceManager &SM, TargetInfo &t,
             IdentifierTable &idents, SelectorTable &sels,
             unsigned size_reserve=0);

  ~ASTContext();
  
  void PrintStats() const;
  const std::vector<Type*>& getTypes() const { return Types; }
  
  //===--------------------------------------------------------------------===//
  //                           Type Constructors
  //===--------------------------------------------------------------------===//
  
  /// getASQualType - Return the uniqued reference to the type for an address
  /// space qualified type with the specified type and address space.  The
  /// resulting type has a union of the qualifiers from T and the address space.
  // If T already has an address space specifier, it is silently replaced.
  QualType getASQualType(QualType T, unsigned AddressSpace);
  
  /// getComplexType - Return the uniqued reference to the type for a complex
  /// number with the specified element type.
  QualType getComplexType(QualType T);
  
  /// getPointerType - Return the uniqued reference to the type for a pointer to
  /// the specified type.
  QualType getPointerType(QualType T);

  /// getBlockPointerType - Return the uniqued reference to the type for a block
  /// of the specified type.
  QualType getBlockPointerType(QualType T);
  
  /// getReferenceType - Return the uniqued reference to the type for a
  /// reference to the specified type.
  QualType getReferenceType(QualType T);
  
  /// getVariableArrayType - Returns a non-unique reference to the type for a
  /// variable array of the specified element type.
  QualType getVariableArrayType(QualType EltTy, Expr *NumElts,
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

  /// getFunctionTypeNoProto - Return a K&R style C function type like 'int()'.
  ///
  QualType getFunctionTypeNoProto(QualType ResultTy);
  
  /// getFunctionType - Return a normal function type with a typed argument
  /// list.  isVariadic indicates whether the argument list includes '...'.
  QualType getFunctionType(QualType ResultTy, const QualType *ArgArray,
                           unsigned NumArgs, bool isVariadic);

  /// getTypeDeclType - Return the unique reference to the type for
  /// the specified type declaration.
  QualType getTypeDeclType(TypeDecl *Decl, TypeDecl* PrevDecl=0);

  /// getTypedefType - Return the unique reference to the type for the
  /// specified typename decl.
  QualType getTypedefType(TypedefDecl *Decl);
  QualType getObjCInterfaceType(ObjCInterfaceDecl *Decl);
  
  /// getObjCQualifiedInterfaceType - Return a 
  /// ObjCQualifiedInterfaceType type for the given interface decl and
  /// the conforming protocol list.
  QualType getObjCQualifiedInterfaceType(ObjCInterfaceDecl *Decl,
             ObjCProtocolDecl **ProtocolList, unsigned NumProtocols);
  
  /// getObjCQualifiedIdType - Return an ObjCQualifiedIdType for a 
  /// given 'id' and conforming protocol list.
  QualType getObjCQualifiedIdType(ObjCProtocolDecl **ProtocolList, 
                                  unsigned NumProtocols);
                                  

  /// getTypeOfType - GCC extension.
  QualType getTypeOfExpr(Expr *e);
  QualType getTypeOfType(QualType t);
  
  /// getTagDeclType - Return the unique reference to the type for the
  /// specified TagDecl (struct/union/class/enum) decl.
  QualType getTagDeclType(TagDecl *Decl);
  
  /// getSizeType - Return the unique type for "size_t" (C99 7.17), defined
  /// in <stddef.h>. The sizeof operator requires this (C99 6.5.3.4p4).
  QualType getSizeType() const;

  /// getWCharType - Return the unique type for "wchar_t" (C99 7.17), defined
  /// in <stddef.h>. Wide strings require this (C99 6.4.5p5).
  QualType getWCharType() const;

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
                              bool NameFields=false) const;
  
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
    
  //===--------------------------------------------------------------------===//
  //                         Type Predicates.
  //===--------------------------------------------------------------------===//
  
  /// isObjCObjectPointerType - Returns true if type is an Objective-C pointer
  /// to an object type.  This includes "id" and "Class" (two 'special' pointers
  /// to struct), Interface* (pointer to ObjCInterfaceType) and id<P> (qualified
  /// ID type).
  bool isObjCObjectPointerType(QualType Ty) const;
    
  //===--------------------------------------------------------------------===//
  //                         Type Sizing and Analysis
  //===--------------------------------------------------------------------===//
  
  /// getFloatTypeSemantics - Return the APFloat 'semantics' for the specified
  /// scalar floating point type.
  const llvm::fltSemantics &getFloatTypeSemantics(QualType T) const;
  
  /// getTypeInfo - Get the size and alignment of the specified complete type in
  /// bits.
  std::pair<uint64_t, unsigned> getTypeInfo(QualType T);
  
  /// getTypeSize - Return the size of the specified type, in bits.  This method
  /// does not work on incomplete types.
  uint64_t getTypeSize(QualType T) {
    return getTypeInfo(T).first;
  }
  
  /// getTypeAlign - Return the alignment of the specified type, in bits.  This
  /// method does not work on incomplete types.
  unsigned getTypeAlign(QualType T) {
    return getTypeInfo(T).second;
  }
  
  /// getASTRecordLayout - Get or compute information about the layout of the
  /// specified record (struct/union/class), which indicates its size and field
  /// position information.
  const ASTRecordLayout &getASTRecordLayout(const RecordDecl *D);
  
  const ASTRecordLayout &getASTObjCInterfaceLayout(const ObjCInterfaceDecl *D);
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

  //===--------------------------------------------------------------------===//
  //                    Type Compatibility Predicates
  //===--------------------------------------------------------------------===//
                                             
  /// Compatibility predicates used to check assignment expressions.
  bool typesAreCompatible(QualType, QualType); // C99 6.2.7p1
  bool typesAreBlockCompatible(QualType lhs, QualType rhs);
  
  bool isObjCIdType(QualType T) const {
    if (!IdStructType) // ObjC isn't enabled
      return false;
    return T->getAsStructureType() == IdStructType;
  }
  bool isObjCClassType(QualType T) const {
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

  void Emit(llvm::Serializer& S) const;
  static ASTContext* Create(llvm::Deserializer& D);  
  
private:
  ASTContext(const ASTContext&); // DO NOT IMPLEMENT
  void operator=(const ASTContext&); // DO NOT IMPLEMENT
  
  void InitBuiltinTypes();
  void InitBuiltinType(QualType &R, BuiltinType::Kind K);
  
  /// setRecordDefinition - Used by RecordDecl::defineBody to inform ASTContext
  ///  about which RecordDecl serves as the definition of a particular
  ///  struct/union/class.  This will eventually be used by enums as well.
  void setTagDefinition(TagDecl* R);
  friend class RecordDecl;

  // Return the ObjC type encoding for a given type.
  void getObjCEncodingForTypeImpl(QualType t, std::string &S, 
                                  bool ExpandPointedToStructures,
                                  bool ExpandStructures,
                                  bool NameFields) const;
  
};
  
}  // end namespace clang

#endif
