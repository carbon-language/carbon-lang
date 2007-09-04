//===--- Type.h - C Language Family Type Representation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Type interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TYPE_H
#define LLVM_CLANG_AST_TYPE_H

#include "llvm/Support/Casting.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/APSInt.h"

using llvm::isa;
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;

namespace clang {
  class ASTContext;
  class Type;
  class TypedefDecl;
  class TagDecl;
  class RecordDecl;
  class EnumDecl;
  class Expr;
  class SourceLocation;
  class PointerType;
  class ReferenceType;
  class VectorType;
  class ArrayType;
  class ConstantArrayType;
  class VariableArrayType;
  class RecordType;
  class ComplexType;
  class TagType;
  class FunctionType;
  class OCUVectorType;
  class BuiltinType;
  
/// QualType - For efficiency, we don't store CVR-qualified types as nodes on
/// their own: instead each reference to a type stores the qualifiers.  This
/// greatly reduces the number of nodes we need to allocate for types (for
/// example we only need one for 'int', 'const int', 'volatile int',
/// 'const volatile int', etc).
///
/// As an added efficiency bonus, instead of making this a pair, we just store
/// the three bits we care about in the low bits of the pointer.  To handle the
/// packing/unpacking, we make QualType be a simple wrapper class that acts like
/// a smart pointer.
class QualType {
  uintptr_t ThePtr;
public:
  enum TQ {   // NOTE: These flags must be kept in sync with DeclSpec::TQ.
    Const    = 0x1,
    Restrict = 0x2,
    Volatile = 0x4,
    CVRFlags = Const|Restrict|Volatile
  };
  
  QualType() : ThePtr(0) {}
  
  QualType(Type *Ptr, unsigned Quals) {
    assert((Quals & ~CVRFlags) == 0 && "Invalid type qualifiers!");
    ThePtr = reinterpret_cast<uintptr_t>(Ptr);
    assert((ThePtr & CVRFlags) == 0 && "Type pointer not 8-byte aligned?");
    ThePtr |= Quals;
  }

  static QualType getFromOpaquePtr(void *Ptr) {
    QualType T;
    T.ThePtr = reinterpret_cast<uintptr_t>(Ptr);
    return T;
  }
  
  unsigned getQualifiers() const {
    return ThePtr & CVRFlags;
  }
  Type *getTypePtr() const {
    return reinterpret_cast<Type*>(ThePtr & ~CVRFlags);
  }
  
  void *getAsOpaquePtr() const {
    return reinterpret_cast<void*>(ThePtr);
  }
  
  Type &operator*() const {
    return *getTypePtr();
  }

  Type *operator->() const {
    return getTypePtr();
  }
  
  /// isNull - Return true if this QualType doesn't point to a type yet.
  bool isNull() const {
    return ThePtr == 0;
  }

  bool isConstQualified() const {
    return (ThePtr & Const) ? true : false;
  }
  bool isVolatileQualified() const {
    return (ThePtr & Volatile) ? true : false;
  }
  bool isRestrictQualified() const {
    return (ThePtr & Restrict) ? true : false;
  }
  
  /// addConst/addVolatile/addRestrict - add the specified type qual to this
  /// QualType.
  void addConst()    { ThePtr |= Const; }
  void addVolatile() { ThePtr |= Volatile; }
  void addRestrict() { ThePtr |= Restrict; }

  QualType getQualifiedType(unsigned TQs) const {
    return QualType(getTypePtr(), TQs);
  }
  
  QualType getUnqualifiedType() const {
    return QualType(getTypePtr(), 0);
  }
  
  /// operator==/!= - Indicate whether the specified types and qualifiers are
  /// identical.
  bool operator==(const QualType &RHS) const {
    return ThePtr == RHS.ThePtr;
  }
  bool operator!=(const QualType &RHS) const {
    return ThePtr != RHS.ThePtr;
  }
  std::string getAsString() const {
    std::string S;
    getAsStringInternal(S);
    return S;
  }
  void getAsStringInternal(std::string &Str) const;
  
  void dump(const char *s = 0) const;

  /// getCanonicalType - Return the canonical version of this type, with the
  /// appropriate type qualifiers on it.
  inline QualType getCanonicalType() const;
  
private:
};

} // end clang.

namespace llvm {
/// Implement simplify_type for QualType, so that we can dyn_cast from QualType
/// to a specific Type class.
template<> struct simplify_type<const ::clang::QualType> {
  typedef ::clang::Type* SimpleType;
  static SimpleType getSimplifiedValue(const ::clang::QualType &Val) {
    return Val.getTypePtr();
  }
};
template<> struct simplify_type< ::clang::QualType>
  : public simplify_type<const ::clang::QualType> {};
}

namespace clang {

/// Type - This is the base class of the type hierarchy.  A central concept
/// with types is that each type always has a canonical type.  A canonical type
/// is the type with any typedef names stripped out of it or the types it
/// references.  For example, consider:
///
///  typedef int  foo;
///  typedef foo* bar;
///    'int *'    'foo *'    'bar'
///
/// There will be a Type object created for 'int'.  Since int is canonical, its
/// canonicaltype pointer points to itself.  There is also a Type for 'foo' (a
/// TypeNameType).  Its CanonicalType pointer points to the 'int' Type.  Next
/// there is a PointerType that represents 'int*', which, like 'int', is
/// canonical.  Finally, there is a PointerType type for 'foo*' whose canonical
/// type is 'int*', and there is a TypeNameType for 'bar', whose canonical type
/// is also 'int*'.
///
/// Non-canonical types are useful for emitting diagnostics, without losing
/// information about typedefs being used.  Canonical types are useful for type
/// comparisons (they allow by-pointer equality tests) and useful for reasoning
/// about whether something has a particular form (e.g. is a function type),
/// because they implicitly, recursively, strip all typedefs out of a type.
///
/// Types, once created, are immutable.
///
class Type {
public:
  enum TypeClass {
    Builtin, Complex, Pointer, Reference, 
    ConstantArray, VariableArray, 
    Vector, OCUVector,
    FunctionNoProto, FunctionProto,
    TypeName, Tagged, 
    TypeOfExp, TypeOfTyp // GNU typeof extension.
  };
private:
  QualType CanonicalType;

  /// TypeClass bitfield - Enum that specifies what subclass this belongs to.
  /// Note that this should stay at the end of the ivars for Type so that
  /// subclasses can pack their bitfields into the same word.
  TypeClass TC : 4;
protected:
  Type(TypeClass tc, QualType Canonical)
    : CanonicalType(Canonical.isNull() ? QualType(this,0) : Canonical), TC(tc){}
  virtual ~Type();
  friend class ASTContext;
public:
  TypeClass getTypeClass() const { return TC; }
  
  bool isCanonical() const { return CanonicalType.getTypePtr() == this; }

  /// Types are partitioned into 3 broad categories (C99 6.2.5p1): 
  /// object types, function types, and incomplete types.
  
  /// isObjectType - types that fully describe objects. An object is a region
  /// of memory that can be examined and stored into (H&S).
  bool isObjectType() const;

  /// isIncompleteType - Return true if this is an incomplete type.
  /// A type that can describe objects, but which lacks information needed to
  /// determine its size (e.g. void, or a fwd declared struct). Clients of this
  /// routine will need to determine if the size is actually required.  
  bool isIncompleteType() const;
  
  /// isVariablyModifiedType (C99 6.7.5.2p2) - Return true for variable array
  /// types that have a non-constant expression. This does not include "[]".
  bool isVariablyModifiedType() const;
  
  /// Helper methods to distinguish type categories. All type predicates
  /// operate on the canonical type, ignoring typedefs.
  bool isIntegerType() const;     // C99 6.2.5p17 (int, char, bool, enum)
  bool isEnumeralType() const;
  bool isBooleanType() const;
  bool isCharType() const;
  
  /// Floating point categories.
  bool isRealFloatingType() const; // C99 6.2.5p10 (float, double, long double)
  bool isComplexType() const;      // C99 6.2.5p11 (complex)
  bool isFloatingType() const;     // C99 6.2.5p11 (real floating + complex)
  bool isRealType() const;         // C99 6.2.5p17 (real floating + integer)
  bool isArithmeticType() const;   // C99 6.2.5p18 (integer + floating)
  bool isVoidType() const;         // C99 6.2.5p19
  bool isDerivedType() const;      // C99 6.2.5p20
  bool isScalarType() const;       // C99 6.2.5p21 (arithmetic + pointers)
  bool isAggregateType() const;    // C99 6.2.5p21 (arrays, structures)
  
  // Type Predicates: Check to see if this type is structurally the specified
  // type, ignoring typedefs.
  bool isFunctionType() const;
  bool isPointerType() const;
  bool isReferenceType() const;
  bool isArrayType() const;
  bool isRecordType() const;
  bool isStructureType() const;   
  bool isUnionType() const;
  bool isVectorType() const; // GCC vector type.
  bool isOCUVectorType() const; // OCU vector type.
  
  // Type Checking Functions: Check to see if this type is structurally the
  // specified type, ignoring typedefs, and return a pointer to the best type
  // we can.
  const BuiltinType *getAsBuiltinType() const;   
  const FunctionType *getAsFunctionType() const;   
  const PointerType *getAsPointerType() const;
  const ReferenceType *getAsReferenceType() const;
  const ArrayType *getAsArrayType() const;
  const ConstantArrayType *getAsConstantArrayType() const;
  const VariableArrayType *getAsVariableArrayType() const;
  const VariableArrayType *getAsVariablyModifiedType() const;
  const RecordType *getAsRecordType() const;
  const RecordType *getAsStructureType() const;   
  const RecordType *getAsUnionType() const;
  const VectorType *getAsVectorType() const; // GCC vector type.
  const ComplexType *getAsComplexType() const;
  const OCUVectorType *getAsOCUVectorType() const; // OCU vector type.
  
  /// More type predicates useful for type checking/promotion
  bool isPromotableIntegerType() const; // C99 6.3.1.1p2

  /// isSignedIntegerType - Return true if this is an integer type that is
  /// signed, according to C99 6.2.5p4 [char, signed char, short, int, long..],
  /// an enum decl which has a signed representation, or a vector of signed
  /// integer element type.
  bool isSignedIntegerType() const;

  /// isUnsignedIntegerType - Return true if this is an integer type that is
  /// unsigned, according to C99 6.2.5p6 [which returns true for _Bool], an enum
  /// decl which has an unsigned representation, or a vector of unsigned integer
  /// element type.
  bool isUnsignedIntegerType() const;

  /// isConstantSizeType - Return true if this is not a variable sized type,
  /// according to the rules of C99 6.7.5p3.  If Loc is non-null, it is set to
  /// the location of the subexpression that makes it a vla type.  It is not
  /// legal to call this on incomplete types.
  bool isConstantSizeType(ASTContext &Ctx, SourceLocation *Loc = 0) const;

  /// Compatibility predicates used to check assignment expressions.
  static bool typesAreCompatible(QualType, QualType); // C99 6.2.7p1
  static bool tagTypesAreCompatible(QualType, QualType); // C99 6.2.7p1
  static bool pointerTypesAreCompatible(QualType, QualType);  // C99 6.7.5.1p2
  static bool referenceTypesAreCompatible(QualType, QualType); // C++ 5.17p6
  static bool functionTypesAreCompatible(QualType, QualType); // C99 6.7.5.3p15
  static bool arrayTypesAreCompatible(QualType, QualType); // C99 6.7.5.2p6
  static bool builtinTypesAreCompatible(QualType, QualType);
private:  
  QualType getCanonicalTypeInternal() const { return CanonicalType; }
  friend class QualType;
public:
  virtual void getAsStringInternal(std::string &InnerString) const = 0;
  
  static bool classof(const Type *) { return true; }
};

/// BuiltinType - This class is used for builtin types like 'int'.  Builtin
/// types are always canonical and have a literal name field.
class BuiltinType : public Type {
public:
  enum Kind {
    Void,
    
    Bool,     // This is bool and/or _Bool.
    Char_U,   // This is 'char' for targets where char is unsigned.
    UChar,    // This is explicitly qualified unsigned char.
    UShort,
    UInt,
    ULong,
    ULongLong,
    
    Char_S,   // This is 'char' for targets where char is signed.
    SChar,    // This is explicitly qualified signed char.
    Short,
    Int,
    Long,
    LongLong,
    
    Float, Double, LongDouble
  };
private:
  Kind TypeKind;
public:
  BuiltinType(Kind K) : Type(Builtin, QualType()), TypeKind(K) {}
  
  Kind getKind() const { return TypeKind; }
  const char *getName() const;
  
  virtual void getAsStringInternal(std::string &InnerString) const;
  
  static bool classof(const Type *T) { return T->getTypeClass() == Builtin; }
  static bool classof(const BuiltinType *) { return true; }
};

/// ComplexType - C99 6.2.5p11 - Complex values.  This supports the C99 complex
/// types (_Complex float etc) as well as the GCC integer complex extensions.
///
class ComplexType : public Type, public llvm::FoldingSetNode {
  QualType ElementType;
  ComplexType(QualType Element, QualType CanonicalPtr) :
    Type(Complex, CanonicalPtr), ElementType(Element) {
  }
  friend class ASTContext;  // ASTContext creates these.
public:
  QualType getElementType() const { return ElementType; }
  
  virtual void getAsStringInternal(std::string &InnerString) const;
  
  
  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getElementType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Element) {
    ID.AddPointer(Element.getAsOpaquePtr());
  }
  
  static bool classof(const Type *T) { return T->getTypeClass() == Complex; }
  static bool classof(const ComplexType *) { return true; }
};


/// PointerType - C99 6.7.5.1 - Pointer Declarators.
///
class PointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;
  PointerType(QualType Pointee, QualType CanonicalPtr) :
    Type(Pointer, CanonicalPtr), PointeeType(Pointee) {
  }
  friend class ASTContext;  // ASTContext creates these.
public:
    
  QualType getPointeeType() const { return PointeeType; }
  
  virtual void getAsStringInternal(std::string &InnerString) const;
  
  
  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getPointeeType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Pointee) {
    ID.AddPointer(Pointee.getAsOpaquePtr());
  }
  
  static bool classof(const Type *T) { return T->getTypeClass() == Pointer; }
  static bool classof(const PointerType *) { return true; }
};

/// ReferenceType - C++ 8.3.2 - Reference Declarators.
///
class ReferenceType : public Type, public llvm::FoldingSetNode {
  QualType ReferenceeType;
  ReferenceType(QualType Referencee, QualType CanonicalRef) :
    Type(Reference, CanonicalRef), ReferenceeType(Referencee) {
  }
  friend class ASTContext;  // ASTContext creates these.
public:
  virtual void getAsStringInternal(std::string &InnerString) const;

  QualType getReferenceeType() const { return ReferenceeType; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getReferenceeType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Referencee) {
    ID.AddPointer(Referencee.getAsOpaquePtr());
  }

  static bool classof(const Type *T) { return T->getTypeClass() == Reference; }
  static bool classof(const ReferenceType *) { return true; }
};

/// ArrayType - C99 6.7.5.2 - Array Declarators.
///
class ArrayType : public Type {
public:
  /// ArraySizeModifier - Capture whether this is a normal array (e.g. int X[4])
  /// an array with a static size (e.g. int X[static 4]), or with a star size
  /// (e.g. int X[*]). 'static' is only allowed on function parameters.
  enum ArraySizeModifier {
    Normal, Static, Star
  };
private:
  /// ElementType - The element type of the array.
  QualType ElementType;
  
  /// NOTE: These fields are packed into the bitfields space in the Type class.
  ArraySizeModifier SizeModifier : 2;
  
  /// IndexTypeQuals - Capture qualifiers in declarations like:
  /// 'int X[static restrict 4]'. For function parameters only.
  unsigned IndexTypeQuals : 3;
  
protected:
  ArrayType(TypeClass tc, QualType et, QualType can,
            ArraySizeModifier sm, unsigned tq)
    : Type(tc, can), ElementType(et), SizeModifier(sm), IndexTypeQuals(tq) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  QualType getElementType() const { return ElementType; }
  ArraySizeModifier getSizeModifier() const { return SizeModifier; }
  unsigned getIndexTypeQualifier() const { return IndexTypeQuals; }
  
  QualType getBaseType() const {
    const ArrayType *AT;
    QualType ElmtType = getElementType();
    // If we have a multi-dimensional array, navigate to the base type.
    while ((AT = ElmtType->getAsArrayType()))
      ElmtType = AT->getElementType();
    return ElmtType;
  }
  static bool classof(const Type *T) {
    return T->getTypeClass() == ConstantArray ||
           T->getTypeClass() == VariableArray;
  }
  static bool classof(const ArrayType *) { return true; }
};

class ConstantArrayType : public ArrayType, public llvm::FoldingSetNode {
  llvm::APInt Size; // Allows us to unique the type.
  
  ConstantArrayType(QualType et, QualType can, llvm::APInt sz,
                    ArraySizeModifier sm, unsigned tq)
    : ArrayType(ConstantArray, et, can, sm, tq), Size(sz) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  llvm::APInt getSize() const { return Size; }
  int getMaximumElements() const {
    QualType ElmtType = getElementType();
    int maxElements = getSize().getZExtValue();

    const ConstantArrayType *CAT;
    // If we have a multi-dimensional array, include it's elements.
    while ((CAT = ElmtType->getAsConstantArrayType())) {
      ElmtType = CAT->getElementType();
      maxElements *= CAT->getSize().getZExtValue();
    }
    return maxElements;
  }
  virtual void getAsStringInternal(std::string &InnerString) const;
  
  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getElementType(), getSize());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType ET,
                      llvm::APInt ArraySize) {
    ID.AddPointer(ET.getAsOpaquePtr());
    ID.AddInteger(ArraySize.getZExtValue());
  }
  static bool classof(const Type *T) { 
    return T->getTypeClass() == ConstantArray; 
  }
  static bool classof(const ConstantArrayType *) { return true; }
};

// FIXME: VariableArrayType's aren't uniqued (since expressions aren't).
class VariableArrayType : public ArrayType {
  /// SizeExpr - An assignment expression. VLA's are only permitted within 
  /// a function block. 
  Expr *SizeExpr;
  
  VariableArrayType(QualType et, QualType can, Expr *e,
                    ArraySizeModifier sm, unsigned tq)
    : ArrayType(VariableArray, et, can, sm, tq), SizeExpr(e) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  Expr *getSizeExpr() const { return SizeExpr; }
  
  virtual void getAsStringInternal(std::string &InnerString) const;
  
  static bool classof(const Type *T) { 
    return T->getTypeClass() == VariableArray; 
  }
  static bool classof(const VariableArrayType *) { return true; }
};

/// VectorType - GCC generic vector type. This type is created using
/// __attribute__((vector_size(n)), where "n" specifies the vector size in 
/// bytes. Since the constructor takes the number of vector elements, the 
/// client is responsible for converting the size into the number of elements.
class VectorType : public Type, public llvm::FoldingSetNode {
protected:
  /// ElementType - The element type of the vector.
  QualType ElementType;
  
  /// NumElements - The number of elements in the vector.
  unsigned NumElements;
  
  VectorType(QualType vecType, unsigned nElements, QualType canonType) :
    Type(Vector, canonType), ElementType(vecType), NumElements(nElements) {} 
  VectorType(TypeClass tc, QualType vecType, unsigned nElements, 
    QualType canonType) : Type(tc, canonType), ElementType(vecType), 
    NumElements(nElements) {} 
  friend class ASTContext;  // ASTContext creates these.
public:
    
  QualType getElementType() const { return ElementType; }
  unsigned getNumElements() const { return NumElements; } 

  virtual void getAsStringInternal(std::string &InnerString) const;
  
  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getElementType(), getNumElements(), getTypeClass());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType ElementType, 
                      unsigned NumElements, TypeClass TypeClass) {
    ID.AddPointer(ElementType.getAsOpaquePtr());
    ID.AddInteger(NumElements);
    ID.AddInteger(TypeClass);
  }
  static bool classof(const Type *T) { 
    return T->getTypeClass() == Vector || T->getTypeClass() == OCUVector; 
  }
  static bool classof(const VectorType *) { return true; }
};

/// OCUVectorType - Extended vector type. This type is created using
/// __attribute__((ocu_vector_type(n)), where "n" is the number of elements.
/// Unlike vector_size, ocu_vector_type is only allowed on typedef's. This
/// class enables syntactic extensions, like Vector Components for accessing
/// points, colors, and textures (modeled after OpenGL Shading Language).
class OCUVectorType : public VectorType {
  OCUVectorType(QualType vecType, unsigned nElements, QualType canonType) :
    VectorType(OCUVector, vecType, nElements, canonType) {} 
  friend class ASTContext;  // ASTContext creates these.
public:
  static int getPointAccessorIdx(char c) {
    switch (c) {
    default: return -1;
    case 'x': return 0;
    case 'y': return 1;
    case 'z': return 2;
    case 'w': return 3;
    }
  }
  static int getColorAccessorIdx(char c) {
    switch (c) {
    default: return -1;
    case 'r': return 0;
    case 'g': return 1;
    case 'b': return 2;
    case 'a': return 3;
    }
  }
  static int getTextureAccessorIdx(char c) {
    switch (c) {
    default: return -1;
    case 's': return 0;
    case 't': return 1;
    case 'p': return 2;
    case 'q': return 3;
    }
  };
  
  static int getAccessorIdx(char c) {
    if (int idx = getPointAccessorIdx(c)+1) return idx-1;
    if (int idx = getColorAccessorIdx(c)+1) return idx-1;
    return getTextureAccessorIdx(c);
  }
  
  bool isAccessorWithinNumElements(char c) const {
    if (int idx = getAccessorIdx(c)+1)
      return unsigned(idx-1) < NumElements;
    return false;
  }
  virtual void getAsStringInternal(std::string &InnerString) const;

  static bool classof(const Type *T) { 
    return T->getTypeClass() == OCUVector; 
  }
  static bool classof(const OCUVectorType *) { return true; }
};

/// FunctionType - C99 6.7.5.3 - Function Declarators.  This is the common base
/// class of FunctionTypeNoProto and FunctionTypeProto.
///
class FunctionType : public Type {
  /// SubClassData - This field is owned by the subclass, put here to pack
  /// tightly with the ivars in Type.
  bool SubClassData : 1;
  
  // The type returned by the function.
  QualType ResultType;
protected:
  FunctionType(TypeClass tc, QualType res, bool SubclassInfo,QualType Canonical)
    : Type(tc, Canonical), SubClassData(SubclassInfo), ResultType(res) {}
  bool getSubClassData() const { return SubClassData; }
public:
  
  QualType getResultType() const { return ResultType; }

  
  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionNoProto ||
           T->getTypeClass() == FunctionProto;
  }
  static bool classof(const FunctionType *) { return true; }
};

/// FunctionTypeNoProto - Represents a K&R-style 'int foo()' function, which has
/// no information available about its arguments.
class FunctionTypeNoProto : public FunctionType, public llvm::FoldingSetNode {
  FunctionTypeNoProto(QualType Result, QualType Canonical)
    : FunctionType(FunctionNoProto, Result, false, Canonical) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  // No additional state past what FunctionType provides.
  
  virtual void getAsStringInternal(std::string &InnerString) const;

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getResultType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType ResultType) {
    ID.AddPointer(ResultType.getAsOpaquePtr());
  }
  
  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionNoProto;
  }
  static bool classof(const FunctionTypeNoProto *) { return true; }
};

/// FunctionTypeProto - Represents a prototype with argument type info, e.g.
/// 'int foo(int)' or 'int foo(void)'.  'void' is represented as having no
/// arguments, not as having a single void argument.
class FunctionTypeProto : public FunctionType, public llvm::FoldingSetNode {
  FunctionTypeProto(QualType Result, QualType *ArgArray, unsigned numArgs,
                    bool isVariadic, QualType Canonical)
    : FunctionType(FunctionProto, Result, isVariadic, Canonical),
      NumArgs(numArgs) {
    // Fill in the trailing argument array.
    QualType *ArgInfo = reinterpret_cast<QualType *>(this+1);;
    for (unsigned i = 0; i != numArgs; ++i)
      ArgInfo[i] = ArgArray[i];
  }
  
  /// NumArgs - The number of arguments this function has, not counting '...'.
  unsigned NumArgs;
  
  /// ArgInfo - There is an variable size array after the class in memory that
  /// holds the argument types.
  friend class ASTContext;  // ASTContext creates these.
public:
  unsigned getNumArgs() const { return NumArgs; }
  QualType getArgType(unsigned i) const {
    assert(i < NumArgs && "Invalid argument number!");
    return arg_type_begin()[i];
  }
    
  bool isVariadic() const { return getSubClassData(); }
  
  typedef const QualType *arg_type_iterator;
  arg_type_iterator arg_type_begin() const {
    return reinterpret_cast<const QualType *>(this+1);
  }
  arg_type_iterator arg_type_end() const { return arg_type_begin()+NumArgs; }
  
  virtual void getAsStringInternal(std::string &InnerString) const;

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionProto;
  }
  static bool classof(const FunctionTypeProto *) { return true; }
  
  void Profile(llvm::FoldingSetNodeID &ID);
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Result,
                      arg_type_iterator ArgTys, unsigned NumArgs,
                      bool isVariadic);
};


class TypedefType : public Type {
  TypedefDecl *Decl;
  TypedefType(TypedefDecl *D, QualType can) : Type(TypeName, can), Decl(D) {
    assert(!isa<TypedefType>(can) && "Invalid canonical type");
  }
  friend class ASTContext;  // ASTContext creates these.
public:
  
  TypedefDecl *getDecl() const { return Decl; }
  
  /// LookThroughTypedefs - Return the ultimate type this typedef corresponds to
  /// potentially looking through *all* consequtive typedefs.  This returns the
  /// sum of the type qualifiers, so if you have:
  ///   typedef const int A;
  ///   typedef volatile A B;
  /// looking through the typedefs for B will give you "const volatile A".
  QualType LookThroughTypedefs() const;
    
  virtual void getAsStringInternal(std::string &InnerString) const;

  static bool classof(const Type *T) { return T->getTypeClass() == TypeName; }
  static bool classof(const TypedefType *) { return true; }
};

/// TypeOfExpr (GCC extension).
class TypeOfExpr : public Type {
  Expr *TOExpr;
  TypeOfExpr(Expr *E, QualType can) : Type(TypeOfExp, can), TOExpr(E) {
    assert(!isa<TypedefType>(can) && "Invalid canonical type");
  }
  friend class ASTContext;  // ASTContext creates these.
public:
  Expr *getUnderlyingExpr() const { return TOExpr; }
  
  virtual void getAsStringInternal(std::string &InnerString) const;

  static bool classof(const Type *T) { return T->getTypeClass() == TypeOfExp; }
  static bool classof(const TypeOfExpr *) { return true; }
};

/// TypeOfType (GCC extension).
class TypeOfType : public Type {
  QualType TOType;
  TypeOfType(QualType T, QualType can) : Type(TypeOfTyp, can), TOType(T) {
    assert(!isa<TypedefType>(can) && "Invalid canonical type");
  }
  friend class ASTContext;  // ASTContext creates these.
public:
  QualType getUnderlyingType() const { return TOType; }
  
  virtual void getAsStringInternal(std::string &InnerString) const;

  static bool classof(const Type *T) { return T->getTypeClass() == TypeOfTyp; }
  static bool classof(const TypeOfType *) { return true; }
};

class TagType : public Type {
  TagDecl *Decl;
  TagType(TagDecl *D, QualType can) : Type(Tagged, can), Decl(D) {}
  friend class ASTContext;  // ASTContext creates these.
public:
    
  TagDecl *getDecl() const { return Decl; }
  
  virtual void getAsStringInternal(std::string &InnerString) const;
  
  static bool classof(const Type *T) { return T->getTypeClass() == Tagged; }
  static bool classof(const TagType *) { return true; }
};

/// RecordType - This is a helper class that allows the use of isa/cast/dyncast
/// to detect TagType objects of structs/unions/classes.
class RecordType : public TagType {
  RecordType(); // DO NOT IMPLEMENT
public:
    
  RecordDecl *getDecl() const {
    return reinterpret_cast<RecordDecl*>(TagType::getDecl());
  }
  
  // FIXME: This predicate is a helper to QualType/Type. It needs to 
  // recursively check all fields for const-ness. If any field is declared
  // const, it needs to return false. 
  bool hasConstFields() const { return false; } 
  
  static bool classof(const Type *T);
  static bool classof(const RecordType *) { return true; }
};


// Inline function definitions.

/// getCanonicalType - Return the canonical version of this type, with the
/// appropriate type qualifiers on it.
inline QualType QualType::getCanonicalType() const {
  return QualType(getTypePtr()->getCanonicalTypeInternal().getTypePtr(),
                  getQualifiers() |
                  getTypePtr()->getCanonicalTypeInternal().getQualifiers());
}


inline bool Type::isFunctionType() const {
  return isa<FunctionType>(CanonicalType);
}
inline bool Type::isPointerType() const {
  return isa<PointerType>(CanonicalType);
}
inline bool Type::isReferenceType() const {
  return isa<ReferenceType>(CanonicalType);
}
inline bool Type::isArrayType() const {
  return isa<ArrayType>(CanonicalType);
}
inline bool Type::isRecordType() const {
  return isa<RecordType>(CanonicalType);
}
inline bool Type::isVectorType() const {
  return isa<VectorType>(CanonicalType);
}
inline bool Type::isOCUVectorType() const {
  return isa<OCUVectorType>(CanonicalType);
}

}  // end namespace clang

#endif
