//===--- Type.h - C Language Family Type Representation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Type interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TYPE_H
#define LLVM_CLANG_AST_TYPE_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateName.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/type_traits.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"

using llvm::isa;
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
namespace clang {
  enum {
    TypeAlignmentInBits = 3,
    TypeAlignment = 1 << TypeAlignmentInBits
  };
  class Type;
  class ExtQuals;
  class QualType;
}

namespace llvm {
  template <typename T>
  class PointerLikeTypeTraits;
  template<>
  class PointerLikeTypeTraits< ::clang::Type*> {
  public:
    static inline void *getAsVoidPointer(::clang::Type *P) { return P; }
    static inline ::clang::Type *getFromVoidPointer(void *P) {
      return static_cast< ::clang::Type*>(P);
    }
    enum { NumLowBitsAvailable = clang::TypeAlignmentInBits };
  };
  template<>
  class PointerLikeTypeTraits< ::clang::ExtQuals*> {
  public:
    static inline void *getAsVoidPointer(::clang::ExtQuals *P) { return P; }
    static inline ::clang::ExtQuals *getFromVoidPointer(void *P) {
      return static_cast< ::clang::ExtQuals*>(P);
    }
    enum { NumLowBitsAvailable = clang::TypeAlignmentInBits };
  };

  template <>
  struct isPodLike<clang::QualType> { static const bool value = true; };
}

namespace clang {
  class ASTContext;
  class TypedefDecl;
  class TemplateDecl;
  class TemplateTypeParmDecl;
  class NonTypeTemplateParmDecl;
  class TemplateTemplateParmDecl;
  class TagDecl;
  class RecordDecl;
  class CXXRecordDecl;
  class EnumDecl;
  class FieldDecl;
  class ObjCInterfaceDecl;
  class ObjCProtocolDecl;
  class ObjCMethodDecl;
  class UnresolvedUsingTypenameDecl;
  class Expr;
  class Stmt;
  class SourceLocation;
  class StmtIteratorBase;
  class TemplateArgument;
  class TemplateArgumentLoc;
  class TemplateArgumentListInfo;
  class QualifiedNameType;
  struct PrintingPolicy;

  // Provide forward declarations for all of the *Type classes
#define TYPE(Class, Base) class Class##Type;
#include "clang/AST/TypeNodes.def"

/// Qualifiers - The collection of all-type qualifiers we support.
/// Clang supports five independent qualifiers:
/// * C99: const, volatile, and restrict
/// * Embedded C (TR18037): address spaces
/// * Objective C: the GC attributes (none, weak, or strong)
class Qualifiers {
public:
  enum TQ { // NOTE: These flags must be kept in sync with DeclSpec::TQ.
    Const    = 0x1,
    Restrict = 0x2,
    Volatile = 0x4,
    CVRMask = Const | Volatile | Restrict
  };

  enum GC {
    GCNone = 0,
    Weak,
    Strong
  };

  enum {
    /// The maximum supported address space number.
    /// 24 bits should be enough for anyone.
    MaxAddressSpace = 0xffffffu,

    /// The width of the "fast" qualifier mask.
    FastWidth = 2,

    /// The fast qualifier mask.
    FastMask = (1 << FastWidth) - 1
  };

  Qualifiers() : Mask(0) {}

  static Qualifiers fromFastMask(unsigned Mask) {
    Qualifiers Qs;
    Qs.addFastQualifiers(Mask);
    return Qs;
  }

  static Qualifiers fromCVRMask(unsigned CVR) {
    Qualifiers Qs;
    Qs.addCVRQualifiers(CVR);
    return Qs;
  }

  // Deserialize qualifiers from an opaque representation.
  static Qualifiers fromOpaqueValue(unsigned opaque) {
    Qualifiers Qs;
    Qs.Mask = opaque;
    return Qs;
  }

  // Serialize these qualifiers into an opaque representation.
  unsigned getAsOpaqueValue() const {
    return Mask;
  }

  bool hasConst() const { return Mask & Const; }
  void setConst(bool flag) {
    Mask = (Mask & ~Const) | (flag ? Const : 0);
  }
  void removeConst() { Mask &= ~Const; }
  void addConst() { Mask |= Const; }

  bool hasVolatile() const { return Mask & Volatile; }
  void setVolatile(bool flag) {
    Mask = (Mask & ~Volatile) | (flag ? Volatile : 0);
  }
  void removeVolatile() { Mask &= ~Volatile; }
  void addVolatile() { Mask |= Volatile; }

  bool hasRestrict() const { return Mask & Restrict; }
  void setRestrict(bool flag) {
    Mask = (Mask & ~Restrict) | (flag ? Restrict : 0);
  }
  void removeRestrict() { Mask &= ~Restrict; }
  void addRestrict() { Mask |= Restrict; }

  bool hasCVRQualifiers() const { return getCVRQualifiers(); }
  unsigned getCVRQualifiers() const { return Mask & CVRMask; }
  void setCVRQualifiers(unsigned mask) {
    assert(!(mask & ~CVRMask) && "bitmask contains non-CVR bits");
    Mask = (Mask & ~CVRMask) | mask;
  }
  void removeCVRQualifiers(unsigned mask) {
    assert(!(mask & ~CVRMask) && "bitmask contains non-CVR bits");
    Mask &= ~mask;
  }
  void removeCVRQualifiers() {
    removeCVRQualifiers(CVRMask);
  }
  void addCVRQualifiers(unsigned mask) {
    assert(!(mask & ~CVRMask) && "bitmask contains non-CVR bits");
    Mask |= mask;
  }

  bool hasObjCGCAttr() const { return Mask & GCAttrMask; }
  GC getObjCGCAttr() const { return GC((Mask & GCAttrMask) >> GCAttrShift); }
  void setObjCGCAttr(GC type) {
    Mask = (Mask & ~GCAttrMask) | (type << GCAttrShift);
  }
  void removeObjCGCAttr() { setObjCGCAttr(GCNone); }
  void addObjCGCAttr(GC type) {
    assert(type);
    setObjCGCAttr(type);
  }

  bool hasAddressSpace() const { return Mask & AddressSpaceMask; }
  unsigned getAddressSpace() const { return Mask >> AddressSpaceShift; }
  void setAddressSpace(unsigned space) {
    assert(space <= MaxAddressSpace);
    Mask = (Mask & ~AddressSpaceMask)
         | (((uint32_t) space) << AddressSpaceShift);
  }
  void removeAddressSpace() { setAddressSpace(0); }
  void addAddressSpace(unsigned space) {
    assert(space);
    setAddressSpace(space);
  }

  // Fast qualifiers are those that can be allocated directly
  // on a QualType object.
  bool hasFastQualifiers() const { return getFastQualifiers(); }
  unsigned getFastQualifiers() const { return Mask & FastMask; }
  void setFastQualifiers(unsigned mask) {
    assert(!(mask & ~FastMask) && "bitmask contains non-fast qualifier bits");
    Mask = (Mask & ~FastMask) | mask;
  }
  void removeFastQualifiers(unsigned mask) {
    assert(!(mask & ~FastMask) && "bitmask contains non-fast qualifier bits");
    Mask &= ~mask;
  }
  void removeFastQualifiers() {
    removeFastQualifiers(FastMask);
  }
  void addFastQualifiers(unsigned mask) {
    assert(!(mask & ~FastMask) && "bitmask contains non-fast qualifier bits");
    Mask |= mask;
  }

  /// hasNonFastQualifiers - Return true if the set contains any
  /// qualifiers which require an ExtQuals node to be allocated.
  bool hasNonFastQualifiers() const { return Mask & ~FastMask; }
  Qualifiers getNonFastQualifiers() const {
    Qualifiers Quals = *this;
    Quals.setFastQualifiers(0);
    return Quals;
  }

  /// hasQualifiers - Return true if the set contains any qualifiers.
  bool hasQualifiers() const { return Mask; }
  bool empty() const { return !Mask; }

  /// \brief Add the qualifiers from the given set to this set.
  void addQualifiers(Qualifiers Q) {
    // If the other set doesn't have any non-boolean qualifiers, just
    // bit-or it in.
    if (!(Q.Mask & ~CVRMask))
      Mask |= Q.Mask;
    else {
      Mask |= (Q.Mask & CVRMask);
      if (Q.hasAddressSpace())
        addAddressSpace(Q.getAddressSpace());
      if (Q.hasObjCGCAttr())
        addObjCGCAttr(Q.getObjCGCAttr());
    }
  }

  bool operator==(Qualifiers Other) const { return Mask == Other.Mask; }
  bool operator!=(Qualifiers Other) const { return Mask != Other.Mask; }

  operator bool() const { return hasQualifiers(); }

  Qualifiers &operator+=(Qualifiers R) {
    addQualifiers(R);
    return *this;
  }

  // Union two qualifier sets.  If an enumerated qualifier appears
  // in both sets, use the one from the right.
  friend Qualifiers operator+(Qualifiers L, Qualifiers R) {
    L += R;
    return L;
  }

  std::string getAsString() const;
  std::string getAsString(const PrintingPolicy &Policy) const {
    std::string Buffer;
    getAsStringInternal(Buffer, Policy);
    return Buffer;
  }
  void getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(Mask);
  }

private:

  // bits:     |0 1 2|3 .. 4|5  ..  31|
  //           |C R V|GCAttr|AddrSpace|
  uint32_t Mask;

  static const uint32_t GCAttrMask = 0x18;
  static const uint32_t GCAttrShift = 3;
  static const uint32_t AddressSpaceMask = ~(CVRMask | GCAttrMask);
  static const uint32_t AddressSpaceShift = 5;
};


/// ExtQuals - We can encode up to three bits in the low bits of a
/// type pointer, but there are many more type qualifiers that we want
/// to be able to apply to an arbitrary type.  Therefore we have this
/// struct, intended to be heap-allocated and used by QualType to
/// store qualifiers.
///
/// The current design tags the 'const' and 'restrict' qualifiers in
/// two low bits on the QualType pointer; a third bit records whether
/// the pointer is an ExtQuals node.  'const' was chosen because it is
/// orders of magnitude more common than the other two qualifiers, in
/// both library and user code.  It's relatively rare to see
/// 'restrict' in user code, but many standard C headers are saturated
/// with 'restrict' declarations, so that representing them efficiently
/// is a critical goal of this representation.
class ExtQuals : public llvm::FoldingSetNode {
  // NOTE: changing the fast qualifiers should be straightforward as
  // long as you don't make 'const' non-fast.
  // 1. Qualifiers:
  //    a) Modify the bitmasks (Qualifiers::TQ and DeclSpec::TQ).
  //       Fast qualifiers must occupy the low-order bits.
  //    b) Update Qualifiers::FastWidth and FastMask.
  // 2. QualType:
  //    a) Update is{Volatile,Restrict}Qualified(), defined inline.
  //    b) Update remove{Volatile,Restrict}, defined near the end of
  //       this header.
  // 3. ASTContext:
  //    a) Update get{Volatile,Restrict}Type.

  /// Context - the context to which this set belongs.  We save this
  /// here so that QualifierCollector can use it to reapply extended
  /// qualifiers to an arbitrary type without requiring a context to
  /// be pushed through every single API dealing with qualifiers.
  ASTContext& Context;

  /// BaseType - the underlying type that this qualifies
  const Type *BaseType;

  /// Quals - the immutable set of qualifiers applied by this
  /// node;  always contains extended qualifiers.
  Qualifiers Quals;

public:
  ExtQuals(ASTContext& Context, const Type *Base, Qualifiers Quals)
    : Context(Context), BaseType(Base), Quals(Quals)
  {
    assert(Quals.hasNonFastQualifiers()
           && "ExtQuals created with no fast qualifiers");
    assert(!Quals.hasFastQualifiers()
           && "ExtQuals created with fast qualifiers");
  }

  Qualifiers getQualifiers() const { return Quals; }

  bool hasVolatile() const { return Quals.hasVolatile(); }

  bool hasObjCGCAttr() const { return Quals.hasObjCGCAttr(); }
  Qualifiers::GC getObjCGCAttr() const { return Quals.getObjCGCAttr(); }

  bool hasAddressSpace() const { return Quals.hasAddressSpace(); }
  unsigned getAddressSpace() const { return Quals.getAddressSpace(); }

  const Type *getBaseType() const { return BaseType; }

  ASTContext &getContext() const { return Context; }

public:
  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, getBaseType(), Quals);
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      const Type *BaseType,
                      Qualifiers Quals) {
    assert(!Quals.hasFastQualifiers() && "fast qualifiers in ExtQuals hash!");
    ID.AddPointer(BaseType);
    Quals.Profile(ID);
  }
};


/// QualType - For efficiency, we don't store CV-qualified types as nodes on
/// their own: instead each reference to a type stores the qualifiers.  This
/// greatly reduces the number of nodes we need to allocate for types (for
/// example we only need one for 'int', 'const int', 'volatile int',
/// 'const volatile int', etc).
///
/// As an added efficiency bonus, instead of making this a pair, we
/// just store the two bits we care about in the low bits of the
/// pointer.  To handle the packing/unpacking, we make QualType be a
/// simple wrapper class that acts like a smart pointer.  A third bit
/// indicates whether there are extended qualifiers present, in which
/// case the pointer points to a special structure.
class QualType {
  // Thankfully, these are efficiently composable.
  llvm::PointerIntPair<llvm::PointerUnion<const Type*,const ExtQuals*>,
                       Qualifiers::FastWidth> Value;

  const ExtQuals *getExtQualsUnsafe() const {
    return Value.getPointer().get<const ExtQuals*>();
  }

  const Type *getTypePtrUnsafe() const {
    return Value.getPointer().get<const Type*>();
  }

  QualType getUnqualifiedTypeSlow() const;
  
  friend class QualifierCollector;
public:
  QualType() {}

  QualType(const Type *Ptr, unsigned Quals)
    : Value(Ptr, Quals) {}
  QualType(const ExtQuals *Ptr, unsigned Quals)
    : Value(Ptr, Quals) {}

  unsigned getLocalFastQualifiers() const { return Value.getInt(); }
  void setLocalFastQualifiers(unsigned Quals) { Value.setInt(Quals); }

  /// Retrieves a pointer to the underlying (unqualified) type.
  /// This should really return a const Type, but it's not worth
  /// changing all the users right now.
  Type *getTypePtr() const {
    if (hasLocalNonFastQualifiers())
      return const_cast<Type*>(getExtQualsUnsafe()->getBaseType());
    return const_cast<Type*>(getTypePtrUnsafe());
  }

  void *getAsOpaquePtr() const { return Value.getOpaqueValue(); }
  static QualType getFromOpaquePtr(void *Ptr) {
    QualType T;
    T.Value.setFromOpaqueValue(Ptr);
    return T;
  }

  Type &operator*() const {
    return *getTypePtr();
  }

  Type *operator->() const {
    return getTypePtr();
  }

  bool isCanonical() const;
  bool isCanonicalAsParam() const;

  /// isNull - Return true if this QualType doesn't point to a type yet.
  bool isNull() const {
    return Value.getPointer().isNull();
  }

  /// \brief Determine whether this particular QualType instance has the 
  /// "const" qualifier set, without looking through typedefs that may have
  /// added "const" at a different level.
  bool isLocalConstQualified() const {
    return (getLocalFastQualifiers() & Qualifiers::Const);
  }
  
  /// \brief Determine whether this type is const-qualified.
  bool isConstQualified() const;
  
  /// \brief Determine whether this particular QualType instance has the 
  /// "restrict" qualifier set, without looking through typedefs that may have
  /// added "restrict" at a different level.
  bool isLocalRestrictQualified() const {
    return (getLocalFastQualifiers() & Qualifiers::Restrict);
  }
  
  /// \brief Determine whether this type is restrict-qualified.
  bool isRestrictQualified() const;
  
  /// \brief Determine whether this particular QualType instance has the 
  /// "volatile" qualifier set, without looking through typedefs that may have
  /// added "volatile" at a different level.
  bool isLocalVolatileQualified() const {
    return (hasLocalNonFastQualifiers() && getExtQualsUnsafe()->hasVolatile());
  }

  /// \brief Determine whether this type is volatile-qualified.
  bool isVolatileQualified() const;
  
  /// \brief Determine whether this particular QualType instance has any
  /// qualifiers, without looking through any typedefs that might add 
  /// qualifiers at a different level.
  bool hasLocalQualifiers() const {
    return getLocalFastQualifiers() || hasLocalNonFastQualifiers();
  }

  /// \brief Determine whether this type has any qualifiers.
  bool hasQualifiers() const;
  
  /// \brief Determine whether this particular QualType instance has any
  /// "non-fast" qualifiers, e.g., those that are stored in an ExtQualType
  /// instance.
  bool hasLocalNonFastQualifiers() const {
    return Value.getPointer().is<const ExtQuals*>();
  }

  /// \brief Retrieve the set of qualifiers local to this particular QualType
  /// instance, not including any qualifiers acquired through typedefs or
  /// other sugar.
  Qualifiers getLocalQualifiers() const {
    Qualifiers Quals;
    if (hasLocalNonFastQualifiers())
      Quals = getExtQualsUnsafe()->getQualifiers();
    Quals.addFastQualifiers(getLocalFastQualifiers());
    return Quals;
  }

  /// \brief Retrieve the set of qualifiers applied to this type.
  Qualifiers getQualifiers() const;
  
  /// \brief Retrieve the set of CVR (const-volatile-restrict) qualifiers 
  /// local to this particular QualType instance, not including any qualifiers
  /// acquired through typedefs or other sugar.
  unsigned getLocalCVRQualifiers() const {
    unsigned CVR = getLocalFastQualifiers();
    if (isLocalVolatileQualified())
      CVR |= Qualifiers::Volatile;
    return CVR;
  }

  /// \brief Retrieve the set of CVR (const-volatile-restrict) qualifiers 
  /// applied to this type.
  unsigned getCVRQualifiers() const;

  /// \brief Retrieve the set of CVR (const-volatile-restrict) qualifiers
  /// applied to this type, looking through any number of unqualified array
  /// types to their element types' qualifiers.
  unsigned getCVRQualifiersThroughArrayTypes() const;

  bool isConstant(ASTContext& Ctx) const {
    return QualType::isConstant(*this, Ctx);
  }

  // Don't promise in the API that anything besides 'const' can be
  // easily added.

  /// addConst - add the specified type qualifier to this QualType.  
  void addConst() {
    addFastQualifiers(Qualifiers::Const);
  }
  QualType withConst() const {
    return withFastQualifiers(Qualifiers::Const);
  }

  void addFastQualifiers(unsigned TQs) {
    assert(!(TQs & ~Qualifiers::FastMask)
           && "non-fast qualifier bits set in mask!");
    Value.setInt(Value.getInt() | TQs);
  }

  // FIXME: The remove* functions are semantically broken, because they might
  // not remove a qualifier stored on a typedef. Most of the with* functions
  // have the same problem.
  void removeConst();
  void removeVolatile();
  void removeRestrict();
  void removeCVRQualifiers(unsigned Mask);

  void removeFastQualifiers() { Value.setInt(0); }
  void removeFastQualifiers(unsigned Mask) {
    assert(!(Mask & ~Qualifiers::FastMask) && "mask has non-fast qualifiers");
    Value.setInt(Value.getInt() & ~Mask);
  }

  // Creates a type with the given qualifiers in addition to any
  // qualifiers already on this type.
  QualType withFastQualifiers(unsigned TQs) const {
    QualType T = *this;
    T.addFastQualifiers(TQs);
    return T;
  }

  // Creates a type with exactly the given fast qualifiers, removing
  // any existing fast qualifiers.
  QualType withExactFastQualifiers(unsigned TQs) const {
    return withoutFastQualifiers().withFastQualifiers(TQs);
  }

  // Removes fast qualifiers, but leaves any extended qualifiers in place.
  QualType withoutFastQualifiers() const {
    QualType T = *this;
    T.removeFastQualifiers();
    return T;
  }

  /// \brief Return this type with all of the instance-specific qualifiers
  /// removed, but without removing any qualifiers that may have been applied
  /// through typedefs.
  QualType getLocalUnqualifiedType() const { return QualType(getTypePtr(), 0); }

  /// \brief Return the unqualified form of the given type, which might be
  /// desugared to eliminate qualifiers introduced via typedefs.
  QualType getUnqualifiedType() const {
    QualType T = getLocalUnqualifiedType();
    if (!T.hasQualifiers())
      return T;
    
    return getUnqualifiedTypeSlow();
  }
  
  bool isMoreQualifiedThan(QualType Other) const;
  bool isAtLeastAsQualifiedAs(QualType Other) const;
  QualType getNonReferenceType() const;

  /// getDesugaredType - Return the specified type with any "sugar" removed from
  /// the type.  This takes off typedefs, typeof's etc.  If the outer level of
  /// the type is already concrete, it returns it unmodified.  This is similar
  /// to getting the canonical type, but it doesn't remove *all* typedefs.  For
  /// example, it returns "T*" as "T*", (not as "int*"), because the pointer is
  /// concrete.
  ///
  /// Qualifiers are left in place.
  QualType getDesugaredType() const {
    return QualType::getDesugaredType(*this);
  }

  /// operator==/!= - Indicate whether the specified types and qualifiers are
  /// identical.
  friend bool operator==(const QualType &LHS, const QualType &RHS) {
    return LHS.Value == RHS.Value;
  }
  friend bool operator!=(const QualType &LHS, const QualType &RHS) {
    return LHS.Value != RHS.Value;
  }
  std::string getAsString() const;

  std::string getAsString(const PrintingPolicy &Policy) const {
    std::string S;
    getAsStringInternal(S, Policy);
    return S;
  }
  void getAsStringInternal(std::string &Str,
                           const PrintingPolicy &Policy) const;

  void dump(const char *s) const;
  void dump() const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(getAsOpaquePtr());
  }

  /// getAddressSpace - Return the address space of this type.
  inline unsigned getAddressSpace() const;

  /// GCAttrTypesAttr - Returns gc attribute of this type.
  inline Qualifiers::GC getObjCGCAttr() const;

  /// isObjCGCWeak true when Type is objc's weak.
  bool isObjCGCWeak() const {
    return getObjCGCAttr() == Qualifiers::Weak;
  }

  /// isObjCGCStrong true when Type is objc's strong.
  bool isObjCGCStrong() const {
    return getObjCGCAttr() == Qualifiers::Strong;
  }

  /// getNoReturnAttr - Returns true if the type has the noreturn attribute,
  /// false otherwise.
  bool getNoReturnAttr() const;

private:
  // These methods are implemented in a separate translation unit;
  // "static"-ize them to avoid creating temporary QualTypes in the
  // caller.
  static bool isConstant(QualType T, ASTContext& Ctx);
  static QualType getDesugaredType(QualType T);
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

// Teach SmallPtrSet that QualType is "basically a pointer".
template<>
class PointerLikeTypeTraits<clang::QualType> {
public:
  static inline void *getAsVoidPointer(clang::QualType P) {
    return P.getAsOpaquePtr();
  }
  static inline clang::QualType getFromVoidPointer(void *P) {
    return clang::QualType::getFromOpaquePtr(P);
  }
  // Various qualifiers go in low bits.
  enum { NumLowBitsAvailable = 0 };
};

} // end namespace llvm

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
/// TypedefType).  Its CanonicalType pointer points to the 'int' Type.  Next
/// there is a PointerType that represents 'int*', which, like 'int', is
/// canonical.  Finally, there is a PointerType type for 'foo*' whose canonical
/// type is 'int*', and there is a TypedefType for 'bar', whose canonical type
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
#define TYPE(Class, Base) Class,
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
    TagFirst = Record, TagLast = Enum
  };

protected:
  enum { TypeClassBitSize = 6 };

private:
  QualType CanonicalType;

  /// Dependent - Whether this type is a dependent type (C++ [temp.dep.type]).
  bool Dependent : 1;

  /// TypeClass bitfield - Enum that specifies what subclass this belongs to.
  /// Note that this should stay at the end of the ivars for Type so that
  /// subclasses can pack their bitfields into the same word.
  unsigned TC : TypeClassBitSize;

  Type(const Type&);           // DO NOT IMPLEMENT.
  void operator=(const Type&); // DO NOT IMPLEMENT.
protected:
  // silence VC++ warning C4355: 'this' : used in base member initializer list
  Type *this_() { return this; }
  Type(TypeClass tc, QualType Canonical, bool dependent)
    : CanonicalType(Canonical.isNull() ? QualType(this_(), 0) : Canonical),
      Dependent(dependent), TC(tc) {}
  virtual ~Type() {}
  virtual void Destroy(ASTContext& C);
  friend class ASTContext;

public:
  TypeClass getTypeClass() const { return static_cast<TypeClass>(TC); }

  bool isCanonicalUnqualified() const {
    return CanonicalType.getTypePtr() == this;
  }

  /// Types are partitioned into 3 broad categories (C99 6.2.5p1):
  /// object types, function types, and incomplete types.

  /// \brief Determines whether the type describes an object in memory.
  ///
  /// Note that this definition of object type corresponds to the C++
  /// definition of object type, which includes incomplete types, as
  /// opposed to the C definition (which does not include incomplete
  /// types).
  bool isObjectType() const;

  /// isIncompleteType - Return true if this is an incomplete type.
  /// A type that can describe objects, but which lacks information needed to
  /// determine its size (e.g. void, or a fwd declared struct). Clients of this
  /// routine will need to determine if the size is actually required.
  bool isIncompleteType() const;

  /// isIncompleteOrObjectType - Return true if this is an incomplete or object
  /// type, in other words, not a function type.
  bool isIncompleteOrObjectType() const {
    return !isFunctionType();
  }

  /// isPODType - Return true if this is a plain-old-data type (C++ 3.9p10).
  bool isPODType() const;

  /// isLiteralType - Return true if this is a literal type
  /// (C++0x [basic.types]p10)
  bool isLiteralType() const;

  /// isVariablyModifiedType (C99 6.7.5.2p2) - Return true for variable array
  /// types that have a non-constant expression. This does not include "[]".
  bool isVariablyModifiedType() const;

  /// Helper methods to distinguish type categories. All type predicates
  /// operate on the canonical type, ignoring typedefs and qualifiers.

  /// isSpecificBuiltinType - Test for a particular builtin type.
  bool isSpecificBuiltinType(unsigned K) const;

  /// isIntegerType() does *not* include complex integers (a GCC extension).
  /// isComplexIntegerType() can be used to test for complex integers.
  bool isIntegerType() const;     // C99 6.2.5p17 (int, char, bool, enum)
  bool isEnumeralType() const;
  bool isBooleanType() const;
  bool isCharType() const;
  bool isWideCharType() const;
  bool isAnyCharacterType() const;
  bool isIntegralType() const;
  
  /// Floating point categories.
  bool isRealFloatingType() const; // C99 6.2.5p10 (float, double, long double)
  /// isComplexType() does *not* include complex integers (a GCC extension).
  /// isComplexIntegerType() can be used to test for complex integers.
  bool isComplexType() const;      // C99 6.2.5p11 (complex)
  bool isAnyComplexType() const;   // C99 6.2.5p11 (complex) + Complex Int.
  bool isFloatingType() const;     // C99 6.2.5p11 (real floating + complex)
  bool isRealType() const;         // C99 6.2.5p17 (real floating + integer)
  bool isArithmeticType() const;   // C99 6.2.5p18 (integer + floating)
  bool isVoidType() const;         // C99 6.2.5p19
  bool isDerivedType() const;      // C99 6.2.5p20
  bool isScalarType() const;       // C99 6.2.5p21 (arithmetic + pointers)
  bool isAggregateType() const;

  // Type Predicates: Check to see if this type is structurally the specified
  // type, ignoring typedefs and qualifiers.
  bool isFunctionType() const;
  bool isFunctionNoProtoType() const { return getAs<FunctionNoProtoType>(); }
  bool isFunctionProtoType() const { return getAs<FunctionProtoType>(); }
  bool isPointerType() const;
  bool isAnyPointerType() const;   // Any C pointer or ObjC object pointer
  bool isBlockPointerType() const;
  bool isVoidPointerType() const;
  bool isReferenceType() const;
  bool isLValueReferenceType() const;
  bool isRValueReferenceType() const;
  bool isFunctionPointerType() const;
  bool isMemberPointerType() const;
  bool isMemberFunctionPointerType() const;
  bool isArrayType() const;
  bool isConstantArrayType() const;
  bool isIncompleteArrayType() const;
  bool isVariableArrayType() const;
  bool isDependentSizedArrayType() const;
  bool isRecordType() const;
  bool isClassType() const;
  bool isStructureType() const;
  bool isUnionType() const;
  bool isComplexIntegerType() const;            // GCC _Complex integer type.
  bool isVectorType() const;                    // GCC vector type.
  bool isExtVectorType() const;                 // Extended vector type.
  bool isObjCObjectPointerType() const;         // Pointer to *any* ObjC object.
  // FIXME: change this to 'raw' interface type, so we can used 'interface' type
  // for the common case.
  bool isObjCInterfaceType() const;             // NSString or NSString<foo>
  bool isObjCQualifiedInterfaceType() const;    // NSString<foo>
  bool isObjCQualifiedIdType() const;           // id<foo>
  bool isObjCQualifiedClassType() const;        // Class<foo>
  bool isObjCIdType() const;                    // id
  bool isObjCClassType() const;                 // Class
  bool isObjCSelType() const;                 // Class
  bool isObjCBuiltinType() const;               // 'id' or 'Class'
  bool isTemplateTypeParmType() const;          // C++ template type parameter
  bool isNullPtrType() const;                   // C++0x nullptr_t

  /// isDependentType - Whether this type is a dependent type, meaning
  /// that its definition somehow depends on a template parameter
  /// (C++ [temp.dep.type]).
  bool isDependentType() const { return Dependent; }
  bool isOverloadableType() const;

  /// hasPointerRepresentation - Whether this type is represented
  /// natively as a pointer; this includes pointers, references, block
  /// pointers, and Objective-C interface, qualified id, and qualified
  /// interface types, as well as nullptr_t.
  bool hasPointerRepresentation() const;

  /// hasObjCPointerRepresentation - Whether this type can represent
  /// an objective pointer type for the purpose of GC'ability
  bool hasObjCPointerRepresentation() const;

  // Type Checking Functions: Check to see if this type is structurally the
  // specified type, ignoring typedefs and qualifiers, and return a pointer to
  // the best type we can.
  const RecordType *getAsStructureType() const;
  /// NOTE: getAs*ArrayType are methods on ASTContext.
  const RecordType *getAsUnionType() const;
  const ComplexType *getAsComplexIntegerType() const; // GCC complex int type.
  // The following is a convenience method that returns an ObjCObjectPointerType
  // for object declared using an interface.
  const ObjCObjectPointerType *getAsObjCInterfacePointerType() const;
  const ObjCObjectPointerType *getAsObjCQualifiedIdType() const;
  const ObjCInterfaceType *getAsObjCQualifiedInterfaceType() const;
  const CXXRecordDecl *getCXXRecordDeclForPointerType() const;

  // Member-template getAs<specific type>'.  This scheme will eventually
  // replace the specific getAsXXXX methods above.
  //
  // There are some specializations of this member template listed
  // immediately following this class.
  template <typename T> const T *getAs() const;

  /// getAsPointerToObjCInterfaceType - If this is a pointer to an ObjC
  /// interface, return the interface type, otherwise return null.
  const ObjCInterfaceType *getAsPointerToObjCInterfaceType() const;

  /// getArrayElementTypeNoTypeQual - If this is an array type, return the
  /// element type of the array, potentially with type qualifiers missing.
  /// This method should never be used when type qualifiers are meaningful.
  const Type *getArrayElementTypeNoTypeQual() const;

  /// getPointeeType - If this is a pointer, ObjC object pointer, or block
  /// pointer, this returns the respective pointee.
  QualType getPointeeType() const;

  /// getUnqualifiedDesugaredType() - Return the specified type with
  /// any "sugar" removed from the type, removing any typedefs,
  /// typeofs, etc., as well as any qualifiers.
  const Type *getUnqualifiedDesugaredType() const;

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
  /// according to the rules of C99 6.7.5p3.  It is not legal to call this on
  /// incomplete types.
  bool isConstantSizeType() const;

  /// isSpecifierType - Returns true if this type can be represented by some
  /// set of type specifiers.
  bool isSpecifierType() const;

  const char *getTypeClassName() const;

  QualType getCanonicalTypeInternal() const { return CanonicalType; }
  void dump() const;
  static bool classof(const Type *) { return true; }
};

template <> inline const TypedefType *Type::getAs() const {
  return dyn_cast<TypedefType>(this);
}

// We can do canonical leaf types faster, because we don't have to
// worry about preserving child type decoration.
#define TYPE(Class, Base)
#define LEAF_TYPE(Class) \
template <> inline const Class##Type *Type::getAs() const { \
  return dyn_cast<Class##Type>(CanonicalType); \
}
#include "clang/AST/TypeNodes.def"


/// BuiltinType - This class is used for builtin types like 'int'.  Builtin
/// types are always canonical and have a literal name field.
class BuiltinType : public Type {
public:
  enum Kind {
    Void,

    Bool,     // This is bool and/or _Bool.
    Char_U,   // This is 'char' for targets where char is unsigned.
    UChar,    // This is explicitly qualified unsigned char.
    Char16,   // This is 'char16_t' for C++.
    Char32,   // This is 'char32_t' for C++.
    UShort,
    UInt,
    ULong,
    ULongLong,
    UInt128,  // __uint128_t

    Char_S,   // This is 'char' for targets where char is signed.
    SChar,    // This is explicitly qualified signed char.
    WChar,    // This is 'wchar_t' for C++.
    Short,
    Int,
    Long,
    LongLong,
    Int128,   // __int128_t

    Float, Double, LongDouble,

    NullPtr,  // This is the type of C++0x 'nullptr'.

    Overload,  // This represents the type of an overloaded function declaration.
    Dependent, // This represents the type of a type-dependent expression.

    UndeducedAuto, // In C++0x, this represents the type of an auto variable
                   // that has not been deduced yet.
    ObjCId,    // This represents the ObjC 'id' type.
    ObjCClass, // This represents the ObjC 'Class' type.
    ObjCSel    // This represents the ObjC 'SEL' type.
  };
private:
  Kind TypeKind;
public:
  BuiltinType(Kind K)
    : Type(Builtin, QualType(), /*Dependent=*/(K == Dependent)),
      TypeKind(K) {}

  Kind getKind() const { return TypeKind; }
  const char *getName(const LangOptions &LO) const;

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  bool isInteger() const {
    return TypeKind >= Bool && TypeKind <= Int128;
  }

  bool isSignedInteger() const {
    return TypeKind >= Char_S && TypeKind <= Int128;
  }

  bool isUnsignedInteger() const {
    return TypeKind >= Bool && TypeKind <= UInt128;
  }

  bool isFloatingPoint() const {
    return TypeKind >= Float && TypeKind <= LongDouble;
  }

  static bool classof(const Type *T) { return T->getTypeClass() == Builtin; }
  static bool classof(const BuiltinType *) { return true; }
};

/// ComplexType - C99 6.2.5p11 - Complex values.  This supports the C99 complex
/// types (_Complex float etc) as well as the GCC integer complex extensions.
///
class ComplexType : public Type, public llvm::FoldingSetNode {
  QualType ElementType;
  ComplexType(QualType Element, QualType CanonicalPtr) :
    Type(Complex, CanonicalPtr, Element->isDependentType()),
    ElementType(Element) {
  }
  friend class ASTContext;  // ASTContext creates these.
public:
  QualType getElementType() const { return ElementType; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

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
    Type(Pointer, CanonicalPtr, Pointee->isDependentType()), PointeeType(Pointee) {
  }
  friend class ASTContext;  // ASTContext creates these.
public:

  QualType getPointeeType() const { return PointeeType; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getPointeeType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Pointee) {
    ID.AddPointer(Pointee.getAsOpaquePtr());
  }

  static bool classof(const Type *T) { return T->getTypeClass() == Pointer; }
  static bool classof(const PointerType *) { return true; }
};

/// BlockPointerType - pointer to a block type.
/// This type is to represent types syntactically represented as
/// "void (^)(int)", etc. Pointee is required to always be a function type.
///
class BlockPointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;  // Block is some kind of pointer type
  BlockPointerType(QualType Pointee, QualType CanonicalCls) :
    Type(BlockPointer, CanonicalCls, Pointee->isDependentType()),
    PointeeType(Pointee) {
  }
  friend class ASTContext;  // ASTContext creates these.
public:

  // Get the pointee type. Pointee is required to always be a function type.
  QualType getPointeeType() const { return PointeeType; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
      Profile(ID, getPointeeType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Pointee) {
      ID.AddPointer(Pointee.getAsOpaquePtr());
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == BlockPointer;
  }
  static bool classof(const BlockPointerType *) { return true; }
};

/// ReferenceType - Base for LValueReferenceType and RValueReferenceType
///
class ReferenceType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;

  /// True if the type was originally spelled with an lvalue sigil.
  /// This is never true of rvalue references but can also be false
  /// on lvalue references because of C++0x [dcl.typedef]p9,
  /// as follows:
  ///
  ///   typedef int &ref;    // lvalue, spelled lvalue
  ///   typedef int &&rvref; // rvalue
  ///   ref &a;              // lvalue, inner ref, spelled lvalue
  ///   ref &&a;             // lvalue, inner ref
  ///   rvref &a;            // lvalue, inner ref, spelled lvalue
  ///   rvref &&a;           // rvalue, inner ref
  bool SpelledAsLValue;

  /// True if the inner type is a reference type.  This only happens
  /// in non-canonical forms.
  bool InnerRef;

protected:
  ReferenceType(TypeClass tc, QualType Referencee, QualType CanonicalRef,
                bool SpelledAsLValue) :
    Type(tc, CanonicalRef, Referencee->isDependentType()),
    PointeeType(Referencee), SpelledAsLValue(SpelledAsLValue),
    InnerRef(Referencee->isReferenceType()) {
  }
public:
  bool isSpelledAsLValue() const { return SpelledAsLValue; }

  QualType getPointeeTypeAsWritten() const { return PointeeType; }
  QualType getPointeeType() const {
    // FIXME: this might strip inner qualifiers; okay?
    const ReferenceType *T = this;
    while (T->InnerRef)
      T = T->PointeeType->getAs<ReferenceType>();
    return T->PointeeType;
  }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, PointeeType, SpelledAsLValue);
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      QualType Referencee,
                      bool SpelledAsLValue) {
    ID.AddPointer(Referencee.getAsOpaquePtr());
    ID.AddBoolean(SpelledAsLValue);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == LValueReference ||
           T->getTypeClass() == RValueReference;
  }
  static bool classof(const ReferenceType *) { return true; }
};

/// LValueReferenceType - C++ [dcl.ref] - Lvalue reference
///
class LValueReferenceType : public ReferenceType {
  LValueReferenceType(QualType Referencee, QualType CanonicalRef,
                      bool SpelledAsLValue) :
    ReferenceType(LValueReference, Referencee, CanonicalRef, SpelledAsLValue)
  {}
  friend class ASTContext; // ASTContext creates these
public:
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == LValueReference;
  }
  static bool classof(const LValueReferenceType *) { return true; }
};

/// RValueReferenceType - C++0x [dcl.ref] - Rvalue reference
///
class RValueReferenceType : public ReferenceType {
  RValueReferenceType(QualType Referencee, QualType CanonicalRef) :
    ReferenceType(RValueReference, Referencee, CanonicalRef, false) {
  }
  friend class ASTContext; // ASTContext creates these
public:
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == RValueReference;
  }
  static bool classof(const RValueReferenceType *) { return true; }
};

/// MemberPointerType - C++ 8.3.3 - Pointers to members
///
class MemberPointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;
  /// The class of which the pointee is a member. Must ultimately be a
  /// RecordType, but could be a typedef or a template parameter too.
  const Type *Class;

  MemberPointerType(QualType Pointee, const Type *Cls, QualType CanonicalPtr) :
    Type(MemberPointer, CanonicalPtr,
         Cls->isDependentType() || Pointee->isDependentType()),
    PointeeType(Pointee), Class(Cls) {
  }
  friend class ASTContext; // ASTContext creates these.
public:

  QualType getPointeeType() const { return PointeeType; }

  const Type *getClass() const { return Class; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getPointeeType(), getClass());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Pointee,
                      const Type *Class) {
    ID.AddPointer(Pointee.getAsOpaquePtr());
    ID.AddPointer(Class);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == MemberPointer;
  }
  static bool classof(const MemberPointerType *) { return true; }
};

/// ArrayType - C99 6.7.5.2 - Array Declarators.
///
class ArrayType : public Type, public llvm::FoldingSetNode {
public:
  /// ArraySizeModifier - Capture whether this is a normal array (e.g. int X[4])
  /// an array with a static size (e.g. int X[static 4]), or an array
  /// with a star size (e.g. int X[*]).
  /// 'static' is only allowed on function parameters.
  enum ArraySizeModifier {
    Normal, Static, Star
  };
private:
  /// ElementType - The element type of the array.
  QualType ElementType;

  // NOTE: VC++ treats enums as signed, avoid using the ArraySizeModifier enum
  /// NOTE: These fields are packed into the bitfields space in the Type class.
  unsigned SizeModifier : 2;

  /// IndexTypeQuals - Capture qualifiers in declarations like:
  /// 'int X[static restrict 4]'. For function parameters only.
  unsigned IndexTypeQuals : 3;

protected:
  // C++ [temp.dep.type]p1:
  //   A type is dependent if it is...
  //     - an array type constructed from any dependent type or whose
  //       size is specified by a constant expression that is
  //       value-dependent,
  ArrayType(TypeClass tc, QualType et, QualType can,
            ArraySizeModifier sm, unsigned tq)
    : Type(tc, can, et->isDependentType() || tc == DependentSizedArray),
      ElementType(et), SizeModifier(sm), IndexTypeQuals(tq) {}

  friend class ASTContext;  // ASTContext creates these.
public:
  QualType getElementType() const { return ElementType; }
  ArraySizeModifier getSizeModifier() const {
    return ArraySizeModifier(SizeModifier);
  }
  Qualifiers getIndexTypeQualifiers() const {
    return Qualifiers::fromCVRMask(IndexTypeQuals);
  }
  unsigned getIndexTypeCVRQualifiers() const { return IndexTypeQuals; }

  static bool classof(const Type *T) {
    return T->getTypeClass() == ConstantArray ||
           T->getTypeClass() == VariableArray ||
           T->getTypeClass() == IncompleteArray ||
           T->getTypeClass() == DependentSizedArray;
  }
  static bool classof(const ArrayType *) { return true; }
};

/// ConstantArrayType - This class represents the canonical version of
/// C arrays with a specified constant size.  For example, the canonical
/// type for 'int A[4 + 4*100]' is a ConstantArrayType where the element
/// type is 'int' and the size is 404.
class ConstantArrayType : public ArrayType {
  llvm::APInt Size; // Allows us to unique the type.

  ConstantArrayType(QualType et, QualType can, const llvm::APInt &size,
                    ArraySizeModifier sm, unsigned tq)
    : ArrayType(ConstantArray, et, can, sm, tq),
      Size(size) {}
protected:
  ConstantArrayType(TypeClass tc, QualType et, QualType can,
                    const llvm::APInt &size, ArraySizeModifier sm, unsigned tq)
    : ArrayType(tc, et, can, sm, tq), Size(size) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  const llvm::APInt &getSize() const { return Size; }
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getElementType(), getSize(),
            getSizeModifier(), getIndexTypeCVRQualifiers());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType ET,
                      const llvm::APInt &ArraySize, ArraySizeModifier SizeMod,
                      unsigned TypeQuals) {
    ID.AddPointer(ET.getAsOpaquePtr());
    ID.AddInteger(ArraySize.getZExtValue());
    ID.AddInteger(SizeMod);
    ID.AddInteger(TypeQuals);
  }
  static bool classof(const Type *T) {
    return T->getTypeClass() == ConstantArray;
  }
  static bool classof(const ConstantArrayType *) { return true; }
};

/// IncompleteArrayType - This class represents C arrays with an unspecified
/// size.  For example 'int A[]' has an IncompleteArrayType where the element
/// type is 'int' and the size is unspecified.
class IncompleteArrayType : public ArrayType {

  IncompleteArrayType(QualType et, QualType can,
                      ArraySizeModifier sm, unsigned tq)
    : ArrayType(IncompleteArray, et, can, sm, tq) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == IncompleteArray;
  }
  static bool classof(const IncompleteArrayType *) { return true; }

  friend class StmtIteratorBase;

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getElementType(), getSizeModifier(),
            getIndexTypeCVRQualifiers());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, QualType ET,
                      ArraySizeModifier SizeMod, unsigned TypeQuals) {
    ID.AddPointer(ET.getAsOpaquePtr());
    ID.AddInteger(SizeMod);
    ID.AddInteger(TypeQuals);
  }
};

/// VariableArrayType - This class represents C arrays with a specified size
/// which is not an integer-constant-expression.  For example, 'int s[x+foo()]'.
/// Since the size expression is an arbitrary expression, we store it as such.
///
/// Note: VariableArrayType's aren't uniqued (since the expressions aren't) and
/// should not be: two lexically equivalent variable array types could mean
/// different things, for example, these variables do not have the same type
/// dynamically:
///
/// void foo(int x) {
///   int Y[x];
///   ++x;
///   int Z[x];
/// }
///
class VariableArrayType : public ArrayType {
  /// SizeExpr - An assignment expression. VLA's are only permitted within
  /// a function block.
  Stmt *SizeExpr;
  /// Brackets - The left and right array brackets.
  SourceRange Brackets;

  VariableArrayType(QualType et, QualType can, Expr *e,
                    ArraySizeModifier sm, unsigned tq,
                    SourceRange brackets)
    : ArrayType(VariableArray, et, can, sm, tq),
      SizeExpr((Stmt*) e), Brackets(brackets) {}
  friend class ASTContext;  // ASTContext creates these.
  virtual void Destroy(ASTContext& C);

public:
  Expr *getSizeExpr() const {
    // We use C-style casts instead of cast<> here because we do not wish
    // to have a dependency of Type.h on Stmt.h/Expr.h.
    return (Expr*) SizeExpr;
  }
  SourceRange getBracketsRange() const { return Brackets; }
  SourceLocation getLBracketLoc() const { return Brackets.getBegin(); }
  SourceLocation getRBracketLoc() const { return Brackets.getEnd(); }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == VariableArray;
  }
  static bool classof(const VariableArrayType *) { return true; }

  friend class StmtIteratorBase;

  void Profile(llvm::FoldingSetNodeID &ID) {
    assert(0 && "Cannnot unique VariableArrayTypes.");
  }
};

/// DependentSizedArrayType - This type represents an array type in
/// C++ whose size is a value-dependent expression. For example:
///
/// \code
/// template<typename T, int Size>
/// class array {
///   T data[Size];
/// };
/// \endcode
///
/// For these types, we won't actually know what the array bound is
/// until template instantiation occurs, at which point this will
/// become either a ConstantArrayType or a VariableArrayType.
class DependentSizedArrayType : public ArrayType {
  ASTContext &Context;

  /// \brief An assignment expression that will instantiate to the
  /// size of the array.
  ///
  /// The expression itself might be NULL, in which case the array
  /// type will have its size deduced from an initializer.
  Stmt *SizeExpr;

  /// Brackets - The left and right array brackets.
  SourceRange Brackets;

  DependentSizedArrayType(ASTContext &Context, QualType et, QualType can,
                          Expr *e, ArraySizeModifier sm, unsigned tq,
                          SourceRange brackets)
    : ArrayType(DependentSizedArray, et, can, sm, tq),
      Context(Context), SizeExpr((Stmt*) e), Brackets(brackets) {}
  friend class ASTContext;  // ASTContext creates these.
  virtual void Destroy(ASTContext& C);

public:
  Expr *getSizeExpr() const {
    // We use C-style casts instead of cast<> here because we do not wish
    // to have a dependency of Type.h on Stmt.h/Expr.h.
    return (Expr*) SizeExpr;
  }
  SourceRange getBracketsRange() const { return Brackets; }
  SourceLocation getLBracketLoc() const { return Brackets.getBegin(); }
  SourceLocation getRBracketLoc() const { return Brackets.getEnd(); }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == DependentSizedArray;
  }
  static bool classof(const DependentSizedArrayType *) { return true; }

  friend class StmtIteratorBase;


  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getElementType(),
            getSizeModifier(), getIndexTypeCVRQualifiers(), getSizeExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, ASTContext &Context,
                      QualType ET, ArraySizeModifier SizeMod,
                      unsigned TypeQuals, Expr *E);
};

/// DependentSizedExtVectorType - This type represent an extended vector type
/// where either the type or size is dependent. For example:
/// @code
/// template<typename T, int Size>
/// class vector {
///   typedef T __attribute__((ext_vector_type(Size))) type;
/// }
/// @endcode
class DependentSizedExtVectorType : public Type, public llvm::FoldingSetNode {
  ASTContext &Context;
  Expr *SizeExpr;
  /// ElementType - The element type of the array.
  QualType ElementType;
  SourceLocation loc;

  DependentSizedExtVectorType(ASTContext &Context, QualType ElementType,
                              QualType can, Expr *SizeExpr, SourceLocation loc)
    : Type (DependentSizedExtVector, can, true),
      Context(Context), SizeExpr(SizeExpr), ElementType(ElementType),
      loc(loc) {}
  friend class ASTContext;
  virtual void Destroy(ASTContext& C);

public:
  Expr *getSizeExpr() const { return SizeExpr; }
  QualType getElementType() const { return ElementType; }
  SourceLocation getAttributeLoc() const { return loc; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == DependentSizedExtVector;
  }
  static bool classof(const DependentSizedExtVectorType *) { return true; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getElementType(), getSizeExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, ASTContext &Context,
                      QualType ElementType, Expr *SizeExpr);
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
    Type(Vector, canonType, vecType->isDependentType()),
    ElementType(vecType), NumElements(nElements) {}
  VectorType(TypeClass tc, QualType vecType, unsigned nElements,
             QualType canonType)
    : Type(tc, canonType, vecType->isDependentType()), ElementType(vecType),
      NumElements(nElements) {}
  friend class ASTContext;  // ASTContext creates these.
public:

  QualType getElementType() const { return ElementType; }
  unsigned getNumElements() const { return NumElements; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

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
    return T->getTypeClass() == Vector || T->getTypeClass() == ExtVector;
  }
  static bool classof(const VectorType *) { return true; }
};

/// ExtVectorType - Extended vector type. This type is created using
/// __attribute__((ext_vector_type(n)), where "n" is the number of elements.
/// Unlike vector_size, ext_vector_type is only allowed on typedef's. This
/// class enables syntactic extensions, like Vector Components for accessing
/// points, colors, and textures (modeled after OpenGL Shading Language).
class ExtVectorType : public VectorType {
  ExtVectorType(QualType vecType, unsigned nElements, QualType canonType) :
    VectorType(ExtVector, vecType, nElements, canonType) {}
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
  static int getNumericAccessorIdx(char c) {
    switch (c) {
      default: return -1;
      case '0': return 0;
      case '1': return 1;
      case '2': return 2;
      case '3': return 3;
      case '4': return 4;
      case '5': return 5;
      case '6': return 6;
      case '7': return 7;
      case '8': return 8;
      case '9': return 9;
      case 'A':
      case 'a': return 10;
      case 'B':
      case 'b': return 11;
      case 'C':
      case 'c': return 12;
      case 'D':
      case 'd': return 13;
      case 'E':
      case 'e': return 14;
      case 'F':
      case 'f': return 15;
    }
  }

  static int getAccessorIdx(char c) {
    if (int idx = getPointAccessorIdx(c)+1) return idx-1;
    return getNumericAccessorIdx(c);
  }

  bool isAccessorWithinNumElements(char c) const {
    if (int idx = getAccessorIdx(c)+1)
      return unsigned(idx-1) < NumElements;
    return false;
  }
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == ExtVector;
  }
  static bool classof(const ExtVectorType *) { return true; }
};

/// FunctionType - C99 6.7.5.3 - Function Declarators.  This is the common base
/// class of FunctionNoProtoType and FunctionProtoType.
///
class FunctionType : public Type {
  /// SubClassData - This field is owned by the subclass, put here to pack
  /// tightly with the ivars in Type.
  bool SubClassData : 1;

  /// TypeQuals - Used only by FunctionProtoType, put here to pack with the
  /// other bitfields.
  /// The qualifiers are part of FunctionProtoType because...
  ///
  /// C++ 8.3.5p4: The return type, the parameter type list and the
  /// cv-qualifier-seq, [...], are part of the function type.
  ///
  unsigned TypeQuals : 3;

  /// NoReturn - Indicates if the function type is attribute noreturn.
  unsigned NoReturn : 1;

  // The type returned by the function.
  QualType ResultType;
protected:
  FunctionType(TypeClass tc, QualType res, bool SubclassInfo,
               unsigned typeQuals, QualType Canonical, bool Dependent,
               bool noReturn = false)
    : Type(tc, Canonical, Dependent),
      SubClassData(SubclassInfo), TypeQuals(typeQuals), NoReturn(noReturn),
      ResultType(res) {}
  bool getSubClassData() const { return SubClassData; }
  unsigned getTypeQuals() const { return TypeQuals; }
public:

  QualType getResultType() const { return ResultType; }
  bool getNoReturnAttr() const { return NoReturn; }

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionNoProto ||
           T->getTypeClass() == FunctionProto;
  }
  static bool classof(const FunctionType *) { return true; }
};

/// FunctionNoProtoType - Represents a K&R-style 'int foo()' function, which has
/// no information available about its arguments.
class FunctionNoProtoType : public FunctionType, public llvm::FoldingSetNode {
  FunctionNoProtoType(QualType Result, QualType Canonical,
                      bool NoReturn = false)
    : FunctionType(FunctionNoProto, Result, false, 0, Canonical,
                   /*Dependent=*/false, NoReturn) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  // No additional state past what FunctionType provides.

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getResultType(), getNoReturnAttr());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType ResultType,
                      bool NoReturn) {
    ID.AddInteger(NoReturn);
    ID.AddPointer(ResultType.getAsOpaquePtr());
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionNoProto;
  }
  static bool classof(const FunctionNoProtoType *) { return true; }
};

/// FunctionProtoType - Represents a prototype with argument type info, e.g.
/// 'int foo(int)' or 'int foo(void)'.  'void' is represented as having no
/// arguments, not as having a single void argument. Such a type can have an
/// exception specification, but this specification is not part of the canonical
/// type.
class FunctionProtoType : public FunctionType, public llvm::FoldingSetNode {
  /// hasAnyDependentType - Determine whether there are any dependent
  /// types within the arguments passed in.
  static bool hasAnyDependentType(const QualType *ArgArray, unsigned numArgs) {
    for (unsigned Idx = 0; Idx < numArgs; ++Idx)
      if (ArgArray[Idx]->isDependentType())
    return true;

    return false;
  }

  FunctionProtoType(QualType Result, const QualType *ArgArray, unsigned numArgs,
                    bool isVariadic, unsigned typeQuals, bool hasExs,
                    bool hasAnyExs, const QualType *ExArray,
                    unsigned numExs, QualType Canonical, bool NoReturn)
    : FunctionType(FunctionProto, Result, isVariadic, typeQuals, Canonical,
                   (Result->isDependentType() ||
                    hasAnyDependentType(ArgArray, numArgs)), NoReturn),
      NumArgs(numArgs), NumExceptions(numExs), HasExceptionSpec(hasExs),
      AnyExceptionSpec(hasAnyExs) {
    // Fill in the trailing argument array.
    QualType *ArgInfo = reinterpret_cast<QualType*>(this+1);
    for (unsigned i = 0; i != numArgs; ++i)
      ArgInfo[i] = ArgArray[i];
    // Fill in the exception array.
    QualType *Ex = ArgInfo + numArgs;
    for (unsigned i = 0; i != numExs; ++i)
      Ex[i] = ExArray[i];
  }

  /// NumArgs - The number of arguments this function has, not counting '...'.
  unsigned NumArgs : 20;

  /// NumExceptions - The number of types in the exception spec, if any.
  unsigned NumExceptions : 10;

  /// HasExceptionSpec - Whether this function has an exception spec at all.
  bool HasExceptionSpec : 1;

  /// AnyExceptionSpec - Whether this function has a throw(...) spec.
  bool AnyExceptionSpec : 1;

  /// ArgInfo - There is an variable size array after the class in memory that
  /// holds the argument types.

  /// Exceptions - There is another variable size array after ArgInfo that
  /// holds the exception types.

  friend class ASTContext;  // ASTContext creates these.

public:
  unsigned getNumArgs() const { return NumArgs; }
  QualType getArgType(unsigned i) const {
    assert(i < NumArgs && "Invalid argument number!");
    return arg_type_begin()[i];
  }

  bool hasExceptionSpec() const { return HasExceptionSpec; }
  bool hasAnyExceptionSpec() const { return AnyExceptionSpec; }
  unsigned getNumExceptions() const { return NumExceptions; }
  QualType getExceptionType(unsigned i) const {
    assert(i < NumExceptions && "Invalid exception number!");
    return exception_begin()[i];
  }
  bool hasEmptyExceptionSpec() const {
    return hasExceptionSpec() && !hasAnyExceptionSpec() &&
      getNumExceptions() == 0;
  }

  bool isVariadic() const { return getSubClassData(); }
  unsigned getTypeQuals() const { return FunctionType::getTypeQuals(); }

  typedef const QualType *arg_type_iterator;
  arg_type_iterator arg_type_begin() const {
    return reinterpret_cast<const QualType *>(this+1);
  }
  arg_type_iterator arg_type_end() const { return arg_type_begin()+NumArgs; }

  typedef const QualType *exception_iterator;
  exception_iterator exception_begin() const {
    // exceptions begin where arguments end
    return arg_type_end();
  }
  exception_iterator exception_end() const {
    return exception_begin() + NumExceptions;
  }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionProto;
  }
  static bool classof(const FunctionProtoType *) { return true; }

  void Profile(llvm::FoldingSetNodeID &ID);
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Result,
                      arg_type_iterator ArgTys, unsigned NumArgs,
                      bool isVariadic, unsigned TypeQuals,
                      bool hasExceptionSpec, bool anyExceptionSpec,
                      unsigned NumExceptions, exception_iterator Exs,
                      bool NoReturn);
};


/// \brief Represents the dependent type named by a dependently-scoped
/// typename using declaration, e.g.
///   using typename Base<T>::foo;
/// Template instantiation turns these into the underlying type.
class UnresolvedUsingType : public Type {
  UnresolvedUsingTypenameDecl *Decl;

  UnresolvedUsingType(UnresolvedUsingTypenameDecl *D)
    : Type(UnresolvedUsing, QualType(), true), Decl(D) {}
  friend class ASTContext; // ASTContext creates these.
public:

  UnresolvedUsingTypenameDecl *getDecl() const { return Decl; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == UnresolvedUsing;
  }
  static bool classof(const UnresolvedUsingType *) { return true; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    return Profile(ID, Decl);
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      UnresolvedUsingTypenameDecl *D) {
    ID.AddPointer(D);
  }
};


class TypedefType : public Type {
  TypedefDecl *Decl;
protected:
  TypedefType(TypeClass tc, TypedefDecl *D, QualType can)
    : Type(tc, can, can->isDependentType()), Decl(D) {
    assert(!isa<TypedefType>(can) && "Invalid canonical type");
  }
  friend class ASTContext;  // ASTContext creates these.
public:

  TypedefDecl *getDecl() const { return Decl; }

  /// LookThroughTypedefs - Return the ultimate type this typedef corresponds to
  /// potentially looking through *all* consecutive typedefs.  This returns the
  /// sum of the type qualifiers, so if you have:
  ///   typedef const int A;
  ///   typedef volatile A B;
  /// looking through the typedefs for B will give you "const volatile A".
  QualType LookThroughTypedefs() const;

  bool isSugared() const { return true; }
  QualType desugar() const;

  static bool classof(const Type *T) { return T->getTypeClass() == Typedef; }
  static bool classof(const TypedefType *) { return true; }
};

/// TypeOfExprType (GCC extension).
class TypeOfExprType : public Type {
  Expr *TOExpr;

protected:
  TypeOfExprType(Expr *E, QualType can = QualType());
  friend class ASTContext;  // ASTContext creates these.
public:
  Expr *getUnderlyingExpr() const { return TOExpr; }

  /// \brief Remove a single level of sugar.
  QualType desugar() const;

  /// \brief Returns whether this type directly provides sugar.
  bool isSugared() const { return true; }

  static bool classof(const Type *T) { return T->getTypeClass() == TypeOfExpr; }
  static bool classof(const TypeOfExprType *) { return true; }
};

/// Subclass of TypeOfExprType that is used for canonical, dependent
/// typeof(expr) types.
class DependentTypeOfExprType
  : public TypeOfExprType, public llvm::FoldingSetNode {
  ASTContext &Context;

public:
  DependentTypeOfExprType(ASTContext &Context, Expr *E)
    : TypeOfExprType(E), Context(Context) { }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getUnderlyingExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, ASTContext &Context,
                      Expr *E);
};

/// TypeOfType (GCC extension).
class TypeOfType : public Type {
  QualType TOType;
  TypeOfType(QualType T, QualType can)
    : Type(TypeOf, can, T->isDependentType()), TOType(T) {
    assert(!isa<TypedefType>(can) && "Invalid canonical type");
  }
  friend class ASTContext;  // ASTContext creates these.
public:
  QualType getUnderlyingType() const { return TOType; }

  /// \brief Remove a single level of sugar.
  QualType desugar() const { return getUnderlyingType(); }

  /// \brief Returns whether this type directly provides sugar.
  bool isSugared() const { return true; }

  static bool classof(const Type *T) { return T->getTypeClass() == TypeOf; }
  static bool classof(const TypeOfType *) { return true; }
};

/// DecltypeType (C++0x)
class DecltypeType : public Type {
  Expr *E;

  // FIXME: We could get rid of UnderlyingType if we wanted to: We would have to
  // Move getDesugaredType to ASTContext so that it can call getDecltypeForExpr
  // from it.
  QualType UnderlyingType;

protected:
  DecltypeType(Expr *E, QualType underlyingType, QualType can = QualType());
  friend class ASTContext;  // ASTContext creates these.
public:
  Expr *getUnderlyingExpr() const { return E; }
  QualType getUnderlyingType() const { return UnderlyingType; }

  /// \brief Remove a single level of sugar.
  QualType desugar() const { return getUnderlyingType(); }

  /// \brief Returns whether this type directly provides sugar.
  bool isSugared() const { return !isDependentType(); }

  static bool classof(const Type *T) { return T->getTypeClass() == Decltype; }
  static bool classof(const DecltypeType *) { return true; }
};

/// Subclass of DecltypeType that is used for canonical, dependent
/// C++0x decltype types.
class DependentDecltypeType : public DecltypeType, public llvm::FoldingSetNode {
  ASTContext &Context;

public:
  DependentDecltypeType(ASTContext &Context, Expr *E);

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getUnderlyingExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, ASTContext &Context,
                      Expr *E);
};

class TagType : public Type {
  /// Stores the TagDecl associated with this type. The decl will
  /// point to the TagDecl that actually defines the entity (or is a
  /// definition in progress), if there is such a definition. The
  /// single-bit value will be non-zero when this tag is in the
  /// process of being defined.
  mutable llvm::PointerIntPair<TagDecl *, 1> decl;
  friend class ASTContext;
  friend class TagDecl;

protected:
  TagType(TypeClass TC, TagDecl *D, QualType can);

public:
  TagDecl *getDecl() const { return decl.getPointer(); }

  /// @brief Determines whether this type is in the process of being
  /// defined.
  bool isBeingDefined() const { return decl.getInt(); }
  void setBeingDefined(bool Def) const { decl.setInt(Def? 1 : 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() >= TagFirst && T->getTypeClass() <= TagLast;
  }
  static bool classof(const TagType *) { return true; }
  static bool classof(const RecordType *) { return true; }
  static bool classof(const EnumType *) { return true; }
};

/// RecordType - This is a helper class that allows the use of isa/cast/dyncast
/// to detect TagType objects of structs/unions/classes.
class RecordType : public TagType {
protected:
  explicit RecordType(RecordDecl *D)
    : TagType(Record, reinterpret_cast<TagDecl*>(D), QualType()) { }
  explicit RecordType(TypeClass TC, RecordDecl *D)
    : TagType(TC, reinterpret_cast<TagDecl*>(D), QualType()) { }
  friend class ASTContext;   // ASTContext creates these.
public:

  RecordDecl *getDecl() const {
    return reinterpret_cast<RecordDecl*>(TagType::getDecl());
  }

  // FIXME: This predicate is a helper to QualType/Type. It needs to
  // recursively check all fields for const-ness. If any field is declared
  // const, it needs to return false.
  bool hasConstFields() const { return false; }

  // FIXME: RecordType needs to check when it is created that all fields are in
  // the same address space, and return that.
  unsigned getAddressSpace() const { return 0; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const TagType *T);
  static bool classof(const Type *T) {
    return isa<TagType>(T) && classof(cast<TagType>(T));
  }
  static bool classof(const RecordType *) { return true; }
};

/// EnumType - This is a helper class that allows the use of isa/cast/dyncast
/// to detect TagType objects of enums.
class EnumType : public TagType {
  explicit EnumType(EnumDecl *D)
    : TagType(Enum, reinterpret_cast<TagDecl*>(D), QualType()) { }
  friend class ASTContext;   // ASTContext creates these.
public:

  EnumDecl *getDecl() const {
    return reinterpret_cast<EnumDecl*>(TagType::getDecl());
  }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const TagType *T);
  static bool classof(const Type *T) {
    return isa<TagType>(T) && classof(cast<TagType>(T));
  }
  static bool classof(const EnumType *) { return true; }
};

/// ElaboratedType - A non-canonical type used to represents uses of
/// elaborated type specifiers in C++.  For example:
///
///   void foo(union MyUnion);
///            ^^^^^^^^^^^^^
///
/// At the moment, for efficiency we do not create elaborated types in
/// C, since outside of typedefs all references to structs would
/// necessarily be elaborated.
class ElaboratedType : public Type, public llvm::FoldingSetNode {
public:
  enum TagKind {
    TK_struct,
    TK_union,
    TK_class,
    TK_enum
  };

private:
  /// The tag that was used in this elaborated type specifier.
  TagKind Tag;

  /// The underlying type.
  QualType UnderlyingType;

  explicit ElaboratedType(QualType Ty, TagKind Tag, QualType Canon)
    : Type(Elaborated, Canon, Canon->isDependentType()),
      Tag(Tag), UnderlyingType(Ty) { }
  friend class ASTContext;   // ASTContext creates these.

public:
  TagKind getTagKind() const { return Tag; }
  QualType getUnderlyingType() const { return UnderlyingType; }

  /// \brief Remove a single level of sugar.
  QualType desugar() const { return getUnderlyingType(); }

  /// \brief Returns whether this type directly provides sugar.
  bool isSugared() const { return true; }

  static const char *getNameForTagKind(TagKind Kind) {
    switch (Kind) {
    default: assert(0 && "Unknown TagKind!");
    case TK_struct: return "struct";
    case TK_union:  return "union";
    case TK_class:  return "class";
    case TK_enum:   return "enum";
    }
  }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getUnderlyingType(), getTagKind());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType T, TagKind Tag) {
    ID.AddPointer(T.getAsOpaquePtr());
    ID.AddInteger(Tag);
  }

  static bool classof(const ElaboratedType*) { return true; }
  static bool classof(const Type *T) { return T->getTypeClass() == Elaborated; }
};

class TemplateTypeParmType : public Type, public llvm::FoldingSetNode {
  unsigned Depth : 15;
  unsigned Index : 16;
  unsigned ParameterPack : 1;
  IdentifierInfo *Name;

  TemplateTypeParmType(unsigned D, unsigned I, bool PP, IdentifierInfo *N,
                       QualType Canon)
    : Type(TemplateTypeParm, Canon, /*Dependent=*/true),
      Depth(D), Index(I), ParameterPack(PP), Name(N) { }

  TemplateTypeParmType(unsigned D, unsigned I, bool PP)
    : Type(TemplateTypeParm, QualType(this, 0), /*Dependent=*/true),
      Depth(D), Index(I), ParameterPack(PP), Name(0) { }

  friend class ASTContext;  // ASTContext creates these

public:
  unsigned getDepth() const { return Depth; }
  unsigned getIndex() const { return Index; }
  bool isParameterPack() const { return ParameterPack; }
  IdentifierInfo *getName() const { return Name; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Depth, Index, ParameterPack, Name);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, unsigned Depth,
                      unsigned Index, bool ParameterPack,
                      IdentifierInfo *Name) {
    ID.AddInteger(Depth);
    ID.AddInteger(Index);
    ID.AddBoolean(ParameterPack);
    ID.AddPointer(Name);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == TemplateTypeParm;
  }
  static bool classof(const TemplateTypeParmType *T) { return true; }
};

/// \brief Represents the result of substituting a type for a template
/// type parameter.
///
/// Within an instantiated template, all template type parameters have
/// been replaced with these.  They are used solely to record that a
/// type was originally written as a template type parameter;
/// therefore they are never canonical.
class SubstTemplateTypeParmType : public Type, public llvm::FoldingSetNode {
  // The original type parameter.
  const TemplateTypeParmType *Replaced;

  SubstTemplateTypeParmType(const TemplateTypeParmType *Param, QualType Canon)
    : Type(SubstTemplateTypeParm, Canon, Canon->isDependentType()),
      Replaced(Param) { }

  friend class ASTContext;

public:
  IdentifierInfo *getName() const { return Replaced->getName(); }

  /// Gets the template parameter that was substituted for.
  const TemplateTypeParmType *getReplacedParameter() const {
    return Replaced;
  }

  /// Gets the type that was substituted for the template
  /// parameter.
  QualType getReplacementType() const {
    return getCanonicalTypeInternal();
  }

  bool isSugared() const { return true; }
  QualType desugar() const { return getReplacementType(); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getReplacedParameter(), getReplacementType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      const TemplateTypeParmType *Replaced,
                      QualType Replacement) {
    ID.AddPointer(Replaced);
    ID.AddPointer(Replacement.getAsOpaquePtr());
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == SubstTemplateTypeParm;
  }
  static bool classof(const SubstTemplateTypeParmType *T) { return true; }
};

/// \brief Represents the type of a template specialization as written
/// in the source code.
///
/// Template specialization types represent the syntactic form of a
/// template-id that refers to a type, e.g., @c vector<int>. Some
/// template specialization types are syntactic sugar, whose canonical
/// type will point to some other type node that represents the
/// instantiation or class template specialization. For example, a
/// class template specialization type of @c vector<int> will refer to
/// a tag type for the instantiation
/// @c std::vector<int, std::allocator<int>>.
///
/// Other template specialization types, for which the template name
/// is dependent, may be canonical types. These types are always
/// dependent.
class TemplateSpecializationType
  : public Type, public llvm::FoldingSetNode {

  // FIXME: Currently needed for profiling expressions; can we avoid this?
  ASTContext &Context;

    /// \brief The name of the template being specialized.
  TemplateName Template;

  /// \brief - The number of template arguments named in this class
  /// template specialization.
  unsigned NumArgs;

  TemplateSpecializationType(ASTContext &Context,
                             TemplateName T,
                             const TemplateArgument *Args,
                             unsigned NumArgs, QualType Canon);

  virtual void Destroy(ASTContext& C);

  friend class ASTContext;  // ASTContext creates these

public:
  /// \brief Determine whether any of the given template arguments are
  /// dependent.
  static bool anyDependentTemplateArguments(const TemplateArgument *Args,
                                            unsigned NumArgs);

  static bool anyDependentTemplateArguments(const TemplateArgumentLoc *Args,
                                            unsigned NumArgs);

  static bool anyDependentTemplateArguments(const TemplateArgumentListInfo &);

  /// \brief Print a template argument list, including the '<' and '>'
  /// enclosing the template arguments.
  static std::string PrintTemplateArgumentList(const TemplateArgument *Args,
                                               unsigned NumArgs,
                                               const PrintingPolicy &Policy);

  static std::string PrintTemplateArgumentList(const TemplateArgumentLoc *Args,
                                               unsigned NumArgs,
                                               const PrintingPolicy &Policy);

  static std::string PrintTemplateArgumentList(const TemplateArgumentListInfo &,
                                               const PrintingPolicy &Policy);

  typedef const TemplateArgument * iterator;

  iterator begin() const { return getArgs(); }
  iterator end() const;

  /// \brief Retrieve the name of the template that we are specializing.
  TemplateName getTemplateName() const { return Template; }

  /// \brief Retrieve the template arguments.
  const TemplateArgument *getArgs() const {
    return reinterpret_cast<const TemplateArgument *>(this + 1);
  }

  /// \brief Retrieve the number of template arguments.
  unsigned getNumArgs() const { return NumArgs; }

  /// \brief Retrieve a specific template argument as a type.
  /// \precondition @c isArgType(Arg)
  const TemplateArgument &getArg(unsigned Idx) const;

  bool isSugared() const { return !isDependentType(); }
  QualType desugar() const { return getCanonicalTypeInternal(); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Template, getArgs(), NumArgs, Context);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, TemplateName T,
                      const TemplateArgument *Args, unsigned NumArgs,
                      ASTContext &Context);

  static bool classof(const Type *T) {
    return T->getTypeClass() == TemplateSpecialization;
  }
  static bool classof(const TemplateSpecializationType *T) { return true; }
};

/// \brief Represents a type that was referred to via a qualified
/// name, e.g., N::M::type.
///
/// This type is used to keep track of a type name as written in the
/// source code, including any nested-name-specifiers. The type itself
/// is always "sugar", used to express what was written in the source
/// code but containing no additional semantic information.
class QualifiedNameType : public Type, public llvm::FoldingSetNode {
  /// \brief The nested name specifier containing the qualifier.
  NestedNameSpecifier *NNS;

  /// \brief The type that this qualified name refers to.
  QualType NamedType;

  QualifiedNameType(NestedNameSpecifier *NNS, QualType NamedType,
                    QualType CanonType)
    : Type(QualifiedName, CanonType, NamedType->isDependentType()),
      NNS(NNS), NamedType(NamedType) { }

  friend class ASTContext;  // ASTContext creates these

public:
  /// \brief Retrieve the qualification on this type.
  NestedNameSpecifier *getQualifier() const { return NNS; }

  /// \brief Retrieve the type named by the qualified-id.
  QualType getNamedType() const { return NamedType; }

  /// \brief Remove a single level of sugar.
  QualType desugar() const { return getNamedType(); }

  /// \brief Returns whether this type directly provides sugar.
  bool isSugared() const { return true; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, NNS, NamedType);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, NestedNameSpecifier *NNS,
                      QualType NamedType) {
    ID.AddPointer(NNS);
    NamedType.Profile(ID);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == QualifiedName;
  }
  static bool classof(const QualifiedNameType *T) { return true; }
};

/// \brief Represents a 'typename' specifier that names a type within
/// a dependent type, e.g., "typename T::type".
///
/// TypenameType has a very similar structure to QualifiedNameType,
/// which also involves a nested-name-specifier following by a type,
/// and (FIXME!) both can even be prefixed by the 'typename'
/// keyword. However, the two types serve very different roles:
/// QualifiedNameType is a non-semantic type that serves only as sugar
/// to show how a particular type was written in the source
/// code. TypenameType, on the other hand, only occurs when the
/// nested-name-specifier is dependent, such that we cannot resolve
/// the actual type until after instantiation.
class TypenameType : public Type, public llvm::FoldingSetNode {
  /// \brief The nested name specifier containing the qualifier.
  NestedNameSpecifier *NNS;

  typedef llvm::PointerUnion<const IdentifierInfo *,
                             const TemplateSpecializationType *> NameType;

  /// \brief The type that this typename specifier refers to.
  NameType Name;

  TypenameType(NestedNameSpecifier *NNS, const IdentifierInfo *Name,
               QualType CanonType)
    : Type(Typename, CanonType, true), NNS(NNS), Name(Name) {
    assert(NNS->isDependent() &&
           "TypenameType requires a dependent nested-name-specifier");
  }

  TypenameType(NestedNameSpecifier *NNS, const TemplateSpecializationType *Ty,
               QualType CanonType)
    : Type(Typename, CanonType, true), NNS(NNS), Name(Ty) {
    assert(NNS->isDependent() &&
           "TypenameType requires a dependent nested-name-specifier");
  }

  friend class ASTContext;  // ASTContext creates these

public:
  /// \brief Retrieve the qualification on this type.
  NestedNameSpecifier *getQualifier() const { return NNS; }

  /// \brief Retrieve the type named by the typename specifier as an
  /// identifier.
  ///
  /// This routine will return a non-NULL identifier pointer when the
  /// form of the original typename was terminated by an identifier,
  /// e.g., "typename T::type".
  const IdentifierInfo *getIdentifier() const {
    return Name.dyn_cast<const IdentifierInfo *>();
  }

  /// \brief Retrieve the type named by the typename specifier as a
  /// type specialization.
  const TemplateSpecializationType *getTemplateId() const {
    return Name.dyn_cast<const TemplateSpecializationType *>();
  }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, NNS, Name);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, NestedNameSpecifier *NNS,
                      NameType Name) {
    ID.AddPointer(NNS);
    ID.AddPointer(Name.getOpaqueValue());
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == Typename;
  }
  static bool classof(const TypenameType *T) { return true; }
};

/// ObjCInterfaceType - Interfaces are the core concept in Objective-C for
/// object oriented design.  They basically correspond to C++ classes.  There
/// are two kinds of interface types, normal interfaces like "NSString" and
/// qualified interfaces, which are qualified with a protocol list like
/// "NSString<NSCopyable, NSAmazing>".
class ObjCInterfaceType : public Type, public llvm::FoldingSetNode {
  ObjCInterfaceDecl *Decl;

  // List of protocols for this protocol conforming object type
  // List is sorted on protocol name. No protocol is enterred more than once.
  llvm::SmallVector<ObjCProtocolDecl*, 4> Protocols;

  ObjCInterfaceType(QualType Canonical, ObjCInterfaceDecl *D,
                    ObjCProtocolDecl **Protos, unsigned NumP) :
    Type(ObjCInterface, Canonical, /*Dependent=*/false),
    Decl(D), Protocols(Protos, Protos+NumP) { }
  friend class ASTContext;  // ASTContext creates these.
public:
  ObjCInterfaceDecl *getDecl() const { return Decl; }

  /// getNumProtocols - Return the number of qualifying protocols in this
  /// interface type, or 0 if there are none.
  unsigned getNumProtocols() const { return Protocols.size(); }

  /// qual_iterator and friends: this provides access to the (potentially empty)
  /// list of protocols qualifying this interface.
  typedef llvm::SmallVector<ObjCProtocolDecl*, 8>::const_iterator qual_iterator;
  qual_iterator qual_begin() const { return Protocols.begin(); }
  qual_iterator qual_end() const   { return Protocols.end(); }
  bool qual_empty() const { return Protocols.size() == 0; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID);
  static void Profile(llvm::FoldingSetNodeID &ID,
                      const ObjCInterfaceDecl *Decl,
                      ObjCProtocolDecl **protocols, unsigned NumProtocols);

  static bool classof(const Type *T) {
    return T->getTypeClass() == ObjCInterface;
  }
  static bool classof(const ObjCInterfaceType *) { return true; }
};

/// ObjCObjectPointerType - Used to represent 'id', 'Interface *', 'id <p>',
/// and 'Interface <p> *'.
///
/// Duplicate protocols are removed and protocol list is canonicalized to be in
/// alphabetical order.
class ObjCObjectPointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType; // A builtin or interface type.

  // List of protocols for this protocol conforming object type
  // List is sorted on protocol name. No protocol is entered more than once.
  llvm::SmallVector<ObjCProtocolDecl*, 8> Protocols;

  ObjCObjectPointerType(QualType Canonical, QualType T,
                        ObjCProtocolDecl **Protos, unsigned NumP) :
    Type(ObjCObjectPointer, Canonical, /*Dependent=*/false),
    PointeeType(T), Protocols(Protos, Protos+NumP) { }
  friend class ASTContext;  // ASTContext creates these.

public:
  // Get the pointee type. Pointee will either be:
  // - a built-in type (for 'id' and 'Class').
  // - an interface type (for user-defined types).
  // - a TypedefType whose canonical type is an interface (as in 'T' below).
  //   For example: typedef NSObject T; T *var;
  QualType getPointeeType() const { return PointeeType; }

  const ObjCInterfaceType *getInterfaceType() const {
    return PointeeType->getAs<ObjCInterfaceType>();
  }
  /// getInterfaceDecl - returns an interface decl for user-defined types.
  ObjCInterfaceDecl *getInterfaceDecl() const {
    return getInterfaceType() ? getInterfaceType()->getDecl() : 0;
  }
  /// isObjCIdType - true for "id".
  bool isObjCIdType() const {
    return getPointeeType()->isSpecificBuiltinType(BuiltinType::ObjCId) &&
           !Protocols.size();
  }
  /// isObjCClassType - true for "Class".
  bool isObjCClassType() const {
    return getPointeeType()->isSpecificBuiltinType(BuiltinType::ObjCClass) &&
           !Protocols.size();
  }
  
  /// isObjCQualifiedIdType - true for "id <p>".
  bool isObjCQualifiedIdType() const {
    return getPointeeType()->isSpecificBuiltinType(BuiltinType::ObjCId) &&
           Protocols.size();
  }
  /// isObjCQualifiedClassType - true for "Class <p>".
  bool isObjCQualifiedClassType() const {
    return getPointeeType()->isSpecificBuiltinType(BuiltinType::ObjCClass) &&
           Protocols.size();
  }
  /// qual_iterator and friends: this provides access to the (potentially empty)
  /// list of protocols qualifying this interface.
  typedef llvm::SmallVector<ObjCProtocolDecl*, 8>::const_iterator qual_iterator;

  qual_iterator qual_begin() const { return Protocols.begin(); }
  qual_iterator qual_end() const   { return Protocols.end(); }
  bool qual_empty() const { return Protocols.size() == 0; }

  /// getNumProtocols - Return the number of qualifying protocols in this
  /// interface type, or 0 if there are none.
  unsigned getNumProtocols() const { return Protocols.size(); }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID);
  static void Profile(llvm::FoldingSetNodeID &ID, QualType T,
                      ObjCProtocolDecl **protocols, unsigned NumProtocols);
  static bool classof(const Type *T) {
    return T->getTypeClass() == ObjCObjectPointer;
  }
  static bool classof(const ObjCObjectPointerType *) { return true; }
};

/// A qualifier set is used to build a set of qualifiers.
class QualifierCollector : public Qualifiers {
  ASTContext *Context;

public:
  QualifierCollector(Qualifiers Qs = Qualifiers())
    : Qualifiers(Qs), Context(0) {}
  QualifierCollector(ASTContext &Context, Qualifiers Qs = Qualifiers())
    : Qualifiers(Qs), Context(&Context) {}

  void setContext(ASTContext &C) { Context = &C; }

  /// Collect any qualifiers on the given type and return an
  /// unqualified type.
  const Type *strip(QualType QT) {
    addFastQualifiers(QT.getLocalFastQualifiers());
    if (QT.hasLocalNonFastQualifiers()) {
      const ExtQuals *EQ = QT.getExtQualsUnsafe();
      Context = &EQ->getContext();
      addQualifiers(EQ->getQualifiers());
      return EQ->getBaseType();
    }
    return QT.getTypePtrUnsafe();
  }

  /// Apply the collected qualifiers to the given type.
  QualType apply(QualType QT) const;

  /// Apply the collected qualifiers to the given type.
  QualType apply(const Type* T) const;

};


// Inline function definitions.

inline bool QualType::isCanonical() const {
  const Type *T = getTypePtr();
  if (hasLocalQualifiers())
    return T->isCanonicalUnqualified() && !isa<ArrayType>(T);
  return T->isCanonicalUnqualified();
}

inline bool QualType::isCanonicalAsParam() const {
  if (hasLocalQualifiers()) return false;
  const Type *T = getTypePtr();
  return T->isCanonicalUnqualified() &&
           !isa<FunctionType>(T) && !isa<ArrayType>(T);
}

inline bool QualType::isConstQualified() const {
  return isLocalConstQualified() || 
              getTypePtr()->getCanonicalTypeInternal().isLocalConstQualified();
}

inline bool QualType::isRestrictQualified() const {
  return isLocalRestrictQualified() || 
            getTypePtr()->getCanonicalTypeInternal().isLocalRestrictQualified();
}


inline bool QualType::isVolatileQualified() const {
  return isLocalVolatileQualified() || 
  getTypePtr()->getCanonicalTypeInternal().isLocalVolatileQualified();
}
  
inline bool QualType::hasQualifiers() const {
  return hasLocalQualifiers() ||
                  getTypePtr()->getCanonicalTypeInternal().hasLocalQualifiers();
}
  
inline Qualifiers QualType::getQualifiers() const {
  Qualifiers Quals = getLocalQualifiers();
  Quals.addQualifiers(
                 getTypePtr()->getCanonicalTypeInternal().getLocalQualifiers());
  return Quals;
}
  
inline unsigned QualType::getCVRQualifiers() const {
  return getLocalCVRQualifiers() | 
              getTypePtr()->getCanonicalTypeInternal().getLocalCVRQualifiers();
}

/// getCVRQualifiersThroughArrayTypes - If there are CVR qualifiers for this
/// type, returns them. Otherwise, if this is an array type, recurses
/// on the element type until some qualifiers have been found or a non-array
/// type reached.
inline unsigned QualType::getCVRQualifiersThroughArrayTypes() const {
  if (unsigned Quals = getCVRQualifiers())
    return Quals;
  QualType CT = getTypePtr()->getCanonicalTypeInternal();
  if (const ArrayType *AT = dyn_cast<ArrayType>(CT))
    return AT->getElementType().getCVRQualifiersThroughArrayTypes();
  return 0;
}

inline void QualType::removeConst() {
  removeFastQualifiers(Qualifiers::Const);
}

inline void QualType::removeRestrict() {
  removeFastQualifiers(Qualifiers::Restrict);
}

inline void QualType::removeVolatile() {
  QualifierCollector Qc;
  const Type *Ty = Qc.strip(*this);
  if (Qc.hasVolatile()) {
    Qc.removeVolatile();
    *this = Qc.apply(Ty);
  }
}

inline void QualType::removeCVRQualifiers(unsigned Mask) {
  assert(!(Mask & ~Qualifiers::CVRMask) && "mask has non-CVR bits");

  // Fast path: we don't need to touch the slow qualifiers.
  if (!(Mask & ~Qualifiers::FastMask)) {
    removeFastQualifiers(Mask);
    return;
  }

  QualifierCollector Qc;
  const Type *Ty = Qc.strip(*this);
  Qc.removeCVRQualifiers(Mask);
  *this = Qc.apply(Ty);
}

/// getAddressSpace - Return the address space of this type.
inline unsigned QualType::getAddressSpace() const {
  if (hasLocalNonFastQualifiers()) {
    const ExtQuals *EQ = getExtQualsUnsafe();
    if (EQ->hasAddressSpace())
      return EQ->getAddressSpace();
  }

  QualType CT = getTypePtr()->getCanonicalTypeInternal();
  if (CT.hasLocalNonFastQualifiers()) {
    const ExtQuals *EQ = CT.getExtQualsUnsafe();
    if (EQ->hasAddressSpace())
      return EQ->getAddressSpace();
  }

  if (const ArrayType *AT = dyn_cast<ArrayType>(CT))
    return AT->getElementType().getAddressSpace();
  if (const RecordType *RT = dyn_cast<RecordType>(CT))
    return RT->getAddressSpace();
  return 0;
}

/// getObjCGCAttr - Return the gc attribute of this type.
inline Qualifiers::GC QualType::getObjCGCAttr() const {
  if (hasLocalNonFastQualifiers()) {
    const ExtQuals *EQ = getExtQualsUnsafe();
    if (EQ->hasObjCGCAttr())
      return EQ->getObjCGCAttr();
  }

  QualType CT = getTypePtr()->getCanonicalTypeInternal();
  if (CT.hasLocalNonFastQualifiers()) {
    const ExtQuals *EQ = CT.getExtQualsUnsafe();
    if (EQ->hasObjCGCAttr())
      return EQ->getObjCGCAttr();
  }

  if (const ArrayType *AT = dyn_cast<ArrayType>(CT))
      return AT->getElementType().getObjCGCAttr();
  if (const ObjCObjectPointerType *PT = CT->getAs<ObjCObjectPointerType>())
    return PT->getPointeeType().getObjCGCAttr();
  // We most look at all pointer types, not just pointer to interface types.
  if (const PointerType *PT = CT->getAs<PointerType>())
    return PT->getPointeeType().getObjCGCAttr();
  return Qualifiers::GCNone;
}

  /// getNoReturnAttr - Returns true if the type has the noreturn attribute,
  /// false otherwise.
inline bool QualType::getNoReturnAttr() const {
  QualType CT = getTypePtr()->getCanonicalTypeInternal();
  if (const PointerType *PT = getTypePtr()->getAs<PointerType>()) {
    if (const FunctionType *FT = PT->getPointeeType()->getAs<FunctionType>())
      return FT->getNoReturnAttr();
  } else if (const FunctionType *FT = getTypePtr()->getAs<FunctionType>())
    return FT->getNoReturnAttr();

  return false;
}

/// isMoreQualifiedThan - Determine whether this type is more
/// qualified than the Other type. For example, "const volatile int"
/// is more qualified than "const int", "volatile int", and
/// "int". However, it is not more qualified than "const volatile
/// int".
inline bool QualType::isMoreQualifiedThan(QualType Other) const {
  // FIXME: work on arbitrary qualifiers
  unsigned MyQuals = this->getCVRQualifiersThroughArrayTypes();
  unsigned OtherQuals = Other.getCVRQualifiersThroughArrayTypes();
  if (getAddressSpace() != Other.getAddressSpace())
    return false;
  return MyQuals != OtherQuals && (MyQuals | OtherQuals) == MyQuals;
}

/// isAtLeastAsQualifiedAs - Determine whether this type is at last
/// as qualified as the Other type. For example, "const volatile
/// int" is at least as qualified as "const int", "volatile int",
/// "int", and "const volatile int".
inline bool QualType::isAtLeastAsQualifiedAs(QualType Other) const {
  // FIXME: work on arbitrary qualifiers
  unsigned MyQuals = this->getCVRQualifiersThroughArrayTypes();
  unsigned OtherQuals = Other.getCVRQualifiersThroughArrayTypes();
  if (getAddressSpace() != Other.getAddressSpace())
    return false;
  return (MyQuals | OtherQuals) == MyQuals;
}

/// getNonReferenceType - If Type is a reference type (e.g., const
/// int&), returns the type that the reference refers to ("const
/// int"). Otherwise, returns the type itself. This routine is used
/// throughout Sema to implement C++ 5p6:
///
///   If an expression initially has the type "reference to T" (8.3.2,
///   8.5.3), the type is adjusted to "T" prior to any further
///   analysis, the expression designates the object or function
///   denoted by the reference, and the expression is an lvalue.
inline QualType QualType::getNonReferenceType() const {
  if (const ReferenceType *RefType = (*this)->getAs<ReferenceType>())
    return RefType->getPointeeType();
  else
    return *this;
}

inline const ObjCInterfaceType *Type::getAsPointerToObjCInterfaceType() const {
  if (const PointerType *PT = getAs<PointerType>())
    return PT->getPointeeType()->getAs<ObjCInterfaceType>();
  return 0;
}

inline bool Type::isFunctionType() const {
  return isa<FunctionType>(CanonicalType);
}
inline bool Type::isPointerType() const {
  return isa<PointerType>(CanonicalType);
}
inline bool Type::isAnyPointerType() const {
  return isPointerType() || isObjCObjectPointerType();
}
inline bool Type::isBlockPointerType() const {
  return isa<BlockPointerType>(CanonicalType);
}
inline bool Type::isReferenceType() const {
  return isa<ReferenceType>(CanonicalType);
}
inline bool Type::isLValueReferenceType() const {
  return isa<LValueReferenceType>(CanonicalType);
}
inline bool Type::isRValueReferenceType() const {
  return isa<RValueReferenceType>(CanonicalType);
}
inline bool Type::isFunctionPointerType() const {
  if (const PointerType* T = getAs<PointerType>())
    return T->getPointeeType()->isFunctionType();
  else
    return false;
}
inline bool Type::isMemberPointerType() const {
  return isa<MemberPointerType>(CanonicalType);
}
inline bool Type::isMemberFunctionPointerType() const {
  if (const MemberPointerType* T = getAs<MemberPointerType>())
    return T->getPointeeType()->isFunctionType();
  else
    return false;
}
inline bool Type::isArrayType() const {
  return isa<ArrayType>(CanonicalType);
}
inline bool Type::isConstantArrayType() const {
  return isa<ConstantArrayType>(CanonicalType);
}
inline bool Type::isIncompleteArrayType() const {
  return isa<IncompleteArrayType>(CanonicalType);
}
inline bool Type::isVariableArrayType() const {
  return isa<VariableArrayType>(CanonicalType);
}
inline bool Type::isDependentSizedArrayType() const {
  return isa<DependentSizedArrayType>(CanonicalType);
}
inline bool Type::isRecordType() const {
  return isa<RecordType>(CanonicalType);
}
inline bool Type::isAnyComplexType() const {
  return isa<ComplexType>(CanonicalType);
}
inline bool Type::isVectorType() const {
  return isa<VectorType>(CanonicalType);
}
inline bool Type::isExtVectorType() const {
  return isa<ExtVectorType>(CanonicalType);
}
inline bool Type::isObjCObjectPointerType() const {
  return isa<ObjCObjectPointerType>(CanonicalType);
}
inline bool Type::isObjCInterfaceType() const {
  return isa<ObjCInterfaceType>(CanonicalType);
}
inline bool Type::isObjCQualifiedIdType() const {
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>())
    return OPT->isObjCQualifiedIdType();
  return false;
}
inline bool Type::isObjCQualifiedClassType() const {
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>())
    return OPT->isObjCQualifiedClassType();
  return false;
}
inline bool Type::isObjCIdType() const {
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>())
    return OPT->isObjCIdType();
  return false;
}
inline bool Type::isObjCClassType() const {
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>())
    return OPT->isObjCClassType();
  return false;
}
inline bool Type::isObjCSelType() const {
  if (const PointerType *OPT = getAs<PointerType>())
    return OPT->getPointeeType()->isSpecificBuiltinType(BuiltinType::ObjCSel);
  return false;
}
inline bool Type::isObjCBuiltinType() const {
  return isObjCIdType() || isObjCClassType() || isObjCSelType();
}
inline bool Type::isTemplateTypeParmType() const {
  return isa<TemplateTypeParmType>(CanonicalType);
}

inline bool Type::isSpecificBuiltinType(unsigned K) const {
  if (const BuiltinType *BT = getAs<BuiltinType>())
    if (BT->getKind() == (BuiltinType::Kind) K)
      return true;
  return false;
}

/// \brief Determines whether this is a type for which one can define
/// an overloaded operator.
inline bool Type::isOverloadableType() const {
  return isDependentType() || isRecordType() || isEnumeralType();
}

inline bool Type::hasPointerRepresentation() const {
  return (isPointerType() || isReferenceType() || isBlockPointerType() ||
          isObjCInterfaceType() || isObjCObjectPointerType() ||
          isObjCQualifiedInterfaceType() || isNullPtrType());
}

inline bool Type::hasObjCPointerRepresentation() const {
  return (isObjCInterfaceType() || isObjCObjectPointerType() ||
          isObjCQualifiedInterfaceType());
}

/// Insertion operator for diagnostics.  This allows sending QualType's into a
/// diagnostic with <<.
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           QualType T) {
  DB.AddTaggedVal(reinterpret_cast<intptr_t>(T.getAsOpaquePtr()),
                  Diagnostic::ak_qualtype);
  return DB;
}

// Helper class template that is used by Type::getAs to ensure that one does
// not try to look through a qualified type to get to an array type.
template<typename T,
         bool isArrayType = (llvm::is_same<T, ArrayType>::value ||
                             llvm::is_base_of<ArrayType, T>::value)>
struct ArrayType_cannot_be_used_with_getAs { };
  
template<typename T>
struct ArrayType_cannot_be_used_with_getAs<T, true>;
  
/// Member-template getAs<specific type>'.
template <typename T> const T *Type::getAs() const {
  ArrayType_cannot_be_used_with_getAs<T> at;
  (void)at;
  
  // If this is directly a T type, return it.
  if (const T *Ty = dyn_cast<T>(this))
    return Ty;

  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<T>(CanonicalType))
    return 0;

  // If this is a typedef for the type, strip the typedef off without
  // losing all typedef information.
  return cast<T>(getUnqualifiedDesugaredType());
}

}  // end namespace clang

#endif
