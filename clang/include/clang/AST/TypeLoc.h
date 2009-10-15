//===--- TypeLoc.h - Type Source Info Wrapper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TypeLoc interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TYPELOC_H
#define LLVM_CLANG_AST_TYPELOC_H

#include "clang/AST/Type.h"

namespace clang {
  class ParmVarDecl;
  class TypeSpecLoc;
  class DeclaratorInfo;
  class UnqualTypeLoc;

/// \brief Base wrapper for a particular "section" of type source info.
///
/// A client should use the TypeLoc subclasses through cast/dyn_cast in order to
/// get at the actual information.
class TypeLoc {
protected:
  // The correctness of this relies on the property that, for Type *Ty,
  //   QualType(Ty, 0).getAsOpaquePtr() == (void*) Ty
  void *Ty;
  void *Data;

public:
  TypeLoc() : Ty(0), Data(0) { }
  TypeLoc(QualType ty, void *opaqueData)
    : Ty(ty.getAsOpaquePtr()), Data(opaqueData) { }
  TypeLoc(Type *ty, void *opaqueData)
    : Ty(ty), Data(opaqueData) { }

  bool isNull() const { return !Ty; }
  operator bool() const { return Ty; }

  /// \brief Returns the size of type source info data block for the given type.
  static unsigned getFullDataSizeForType(QualType Ty);

  /// \brief Get the type for which this source info wrapper provides
  /// information.
  QualType getSourceType() const { return QualType::getFromOpaquePtr(Ty); }

  Type *getSourceTypePtr() const {
    return QualType::getFromOpaquePtr(Ty).getTypePtr();
  }

  /// \brief Get the pointer where source information is stored.
  void *getOpaqueData() const { return Data; }

  SourceRange getSourceRange() const;

  /// \brief Find the TypeSpecLoc that is part of this TypeLoc.
  TypeSpecLoc getTypeSpecLoc() const;

  /// \brief Find the TypeSpecLoc that is part of this TypeLoc and return its
  /// SourceRange.
  SourceRange getTypeSpecRange() const;

  /// \brief Returns the size of the type source info data block.
  unsigned getFullDataSize() const {
    return getFullDataSizeForType(getSourceType());
  }

  /// \brief Get the next TypeLoc pointed by this TypeLoc, e.g for "int*" the
  /// TypeLoc is a PointerLoc and next TypeLoc is for "int".
  TypeLoc getNextTypeLoc() const;

  /// \brief Skips past any qualifiers, if this is qualified.
  UnqualTypeLoc getUnqualifiedLoc() const;

  friend bool operator==(const TypeLoc &LHS, const TypeLoc &RHS) {
    return LHS.Ty == RHS.Ty && LHS.Data == RHS.Data;
  }

  friend bool operator!=(const TypeLoc &LHS, const TypeLoc &RHS) {
    return !(LHS == RHS);
  }

  static bool classof(const TypeLoc *TL) { return true; }
};

/// \brief Wrapper of type source information for a type with
/// no direct quqlaifiers.
class UnqualTypeLoc : public TypeLoc {
public:
  UnqualTypeLoc() {}
  UnqualTypeLoc(Type *Ty, void *Data) : TypeLoc(Ty, Data) {}

  Type *getSourceTypePtr() const {
    return reinterpret_cast<Type*>(Ty);
  }

  static bool classof(const TypeLoc *TL) {
    return !TL->getSourceType().hasQualifiers();
  }
  static bool classof(const UnqualTypeLoc *TL) { return true; }
};

/// \brief Wrapper of type source information for a type with
/// non-trivial direct qualifiers.
///
/// Currently, we intentionally do not provide source location for
/// type qualifiers.
class QualifiedLoc : public TypeLoc {
public:
  SourceRange getSourceRange() const {
    return SourceRange();
  }

  UnqualTypeLoc getUnqualifiedLoc() const {
    return UnqualTypeLoc(getSourceTypePtr(), Data);
  }

  /// \brief Returns the size of the type source info data block that is
  /// specific to this type.
  unsigned getLocalDataSize() const {
    // In fact, we don't currently preserve any location information
    // for qualifiers.
    return 0;
  }

  /// \brief Returns the size of the type source info data block.
  unsigned getFullDataSize() const {
    return getLocalDataSize() + 
      getFullDataSizeForType(getSourceType().getUnqualifiedType());
  }

  static bool classof(const TypeLoc *TL) {
    return TL->getSourceType().hasQualifiers();
  }
  static bool classof(const QualifiedLoc *TL) { return true; }
};

inline UnqualTypeLoc TypeLoc::getUnqualifiedLoc() const {
  if (isa<QualifiedLoc>(this))
    return cast<QualifiedLoc>(this)->getUnqualifiedLoc();
  return cast<UnqualTypeLoc>(*this);
}

/// \brief Base wrapper of type source info data for type-spec types.
class TypeSpecLoc : public UnqualTypeLoc  {
public:
  static bool classof(const TypeLoc *TL) {
    return (UnqualTypeLoc::classof(TL) &&
            classof(static_cast<const UnqualTypeLoc*>(TL)));
  }
  static bool classof(const UnqualTypeLoc *TL);
  static bool classof(const TypeSpecLoc *TL) { return true; }
};

inline SourceRange TypeLoc::getTypeSpecRange() const {
  return getTypeSpecLoc().getSourceRange();
}
  
/// \brief Base wrapper of type source info data for types part of a declarator,
/// excluding type-spec types.
class DeclaratorLoc : public UnqualTypeLoc  {
public:
  /// \brief Find the TypeSpecLoc that is part of this DeclaratorLoc.
  TypeSpecLoc getTypeSpecLoc() const;

  static bool classof(const TypeLoc *TL) {
    return (UnqualTypeLoc::classof(TL) &&
            classof(static_cast<const UnqualTypeLoc*>(TL)));
  }
  static bool classof(const UnqualTypeLoc *TL);
  static bool classof(const DeclaratorLoc *TL) { return true; }
};


/// A metaprogramming base class for TypeLoc classes which correspond
/// to a particular Type subclass.
///
/// \param Base a class from which to derive
/// \param Derived the class deriving from this one
/// \param TypeClass the concrete Type subclass which this 
/// \param LocalData the structure type of local location data for
///   this type
///
/// sizeof(LocalData) needs to be a multiple of sizeof(void*) or
/// else the world will end.
///
/// TypeLocs with non-constant amounts of local data should override
/// getExtraLocalDataSize(); getExtraLocalData() will then point to
/// this extra memory.
///
/// TypeLocs with an inner type should override ha
template <class Base, class Derived, class TypeClass, class LocalData>
class ConcreteTypeLoc : public Base {

  const Derived *asDerived() const {
    return static_cast<const Derived*>(this);
  }

public:
  unsigned getLocalDataSize() const {
    return sizeof(LocalData) + asDerived()->getExtraLocalDataSize();
  }
  // Give a default implementation that's useful for leaf types.
  unsigned getFullDataSize() const {
    return asDerived()->getLocalDataSize() + getInnerTypeSize();
  }

  static bool classof(const TypeLoc *TL) {
    return Derived::classofType(TL->getSourceTypePtr());
  }
  static bool classof(const UnqualTypeLoc *TL) {
    return Derived::classofType(TL->getSourceTypePtr());
  }
  static bool classof(const Derived *TL) {
    return true;
  }

  static bool classofType(const Type *Ty) {
    return TypeClass::classof(Ty);
  }

protected:
  TypeClass *getTypePtr() const {
    return cast<TypeClass>(Base::getSourceTypePtr());
  }

  unsigned getExtraLocalDataSize() const {
    return 0;
  }

  LocalData *getLocalData() const {
    return static_cast<LocalData*>(Base::Data);
  }

  /// Gets a pointer past the Info structure; useful for classes with
  /// local data that can't be captured in the Info (e.g. because it's
  /// of variable size).
  void *getExtraLocalData() const {
    return getLocalData() + 1;
  }
  
  void *getNonLocalData() const {
    return static_cast<char*>(Base::Data) + asDerived()->getLocalDataSize();
  }

  bool hasInnerType() const {
    return false;
  }

  TypeLoc getInnerTypeLoc() const {
    assert(asDerived()->hasInnerType());
    return TypeLoc(asDerived()->getInnerType(), getNonLocalData());
  }

private:
  unsigned getInnerTypeSize() const {
    if (asDerived()->hasInnerType())
      return getInnerTypeLoc().getFullDataSize();
    return 0;
  }

  // Required here because my metaprogramming is too weak to avoid it.
  QualType getInnerType() const {
    assert(0 && "getInnerType() not overridden");
    return QualType();
  }
};


struct DefaultTypeSpecLocInfo {
  SourceLocation StartLoc;
};

/// \brief The default wrapper for type-spec types that are not handled by
/// another specific wrapper.
class DefaultTypeSpecLoc : public ConcreteTypeLoc<TypeSpecLoc,
                                                  DefaultTypeSpecLoc,
                                                  Type,
                                                  DefaultTypeSpecLocInfo> {
public:
  SourceLocation getStartLoc() const {
    return getLocalData()->StartLoc;
  }
  void setStartLoc(SourceLocation Loc) {
    getLocalData()->StartLoc = Loc;
  }
  SourceRange getSourceRange() const {
    return SourceRange(getStartLoc(), getStartLoc());
  }

  static bool classofType(const Type *T);
};


struct TypedefLocInfo {
  SourceLocation NameLoc;
};

/// \brief Wrapper for source info for typedefs.
class TypedefLoc : public ConcreteTypeLoc<TypeSpecLoc,TypedefLoc,
                                          TypedefType,TypedefLocInfo> {
public:
  SourceLocation getNameLoc() const {
    return getLocalData()->NameLoc;
  }
  void setNameLoc(SourceLocation Loc) {
    getLocalData()->NameLoc = Loc;
  }
  SourceRange getSourceRange() const {
    return SourceRange(getNameLoc(), getNameLoc());
  }

  TypedefDecl *getTypedefDecl() const {
    return getTypePtr()->getDecl();
  }
};


struct ObjCInterfaceLocInfo {
  SourceLocation NameLoc;
};

/// \brief Wrapper for source info for ObjC interfaces.
class ObjCInterfaceLoc : public ConcreteTypeLoc<TypeSpecLoc,
                                                ObjCInterfaceLoc,
                                                ObjCInterfaceType,
                                                ObjCInterfaceLocInfo> {
public:
  SourceLocation getNameLoc() const {
    return getLocalData()->NameLoc;
  }
  void setNameLoc(SourceLocation Loc) {
    getLocalData()->NameLoc = Loc;
  }
  SourceRange getSourceRange() const {
    return SourceRange(getNameLoc(), getNameLoc());
  }

  ObjCInterfaceDecl *getIFaceDecl() const {
    return getTypePtr()->getDecl();
  }
};


struct ObjCProtocolListLocInfo {
  SourceLocation LAngleLoc, RAngleLoc;
};

/// \brief Wrapper for source info for ObjC protocol lists.
class ObjCProtocolListLoc : public ConcreteTypeLoc<TypeSpecLoc,
                                                   ObjCProtocolListLoc,
                                                   ObjCProtocolListType,
                                                   ObjCProtocolListLocInfo> {
  // SourceLocations are stored after Info, one for each Protocol.
  SourceLocation *getProtocolLocArray() const {
    return (SourceLocation*) getExtraLocalData();
  }

public:
  SourceLocation getLAngleLoc() const {
    return getLocalData()->LAngleLoc;
  }
  void setLAngleLoc(SourceLocation Loc) {
    getLocalData()->LAngleLoc = Loc;
  }

  SourceLocation getRAngleLoc() const {
    return getLocalData()->RAngleLoc;
  }
  void setRAngleLoc(SourceLocation Loc) {
    getLocalData()->RAngleLoc = Loc;
  }

  unsigned getNumProtocols() const {
    return getTypePtr()->getNumProtocols();
  }

  SourceLocation getProtocolLoc(unsigned i) const {
    assert(i < getNumProtocols() && "Index is out of bounds!");
    return getProtocolLocArray()[i];
  }
  void setProtocolLoc(unsigned i, SourceLocation Loc) {
    assert(i < getNumProtocols() && "Index is out of bounds!");
    getProtocolLocArray()[i] = Loc;
  }

  ObjCProtocolDecl *getProtocol(unsigned i) const {
    assert(i < getNumProtocols() && "Index is out of bounds!");
    return *(getTypePtr()->qual_begin() + i);
  }
  
  TypeLoc getBaseTypeLoc() const {
    return getInnerTypeLoc();
  }

  SourceRange getSourceRange() const {
    return SourceRange(getLAngleLoc(), getRAngleLoc());
  }

  /// \brief Returns the size of the type source info data block that is
  /// specific to this type.
  unsigned getExtraLocalDataSize() const {
    return getNumProtocols() * sizeof(SourceLocation);
  }

  bool hasInnerType() const { return true; }
  QualType getInnerType() const { return getTypePtr()->getBaseType(); }
};


struct PointerLocInfo {
  SourceLocation StarLoc;
};

/// \brief Wrapper for source info for pointers.
class PointerLoc : public ConcreteTypeLoc<DeclaratorLoc,
                                          PointerLoc,
                                          PointerType,
                                          PointerLocInfo> {
public:
  SourceLocation getStarLoc() const {
    return getLocalData()->StarLoc;
  }
  void setStarLoc(SourceLocation Loc) {
    getLocalData()->StarLoc = Loc;
  }

  TypeLoc getPointeeLoc() const {
    return getInnerTypeLoc();
  }

  /// \brief Find the TypeSpecLoc that is part of this PointerLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getPointeeLoc().getTypeSpecLoc();
  }

  SourceRange getSourceRange() const {
    return SourceRange(getStarLoc(), getStarLoc());
  }

  bool hasInnerType() const { return true; }
  QualType getInnerType() const { return getTypePtr()->getPointeeType(); }
};


struct BlockPointerLocInfo {
  SourceLocation CaretLoc;
};

/// \brief Wrapper for source info for block pointers.
class BlockPointerLoc : public ConcreteTypeLoc<DeclaratorLoc,
                                               BlockPointerLoc,
                                               BlockPointerType,
                                               BlockPointerLocInfo> {
public:
  SourceLocation getCaretLoc() const {
    return getLocalData()->CaretLoc;
  }
  void setCaretLoc(SourceLocation Loc) {
    getLocalData()->CaretLoc = Loc;
  }

  TypeLoc getPointeeLoc() const {
    return getInnerTypeLoc();
  }

  /// \brief Find the TypeSpecLoc that is part of this BlockPointerLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getPointeeLoc().getTypeSpecLoc();
  }

  SourceRange getSourceRange() const {
    return SourceRange(getCaretLoc(), getCaretLoc());
  }

  bool hasInnerType() const { return true; }
  QualType getInnerType() const { return getTypePtr()->getPointeeType(); }
};


struct MemberPointerLocInfo {
  SourceLocation StarLoc;
};

/// \brief Wrapper for source info for member pointers.
class MemberPointerLoc : public ConcreteTypeLoc<DeclaratorLoc,
                                                MemberPointerLoc,
                                                MemberPointerType,
                                                MemberPointerLocInfo> {
public:
  SourceLocation getStarLoc() const {
    return getLocalData()->StarLoc;
  }
  void setStarLoc(SourceLocation Loc) {
    getLocalData()->StarLoc = Loc;
  }

  TypeLoc getPointeeLoc() const {
    return getInnerTypeLoc();
  }

  /// \brief Find the TypeSpecLoc that is part of this MemberPointerLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getPointeeLoc().getTypeSpecLoc();
  }

  SourceRange getSourceRange() const {
    return SourceRange(getStarLoc(), getStarLoc());
  }

  bool hasInnerType() const { return true; }
  QualType getInnerType() const { return getTypePtr()->getPointeeType(); }
};


struct ReferenceLocInfo {
  SourceLocation AmpLoc;
};

/// \brief Wrapper for source info for references.
class ReferenceLoc : public ConcreteTypeLoc<DeclaratorLoc,
                                            ReferenceLoc,
                                            ReferenceType,
                                            ReferenceLocInfo> {
public:
  SourceLocation getAmpLoc() const {
    return getLocalData()->AmpLoc;
  }
  void setAmpLoc(SourceLocation Loc) {
    getLocalData()->AmpLoc = Loc;
  }

  TypeLoc getPointeeLoc() const {
    return TypeLoc(getTypePtr()->getPointeeType(), getNonLocalData());
  }

  /// \brief Find the TypeSpecLoc that is part of this ReferenceLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getPointeeLoc().getTypeSpecLoc();
  }

  SourceRange getSourceRange() const {
    return SourceRange(getAmpLoc(), getAmpLoc());
  }

  bool hasInnerType() const { return true; }
  QualType getInnerType() const { return getTypePtr()->getPointeeType(); }
};


struct FunctionLocInfo {
  SourceLocation LParenLoc, RParenLoc;
};

/// \brief Wrapper for source info for functions.
class FunctionLoc : public ConcreteTypeLoc<DeclaratorLoc,
                                           FunctionLoc,
                                           FunctionType,
                                           FunctionLocInfo> {
  // ParmVarDecls* are stored after Info, one for each argument.
  ParmVarDecl **getParmArray() const {
    return (ParmVarDecl**) getExtraLocalData();
  }

public:
  SourceLocation getLParenLoc() const {
    return getLocalData()->LParenLoc;
  }
  void setLParenLoc(SourceLocation Loc) {
    getLocalData()->LParenLoc = Loc;
  }

  SourceLocation getRParenLoc() const {
    return getLocalData()->RParenLoc;
  }
  void setRParenLoc(SourceLocation Loc) {
    getLocalData()->RParenLoc = Loc;
  }

  unsigned getNumArgs() const {
    if (isa<FunctionNoProtoType>(getTypePtr()))
      return 0;
    return cast<FunctionProtoType>(getTypePtr())->getNumArgs();
  }
  ParmVarDecl *getArg(unsigned i) const { return getParmArray()[i]; }
  void setArg(unsigned i, ParmVarDecl *VD) { getParmArray()[i] = VD; }

  TypeLoc getArgLoc(unsigned i) const;

  TypeLoc getResultLoc() const {
    return getInnerTypeLoc();
  }

  /// \brief Find the TypeSpecLoc that is part of this FunctionLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getResultLoc().getTypeSpecLoc();
  }
  SourceRange getSourceRange() const {
    return SourceRange(getLParenLoc(), getRParenLoc());
  }

  /// \brief Returns the size of the type source info data block that is
  /// specific to this type.
  unsigned getExtraLocalDataSize() const {
    return getNumArgs() * sizeof(ParmVarDecl*);
  }

  bool hasInnerType() const { return true; }
  QualType getInnerType() const { return getTypePtr()->getResultType(); }
};


struct ArrayLocInfo {
  SourceLocation LBracketLoc, RBracketLoc;
  Expr *Size;
};

/// \brief Wrapper for source info for arrays.
class ArrayLoc : public ConcreteTypeLoc<DeclaratorLoc,
                                        ArrayLoc,
                                        ArrayType,
                                        ArrayLocInfo> {
public:
  SourceLocation getLBracketLoc() const {
    return getLocalData()->LBracketLoc;
  }
  void setLBracketLoc(SourceLocation Loc) {
    getLocalData()->LBracketLoc = Loc;
  }

  SourceLocation getRBracketLoc() const {
    return getLocalData()->RBracketLoc;
  }
  void setRBracketLoc(SourceLocation Loc) {
    getLocalData()->RBracketLoc = Loc;
  }

  Expr *getSizeExpr() const {
    return getLocalData()->Size;
  }
  void setSizeExpr(Expr *Size) {
    getLocalData()->Size = Size;
  }

  TypeLoc getElementLoc() const {
    return getInnerTypeLoc();
  }

  /// \brief Find the TypeSpecLoc that is part of this ArrayLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getElementLoc().getTypeSpecLoc();
  }
  SourceRange getSourceRange() const {
    return SourceRange(getLBracketLoc(), getRBracketLoc());
  }

  bool hasInnerType() const { return true; }
  QualType getInnerType() const { return getTypePtr()->getElementType(); }
};

}

#endif
