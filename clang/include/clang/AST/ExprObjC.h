//===--- ExprObjC.h - Classes for representing ObjC expressions -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ExprObjC interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_EXPROBJC_H
#define LLVM_CLANG_AST_EXPROBJC_H

#include "clang/AST/Expr.h"
#include "clang/Basic/IdentifierTable.h"

namespace clang {
  class IdentifierInfo;
  class ASTContext;
  class ObjCMethodDecl;
  class ObjCPropertyDecl;

/// ObjCStringLiteral, used for Objective-C string literals
/// i.e. @"foo".
class ObjCStringLiteral : public Expr {
  Stmt *String;
  SourceLocation AtLoc;
public:
  ObjCStringLiteral(StringLiteral *SL, QualType T, SourceLocation L)
    : Expr(ObjCStringLiteralClass, T, false, false), String(SL), AtLoc(L) {}
  explicit ObjCStringLiteral(EmptyShell Empty)
    : Expr(ObjCStringLiteralClass, Empty) {}

  StringLiteral *getString() { return cast<StringLiteral>(String); }
  const StringLiteral *getString() const { return cast<StringLiteral>(String); }
  void setString(StringLiteral *S) { String = S; }

  SourceLocation getAtLoc() const { return AtLoc; }
  void setAtLoc(SourceLocation L) { AtLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(AtLoc, String->getLocEnd());
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCStringLiteralClass;
  }
  static bool classof(const ObjCStringLiteral *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ObjCEncodeExpr, used for @encode in Objective-C.  @encode has the same type
/// and behavior as StringLiteral except that the string initializer is obtained
/// from ASTContext with the encoding type as an argument.
class ObjCEncodeExpr : public Expr {
  TypeSourceInfo *EncodedType;
  SourceLocation AtLoc, RParenLoc;
public:
  ObjCEncodeExpr(QualType T, TypeSourceInfo *EncodedType,
                 SourceLocation at, SourceLocation rp)
    : Expr(ObjCEncodeExprClass, T, EncodedType->getType()->isDependentType(),
           EncodedType->getType()->isDependentType()), 
      EncodedType(EncodedType), AtLoc(at), RParenLoc(rp) {}

  explicit ObjCEncodeExpr(EmptyShell Empty) : Expr(ObjCEncodeExprClass, Empty){}


  SourceLocation getAtLoc() const { return AtLoc; }
  void setAtLoc(SourceLocation L) { AtLoc = L; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  QualType getEncodedType() const { return EncodedType->getType(); }

  TypeSourceInfo *getEncodedTypeSourceInfo() const { return EncodedType; }
  void setEncodedTypeSourceInfo(TypeSourceInfo *EncType) { 
    EncodedType = EncType; 
  }

  virtual SourceRange getSourceRange() const {
    return SourceRange(AtLoc, RParenLoc);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCEncodeExprClass;
  }
  static bool classof(const ObjCEncodeExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ObjCSelectorExpr used for @selector in Objective-C.
class ObjCSelectorExpr : public Expr {
  Selector SelName;
  SourceLocation AtLoc, RParenLoc;
public:
  ObjCSelectorExpr(QualType T, Selector selInfo,
                   SourceLocation at, SourceLocation rp)
  : Expr(ObjCSelectorExprClass, T, false, false), SelName(selInfo), AtLoc(at),
    RParenLoc(rp){}
  explicit ObjCSelectorExpr(EmptyShell Empty)
   : Expr(ObjCSelectorExprClass, Empty) {}

  Selector getSelector() const { return SelName; }
  void setSelector(Selector S) { SelName = S; }

  SourceLocation getAtLoc() const { return AtLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setAtLoc(SourceLocation L) { AtLoc = L; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(AtLoc, RParenLoc);
  }

  /// getNumArgs - Return the number of actual arguments to this call.
  unsigned getNumArgs() const { return SelName.getNumArgs(); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCSelectorExprClass;
  }
  static bool classof(const ObjCSelectorExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ObjCProtocolExpr used for protocol expression in Objective-C.  This is used
/// as: @protocol(foo), as in:
///   obj conformsToProtocol:@protocol(foo)]
/// The return type is "Protocol*".
class ObjCProtocolExpr : public Expr {
  ObjCProtocolDecl *TheProtocol;
  SourceLocation AtLoc, RParenLoc;
public:
  ObjCProtocolExpr(QualType T, ObjCProtocolDecl *protocol,
                   SourceLocation at, SourceLocation rp)
  : Expr(ObjCProtocolExprClass, T, false, false), TheProtocol(protocol),
    AtLoc(at), RParenLoc(rp) {}
  explicit ObjCProtocolExpr(EmptyShell Empty)
    : Expr(ObjCProtocolExprClass, Empty) {}

  ObjCProtocolDecl *getProtocol() const { return TheProtocol; }
  void setProtocol(ObjCProtocolDecl *P) { TheProtocol = P; }

  SourceLocation getAtLoc() const { return AtLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setAtLoc(SourceLocation L) { AtLoc = L; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(AtLoc, RParenLoc);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCProtocolExprClass;
  }
  static bool classof(const ObjCProtocolExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ObjCIvarRefExpr - A reference to an ObjC instance variable.
class ObjCIvarRefExpr : public Expr {
  class ObjCIvarDecl *D;
  SourceLocation Loc;
  Stmt *Base;
  bool IsArrow:1;      // True if this is "X->F", false if this is "X.F".
  bool IsFreeIvar:1;   // True if ivar reference has no base (self assumed).

public:
  ObjCIvarRefExpr(ObjCIvarDecl *d,
                  QualType t, SourceLocation l, Expr *base,
                  bool arrow = false, bool freeIvar = false) :
    Expr(ObjCIvarRefExprClass, t, /*TypeDependent=*/false,
         base->isValueDependent()), D(d),
         Loc(l), Base(base), IsArrow(arrow),
         IsFreeIvar(freeIvar) {}

  explicit ObjCIvarRefExpr(EmptyShell Empty)
    : Expr(ObjCIvarRefExprClass, Empty) {}

  ObjCIvarDecl *getDecl() { return D; }
  const ObjCIvarDecl *getDecl() const { return D; }
  void setDecl(ObjCIvarDecl *d) { D = d; }

  const Expr *getBase() const { return cast<Expr>(Base); }
  Expr *getBase() { return cast<Expr>(Base); }
  void setBase(Expr * base) { Base = base; }

  bool isArrow() const { return IsArrow; }
  bool isFreeIvar() const { return IsFreeIvar; }
  void setIsArrow(bool A) { IsArrow = A; }
  void setIsFreeIvar(bool A) { IsFreeIvar = A; }

  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }

  virtual SourceRange getSourceRange() const {
    return isFreeIvar() ? SourceRange(Loc)
    : SourceRange(getBase()->getLocStart(), Loc);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCIvarRefExprClass;
  }
  static bool classof(const ObjCIvarRefExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ObjCPropertyRefExpr - A dot-syntax expression to access an ObjC
/// property.
///
class ObjCPropertyRefExpr : public Expr {
private:
  ObjCPropertyDecl *AsProperty;
  SourceLocation IdLoc;
  
  /// \brief When the receiver in property access is 'super', this is
  /// the location of the 'super' keyword.
  SourceLocation SuperLoc;
  
  /// \brief When the receiver in property access is 'super', this is
  /// the type associated with 'super' keyword. A null type indicates
  /// that this is not a 'super' receiver.
  llvm::PointerUnion<Stmt*, Type*> BaseExprOrSuperType;
  
public:
  ObjCPropertyRefExpr(ObjCPropertyDecl *PD, QualType t,
                      SourceLocation l, Expr *base)
    : Expr(ObjCPropertyRefExprClass, t, /*TypeDependent=*/false, 
           base->isValueDependent()), 
      AsProperty(PD), IdLoc(l), BaseExprOrSuperType(base) {
  }
  
  ObjCPropertyRefExpr(ObjCPropertyDecl *PD, QualType t,
                      SourceLocation l, SourceLocation sl, QualType st)
  : Expr(ObjCPropertyRefExprClass, t, /*TypeDependent=*/false, false), 
    AsProperty(PD), IdLoc(l), SuperLoc(sl), 
    BaseExprOrSuperType(st.getTypePtr()) {
  }

  explicit ObjCPropertyRefExpr(EmptyShell Empty)
    : Expr(ObjCPropertyRefExprClass, Empty) {}

  ObjCPropertyDecl *getProperty() const { return AsProperty; }

  const Expr *getBase() const { 
    return cast<Expr>(BaseExprOrSuperType.get<Stmt*>()); 
  }
  Expr *getBase() { 
    return cast<Expr>(BaseExprOrSuperType.get<Stmt*>()); 
  }

  SourceLocation getLocation() const { return IdLoc; }
  
  SourceLocation getSuperLocation() const { return SuperLoc; }
  QualType getSuperType() const { 
    Type *t = BaseExprOrSuperType.get<Type*>();
    return QualType(t, 0); 
  }
  bool isSuperReceiver() const { return BaseExprOrSuperType.is<Type*>(); }

  virtual SourceRange getSourceRange() const {
    return SourceRange(
                  (BaseExprOrSuperType.is<Stmt*>() ? getBase()->getLocStart() 
                                                   : getSuperLocation()), 
                  IdLoc);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCPropertyRefExprClass;
  }
  static bool classof(const ObjCPropertyRefExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
private:
  friend class ASTStmtReader;
  void setProperty(ObjCPropertyDecl *D) { AsProperty = D; }
  void setBase(Expr *base) { BaseExprOrSuperType = base; }
  void setLocation(SourceLocation L) { IdLoc = L; }
  void setSuperLocation(SourceLocation Loc) { SuperLoc = Loc; }
  void setSuperType(QualType T) { BaseExprOrSuperType = T.getTypePtr(); }
};

/// ObjCImplicitSetterGetterRefExpr - A dot-syntax expression to access two
/// methods; one to set a value to an 'ivar' (Setter) and the other to access
/// an 'ivar' (Setter).
/// An example for use of this AST is:
/// @code
///  @interface Test { }
///  - (Test *)crash;
///  - (void)setCrash: (Test*)value;
/// @end
/// void  foo(Test *p1, Test *p2)
/// {
///    p2.crash  = p1.crash; // Uses ObjCImplicitSetterGetterRefExpr AST
/// }
/// @endcode
class ObjCImplicitSetterGetterRefExpr : public Expr {
  /// Setter - Setter method user declared for setting its 'ivar' to a value
  ObjCMethodDecl *Setter;
  /// Getter - Getter method user declared for accessing 'ivar' it controls.
  ObjCMethodDecl *Getter;
  /// Location of the member in the dot syntax notation. This is location
  /// of the getter method.
  SourceLocation MemberLoc;
  // FIXME: Swizzle these into a single pointer.
  Stmt *Base;
  ObjCInterfaceDecl *InterfaceDecl;
  /// \brief Location of the receiver class in the dot syntax notation
  /// used to call a class method setter/getter.
  SourceLocation ClassLoc;

  /// \brief When the receiver in dot-syntax expression is 'super',
  /// this is the location of the 'super' keyword.
  SourceLocation SuperLoc;
  
  /// \brief When the receiver in dot-syntax expression is 'super', this is
  /// the type associated with 'super' keyword.
  QualType SuperTy;
  
public:
  ObjCImplicitSetterGetterRefExpr(ObjCMethodDecl *getter,
                 QualType t,
                 ObjCMethodDecl *setter,
                 SourceLocation l, Expr *base)
    : Expr(ObjCImplicitSetterGetterRefExprClass, t, /*TypeDependent=*/false, 
           base->isValueDependent()),
      Setter(setter), Getter(getter), MemberLoc(l), Base(base),
      InterfaceDecl(0), ClassLoc(SourceLocation()) {}
  
  ObjCImplicitSetterGetterRefExpr(ObjCMethodDecl *getter,
                                  QualType t,
                                  ObjCMethodDecl *setter,
                                  SourceLocation l,
                                  SourceLocation sl, 
                                  QualType st)
  : Expr(ObjCImplicitSetterGetterRefExprClass, t, /*TypeDependent=*/false, 
         false),
  Setter(setter), Getter(getter), MemberLoc(l),
  Base(0), InterfaceDecl(0), ClassLoc(SourceLocation()), 
  SuperLoc(sl), SuperTy(st) {
  }
  
  ObjCImplicitSetterGetterRefExpr(ObjCMethodDecl *getter,
                 QualType t,
                 ObjCMethodDecl *setter,
                 SourceLocation l, ObjCInterfaceDecl *C, SourceLocation CL)
    : Expr(ObjCImplicitSetterGetterRefExprClass, t, false, false),
      Setter(setter), Getter(getter), MemberLoc(l), Base(0), InterfaceDecl(C),
      ClassLoc(CL) {}
  explicit ObjCImplicitSetterGetterRefExpr(EmptyShell Empty)
           : Expr(ObjCImplicitSetterGetterRefExprClass, Empty){}

  ObjCMethodDecl *getGetterMethod() const { return Getter; }
  ObjCMethodDecl *getSetterMethod() const { return Setter; }
  ObjCInterfaceDecl *getInterfaceDecl() const { return InterfaceDecl; }
  void setGetterMethod(ObjCMethodDecl *D) { Getter = D; }
  void setSetterMethod(ObjCMethodDecl *D) { Setter = D; }
  void setInterfaceDecl(ObjCInterfaceDecl *D) { InterfaceDecl = D; }

  virtual SourceRange getSourceRange() const {
    if (isSuperReceiver())
      return SourceRange(getSuperLocation(), MemberLoc);
    if (Base)
      return SourceRange(getBase()->getLocStart(), MemberLoc);
    return SourceRange(ClassLoc, MemberLoc);
  }
  const Expr *getBase() const { return cast_or_null<Expr>(Base); }
  Expr *getBase() { return cast_or_null<Expr>(Base); }
  void setBase(Expr *base) { Base = base; }

  SourceLocation getLocation() const { return MemberLoc; }
  void setLocation(SourceLocation L) { MemberLoc = L; }
  SourceLocation getClassLoc() const { return ClassLoc; }
  void setClassLoc(SourceLocation L) { ClassLoc = L; }
  
  SourceLocation getSuperLocation() const { return SuperLoc; }
  QualType getSuperType() const { return SuperTy; }
  /// \brief When the receiver in dot-syntax expression is 'super', this
  /// method returns true if both Base expression and Interface are null.
  bool isSuperReceiver() const { return InterfaceDecl == 0 && Base == 0; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCImplicitSetterGetterRefExprClass;
  }
  static bool classof(const ObjCImplicitSetterGetterRefExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
private:
  friend class ASTStmtReader;
  void setSuperLocation(SourceLocation Loc) { SuperLoc = Loc; }
  void setSuperType(QualType T) { SuperTy = T; }
};

/// \brief An expression that sends a message to the given Objective-C
/// object or class.
///
/// The following contains two message send expressions:
///
/// \code
///   [[NSString alloc] initWithString:@"Hello"]
/// \endcode
///
/// The innermost message send invokes the "alloc" class method on the
/// NSString class, while the outermost message send invokes the
/// "initWithString" instance method on the object returned from
/// NSString's "alloc". In all, an Objective-C message send can take
/// on four different (although related) forms:
///
///   1. Send to an object instance.
///   2. Send to a class.
///   3. Send to the superclass instance of the current class.
///   4. Send to the superclass of the current class.
///
/// All four kinds of message sends are modeled by the ObjCMessageExpr
/// class, and can be distinguished via \c getReceiverKind(). Example:
///
class ObjCMessageExpr : public Expr {
  /// \brief The number of arguments in the message send, not
  /// including the receiver.
  unsigned NumArgs : 16;

  /// \brief The kind of message send this is, which is one of the
  /// ReceiverKind values.
  ///
  /// We pad this out to a byte to avoid excessive masking and shifting.
  unsigned Kind : 8;

  /// \brief Whether we have an actual method prototype in \c
  /// SelectorOrMethod.
  ///
  /// When non-zero, we have a method declaration; otherwise, we just
  /// have a selector.
  unsigned HasMethod : 8;

  /// \brief When the message expression is a send to 'super', this is
  /// the location of the 'super' keyword.
  SourceLocation SuperLoc;

  /// \brief Stores either the selector that this message is sending
  /// to (when \c HasMethod is zero) or an \c ObjCMethodDecl pointer
  /// referring to the method that we type-checked against.
  uintptr_t SelectorOrMethod;

  /// \brief The source locations of the open and close square
  /// brackets ('[' and ']', respectively).
  SourceLocation LBracLoc, RBracLoc;

  ObjCMessageExpr(EmptyShell Empty, unsigned NumArgs)
    : Expr(ObjCMessageExprClass, Empty), NumArgs(NumArgs), Kind(0), 
      HasMethod(0), SelectorOrMethod(0) { }

  ObjCMessageExpr(QualType T,
                  SourceLocation LBracLoc,
                  SourceLocation SuperLoc,
                  bool IsInstanceSuper,
                  QualType SuperType,
                  Selector Sel, 
                  ObjCMethodDecl *Method,
                  Expr **Args, unsigned NumArgs,
                  SourceLocation RBracLoc);
  ObjCMessageExpr(QualType T,
                  SourceLocation LBracLoc,
                  TypeSourceInfo *Receiver,
                  Selector Sel, 
                  ObjCMethodDecl *Method,
                  Expr **Args, unsigned NumArgs,
                  SourceLocation RBracLoc);
  ObjCMessageExpr(QualType T,
                  SourceLocation LBracLoc,
                  Expr *Receiver,
                  Selector Sel, 
                  ObjCMethodDecl *Method,
                  Expr **Args, unsigned NumArgs,
                  SourceLocation RBracLoc);

  /// \brief Retrieve the pointer value of the message receiver.
  void *getReceiverPointer() const {
    return *const_cast<void **>(
                             reinterpret_cast<const void * const*>(this + 1));
  }

  /// \brief Set the pointer value of the message receiver.
  void setReceiverPointer(void *Value) {
    *reinterpret_cast<void **>(this + 1) = Value;
  }

public:
  /// \brief The kind of receiver this message is sending to.
  enum ReceiverKind {
    /// \brief The receiver is a class.
    Class = 0,
    /// \brief The receiver is an object instance.
    Instance,
    /// \brief The receiver is a superclass.
    SuperClass,
    /// \brief The receiver is the instance of the superclass object.
    SuperInstance
  };

  /// \brief Create a message send to super.
  ///
  /// \param Context The ASTContext in which this expression will be created.
  ///
  /// \param T The result type of this message.
  ///
  /// \param LBrac The location of the open square bracket '['.
  ///
  /// \param SuperLoc The location of the "super" keyword.
  ///
  /// \param IsInstanceSuper Whether this is an instance "super"
  /// message (otherwise, it's a class "super" message).
  ///
  /// \param Sel The selector used to determine which method gets called.
  ///
  /// \param Method The Objective-C method against which this message
  /// send was type-checked. May be NULL.
  ///
  /// \param Args The message send arguments.
  ///
  /// \param NumArgs The number of arguments.
  ///
  /// \param RBracLoc The location of the closing square bracket ']'.
  static ObjCMessageExpr *Create(ASTContext &Context, QualType T,
                                 SourceLocation LBracLoc,
                                 SourceLocation SuperLoc,
                                 bool IsInstanceSuper,
                                 QualType SuperType,
                                 Selector Sel, 
                                 ObjCMethodDecl *Method,
                                 Expr **Args, unsigned NumArgs,
                                 SourceLocation RBracLoc);

  /// \brief Create a class message send.
  ///
  /// \param Context The ASTContext in which this expression will be created.
  ///
  /// \param T The result type of this message.
  ///
  /// \param LBrac The location of the open square bracket '['.
  ///
  /// \param Receiver The type of the receiver, including
  /// source-location information.
  ///
  /// \param Sel The selector used to determine which method gets called.
  ///
  /// \param Method The Objective-C method against which this message
  /// send was type-checked. May be NULL.
  ///
  /// \param Args The message send arguments.
  ///
  /// \param NumArgs The number of arguments.
  ///
  /// \param RBracLoc The location of the closing square bracket ']'.
  static ObjCMessageExpr *Create(ASTContext &Context, QualType T,
                                 SourceLocation LBracLoc,
                                 TypeSourceInfo *Receiver,
                                 Selector Sel, 
                                 ObjCMethodDecl *Method,
                                 Expr **Args, unsigned NumArgs,
                                 SourceLocation RBracLoc);

  /// \brief Create an instance message send.
  ///
  /// \param Context The ASTContext in which this expression will be created.
  ///
  /// \param T The result type of this message.
  ///
  /// \param LBrac The location of the open square bracket '['.
  ///
  /// \param Receiver The expression used to produce the object that
  /// will receive this message.
  ///
  /// \param Sel The selector used to determine which method gets called.
  ///
  /// \param Method The Objective-C method against which this message
  /// send was type-checked. May be NULL.
  ///
  /// \param Args The message send arguments.
  ///
  /// \param NumArgs The number of arguments.
  ///
  /// \param RBracLoc The location of the closing square bracket ']'.
  static ObjCMessageExpr *Create(ASTContext &Context, QualType T,
                                 SourceLocation LBracLoc,
                                 Expr *Receiver,
                                 Selector Sel, 
                                 ObjCMethodDecl *Method,
                                 Expr **Args, unsigned NumArgs,
                                 SourceLocation RBracLoc);

  /// \brief Create an empty Objective-C message expression, to be
  /// filled in by subsequent calls.
  ///
  /// \param Context The context in which the message send will be created.
  ///
  /// \param NumArgs The number of message arguments, not including
  /// the receiver.
  static ObjCMessageExpr *CreateEmpty(ASTContext &Context, unsigned NumArgs);

  /// \brief Determine the kind of receiver that this message is being
  /// sent to.
  ReceiverKind getReceiverKind() const { return (ReceiverKind)Kind; }

  /// \brief Determine whether this is an instance message to either a
  /// computed object or to super.
  bool isInstanceMessage() const {
    return getReceiverKind() == Instance || getReceiverKind() == SuperInstance;
  }

  /// \brief Determine whether this is an class message to either a
  /// specified class or to super.
  bool isClassMessage() const {
    return getReceiverKind() == Class || getReceiverKind() == SuperClass;
  }

  /// \brief Returns the receiver of an instance message.
  ///
  /// \brief Returns the object expression for an instance message, or
  /// NULL for a message that is not an instance message.
  Expr *getInstanceReceiver() {
    if (getReceiverKind() == Instance)
      return static_cast<Expr *>(getReceiverPointer());

    return 0;
  }
  const Expr *getInstanceReceiver() const {
    return const_cast<ObjCMessageExpr*>(this)->getInstanceReceiver();
  }

  /// \brief Turn this message send into an instance message that
  /// computes the receiver object with the given expression.
  void setInstanceReceiver(Expr *rec) { 
    Kind = Instance;
    setReceiverPointer(rec);
  }
  
  /// \brief Returns the type of a class message send, or NULL if the
  /// message is not a class message.
  QualType getClassReceiver() const { 
    if (TypeSourceInfo *TSInfo = getClassReceiverTypeInfo())
      return TSInfo->getType();

    return QualType();
  }

  /// \brief Returns a type-source information of a class message
  /// send, or NULL if the message is not a class message.
  TypeSourceInfo *getClassReceiverTypeInfo() const {
    if (getReceiverKind() == Class)
      return reinterpret_cast<TypeSourceInfo *>(getReceiverPointer());
    return 0;
  }

  void setClassReceiver(TypeSourceInfo *TSInfo) {
    Kind = Class;
    setReceiverPointer(TSInfo);
  }

  /// \brief Retrieve the location of the 'super' keyword for a class
  /// or instance message to 'super', otherwise an invalid source location.
  SourceLocation getSuperLoc() const { 
    if (getReceiverKind() == SuperInstance || getReceiverKind() == SuperClass)
      return SuperLoc;

    return SourceLocation();
  }

  /// \brief Retrieve the Objective-C interface to which this message
  /// is being directed, if known.
  ///
  /// This routine cross-cuts all of the different kinds of message
  /// sends to determine what the underlying (statically known) type
  /// of the receiver will be; use \c getReceiverKind() to determine
  /// whether the message is a class or an instance method, whether it
  /// is a send to super or not, etc.
  ///
  /// \returns The Objective-C interface if known, otherwise NULL.
  ObjCInterfaceDecl *getReceiverInterface() const;

  /// \brief Retrieve the type referred to by 'super'. 
  ///
  /// The returned type will either be an ObjCInterfaceType (for an
  /// class message to super) or an ObjCObjectPointerType that refers
  /// to a class (for an instance message to super);
  QualType getSuperType() const {
    if (getReceiverKind() == SuperInstance || getReceiverKind() == SuperClass)
      return QualType::getFromOpaquePtr(getReceiverPointer());

    return QualType();
  }

  void setSuper(SourceLocation Loc, QualType T, bool IsInstanceSuper) {
    Kind = IsInstanceSuper? SuperInstance : SuperClass;
    SuperLoc = Loc;
    setReceiverPointer(T.getAsOpaquePtr());
  }

  Selector getSelector() const;

  void setSelector(Selector S) { 
    HasMethod = false;
    SelectorOrMethod = reinterpret_cast<uintptr_t>(S.getAsOpaquePtr());
  }

  const ObjCMethodDecl *getMethodDecl() const { 
    if (HasMethod)
      return reinterpret_cast<const ObjCMethodDecl *>(SelectorOrMethod);

    return 0;
  }

  ObjCMethodDecl *getMethodDecl() { 
    if (HasMethod)
      return reinterpret_cast<ObjCMethodDecl *>(SelectorOrMethod);

    return 0;
  }

  void setMethodDecl(ObjCMethodDecl *MD) { 
    HasMethod = true;
    SelectorOrMethod = reinterpret_cast<uintptr_t>(MD);
  }

  /// \brief Return the number of actual arguments in this message,
  /// not counting the receiver.
  unsigned getNumArgs() const { return NumArgs; }

  /// \brief Retrieve the arguments to this message, not including the
  /// receiver.
  Stmt **getArgs() {
    return reinterpret_cast<Stmt **>(this + 1) + 1;
  }
  const Stmt * const *getArgs() const {
    return reinterpret_cast<const Stmt * const *>(this + 1) + 1;
  }

  /// getArg - Return the specified argument.
  Expr *getArg(unsigned Arg) {
    assert(Arg < NumArgs && "Arg access out of range!");
    return cast<Expr>(getArgs()[Arg]);
  }
  const Expr *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return cast<Expr>(getArgs()[Arg]);
  }
  /// setArg - Set the specified argument.
  void setArg(unsigned Arg, Expr *ArgExpr) {
    assert(Arg < NumArgs && "Arg access out of range!");
    getArgs()[Arg] = ArgExpr;
  }

  SourceLocation getLeftLoc() const { return LBracLoc; }
  SourceLocation getRightLoc() const { return RBracLoc; }

  void setLeftLoc(SourceLocation L) { LBracLoc = L; }
  void setRightLoc(SourceLocation L) { RBracLoc = L; }

  void setSourceRange(SourceRange R) {
    LBracLoc = R.getBegin();
    RBracLoc = R.getEnd();
  }
  virtual SourceRange getSourceRange() const {
    return SourceRange(LBracLoc, RBracLoc);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCMessageExprClass;
  }
  static bool classof(const ObjCMessageExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();

  typedef ExprIterator arg_iterator;
  typedef ConstExprIterator const_arg_iterator;

  arg_iterator arg_begin() { return getArgs(); }
  arg_iterator arg_end()   { return getArgs() + NumArgs; }
  const_arg_iterator arg_begin() const { return getArgs(); }
  const_arg_iterator arg_end() const { return getArgs() + NumArgs; }
};

/// ObjCIsaExpr - Represent X->isa and X.isa when X is an ObjC 'id' type.
/// (similiar in spirit to MemberExpr).
class ObjCIsaExpr : public Expr {
  /// Base - the expression for the base object pointer.
  Stmt *Base;

  /// IsaMemberLoc - This is the location of the 'isa'.
  SourceLocation IsaMemberLoc;

  /// IsArrow - True if this is "X->F", false if this is "X.F".
  bool IsArrow;
public:
  ObjCIsaExpr(Expr *base, bool isarrow, SourceLocation l, QualType ty)
    : Expr(ObjCIsaExprClass, ty, /*TypeDependent=*/false, 
           base->isValueDependent()),
      Base(base), IsaMemberLoc(l), IsArrow(isarrow) {}

  /// \brief Build an empty expression.
  explicit ObjCIsaExpr(EmptyShell Empty) : Expr(ObjCIsaExprClass, Empty) { }

  void setBase(Expr *E) { Base = E; }
  Expr *getBase() const { return cast<Expr>(Base); }

  bool isArrow() const { return IsArrow; }
  void setArrow(bool A) { IsArrow = A; }

  /// getMemberLoc - Return the location of the "member", in X->F, it is the
  /// location of 'F'.
  SourceLocation getIsaMemberLoc() const { return IsaMemberLoc; }
  void setIsaMemberLoc(SourceLocation L) { IsaMemberLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(getBase()->getLocStart(), IsaMemberLoc);
  }

  virtual SourceLocation getExprLoc() const { return IsaMemberLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCIsaExprClass;
  }
  static bool classof(const ObjCIsaExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

}  // end namespace clang

#endif
