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
    : Expr(ObjCStringLiteralClass, T), String(SL), AtLoc(L) {}
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
  QualType EncType;
  SourceLocation AtLoc, RParenLoc;
public:
  ObjCEncodeExpr(QualType T, QualType ET, 
                 SourceLocation at, SourceLocation rp)
    : Expr(ObjCEncodeExprClass, T, ET->isDependentType(), 
           ET->isDependentType()), EncType(ET), AtLoc(at), RParenLoc(rp) {}
  
  explicit ObjCEncodeExpr(EmptyShell Empty) : Expr(ObjCEncodeExprClass, Empty){}

  
  SourceLocation getAtLoc() const { return AtLoc; }
  void setAtLoc(SourceLocation L) { AtLoc = L; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }
  
  QualType getEncodedType() const { return EncType; }
  void setEncodedType(QualType T) { EncType = T; }

  
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
  : Expr(ObjCSelectorExprClass, T), SelName(selInfo), AtLoc(at), RParenLoc(rp){}
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
  : Expr(ObjCProtocolExprClass, T), TheProtocol(protocol),
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
                  QualType t, SourceLocation l, Expr *base=0, 
                  bool arrow = false, bool freeIvar = false) : 
    Expr(ObjCIvarRefExprClass, t), D(d),
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
  Stmt *Base;
public:
  ObjCPropertyRefExpr(ObjCPropertyDecl *PD, QualType t, 
                      SourceLocation l, Expr *base)
    : Expr(ObjCPropertyRefExprClass, t), AsProperty(PD), IdLoc(l), Base(base) {
  }
  
  explicit ObjCPropertyRefExpr(EmptyShell Empty)
    : Expr(ObjCPropertyRefExprClass, Empty) {}

  ObjCPropertyDecl *getProperty() const { return AsProperty; }
  void setProperty(ObjCPropertyDecl *D) { AsProperty = D; }
  
  const Expr *getBase() const { return cast<Expr>(Base); }
  Expr *getBase() { return cast<Expr>(Base); }
  void setBase(Expr *base) { Base = base; }
  
  SourceLocation getLocation() const { return IdLoc; }
  void setLocation(SourceLocation L) { IdLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(getBase()->getLocStart(), IdLoc);
  }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ObjCPropertyRefExprClass; 
  }
  static bool classof(const ObjCPropertyRefExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ObjCImplctSetterGetterRefExpr - A dot-syntax expression to access two 
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
///    p2.crash  = p1.crash; // Uses ObjCImplctSetterGetterRefExpr AST
/// }
/// @endcode
class ObjCImplctSetterGetterRefExpr : public Expr {
  /// Setter - Setter method user declared for setting its 'ivar' to a value
  ObjCMethodDecl *Setter;
  /// Getter - Getter method user declared for accessing 'ivar' it controls.
  ObjCMethodDecl *Getter;
  SourceLocation Loc;
  // FIXME: Swizzle these into a single pointer.
  Stmt *Base;
  ObjCInterfaceDecl *InterfaceDecl;
  SourceLocation ClassLoc;
    
public:
  ObjCImplctSetterGetterRefExpr(ObjCMethodDecl *getter,
                 QualType t, 
                 ObjCMethodDecl *setter,
                 SourceLocation l, Expr *base)
    : Expr(ObjCImplctSetterGetterRefExprClass, t), Setter(setter),
      Getter(getter), Loc(l), Base(base), InterfaceDecl(0),
      ClassLoc(SourceLocation()) {
    }
  ObjCImplctSetterGetterRefExpr(ObjCMethodDecl *getter,
                 QualType t, 
                 ObjCMethodDecl *setter,
                 SourceLocation l, ObjCInterfaceDecl *C, SourceLocation CL)
    : Expr(ObjCImplctSetterGetterRefExprClass, t), Setter(setter),
      Getter(getter), Loc(l), Base(0), InterfaceDecl(C), ClassLoc(CL) {
    }
  explicit ObjCImplctSetterGetterRefExpr(EmptyShell Empty) 
           : Expr(ObjCImplctSetterGetterRefExprClass, Empty){}

  ObjCMethodDecl *getGetterMethod() const { return Getter; }
  ObjCMethodDecl *getSetterMethod() const { return Setter; }
  ObjCInterfaceDecl *getInterfaceDecl() const { return InterfaceDecl; }
  void setGetterMethod(ObjCMethodDecl *D) { Getter = D; }
  void setSetterMethod(ObjCMethodDecl *D) { Setter = D; }
  void setInterfaceDecl(ObjCInterfaceDecl *D) { InterfaceDecl = D; }
  
  virtual SourceRange getSourceRange() const {
    if (Base)
      return SourceRange(getBase()->getLocStart(), Loc);
    return SourceRange(ClassLoc, Loc);
  }
  const Expr *getBase() const { return cast_or_null<Expr>(Base); }
  Expr *getBase() { return cast_or_null<Expr>(Base); }
  void setBase(Expr *base) { Base = base; }
    
  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }
  SourceLocation getClassLoc() const { return ClassLoc; }
  void setClassLoc(SourceLocation L) { ClassLoc = L; }
    
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ObjCImplctSetterGetterRefExprClass; 
  }
  static bool classof(const ObjCImplctSetterGetterRefExpr *) { return true; }
    
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};
  
class ObjCMessageExpr : public Expr {
  // SubExprs - The receiver and arguments of the message expression.
  Stmt **SubExprs;
  
  // NumArgs - The number of arguments (not including the receiver) to the
  //  message expression.
  unsigned NumArgs;
  
  // A unigue name for this message.
  Selector SelName;
  
  // A method prototype for this message (optional). 
  // FIXME: Since method decls contain the selector, and most messages have a
  // prototype, consider devising a scheme for unifying SelName/MethodProto.
  ObjCMethodDecl *MethodProto;

  SourceLocation LBracloc, RBracloc;

  // Constants for indexing into SubExprs.
  enum { RECEIVER=0, ARGS_START=1 };

  // Bit-swizzling flags.
  enum { IsInstMeth=0, IsClsMethDeclUnknown, IsClsMethDeclKnown, Flags=0x3 };
  unsigned getFlag() const { return (uintptr_t) SubExprs[RECEIVER] & Flags; }
  
public:
  /// This constructor is used to represent class messages where the
  /// ObjCInterfaceDecl* of the receiver is not known.
  ObjCMessageExpr(IdentifierInfo *clsName, Selector selInfo,
                  QualType retType, ObjCMethodDecl *methDecl,
                  SourceLocation LBrac, SourceLocation RBrac,
                  Expr **ArgExprs, unsigned NumArgs);

  /// This constructor is used to represent class messages where the
  /// ObjCInterfaceDecl* of the receiver is known.
  // FIXME: clsName should be typed to ObjCInterfaceType
  ObjCMessageExpr(ObjCInterfaceDecl *cls, Selector selInfo,
                  QualType retType, ObjCMethodDecl *methDecl,
                  SourceLocation LBrac, SourceLocation RBrac,
                  Expr **ArgExprs, unsigned NumArgs);
  
  // constructor for instance messages.
  ObjCMessageExpr(Expr *receiver, Selector selInfo,
                  QualType retType, ObjCMethodDecl *methDecl,
                  SourceLocation LBrac, SourceLocation RBrac,
                  Expr **ArgExprs, unsigned NumArgs);
                  
  explicit ObjCMessageExpr(EmptyShell Empty)
    : Expr(ObjCMessageExprClass, Empty), SubExprs(0), NumArgs(0) {}
  
  ~ObjCMessageExpr() {
    delete [] SubExprs;
  }
  
  /// getReceiver - Returns the receiver of the message expression.
  ///  This can be NULL if the message is for class methods.  For
  ///  class methods, use getClassName.
  /// FIXME: need to handle/detect 'super' usage within a class method.
  Expr *getReceiver() { 
    uintptr_t x = (uintptr_t) SubExprs[RECEIVER];
    return (x & Flags) == IsInstMeth ? (Expr*) x : 0;
  }  
  const Expr *getReceiver() const {
    return const_cast<ObjCMessageExpr*>(this)->getReceiver();
  }
  // FIXME: need setters for different receiver types.
  void setReceiver(Expr *rec) { SubExprs[RECEIVER] = rec; }
  Selector getSelector() const { return SelName; }
  void setSelector(Selector S) { SelName = S; }
  
  const ObjCMethodDecl *getMethodDecl() const { return MethodProto; }
  ObjCMethodDecl *getMethodDecl() { return MethodProto; }
  void setMethodDecl(ObjCMethodDecl *MD) { MethodProto = MD; }
  
  typedef std::pair<ObjCInterfaceDecl*, IdentifierInfo*> ClassInfo;
  
  /// getClassInfo - For class methods, this returns both the ObjCInterfaceDecl*
  ///  and IdentifierInfo* of the invoked class.  Both can be NULL if this
  ///  is an instance message, and the ObjCInterfaceDecl* can be NULL if none
  ///  was available when this ObjCMessageExpr object was constructed.  
  ClassInfo getClassInfo() const; 
  void setClassInfo(const ClassInfo &C);
  
  /// getClassName - For class methods, this returns the invoked class,
  ///  and returns NULL otherwise.  For instance methods, use getReceiver.  
  IdentifierInfo *getClassName() const {
    return getClassInfo().second;
  }
  
  /// getNumArgs - Return the number of actual arguments to this call.
  unsigned getNumArgs() const { return NumArgs; }
  void setNumArgs(unsigned nArgs) { 
    NumArgs = nArgs; 
    // FIXME: should always allocate SubExprs via the ASTContext's
    // allocator.
    if (!SubExprs)
      SubExprs = new Stmt* [NumArgs + 1];
  }
  
  /// getArg - Return the specified argument.
  Expr *getArg(unsigned Arg) {
    assert(Arg < NumArgs && "Arg access out of range!");
    return cast<Expr>(SubExprs[Arg+ARGS_START]);
  }
  const Expr *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return cast<Expr>(SubExprs[Arg+ARGS_START]);
  }
  /// setArg - Set the specified argument.
  void setArg(unsigned Arg, Expr *ArgExpr) {
    assert(Arg < NumArgs && "Arg access out of range!");
    SubExprs[Arg+ARGS_START] = ArgExpr;
  }
  
  SourceLocation getLeftLoc() const { return LBracloc; }
  SourceLocation getRightLoc() const { return RBracloc; }

  void setLeftLoc(SourceLocation L) { LBracloc = L; }
  void setRightLoc(SourceLocation L) { RBracloc = L; }
  
  void setSourceRange(SourceRange R) {
    LBracloc = R.getBegin();
    RBracloc = R.getEnd();
  }
  virtual SourceRange getSourceRange() const {
    return SourceRange(LBracloc, RBracloc);
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
  
  arg_iterator arg_begin() { return &SubExprs[ARGS_START]; }
  arg_iterator arg_end()   { return &SubExprs[ARGS_START] + NumArgs; }
  const_arg_iterator arg_begin() const { return &SubExprs[ARGS_START]; }
  const_arg_iterator arg_end() const { return &SubExprs[ARGS_START] + NumArgs; }
};

/// ObjCSuperExpr - Represents the "super" expression in Objective-C,
/// which refers to the object on which the current method is executing.
class ObjCSuperExpr : public Expr {
  SourceLocation Loc;
public:
  ObjCSuperExpr(SourceLocation L, QualType Type) 
    : Expr(ObjCSuperExprClass, Type), Loc(L) { }
  explicit ObjCSuperExpr(EmptyShell Empty) : Expr(ObjCSuperExprClass, Empty) {}

  SourceLocation getLoc() const { return Loc; }
  void setLoc(SourceLocation L) { Loc = L; }
  
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ObjCSuperExprClass;
  }
  static bool classof(const ObjCSuperExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
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
    : Expr(ObjCIsaExprClass, ty),
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
