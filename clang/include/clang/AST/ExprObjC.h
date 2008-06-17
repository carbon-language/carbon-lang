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

namespace clang {
  class IdentifierInfo;
  class Selector;
  class ASTContext;
  
/// ObjCStringLiteral, used for Objective-C string literals
/// i.e. @"foo".
class ObjCStringLiteral : public Expr {
  StringLiteral *String;
  SourceLocation AtLoc;
public:
  ObjCStringLiteral(StringLiteral *SL, QualType T, SourceLocation L)
    : Expr(ObjCStringLiteralClass, T), String(SL), AtLoc(L) {}
  
  StringLiteral* getString() { return String; }

  const StringLiteral* getString() const { return String; }

  SourceLocation getAtLoc() const { return AtLoc; }

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
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCStringLiteral* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};
  
/// ObjCEncodeExpr, used for @encode in Objective-C.
class ObjCEncodeExpr : public Expr {
  QualType EncType;
  SourceLocation AtLoc, RParenLoc;
public:
  ObjCEncodeExpr(QualType T, QualType ET, 
                 SourceLocation at, SourceLocation rp)
    : Expr(ObjCEncodeExprClass, T), EncType(ET), AtLoc(at), RParenLoc(rp) {}
  
  SourceLocation getAtLoc() const { return AtLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(AtLoc, RParenLoc);
  }

  QualType getEncodedType() const { return EncType; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCEncodeExprClass;
  }
  static bool classof(const ObjCEncodeExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCEncodeExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// ObjCSelectorExpr used for @selector in Objective-C.
class ObjCSelectorExpr : public Expr {
  Selector SelName;
  SourceLocation AtLoc, RParenLoc;
public:
  ObjCSelectorExpr(QualType T, Selector selInfo,
                   SourceLocation at, SourceLocation rp)
  : Expr(ObjCSelectorExprClass, T), SelName(selInfo), 
  AtLoc(at), RParenLoc(rp) {}
  
  Selector getSelector() const { return SelName; }
  
  SourceLocation getAtLoc() const { return AtLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }

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

  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCSelectorExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};
  
/// ObjCProtocolExpr used for protocol in Objective-C.
class ObjCProtocolExpr : public Expr {    
  ObjCProtocolDecl *Protocol;    
  SourceLocation AtLoc, RParenLoc;
public:
  ObjCProtocolExpr(QualType T, ObjCProtocolDecl *protocol,
                   SourceLocation at, SourceLocation rp)
  : Expr(ObjCProtocolExprClass, T), Protocol(protocol), 
  AtLoc(at), RParenLoc(rp) {}
    
  ObjCProtocolDecl *getProtocol() const { return Protocol; }
    
  SourceLocation getAtLoc() const { return AtLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }

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
  ObjCIvarRefExpr(ObjCIvarDecl *d, QualType t, SourceLocation l, Expr *base=0, 
                  bool arrow = false, bool freeIvar = false) : 
    Expr(ObjCIvarRefExprClass, t), D(d), Loc(l), Base(base), IsArrow(arrow),
    IsFreeIvar(freeIvar) {}
  
  ObjCIvarDecl *getDecl() { return D; }
  const ObjCIvarDecl *getDecl() const { return D; }
  virtual SourceRange getSourceRange() const { 
    return isFreeIvar() ? SourceRange(Loc)
                        : SourceRange(getBase()->getLocStart(), Loc); 
  }
  const Expr *getBase() const { return cast<Expr>(Base); }
  Expr *getBase() { return cast<Expr>(Base); }
  void setBase(Expr * base) { Base = base; }
  bool isArrow() const { return IsArrow; }
  bool isFreeIvar() const { return IsFreeIvar; }
  
  SourceLocation getLocation() const { return Loc; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ObjCIvarRefExprClass; 
  }
  static bool classof(const ObjCIvarRefExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCIvarRefExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// ObjCPropertyRefExpr - A reference to an ObjC property.
class ObjCPropertyRefExpr : public Expr {
  class Decl *D; // an ObjCMethodDecl or ObjCPropertyDecl
  SourceLocation Loc;
  Stmt *Base;
  
public:
  ObjCPropertyRefExpr(Decl *d, QualType t, SourceLocation l, Expr *base) : 
    Expr(ObjCPropertyRefExprClass, t), D(d), Loc(l), Base(base) {}
  
  Decl *getDecl() { return D; }
  const Decl *getDecl() const { return D; }
  
  virtual SourceRange getSourceRange() const { 
    return SourceRange(getBase()->getLocStart(), Loc); 
  }
  const Expr *getBase() const { return cast<Expr>(Base); }
  Expr *getBase() { return cast<Expr>(Base); }
  void setBase(Expr * base) { Base = base; }
  
  SourceLocation getLocation() const { return Loc; }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ObjCPropertyRefExprClass; 
  }
  static bool classof(const ObjCPropertyRefExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCPropertyRefExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

class ObjCMessageExpr : public Expr {
  enum { RECEIVER=0, ARGS_START=1 };

  Stmt **SubExprs;
  
  unsigned NumArgs;
  
  // A unigue name for this message.
  Selector SelName;
  
  // A method prototype for this message (optional). 
  // FIXME: Since method decls contain the selector, and most messages have a
  // prototype, consider devising a scheme for unifying SelName/MethodProto.
  ObjCMethodDecl *MethodProto;

  SourceLocation LBracloc, RBracloc;
  
  // constructor used during deserialization
  ObjCMessageExpr(Selector selInfo, QualType retType,
                  SourceLocation LBrac, SourceLocation RBrac,
                  Expr **ArgExprs, unsigned nargs)
  : Expr(ObjCMessageExprClass, retType), NumArgs(nargs), SelName(selInfo),
    MethodProto(NULL), LBracloc(LBrac), RBracloc(RBrac) {}
  
public:
  // constructor for class messages. 
  // FIXME: clsName should be typed to ObjCInterfaceType
  ObjCMessageExpr(IdentifierInfo *clsName, Selector selInfo,
                  QualType retType, ObjCMethodDecl *methDecl,
                  SourceLocation LBrac, SourceLocation RBrac,
                  Expr **ArgExprs, unsigned NumArgs);
  // constructor for instance messages.
  ObjCMessageExpr(Expr *receiver, Selector selInfo,
                  QualType retType, ObjCMethodDecl *methDecl,
                  SourceLocation LBrac, SourceLocation RBrac,
                  Expr **ArgExprs, unsigned NumArgs);
  
  ~ObjCMessageExpr() {
    delete [] SubExprs;
  }
  
  /// getReceiver - Returns the receiver of the message expression.
  ///  This can be NULL if the message is for instance methods.  For
  ///  instance methods, use getClassName.
  Expr *getReceiver() { 
    uintptr_t x = (uintptr_t) SubExprs[RECEIVER];
    return x & 0x1 ? NULL : (Expr*) x;
  }  
  const Expr *getReceiver() const {
    return const_cast<ObjCMessageExpr*>(this)->getReceiver();
  }
  
  Selector getSelector() const { return SelName; }

  const ObjCMethodDecl *getMethodDecl() const { return MethodProto; }
  ObjCMethodDecl *getMethodDecl() { return MethodProto; }
  
  /// getClassName - For instance methods, this returns the invoked class,
  ///  and returns NULL otherwise.  For regular methods, use getReceiver.  
  IdentifierInfo *getClassName() {
    uintptr_t x = (uintptr_t) SubExprs[RECEIVER];
    return x & 0x1 ? (IdentifierInfo*) (x & ~0x1) : NULL;
  }  
  const IdentifierInfo *getClassName() const {
    return const_cast<ObjCMessageExpr*>(this)->getClassName();
  }
  
  /// getNumArgs - Return the number of actual arguments to this call.
  unsigned getNumArgs() const { return NumArgs; }

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
  
  // Serialization.
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCMessageExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// ObjCSuperRefExpr - A reference to super.
class ObjCSuperRefExpr : public Expr {
  SourceLocation Loc;
public:
  ObjCSuperRefExpr(QualType t, SourceLocation l) : 
    Expr(ObjCSuperRefExprClass, t), Loc(l) {}
  
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ObjCSuperRefExprClass; 
  }
  static bool classof(const ObjCSuperRefExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCSuperRefExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

}  // end namespace clang

#endif
