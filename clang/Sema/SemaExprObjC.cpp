//===--- SemaExprObjC.cpp - Semantic Analysis for ObjC Expressions --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for Objective-C expressions.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
using namespace clang;

Sema::ExprResult Sema::ParseObjCStringLiteral(SourceLocation *AtLocs, 
                                              ExprTy **Strings,
                                              unsigned NumStrings) {
  SourceLocation AtLoc = AtLocs[0];
  StringLiteral* S = static_cast<StringLiteral *>(Strings[0]);
  if (NumStrings > 1) {
    // Concatenate objc strings.
    StringLiteral* ES = static_cast<StringLiteral *>(Strings[NumStrings-1]);
    SourceLocation EndLoc = ES->getSourceRange().getEnd();
    unsigned Length = 0;
    for (unsigned i = 0; i < NumStrings; i++)
      Length += static_cast<StringLiteral *>(Strings[i])->getByteLength();
    char *strBuf = new char [Length];
    char *p = strBuf;
    bool isWide = false;
    for (unsigned i = 0; i < NumStrings; i++) {
      S = static_cast<StringLiteral *>(Strings[i]);
      if (S->isWide())
        isWide = true;
      memcpy(p, S->getStrData(), S->getByteLength());
      p += S->getByteLength();
      delete S;
    }
    S = new StringLiteral(strBuf, Length,
                          isWide, Context.getPointerType(Context.CharTy),
                          AtLoc, EndLoc);
  }
  
  if (CheckBuiltinCFStringArgument(S))
    return true;
  
  if (Context.getObjCConstantStringInterface().isNull()) {
    // Initialize the constant string interface lazily. This assumes
    // the NSConstantString interface is seen in this translation unit.
    IdentifierInfo *NSIdent = &Context.Idents.get("NSConstantString");
    ScopedDecl *IFace = LookupScopedDecl(NSIdent, Decl::IDNS_Ordinary, 
                                         SourceLocation(), TUScope);
    ObjCInterfaceDecl *strIFace = dyn_cast_or_null<ObjCInterfaceDecl>(IFace);
    if (!strIFace)
      return Diag(S->getLocStart(), diag::err_undef_interface,
                  NSIdent->getName());
    Context.setObjCConstantStringInterface(strIFace);
  }
  QualType t = Context.getObjCConstantStringInterface();
  t = Context.getPointerType(t);
  return new ObjCStringLiteral(S, t, AtLoc);
}

Sema::ExprResult Sema::ParseObjCEncodeExpression(SourceLocation AtLoc,
                                                 SourceLocation EncodeLoc,
                                                 SourceLocation LParenLoc,
                                                 TypeTy *Ty,
                                                 SourceLocation RParenLoc) {
  QualType EncodedType = QualType::getFromOpaquePtr(Ty);

  QualType t = Context.getPointerType(Context.CharTy);
  return new ObjCEncodeExpr(t, EncodedType, AtLoc, RParenLoc);
}

Sema::ExprResult Sema::ParseObjCSelectorExpression(Selector Sel,
                                                   SourceLocation AtLoc,
                                                   SourceLocation SelLoc,
                                                   SourceLocation LParenLoc,
                                                   SourceLocation RParenLoc) {
  QualType t = Context.getObjCSelType();
  return new ObjCSelectorExpr(t, Sel, AtLoc, RParenLoc);
}

Sema::ExprResult Sema::ParseObjCProtocolExpression(IdentifierInfo *ProtocolId,
                                                   SourceLocation AtLoc,
                                                   SourceLocation ProtoLoc,
                                                   SourceLocation LParenLoc,
                                                   SourceLocation RParenLoc) {
  ObjCProtocolDecl* PDecl = ObjCProtocols[ProtocolId];
  if (!PDecl) {
    Diag(ProtoLoc, diag::err_undeclared_protocol, ProtocolId->getName());
    return true;
  }
  
  QualType t = Context.getObjCProtoType();
  if (t.isNull())
    return true;
  t = Context.getPointerType(t);
  return new ObjCProtocolExpr(t, PDecl, AtLoc, RParenLoc);
}

bool Sema::CheckMessageArgumentTypes(Expr **Args, unsigned NumArgs,
                                     ObjCMethodDecl *Method) {
  bool anyIncompatibleArgs = false;
  
  for (unsigned i = 0; i < NumArgs; i++) {
    Expr *argExpr = Args[i];
    assert(argExpr && "CheckMessageArgumentTypes(): missing expression");
    
    QualType lhsType = Method->getParamDecl(i)->getType();
    QualType rhsType = argExpr->getType();

    // If necessary, apply function/array conversion. C99 6.7.5.3p[7,8]. 
    if (const ArrayType *ary = lhsType->getAsArrayType())
      lhsType = Context.getPointerType(ary->getElementType());
    else if (lhsType->isFunctionType())
      lhsType = Context.getPointerType(lhsType);

    AssignConvertType Result = CheckSingleAssignmentConstraints(lhsType,
                                                                argExpr);
    if (Args[i] != argExpr) // The expression was converted.
      Args[i] = argExpr; // Make sure we store the converted expression.
    
    anyIncompatibleArgs |= 
      DiagnoseAssignmentResult(Result, argExpr->getLocStart(), lhsType, rhsType,
                               argExpr, "sending");
  }
  return anyIncompatibleArgs;
}

// ActOnClassMessage - used for both unary and keyword messages.
// ArgExprs is optional - if it is present, the number of expressions
// is obtained from Sel.getNumArgs().
Sema::ExprResult Sema::ActOnClassMessage(
  Scope *S,
  IdentifierInfo *receiverName, Selector Sel,
  SourceLocation lbrac, SourceLocation rbrac, ExprTy **Args, unsigned NumArgs)
{
  assert(receiverName && "missing receiver class name");

  Expr **ArgExprs = reinterpret_cast<Expr **>(Args);
  ObjCInterfaceDecl* ClassDecl = 0;
  if (!strcmp(receiverName->getName(), "super") && CurMethodDecl) {
    ClassDecl = CurMethodDecl->getClassInterface()->getSuperClass();
    if (ClassDecl && CurMethodDecl->isInstance()) {
      // Synthesize a cast to the super class. This hack allows us to loosely
      // represent super without creating a special expression node.
      IdentifierInfo &II = Context.Idents.get("self");
      ExprResult ReceiverExpr = ActOnIdentifierExpr(S, lbrac, II, false);
      QualType superTy = Context.getObjCInterfaceType(ClassDecl);
      superTy = Context.getPointerType(superTy);
      ReceiverExpr = ActOnCastExpr(SourceLocation(), superTy.getAsOpaquePtr(),
                                   SourceLocation(), ReceiverExpr.Val);
      // We are really in an instance method, redirect.
      return ActOnInstanceMessage(ReceiverExpr.Val, Sel, lbrac, rbrac,
                                  Args, NumArgs);
    }
    // We are sending a message to 'super' within a class method. Do nothing,
    // the receiver will pass through as 'super' (how convenient:-).
  } else
    ClassDecl = getObjCInterfaceDecl(receiverName);
  
  // FIXME: can ClassDecl ever be null?
  ObjCMethodDecl *Method = ClassDecl->lookupClassMethod(Sel);
  QualType returnType;
  
  // Before we give up, check if the selector is an instance method.
  if (!Method)
    Method = ClassDecl->lookupInstanceMethod(Sel);
  if (!Method) {
    Diag(lbrac, diag::warn_method_not_found, std::string("+"), Sel.getName(),
         SourceRange(lbrac, rbrac));
    returnType = Context.getObjCIdType();
  } else {
    returnType = Method->getResultType();
    if (Sel.getNumArgs()) {
      if (CheckMessageArgumentTypes(ArgExprs, Sel.getNumArgs(), Method))
        return true;
    }
  }
  return new ObjCMessageExpr(receiverName, Sel, returnType, Method,
                             lbrac, rbrac, ArgExprs, NumArgs);
}

// ActOnInstanceMessage - used for both unary and keyword messages.
// ArgExprs is optional - if it is present, the number of expressions
// is obtained from Sel.getNumArgs().
Sema::ExprResult Sema::ActOnInstanceMessage(
  ExprTy *receiver, Selector Sel,
  SourceLocation lbrac, SourceLocation rbrac, ExprTy **Args, unsigned NumArgs) 
{
  assert(receiver && "missing receiver expression");
  
  Expr **ArgExprs = reinterpret_cast<Expr **>(Args);
  Expr *RExpr = static_cast<Expr *>(receiver);
  QualType receiverType = RExpr->getType();
  QualType returnType;
  ObjCMethodDecl *Method = 0;
  
  if (receiverType == Context.getObjCIdType() ||
      receiverType == Context.getObjCClassType()) {
    Method = InstanceMethodPool[Sel].Method;
	if (!Method)
	  Method = FactoryMethodPool[Sel].Method;
    if (!Method) {
      Diag(lbrac, diag::warn_method_not_found, std::string("-"), Sel.getName(),
           SourceRange(lbrac, rbrac));
      returnType = Context.getObjCIdType();
    } else {
      returnType = Method->getResultType();
      if (Sel.getNumArgs())
        if (CheckMessageArgumentTypes(ArgExprs, Sel.getNumArgs(), Method))
          return true;
    }
  } else {
    bool receiverIsQualId = isa<ObjCQualifiedIdType>(receiverType);
    // FIXME (snaroff): checking in this code from Patrick. Needs to be
    // revisited. how do we get the ClassDecl from the receiver expression?
    if (!receiverIsQualId)
      while (receiverType->isPointerType()) {
        PointerType *pointerType =
          static_cast<PointerType*>(receiverType.getTypePtr());
        receiverType = pointerType->getPointeeType();
      }
    ObjCInterfaceDecl* ClassDecl = 0;
    if (ObjCQualifiedInterfaceType *QIT = 
        dyn_cast<ObjCQualifiedInterfaceType>(receiverType)) {
      ClassDecl = QIT->getDecl();
      Method = ClassDecl->lookupInstanceMethod(Sel);
      if (!Method) {
        // search protocols
        for (unsigned i = 0; i < QIT->getNumProtocols(); i++) {
          ObjCProtocolDecl *PDecl = QIT->getProtocols(i);
          if (PDecl && (Method = PDecl->lookupInstanceMethod(Sel)))
            break;
        }
      }
      if (!Method)
        Diag(lbrac, diag::warn_method_not_found_in_protocol, 
             std::string("-"), Sel.getName(),
             SourceRange(lbrac, rbrac));
    }
    else if (ObjCQualifiedIdType *QIT = 
             dyn_cast<ObjCQualifiedIdType>(receiverType)) {
      // search protocols
      for (unsigned i = 0; i < QIT->getNumProtocols(); i++) {
        ObjCProtocolDecl *PDecl = QIT->getProtocols(i);
        if (PDecl && (Method = PDecl->lookupInstanceMethod(Sel)))
          break;
      }
      if (!Method)
        Diag(lbrac, diag::warn_method_not_found_in_protocol, 
             std::string("-"), Sel.getName(),
             SourceRange(lbrac, rbrac));
    }
    else {
      assert(ObjCInterfaceType::classof(receiverType.getTypePtr()) &&
             "bad receiver type");
      ClassDecl = static_cast<ObjCInterfaceType*>(
                    receiverType.getTypePtr())->getDecl();
      // FIXME: consider using InstanceMethodPool, since it will be faster
      // than the following method (which can do *many* linear searches). The
      // idea is to add class info to InstanceMethodPool...
      Method = ClassDecl->lookupInstanceMethod(Sel);
    }
    if (!Method) {
      // If we have an implementation in scope, check "private" methods.
      if (ClassDecl)
        if (ObjCImplementationDecl *ImpDecl = 
            ObjCImplementations[ClassDecl->getIdentifier()])
          Method = ImpDecl->getInstanceMethod(Sel);
	  // If we still haven't found a method, look in the global pool. This
	  // behavior isn't very desirable, however we need it for GCC
          // compatibility.
	  if (!Method)
	    Method = InstanceMethodPool[Sel].Method;
    }
    if (!Method) {
      Diag(lbrac, diag::warn_method_not_found, std::string("-"), Sel.getName(),
           SourceRange(lbrac, rbrac));
      returnType = Context.getObjCIdType();
    } else {
      returnType = Method->getResultType();
      if (Sel.getNumArgs())
        if (CheckMessageArgumentTypes(ArgExprs, Sel.getNumArgs(), Method))
          return true;
    }
  }
  return new ObjCMessageExpr(RExpr, Sel, returnType, Method, lbrac, rbrac, 
                             ArgExprs, NumArgs);
}
