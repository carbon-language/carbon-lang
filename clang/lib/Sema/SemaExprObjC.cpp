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
#include "clang/AST/ExprObjC.h"
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
    Decl *IFace = LookupDecl(NSIdent, Decl::IDNS_Ordinary, TUScope);
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
    if (lhsType->isArrayType())
      lhsType = Context.getArrayDecayedType(lhsType);
    else if (lhsType->isFunctionType())
      lhsType = Context.getPointerType(lhsType);

    AssignConvertType Result = 
      CheckSingleAssignmentConstraints(lhsType, argExpr);
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
    if (!ClassDecl)
      return Diag(lbrac, diag::error_no_super_class,
                  CurMethodDecl->getClassInterface()->getName());
    if (CurMethodDecl->isInstance()) {
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
  QualType receiverType = RExpr->getType().getCanonicalType();
  QualType returnType;
  ObjCMethodDecl *Method = 0;
  
  // FIXME: This code is not stripping off type qualifiers! Should it?
  if (receiverType == Context.getObjCIdType().getCanonicalType() ||
      receiverType == Context.getObjCClassType().getCanonicalType()) {
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
      while (const PointerType *PTy = receiverType->getAsPointerType())
        receiverType = PTy->getPointeeType();
    
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
      ObjCInterfaceType *OCIReceiver =dyn_cast<ObjCInterfaceType>(receiverType);
      if (OCIReceiver == 0) {
        Diag(lbrac, diag::error_bad_receiver_type,
             RExpr->getType().getAsString());
        return true;
      }
      ClassDecl = OCIReceiver->getDecl();
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

//===----------------------------------------------------------------------===//
// ObjCQualifiedIdTypesAreCompatible - Compatibility testing for qualified id's.
//===----------------------------------------------------------------------===//

/// ProtocolCompatibleWithProtocol - return 'true' if 'lProto' is in the
/// inheritance hierarchy of 'rProto'.
static bool ProtocolCompatibleWithProtocol(ObjCProtocolDecl *lProto,
                                           ObjCProtocolDecl *rProto) {
  if (lProto == rProto)
    return true;
  ObjCProtocolDecl** RefPDecl = rProto->getReferencedProtocols();
  for (unsigned i = 0; i < rProto->getNumReferencedProtocols(); i++)
    if (ProtocolCompatibleWithProtocol(lProto, RefPDecl[i]))
      return true;
  return false;
}

/// ClassImplementsProtocol - Checks that 'lProto' protocol
/// has been implemented in IDecl class, its super class or categories (if
/// lookupCategory is true). 
static bool ClassImplementsProtocol(ObjCProtocolDecl *lProto,
                                    ObjCInterfaceDecl *IDecl, 
                                    bool lookupCategory) {
  
  // 1st, look up the class.
  ObjCProtocolDecl **protoList = IDecl->getReferencedProtocols();
  for (unsigned i = 0; i < IDecl->getNumIntfRefProtocols(); i++) {
    if (ProtocolCompatibleWithProtocol(lProto, protoList[i]))
      return true;
  }
  
  // 2nd, look up the category.
  if (lookupCategory)
    for (ObjCCategoryDecl *CDecl = IDecl->getCategoryList(); CDecl;
         CDecl = CDecl->getNextClassCategory()) {
      protoList = CDecl->getReferencedProtocols();
      for (unsigned i = 0; i < CDecl->getNumReferencedProtocols(); i++) {
        if (ProtocolCompatibleWithProtocol(lProto, protoList[i]))
          return true;
      }
    }
  
  // 3rd, look up the super class(s)
  if (IDecl->getSuperClass())
    return 
      ClassImplementsProtocol(lProto, IDecl->getSuperClass(), lookupCategory);
  
  return false;
}

/// ObjCQualifiedIdTypesAreCompatible - We know that one of lhs/rhs is an
/// ObjCQualifiedIDType.
bool Sema::ObjCQualifiedIdTypesAreCompatible(QualType lhs, QualType rhs,
                                             bool compare) {
  // Allow id<P..> and an 'id' or void* type in all cases.
  if (const PointerType *PT = lhs->getAsPointerType()) {
    QualType PointeeTy = PT->getPointeeType();
    if (Context.isObjCIdType(PointeeTy) || PointeeTy->isVoidType())
      return true;
  } else if (const PointerType *PT = rhs->getAsPointerType()) {
    QualType PointeeTy = PT->getPointeeType();
    if (Context.isObjCIdType(PointeeTy) || PointeeTy->isVoidType())
      return true;
  }
  
  if (const ObjCQualifiedIdType *lhsQID = lhs->getAsObjCQualifiedIdType()) {
    const ObjCQualifiedIdType *rhsQID = rhs->getAsObjCQualifiedIdType();
    const ObjCQualifiedInterfaceType *rhsQI = 0;
    if (!rhsQID) {
      // Not comparing two ObjCQualifiedIdType's?
      if (!rhs->isPointerType()) return false;
      QualType rtype = rhs->getAsPointerType()->getPointeeType();

      rhsQI = rtype->getAsObjCQualifiedInterfaceType();
      if (rhsQI == 0) {
        // If the RHS is an interface pointer ('NSString*'), handle it.
        if (const ObjCInterfaceType *IT = rtype->getAsObjCInterfaceType()) {
          ObjCInterfaceDecl *rhsID = IT->getDecl();
          for (unsigned i = 0; i != lhsQID->getNumProtocols(); ++i) {
            // when comparing an id<P> on lhs with a static type on rhs,
            // see if static class implements all of id's protocols, directly or
            // through its super class and categories.
            if (!ClassImplementsProtocol(lhsQID->getProtocols(i), rhsID, true))
              return false;
          }
          return true;
        }
      }      
    }
    
    ObjCQualifiedIdType::qual_iterator RHSProtoI, RHSProtoE;
    if (rhsQI) {
      RHSProtoI = rhsQI->qual_begin();
      RHSProtoE = rhsQI->qual_end();
    } else if (rhsQID) {
      RHSProtoI = rhsQID->qual_begin();
      RHSProtoE = rhsQID->qual_end();
    } else {
      return false;
    }
    
    for (unsigned i =0; i < lhsQID->getNumProtocols(); i++) {
      ObjCProtocolDecl *lhsProto = lhsQID->getProtocols(i);
      bool match = false;

      // when comparing an id<P> on lhs with a static type on rhs,
      // see if static class implements all of id's protocols, directly or
      // through its super class and categories.
      for (; RHSProtoI != RHSProtoE; ++RHSProtoI) {
        ObjCProtocolDecl *rhsProto = *RHSProtoI;
        if (ProtocolCompatibleWithProtocol(lhsProto, rhsProto) ||
            compare && ProtocolCompatibleWithProtocol(rhsProto, lhsProto)) {
          match = true;
          break;
        }
      }
      if (!match)
        return false;
    }
    
    return true;
  }
  
  const ObjCQualifiedIdType *rhsQID = rhs->getAsObjCQualifiedIdType();
  assert(rhsQID && "One of the LHS/RHS should be id<x>");
    
  if (!lhs->isPointerType())
    return false;
  
  QualType ltype = lhs->getAsPointerType()->getPointeeType();
  if (const ObjCQualifiedInterfaceType *lhsQI =
         ltype->getAsObjCQualifiedInterfaceType()) {
    ObjCQualifiedIdType::qual_iterator LHSProtoI = lhsQI->qual_begin();
    ObjCQualifiedIdType::qual_iterator LHSProtoE = lhsQI->qual_end();
    for (; LHSProtoI != LHSProtoE; ++LHSProtoI) {
      bool match = false;
      ObjCProtocolDecl *lhsProto = *LHSProtoI;
      for (unsigned j = 0; j < rhsQID->getNumProtocols(); j++) {
        ObjCProtocolDecl *rhsProto = rhsQID->getProtocols(j);
        if (ProtocolCompatibleWithProtocol(lhsProto, rhsProto) ||
            compare && ProtocolCompatibleWithProtocol(rhsProto, lhsProto)) {
          match = true;
          break;
        }
      }
      if (!match)
        return false;
    }
    return true;
  }
  
  if (const ObjCInterfaceType *IT = ltype->getAsObjCInterfaceType()) {
    // for static type vs. qualified 'id' type, check that class implements
    // all of 'id's protocols.
    ObjCInterfaceDecl *lhsID = IT->getDecl();
    for (unsigned j = 0; j < rhsQID->getNumProtocols(); j++) {
      ObjCProtocolDecl *rhsProto = rhsQID->getProtocols(j);
      if (!ClassImplementsProtocol(rhsProto, lhsID, compare))
        return false;
    }
    return true;
  }
  return false;
}

