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
    if (strIFace)
      Context.setObjCConstantStringInterface(strIFace);
  }
  QualType t = Context.getObjCConstantStringInterface();
  // If there is no NSConstantString interface defined then treat constant
  // strings as untyped objects and let the runtime figure it out later.
  if (t == QualType()) {
    t = Context.getObjCIdType();
  } else {
    t = Context.getPointerType(t);
  }
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
  if (!strcmp(receiverName->getName(), "super") && getCurMethodDecl()) {
    ClassDecl = getCurMethodDecl()->getClassInterface()->getSuperClass();
    if (!ClassDecl)
      return Diag(lbrac, diag::error_no_super_class,
                  getCurMethodDecl()->getClassInterface()->getName());
    if (getCurMethodDecl()->isInstance()) {
      QualType superTy = Context.getObjCInterfaceType(ClassDecl);
      superTy = Context.getPointerType(superTy);
      ExprResult ReceiverExpr = new PreDefinedExpr(SourceLocation(), superTy,
          PreDefinedExpr::ObjCSuper);
      // We are really in an instance method, redirect.
      return ActOnInstanceMessage(ReceiverExpr.Val, Sel, lbrac, rbrac,
                                  Args, NumArgs);
    }
    // We are sending a message to 'super' within a class method. Do nothing,
    // the receiver will pass through as 'super' (how convenient:-).
  } else
    ClassDecl = getObjCInterfaceDecl(receiverName);
  
  // ClassDecl is null in the following case.
  //
  //  typedef XCElementDisplayRect XCElementGraphicsRect;
  //
  //  @implementation XCRASlice
  //  - whatever { // Note that XCElementGraphicsRect is a typedef name.
  //    _sGraphicsDelegate =[[XCElementGraphicsRect alloc] init];
  //  }
  //
  // FIXME: Investigate why GCC allows the above.
  ObjCMethodDecl *Method = 0;
  QualType returnType;
  if (ClassDecl) {
    Method = ClassDecl->lookupClassMethod(Sel);
    
    // If we have an implementation in scope, check "private" methods.
    if (!Method) {
      if (ObjCImplementationDecl *ImpDecl = 
          ObjCImplementations[ClassDecl->getIdentifier()])
        Method = ImpDecl->getClassMethod(Sel);
    }
    // Before we give up, check if the selector is an instance method.
    if (!Method)
      Method = ClassDecl->lookupInstanceMethod(Sel);
  }
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

  // If we have the ObjCInterfaceDecl* for the class that is receiving
  // the message, use that to construct the ObjCMessageExpr.  Otherwise
  // pass on the IdentifierInfo* for the class.
  if (ClassDecl)
    return new ObjCMessageExpr(ClassDecl, Sel, returnType, Method,
                               lbrac, rbrac, ArgExprs, NumArgs);
  else
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
  QualType returnType;

  QualType receiverType = 
    RExpr->getType().getCanonicalType().getUnqualifiedType();
  
  // Handle messages to id.
  if (receiverType == Context.getObjCIdType().getCanonicalType()) {
    ObjCMethodDecl *Method = InstanceMethodPool[Sel].Method;
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
    return new ObjCMessageExpr(RExpr, Sel, returnType, Method, lbrac, rbrac, 
                               ArgExprs, NumArgs);
  }
  
  // Handle messages to Class.
  if (receiverType == Context.getObjCClassType().getCanonicalType()) {
    ObjCMethodDecl *Method = 0;
    if (getCurMethodDecl()) {
      ObjCInterfaceDecl* ClassDecl = getCurMethodDecl()->getClassInterface();
      // If we have an implementation in scope, check "private" methods.
      if (ClassDecl)
        if (ObjCImplementationDecl *ImpDecl = 
              ObjCImplementations[ClassDecl->getIdentifier()])
          Method = ImpDecl->getClassMethod(Sel);
    }
    if (!Method)
      Method = FactoryMethodPool[Sel].Method;
    if (!Method)
      Method = InstanceMethodPool[Sel].Method;
    if (!Method) {
      Diag(lbrac, diag::warn_method_not_found, std::string("-"), Sel.getName(),
           RExpr->getSourceRange());
      returnType = Context.getObjCIdType();
    } else {
      returnType = Method->getResultType();
      if (Sel.getNumArgs())
        if (CheckMessageArgumentTypes(ArgExprs, Sel.getNumArgs(), Method))
          return true;
    }

    return new ObjCMessageExpr(RExpr, Sel, returnType, Method, lbrac, rbrac, 
                               ArgExprs, NumArgs);
  }
  
  ObjCMethodDecl *Method = 0;
  ObjCInterfaceDecl* ClassDecl = 0;
  
  // We allow sending a message to a qualified ID ("id<foo>"), which is ok as 
  // long as one of the protocols implements the selector (if not, warn).
  if (ObjCQualifiedIdType *QIT = 
           dyn_cast<ObjCQualifiedIdType>(receiverType)) {
    // Search protocols
    for (unsigned i = 0; i < QIT->getNumProtocols(); i++) {
      ObjCProtocolDecl *PDecl = QIT->getProtocols(i);
      if (PDecl && (Method = PDecl->lookupInstanceMethod(Sel)))
        break;
    }
    if (!Method)
      Diag(lbrac, diag::warn_method_not_found_in_protocol, 
           std::string("-"), Sel.getName(),
           RExpr->getSourceRange());
  } else if (const ObjCInterfaceType *OCIReceiver = 
                receiverType->getAsPointerToObjCInterfaceType()) {
    // We allow sending a message to a pointer to an interface (an object).
    
    ClassDecl = OCIReceiver->getDecl();
    // FIXME: consider using InstanceMethodPool, since it will be faster
    // than the following method (which can do *many* linear searches). The
    // idea is to add class info to InstanceMethodPool.
    Method = ClassDecl->lookupInstanceMethod(Sel);
    
    if (!Method) {
      // Search protocol qualifiers.
      for (ObjCQualifiedIdType::qual_iterator QI = OCIReceiver->qual_begin(),
           E = OCIReceiver->qual_end(); QI != E; ++QI) {
        if ((Method = (*QI)->lookupInstanceMethod(Sel)))
          break;
      }
    }
    
    if (!Method && !OCIReceiver->qual_empty())
      Diag(lbrac, diag::warn_method_not_found_in_protocol, 
           std::string("-"), Sel.getName(),
           SourceRange(lbrac, rbrac));
  } else {
    Diag(lbrac, diag::error_bad_receiver_type,
         RExpr->getType().getAsString(), RExpr->getSourceRange());
    return true;
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
                                    bool lookupCategory,
                                    bool RHSIsQualifiedID = false) {
  
  // 1st, look up the class.
  ObjCProtocolDecl **protoList = IDecl->getReferencedProtocols();
  for (unsigned i = 0; i < IDecl->getNumIntfRefProtocols(); i++) {
    if (ProtocolCompatibleWithProtocol(lProto, protoList[i]))
      return true;
    // This is dubious and is added to be compatible with gcc.
    // In gcc, it is also allowed assigning a protocol-qualified 'id'
    // type to a LHS object when protocol in qualified LHS is in list
    // of protocols in the rhs 'id' object. This IMO, should be a bug.
    // FIXME: Treat this as an extension, and flag this as an error when
    //  GCC extensions are not enabled.
    else if (RHSIsQualifiedID &&
             ProtocolCompatibleWithProtocol(protoList[i], lProto))
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
      ClassImplementsProtocol(lProto, IDecl->getSuperClass(), lookupCategory,
                              RHSIsQualifiedID);
  
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
    QualType rtype;
    
    if (!rhsQID) {
      // Not comparing two ObjCQualifiedIdType's?
      if (!rhs->isPointerType()) return false;
      
      rtype = rhs->getAsPointerType()->getPointeeType();
      rhsQI = rtype->getAsObjCQualifiedInterfaceType();
      if (rhsQI == 0) {
        // If the RHS is a unqualified interface pointer "NSString*", 
        // make sure we check the class hierarchy.
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
    if (rhsQI) { // We have a qualified interface (e.g. "NSObject<Proto> *").
      RHSProtoI = rhsQI->qual_begin();
      RHSProtoE = rhsQI->qual_end();
    } else if (rhsQID) { // We have a qualified id (e.g. "id<Proto> *").
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
      if (rhsQI) {
        // If the RHS is a qualified interface pointer "NSString<P>*", 
        // make sure we check the class hierarchy.
        if (const ObjCInterfaceType *IT = rtype->getAsObjCInterfaceType()) {
          ObjCInterfaceDecl *rhsID = IT->getDecl();
          for (unsigned i = 0; i != lhsQID->getNumProtocols(); ++i) {
            // when comparing an id<P> on lhs with a static type on rhs,
            // see if static class implements all of id's protocols, directly or
            // through its super class and categories.
            if (ClassImplementsProtocol(lhsQID->getProtocols(i), rhsID, true)) {
              match = true;
              break;
            }
          }
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
      if (!ClassImplementsProtocol(rhsProto, lhsID, compare, true))
        return false;
    }
    return true;
  }
  return false;
}

