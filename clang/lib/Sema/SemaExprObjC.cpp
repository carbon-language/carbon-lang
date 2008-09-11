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
#include "clang/Basic/Diagnostic.h"
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

bool Sema::CheckMessageArgumentTypes(Expr **Args, Selector Sel,
                                     ObjCMethodDecl *Method, 
                                     const char *PrefixStr,
                                     SourceLocation lbrac, SourceLocation rbrac,
                                     QualType &ReturnType) {  
  unsigned NumArgs = Sel.getNumArgs();  
  if (!Method) {
    Diag(lbrac, diag::warn_method_not_found, std::string(PrefixStr),
         Sel.getName(), SourceRange(lbrac, rbrac));
    ReturnType = Context.getObjCIdType();
    return false;
  } else {
    ReturnType = Method->getResultType();
  }
   
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
  bool isSuper = false;
  
  if (receiverName == SuperID && getCurMethodDecl()) {
    isSuper = true;
    ClassDecl = getCurMethodDecl()->getClassInterface()->getSuperClass();
    if (!ClassDecl)
      return Diag(lbrac, diag::error_no_super_class,
                  getCurMethodDecl()->getClassInterface()->getName());
    if (getCurMethodDecl()->isInstance()) {
      QualType superTy = Context.getObjCInterfaceType(ClassDecl);
      superTy = Context.getPointerType(superTy);
      ExprResult ReceiverExpr = new PredefinedExpr(SourceLocation(), superTy,
          PredefinedExpr::ObjCSuper);
      // We are really in an instance method, redirect.
      return ActOnInstanceMessage(ReceiverExpr.Val, Sel, lbrac, rbrac,
                                  Args, NumArgs);
    }
    // We are sending a message to 'super' within a class method. Do nothing,
    // the receiver will pass through as 'super' (how convenient:-).
  } else
    ClassDecl = getObjCInterfaceDecl(receiverName);
  
  // The following code allows for the following GCC-ism:
  //
  //  typedef XCElementDisplayRect XCElementGraphicsRect;
  //
  //  @implementation XCRASlice
  //  - whatever { // Note that XCElementGraphicsRect is a typedef name.
  //    _sGraphicsDelegate =[[XCElementGraphicsRect alloc] init];
  //  }
  //
  // If necessary, the following lookup could move to getObjCInterfaceDecl().
  if (!ClassDecl) {
    Decl *IDecl = LookupDecl(receiverName, Decl::IDNS_Ordinary, 0, false);
    if (TypedefDecl *OCTD = dyn_cast_or_null<TypedefDecl>(IDecl)) {
      const ObjCInterfaceType *OCIT;
      OCIT = OCTD->getUnderlyingType()->getAsObjCInterfaceType();
      if (OCIT)
        ClassDecl = OCIT->getDecl();
    }
  }
  assert(ClassDecl && "missing interface declaration");
  ObjCMethodDecl *Method = 0;
  QualType returnType;
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

  if (CheckMessageArgumentTypes(ArgExprs, Sel, Method, "+", 
                                lbrac, rbrac, returnType))
    return true;

  // If we have the ObjCInterfaceDecl* for the class that is receiving
  // the message, use that to construct the ObjCMessageExpr.  Otherwise
  // pass on the IdentifierInfo* for the class.
  // FIXME: need to do a better job handling 'super' usage within a class 
  // For now, we simply pass the "super" identifier through (which isn't
  // consistent with instance methods.
  if (isSuper)
    return new ObjCMessageExpr(receiverName, Sel, returnType, Method,
                               lbrac, rbrac, ArgExprs, NumArgs);
  else
    return new ObjCMessageExpr(ClassDecl, Sel, returnType, Method,
                               lbrac, rbrac, ArgExprs, NumArgs);
}

// ActOnInstanceMessage - used for both unary and keyword messages.
// ArgExprs is optional - if it is present, the number of expressions
// is obtained from Sel.getNumArgs().
Sema::ExprResult Sema::ActOnInstanceMessage(ExprTy *receiver, Selector Sel,
                                            SourceLocation lbrac, 
                                            SourceLocation rbrac,
                                            ExprTy **Args, unsigned NumArgs) {
  assert(receiver && "missing receiver expression");
  
  Expr **ArgExprs = reinterpret_cast<Expr **>(Args);
  Expr *RExpr = static_cast<Expr *>(receiver);
  QualType returnType;

  QualType ReceiverCType =
    Context.getCanonicalType(RExpr->getType()).getUnqualifiedType();
  
  // Handle messages to id.
  if (ReceiverCType == Context.getCanonicalType(Context.getObjCIdType())) {
    ObjCMethodDecl *Method = InstanceMethodPool[Sel].Method;
    if (!Method)
      Method = FactoryMethodPool[Sel].Method;
    if (CheckMessageArgumentTypes(ArgExprs, Sel, Method, "-", 
                                  lbrac, rbrac, returnType))
      return true;
    return new ObjCMessageExpr(RExpr, Sel, returnType, Method, lbrac, rbrac, 
                               ArgExprs, NumArgs);
  }
  
  // Handle messages to Class.
  if (ReceiverCType == Context.getCanonicalType(Context.getObjCClassType())) {
    ObjCMethodDecl *Method = 0;
    if (ObjCMethodDecl *CurMeth = getCurMethodDecl()) {
      // If we have an implementation in scope, check "private" methods.
      if (ObjCInterfaceDecl *ClassDecl = CurMeth->getClassInterface())
        if (ObjCImplementationDecl *ImpDecl = 
              ObjCImplementations[ClassDecl->getIdentifier()])
          Method = ImpDecl->getClassMethod(Sel);
    }
    if (!Method)
      Method = FactoryMethodPool[Sel].Method;
    if (!Method)
      Method = InstanceMethodPool[Sel].Method;
    if (CheckMessageArgumentTypes(ArgExprs, Sel, Method, "-", 
                                  lbrac, rbrac, returnType))
      return true;
    return new ObjCMessageExpr(RExpr, Sel, returnType, Method, lbrac, rbrac, 
                               ArgExprs, NumArgs);
  }
  
  ObjCMethodDecl *Method = 0;
  ObjCInterfaceDecl* ClassDecl = 0;
  
  // We allow sending a message to a qualified ID ("id<foo>"), which is ok as 
  // long as one of the protocols implements the selector (if not, warn).
  if (ObjCQualifiedIdType *QIT = dyn_cast<ObjCQualifiedIdType>(ReceiverCType)) {
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
                ReceiverCType->getAsPointerToObjCInterfaceType()) {
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
  if (CheckMessageArgumentTypes(ArgExprs, Sel, Method, "-", 
                                lbrac, rbrac, returnType))
    return true;
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
  for (ObjCProtocolDecl::protocol_iterator PI = rProto->protocol_begin(),
       E = rProto->protocol_end(); PI != E; ++PI)
    if (ProtocolCompatibleWithProtocol(lProto, *PI))
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
  const ObjCList<ObjCProtocolDecl> &Protocols =
    IDecl->getReferencedProtocols();

  for (ObjCList<ObjCProtocolDecl>::iterator PI = Protocols.begin(),
       E = Protocols.end(); PI != E; ++PI) {
    if (ProtocolCompatibleWithProtocol(lProto, *PI))
      return true;
    // This is dubious and is added to be compatible with gcc.
    // In gcc, it is also allowed assigning a protocol-qualified 'id'
    // type to a LHS object when protocol in qualified LHS is in list
    // of protocols in the rhs 'id' object. This IMO, should be a bug.
    // FIXME: Treat this as an extension, and flag this as an error when
    //  GCC extensions are not enabled.
    if (RHSIsQualifiedID && ProtocolCompatibleWithProtocol(*PI, lProto))
      return true;
  }
  
  // 2nd, look up the category.
  if (lookupCategory)
    for (ObjCCategoryDecl *CDecl = IDecl->getCategoryList(); CDecl;
         CDecl = CDecl->getNextClassCategory()) {
      for (ObjCCategoryDecl::protocol_iterator PI = CDecl->protocol_begin(),
           E = CDecl->protocol_end(); PI != E; ++PI)
        if (ProtocolCompatibleWithProtocol(lProto, *PI))
          return true;
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

