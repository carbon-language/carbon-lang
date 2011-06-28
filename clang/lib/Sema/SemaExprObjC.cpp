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

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Initialization.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "llvm/ADT/SmallString.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang;
using namespace sema;

ExprResult Sema::ParseObjCStringLiteral(SourceLocation *AtLocs,
                                        Expr **strings,
                                        unsigned NumStrings) {
  StringLiteral **Strings = reinterpret_cast<StringLiteral**>(strings);

  // Most ObjC strings are formed out of a single piece.  However, we *can*
  // have strings formed out of multiple @ strings with multiple pptokens in
  // each one, e.g. @"foo" "bar" @"baz" "qux"   which need to be turned into one
  // StringLiteral for ObjCStringLiteral to hold onto.
  StringLiteral *S = Strings[0];

  // If we have a multi-part string, merge it all together.
  if (NumStrings != 1) {
    // Concatenate objc strings.
    llvm::SmallString<128> StrBuf;
    llvm::SmallVector<SourceLocation, 8> StrLocs;

    for (unsigned i = 0; i != NumStrings; ++i) {
      S = Strings[i];

      // ObjC strings can't be wide.
      if (S->isWide()) {
        Diag(S->getLocStart(), diag::err_cfstring_literal_not_string_constant)
          << S->getSourceRange();
        return true;
      }

      // Append the string.
      StrBuf += S->getString();

      // Get the locations of the string tokens.
      StrLocs.append(S->tokloc_begin(), S->tokloc_end());
    }

    // Create the aggregate string with the appropriate content and location
    // information.
    S = StringLiteral::Create(Context, StrBuf,
                              /*Wide=*/false, /*Pascal=*/false,
                              Context.getPointerType(Context.CharTy),
                              &StrLocs[0], StrLocs.size());
  }

  // Verify that this composite string is acceptable for ObjC strings.
  if (CheckObjCString(S))
    return true;

  // Initialize the constant string interface lazily. This assumes
  // the NSString interface is seen in this translation unit. Note: We
  // don't use NSConstantString, since the runtime team considers this
  // interface private (even though it appears in the header files).
  QualType Ty = Context.getObjCConstantStringInterface();
  if (!Ty.isNull()) {
    Ty = Context.getObjCObjectPointerType(Ty);
  } else if (getLangOptions().NoConstantCFStrings) {
    IdentifierInfo *NSIdent=0;
    std::string StringClass(getLangOptions().ObjCConstantStringClass);
    
    if (StringClass.empty())
      NSIdent = &Context.Idents.get("NSConstantString");
    else
      NSIdent = &Context.Idents.get(StringClass);
    
    NamedDecl *IF = LookupSingleName(TUScope, NSIdent, AtLocs[0],
                                     LookupOrdinaryName);
    if (ObjCInterfaceDecl *StrIF = dyn_cast_or_null<ObjCInterfaceDecl>(IF)) {
      Context.setObjCConstantStringInterface(StrIF);
      Ty = Context.getObjCConstantStringInterface();
      Ty = Context.getObjCObjectPointerType(Ty);
    } else {
      // If there is no NSConstantString interface defined then treat this
      // as error and recover from it.
      Diag(S->getLocStart(), diag::err_no_nsconstant_string_class) << NSIdent
        << S->getSourceRange();
      Ty = Context.getObjCIdType();
    }
  } else {
    IdentifierInfo *NSIdent = &Context.Idents.get("NSString");
    NamedDecl *IF = LookupSingleName(TUScope, NSIdent, AtLocs[0],
                                     LookupOrdinaryName);
    if (ObjCInterfaceDecl *StrIF = dyn_cast_or_null<ObjCInterfaceDecl>(IF)) {
      Context.setObjCConstantStringInterface(StrIF);
      Ty = Context.getObjCConstantStringInterface();
      Ty = Context.getObjCObjectPointerType(Ty);
    } else {
      // If there is no NSString interface defined then treat constant
      // strings as untyped objects and let the runtime figure it out later.
      Ty = Context.getObjCIdType();
    }
  }

  return new (Context) ObjCStringLiteral(S, Ty, AtLocs[0]);
}

ExprResult Sema::BuildObjCEncodeExpression(SourceLocation AtLoc,
                                      TypeSourceInfo *EncodedTypeInfo,
                                      SourceLocation RParenLoc) {
  QualType EncodedType = EncodedTypeInfo->getType();
  QualType StrTy;
  if (EncodedType->isDependentType())
    StrTy = Context.DependentTy;
  else {
    if (!EncodedType->getAsArrayTypeUnsafe() && //// Incomplete array is handled.
        !EncodedType->isVoidType()) // void is handled too.
      if (RequireCompleteType(AtLoc, EncodedType,
                         PDiag(diag::err_incomplete_type_objc_at_encode)
                             << EncodedTypeInfo->getTypeLoc().getSourceRange()))
        return ExprError();

    std::string Str;
    Context.getObjCEncodingForType(EncodedType, Str);

    // The type of @encode is the same as the type of the corresponding string,
    // which is an array type.
    StrTy = Context.CharTy;
    // A C++ string literal has a const-qualified element type (C++ 2.13.4p1).
    if (getLangOptions().CPlusPlus || getLangOptions().ConstStrings)
      StrTy.addConst();
    StrTy = Context.getConstantArrayType(StrTy, llvm::APInt(32, Str.size()+1),
                                         ArrayType::Normal, 0);
  }

  return new (Context) ObjCEncodeExpr(StrTy, EncodedTypeInfo, AtLoc, RParenLoc);
}

ExprResult Sema::ParseObjCEncodeExpression(SourceLocation AtLoc,
                                           SourceLocation EncodeLoc,
                                           SourceLocation LParenLoc,
                                           ParsedType ty,
                                           SourceLocation RParenLoc) {
  // FIXME: Preserve type source info ?
  TypeSourceInfo *TInfo;
  QualType EncodedType = GetTypeFromParser(ty, &TInfo);
  if (!TInfo)
    TInfo = Context.getTrivialTypeSourceInfo(EncodedType,
                                             PP.getLocForEndOfToken(LParenLoc));

  return BuildObjCEncodeExpression(AtLoc, TInfo, RParenLoc);
}

ExprResult Sema::ParseObjCSelectorExpression(Selector Sel,
                                             SourceLocation AtLoc,
                                             SourceLocation SelLoc,
                                             SourceLocation LParenLoc,
                                             SourceLocation RParenLoc) {
  ObjCMethodDecl *Method = LookupInstanceMethodInGlobalPool(Sel,
                             SourceRange(LParenLoc, RParenLoc), false, false);
  if (!Method)
    Method = LookupFactoryMethodInGlobalPool(Sel,
                                          SourceRange(LParenLoc, RParenLoc));
  if (!Method)
    Diag(SelLoc, diag::warn_undeclared_selector) << Sel;

  llvm::DenseMap<Selector, SourceLocation>::iterator Pos
    = ReferencedSelectors.find(Sel);
  if (Pos == ReferencedSelectors.end())
    ReferencedSelectors.insert(std::make_pair(Sel, SelLoc));

  // In ARC, forbid the user from using @selector for 
  // retain/release/autorelease/dealloc/retainCount.
  if (getLangOptions().ObjCAutoRefCount) {
    switch (Sel.getMethodFamily()) {
    case OMF_retain:
    case OMF_release:
    case OMF_autorelease:
    case OMF_retainCount:
    case OMF_dealloc:
      Diag(AtLoc, diag::err_arc_illegal_selector) << 
        Sel << SourceRange(LParenLoc, RParenLoc);
      break;

    case OMF_None:
    case OMF_alloc:
    case OMF_copy:
    case OMF_init:
    case OMF_mutableCopy:
    case OMF_new:
    case OMF_self:
      break;
    }
  }
  QualType Ty = Context.getObjCSelType();
  return new (Context) ObjCSelectorExpr(Ty, Sel, AtLoc, RParenLoc);
}

ExprResult Sema::ParseObjCProtocolExpression(IdentifierInfo *ProtocolId,
                                             SourceLocation AtLoc,
                                             SourceLocation ProtoLoc,
                                             SourceLocation LParenLoc,
                                             SourceLocation RParenLoc) {
  ObjCProtocolDecl* PDecl = LookupProtocol(ProtocolId, ProtoLoc);
  if (!PDecl) {
    Diag(ProtoLoc, diag::err_undeclared_protocol) << ProtocolId;
    return true;
  }

  QualType Ty = Context.getObjCProtoType();
  if (Ty.isNull())
    return true;
  Ty = Context.getObjCObjectPointerType(Ty);
  return new (Context) ObjCProtocolExpr(Ty, PDecl, AtLoc, RParenLoc);
}

/// Try to capture an implicit reference to 'self'.
ObjCMethodDecl *Sema::tryCaptureObjCSelf() {
  // Ignore block scopes: we can capture through them.
  DeclContext *DC = CurContext;
  while (true) {
    if (isa<BlockDecl>(DC)) DC = cast<BlockDecl>(DC)->getDeclContext();
    else if (isa<EnumDecl>(DC)) DC = cast<EnumDecl>(DC)->getDeclContext();
    else break;
  }

  // If we're not in an ObjC method, error out.  Note that, unlike the
  // C++ case, we don't require an instance method --- class methods
  // still have a 'self', and we really do still need to capture it!
  ObjCMethodDecl *method = dyn_cast<ObjCMethodDecl>(DC);
  if (!method)
    return 0;

  ImplicitParamDecl *self = method->getSelfDecl();
  assert(self && "capturing 'self' in non-definition?");

  // Mark that we're closing on 'this' in all the block scopes, if applicable.
  for (unsigned idx = FunctionScopes.size() - 1;
       isa<BlockScopeInfo>(FunctionScopes[idx]);
       --idx) {
    BlockScopeInfo *blockScope = cast<BlockScopeInfo>(FunctionScopes[idx]);
    unsigned &captureIndex = blockScope->CaptureMap[self];
    if (captureIndex) break;

    bool nested = isa<BlockScopeInfo>(FunctionScopes[idx-1]);
    blockScope->Captures.push_back(
              BlockDecl::Capture(self, /*byref*/ false, nested, /*copy*/ 0));
    captureIndex = blockScope->Captures.size(); // +1
  }

  return method;
}

QualType Sema::getMessageSendResultType(QualType ReceiverType,
                                        ObjCMethodDecl *Method,
                                    bool isClassMessage, bool isSuperMessage) {
  assert(Method && "Must have a method");
  if (!Method->hasRelatedResultType())
    return Method->getSendResultType();
  
  // If a method has a related return type:
  //   - if the method found is an instance method, but the message send
  //     was a class message send, T is the declared return type of the method
  //     found
  if (Method->isInstanceMethod() && isClassMessage)
    return Method->getSendResultType();
  
  //   - if the receiver is super, T is a pointer to the class of the 
  //     enclosing method definition
  if (isSuperMessage) {
    if (ObjCMethodDecl *CurMethod = getCurMethodDecl())
      if (ObjCInterfaceDecl *Class = CurMethod->getClassInterface())
        return Context.getObjCObjectPointerType(
                                        Context.getObjCInterfaceType(Class));
  }
    
  //   - if the receiver is the name of a class U, T is a pointer to U
  if (ReceiverType->getAs<ObjCInterfaceType>() ||
      ReceiverType->isObjCQualifiedInterfaceType())
    return Context.getObjCObjectPointerType(ReceiverType);
  //   - if the receiver is of type Class or qualified Class type, 
  //     T is the declared return type of the method.
  if (ReceiverType->isObjCClassType() ||
      ReceiverType->isObjCQualifiedClassType())
    return  Method->getSendResultType();
  
  //   - if the receiver is id, qualified id, Class, or qualified Class, T
  //     is the receiver type, otherwise
  //   - T is the type of the receiver expression.
  return ReceiverType;
}

void Sema::EmitRelatedResultTypeNote(const Expr *E) {
  E = E->IgnoreParenImpCasts();
  const ObjCMessageExpr *MsgSend = dyn_cast<ObjCMessageExpr>(E);
  if (!MsgSend)
    return;
  
  const ObjCMethodDecl *Method = MsgSend->getMethodDecl();
  if (!Method)
    return;
  
  if (!Method->hasRelatedResultType())
    return;
  
  if (Context.hasSameUnqualifiedType(Method->getResultType()
                                                        .getNonReferenceType(),
                                     MsgSend->getType()))
    return;
  
  Diag(Method->getLocation(), diag::note_related_result_type_inferred)
    << Method->isInstanceMethod() << Method->getSelector()
    << MsgSend->getType();
}

bool Sema::CheckMessageArgumentTypes(QualType ReceiverType,
                                     Expr **Args, unsigned NumArgs,
                                     Selector Sel, ObjCMethodDecl *Method,
                                     bool isClassMessage, bool isSuperMessage,
                                     SourceLocation lbrac, SourceLocation rbrac,
                                     QualType &ReturnType, ExprValueKind &VK) {
  if (!Method) {
    // Apply default argument promotion as for (C99 6.5.2.2p6).
    for (unsigned i = 0; i != NumArgs; i++) {
      if (Args[i]->isTypeDependent())
        continue;

      ExprResult Result = DefaultArgumentPromotion(Args[i]);
      if (Result.isInvalid())
        return true;
      Args[i] = Result.take();
    }

    unsigned DiagID;
    if (getLangOptions().ObjCAutoRefCount)
      DiagID = diag::err_arc_method_not_found;
    else
      DiagID = isClassMessage ? diag::warn_class_method_not_found
                              : diag::warn_inst_method_not_found;
    Diag(lbrac, DiagID)
      << Sel << isClassMessage << SourceRange(lbrac, rbrac);
    ReturnType = Context.getObjCIdType();
    VK = VK_RValue;
    return false;
  }

  ReturnType = getMessageSendResultType(ReceiverType, Method, isClassMessage, 
                                        isSuperMessage);
  VK = Expr::getValueKindForType(Method->getResultType());

  unsigned NumNamedArgs = Sel.getNumArgs();
  // Method might have more arguments than selector indicates. This is due
  // to addition of c-style arguments in method.
  if (Method->param_size() > Sel.getNumArgs())
    NumNamedArgs = Method->param_size();
  // FIXME. This need be cleaned up.
  if (NumArgs < NumNamedArgs) {
    Diag(lbrac, diag::err_typecheck_call_too_few_args)
      << 2 << NumNamedArgs << NumArgs;
    return false;
  }

  bool IsError = false;
  for (unsigned i = 0; i < NumNamedArgs; i++) {
    // We can't do any type-checking on a type-dependent argument.
    if (Args[i]->isTypeDependent())
      continue;

    Expr *argExpr = Args[i];

    ParmVarDecl *Param = Method->param_begin()[i];
    assert(argExpr && "CheckMessageArgumentTypes(): missing expression");

    if (RequireCompleteType(argExpr->getSourceRange().getBegin(),
                            Param->getType(),
                            PDiag(diag::err_call_incomplete_argument)
                              << argExpr->getSourceRange()))
      return true;

    InitializedEntity Entity = InitializedEntity::InitializeParameter(Context,
                                                                      Param);
    ExprResult ArgE = PerformCopyInitialization(Entity, lbrac, Owned(argExpr));
    if (ArgE.isInvalid())
      IsError = true;
    else
      Args[i] = ArgE.takeAs<Expr>();
  }

  // Promote additional arguments to variadic methods.
  if (Method->isVariadic()) {
    for (unsigned i = NumNamedArgs; i < NumArgs; ++i) {
      if (Args[i]->isTypeDependent())
        continue;

      ExprResult Arg = DefaultVariadicArgumentPromotion(Args[i], VariadicMethod, 0);
      IsError |= Arg.isInvalid();
      Args[i] = Arg.take();
    }
  } else {
    // Check for extra arguments to non-variadic methods.
    if (NumArgs != NumNamedArgs) {
      Diag(Args[NumNamedArgs]->getLocStart(),
           diag::err_typecheck_call_too_many_args)
        << 2 /*method*/ << NumNamedArgs << NumArgs
        << Method->getSourceRange()
        << SourceRange(Args[NumNamedArgs]->getLocStart(),
                       Args[NumArgs-1]->getLocEnd());
    }
  }
  // diagnose nonnull arguments.
  for (specific_attr_iterator<NonNullAttr>
       i = Method->specific_attr_begin<NonNullAttr>(),
       e = Method->specific_attr_end<NonNullAttr>(); i != e; ++i) {
    CheckNonNullArguments(*i, Args, lbrac);
  }

  DiagnoseSentinelCalls(Method, lbrac, Args, NumArgs);
  return IsError;
}

bool Sema::isSelfExpr(Expr *receiver) {
  // 'self' is objc 'self' in an objc method only.
  DeclContext *DC = CurContext;
  while (isa<BlockDecl>(DC))
    DC = DC->getParent();
  if (DC && !isa<ObjCMethodDecl>(DC))
    return false;
  receiver = receiver->IgnoreParenLValueCasts();
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(receiver))
    if (DRE->getDecl()->getIdentifier() == &Context.Idents.get("self"))
      return true;
  return false;
}

// Helper method for ActOnClassMethod/ActOnInstanceMethod.
// Will search "local" class/category implementations for a method decl.
// If failed, then we search in class's root for an instance method.
// Returns 0 if no method is found.
ObjCMethodDecl *Sema::LookupPrivateClassMethod(Selector Sel,
                                          ObjCInterfaceDecl *ClassDecl) {
  ObjCMethodDecl *Method = 0;
  // lookup in class and all superclasses
  while (ClassDecl && !Method) {
    if (ObjCImplementationDecl *ImpDecl = ClassDecl->getImplementation())
      Method = ImpDecl->getClassMethod(Sel);

    // Look through local category implementations associated with the class.
    if (!Method)
      Method = ClassDecl->getCategoryClassMethod(Sel);

    // Before we give up, check if the selector is an instance method.
    // But only in the root. This matches gcc's behaviour and what the
    // runtime expects.
    if (!Method && !ClassDecl->getSuperClass()) {
      Method = ClassDecl->lookupInstanceMethod(Sel);
      // Look through local category implementations associated
      // with the root class.
      if (!Method)
        Method = LookupPrivateInstanceMethod(Sel, ClassDecl);
    }

    ClassDecl = ClassDecl->getSuperClass();
  }
  return Method;
}

ObjCMethodDecl *Sema::LookupPrivateInstanceMethod(Selector Sel,
                                              ObjCInterfaceDecl *ClassDecl) {
  ObjCMethodDecl *Method = 0;
  while (ClassDecl && !Method) {
    // If we have implementations in scope, check "private" methods.
    if (ObjCImplementationDecl *ImpDecl = ClassDecl->getImplementation())
      Method = ImpDecl->getInstanceMethod(Sel);

    // Look through local category implementations associated with the class.
    if (!Method)
      Method = ClassDecl->getCategoryInstanceMethod(Sel);
    ClassDecl = ClassDecl->getSuperClass();
  }
  return Method;
}

/// LookupMethodInQualifiedType - Lookups up a method in protocol qualifier 
/// list of a qualified objective pointer type.
ObjCMethodDecl *Sema::LookupMethodInQualifiedType(Selector Sel,
                                              const ObjCObjectPointerType *OPT,
                                              bool Instance)
{
  ObjCMethodDecl *MD = 0;
  for (ObjCObjectPointerType::qual_iterator I = OPT->qual_begin(),
       E = OPT->qual_end(); I != E; ++I) {
    ObjCProtocolDecl *PROTO = (*I);
    if ((MD = PROTO->lookupMethod(Sel, Instance))) {
      return MD;
    }
  }
  return 0;
}

/// HandleExprPropertyRefExpr - Handle foo.bar where foo is a pointer to an
/// objective C interface.  This is a property reference expression.
ExprResult Sema::
HandleExprPropertyRefExpr(const ObjCObjectPointerType *OPT,
                          Expr *BaseExpr, SourceLocation OpLoc,
                          DeclarationName MemberName,
                          SourceLocation MemberLoc,
                          SourceLocation SuperLoc, QualType SuperType,
                          bool Super) {
  const ObjCInterfaceType *IFaceT = OPT->getInterfaceType();
  ObjCInterfaceDecl *IFace = IFaceT->getDecl();
  
  if (MemberName.getNameKind() != DeclarationName::Identifier) {
    Diag(MemberLoc, diag::err_invalid_property_name)
      << MemberName << QualType(OPT, 0);
    return ExprError();
  }
  
  IdentifierInfo *Member = MemberName.getAsIdentifierInfo();

  if (IFace->isForwardDecl()) {
    Diag(MemberLoc, diag::err_property_not_found_forward_class)
         << MemberName << QualType(OPT, 0);
    Diag(IFace->getLocation(), diag::note_forward_class);
    return ExprError();
  }
  // Search for a declared property first.
  if (ObjCPropertyDecl *PD = IFace->FindPropertyDeclaration(Member)) {
    // Check whether we can reference this property.
    if (DiagnoseUseOfDecl(PD, MemberLoc))
      return ExprError();
    QualType ResTy = PD->getType();
    ResTy = ResTy.getNonLValueExprType(Context);
    Selector Sel = PP.getSelectorTable().getNullarySelector(Member);
    ObjCMethodDecl *Getter = IFace->lookupInstanceMethod(Sel);
    if (Getter &&
        (Getter->hasRelatedResultType()
         || DiagnosePropertyAccessorMismatch(PD, Getter, MemberLoc)))
        ResTy = getMessageSendResultType(QualType(OPT, 0), Getter, false, 
                                         Super);
             
    if (Super)
      return Owned(new (Context) ObjCPropertyRefExpr(PD, ResTy,
                                                     VK_LValue, OK_ObjCProperty,
                                                     MemberLoc, 
                                                     SuperLoc, SuperType));
    else
      return Owned(new (Context) ObjCPropertyRefExpr(PD, ResTy,
                                                     VK_LValue, OK_ObjCProperty,
                                                     MemberLoc, BaseExpr));
  }
  // Check protocols on qualified interfaces.
  for (ObjCObjectPointerType::qual_iterator I = OPT->qual_begin(),
       E = OPT->qual_end(); I != E; ++I)
    if (ObjCPropertyDecl *PD = (*I)->FindPropertyDeclaration(Member)) {
      // Check whether we can reference this property.
      if (DiagnoseUseOfDecl(PD, MemberLoc))
        return ExprError();
      
      QualType T = PD->getType();
      if (ObjCMethodDecl *Getter = PD->getGetterMethodDecl())
        T = getMessageSendResultType(QualType(OPT, 0), Getter, false, Super);
      if (Super)
        return Owned(new (Context) ObjCPropertyRefExpr(PD, T,
                                                       VK_LValue,
                                                       OK_ObjCProperty,
                                                       MemberLoc, 
                                                       SuperLoc, SuperType));
      else
        return Owned(new (Context) ObjCPropertyRefExpr(PD, T,
                                                       VK_LValue,
                                                       OK_ObjCProperty,
                                                       MemberLoc,
                                                       BaseExpr));
    }
  // If that failed, look for an "implicit" property by seeing if the nullary
  // selector is implemented.

  // FIXME: The logic for looking up nullary and unary selectors should be
  // shared with the code in ActOnInstanceMessage.

  Selector Sel = PP.getSelectorTable().getNullarySelector(Member);
  ObjCMethodDecl *Getter = IFace->lookupInstanceMethod(Sel);
  
  // May be founf in property's qualified list.
  if (!Getter)
    Getter = LookupMethodInQualifiedType(Sel, OPT, true);

  // If this reference is in an @implementation, check for 'private' methods.
  if (!Getter)
    Getter = IFace->lookupPrivateMethod(Sel);

  // Look through local category implementations associated with the class.
  if (!Getter)
    Getter = IFace->getCategoryInstanceMethod(Sel);
  if (Getter) {
    // Check if we can reference this property.
    if (DiagnoseUseOfDecl(Getter, MemberLoc))
      return ExprError();
  }
  // If we found a getter then this may be a valid dot-reference, we
  // will look for the matching setter, in case it is needed.
  Selector SetterSel =
    SelectorTable::constructSetterName(PP.getIdentifierTable(),
                                       PP.getSelectorTable(), Member);
  ObjCMethodDecl *Setter = IFace->lookupInstanceMethod(SetterSel);
  
  // May be founf in property's qualified list.
  if (!Setter)
    Setter = LookupMethodInQualifiedType(SetterSel, OPT, true);
  
  if (!Setter) {
    // If this reference is in an @implementation, also check for 'private'
    // methods.
    Setter = IFace->lookupPrivateMethod(SetterSel);
  }
  // Look through local category implementations associated with the class.
  if (!Setter)
    Setter = IFace->getCategoryInstanceMethod(SetterSel);
    
  if (Setter && DiagnoseUseOfDecl(Setter, MemberLoc))
    return ExprError();

  if (Getter || Setter) {
    QualType PType;
    if (Getter)
      PType = getMessageSendResultType(QualType(OPT, 0), Getter, false, Super);
    else {
      ParmVarDecl *ArgDecl = *Setter->param_begin();
      PType = ArgDecl->getType();
    }
    
    ExprValueKind VK = VK_LValue;
    ExprObjectKind OK = OK_ObjCProperty;
    if (!getLangOptions().CPlusPlus && !PType.hasQualifiers() &&
        PType->isVoidType())
      VK = VK_RValue, OK = OK_Ordinary;

    if (Super)
      return Owned(new (Context) ObjCPropertyRefExpr(Getter, Setter,
                                                     PType, VK, OK,
                                                     MemberLoc,
                                                     SuperLoc, SuperType));
    else
      return Owned(new (Context) ObjCPropertyRefExpr(Getter, Setter,
                                                     PType, VK, OK,
                                                     MemberLoc, BaseExpr));

  }

  // Attempt to correct for typos in property names.
  TypoCorrection Corrected = CorrectTypo(
      DeclarationNameInfo(MemberName, MemberLoc), LookupOrdinaryName, NULL,
      NULL, IFace, false, CTC_NoKeywords, OPT);
  if (ObjCPropertyDecl *Property =
      Corrected.getCorrectionDeclAs<ObjCPropertyDecl>()) {
    DeclarationName TypoResult = Corrected.getCorrection();
    Diag(MemberLoc, diag::err_property_not_found_suggest)
      << MemberName << QualType(OPT, 0) << TypoResult
      << FixItHint::CreateReplacement(MemberLoc, TypoResult.getAsString());
    Diag(Property->getLocation(), diag::note_previous_decl)
      << Property->getDeclName();
    return HandleExprPropertyRefExpr(OPT, BaseExpr, OpLoc,
                                     TypoResult, MemberLoc,
                                     SuperLoc, SuperType, Super);
  }
  ObjCInterfaceDecl *ClassDeclared;
  if (ObjCIvarDecl *Ivar = 
      IFace->lookupInstanceVariable(Member, ClassDeclared)) {
    QualType T = Ivar->getType();
    if (const ObjCObjectPointerType * OBJPT = 
        T->getAsObjCInterfacePointerType()) {
      const ObjCInterfaceType *IFaceT = OBJPT->getInterfaceType();
      if (ObjCInterfaceDecl *IFace = IFaceT->getDecl())
        if (IFace->isForwardDecl()) {
          Diag(MemberLoc, diag::err_property_not_as_forward_class)
          << MemberName << IFace;
          Diag(IFace->getLocation(), diag::note_forward_class);
          return ExprError();
        }
    }
    Diag(MemberLoc, 
         diag::err_ivar_access_using_property_syntax_suggest)
    << MemberName << QualType(OPT, 0) << Ivar->getDeclName()
    << FixItHint::CreateReplacement(OpLoc, "->");
    return ExprError();
  }
  
  Diag(MemberLoc, diag::err_property_not_found)
    << MemberName << QualType(OPT, 0);
  if (Setter)
    Diag(Setter->getLocation(), diag::note_getter_unavailable)
          << MemberName << BaseExpr->getSourceRange();
  return ExprError();
}



ExprResult Sema::
ActOnClassPropertyRefExpr(IdentifierInfo &receiverName,
                          IdentifierInfo &propertyName,
                          SourceLocation receiverNameLoc,
                          SourceLocation propertyNameLoc) {

  IdentifierInfo *receiverNamePtr = &receiverName;
  ObjCInterfaceDecl *IFace = getObjCInterfaceDecl(receiverNamePtr,
                                                  receiverNameLoc);

  bool IsSuper = false;
  if (IFace == 0) {
    // If the "receiver" is 'super' in a method, handle it as an expression-like
    // property reference.
    if (receiverNamePtr->isStr("super")) {
      IsSuper = true;

      if (ObjCMethodDecl *CurMethod = tryCaptureObjCSelf()) {
        if (CurMethod->isInstanceMethod()) {
          QualType T = 
            Context.getObjCInterfaceType(CurMethod->getClassInterface());
          T = Context.getObjCObjectPointerType(T);
        
          return HandleExprPropertyRefExpr(T->getAsObjCInterfacePointerType(),
                                           /*BaseExpr*/0, 
                                           SourceLocation()/*OpLoc*/, 
                                           &propertyName,
                                           propertyNameLoc,
                                           receiverNameLoc, T, true);
        }

        // Otherwise, if this is a class method, try dispatching to our
        // superclass.
        IFace = CurMethod->getClassInterface()->getSuperClass();
      }
    }
    
    if (IFace == 0) {
      Diag(receiverNameLoc, diag::err_expected_ident_or_lparen);
      return ExprError();
    }
  }

  // Search for a declared property first.
  Selector Sel = PP.getSelectorTable().getNullarySelector(&propertyName);
  ObjCMethodDecl *Getter = IFace->lookupClassMethod(Sel);

  // If this reference is in an @implementation, check for 'private' methods.
  if (!Getter)
    if (ObjCMethodDecl *CurMeth = getCurMethodDecl())
      if (ObjCInterfaceDecl *ClassDecl = CurMeth->getClassInterface())
        if (ObjCImplementationDecl *ImpDecl = ClassDecl->getImplementation())
          Getter = ImpDecl->getClassMethod(Sel);

  if (Getter) {
    // FIXME: refactor/share with ActOnMemberReference().
    // Check if we can reference this property.
    if (DiagnoseUseOfDecl(Getter, propertyNameLoc))
      return ExprError();
  }

  // Look for the matching setter, in case it is needed.
  Selector SetterSel =
    SelectorTable::constructSetterName(PP.getIdentifierTable(),
                                       PP.getSelectorTable(), &propertyName);

  ObjCMethodDecl *Setter = IFace->lookupClassMethod(SetterSel);
  if (!Setter) {
    // If this reference is in an @implementation, also check for 'private'
    // methods.
    if (ObjCMethodDecl *CurMeth = getCurMethodDecl())
      if (ObjCInterfaceDecl *ClassDecl = CurMeth->getClassInterface())
        if (ObjCImplementationDecl *ImpDecl = ClassDecl->getImplementation())
          Setter = ImpDecl->getClassMethod(SetterSel);
  }
  // Look through local category implementations associated with the class.
  if (!Setter)
    Setter = IFace->getCategoryClassMethod(SetterSel);

  if (Setter && DiagnoseUseOfDecl(Setter, propertyNameLoc))
    return ExprError();

  if (Getter || Setter) {
    QualType PType;

    ExprValueKind VK = VK_LValue;
    if (Getter) {
      PType = getMessageSendResultType(Context.getObjCInterfaceType(IFace),
                                       Getter, true, 
                                       receiverNamePtr->isStr("super"));
      if (!getLangOptions().CPlusPlus &&
          !PType.hasQualifiers() && PType->isVoidType())
        VK = VK_RValue;
    } else {
      for (ObjCMethodDecl::param_iterator PI = Setter->param_begin(),
           E = Setter->param_end(); PI != E; ++PI)
        PType = (*PI)->getType();
      VK = VK_LValue;
    }

    ExprObjectKind OK = (VK == VK_RValue ? OK_Ordinary : OK_ObjCProperty);

    if (IsSuper)
    return Owned(new (Context) ObjCPropertyRefExpr(Getter, Setter,
                                                   PType, VK, OK,
                                                   propertyNameLoc,
                                                   receiverNameLoc, 
                                          Context.getObjCInterfaceType(IFace)));

    return Owned(new (Context) ObjCPropertyRefExpr(Getter, Setter,
                                                   PType, VK, OK,
                                                   propertyNameLoc,
                                                   receiverNameLoc, IFace));
  }
  return ExprError(Diag(propertyNameLoc, diag::err_property_not_found)
                     << &propertyName << Context.getObjCInterfaceType(IFace));
}

Sema::ObjCMessageKind Sema::getObjCMessageKind(Scope *S,
                                               IdentifierInfo *Name,
                                               SourceLocation NameLoc,
                                               bool IsSuper,
                                               bool HasTrailingDot,
                                               ParsedType &ReceiverType) {
  ReceiverType = ParsedType();

  // If the identifier is "super" and there is no trailing dot, we're
  // messaging super. If the identifier is "super" and there is a
  // trailing dot, it's an instance message.
  if (IsSuper && S->isInObjcMethodScope())
    return HasTrailingDot? ObjCInstanceMessage : ObjCSuperMessage;
  
  LookupResult Result(*this, Name, NameLoc, LookupOrdinaryName);
  LookupName(Result, S);
  
  switch (Result.getResultKind()) {
  case LookupResult::NotFound:
    // Normal name lookup didn't find anything. If we're in an
    // Objective-C method, look for ivars. If we find one, we're done!
    // FIXME: This is a hack. Ivar lookup should be part of normal
    // lookup.
    if (ObjCMethodDecl *Method = getCurMethodDecl()) {
      ObjCInterfaceDecl *ClassDeclared;
      if (Method->getClassInterface()->lookupInstanceVariable(Name, 
                                                              ClassDeclared))
        return ObjCInstanceMessage;
    }
  
    // Break out; we'll perform typo correction below.
    break;

  case LookupResult::NotFoundInCurrentInstantiation:
  case LookupResult::FoundOverloaded:
  case LookupResult::FoundUnresolvedValue:
  case LookupResult::Ambiguous:
    Result.suppressDiagnostics();
    return ObjCInstanceMessage;

  case LookupResult::Found: {
    // If the identifier is a class or not, and there is a trailing dot,
    // it's an instance message.
    if (HasTrailingDot)
      return ObjCInstanceMessage;
    // We found something. If it's a type, then we have a class
    // message. Otherwise, it's an instance message.
    NamedDecl *ND = Result.getFoundDecl();
    QualType T;
    if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(ND))
      T = Context.getObjCInterfaceType(Class);
    else if (TypeDecl *Type = dyn_cast<TypeDecl>(ND))
      T = Context.getTypeDeclType(Type);
    else 
      return ObjCInstanceMessage;

    //  We have a class message, and T is the type we're
    //  messaging. Build source-location information for it.
    TypeSourceInfo *TSInfo = Context.getTrivialTypeSourceInfo(T, NameLoc);
    ReceiverType = CreateParsedType(T, TSInfo);
    return ObjCClassMessage;
  }
  }

  // Determine our typo-correction context.
  CorrectTypoContext CTC = CTC_Expression;
  if (ObjCMethodDecl *Method = getCurMethodDecl())
    if (Method->getClassInterface() &&
        Method->getClassInterface()->getSuperClass())
      CTC = CTC_ObjCMessageReceiver;
      
  if (TypoCorrection Corrected = CorrectTypo(Result.getLookupNameInfo(),
                                             Result.getLookupKind(), S, NULL,
                                             NULL, false, CTC)) {
    if (NamedDecl *ND = Corrected.getCorrectionDecl()) {
      // If we found a declaration, correct when it refers to an Objective-C
      // class.
      if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(ND)) {
        Diag(NameLoc, diag::err_unknown_receiver_suggest)
          << Name << Corrected.getCorrection()
          << FixItHint::CreateReplacement(SourceRange(NameLoc),
                                          ND->getNameAsString());
        Diag(ND->getLocation(), diag::note_previous_decl)
          << Corrected.getCorrection();

        QualType T = Context.getObjCInterfaceType(Class);
        TypeSourceInfo *TSInfo = Context.getTrivialTypeSourceInfo(T, NameLoc);
        ReceiverType = CreateParsedType(T, TSInfo);
        return ObjCClassMessage;
      }
    } else if (Corrected.isKeyword() &&
               Corrected.getCorrectionAsIdentifierInfo()->isStr("super")) {
      // If we've found the keyword "super", this is a send to super.
      Diag(NameLoc, diag::err_unknown_receiver_suggest)
        << Name << Corrected.getCorrection()
        << FixItHint::CreateReplacement(SourceRange(NameLoc), "super");
      return ObjCSuperMessage;
    }
  }
  
  // Fall back: let the parser try to parse it as an instance message.
  return ObjCInstanceMessage;
}

ExprResult Sema::ActOnSuperMessage(Scope *S, 
                                   SourceLocation SuperLoc,
                                   Selector Sel,
                                   SourceLocation LBracLoc,
                                   SourceLocation SelectorLoc,
                                   SourceLocation RBracLoc,
                                   MultiExprArg Args) {
  // Determine whether we are inside a method or not.
  ObjCMethodDecl *Method = tryCaptureObjCSelf();
  if (!Method) {
    Diag(SuperLoc, diag::err_invalid_receiver_to_message_super);
    return ExprError();
  }

  ObjCInterfaceDecl *Class = Method->getClassInterface();
  if (!Class) {
    Diag(SuperLoc, diag::error_no_super_class_message)
      << Method->getDeclName();
    return ExprError();
  }

  ObjCInterfaceDecl *Super = Class->getSuperClass();
  if (!Super) {
    // The current class does not have a superclass.
    Diag(SuperLoc, diag::error_root_class_cannot_use_super)
      << Class->getIdentifier();
    return ExprError();
  }

  // We are in a method whose class has a superclass, so 'super'
  // is acting as a keyword.
  if (Method->isInstanceMethod()) {
    // Since we are in an instance method, this is an instance
    // message to the superclass instance.
    QualType SuperTy = Context.getObjCInterfaceType(Super);
    SuperTy = Context.getObjCObjectPointerType(SuperTy);
    return BuildInstanceMessage(0, SuperTy, SuperLoc,
                                Sel, /*Method=*/0,
                                LBracLoc, SelectorLoc, RBracLoc, move(Args));
  }
  
  // Since we are in a class method, this is a class message to
  // the superclass.
  return BuildClassMessage(/*ReceiverTypeInfo=*/0,
                           Context.getObjCInterfaceType(Super),
                           SuperLoc, Sel, /*Method=*/0,
                           LBracLoc, SelectorLoc, RBracLoc, move(Args));
}

/// \brief Build an Objective-C class message expression.
///
/// This routine takes care of both normal class messages and
/// class messages to the superclass.
///
/// \param ReceiverTypeInfo Type source information that describes the
/// receiver of this message. This may be NULL, in which case we are
/// sending to the superclass and \p SuperLoc must be a valid source
/// location.

/// \param ReceiverType The type of the object receiving the
/// message. When \p ReceiverTypeInfo is non-NULL, this is the same
/// type as that refers to. For a superclass send, this is the type of
/// the superclass.
///
/// \param SuperLoc The location of the "super" keyword in a
/// superclass message.
///
/// \param Sel The selector to which the message is being sent.
///
/// \param Method The method that this class message is invoking, if
/// already known.
///
/// \param LBracLoc The location of the opening square bracket ']'.
///
/// \param RBrac The location of the closing square bracket ']'.
///
/// \param Args The message arguments.
ExprResult Sema::BuildClassMessage(TypeSourceInfo *ReceiverTypeInfo,
                                   QualType ReceiverType,
                                   SourceLocation SuperLoc,
                                   Selector Sel,
                                   ObjCMethodDecl *Method,
                                   SourceLocation LBracLoc, 
                                   SourceLocation SelectorLoc,
                                   SourceLocation RBracLoc,
                                   MultiExprArg ArgsIn) {
  SourceLocation Loc = SuperLoc.isValid()? SuperLoc
    : ReceiverTypeInfo->getTypeLoc().getSourceRange().getBegin();
  if (LBracLoc.isInvalid()) {
    Diag(Loc, diag::err_missing_open_square_message_send)
      << FixItHint::CreateInsertion(Loc, "[");
    LBracLoc = Loc;
  }
  
  if (ReceiverType->isDependentType()) {
    // If the receiver type is dependent, we can't type-check anything
    // at this point. Build a dependent expression.
    unsigned NumArgs = ArgsIn.size();
    Expr **Args = reinterpret_cast<Expr **>(ArgsIn.release());
    assert(SuperLoc.isInvalid() && "Message to super with dependent type");
    return Owned(ObjCMessageExpr::Create(Context, ReceiverType,
                                         VK_RValue, LBracLoc, ReceiverTypeInfo,
                                         Sel, SelectorLoc, /*Method=*/0,
                                         Args, NumArgs, RBracLoc));
  }
  
  // Find the class to which we are sending this message.
  ObjCInterfaceDecl *Class = 0;
  const ObjCObjectType *ClassType = ReceiverType->getAs<ObjCObjectType>();
  if (!ClassType || !(Class = ClassType->getInterface())) {
    Diag(Loc, diag::err_invalid_receiver_class_message)
      << ReceiverType;
    return ExprError();
  }
  assert(Class && "We don't know which class we're messaging?");
  (void)DiagnoseUseOfDecl(Class, Loc);
  // Find the method we are messaging.
  if (!Method) {
    if (Class->isForwardDecl()) {
      if (getLangOptions().ObjCAutoRefCount) {
        Diag(Loc, diag::err_arc_receiver_forward_class) << ReceiverType;
      } else {
        Diag(Loc, diag::warn_receiver_forward_class) << Class->getDeclName();
      }

      // A forward class used in messaging is treated as a 'Class'
      Method = LookupFactoryMethodInGlobalPool(Sel, 
                                               SourceRange(LBracLoc, RBracLoc));
      if (Method && !getLangOptions().ObjCAutoRefCount)
        Diag(Method->getLocation(), diag::note_method_sent_forward_class)
          << Method->getDeclName();
    }
    if (!Method)
      Method = Class->lookupClassMethod(Sel);

    // If we have an implementation in scope, check "private" methods.
    if (!Method)
      Method = LookupPrivateClassMethod(Sel, Class);

    if (Method && DiagnoseUseOfDecl(Method, Loc))
      return ExprError();
  }

  // Check the argument types and determine the result type.
  QualType ReturnType;
  ExprValueKind VK = VK_RValue;

  unsigned NumArgs = ArgsIn.size();
  Expr **Args = reinterpret_cast<Expr **>(ArgsIn.release());
  if (CheckMessageArgumentTypes(ReceiverType, Args, NumArgs, Sel, Method, true,
                                SuperLoc.isValid(), LBracLoc, RBracLoc, 
                                ReturnType, VK))
    return ExprError();

  if (Method && !Method->getResultType()->isVoidType() &&
      RequireCompleteType(LBracLoc, Method->getResultType(), 
                          diag::err_illegal_message_expr_incomplete_type))
    return ExprError();

  // Construct the appropriate ObjCMessageExpr.
  Expr *Result;
  if (SuperLoc.isValid())
    Result = ObjCMessageExpr::Create(Context, ReturnType, VK, LBracLoc, 
                                     SuperLoc, /*IsInstanceSuper=*/false, 
                                     ReceiverType, Sel, SelectorLoc,
                                     Method, Args, NumArgs, RBracLoc);
  else
    Result = ObjCMessageExpr::Create(Context, ReturnType, VK, LBracLoc, 
                                     ReceiverTypeInfo, Sel, SelectorLoc,
                                     Method, Args, NumArgs, RBracLoc);
  return MaybeBindToTemporary(Result);
}

// ActOnClassMessage - used for both unary and keyword messages.
// ArgExprs is optional - if it is present, the number of expressions
// is obtained from Sel.getNumArgs().
ExprResult Sema::ActOnClassMessage(Scope *S, 
                                   ParsedType Receiver,
                                   Selector Sel,
                                   SourceLocation LBracLoc,
                                   SourceLocation SelectorLoc,
                                   SourceLocation RBracLoc,
                                   MultiExprArg Args) {
  TypeSourceInfo *ReceiverTypeInfo;
  QualType ReceiverType = GetTypeFromParser(Receiver, &ReceiverTypeInfo);
  if (ReceiverType.isNull())
    return ExprError();


  if (!ReceiverTypeInfo)
    ReceiverTypeInfo = Context.getTrivialTypeSourceInfo(ReceiverType, LBracLoc);

  return BuildClassMessage(ReceiverTypeInfo, ReceiverType, 
                           /*SuperLoc=*/SourceLocation(), Sel, /*Method=*/0,
                           LBracLoc, SelectorLoc, RBracLoc, move(Args));
}

/// \brief Build an Objective-C instance message expression.
///
/// This routine takes care of both normal instance messages and
/// instance messages to the superclass instance.
///
/// \param Receiver The expression that computes the object that will
/// receive this message. This may be empty, in which case we are
/// sending to the superclass instance and \p SuperLoc must be a valid
/// source location.
///
/// \param ReceiverType The (static) type of the object receiving the
/// message. When a \p Receiver expression is provided, this is the
/// same type as that expression. For a superclass instance send, this
/// is a pointer to the type of the superclass.
///
/// \param SuperLoc The location of the "super" keyword in a
/// superclass instance message.
///
/// \param Sel The selector to which the message is being sent.
///
/// \param Method The method that this instance message is invoking, if
/// already known.
///
/// \param LBracLoc The location of the opening square bracket ']'.
///
/// \param RBrac The location of the closing square bracket ']'.
///
/// \param Args The message arguments.
ExprResult Sema::BuildInstanceMessage(Expr *Receiver,
                                      QualType ReceiverType,
                                      SourceLocation SuperLoc,
                                      Selector Sel,
                                      ObjCMethodDecl *Method,
                                      SourceLocation LBracLoc, 
                                      SourceLocation SelectorLoc,
                                      SourceLocation RBracLoc,
                                      MultiExprArg ArgsIn) {
  // The location of the receiver.
  SourceLocation Loc = SuperLoc.isValid()? SuperLoc : Receiver->getLocStart();
  
  if (LBracLoc.isInvalid()) {
    Diag(Loc, diag::err_missing_open_square_message_send)
      << FixItHint::CreateInsertion(Loc, "[");
    LBracLoc = Loc;
  }

  // If we have a receiver expression, perform appropriate promotions
  // and determine receiver type.
  if (Receiver) {
    if (Receiver->isTypeDependent()) {
      // If the receiver is type-dependent, we can't type-check anything
      // at this point. Build a dependent expression.
      unsigned NumArgs = ArgsIn.size();
      Expr **Args = reinterpret_cast<Expr **>(ArgsIn.release());
      assert(SuperLoc.isInvalid() && "Message to super with dependent type");
      return Owned(ObjCMessageExpr::Create(Context, Context.DependentTy,
                                           VK_RValue, LBracLoc, Receiver, Sel, 
                                           SelectorLoc, /*Method=*/0,
                                           Args, NumArgs, RBracLoc));
    }

    // If necessary, apply function/array conversion to the receiver.
    // C99 6.7.5.3p[7,8].
    ExprResult Result = DefaultFunctionArrayLvalueConversion(Receiver);
    if (Result.isInvalid())
      return ExprError();
    Receiver = Result.take();
    ReceiverType = Receiver->getType();
  }

  if (!Method) {
    // Handle messages to id.
    bool receiverIsId = ReceiverType->isObjCIdType();
    if (receiverIsId || ReceiverType->isBlockPointerType() ||
        (Receiver && Context.isObjCNSObjectType(Receiver->getType()))) {
      Method = LookupInstanceMethodInGlobalPool(Sel, 
                                                SourceRange(LBracLoc, RBracLoc),
                                                receiverIsId);
      if (!Method)
        Method = LookupFactoryMethodInGlobalPool(Sel, 
                                                 SourceRange(LBracLoc, RBracLoc),
                                                 receiverIsId);
    } else if (ReceiverType->isObjCClassType() ||
               ReceiverType->isObjCQualifiedClassType()) {
      // Handle messages to Class.
      // We allow sending a message to a qualified Class ("Class<foo>"), which 
      // is ok as long as one of the protocols implements the selector (if not, warn).
      if (const ObjCObjectPointerType *QClassTy 
            = ReceiverType->getAsObjCQualifiedClassType()) {
        // Search protocols for class methods.
        Method = LookupMethodInQualifiedType(Sel, QClassTy, false);
        if (!Method) {
          Method = LookupMethodInQualifiedType(Sel, QClassTy, true);
          // warn if instance method found for a Class message.
          if (Method) {
            Diag(Loc, diag::warn_instance_method_on_class_found)
              << Method->getSelector() << Sel;
            Diag(Method->getLocation(), diag::note_method_declared_at);
          }
        }
      } else {
        if (ObjCMethodDecl *CurMeth = getCurMethodDecl()) {
          if (ObjCInterfaceDecl *ClassDecl = CurMeth->getClassInterface()) {
            // First check the public methods in the class interface.
            Method = ClassDecl->lookupClassMethod(Sel);

            if (!Method)
              Method = LookupPrivateClassMethod(Sel, ClassDecl);
          }
          if (Method && DiagnoseUseOfDecl(Method, Loc))
            return ExprError();
        }
        if (!Method) {
          // If not messaging 'self', look for any factory method named 'Sel'.
          if (!Receiver || !isSelfExpr(Receiver)) {
            Method = LookupFactoryMethodInGlobalPool(Sel, 
                                                SourceRange(LBracLoc, RBracLoc),
                                                     true);
            if (!Method) {
              // If no class (factory) method was found, check if an _instance_
              // method of the same name exists in the root class only.
              Method = LookupInstanceMethodInGlobalPool(Sel,
                                               SourceRange(LBracLoc, RBracLoc),
                                                        true);
              if (Method)
                  if (const ObjCInterfaceDecl *ID =
                      dyn_cast<ObjCInterfaceDecl>(Method->getDeclContext())) {
                    if (ID->getSuperClass())
                      Diag(Loc, diag::warn_root_inst_method_not_found)
                      << Sel << SourceRange(LBracLoc, RBracLoc);
                  }
            }
          }
        }
      }
    } else {
      ObjCInterfaceDecl* ClassDecl = 0;

      // We allow sending a message to a qualified ID ("id<foo>"), which is ok as
      // long as one of the protocols implements the selector (if not, warn).
      if (const ObjCObjectPointerType *QIdTy 
                                   = ReceiverType->getAsObjCQualifiedIdType()) {
        // Search protocols for instance methods.
        Method = LookupMethodInQualifiedType(Sel, QIdTy, true);
        if (!Method)
          Method = LookupMethodInQualifiedType(Sel, QIdTy, false);
      } else if (const ObjCObjectPointerType *OCIType
                   = ReceiverType->getAsObjCInterfacePointerType()) {
        // We allow sending a message to a pointer to an interface (an object).
        ClassDecl = OCIType->getInterfaceDecl();

        if (ClassDecl->isForwardDecl() && getLangOptions().ObjCAutoRefCount) {
          Diag(Loc, diag::err_arc_receiver_forward_instance)
            << OCIType->getPointeeType()
            << (Receiver ? Receiver->getSourceRange() : SourceRange(SuperLoc));
          return ExprError();
        }

        // FIXME: consider using LookupInstanceMethodInGlobalPool, since it will be
        // faster than the following method (which can do *many* linear searches).
        // The idea is to add class info to MethodPool.
        Method = ClassDecl->lookupInstanceMethod(Sel);

        if (!Method)
          // Search protocol qualifiers.
          Method = LookupMethodInQualifiedType(Sel, OCIType, true);
        
        const ObjCInterfaceDecl *forwardClass = 0;
        if (!Method) {
          // If we have implementations in scope, check "private" methods.
          Method = LookupPrivateInstanceMethod(Sel, ClassDecl);

          if (!Method && getLangOptions().ObjCAutoRefCount) {
            Diag(Loc, diag::err_arc_may_not_respond)
              << OCIType->getPointeeType() << Sel;
            return ExprError();
          }

          if (!Method && (!Receiver || !isSelfExpr(Receiver))) {
            // If we still haven't found a method, look in the global pool. This
            // behavior isn't very desirable, however we need it for GCC
            // compatibility. FIXME: should we deviate??
            if (OCIType->qual_empty()) {
              Method = LookupInstanceMethodInGlobalPool(Sel,
                                                 SourceRange(LBracLoc, RBracLoc));
              if (OCIType->getInterfaceDecl()->isForwardDecl())
                forwardClass = OCIType->getInterfaceDecl();
              if (Method && !forwardClass)
                Diag(Loc, diag::warn_maynot_respond)
                  << OCIType->getInterfaceDecl()->getIdentifier() << Sel;
            }
          }
        }
        if (Method && DiagnoseUseOfDecl(Method, Loc, forwardClass))
          return ExprError();
      } else if (!getLangOptions().ObjCAutoRefCount &&
                 !Context.getObjCIdType().isNull() &&
                 (ReceiverType->isPointerType() || 
                  ReceiverType->isIntegerType())) {
        // Implicitly convert integers and pointers to 'id' but emit a warning.
        // But not in ARC.
        Diag(Loc, diag::warn_bad_receiver_type)
          << ReceiverType 
          << Receiver->getSourceRange();
        if (ReceiverType->isPointerType())
          Receiver = ImpCastExprToType(Receiver, Context.getObjCIdType(), 
                            CK_BitCast).take();
        else {
          // TODO: specialized warning on null receivers?
          bool IsNull = Receiver->isNullPointerConstant(Context,
                                              Expr::NPC_ValueDependentIsNull);
          Receiver = ImpCastExprToType(Receiver, Context.getObjCIdType(),
                            IsNull ? CK_NullToPointer : CK_IntegralToPointer).take();
        }
        ReceiverType = Receiver->getType();
      } 
      else {
        ExprResult ReceiverRes;
        if (getLangOptions().CPlusPlus)
          ReceiverRes = PerformContextuallyConvertToObjCId(Receiver);
        if (ReceiverRes.isUsable()) {
          Receiver = ReceiverRes.take();
          if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(Receiver)) {
            Receiver = ICE->getSubExpr();
            ReceiverType = Receiver->getType();
          }
          return BuildInstanceMessage(Receiver,
                                      ReceiverType,
                                      SuperLoc,
                                      Sel,
                                      Method,
                                      LBracLoc,
                                      SelectorLoc,
                                      RBracLoc,
                                      move(ArgsIn));
        } else {
          // Reject other random receiver types (e.g. structs).
          Diag(Loc, diag::err_bad_receiver_type)
            << ReceiverType << Receiver->getSourceRange();
          return ExprError();
        }
      }
    }
  }

  // Check the message arguments.
  unsigned NumArgs = ArgsIn.size();
  Expr **Args = reinterpret_cast<Expr **>(ArgsIn.release());
  QualType ReturnType;
  ExprValueKind VK = VK_RValue;
  bool ClassMessage = (ReceiverType->isObjCClassType() ||
                       ReceiverType->isObjCQualifiedClassType());
  if (CheckMessageArgumentTypes(ReceiverType, Args, NumArgs, Sel, Method, 
                                ClassMessage, SuperLoc.isValid(), 
                                LBracLoc, RBracLoc, ReturnType, VK))
    return ExprError();
  
  if (Method && !Method->getResultType()->isVoidType() &&
      RequireCompleteType(LBracLoc, Method->getResultType(), 
                          diag::err_illegal_message_expr_incomplete_type))
    return ExprError();

  // In ARC, forbid the user from sending messages to 
  // retain/release/autorelease/dealloc/retainCount explicitly.
  if (getLangOptions().ObjCAutoRefCount) {
    ObjCMethodFamily family =
      (Method ? Method->getMethodFamily() : Sel.getMethodFamily());
    switch (family) {
    case OMF_init:
      if (Method)
        checkInitMethod(Method, ReceiverType);

    case OMF_None:
    case OMF_alloc:
    case OMF_copy:
    case OMF_mutableCopy:
    case OMF_new:
    case OMF_self:
      break;

    case OMF_dealloc:
    case OMF_retain:
    case OMF_release:
    case OMF_autorelease:
    case OMF_retainCount:
      Diag(Loc, diag::err_arc_illegal_explicit_message)
        << Sel << SelectorLoc;
      break;
    }
  }

  // Construct the appropriate ObjCMessageExpr instance.
  ObjCMessageExpr *Result;
  if (SuperLoc.isValid())
    Result = ObjCMessageExpr::Create(Context, ReturnType, VK, LBracLoc,
                                     SuperLoc,  /*IsInstanceSuper=*/true,
                                     ReceiverType, Sel, SelectorLoc, Method, 
                                     Args, NumArgs, RBracLoc);
  else
    Result = ObjCMessageExpr::Create(Context, ReturnType, VK, LBracLoc,
                                     Receiver, Sel, SelectorLoc, Method,
                                     Args, NumArgs, RBracLoc);

  if (getLangOptions().ObjCAutoRefCount) {
    // In ARC, annotate delegate init calls.
    if (Result->getMethodFamily() == OMF_init &&
        (SuperLoc.isValid() || isSelfExpr(Receiver))) {
      // Only consider init calls *directly* in init implementations,
      // not within blocks.
      ObjCMethodDecl *method = dyn_cast<ObjCMethodDecl>(CurContext);
      if (method && method->getMethodFamily() == OMF_init) {
        // The implicit assignment to self means we also don't want to
        // consume the result.
        Result->setDelegateInitCall(true);
        return Owned(Result);
      }
    }

    // In ARC, check for message sends which are likely to introduce
    // retain cycles.
    checkRetainCycles(Result);
  }
      
  return MaybeBindToTemporary(Result);
}

// ActOnInstanceMessage - used for both unary and keyword messages.
// ArgExprs is optional - if it is present, the number of expressions
// is obtained from Sel.getNumArgs().
ExprResult Sema::ActOnInstanceMessage(Scope *S,
                                      Expr *Receiver, 
                                      Selector Sel,
                                      SourceLocation LBracLoc,
                                      SourceLocation SelectorLoc,
                                      SourceLocation RBracLoc,
                                      MultiExprArg Args) {
  if (!Receiver)
    return ExprError();

  return BuildInstanceMessage(Receiver, Receiver->getType(),
                              /*SuperLoc=*/SourceLocation(), Sel, /*Method=*/0, 
                              LBracLoc, SelectorLoc, RBracLoc, move(Args));
}

enum ARCConversionTypeClass {
  ACTC_none,
  ACTC_retainable,
  ACTC_indirectRetainable
};
static ARCConversionTypeClass classifyTypeForARCConversion(QualType type) {
  ARCConversionTypeClass ACTC = ACTC_retainable;
  
  // Ignore an outermost reference type.
  if (const ReferenceType *ref = type->getAs<ReferenceType>())
    type = ref->getPointeeType();
  
  // Drill through pointers and arrays recursively.
  while (true) {
    if (const PointerType *ptr = type->getAs<PointerType>()) {
      type = ptr->getPointeeType();
    } else if (const ArrayType *array = type->getAsArrayTypeUnsafe()) {
      type = QualType(array->getElementType()->getBaseElementTypeUnsafe(), 0);
    } else {
      break;
    }
    ACTC = ACTC_indirectRetainable;
  }
  
  if (!type->isObjCRetainableType()) return ACTC_none;
  return ACTC;
}

namespace {
  /// Return true if the given expression can be reasonably converted
  /// between a retainable pointer type and a C pointer type.
  struct ARCCastChecker : StmtVisitor<ARCCastChecker, bool> {
    ASTContext &Context;
    ARCCastChecker(ASTContext &Context) : Context(Context) {}
    bool VisitStmt(Stmt *s) {
      return false;
    }
    bool VisitExpr(Expr *e) {
      return e->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull);
    }
    
    bool VisitParenExpr(ParenExpr *e) {
      return Visit(e->getSubExpr());
    }
    bool VisitCastExpr(CastExpr *e) {
      switch (e->getCastKind()) {
        case CK_NullToPointer:
          return true;
        case CK_NoOp:
        case CK_LValueToRValue:
        case CK_BitCast:
        case CK_AnyPointerToObjCPointerCast:
        case CK_AnyPointerToBlockPointerCast:
          return Visit(e->getSubExpr());
        default:
          return false;
      }
    }
    bool VisitUnaryExtension(UnaryOperator *e) {
      return Visit(e->getSubExpr());
    }
    bool VisitBinComma(BinaryOperator *e) {
      return Visit(e->getRHS());
    }
    bool VisitConditionalOperator(ConditionalOperator *e) {
      // Conditional operators are okay if both sides are okay.
      return Visit(e->getTrueExpr()) && Visit(e->getFalseExpr());
    }
    bool VisitObjCStringLiteral(ObjCStringLiteral *e) {
      // Always white-list Objective-C string literals.
      return true;
    }
    bool VisitStmtExpr(StmtExpr *e) {
      return Visit(e->getSubStmt()->body_back());
    }
    bool VisitDeclRefExpr(DeclRefExpr *e) {
      // White-list references to global extern strings from system
      // headers.
      if (VarDecl *var = dyn_cast<VarDecl>(e->getDecl()))
        if (var->getStorageClass() == SC_Extern &&
            var->getType().isConstQualified() &&
            Context.getSourceManager().isInSystemHeader(var->getLocation()))
          return true;
      return false;
    }
  };
}

bool
Sema::ValidObjCARCNoBridgeCastExpr(Expr *&Exp, QualType castType) {
  Expr *NewExp = Exp->IgnoreParenCasts();
  
  if (!isa<ObjCMessageExpr>(NewExp) && !isa<ObjCPropertyRefExpr>(NewExp)
      && !isa<CallExpr>(NewExp))
    return false;
  ObjCMethodDecl *method = 0;
  bool MethodReturnsPlusOne = false;
  
  if (ObjCPropertyRefExpr *PRE = dyn_cast<ObjCPropertyRefExpr>(NewExp)) {
    method = PRE->getExplicitProperty()->getGetterMethodDecl();
  }
  else if (ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(NewExp))
    method = ME->getMethodDecl();
  else {
    CallExpr *CE = cast<CallExpr>(NewExp);
    Decl *CallDecl = CE->getCalleeDecl();
    if (!CallDecl)
      return false;
    if (CallDecl->hasAttr<CFReturnsNotRetainedAttr>())
      return true;
    MethodReturnsPlusOne = CallDecl->hasAttr<CFReturnsRetainedAttr>();
    if (!MethodReturnsPlusOne) {
      if (NamedDecl *ND = dyn_cast<NamedDecl>(CallDecl))
        if (const IdentifierInfo *Id = ND->getIdentifier())
          if (Id->isStr("__builtin___CFStringMakeConstantString"))
            return true;
    }
  }
  
  if (!MethodReturnsPlusOne) {
    if (!method)
      return false;
    if (method->hasAttr<CFReturnsNotRetainedAttr>())
      return true;
    MethodReturnsPlusOne = method->hasAttr<CFReturnsRetainedAttr>();
    if (!MethodReturnsPlusOne) {
      ObjCMethodFamily family = method->getSelector().getMethodFamily();
      switch (family) {
        case OMF_alloc:
        case OMF_copy:
        case OMF_mutableCopy:
        case OMF_new:
          MethodReturnsPlusOne = true;
          break;
        default:
          break;
      }
    }
  }
  
  if (MethodReturnsPlusOne) {
    TypeSourceInfo *TSInfo = 
      Context.getTrivialTypeSourceInfo(castType, SourceLocation());
    ExprResult ExpRes = BuildObjCBridgedCast(SourceLocation(), OBC_BridgeTransfer,
                                             SourceLocation(), TSInfo, Exp);
    Exp = ExpRes.take();
  }
  return true;
}

void 
Sema::CheckObjCARCConversion(SourceRange castRange, QualType castType,
                             Expr *&castExpr, CheckedConversionKind CCK) {
  QualType castExprType = castExpr->getType();
  
  ARCConversionTypeClass exprACTC = classifyTypeForARCConversion(castExprType);
  ARCConversionTypeClass castACTC = classifyTypeForARCConversion(castType);
  if (exprACTC == castACTC) return;
  if (exprACTC && castType->isIntegralType(Context)) return;
  
  // Allow casts between pointers to lifetime types (e.g., __strong id*)
  // and pointers to void (e.g., cv void *). Casting from void* to lifetime*
  // must be explicit.
  if (const PointerType *CastPtr = castType->getAs<PointerType>()) {
    if (const PointerType *CastExprPtr = castExprType->getAs<PointerType>()) {
      QualType CastPointee = CastPtr->getPointeeType();
      QualType CastExprPointee = CastExprPtr->getPointeeType();
      if ((CCK != CCK_ImplicitConversion && 
           CastPointee->isObjCIndirectLifetimeType() && 
           CastExprPointee->isVoidType()) ||
          (CastPointee->isVoidType() && 
           CastExprPointee->isObjCIndirectLifetimeType()))
        return;
    }
  }
  
  if (ARCCastChecker(Context).Visit(castExpr))
    return;
  
  SourceLocation loc =
  (castRange.isValid() ? castRange.getBegin() : castExpr->getExprLoc());
  
  if (makeUnavailableInSystemHeader(loc,
                                    "converts between Objective-C and C pointers in -fobjc-arc"))
    return;
  
  unsigned srcKind = 0;
  switch (exprACTC) {
    case ACTC_none:
      srcKind = (castExprType->isPointerType() ? 1 : 0);
      break;
    case ACTC_retainable:
      srcKind = (castExprType->isBlockPointerType() ? 2 : 3);
      break;
    case ACTC_indirectRetainable:
      srcKind = 4;
      break;
  }
  
  if (CCK == CCK_CStyleCast) {
    // Check whether this could be fixed with a bridge cast.
    SourceLocation AfterLParen = PP.getLocForEndOfToken(castRange.getBegin());
    SourceLocation NoteLoc = AfterLParen.isValid()? AfterLParen : loc;
    
    if (castType->isObjCARCBridgableType() && 
        castExprType->isCARCBridgableType()) {
      // explicit unbridged casts are allowed if the source of the cast is a 
      // message sent to an objc method (or property access)
      if (ValidObjCARCNoBridgeCastExpr(castExpr, castType))
        return;
      Diag(loc, diag::err_arc_cast_requires_bridge)
        << 2
        << castExprType
        << (castType->isBlockPointerType()? 1 : 0)
        << castType
        << castRange
        << castExpr->getSourceRange();
      Diag(NoteLoc, diag::note_arc_bridge)
        << FixItHint::CreateInsertion(AfterLParen, "__bridge ");
      Diag(NoteLoc, diag::note_arc_bridge_transfer)
        << castExprType
        << FixItHint::CreateInsertion(AfterLParen, "__bridge_transfer ");
      
      return;
    }
    
    if (castType->isCARCBridgableType() && 
        castExprType->isObjCARCBridgableType()){
      Diag(loc, diag::err_arc_cast_requires_bridge)
        << (castExprType->isBlockPointerType()? 1 : 0)
        << castExprType
        << 2
        << castType
        << castRange
        << castExpr->getSourceRange();

      Diag(NoteLoc, diag::note_arc_bridge)
        << FixItHint::CreateInsertion(AfterLParen, "__bridge ");
      Diag(NoteLoc, diag::note_arc_bridge_retained)
        << castType
        << FixItHint::CreateInsertion(AfterLParen, "__bridge_retained ");
      return;
    }
  }
  
  Diag(loc, diag::err_arc_mismatched_cast)
    << (CCK != CCK_ImplicitConversion) << srcKind << castExprType << castType
    << castRange << castExpr->getSourceRange();
}

ExprResult Sema::BuildObjCBridgedCast(SourceLocation LParenLoc,
                                      ObjCBridgeCastKind Kind,
                                      SourceLocation BridgeKeywordLoc,
                                      TypeSourceInfo *TSInfo,
                                      Expr *SubExpr) {
  QualType T = TSInfo->getType();
  QualType FromType = SubExpr->getType();

  bool MustConsume = false;
  if (T->isDependentType() || SubExpr->isTypeDependent()) {
    // Okay: we'll build a dependent expression type.
  } else if (T->isObjCARCBridgableType() && FromType->isCARCBridgableType()) {
    // Casting CF -> id
    switch (Kind) {
    case OBC_Bridge:
      break;
      
    case OBC_BridgeRetained:
      Diag(BridgeKeywordLoc, diag::err_arc_bridge_cast_wrong_kind)
        << 2
        << FromType
        << (T->isBlockPointerType()? 1 : 0)
        << T
        << SubExpr->getSourceRange()
        << Kind;
      Diag(BridgeKeywordLoc, diag::note_arc_bridge)
        << FixItHint::CreateReplacement(BridgeKeywordLoc, "__bridge");
      Diag(BridgeKeywordLoc, diag::note_arc_bridge_transfer)
        << FromType
        << FixItHint::CreateReplacement(BridgeKeywordLoc, 
                                        "__bridge_transfer ");

      Kind = OBC_Bridge;
      break;
      
    case OBC_BridgeTransfer:
      // We must consume the Objective-C object produced by the cast.
      MustConsume = true;
      break;
    }
  } else if (T->isCARCBridgableType() && FromType->isObjCARCBridgableType()) {
    // Okay: id -> CF
    switch (Kind) {
    case OBC_Bridge:
      break;
      
    case OBC_BridgeRetained:        
      // Produce the object before casting it.
      SubExpr = ImplicitCastExpr::Create(Context, FromType,
                                         CK_ObjCProduceObject,
                                         SubExpr, 0, VK_RValue);
      break;
      
    case OBC_BridgeTransfer:
      Diag(BridgeKeywordLoc, diag::err_arc_bridge_cast_wrong_kind)
        << (FromType->isBlockPointerType()? 1 : 0)
        << FromType
        << 2
        << T
        << SubExpr->getSourceRange()
        << Kind;
        
      Diag(BridgeKeywordLoc, diag::note_arc_bridge)
        << FixItHint::CreateReplacement(BridgeKeywordLoc, "__bridge ");
      Diag(BridgeKeywordLoc, diag::note_arc_bridge_retained)
        << T
        << FixItHint::CreateReplacement(BridgeKeywordLoc, "__bridge_retained ");
        
      Kind = OBC_Bridge;
      break;
    }
  } else {
    Diag(LParenLoc, diag::err_arc_bridge_cast_incompatible)
      << FromType << T << Kind
      << SubExpr->getSourceRange()
      << TSInfo->getTypeLoc().getSourceRange();
    return ExprError();
  }

  Expr *Result = new (Context) ObjCBridgedCastExpr(LParenLoc, Kind, 
                                                   BridgeKeywordLoc,
                                                   TSInfo, SubExpr);
  
  if (MustConsume) {
    ExprNeedsCleanups = true;
    Result = ImplicitCastExpr::Create(Context, T, CK_ObjCConsumeObject, Result, 
                                      0, VK_RValue);    
  }
  
  return Result;
}

ExprResult Sema::ActOnObjCBridgedCast(Scope *S,
                                      SourceLocation LParenLoc,
                                      ObjCBridgeCastKind Kind,
                                      SourceLocation BridgeKeywordLoc,
                                      ParsedType Type,
                                      SourceLocation RParenLoc,
                                      Expr *SubExpr) {
  TypeSourceInfo *TSInfo = 0;
  QualType T = GetTypeFromParser(Type, &TSInfo);
  if (!TSInfo)
    TSInfo = Context.getTrivialTypeSourceInfo(T, LParenLoc);
  return BuildObjCBridgedCast(LParenLoc, Kind, BridgeKeywordLoc, TSInfo, 
                              SubExpr);
}
