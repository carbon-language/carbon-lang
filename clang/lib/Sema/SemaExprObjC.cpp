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
#include "clang/Sema/Initialization.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/TypeLoc.h"
#include "llvm/ADT/SmallString.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang;

Sema::ExprResult Sema::ParseObjCStringLiteral(SourceLocation *AtLocs,
                                              ExprTy **strings,
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
    S = StringLiteral::Create(Context, &StrBuf[0], StrBuf.size(), false,
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
    IdentifierInfo *NSIdent = &Context.Idents.get("NSConstantString");
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

Expr *Sema::BuildObjCEncodeExpression(SourceLocation AtLoc,
                                      TypeSourceInfo *EncodedTypeInfo,
                                      SourceLocation RParenLoc) {
  QualType EncodedType = EncodedTypeInfo->getType();
  QualType StrTy;
  if (EncodedType->isDependentType())
    StrTy = Context.DependentTy;
  else {
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

Sema::ExprResult Sema::ParseObjCEncodeExpression(SourceLocation AtLoc,
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

Sema::ExprResult Sema::ParseObjCSelectorExpression(Selector Sel,
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

  QualType Ty = Context.getObjCSelType();
  return new (Context) ObjCSelectorExpr(Ty, Sel, AtLoc, RParenLoc);
}

Sema::ExprResult Sema::ParseObjCProtocolExpression(IdentifierInfo *ProtocolId,
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

bool Sema::CheckMessageArgumentTypes(Expr **Args, unsigned NumArgs,
                                     Selector Sel, ObjCMethodDecl *Method,
                                     bool isClassMessage,
                                     SourceLocation lbrac, SourceLocation rbrac,
                                     QualType &ReturnType) {
  if (!Method) {
    // Apply default argument promotion as for (C99 6.5.2.2p6).
    for (unsigned i = 0; i != NumArgs; i++) {
      if (Args[i]->isTypeDependent())
        continue;

      DefaultArgumentPromotion(Args[i]);
    }

    unsigned DiagID = isClassMessage ? diag::warn_class_method_not_found :
                                       diag::warn_inst_method_not_found;
    Diag(lbrac, DiagID)
      << Sel << isClassMessage << SourceRange(lbrac, rbrac);
    ReturnType = Context.getObjCIdType();
    return false;
  }

  ReturnType = Method->getSendResultType();

  unsigned NumNamedArgs = Sel.getNumArgs();
  // Method might have more arguments than selector indicates. This is due
  // to addition of c-style arguments in method.
  if (Method->param_size() > Sel.getNumArgs())
    NumNamedArgs = Method->param_size();
  // FIXME. This need be cleaned up.
  if (NumArgs < NumNamedArgs) {
    Diag(lbrac, diag::err_typecheck_call_too_few_args) << 2
    << NumNamedArgs << NumArgs;
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

    InitializedEntity Entity = InitializedEntity::InitializeParameter(Param);
    ExprResult ArgE = PerformCopyInitialization(Entity,
                                                      SourceLocation(),
                                                      Owned(argExpr->Retain()));
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

      IsError |= DefaultVariadicArgumentPromotion(Args[i], VariadicMethod, 0);
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

  DiagnoseSentinelCalls(Method, lbrac, Args, NumArgs);
  return IsError;
}

bool Sema::isSelfExpr(Expr *RExpr) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(RExpr))
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

/// HandleExprPropertyRefExpr - Handle foo.bar where foo is a pointer to an
/// objective C interface.  This is a property reference expression.
ExprResult Sema::
HandleExprPropertyRefExpr(const ObjCObjectPointerType *OPT,
                          Expr *BaseExpr, DeclarationName MemberName,
                          SourceLocation MemberLoc) {
  const ObjCInterfaceType *IFaceT = OPT->getInterfaceType();
  ObjCInterfaceDecl *IFace = IFaceT->getDecl();
  IdentifierInfo *Member = MemberName.getAsIdentifierInfo();

  // Search for a declared property first.
  if (ObjCPropertyDecl *PD = IFace->FindPropertyDeclaration(Member)) {
    // Check whether we can reference this property.
    if (DiagnoseUseOfDecl(PD, MemberLoc))
      return ExprError();
    QualType ResTy = PD->getType();
    Selector Sel = PP.getSelectorTable().getNullarySelector(Member);
    ObjCMethodDecl *Getter = IFace->lookupInstanceMethod(Sel);
    if (DiagnosePropertyAccessorMismatch(PD, Getter, MemberLoc))
      ResTy = Getter->getSendResultType();
    return Owned(new (Context) ObjCPropertyRefExpr(PD, ResTy,
                                                   MemberLoc, BaseExpr));
  }
  // Check protocols on qualified interfaces.
  for (ObjCObjectPointerType::qual_iterator I = OPT->qual_begin(),
       E = OPT->qual_end(); I != E; ++I)
    if (ObjCPropertyDecl *PD = (*I)->FindPropertyDeclaration(Member)) {
      // Check whether we can reference this property.
      if (DiagnoseUseOfDecl(PD, MemberLoc))
        return ExprError();

      return Owned(new (Context) ObjCPropertyRefExpr(PD, PD->getType(),
                                                     MemberLoc, BaseExpr));
    }
  // If that failed, look for an "implicit" property by seeing if the nullary
  // selector is implemented.

  // FIXME: The logic for looking up nullary and unary selectors should be
  // shared with the code in ActOnInstanceMessage.

  Selector Sel = PP.getSelectorTable().getNullarySelector(Member);
  ObjCMethodDecl *Getter = IFace->lookupInstanceMethod(Sel);

  // If this reference is in an @implementation, check for 'private' methods.
  if (!Getter)
    Getter = IFace->lookupPrivateInstanceMethod(Sel);

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
  if (!Setter) {
    // If this reference is in an @implementation, also check for 'private'
    // methods.
    Setter = IFace->lookupPrivateInstanceMethod(SetterSel);
  }
  // Look through local category implementations associated with the class.
  if (!Setter)
    Setter = IFace->getCategoryInstanceMethod(SetterSel);

  if (Setter && DiagnoseUseOfDecl(Setter, MemberLoc))
    return ExprError();

  if (Getter) {
    QualType PType;
    PType = Getter->getSendResultType();
    return Owned(new (Context) ObjCImplicitSetterGetterRefExpr(Getter, PType,
                                    Setter, MemberLoc, BaseExpr));
  }

  // Attempt to correct for typos in property names.
  LookupResult Res(*this, MemberName, MemberLoc, LookupOrdinaryName);
  if (CorrectTypo(Res, 0, 0, IFace, false, CTC_NoKeywords, OPT) &&
      Res.getAsSingle<ObjCPropertyDecl>()) {
    DeclarationName TypoResult = Res.getLookupName();
    Diag(MemberLoc, diag::err_property_not_found_suggest)
      << MemberName << QualType(OPT, 0) << TypoResult
      << FixItHint::CreateReplacement(MemberLoc, TypoResult.getAsString());
    ObjCPropertyDecl *Property = Res.getAsSingle<ObjCPropertyDecl>();
    Diag(Property->getLocation(), diag::note_previous_decl)
      << Property->getDeclName();
    return HandleExprPropertyRefExpr(OPT, BaseExpr, TypoResult, MemberLoc);
  }
  
  Diag(MemberLoc, diag::err_property_not_found)
    << MemberName << QualType(OPT, 0);
  if (Setter && !Getter)
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
  if (IFace == 0) {
    // If the "receiver" is 'super' in a method, handle it as an expression-like
    // property reference.
    if (ObjCMethodDecl *CurMethod = getCurMethodDecl())
      if (receiverNamePtr->isStr("super")) {
        if (CurMethod->isInstanceMethod()) {
          QualType T = 
            Context.getObjCInterfaceType(CurMethod->getClassInterface());
          T = Context.getObjCObjectPointerType(T);
          Expr *SuperExpr = new (Context) ObjCSuperExpr(receiverNameLoc, T);
        
          return HandleExprPropertyRefExpr(T->getAsObjCInterfacePointerType(),
                                           SuperExpr, &propertyName,
                                           propertyNameLoc);
        }

        // Otherwise, if this is a class method, try dispatching to our
        // superclass.
        IFace = CurMethod->getClassInterface()->getSuperClass();
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

    if (Getter)
      PType = Getter->getSendResultType();
    else {
      for (ObjCMethodDecl::param_iterator PI = Setter->param_begin(),
           E = Setter->param_end(); PI != E; ++PI)
        PType = (*PI)->getType();
    }
    return Owned(new (Context) ObjCImplicitSetterGetterRefExpr(
                                  Getter, PType, Setter,
                                  propertyNameLoc, IFace, receiverNameLoc));
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
  // messaging super.
  if (IsSuper && !HasTrailingDot && S->isInObjcMethodScope())
    return ObjCSuperMessage;
  
  LookupResult Result(*this, Name, NameLoc, LookupOrdinaryName);
  LookupName(Result, S);
  
  switch (Result.getResultKind()) {
  case LookupResult::NotFound:
    // Normal name lookup didn't find anything. If we're in an
    // Objective-C method, look for ivars. If we find one, we're done!
    // FIXME: This is a hack. Ivar lookup should be part of normal lookup.
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
      
  if (DeclarationName Corrected = CorrectTypo(Result, S, 0, 0, false, CTC)) {
    if (Result.isSingleResult()) {
      // If we found a declaration, correct when it refers to an Objective-C
      // class.
      NamedDecl *ND = Result.getFoundDecl();
      if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(ND)) {
        Diag(NameLoc, diag::err_unknown_receiver_suggest)
          << Name << Result.getLookupName()
          << FixItHint::CreateReplacement(SourceRange(NameLoc),
                                          ND->getNameAsString());
        Diag(ND->getLocation(), diag::note_previous_decl)
          << Corrected;

        QualType T = Context.getObjCInterfaceType(Class);
        TypeSourceInfo *TSInfo = Context.getTrivialTypeSourceInfo(T, NameLoc);
        ReceiverType = CreateParsedType(T, TSInfo);
        return ObjCClassMessage;
      }
    } else if (Result.empty() && Corrected.getAsIdentifierInfo() &&
               Corrected.getAsIdentifierInfo()->isStr("super")) {
      // If we've found the keyword "super", this is a send to super.
      Diag(NameLoc, diag::err_unknown_receiver_suggest)
        << Name << Corrected
        << FixItHint::CreateReplacement(SourceRange(NameLoc), "super");
      Name = Corrected.getAsIdentifierInfo();
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
  ObjCMethodDecl *Method = getCurMethodDecl();
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
    Diag(SuperLoc, diag::error_no_super_class) << Class->getIdentifier();
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
                                Sel, /*Method=*/0, LBracLoc, RBracLoc, 
                                move(Args));
  }
  
  // Since we are in a class method, this is a class message to
  // the superclass.
  return BuildClassMessage(/*ReceiverTypeInfo=*/0,
                           Context.getObjCInterfaceType(Super),
                           SuperLoc, Sel, /*Method=*/0, LBracLoc, RBracLoc, 
                           move(Args));
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
                                               SourceLocation RBracLoc,
                                               MultiExprArg ArgsIn) {
  if (ReceiverType->isDependentType()) {
    // If the receiver type is dependent, we can't type-check anything
    // at this point. Build a dependent expression.
    unsigned NumArgs = ArgsIn.size();
    Expr **Args = reinterpret_cast<Expr **>(ArgsIn.release());
    assert(SuperLoc.isInvalid() && "Message to super with dependent type");
    return Owned(ObjCMessageExpr::Create(Context, ReceiverType, LBracLoc,
                                         ReceiverTypeInfo, Sel, /*Method=*/0, 
                                         Args, NumArgs, RBracLoc));
  }
  
  SourceLocation Loc = SuperLoc.isValid()? SuperLoc
             : ReceiverTypeInfo->getTypeLoc().getLocalSourceRange().getBegin();

  // Find the class to which we are sending this message.
  ObjCInterfaceDecl *Class = 0;
  const ObjCObjectType *ClassType = ReceiverType->getAs<ObjCObjectType>();
  if (!ClassType || !(Class = ClassType->getInterface())) {
    Diag(Loc, diag::err_invalid_receiver_class_message)
      << ReceiverType;
    return ExprError();
  }
  assert(Class && "We don't know which class we're messaging?");

  // Find the method we are messaging.
  if (!Method) {
    if (Class->isForwardDecl()) {
      // A forward class used in messaging is treated as a 'Class'
      Diag(Loc, diag::warn_receiver_forward_class) << Class->getDeclName();
      Method = LookupFactoryMethodInGlobalPool(Sel, 
                                               SourceRange(LBracLoc, RBracLoc));
      if (Method)
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
  unsigned NumArgs = ArgsIn.size();
  Expr **Args = reinterpret_cast<Expr **>(ArgsIn.release());
  if (CheckMessageArgumentTypes(Args, NumArgs, Sel, Method, true,
                                LBracLoc, RBracLoc, ReturnType))
    return ExprError();

  // Construct the appropriate ObjCMessageExpr.
  Expr *Result;
  if (SuperLoc.isValid())
    Result = ObjCMessageExpr::Create(Context, ReturnType, LBracLoc, 
                                     SuperLoc, /*IsInstanceSuper=*/false, 
                                     ReceiverType, Sel, Method, Args, 
                                     NumArgs, RBracLoc);
  else
    Result = ObjCMessageExpr::Create(Context, ReturnType, LBracLoc, 
                                     ReceiverTypeInfo, Sel, Method, Args, 
                                     NumArgs, RBracLoc);
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
                           LBracLoc, RBracLoc, move(Args));
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
                                                  SourceLocation RBracLoc,
                                                  MultiExprArg ArgsIn) {
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
                                           LBracLoc, Receiver, Sel, 
                                           /*Method=*/0, Args, NumArgs, 
                                           RBracLoc));
    }

    // If necessary, apply function/array conversion to the receiver.
    // C99 6.7.5.3p[7,8].
    DefaultFunctionArrayLvalueConversion(Receiver);
    ReceiverType = Receiver->getType();
  }

  // The location of the receiver.
  SourceLocation Loc = SuperLoc.isValid()? SuperLoc : Receiver->getLocStart();

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
      if (ObjCMethodDecl *CurMeth = getCurMethodDecl()) {
        if (ObjCInterfaceDecl *ClassDecl = CurMeth->getClassInterface()) {
          // First check the public methods in the class interface.
          Method = ClassDecl->lookupClassMethod(Sel);

          if (!Method)
            Method = LookupPrivateClassMethod(Sel, ClassDecl);

          // FIXME: if we still haven't found a method, we need to look in
          // protocols (if we have qualifiers).
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
    } else {
      ObjCInterfaceDecl* ClassDecl = 0;

      // We allow sending a message to a qualified ID ("id<foo>"), which is ok as
      // long as one of the protocols implements the selector (if not, warn).
      if (const ObjCObjectPointerType *QIdTy 
                                   = ReceiverType->getAsObjCQualifiedIdType()) {
        // Search protocols for instance methods.
        for (ObjCObjectPointerType::qual_iterator I = QIdTy->qual_begin(),
               E = QIdTy->qual_end(); I != E; ++I) {
          ObjCProtocolDecl *PDecl = *I;
          if (PDecl && (Method = PDecl->lookupInstanceMethod(Sel)))
            break;
          // Since we aren't supporting "Class<foo>", look for a class method.
          if (PDecl && (Method = PDecl->lookupClassMethod(Sel)))
            break;
        }
      } else if (const ObjCObjectPointerType *OCIType
                   = ReceiverType->getAsObjCInterfacePointerType()) {
        // We allow sending a message to a pointer to an interface (an object).
        ClassDecl = OCIType->getInterfaceDecl();
        // FIXME: consider using LookupInstanceMethodInGlobalPool, since it will be
        // faster than the following method (which can do *many* linear searches).
        // The idea is to add class info to MethodPool.
        Method = ClassDecl->lookupInstanceMethod(Sel);

        if (!Method) {
          // Search protocol qualifiers.
          for (ObjCObjectPointerType::qual_iterator QI = OCIType->qual_begin(),
                 E = OCIType->qual_end(); QI != E; ++QI) {
            if ((Method = (*QI)->lookupInstanceMethod(Sel)))
              break;
          }
        }
        if (!Method) {
          // If we have implementations in scope, check "private" methods.
          Method = LookupPrivateInstanceMethod(Sel, ClassDecl);

          if (!Method && (!Receiver || !isSelfExpr(Receiver))) {
            // If we still haven't found a method, look in the global pool. This
            // behavior isn't very desirable, however we need it for GCC
            // compatibility. FIXME: should we deviate??
            if (OCIType->qual_empty()) {
              Method = LookupInstanceMethodInGlobalPool(Sel,
                                                 SourceRange(LBracLoc, RBracLoc)); 
              if (Method && !OCIType->getInterfaceDecl()->isForwardDecl())
                Diag(Loc, diag::warn_maynot_respond)
                  << OCIType->getInterfaceDecl()->getIdentifier() << Sel;
            }
          }
        }
        if (Method && DiagnoseUseOfDecl(Method, Loc))
          return ExprError();
      } else if (!Context.getObjCIdType().isNull() &&
                 (ReceiverType->isPointerType() || 
                  ReceiverType->isIntegerType())) {
        // Implicitly convert integers and pointers to 'id' but emit a warning.
        Diag(Loc, diag::warn_bad_receiver_type)
          << ReceiverType 
          << Receiver->getSourceRange();
        if (ReceiverType->isPointerType())
          ImpCastExprToType(Receiver, Context.getObjCIdType(), 
                            CK_BitCast);
        else
          ImpCastExprToType(Receiver, Context.getObjCIdType(),
                            CK_IntegralToPointer);
        ReceiverType = Receiver->getType();
      } 
      else if (getLangOptions().CPlusPlus &&
               !PerformContextuallyConvertToObjCId(Receiver)) {
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

  // Check the message arguments.
  unsigned NumArgs = ArgsIn.size();
  Expr **Args = reinterpret_cast<Expr **>(ArgsIn.release());
  QualType ReturnType;
  if (CheckMessageArgumentTypes(Args, NumArgs, Sel, Method, false,
                                LBracLoc, RBracLoc, ReturnType))
    return ExprError();
  
  if (!ReturnType->isVoidType()) {
    if (RequireCompleteType(LBracLoc, ReturnType, 
                            diag::err_illegal_message_expr_incomplete_type))
      return ExprError();
  }

  // Construct the appropriate ObjCMessageExpr instance.
  Expr *Result;
  if (SuperLoc.isValid())
    Result = ObjCMessageExpr::Create(Context, ReturnType, LBracLoc,
                                     SuperLoc,  /*IsInstanceSuper=*/true,
                                     ReceiverType, Sel, Method, 
                                     Args, NumArgs, RBracLoc);
  else
    Result = ObjCMessageExpr::Create(Context, ReturnType, LBracLoc, Receiver, 
                                     Sel, Method, Args, NumArgs, RBracLoc);
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
                              LBracLoc, RBracLoc, move(Args));
}

