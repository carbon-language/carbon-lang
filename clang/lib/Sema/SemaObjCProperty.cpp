//===--- SemaObjCProperty.cpp - Semantic Analysis for ObjC @property ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for Objective C @property and
//  @synthesize declarations.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/Sema.h"
#include "clang/Sema/Initialization.h"
#include "clang/AST/ExprObjC.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Grammar actions.
//===----------------------------------------------------------------------===//

Sema::DeclPtrTy Sema::ActOnProperty(Scope *S, SourceLocation AtLoc,
                                    FieldDeclarator &FD,
                                    ObjCDeclSpec &ODS,
                                    Selector GetterSel,
                                    Selector SetterSel,
                                    DeclPtrTy ClassCategory,
                                    bool *isOverridingProperty,
                                    tok::ObjCKeywordKind MethodImplKind) {
  unsigned Attributes = ODS.getPropertyAttributes();
  bool isReadWrite = ((Attributes & ObjCDeclSpec::DQ_PR_readwrite) ||
                      // default is readwrite!
                      !(Attributes & ObjCDeclSpec::DQ_PR_readonly));
  // property is defaulted to 'assign' if it is readwrite and is
  // not retain or copy
  bool isAssign = ((Attributes & ObjCDeclSpec::DQ_PR_assign) ||
                   (isReadWrite &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_retain) &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_copy)));

  TypeSourceInfo *TSI = GetTypeForDeclarator(FD.D, S);
  QualType T = TSI->getType();
  if (T->isReferenceType()) {
    Diag(AtLoc, diag::error_reference_property);
    return DeclPtrTy();
  }
  // Proceed with constructing the ObjCPropertDecls.
  ObjCContainerDecl *ClassDecl =
    cast<ObjCContainerDecl>(ClassCategory.getAs<Decl>());

  if (ObjCCategoryDecl *CDecl = dyn_cast<ObjCCategoryDecl>(ClassDecl))
    if (CDecl->IsClassExtension()) {
      DeclPtrTy Res = HandlePropertyInClassExtension(S, CDecl, AtLoc,
                                           FD, GetterSel, SetterSel,
                                           isAssign, isReadWrite,
                                           Attributes,
                                           isOverridingProperty, TSI,
                                           MethodImplKind);
      if (Res)
        CheckObjCPropertyAttributes(Res, AtLoc, Attributes);
      return Res;
    }
  
  DeclPtrTy Res =  DeclPtrTy::make(CreatePropertyDecl(S, ClassDecl, AtLoc, FD,
                                              GetterSel, SetterSel,
                                              isAssign, isReadWrite,
                                              Attributes, TSI, MethodImplKind));
  // Validate the attributes on the @property.
  CheckObjCPropertyAttributes(Res, AtLoc, Attributes);
  return Res;
}

Sema::DeclPtrTy
Sema::HandlePropertyInClassExtension(Scope *S, ObjCCategoryDecl *CDecl,
                                     SourceLocation AtLoc, FieldDeclarator &FD,
                                     Selector GetterSel, Selector SetterSel,
                                     const bool isAssign,
                                     const bool isReadWrite,
                                     const unsigned Attributes,
                                     bool *isOverridingProperty,
                                     TypeSourceInfo *T,
                                     tok::ObjCKeywordKind MethodImplKind) {

  // Diagnose if this property is already in continuation class.
  DeclContext *DC = cast<DeclContext>(CDecl);
  IdentifierInfo *PropertyId = FD.D.getIdentifier();

  if (ObjCPropertyDecl *prevDecl =
        ObjCPropertyDecl::findPropertyDecl(DC, PropertyId)) {
    Diag(AtLoc, diag::err_duplicate_property);
    Diag(prevDecl->getLocation(), diag::note_property_declare);
    return DeclPtrTy();
  }

  // Create a new ObjCPropertyDecl with the DeclContext being
  // the class extension.
  ObjCPropertyDecl *PDecl =
    ObjCPropertyDecl::Create(Context, DC, FD.D.getIdentifierLoc(),
                             PropertyId, AtLoc, T);
  if (Attributes & ObjCDeclSpec::DQ_PR_readonly)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readonly);
  if (Attributes & ObjCDeclSpec::DQ_PR_readwrite)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readwrite);

  DC->addDecl(PDecl);

  // We need to look in the @interface to see if the @property was
  // already declared.
  ObjCInterfaceDecl *CCPrimary = CDecl->getClassInterface();
  if (!CCPrimary) {
    Diag(CDecl->getLocation(), diag::err_continuation_class);
    *isOverridingProperty = true;
    return DeclPtrTy();
  }

  // Find the property in continuation class's primary class only.
  ObjCPropertyDecl *PIDecl =
    CCPrimary->FindPropertyVisibleInPrimaryClass(PropertyId);

  if (!PIDecl) {
    // No matching property found in the primary class. Just fall thru
    // and add property to continuation class's primary class.
    ObjCPropertyDecl *PDecl =
      CreatePropertyDecl(S, CCPrimary, AtLoc,
                         FD, GetterSel, SetterSel, isAssign, isReadWrite,
                         Attributes, T, MethodImplKind, DC);
    // Mark written attribute as having no attribute because
    // this is not a user-written property declaration in primary
    // class.
    PDecl->setPropertyAttributesAsWritten(ObjCPropertyDecl::OBJC_PR_noattr);

    // A case of continuation class adding a new property in the class. This
    // is not what it was meant for. However, gcc supports it and so should we.
    // Make sure setter/getters are declared here.
    ProcessPropertyDecl(PDecl, CCPrimary);
    return DeclPtrTy::make(PDecl);

  }

  // The property 'PIDecl's readonly attribute will be over-ridden
  // with continuation class's readwrite property attribute!
  unsigned PIkind = PIDecl->getPropertyAttributesAsWritten();
  if (isReadWrite && (PIkind & ObjCPropertyDecl::OBJC_PR_readonly)) {
    unsigned retainCopyNonatomic =
    (ObjCPropertyDecl::OBJC_PR_retain |
     ObjCPropertyDecl::OBJC_PR_copy |
     ObjCPropertyDecl::OBJC_PR_nonatomic);
    if ((Attributes & retainCopyNonatomic) !=
        (PIkind & retainCopyNonatomic)) {
      Diag(AtLoc, diag::warn_property_attr_mismatch);
      Diag(PIDecl->getLocation(), diag::note_property_declare);
    }
    DeclContext *DC = cast<DeclContext>(CCPrimary);
    if (!ObjCPropertyDecl::findPropertyDecl(DC,
                                 PIDecl->getDeclName().getAsIdentifierInfo())) {
      // Protocol is not in the primary class. Must build one for it.
      ObjCDeclSpec ProtocolPropertyODS;
      // FIXME. Assuming that ObjCDeclSpec::ObjCPropertyAttributeKind
      // and ObjCPropertyDecl::PropertyAttributeKind have identical
      // values.  Should consolidate both into one enum type.
      ProtocolPropertyODS.
      setPropertyAttributes((ObjCDeclSpec::ObjCPropertyAttributeKind)
                            PIkind);

      DeclPtrTy ProtocolPtrTy =
        ActOnProperty(S, AtLoc, FD, ProtocolPropertyODS,
                      PIDecl->getGetterName(),
                      PIDecl->getSetterName(),
                      DeclPtrTy::make(CCPrimary), isOverridingProperty,
                      MethodImplKind);
      PIDecl = cast<ObjCPropertyDecl>(ProtocolPtrTy.getAs<Decl>());
    }
    PIDecl->makeitReadWriteAttribute();
    if (Attributes & ObjCDeclSpec::DQ_PR_retain)
      PIDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_retain);
    if (Attributes & ObjCDeclSpec::DQ_PR_copy)
      PIDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_copy);
    PIDecl->setSetterName(SetterSel);
  } else {
    Diag(AtLoc, diag::err_use_continuation_class)
      << CCPrimary->getDeclName();
    Diag(PIDecl->getLocation(), diag::note_property_declare);
  }
  *isOverridingProperty = true;
  // Make sure setter decl is synthesized, and added to primary class's list.
  ProcessPropertyDecl(PIDecl, CCPrimary);
  return DeclPtrTy();
}

ObjCPropertyDecl *Sema::CreatePropertyDecl(Scope *S,
                                           ObjCContainerDecl *CDecl,
                                           SourceLocation AtLoc,
                                           FieldDeclarator &FD,
                                           Selector GetterSel,
                                           Selector SetterSel,
                                           const bool isAssign,
                                           const bool isReadWrite,
                                           const unsigned Attributes,
                                           TypeSourceInfo *TInfo,
                                           tok::ObjCKeywordKind MethodImplKind,
                                           DeclContext *lexicalDC){
  IdentifierInfo *PropertyId = FD.D.getIdentifier();
  QualType T = TInfo->getType();

  // Issue a warning if property is 'assign' as default and its object, which is
  // gc'able conforms to NSCopying protocol
  if (getLangOptions().getGCMode() != LangOptions::NonGC &&
      isAssign && !(Attributes & ObjCDeclSpec::DQ_PR_assign))
    if (const ObjCObjectPointerType *ObjPtrTy =
          T->getAs<ObjCObjectPointerType>()) {
      ObjCInterfaceDecl *IDecl = ObjPtrTy->getObjectType()->getInterface();
      if (IDecl)
        if (ObjCProtocolDecl* PNSCopying =
            LookupProtocol(&Context.Idents.get("NSCopying"), AtLoc))
          if (IDecl->ClassImplementsProtocol(PNSCopying, true))
            Diag(AtLoc, diag::warn_implements_nscopying) << PropertyId;
    }
  if (T->isObjCObjectType())
    Diag(FD.D.getIdentifierLoc(), diag::err_statically_allocated_object);

  DeclContext *DC = cast<DeclContext>(CDecl);
  ObjCPropertyDecl *PDecl = ObjCPropertyDecl::Create(Context, DC,
                                                     FD.D.getIdentifierLoc(),
                                                     PropertyId, AtLoc, TInfo);

  if (ObjCPropertyDecl *prevDecl =
        ObjCPropertyDecl::findPropertyDecl(DC, PropertyId)) {
    Diag(PDecl->getLocation(), diag::err_duplicate_property);
    Diag(prevDecl->getLocation(), diag::note_property_declare);
    PDecl->setInvalidDecl();
  }
  else {
    DC->addDecl(PDecl);
    if (lexicalDC)
      PDecl->setLexicalDeclContext(lexicalDC);
  }

  if (T->isArrayType() || T->isFunctionType()) {
    Diag(AtLoc, diag::err_property_type) << T;
    PDecl->setInvalidDecl();
  }

  ProcessDeclAttributes(S, PDecl, FD.D);

  // Regardless of setter/getter attribute, we save the default getter/setter
  // selector names in anticipation of declaration of setter/getter methods.
  PDecl->setGetterName(GetterSel);
  PDecl->setSetterName(SetterSel);

  if (Attributes & ObjCDeclSpec::DQ_PR_readonly)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readonly);

  if (Attributes & ObjCDeclSpec::DQ_PR_getter)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_getter);

  if (Attributes & ObjCDeclSpec::DQ_PR_setter)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_setter);

  if (isReadWrite)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readwrite);

  if (Attributes & ObjCDeclSpec::DQ_PR_retain)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_retain);

  if (Attributes & ObjCDeclSpec::DQ_PR_copy)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_copy);

  if (isAssign)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_assign);

  if (Attributes & ObjCDeclSpec::DQ_PR_nonatomic)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_nonatomic);

  PDecl->setPropertyAttributesAsWritten(PDecl->getPropertyAttributes());
  
  if (MethodImplKind == tok::objc_required)
    PDecl->setPropertyImplementation(ObjCPropertyDecl::Required);
  else if (MethodImplKind == tok::objc_optional)
    PDecl->setPropertyImplementation(ObjCPropertyDecl::Optional);

  return PDecl;
}


/// ActOnPropertyImplDecl - This routine performs semantic checks and
/// builds the AST node for a property implementation declaration; declared
/// as @synthesize or @dynamic.
///
Sema::DeclPtrTy Sema::ActOnPropertyImplDecl(Scope *S,
                                            SourceLocation AtLoc,
                                            SourceLocation PropertyLoc,
                                            bool Synthesize,
                                            DeclPtrTy ClassCatImpDecl,
                                            IdentifierInfo *PropertyId,
                                            IdentifierInfo *PropertyIvar) {
  ObjCContainerDecl *ClassImpDecl =
    cast_or_null<ObjCContainerDecl>(ClassCatImpDecl.getAs<Decl>());
  // Make sure we have a context for the property implementation declaration.
  if (!ClassImpDecl) {
    Diag(AtLoc, diag::error_missing_property_context);
    return DeclPtrTy();
  }
  ObjCPropertyDecl *property = 0;
  ObjCInterfaceDecl* IDecl = 0;
  // Find the class or category class where this property must have
  // a declaration.
  ObjCImplementationDecl *IC = 0;
  ObjCCategoryImplDecl* CatImplClass = 0;
  if ((IC = dyn_cast<ObjCImplementationDecl>(ClassImpDecl))) {
    IDecl = IC->getClassInterface();
    // We always synthesize an interface for an implementation
    // without an interface decl. So, IDecl is always non-zero.
    assert(IDecl &&
           "ActOnPropertyImplDecl - @implementation without @interface");

    // Look for this property declaration in the @implementation's @interface
    property = IDecl->FindPropertyDeclaration(PropertyId);
    if (!property) {
      Diag(PropertyLoc, diag::error_bad_property_decl) << IDecl->getDeclName();
      return DeclPtrTy();
    }
    if (const ObjCCategoryDecl *CD =
        dyn_cast<ObjCCategoryDecl>(property->getDeclContext())) {
      if (!CD->IsClassExtension()) {
        Diag(PropertyLoc, diag::error_category_property) << CD->getDeclName();
        Diag(property->getLocation(), diag::note_property_declare);
        return DeclPtrTy();
      }
    }
  } else if ((CatImplClass = dyn_cast<ObjCCategoryImplDecl>(ClassImpDecl))) {
    if (Synthesize) {
      Diag(AtLoc, diag::error_synthesize_category_decl);
      return DeclPtrTy();
    }
    IDecl = CatImplClass->getClassInterface();
    if (!IDecl) {
      Diag(AtLoc, diag::error_missing_property_interface);
      return DeclPtrTy();
    }
    ObjCCategoryDecl *Category =
    IDecl->FindCategoryDeclaration(CatImplClass->getIdentifier());

    // If category for this implementation not found, it is an error which
    // has already been reported eralier.
    if (!Category)
      return DeclPtrTy();
    // Look for this property declaration in @implementation's category
    property = Category->FindPropertyDeclaration(PropertyId);
    if (!property) {
      Diag(PropertyLoc, diag::error_bad_category_property_decl)
      << Category->getDeclName();
      return DeclPtrTy();
    }
  } else {
    Diag(AtLoc, diag::error_bad_property_context);
    return DeclPtrTy();
  }
  ObjCIvarDecl *Ivar = 0;
  // Check that we have a valid, previously declared ivar for @synthesize
  if (Synthesize) {
    // @synthesize
    if (!PropertyIvar)
      PropertyIvar = PropertyId;
    QualType PropType = Context.getCanonicalType(property->getType());
    // Check that this is a previously declared 'ivar' in 'IDecl' interface
    ObjCInterfaceDecl *ClassDeclared;
    Ivar = IDecl->lookupInstanceVariable(PropertyIvar, ClassDeclared);
    if (!Ivar) {
      Ivar = ObjCIvarDecl::Create(Context, ClassImpDecl, PropertyLoc,
                                  PropertyIvar, PropType, /*Dinfo=*/0,
                                  ObjCIvarDecl::Protected,
                                  (Expr *)0, true);
      ClassImpDecl->addDecl(Ivar);
      IDecl->makeDeclVisibleInContext(Ivar, false);
      property->setPropertyIvarDecl(Ivar);

      if (!getLangOptions().ObjCNonFragileABI)
        Diag(PropertyLoc, diag::error_missing_property_ivar_decl) << PropertyId;
      // Note! I deliberately want it to fall thru so, we have a
      // a property implementation and to avoid future warnings.
    } else if (getLangOptions().ObjCNonFragileABI &&
               ClassDeclared != IDecl) {
      Diag(PropertyLoc, diag::error_ivar_in_superclass_use)
      << property->getDeclName() << Ivar->getDeclName()
      << ClassDeclared->getDeclName();
      Diag(Ivar->getLocation(), diag::note_previous_access_declaration)
      << Ivar << Ivar->getName();
      // Note! I deliberately want it to fall thru so more errors are caught.
    }
    QualType IvarType = Context.getCanonicalType(Ivar->getType());

    // Check that type of property and its ivar are type compatible.
    if (PropType != IvarType) {
      bool compat = false;
      if (isa<ObjCObjectPointerType>(PropType) 
            && isa<ObjCObjectPointerType>(IvarType))
        compat = 
          Context.canAssignObjCInterfaces(
                                  PropType->getAs<ObjCObjectPointerType>(),
                                  IvarType->getAs<ObjCObjectPointerType>());
      else 
        compat = (CheckAssignmentConstraints(PropType, IvarType) == Compatible);
      if (!compat) {
        Diag(PropertyLoc, diag::error_property_ivar_type)
          << property->getDeclName() << PropType
          << Ivar->getDeclName() << IvarType;
        Diag(Ivar->getLocation(), diag::note_ivar_decl);
        // Note! I deliberately want it to fall thru so, we have a
        // a property implementation and to avoid future warnings.
      }

      // FIXME! Rules for properties are somewhat different that those
      // for assignments. Use a new routine to consolidate all cases;
      // specifically for property redeclarations as well as for ivars.
      QualType lhsType =Context.getCanonicalType(PropType).getUnqualifiedType();
      QualType rhsType =Context.getCanonicalType(IvarType).getUnqualifiedType();
      if (lhsType != rhsType &&
          lhsType->isArithmeticType()) {
        Diag(PropertyLoc, diag::error_property_ivar_type)
          << property->getDeclName() << PropType
          << Ivar->getDeclName() << IvarType;
        Diag(Ivar->getLocation(), diag::note_ivar_decl);
        // Fall thru - see previous comment
      }
      // __weak is explicit. So it works on Canonical type.
      if (PropType.isObjCGCWeak() && !IvarType.isObjCGCWeak() &&
          getLangOptions().getGCMode() != LangOptions::NonGC) {
        Diag(PropertyLoc, diag::error_weak_property)
        << property->getDeclName() << Ivar->getDeclName();
        // Fall thru - see previous comment
      }
      if ((property->getType()->isObjCObjectPointerType() ||
           PropType.isObjCGCStrong()) && IvarType.isObjCGCWeak() &&
          getLangOptions().getGCMode() != LangOptions::NonGC) {
        Diag(PropertyLoc, diag::error_strong_property)
        << property->getDeclName() << Ivar->getDeclName();
        // Fall thru - see previous comment
      }
    }
  } else if (PropertyIvar)
    // @dynamic
    Diag(PropertyLoc, diag::error_dynamic_property_ivar_decl);
  assert (property && "ActOnPropertyImplDecl - property declaration missing");
  ObjCPropertyImplDecl *PIDecl =
  ObjCPropertyImplDecl::Create(Context, CurContext, AtLoc, PropertyLoc,
                               property,
                               (Synthesize ?
                                ObjCPropertyImplDecl::Synthesize
                                : ObjCPropertyImplDecl::Dynamic),
                               Ivar);
  if (ObjCMethodDecl *getterMethod = property->getGetterMethodDecl()) {
    getterMethod->createImplicitParams(Context, IDecl);
    if (getLangOptions().CPlusPlus && Synthesize) {
      // For Objective-C++, need to synthesize the AST for the IVAR object to be
      // returned by the getter as it must conform to C++'s copy-return rules.
      // FIXME. Eventually we want to do this for Objective-C as well.
      ImplicitParamDecl *SelfDecl = getterMethod->getSelfDecl();
      DeclRefExpr *SelfExpr = 
        new (Context) DeclRefExpr(SelfDecl,SelfDecl->getType(),
                                  SourceLocation());
      Expr *IvarRefExpr =
        new (Context) ObjCIvarRefExpr(Ivar, Ivar->getType(), AtLoc,
                                      SelfExpr, true, true);
      OwningExprResult Res = 
        PerformCopyInitialization(InitializedEntity::InitializeResult(
                                    SourceLocation(),
                                    getterMethod->getResultType(),
                                    /*NRVO=*/false),
                                  SourceLocation(),
                                  Owned(IvarRefExpr));
      if (!Res.isInvalid()) {
        Expr *ResExpr = Res.takeAs<Expr>();
        if (ResExpr)
          ResExpr = MaybeCreateCXXExprWithTemporaries(ResExpr);
        PIDecl->setGetterCXXConstructor(ResExpr);
      }
    }
  }
  if (ObjCMethodDecl *setterMethod = property->getSetterMethodDecl()) {
    setterMethod->createImplicitParams(Context, IDecl);
    if (getLangOptions().CPlusPlus && Synthesize) {
      // FIXME. Eventually we want to do this for Objective-C as well.
      ImplicitParamDecl *SelfDecl = setterMethod->getSelfDecl();
      DeclRefExpr *SelfExpr = 
        new (Context) DeclRefExpr(SelfDecl,SelfDecl->getType(),
                                  SourceLocation());
      Expr *lhs =
        new (Context) ObjCIvarRefExpr(Ivar, Ivar->getType(), AtLoc,
                                      SelfExpr, true, true);
      ObjCMethodDecl::param_iterator P = setterMethod->param_begin();
      ParmVarDecl *Param = (*P);
      Expr *rhs = new (Context) DeclRefExpr(Param,Param->getType(),
                                            SourceLocation());
      OwningExprResult Res = BuildBinOp(S, SourceLocation(), 
                                        BinaryOperator::Assign, lhs, rhs);
      PIDecl->setSetterCXXAssignment(Res.takeAs<Expr>());
    }
  }
  
  if (IC) {
    if (Synthesize)
      if (ObjCPropertyImplDecl *PPIDecl =
          IC->FindPropertyImplIvarDecl(PropertyIvar)) {
        Diag(PropertyLoc, diag::error_duplicate_ivar_use)
        << PropertyId << PPIDecl->getPropertyDecl()->getIdentifier()
        << PropertyIvar;
        Diag(PPIDecl->getLocation(), diag::note_previous_use);
      }

    if (ObjCPropertyImplDecl *PPIDecl
        = IC->FindPropertyImplDecl(PropertyId)) {
      Diag(PropertyLoc, diag::error_property_implemented) << PropertyId;
      Diag(PPIDecl->getLocation(), diag::note_previous_declaration);
      return DeclPtrTy();
    }
    IC->addPropertyImplementation(PIDecl);
    if (getLangOptions().ObjCNonFragileABI2) {
      // Diagnose if an ivar was lazily synthesdized due to a previous
      // use and if 1) property is @dynamic or 2) property is synthesized
      // but it requires a dirreferently named ivar.
      ObjCInterfaceDecl *ClassDeclared;
      ObjCIvarDecl *Ivar = 0;
      if (!Synthesize)
        Ivar = IDecl->lookupInstanceVariable(PropertyId, ClassDeclared);
      else {
        if (PropertyIvar && PropertyIvar != PropertyId)
          Ivar = IDecl->lookupInstanceVariable(PropertyId, ClassDeclared);
      }
      if (Ivar && Ivar->getSynthesize()) {
        Diag(Ivar->getLocation(), diag::err_undeclared_var_use) 
        << PropertyId;
        Ivar->setInvalidDecl();
      }
    }
  } else {
    if (Synthesize)
      if (ObjCPropertyImplDecl *PPIDecl =
          CatImplClass->FindPropertyImplIvarDecl(PropertyIvar)) {
        Diag(PropertyLoc, diag::error_duplicate_ivar_use)
        << PropertyId << PPIDecl->getPropertyDecl()->getIdentifier()
        << PropertyIvar;
        Diag(PPIDecl->getLocation(), diag::note_previous_use);
      }

    if (ObjCPropertyImplDecl *PPIDecl =
        CatImplClass->FindPropertyImplDecl(PropertyId)) {
      Diag(PropertyLoc, diag::error_property_implemented) << PropertyId;
      Diag(PPIDecl->getLocation(), diag::note_previous_declaration);
      return DeclPtrTy();
    }
    CatImplClass->addPropertyImplementation(PIDecl);
  }

  return DeclPtrTy::make(PIDecl);
}

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// DiagnosePropertyMismatch - Compares two properties for their
/// attributes and types and warns on a variety of inconsistencies.
///
void
Sema::DiagnosePropertyMismatch(ObjCPropertyDecl *Property,
                               ObjCPropertyDecl *SuperProperty,
                               const IdentifierInfo *inheritedName) {
  ObjCPropertyDecl::PropertyAttributeKind CAttr =
  Property->getPropertyAttributes();
  ObjCPropertyDecl::PropertyAttributeKind SAttr =
  SuperProperty->getPropertyAttributes();
  if ((CAttr & ObjCPropertyDecl::OBJC_PR_readonly)
      && (SAttr & ObjCPropertyDecl::OBJC_PR_readwrite))
    Diag(Property->getLocation(), diag::warn_readonly_property)
      << Property->getDeclName() << inheritedName;
  if ((CAttr & ObjCPropertyDecl::OBJC_PR_copy)
      != (SAttr & ObjCPropertyDecl::OBJC_PR_copy))
    Diag(Property->getLocation(), diag::warn_property_attribute)
      << Property->getDeclName() << "copy" << inheritedName;
  else if ((CAttr & ObjCPropertyDecl::OBJC_PR_retain)
           != (SAttr & ObjCPropertyDecl::OBJC_PR_retain))
    Diag(Property->getLocation(), diag::warn_property_attribute)
      << Property->getDeclName() << "retain" << inheritedName;

  if ((CAttr & ObjCPropertyDecl::OBJC_PR_nonatomic)
      != (SAttr & ObjCPropertyDecl::OBJC_PR_nonatomic))
    Diag(Property->getLocation(), diag::warn_property_attribute)
      << Property->getDeclName() << "atomic" << inheritedName;
  if (Property->getSetterName() != SuperProperty->getSetterName())
    Diag(Property->getLocation(), diag::warn_property_attribute)
      << Property->getDeclName() << "setter" << inheritedName;
  if (Property->getGetterName() != SuperProperty->getGetterName())
    Diag(Property->getLocation(), diag::warn_property_attribute)
      << Property->getDeclName() << "getter" << inheritedName;

  QualType LHSType =
    Context.getCanonicalType(SuperProperty->getType());
  QualType RHSType =
    Context.getCanonicalType(Property->getType());

  if (!Context.typesAreCompatible(LHSType, RHSType)) {
    // FIXME: Incorporate this test with typesAreCompatible.
    if (LHSType->isObjCQualifiedIdType() && RHSType->isObjCQualifiedIdType())
      if (Context.ObjCQualifiedIdTypesAreCompatible(LHSType, RHSType, false))
        return;
    Diag(Property->getLocation(), diag::warn_property_types_are_incompatible)
      << Property->getType() << SuperProperty->getType() << inheritedName;
  }
}

bool Sema::DiagnosePropertyAccessorMismatch(ObjCPropertyDecl *property,
                                            ObjCMethodDecl *GetterMethod,
                                            SourceLocation Loc) {
  if (GetterMethod &&
      GetterMethod->getResultType() != property->getType()) {
    AssignConvertType result = Incompatible;
    if (property->getType()->isObjCObjectPointerType())
      result = CheckAssignmentConstraints(GetterMethod->getResultType(),
                                          property->getType());
    if (result != Compatible) {
      Diag(Loc, diag::warn_accessor_property_type_mismatch)
      << property->getDeclName()
      << GetterMethod->getSelector();
      Diag(GetterMethod->getLocation(), diag::note_declared_at);
      return true;
    }
  }
  return false;
}

/// ComparePropertiesInBaseAndSuper - This routine compares property
/// declarations in base and its super class, if any, and issues
/// diagnostics in a variety of inconsistant situations.
///
void Sema::ComparePropertiesInBaseAndSuper(ObjCInterfaceDecl *IDecl) {
  ObjCInterfaceDecl *SDecl = IDecl->getSuperClass();
  if (!SDecl)
    return;
  // FIXME: O(N^2)
  for (ObjCInterfaceDecl::prop_iterator S = SDecl->prop_begin(),
       E = SDecl->prop_end(); S != E; ++S) {
    ObjCPropertyDecl *SuperPDecl = (*S);
    // Does property in super class has declaration in current class?
    for (ObjCInterfaceDecl::prop_iterator I = IDecl->prop_begin(),
         E = IDecl->prop_end(); I != E; ++I) {
      ObjCPropertyDecl *PDecl = (*I);
      if (SuperPDecl->getIdentifier() == PDecl->getIdentifier())
          DiagnosePropertyMismatch(PDecl, SuperPDecl,
                                   SDecl->getIdentifier());
    }
  }
}

/// MatchOneProtocolPropertiesInClass - This routine goes thru the list
/// of properties declared in a protocol and compares their attribute against
/// the same property declared in the class or category.
void
Sema::MatchOneProtocolPropertiesInClass(Decl *CDecl,
                                          ObjCProtocolDecl *PDecl) {
  ObjCInterfaceDecl *IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(CDecl);
  if (!IDecl) {
    // Category
    ObjCCategoryDecl *CatDecl = static_cast<ObjCCategoryDecl*>(CDecl);
    assert (CatDecl && "MatchOneProtocolPropertiesInClass");
    if (!CatDecl->IsClassExtension())
      for (ObjCProtocolDecl::prop_iterator P = PDecl->prop_begin(),
           E = PDecl->prop_end(); P != E; ++P) {
        ObjCPropertyDecl *Pr = (*P);
        ObjCCategoryDecl::prop_iterator CP, CE;
        // Is this property already in  category's list of properties?
        for (CP = CatDecl->prop_begin(), CE = CatDecl->prop_end(); CP!=CE; ++CP)
          if ((*CP)->getIdentifier() == Pr->getIdentifier())
            break;
        if (CP != CE)
          // Property protocol already exist in class. Diagnose any mismatch.
          DiagnosePropertyMismatch((*CP), Pr, PDecl->getIdentifier());
      }
    return;
  }
  for (ObjCProtocolDecl::prop_iterator P = PDecl->prop_begin(),
       E = PDecl->prop_end(); P != E; ++P) {
    ObjCPropertyDecl *Pr = (*P);
    ObjCInterfaceDecl::prop_iterator CP, CE;
    // Is this property already in  class's list of properties?
    for (CP = IDecl->prop_begin(), CE = IDecl->prop_end(); CP != CE; ++CP)
      if ((*CP)->getIdentifier() == Pr->getIdentifier())
        break;
    if (CP != CE)
      // Property protocol already exist in class. Diagnose any mismatch.
      DiagnosePropertyMismatch((*CP), Pr, PDecl->getIdentifier());
    }
}

/// CompareProperties - This routine compares properties
/// declared in 'ClassOrProtocol' objects (which can be a class or an
/// inherited protocol with the list of properties for class/category 'CDecl'
///
void Sema::CompareProperties(Decl *CDecl,
                             DeclPtrTy ClassOrProtocol) {
  Decl *ClassDecl = ClassOrProtocol.getAs<Decl>();
  ObjCInterfaceDecl *IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(CDecl);

  if (!IDecl) {
    // Category
    ObjCCategoryDecl *CatDecl = static_cast<ObjCCategoryDecl*>(CDecl);
    assert (CatDecl && "CompareProperties");
    if (ObjCCategoryDecl *MDecl = dyn_cast<ObjCCategoryDecl>(ClassDecl)) {
      for (ObjCCategoryDecl::protocol_iterator P = MDecl->protocol_begin(),
           E = MDecl->protocol_end(); P != E; ++P)
      // Match properties of category with those of protocol (*P)
      MatchOneProtocolPropertiesInClass(CatDecl, *P);

      // Go thru the list of protocols for this category and recursively match
      // their properties with those in the category.
      for (ObjCCategoryDecl::protocol_iterator P = CatDecl->protocol_begin(),
           E = CatDecl->protocol_end(); P != E; ++P)
        CompareProperties(CatDecl, DeclPtrTy::make(*P));
    } else {
      ObjCProtocolDecl *MD = cast<ObjCProtocolDecl>(ClassDecl);
      for (ObjCProtocolDecl::protocol_iterator P = MD->protocol_begin(),
           E = MD->protocol_end(); P != E; ++P)
        MatchOneProtocolPropertiesInClass(CatDecl, *P);
    }
    return;
  }

  if (ObjCInterfaceDecl *MDecl = dyn_cast<ObjCInterfaceDecl>(ClassDecl)) {
    for (ObjCInterfaceDecl::protocol_iterator P = MDecl->protocol_begin(),
         E = MDecl->protocol_end(); P != E; ++P)
      // Match properties of class IDecl with those of protocol (*P).
      MatchOneProtocolPropertiesInClass(IDecl, *P);

    // Go thru the list of protocols for this class and recursively match
    // their properties with those declared in the class.
    for (ObjCInterfaceDecl::protocol_iterator P = IDecl->protocol_begin(),
         E = IDecl->protocol_end(); P != E; ++P)
      CompareProperties(IDecl, DeclPtrTy::make(*P));
  } else {
    ObjCProtocolDecl *MD = cast<ObjCProtocolDecl>(ClassDecl);
    for (ObjCProtocolDecl::protocol_iterator P = MD->protocol_begin(),
         E = MD->protocol_end(); P != E; ++P)
      MatchOneProtocolPropertiesInClass(IDecl, *P);
  }
}

/// isPropertyReadonly - Return true if property is readonly, by searching
/// for the property in the class and in its categories and implementations
///
bool Sema::isPropertyReadonly(ObjCPropertyDecl *PDecl,
                              ObjCInterfaceDecl *IDecl) {
  // by far the most common case.
  if (!PDecl->isReadOnly())
    return false;
  // Even if property is ready only, if interface has a user defined setter,
  // it is not considered read only.
  if (IDecl->getInstanceMethod(PDecl->getSetterName()))
    return false;

  // Main class has the property as 'readonly'. Must search
  // through the category list to see if the property's
  // attribute has been over-ridden to 'readwrite'.
  for (ObjCCategoryDecl *Category = IDecl->getCategoryList();
       Category; Category = Category->getNextClassCategory()) {
    // Even if property is ready only, if a category has a user defined setter,
    // it is not considered read only.
    if (Category->getInstanceMethod(PDecl->getSetterName()))
      return false;
    ObjCPropertyDecl *P =
      Category->FindPropertyDeclaration(PDecl->getIdentifier());
    if (P && !P->isReadOnly())
      return false;
  }

  // Also, check for definition of a setter method in the implementation if
  // all else failed.
  if (ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(CurContext)) {
    if (ObjCImplementationDecl *IMD =
        dyn_cast<ObjCImplementationDecl>(OMD->getDeclContext())) {
      if (IMD->getInstanceMethod(PDecl->getSetterName()))
        return false;
    } else if (ObjCCategoryImplDecl *CIMD =
               dyn_cast<ObjCCategoryImplDecl>(OMD->getDeclContext())) {
      if (CIMD->getInstanceMethod(PDecl->getSetterName()))
        return false;
    }
  }
  // Lastly, look through the implementation (if one is in scope).
  if (ObjCImplementationDecl *ImpDecl = IDecl->getImplementation())
    if (ImpDecl->getInstanceMethod(PDecl->getSetterName()))
      return false;
  // If all fails, look at the super class.
  if (ObjCInterfaceDecl *SIDecl = IDecl->getSuperClass())
    return isPropertyReadonly(PDecl, SIDecl);
  return true;
}

/// CollectImmediateProperties - This routine collects all properties in
/// the class and its conforming protocols; but not those it its super class.
void Sema::CollectImmediateProperties(ObjCContainerDecl *CDecl,
            llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*>& PropMap,
            llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*>& SuperPropMap) {
  if (ObjCInterfaceDecl *IDecl = dyn_cast<ObjCInterfaceDecl>(CDecl)) {
    for (ObjCContainerDecl::prop_iterator P = IDecl->prop_begin(),
         E = IDecl->prop_end(); P != E; ++P) {
      ObjCPropertyDecl *Prop = (*P);
      PropMap[Prop->getIdentifier()] = Prop;
    }
    // scan through class's protocols.
    for (ObjCInterfaceDecl::protocol_iterator PI = IDecl->protocol_begin(),
         E = IDecl->protocol_end(); PI != E; ++PI)
        CollectImmediateProperties((*PI), PropMap, SuperPropMap);
  }
  if (ObjCCategoryDecl *CATDecl = dyn_cast<ObjCCategoryDecl>(CDecl)) {
    if (!CATDecl->IsClassExtension())
      for (ObjCContainerDecl::prop_iterator P = CATDecl->prop_begin(),
           E = CATDecl->prop_end(); P != E; ++P) {
        ObjCPropertyDecl *Prop = (*P);
        PropMap[Prop->getIdentifier()] = Prop;
      }
    // scan through class's protocols.
    for (ObjCInterfaceDecl::protocol_iterator PI = CATDecl->protocol_begin(),
         E = CATDecl->protocol_end(); PI != E; ++PI)
      CollectImmediateProperties((*PI), PropMap, SuperPropMap);
  }
  else if (ObjCProtocolDecl *PDecl = dyn_cast<ObjCProtocolDecl>(CDecl)) {
    for (ObjCProtocolDecl::prop_iterator P = PDecl->prop_begin(),
         E = PDecl->prop_end(); P != E; ++P) {
      ObjCPropertyDecl *Prop = (*P);
      ObjCPropertyDecl *PropertyFromSuper = SuperPropMap[Prop->getIdentifier()];
      // Exclude property for protocols which conform to class's super-class, 
      // as super-class has to implement the property.
      if (!PropertyFromSuper || PropertyFromSuper != Prop) {
        ObjCPropertyDecl *&PropEntry = PropMap[Prop->getIdentifier()];
        if (!PropEntry)
          PropEntry = Prop;
      }
    }
    // scan through protocol's protocols.
    for (ObjCProtocolDecl::protocol_iterator PI = PDecl->protocol_begin(),
         E = PDecl->protocol_end(); PI != E; ++PI)
      CollectImmediateProperties((*PI), PropMap, SuperPropMap);
  }
}

/// CollectClassPropertyImplementations - This routine collects list of
/// properties to be implemented in the class. This includes, class's
/// and its conforming protocols' properties.
static void CollectClassPropertyImplementations(ObjCContainerDecl *CDecl,
                llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*>& PropMap) {
  if (ObjCInterfaceDecl *IDecl = dyn_cast<ObjCInterfaceDecl>(CDecl)) {
    for (ObjCContainerDecl::prop_iterator P = IDecl->prop_begin(),
         E = IDecl->prop_end(); P != E; ++P) {
      ObjCPropertyDecl *Prop = (*P);
      PropMap[Prop->getIdentifier()] = Prop;
    }
    for (ObjCInterfaceDecl::protocol_iterator PI = IDecl->protocol_begin(),
         E = IDecl->protocol_end(); PI != E; ++PI)
      CollectClassPropertyImplementations((*PI), PropMap);
  }
  else if (ObjCProtocolDecl *PDecl = dyn_cast<ObjCProtocolDecl>(CDecl)) {
    for (ObjCProtocolDecl::prop_iterator P = PDecl->prop_begin(),
         E = PDecl->prop_end(); P != E; ++P) {
      ObjCPropertyDecl *Prop = (*P);
      PropMap[Prop->getIdentifier()] = Prop;
    }
    // scan through protocol's protocols.
    for (ObjCProtocolDecl::protocol_iterator PI = PDecl->protocol_begin(),
         E = PDecl->protocol_end(); PI != E; ++PI)
      CollectClassPropertyImplementations((*PI), PropMap);
  }
}

/// CollectSuperClassPropertyImplementations - This routine collects list of
/// properties to be implemented in super class(s) and also coming from their
/// conforming protocols.
static void CollectSuperClassPropertyImplementations(ObjCInterfaceDecl *CDecl,
                llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*>& PropMap) {
  if (ObjCInterfaceDecl *SDecl = CDecl->getSuperClass()) {
    while (SDecl) {
      CollectClassPropertyImplementations(SDecl, PropMap);
      SDecl = SDecl->getSuperClass();
    }
  }
}

/// LookupPropertyDecl - Looks up a property in the current class and all
/// its protocols.
ObjCPropertyDecl *Sema::LookupPropertyDecl(const ObjCContainerDecl *CDecl,
                                     IdentifierInfo *II) {
  if (const ObjCInterfaceDecl *IDecl =
        dyn_cast<ObjCInterfaceDecl>(CDecl)) {
    for (ObjCContainerDecl::prop_iterator P = IDecl->prop_begin(),
         E = IDecl->prop_end(); P != E; ++P) {
      ObjCPropertyDecl *Prop = (*P);
      if (Prop->getIdentifier() == II)
        return Prop;
    }
    // scan through class's protocols.
    for (ObjCInterfaceDecl::protocol_iterator PI = IDecl->protocol_begin(),
         E = IDecl->protocol_end(); PI != E; ++PI) {
      ObjCPropertyDecl *Prop = LookupPropertyDecl((*PI), II);
      if (Prop)
        return Prop;
    }
  }
  else if (const ObjCProtocolDecl *PDecl =
            dyn_cast<ObjCProtocolDecl>(CDecl)) {
    for (ObjCProtocolDecl::prop_iterator P = PDecl->prop_begin(),
         E = PDecl->prop_end(); P != E; ++P) {
      ObjCPropertyDecl *Prop = (*P);
      if (Prop->getIdentifier() == II)
        return Prop;
    }
    // scan through protocol's protocols.
    for (ObjCProtocolDecl::protocol_iterator PI = PDecl->protocol_begin(),
         E = PDecl->protocol_end(); PI != E; ++PI) {
      ObjCPropertyDecl *Prop = LookupPropertyDecl((*PI), II);
      if (Prop)
        return Prop;
    }
  }
  return 0;
}

/// DefaultSynthesizeProperties - This routine default synthesizes all
/// properties which must be synthesized in class's @implementation.
void Sema::DefaultSynthesizeProperties (Scope *S, ObjCImplDecl* IMPDecl,
                                        ObjCInterfaceDecl *IDecl) {
  
  llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*> PropMap;
  CollectClassPropertyImplementations(IDecl, PropMap);
  if (PropMap.empty())
    return;
  llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*> SuperPropMap;
  CollectSuperClassPropertyImplementations(IDecl, SuperPropMap);
  
  for (llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*>::iterator
       P = PropMap.begin(), E = PropMap.end(); P != E; ++P) {
    ObjCPropertyDecl *Prop = P->second;
    // If property to be implemented in the super class, ignore.
    if (SuperPropMap[Prop->getIdentifier()])
      continue;
    // Is there a matching propery synthesize/dynamic?
    if (Prop->isInvalidDecl() ||
        Prop->getPropertyImplementation() == ObjCPropertyDecl::Optional ||
        IMPDecl->FindPropertyImplIvarDecl(Prop->getIdentifier()))
      continue;
    // Property may have been synthesized by user.
    if (IMPDecl->FindPropertyImplDecl(Prop->getIdentifier()))
      continue;

    ActOnPropertyImplDecl(S, IMPDecl->getLocation(), IMPDecl->getLocation(),
                          true, DeclPtrTy::make(IMPDecl),
                          Prop->getIdentifier(), Prop->getIdentifier());
  }    
}

void Sema::DiagnoseUnimplementedProperties(Scope *S, ObjCImplDecl* IMPDecl,
                                      ObjCContainerDecl *CDecl,
                                      const llvm::DenseSet<Selector>& InsMap) {
  llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*> SuperPropMap;
  if (ObjCInterfaceDecl *IDecl = dyn_cast<ObjCInterfaceDecl>(CDecl))
    CollectSuperClassPropertyImplementations(IDecl, SuperPropMap);
  
  llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*> PropMap;
  CollectImmediateProperties(CDecl, PropMap, SuperPropMap);
  if (PropMap.empty())
    return;

  llvm::DenseSet<ObjCPropertyDecl *> PropImplMap;
  for (ObjCImplDecl::propimpl_iterator
       I = IMPDecl->propimpl_begin(),
       EI = IMPDecl->propimpl_end(); I != EI; ++I)
    PropImplMap.insert((*I)->getPropertyDecl());

  for (llvm::DenseMap<IdentifierInfo *, ObjCPropertyDecl*>::iterator
       P = PropMap.begin(), E = PropMap.end(); P != E; ++P) {
    ObjCPropertyDecl *Prop = P->second;
    // Is there a matching propery synthesize/dynamic?
    if (Prop->isInvalidDecl() ||
        Prop->getPropertyImplementation() == ObjCPropertyDecl::Optional ||
        PropImplMap.count(Prop))
      continue;
    if (!InsMap.count(Prop->getGetterName())) {
      Diag(Prop->getLocation(),
           isa<ObjCCategoryDecl>(CDecl) ?
            diag::warn_setter_getter_impl_required_in_category :
            diag::warn_setter_getter_impl_required)
      << Prop->getDeclName() << Prop->getGetterName();
      Diag(IMPDecl->getLocation(),
           diag::note_property_impl_required);
    }

    if (!Prop->isReadOnly() && !InsMap.count(Prop->getSetterName())) {
      Diag(Prop->getLocation(),
           isa<ObjCCategoryDecl>(CDecl) ?
           diag::warn_setter_getter_impl_required_in_category :
           diag::warn_setter_getter_impl_required)
      << Prop->getDeclName() << Prop->getSetterName();
      Diag(IMPDecl->getLocation(),
           diag::note_property_impl_required);
    }
  }
}

void
Sema::AtomicPropertySetterGetterRules (ObjCImplDecl* IMPDecl,
                                       ObjCContainerDecl* IDecl) {
  // Rules apply in non-GC mode only
  if (getLangOptions().getGCMode() != LangOptions::NonGC)
    return;
  for (ObjCContainerDecl::prop_iterator I = IDecl->prop_begin(),
       E = IDecl->prop_end();
       I != E; ++I) {
    ObjCPropertyDecl *Property = (*I);
    unsigned Attributes = Property->getPropertyAttributes();
    // We only care about readwrite atomic property.
    if ((Attributes & ObjCPropertyDecl::OBJC_PR_nonatomic) ||
        !(Attributes & ObjCPropertyDecl::OBJC_PR_readwrite))
      continue;
    if (const ObjCPropertyImplDecl *PIDecl
         = IMPDecl->FindPropertyImplDecl(Property->getIdentifier())) {
      if (PIDecl->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic)
        continue;
      ObjCMethodDecl *GetterMethod =
        IMPDecl->getInstanceMethod(Property->getGetterName());
      ObjCMethodDecl *SetterMethod =
        IMPDecl->getInstanceMethod(Property->getSetterName());
      if ((GetterMethod && !SetterMethod) || (!GetterMethod && SetterMethod)) {
        SourceLocation MethodLoc =
          (GetterMethod ? GetterMethod->getLocation()
                        : SetterMethod->getLocation());
        Diag(MethodLoc, diag::warn_atomic_property_rule)
          << Property->getIdentifier();
        Diag(Property->getLocation(), diag::note_property_declare);
      }
    }
  }
}

/// ProcessPropertyDecl - Make sure that any user-defined setter/getter methods
/// have the property type and issue diagnostics if they don't.
/// Also synthesize a getter/setter method if none exist (and update the
/// appropriate lookup tables. FIXME: Should reconsider if adding synthesized
/// methods is the "right" thing to do.
void Sema::ProcessPropertyDecl(ObjCPropertyDecl *property,
                               ObjCContainerDecl *CD) {
  ObjCMethodDecl *GetterMethod, *SetterMethod;

  GetterMethod = CD->getInstanceMethod(property->getGetterName());
  SetterMethod = CD->getInstanceMethod(property->getSetterName());
  DiagnosePropertyAccessorMismatch(property, GetterMethod,
                                   property->getLocation());

  if (SetterMethod) {
    ObjCPropertyDecl::PropertyAttributeKind CAttr =
      property->getPropertyAttributes();
    if ((!(CAttr & ObjCPropertyDecl::OBJC_PR_readonly)) &&
        Context.getCanonicalType(SetterMethod->getResultType()) !=
          Context.VoidTy)
      Diag(SetterMethod->getLocation(), diag::err_setter_type_void);
    if (SetterMethod->param_size() != 1 ||
        ((*SetterMethod->param_begin())->getType() != property->getType())) {
      Diag(property->getLocation(),
           diag::warn_accessor_property_type_mismatch)
        << property->getDeclName()
        << SetterMethod->getSelector();
      Diag(SetterMethod->getLocation(), diag::note_declared_at);
    }
  }

  // Synthesize getter/setter methods if none exist.
  // Find the default getter and if one not found, add one.
  // FIXME: The synthesized property we set here is misleading. We almost always
  // synthesize these methods unless the user explicitly provided prototypes
  // (which is odd, but allowed). Sema should be typechecking that the
  // declarations jive in that situation (which it is not currently).
  if (!GetterMethod) {
    // No instance method of same name as property getter name was found.
    // Declare a getter method and add it to the list of methods
    // for this class.
    GetterMethod = ObjCMethodDecl::Create(Context, property->getLocation(),
                             property->getLocation(), property->getGetterName(),
                             property->getType(), 0, CD, true, false, true, 
                             false,
                             (property->getPropertyImplementation() ==
                              ObjCPropertyDecl::Optional) ?
                             ObjCMethodDecl::Optional :
                             ObjCMethodDecl::Required);
    CD->addDecl(GetterMethod);
    // FIXME: Eventually this shouldn't be needed, as the lexical context
    // and the real context should be the same.
    if (DeclContext *lexicalDC = property->getLexicalDeclContext())
      GetterMethod->setLexicalDeclContext(lexicalDC);

  } else
    // A user declared getter will be synthesize when @synthesize of
    // the property with the same name is seen in the @implementation
    GetterMethod->setSynthesized(true);
  property->setGetterMethodDecl(GetterMethod);

  // Skip setter if property is read-only.
  if (!property->isReadOnly()) {
    // Find the default setter and if one not found, add one.
    if (!SetterMethod) {
      // No instance method of same name as property setter name was found.
      // Declare a setter method and add it to the list of methods
      // for this class.
      SetterMethod = ObjCMethodDecl::Create(Context, property->getLocation(),
                               property->getLocation(),
                               property->getSetterName(),
                               Context.VoidTy, 0, CD, true, false, true,
                               false,
                               (property->getPropertyImplementation() ==
                                ObjCPropertyDecl::Optional) ?
                               ObjCMethodDecl::Optional :
                               ObjCMethodDecl::Required);
      // Invent the arguments for the setter. We don't bother making a
      // nice name for the argument.
      ParmVarDecl *Argument = ParmVarDecl::Create(Context, SetterMethod,
                                                  property->getLocation(),
                                                  property->getIdentifier(),
                                                  property->getType(),
                                                  /*TInfo=*/0,
                                                  VarDecl::None,
                                                  VarDecl::None,
                                                  0);
      SetterMethod->setMethodParams(Context, &Argument, 1, 1);
      CD->addDecl(SetterMethod);
      // FIXME: Eventually this shouldn't be needed, as the lexical context
      // and the real context should be the same.
      if (DeclContext *lexicalDC = property->getLexicalDeclContext())
        SetterMethod->setLexicalDeclContext(lexicalDC);
    } else
      // A user declared setter will be synthesize when @synthesize of
      // the property with the same name is seen in the @implementation
      SetterMethod->setSynthesized(true);
    property->setSetterMethodDecl(SetterMethod);
  }
  // Add any synthesized methods to the global pool. This allows us to
  // handle the following, which is supported by GCC (and part of the design).
  //
  // @interface Foo
  // @property double bar;
  // @end
  //
  // void thisIsUnfortunate() {
  //   id foo;
  //   double bar = [foo bar];
  // }
  //
  if (GetterMethod)
    AddInstanceMethodToGlobalPool(GetterMethod);
  if (SetterMethod)
    AddInstanceMethodToGlobalPool(SetterMethod);
}

void Sema::CheckObjCPropertyAttributes(DeclPtrTy PropertyPtrTy,
                                       SourceLocation Loc,
                                       unsigned &Attributes) {
  // FIXME: Improve the reported location.
  Decl *PDecl = PropertyPtrTy.getAs<Decl>();
  if (!PDecl)
    return;

  ObjCPropertyDecl *PropertyDecl = cast<ObjCPropertyDecl>(PDecl);
  QualType PropertyTy = PropertyDecl->getType(); 

  // readonly and readwrite/assign/retain/copy conflict.
  if ((Attributes & ObjCDeclSpec::DQ_PR_readonly) &&
      (Attributes & (ObjCDeclSpec::DQ_PR_readwrite |
                     ObjCDeclSpec::DQ_PR_assign |
                     ObjCDeclSpec::DQ_PR_copy |
                     ObjCDeclSpec::DQ_PR_retain))) {
    const char * which = (Attributes & ObjCDeclSpec::DQ_PR_readwrite) ?
                          "readwrite" :
                         (Attributes & ObjCDeclSpec::DQ_PR_assign) ?
                          "assign" :
                         (Attributes & ObjCDeclSpec::DQ_PR_copy) ?
                          "copy" : "retain";

    Diag(Loc, (Attributes & (ObjCDeclSpec::DQ_PR_readwrite)) ?
                 diag::err_objc_property_attr_mutually_exclusive :
                 diag::warn_objc_property_attr_mutually_exclusive)
      << "readonly" << which;
  }

  // Check for copy or retain on non-object types.
  if ((Attributes & (ObjCDeclSpec::DQ_PR_copy | ObjCDeclSpec::DQ_PR_retain)) &&
      !PropertyTy->isObjCObjectPointerType() &&
      !PropertyTy->isBlockPointerType() &&
      !Context.isObjCNSObjectType(PropertyTy) &&
      !PropertyDecl->getAttr<ObjCNSObjectAttr>()) {
    Diag(Loc, diag::err_objc_property_requires_object)
      << (Attributes & ObjCDeclSpec::DQ_PR_copy ? "copy" : "retain");
    Attributes &= ~(ObjCDeclSpec::DQ_PR_copy | ObjCDeclSpec::DQ_PR_retain);
  }

  // Check for more than one of { assign, copy, retain }.
  if (Attributes & ObjCDeclSpec::DQ_PR_assign) {
    if (Attributes & ObjCDeclSpec::DQ_PR_copy) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "assign" << "copy";
      Attributes &= ~ObjCDeclSpec::DQ_PR_copy;
    }
    if (Attributes & ObjCDeclSpec::DQ_PR_retain) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "assign" << "retain";
      Attributes &= ~ObjCDeclSpec::DQ_PR_retain;
    }
  } else if (Attributes & ObjCDeclSpec::DQ_PR_copy) {
    if (Attributes & ObjCDeclSpec::DQ_PR_retain) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "copy" << "retain";
      Attributes &= ~ObjCDeclSpec::DQ_PR_retain;
    }
  }

  // Warn if user supplied no assignment attribute, property is
  // readwrite, and this is an object type.
  if (!(Attributes & (ObjCDeclSpec::DQ_PR_assign | ObjCDeclSpec::DQ_PR_copy |
                      ObjCDeclSpec::DQ_PR_retain)) &&
      !(Attributes & ObjCDeclSpec::DQ_PR_readonly) &&
      PropertyTy->isObjCObjectPointerType()) {
    // Skip this warning in gc-only mode.
    if (getLangOptions().getGCMode() != LangOptions::GCOnly)
      Diag(Loc, diag::warn_objc_property_no_assignment_attribute);

    // If non-gc code warn that this is likely inappropriate.
    if (getLangOptions().getGCMode() == LangOptions::NonGC)
      Diag(Loc, diag::warn_objc_property_default_assign_on_object);

    // FIXME: Implement warning dependent on NSCopying being
    // implemented. See also:
    // <rdar://5168496&4855821&5607453&5096644&4947311&5698469&4947014&5168496>
    // (please trim this list while you are at it).
  }

  if (!(Attributes & ObjCDeclSpec::DQ_PR_copy)
      && getLangOptions().getGCMode() == LangOptions::GCOnly
      && PropertyTy->isBlockPointerType())
    Diag(Loc, diag::warn_objc_property_copy_missing_on_block);
}
