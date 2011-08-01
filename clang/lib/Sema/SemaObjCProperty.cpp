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

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Initialization.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprObjC.h"
#include "llvm/ADT/DenseSet.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Grammar actions.
//===----------------------------------------------------------------------===//

/// Check the internal consistency of a property declaration.
static void checkARCPropertyDecl(Sema &S, ObjCPropertyDecl *property) {
  if (property->isInvalidDecl()) return;

  ObjCPropertyDecl::PropertyAttributeKind propertyKind
    = property->getPropertyAttributes();
  Qualifiers::ObjCLifetime propertyLifetime
    = property->getType().getObjCLifetime();

  // Nothing to do if we don't have a lifetime.
  if (propertyLifetime == Qualifiers::OCL_None) return;

  Qualifiers::ObjCLifetime expectedLifetime;
  unsigned selector;

  // Strong properties should have either strong or no lifetime.
  if (propertyKind & (ObjCPropertyDecl::OBJC_PR_retain |
                      ObjCPropertyDecl::OBJC_PR_strong |
                      ObjCPropertyDecl::OBJC_PR_copy)) {
    expectedLifetime = Qualifiers::OCL_Strong;
    selector = 0;
  } else if (propertyKind & ObjCPropertyDecl::OBJC_PR_weak) {
    expectedLifetime = Qualifiers::OCL_Weak;
    selector = 1;
  } else if (propertyKind & (ObjCPropertyDecl::OBJC_PR_assign |
                             ObjCPropertyDecl::OBJC_PR_unsafe_unretained) &&
             property->getType()->isObjCRetainableType()) {
    expectedLifetime = Qualifiers::OCL_ExplicitNone;
    selector = 2;
  } else {
    // We have a lifetime qualifier but no dominating property
    // attribute.  That's okay.
    return;
  }

  if (propertyLifetime == expectedLifetime) return;

  property->setInvalidDecl();
  S.Diag(property->getLocation(),
         diag::err_arc_inconsistent_property_ownership)
    << property->getDeclName()
    << selector
    << propertyLifetime;
}

Decl *Sema::ActOnProperty(Scope *S, SourceLocation AtLoc,
                          FieldDeclarator &FD,
                          ObjCDeclSpec &ODS,
                          Selector GetterSel,
                          Selector SetterSel,
                          Decl *ClassCategory,
                          bool *isOverridingProperty,
                          tok::ObjCKeywordKind MethodImplKind,
                          DeclContext *lexicalDC) {
  unsigned Attributes = ODS.getPropertyAttributes();
  TypeSourceInfo *TSI = GetTypeForDeclarator(FD.D, S);
  QualType T = TSI->getType();
  if ((getLangOptions().getGCMode() != LangOptions::NonGC && 
       T.isObjCGCWeak()) ||
      (getLangOptions().ObjCAutoRefCount &&
       T.getObjCLifetime() == Qualifiers::OCL_Weak))
    Attributes |= ObjCDeclSpec::DQ_PR_weak;

  bool isReadWrite = ((Attributes & ObjCDeclSpec::DQ_PR_readwrite) ||
                      // default is readwrite!
                      !(Attributes & ObjCDeclSpec::DQ_PR_readonly));
  // property is defaulted to 'assign' if it is readwrite and is
  // not retain or copy
  bool isAssign = ((Attributes & ObjCDeclSpec::DQ_PR_assign) ||
                   (isReadWrite &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_retain) &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_strong) &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_copy) &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained) &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_weak)));

  // Proceed with constructing the ObjCPropertDecls.
  ObjCContainerDecl *ClassDecl =
    cast<ObjCContainerDecl>(ClassCategory);

  if (ObjCCategoryDecl *CDecl = dyn_cast<ObjCCategoryDecl>(ClassDecl))
    if (CDecl->IsClassExtension()) {
      Decl *Res = HandlePropertyInClassExtension(S, CDecl, AtLoc,
                                           FD, GetterSel, SetterSel,
                                           isAssign, isReadWrite,
                                           Attributes,
                                           isOverridingProperty, TSI,
                                           MethodImplKind);
      if (Res) {
        CheckObjCPropertyAttributes(Res, AtLoc, Attributes);
        if (getLangOptions().ObjCAutoRefCount)
          checkARCPropertyDecl(*this, cast<ObjCPropertyDecl>(Res));
      }
      return Res;
    }
  
  ObjCPropertyDecl *Res = CreatePropertyDecl(S, ClassDecl, AtLoc, FD,
                                             GetterSel, SetterSel,
                                             isAssign, isReadWrite,
                                             Attributes, TSI, MethodImplKind);
  if (lexicalDC)
    Res->setLexicalDeclContext(lexicalDC);

  // Validate the attributes on the @property.
  CheckObjCPropertyAttributes(Res, AtLoc, Attributes);

  if (getLangOptions().ObjCAutoRefCount)
    checkARCPropertyDecl(*this, Res);

  return Res;
}

Decl *
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
  ObjCInterfaceDecl *CCPrimary = CDecl->getClassInterface();
  
  if (CCPrimary)
    // Check for duplicate declaration of this property in current and
    // other class extensions.
    for (const ObjCCategoryDecl *ClsExtDecl = 
         CCPrimary->getFirstClassExtension();
         ClsExtDecl; ClsExtDecl = ClsExtDecl->getNextClassExtension()) {
      if (ObjCPropertyDecl *prevDecl =
          ObjCPropertyDecl::findPropertyDecl(ClsExtDecl, PropertyId)) {
        Diag(AtLoc, diag::err_duplicate_property);
        Diag(prevDecl->getLocation(), diag::note_property_declare);
        return 0;
      }
    }
  
  // Create a new ObjCPropertyDecl with the DeclContext being
  // the class extension.
  // FIXME. We should really be using CreatePropertyDecl for this.
  ObjCPropertyDecl *PDecl =
    ObjCPropertyDecl::Create(Context, DC, FD.D.getIdentifierLoc(),
                             PropertyId, AtLoc, T);
  if (Attributes & ObjCDeclSpec::DQ_PR_readonly)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readonly);
  if (Attributes & ObjCDeclSpec::DQ_PR_readwrite)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readwrite);
  // Set setter/getter selector name. Needed later.
  PDecl->setGetterName(GetterSel);
  PDecl->setSetterName(SetterSel);
  ProcessDeclAttributes(S, PDecl, FD.D);
  DC->addDecl(PDecl);

  // We need to look in the @interface to see if the @property was
  // already declared.
  if (!CCPrimary) {
    Diag(CDecl->getLocation(), diag::err_continuation_class);
    *isOverridingProperty = true;
    return 0;
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

    // A case of continuation class adding a new property in the class. This
    // is not what it was meant for. However, gcc supports it and so should we.
    // Make sure setter/getters are declared here.
    ProcessPropertyDecl(PDecl, CCPrimary, /* redeclaredProperty = */ 0,
                        /* lexicalDC = */ CDecl);
    return PDecl;
  }

  // The property 'PIDecl's readonly attribute will be over-ridden
  // with continuation class's readwrite property attribute!
  unsigned PIkind = PIDecl->getPropertyAttributesAsWritten();
  if (isReadWrite && (PIkind & ObjCPropertyDecl::OBJC_PR_readonly)) {
    unsigned retainCopyNonatomic =
    (ObjCPropertyDecl::OBJC_PR_retain |
     ObjCPropertyDecl::OBJC_PR_strong |
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

      Decl *ProtocolPtrTy =
        ActOnProperty(S, AtLoc, FD, ProtocolPropertyODS,
                      PIDecl->getGetterName(),
                      PIDecl->getSetterName(),
                      CCPrimary, isOverridingProperty,
                      MethodImplKind,
                      /* lexicalDC = */ CDecl);
      PIDecl = cast<ObjCPropertyDecl>(ProtocolPtrTy);
    }
    PIDecl->makeitReadWriteAttribute();
    if (Attributes & ObjCDeclSpec::DQ_PR_retain)
      PIDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_retain);
    if (Attributes & ObjCDeclSpec::DQ_PR_strong)
      PIDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_strong);
    if (Attributes & ObjCDeclSpec::DQ_PR_copy)
      PIDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_copy);
    PIDecl->setSetterName(SetterSel);
  } else {
    // Tailor the diagnostics for the common case where a readwrite
    // property is declared both in the @interface and the continuation.
    // This is a common error where the user often intended the original
    // declaration to be readonly.
    unsigned diag =
      (Attributes & ObjCDeclSpec::DQ_PR_readwrite) &&
      (PIkind & ObjCPropertyDecl::OBJC_PR_readwrite)
      ? diag::err_use_continuation_class_redeclaration_readwrite
      : diag::err_use_continuation_class;
    Diag(AtLoc, diag)
      << CCPrimary->getDeclName();
    Diag(PIDecl->getLocation(), diag::note_property_declare);
  }
  *isOverridingProperty = true;
  // Make sure setter decl is synthesized, and added to primary class's list.
  ProcessPropertyDecl(PIDecl, CCPrimary, PDecl, CDecl);
  return 0;
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

  unsigned attributesAsWritten = 0;
  if (Attributes & ObjCDeclSpec::DQ_PR_readonly)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_readonly;
  if (Attributes & ObjCDeclSpec::DQ_PR_readwrite)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_readwrite;
  if (Attributes & ObjCDeclSpec::DQ_PR_getter)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_getter;
  if (Attributes & ObjCDeclSpec::DQ_PR_setter)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_setter;
  if (Attributes & ObjCDeclSpec::DQ_PR_assign)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_assign;
  if (Attributes & ObjCDeclSpec::DQ_PR_retain)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_retain;
  if (Attributes & ObjCDeclSpec::DQ_PR_strong)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_strong;
  if (Attributes & ObjCDeclSpec::DQ_PR_weak)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_weak;
  if (Attributes & ObjCDeclSpec::DQ_PR_copy)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_copy;
  if (Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_unsafe_unretained;
  if (Attributes & ObjCDeclSpec::DQ_PR_nonatomic)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_nonatomic;
  if (Attributes & ObjCDeclSpec::DQ_PR_atomic)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_atomic;

  PDecl->setPropertyAttributesAsWritten(
                  (ObjCPropertyDecl::PropertyAttributeKind)attributesAsWritten);

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

  if (Attributes & ObjCDeclSpec::DQ_PR_strong)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_strong);

  if (Attributes & ObjCDeclSpec::DQ_PR_weak)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_weak);

  if (Attributes & ObjCDeclSpec::DQ_PR_copy)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_copy);

  if (Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_unsafe_unretained);

  if (isAssign)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_assign);

  if (Attributes & ObjCDeclSpec::DQ_PR_nonatomic)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_nonatomic);
  else if (Attributes & ObjCDeclSpec::DQ_PR_atomic)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_atomic);

  // 'unsafe_unretained' is alias for 'assign'.
  if (Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_assign);
  if (isAssign)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_unsafe_unretained);

  if (MethodImplKind == tok::objc_required)
    PDecl->setPropertyImplementation(ObjCPropertyDecl::Required);
  else if (MethodImplKind == tok::objc_optional)
    PDecl->setPropertyImplementation(ObjCPropertyDecl::Optional);

  return PDecl;
}

static void checkARCPropertyImpl(Sema &S, SourceLocation propertyImplLoc,
                                 ObjCPropertyDecl *property,
                                 ObjCIvarDecl *ivar) {
  if (property->isInvalidDecl() || ivar->isInvalidDecl()) return;

  QualType propertyType = property->getType();
  Qualifiers::ObjCLifetime propertyLifetime = propertyType.getObjCLifetime();
  ObjCPropertyDecl::PropertyAttributeKind propertyKind
    = property->getPropertyAttributes();

  QualType ivarType = ivar->getType();
  Qualifiers::ObjCLifetime ivarLifetime = ivarType.getObjCLifetime();
          
  // Case 1: strong properties.
  if (propertyLifetime == Qualifiers::OCL_Strong ||
      (propertyKind & (ObjCPropertyDecl::OBJC_PR_retain |
                       ObjCPropertyDecl::OBJC_PR_strong |
                       ObjCPropertyDecl::OBJC_PR_copy))) {
    switch (ivarLifetime) {
    case Qualifiers::OCL_Strong:
      // Okay.
      return;

    case Qualifiers::OCL_None:
    case Qualifiers::OCL_Autoreleasing:
      // These aren't valid lifetimes for object ivars;  don't diagnose twice.
      return;

    case Qualifiers::OCL_ExplicitNone:
    case Qualifiers::OCL_Weak:
      S.Diag(propertyImplLoc, diag::err_arc_strong_property_ownership)
        << property->getDeclName()
        << ivar->getDeclName()
        << ivarLifetime;
      break;
    }

  // Case 2: weak properties.
  } else if (propertyLifetime == Qualifiers::OCL_Weak ||
             (propertyKind & ObjCPropertyDecl::OBJC_PR_weak)) {
    switch (ivarLifetime) {
    case Qualifiers::OCL_Weak:
      // Okay.
      return;

    case Qualifiers::OCL_None:
    case Qualifiers::OCL_Autoreleasing:
      // These aren't valid lifetimes for object ivars;  don't diagnose twice.
      return;

    case Qualifiers::OCL_ExplicitNone:
    case Qualifiers::OCL_Strong:
      S.Diag(propertyImplLoc, diag::error_weak_property)
        << property->getDeclName()
        << ivar->getDeclName();
      break;
    }

  // Case 3: assign properties.
  } else if ((propertyKind & ObjCPropertyDecl::OBJC_PR_assign) &&
             propertyType->isObjCRetainableType()) {
    switch (ivarLifetime) {
    case Qualifiers::OCL_ExplicitNone:
      // Okay.
      return;

    case Qualifiers::OCL_None:
    case Qualifiers::OCL_Autoreleasing:
      // These aren't valid lifetimes for object ivars;  don't diagnose twice.
      return;

    case Qualifiers::OCL_Weak:
    case Qualifiers::OCL_Strong:
      S.Diag(propertyImplLoc, diag::err_arc_assign_property_ownership)
        << property->getDeclName()
        << ivar->getDeclName();
      break;
    }

  // Any other property should be ignored.
  } else {
    return;
  }

  S.Diag(property->getLocation(), diag::note_property_declare);
}


/// ActOnPropertyImplDecl - This routine performs semantic checks and
/// builds the AST node for a property implementation declaration; declared
/// as @synthesize or @dynamic.
///
Decl *Sema::ActOnPropertyImplDecl(Scope *S,
                                  SourceLocation AtLoc,
                                  SourceLocation PropertyLoc,
                                  bool Synthesize,
                                  Decl *ClassCatImpDecl,
                                  IdentifierInfo *PropertyId,
                                  IdentifierInfo *PropertyIvar,
                                  SourceLocation PropertyIvarLoc) {
  ObjCContainerDecl *ClassImpDecl =
    cast_or_null<ObjCContainerDecl>(ClassCatImpDecl);
  // Make sure we have a context for the property implementation declaration.
  if (!ClassImpDecl) {
    Diag(AtLoc, diag::error_missing_property_context);
    return 0;
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
      return 0;
    }
    unsigned PIkind = property->getPropertyAttributesAsWritten();
    if ((PIkind & (ObjCPropertyDecl::OBJC_PR_atomic |
                   ObjCPropertyDecl::OBJC_PR_nonatomic) ) == 0) {
      if (AtLoc.isValid())
        Diag(AtLoc, diag::warn_implicit_atomic_property);
      else
        Diag(IC->getLocation(), diag::warn_auto_implicit_atomic_property);
      Diag(property->getLocation(), diag::note_property_declare);
    }
    
    if (const ObjCCategoryDecl *CD =
        dyn_cast<ObjCCategoryDecl>(property->getDeclContext())) {
      if (!CD->IsClassExtension()) {
        Diag(PropertyLoc, diag::error_category_property) << CD->getDeclName();
        Diag(property->getLocation(), diag::note_property_declare);
        return 0;
      }
    }
  } else if ((CatImplClass = dyn_cast<ObjCCategoryImplDecl>(ClassImpDecl))) {
    if (Synthesize) {
      Diag(AtLoc, diag::error_synthesize_category_decl);
      return 0;
    }
    IDecl = CatImplClass->getClassInterface();
    if (!IDecl) {
      Diag(AtLoc, diag::error_missing_property_interface);
      return 0;
    }
    ObjCCategoryDecl *Category =
    IDecl->FindCategoryDeclaration(CatImplClass->getIdentifier());

    // If category for this implementation not found, it is an error which
    // has already been reported eralier.
    if (!Category)
      return 0;
    // Look for this property declaration in @implementation's category
    property = Category->FindPropertyDeclaration(PropertyId);
    if (!property) {
      Diag(PropertyLoc, diag::error_bad_category_property_decl)
      << Category->getDeclName();
      return 0;
    }
  } else {
    Diag(AtLoc, diag::error_bad_property_context);
    return 0;
  }
  ObjCIvarDecl *Ivar = 0;
  // Check that we have a valid, previously declared ivar for @synthesize
  if (Synthesize) {
    // @synthesize
    if (!PropertyIvar)
      PropertyIvar = PropertyId;
    ObjCPropertyDecl::PropertyAttributeKind kind 
      = property->getPropertyAttributes();
    QualType PropType = Context.getCanonicalType(property->getType());
    QualType PropertyIvarType = PropType;
    if (PropType->isReferenceType())
      PropertyIvarType = cast<ReferenceType>(PropType)->getPointeeType();
    // Check that this is a previously declared 'ivar' in 'IDecl' interface
    ObjCInterfaceDecl *ClassDeclared;
    Ivar = IDecl->lookupInstanceVariable(PropertyIvar, ClassDeclared);
    if (!Ivar) {
      // In ARC, give the ivar a lifetime qualifier based on its
      // property attributes.
      if (getLangOptions().ObjCAutoRefCount &&
          !PropertyIvarType.getObjCLifetime()) {

        if (!property->hasWrittenStorageAttribute() &&
            property->getType()->isObjCRetainableType()) {
          Diag(PropertyLoc,
               diag::err_arc_objc_property_default_assign_on_object);
          Diag(property->getLocation(), diag::note_property_declare);
        }

        // retain/copy have retaining lifetime.
        if (kind & (ObjCPropertyDecl::OBJC_PR_retain |
                    ObjCPropertyDecl::OBJC_PR_strong |
                    ObjCPropertyDecl::OBJC_PR_copy)) {
          Qualifiers qs;
          qs.addObjCLifetime(Qualifiers::OCL_Strong);
          PropertyIvarType = Context.getQualifiedType(PropertyIvarType, qs);
        }
        else if (kind & ObjCPropertyDecl::OBJC_PR_weak) {
          if (!getLangOptions().ObjCRuntimeHasWeak) {
            Diag(PropertyLoc, diag::err_arc_weak_no_runtime);
            Diag(property->getLocation(), diag::note_property_declare);
          }
          Qualifiers qs;
          qs.addObjCLifetime(Qualifiers::OCL_Weak);
          PropertyIvarType = Context.getQualifiedType(PropertyIvarType, qs);   
        }
        else if (kind & ObjCPropertyDecl::OBJC_PR_assign &&
                 PropertyIvarType->isObjCRetainableType()) {
            // assume that an 'assign' property synthesizes __unsafe_unretained
            // ivar
            Qualifiers qs;
            qs.addObjCLifetime(Qualifiers::OCL_ExplicitNone);
            PropertyIvarType = Context.getQualifiedType(PropertyIvarType, qs);  
        }
      }

      if (kind & ObjCPropertyDecl::OBJC_PR_weak &&
          !getLangOptions().ObjCAutoRefCount &&
          getLangOptions().getGCMode() == LangOptions::NonGC) {
        Diag(PropertyLoc, diag::error_synthesize_weak_non_arc_or_gc);
        Diag(property->getLocation(), diag::note_property_declare);
      }

      Ivar = ObjCIvarDecl::Create(Context, ClassImpDecl,
                                  PropertyLoc, PropertyLoc, PropertyIvar,
                                  PropertyIvarType, /*Dinfo=*/0,
                                  ObjCIvarDecl::Private,
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
    if (PropertyIvarType != IvarType) {
      bool compat = false;
      if (isa<ObjCObjectPointerType>(PropertyIvarType) 
          && isa<ObjCObjectPointerType>(IvarType))
        compat = 
          Context.canAssignObjCInterfaces(
                                  PropertyIvarType->getAs<ObjCObjectPointerType>(),
                                  IvarType->getAs<ObjCObjectPointerType>());
      else {
        SourceLocation Loc = PropertyIvarLoc;
        if (Loc.isInvalid())
          Loc = PropertyLoc;
        compat = (CheckAssignmentConstraints(Loc, PropertyIvarType, IvarType)
                    == Compatible);
      }
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
      QualType lhsType =Context.getCanonicalType(PropertyIvarType).getUnqualifiedType();
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
      if ((PropType.isObjCGCWeak() && !IvarType.isObjCGCWeak() &&
           getLangOptions().getGCMode() != LangOptions::NonGC)) {
        Diag(PropertyLoc, diag::error_weak_property)
        << property->getDeclName() << Ivar->getDeclName();
        // Fall thru - see previous comment
      }
      // Fall thru - see previous comment
      if ((property->getType()->isObjCObjectPointerType() ||
           PropType.isObjCGCStrong()) && IvarType.isObjCGCWeak() &&
          getLangOptions().getGCMode() != LangOptions::NonGC) {
        Diag(PropertyLoc, diag::error_strong_property)
        << property->getDeclName() << Ivar->getDeclName();
        // Fall thru - see previous comment
      }
    }
    if (getLangOptions().ObjCAutoRefCount)
      checkARCPropertyImpl(*this, PropertyLoc, property, Ivar);
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
                               Ivar, PropertyIvarLoc);
  if (ObjCMethodDecl *getterMethod = property->getGetterMethodDecl()) {
    getterMethod->createImplicitParams(Context, IDecl);
    if (getLangOptions().CPlusPlus && Synthesize &&
        Ivar->getType()->isRecordType()) {
      // For Objective-C++, need to synthesize the AST for the IVAR object to be
      // returned by the getter as it must conform to C++'s copy-return rules.
      // FIXME. Eventually we want to do this for Objective-C as well.
      ImplicitParamDecl *SelfDecl = getterMethod->getSelfDecl();
      DeclRefExpr *SelfExpr = 
        new (Context) DeclRefExpr(SelfDecl, SelfDecl->getType(),
                                  VK_RValue, SourceLocation());
      Expr *IvarRefExpr =
        new (Context) ObjCIvarRefExpr(Ivar, Ivar->getType(), AtLoc,
                                      SelfExpr, true, true);
      ExprResult Res = 
        PerformCopyInitialization(InitializedEntity::InitializeResult(
                                    SourceLocation(),
                                    getterMethod->getResultType(),
                                    /*NRVO=*/false),
                                  SourceLocation(),
                                  Owned(IvarRefExpr));
      if (!Res.isInvalid()) {
        Expr *ResExpr = Res.takeAs<Expr>();
        if (ResExpr)
          ResExpr = MaybeCreateExprWithCleanups(ResExpr);
        PIDecl->setGetterCXXConstructor(ResExpr);
      }
    }
    if (property->hasAttr<NSReturnsNotRetainedAttr>() &&
        !getterMethod->hasAttr<NSReturnsNotRetainedAttr>()) {
      Diag(getterMethod->getLocation(), 
           diag::warn_property_getter_owning_mismatch);
      Diag(property->getLocation(), diag::note_property_declare);
    }
  }
  if (ObjCMethodDecl *setterMethod = property->getSetterMethodDecl()) {
    setterMethod->createImplicitParams(Context, IDecl);
    if (getLangOptions().CPlusPlus && Synthesize
        && Ivar->getType()->isRecordType()) {
      // FIXME. Eventually we want to do this for Objective-C as well.
      ImplicitParamDecl *SelfDecl = setterMethod->getSelfDecl();
      DeclRefExpr *SelfExpr = 
        new (Context) DeclRefExpr(SelfDecl, SelfDecl->getType(),
                                  VK_RValue, SourceLocation());
      Expr *lhs =
        new (Context) ObjCIvarRefExpr(Ivar, Ivar->getType(), AtLoc,
                                      SelfExpr, true, true);
      ObjCMethodDecl::param_iterator P = setterMethod->param_begin();
      ParmVarDecl *Param = (*P);
      QualType T = Param->getType();
      if (T->isReferenceType())
        T = T->getAs<ReferenceType>()->getPointeeType();
      Expr *rhs = new (Context) DeclRefExpr(Param, T,
                                            VK_LValue, SourceLocation());
      ExprResult Res = BuildBinOp(S, lhs->getLocEnd(), 
                                  BO_Assign, lhs, rhs);
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
      return 0;
    }
    IC->addPropertyImplementation(PIDecl);
    if (getLangOptions().ObjCDefaultSynthProperties &&
        getLangOptions().ObjCNonFragileABI2) {
      // Diagnose if an ivar was lazily synthesdized due to a previous
      // use and if 1) property is @dynamic or 2) property is synthesized
      // but it requires an ivar of different name.
      ObjCInterfaceDecl *ClassDeclared=0;
      ObjCIvarDecl *Ivar = 0;
      if (!Synthesize)
        Ivar = IDecl->lookupInstanceVariable(PropertyId, ClassDeclared);
      else {
        if (PropertyIvar && PropertyIvar != PropertyId)
          Ivar = IDecl->lookupInstanceVariable(PropertyId, ClassDeclared);
      }
      // Issue diagnostics only if Ivar belongs to current class.
      if (Ivar && Ivar->getSynthesize() && 
          IC->getClassInterface() == ClassDeclared) {
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
      return 0;
    }
    CatImplClass->addPropertyImplementation(PIDecl);
  }

  return PIDecl;
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
  else {
    unsigned CAttrRetain = 
      (CAttr & 
       (ObjCPropertyDecl::OBJC_PR_retain | ObjCPropertyDecl::OBJC_PR_strong));
    unsigned SAttrRetain = 
      (SAttr & 
       (ObjCPropertyDecl::OBJC_PR_retain | ObjCPropertyDecl::OBJC_PR_strong));
    bool CStrong = (CAttrRetain != 0);
    bool SStrong = (SAttrRetain != 0);
    if (CStrong != SStrong)
      Diag(Property->getLocation(), diag::warn_property_attribute)
        << Property->getDeclName() << "retain (or strong)" << inheritedName;
  }

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

  if (!Context.propertyTypesAreCompatible(LHSType, RHSType)) {
    // Do cases not handled in above.
    // FIXME. For future support of covariant property types, revisit this.
    bool IncompatibleObjC = false;
    QualType ConvertedType;
    if (!isObjCPointerConversion(RHSType, LHSType, 
                                 ConvertedType, IncompatibleObjC) ||
        IncompatibleObjC)
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
      result = CheckAssignmentConstraints(Loc, GetterMethod->getResultType(),
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
/// diagnostics in a variety of inconsistent situations.
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
void Sema::CompareProperties(Decl *CDecl, Decl *ClassOrProtocol) {
  Decl *ClassDecl = ClassOrProtocol;
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
        CompareProperties(CatDecl, *P);
    } else {
      ObjCProtocolDecl *MD = cast<ObjCProtocolDecl>(ClassDecl);
      for (ObjCProtocolDecl::protocol_iterator P = MD->protocol_begin(),
           E = MD->protocol_end(); P != E; ++P)
        MatchOneProtocolPropertiesInClass(CatDecl, *P);
    }
    return;
  }

  if (ObjCInterfaceDecl *MDecl = dyn_cast<ObjCInterfaceDecl>(ClassDecl)) {
    for (ObjCInterfaceDecl::all_protocol_iterator
          P = MDecl->all_referenced_protocol_begin(),
          E = MDecl->all_referenced_protocol_end(); P != E; ++P)
      // Match properties of class IDecl with those of protocol (*P).
      MatchOneProtocolPropertiesInClass(IDecl, *P);

    // Go thru the list of protocols for this class and recursively match
    // their properties with those declared in the class.
    for (ObjCInterfaceDecl::all_protocol_iterator
          P = IDecl->all_referenced_protocol_begin(),
          E = IDecl->all_referenced_protocol_end(); P != E; ++P)
      CompareProperties(IDecl, *P);
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
    for (ObjCInterfaceDecl::all_protocol_iterator
         PI = IDecl->all_referenced_protocol_begin(),
         E = IDecl->all_referenced_protocol_end(); PI != E; ++PI)
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
    for (ObjCCategoryDecl::protocol_iterator PI = CATDecl->protocol_begin(),
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
    for (ObjCInterfaceDecl::all_protocol_iterator
         PI = IDecl->all_referenced_protocol_begin(),
         E = IDecl->all_referenced_protocol_end(); PI != E; ++PI)
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
    for (ObjCInterfaceDecl::all_protocol_iterator
         PI = IDecl->all_referenced_protocol_begin(),
         E = IDecl->all_referenced_protocol_end(); PI != E; ++PI) {
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
    if (IMPDecl->getInstanceMethod(Prop->getGetterName())) {
      if (Prop->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_readonly)
        continue;
      if (IMPDecl->getInstanceMethod(Prop->getSetterName()))
        continue;
    }


    // We use invalid SourceLocations for the synthesized ivars since they
    // aren't really synthesized at a particular location; they just exist.
    // Saying that they are located at the @implementation isn't really going
    // to help users.
    ActOnPropertyImplDecl(S, SourceLocation(), SourceLocation(),
                          true,IMPDecl,
                          Prop->getIdentifier(), Prop->getIdentifier(),
                          SourceLocation());
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
        PropImplMap.count(Prop) || Prop->hasAttr<UnavailableAttr>())
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
    ObjCMethodDecl *GetterMethod = 0;
    ObjCMethodDecl *SetterMethod = 0;
    bool LookedUpGetterSetter = false;

    unsigned Attributes = Property->getPropertyAttributes();
    unsigned AttributesAsWrittern = Property->getPropertyAttributesAsWritten();

    if (!(AttributesAsWrittern & ObjCPropertyDecl::OBJC_PR_atomic) &&
        !(AttributesAsWrittern & ObjCPropertyDecl::OBJC_PR_nonatomic)) {
      GetterMethod = IMPDecl->getInstanceMethod(Property->getGetterName());
      SetterMethod = IMPDecl->getInstanceMethod(Property->getSetterName());
      LookedUpGetterSetter = true;
      if (GetterMethod) {
        Diag(GetterMethod->getLocation(),
             diag::warn_default_atomic_custom_getter_setter)
          << Property->getIdentifier() << 0;
        Diag(Property->getLocation(), diag::note_property_declare);
      }
      if (SetterMethod) {
        Diag(SetterMethod->getLocation(),
             diag::warn_default_atomic_custom_getter_setter)
          << Property->getIdentifier() << 1;
        Diag(Property->getLocation(), diag::note_property_declare);
      }
    }

    // We only care about readwrite atomic property.
    if ((Attributes & ObjCPropertyDecl::OBJC_PR_nonatomic) ||
        !(Attributes & ObjCPropertyDecl::OBJC_PR_readwrite))
      continue;
    if (const ObjCPropertyImplDecl *PIDecl
         = IMPDecl->FindPropertyImplDecl(Property->getIdentifier())) {
      if (PIDecl->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic)
        continue;
      if (!LookedUpGetterSetter) {
        GetterMethod = IMPDecl->getInstanceMethod(Property->getGetterName());
        SetterMethod = IMPDecl->getInstanceMethod(Property->getSetterName());
        LookedUpGetterSetter = true;
      }
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

void Sema::DiagnoseOwningPropertyGetterSynthesis(const ObjCImplementationDecl *D) {
  if (getLangOptions().getGCMode() == LangOptions::GCOnly)
    return;

  for (ObjCImplementationDecl::propimpl_iterator
         i = D->propimpl_begin(), e = D->propimpl_end(); i != e; ++i) {
    ObjCPropertyImplDecl *PID = *i;
    if (PID->getPropertyImplementation() != ObjCPropertyImplDecl::Synthesize)
      continue;
    
    const ObjCPropertyDecl *PD = PID->getPropertyDecl();
    if (PD && !PD->hasAttr<NSReturnsNotRetainedAttr>() &&
        !D->getInstanceMethod(PD->getGetterName())) {
      ObjCMethodDecl *method = PD->getGetterMethodDecl();
      if (!method)
        continue;
      ObjCMethodFamily family = method->getMethodFamily();
      if (family == OMF_alloc || family == OMF_copy ||
          family == OMF_mutableCopy || family == OMF_new) {
        if (getLangOptions().ObjCAutoRefCount)
          Diag(PID->getLocation(), diag::err_ownin_getter_rule);
        else
          Diag(PID->getLocation(), diag::warn_ownin_getter_rule);
        Diag(PD->getLocation(), diag::note_property_declare);
      }
    }
  }
}

/// AddPropertyAttrs - Propagates attributes from a property to the
/// implicitly-declared getter or setter for that property.
static void AddPropertyAttrs(Sema &S, ObjCMethodDecl *PropertyMethod,
                             ObjCPropertyDecl *Property) {
  // Should we just clone all attributes over?
  for (Decl::attr_iterator A = Property->attr_begin(), 
                        AEnd = Property->attr_end(); 
       A != AEnd; ++A) {
    if (isa<DeprecatedAttr>(*A) || 
        isa<UnavailableAttr>(*A) || 
        isa<AvailabilityAttr>(*A))
      PropertyMethod->addAttr((*A)->clone(S.Context));
  }
}

/// ProcessPropertyDecl - Make sure that any user-defined setter/getter methods
/// have the property type and issue diagnostics if they don't.
/// Also synthesize a getter/setter method if none exist (and update the
/// appropriate lookup tables. FIXME: Should reconsider if adding synthesized
/// methods is the "right" thing to do.
void Sema::ProcessPropertyDecl(ObjCPropertyDecl *property,
                               ObjCContainerDecl *CD,
                               ObjCPropertyDecl *redeclaredProperty,
                               ObjCContainerDecl *lexicalDC) {

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
    SourceLocation Loc = redeclaredProperty ? 
      redeclaredProperty->getLocation() :
      property->getLocation();

    GetterMethod = ObjCMethodDecl::Create(Context, Loc, Loc,
                             property->getGetterName(),
                             property->getType(), 0, CD, true, false, true, 
                             false,
                             (property->getPropertyImplementation() ==
                              ObjCPropertyDecl::Optional) ?
                             ObjCMethodDecl::Optional :
                             ObjCMethodDecl::Required);
    CD->addDecl(GetterMethod);

    AddPropertyAttrs(*this, GetterMethod, property);

    // FIXME: Eventually this shouldn't be needed, as the lexical context
    // and the real context should be the same.
    if (lexicalDC)
      GetterMethod->setLexicalDeclContext(lexicalDC);
    if (property->hasAttr<NSReturnsNotRetainedAttr>())
      GetterMethod->addAttr(
        ::new (Context) NSReturnsNotRetainedAttr(Loc, Context));
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
      SourceLocation Loc = redeclaredProperty ? 
        redeclaredProperty->getLocation() :
        property->getLocation();

      SetterMethod =
        ObjCMethodDecl::Create(Context, Loc, Loc,
                               property->getSetterName(), Context.VoidTy, 0,
                               CD, true, false, true, false,
                               (property->getPropertyImplementation() ==
                                ObjCPropertyDecl::Optional) ?
                                ObjCMethodDecl::Optional :
                                ObjCMethodDecl::Required);

      // Invent the arguments for the setter. We don't bother making a
      // nice name for the argument.
      ParmVarDecl *Argument = ParmVarDecl::Create(Context, SetterMethod,
                                                  Loc, Loc,
                                                  property->getIdentifier(),
                                    property->getType().getUnqualifiedType(),
                                                  /*TInfo=*/0,
                                                  SC_None,
                                                  SC_None,
                                                  0);
      SetterMethod->setMethodParams(Context, &Argument, 1, 1);

      AddPropertyAttrs(*this, SetterMethod, property);

      CD->addDecl(SetterMethod);
      // FIXME: Eventually this shouldn't be needed, as the lexical context
      // and the real context should be the same.
      if (lexicalDC)
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

void Sema::CheckObjCPropertyAttributes(Decl *PDecl,
                                       SourceLocation Loc,
                                       unsigned &Attributes) {
  // FIXME: Improve the reported location.
  if (!PDecl || PDecl->isInvalidDecl())
    return;

  ObjCPropertyDecl *PropertyDecl = cast<ObjCPropertyDecl>(PDecl);
  QualType PropertyTy = PropertyDecl->getType(); 

  // readonly and readwrite/assign/retain/copy conflict.
  if ((Attributes & ObjCDeclSpec::DQ_PR_readonly) &&
      (Attributes & (ObjCDeclSpec::DQ_PR_readwrite |
                     ObjCDeclSpec::DQ_PR_assign |
                     ObjCDeclSpec::DQ_PR_unsafe_unretained |
                     ObjCDeclSpec::DQ_PR_copy |
                     ObjCDeclSpec::DQ_PR_retain |
                     ObjCDeclSpec::DQ_PR_strong))) {
    const char * which = (Attributes & ObjCDeclSpec::DQ_PR_readwrite) ?
                          "readwrite" :
                         (Attributes & ObjCDeclSpec::DQ_PR_assign) ?
                          "assign" :
                         (Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained) ?
                          "unsafe_unretained" :
                         (Attributes & ObjCDeclSpec::DQ_PR_copy) ?
                          "copy" : "retain";

    Diag(Loc, (Attributes & (ObjCDeclSpec::DQ_PR_readwrite)) ?
                 diag::err_objc_property_attr_mutually_exclusive :
                 diag::warn_objc_property_attr_mutually_exclusive)
      << "readonly" << which;
  }

  // Check for copy or retain on non-object types.
  if ((Attributes & (ObjCDeclSpec::DQ_PR_weak | ObjCDeclSpec::DQ_PR_copy |
                    ObjCDeclSpec::DQ_PR_retain | ObjCDeclSpec::DQ_PR_strong)) &&
      !PropertyTy->isObjCRetainableType() &&
      !PropertyDecl->getAttr<ObjCNSObjectAttr>()) {
    Diag(Loc, diag::err_objc_property_requires_object)
      << (Attributes & ObjCDeclSpec::DQ_PR_weak ? "weak" :
          Attributes & ObjCDeclSpec::DQ_PR_copy ? "copy" : "retain (or strong)");
    Attributes &= ~(ObjCDeclSpec::DQ_PR_weak   | ObjCDeclSpec::DQ_PR_copy |
                    ObjCDeclSpec::DQ_PR_retain | ObjCDeclSpec::DQ_PR_strong);
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
    if (Attributes & ObjCDeclSpec::DQ_PR_strong) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "assign" << "strong";
      Attributes &= ~ObjCDeclSpec::DQ_PR_strong;
    }
    if (getLangOptions().ObjCAutoRefCount  &&
        (Attributes & ObjCDeclSpec::DQ_PR_weak)) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "assign" << "weak";
      Attributes &= ~ObjCDeclSpec::DQ_PR_weak;
    }
  } else if (Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained) {
    if (Attributes & ObjCDeclSpec::DQ_PR_copy) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "unsafe_unretained" << "copy";
      Attributes &= ~ObjCDeclSpec::DQ_PR_copy;
    }
    if (Attributes & ObjCDeclSpec::DQ_PR_retain) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "unsafe_unretained" << "retain";
      Attributes &= ~ObjCDeclSpec::DQ_PR_retain;
    }
    if (Attributes & ObjCDeclSpec::DQ_PR_strong) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "unsafe_unretained" << "strong";
      Attributes &= ~ObjCDeclSpec::DQ_PR_strong;
    }
    if (getLangOptions().ObjCAutoRefCount  &&
        (Attributes & ObjCDeclSpec::DQ_PR_weak)) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "unsafe_unretained" << "weak";
      Attributes &= ~ObjCDeclSpec::DQ_PR_weak;
    }
  } else if (Attributes & ObjCDeclSpec::DQ_PR_copy) {
    if (Attributes & ObjCDeclSpec::DQ_PR_retain) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "copy" << "retain";
      Attributes &= ~ObjCDeclSpec::DQ_PR_retain;
    }
    if (Attributes & ObjCDeclSpec::DQ_PR_strong) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "copy" << "strong";
      Attributes &= ~ObjCDeclSpec::DQ_PR_strong;
    }
    if (Attributes & ObjCDeclSpec::DQ_PR_weak) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "copy" << "weak";
      Attributes &= ~ObjCDeclSpec::DQ_PR_weak;
    }
  }
  else if ((Attributes & ObjCDeclSpec::DQ_PR_retain) &&
           (Attributes & ObjCDeclSpec::DQ_PR_weak)) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "retain" << "weak";
      Attributes &= ~ObjCDeclSpec::DQ_PR_weak;
  }
  else if ((Attributes & ObjCDeclSpec::DQ_PR_strong) &&
           (Attributes & ObjCDeclSpec::DQ_PR_weak)) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "strong" << "weak";
      Attributes &= ~ObjCDeclSpec::DQ_PR_weak;
  }

  // Warn if user supplied no assignment attribute, property is
  // readwrite, and this is an object type.
  if (!(Attributes & (ObjCDeclSpec::DQ_PR_assign | ObjCDeclSpec::DQ_PR_copy |
                      ObjCDeclSpec::DQ_PR_unsafe_unretained |
                      ObjCDeclSpec::DQ_PR_retain | ObjCDeclSpec::DQ_PR_strong |
                      ObjCDeclSpec::DQ_PR_weak)) &&
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
      &&!(Attributes & ObjCDeclSpec::DQ_PR_readonly)
      && getLangOptions().getGCMode() == LangOptions::GCOnly
      && PropertyTy->isBlockPointerType())
    Diag(Loc, diag::warn_objc_property_copy_missing_on_block);
}
