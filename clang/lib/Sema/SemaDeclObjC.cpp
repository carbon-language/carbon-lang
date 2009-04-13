//===--- SemaDeclObjC.cpp - Semantic Analysis for ObjC Declarations -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for Objective C declarations.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Parse/DeclSpec.h"
using namespace clang;

/// ActOnStartOfObjCMethodDef - This routine sets up parameters; invisible
/// and user declared, in the method definition's AST.
void Sema::ActOnStartOfObjCMethodDef(Scope *FnBodyScope, DeclPtrTy D) {
  assert(getCurMethodDecl() == 0 && "Method parsing confused");
  ObjCMethodDecl *MDecl = dyn_cast_or_null<ObjCMethodDecl>(D.getAs<Decl>());
  
  // If we don't have a valid method decl, simply return.
  if (!MDecl)
    return;

  // Allow the rest of sema to find private method decl implementations.
  if (MDecl->isInstanceMethod())
    AddInstanceMethodToGlobalPool(MDecl);
  else
    AddFactoryMethodToGlobalPool(MDecl);
  
  // Allow all of Sema to see that we are entering a method definition.
  PushDeclContext(FnBodyScope, MDecl);

  // Create Decl objects for each parameter, entrring them in the scope for
  // binding to their use.

  // Insert the invisible arguments, self and _cmd!
  MDecl->createImplicitParams(Context, MDecl->getClassInterface());
  
  PushOnScopeChains(MDecl->getSelfDecl(), FnBodyScope);
  PushOnScopeChains(MDecl->getCmdDecl(), FnBodyScope);

  // Introduce all of the other parameters into this scope.
  for (ObjCMethodDecl::param_iterator PI = MDecl->param_begin(),
       E = MDecl->param_end(); PI != E; ++PI)
    if ((*PI)->getIdentifier())
      PushOnScopeChains(*PI, FnBodyScope);
}

Sema::DeclPtrTy Sema::
ActOnStartClassInterface(SourceLocation AtInterfaceLoc,
                         IdentifierInfo *ClassName, SourceLocation ClassLoc,
                         IdentifierInfo *SuperName, SourceLocation SuperLoc,
                         const DeclPtrTy *ProtoRefs, unsigned NumProtoRefs,
                         SourceLocation EndProtoLoc, AttributeList *AttrList) {
  assert(ClassName && "Missing class identifier");
  
  // Check for another declaration kind with the same name.
  NamedDecl *PrevDecl = LookupName(TUScope, ClassName, LookupOrdinaryName);
  if (PrevDecl && PrevDecl->isTemplateParameter()) {
    // Maybe we will complain about the shadowed template parameter.
    DiagnoseTemplateParameterShadow(ClassLoc, PrevDecl);
    // Just pretend that we didn't see the previous declaration.
    PrevDecl = 0;
  }

  if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
    Diag(ClassLoc, diag::err_redefinition_different_kind) << ClassName;
    Diag(PrevDecl->getLocation(), diag::note_previous_definition);
  }
  
  ObjCInterfaceDecl* IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl);
  if (IDecl) {
    // Class already seen. Is it a forward declaration?
    if (!IDecl->isForwardDecl()) {
      IDecl->setInvalidDecl();
      Diag(AtInterfaceLoc, diag::err_duplicate_class_def)<<IDecl->getDeclName();
      Diag(IDecl->getLocation(), diag::note_previous_definition);

      // Return the previous class interface.
      // FIXME: don't leak the objects passed in!
      return DeclPtrTy::make(IDecl);
    } else {
      IDecl->setLocation(AtInterfaceLoc);
      IDecl->setForwardDecl(false);
    }
  } else {
    IDecl = ObjCInterfaceDecl::Create(Context, CurContext, AtInterfaceLoc, 
                                      ClassName, ClassLoc);
    if (AttrList)
      ProcessDeclAttributeList(IDecl, AttrList);
  
    ObjCInterfaceDecls[ClassName] = IDecl;
    // FIXME: PushOnScopeChains
    CurContext->addDecl(Context, IDecl);
    // Remember that this needs to be removed when the scope is popped.
    TUScope->AddDecl(DeclPtrTy::make(IDecl));
  }
  
  if (SuperName) {
    // Check if a different kind of symbol declared in this scope.
    PrevDecl = LookupName(TUScope, SuperName, LookupOrdinaryName);

    ObjCInterfaceDecl *SuperClassDecl = 
                                  dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl);

    // Diagnose classes that inherit from deprecated classes.
    if (SuperClassDecl)
      (void)DiagnoseUseOfDecl(SuperClassDecl, SuperLoc);
    
    if (PrevDecl && SuperClassDecl == 0) {
      // The previous declaration was not a class decl. Check if we have a
      // typedef. If we do, get the underlying class type.
      if (const TypedefDecl *TDecl = dyn_cast_or_null<TypedefDecl>(PrevDecl)) {
        QualType T = TDecl->getUnderlyingType();
        if (T->isObjCInterfaceType()) {
          if (NamedDecl *IDecl = T->getAsObjCInterfaceType()->getDecl())
            SuperClassDecl = dyn_cast<ObjCInterfaceDecl>(IDecl);
        }
      }
      
      // This handles the following case:
      //
      // typedef int SuperClass;
      // @interface MyClass : SuperClass {} @end
      //
      if (!SuperClassDecl) {
        Diag(SuperLoc, diag::err_redefinition_different_kind) << SuperName;
        Diag(PrevDecl->getLocation(), diag::note_previous_definition);
      }
    }
    
    if (!dyn_cast_or_null<TypedefDecl>(PrevDecl)) {
      if (!SuperClassDecl)
        Diag(SuperLoc, diag::err_undef_superclass)
          << SuperName << ClassName << SourceRange(AtInterfaceLoc, ClassLoc);
      else if (SuperClassDecl->isForwardDecl())
        Diag(SuperLoc, diag::err_undef_superclass)
          << SuperClassDecl->getDeclName() << ClassName
          << SourceRange(AtInterfaceLoc, ClassLoc);
    }
    IDecl->setSuperClass(SuperClassDecl);
    IDecl->setSuperClassLoc(SuperLoc);
    IDecl->setLocEnd(SuperLoc);
  } else { // we have a root class.
    IDecl->setLocEnd(ClassLoc);
  }
  
  /// Check then save referenced protocols.
  if (NumProtoRefs) {
    IDecl->setProtocolList((ObjCProtocolDecl**)ProtoRefs, NumProtoRefs,
                           Context);
    IDecl->setLocEnd(EndProtoLoc);
  }
  
  CheckObjCDeclScope(IDecl);
  return DeclPtrTy::make(IDecl);
}

/// ActOnCompatiblityAlias - this action is called after complete parsing of
/// @compatibility_alias declaration. It sets up the alias relationships.
Sema::DeclPtrTy Sema::ActOnCompatiblityAlias(SourceLocation AtLoc,
                                             IdentifierInfo *AliasName, 
                                             SourceLocation AliasLocation,
                                             IdentifierInfo *ClassName,
                                             SourceLocation ClassLocation) {
  // Look for previous declaration of alias name
  NamedDecl *ADecl = LookupName(TUScope, AliasName, LookupOrdinaryName);
  if (ADecl) {
    if (isa<ObjCCompatibleAliasDecl>(ADecl))
      Diag(AliasLocation, diag::warn_previous_alias_decl);
    else
      Diag(AliasLocation, diag::err_conflicting_aliasing_type) << AliasName;
    Diag(ADecl->getLocation(), diag::note_previous_declaration);
    return DeclPtrTy();
  }
  // Check for class declaration
  NamedDecl *CDeclU = LookupName(TUScope, ClassName, LookupOrdinaryName);
  if (const TypedefDecl *TDecl = dyn_cast_or_null<TypedefDecl>(CDeclU)) {
    QualType T = TDecl->getUnderlyingType();
    if (T->isObjCInterfaceType()) {
      if (NamedDecl *IDecl = T->getAsObjCInterfaceType()->getDecl()) {
        ClassName = IDecl->getIdentifier();
        CDeclU = LookupName(TUScope, ClassName, LookupOrdinaryName);
      }
    }
  }
  ObjCInterfaceDecl *CDecl = dyn_cast_or_null<ObjCInterfaceDecl>(CDeclU);
  if (CDecl == 0) {
    Diag(ClassLocation, diag::warn_undef_interface) << ClassName;
    if (CDeclU)
      Diag(CDeclU->getLocation(), diag::note_previous_declaration);
    return DeclPtrTy();
  }
  
  // Everything checked out, instantiate a new alias declaration AST.
  ObjCCompatibleAliasDecl *AliasDecl = 
    ObjCCompatibleAliasDecl::Create(Context, CurContext, AtLoc, AliasName, CDecl);
  
  ObjCAliasDecls[AliasName] = AliasDecl;

  // FIXME: PushOnScopeChains?
  CurContext->addDecl(Context, AliasDecl);
  if (!CheckObjCDeclScope(AliasDecl))
    TUScope->AddDecl(DeclPtrTy::make(AliasDecl));

  return DeclPtrTy::make(AliasDecl);
}

void Sema::CheckForwardProtocolDeclarationForCircularDependency(
  IdentifierInfo *PName,
  SourceLocation &Ploc, SourceLocation PrevLoc,
  const ObjCList<ObjCProtocolDecl> &PList) 
{
  for (ObjCList<ObjCProtocolDecl>::iterator I = PList.begin(),
       E = PList.end(); I != E; ++I) {
       
    if (ObjCProtocolDecl *PDecl = ObjCProtocols[(*I)->getIdentifier()]) {
      if (PDecl->getIdentifier() == PName) {
        Diag(Ploc, diag::err_protocol_has_circular_dependency);
        Diag(PrevLoc, diag::note_previous_definition);
      }
      CheckForwardProtocolDeclarationForCircularDependency(PName, Ploc, 
        PDecl->getLocation(), PDecl->getReferencedProtocols());
    }
  }
}

Sema::DeclPtrTy
Sema::ActOnStartProtocolInterface(SourceLocation AtProtoInterfaceLoc,
                                  IdentifierInfo *ProtocolName,
                                  SourceLocation ProtocolLoc,
                                  const DeclPtrTy *ProtoRefs,
                                  unsigned NumProtoRefs,
                                  SourceLocation EndProtoLoc,
                                  AttributeList *AttrList) {
  // FIXME: Deal with AttrList.
  assert(ProtocolName && "Missing protocol identifier");
  ObjCProtocolDecl *PDecl = ObjCProtocols[ProtocolName];
  if (PDecl) {
    // Protocol already seen. Better be a forward protocol declaration
    if (!PDecl->isForwardDecl()) {
      Diag(ProtocolLoc, diag::warn_duplicate_protocol_def) << ProtocolName;
      Diag(PDecl->getLocation(), diag::note_previous_definition);
      // Just return the protocol we already had.
      // FIXME: don't leak the objects passed in!
      return DeclPtrTy::make(PDecl);
    }
    ObjCList<ObjCProtocolDecl> PList;
    PList.set((ObjCProtocolDecl *const*)ProtoRefs, NumProtoRefs, Context); 
    CheckForwardProtocolDeclarationForCircularDependency(
      ProtocolName, ProtocolLoc, PDecl->getLocation(), PList);
    PList.Destroy(Context);
    
    // Make sure the cached decl gets a valid start location.
    PDecl->setLocation(AtProtoInterfaceLoc);
    PDecl->setForwardDecl(false);
  } else {
    PDecl = ObjCProtocolDecl::Create(Context, CurContext, 
                                     AtProtoInterfaceLoc,ProtocolName);
    // FIXME: PushOnScopeChains?
    CurContext->addDecl(Context, PDecl);
    PDecl->setForwardDecl(false);
    ObjCProtocols[ProtocolName] = PDecl;
  }
  if (AttrList)
    ProcessDeclAttributeList(PDecl, AttrList);
  if (NumProtoRefs) {
    /// Check then save referenced protocols.
    PDecl->setProtocolList((ObjCProtocolDecl**)ProtoRefs, NumProtoRefs,Context);
    PDecl->setLocEnd(EndProtoLoc);
  }
  
  CheckObjCDeclScope(PDecl);  
  return DeclPtrTy::make(PDecl);
}

/// FindProtocolDeclaration - This routine looks up protocols and
/// issues an error if they are not declared. It returns list of
/// protocol declarations in its 'Protocols' argument.
void
Sema::FindProtocolDeclaration(bool WarnOnDeclarations,
                              const IdentifierLocPair *ProtocolId,
                              unsigned NumProtocols,
                              llvm::SmallVectorImpl<DeclPtrTy> &Protocols) {
  for (unsigned i = 0; i != NumProtocols; ++i) {
    ObjCProtocolDecl *PDecl = ObjCProtocols[ProtocolId[i].first];
    if (!PDecl) {
      Diag(ProtocolId[i].second, diag::err_undeclared_protocol)
        << ProtocolId[i].first;
      continue;
    }
    
    (void)DiagnoseUseOfDecl(PDecl, ProtocolId[i].second);

    // If this is a forward declaration and we are supposed to warn in this
    // case, do it.
    if (WarnOnDeclarations && PDecl->isForwardDecl())
      Diag(ProtocolId[i].second, diag::warn_undef_protocolref)
        << ProtocolId[i].first;
    Protocols.push_back(DeclPtrTy::make(PDecl));
  }
}

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
      if (ObjCQualifiedIdTypesAreCompatible(LHSType, RHSType, false))
        return;
    Diag(Property->getLocation(), diag::warn_property_types_are_incompatible)
      << Property->getType() << SuperProperty->getType() << inheritedName;
  }
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
  for (ObjCInterfaceDecl::prop_iterator S = SDecl->prop_begin(Context),
       E = SDecl->prop_end(Context); S != E; ++S) {
    ObjCPropertyDecl *SuperPDecl = (*S);
    // Does property in super class has declaration in current class?
    for (ObjCInterfaceDecl::prop_iterator I = IDecl->prop_begin(Context),
         E = IDecl->prop_end(Context); I != E; ++I) {
      ObjCPropertyDecl *PDecl = (*I);
      if (SuperPDecl->getIdentifier() == PDecl->getIdentifier())
          DiagnosePropertyMismatch(PDecl, SuperPDecl, 
                                   SDecl->getIdentifier());
    }
  }
}

/// MergeOneProtocolPropertiesIntoClass - This routine goes thru the list
/// of properties declared in a protocol and adds them to the list
/// of properties for current class/category if it is not there already.
void
Sema::MergeOneProtocolPropertiesIntoClass(Decl *CDecl,
                                          ObjCProtocolDecl *PDecl) {
  ObjCInterfaceDecl *IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(CDecl);
  if (!IDecl) {
    // Category
    ObjCCategoryDecl *CatDecl = static_cast<ObjCCategoryDecl*>(CDecl);
    assert (CatDecl && "MergeOneProtocolPropertiesIntoClass");
    for (ObjCProtocolDecl::prop_iterator P = PDecl->prop_begin(Context),
         E = PDecl->prop_end(Context); P != E; ++P) {
      ObjCPropertyDecl *Pr = (*P);
      ObjCCategoryDecl::prop_iterator CP, CE;
      // Is this property already in  category's list of properties?
      for (CP = CatDecl->prop_begin(Context), CE = CatDecl->prop_end(Context); 
           CP != CE; ++CP)
        if ((*CP)->getIdentifier() == Pr->getIdentifier())
          break;
      if (CP != CE)
        // Property protocol already exist in class. Diagnose any mismatch.
        DiagnosePropertyMismatch((*CP), Pr, PDecl->getIdentifier());
    }
    return;
  }
  for (ObjCProtocolDecl::prop_iterator P = PDecl->prop_begin(Context),
       E = PDecl->prop_end(Context); P != E; ++P) {
    ObjCPropertyDecl *Pr = (*P);
    ObjCInterfaceDecl::prop_iterator CP, CE;
    // Is this property already in  class's list of properties?
    for (CP = IDecl->prop_begin(Context), CE = IDecl->prop_end(Context); 
         CP != CE; ++CP)
      if ((*CP)->getIdentifier() == Pr->getIdentifier())
        break;
    if (CP != CE)
      // Property protocol already exist in class. Diagnose any mismatch.
      DiagnosePropertyMismatch((*CP), Pr, PDecl->getIdentifier());
    }
}

/// MergeProtocolPropertiesIntoClass - This routine merges properties
/// declared in 'MergeItsProtocols' objects (which can be a class or an
/// inherited protocol into the list of properties for class/category 'CDecl'
///
void Sema::MergeProtocolPropertiesIntoClass(Decl *CDecl,
                                            DeclPtrTy MergeItsProtocols) {
  Decl *ClassDecl = MergeItsProtocols.getAs<Decl>();
  ObjCInterfaceDecl *IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(CDecl);

  if (!IDecl) {
    // Category
    ObjCCategoryDecl *CatDecl = static_cast<ObjCCategoryDecl*>(CDecl);
    assert (CatDecl && "MergeProtocolPropertiesIntoClass");
    if (ObjCCategoryDecl *MDecl = dyn_cast<ObjCCategoryDecl>(ClassDecl)) {
      for (ObjCCategoryDecl::protocol_iterator P = MDecl->protocol_begin(),
           E = MDecl->protocol_end(); P != E; ++P)
      // Merge properties of category (*P) into IDECL's
      MergeOneProtocolPropertiesIntoClass(CatDecl, *P);
    
      // Go thru the list of protocols for this category and recursively merge
      // their properties into this class as well.
      for (ObjCCategoryDecl::protocol_iterator P = CatDecl->protocol_begin(),
           E = CatDecl->protocol_end(); P != E; ++P)
        MergeProtocolPropertiesIntoClass(CatDecl, DeclPtrTy::make(*P));
    } else {
      ObjCProtocolDecl *MD = cast<ObjCProtocolDecl>(ClassDecl);
      for (ObjCProtocolDecl::protocol_iterator P = MD->protocol_begin(),
           E = MD->protocol_end(); P != E; ++P)
        MergeOneProtocolPropertiesIntoClass(CatDecl, *P);
    }
    return;
  }

  if (ObjCInterfaceDecl *MDecl = dyn_cast<ObjCInterfaceDecl>(ClassDecl)) {
    for (ObjCInterfaceDecl::protocol_iterator P = MDecl->protocol_begin(),
         E = MDecl->protocol_end(); P != E; ++P)
      // Merge properties of class (*P) into IDECL's
      MergeOneProtocolPropertiesIntoClass(IDecl, *P);
    
    // Go thru the list of protocols for this class and recursively merge
    // their properties into this class as well.
    for (ObjCInterfaceDecl::protocol_iterator P = IDecl->protocol_begin(),
         E = IDecl->protocol_end(); P != E; ++P)
      MergeProtocolPropertiesIntoClass(IDecl, DeclPtrTy::make(*P));
  } else {
    ObjCProtocolDecl *MD = cast<ObjCProtocolDecl>(ClassDecl);
    for (ObjCProtocolDecl::protocol_iterator P = MD->protocol_begin(),
         E = MD->protocol_end(); P != E; ++P)
      MergeOneProtocolPropertiesIntoClass(IDecl, *P);
  }
}

/// DiagnoseClassExtensionDupMethods - Check for duplicate declaration of
/// a class method in its extension.
///
void Sema::DiagnoseClassExtensionDupMethods(ObjCCategoryDecl *CAT, 
                                            ObjCInterfaceDecl *ID) {
  if (!ID)
    return;  // Possibly due to previous error

  llvm::DenseMap<Selector, const ObjCMethodDecl*> MethodMap;
  for (ObjCInterfaceDecl::method_iterator i = ID->meth_begin(Context),
       e =  ID->meth_end(Context); i != e; ++i) {
    ObjCMethodDecl *MD = *i;
    MethodMap[MD->getSelector()] = MD;
  }

  if (MethodMap.empty())
    return;
  for (ObjCCategoryDecl::method_iterator i = CAT->meth_begin(Context),
       e =  CAT->meth_end(Context); i != e; ++i) {
    ObjCMethodDecl *Method = *i;
    const ObjCMethodDecl *&PrevMethod = MethodMap[Method->getSelector()];
    if (PrevMethod && !MatchTwoMethodDeclarations(Method, PrevMethod)) {
      Diag(Method->getLocation(), diag::err_duplicate_method_decl)
            << Method->getDeclName();
      Diag(PrevMethod->getLocation(), diag::note_previous_declaration);
    }
  }
}

/// ActOnForwardProtocolDeclaration - Handle @protocol foo;
Action::DeclPtrTy
Sema::ActOnForwardProtocolDeclaration(SourceLocation AtProtocolLoc,
                                      const IdentifierLocPair *IdentList,
                                      unsigned NumElts,
                                      AttributeList *attrList) {
  llvm::SmallVector<ObjCProtocolDecl*, 32> Protocols;
  
  for (unsigned i = 0; i != NumElts; ++i) {
    IdentifierInfo *Ident = IdentList[i].first;
    ObjCProtocolDecl *&PDecl = ObjCProtocols[Ident];
    if (PDecl == 0) { // Not already seen?
      PDecl = ObjCProtocolDecl::Create(Context, CurContext, 
                                       IdentList[i].second, Ident);
      // FIXME: PushOnScopeChains?
      CurContext->addDecl(Context, PDecl);
    }
    if (attrList)
      ProcessDeclAttributeList(PDecl, attrList);
    Protocols.push_back(PDecl);
  }
  
  ObjCForwardProtocolDecl *PDecl = 
    ObjCForwardProtocolDecl::Create(Context, CurContext, AtProtocolLoc,
                                    &Protocols[0], Protocols.size());
  CurContext->addDecl(Context, PDecl);
  CheckObjCDeclScope(PDecl);
  return DeclPtrTy::make(PDecl);
}

Sema::DeclPtrTy Sema::
ActOnStartCategoryInterface(SourceLocation AtInterfaceLoc,
                            IdentifierInfo *ClassName, SourceLocation ClassLoc,
                            IdentifierInfo *CategoryName,
                            SourceLocation CategoryLoc,
                            const DeclPtrTy *ProtoRefs,
                            unsigned NumProtoRefs,
                            SourceLocation EndProtoLoc) {
  ObjCCategoryDecl *CDecl = 
    ObjCCategoryDecl::Create(Context, CurContext, AtInterfaceLoc, CategoryName);
  // FIXME: PushOnScopeChains?
  CurContext->addDecl(Context, CDecl);

  ObjCInterfaceDecl *IDecl = getObjCInterfaceDecl(ClassName);
  /// Check that class of this category is already completely declared.
  if (!IDecl || IDecl->isForwardDecl()) {
    CDecl->setInvalidDecl();
    Diag(ClassLoc, diag::err_undef_interface) << ClassName;
    return DeclPtrTy::make(CDecl);
  }

  CDecl->setClassInterface(IDecl);
  
  // If the interface is deprecated, warn about it.
  (void)DiagnoseUseOfDecl(IDecl, ClassLoc);

  /// Check for duplicate interface declaration for this category
  ObjCCategoryDecl *CDeclChain;
  for (CDeclChain = IDecl->getCategoryList(); CDeclChain;
       CDeclChain = CDeclChain->getNextClassCategory()) {
    if (CategoryName && CDeclChain->getIdentifier() == CategoryName) {
      Diag(CategoryLoc, diag::warn_dup_category_def)
      << ClassName << CategoryName;
      Diag(CDeclChain->getLocation(), diag::note_previous_definition);
      break;
    }
  }
  if (!CDeclChain)
    CDecl->insertNextClassCategory();

  if (NumProtoRefs) {
    CDecl->setProtocolList((ObjCProtocolDecl**)ProtoRefs, NumProtoRefs,Context);
    CDecl->setLocEnd(EndProtoLoc);
  }
  
  CheckObjCDeclScope(CDecl);
  return DeclPtrTy::make(CDecl);
}

/// ActOnStartCategoryImplementation - Perform semantic checks on the
/// category implementation declaration and build an ObjCCategoryImplDecl
/// object.
Sema::DeclPtrTy Sema::ActOnStartCategoryImplementation(
                      SourceLocation AtCatImplLoc,
                      IdentifierInfo *ClassName, SourceLocation ClassLoc,
                      IdentifierInfo *CatName, SourceLocation CatLoc) {
  ObjCInterfaceDecl *IDecl = getObjCInterfaceDecl(ClassName);
  ObjCCategoryImplDecl *CDecl = 
    ObjCCategoryImplDecl::Create(Context, CurContext, AtCatImplLoc, CatName,
                                 IDecl);
  /// Check that class of this category is already completely declared.
  if (!IDecl || IDecl->isForwardDecl())
    Diag(ClassLoc, diag::err_undef_interface) << ClassName;

  // FIXME: PushOnScopeChains?
  CurContext->addDecl(Context, CDecl);

  /// TODO: Check that CatName, category name, is not used in another
  // implementation.
  ObjCCategoryImpls.push_back(CDecl);
  
  CheckObjCDeclScope(CDecl);
  return DeclPtrTy::make(CDecl);
}

Sema::DeclPtrTy Sema::ActOnStartClassImplementation(
                      SourceLocation AtClassImplLoc,
                      IdentifierInfo *ClassName, SourceLocation ClassLoc,
                      IdentifierInfo *SuperClassname, 
                      SourceLocation SuperClassLoc) {
  ObjCInterfaceDecl* IDecl = 0;
  // Check for another declaration kind with the same name.
  NamedDecl *PrevDecl = LookupName(TUScope, ClassName, LookupOrdinaryName);
  if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
    Diag(ClassLoc, diag::err_redefinition_different_kind) << ClassName;
    Diag(PrevDecl->getLocation(), diag::note_previous_definition);
  }  else {
    // Is there an interface declaration of this class; if not, warn!
    IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl); 
    if (!IDecl)
      Diag(ClassLoc, diag::warn_undef_interface) << ClassName;
  }
  
  // Check that super class name is valid class name
  ObjCInterfaceDecl* SDecl = 0;
  if (SuperClassname) {
    // Check if a different kind of symbol declared in this scope.
    PrevDecl = LookupName(TUScope, SuperClassname, LookupOrdinaryName);
    if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
      Diag(SuperClassLoc, diag::err_redefinition_different_kind)
        << SuperClassname;
      Diag(PrevDecl->getLocation(), diag::note_previous_definition);
    } else {
      SDecl = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl); 
      if (!SDecl)
        Diag(SuperClassLoc, diag::err_undef_superclass)
          << SuperClassname << ClassName;
      else if (IDecl && IDecl->getSuperClass() != SDecl) {
        // This implementation and its interface do not have the same
        // super class.
        Diag(SuperClassLoc, diag::err_conflicting_super_class)
          << SDecl->getDeclName();
        Diag(SDecl->getLocation(), diag::note_previous_definition);
      }
    }
  }
  
  if (!IDecl) {
    // Legacy case of @implementation with no corresponding @interface.
    // Build, chain & install the interface decl into the identifier.

    // FIXME: Do we support attributes on the @implementation? If so
    // we should copy them over.
    IDecl = ObjCInterfaceDecl::Create(Context, CurContext, AtClassImplLoc, 
                                      ClassName, ClassLoc, false, true);
    ObjCInterfaceDecls[ClassName] = IDecl;
    IDecl->setSuperClass(SDecl);
    IDecl->setLocEnd(ClassLoc);
    
    // FIXME: PushOnScopeChains?
    CurContext->addDecl(Context, IDecl);
    // Remember that this needs to be removed when the scope is popped.
    TUScope->AddDecl(DeclPtrTy::make(IDecl));
  }
  
  ObjCImplementationDecl* IMPDecl = 
    ObjCImplementationDecl::Create(Context, CurContext, AtClassImplLoc, 
                                   IDecl, SDecl);
  
  // FIXME: PushOnScopeChains?
  CurContext->addDecl(Context, IMPDecl);

  if (CheckObjCDeclScope(IMPDecl))
    return DeclPtrTy::make(IMPDecl);
  
  // Check that there is no duplicate implementation of this class.
  if (ObjCImplementations[ClassName])
    // FIXME: Don't leak everything!
    Diag(ClassLoc, diag::err_dup_implementation_class) << ClassName;
  else // add it to the list.
    ObjCImplementations[ClassName] = IMPDecl;
  return DeclPtrTy::make(IMPDecl);
}

void Sema::CheckImplementationIvars(ObjCImplementationDecl *ImpDecl,
                                    ObjCIvarDecl **ivars, unsigned numIvars,
                                    SourceLocation RBrace) {
  assert(ImpDecl && "missing implementation decl");
  ObjCInterfaceDecl* IDecl = ImpDecl->getClassInterface();
  if (!IDecl)
    return;
  /// Check case of non-existing @interface decl.
  /// (legacy objective-c @implementation decl without an @interface decl).
  /// Add implementations's ivar to the synthesize class's ivar list.
  if (IDecl->ImplicitInterfaceDecl()) {
    IDecl->setIVarList(ivars, numIvars, Context);
    IDecl->setLocEnd(RBrace);
    return;
  }
  // If implementation has empty ivar list, just return.
  if (numIvars == 0)
    return;
  
  assert(ivars && "missing @implementation ivars");
  
  // Check interface's Ivar list against those in the implementation.
  // names and types must match.
  //
  unsigned j = 0;
  ObjCInterfaceDecl::ivar_iterator 
    IVI = IDecl->ivar_begin(), IVE = IDecl->ivar_end();
  for (; numIvars > 0 && IVI != IVE; ++IVI) {
    ObjCIvarDecl* ImplIvar = ivars[j++];
    ObjCIvarDecl* ClsIvar = *IVI;
    assert (ImplIvar && "missing implementation ivar");
    assert (ClsIvar && "missing class ivar");
    
    // First, make sure the types match.
    if (Context.getCanonicalType(ImplIvar->getType()) !=
        Context.getCanonicalType(ClsIvar->getType())) {
      Diag(ImplIvar->getLocation(), diag::err_conflicting_ivar_type)
        << ImplIvar->getIdentifier()
        << ImplIvar->getType() << ClsIvar->getType();
      Diag(ClsIvar->getLocation(), diag::note_previous_definition);
    } else if (ImplIvar->isBitField() && ClsIvar->isBitField()) {
      Expr *ImplBitWidth = ImplIvar->getBitWidth();
      Expr *ClsBitWidth = ClsIvar->getBitWidth();
      if (ImplBitWidth->getIntegerConstantExprValue(Context).getZExtValue() !=
          ClsBitWidth->getIntegerConstantExprValue(Context).getZExtValue()) {
        Diag(ImplBitWidth->getLocStart(), diag::err_conflicting_ivar_bitwidth)
          << ImplIvar->getIdentifier();
        Diag(ClsBitWidth->getLocStart(), diag::note_previous_definition);
      }
    } 
    // Make sure the names are identical.
    if (ImplIvar->getIdentifier() != ClsIvar->getIdentifier()) {
      Diag(ImplIvar->getLocation(), diag::err_conflicting_ivar_name)
        << ImplIvar->getIdentifier() << ClsIvar->getIdentifier();
      Diag(ClsIvar->getLocation(), diag::note_previous_definition);
    }
    --numIvars;
  }
  
  if (numIvars > 0)
    Diag(ivars[j]->getLocation(), diag::err_inconsistant_ivar_count);
  else if (IVI != IVE)
    Diag((*IVI)->getLocation(), diag::err_inconsistant_ivar_count);
}

void Sema::WarnUndefinedMethod(SourceLocation ImpLoc, ObjCMethodDecl *method,
                               bool &IncompleteImpl) {
  if (!IncompleteImpl) {
    Diag(ImpLoc, diag::warn_incomplete_impl);
    IncompleteImpl = true;
  }
  Diag(ImpLoc, diag::warn_undef_method_impl) << method->getDeclName();
}

void Sema::WarnConflictingTypedMethods(ObjCMethodDecl *ImpMethodDecl,
                                       ObjCMethodDecl *IntfMethodDecl) {
  if (!Context.typesAreCompatible(IntfMethodDecl->getResultType(),
                                  ImpMethodDecl->getResultType())) {
    Diag(ImpMethodDecl->getLocation(), diag::warn_conflicting_ret_types) 
      << ImpMethodDecl->getDeclName() << IntfMethodDecl->getResultType()
      << ImpMethodDecl->getResultType();
    Diag(IntfMethodDecl->getLocation(), diag::note_previous_definition);
  }
  
  for (ObjCMethodDecl::param_iterator IM = ImpMethodDecl->param_begin(),
       IF = IntfMethodDecl->param_begin(), EM = ImpMethodDecl->param_end();
       IM != EM; ++IM, ++IF) {
    if (Context.typesAreCompatible((*IF)->getType(), (*IM)->getType()))
      continue;
    
    Diag((*IM)->getLocation(), diag::warn_conflicting_param_types) 
      << ImpMethodDecl->getDeclName() << (*IF)->getType()
      << (*IM)->getType();
    Diag((*IF)->getLocation(), diag::note_previous_definition);
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
  if (IDecl->getInstanceMethod(Context, PDecl->getSetterName()))
    return false;
  
  // Main class has the property as 'readonly'. Must search
  // through the category list to see if the property's 
  // attribute has been over-ridden to 'readwrite'.
  for (ObjCCategoryDecl *Category = IDecl->getCategoryList();
       Category; Category = Category->getNextClassCategory()) {
    // Even if property is ready only, if a category has a user defined setter, 
    // it is not considered read only. 
    if (Category->getInstanceMethod(Context, PDecl->getSetterName()))
      return false;
    ObjCPropertyDecl *P = 
      Category->FindPropertyDeclaration(Context, PDecl->getIdentifier());
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
    }
    else if (ObjCCategoryImplDecl *CIMD = 
             dyn_cast<ObjCCategoryImplDecl>(OMD->getDeclContext())) {
      if (CIMD->getInstanceMethod(PDecl->getSetterName()))
        return false;
    }
  }
  // Lastly, look through the implementation (if one is in scope).
  if (ObjCImplementationDecl *ImpDecl = 
        ObjCImplementations[IDecl->getIdentifier()])
    if (ImpDecl->getInstanceMethod(PDecl->getSetterName()))
      return false;
  // If all fails, look at the super class.
  if (ObjCInterfaceDecl *SIDecl = IDecl->getSuperClass())
    return isPropertyReadonly(PDecl, SIDecl);
  return true;
}

/// FIXME: Type hierarchies in Objective-C can be deep. We could most
/// likely improve the efficiency of selector lookups and type
/// checking by associating with each protocol / interface / category
/// the flattened instance tables. If we used an immutable set to keep
/// the table then it wouldn't add significant memory cost and it
/// would be handy for lookups.

/// CheckProtocolMethodDefs - This routine checks unimplemented methods
/// Declared in protocol, and those referenced by it.
void Sema::CheckProtocolMethodDefs(SourceLocation ImpLoc,
                                   ObjCProtocolDecl *PDecl,
                                   bool& IncompleteImpl,
                                   const llvm::DenseSet<Selector> &InsMap,
                                   const llvm::DenseSet<Selector> &ClsMap,
                                   ObjCInterfaceDecl *IDecl) {
  ObjCInterfaceDecl *Super = IDecl->getSuperClass();

  // If a method lookup fails locally we still need to look and see if
  // the method was implemented by a base class or an inherited
  // protocol. This lookup is slow, but occurs rarely in correct code
  // and otherwise would terminate in a warning.

  // check unimplemented instance methods.
  for (ObjCProtocolDecl::instmeth_iterator I = PDecl->instmeth_begin(Context), 
       E = PDecl->instmeth_end(Context); I != E; ++I) {
    ObjCMethodDecl *method = *I;
    if (method->getImplementationControl() != ObjCMethodDecl::Optional && 
        !method->isSynthesized() && !InsMap.count(method->getSelector()) &&
        (!Super || 
         !Super->lookupInstanceMethod(Context, method->getSelector()))) {
        // Ugly, but necessary. Method declared in protcol might have
        // have been synthesized due to a property declared in the class which
        // uses the protocol.
        ObjCMethodDecl *MethodInClass = 
          IDecl->lookupInstanceMethod(Context, method->getSelector());
        if (!MethodInClass || !MethodInClass->isSynthesized())
          WarnUndefinedMethod(ImpLoc, method, IncompleteImpl);
      }
  }
  // check unimplemented class methods
  for (ObjCProtocolDecl::classmeth_iterator 
         I = PDecl->classmeth_begin(Context),
         E = PDecl->classmeth_end(Context);
       I != E; ++I) {
    ObjCMethodDecl *method = *I;
    if (method->getImplementationControl() != ObjCMethodDecl::Optional &&
        !ClsMap.count(method->getSelector()) &&
        (!Super || !Super->lookupClassMethod(Context, method->getSelector())))
      WarnUndefinedMethod(ImpLoc, method, IncompleteImpl);
  }
  // Check on this protocols's referenced protocols, recursively.
  for (ObjCProtocolDecl::protocol_iterator PI = PDecl->protocol_begin(),
       E = PDecl->protocol_end(); PI != E; ++PI)
    CheckProtocolMethodDefs(ImpLoc, *PI, IncompleteImpl, InsMap, ClsMap, IDecl);
}

void Sema::ImplMethodsVsClassMethods(ObjCImplDecl* IMPDecl, 
                                     ObjCContainerDecl* CDecl, 
                                     bool IncompleteImpl) {
  llvm::DenseSet<Selector> InsMap;
  // Check and see if instance methods in class interface have been
  // implemented in the implementation class.
  for (ObjCImplementationDecl::instmeth_iterator I = IMPDecl->instmeth_begin(),
         E = IMPDecl->instmeth_end(); I != E; ++I)
    InsMap.insert((*I)->getSelector());
  
  for (ObjCInterfaceDecl::instmeth_iterator I = CDecl->instmeth_begin(Context),
       E = CDecl->instmeth_end(Context); I != E; ++I) {
    if (!(*I)->isSynthesized() && !InsMap.count((*I)->getSelector())) {
      WarnUndefinedMethod(IMPDecl->getLocation(), *I, IncompleteImpl);
      continue;
    }
    
    ObjCMethodDecl *ImpMethodDecl = 
      IMPDecl->getInstanceMethod((*I)->getSelector());
    ObjCMethodDecl *IntfMethodDecl = 
      CDecl->getInstanceMethod(Context, (*I)->getSelector());
    assert(IntfMethodDecl && 
           "IntfMethodDecl is null in ImplMethodsVsClassMethods");
    // ImpMethodDecl may be null as in a @dynamic property.
    if (ImpMethodDecl)
      WarnConflictingTypedMethods(ImpMethodDecl, IntfMethodDecl);
  }
      
  llvm::DenseSet<Selector> ClsMap;
  // Check and see if class methods in class interface have been
  // implemented in the implementation class.
  for (ObjCImplementationDecl::classmeth_iterator I =IMPDecl->classmeth_begin(),
       E = IMPDecl->classmeth_end(); I != E; ++I)
    ClsMap.insert((*I)->getSelector());
  
  for (ObjCInterfaceDecl::classmeth_iterator 
         I = CDecl->classmeth_begin(Context),
         E = CDecl->classmeth_end(Context);
       I != E; ++I)
    if (!ClsMap.count((*I)->getSelector()))
      WarnUndefinedMethod(IMPDecl->getLocation(), *I, IncompleteImpl);
    else {
      ObjCMethodDecl *ImpMethodDecl = 
        IMPDecl->getClassMethod((*I)->getSelector());
      ObjCMethodDecl *IntfMethodDecl = 
        CDecl->getClassMethod(Context, (*I)->getSelector());
      WarnConflictingTypedMethods(ImpMethodDecl, IntfMethodDecl);
    }
  
  
  // Check the protocol list for unimplemented methods in the @implementation
  // class.
  if (ObjCInterfaceDecl *I = dyn_cast<ObjCInterfaceDecl> (CDecl)) {
    for (ObjCCategoryDecl::protocol_iterator PI = I->protocol_begin(),
         E = I->protocol_end(); PI != E; ++PI)
      CheckProtocolMethodDefs(IMPDecl->getLocation(), *PI, IncompleteImpl, 
                              InsMap, ClsMap, I);
    // Check class extensions (unnamed categories)
    for (ObjCCategoryDecl *Categories = I->getCategoryList();
         Categories; Categories = Categories->getNextClassCategory()) {
      if (!Categories->getIdentifier()) {
        ImplMethodsVsClassMethods(IMPDecl, Categories, IncompleteImpl);
        break;
      }
    }
  } else if (ObjCCategoryDecl *C = dyn_cast<ObjCCategoryDecl>(CDecl)) {
    for (ObjCCategoryDecl::protocol_iterator PI = C->protocol_begin(),
         E = C->protocol_end(); PI != E; ++PI)
      CheckProtocolMethodDefs(IMPDecl->getLocation(), *PI, IncompleteImpl, 
                              InsMap, ClsMap, C->getClassInterface());
  } else
    assert(false && "invalid ObjCContainerDecl type.");
}

/// ActOnForwardClassDeclaration - 
Action::DeclPtrTy
Sema::ActOnForwardClassDeclaration(SourceLocation AtClassLoc,
                                   IdentifierInfo **IdentList,
                                   unsigned NumElts) {
  llvm::SmallVector<ObjCInterfaceDecl*, 32> Interfaces;
  
  for (unsigned i = 0; i != NumElts; ++i) {
    // Check for another declaration kind with the same name.
    NamedDecl *PrevDecl = LookupName(TUScope, IdentList[i], LookupOrdinaryName);
    if (PrevDecl && PrevDecl->isTemplateParameter()) {
      // Maybe we will complain about the shadowed template parameter.
      DiagnoseTemplateParameterShadow(AtClassLoc, PrevDecl);
      // Just pretend that we didn't see the previous declaration.
      PrevDecl = 0;
    }

    if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
      // GCC apparently allows the following idiom:
      //
      // typedef NSObject < XCElementTogglerP > XCElementToggler;
      // @class XCElementToggler;
      //
      // FIXME: Make an extension? 
      TypedefDecl *TDD = dyn_cast<TypedefDecl>(PrevDecl);
      if (!TDD || !isa<ObjCInterfaceType>(TDD->getUnderlyingType())) {
        Diag(AtClassLoc, diag::err_redefinition_different_kind) << IdentList[i];
        Diag(PrevDecl->getLocation(), diag::note_previous_definition);
      }
    }
    ObjCInterfaceDecl *IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl); 
    if (!IDecl) {  // Not already seen?  Make a forward decl.
      IDecl = ObjCInterfaceDecl::Create(Context, CurContext, AtClassLoc, 
                                        IdentList[i], SourceLocation(), true);
      ObjCInterfaceDecls[IdentList[i]] = IDecl;

      // FIXME: PushOnScopeChains?
      CurContext->addDecl(Context, IDecl);
      // Remember that this needs to be removed when the scope is popped.
      TUScope->AddDecl(DeclPtrTy::make(IDecl));
    }

    Interfaces.push_back(IDecl);
  }
  
  ObjCClassDecl *CDecl = ObjCClassDecl::Create(Context, CurContext, AtClassLoc,
                                               &Interfaces[0],
                                               Interfaces.size());
  CurContext->addDecl(Context, CDecl);
  CheckObjCDeclScope(CDecl);
  return DeclPtrTy::make(CDecl);
}


/// MatchTwoMethodDeclarations - Checks that two methods have matching type and
/// returns true, or false, accordingly.
/// TODO: Handle protocol list; such as id<p1,p2> in type comparisons
bool Sema::MatchTwoMethodDeclarations(const ObjCMethodDecl *Method, 
                                      const ObjCMethodDecl *PrevMethod,
                                      bool matchBasedOnSizeAndAlignment) {
  QualType T1 = Context.getCanonicalType(Method->getResultType());
  QualType T2 = Context.getCanonicalType(PrevMethod->getResultType());
  
  if (T1 != T2) {
    // The result types are different.
    if (!matchBasedOnSizeAndAlignment)
      return false;
    // Incomplete types don't have a size and alignment.
    if (T1->isIncompleteType() || T2->isIncompleteType())
      return false;
    // Check is based on size and alignment.
    if (Context.getTypeInfo(T1) != Context.getTypeInfo(T2))
      return false;
  }
  
  ObjCMethodDecl::param_iterator ParamI = Method->param_begin(),
       E = Method->param_end();
  ObjCMethodDecl::param_iterator PrevI = PrevMethod->param_begin();
  
  for (; ParamI != E; ++ParamI, ++PrevI) {
    assert(PrevI != PrevMethod->param_end() && "Param mismatch");
    T1 = Context.getCanonicalType((*ParamI)->getType());
    T2 = Context.getCanonicalType((*PrevI)->getType());
    if (T1 != T2) {
      // The result types are different.
      if (!matchBasedOnSizeAndAlignment)
        return false;
      // Incomplete types don't have a size and alignment.
      if (T1->isIncompleteType() || T2->isIncompleteType())
        return false;
      // Check is based on size and alignment.
      if (Context.getTypeInfo(T1) != Context.getTypeInfo(T2))
        return false;
    }
  }
  return true;
}

void Sema::AddInstanceMethodToGlobalPool(ObjCMethodDecl *Method) {
  ObjCMethodList &Entry = InstanceMethodPool[Method->getSelector()];
  if (Entry.Method == 0) {
    // Haven't seen a method with this selector name yet - add it.
    Entry.Method = Method;
    Entry.Next = 0;
    return;
  }
  
  // We've seen a method with this name, see if we have already seen this type
  // signature.
  for (ObjCMethodList *List = &Entry; List; List = List->Next)
    if (MatchTwoMethodDeclarations(Method, List->Method))
      return;
    
  // We have a new signature for an existing method - add it.
  // This is extremely rare. Only 1% of Cocoa selectors are "overloaded".
  Entry.Next = new ObjCMethodList(Method, Entry.Next);
}

// FIXME: Finish implementing -Wno-strict-selector-match.
ObjCMethodDecl *Sema::LookupInstanceMethodInGlobalPool(Selector Sel, 
                                                       SourceRange R) {
  ObjCMethodList &MethList = InstanceMethodPool[Sel];
  bool issueWarning = false;
  
  if (MethList.Method && MethList.Next) {
    for (ObjCMethodList *Next = MethList.Next; Next; Next = Next->Next)
      // This checks if the methods differ by size & alignment.
      if (!MatchTwoMethodDeclarations(MethList.Method, Next->Method, true))
        issueWarning = true;
  }
  if (issueWarning && (MethList.Method && MethList.Next)) {
    Diag(R.getBegin(), diag::warn_multiple_method_decl) << Sel << R;
    Diag(MethList.Method->getLocStart(), diag::note_using_decl)
      << MethList.Method->getSourceRange();
    for (ObjCMethodList *Next = MethList.Next; Next; Next = Next->Next)
      Diag(Next->Method->getLocStart(), diag::note_also_found_decl)
        << Next->Method->getSourceRange();
  }
  return MethList.Method;
}

void Sema::AddFactoryMethodToGlobalPool(ObjCMethodDecl *Method) {
  ObjCMethodList &FirstMethod = FactoryMethodPool[Method->getSelector()];
  if (!FirstMethod.Method) {
    // Haven't seen a method with this selector name yet - add it.
    FirstMethod.Method = Method;
    FirstMethod.Next = 0;
  } else {
    // We've seen a method with this name, now check the type signature(s).
    bool match = MatchTwoMethodDeclarations(Method, FirstMethod.Method);
    
    for (ObjCMethodList *Next = FirstMethod.Next; !match && Next; 
         Next = Next->Next)
      match = MatchTwoMethodDeclarations(Method, Next->Method);
      
    if (!match) {
      // We have a new signature for an existing method - add it.
      // This is extremely rare. Only 1% of Cocoa selectors are "overloaded".
      struct ObjCMethodList *OMI = new ObjCMethodList(Method, FirstMethod.Next);
      FirstMethod.Next = OMI;
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
  
  GetterMethod = CD->getInstanceMethod(Context, property->getGetterName());  
  SetterMethod = CD->getInstanceMethod(Context, property->getSetterName());
  
  if (GetterMethod &&
      GetterMethod->getResultType() != property->getType()) {
    Diag(property->getLocation(), 
         diag::err_accessor_property_type_mismatch) 
      << property->getDeclName()
      << GetterMethod->getSelector();
    Diag(GetterMethod->getLocation(), diag::note_declared_at);
  }
  
  if (SetterMethod) {
    if (Context.getCanonicalType(SetterMethod->getResultType()) 
        != Context.VoidTy)
      Diag(SetterMethod->getLocation(), diag::err_setter_type_void);
    if (SetterMethod->param_size() != 1 ||
        ((*SetterMethod->param_begin())->getType() != property->getType())) {
      Diag(property->getLocation(), 
           diag::err_accessor_property_type_mismatch) 
        << property->getDeclName()
        << SetterMethod->getSelector();
      Diag(SetterMethod->getLocation(), diag::note_declared_at);
    }
  }

  // Synthesize getter/setter methods if none exist.
  // Find the default getter and if one not found, add one.
  // FIXME: The synthesized property we set here is misleading. We
  // almost always synthesize these methods unless the user explicitly
  // provided prototypes (which is odd, but allowed). Sema should be
  // typechecking that the declarations jive in that situation (which
  // it is not currently).
  if (!GetterMethod) {
    // No instance method of same name as property getter name was found.
    // Declare a getter method and add it to the list of methods 
    // for this class.
    GetterMethod = ObjCMethodDecl::Create(Context, property->getLocation(), 
                             property->getLocation(), property->getGetterName(), 
                             property->getType(), CD, true, false, true, 
                             (property->getPropertyImplementation() == 
                              ObjCPropertyDecl::Optional) ? 
                             ObjCMethodDecl::Optional : 
                             ObjCMethodDecl::Required);
    CD->addDecl(Context, GetterMethod);
  } else
    // A user declared getter will be synthesize when @synthesize of
    // the property with the same name is seen in the @implementation
    GetterMethod->setIsSynthesized();
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
                               Context.VoidTy, CD, true, false, true,
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
                                                  VarDecl::None,
                                                  0);
      SetterMethod->setMethodParams(&Argument, 1, Context);
      CD->addDecl(Context, SetterMethod);
    } else
      // A user declared setter will be synthesize when @synthesize of
      // the property with the same name is seen in the @implementation
      SetterMethod->setIsSynthesized();
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

// Note: For class/category implemenations, allMethods/allProperties is
// always null.
void Sema::ActOnAtEnd(SourceLocation AtEndLoc, DeclPtrTy classDecl,
                      DeclPtrTy *allMethods, unsigned allNum,
                      DeclPtrTy *allProperties, unsigned pNum,
                      DeclGroupPtrTy *allTUVars, unsigned tuvNum) {
  Decl *ClassDecl = classDecl.getAs<Decl>();

  // FIXME: If we don't have a ClassDecl, we have an error. We should consider
  // always passing in a decl. If the decl has an error, isInvalidDecl()
  // should be true.
  if (!ClassDecl)
    return;
    
  bool isInterfaceDeclKind = 
        isa<ObjCInterfaceDecl>(ClassDecl) || isa<ObjCCategoryDecl>(ClassDecl)
         || isa<ObjCProtocolDecl>(ClassDecl);
  bool checkIdenticalMethods = isa<ObjCImplementationDecl>(ClassDecl);

  DeclContext *DC = dyn_cast<DeclContext>(ClassDecl);

  // FIXME: Remove these and use the ObjCContainerDecl/DeclContext.
  llvm::DenseMap<Selector, const ObjCMethodDecl*> InsMap;
  llvm::DenseMap<Selector, const ObjCMethodDecl*> ClsMap;

  for (unsigned i = 0; i < allNum; i++ ) {
    ObjCMethodDecl *Method =
      cast_or_null<ObjCMethodDecl>(allMethods[i].getAs<Decl>());

    if (!Method) continue;  // Already issued a diagnostic.
    if (Method->isInstanceMethod()) {
      /// Check for instance method of the same name with incompatible types
      const ObjCMethodDecl *&PrevMethod = InsMap[Method->getSelector()];
      bool match = PrevMethod ? MatchTwoMethodDeclarations(Method, PrevMethod) 
                              : false;
      if ((isInterfaceDeclKind && PrevMethod && !match) 
          || (checkIdenticalMethods && match)) {
          Diag(Method->getLocation(), diag::err_duplicate_method_decl)
            << Method->getDeclName();
          Diag(PrevMethod->getLocation(), diag::note_previous_declaration);
      } else {
        DC->addDecl(Context, Method);
        InsMap[Method->getSelector()] = Method;
        /// The following allows us to typecheck messages to "id".
        AddInstanceMethodToGlobalPool(Method);
      }
    }
    else {
      /// Check for class method of the same name with incompatible types
      const ObjCMethodDecl *&PrevMethod = ClsMap[Method->getSelector()];
      bool match = PrevMethod ? MatchTwoMethodDeclarations(Method, PrevMethod) 
                              : false;
      if ((isInterfaceDeclKind && PrevMethod && !match) 
          || (checkIdenticalMethods && match)) {
        Diag(Method->getLocation(), diag::err_duplicate_method_decl)
          << Method->getDeclName();
        Diag(PrevMethod->getLocation(), diag::note_previous_declaration);
      } else {
        DC->addDecl(Context, Method);
        ClsMap[Method->getSelector()] = Method;
        /// The following allows us to typecheck messages to "Class".
        AddFactoryMethodToGlobalPool(Method);
      }
    }
  }
  if (ObjCInterfaceDecl *I = dyn_cast<ObjCInterfaceDecl>(ClassDecl)) {
    // Compares properties declared in this class to those of its 
    // super class.
    ComparePropertiesInBaseAndSuper(I);
    MergeProtocolPropertiesIntoClass(I, DeclPtrTy::make(I));
  } else if (ObjCCategoryDecl *C = dyn_cast<ObjCCategoryDecl>(ClassDecl)) {
    // Categories are used to extend the class by declaring new methods.
    // By the same token, they are also used to add new properties. No 
    // need to compare the added property to those in the class.

    // Merge protocol properties into category
    MergeProtocolPropertiesIntoClass(C, DeclPtrTy::make(C));
    if (C->getIdentifier() == 0)
      DiagnoseClassExtensionDupMethods(C, C->getClassInterface());
  }
  if (ObjCContainerDecl *CDecl = dyn_cast<ObjCContainerDecl>(ClassDecl)) {
    // ProcessPropertyDecl is responsible for diagnosing conflicts with any
    // user-defined setter/getter. It also synthesizes setter/getter methods
    // and adds them to the DeclContext and global method pools.
    for (ObjCContainerDecl::prop_iterator I = CDecl->prop_begin(Context),
                                          E = CDecl->prop_end(Context);
         I != E; ++I)
      ProcessPropertyDecl(*I, CDecl);
    CDecl->setAtEndLoc(AtEndLoc);
  }
  if (ObjCImplementationDecl *IC=dyn_cast<ObjCImplementationDecl>(ClassDecl)) {
    IC->setLocEnd(AtEndLoc);
    if (ObjCInterfaceDecl* IDecl = IC->getClassInterface())
      ImplMethodsVsClassMethods(IC, IDecl);
  } else if (ObjCCategoryImplDecl* CatImplClass = 
                                   dyn_cast<ObjCCategoryImplDecl>(ClassDecl)) {
    CatImplClass->setLocEnd(AtEndLoc);
    
    // Find category interface decl and then check that all methods declared
    // in this interface are implemented in the category @implementation.
    if (ObjCInterfaceDecl* IDecl = CatImplClass->getClassInterface()) {
      for (ObjCCategoryDecl *Categories = IDecl->getCategoryList();
           Categories; Categories = Categories->getNextClassCategory()) {
        if (Categories->getIdentifier() == CatImplClass->getIdentifier()) {
          ImplMethodsVsClassMethods(CatImplClass, Categories);
          break;
        }
      }
    }
  }
  if (isInterfaceDeclKind) {
    // Reject invalid vardecls.
    for (unsigned i = 0; i != tuvNum; i++) {
      DeclGroupRef DG = allTUVars[i].getAsVal<DeclGroupRef>();
      for (DeclGroupRef::iterator I = DG.begin(), E = DG.end(); I != E; ++I)
        if (VarDecl *VDecl = dyn_cast<VarDecl>(*I)) {
          if (VDecl->getStorageClass() != VarDecl::Extern &&
              VDecl->getStorageClass() != VarDecl::PrivateExtern)
            Diag(VDecl->getLocation(), diag::err_objc_var_decl_inclass);
        }
    }
  }
}


/// CvtQTToAstBitMask - utility routine to produce an AST bitmask for
/// objective-c's type qualifier from the parser version of the same info.
static Decl::ObjCDeclQualifier 
CvtQTToAstBitMask(ObjCDeclSpec::ObjCDeclQualifier PQTVal) {
  Decl::ObjCDeclQualifier ret = Decl::OBJC_TQ_None;
  if (PQTVal & ObjCDeclSpec::DQ_In)
    ret = (Decl::ObjCDeclQualifier)(ret | Decl::OBJC_TQ_In);
  if (PQTVal & ObjCDeclSpec::DQ_Inout)
    ret = (Decl::ObjCDeclQualifier)(ret | Decl::OBJC_TQ_Inout);
  if (PQTVal & ObjCDeclSpec::DQ_Out)
    ret = (Decl::ObjCDeclQualifier)(ret | Decl::OBJC_TQ_Out);
  if (PQTVal & ObjCDeclSpec::DQ_Bycopy)
    ret = (Decl::ObjCDeclQualifier)(ret | Decl::OBJC_TQ_Bycopy);
  if (PQTVal & ObjCDeclSpec::DQ_Byref)
    ret = (Decl::ObjCDeclQualifier)(ret | Decl::OBJC_TQ_Byref);
  if (PQTVal & ObjCDeclSpec::DQ_Oneway)
    ret = (Decl::ObjCDeclQualifier)(ret | Decl::OBJC_TQ_Oneway);

  return ret;
}

Sema::DeclPtrTy Sema::ActOnMethodDeclaration(
    SourceLocation MethodLoc, SourceLocation EndLoc,
    tok::TokenKind MethodType, DeclPtrTy classDecl,
    ObjCDeclSpec &ReturnQT, TypeTy *ReturnType,
    Selector Sel,
    // optional arguments. The number of types/arguments is obtained
    // from the Sel.getNumArgs().
    ObjCArgInfo *ArgInfo,
    llvm::SmallVectorImpl<Declarator> &Cdecls,
    AttributeList *AttrList, tok::ObjCKeywordKind MethodDeclKind,
    bool isVariadic) {
  Decl *ClassDecl = classDecl.getAs<Decl>();

  // Make sure we can establish a context for the method.
  if (!ClassDecl) {
    Diag(MethodLoc, diag::error_missing_method_context);
    return DeclPtrTy();
  }
  QualType resultDeclType;
  
  if (ReturnType) {
    resultDeclType = QualType::getFromOpaquePtr(ReturnType);
    
    // Methods cannot return interface types. All ObjC objects are
    // passed by reference.
    if (resultDeclType->isObjCInterfaceType()) {
      Diag(MethodLoc, diag::err_object_cannot_be_passed_returned_by_value)
        << 0 << resultDeclType;
      return DeclPtrTy();
    }
  } else // get the type for "id".
    resultDeclType = Context.getObjCIdType();
  
  ObjCMethodDecl* ObjCMethod = 
    ObjCMethodDecl::Create(Context, MethodLoc, EndLoc, Sel, resultDeclType,
                           cast<DeclContext>(ClassDecl), 
                           MethodType == tok::minus, isVariadic,
                           false,
                           MethodDeclKind == tok::objc_optional ? 
                           ObjCMethodDecl::Optional : 
                           ObjCMethodDecl::Required);
  
  llvm::SmallVector<ParmVarDecl*, 16> Params;
  
  for (unsigned i = 0, e = Sel.getNumArgs(); i != e; ++i) {
    QualType ArgType, UnpromotedArgType;
    
    if (ArgInfo[i].Type == 0) {
      UnpromotedArgType = ArgType = Context.getObjCIdType();
    } else {
      UnpromotedArgType = ArgType = QualType::getFromOpaquePtr(ArgInfo[i].Type);
      // Perform the default array/function conversions (C99 6.7.5.3p[7,8]).
      ArgType = adjustParameterType(ArgType);
    }
    
    ParmVarDecl* Param;
    if (ArgType == UnpromotedArgType)
      Param = ParmVarDecl::Create(Context, ObjCMethod, ArgInfo[i].NameLoc,
                                  ArgInfo[i].Name, ArgType,
                                  VarDecl::None, 0);
    else
      Param = OriginalParmVarDecl::Create(Context, ObjCMethod,
                                          ArgInfo[i].NameLoc,
                                          ArgInfo[i].Name, ArgType,
                                          UnpromotedArgType,
                                          VarDecl::None, 0);
    
    if (ArgType->isObjCInterfaceType()) {
      Diag(ArgInfo[i].NameLoc,
           diag::err_object_cannot_be_passed_returned_by_value)
        << 1 << ArgType;
      Param->setInvalidDecl();
    }
    
    Param->setObjCDeclQualifier(
      CvtQTToAstBitMask(ArgInfo[i].DeclSpec.getObjCDeclQualifier()));
    
    // Apply the attributes to the parameter.
    ProcessDeclAttributeList(Param, ArgInfo[i].ArgAttrs);
    
    Params.push_back(Param);
  }

  ObjCMethod->setMethodParams(&Params[0], Sel.getNumArgs(), Context);
  ObjCMethod->setObjCDeclQualifier(
    CvtQTToAstBitMask(ReturnQT.getObjCDeclQualifier()));
  const ObjCMethodDecl *PrevMethod = 0;

  if (AttrList)
    ProcessDeclAttributeList(ObjCMethod, AttrList);
 
  // For implementations (which can be very "coarse grain"), we add the 
  // method now. This allows the AST to implement lookup methods that work 
  // incrementally (without waiting until we parse the @end). It also allows 
  // us to flag multiple declaration errors as they occur.
  if (ObjCImplementationDecl *ImpDecl = 
        dyn_cast<ObjCImplementationDecl>(ClassDecl)) {
    if (MethodType == tok::minus) {
      PrevMethod = ImpDecl->getInstanceMethod(Sel);
      ImpDecl->addInstanceMethod(ObjCMethod);
    } else {
      PrevMethod = ImpDecl->getClassMethod(Sel);
      ImpDecl->addClassMethod(ObjCMethod);
    }
  } 
  else if (ObjCCategoryImplDecl *CatImpDecl = 
            dyn_cast<ObjCCategoryImplDecl>(ClassDecl)) {
    if (MethodType == tok::minus) {
      PrevMethod = CatImpDecl->getInstanceMethod(Sel);
      CatImpDecl->addInstanceMethod(ObjCMethod);
    } else {
      PrevMethod = CatImpDecl->getClassMethod(Sel);
      CatImpDecl->addClassMethod(ObjCMethod);
    }
  }
  if (PrevMethod) {
    // You can never have two method definitions with the same name.
    Diag(ObjCMethod->getLocation(), diag::err_duplicate_method_decl)
      << ObjCMethod->getDeclName();
    Diag(PrevMethod->getLocation(), diag::note_previous_declaration);
  } 
  return DeclPtrTy::make(ObjCMethod);
}

void Sema::CheckObjCPropertyAttributes(QualType PropertyTy, 
                                       SourceLocation Loc,
                                       unsigned &Attributes) {
  // FIXME: Improve the reported location.

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
      !Context.isObjCObjectPointerType(PropertyTy)) {
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
      Context.isObjCObjectPointerType(PropertyTy)) {
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
}

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
  QualType T = GetTypeForDeclarator(FD.D, S);
  Decl *ClassDecl = ClassCategory.getAs<Decl>();
  ObjCInterfaceDecl *CCPrimary = 0; // continuation class's primary class
  // May modify Attributes.
  CheckObjCPropertyAttributes(T, AtLoc, Attributes);
  if (ObjCCategoryDecl *CDecl = dyn_cast<ObjCCategoryDecl>(ClassDecl))
    if (!CDecl->getIdentifier()) {
      // This is a continuation class. property requires special 
      // handling.
      if ((CCPrimary = CDecl->getClassInterface())) {
        // Find the property in continuation class's primary class only.
        ObjCPropertyDecl *PIDecl = 0;
        IdentifierInfo *PropertyId = FD.D.getIdentifier();
        for (ObjCInterfaceDecl::prop_iterator 
               I = CCPrimary->prop_begin(Context),
               E = CCPrimary->prop_end(Context);
             I != E; ++I)
          if ((*I)->getIdentifier() == PropertyId) {
            PIDecl = *I;
            break;
          }
            
        if (PIDecl) {
          // property 'PIDecl's readonly attribute will be over-ridden
          // with continuation class's readwrite property attribute!
          unsigned PIkind = PIDecl->getPropertyAttributes();
          if (isReadWrite && (PIkind & ObjCPropertyDecl::OBJC_PR_readonly)) {
            if ((Attributes & ObjCPropertyDecl::OBJC_PR_nonatomic) !=
                (PIkind & ObjCPropertyDecl::OBJC_PR_nonatomic))
              Diag(AtLoc, diag::warn_property_attr_mismatch);
            PIDecl->makeitReadWriteAttribute();
            if (Attributes & ObjCDeclSpec::DQ_PR_retain)
              PIDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_retain);
            if (Attributes & ObjCDeclSpec::DQ_PR_copy)
              PIDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_copy);
            PIDecl->setSetterName(SetterSel);
            // FIXME: use a common routine with addPropertyMethods.
            ObjCMethodDecl *SetterDecl =
              ObjCMethodDecl::Create(Context, AtLoc, AtLoc, SetterSel,
                                     Context.VoidTy,
                                     CCPrimary,
                                     true, false, true, 
                                     ObjCMethodDecl::Required);
            ParmVarDecl *Argument = ParmVarDecl::Create(Context, SetterDecl,
                                                        FD.D.getIdentifierLoc(),
                                                        PropertyId,
                                                        T, VarDecl::None, 0);
            SetterDecl->setMethodParams(&Argument, 1, Context);
            PIDecl->setSetterMethodDecl(SetterDecl);
          }
          else
            Diag(AtLoc, diag::err_use_continuation_class) 
              << CCPrimary->getDeclName();
          *isOverridingProperty = true;
          return DeclPtrTy();
        }
        // No matching property found in the primary class. Just fall thru
        // and add property to continuation class's primary class.
        ClassDecl = CCPrimary;
      } else {
        Diag(CDecl->getLocation(), diag::err_continuation_class);
        *isOverridingProperty = true;
        return DeclPtrTy();
      } 
    }

  DeclContext *DC = dyn_cast<DeclContext>(ClassDecl);
  assert(DC && "ClassDecl is not a DeclContext");
  ObjCPropertyDecl *PDecl = ObjCPropertyDecl::Create(Context, DC,
                                                     FD.D.getIdentifierLoc(), 
                                                     FD.D.getIdentifier(), T);
  DC->addDecl(Context, PDecl);
  
  if (T->isArrayType() || T->isFunctionType()) {
    Diag(AtLoc, diag::err_property_type) << T;
    PDecl->setInvalidDecl();
  }
  
  ProcessDeclAttributes(PDecl, FD.D);

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
  
  if (MethodImplKind == tok::objc_required)
    PDecl->setPropertyImplementation(ObjCPropertyDecl::Required);
  else if (MethodImplKind == tok::objc_optional)
    PDecl->setPropertyImplementation(ObjCPropertyDecl::Optional);
  // A case of continuation class adding a new property in the class. This
  // is not what it was meant for. However, gcc supports it and so should we.
  // Make sure setter/getters are declared here.
  if (CCPrimary)
    ProcessPropertyDecl(PDecl, CCPrimary);
  
  return DeclPtrTy::make(PDecl);
}

/// ActOnPropertyImplDecl - This routine performs semantic checks and
/// builds the AST node for a property implementation declaration; declared
/// as @synthesize or @dynamic.
///
Sema::DeclPtrTy Sema::ActOnPropertyImplDecl(SourceLocation AtLoc, 
                                            SourceLocation PropertyLoc,
                                            bool Synthesize, 
                                            DeclPtrTy ClassCatImpDecl,
                                            IdentifierInfo *PropertyId,
                                            IdentifierInfo *PropertyIvar) {
  Decl *ClassImpDecl = ClassCatImpDecl.getAs<Decl>();
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
    property = IDecl->FindPropertyDeclaration(Context, PropertyId);
    if (!property) {
      Diag(PropertyLoc, diag::error_bad_property_decl) << IDecl->getDeclName();
      return DeclPtrTy();
    }
  }
  else if ((CatImplClass = dyn_cast<ObjCCategoryImplDecl>(ClassImpDecl))) {
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
    property = Category->FindPropertyDeclaration(Context, PropertyId);
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
    Ivar = IDecl->lookupInstanceVariable(Context, PropertyIvar);
    if (!Ivar) {
      if (getLangOptions().ObjCNonFragileABI) {
        Ivar = ObjCIvarDecl::Create(Context, CurContext, PropertyLoc, 
                                    PropertyIvar, PropType, 
                                    ObjCIvarDecl::Public,
                                    (Expr *)0);
        property->setPropertyIvarDecl(Ivar);
      }
      else {
        Diag(PropertyLoc, diag::error_missing_property_ivar_decl) << PropertyId;
        return DeclPtrTy();
      }
    }
    QualType IvarType = Context.getCanonicalType(Ivar->getType());
    
    // Check that type of property and its ivar are type compatible.
    if (PropType != IvarType) {
      if (CheckAssignmentConstraints(PropType, IvarType) != Compatible) {
        Diag(PropertyLoc, diag::error_property_ivar_type)
          << property->getDeclName() << Ivar->getDeclName();
        return DeclPtrTy();
      }
      
      // FIXME! Rules for properties are somewhat different that those
      // for assignments. Use a new routine to consolidate all cases;
      // specifically for property redeclarations as well as for ivars.
      QualType lhsType =Context.getCanonicalType(PropType).getUnqualifiedType();
      QualType rhsType =Context.getCanonicalType(IvarType).getUnqualifiedType();
      if (lhsType != rhsType && 
          lhsType->isArithmeticType()) {
        Diag(PropertyLoc, diag::error_property_ivar_type)
        << property->getDeclName() << Ivar->getDeclName();
        return DeclPtrTy();
      }
      // __weak is explicit. So it works on Canonical type.
      if (PropType.isObjCGCWeak() && !IvarType.isObjCGCWeak() &&
          getLangOptions().getGCMode() != LangOptions::NonGC) {
        Diag(PropertyLoc, diag::error_weak_property)
        << property->getDeclName() << Ivar->getDeclName();
        return DeclPtrTy();
      }
      if ((Context.isObjCObjectPointerType(property->getType()) || 
           PropType.isObjCGCStrong()) && IvarType.isObjCGCWeak() &&
           getLangOptions().getGCMode() != LangOptions::NonGC) {
        Diag(PropertyLoc, diag::error_strong_property)
        << property->getDeclName() << Ivar->getDeclName();
        return DeclPtrTy();
      }
    }
  } else if (PropertyIvar) {
    // @dynamic
    Diag(PropertyLoc, diag::error_dynamic_property_ivar_decl);
    return DeclPtrTy();
  }
  assert (property && "ActOnPropertyImplDecl - property declaration missing");
  ObjCPropertyImplDecl *PIDecl = 
    ObjCPropertyImplDecl::Create(Context, CurContext, AtLoc, PropertyLoc, 
                                 property, 
                                 (Synthesize ? 
                                  ObjCPropertyImplDecl::Synthesize 
                                  : ObjCPropertyImplDecl::Dynamic),
                                 Ivar);
  CurContext->addDecl(Context, PIDecl);
  if (IC) {
    if (Synthesize)
      if (ObjCPropertyImplDecl *PPIDecl = 
          IC->FindPropertyImplIvarDecl(PropertyIvar)) {
        Diag(PropertyLoc, diag::error_duplicate_ivar_use) 
          << PropertyId << PPIDecl->getPropertyDecl()->getIdentifier() 
          << PropertyIvar;
        Diag(PPIDecl->getLocation(), diag::note_previous_use);
      }
    
    if (ObjCPropertyImplDecl *PPIDecl = IC->FindPropertyImplDecl(PropertyId)) {
      Diag(PropertyLoc, diag::error_property_implemented) << PropertyId;
      Diag(PPIDecl->getLocation(), diag::note_previous_declaration);
      return DeclPtrTy();
    }
    IC->addPropertyImplementation(PIDecl);
  }
  else {
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

bool Sema::CheckObjCDeclScope(Decl *D) {
  if (isa<TranslationUnitDecl>(CurContext->getLookupContext()))
    return false;
  
  Diag(D->getLocation(), diag::err_objc_decls_may_only_appear_in_global_scope);
  D->setInvalidDecl();
  
  return true;
}

/// Collect the instance variables declared in an Objective-C object.  Used in
/// the creation of structures from objects using the @defs directive.
/// FIXME: This should be consolidated with CollectObjCIvars as it is also
/// part of the AST generation logic of @defs.
static void CollectIvars(ObjCInterfaceDecl *Class, RecordDecl *Record,
                         ASTContext& Ctx,
                         llvm::SmallVectorImpl<Sema::DeclPtrTy> &ivars) {
  if (Class->getSuperClass())
    CollectIvars(Class->getSuperClass(), Record, Ctx, ivars);
  
  // For each ivar, create a fresh ObjCAtDefsFieldDecl.
  for (ObjCInterfaceDecl::ivar_iterator I = Class->ivar_begin(),
       E = Class->ivar_end(); I != E; ++I) {
    ObjCIvarDecl* ID = *I;
    Decl *FD = ObjCAtDefsFieldDecl::Create(Ctx, Record, ID->getLocation(),
                                           ID->getIdentifier(), ID->getType(),
                                           ID->getBitWidth());
    ivars.push_back(Sema::DeclPtrTy::make(FD));
  }
}

/// Called whenever @defs(ClassName) is encountered in the source.  Inserts the
/// instance variables of ClassName into Decls.
void Sema::ActOnDefs(Scope *S, DeclPtrTy TagD, SourceLocation DeclStart, 
                     IdentifierInfo *ClassName,
                     llvm::SmallVectorImpl<DeclPtrTy> &Decls) {
  // Check that ClassName is a valid class
  ObjCInterfaceDecl *Class = getObjCInterfaceDecl(ClassName);
  if (!Class) {
    Diag(DeclStart, diag::err_undef_interface) << ClassName;
    return;
  }
  // Collect the instance variables
  CollectIvars(Class, dyn_cast<RecordDecl>(TagD.getAs<Decl>()), Context, Decls);
  
  // Introduce all of these fields into the appropriate scope.
  for (llvm::SmallVectorImpl<DeclPtrTy>::iterator D = Decls.begin();
       D != Decls.end(); ++D) {
    FieldDecl *FD = cast<FieldDecl>(D->getAs<Decl>());
    if (getLangOptions().CPlusPlus)
      PushOnScopeChains(cast<FieldDecl>(FD), S);
    else if (RecordDecl *Record = dyn_cast<RecordDecl>(TagD.getAs<Decl>()))
      Record->addDecl(Context, FD);
  }
}

