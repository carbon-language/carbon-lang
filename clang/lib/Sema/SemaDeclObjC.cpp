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
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Parse/Scope.h"

using namespace clang;

/// ObjCActOnStartOfMethodDef - This routine sets up parameters; invisible
/// and user declared, in the method definition's AST.
void Sema::ObjCActOnStartOfMethodDef(Scope *FnBodyScope, DeclTy *D) {
  assert(getCurMethodDecl() == 0 && "Method parsing confused");
  ObjCMethodDecl *MDecl = dyn_cast_or_null<ObjCMethodDecl>((Decl *)D);
  
  // If we don't have a valid method decl, simply return.
  if (!MDecl)
    return;

  // Allow the rest of sema to find private method decl implementations.
  if (MDecl->isInstance())
    AddInstanceMethodToGlobalPool(MDecl);
  else
    AddFactoryMethodToGlobalPool(MDecl);
  
  // Allow all of Sema to see that we are entering a method definition.
  PushDeclContext(MDecl);

  // Create Decl objects for each parameter, entrring them in the scope for
  // binding to their use.
  struct DeclaratorChunk::ParamInfo PI;

  // Insert the invisible arguments, self and _cmd!
  PI.Ident = &Context.Idents.get("self");
  PI.IdentLoc = SourceLocation(); // synthesized vars have a null location.
  QualType selfTy;
  if (MDecl->isInstance()) {
    selfTy = Context.getObjCIdType();
    if (ObjCInterfaceDecl *OID = MDecl->getClassInterface()) {
      // There may be no interface context due to error in declaration of the 
      // interface (which has been reported). Recover gracefully
      selfTy = Context.getObjCInterfaceType(OID);
      selfTy = Context.getPointerType(selfTy);
    }
  } else // we have a factory method.
    selfTy = Context.getObjCClassType();
  getCurMethodDecl()->setSelfDecl(CreateImplicitParameter(FnBodyScope,
        PI.Ident, PI.IdentLoc, selfTy));
  
  PI.Ident = &Context.Idents.get("_cmd");
  getCurMethodDecl()->setCmdDecl(CreateImplicitParameter(FnBodyScope,
        PI.Ident, PI.IdentLoc, Context.getObjCSelType()));

  // Introduce all of the other parameters into this scope.
  for (unsigned i = 0, e = MDecl->getNumParams(); i != e; ++i) {
    ParmVarDecl *PDecl = MDecl->getParamDecl(i);
    IdentifierInfo *II = PDecl->getIdentifier();
    if (II)
      PushOnScopeChains(PDecl, FnBodyScope);
  }
}

Sema::DeclTy *Sema::
ActOnStartClassInterface(SourceLocation AtInterfaceLoc,
                         IdentifierInfo *ClassName, SourceLocation ClassLoc,
                         IdentifierInfo *SuperName, SourceLocation SuperLoc,
                         const IdentifierLocPair *ProtocolNames,
                         unsigned NumProtocols,
                         SourceLocation EndProtoLoc, AttributeList *AttrList) {
  assert(ClassName && "Missing class identifier");
  
  // Check for another declaration kind with the same name.
  Decl *PrevDecl = LookupDecl(ClassName, Decl::IDNS_Ordinary, TUScope);
  if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
    Diag(ClassLoc, diag::err_redefinition_different_kind,
         ClassName->getName());
    Diag(PrevDecl->getLocation(), diag::err_previous_definition);
  }
  
  ObjCInterfaceDecl* IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl);
  if (IDecl) {
    // Class already seen. Is it a forward declaration?
    if (!IDecl->isForwardDecl())
      Diag(AtInterfaceLoc, diag::err_duplicate_class_def, IDecl->getName());
    else {
      IDecl->setLocation(AtInterfaceLoc);
      IDecl->setForwardDecl(false);
    }
  } else {
    IDecl = ObjCInterfaceDecl::Create(Context, AtInterfaceLoc,
                                      ClassName, ClassLoc);
  
    ObjCInterfaceDecls[ClassName] = IDecl;
    // Remember that this needs to be removed when the scope is popped.
    TUScope->AddDecl(IDecl);
  }
  
  if (SuperName) {
    ObjCInterfaceDecl* SuperClassEntry = 0;
    // Check if a different kind of symbol declared in this scope.
    PrevDecl = LookupDecl(SuperName, Decl::IDNS_Ordinary, TUScope);
    if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
      Diag(SuperLoc, diag::err_redefinition_different_kind,
           SuperName->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
    }
    else {
      // Check that super class is previously defined
      SuperClassEntry = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl); 
                              
      if (!SuperClassEntry || SuperClassEntry->isForwardDecl()) {
        Diag(SuperLoc, diag::err_undef_superclass, 
             SuperClassEntry ? SuperClassEntry->getName() 
                             : SuperName->getName(),
             ClassName->getName(), SourceRange(AtInterfaceLoc, ClassLoc));
      }
    }
    IDecl->setSuperClass(SuperClassEntry);
    IDecl->setSuperClassLoc(SuperLoc);
    IDecl->setLocEnd(SuperLoc);
  } else { // we have a root class.
    IDecl->setLocEnd(ClassLoc);
  }
  
  /// Check then save referenced protocols
  if (NumProtocols) {
    llvm::SmallVector<ObjCProtocolDecl*, 8> RefProtos;
    for (unsigned int i = 0; i != NumProtocols; i++) {
      ObjCProtocolDecl* RefPDecl = ObjCProtocols[ProtocolNames[i].first];
      if (!RefPDecl)
        Diag(ProtocolNames[i].second, diag::err_undeclared_protocol,
             ProtocolNames[i].first->getName());
      else {
        if (RefPDecl->isForwardDecl())
          Diag(ProtocolNames[i].second, diag::warn_undef_protocolref,
               ProtocolNames[i].first->getName());
        RefProtos.push_back(RefPDecl);
      }
    }
    if (!RefProtos.empty())
      IDecl->addReferencedProtocols(&RefProtos[0], RefProtos.size());
    IDecl->setLocEnd(EndProtoLoc);
  }
  return IDecl;
}

/// ActOnCompatiblityAlias - this action is called after complete parsing of
/// @compaatibility_alias declaration. It sets up the alias relationships.
Sema::DeclTy *Sema::ActOnCompatiblityAlias(SourceLocation AtLoc,
                                           IdentifierInfo *AliasName, 
                                           SourceLocation AliasLocation,
                                           IdentifierInfo *ClassName,
                                           SourceLocation ClassLocation) {
  // Look for previous declaration of alias name
  Decl *ADecl = LookupDecl(AliasName, Decl::IDNS_Ordinary, TUScope);
  if (ADecl) {
    if (isa<ObjCCompatibleAliasDecl>(ADecl)) {
      Diag(AliasLocation, diag::warn_previous_alias_decl);
      Diag(ADecl->getLocation(), diag::warn_previous_declaration);
    }
    else {
      Diag(AliasLocation, diag::err_conflicting_aliasing_type,
           AliasName->getName());
      Diag(ADecl->getLocation(), diag::err_previous_declaration);
    }
    return 0;
  }
  // Check for class declaration
  Decl *CDeclU = LookupDecl(ClassName, Decl::IDNS_Ordinary, TUScope);
  ObjCInterfaceDecl *CDecl = dyn_cast_or_null<ObjCInterfaceDecl>(CDeclU);
  if (CDecl == 0) {
    Diag(ClassLocation, diag::warn_undef_interface, ClassName->getName());
    if (CDeclU)
      Diag(CDeclU->getLocation(), diag::warn_previous_declaration);
    return 0;
  }
  
  // Everything checked out, instantiate a new alias declaration AST.
  ObjCCompatibleAliasDecl *AliasDecl = 
    ObjCCompatibleAliasDecl::Create(Context, AtLoc, AliasName, CDecl);
  
  ObjCAliasDecls[AliasName] = AliasDecl;
  TUScope->AddDecl(AliasDecl);
  return AliasDecl;
}

Sema::DeclTy *
Sema::ActOnStartProtocolInterface(SourceLocation AtProtoInterfaceLoc,
                                  IdentifierInfo *ProtocolName,
                                  SourceLocation ProtocolLoc,
                                  DeclTy * const *ProtoRefs,
                                  unsigned NumProtoRefs,
                                  SourceLocation EndProtoLoc) {
  assert(ProtocolName && "Missing protocol identifier");
  ObjCProtocolDecl *PDecl = ObjCProtocols[ProtocolName];
  if (PDecl) {
    // Protocol already seen. Better be a forward protocol declaration
    if (!PDecl->isForwardDecl()) {
      Diag(ProtocolLoc, diag::err_duplicate_protocol_def, 
           ProtocolName->getName());
      // Just return the protocol we already had.
      // FIXME: don't leak the objects passed in!
      return PDecl;
    }
    
    PDecl->setForwardDecl(false);
  } else {
    PDecl = ObjCProtocolDecl::Create(Context, AtProtoInterfaceLoc,ProtocolName);
    PDecl->setForwardDecl(false);
    ObjCProtocols[ProtocolName] = PDecl;
  }
  
  if (NumProtoRefs) {
    /// Check then save referenced protocols.
    PDecl->addReferencedProtocols((ObjCProtocolDecl**)ProtoRefs, NumProtoRefs);
    PDecl->setLocEnd(EndProtoLoc);
  }
  return PDecl;
}

/// FindProtocolDeclaration - This routine looks up protocols and
/// issuer error if they are not declared. It returns list of protocol
/// declarations in its 'Protocols' argument.
void
Sema::FindProtocolDeclaration(bool WarnOnDeclarations,
                              const IdentifierLocPair *ProtocolId,
                              unsigned NumProtocols,
                              llvm::SmallVectorImpl<DeclTy*> &Protocols) {
  for (unsigned i = 0; i != NumProtocols; ++i) {
    ObjCProtocolDecl *PDecl = ObjCProtocols[ProtocolId[i].first];
    if (!PDecl) {
      Diag(ProtocolId[i].second, diag::err_undeclared_protocol, 
           ProtocolId[i].first->getName());
      continue;
    }

    // If this is a forward declaration and we are supposed to warn in this
    // case, do it.
    if (WarnOnDeclarations && PDecl->isForwardDecl())
      Diag(ProtocolId[i].second, diag::warn_undef_protocolref,
           ProtocolId[i].first->getName());
    Protocols.push_back(PDecl); 
  }
}

/// DiagnosePropertyMismatch - Compares two properties for their
/// attributes and types and warns on a variety of inconsistancies.
///
void
Sema::DiagnosePropertyMismatch(ObjCPropertyDecl *Property, 
                               ObjCPropertyDecl *SuperProperty,
                               const char *inheritedName) {
  ObjCPropertyDecl::PropertyAttributeKind CAttr = 
  Property->getPropertyAttributes();
  ObjCPropertyDecl::PropertyAttributeKind SAttr = 
  SuperProperty->getPropertyAttributes();
  if ((CAttr & ObjCPropertyDecl::OBJC_PR_readonly)
      && (SAttr & ObjCPropertyDecl::OBJC_PR_readwrite))
    Diag(Property->getLocation(), diag::warn_readonly_property, 
               Property->getName(), inheritedName);
  if ((CAttr & ObjCPropertyDecl::OBJC_PR_copy)
      != (SAttr & ObjCPropertyDecl::OBJC_PR_copy))
    Diag(Property->getLocation(), diag::warn_property_attribute,
         Property->getName(), "copy", inheritedName, 
         SourceRange());
  else if ((CAttr & ObjCPropertyDecl::OBJC_PR_retain)
           != (SAttr & ObjCPropertyDecl::OBJC_PR_retain))
    Diag(Property->getLocation(), diag::warn_property_attribute,
         Property->getName(), "retain", inheritedName, 
         SourceRange());
  
  if ((CAttr & ObjCPropertyDecl::OBJC_PR_nonatomic)
      != (SAttr & ObjCPropertyDecl::OBJC_PR_nonatomic))
    Diag(Property->getLocation(), diag::warn_property_attribute,
         Property->getName(), "atomic", inheritedName, 
         SourceRange());
  if (Property->getSetterName() != SuperProperty->getSetterName())
    Diag(Property->getLocation(), diag::warn_property_attribute,
         Property->getName(), "setter", inheritedName, 
         SourceRange());
  if (Property->getGetterName() != SuperProperty->getGetterName())
    Diag(Property->getLocation(), diag::warn_property_attribute,
         Property->getName(), "getter", inheritedName, 
         SourceRange());
  
  if (Property->getCanonicalType() != SuperProperty->getCanonicalType())
    Diag(Property->getLocation(), diag::warn_property_type,
         Property->getType().getAsString(),  
         inheritedName);
  
}

/// ComparePropertiesInBaseAndSuper - This routine compares property
/// declarations in base and its super class, if any, and issues
/// diagnostics in a variety of inconsistant situations.
///
void 
Sema::ComparePropertiesInBaseAndSuper(ObjCInterfaceDecl *IDecl) {
  ObjCInterfaceDecl *SDecl = IDecl->getSuperClass();
  if (!SDecl)
    return;
  for (ObjCInterfaceDecl::classprop_iterator S = SDecl->classprop_begin(),
       E = SDecl->classprop_end(); S != E; ++S) {
    ObjCPropertyDecl *SuperPDecl = (*S);
    // Does property in super class has declaration in current class?
    for (ObjCInterfaceDecl::classprop_iterator I = IDecl->classprop_begin(),
         E = IDecl->classprop_end(); I != E; ++I) {
      ObjCPropertyDecl *PDecl = (*I);
      if (SuperPDecl->getIdentifier() == PDecl->getIdentifier())
          DiagnosePropertyMismatch(PDecl, SuperPDecl, SDecl->getName());
    }
  }
}

/// MergeOneProtocolPropertiesIntoClass - This routine goes thru the list
/// of properties declared in a protocol and adds them to the list
/// of properties for current class if it is not there already.
void
Sema::MergeOneProtocolPropertiesIntoClass(ObjCInterfaceDecl *IDecl,
                                          ObjCProtocolDecl *PDecl)
{
  llvm::SmallVector<ObjCPropertyDecl*, 16> mergeProperties;
  for (ObjCProtocolDecl::classprop_iterator P = PDecl->classprop_begin(),
       E = PDecl->classprop_end(); P != E; ++P) {
    ObjCPropertyDecl *Pr = (*P);
    ObjCInterfaceDecl::classprop_iterator CP, CE;
    // Is this property already in  class's list of properties?
    for (CP = IDecl->classprop_begin(), CE = IDecl->classprop_end(); 
         CP != CE; ++CP)
      if ((*CP)->getIdentifier() == Pr->getIdentifier())
        break;
    if (CP == CE)
      // Add this property to list of properties for thie class.
      mergeProperties.push_back(Pr);
    else
      // Property protocol already exist in class. Diagnose any mismatch.
      DiagnosePropertyMismatch((*CP), Pr, PDecl->getName());
    }
  IDecl->mergeProperties(&mergeProperties[0], mergeProperties.size());
}

/// MergeProtocolPropertiesIntoClass - This routine merges properties
/// declared in 'MergeItsProtocols' objects (which can be a class or an
/// inherited protocol into the list of properties for class 'IDecl'
///

void
Sema::MergeProtocolPropertiesIntoClass(ObjCInterfaceDecl *IDecl,
                                       DeclTy *MergeItsProtocols) {
  Decl *ClassDecl = static_cast<Decl *>(MergeItsProtocols);
  if (ObjCInterfaceDecl *MDecl = dyn_cast<ObjCInterfaceDecl>(ClassDecl)) {
    for (ObjCInterfaceDecl::protocol_iterator P = MDecl->protocol_begin(),
         E = MDecl->protocol_end(); P != E; ++P)
      // Merge properties of class (*P) into IDECL's
      MergeOneProtocolPropertiesIntoClass(IDecl, *P);
    
    // Go thru the list of protocols for this class and recursively merge
    // their properties into this class as well.
    for (ObjCInterfaceDecl::protocol_iterator P = IDecl->protocol_begin(),
         E = IDecl->protocol_end(); P != E; ++P)
      MergeProtocolPropertiesIntoClass(IDecl, *P);
  } else {
    ObjCProtocolDecl *MD = cast<ObjCProtocolDecl>(ClassDecl);
    for (ObjCProtocolDecl::protocol_iterator P = MD->protocol_begin(),
         E = MD->protocol_end(); P != E; ++P)
      MergeOneProtocolPropertiesIntoClass(IDecl, (*P));
  }
}

/// ActOnForwardProtocolDeclaration - 
Action::DeclTy *
Sema::ActOnForwardProtocolDeclaration(SourceLocation AtProtocolLoc,
                                      const IdentifierLocPair *IdentList,
                                      unsigned NumElts) {
  llvm::SmallVector<ObjCProtocolDecl*, 32> Protocols;
  
  for (unsigned i = 0; i != NumElts; ++i) {
    IdentifierInfo *Ident = IdentList[i].first;
    ObjCProtocolDecl *&PDecl = ObjCProtocols[Ident];
    if (PDecl == 0) // Not already seen?
      PDecl = ObjCProtocolDecl::Create(Context, IdentList[i].second, Ident);
    
    Protocols.push_back(PDecl);
  }
  return ObjCForwardProtocolDecl::Create(Context, AtProtocolLoc,
                                         &Protocols[0], Protocols.size());
}

Sema::DeclTy *Sema::
ActOnStartCategoryInterface(SourceLocation AtInterfaceLoc,
                            IdentifierInfo *ClassName, SourceLocation ClassLoc,
                            IdentifierInfo *CategoryName,
                            SourceLocation CategoryLoc,
                            const IdentifierLocPair *ProtoRefNames,
                            unsigned NumProtoRefs,
                            SourceLocation EndProtoLoc) {
  ObjCInterfaceDecl *IDecl = getObjCInterfaceDecl(ClassName);
  
  ObjCCategoryDecl *CDecl = 
    ObjCCategoryDecl::Create(Context, AtInterfaceLoc, CategoryName);
  CDecl->setClassInterface(IDecl);
  
  /// Check that class of this category is already completely declared.
  if (!IDecl || IDecl->isForwardDecl())
    Diag(ClassLoc, diag::err_undef_interface, ClassName->getName());
  else {
    /// Check for duplicate interface declaration for this category
    ObjCCategoryDecl *CDeclChain;
    for (CDeclChain = IDecl->getCategoryList(); CDeclChain;
         CDeclChain = CDeclChain->getNextClassCategory()) {
      if (CategoryName && CDeclChain->getIdentifier() == CategoryName) {
        Diag(CategoryLoc, diag::warn_dup_category_def, ClassName->getName(),
             CategoryName->getName());
        break;
      }
    }
    if (!CDeclChain)
      CDecl->insertNextClassCategory();
  }

  if (NumProtoRefs) {
    llvm::SmallVector<ObjCProtocolDecl*, 32> RefProtocols;
    /// Check and then save the referenced protocols.
    for (unsigned int i = 0; i != NumProtoRefs; i++) {
      ObjCProtocolDecl* RefPDecl = ObjCProtocols[ProtoRefNames[i].first];
      if (!RefPDecl)
        Diag(ProtoRefNames[i].second, diag::err_undeclared_protocol,
             ProtoRefNames[i].first->getName());
      else {
        if (RefPDecl->isForwardDecl())
          Diag(ProtoRefNames[i].second, diag::warn_undef_protocolref,
               ProtoRefNames[i].first->getName());
        RefProtocols.push_back(RefPDecl);
      }
    }
    if (!RefProtocols.empty())
      CDecl->addReferencedProtocols(&RefProtocols[0], RefProtocols.size());
  }
  CDecl->setLocEnd(EndProtoLoc);
  return CDecl;
}

/// ActOnStartCategoryImplementation - Perform semantic checks on the
/// category implementation declaration and build an ObjCCategoryImplDecl
/// object.
Sema::DeclTy *Sema::ActOnStartCategoryImplementation(
                      SourceLocation AtCatImplLoc,
                      IdentifierInfo *ClassName, SourceLocation ClassLoc,
                      IdentifierInfo *CatName, SourceLocation CatLoc) {
  ObjCInterfaceDecl *IDecl = getObjCInterfaceDecl(ClassName);
  ObjCCategoryImplDecl *CDecl = 
    ObjCCategoryImplDecl::Create(Context, AtCatImplLoc, CatName, IDecl);
  /// Check that class of this category is already completely declared.
  if (!IDecl || IDecl->isForwardDecl())
    Diag(ClassLoc, diag::err_undef_interface, ClassName->getName());

  /// TODO: Check that CatName, category name, is not used in another
  // implementation.
  return CDecl;
}

Sema::DeclTy *Sema::ActOnStartClassImplementation(
                      SourceLocation AtClassImplLoc,
                      IdentifierInfo *ClassName, SourceLocation ClassLoc,
                      IdentifierInfo *SuperClassname, 
                      SourceLocation SuperClassLoc) {
  ObjCInterfaceDecl* IDecl = 0;
  // Check for another declaration kind with the same name.
  Decl *PrevDecl = LookupDecl(ClassName, Decl::IDNS_Ordinary, TUScope);
  if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
    Diag(ClassLoc, diag::err_redefinition_different_kind,
         ClassName->getName());
    Diag(PrevDecl->getLocation(), diag::err_previous_definition);
  }
  else {
    // Is there an interface declaration of this class; if not, warn!
    IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl); 
    if (!IDecl)
      Diag(ClassLoc, diag::warn_undef_interface, ClassName->getName());
  }
  
  // Check that super class name is valid class name
  ObjCInterfaceDecl* SDecl = 0;
  if (SuperClassname) {
    // Check if a different kind of symbol declared in this scope.
    PrevDecl = LookupDecl(SuperClassname, Decl::IDNS_Ordinary, TUScope);
    if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
      Diag(SuperClassLoc, diag::err_redefinition_different_kind,
           SuperClassname->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
    }
    else {
      SDecl = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl); 
      if (!SDecl)
        Diag(SuperClassLoc, diag::err_undef_superclass, 
             SuperClassname->getName(), ClassName->getName());
      else if (IDecl && IDecl->getSuperClass() != SDecl) {
        // This implementation and its interface do not have the same
        // super class.
        Diag(SuperClassLoc, diag::err_conflicting_super_class, 
             SDecl->getName());
        Diag(SDecl->getLocation(), diag::err_previous_definition);
      }
    }
  }
  
  if (!IDecl) {
    // Legacy case of @implementation with no corresponding @interface.
    // Build, chain & install the interface decl into the identifier.
    IDecl = ObjCInterfaceDecl::Create(Context, AtClassImplLoc, ClassName, 
                                      ClassLoc, false, true);
    ObjCInterfaceDecls[ClassName] = IDecl;
    IDecl->setSuperClass(SDecl);
    IDecl->setLocEnd(ClassLoc);
    
    // Remember that this needs to be removed when the scope is popped.
    TUScope->AddDecl(IDecl);
  }
  
  ObjCImplementationDecl* IMPDecl = 
    ObjCImplementationDecl::Create(Context, AtClassImplLoc, ClassName, 
                                   IDecl, SDecl);
  
  // Check that there is no duplicate implementation of this class.
  if (ObjCImplementations[ClassName])
    // FIXME: Don't leak everything!
    Diag(ClassLoc, diag::err_dup_implementation_class, ClassName->getName());
  else // add it to the list.
    ObjCImplementations[ClassName] = IMPDecl;
  return IMPDecl;
}

void Sema::CheckImplementationIvars(ObjCImplementationDecl *ImpDecl,
                                    ObjCIvarDecl **ivars, unsigned numIvars,
                                    SourceLocation RBrace) {
  assert(ImpDecl && "missing implementation decl");
  ObjCInterfaceDecl* IDecl = getObjCInterfaceDecl(ImpDecl->getIdentifier());
  if (!IDecl)
    return;
  /// Check case of non-existing @interface decl.
  /// (legacy objective-c @implementation decl without an @interface decl).
  /// Add implementations's ivar to the synthesize class's ivar list.
  if (IDecl->ImplicitInterfaceDecl()) {
    IDecl->addInstanceVariablesToClass(ivars, numIvars, RBrace);
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
    if (ImplIvar->getCanonicalType() != ClsIvar->getCanonicalType()) {
      Diag(ImplIvar->getLocation(), diag::err_conflicting_ivar_type,
           ImplIvar->getIdentifier()->getName());
      Diag(ClsIvar->getLocation(), diag::err_previous_definition,
           ClsIvar->getIdentifier()->getName());
    }
    // TODO: Two mismatched (unequal width) Ivar bitfields should be diagnosed 
    // as error.
    else if (ImplIvar->getIdentifier() != ClsIvar->getIdentifier()) {
      Diag(ImplIvar->getLocation(), diag::err_conflicting_ivar_name,
           ImplIvar->getIdentifier()->getName());
      Diag(ClsIvar->getLocation(), diag::err_previous_definition,
           ClsIvar->getIdentifier()->getName());
      return;
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
  Diag(ImpLoc, diag::warn_undef_method_impl, method->getSelector().getName());
}

/// CheckProtocolMethodDefs - This routine checks unimplemented methods
/// Declared in protocol, and those referenced by it.
void Sema::CheckProtocolMethodDefs(SourceLocation ImpLoc,
                                   ObjCProtocolDecl *PDecl,
                                   bool& IncompleteImpl,
                                   const llvm::DenseSet<Selector> &InsMap,
                                   const llvm::DenseSet<Selector> &ClsMap) {
  // check unimplemented instance methods.
  for (ObjCProtocolDecl::instmeth_iterator I = PDecl->instmeth_begin(), 
       E = PDecl->instmeth_end(); I != E; ++I) {
    ObjCMethodDecl *method = *I;
    if (!InsMap.count(method->getSelector()) && 
        method->getImplementationControl() != ObjCMethodDecl::Optional)
      WarnUndefinedMethod(ImpLoc, method, IncompleteImpl);
  }
  // check unimplemented class methods
  for (ObjCProtocolDecl::classmeth_iterator I = PDecl->classmeth_begin(), 
       E = PDecl->classmeth_end(); I != E; ++I) {
    ObjCMethodDecl *method = *I;
    if (!ClsMap.count(method->getSelector()) &&
        method->getImplementationControl() != ObjCMethodDecl::Optional)
      WarnUndefinedMethod(ImpLoc, method, IncompleteImpl);
  }
  // Check on this protocols's referenced protocols, recursively.
  for (ObjCProtocolDecl::protocol_iterator PI = PDecl->protocol_begin(),
       E = PDecl->protocol_end(); PI != E; ++PI)
    CheckProtocolMethodDefs(ImpLoc, *PI, IncompleteImpl, InsMap, ClsMap);
}

void Sema::ImplMethodsVsClassMethods(ObjCImplementationDecl* IMPDecl, 
                                     ObjCInterfaceDecl* IDecl) {
  llvm::DenseSet<Selector> InsMap;
  // Check and see if instance methods in class interface have been
  // implemented in the implementation class.
  for (ObjCImplementationDecl::instmeth_iterator I = IMPDecl->instmeth_begin(),
       E = IMPDecl->instmeth_end(); I != E; ++I)
    InsMap.insert((*I)->getSelector());
  
  bool IncompleteImpl = false;
  for (ObjCInterfaceDecl::instmeth_iterator I = IDecl->instmeth_begin(),
       E = IDecl->instmeth_end(); I != E; ++I)
    if (!(*I)->isSynthesized() && !InsMap.count((*I)->getSelector()))
      WarnUndefinedMethod(IMPDecl->getLocation(), *I, IncompleteImpl);
      
  llvm::DenseSet<Selector> ClsMap;
  // Check and see if class methods in class interface have been
  // implemented in the implementation class.
  for (ObjCImplementationDecl::classmeth_iterator I =IMPDecl->classmeth_begin(),
       E = IMPDecl->classmeth_end(); I != E; ++I)
    ClsMap.insert((*I)->getSelector());
  
  for (ObjCInterfaceDecl::classmeth_iterator I = IDecl->classmeth_begin(),
       E = IDecl->classmeth_end(); I != E; ++I)
    if (!ClsMap.count((*I)->getSelector()))
      WarnUndefinedMethod(IMPDecl->getLocation(), *I, IncompleteImpl);
  
  // Check the protocol list for unimplemented methods in the @implementation
  // class.
  const ObjCList<ObjCProtocolDecl> &Protocols =
    IDecl->getReferencedProtocols();
  for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
       E = Protocols.end(); I != E; ++I)
    CheckProtocolMethodDefs(IMPDecl->getLocation(), *I, 
                            IncompleteImpl, InsMap, ClsMap);
}

/// ImplCategoryMethodsVsIntfMethods - Checks that methods declared in the
/// category interface is implemented in the category @implementation.
void Sema::ImplCategoryMethodsVsIntfMethods(ObjCCategoryImplDecl *CatImplDecl,
                                            ObjCCategoryDecl *CatClassDecl) {
  llvm::DenseSet<Selector> InsMap;
  // Check and see if instance methods in category interface have been
  // implemented in its implementation class.
  for (ObjCCategoryImplDecl::instmeth_iterator I =CatImplDecl->instmeth_begin(),
       E = CatImplDecl->instmeth_end(); I != E; ++I)
    InsMap.insert((*I)->getSelector());
  
  bool IncompleteImpl = false;
  for (ObjCCategoryDecl::instmeth_iterator I = CatClassDecl->instmeth_begin(),
       E = CatClassDecl->instmeth_end(); I != E; ++I)
    if (!InsMap.count((*I)->getSelector()))
      WarnUndefinedMethod(CatImplDecl->getLocation(), *I, IncompleteImpl);

  llvm::DenseSet<Selector> ClsMap;
  // Check and see if class methods in category interface have been
  // implemented in its implementation class.
  for (ObjCCategoryImplDecl::classmeth_iterator
       I = CatImplDecl->classmeth_begin(), E = CatImplDecl->classmeth_end();
       I != E; ++I)
    ClsMap.insert((*I)->getSelector());
  
  for (ObjCCategoryDecl::classmeth_iterator I = CatClassDecl->classmeth_begin(),
       E = CatClassDecl->classmeth_end(); I != E; ++I)
    if (!ClsMap.count((*I)->getSelector()))
      WarnUndefinedMethod(CatImplDecl->getLocation(), *I, IncompleteImpl);
  
  // Check the protocol list for unimplemented methods in the @implementation
  // class.
  for (ObjCCategoryDecl::protocol_iterator PI = CatClassDecl->protocol_begin(),
       E = CatClassDecl->protocol_end(); PI != E; ++PI)
    CheckProtocolMethodDefs(CatImplDecl->getLocation(), *PI, IncompleteImpl, 
                            InsMap, ClsMap);
}

/// ActOnForwardClassDeclaration - 
Action::DeclTy *
Sema::ActOnForwardClassDeclaration(SourceLocation AtClassLoc,
                                   IdentifierInfo **IdentList, unsigned NumElts) 
{
  llvm::SmallVector<ObjCInterfaceDecl*, 32> Interfaces;
  
  for (unsigned i = 0; i != NumElts; ++i) {
    // Check for another declaration kind with the same name.
    Decl *PrevDecl = LookupDecl(IdentList[i], Decl::IDNS_Ordinary, TUScope);
    if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
      // GCC apparently allows the following idiom:
      //
      // typedef NSObject < XCElementTogglerP > XCElementToggler;
      // @class XCElementToggler;
      //
      // FIXME: Make an extension? 
      TypedefDecl *TDD = dyn_cast<TypedefDecl>(PrevDecl);
      if (!TDD || !isa<ObjCInterfaceType>(TDD->getUnderlyingType())) {
        Diag(AtClassLoc, diag::err_redefinition_different_kind,
             IdentList[i]->getName());
        Diag(PrevDecl->getLocation(), diag::err_previous_definition);
      }
    }
    ObjCInterfaceDecl *IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl); 
    if (!IDecl) {  // Not already seen?  Make a forward decl.
      IDecl = ObjCInterfaceDecl::Create(Context, AtClassLoc, IdentList[i],
                                        SourceLocation(), true);
      ObjCInterfaceDecls[IdentList[i]] = IDecl;

      // Remember that this needs to be removed when the scope is popped.
      TUScope->AddDecl(IDecl);
    }

    Interfaces.push_back(IDecl);
  }
  
  return ObjCClassDecl::Create(Context, AtClassLoc,
                               &Interfaces[0], Interfaces.size());
}


/// MatchTwoMethodDeclarations - Checks that two methods have matching type and
/// returns true, or false, accordingly.
/// TODO: Handle protocol list; such as id<p1,p2> in type comparisons
bool Sema::MatchTwoMethodDeclarations(const ObjCMethodDecl *Method, 
                                      const ObjCMethodDecl *PrevMethod) {
  if (Method->getResultType().getCanonicalType() !=
      PrevMethod->getResultType().getCanonicalType())
    return false;
  for (unsigned i = 0, e = Method->getNumParams(); i != e; ++i) {
    ParmVarDecl *ParamDecl = Method->getParamDecl(i);
    ParmVarDecl *PrevParamDecl = PrevMethod->getParamDecl(i);
    if (Context.getCanonicalType(ParamDecl->getType()) !=
        Context.getCanonicalType(PrevParamDecl->getType()))
      return false;
  }
  return true;
}

void Sema::AddInstanceMethodToGlobalPool(ObjCMethodDecl *Method) {
  ObjCMethodList &FirstMethod = InstanceMethodPool[Method->getSelector()];
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

// Note: For class/category implemenations, allMethods/allProperties is
// always null.
void Sema::ActOnAtEnd(SourceLocation AtEndLoc, DeclTy *classDecl,
                      DeclTy **allMethods, unsigned allNum,
                      DeclTy **allProperties, unsigned pNum) {
  Decl *ClassDecl = static_cast<Decl *>(classDecl);

  // FIXME: If we don't have a ClassDecl, we have an error. We should consider
  // always passing in a decl. If the decl has an error, isInvalidDecl()
  // should be true.
  if (!ClassDecl)
    return;
    
  llvm::SmallVector<ObjCMethodDecl*, 32> insMethods;
  llvm::SmallVector<ObjCMethodDecl*, 16> clsMethods;
  
  llvm::DenseMap<Selector, const ObjCMethodDecl*> InsMap;
  llvm::DenseMap<Selector, const ObjCMethodDecl*> ClsMap;
  
  bool isInterfaceDeclKind = 
        isa<ObjCInterfaceDecl>(ClassDecl) || isa<ObjCCategoryDecl>(ClassDecl)
         || isa<ObjCProtocolDecl>(ClassDecl);
  bool checkIdenticalMethods = isa<ObjCImplementationDecl>(ClassDecl);
  
  if (pNum != 0) {
    if (ObjCInterfaceDecl *IDecl = dyn_cast<ObjCInterfaceDecl>(ClassDecl))
      IDecl->addProperties((ObjCPropertyDecl**)allProperties, pNum);
    else if (ObjCCategoryDecl *CDecl = dyn_cast<ObjCCategoryDecl>(ClassDecl))
      CDecl->addProperties((ObjCPropertyDecl**)allProperties, pNum);
    else if (ObjCProtocolDecl *PDecl = dyn_cast<ObjCProtocolDecl>(ClassDecl))
          PDecl->addProperties((ObjCPropertyDecl**)allProperties, pNum);
    else
      assert(false && "ActOnAtEnd - property declaration misplaced");
  }
  
  for (unsigned i = 0; i < allNum; i++ ) {
    ObjCMethodDecl *Method =
      cast_or_null<ObjCMethodDecl>(static_cast<Decl*>(allMethods[i]));

    if (!Method) continue;  // Already issued a diagnostic.
    if (Method->isInstance()) {
      /// Check for instance method of the same name with incompatible types
      const ObjCMethodDecl *&PrevMethod = InsMap[Method->getSelector()];
      bool match = PrevMethod ? MatchTwoMethodDeclarations(Method, PrevMethod) 
                              : false;
      if (isInterfaceDeclKind && PrevMethod && !match 
          || checkIdenticalMethods && match) {
          Diag(Method->getLocation(), diag::error_duplicate_method_decl,
               Method->getSelector().getName());
          Diag(PrevMethod->getLocation(), diag::err_previous_declaration);
      } else {
        insMethods.push_back(Method);
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
      if (isInterfaceDeclKind && PrevMethod && !match 
          || checkIdenticalMethods && match) {
        Diag(Method->getLocation(), diag::error_duplicate_method_decl,
             Method->getSelector().getName());
        Diag(PrevMethod->getLocation(), diag::err_previous_declaration);
      } else {
        clsMethods.push_back(Method);
        ClsMap[Method->getSelector()] = Method;
        /// The following allows us to typecheck messages to "Class".
        AddFactoryMethodToGlobalPool(Method);
      }
    }
  }
  
  if (ObjCInterfaceDecl *I = dyn_cast<ObjCInterfaceDecl>(ClassDecl)) {
    // Compares properties declaraed in this class to those of its 
    // super class.
    ComparePropertiesInBaseAndSuper(I);
    MergeProtocolPropertiesIntoClass(I, I);
    for (ObjCInterfaceDecl::classprop_iterator P = I->classprop_begin(),
         E = I->classprop_end(); P != E; ++P) {
      // FIXME: It would be really nice if we could avoid this. Injecting 
      // methods into the interface makes it hard to distinguish "real" methods
      // from synthesized "property" methods (that aren't in the source). 
      // This complicicates the rewriter's life.
      I->addPropertyMethods(Context, *P, insMethods);
    }
    I->addMethods(&insMethods[0], insMethods.size(),
                  &clsMethods[0], clsMethods.size(), AtEndLoc);
    
  } else if (ObjCProtocolDecl *P = dyn_cast<ObjCProtocolDecl>(ClassDecl)) {
    P->addMethods(&insMethods[0], insMethods.size(),
                  &clsMethods[0], clsMethods.size(), AtEndLoc);
  }
  else if (ObjCCategoryDecl *C = dyn_cast<ObjCCategoryDecl>(ClassDecl)) {
    C->addMethods(&insMethods[0], insMethods.size(),
                  &clsMethods[0], clsMethods.size(), AtEndLoc);
  }
  else if (ObjCImplementationDecl *IC = 
                dyn_cast<ObjCImplementationDecl>(ClassDecl)) {
    IC->setLocEnd(AtEndLoc);
    if (ObjCInterfaceDecl* IDecl = getObjCInterfaceDecl(IC->getIdentifier()))
      ImplMethodsVsClassMethods(IC, IDecl);
  } else {
    ObjCCategoryImplDecl* CatImplClass = cast<ObjCCategoryImplDecl>(ClassDecl);
    CatImplClass->setLocEnd(AtEndLoc);
    ObjCInterfaceDecl* IDecl = CatImplClass->getClassInterface();
    // Find category interface decl and then check that all methods declared
    // in this interface is implemented in the category @implementation.
    if (IDecl) {
      for (ObjCCategoryDecl *Categories = IDecl->getCategoryList();
           Categories; Categories = Categories->getNextClassCategory()) {
        if (Categories->getIdentifier() == CatImplClass->getIdentifier()) {
          ImplCategoryMethodsVsIntfMethods(CatImplClass, Categories);
          break;
        }
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

Sema::DeclTy *Sema::ActOnMethodDeclaration(
    SourceLocation MethodLoc, SourceLocation EndLoc,
    tok::TokenKind MethodType, DeclTy *classDecl,
    ObjCDeclSpec &ReturnQT, TypeTy *ReturnType,
    Selector Sel,
    // optional arguments. The number of types/arguments is obtained
    // from the Sel.getNumArgs().
    ObjCDeclSpec *ArgQT, TypeTy **ArgTypes, IdentifierInfo **ArgNames,
    AttributeList *AttrList, tok::ObjCKeywordKind MethodDeclKind,
    bool isVariadic) {
  Decl *ClassDecl = static_cast<Decl*>(classDecl);

  // Make sure we can establish a context for the method.
  if (!ClassDecl) {
    Diag(MethodLoc, diag::error_missing_method_context);
    return 0;
  }
  QualType resultDeclType;
  
  if (ReturnType)
    resultDeclType = QualType::getFromOpaquePtr(ReturnType);
  else // get the type for "id".
    resultDeclType = Context.getObjCIdType();
  
  ObjCMethodDecl* ObjCMethod = 
    ObjCMethodDecl::Create(Context, MethodLoc, EndLoc, Sel, resultDeclType,
                           ClassDecl, AttrList, 
                           MethodType == tok::minus, isVariadic,
                           false,
                           MethodDeclKind == tok::objc_optional ? 
                           ObjCMethodDecl::Optional : 
                           ObjCMethodDecl::Required);
  
  llvm::SmallVector<ParmVarDecl*, 16> Params;
  
  for (unsigned i = 0; i < Sel.getNumArgs(); i++) {
    // FIXME: arg->AttrList must be stored too!
    QualType argType;
    
    if (ArgTypes[i])
      argType = QualType::getFromOpaquePtr(ArgTypes[i]);
    else
      argType = Context.getObjCIdType();
    ParmVarDecl* Param = ParmVarDecl::Create(Context, ObjCMethod,
                                             SourceLocation(/*FIXME*/),
                                             ArgNames[i], argType,
                                             VarDecl::None, 0, 0);
    Param->setObjCDeclQualifier(
      CvtQTToAstBitMask(ArgQT[i].getObjCDeclQualifier()));
    Params.push_back(Param);
  }

  ObjCMethod->setMethodParams(&Params[0], Sel.getNumArgs());
  ObjCMethod->setObjCDeclQualifier(
    CvtQTToAstBitMask(ReturnQT.getObjCDeclQualifier()));
  const ObjCMethodDecl *PrevMethod = 0;
 
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
    Diag(ObjCMethod->getLocation(), diag::error_duplicate_method_decl,
        ObjCMethod->getSelector().getName());
    Diag(PrevMethod->getLocation(), diag::err_previous_declaration);
  } 
  return ObjCMethod;
}

Sema::DeclTy *Sema::ActOnProperty(Scope *S, SourceLocation AtLoc, 
                                  FieldDeclarator &FD,
                                  ObjCDeclSpec &ODS,
                                  Selector GetterSel,
                                  Selector SetterSel,
                                  tok::ObjCKeywordKind MethodImplKind) {
  QualType T = GetTypeForDeclarator(FD.D, S);
  ObjCPropertyDecl *PDecl = ObjCPropertyDecl::Create(Context, AtLoc, 
                                                     FD.D.getIdentifier(), T);
  // Regardless of setter/getter attribute, we save the default getter/setter
  // selector names in anticipation of declaration of setter/getter methods.
  PDecl->setGetterName(GetterSel);
  PDecl->setSetterName(SetterSel);
  
  if (ODS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_readonly)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readonly);
  
  if (ODS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_getter)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_getter);
  
  if (ODS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_setter)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_setter);
  
  if (ODS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_assign)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_assign);
  
  if (ODS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_readwrite)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readwrite);
  
  if (ODS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_retain)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_retain);
  
  if (ODS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_copy)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_copy);
  
  if (ODS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_nonatomic)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_nonatomic);
  
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
Sema::DeclTy *Sema::ActOnPropertyImplDecl(SourceLocation AtLoc, 
                                          SourceLocation PropertyLoc,
                                          bool Synthesize, 
                                          DeclTy *ClassCatImpDecl,
                                          IdentifierInfo *PropertyId,
                                          IdentifierInfo *PropertyIvar) {
  Decl *ClassImpDecl = static_cast<Decl*>(ClassCatImpDecl);
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
    IDecl = getObjCInterfaceDecl(IC->getIdentifier());
    // We always synthesize an interface for an implementation
    // without an interface decl. So, IDecl is always non-zero.
    assert(IDecl && 
           "ActOnPropertyImplDecl - @implementation without @interface");
    
    // Look for this property declaration in the @implementation's @interface
    property = IDecl->FindPropertyDeclaration(PropertyId);
    if (!property) {
       Diag(PropertyLoc, diag::error_bad_property_decl, IDecl->getName());
      return 0;
    }
  }
  else if ((CatImplClass = dyn_cast<ObjCCategoryImplDecl>(ClassImpDecl))) {
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
      Diag(PropertyLoc, diag::error_bad_category_property_decl, 
           Category->getName());
      return 0;
    }
  }
  else {
    Diag(AtLoc, diag::error_bad_property_context);
    return 0;
  }
  ObjCIvarDecl *Ivar = 0;
  // Check that we have a valid, previously declared ivar for @synthesize
  if (Synthesize) {
    // @synthesize
    if (!PropertyIvar)
      PropertyIvar = PropertyId;
    // Check that this is a previously declared 'ivar' in 'IDecl' interface
    Ivar = IDecl->FindIvarDeclaration(PropertyIvar);
    if (!Ivar) {
      Diag(PropertyLoc, diag::error_missing_property_ivar_decl, 
           PropertyId->getName());
      return 0;
    }
    // Check that type of property and its ivar match. 
    if (Ivar->getCanonicalType() != property->getCanonicalType()) {
      Diag(PropertyLoc, diag::error_property_ivar_type, property->getName(),
           Ivar->getName());
      return 0;
    }
      
  } else if (PropertyIvar) {
    // @dynamic
    Diag(PropertyLoc, diag::error_dynamic_property_ivar_decl);
    return 0;
  }
  assert (property && "ActOnPropertyImplDecl - property declaration missing");
  ObjCPropertyImplDecl *PIDecl = 
    ObjCPropertyImplDecl::Create(Context, AtLoc, PropertyLoc, property, 
                                 (Synthesize ? 
                                  ObjCPropertyImplDecl::OBJC_PR_IMPL_SYNTHSIZE 
                                  : ObjCPropertyImplDecl::OBJC_PR_IMPL_DYNAMIC),
                                  Ivar);
  if (IC)
    IC->addPropertyImplementation(PIDecl);
  else
    CatImplClass->addPropertyImplementation(PIDecl);
    
  return PIDecl;
}
