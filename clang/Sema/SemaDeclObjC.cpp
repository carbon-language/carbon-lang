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
  assert(CurFunctionDecl == 0 && "Method parsing confused");
  ObjCMethodDecl *MDecl = dyn_cast<ObjCMethodDecl>(static_cast<Decl *>(D));
  assert(MDecl != 0 && "Not a method declarator!");

  // Allow the rest of sema to find private method decl implementations.
  if (MDecl->isInstance())
    AddInstanceMethodToGlobalPool(MDecl);
  else
    AddFactoryMethodToGlobalPool(MDecl);
  
  // Allow all of Sema to see that we are entering a method definition.
  CurMethodDecl = MDecl;

  // Create Decl objects for each parameter, entrring them in the scope for
  // binding to their use.
  struct DeclaratorChunk::ParamInfo PI;

  // Insert the invisible arguments, self and _cmd!
  PI.Ident = &Context.Idents.get("self");
  PI.IdentLoc = SourceLocation(); // synthesized vars have a null location.
  PI.InvalidType = false;
  if (MDecl->isInstance()) {
    QualType selfTy = Context.getObjCInterfaceType(MDecl->getClassInterface());
    selfTy = Context.getPointerType(selfTy);
    PI.TypeInfo = selfTy.getAsOpaquePtr();
  } else
    PI.TypeInfo = Context.getObjCIdType().getAsOpaquePtr();
  CurMethodDecl->setSelfDecl(ActOnParamDeclarator(PI, FnBodyScope));
  
  PI.Ident = &Context.Idents.get("_cmd");
  PI.TypeInfo = Context.getObjCSelType().getAsOpaquePtr();
  ActOnParamDeclarator(PI, FnBodyScope);
  
  for (int i = 0; i <  MDecl->getNumParams(); i++) {
    ParmVarDecl *PDecl = MDecl->getParamDecl(i);
    PI.Ident = PDecl->getIdentifier();
    PI.IdentLoc = PDecl->getLocation(); // user vars have a real location.
    PI.TypeInfo = PDecl->getType().getAsOpaquePtr();
    ActOnParamDeclarator(PI, FnBodyScope);
  }
}

Sema::DeclTy *Sema::ActOnStartClassInterface(
                    SourceLocation AtInterfaceLoc,
                    IdentifierInfo *ClassName, SourceLocation ClassLoc,
                    IdentifierInfo *SuperName, SourceLocation SuperLoc,
                    IdentifierInfo **ProtocolNames, unsigned NumProtocols,
                    SourceLocation EndProtoLoc, AttributeList *AttrList) {
  assert(ClassName && "Missing class identifier");
  
  // Check for another declaration kind with the same name.
  ScopedDecl *PrevDecl = LookupInterfaceDecl(ClassName);
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
      IDecl->AllocIntfRefProtocols(NumProtocols);
    }
  }
  else {
    IDecl = new ObjCInterfaceDecl(AtInterfaceLoc, NumProtocols, ClassName);
  
    // Chain & install the interface decl into the identifier.
    IDecl->setNext(ClassName->getFETokenInfo<ScopedDecl>());
    ClassName->setFETokenInfo(IDecl);
    
    // Remember that this needs to be removed when the scope is popped.
    TUScope->AddDecl(IDecl);
  }
  
  if (SuperName) {
    ObjCInterfaceDecl* SuperClassEntry = 0;
    // Check if a different kind of symbol declared in this scope.
    PrevDecl = LookupInterfaceDecl(SuperName);
    if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
      Diag(SuperLoc, diag::err_redefinition_different_kind,
           SuperName->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
    }
    else {
      // Check that super class is previously defined
      SuperClassEntry = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl); 
                              
      if (!SuperClassEntry || SuperClassEntry->isForwardDecl()) {
        Diag(AtInterfaceLoc, diag::err_undef_superclass, 
             SuperClassEntry ? SuperClassEntry->getName() 
                             : SuperName->getName(),
             ClassName->getName()); 
      }
    }
    IDecl->setSuperClass(SuperClassEntry);
    IDecl->setLocEnd(SuperLoc);
  } else { // we have a root class.
    IDecl->setLocEnd(ClassLoc);
  }
  
  /// Check then save referenced protocols
  if (NumProtocols) {
    for (unsigned int i = 0; i != NumProtocols; i++) {
      ObjCProtocolDecl* RefPDecl = ObjCProtocols[ProtocolNames[i]];
      if (!RefPDecl || RefPDecl->isForwardDecl())
        Diag(ClassLoc, diag::warn_undef_protocolref,
             ProtocolNames[i]->getName(),
             ClassName->getName());
      IDecl->setIntfRefProtocols(i, RefPDecl);
    }
    IDecl->setLocEnd(EndProtoLoc);
  }
  return IDecl;
}

/// ActOnCompatiblityAlias - this action is called after complete parsing of
/// @compaatibility_alias declaration. It sets up the alias relationships.
Sema::DeclTy *Sema::ActOnCompatiblityAlias(
                      SourceLocation AtCompatibilityAliasLoc,
                      IdentifierInfo *AliasName,  SourceLocation AliasLocation,
                      IdentifierInfo *ClassName, SourceLocation ClassLocation) {
  // Look for previous declaration of alias name
  ScopedDecl *ADecl = LookupScopedDecl(AliasName, Decl::IDNS_Ordinary,
                                       AliasLocation, TUScope);
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
  ScopedDecl *CDecl = LookupScopedDecl(ClassName, Decl::IDNS_Ordinary,
                                       ClassLocation, TUScope);
  if (!CDecl || !isa<ObjCInterfaceDecl>(CDecl)) {
    Diag(ClassLocation, diag::warn_undef_interface,
         ClassName->getName());
    if (CDecl)
      Diag(CDecl->getLocation(), diag::warn_previous_declaration);
    return 0;
  }
  // Everything checked out, instantiate a new alias declaration ast
  ObjCCompatibleAliasDecl *AliasDecl = 
    new ObjCCompatibleAliasDecl(AtCompatibilityAliasLoc, 
                                AliasName,
                                dyn_cast<ObjCInterfaceDecl>(CDecl));
    
  // Chain & install the interface decl into the identifier.
  AliasDecl->setNext(AliasName->getFETokenInfo<ScopedDecl>());
  AliasName->setFETokenInfo(AliasDecl);
  return AliasDecl;
}

Sema::DeclTy *Sema::ActOnStartProtocolInterface(
                SourceLocation AtProtoInterfaceLoc,
                IdentifierInfo *ProtocolName, SourceLocation ProtocolLoc,
                IdentifierInfo **ProtoRefNames, unsigned NumProtoRefs,
                SourceLocation EndProtoLoc) {
  assert(ProtocolName && "Missing protocol identifier");
  ObjCProtocolDecl *PDecl = ObjCProtocols[ProtocolName];
  if (PDecl) {
    // Protocol already seen. Better be a forward protocol declaration
    if (!PDecl->isForwardDecl())
      Diag(ProtocolLoc, diag::err_duplicate_protocol_def, 
           ProtocolName->getName());
    else {
      PDecl->setForwardDecl(false);
      PDecl->AllocReferencedProtocols(NumProtoRefs);
    }
  }
  else {
    PDecl = new ObjCProtocolDecl(AtProtoInterfaceLoc, NumProtoRefs, 
                                 ProtocolName);
    ObjCProtocols[ProtocolName] = PDecl;
  }    
  
  if (NumProtoRefs) {
    /// Check then save referenced protocols
    for (unsigned int i = 0; i != NumProtoRefs; i++) {
      ObjCProtocolDecl* RefPDecl = ObjCProtocols[ProtoRefNames[i]];
      if (!RefPDecl || RefPDecl->isForwardDecl())
        Diag(ProtocolLoc, diag::warn_undef_protocolref,
             ProtoRefNames[i]->getName(),
             ProtocolName->getName());
      PDecl->setReferencedProtocols(i, RefPDecl);
    }
    PDecl->setLocEnd(EndProtoLoc);
  }
  return PDecl;
}

/// FindProtocolDeclaration - This routine looks up protocols and
/// issuer error if they are not declared. It returns list of protocol
/// declarations in its 'Protocols' argument.
void
Sema::FindProtocolDeclaration(SourceLocation TypeLoc,
                              IdentifierInfo **ProtocolId,
                              unsigned NumProtocols,
                              llvm::SmallVector<DeclTy *,8> &Protocols) {
  for (unsigned i = 0; i != NumProtocols; ++i) {
    ObjCProtocolDecl *PDecl = ObjCProtocols[ProtocolId[i]];
    if (!PDecl)
      Diag(TypeLoc, diag::err_undeclared_protocol, 
           ProtocolId[i]->getName());
    else
      Protocols.push_back(PDecl); 
  }
}

/// ActOnForwardProtocolDeclaration - 
Action::DeclTy *
Sema::ActOnForwardProtocolDeclaration(SourceLocation AtProtocolLoc,
        IdentifierInfo **IdentList, unsigned NumElts) {
  llvm::SmallVector<ObjCProtocolDecl*, 32> Protocols;
  
  for (unsigned i = 0; i != NumElts; ++i) {
    IdentifierInfo *P = IdentList[i];
    ObjCProtocolDecl *PDecl = ObjCProtocols[P];
    if (!PDecl)  { // Not already seen?
      // FIXME: Pass in the location of the identifier!
      PDecl = new ObjCProtocolDecl(AtProtocolLoc, 0, P, true);
      ObjCProtocols[P] = PDecl;
    }
    
    Protocols.push_back(PDecl);
  }
  return new ObjCForwardProtocolDecl(AtProtocolLoc,
                                     &Protocols[0], Protocols.size());
}

Sema::DeclTy *Sema::ActOnStartCategoryInterface(
                      SourceLocation AtInterfaceLoc,
                      IdentifierInfo *ClassName, SourceLocation ClassLoc,
                      IdentifierInfo *CategoryName, SourceLocation CategoryLoc,
                      IdentifierInfo **ProtoRefNames, unsigned NumProtoRefs,
                      SourceLocation EndProtoLoc) {
  ObjCInterfaceDecl *IDecl = getObjCInterfaceDecl(ClassName);
  
  /// Check that class of this category is already completely declared.
  if (!IDecl || IDecl->isForwardDecl()) {
    Diag(ClassLoc, diag::err_undef_interface, ClassName->getName());
    return 0;
  }
  ObjCCategoryDecl *CDecl = new ObjCCategoryDecl(AtInterfaceLoc, NumProtoRefs,
                                                 CategoryName);
  CDecl->setClassInterface(IDecl);
  /// Check for duplicate interface declaration for this category
  ObjCCategoryDecl *CDeclChain;
  for (CDeclChain = IDecl->getCategoryList(); CDeclChain;
       CDeclChain = CDeclChain->getNextClassCategory()) {
    if (CDeclChain->getIdentifier() == CategoryName) {
      Diag(CategoryLoc, diag::err_dup_category_def, ClassName->getName(),
           CategoryName->getName());
      break;
    }
  }
  if (!CDeclChain)
    CDecl->insertNextClassCategory();

  if (NumProtoRefs) {
    /// Check then save referenced protocols
    for (unsigned int i = 0; i != NumProtoRefs; i++) {
      ObjCProtocolDecl* RefPDecl = ObjCProtocols[ProtoRefNames[i]];
      if (!RefPDecl || RefPDecl->isForwardDecl()) {
        Diag(CategoryLoc, diag::warn_undef_protocolref,
             ProtoRefNames[i]->getName(),
             CategoryName->getName());
      }
      CDecl->setCatReferencedProtocols(i, RefPDecl);
    }
    CDecl->setLocEnd(EndProtoLoc);
  }
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
  ObjCCategoryImplDecl *CDecl = new ObjCCategoryImplDecl(AtCatImplLoc, 
                                                         CatName, IDecl);
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
  ScopedDecl *PrevDecl = LookupInterfaceDecl(ClassName);
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
    PrevDecl = LookupInterfaceDecl(SuperClassname);
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
    IDecl = new ObjCInterfaceDecl(AtClassImplLoc, 0, ClassName, 
				  false, true);
    IDecl->setNext(ClassName->getFETokenInfo<ScopedDecl>());
    ClassName->setFETokenInfo(IDecl);
    IDecl->setSuperClass(SDecl);
    IDecl->setLocEnd(ClassLoc);
    
    // Remember that this needs to be removed when the scope is popped.
    TUScope->AddDecl(IDecl);
  }
  
  ObjCImplementationDecl* IMPDecl = 
  new ObjCImplementationDecl(AtClassImplLoc, ClassName, IDecl, SDecl);
  
  // Check that there is no duplicate implementation of this class.
  if (ObjCImplementations[ClassName])
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

/// CheckProtocolMethodDefs - This routine checks unimpletented methods
/// Declared in protocol, and those referenced by it.
void Sema::CheckProtocolMethodDefs(ObjCProtocolDecl *PDecl,
                                   bool& IncompleteImpl,
             const llvm::DenseSet<Selector> &InsMap,
             const llvm::DenseSet<Selector> &ClsMap) {
  // check unimplemented instance methods.
  for (ObjCProtocolDecl::instmeth_iterator I = PDecl->instmeth_begin(), 
       E = PDecl->instmeth_end(); I != E; ++I) {
    ObjCMethodDecl *method = *I;
    if (!InsMap.count(method->getSelector()) && 
        method->getImplementationControl() != ObjCMethodDecl::Optional) {
      Diag(method->getLocation(), diag::warn_undef_method_impl,
           method->getSelector().getName());
      IncompleteImpl = true;
    }
  }
  // check unimplemented class methods
  for (ObjCProtocolDecl::classmeth_iterator I = PDecl->classmeth_begin(), 
       E = PDecl->classmeth_end(); I != E; ++I) {
    ObjCMethodDecl *method = *I;
    if (!ClsMap.count(method->getSelector()) &&
        method->getImplementationControl() != ObjCMethodDecl::Optional) {
      Diag(method->getLocation(), diag::warn_undef_method_impl,
           method->getSelector().getName());
      IncompleteImpl = true;
    }
  }
  // Check on this protocols's referenced protocols, recursively
  ObjCProtocolDecl** RefPDecl = PDecl->getReferencedProtocols();
  for (unsigned i = 0; i < PDecl->getNumReferencedProtocols(); i++)
    CheckProtocolMethodDefs(RefPDecl[i], IncompleteImpl, InsMap, ClsMap);
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
    if (!InsMap.count((*I)->getSelector())) {
      Diag((*I)->getLocation(), diag::warn_undef_method_impl,
           (*I)->getSelector().getName());
      IncompleteImpl = true;
    }
      
  llvm::DenseSet<Selector> ClsMap;
  // Check and see if class methods in class interface have been
  // implemented in the implementation class.
  for (ObjCImplementationDecl::classmeth_iterator I =IMPDecl->classmeth_begin(),
       E = IMPDecl->classmeth_end(); I != E; ++I)
    ClsMap.insert((*I)->getSelector());
  
  for (ObjCInterfaceDecl::classmeth_iterator I = IDecl->classmeth_begin(),
       E = IDecl->classmeth_end(); I != E; ++I)
    if (!ClsMap.count((*I)->getSelector())) {
      Diag((*I)->getLocation(), diag::warn_undef_method_impl,
           (*I)->getSelector().getName());
      IncompleteImpl = true;
    }
  
  // Check the protocol list for unimplemented methods in the @implementation
  // class.
  ObjCProtocolDecl** protocols = IDecl->getReferencedProtocols();
  for (unsigned i = 0; i < IDecl->getNumIntfRefProtocols(); i++)
    CheckProtocolMethodDefs(protocols[i], IncompleteImpl, InsMap, ClsMap);

  if (IncompleteImpl)
    Diag(IMPDecl->getLocation(), diag::warn_incomplete_impl_class, 
         IMPDecl->getName());
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
    if (!InsMap.count((*I)->getSelector())) {
      Diag((*I)->getLocation(), diag::warn_undef_method_impl,
           (*I)->getSelector().getName());
      IncompleteImpl = true;
    }
  llvm::DenseSet<Selector> ClsMap;
  // Check and see if class methods in category interface have been
  // implemented in its implementation class.
  for (ObjCCategoryImplDecl::classmeth_iterator
       I = CatImplDecl->classmeth_begin(), E = CatImplDecl->classmeth_end();
       I != E; ++I)
    ClsMap.insert((*I)->getSelector());
  
  for (ObjCCategoryDecl::classmeth_iterator I = CatClassDecl->classmeth_begin(),
       E = CatClassDecl->classmeth_end(); I != E; ++I)
    if (!ClsMap.count((*I)->getSelector())) {
      Diag((*I)->getLocation(), diag::warn_undef_method_impl,
           (*I)->getSelector().getName());
      IncompleteImpl = true;
    }
  
  // Check the protocol list for unimplemented methods in the @implementation
  // class.
  ObjCProtocolDecl** protocols = CatClassDecl->getReferencedProtocols();
  for (unsigned i = 0; i < CatClassDecl->getNumReferencedProtocols(); i++) {
    ObjCProtocolDecl* PDecl = protocols[i];
    CheckProtocolMethodDefs(PDecl, IncompleteImpl, InsMap, ClsMap);
  }
  if (IncompleteImpl)
    Diag(CatImplDecl->getLocation(), diag::warn_incomplete_impl_category, 
         CatClassDecl->getName());
}

/// ActOnForwardClassDeclaration - 
Action::DeclTy *
Sema::ActOnForwardClassDeclaration(SourceLocation AtClassLoc,
                                   IdentifierInfo **IdentList, unsigned NumElts) 
{
  llvm::SmallVector<ObjCInterfaceDecl*, 32> Interfaces;
  
  for (unsigned i = 0; i != NumElts; ++i) {
    // Check for another declaration kind with the same name.
    ScopedDecl *PrevDecl = LookupInterfaceDecl(IdentList[i]);
    if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
      Diag(AtClassLoc, diag::err_redefinition_different_kind,
           IdentList[i]->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
    }
    ObjCInterfaceDecl *IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl); 
    if (!IDecl) {  // Not already seen?  Make a forward decl.
      IDecl = new ObjCInterfaceDecl(AtClassLoc, 0, IdentList[i], true);
      // Chain & install the interface decl into the identifier.
      IDecl->setNext(IdentList[i]->getFETokenInfo<ScopedDecl>());
      IdentList[i]->setFETokenInfo(IDecl);

      // Remember that this needs to be removed when the scope is popped.
      TUScope->AddDecl(IDecl);
    }

    Interfaces.push_back(IDecl);
  }
  
  return new ObjCClassDecl(AtClassLoc, &Interfaces[0], Interfaces.size());
}


/// MatchTwoMethodDeclarations - Checks that two methods have matching type and
/// returns true, or false, accordingly.
/// TODO: Handle protocol list; such as id<p1,p2> in type comparisons
bool Sema::MatchTwoMethodDeclarations(const ObjCMethodDecl *Method, 
                                      const ObjCMethodDecl *PrevMethod) {
  if (Method->getResultType().getCanonicalType() !=
      PrevMethod->getResultType().getCanonicalType())
    return false;
  for (int i = 0; i < Method->getNumParams(); i++) {
    ParmVarDecl *ParamDecl = Method->getParamDecl(i);
    ParmVarDecl *PrevParamDecl = PrevMethod->getParamDecl(i);
    if (ParamDecl->getCanonicalType() != PrevParamDecl->getCanonicalType())
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
        (isa<ObjCInterfaceDecl>(ClassDecl) || isa<ObjCCategoryDecl>(ClassDecl)
         || isa<ObjCProtocolDecl>(ClassDecl));
  bool checkIdenticalMethods = isa<ObjCImplementationDecl>(ClassDecl);
  
  // TODO: property declaration in category and protocols.
  if (pNum != 0 && isa<ObjCInterfaceDecl>(ClassDecl)) {
    ObjCPropertyDecl **properties = new ObjCPropertyDecl*[pNum];
    memcpy(properties, allProperties, pNum*sizeof(ObjCPropertyDecl*));
    dyn_cast<ObjCInterfaceDecl>(ClassDecl)->setPropertyDecls(properties);
    dyn_cast<ObjCInterfaceDecl>(ClassDecl)->setNumPropertyDecl(pNum);
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
    tok::TokenKind MethodType, DeclTy *ClassDecl,
    ObjCDeclSpec &ReturnQT, TypeTy *ReturnType,
    Selector Sel,
    // optional arguments. The number of types/arguments is obtained
    // from the Sel.getNumArgs().
    ObjCDeclSpec *ArgQT, TypeTy **ArgTypes, IdentifierInfo **ArgNames,
    AttributeList *AttrList, tok::ObjCKeywordKind MethodDeclKind,
    bool isVariadic) {
  llvm::SmallVector<ParmVarDecl*, 16> Params;
  
  for (unsigned i = 0; i < Sel.getNumArgs(); i++) {
    // FIXME: arg->AttrList must be stored too!
    QualType argType;
    
    if (ArgTypes[i])
      argType = QualType::getFromOpaquePtr(ArgTypes[i]);
    else
      argType = Context.getObjCIdType();
    ParmVarDecl* Param = new ParmVarDecl(SourceLocation(/*FIXME*/), ArgNames[i], 
                                         argType, VarDecl::None, 0);
    Param->setObjCDeclQualifier(
      CvtQTToAstBitMask(ArgQT[i].getObjCDeclQualifier()));
    Params.push_back(Param);
  }
  QualType resultDeclType;
  
  if (ReturnType)
    resultDeclType = QualType::getFromOpaquePtr(ReturnType);
  else // get the type for "id".
    resultDeclType = Context.getObjCIdType();
  
  Decl *CDecl = static_cast<Decl*>(ClassDecl);
  ObjCMethodDecl* ObjCMethod =  new ObjCMethodDecl(MethodLoc, EndLoc, Sel,
                                      resultDeclType,
                                      CDecl,
                                      0, -1, AttrList, 
                                      MethodType == tok::minus, isVariadic,
                                      MethodDeclKind == tok::objc_optional ? 
                                      ObjCMethodDecl::Optional : 
                                      ObjCMethodDecl::Required);
  ObjCMethod->setMethodParams(&Params[0], Sel.getNumArgs());
  ObjCMethod->setObjCDeclQualifier(
    CvtQTToAstBitMask(ReturnQT.getObjCDeclQualifier()));
  const ObjCMethodDecl *PrevMethod = 0;
 
  // For implementations (which can be very "coarse grain"), we add the 
  // method now. This allows the AST to implement lookup methods that work 
  // incrementally (without waiting until we parse the @end). It also allows 
  // us to flag multiple declaration errors as they occur.
  if (ObjCImplementationDecl *ImpDecl = 
        dyn_cast<ObjCImplementationDecl>(CDecl)) {
    if (MethodType == tok::minus) {
      PrevMethod = ImpDecl->getInstanceMethod(Sel);
      ImpDecl->addInstanceMethod(ObjCMethod);
    } else {
      PrevMethod = ImpDecl->getClassMethod(Sel);
      ImpDecl->addClassMethod(ObjCMethod);
    }
  } 
  else if (ObjCCategoryImplDecl *CatImpDecl = 
            dyn_cast<ObjCCategoryImplDecl>(CDecl)) {
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

Sema::DeclTy *Sema::ActOnAddObjCProperties(SourceLocation AtLoc, 
  DeclTy **allProperties, unsigned NumProperties, ObjCDeclSpec &DS) {
  ObjCPropertyDecl *PDecl = new ObjCPropertyDecl(AtLoc);
  
  if(DS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_readonly)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readonly);
  
  if(DS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_getter) {
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_getter);
    PDecl->setGetterName(DS.getGetterName());
  }
  
  if(DS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_setter) {
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_setter);
    PDecl->setSetterName(DS.getSetterName());
  }
  
  if(DS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_assign)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_assign);
  
  if(DS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_readwrite)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readwrite);
  
  if(DS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_retain)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_retain);
  
  if(DS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_copy)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_copy);
  
  if(DS.getPropertyAttributes() & ObjCDeclSpec::DQ_PR_nonatomic)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_nonatomic);
  
  PDecl->setNumPropertyDecls(NumProperties);
  if (NumProperties != 0) {
    ObjCIvarDecl **properties = new ObjCIvarDecl*[NumProperties];
    memcpy(properties, allProperties, NumProperties*sizeof(ObjCIvarDecl*));
    PDecl->setPropertyDecls(properties);
  }
  return PDecl;
}

