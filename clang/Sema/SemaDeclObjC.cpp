//===--- SemaDeclObjC.cpp - Semantic Analysis for ObjC Declarations -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

/// ObjcActOnStartOfMethodDef - This routine sets up parameters; invisible
/// and user declared, in the method definition's AST.
void Sema::ObjcActOnStartOfMethodDef(Scope *FnBodyScope, DeclTy *D) {
  assert(CurFunctionDecl == 0 && "Method parsing confused");
  ObjcMethodDecl *MDecl = dyn_cast<ObjcMethodDecl>(static_cast<Decl *>(D));
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
    QualType selfTy = Context.getObjcInterfaceType(MDecl->getClassInterface());
    selfTy = Context.getPointerType(selfTy);
    PI.TypeInfo = selfTy.getAsOpaquePtr();
  } else
    PI.TypeInfo = Context.getObjcIdType().getAsOpaquePtr();
  CurMethodDecl->setSelfDecl(ActOnParamDeclarator(PI, FnBodyScope));
  
  PI.Ident = &Context.Idents.get("_cmd");
  PI.TypeInfo = Context.getObjcSelType().getAsOpaquePtr();
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
  if (PrevDecl && !isa<ObjcInterfaceDecl>(PrevDecl)) {
    Diag(ClassLoc, diag::err_redefinition_different_kind,
         ClassName->getName());
    Diag(PrevDecl->getLocation(), diag::err_previous_definition);
  }
  
  ObjcInterfaceDecl* IDecl = dyn_cast_or_null<ObjcInterfaceDecl>(PrevDecl);
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
    IDecl = new ObjcInterfaceDecl(AtInterfaceLoc, NumProtocols, ClassName);
  
    // Chain & install the interface decl into the identifier.
    IDecl->setNext(ClassName->getFETokenInfo<ScopedDecl>());
    ClassName->setFETokenInfo(IDecl);
    
    // Remember that this needs to be removed when the scope is popped.
    TUScope->AddDecl(IDecl);
  }
  
  if (SuperName) {
    ObjcInterfaceDecl* SuperClassEntry = 0;
    // Check if a different kind of symbol declared in this scope.
    PrevDecl = LookupInterfaceDecl(SuperName);
    if (PrevDecl && !isa<ObjcInterfaceDecl>(PrevDecl)) {
      Diag(SuperLoc, diag::err_redefinition_different_kind,
           SuperName->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
    }
    else {
      // Check that super class is previously defined
      SuperClassEntry = dyn_cast_or_null<ObjcInterfaceDecl>(PrevDecl); 
                              
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
      ObjcProtocolDecl* RefPDecl = ObjcProtocols[ProtocolNames[i]];
      if (!RefPDecl || RefPDecl->isForwardDecl())
        Diag(ClassLoc, diag::warn_undef_protocolref,
             ProtocolNames[i]->getName(),
             ClassName->getName());
      IDecl->setIntfRefProtocols((int)i, RefPDecl);
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
    if (isa<ObjcCompatibleAliasDecl>(ADecl)) {
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
  if (!CDecl || !isa<ObjcInterfaceDecl>(CDecl)) {
    Diag(ClassLocation, diag::warn_undef_interface,
         ClassName->getName());
    if (CDecl)
      Diag(CDecl->getLocation(), diag::warn_previous_declaration);
    return 0;
  }
  // Everything checked out, instantiate a new alias declaration ast
  ObjcCompatibleAliasDecl *AliasDecl = 
    new ObjcCompatibleAliasDecl(AtCompatibilityAliasLoc, 
                                AliasName,
                                dyn_cast<ObjcInterfaceDecl>(CDecl));
    
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
  ObjcProtocolDecl *PDecl = ObjcProtocols[ProtocolName];
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
    PDecl = new ObjcProtocolDecl(AtProtoInterfaceLoc, NumProtoRefs, 
                                 ProtocolName);
    ObjcProtocols[ProtocolName] = PDecl;
  }    
  
  if (NumProtoRefs) {
    /// Check then save referenced protocols
    for (unsigned int i = 0; i != NumProtoRefs; i++) {
      ObjcProtocolDecl* RefPDecl = ObjcProtocols[ProtoRefNames[i]];
      if (!RefPDecl || RefPDecl->isForwardDecl())
        Diag(ProtocolLoc, diag::warn_undef_protocolref,
             ProtoRefNames[i]->getName(),
             ProtocolName->getName());
      PDecl->setReferencedProtocols((int)i, RefPDecl);
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
    ObjcProtocolDecl *PDecl = ObjcProtocols[ProtocolId[i]];
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
  llvm::SmallVector<ObjcProtocolDecl*, 32> Protocols;
  
  for (unsigned i = 0; i != NumElts; ++i) {
    IdentifierInfo *P = IdentList[i];
    ObjcProtocolDecl *PDecl = ObjcProtocols[P];
    if (!PDecl)  { // Not already seen?
      // FIXME: Pass in the location of the identifier!
      PDecl = new ObjcProtocolDecl(AtProtocolLoc, 0, P, true);
      ObjcProtocols[P] = PDecl;
    }
    
    Protocols.push_back(PDecl);
  }
  return new ObjcForwardProtocolDecl(AtProtocolLoc,
                                     &Protocols[0], Protocols.size());
}

Sema::DeclTy *Sema::ActOnStartCategoryInterface(
                      SourceLocation AtInterfaceLoc,
                      IdentifierInfo *ClassName, SourceLocation ClassLoc,
                      IdentifierInfo *CategoryName, SourceLocation CategoryLoc,
                      IdentifierInfo **ProtoRefNames, unsigned NumProtoRefs,
                      SourceLocation EndProtoLoc) {
  ObjcInterfaceDecl *IDecl = getObjCInterfaceDecl(ClassName);
  
  /// Check that class of this category is already completely declared.
  if (!IDecl || IDecl->isForwardDecl()) {
    Diag(ClassLoc, diag::err_undef_interface, ClassName->getName());
    return 0;
  }
  ObjcCategoryDecl *CDecl = new ObjcCategoryDecl(AtInterfaceLoc, NumProtoRefs,
                                                 CategoryName);
  CDecl->setClassInterface(IDecl);
  /// Check for duplicate interface declaration for this category
  ObjcCategoryDecl *CDeclChain;
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
      ObjcProtocolDecl* RefPDecl = ObjcProtocols[ProtoRefNames[i]];
      if (!RefPDecl || RefPDecl->isForwardDecl()) {
        Diag(CategoryLoc, diag::warn_undef_protocolref,
             ProtoRefNames[i]->getName(),
             CategoryName->getName());
      }
      CDecl->setCatReferencedProtocols((int)i, RefPDecl);
    }
    CDecl->setLocEnd(EndProtoLoc);
  }
  return CDecl;
}

/// ActOnStartCategoryImplementation - Perform semantic checks on the
/// category implementation declaration and build an ObjcCategoryImplDecl
/// object.
Sema::DeclTy *Sema::ActOnStartCategoryImplementation(
                      SourceLocation AtCatImplLoc,
                      IdentifierInfo *ClassName, SourceLocation ClassLoc,
                      IdentifierInfo *CatName, SourceLocation CatLoc) {
  ObjcInterfaceDecl *IDecl = getObjCInterfaceDecl(ClassName);
  ObjcCategoryImplDecl *CDecl = new ObjcCategoryImplDecl(AtCatImplLoc, 
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
  ObjcInterfaceDecl* IDecl = 0;
  // Check for another declaration kind with the same name.
  ScopedDecl *PrevDecl = LookupInterfaceDecl(ClassName);
  if (PrevDecl && !isa<ObjcInterfaceDecl>(PrevDecl)) {
    Diag(ClassLoc, diag::err_redefinition_different_kind,
         ClassName->getName());
    Diag(PrevDecl->getLocation(), diag::err_previous_definition);
  }
  else {
    // Is there an interface declaration of this class; if not, warn!
    IDecl = dyn_cast_or_null<ObjcInterfaceDecl>(PrevDecl); 
    if (!IDecl)
      Diag(ClassLoc, diag::warn_undef_interface, ClassName->getName());
  }
  
  // Check that super class name is valid class name
  ObjcInterfaceDecl* SDecl = 0;
  if (SuperClassname) {
    // Check if a different kind of symbol declared in this scope.
    PrevDecl = LookupInterfaceDecl(SuperClassname);
    if (PrevDecl && !isa<ObjcInterfaceDecl>(PrevDecl)) {
      Diag(SuperClassLoc, diag::err_redefinition_different_kind,
           SuperClassname->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
    }
    else {
      SDecl = dyn_cast_or_null<ObjcInterfaceDecl>(PrevDecl); 
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
    IDecl = new ObjcInterfaceDecl(AtClassImplLoc, 0, ClassName, 
				  false, true);
    IDecl->setNext(ClassName->getFETokenInfo<ScopedDecl>());
    ClassName->setFETokenInfo(IDecl);
    IDecl->setSuperClass(SDecl);
    IDecl->setLocEnd(ClassLoc);
    
    // Remember that this needs to be removed when the scope is popped.
    TUScope->AddDecl(IDecl);
  }
  
  ObjcImplementationDecl* IMPDecl = 
  new ObjcImplementationDecl(AtClassImplLoc, ClassName, IDecl, SDecl);
  
  // Check that there is no duplicate implementation of this class.
  if (ObjcImplementations[ClassName])
    Diag(ClassLoc, diag::err_dup_implementation_class, ClassName->getName());
  else // add it to the list.
    ObjcImplementations[ClassName] = IMPDecl;
  return IMPDecl;
}

void Sema::CheckImplementationIvars(ObjcImplementationDecl *ImpDecl,
                                    ObjcIvarDecl **ivars, unsigned numIvars,
                                    SourceLocation RBrace) {
  assert(ImpDecl && "missing implementation decl");
  ObjcInterfaceDecl* IDecl = getObjCInterfaceDecl(ImpDecl->getIdentifier());
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
  ObjcInterfaceDecl::ivar_iterator 
    IVI = IDecl->ivar_begin(), IVE = IDecl->ivar_end();
  for (; numIvars > 0 && IVI != IVE; ++IVI) {
    ObjcIvarDecl* ImplIvar = ivars[j++];
    ObjcIvarDecl* ClsIvar = *IVI;
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
void Sema::CheckProtocolMethodDefs(ObjcProtocolDecl *PDecl,
                                   bool& IncompleteImpl,
             const llvm::DenseSet<Selector> &InsMap,
             const llvm::DenseSet<Selector> &ClsMap) {
  // check unimplemented instance methods.
  for (ObjcProtocolDecl::instmeth_iterator I = PDecl->instmeth_begin(), 
       E = PDecl->instmeth_end(); I != E; ++I) {
    ObjcMethodDecl *method = *I;
    if (!InsMap.count(method->getSelector()) && 
        method->getImplementationControl() != ObjcMethodDecl::Optional) {
      Diag(method->getLocation(), diag::warn_undef_method_impl,
           method->getSelector().getName());
      IncompleteImpl = true;
    }
  }
  // check unimplemented class methods
  for (ObjcProtocolDecl::classmeth_iterator I = PDecl->classmeth_begin(), 
       E = PDecl->classmeth_end(); I != E; ++I) {
    ObjcMethodDecl *method = *I;
    if (!ClsMap.count(method->getSelector()) &&
        method->getImplementationControl() != ObjcMethodDecl::Optional) {
      Diag(method->getLocation(), diag::warn_undef_method_impl,
           method->getSelector().getName());
      IncompleteImpl = true;
    }
  }
  // Check on this protocols's referenced protocols, recursively
  ObjcProtocolDecl** RefPDecl = PDecl->getReferencedProtocols();
  for (int i = 0; i < PDecl->getNumReferencedProtocols(); i++)
    CheckProtocolMethodDefs(RefPDecl[i], IncompleteImpl, InsMap, ClsMap);
}

void Sema::ImplMethodsVsClassMethods(ObjcImplementationDecl* IMPDecl, 
                                     ObjcInterfaceDecl* IDecl) {
  llvm::DenseSet<Selector> InsMap;
  // Check and see if instance methods in class interface have been
  // implemented in the implementation class.
  for (ObjcImplementationDecl::instmeth_iterator I = IMPDecl->instmeth_begin(),
       E = IMPDecl->instmeth_end(); I != E; ++I)
    InsMap.insert((*I)->getSelector());
  
  bool IncompleteImpl = false;
  for (ObjcInterfaceDecl::instmeth_iterator I = IDecl->instmeth_begin(),
       E = IDecl->instmeth_end(); I != E; ++I)
    if (!InsMap.count((*I)->getSelector())) {
      Diag((*I)->getLocation(), diag::warn_undef_method_impl,
           (*I)->getSelector().getName());
      IncompleteImpl = true;
    }
      
  llvm::DenseSet<Selector> ClsMap;
  // Check and see if class methods in class interface have been
  // implemented in the implementation class.
  for (ObjcImplementationDecl::classmeth_iterator I =IMPDecl->classmeth_begin(),
       E = IMPDecl->classmeth_end(); I != E; ++I)
    ClsMap.insert((*I)->getSelector());
  
  for (ObjcInterfaceDecl::classmeth_iterator I = IDecl->classmeth_begin(),
       E = IDecl->classmeth_end(); I != E; ++I)
    if (!ClsMap.count((*I)->getSelector())) {
      Diag((*I)->getLocation(), diag::warn_undef_method_impl,
           (*I)->getSelector().getName());
      IncompleteImpl = true;
    }
  
  // Check the protocol list for unimplemented methods in the @implementation
  // class.
  ObjcProtocolDecl** protocols = IDecl->getReferencedProtocols();
  for (int i = 0; i < IDecl->getNumIntfRefProtocols(); i++)
    CheckProtocolMethodDefs(protocols[i], IncompleteImpl, InsMap, ClsMap);

  if (IncompleteImpl)
    Diag(IMPDecl->getLocation(), diag::warn_incomplete_impl_class, 
         IMPDecl->getName());
}

/// ImplCategoryMethodsVsIntfMethods - Checks that methods declared in the
/// category interface is implemented in the category @implementation.
void Sema::ImplCategoryMethodsVsIntfMethods(ObjcCategoryImplDecl *CatImplDecl,
                                            ObjcCategoryDecl *CatClassDecl) {
  llvm::DenseSet<Selector> InsMap;
  // Check and see if instance methods in category interface have been
  // implemented in its implementation class.
  for (ObjcCategoryImplDecl::instmeth_iterator I =CatImplDecl->instmeth_begin(),
       E = CatImplDecl->instmeth_end(); I != E; ++I)
    InsMap.insert((*I)->getSelector());
  
  bool IncompleteImpl = false;
  for (ObjcCategoryDecl::instmeth_iterator I = CatClassDecl->instmeth_begin(),
       E = CatClassDecl->instmeth_end(); I != E; ++I)
    if (!InsMap.count((*I)->getSelector())) {
      Diag((*I)->getLocation(), diag::warn_undef_method_impl,
           (*I)->getSelector().getName());
      IncompleteImpl = true;
    }
  llvm::DenseSet<Selector> ClsMap;
  // Check and see if class methods in category interface have been
  // implemented in its implementation class.
  for (ObjcCategoryImplDecl::classmeth_iterator
       I = CatImplDecl->classmeth_begin(), E = CatImplDecl->classmeth_end();
       I != E; ++I)
    ClsMap.insert((*I)->getSelector());
  
  for (ObjcCategoryDecl::classmeth_iterator I = CatClassDecl->classmeth_begin(),
       E = CatClassDecl->classmeth_end(); I != E; ++I)
    if (!ClsMap.count((*I)->getSelector())) {
      Diag((*I)->getLocation(), diag::warn_undef_method_impl,
           (*I)->getSelector().getName());
      IncompleteImpl = true;
    }
  
  // Check the protocol list for unimplemented methods in the @implementation
  // class.
  ObjcProtocolDecl** protocols = CatClassDecl->getReferencedProtocols();
  for (int i = 0; i < CatClassDecl->getNumReferencedProtocols(); i++) {
    ObjcProtocolDecl* PDecl = protocols[i];
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
  llvm::SmallVector<ObjcInterfaceDecl*, 32> Interfaces;
  
  for (unsigned i = 0; i != NumElts; ++i) {
    // Check for another declaration kind with the same name.
    ScopedDecl *PrevDecl = LookupInterfaceDecl(IdentList[i]);
    if (PrevDecl && !isa<ObjcInterfaceDecl>(PrevDecl)) {
      Diag(AtClassLoc, diag::err_redefinition_different_kind,
           IdentList[i]->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
    }
    ObjcInterfaceDecl *IDecl = dyn_cast_or_null<ObjcInterfaceDecl>(PrevDecl); 
    if (!IDecl) {  // Not already seen?  Make a forward decl.
      IDecl = new ObjcInterfaceDecl(AtClassLoc, 0, IdentList[i], true);
      // Chain & install the interface decl into the identifier.
      IDecl->setNext(IdentList[i]->getFETokenInfo<ScopedDecl>());
      IdentList[i]->setFETokenInfo(IDecl);

      // Remember that this needs to be removed when the scope is popped.
      TUScope->AddDecl(IDecl);
    }

    Interfaces.push_back(IDecl);
  }
  
  return new ObjcClassDecl(AtClassLoc, &Interfaces[0], Interfaces.size());
}


/// MatchTwoMethodDeclarations - Checks that two methods have matching type and
/// returns true, or false, accordingly.
/// TODO: Handle protocol list; such as id<p1,p2> in type comparisons
bool Sema::MatchTwoMethodDeclarations(const ObjcMethodDecl *Method, 
                                      const ObjcMethodDecl *PrevMethod) {
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

void Sema::AddInstanceMethodToGlobalPool(ObjcMethodDecl *Method) {
  ObjcMethodList &FirstMethod = InstanceMethodPool[Method->getSelector()];
  if (!FirstMethod.Method) {
    // Haven't seen a method with this selector name yet - add it.
    FirstMethod.Method = Method;
    FirstMethod.Next = 0;
  } else {
    // We've seen a method with this name, now check the type signature(s).
    bool match = MatchTwoMethodDeclarations(Method, FirstMethod.Method);
    
    for (ObjcMethodList *Next = FirstMethod.Next; !match && Next; 
         Next = Next->Next)
      match = MatchTwoMethodDeclarations(Method, Next->Method);
      
    if (!match) {
      // We have a new signature for an existing method - add it.
      // This is extremely rare. Only 1% of Cocoa selectors are "overloaded".
      struct ObjcMethodList *OMI = new ObjcMethodList(Method, FirstMethod.Next);
      FirstMethod.Next = OMI;
    }
  }
}

void Sema::AddFactoryMethodToGlobalPool(ObjcMethodDecl *Method) {
  ObjcMethodList &FirstMethod = FactoryMethodPool[Method->getSelector()];
  if (!FirstMethod.Method) {
    // Haven't seen a method with this selector name yet - add it.
    FirstMethod.Method = Method;
    FirstMethod.Next = 0;
  } else {
    // We've seen a method with this name, now check the type signature(s).
    bool match = MatchTwoMethodDeclarations(Method, FirstMethod.Method);
    
    for (ObjcMethodList *Next = FirstMethod.Next; !match && Next; 
         Next = Next->Next)
      match = MatchTwoMethodDeclarations(Method, Next->Method);
      
    if (!match) {
      // We have a new signature for an existing method - add it.
      // This is extremely rare. Only 1% of Cocoa selectors are "overloaded".
      struct ObjcMethodList *OMI = new ObjcMethodList(Method, FirstMethod.Next);
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
    
  llvm::SmallVector<ObjcMethodDecl*, 32> insMethods;
  llvm::SmallVector<ObjcMethodDecl*, 16> clsMethods;
  
  llvm::DenseMap<Selector, const ObjcMethodDecl*> InsMap;
  llvm::DenseMap<Selector, const ObjcMethodDecl*> ClsMap;
  
  bool isInterfaceDeclKind = 
        (isa<ObjcInterfaceDecl>(ClassDecl) || isa<ObjcCategoryDecl>(ClassDecl)
         || isa<ObjcProtocolDecl>(ClassDecl));
  bool checkIdenticalMethods = isa<ObjcImplementationDecl>(ClassDecl);
  
  // TODO: property declaration in category and protocols.
  if (pNum != 0 && isa<ObjcInterfaceDecl>(ClassDecl)) {
    ObjcPropertyDecl **properties = new ObjcPropertyDecl*[pNum];
    memcpy(properties, allProperties, pNum*sizeof(ObjcPropertyDecl*));
    dyn_cast<ObjcInterfaceDecl>(ClassDecl)->setPropertyDecls(properties);
    dyn_cast<ObjcInterfaceDecl>(ClassDecl)->setNumPropertyDecl(pNum);
  }
  
  for (unsigned i = 0; i < allNum; i++ ) {
    ObjcMethodDecl *Method =
      cast_or_null<ObjcMethodDecl>(static_cast<Decl*>(allMethods[i]));

    if (!Method) continue;  // Already issued a diagnostic.
    if (Method->isInstance()) {
      /// Check for instance method of the same name with incompatible types
      const ObjcMethodDecl *&PrevMethod = InsMap[Method->getSelector()];
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
      const ObjcMethodDecl *&PrevMethod = ClsMap[Method->getSelector()];
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
  
  if (ObjcInterfaceDecl *I = dyn_cast<ObjcInterfaceDecl>(ClassDecl)) {
    I->addMethods(&insMethods[0], insMethods.size(),
                  &clsMethods[0], clsMethods.size(), AtEndLoc);
  } else if (ObjcProtocolDecl *P = dyn_cast<ObjcProtocolDecl>(ClassDecl)) {
    P->addMethods(&insMethods[0], insMethods.size(),
                  &clsMethods[0], clsMethods.size(), AtEndLoc);
  }
  else if (ObjcCategoryDecl *C = dyn_cast<ObjcCategoryDecl>(ClassDecl)) {
    C->addMethods(&insMethods[0], insMethods.size(),
                  &clsMethods[0], clsMethods.size(), AtEndLoc);
  }
  else if (ObjcImplementationDecl *IC = 
                dyn_cast<ObjcImplementationDecl>(ClassDecl)) {
    IC->setLocEnd(AtEndLoc);
    if (ObjcInterfaceDecl* IDecl = getObjCInterfaceDecl(IC->getIdentifier()))
      ImplMethodsVsClassMethods(IC, IDecl);
  } else {
    ObjcCategoryImplDecl* CatImplClass = cast<ObjcCategoryImplDecl>(ClassDecl);
    CatImplClass->setLocEnd(AtEndLoc);
    ObjcInterfaceDecl* IDecl = CatImplClass->getClassInterface();
    // Find category interface decl and then check that all methods declared
    // in this interface is implemented in the category @implementation.
    if (IDecl) {
      for (ObjcCategoryDecl *Categories = IDecl->getCategoryList();
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
static Decl::ObjcDeclQualifier 
CvtQTToAstBitMask(ObjcDeclSpec::ObjcDeclQualifier PQTVal) {
  Decl::ObjcDeclQualifier ret = Decl::OBJC_TQ_None;
  if (PQTVal & ObjcDeclSpec::DQ_In)
    ret = (Decl::ObjcDeclQualifier)(ret | Decl::OBJC_TQ_In);
  if (PQTVal & ObjcDeclSpec::DQ_Inout)
    ret = (Decl::ObjcDeclQualifier)(ret | Decl::OBJC_TQ_Inout);
  if (PQTVal & ObjcDeclSpec::DQ_Out)
    ret = (Decl::ObjcDeclQualifier)(ret | Decl::OBJC_TQ_Out);
  if (PQTVal & ObjcDeclSpec::DQ_Bycopy)
    ret = (Decl::ObjcDeclQualifier)(ret | Decl::OBJC_TQ_Bycopy);
  if (PQTVal & ObjcDeclSpec::DQ_Byref)
    ret = (Decl::ObjcDeclQualifier)(ret | Decl::OBJC_TQ_Byref);
  if (PQTVal & ObjcDeclSpec::DQ_Oneway)
    ret = (Decl::ObjcDeclQualifier)(ret | Decl::OBJC_TQ_Oneway);

  return ret;
}

Sema::DeclTy *Sema::ActOnMethodDeclaration(
    SourceLocation MethodLoc, SourceLocation EndLoc,
    tok::TokenKind MethodType, DeclTy *ClassDecl,
    ObjcDeclSpec &ReturnQT, TypeTy *ReturnType,
    Selector Sel,
    // optional arguments. The number of types/arguments is obtained
    // from the Sel.getNumArgs().
    ObjcDeclSpec *ArgQT, TypeTy **ArgTypes, IdentifierInfo **ArgNames,
    AttributeList *AttrList, tok::ObjCKeywordKind MethodDeclKind,
    bool isVariadic) {
  llvm::SmallVector<ParmVarDecl*, 16> Params;
  
  for (unsigned i = 0; i < Sel.getNumArgs(); i++) {
    // FIXME: arg->AttrList must be stored too!
    QualType argType;
    
    if (ArgTypes[i])
      argType = QualType::getFromOpaquePtr(ArgTypes[i]);
    else
      argType = Context.getObjcIdType();
    ParmVarDecl* Param = new ParmVarDecl(SourceLocation(/*FIXME*/), ArgNames[i], 
                                         argType, VarDecl::None, 0);
    Param->setObjcDeclQualifier(
      CvtQTToAstBitMask(ArgQT[i].getObjcDeclQualifier()));
    Params.push_back(Param);
  }
  QualType resultDeclType;
  
  if (ReturnType)
    resultDeclType = QualType::getFromOpaquePtr(ReturnType);
  else // get the type for "id".
    resultDeclType = Context.getObjcIdType();
  
  Decl *CDecl = static_cast<Decl*>(ClassDecl);
  ObjcMethodDecl* ObjcMethod =  new ObjcMethodDecl(MethodLoc, EndLoc, Sel,
                                      resultDeclType,
                                      CDecl,
                                      0, -1, AttrList, 
                                      MethodType == tok::minus, isVariadic,
                                      MethodDeclKind == tok::objc_optional ? 
                                      ObjcMethodDecl::Optional : 
                                      ObjcMethodDecl::Required);
  ObjcMethod->setMethodParams(&Params[0], Sel.getNumArgs());
  ObjcMethod->setObjcDeclQualifier(
    CvtQTToAstBitMask(ReturnQT.getObjcDeclQualifier()));
  const ObjcMethodDecl *PrevMethod = 0;
 
  // For implementations (which can be very "coarse grain"), we add the 
  // method now. This allows the AST to implement lookup methods that work 
  // incrementally (without waiting until we parse the @end). It also allows 
  // us to flag multiple declaration errors as they occur.
  if (ObjcImplementationDecl *ImpDecl = 
        dyn_cast<ObjcImplementationDecl>(CDecl)) {
    if (MethodType == tok::minus) {
      PrevMethod = ImpDecl->getInstanceMethod(Sel);
      ImpDecl->addInstanceMethod(ObjcMethod);
    } else {
      PrevMethod = ImpDecl->getClassMethod(Sel);
      ImpDecl->addClassMethod(ObjcMethod);
    }
  } 
  else if (ObjcCategoryImplDecl *CatImpDecl = 
            dyn_cast<ObjcCategoryImplDecl>(CDecl)) {
    if (MethodType == tok::minus) {
      PrevMethod = CatImpDecl->getInstanceMethod(Sel);
      CatImpDecl->addInstanceMethod(ObjcMethod);
    } else {
      PrevMethod = CatImpDecl->getClassMethod(Sel);
      CatImpDecl->addClassMethod(ObjcMethod);
    }
  }
  if (PrevMethod) {
    // You can never have two method definitions with the same name.
    Diag(ObjcMethod->getLocation(), diag::error_duplicate_method_decl,
        ObjcMethod->getSelector().getName());
    Diag(PrevMethod->getLocation(), diag::err_previous_declaration);
  } 
  return ObjcMethod;
}

Sema::DeclTy *Sema::ActOnAddObjcProperties(SourceLocation AtLoc, 
  DeclTy **allProperties, unsigned NumProperties, ObjcDeclSpec &DS) {
  ObjcPropertyDecl *PDecl = new ObjcPropertyDecl(AtLoc);
  
  if(DS.getPropertyAttributes() & ObjcDeclSpec::DQ_PR_readonly)
    PDecl->setPropertyAttributes(ObjcPropertyDecl::OBJC_PR_readonly);
  
  if(DS.getPropertyAttributes() & ObjcDeclSpec::DQ_PR_getter) {
    PDecl->setPropertyAttributes(ObjcPropertyDecl::OBJC_PR_getter);
    PDecl->setGetterName(DS.getGetterName());
  }
  
  if(DS.getPropertyAttributes() & ObjcDeclSpec::DQ_PR_setter) {
    PDecl->setPropertyAttributes(ObjcPropertyDecl::OBJC_PR_setter);
    PDecl->setSetterName(DS.getSetterName());
  }
  
  if(DS.getPropertyAttributes() & ObjcDeclSpec::DQ_PR_assign)
    PDecl->setPropertyAttributes(ObjcPropertyDecl::OBJC_PR_assign);
  
  if(DS.getPropertyAttributes() & ObjcDeclSpec::DQ_PR_readwrite)
    PDecl->setPropertyAttributes(ObjcPropertyDecl::OBJC_PR_readwrite);
  
  if(DS.getPropertyAttributes() & ObjcDeclSpec::DQ_PR_retain)
    PDecl->setPropertyAttributes(ObjcPropertyDecl::OBJC_PR_retain);
  
  if(DS.getPropertyAttributes() & ObjcDeclSpec::DQ_PR_copy)
    PDecl->setPropertyAttributes(ObjcPropertyDecl::OBJC_PR_copy);
  
  if(DS.getPropertyAttributes() & ObjcDeclSpec::DQ_PR_nonatomic)
    PDecl->setPropertyAttributes(ObjcPropertyDecl::OBJC_PR_nonatomic);
  
  PDecl->setNumPropertyDecls(NumProperties);
  if (NumProperties != 0) {
    ObjcIvarDecl **properties = new ObjcIvarDecl*[NumProperties];
    memcpy(properties, allProperties, NumProperties*sizeof(ObjcIvarDecl*));
    PDecl->setPropertyDecls(properties);
  }
  return PDecl;
}

