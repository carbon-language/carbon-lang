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
#include "Lookup.h"
#include "clang/Sema/ExternalSemaSource.h"
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
  PushFunctionScope();
  
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
                         const SourceLocation *ProtoLocs, 
                         SourceLocation EndProtoLoc, AttributeList *AttrList) {
  assert(ClassName && "Missing class identifier");

  // Check for another declaration kind with the same name.
  NamedDecl *PrevDecl = LookupSingleName(TUScope, ClassName, ClassLoc,
                                         LookupOrdinaryName, ForRedeclaration);

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
      IDecl->setClassLoc(ClassLoc);
      
      // Since this ObjCInterfaceDecl was created by a forward declaration,
      // we now add it to the DeclContext since it wasn't added before
      // (see ActOnForwardClassDeclaration).
      IDecl->setLexicalDeclContext(CurContext);
      CurContext->addDecl(IDecl);
      
      if (AttrList)
        ProcessDeclAttributeList(TUScope, IDecl, AttrList);
    }
  } else {
    IDecl = ObjCInterfaceDecl::Create(Context, CurContext, AtInterfaceLoc,
                                      ClassName, ClassLoc);
    if (AttrList)
      ProcessDeclAttributeList(TUScope, IDecl, AttrList);

    PushOnScopeChains(IDecl, TUScope);
  }

  if (SuperName) {
    // Check if a different kind of symbol declared in this scope.
    PrevDecl = LookupSingleName(TUScope, SuperName, SuperLoc,
                                LookupOrdinaryName);

    if (!PrevDecl) {
      // Try to correct for a typo in the superclass name.
      LookupResult R(*this, SuperName, SuperLoc, LookupOrdinaryName);
      if (CorrectTypo(R, TUScope, 0, 0, false, CTC_NoKeywords) &&
          (PrevDecl = R.getAsSingle<ObjCInterfaceDecl>())) {
        Diag(SuperLoc, diag::err_undef_superclass_suggest)
          << SuperName << ClassName << PrevDecl->getDeclName();
        Diag(PrevDecl->getLocation(), diag::note_previous_decl)
          << PrevDecl->getDeclName();
      }
    }

    if (PrevDecl == IDecl) {
      Diag(SuperLoc, diag::err_recursive_superclass)
        << SuperName << ClassName << SourceRange(AtInterfaceLoc, ClassLoc);
      IDecl->setLocEnd(ClassLoc);
    } else {
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
          if (T->isObjCObjectType()) {
            if (NamedDecl *IDecl = T->getAs<ObjCObjectType>()->getInterface())
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
    }
  } else { // we have a root class.
    IDecl->setLocEnd(ClassLoc);
  }

  /// Check then save referenced protocols.
  if (NumProtoRefs) {
    IDecl->setProtocolList((ObjCProtocolDecl**)ProtoRefs, NumProtoRefs,
                           ProtoLocs, Context);
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
  NamedDecl *ADecl = LookupSingleName(TUScope, AliasName, AliasLocation,
                                      LookupOrdinaryName, ForRedeclaration);
  if (ADecl) {
    if (isa<ObjCCompatibleAliasDecl>(ADecl))
      Diag(AliasLocation, diag::warn_previous_alias_decl);
    else
      Diag(AliasLocation, diag::err_conflicting_aliasing_type) << AliasName;
    Diag(ADecl->getLocation(), diag::note_previous_declaration);
    return DeclPtrTy();
  }
  // Check for class declaration
  NamedDecl *CDeclU = LookupSingleName(TUScope, ClassName, ClassLocation,
                                       LookupOrdinaryName, ForRedeclaration);
  if (const TypedefDecl *TDecl = dyn_cast_or_null<TypedefDecl>(CDeclU)) {
    QualType T = TDecl->getUnderlyingType();
    if (T->isObjCObjectType()) {
      if (NamedDecl *IDecl = T->getAs<ObjCObjectType>()->getInterface()) {
        ClassName = IDecl->getIdentifier();
        CDeclU = LookupSingleName(TUScope, ClassName, ClassLocation,
                                  LookupOrdinaryName, ForRedeclaration);
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

  if (!CheckObjCDeclScope(AliasDecl))
    PushOnScopeChains(AliasDecl, TUScope);

  return DeclPtrTy::make(AliasDecl);
}

void Sema::CheckForwardProtocolDeclarationForCircularDependency(
  IdentifierInfo *PName,
  SourceLocation &Ploc, SourceLocation PrevLoc,
  const ObjCList<ObjCProtocolDecl> &PList) {
  for (ObjCList<ObjCProtocolDecl>::iterator I = PList.begin(),
       E = PList.end(); I != E; ++I) {

    if (ObjCProtocolDecl *PDecl = LookupProtocol((*I)->getIdentifier(),
                                                 Ploc)) {
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
                                  const SourceLocation *ProtoLocs,
                                  SourceLocation EndProtoLoc,
                                  AttributeList *AttrList) {
  // FIXME: Deal with AttrList.
  assert(ProtocolName && "Missing protocol identifier");
  ObjCProtocolDecl *PDecl = LookupProtocol(ProtocolName, ProtocolLoc);
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
    PushOnScopeChains(PDecl, TUScope);
    PDecl->setForwardDecl(false);
  }
  if (AttrList)
    ProcessDeclAttributeList(TUScope, PDecl, AttrList);
  if (NumProtoRefs) {
    /// Check then save referenced protocols.
    PDecl->setProtocolList((ObjCProtocolDecl**)ProtoRefs, NumProtoRefs,
                           ProtoLocs, Context);
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
    ObjCProtocolDecl *PDecl = LookupProtocol(ProtocolId[i].first,
                                             ProtocolId[i].second);
    if (!PDecl) {
      LookupResult R(*this, ProtocolId[i].first, ProtocolId[i].second,
                     LookupObjCProtocolName);
      if (CorrectTypo(R, TUScope, 0, 0, false, CTC_NoKeywords) &&
          (PDecl = R.getAsSingle<ObjCProtocolDecl>())) {
        Diag(ProtocolId[i].second, diag::err_undeclared_protocol_suggest)
          << ProtocolId[i].first << R.getLookupName();
        Diag(PDecl->getLocation(), diag::note_previous_decl)
          << PDecl->getDeclName();
      }
    }

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

/// DiagnoseClassExtensionDupMethods - Check for duplicate declaration of
/// a class method in its extension.
///
void Sema::DiagnoseClassExtensionDupMethods(ObjCCategoryDecl *CAT,
                                            ObjCInterfaceDecl *ID) {
  if (!ID)
    return;  // Possibly due to previous error

  llvm::DenseMap<Selector, const ObjCMethodDecl*> MethodMap;
  for (ObjCInterfaceDecl::method_iterator i = ID->meth_begin(),
       e =  ID->meth_end(); i != e; ++i) {
    ObjCMethodDecl *MD = *i;
    MethodMap[MD->getSelector()] = MD;
  }

  if (MethodMap.empty())
    return;
  for (ObjCCategoryDecl::method_iterator i = CAT->meth_begin(),
       e =  CAT->meth_end(); i != e; ++i) {
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
  llvm::SmallVector<SourceLocation, 8> ProtoLocs;

  for (unsigned i = 0; i != NumElts; ++i) {
    IdentifierInfo *Ident = IdentList[i].first;
    ObjCProtocolDecl *PDecl = LookupProtocol(Ident, IdentList[i].second);
    if (PDecl == 0) { // Not already seen?
      PDecl = ObjCProtocolDecl::Create(Context, CurContext,
                                       IdentList[i].second, Ident);
      PushOnScopeChains(PDecl, TUScope);
    }
    if (attrList)
      ProcessDeclAttributeList(TUScope, PDecl, attrList);
    Protocols.push_back(PDecl);
    ProtoLocs.push_back(IdentList[i].second);
  }

  ObjCForwardProtocolDecl *PDecl =
    ObjCForwardProtocolDecl::Create(Context, CurContext, AtProtocolLoc,
                                    Protocols.data(), Protocols.size(),
                                    ProtoLocs.data());
  CurContext->addDecl(PDecl);
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
                            const SourceLocation *ProtoLocs,
                            SourceLocation EndProtoLoc) {
  ObjCCategoryDecl *CDecl = 0;
  ObjCInterfaceDecl *IDecl = getObjCInterfaceDecl(ClassName, ClassLoc, true);

  /// Check that class of this category is already completely declared.
  if (!IDecl || IDecl->isForwardDecl()) {
    // Create an invalid ObjCCategoryDecl to serve as context for
    // the enclosing method declarations.  We mark the decl invalid
    // to make it clear that this isn't a valid AST.
    CDecl = ObjCCategoryDecl::Create(Context, CurContext, AtInterfaceLoc,
                                     ClassLoc, CategoryLoc, CategoryName);
    CDecl->setInvalidDecl();
    Diag(ClassLoc, diag::err_undef_interface) << ClassName;
    return DeclPtrTy::make(CDecl);
  }

  if (!CategoryName) {
    // Class extensions require a special treatment. Use an existing one.
    // Note that 'getClassExtension()' can return NULL.
    CDecl = IDecl->getClassExtension();
    if (IDecl->getImplementation()) {
      Diag(ClassLoc, diag::err_class_extension_after_impl) << ClassName;
      Diag(IDecl->getImplementation()->getLocation(), 
           diag::note_implementation_declared);
    }
  }

  if (!CDecl) {
    CDecl = ObjCCategoryDecl::Create(Context, CurContext, AtInterfaceLoc,
                                     ClassLoc, CategoryLoc, CategoryName);
    // FIXME: PushOnScopeChains?
    CurContext->addDecl(CDecl);

    CDecl->setClassInterface(IDecl);
    // Insert first use of class extension to the list of class's categories.
    if (!CategoryName)
      CDecl->insertNextClassCategory();
  }

  // If the interface is deprecated, warn about it.
  (void)DiagnoseUseOfDecl(IDecl, ClassLoc);

  if (CategoryName) {
    /// Check for duplicate interface declaration for this category
    ObjCCategoryDecl *CDeclChain;
    for (CDeclChain = IDecl->getCategoryList(); CDeclChain;
         CDeclChain = CDeclChain->getNextClassCategory()) {
      if (CDeclChain->getIdentifier() == CategoryName) {
        // Class extensions can be declared multiple times.
        Diag(CategoryLoc, diag::warn_dup_category_def)
          << ClassName << CategoryName;
        Diag(CDeclChain->getLocation(), diag::note_previous_definition);
        break;
      }
    }
    if (!CDeclChain)
      CDecl->insertNextClassCategory();
  }

  if (NumProtoRefs) {
    CDecl->setProtocolList((ObjCProtocolDecl**)ProtoRefs, NumProtoRefs, 
                           ProtoLocs, Context);
    // Protocols in the class extension belong to the class.
    if (CDecl->IsClassExtension())
     IDecl->mergeClassExtensionProtocolList((ObjCProtocolDecl**)ProtoRefs, 
                                            NumProtoRefs, ProtoLocs,
                                            Context); 
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
  ObjCInterfaceDecl *IDecl = getObjCInterfaceDecl(ClassName, ClassLoc, true);
  ObjCCategoryDecl *CatIDecl = 0;
  if (IDecl) {
    CatIDecl = IDecl->FindCategoryDeclaration(CatName);
    if (!CatIDecl) {
      // Category @implementation with no corresponding @interface.
      // Create and install one.
      CatIDecl = ObjCCategoryDecl::Create(Context, CurContext, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          CatName);
      CatIDecl->setClassInterface(IDecl);
      CatIDecl->insertNextClassCategory();
    }
  }

  ObjCCategoryImplDecl *CDecl =
    ObjCCategoryImplDecl::Create(Context, CurContext, AtCatImplLoc, CatName,
                                 IDecl);
  /// Check that class of this category is already completely declared.
  if (!IDecl || IDecl->isForwardDecl())
    Diag(ClassLoc, diag::err_undef_interface) << ClassName;

  // FIXME: PushOnScopeChains?
  CurContext->addDecl(CDecl);

  /// Check that CatName, category name, is not used in another implementation.
  if (CatIDecl) {
    if (CatIDecl->getImplementation()) {
      Diag(ClassLoc, diag::err_dup_implementation_category) << ClassName
        << CatName;
      Diag(CatIDecl->getImplementation()->getLocation(),
           diag::note_previous_definition);
    } else
      CatIDecl->setImplementation(CDecl);
  }

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
  NamedDecl *PrevDecl
    = LookupSingleName(TUScope, ClassName, ClassLoc, LookupOrdinaryName,
                       ForRedeclaration);
  if (PrevDecl && !isa<ObjCInterfaceDecl>(PrevDecl)) {
    Diag(ClassLoc, diag::err_redefinition_different_kind) << ClassName;
    Diag(PrevDecl->getLocation(), diag::note_previous_definition);
  } else if ((IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl))) {
    // If this is a forward declaration of an interface, warn.
    if (IDecl->isForwardDecl()) {
      Diag(ClassLoc, diag::warn_undef_interface) << ClassName;
      IDecl = 0;
    }
  } else {
    // We did not find anything with the name ClassName; try to correct for 
    // typos in the class name.
    LookupResult R(*this, ClassName, ClassLoc, LookupOrdinaryName);
    if (CorrectTypo(R, TUScope, 0, 0, false, CTC_NoKeywords) &&
        (IDecl = R.getAsSingle<ObjCInterfaceDecl>())) {
      // Suggest the (potentially) correct interface name. However, put the
      // fix-it hint itself in a separate note, since changing the name in 
      // the warning would make the fix-it change semantics.However, don't
      // provide a code-modification hint or use the typo name for recovery,
      // because this is just a warning. The program may actually be correct.
      Diag(ClassLoc, diag::warn_undef_interface_suggest)
        << ClassName << R.getLookupName();
      Diag(IDecl->getLocation(), diag::note_previous_decl)
        << R.getLookupName()
        << FixItHint::CreateReplacement(ClassLoc,
                                        R.getLookupName().getAsString());
      IDecl = 0;
    } else {
      Diag(ClassLoc, diag::warn_undef_interface) << ClassName;
    }
  }

  // Check that super class name is valid class name
  ObjCInterfaceDecl* SDecl = 0;
  if (SuperClassname) {
    // Check if a different kind of symbol declared in this scope.
    PrevDecl = LookupSingleName(TUScope, SuperClassname, SuperClassLoc,
                                LookupOrdinaryName);
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

    // FIXME: Do we support attributes on the @implementation? If so we should
    // copy them over.
    IDecl = ObjCInterfaceDecl::Create(Context, CurContext, AtClassImplLoc,
                                      ClassName, ClassLoc, false, true);
    IDecl->setSuperClass(SDecl);
    IDecl->setLocEnd(ClassLoc);

    PushOnScopeChains(IDecl, TUScope);
  } else {
    // Mark the interface as being completed, even if it was just as
    //   @class ....;
    // declaration; the user cannot reopen it.
    IDecl->setForwardDecl(false);
  }

  ObjCImplementationDecl* IMPDecl =
    ObjCImplementationDecl::Create(Context, CurContext, AtClassImplLoc,
                                   IDecl, SDecl);

  if (CheckObjCDeclScope(IMPDecl))
    return DeclPtrTy::make(IMPDecl);

  // Check that there is no duplicate implementation of this class.
  if (IDecl->getImplementation()) {
    // FIXME: Don't leak everything!
    Diag(ClassLoc, diag::err_dup_implementation_class) << ClassName;
    Diag(IDecl->getImplementation()->getLocation(),
         diag::note_previous_definition);
  } else { // add it to the list.
    IDecl->setImplementation(IMPDecl);
    PushOnScopeChains(IMPDecl, TUScope);
  }
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
  if (IDecl->isImplicitInterfaceDecl()) {
    IDecl->setLocEnd(RBrace);
    // Add ivar's to class's DeclContext.
    for (unsigned i = 0, e = numIvars; i != e; ++i) {
      ivars[i]->setLexicalDeclContext(ImpDecl);
      IDecl->makeDeclVisibleInContext(ivars[i], false);
      ImpDecl->addDecl(ivars[i]);
    }
    
    return;
  }
  // If implementation has empty ivar list, just return.
  if (numIvars == 0)
    return;

  assert(ivars && "missing @implementation ivars");
  if (LangOpts.ObjCNonFragileABI2) {
    if (ImpDecl->getSuperClass())
      Diag(ImpDecl->getLocation(), diag::warn_on_superclass_use);
    for (unsigned i = 0; i < numIvars; i++) {
      ObjCIvarDecl* ImplIvar = ivars[i];
      if (const ObjCIvarDecl *ClsIvar = 
            IDecl->getIvarDecl(ImplIvar->getIdentifier())) {
        Diag(ImplIvar->getLocation(), diag::err_duplicate_ivar_declaration); 
        Diag(ClsIvar->getLocation(), diag::note_previous_definition);
        continue;
      }
      // Instance ivar to Implementation's DeclContext.
      ImplIvar->setLexicalDeclContext(ImpDecl);
      IDecl->makeDeclVisibleInContext(ImplIvar, false);
      ImpDecl->addDecl(ImplIvar);
    }
    return;
  }
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
      if (ImplBitWidth->EvaluateAsInt(Context).getZExtValue() !=
          ClsBitWidth->EvaluateAsInt(Context).getZExtValue()) {
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
                               bool &IncompleteImpl, unsigned DiagID) {
  if (!IncompleteImpl) {
    Diag(ImpLoc, diag::warn_incomplete_impl);
    IncompleteImpl = true;
  }
  Diag(method->getLocation(), DiagID) 
    << method->getDeclName();
}

void Sema::WarnConflictingTypedMethods(ObjCMethodDecl *ImpMethodDecl,
                                       ObjCMethodDecl *IntfMethodDecl) {
  if (!Context.typesAreCompatible(IntfMethodDecl->getResultType(),
                                  ImpMethodDecl->getResultType()) &&
      !Context.QualifiedIdConformsQualifiedId(IntfMethodDecl->getResultType(),
                                              ImpMethodDecl->getResultType())) {
    Diag(ImpMethodDecl->getLocation(), diag::warn_conflicting_ret_types)
      << ImpMethodDecl->getDeclName() << IntfMethodDecl->getResultType()
      << ImpMethodDecl->getResultType();
    Diag(IntfMethodDecl->getLocation(), diag::note_previous_definition);
  }

  for (ObjCMethodDecl::param_iterator IM = ImpMethodDecl->param_begin(),
       IF = IntfMethodDecl->param_begin(), EM = ImpMethodDecl->param_end();
       IM != EM; ++IM, ++IF) {
    QualType ParmDeclTy = (*IF)->getType().getUnqualifiedType();
    QualType ParmImpTy = (*IM)->getType().getUnqualifiedType();
    if (Context.typesAreCompatible(ParmDeclTy, ParmImpTy) ||
        Context.QualifiedIdConformsQualifiedId(ParmDeclTy, ParmImpTy))
      continue;

    Diag((*IM)->getLocation(), diag::warn_conflicting_param_types)
      << ImpMethodDecl->getDeclName() << (*IF)->getType()
      << (*IM)->getType();
    Diag((*IF)->getLocation(), diag::note_previous_definition);
  }
  if (ImpMethodDecl->isVariadic() != IntfMethodDecl->isVariadic()) {
    Diag(ImpMethodDecl->getLocation(), diag::warn_conflicting_variadic);
    Diag(IntfMethodDecl->getLocation(), diag::note_previous_declaration);
  }
}

/// FIXME: Type hierarchies in Objective-C can be deep. We could most likely
/// improve the efficiency of selector lookups and type checking by associating
/// with each protocol / interface / category the flattened instance tables. If
/// we used an immutable set to keep the table then it wouldn't add significant
/// memory cost and it would be handy for lookups.

/// CheckProtocolMethodDefs - This routine checks unimplemented methods
/// Declared in protocol, and those referenced by it.
void Sema::CheckProtocolMethodDefs(SourceLocation ImpLoc,
                                   ObjCProtocolDecl *PDecl,
                                   bool& IncompleteImpl,
                                   const llvm::DenseSet<Selector> &InsMap,
                                   const llvm::DenseSet<Selector> &ClsMap,
                                   ObjCContainerDecl *CDecl) {
  ObjCInterfaceDecl *IDecl;
  if (ObjCCategoryDecl *C = dyn_cast<ObjCCategoryDecl>(CDecl))
    IDecl = C->getClassInterface();
  else
    IDecl = dyn_cast<ObjCInterfaceDecl>(CDecl);
  assert (IDecl && "CheckProtocolMethodDefs - IDecl is null");
  
  ObjCInterfaceDecl *Super = IDecl->getSuperClass();
  ObjCInterfaceDecl *NSIDecl = 0;
  if (getLangOptions().NeXTRuntime) {
    // check to see if class implements forwardInvocation method and objects
    // of this class are derived from 'NSProxy' so that to forward requests
    // from one object to another.
    // Under such conditions, which means that every method possible is
    // implemented in the class, we should not issue "Method definition not
    // found" warnings.
    // FIXME: Use a general GetUnarySelector method for this.
    IdentifierInfo* II = &Context.Idents.get("forwardInvocation");
    Selector fISelector = Context.Selectors.getSelector(1, &II);
    if (InsMap.count(fISelector))
      // Is IDecl derived from 'NSProxy'? If so, no instance methods
      // need be implemented in the implementation.
      NSIDecl = IDecl->lookupInheritedClass(&Context.Idents.get("NSProxy"));
  }

  // If a method lookup fails locally we still need to look and see if
  // the method was implemented by a base class or an inherited
  // protocol. This lookup is slow, but occurs rarely in correct code
  // and otherwise would terminate in a warning.

  // check unimplemented instance methods.
  if (!NSIDecl)
    for (ObjCProtocolDecl::instmeth_iterator I = PDecl->instmeth_begin(),
         E = PDecl->instmeth_end(); I != E; ++I) {
      ObjCMethodDecl *method = *I;
      if (method->getImplementationControl() != ObjCMethodDecl::Optional &&
          !method->isSynthesized() && !InsMap.count(method->getSelector()) &&
          (!Super ||
           !Super->lookupInstanceMethod(method->getSelector()))) {
            // Ugly, but necessary. Method declared in protcol might have
            // have been synthesized due to a property declared in the class which
            // uses the protocol.
            ObjCMethodDecl *MethodInClass =
            IDecl->lookupInstanceMethod(method->getSelector());
            if (!MethodInClass || !MethodInClass->isSynthesized()) {
              unsigned DIAG = diag::warn_unimplemented_protocol_method;
              if (Diags.getDiagnosticLevel(DIAG) != Diagnostic::Ignored) {
                WarnUndefinedMethod(ImpLoc, method, IncompleteImpl, DIAG);
                Diag(CDecl->getLocation(), diag::note_required_for_protocol_at)
                  << PDecl->getDeclName();
              }
            }
          }
    }
  // check unimplemented class methods
  for (ObjCProtocolDecl::classmeth_iterator
         I = PDecl->classmeth_begin(), E = PDecl->classmeth_end();
       I != E; ++I) {
    ObjCMethodDecl *method = *I;
    if (method->getImplementationControl() != ObjCMethodDecl::Optional &&
        !ClsMap.count(method->getSelector()) &&
        (!Super || !Super->lookupClassMethod(method->getSelector()))) {
      unsigned DIAG = diag::warn_unimplemented_protocol_method;
      if (Diags.getDiagnosticLevel(DIAG) != Diagnostic::Ignored) {
        WarnUndefinedMethod(ImpLoc, method, IncompleteImpl, DIAG);
        Diag(IDecl->getLocation(), diag::note_required_for_protocol_at) <<
          PDecl->getDeclName();
      }
    }
  }
  // Check on this protocols's referenced protocols, recursively.
  for (ObjCProtocolDecl::protocol_iterator PI = PDecl->protocol_begin(),
       E = PDecl->protocol_end(); PI != E; ++PI)
    CheckProtocolMethodDefs(ImpLoc, *PI, IncompleteImpl, InsMap, ClsMap, IDecl);
}

/// MatchAllMethodDeclarations - Check methods declaraed in interface or
/// or protocol against those declared in their implementations.
///
void Sema::MatchAllMethodDeclarations(const llvm::DenseSet<Selector> &InsMap,
                                      const llvm::DenseSet<Selector> &ClsMap,
                                      llvm::DenseSet<Selector> &InsMapSeen,
                                      llvm::DenseSet<Selector> &ClsMapSeen,
                                      ObjCImplDecl* IMPDecl,
                                      ObjCContainerDecl* CDecl,
                                      bool &IncompleteImpl,
                                      bool ImmediateClass) {
  // Check and see if instance methods in class interface have been
  // implemented in the implementation class. If so, their types match.
  for (ObjCInterfaceDecl::instmeth_iterator I = CDecl->instmeth_begin(),
       E = CDecl->instmeth_end(); I != E; ++I) {
    if (InsMapSeen.count((*I)->getSelector()))
        continue;
    InsMapSeen.insert((*I)->getSelector());
    if (!(*I)->isSynthesized() &&
        !InsMap.count((*I)->getSelector())) {
      if (ImmediateClass)
        WarnUndefinedMethod(IMPDecl->getLocation(), *I, IncompleteImpl,
                            diag::note_undef_method_impl);
      continue;
    } else {
      ObjCMethodDecl *ImpMethodDecl =
      IMPDecl->getInstanceMethod((*I)->getSelector());
      ObjCMethodDecl *IntfMethodDecl =
      CDecl->getInstanceMethod((*I)->getSelector());
      assert(IntfMethodDecl &&
             "IntfMethodDecl is null in ImplMethodsVsClassMethods");
      // ImpMethodDecl may be null as in a @dynamic property.
      if (ImpMethodDecl)
        WarnConflictingTypedMethods(ImpMethodDecl, IntfMethodDecl);
    }
  }

  // Check and see if class methods in class interface have been
  // implemented in the implementation class. If so, their types match.
   for (ObjCInterfaceDecl::classmeth_iterator
       I = CDecl->classmeth_begin(), E = CDecl->classmeth_end(); I != E; ++I) {
     if (ClsMapSeen.count((*I)->getSelector()))
       continue;
     ClsMapSeen.insert((*I)->getSelector());
    if (!ClsMap.count((*I)->getSelector())) {
      if (ImmediateClass)
        WarnUndefinedMethod(IMPDecl->getLocation(), *I, IncompleteImpl,
                            diag::note_undef_method_impl);
    } else {
      ObjCMethodDecl *ImpMethodDecl =
        IMPDecl->getClassMethod((*I)->getSelector());
      ObjCMethodDecl *IntfMethodDecl =
        CDecl->getClassMethod((*I)->getSelector());
      WarnConflictingTypedMethods(ImpMethodDecl, IntfMethodDecl);
    }
  }
  if (ObjCInterfaceDecl *I = dyn_cast<ObjCInterfaceDecl> (CDecl)) {
    // Check for any implementation of a methods declared in protocol.
    for (ObjCInterfaceDecl::protocol_iterator PI = I->protocol_begin(),
         E = I->protocol_end(); PI != E; ++PI)
      MatchAllMethodDeclarations(InsMap, ClsMap, InsMapSeen, ClsMapSeen,
                                 IMPDecl,
                                 (*PI), IncompleteImpl, false);
    if (I->getSuperClass())
      MatchAllMethodDeclarations(InsMap, ClsMap, InsMapSeen, ClsMapSeen,
                                 IMPDecl,
                                 I->getSuperClass(), IncompleteImpl, false);
  }
}

void Sema::ImplMethodsVsClassMethods(Scope *S, ObjCImplDecl* IMPDecl,
                                     ObjCContainerDecl* CDecl,
                                     bool IncompleteImpl) {
  llvm::DenseSet<Selector> InsMap;
  // Check and see if instance methods in class interface have been
  // implemented in the implementation class.
  for (ObjCImplementationDecl::instmeth_iterator
         I = IMPDecl->instmeth_begin(), E = IMPDecl->instmeth_end(); I!=E; ++I)
    InsMap.insert((*I)->getSelector());

  // Check and see if properties declared in the interface have either 1)
  // an implementation or 2) there is a @synthesize/@dynamic implementation
  // of the property in the @implementation.
  if (isa<ObjCInterfaceDecl>(CDecl) && !LangOpts.ObjCNonFragileABI2)
    DiagnoseUnimplementedProperties(S, IMPDecl, CDecl, InsMap);
      
  llvm::DenseSet<Selector> ClsMap;
  for (ObjCImplementationDecl::classmeth_iterator
       I = IMPDecl->classmeth_begin(),
       E = IMPDecl->classmeth_end(); I != E; ++I)
    ClsMap.insert((*I)->getSelector());

  // Check for type conflict of methods declared in a class/protocol and
  // its implementation; if any.
  llvm::DenseSet<Selector> InsMapSeen, ClsMapSeen;
  MatchAllMethodDeclarations(InsMap, ClsMap, InsMapSeen, ClsMapSeen,
                             IMPDecl, CDecl,
                             IncompleteImpl, true);

  // Check the protocol list for unimplemented methods in the @implementation
  // class.
  // Check and see if class methods in class interface have been
  // implemented in the implementation class.

  if (ObjCInterfaceDecl *I = dyn_cast<ObjCInterfaceDecl> (CDecl)) {
    for (ObjCInterfaceDecl::protocol_iterator PI = I->protocol_begin(),
         E = I->protocol_end(); PI != E; ++PI)
      CheckProtocolMethodDefs(IMPDecl->getLocation(), *PI, IncompleteImpl,
                              InsMap, ClsMap, I);
    // Check class extensions (unnamed categories)
    for (ObjCCategoryDecl *Categories = I->getCategoryList();
         Categories; Categories = Categories->getNextClassCategory()) {
      if (Categories->IsClassExtension()) {
        ImplMethodsVsClassMethods(S, IMPDecl, Categories, IncompleteImpl);
        break;
      }
    }
  } else if (ObjCCategoryDecl *C = dyn_cast<ObjCCategoryDecl>(CDecl)) {
    // For extended class, unimplemented methods in its protocols will
    // be reported in the primary class.
    if (!C->IsClassExtension()) {
      for (ObjCCategoryDecl::protocol_iterator PI = C->protocol_begin(),
           E = C->protocol_end(); PI != E; ++PI)
        CheckProtocolMethodDefs(IMPDecl->getLocation(), *PI, IncompleteImpl,
                                InsMap, ClsMap, CDecl);
      // Report unimplemented properties in the category as well.
      // When reporting on missing setter/getters, do not report when
      // setter/getter is implemented in category's primary class 
      // implementation.
      if (ObjCInterfaceDecl *ID = C->getClassInterface())
        if (ObjCImplDecl *IMP = ID->getImplementation()) {
          for (ObjCImplementationDecl::instmeth_iterator
               I = IMP->instmeth_begin(), E = IMP->instmeth_end(); I!=E; ++I)
            InsMap.insert((*I)->getSelector());
        }
      DiagnoseUnimplementedProperties(S, IMPDecl, CDecl, InsMap);      
    } 
  } else
    assert(false && "invalid ObjCContainerDecl type.");
}

/// ActOnForwardClassDeclaration -
Action::DeclPtrTy
Sema::ActOnForwardClassDeclaration(SourceLocation AtClassLoc,
                                   IdentifierInfo **IdentList,
                                   SourceLocation *IdentLocs,
                                   unsigned NumElts) {
  llvm::SmallVector<ObjCInterfaceDecl*, 32> Interfaces;

  for (unsigned i = 0; i != NumElts; ++i) {
    // Check for another declaration kind with the same name.
    NamedDecl *PrevDecl
      = LookupSingleName(TUScope, IdentList[i], IdentLocs[i], 
                         LookupOrdinaryName, ForRedeclaration);
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
      if (!TDD || !TDD->getUnderlyingType()->isObjCObjectType()) {
        Diag(AtClassLoc, diag::err_redefinition_different_kind) << IdentList[i];
        Diag(PrevDecl->getLocation(), diag::note_previous_definition);
      } else {
        // a forward class declaration matching a typedef name of a class refers
        // to the underlying class.
        if (const ObjCObjectType *OI =
              TDD->getUnderlyingType()->getAs<ObjCObjectType>())
          PrevDecl = OI->getInterface();
      }
    }
    ObjCInterfaceDecl *IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(PrevDecl);
    if (!IDecl) {  // Not already seen?  Make a forward decl.
      IDecl = ObjCInterfaceDecl::Create(Context, CurContext, AtClassLoc,
                                        IdentList[i], IdentLocs[i], true);
      
      // Push the ObjCInterfaceDecl on the scope chain but do *not* add it to
      // the current DeclContext.  This prevents clients that walk DeclContext
      // from seeing the imaginary ObjCInterfaceDecl until it is actually
      // declared later (if at all).  We also take care to explicitly make
      // sure this declaration is visible for name lookup.
      PushOnScopeChains(IDecl, TUScope, false);
      CurContext->makeDeclVisibleInContext(IDecl, true);
    }

    Interfaces.push_back(IDecl);
  }

  assert(Interfaces.size() == NumElts);
  ObjCClassDecl *CDecl = ObjCClassDecl::Create(Context, CurContext, AtClassLoc,
                                               Interfaces.data(), IdentLocs,
                                               Interfaces.size());
  CurContext->addDecl(CDecl);
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

/// \brief Read the contents of the instance and factory method pools
/// for a given selector from external storage.
///
/// This routine should only be called once, when neither the instance
/// nor the factory method pool has an entry for this selector.
Sema::MethodPool::iterator Sema::ReadMethodPool(Selector Sel,
                                                bool isInstance) {
  assert(ExternalSource && "We need an external AST source");
  assert(InstanceMethodPool.find(Sel) == InstanceMethodPool.end() &&
         "Selector data already loaded into the instance method pool");
  assert(FactoryMethodPool.find(Sel) == FactoryMethodPool.end() &&
         "Selector data already loaded into the factory method pool");

  // Read the method list from the external source.
  std::pair<ObjCMethodList, ObjCMethodList> Methods
    = ExternalSource->ReadMethodPool(Sel);

  if (isInstance) {
    if (Methods.second.Method)
      FactoryMethodPool[Sel] = Methods.second;
    return InstanceMethodPool.insert(std::make_pair(Sel, Methods.first)).first;
  }

  if (Methods.first.Method)
    InstanceMethodPool[Sel] = Methods.first;

  return FactoryMethodPool.insert(std::make_pair(Sel, Methods.second)).first;
}

void Sema::AddInstanceMethodToGlobalPool(ObjCMethodDecl *Method) {
  llvm::DenseMap<Selector, ObjCMethodList>::iterator Pos
    = InstanceMethodPool.find(Method->getSelector());
  if (Pos == InstanceMethodPool.end()) {
    if (ExternalSource && !FactoryMethodPool.count(Method->getSelector()))
      Pos = ReadMethodPool(Method->getSelector(), /*isInstance=*/true);
    else
      Pos = InstanceMethodPool.insert(std::make_pair(Method->getSelector(),
                                                     ObjCMethodList())).first;
  }

  ObjCMethodList &Entry = Pos->second;
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
  ObjCMethodList *Mem = BumpAlloc.Allocate<ObjCMethodList>();
  Entry.Next = new (Mem) ObjCMethodList(Method, Entry.Next);
}

// FIXME: Finish implementing -Wno-strict-selector-match.
ObjCMethodDecl *Sema::LookupInstanceMethodInGlobalPool(Selector Sel,
                                                       SourceRange R,
                                                       bool warn) {
  llvm::DenseMap<Selector, ObjCMethodList>::iterator Pos
    = InstanceMethodPool.find(Sel);
  if (Pos == InstanceMethodPool.end()) {
    if (ExternalSource && !FactoryMethodPool.count(Sel))
      Pos = ReadMethodPool(Sel, /*isInstance=*/true);
    else
      return 0;
  }

  ObjCMethodList &MethList = Pos->second;
  bool issueWarning = false;

  if (MethList.Method && MethList.Next) {
    for (ObjCMethodList *Next = MethList.Next; Next; Next = Next->Next)
      // This checks if the methods differ by size & alignment.
      if (!MatchTwoMethodDeclarations(MethList.Method, Next->Method, true))
        issueWarning = warn;
  }
  if (issueWarning && (MethList.Method && MethList.Next)) {
    Diag(R.getBegin(), diag::warn_multiple_method_decl) << Sel << R;
    Diag(MethList.Method->getLocStart(), diag::note_using)
      << MethList.Method->getSourceRange();
    for (ObjCMethodList *Next = MethList.Next; Next; Next = Next->Next)
      Diag(Next->Method->getLocStart(), diag::note_also_found)
        << Next->Method->getSourceRange();
  }
  return MethList.Method;
}

void Sema::AddFactoryMethodToGlobalPool(ObjCMethodDecl *Method) {
  llvm::DenseMap<Selector, ObjCMethodList>::iterator Pos
    = FactoryMethodPool.find(Method->getSelector());
  if (Pos == FactoryMethodPool.end()) {
    if (ExternalSource && !InstanceMethodPool.count(Method->getSelector()))
      Pos = ReadMethodPool(Method->getSelector(), /*isInstance=*/false);
    else
      Pos = FactoryMethodPool.insert(std::make_pair(Method->getSelector(),
                                                    ObjCMethodList())).first;
  }

  ObjCMethodList &FirstMethod = Pos->second;
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
      ObjCMethodList *Mem = BumpAlloc.Allocate<ObjCMethodList>();
      ObjCMethodList *OMI = new (Mem) ObjCMethodList(Method, FirstMethod.Next);
      FirstMethod.Next = OMI;
    }
  }
}

ObjCMethodDecl *Sema::LookupFactoryMethodInGlobalPool(Selector Sel,
                                                      SourceRange R) {
  llvm::DenseMap<Selector, ObjCMethodList>::iterator Pos
    = FactoryMethodPool.find(Sel);
  if (Pos == FactoryMethodPool.end()) {
    if (ExternalSource && !InstanceMethodPool.count(Sel))
      Pos = ReadMethodPool(Sel, /*isInstance=*/false);
    else
      return 0;
  }

  ObjCMethodList &MethList = Pos->second;
  bool issueWarning = false;

  if (MethList.Method && MethList.Next) {
    for (ObjCMethodList *Next = MethList.Next; Next; Next = Next->Next)
      // This checks if the methods differ by size & alignment.
      if (!MatchTwoMethodDeclarations(MethList.Method, Next->Method, true))
        issueWarning = true;
  }
  if (issueWarning && (MethList.Method && MethList.Next)) {
    Diag(R.getBegin(), diag::warn_multiple_method_decl) << Sel << R;
    Diag(MethList.Method->getLocStart(), diag::note_using)
      << MethList.Method->getSourceRange();
    for (ObjCMethodList *Next = MethList.Next; Next; Next = Next->Next)
      Diag(Next->Method->getLocStart(), diag::note_also_found)
        << Next->Method->getSourceRange();
  }
  return MethList.Method;
}

/// CompareMethodParamsInBaseAndSuper - This routine compares methods with
/// identical selector names in current and its super classes and issues
/// a warning if any of their argument types are incompatible.
void Sema::CompareMethodParamsInBaseAndSuper(Decl *ClassDecl,
                                             ObjCMethodDecl *Method,
                                             bool IsInstance)  {
  ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(ClassDecl);
  if (ID == 0) return;

  while (ObjCInterfaceDecl *SD = ID->getSuperClass()) {
    ObjCMethodDecl *SuperMethodDecl =
        SD->lookupMethod(Method->getSelector(), IsInstance);
    if (SuperMethodDecl == 0) {
      ID = SD;
      continue;
    }
    ObjCMethodDecl::param_iterator ParamI = Method->param_begin(),
      E = Method->param_end();
    ObjCMethodDecl::param_iterator PrevI = SuperMethodDecl->param_begin();
    for (; ParamI != E; ++ParamI, ++PrevI) {
      // Number of parameters are the same and is guaranteed by selector match.
      assert(PrevI != SuperMethodDecl->param_end() && "Param mismatch");
      QualType T1 = Context.getCanonicalType((*ParamI)->getType());
      QualType T2 = Context.getCanonicalType((*PrevI)->getType());
      // If type of arguement of method in this class does not match its
      // respective argument type in the super class method, issue warning;
      if (!Context.typesAreCompatible(T1, T2)) {
        Diag((*ParamI)->getLocation(), diag::ext_typecheck_base_super)
          << T1 << T2;
        Diag(SuperMethodDecl->getLocation(), diag::note_previous_declaration);
        return;
      }
    }
    ID = SD;
  }
}

/// DiagnoseDuplicateIvars - 
/// Check for duplicate ivars in the entire class at the start of 
/// @implementation. This becomes necesssary because class extension can
/// add ivars to a class in random order which will not be known until
/// class's @implementation is seen.
void Sema::DiagnoseDuplicateIvars(ObjCInterfaceDecl *ID, 
                                  ObjCInterfaceDecl *SID) {
  for (ObjCInterfaceDecl::ivar_iterator IVI = ID->ivar_begin(),
       IVE = ID->ivar_end(); IVI != IVE; ++IVI) {
    ObjCIvarDecl* Ivar = (*IVI);
    if (Ivar->isInvalidDecl())
      continue;
    if (IdentifierInfo *II = Ivar->getIdentifier()) {
      ObjCIvarDecl* prevIvar = SID->lookupInstanceVariable(II);
      if (prevIvar) {
        Diag(Ivar->getLocation(), diag::err_duplicate_member) << II;
        Diag(prevIvar->getLocation(), diag::note_previous_declaration);
        Ivar->setInvalidDecl();
      }
    }
  }
}

// Note: For class/category implemenations, allMethods/allProperties is
// always null.
void Sema::ActOnAtEnd(Scope *S, SourceRange AtEnd,
                      DeclPtrTy classDecl,
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

  if (!isInterfaceDeclKind && AtEnd.isInvalid()) {
    // FIXME: This is wrong.  We shouldn't be pretending that there is
    //  an '@end' in the declaration.
    SourceLocation L = ClassDecl->getLocation();
    AtEnd.setBegin(L);
    AtEnd.setEnd(L);
    Diag(L, diag::warn_missing_atend);
  }
  
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
        DC->addDecl(Method);
        InsMap[Method->getSelector()] = Method;
        /// The following allows us to typecheck messages to "id".
        AddInstanceMethodToGlobalPool(Method);
        // verify that the instance method conforms to the same definition of
        // parent methods if it shadows one.
        CompareMethodParamsInBaseAndSuper(ClassDecl, Method, true);
      }
    } else {
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
        DC->addDecl(Method);
        ClsMap[Method->getSelector()] = Method;
        /// The following allows us to typecheck messages to "Class".
        AddFactoryMethodToGlobalPool(Method);
        // verify that the class method conforms to the same definition of
        // parent methods if it shadows one.
        CompareMethodParamsInBaseAndSuper(ClassDecl, Method, false);
      }
    }
  }
  if (ObjCInterfaceDecl *I = dyn_cast<ObjCInterfaceDecl>(ClassDecl)) {
    // Compares properties declared in this class to those of its
    // super class.
    ComparePropertiesInBaseAndSuper(I);
    CompareProperties(I, DeclPtrTy::make(I));
  } else if (ObjCCategoryDecl *C = dyn_cast<ObjCCategoryDecl>(ClassDecl)) {
    // Categories are used to extend the class by declaring new methods.
    // By the same token, they are also used to add new properties. No
    // need to compare the added property to those in the class.

    // Compare protocol properties with those in category
    CompareProperties(C, DeclPtrTy::make(C));
    if (C->IsClassExtension())
      DiagnoseClassExtensionDupMethods(C, C->getClassInterface());
  }
  if (ObjCContainerDecl *CDecl = dyn_cast<ObjCContainerDecl>(ClassDecl)) {
    if (CDecl->getIdentifier())
      // ProcessPropertyDecl is responsible for diagnosing conflicts with any
      // user-defined setter/getter. It also synthesizes setter/getter methods
      // and adds them to the DeclContext and global method pools.
      for (ObjCContainerDecl::prop_iterator I = CDecl->prop_begin(),
                                            E = CDecl->prop_end();
           I != E; ++I)
        ProcessPropertyDecl(*I, CDecl);
    CDecl->setAtEndRange(AtEnd);
  }
  if (ObjCImplementationDecl *IC=dyn_cast<ObjCImplementationDecl>(ClassDecl)) {
    IC->setAtEndRange(AtEnd);
    if (ObjCInterfaceDecl* IDecl = IC->getClassInterface()) {
      if (LangOpts.ObjCNonFragileABI2)
        DefaultSynthesizeProperties(S, IC, IDecl);
      ImplMethodsVsClassMethods(S, IC, IDecl);
      AtomicPropertySetterGetterRules(IC, IDecl);
      if (LangOpts.ObjCNonFragileABI2)
        while (IDecl->getSuperClass()) {
          DiagnoseDuplicateIvars(IDecl, IDecl->getSuperClass());
          IDecl = IDecl->getSuperClass();
        }
    }
    SetIvarInitializers(IC);
  } else if (ObjCCategoryImplDecl* CatImplClass =
                                   dyn_cast<ObjCCategoryImplDecl>(ClassDecl)) {
    CatImplClass->setAtEndRange(AtEnd);

    // Find category interface decl and then check that all methods declared
    // in this interface are implemented in the category @implementation.
    if (ObjCInterfaceDecl* IDecl = CatImplClass->getClassInterface()) {
      for (ObjCCategoryDecl *Categories = IDecl->getCategoryList();
           Categories; Categories = Categories->getNextClassCategory()) {
        if (Categories->getIdentifier() == CatImplClass->getIdentifier()) {
          ImplMethodsVsClassMethods(S, CatImplClass, Categories);
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
          if (!VDecl->hasExternalStorage())
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

static inline
bool containsInvalidMethodImplAttribute(const AttributeList *A) {
  // The 'ibaction' attribute is allowed on method definitions because of
  // how the IBAction macro is used on both method declarations and definitions.
  // If the method definitions contains any other attributes, return true.
  while (A && A->getKind() == AttributeList::AT_IBAction)
    A = A->getNext();
  return A != NULL;
}

Sema::DeclPtrTy Sema::ActOnMethodDeclaration(
    SourceLocation MethodLoc, SourceLocation EndLoc,
    tok::TokenKind MethodType, DeclPtrTy classDecl,
    ObjCDeclSpec &ReturnQT, TypeTy *ReturnType,
    Selector Sel,
    // optional arguments. The number of types/arguments is obtained
    // from the Sel.getNumArgs().
    ObjCArgInfo *ArgInfo,
    DeclaratorChunk::ParamInfo *CParamInfo, unsigned CNumArgs, // c-style args
    AttributeList *AttrList, tok::ObjCKeywordKind MethodDeclKind,
    bool isVariadic) {
  Decl *ClassDecl = classDecl.getAs<Decl>();

  // Make sure we can establish a context for the method.
  if (!ClassDecl) {
    Diag(MethodLoc, diag::error_missing_method_context);
    getLabelMap().clear();
    return DeclPtrTy();
  }
  QualType resultDeclType;

  TypeSourceInfo *ResultTInfo = 0;
  if (ReturnType) {
    resultDeclType = GetTypeFromParser(ReturnType, &ResultTInfo);

    // Methods cannot return interface types. All ObjC objects are
    // passed by reference.
    if (resultDeclType->isObjCObjectType()) {
      Diag(MethodLoc, diag::err_object_cannot_be_passed_returned_by_value)
        << 0 << resultDeclType;
      return DeclPtrTy();
    }
  } else // get the type for "id".
    resultDeclType = Context.getObjCIdType();

  ObjCMethodDecl* ObjCMethod =
    ObjCMethodDecl::Create(Context, MethodLoc, EndLoc, Sel, resultDeclType,
                           ResultTInfo,
                           cast<DeclContext>(ClassDecl),
                           MethodType == tok::minus, isVariadic,
                           false,
                           MethodDeclKind == tok::objc_optional ?
                           ObjCMethodDecl::Optional :
                           ObjCMethodDecl::Required);

  llvm::SmallVector<ParmVarDecl*, 16> Params;

  for (unsigned i = 0, e = Sel.getNumArgs(); i != e; ++i) {
    QualType ArgType;
    TypeSourceInfo *DI;

    if (ArgInfo[i].Type == 0) {
      ArgType = Context.getObjCIdType();
      DI = 0;
    } else {
      ArgType = GetTypeFromParser(ArgInfo[i].Type, &DI);
      // Perform the default array/function conversions (C99 6.7.5.3p[7,8]).
      ArgType = adjustParameterType(ArgType);
    }

    ParmVarDecl* Param
      = ParmVarDecl::Create(Context, ObjCMethod, ArgInfo[i].NameLoc,
                            ArgInfo[i].Name, ArgType, DI,
                            VarDecl::None, VarDecl::None, 0);

    if (ArgType->isObjCObjectType()) {
      Diag(ArgInfo[i].NameLoc,
           diag::err_object_cannot_be_passed_returned_by_value)
        << 1 << ArgType;
      Param->setInvalidDecl();
    }

    Param->setObjCDeclQualifier(
      CvtQTToAstBitMask(ArgInfo[i].DeclSpec.getObjCDeclQualifier()));

    // Apply the attributes to the parameter.
    ProcessDeclAttributeList(TUScope, Param, ArgInfo[i].ArgAttrs);

    Params.push_back(Param);
  }

  for (unsigned i = 0, e = CNumArgs; i != e; ++i) {
    ParmVarDecl *Param = CParamInfo[i].Param.getAs<ParmVarDecl>();
    QualType ArgType = Param->getType();
    if (ArgType.isNull())
      ArgType = Context.getObjCIdType();
    else
      // Perform the default array/function conversions (C99 6.7.5.3p[7,8]).
      ArgType = adjustParameterType(ArgType);
    if (ArgType->isObjCObjectType()) {
      Diag(Param->getLocation(),
           diag::err_object_cannot_be_passed_returned_by_value)
      << 1 << ArgType;
      Param->setInvalidDecl();
    }
    Param->setDeclContext(ObjCMethod);
    IdResolver.RemoveDecl(Param);
    Params.push_back(Param);
  }
  
  ObjCMethod->setMethodParams(Context, Params.data(), Params.size(),
                              Sel.getNumArgs());
  ObjCMethod->setObjCDeclQualifier(
    CvtQTToAstBitMask(ReturnQT.getObjCDeclQualifier()));
  const ObjCMethodDecl *PrevMethod = 0;

  if (AttrList)
    ProcessDeclAttributeList(TUScope, ObjCMethod, AttrList);

  const ObjCMethodDecl *InterfaceMD = 0;

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
    InterfaceMD = ImpDecl->getClassInterface()->getMethod(Sel,
                                                   MethodType == tok::minus);
    if (containsInvalidMethodImplAttribute(AttrList))
      Diag(EndLoc, diag::warn_attribute_method_def);
  } else if (ObjCCategoryImplDecl *CatImpDecl =
             dyn_cast<ObjCCategoryImplDecl>(ClassDecl)) {
    if (MethodType == tok::minus) {
      PrevMethod = CatImpDecl->getInstanceMethod(Sel);
      CatImpDecl->addInstanceMethod(ObjCMethod);
    } else {
      PrevMethod = CatImpDecl->getClassMethod(Sel);
      CatImpDecl->addClassMethod(ObjCMethod);
    }
    if (containsInvalidMethodImplAttribute(AttrList))
      Diag(EndLoc, diag::warn_attribute_method_def);
  }
  if (PrevMethod) {
    // You can never have two method definitions with the same name.
    Diag(ObjCMethod->getLocation(), diag::err_duplicate_method_decl)
      << ObjCMethod->getDeclName();
    Diag(PrevMethod->getLocation(), diag::note_previous_declaration);
  }

  // If the interface declared this method, and it was deprecated there,
  // mark it deprecated here.
  if (InterfaceMD && InterfaceMD->hasAttr<DeprecatedAttr>())
    ObjCMethod->addAttr(::new (Context) DeprecatedAttr());

  return DeclPtrTy::make(ObjCMethod);
}

bool Sema::CheckObjCDeclScope(Decl *D) {
  if (isa<TranslationUnitDecl>(CurContext->getLookupContext()))
    return false;

  Diag(D->getLocation(), diag::err_objc_decls_may_only_appear_in_global_scope);
  D->setInvalidDecl();

  return true;
}

/// Called whenever @defs(ClassName) is encountered in the source.  Inserts the
/// instance variables of ClassName into Decls.
void Sema::ActOnDefs(Scope *S, DeclPtrTy TagD, SourceLocation DeclStart,
                     IdentifierInfo *ClassName,
                     llvm::SmallVectorImpl<DeclPtrTy> &Decls) {
  // Check that ClassName is a valid class
  ObjCInterfaceDecl *Class = getObjCInterfaceDecl(ClassName, DeclStart);
  if (!Class) {
    Diag(DeclStart, diag::err_undef_interface) << ClassName;
    return;
  }
  if (LangOpts.ObjCNonFragileABI) {
    Diag(DeclStart, diag::err_atdef_nonfragile_interface);
    return;
  }

  // Collect the instance variables
  llvm::SmallVector<FieldDecl*, 32> RecFields;
  Context.CollectObjCIvars(Class, RecFields);
  // For each ivar, create a fresh ObjCAtDefsFieldDecl.
  for (unsigned i = 0; i < RecFields.size(); i++) {
    FieldDecl* ID = RecFields[i];
    RecordDecl *Record = dyn_cast<RecordDecl>(TagD.getAs<Decl>());
    Decl *FD = ObjCAtDefsFieldDecl::Create(Context, Record, ID->getLocation(),
                                           ID->getIdentifier(), ID->getType(),
                                           ID->getBitWidth());
    Decls.push_back(Sema::DeclPtrTy::make(FD));
  }

  // Introduce all of these fields into the appropriate scope.
  for (llvm::SmallVectorImpl<DeclPtrTy>::iterator D = Decls.begin();
       D != Decls.end(); ++D) {
    FieldDecl *FD = cast<FieldDecl>(D->getAs<Decl>());
    if (getLangOptions().CPlusPlus)
      PushOnScopeChains(cast<FieldDecl>(FD), S);
    else if (RecordDecl *Record = dyn_cast<RecordDecl>(TagD.getAs<Decl>()))
      Record->addDecl(FD);
  }
}

/// \brief Build a type-check a new Objective-C exception variable declaration.
VarDecl *Sema::BuildObjCExceptionDecl(TypeSourceInfo *TInfo, 
                                      QualType T,
                                      IdentifierInfo *Name, 
                                      SourceLocation NameLoc,
                                      bool Invalid) {
  // ISO/IEC TR 18037 S6.7.3: "The type of an object with automatic storage 
  // duration shall not be qualified by an address-space qualifier."
  // Since all parameters have automatic store duration, they can not have
  // an address space.
  if (T.getAddressSpace() != 0) {
    Diag(NameLoc, diag::err_arg_with_address_space);
    Invalid = true;
  }
  
  // An @catch parameter must be an unqualified object pointer type;
  // FIXME: Recover from "NSObject foo" by inserting the * in "NSObject *foo"?
  if (Invalid) {
    // Don't do any further checking.
  } else if (T->isDependentType()) {
    // Okay: we don't know what this type will instantiate to.
  } else if (!T->isObjCObjectPointerType()) {
    Invalid = true;
    Diag(NameLoc ,diag::err_catch_param_not_objc_type);
  } else if (T->isObjCQualifiedIdType()) {
    Invalid = true;
    Diag(NameLoc, diag::err_illegal_qualifiers_on_catch_parm);
  }
  
  VarDecl *New = VarDecl::Create(Context, CurContext, NameLoc, Name, T, TInfo,
                                 VarDecl::None, VarDecl::None);
  New->setExceptionVariable(true);
  
  if (Invalid)
    New->setInvalidDecl();
  return New;
}

Sema::DeclPtrTy Sema::ActOnObjCExceptionDecl(Scope *S, Declarator &D) {
  const DeclSpec &DS = D.getDeclSpec();
  
  // We allow the "register" storage class on exception variables because
  // GCC did, but we drop it completely. Any other storage class is an error.
  if (DS.getStorageClassSpec() == DeclSpec::SCS_register) {
    Diag(DS.getStorageClassSpecLoc(), diag::warn_register_objc_catch_parm)
      << FixItHint::CreateRemoval(SourceRange(DS.getStorageClassSpecLoc()));
  } else if (DS.getStorageClassSpec() != DeclSpec::SCS_unspecified) {
    Diag(DS.getStorageClassSpecLoc(), diag::err_storage_spec_on_catch_parm)
      << DS.getStorageClassSpec();
  }  
  if (D.getDeclSpec().isThreadSpecified())
    Diag(D.getDeclSpec().getThreadSpecLoc(), diag::err_invalid_thread);
  D.getMutableDeclSpec().ClearStorageClassSpecs();

  DiagnoseFunctionSpecifiers(D);
  
  // Check that there are no default arguments inside the type of this
  // exception object (C++ only).
  if (getLangOptions().CPlusPlus)
    CheckExtraCXXDefaultArguments(D);
  
  TypeSourceInfo *TInfo = 0;
  TagDecl *OwnedDecl = 0;
  QualType ExceptionType = GetTypeForDeclarator(D, S, &TInfo, &OwnedDecl);
  
  if (getLangOptions().CPlusPlus && OwnedDecl && OwnedDecl->isDefinition()) {
    // Objective-C++: Types shall not be defined in exception types.
    Diag(OwnedDecl->getLocation(), diag::err_type_defined_in_param_type)
      << Context.getTypeDeclType(OwnedDecl);
  }

  VarDecl *New = BuildObjCExceptionDecl(TInfo, ExceptionType, D.getIdentifier(), 
                                        D.getIdentifierLoc(), 
                                        D.isInvalidType());
  
  // Parameter declarators cannot be qualified (C++ [dcl.meaning]p1).
  if (D.getCXXScopeSpec().isSet()) {
    Diag(D.getIdentifierLoc(), diag::err_qualified_objc_catch_parm)
      << D.getCXXScopeSpec().getRange();
    New->setInvalidDecl();
  }
  
  // Add the parameter declaration into this scope.
  S->AddDecl(DeclPtrTy::make(New));
  if (D.getIdentifier())
    IdResolver.AddDecl(New);
  
  ProcessDeclAttributes(S, New, D);
  
  if (New->hasAttr<BlocksAttr>())
    Diag(New->getLocation(), diag::err_block_on_nonlocal);
  return DeclPtrTy::make(New);
}

/// CollectIvarsToConstructOrDestruct - Collect those ivars which require
/// initialization.
void Sema::CollectIvarsToConstructOrDestruct(const ObjCInterfaceDecl *OI,
                                llvm::SmallVectorImpl<ObjCIvarDecl*> &Ivars) {
  for (ObjCInterfaceDecl::ivar_iterator I = OI->ivar_begin(),
       E = OI->ivar_end(); I != E; ++I) {
    ObjCIvarDecl *Iv = (*I);
    QualType QT = Context.getBaseElementType(Iv->getType());
    if (QT->isRecordType())
      Ivars.push_back(*I);
  }
  
  // Find ivars to construct/destruct in class extension.
  if (const ObjCCategoryDecl *CDecl = OI->getClassExtension()) {
    for (ObjCCategoryDecl::ivar_iterator I = CDecl->ivar_begin(),
         E = CDecl->ivar_end(); I != E; ++I) {
      ObjCIvarDecl *Iv = (*I);
      QualType QT = Context.getBaseElementType(Iv->getType());
      if (QT->isRecordType())
        Ivars.push_back(*I);
    }
  }
  
  // Also add any ivar defined in this class's implementation.  This
  // includes synthesized ivars.
  if (ObjCImplementationDecl *ImplDecl = OI->getImplementation()) {
    for (ObjCImplementationDecl::ivar_iterator I = ImplDecl->ivar_begin(),
         E = ImplDecl->ivar_end(); I != E; ++I) {
      ObjCIvarDecl *Iv = (*I);
      QualType QT = Context.getBaseElementType(Iv->getType());
      if (QT->isRecordType())
        Ivars.push_back(*I);
    }
  }
}

void ObjCImplementationDecl::setIvarInitializers(ASTContext &C,
                                    CXXBaseOrMemberInitializer ** initializers,
                                                 unsigned numInitializers) {
  if (numInitializers > 0) {
    NumIvarInitializers = numInitializers;
    CXXBaseOrMemberInitializer **ivarInitializers =
    new (C) CXXBaseOrMemberInitializer*[NumIvarInitializers];
    memcpy(ivarInitializers, initializers,
           numInitializers * sizeof(CXXBaseOrMemberInitializer*));
    IvarInitializers = ivarInitializers;
  }
}

