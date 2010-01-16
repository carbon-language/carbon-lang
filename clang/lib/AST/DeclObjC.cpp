//===--- DeclObjC.cpp - ObjC Declaration AST Node Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Objective-C related Decl classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclObjC.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/STLExtras.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// ObjCListBase
//===----------------------------------------------------------------------===//

void ObjCListBase::Destroy(ASTContext &Ctx) {
  Ctx.Deallocate(List);
  NumElts = 0;
  List = 0;
}

void ObjCListBase::set(void *const* InList, unsigned Elts, ASTContext &Ctx) {
  assert(List == 0 && "Elements already set!");
  if (Elts == 0) return;  // Setting to an empty list is a noop.


  List = new (Ctx) void*[Elts];
  NumElts = Elts;
  memcpy(List, InList, sizeof(void*)*Elts);
}

void ObjCProtocolList::set(ObjCProtocolDecl* const* InList, unsigned Elts, 
                           const SourceLocation *Locs, ASTContext &Ctx) {
  if (Elts == 0)
    return;

  Locations = new (Ctx) SourceLocation[Elts];
  memcpy(Locations, Locs, sizeof(SourceLocation) * Elts);
  set(InList, Elts, Ctx);
}

void ObjCProtocolList::Destroy(ASTContext &Ctx) {
  Ctx.Deallocate(Locations);
  Locations = 0;
  ObjCList<ObjCProtocolDecl>::Destroy(Ctx);
}

//===----------------------------------------------------------------------===//
// ObjCInterfaceDecl
//===----------------------------------------------------------------------===//

/// getIvarDecl - This method looks up an ivar in this ContextDecl.
///
ObjCIvarDecl *
ObjCContainerDecl::getIvarDecl(IdentifierInfo *Id) const {
  lookup_const_iterator Ivar, IvarEnd;
  for (llvm::tie(Ivar, IvarEnd) = lookup(Id); Ivar != IvarEnd; ++Ivar) {
    if (ObjCIvarDecl *ivar = dyn_cast<ObjCIvarDecl>(*Ivar))
      return ivar;
  }
  return 0;
}

// Get the local instance/class method declared in this interface.
ObjCMethodDecl *
ObjCContainerDecl::getMethod(Selector Sel, bool isInstance) const {
  // Since instance & class methods can have the same name, the loop below
  // ensures we get the correct method.
  //
  // @interface Whatever
  // - (int) class_method;
  // + (float) class_method;
  // @end
  //
  lookup_const_iterator Meth, MethEnd;
  for (llvm::tie(Meth, MethEnd) = lookup(Sel); Meth != MethEnd; ++Meth) {
    ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(*Meth);
    if (MD && MD->isInstanceMethod() == isInstance)
      return MD;
  }
  return 0;
}

/// FindPropertyDeclaration - Finds declaration of the property given its name
/// in 'PropertyId' and returns it. It returns 0, if not found.
/// FIXME: Convert to DeclContext lookup...
///
ObjCPropertyDecl *
ObjCContainerDecl::FindPropertyDeclaration(IdentifierInfo *PropertyId) const {
  for (prop_iterator I = prop_begin(), E = prop_end(); I != E; ++I)
    if ((*I)->getIdentifier() == PropertyId)
      return *I;

  const ObjCProtocolDecl *PID = dyn_cast<ObjCProtocolDecl>(this);
  if (PID) {
    for (ObjCProtocolDecl::protocol_iterator I = PID->protocol_begin(),
         E = PID->protocol_end(); I != E; ++I)
      if (ObjCPropertyDecl *P = (*I)->FindPropertyDeclaration(PropertyId))
        return P;
  }

  if (const ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(this)) {
    // Look through categories.
    for (ObjCCategoryDecl *Category = OID->getCategoryList();
         Category; Category = Category->getNextClassCategory()) {
      if (ObjCPropertyDecl *P = Category->FindPropertyDeclaration(PropertyId))
        return P;
    }
    // Look through protocols.
    for (ObjCInterfaceDecl::protocol_iterator I = OID->protocol_begin(),
         E = OID->protocol_end(); I != E; ++I) {
      if (ObjCPropertyDecl *P = (*I)->FindPropertyDeclaration(PropertyId))
        return P;
    }
    if (OID->getSuperClass())
      return OID->getSuperClass()->FindPropertyDeclaration(PropertyId);
  } else if (const ObjCCategoryDecl *OCD = dyn_cast<ObjCCategoryDecl>(this)) {
    // Look through protocols.
    for (ObjCInterfaceDecl::protocol_iterator I = OCD->protocol_begin(),
         E = OCD->protocol_end(); I != E; ++I) {
      if (ObjCPropertyDecl *P = (*I)->FindPropertyDeclaration(PropertyId))
        return P;
    }
  }
  return 0;
}

/// FindPropertyVisibleInPrimaryClass - Finds declaration of the property
/// with name 'PropertyId' in the primary class; including those in protocols
/// (direct or indirect) used by the promary class.
/// FIXME: Convert to DeclContext lookup...
///
ObjCPropertyDecl *
ObjCContainerDecl::FindPropertyVisibleInPrimaryClass(
                                            IdentifierInfo *PropertyId) const {
  assert(isa<ObjCInterfaceDecl>(this) && "FindPropertyVisibleInPrimaryClass");
  for (prop_iterator I = prop_begin(), E = prop_end(); I != E; ++I)
    if ((*I)->getIdentifier() == PropertyId)
      return *I;
  const ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(this);
  // Look through protocols.
  for (ObjCInterfaceDecl::protocol_iterator I = OID->protocol_begin(),
       E = OID->protocol_end(); I != E; ++I)
    if (ObjCPropertyDecl *P = (*I)->FindPropertyDeclaration(PropertyId))
      return P;
  return 0;
}

void ObjCInterfaceDecl::mergeClassExtensionProtocolList(
                              ObjCProtocolDecl *const* ExtList, unsigned ExtNum,
                              const SourceLocation *Locs,
                              ASTContext &C)
{
  if (ReferencedProtocols.empty()) {
    ReferencedProtocols.set(ExtList, ExtNum, Locs, C);
    return;
  }
  // Check for duplicate protocol in class's protocol list.
  // This is (O)2. But it is extremely rare and number of protocols in
  // class or its extension are very few.
  llvm::SmallVector<ObjCProtocolDecl*, 8> ProtocolRefs;
  llvm::SmallVector<SourceLocation, 8> ProtocolLocs;
  for (unsigned i = 0; i < ExtNum; i++) {
    bool protocolExists = false;
    ObjCProtocolDecl *ProtoInExtension = ExtList[i];
    for (protocol_iterator p = protocol_begin(), e = protocol_end();
         p != e; p++) {
      ObjCProtocolDecl *Proto = (*p);
      if (C.ProtocolCompatibleWithProtocol(ProtoInExtension, Proto)) {
        protocolExists = true;
        break;
      }      
    }
    // Do we want to warn on a protocol in extension class which
    // already exist in the class? Probably not.
    if (!protocolExists) {
      ProtocolRefs.push_back(ProtoInExtension);
      ProtocolLocs.push_back(Locs[i]);
    }
  }
  if (ProtocolRefs.empty())
    return;
  // Merge ProtocolRefs into class's protocol list;
  protocol_loc_iterator pl = protocol_loc_begin();
  for (protocol_iterator p = protocol_begin(), e = protocol_end();
       p != e; ++p, ++pl) {
    ProtocolRefs.push_back(*p);
    ProtocolLocs.push_back(*pl);
  }
  ReferencedProtocols.Destroy(C);
  unsigned NumProtoRefs = ProtocolRefs.size();
  setProtocolList(ProtocolRefs.data(), NumProtoRefs, ProtocolLocs.data(), C);
}

ObjCIvarDecl *ObjCInterfaceDecl::lookupInstanceVariable(IdentifierInfo *ID,
                                              ObjCInterfaceDecl *&clsDeclared) {
  ObjCInterfaceDecl* ClassDecl = this;
  while (ClassDecl != NULL) {
    if (ObjCIvarDecl *I = ClassDecl->getIvarDecl(ID)) {
      clsDeclared = ClassDecl;
      return I;
    }
    ClassDecl = ClassDecl->getSuperClass();
  }
  return NULL;
}

/// lookupInheritedClass - This method returns ObjCInterfaceDecl * of the super
/// class whose name is passed as argument. If it is not one of the super classes
/// the it returns NULL.
ObjCInterfaceDecl *ObjCInterfaceDecl::lookupInheritedClass(
                                        const IdentifierInfo*ICName) {
  ObjCInterfaceDecl* ClassDecl = this;
  while (ClassDecl != NULL) {
    if (ClassDecl->getIdentifier() == ICName)
      return ClassDecl;
    ClassDecl = ClassDecl->getSuperClass();
  }
  return NULL;
}

/// lookupMethod - This method returns an instance/class method by looking in
/// the class, its categories, and its super classes (using a linear search).
ObjCMethodDecl *ObjCInterfaceDecl::lookupMethod(Selector Sel,
                                                bool isInstance) const {
  const ObjCInterfaceDecl* ClassDecl = this;
  ObjCMethodDecl *MethodDecl = 0;

  while (ClassDecl != NULL) {
    if ((MethodDecl = ClassDecl->getMethod(Sel, isInstance)))
      return MethodDecl;

    // Didn't find one yet - look through protocols.
    const ObjCList<ObjCProtocolDecl> &Protocols =
      ClassDecl->getReferencedProtocols();
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
         E = Protocols.end(); I != E; ++I)
      if ((MethodDecl = (*I)->lookupMethod(Sel, isInstance)))
        return MethodDecl;

    // Didn't find one yet - now look through categories.
    ObjCCategoryDecl *CatDecl = ClassDecl->getCategoryList();
    while (CatDecl) {
      if ((MethodDecl = CatDecl->getMethod(Sel, isInstance)))
        return MethodDecl;

      // Didn't find one yet - look through protocols.
      const ObjCList<ObjCProtocolDecl> &Protocols =
        CatDecl->getReferencedProtocols();
      for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
           E = Protocols.end(); I != E; ++I)
        if ((MethodDecl = (*I)->lookupMethod(Sel, isInstance)))
          return MethodDecl;
      CatDecl = CatDecl->getNextClassCategory();
    }
    ClassDecl = ClassDecl->getSuperClass();
  }
  return NULL;
}

ObjCMethodDecl *ObjCInterfaceDecl::lookupPrivateInstanceMethod(
                                   const Selector &Sel) {
  ObjCMethodDecl *Method = 0;
  if (ObjCImplementationDecl *ImpDecl = getImplementation())
    Method = ImpDecl->getInstanceMethod(Sel);
  
  if (!Method && getSuperClass())
    return getSuperClass()->lookupPrivateInstanceMethod(Sel);
  return Method;
}

//===----------------------------------------------------------------------===//
// ObjCMethodDecl
//===----------------------------------------------------------------------===//

ObjCMethodDecl *ObjCMethodDecl::Create(ASTContext &C,
                                       SourceLocation beginLoc,
                                       SourceLocation endLoc,
                                       Selector SelInfo, QualType T,
                                       DeclContext *contextDecl,
                                       bool isInstance,
                                       bool isVariadic,
                                       bool isSynthesized,
                                       ImplementationControl impControl) {
  return new (C) ObjCMethodDecl(beginLoc, endLoc,
                                  SelInfo, T, contextDecl,
                                  isInstance,
                                  isVariadic, isSynthesized, impControl);
}

void ObjCMethodDecl::Destroy(ASTContext &C) {
  if (Body) Body->Destroy(C);
  if (SelfDecl) SelfDecl->Destroy(C);

  for (param_iterator I=param_begin(), E=param_end(); I!=E; ++I)
    if (*I) (*I)->Destroy(C);

  ParamInfo.Destroy(C);

  Decl::Destroy(C);
}

/// \brief A definition will return its interface declaration.
/// An interface declaration will return its definition.
/// Otherwise it will return itself.
ObjCMethodDecl *ObjCMethodDecl::getNextRedeclaration() {
  ASTContext &Ctx = getASTContext();
  ObjCMethodDecl *Redecl = 0;
  Decl *CtxD = cast<Decl>(getDeclContext());

  if (ObjCInterfaceDecl *IFD = dyn_cast<ObjCInterfaceDecl>(CtxD)) {
    if (ObjCImplementationDecl *ImplD = Ctx.getObjCImplementation(IFD))
      Redecl = ImplD->getMethod(getSelector(), isInstanceMethod());

  } else if (ObjCCategoryDecl *CD = dyn_cast<ObjCCategoryDecl>(CtxD)) {
    if (ObjCCategoryImplDecl *ImplD = Ctx.getObjCImplementation(CD))
      Redecl = ImplD->getMethod(getSelector(), isInstanceMethod());

  } else if (ObjCImplementationDecl *ImplD =
               dyn_cast<ObjCImplementationDecl>(CtxD)) {
    if (ObjCInterfaceDecl *IFD = ImplD->getClassInterface())
      Redecl = IFD->getMethod(getSelector(), isInstanceMethod());

  } else if (ObjCCategoryImplDecl *CImplD =
               dyn_cast<ObjCCategoryImplDecl>(CtxD)) {
    if (ObjCCategoryDecl *CatD = CImplD->getCategoryDecl())
      Redecl = CatD->getMethod(getSelector(), isInstanceMethod());
  }

  return Redecl ? Redecl : this;
}

ObjCMethodDecl *ObjCMethodDecl::getCanonicalDecl() {
  Decl *CtxD = cast<Decl>(getDeclContext());

  if (ObjCImplementationDecl *ImplD = dyn_cast<ObjCImplementationDecl>(CtxD)) {
    if (ObjCInterfaceDecl *IFD = ImplD->getClassInterface())
      if (ObjCMethodDecl *MD = IFD->getMethod(getSelector(),
                                              isInstanceMethod()))
        return MD;

  } else if (ObjCCategoryImplDecl *CImplD =
               dyn_cast<ObjCCategoryImplDecl>(CtxD)) {
    if (ObjCCategoryDecl *CatD = CImplD->getCategoryDecl())
      if (ObjCMethodDecl *MD = CatD->getMethod(getSelector(),
                                               isInstanceMethod()))
        return MD;
  }

  return this;
}

void ObjCMethodDecl::createImplicitParams(ASTContext &Context,
                                          const ObjCInterfaceDecl *OID) {
  QualType selfTy;
  if (isInstanceMethod()) {
    // There may be no interface context due to error in declaration
    // of the interface (which has been reported). Recover gracefully.
    if (OID) {
      selfTy = Context.getObjCInterfaceType(OID);
      selfTy = Context.getObjCObjectPointerType(selfTy);
    } else {
      selfTy = Context.getObjCIdType();
    }
  } else // we have a factory method.
    selfTy = Context.getObjCClassType();

  setSelfDecl(ImplicitParamDecl::Create(Context, this, SourceLocation(),
                                        &Context.Idents.get("self"), selfTy));

  setCmdDecl(ImplicitParamDecl::Create(Context, this, SourceLocation(),
                                       &Context.Idents.get("_cmd"),
                                       Context.getObjCSelType()));
}

ObjCInterfaceDecl *ObjCMethodDecl::getClassInterface() {
  if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(getDeclContext()))
    return ID;
  if (ObjCCategoryDecl *CD = dyn_cast<ObjCCategoryDecl>(getDeclContext()))
    return CD->getClassInterface();
  if (ObjCImplDecl *IMD = dyn_cast<ObjCImplDecl>(getDeclContext()))
    return IMD->getClassInterface();

  assert(!isa<ObjCProtocolDecl>(getDeclContext()) && "It's a protocol method");
  assert(false && "unknown method context");
  return 0;
}

//===----------------------------------------------------------------------===//
// ObjCInterfaceDecl
//===----------------------------------------------------------------------===//

ObjCInterfaceDecl *ObjCInterfaceDecl::Create(ASTContext &C,
                                             DeclContext *DC,
                                             SourceLocation atLoc,
                                             IdentifierInfo *Id,
                                             SourceLocation ClassLoc,
                                             bool ForwardDecl, bool isInternal){
  return new (C) ObjCInterfaceDecl(DC, atLoc, Id, ClassLoc, ForwardDecl,
                                     isInternal);
}

ObjCInterfaceDecl::
ObjCInterfaceDecl(DeclContext *DC, SourceLocation atLoc, IdentifierInfo *Id,
                  SourceLocation CLoc, bool FD, bool isInternal)
  : ObjCContainerDecl(ObjCInterface, DC, atLoc, Id),
    TypeForDecl(0), SuperClass(0),
    CategoryList(0), ForwardDecl(FD), InternalInterface(isInternal),
    ClassLoc(CLoc) {
}

void ObjCInterfaceDecl::Destroy(ASTContext &C) {
  for (ivar_iterator I = ivar_begin(), E = ivar_end(); I != E; ++I)
    if (*I) (*I)->Destroy(C);

  IVars.Destroy(C);
  // FIXME: CategoryList?

  // FIXME: Because there is no clear ownership
  //  role between ObjCInterfaceDecls and the ObjCPropertyDecls that they
  //  reference, we destroy ObjCPropertyDecls in ~TranslationUnit.
  Decl::Destroy(C);
}

ObjCImplementationDecl *ObjCInterfaceDecl::getImplementation() const {
  return getASTContext().getObjCImplementation(
                                          const_cast<ObjCInterfaceDecl*>(this));
}

void ObjCInterfaceDecl::setImplementation(ObjCImplementationDecl *ImplD) {
  getASTContext().setObjCImplementation(this, ImplD);
}


/// FindCategoryDeclaration - Finds category declaration in the list of
/// categories for this class and returns it. Name of the category is passed
/// in 'CategoryId'. If category not found, return 0;
///
ObjCCategoryDecl *
ObjCInterfaceDecl::FindCategoryDeclaration(IdentifierInfo *CategoryId) const {
  for (ObjCCategoryDecl *Category = getCategoryList();
       Category; Category = Category->getNextClassCategory())
    if (Category->getIdentifier() == CategoryId)
      return Category;
  return 0;
}

ObjCMethodDecl *
ObjCInterfaceDecl::getCategoryInstanceMethod(Selector Sel) const {
  for (ObjCCategoryDecl *Category = getCategoryList();
       Category; Category = Category->getNextClassCategory())
    if (ObjCCategoryImplDecl *Impl = Category->getImplementation())
      if (ObjCMethodDecl *MD = Impl->getInstanceMethod(Sel))
        return MD;
  return 0;
}

ObjCMethodDecl *ObjCInterfaceDecl::getCategoryClassMethod(Selector Sel) const {
  for (ObjCCategoryDecl *Category = getCategoryList();
       Category; Category = Category->getNextClassCategory())
    if (ObjCCategoryImplDecl *Impl = Category->getImplementation())
      if (ObjCMethodDecl *MD = Impl->getClassMethod(Sel))
        return MD;
  return 0;
}

/// ClassImplementsProtocol - Checks that 'lProto' protocol
/// has been implemented in IDecl class, its super class or categories (if
/// lookupCategory is true).
bool ObjCInterfaceDecl::ClassImplementsProtocol(ObjCProtocolDecl *lProto,
                                    bool lookupCategory,
                                    bool RHSIsQualifiedID) {
  ObjCInterfaceDecl *IDecl = this;
  // 1st, look up the class.
  const ObjCList<ObjCProtocolDecl> &Protocols =
  IDecl->getReferencedProtocols();

  for (ObjCList<ObjCProtocolDecl>::iterator PI = Protocols.begin(),
       E = Protocols.end(); PI != E; ++PI) {
    if (getASTContext().ProtocolCompatibleWithProtocol(lProto, *PI))
      return true;
    // This is dubious and is added to be compatible with gcc.  In gcc, it is
    // also allowed assigning a protocol-qualified 'id' type to a LHS object
    // when protocol in qualified LHS is in list of protocols in the rhs 'id'
    // object. This IMO, should be a bug.
    // FIXME: Treat this as an extension, and flag this as an error when GCC
    // extensions are not enabled.
    if (RHSIsQualifiedID &&
        getASTContext().ProtocolCompatibleWithProtocol(*PI, lProto))
      return true;
  }

  // 2nd, look up the category.
  if (lookupCategory)
    for (ObjCCategoryDecl *CDecl = IDecl->getCategoryList(); CDecl;
         CDecl = CDecl->getNextClassCategory()) {
      for (ObjCCategoryDecl::protocol_iterator PI = CDecl->protocol_begin(),
           E = CDecl->protocol_end(); PI != E; ++PI)
        if (getASTContext().ProtocolCompatibleWithProtocol(lProto, *PI))
          return true;
    }

  // 3rd, look up the super class(s)
  if (IDecl->getSuperClass())
    return
  IDecl->getSuperClass()->ClassImplementsProtocol(lProto, lookupCategory,
                                                  RHSIsQualifiedID);

  return false;
}

//===----------------------------------------------------------------------===//
// ObjCIvarDecl
//===----------------------------------------------------------------------===//

ObjCIvarDecl *ObjCIvarDecl::Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation L, IdentifierInfo *Id,
                                   QualType T, TypeSourceInfo *TInfo,
                                   AccessControl ac, Expr *BW) {
  return new (C) ObjCIvarDecl(DC, L, Id, T, TInfo, ac, BW);
}



//===----------------------------------------------------------------------===//
// ObjCAtDefsFieldDecl
//===----------------------------------------------------------------------===//

ObjCAtDefsFieldDecl
*ObjCAtDefsFieldDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                             IdentifierInfo *Id, QualType T, Expr *BW) {
  return new (C) ObjCAtDefsFieldDecl(DC, L, Id, T, BW);
}

void ObjCAtDefsFieldDecl::Destroy(ASTContext& C) {
  this->~ObjCAtDefsFieldDecl();
  C.Deallocate((void *)this);
}

//===----------------------------------------------------------------------===//
// ObjCProtocolDecl
//===----------------------------------------------------------------------===//

ObjCProtocolDecl *ObjCProtocolDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L,
                                           IdentifierInfo *Id) {
  return new (C) ObjCProtocolDecl(DC, L, Id);
}

void ObjCProtocolDecl::Destroy(ASTContext &C) {
  ReferencedProtocols.Destroy(C);
  ObjCContainerDecl::Destroy(C);
}

ObjCProtocolDecl *ObjCProtocolDecl::lookupProtocolNamed(IdentifierInfo *Name) {
  ObjCProtocolDecl *PDecl = this;

  if (Name == getIdentifier())
    return PDecl;

  for (protocol_iterator I = protocol_begin(), E = protocol_end(); I != E; ++I)
    if ((PDecl = (*I)->lookupProtocolNamed(Name)))
      return PDecl;

  return NULL;
}

// lookupMethod - Lookup a instance/class method in the protocol and protocols
// it inherited.
ObjCMethodDecl *ObjCProtocolDecl::lookupMethod(Selector Sel,
                                               bool isInstance) const {
  ObjCMethodDecl *MethodDecl = NULL;

  if ((MethodDecl = getMethod(Sel, isInstance)))
    return MethodDecl;

  for (protocol_iterator I = protocol_begin(), E = protocol_end(); I != E; ++I)
    if ((MethodDecl = (*I)->lookupMethod(Sel, isInstance)))
      return MethodDecl;
  return NULL;
}

//===----------------------------------------------------------------------===//
// ObjCClassDecl
//===----------------------------------------------------------------------===//

ObjCClassDecl::ObjCClassDecl(DeclContext *DC, SourceLocation L,
                             ObjCInterfaceDecl *const *Elts,
                             const SourceLocation *Locs,
                             unsigned nElts,
                             ASTContext &C)
  : Decl(ObjCClass, DC, L) {
  setClassList(C, Elts, Locs, nElts);
}

void ObjCClassDecl::setClassList(ASTContext &C, ObjCInterfaceDecl*const*List,
                                 const SourceLocation *Locs, unsigned Num) {
  ForwardDecls = (ObjCClassRef*) C.Allocate(sizeof(ObjCClassRef)*Num,
                                            llvm::alignof<ObjCClassRef>());
  for (unsigned i = 0; i < Num; ++i)
    new (&ForwardDecls[i]) ObjCClassRef(List[i], Locs[i]);
  
  NumDecls = Num;
}

ObjCClassDecl *ObjCClassDecl::Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation L,
                                     ObjCInterfaceDecl *const *Elts,
                                     const SourceLocation *Locs,
                                     unsigned nElts) {
  return new (C) ObjCClassDecl(DC, L, Elts, Locs, nElts, C);
}

void ObjCClassDecl::Destroy(ASTContext &C) {
  // ObjCInterfaceDecls registered with a DeclContext will get destroyed
  // when the DeclContext is destroyed.  For those created only by a forward
  // declaration, the first @class that created the ObjCInterfaceDecl gets
  // to destroy it.
  // FIXME: Note that this ownership role is very brittle; a better
  // polict is surely need in the future.
  for (iterator I = begin(), E = end(); I !=E ; ++I) {
    ObjCInterfaceDecl *ID = I->getInterface();
    if (ID->isForwardDecl() && ID->getLocStart() == getLocStart())
      ID->Destroy(C);
  }
  
  C.Deallocate(ForwardDecls);
  Decl::Destroy(C);
}

SourceRange ObjCClassDecl::getSourceRange() const {
  // FIXME: We should include the semicolon
  assert(NumDecls);
  return SourceRange(getLocation(), ForwardDecls[NumDecls-1].getLocation());
}

//===----------------------------------------------------------------------===//
// ObjCForwardProtocolDecl
//===----------------------------------------------------------------------===//

ObjCForwardProtocolDecl::
ObjCForwardProtocolDecl(DeclContext *DC, SourceLocation L,
                        ObjCProtocolDecl *const *Elts, unsigned nElts,
                        const SourceLocation *Locs, ASTContext &C)
: Decl(ObjCForwardProtocol, DC, L) {
  ReferencedProtocols.set(Elts, nElts, Locs, C);
}


ObjCForwardProtocolDecl *
ObjCForwardProtocolDecl::Create(ASTContext &C, DeclContext *DC,
                                SourceLocation L,
                                ObjCProtocolDecl *const *Elts,
                                unsigned NumElts,
                                const SourceLocation *Locs) {
  return new (C) ObjCForwardProtocolDecl(DC, L, Elts, NumElts, Locs, C);
}

void ObjCForwardProtocolDecl::Destroy(ASTContext &C) {
  ReferencedProtocols.Destroy(C);
  Decl::Destroy(C);
}

//===----------------------------------------------------------------------===//
// ObjCCategoryDecl
//===----------------------------------------------------------------------===//

ObjCCategoryDecl *ObjCCategoryDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L,
                                           IdentifierInfo *Id) {
  return new (C) ObjCCategoryDecl(DC, L, Id);
}

ObjCCategoryImplDecl *ObjCCategoryDecl::getImplementation() const {
  return getASTContext().getObjCImplementation(
                                           const_cast<ObjCCategoryDecl*>(this));
}

void ObjCCategoryDecl::setImplementation(ObjCCategoryImplDecl *ImplD) {
  getASTContext().setObjCImplementation(this, ImplD);
}


//===----------------------------------------------------------------------===//
// ObjCCategoryImplDecl
//===----------------------------------------------------------------------===//

ObjCCategoryImplDecl *
ObjCCategoryImplDecl::Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L,IdentifierInfo *Id,
                             ObjCInterfaceDecl *ClassInterface) {
  return new (C) ObjCCategoryImplDecl(DC, L, Id, ClassInterface);
}

ObjCCategoryDecl *ObjCCategoryImplDecl::getCategoryDecl() const {
  return getClassInterface()->FindCategoryDeclaration(getIdentifier());
}


void ObjCImplDecl::addPropertyImplementation(ObjCPropertyImplDecl *property) {
  // FIXME: The context should be correct before we get here.
  property->setLexicalDeclContext(this);
  addDecl(property);
}

void ObjCImplDecl::setClassInterface(ObjCInterfaceDecl *IFace) {
  ASTContext &Ctx = getASTContext();

  if (ObjCImplementationDecl *ImplD
        = dyn_cast_or_null<ObjCImplementationDecl>(this)) {
    if (IFace)
      Ctx.setObjCImplementation(IFace, ImplD);

  } else if (ObjCCategoryImplDecl *ImplD =
             dyn_cast_or_null<ObjCCategoryImplDecl>(this)) {
    if (ObjCCategoryDecl *CD = IFace->FindCategoryDeclaration(getIdentifier()))
      Ctx.setObjCImplementation(CD, ImplD);
  }

  ClassInterface = IFace;
}

/// FindPropertyImplIvarDecl - This method lookup the ivar in the list of
/// properties implemented in this category @implementation block and returns
/// the implemented property that uses it.
///
ObjCPropertyImplDecl *ObjCImplDecl::
FindPropertyImplIvarDecl(IdentifierInfo *ivarId) const {
  for (propimpl_iterator i = propimpl_begin(), e = propimpl_end(); i != e; ++i){
    ObjCPropertyImplDecl *PID = *i;
    if (PID->getPropertyIvarDecl() &&
        PID->getPropertyIvarDecl()->getIdentifier() == ivarId)
      return PID;
  }
  return 0;
}

/// FindPropertyImplDecl - This method looks up a previous ObjCPropertyImplDecl
/// added to the list of those properties @synthesized/@dynamic in this
/// category @implementation block.
///
ObjCPropertyImplDecl *ObjCImplDecl::
FindPropertyImplDecl(IdentifierInfo *Id) const {
  for (propimpl_iterator i = propimpl_begin(), e = propimpl_end(); i != e; ++i){
    ObjCPropertyImplDecl *PID = *i;
    if (PID->getPropertyDecl()->getIdentifier() == Id)
      return PID;
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// ObjCImplementationDecl
//===----------------------------------------------------------------------===//

ObjCImplementationDecl *
ObjCImplementationDecl::Create(ASTContext &C, DeclContext *DC,
                               SourceLocation L,
                               ObjCInterfaceDecl *ClassInterface,
                               ObjCInterfaceDecl *SuperDecl) {
  return new (C) ObjCImplementationDecl(DC, L, ClassInterface, SuperDecl);
}

//===----------------------------------------------------------------------===//
// ObjCCompatibleAliasDecl
//===----------------------------------------------------------------------===//

ObjCCompatibleAliasDecl *
ObjCCompatibleAliasDecl::Create(ASTContext &C, DeclContext *DC,
                                SourceLocation L,
                                IdentifierInfo *Id,
                                ObjCInterfaceDecl* AliasedClass) {
  return new (C) ObjCCompatibleAliasDecl(DC, L, Id, AliasedClass);
}

//===----------------------------------------------------------------------===//
// ObjCPropertyDecl
//===----------------------------------------------------------------------===//

ObjCPropertyDecl *ObjCPropertyDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L,
                                           IdentifierInfo *Id,
                                           QualType T,
                                           PropertyControl propControl) {
  return new (C) ObjCPropertyDecl(DC, L, Id, T);
}


//===----------------------------------------------------------------------===//
// ObjCPropertyImplDecl
//===----------------------------------------------------------------------===//

ObjCPropertyImplDecl *ObjCPropertyImplDecl::Create(ASTContext &C,
                                                   DeclContext *DC,
                                                   SourceLocation atLoc,
                                                   SourceLocation L,
                                                   ObjCPropertyDecl *property,
                                                   Kind PK,
                                                   ObjCIvarDecl *ivar) {
  return new (C) ObjCPropertyImplDecl(DC, atLoc, L, property, PK, ivar);
}


