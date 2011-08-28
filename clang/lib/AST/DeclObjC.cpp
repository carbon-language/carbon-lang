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

void ObjCListBase::set(void *const* InList, unsigned Elts, ASTContext &Ctx) {
  List = 0;
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

ObjCPropertyDecl *
ObjCPropertyDecl::findPropertyDecl(const DeclContext *DC,
                                   IdentifierInfo *propertyID) {

  DeclContext::lookup_const_iterator I, E;
  llvm::tie(I, E) = DC->lookup(propertyID);
  for ( ; I != E; ++I)
    if (ObjCPropertyDecl *PD = dyn_cast<ObjCPropertyDecl>(*I))
      return PD;

  return 0;
}

/// FindPropertyDeclaration - Finds declaration of the property given its name
/// in 'PropertyId' and returns it. It returns 0, if not found.
ObjCPropertyDecl *
ObjCContainerDecl::FindPropertyDeclaration(IdentifierInfo *PropertyId) const {

  if (ObjCPropertyDecl *PD =
        ObjCPropertyDecl::findPropertyDecl(cast<DeclContext>(this), PropertyId))
    return PD;

  switch (getKind()) {
    default:
      break;
    case Decl::ObjCProtocol: {
      const ObjCProtocolDecl *PID = cast<ObjCProtocolDecl>(this);
      for (ObjCProtocolDecl::protocol_iterator I = PID->protocol_begin(),
           E = PID->protocol_end(); I != E; ++I)
        if (ObjCPropertyDecl *P = (*I)->FindPropertyDeclaration(PropertyId))
          return P;
      break;
    }
    case Decl::ObjCInterface: {
      const ObjCInterfaceDecl *OID = cast<ObjCInterfaceDecl>(this);
      // Look through categories.
      for (ObjCCategoryDecl *Cat = OID->getCategoryList();
           Cat; Cat = Cat->getNextClassCategory())
        if (!Cat->IsClassExtension())
          if (ObjCPropertyDecl *P = Cat->FindPropertyDeclaration(PropertyId))
            return P;

      // Look through protocols.
      for (ObjCInterfaceDecl::all_protocol_iterator
            I = OID->all_referenced_protocol_begin(),
            E = OID->all_referenced_protocol_end(); I != E; ++I)
        if (ObjCPropertyDecl *P = (*I)->FindPropertyDeclaration(PropertyId))
          return P;

      // Finally, check the super class.
      if (const ObjCInterfaceDecl *superClass = OID->getSuperClass())
        return superClass->FindPropertyDeclaration(PropertyId);
      break;
    }
    case Decl::ObjCCategory: {
      const ObjCCategoryDecl *OCD = cast<ObjCCategoryDecl>(this);
      // Look through protocols.
      if (!OCD->IsClassExtension())
        for (ObjCCategoryDecl::protocol_iterator
              I = OCD->protocol_begin(), E = OCD->protocol_end(); I != E; ++I)
        if (ObjCPropertyDecl *P = (*I)->FindPropertyDeclaration(PropertyId))
          return P;

      break;
    }
  }
  return 0;
}

/// FindPropertyVisibleInPrimaryClass - Finds declaration of the property
/// with name 'PropertyId' in the primary class; including those in protocols
/// (direct or indirect) used by the primary class.
///
ObjCPropertyDecl *
ObjCInterfaceDecl::FindPropertyVisibleInPrimaryClass(
                                            IdentifierInfo *PropertyId) const {
  if (ExternallyCompleted)
    LoadExternalDefinition();

  if (ObjCPropertyDecl *PD =
      ObjCPropertyDecl::findPropertyDecl(cast<DeclContext>(this), PropertyId))
    return PD;

  // Look through protocols.
  for (ObjCInterfaceDecl::all_protocol_iterator
        I = all_referenced_protocol_begin(),
        E = all_referenced_protocol_end(); I != E; ++I)
    if (ObjCPropertyDecl *P = (*I)->FindPropertyDeclaration(PropertyId))
      return P;

  return 0;
}

void ObjCInterfaceDecl::mergeClassExtensionProtocolList(
                              ObjCProtocolDecl *const* ExtList, unsigned ExtNum,
                              ASTContext &C)
{
  if (ExternallyCompleted)
    LoadExternalDefinition();

  if (AllReferencedProtocols.empty() && ReferencedProtocols.empty()) {
    AllReferencedProtocols.set(ExtList, ExtNum, C);
    return;
  }
  
  // Check for duplicate protocol in class's protocol list.
  // This is O(n*m). But it is extremely rare and number of protocols in
  // class or its extension are very few.
  SmallVector<ObjCProtocolDecl*, 8> ProtocolRefs;
  for (unsigned i = 0; i < ExtNum; i++) {
    bool protocolExists = false;
    ObjCProtocolDecl *ProtoInExtension = ExtList[i];
    for (all_protocol_iterator
          p = all_referenced_protocol_begin(),
          e = all_referenced_protocol_end(); p != e; ++p) {
      ObjCProtocolDecl *Proto = (*p);
      if (C.ProtocolCompatibleWithProtocol(ProtoInExtension, Proto)) {
        protocolExists = true;
        break;
      }      
    }
    // Do we want to warn on a protocol in extension class which
    // already exist in the class? Probably not.
    if (!protocolExists)
      ProtocolRefs.push_back(ProtoInExtension);
  }

  if (ProtocolRefs.empty())
    return;

  // Merge ProtocolRefs into class's protocol list;
  for (all_protocol_iterator p = all_referenced_protocol_begin(), 
        e = all_referenced_protocol_end(); p != e; ++p) {
    ProtocolRefs.push_back(*p);
  }

  AllReferencedProtocols.set(ProtocolRefs.data(), ProtocolRefs.size(), C);
}

/// getFirstClassExtension - Find first class extension of the given class.
ObjCCategoryDecl* ObjCInterfaceDecl::getFirstClassExtension() const {
  for (ObjCCategoryDecl *CDecl = getCategoryList(); CDecl;
       CDecl = CDecl->getNextClassCategory())
    if (CDecl->IsClassExtension())
      return CDecl;
  return 0;
}

/// getNextClassCategory - Find next class extension in list of categories.
const ObjCCategoryDecl* ObjCCategoryDecl::getNextClassExtension() const {
  for (const ObjCCategoryDecl *CDecl = getNextClassCategory(); CDecl; 
        CDecl = CDecl->getNextClassCategory())
    if (CDecl->IsClassExtension())
      return CDecl;
  return 0;
}

ObjCIvarDecl *ObjCInterfaceDecl::lookupInstanceVariable(IdentifierInfo *ID,
                                              ObjCInterfaceDecl *&clsDeclared) {
  ObjCInterfaceDecl* ClassDecl = this;
  while (ClassDecl != NULL) {
    if (ObjCIvarDecl *I = ClassDecl->getIvarDecl(ID)) {
      clsDeclared = ClassDecl;
      return I;
    }
    for (const ObjCCategoryDecl *CDecl = ClassDecl->getFirstClassExtension();
         CDecl; CDecl = CDecl->getNextClassExtension()) {
      if (ObjCIvarDecl *I = CDecl->getIvarDecl(ID)) {
        clsDeclared = ClassDecl;
        return I;
      }
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

  if (ExternallyCompleted)
    LoadExternalDefinition();

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

ObjCMethodDecl *ObjCInterfaceDecl::lookupPrivateMethod(
                                   const Selector &Sel,
                                   bool Instance) {
  ObjCMethodDecl *Method = 0;
  if (ObjCImplementationDecl *ImpDecl = getImplementation())
    Method = Instance ? ImpDecl->getInstanceMethod(Sel) 
                      : ImpDecl->getClassMethod(Sel);
  
  if (!Method && getSuperClass())
    return getSuperClass()->lookupPrivateMethod(Sel, Instance);
  return Method;
}

//===----------------------------------------------------------------------===//
// ObjCMethodDecl
//===----------------------------------------------------------------------===//

ObjCMethodDecl *ObjCMethodDecl::Create(ASTContext &C,
                                       SourceLocation beginLoc,
                                       SourceLocation endLoc,
                                       Selector SelInfo, QualType T,
                                       TypeSourceInfo *ResultTInfo,
                                       DeclContext *contextDecl,
                                       bool isInstance,
                                       bool isVariadic,
                                       bool isSynthesized,
                                       bool isImplicitlyDeclared,
                                       bool isDefined,
                                       ImplementationControl impControl,
                                       bool HasRelatedResultType,
                                       unsigned numSelectorArgs) {
  return new (C) ObjCMethodDecl(beginLoc, endLoc,
                                SelInfo, T, ResultTInfo, contextDecl,
                                isInstance,
                                isVariadic, isSynthesized, isImplicitlyDeclared,
                                isDefined,
                                impControl,
                                HasRelatedResultType,
                                numSelectorArgs);
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

ObjCMethodFamily ObjCMethodDecl::getMethodFamily() const {
  ObjCMethodFamily family = static_cast<ObjCMethodFamily>(Family);
  if (family != static_cast<unsigned>(InvalidObjCMethodFamily))
    return family;

  // Check for an explicit attribute.
  if (const ObjCMethodFamilyAttr *attr = getAttr<ObjCMethodFamilyAttr>()) {
    // The unfortunate necessity of mapping between enums here is due
    // to the attributes framework.
    switch (attr->getFamily()) {
    case ObjCMethodFamilyAttr::OMF_None: family = OMF_None; break;
    case ObjCMethodFamilyAttr::OMF_alloc: family = OMF_alloc; break;
    case ObjCMethodFamilyAttr::OMF_copy: family = OMF_copy; break;
    case ObjCMethodFamilyAttr::OMF_init: family = OMF_init; break;
    case ObjCMethodFamilyAttr::OMF_mutableCopy: family = OMF_mutableCopy; break;
    case ObjCMethodFamilyAttr::OMF_new: family = OMF_new; break;
    }
    Family = static_cast<unsigned>(family);
    return family;
  }

  family = getSelector().getMethodFamily();
  switch (family) {
  case OMF_None: break;

  // init only has a conventional meaning for an instance method, and
  // it has to return an object.
  case OMF_init:
    if (!isInstanceMethod() || !getResultType()->isObjCObjectPointerType())
      family = OMF_None;
    break;

  // alloc/copy/new have a conventional meaning for both class and
  // instance methods, but they require an object return.
  case OMF_alloc:
  case OMF_copy:
  case OMF_mutableCopy:
  case OMF_new:
    if (!getResultType()->isObjCObjectPointerType())
      family = OMF_None;
    break;

  // These selectors have a conventional meaning only for instance methods.
  case OMF_dealloc:
  case OMF_finalize:
  case OMF_retain:
  case OMF_release:
  case OMF_autorelease:
  case OMF_retainCount:
  case OMF_self:
    if (!isInstanceMethod())
      family = OMF_None;
    break;
      
  case OMF_performSelector:
    if (!isInstanceMethod() ||
        !getResultType()->isObjCIdType())
      family = OMF_None;
    else {
      unsigned noParams = param_size();
      if (noParams < 1 || noParams > 3)
        family = OMF_None;
      else {
        ObjCMethodDecl::arg_type_iterator it = arg_type_begin();
        QualType ArgT = (*it);
        if (!ArgT->isObjCSelType()) {
          family = OMF_None;
          break;
        }
        while (--noParams) {
          it++;
          ArgT = (*it);
          if (!ArgT->isObjCIdType()) {
            family = OMF_None;
            break;
          }
        }
      }
    }
    break;
      
  }

  // Cache the result.
  Family = static_cast<unsigned>(family);
  return family;
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

  bool selfIsPseudoStrong = false;
  bool selfIsConsumed = false;
  if (isInstanceMethod() && Context.getLangOptions().ObjCAutoRefCount) {
    selfIsConsumed = hasAttr<NSConsumesSelfAttr>();

    // 'self' is always __strong.  It's actually pseudo-strong except
    // in init methods, though.
    Qualifiers qs;
    qs.setObjCLifetime(Qualifiers::OCL_Strong);
    selfTy = Context.getQualifiedType(selfTy, qs);

    // In addition, 'self' is const unless this is an init method.
    if (getMethodFamily() != OMF_init) {
      selfTy = selfTy.withConst();
      selfIsPseudoStrong = true;
    }
  }

  ImplicitParamDecl *self
    = ImplicitParamDecl::Create(Context, this, SourceLocation(),
                                &Context.Idents.get("self"), selfTy);
  setSelfDecl(self);

  if (selfIsConsumed)
    self->addAttr(new (Context) NSConsumedAttr(SourceLocation(), Context));

  if (selfIsPseudoStrong)
    self->setARCPseudoStrong(true);

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
    CategoryList(0), IvarList(0), 
    ForwardDecl(FD), InternalInterface(isInternal), ExternallyCompleted(false),
    ClassLoc(CLoc) {
}

void ObjCInterfaceDecl::LoadExternalDefinition() const {
  assert(ExternallyCompleted && "Class is not externally completed");
  ExternallyCompleted = false;
  getASTContext().getExternalSource()->CompleteType(
                                        const_cast<ObjCInterfaceDecl *>(this));
}

void ObjCInterfaceDecl::setExternallyCompleted() {
  assert(getASTContext().getExternalSource() && 
         "Class can't be externally completed without an external source");
  assert(!ForwardDecl && 
         "Forward declarations can't be externally completed");
  ExternallyCompleted = true;
}

ObjCImplementationDecl *ObjCInterfaceDecl::getImplementation() const {
  if (ExternallyCompleted)
    LoadExternalDefinition();

  return getASTContext().getObjCImplementation(
                                          const_cast<ObjCInterfaceDecl*>(this));
}

void ObjCInterfaceDecl::setImplementation(ObjCImplementationDecl *ImplD) {
  getASTContext().setObjCImplementation(this, ImplD);
}

/// all_declared_ivar_begin - return first ivar declared in this class,
/// its extensions and its implementation. Lazily build the list on first
/// access.
ObjCIvarDecl *ObjCInterfaceDecl::all_declared_ivar_begin() {
  if (IvarList)
    return IvarList;
  
  ObjCIvarDecl *curIvar = 0;
  if (!ivar_empty()) {
    ObjCInterfaceDecl::ivar_iterator I = ivar_begin(), E = ivar_end();
    IvarList = (*I); ++I;
    for (curIvar = IvarList; I != E; curIvar = *I, ++I)
      curIvar->setNextIvar(*I);
  }
  
  for (const ObjCCategoryDecl *CDecl = getFirstClassExtension(); CDecl;
       CDecl = CDecl->getNextClassExtension()) {
    if (!CDecl->ivar_empty()) {
      ObjCCategoryDecl::ivar_iterator I = CDecl->ivar_begin(),
                                          E = CDecl->ivar_end();
      if (!IvarList) {
        IvarList = (*I); ++I;
        curIvar = IvarList;
      }
      for ( ;I != E; curIvar = *I, ++I)
        curIvar->setNextIvar(*I);
    }
  }
  
  if (ObjCImplementationDecl *ImplDecl = getImplementation()) {
    if (!ImplDecl->ivar_empty()) {
      ObjCImplementationDecl::ivar_iterator I = ImplDecl->ivar_begin(),
                                            E = ImplDecl->ivar_end();
      if (!IvarList) {
        IvarList = (*I); ++I;
        curIvar = IvarList;
      }
      for ( ;I != E; curIvar = *I, ++I)
        curIvar->setNextIvar(*I);
    }
  }
  return IvarList;
}

/// FindCategoryDeclaration - Finds category declaration in the list of
/// categories for this class and returns it. Name of the category is passed
/// in 'CategoryId'. If category not found, return 0;
///
ObjCCategoryDecl *
ObjCInterfaceDecl::FindCategoryDeclaration(IdentifierInfo *CategoryId) const {
  if (ExternallyCompleted)
    LoadExternalDefinition();

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

ObjCIvarDecl *ObjCIvarDecl::Create(ASTContext &C, ObjCContainerDecl *DC,
                                   SourceLocation StartLoc,
                                   SourceLocation IdLoc, IdentifierInfo *Id,
                                   QualType T, TypeSourceInfo *TInfo,
                                   AccessControl ac, Expr *BW,
                                   bool synthesized) {
  if (DC) {
    // Ivar's can only appear in interfaces, implementations (via synthesized
    // properties), and class extensions (via direct declaration, or synthesized
    // properties).
    //
    // FIXME: This should really be asserting this:
    //   (isa<ObjCCategoryDecl>(DC) &&
    //    cast<ObjCCategoryDecl>(DC)->IsClassExtension()))
    // but unfortunately we sometimes place ivars into non-class extension
    // categories on error. This breaks an AST invariant, and should not be
    // fixed.
    assert((isa<ObjCInterfaceDecl>(DC) || isa<ObjCImplementationDecl>(DC) ||
            isa<ObjCCategoryDecl>(DC)) &&
           "Invalid ivar decl context!");
    // Once a new ivar is created in any of class/class-extension/implementation
    // decl contexts, the previously built IvarList must be rebuilt.
    ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(DC);
    if (!ID) {
      if (ObjCImplementationDecl *IM = dyn_cast<ObjCImplementationDecl>(DC)) {
        ID = IM->getClassInterface();
        if (BW)
          IM->setHasSynthBitfield(true);
      } else {
        ObjCCategoryDecl *CD = cast<ObjCCategoryDecl>(DC);
        ID = CD->getClassInterface();
        if (BW)
          CD->setHasSynthBitfield(true);
      }
    }
    ID->setIvarList(0);
  }

  return new (C) ObjCIvarDecl(DC, StartLoc, IdLoc, Id, T, TInfo,
                              ac, BW, synthesized);
}

const ObjCInterfaceDecl *ObjCIvarDecl::getContainingInterface() const {
  const ObjCContainerDecl *DC = cast<ObjCContainerDecl>(getDeclContext());

  switch (DC->getKind()) {
  default:
  case ObjCCategoryImpl:
  case ObjCProtocol:
    assert(0 && "invalid ivar container!");
    return 0;

    // Ivars can only appear in class extension categories.
  case ObjCCategory: {
    const ObjCCategoryDecl *CD = cast<ObjCCategoryDecl>(DC);
    assert(CD->IsClassExtension() && "invalid container for ivar!");
    return CD->getClassInterface();
  }

  case ObjCImplementation:
    return cast<ObjCImplementationDecl>(DC)->getClassInterface();

  case ObjCInterface:
    return cast<ObjCInterfaceDecl>(DC);
  }
}

//===----------------------------------------------------------------------===//
// ObjCAtDefsFieldDecl
//===----------------------------------------------------------------------===//

ObjCAtDefsFieldDecl
*ObjCAtDefsFieldDecl::Create(ASTContext &C, DeclContext *DC,
                             SourceLocation StartLoc,  SourceLocation IdLoc,
                             IdentifierInfo *Id, QualType T, Expr *BW) {
  return new (C) ObjCAtDefsFieldDecl(DC, StartLoc, IdLoc, Id, T, BW);
}

//===----------------------------------------------------------------------===//
// ObjCProtocolDecl
//===----------------------------------------------------------------------===//

ObjCProtocolDecl *ObjCProtocolDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L,
                                           IdentifierInfo *Id) {
  return new (C) ObjCProtocolDecl(DC, L, Id);
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
                             ObjCInterfaceDecl *const Elt,
                             const SourceLocation Loc,
                             ASTContext &C)
  : Decl(ObjCClass, DC, L) {
  setClass(C, Elt, Loc);
}

ObjCClassDecl *ObjCClassDecl::Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation L,
                                     ObjCInterfaceDecl *const Elt,
                                     const SourceLocation Loc) {
  return new (C) ObjCClassDecl(DC, L, Elt, Loc, C);
}

void ObjCClassDecl::setClass(ASTContext &C, ObjCInterfaceDecl*const Cls,
                             const SourceLocation Loc) {
    
  ForwardDecl = (ObjCClassRef*) C.Allocate(sizeof(ObjCClassRef),
                                           llvm::alignOf<ObjCClassRef>());
  new (ForwardDecl) ObjCClassRef(Cls, Loc);
}
    
SourceRange ObjCClassDecl::getSourceRange() const {
  // FIXME: We should include the semicolon
  return SourceRange(getLocation(), ForwardDecl->getLocation());
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

//===----------------------------------------------------------------------===//
// ObjCCategoryDecl
//===----------------------------------------------------------------------===//

ObjCCategoryDecl *ObjCCategoryDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation AtLoc, 
                                           SourceLocation ClassNameLoc,
                                           SourceLocation CategoryNameLoc,
                                           IdentifierInfo *Id) {
  return new (C) ObjCCategoryDecl(DC, AtLoc, ClassNameLoc, CategoryNameLoc, Id);
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
  // The class interface might be NULL if we are working with invalid code.
  if (const ObjCInterfaceDecl *ID = getClassInterface())
    return ID->FindCategoryDeclaration(getIdentifier());
  return 0;
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

raw_ostream &clang::operator<<(raw_ostream &OS,
                                     const ObjCCategoryImplDecl *CID) {
  OS << CID->getName();
  return OS;
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

void ObjCImplementationDecl::setIvarInitializers(ASTContext &C,
                                             CXXCtorInitializer ** initializers,
                                                 unsigned numInitializers) {
  if (numInitializers > 0) {
    NumIvarInitializers = numInitializers;
    CXXCtorInitializer **ivarInitializers =
    new (C) CXXCtorInitializer*[NumIvarInitializers];
    memcpy(ivarInitializers, initializers,
           numInitializers * sizeof(CXXCtorInitializer*));
    IvarInitializers = ivarInitializers;
  }
}

raw_ostream &clang::operator<<(raw_ostream &OS,
                                     const ObjCImplementationDecl *ID) {
  OS << ID->getName();
  return OS;
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
                                           SourceLocation AtLoc,
                                           TypeSourceInfo *T,
                                           PropertyControl propControl) {
  return new (C) ObjCPropertyDecl(DC, L, Id, AtLoc, T);
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
                                                   ObjCIvarDecl *ivar,
                                                   SourceLocation ivarLoc) {
  return new (C) ObjCPropertyImplDecl(DC, atLoc, L, property, PK, ivar,
                                      ivarLoc);
}

SourceRange ObjCPropertyImplDecl::getSourceRange() const {
  SourceLocation EndLoc = getLocation();
  if (IvarLoc.isValid())
    EndLoc = IvarLoc;

  return SourceRange(AtLoc, EndLoc);
}
