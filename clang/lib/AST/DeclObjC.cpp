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
using namespace clang;

//===----------------------------------------------------------------------===//
// ObjC Decl Allocation/Deallocation Method Implementations
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

void ObjCMethodDecl::Destroy(ASTContext& C) {
  if (Body) Body->Destroy(C);
  if (SelfDecl) SelfDecl->Destroy(C);
  
  for (param_iterator I=param_begin(), E=param_end(); I!=E; ++I)
    if (*I) (*I)->Destroy(C);

  ParamInfo.clear();

  Decl::Destroy(C);
}


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
  for (ivar_iterator I=ivar_begin(), E=ivar_end(); I!=E; ++I)
    if (*I) (*I)->Destroy(C);
  
  IVars.clear();
  // FIXME: CategoryList?
  
  // FIXME: Because there is no clear ownership
  //  role between ObjCInterfaceDecls and the ObjCPropertyDecls that they
  //  reference, we destroy ObjCPropertyDecls in ~TranslationUnit.
  Decl::Destroy(C);
}


ObjCIvarDecl *ObjCIvarDecl::Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation L, IdentifierInfo *Id,
                                   QualType T, AccessControl ac, Expr *BW) {
  return new (C) ObjCIvarDecl(DC, L, Id, T, ac, BW);
}


ObjCAtDefsFieldDecl
*ObjCAtDefsFieldDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                             IdentifierInfo *Id, QualType T, Expr *BW) {
  return new (C) ObjCAtDefsFieldDecl(DC, L, Id, T, BW);
}

void ObjCAtDefsFieldDecl::Destroy(ASTContext& C) {
  this->~ObjCAtDefsFieldDecl();
  C.Deallocate((void *)this); 
}

ObjCProtocolDecl *ObjCProtocolDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L, 
                                           IdentifierInfo *Id) {
  return new (C) ObjCProtocolDecl(DC, L, Id);
}

void ObjCProtocolDecl::Destroy(ASTContext &C) {
  ReferencedProtocols.clear();
  ObjCContainerDecl::Destroy(C);
}




ObjCClassDecl *ObjCClassDecl::Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation L,
                                     ObjCInterfaceDecl *const *Elts,
                                     unsigned nElts) {
  return new (C) ObjCClassDecl(DC, L, Elts, nElts);
}

void ObjCClassDecl::Destroy(ASTContext &C) {
  
  // FIXME: There is no clear ownership policy now for referenced
  //  ObjCInterfaceDecls.  Some of them can be forward declarations that
  //  are never later defined (in which case the ObjCClassDecl owns them)
  //  or the ObjCInterfaceDecl later becomes a real definition later.  Ideally
  //  we should have separate objects for forward declarations and definitions,
  //  obviating this problem.  Because of this situation, referenced
  //  ObjCInterfaceDecls are destroyed in ~TranslationUnit.
  
  ForwardDecls.clear();
  Decl::Destroy(C);
}

ObjCForwardProtocolDecl *
ObjCForwardProtocolDecl::Create(ASTContext &C, DeclContext *DC,
                                SourceLocation L, 
                                ObjCProtocolDecl **Elts, unsigned NumElts) {
  return new (C) ObjCForwardProtocolDecl(DC, L, Elts, NumElts);
}

ObjCForwardProtocolDecl::
ObjCForwardProtocolDecl(DeclContext *DC, SourceLocation L,
                        ObjCProtocolDecl **Elts, unsigned nElts)
  : Decl(ObjCForwardProtocol, DC, L) { 
  NumReferencedProtocols = nElts;
  if (nElts) {
    ReferencedProtocols = new ObjCProtocolDecl*[nElts];
    memcpy(ReferencedProtocols, Elts, nElts*sizeof(ObjCProtocolDecl*));
  } else {
    ReferencedProtocols = 0;
  }
}

void ObjCForwardProtocolDecl::Destroy(ASTContext &C) {
  delete [] ReferencedProtocols;
  ReferencedProtocols = 0;
}

ObjCCategoryDecl *ObjCCategoryDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L,
                                           IdentifierInfo *Id) {
  return new (C) ObjCCategoryDecl(DC, L, Id);
}

ObjCCategoryImplDecl *
ObjCCategoryImplDecl::Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L,IdentifierInfo *Id,
                             ObjCInterfaceDecl *ClassInterface) {
  return new (C) ObjCCategoryImplDecl(DC, L, Id, ClassInterface);
}

ObjCImplementationDecl *
ObjCImplementationDecl::Create(ASTContext &C, DeclContext *DC, 
                               SourceLocation L,
                               ObjCInterfaceDecl *ClassInterface,
                               ObjCInterfaceDecl *SuperDecl) {
  return new (C) ObjCImplementationDecl(DC, L, ClassInterface, SuperDecl);
}

ObjCCompatibleAliasDecl *
ObjCCompatibleAliasDecl::Create(ASTContext &C, DeclContext *DC,
                                SourceLocation L,
                                IdentifierInfo *Id, 
                                ObjCInterfaceDecl* AliasedClass) {
  return new (C) ObjCCompatibleAliasDecl(DC, L, Id, AliasedClass);
}

ObjCPropertyDecl *ObjCPropertyDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L,
                                           IdentifierInfo *Id,
                                           QualType T,
                                           PropertyControl propControl) {
  return new (C) ObjCPropertyDecl(DC, L, Id, T);
}

//===----------------------------------------------------------------------===//
// Objective-C Decl Implementation
//===----------------------------------------------------------------------===//

void ObjCMethodDecl::createImplicitParams(ASTContext &Context, 
                                          const ObjCInterfaceDecl *OID) {
  QualType selfTy;
  if (isInstanceMethod()) {
    // There may be no interface context due to error in declaration
    // of the interface (which has been reported). Recover gracefully.
    if (OID) {
      selfTy = Context.getObjCInterfaceType(const_cast<ObjCInterfaceDecl *>(OID));
      selfTy = Context.getPointerType(selfTy);
    } else {
      selfTy = Context.getObjCIdType();
    }
  } else // we have a factory method.
    selfTy = Context.getObjCClassType();

  SelfDecl = ImplicitParamDecl::Create(Context, this, 
                                       SourceLocation(), 
                                       &Context.Idents.get("self"),
                                       selfTy);

  CmdDecl = ImplicitParamDecl::Create(Context, this, 
                                      SourceLocation(), 
                                      &Context.Idents.get("_cmd"), 
                                      Context.getObjCSelType());
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

/// lookupFieldDeclForIvar - looks up a field decl' in the laid out
/// storage which matches this 'ivar'.
///
FieldDecl *ObjCInterfaceDecl::lookupFieldDeclForIvar(ASTContext &Context, 
                                                     const ObjCIvarDecl *ivar) {
  const RecordDecl *RecordForDecl = Context.addRecordToClass(this);
  assert(RecordForDecl && "lookupFieldDeclForIvar no storage for class");
  DeclarationName Member = ivar->getDeclName();
  DeclContext::lookup_result Lookup = (const_cast< RecordDecl *>(RecordForDecl))
                                        ->lookup(Member);
  assert((Lookup.first != Lookup.second) && "field decl not found");
  FieldDecl *MemberDecl = dyn_cast<FieldDecl>(*Lookup.first);
  assert(MemberDecl && "field decl not found");
  return MemberDecl;
}

/// ObjCAddInstanceVariablesToClassImpl - Checks for correctness of Instance 
/// Variables (Ivars) relative to what declared in @implementation;s class. 
/// Ivars into ObjCImplementationDecl's fields.
///
void ObjCImplementationDecl::ObjCAddInstanceVariablesToClassImpl(
                               ObjCIvarDecl **ivars, unsigned numIvars) {
  NumIvars = numIvars;
  if (numIvars) {
    Ivars = new ObjCIvarDecl*[numIvars];
    memcpy(Ivars, ivars, numIvars*sizeof(ObjCIvarDecl*));
  }
}

// Get the local instance method declared in this interface.
// FIXME: handle overloading, instance & class methods can have the same name.
ObjCMethodDecl *ObjCContainerDecl::getInstanceMethod(Selector Sel) const {
  lookup_const_result MethodResult = lookup(Sel);
  if (MethodResult.first)
    return const_cast<ObjCMethodDecl*>(
             dyn_cast<ObjCMethodDecl>(*MethodResult.first));
  return 0;
}

// Get the local class method declared in this interface.
ObjCMethodDecl *ObjCContainerDecl::getClassMethod(Selector Sel) const {
  lookup_const_result MethodResult = lookup(Sel);
  if (MethodResult.first)
    return const_cast<ObjCMethodDecl*>(
             dyn_cast<ObjCMethodDecl>(*MethodResult.first));
  return 0;
}

unsigned ObjCContainerDecl::getNumInstanceMethods() const {
  unsigned sum = 0;
  for (instmeth_iterator I=instmeth_begin(), E=instmeth_end(); I != E; ++I)
    sum++;
  return sum;
}
unsigned ObjCContainerDecl::getNumClassMethods() const { 
  unsigned sum = 0;
  for (classmeth_iterator I=classmeth_begin(), E=classmeth_end(); I != E; ++I)
    sum++;
  return sum;
}
unsigned ObjCContainerDecl::getNumProperties() const { 
  unsigned sum = 0;
  for (prop_iterator I=prop_begin(), E=prop_end(); I != E; ++I)
    sum++;
  return sum;
}

/// FindPropertyDeclaration - Finds declaration of the property given its name
/// in 'PropertyId' and returns it. It returns 0, if not found.
/// FIXME: Convert to DeclContext lookup...
///
ObjCPropertyDecl *
ObjCContainerDecl::FindPropertyDeclaration(IdentifierInfo *PropertyId) const {
  for (prop_iterator I = prop_begin(), E = prop_end(); I != E; ++I) {
    ObjCPropertyDecl *property = *I;
    if (property->getIdentifier() == PropertyId)
      return property;
  }
  const ObjCProtocolDecl *PID = dyn_cast<ObjCProtocolDecl>(this);
  if (PID) {
    for (ObjCProtocolDecl::protocol_iterator P = PID->protocol_begin(), 
         E = PID->protocol_end(); 
         P != E; ++P)
      if (ObjCPropertyDecl *property = 
            (*P)->FindPropertyDeclaration(PropertyId))
        return property;
  }
  
  if (const ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(this)) {
    // Look through categories.
    for (ObjCCategoryDecl *Category = OID->getCategoryList();
         Category; Category = Category->getNextClassCategory()) {
      ObjCPropertyDecl *property = Category->FindPropertyDeclaration(PropertyId);
      if (property)
        return property;
    }
    // Look through protocols.
    for (ObjCInterfaceDecl::protocol_iterator I = OID->protocol_begin(),
         E = OID->protocol_end(); I != E; ++I) {
      ObjCProtocolDecl *Protocol = *I;
      ObjCPropertyDecl *property = Protocol->FindPropertyDeclaration(PropertyId);
      if (property)
        return property;
    }
    if (OID->getSuperClass())
      return OID->getSuperClass()->FindPropertyDeclaration(PropertyId);
  }
  else if (const ObjCCategoryDecl *OCD = dyn_cast<ObjCCategoryDecl>(this)) {
    // Look through protocols.
    for (ObjCInterfaceDecl::protocol_iterator I = OCD->protocol_begin(),
         E = OCD->protocol_end(); I != E; ++I) {
      ObjCProtocolDecl *Protocol = *I;
      ObjCPropertyDecl *property = Protocol->FindPropertyDeclaration(PropertyId);
      if (property)
        return property;
    }
  }
  return 0;
}

ObjCIvarDecl *ObjCInterfaceDecl::lookupInstanceVariable(
  IdentifierInfo *ID, ObjCInterfaceDecl *&clsDeclared) {
  ObjCInterfaceDecl* ClassDecl = this;
  while (ClassDecl != NULL) {
    for (ivar_iterator I = ClassDecl->ivar_begin(), E = ClassDecl->ivar_end();
         I != E; ++I) {
      if ((*I)->getIdentifier() == ID) {
        clsDeclared = ClassDecl;
        return *I;
      }
    }
    ClassDecl = ClassDecl->getSuperClass();
  }
  return NULL;
}

/// lookupInstanceMethod - This method returns an instance method by looking in
/// the class, its categories, and its super classes (using a linear search).
ObjCMethodDecl *ObjCInterfaceDecl::lookupInstanceMethod(Selector Sel) {
  ObjCInterfaceDecl* ClassDecl = this;
  ObjCMethodDecl *MethodDecl = 0;
  
  while (ClassDecl != NULL) {
    if ((MethodDecl = ClassDecl->getInstanceMethod(Sel)))
      return MethodDecl;
      
    // Didn't find one yet - look through protocols.
    const ObjCList<ObjCProtocolDecl> &Protocols =
      ClassDecl->getReferencedProtocols();
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
         E = Protocols.end(); I != E; ++I)
      if ((MethodDecl = (*I)->getInstanceMethod(Sel)))
        return MethodDecl;
    
    // Didn't find one yet - now look through categories.
    ObjCCategoryDecl *CatDecl = ClassDecl->getCategoryList();
    while (CatDecl) {
      if ((MethodDecl = CatDecl->getInstanceMethod(Sel)))
        return MethodDecl;
        
      // Didn't find one yet - look through protocols.
      const ObjCList<ObjCProtocolDecl> &Protocols =
        CatDecl->getReferencedProtocols();
      for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
           E = Protocols.end(); I != E; ++I)
        if ((MethodDecl = (*I)->getInstanceMethod(Sel)))
          return MethodDecl;
      CatDecl = CatDecl->getNextClassCategory();
    }
    ClassDecl = ClassDecl->getSuperClass();
  }
  return NULL;
}

// lookupClassMethod - This method returns a class method by looking in the
// class, its categories, and its super classes (using a linear search).
ObjCMethodDecl *ObjCInterfaceDecl::lookupClassMethod(Selector Sel) {
  ObjCInterfaceDecl* ClassDecl = this;
  ObjCMethodDecl *MethodDecl = 0;

  while (ClassDecl != NULL) {
    if ((MethodDecl = ClassDecl->getClassMethod(Sel)))
      return MethodDecl;

    // Didn't find one yet - look through protocols.
    for (ObjCInterfaceDecl::protocol_iterator I = ClassDecl->protocol_begin(),
         E = ClassDecl->protocol_end(); I != E; ++I)
      if ((MethodDecl = (*I)->getClassMethod(Sel)))
        return MethodDecl;
    
    // Didn't find one yet - now look through categories.
    ObjCCategoryDecl *CatDecl = ClassDecl->getCategoryList();
    while (CatDecl) {
      if ((MethodDecl = CatDecl->getClassMethod(Sel)))
        return MethodDecl;
      CatDecl = CatDecl->getNextClassCategory();
    }
    ClassDecl = ClassDecl->getSuperClass();
  }
  return NULL;
}

/// getInstanceMethod - This method returns an instance method by
/// looking in the class implementation. Unlike interfaces, we don't
/// look outside the implementation.
ObjCMethodDecl *ObjCImplementationDecl::getInstanceMethod(Selector Sel) const {
  for (instmeth_iterator I = instmeth_begin(), E = instmeth_end(); I != E; ++I)
    if ((*I)->getSelector() == Sel)
      return *I;
  return NULL;
}

/// getClassMethod - This method returns a class method by looking in
/// the class implementation. Unlike interfaces, we don't look outside
/// the implementation.
ObjCMethodDecl *ObjCImplementationDecl::getClassMethod(Selector Sel) const {
  for (classmeth_iterator I = classmeth_begin(), E = classmeth_end();
       I != E; ++I)
    if ((*I)->getSelector() == Sel)
      return *I;
  return NULL;
}

/// FindPropertyImplDecl - This method looks up a previous ObjCPropertyImplDecl
/// added to the list of those properties @synthesized/@dynamic in this
/// @implementation block.
///
ObjCPropertyImplDecl *ObjCImplementationDecl::FindPropertyImplDecl(IdentifierInfo *Id) const {
  for (propimpl_iterator i = propimpl_begin(), e = propimpl_end(); i != e; ++i) {
    ObjCPropertyImplDecl *PID = *i;
    if (PID->getPropertyDecl()->getIdentifier() == Id)
      return PID;
  }
  return 0;
}

/// FindPropertyImplIvarDecl - This method lookup the ivar in the list of
/// properties implemented in this @implementation block and returns the
/// implemented property that uses it.
///
ObjCPropertyImplDecl *ObjCImplementationDecl::FindPropertyImplIvarDecl(IdentifierInfo *ivarId) const {
  for (propimpl_iterator i = propimpl_begin(), e = propimpl_end(); i != e; ++i) {
    ObjCPropertyImplDecl *PID = *i;
    if (PID->getPropertyIvarDecl() &&
        PID->getPropertyIvarDecl()->getIdentifier() == ivarId)
      return PID;
  }
  return 0;
}

/// FindPropertyImplIvarDecl - This method lookup the ivar in the list of
/// properties implemented in this category @implementation block and returns
/// the implemented property that uses it.
///
ObjCPropertyImplDecl *ObjCCategoryImplDecl::
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
ObjCPropertyImplDecl *ObjCCategoryImplDecl::
FindPropertyImplDecl(IdentifierInfo *Id) const {
  for (propimpl_iterator i = propimpl_begin(), e = propimpl_end(); i != e; ++i){
    ObjCPropertyImplDecl *PID = *i;
    if (PID->getPropertyDecl()->getIdentifier() == Id)
      return PID;
  }
  return 0;
}

// lookupInstanceMethod - This method returns an instance method by looking in
// the class implementation. Unlike interfaces, we don't look outside the
// implementation.
ObjCMethodDecl *ObjCCategoryImplDecl::getInstanceMethod(Selector Sel) const {
  for (instmeth_iterator I = instmeth_begin(), E = instmeth_end(); I != E; ++I)
    if ((*I)->getSelector() == Sel)
      return *I;
  return NULL;
}

// lookupClassMethod - This method returns an instance method by looking in
// the class implementation. Unlike interfaces, we don't look outside the
// implementation.
ObjCMethodDecl *ObjCCategoryImplDecl::getClassMethod(Selector Sel) const {
  for (classmeth_iterator I = classmeth_begin(), E = classmeth_end();
       I != E; ++I)
    if ((*I)->getSelector() == Sel)
      return *I;
  return NULL;
}

// lookupInstanceMethod - Lookup a instance method in the protocol and protocols
// it inherited.
ObjCMethodDecl *ObjCProtocolDecl::lookupInstanceMethod(Selector Sel) {
  ObjCMethodDecl *MethodDecl = NULL;
  
  if ((MethodDecl = getInstanceMethod(Sel)))
    return MethodDecl;
    
  for (protocol_iterator I = protocol_begin(), E = protocol_end(); I != E; ++I)
    if ((MethodDecl = (*I)->lookupInstanceMethod(Sel)))
      return MethodDecl;
  return NULL;
}

// lookupInstanceMethod - Lookup a class method in the protocol and protocols
// it inherited.
ObjCMethodDecl *ObjCProtocolDecl::lookupClassMethod(Selector Sel) {
  ObjCMethodDecl *MethodDecl = NULL;

  if ((MethodDecl = getClassMethod(Sel)))
    return MethodDecl;
    
  for (protocol_iterator I = protocol_begin(), E = protocol_end(); I != E; ++I)
    if ((MethodDecl = (*I)->lookupClassMethod(Sel)))
      return MethodDecl;
  return NULL;
}

/// getSynthesizedMethodSize - Compute size of synthesized method name
/// as done be the rewrite.
///
unsigned ObjCMethodDecl::getSynthesizedMethodSize() const {
  // syntesized method name is a concatenation of -/+[class-name selector]
  // Get length of this name.
  unsigned length = 3;  // _I_ or _C_
  length += getClassInterface()->getNameAsString().size()+1; // extra for _
  if (const ObjCCategoryImplDecl *CID = 
      dyn_cast<ObjCCategoryImplDecl>(getDeclContext()))
    length += CID->getNameAsString().size()+1;
  length += getSelector().getAsString().size(); // selector name
  return length; 
}

ObjCInterfaceDecl *ObjCMethodDecl::getClassInterface() {
  if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(getDeclContext()))
    return ID;
  if (ObjCCategoryDecl *CD = dyn_cast<ObjCCategoryDecl>(getDeclContext()))
    return CD->getClassInterface();
  if (ObjCImplementationDecl *IMD = 
        dyn_cast<ObjCImplementationDecl>(getDeclContext()))
    return IMD->getClassInterface();
  if (ObjCCategoryImplDecl *CID = 
        dyn_cast<ObjCCategoryImplDecl>(getDeclContext()))
    return CID->getClassInterface();
  assert(false && "unknown method context");
  return 0;
}

ObjCPropertyImplDecl *ObjCPropertyImplDecl::Create(ASTContext &C,
                                                   DeclContext *DC,
                                                   SourceLocation atLoc,
                                                   SourceLocation L,
                                                   ObjCPropertyDecl *property,
                                                   Kind PK,
                                                   ObjCIvarDecl *ivar) {
  return new (C) ObjCPropertyImplDecl(DC, atLoc, L, property, PK, ivar);
}


