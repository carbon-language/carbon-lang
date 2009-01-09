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
  void *Mem = C.getAllocator().Allocate<ObjCMethodDecl>();
  return new (Mem) ObjCMethodDecl(beginLoc, endLoc,
                                  SelInfo, T, contextDecl,
                                  isInstance, 
                                  isVariadic, isSynthesized, impControl);
}

ObjCMethodDecl::~ObjCMethodDecl() {  
  delete [] ParamInfo;
}

void ObjCMethodDecl::Destroy(ASTContext& C) {
  if (Body) Body->Destroy(C);
  if (SelfDecl) SelfDecl->Destroy(C);
  
  for (param_iterator I=param_begin(), E=param_end(); I!=E; ++I)
    if (*I) (*I)->Destroy(C);
  
  Decl::Destroy(C);
}

ObjCInterfaceDecl *ObjCInterfaceDecl::Create(ASTContext &C,
                                             DeclContext *DC,
                                             SourceLocation atLoc,
                                             IdentifierInfo *Id, 
                                             SourceLocation ClassLoc,
                                             bool ForwardDecl, bool isInternal){
  void *Mem = C.getAllocator().Allocate<ObjCInterfaceDecl>();
  return new (Mem) ObjCInterfaceDecl(DC, atLoc, Id, ClassLoc, ForwardDecl,
                                     isInternal);
}

ObjCContainerDecl::~ObjCContainerDecl() {
  delete [] PropertyDecl;
}

ObjCInterfaceDecl::~ObjCInterfaceDecl() {
  delete [] Ivars;
  // FIXME: CategoryList?
}

void ObjCInterfaceDecl::Destroy(ASTContext& C) {  
  for (ivar_iterator I=ivar_begin(), E=ivar_end(); I!=E; ++I)
    if (*I) (*I)->Destroy(C);
  
  for (method_iterator I=meth_begin(), E=meth_end(); I!=E; ++I)
    if (*I) const_cast<ObjCMethodDecl*>((*I))->Destroy(C);

  // FIXME: Because there is no clear ownership
  //  role between ObjCInterfaceDecls and the ObjCPropertyDecls that they
  //  reference, we destroy ObjCPropertyDecls in ~TranslationUnit.

  Decl::Destroy(C);
}


ObjCIvarDecl *ObjCIvarDecl::Create(ASTContext &C, SourceLocation L,
                                   IdentifierInfo *Id, QualType T, 
                                   AccessControl ac, Expr *BW) {
  void *Mem = C.getAllocator().Allocate<ObjCIvarDecl>();
  return new (Mem) ObjCIvarDecl(L, Id, T, ac, BW);
}


ObjCAtDefsFieldDecl
*ObjCAtDefsFieldDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                             IdentifierInfo *Id, QualType T, Expr *BW) {
  void *Mem = C.getAllocator().Allocate<ObjCAtDefsFieldDecl>();
  return new (Mem) ObjCAtDefsFieldDecl(DC, L, Id, T, BW);
}

void ObjCAtDefsFieldDecl::Destroy(ASTContext& C) {
  this->~ObjCAtDefsFieldDecl();
  C.getAllocator().Deallocate((void *)this); 
}

ObjCProtocolDecl *ObjCProtocolDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L, 
                                           IdentifierInfo *Id) {
  void *Mem = C.getAllocator().Allocate<ObjCProtocolDecl>();
  return new (Mem) ObjCProtocolDecl(DC, L, Id);
}

ObjCProtocolDecl::~ObjCProtocolDecl() {
  delete [] PropertyDecl;
}

void ObjCProtocolDecl::Destroy(ASTContext& C) {
  
  // Referenced Protocols are not owned, so don't Destroy them.
  
  for (method_iterator I=meth_begin(), E=meth_end(); I!=E; ++I)
    if (*I) const_cast<ObjCMethodDecl*>((*I))->Destroy(C);
  
  // FIXME: Because there is no clear ownership
  //  role between ObjCProtocolDecls and the ObjCPropertyDecls that they
  //  reference, we destroy ObjCPropertyDecls in ~TranslationUnit.
  
  Decl::Destroy(C);
}


ObjCClassDecl *ObjCClassDecl::Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation L,
                                     ObjCInterfaceDecl **Elts, unsigned nElts) {
  void *Mem = C.getAllocator().Allocate<ObjCClassDecl>();
  return new (Mem) ObjCClassDecl(DC, L, Elts, nElts);
}

ObjCClassDecl::~ObjCClassDecl() {
  delete [] ForwardDecls;
}

void ObjCClassDecl::Destroy(ASTContext& C) {
  
  // FIXME: There is no clear ownership policy now for referenced
  //  ObjCInterfaceDecls.  Some of them can be forward declarations that
  //  are never later defined (in which case the ObjCClassDecl owns them)
  //  or the ObjCInterfaceDecl later becomes a real definition later.  Ideally
  //  we should have separate objects for forward declarations and definitions,
  //  obviating this problem.  Because of this situation, referenced
  //  ObjCInterfaceDecls are destroyed in ~TranslationUnit.
  
  Decl::Destroy(C);
}

ObjCForwardProtocolDecl *
ObjCForwardProtocolDecl::Create(ASTContext &C, DeclContext *DC,
                                SourceLocation L, 
                                ObjCProtocolDecl **Elts, unsigned NumElts) {
  void *Mem = C.getAllocator().Allocate<ObjCForwardProtocolDecl>();
  return new (Mem) ObjCForwardProtocolDecl(DC, L, Elts, NumElts);
}

ObjCForwardProtocolDecl::~ObjCForwardProtocolDecl() {
  delete [] ReferencedProtocols;
}

ObjCCategoryDecl *ObjCCategoryDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L,
                                           IdentifierInfo *Id) {
  void *Mem = C.getAllocator().Allocate<ObjCCategoryDecl>();
  return new (Mem) ObjCCategoryDecl(DC, L, Id);
}

ObjCCategoryImplDecl *
ObjCCategoryImplDecl::Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L,IdentifierInfo *Id,
                             ObjCInterfaceDecl *ClassInterface) {
  void *Mem = C.getAllocator().Allocate<ObjCCategoryImplDecl>();
  return new (Mem) ObjCCategoryImplDecl(DC, L, Id, ClassInterface);
}

ObjCImplementationDecl *
ObjCImplementationDecl::Create(ASTContext &C, DeclContext *DC, 
                               SourceLocation L,
                               IdentifierInfo *Id,
                               ObjCInterfaceDecl *ClassInterface,
                               ObjCInterfaceDecl *SuperDecl) {
  void *Mem = C.getAllocator().Allocate<ObjCImplementationDecl>();
  return new (Mem) ObjCImplementationDecl(DC, L, Id, ClassInterface, SuperDecl);
}

ObjCCompatibleAliasDecl *
ObjCCompatibleAliasDecl::Create(ASTContext &C, DeclContext *DC,
                                SourceLocation L,
                                IdentifierInfo *Id, 
                                ObjCInterfaceDecl* AliasedClass) {
  void *Mem = C.getAllocator().Allocate<ObjCCompatibleAliasDecl>();
  return new (Mem) ObjCCompatibleAliasDecl(DC, L, Id, AliasedClass);
}

ObjCPropertyDecl *ObjCPropertyDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L,
                                           IdentifierInfo *Id,
                                           QualType T,
                                           PropertyControl propControl) {
  void *Mem = C.getAllocator().Allocate<ObjCPropertyDecl>();
  return new (Mem) ObjCPropertyDecl(DC, L, Id, T);
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
                                       selfTy, 0);

  CmdDecl = ImplicitParamDecl::Create(Context, this, 
                                      SourceLocation(), 
                                      &Context.Idents.get("_cmd"), 
                                      Context.getObjCSelType(), 0);
}

void ObjCMethodDecl::setMethodParams(ParmVarDecl **NewParamInfo,
                                     unsigned NumParams) {
  assert(ParamInfo == 0 && "Already has param info!");

  // Zero params -> null pointer.
  if (NumParams) {
    ParamInfo = new ParmVarDecl*[NumParams];
    memcpy(ParamInfo, NewParamInfo, sizeof(ParmVarDecl*)*NumParams);
    NumMethodParams = NumParams;
  }
}

/// isPropertyReadonly - Return true if property is readonly, by searching
/// for the property in the class and in its categories.
///
bool ObjCInterfaceDecl::isPropertyReadonly(ObjCPropertyDecl *PDecl) const
{
  if (!PDecl->isReadOnly())
    return false;

  // Main class has the property as 'readonly'. Must search
  // through the category list to see if the property's 
  // attribute has been over-ridden to 'readwrite'.
  for (ObjCCategoryDecl *Category = getCategoryList();
       Category; Category = Category->getNextClassCategory()) {
    ObjCPropertyDecl *P = 
      Category->FindPropertyDeclaration(PDecl->getIdentifier());
    if (P && !P->isReadOnly())
      return false;
  }

  return true;
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

/// FindIvarDeclaration - Find an Ivar declaration in this class given its
/// name in 'IvarId'. On failure to find, return 0;
///
ObjCIvarDecl *
  ObjCInterfaceDecl::FindIvarDeclaration(IdentifierInfo *IvarId) const {
  for (ObjCInterfaceDecl::ivar_iterator IVI = ivar_begin(), 
       IVE = ivar_end(); IVI != IVE; ++IVI) {
    ObjCIvarDecl* Ivar = (*IVI);
    if (Ivar->getIdentifier() == IvarId)
      return Ivar;
  }
  if (getSuperClass())
    return getSuperClass()->FindIvarDeclaration(IvarId);
  return 0;
}

/// ObjCAddInstanceVariablesToClass - Inserts instance variables
/// into ObjCInterfaceDecl's fields.
///
void ObjCInterfaceDecl::addInstanceVariablesToClass(ObjCIvarDecl **ivars,
                                                    unsigned numIvars,
                                                    SourceLocation RBrac) {
  NumIvars = numIvars;
  if (numIvars) {
    Ivars = new ObjCIvarDecl*[numIvars];
    memcpy(Ivars, ivars, numIvars*sizeof(ObjCIvarDecl*));
  }
  setLocEnd(RBrac);
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

/// mergeProperties - Adds properties to the end of list of current properties
/// for this class.

void ObjCContainerDecl::mergeProperties(ObjCPropertyDecl **Properties, 
                                        unsigned NumNewProperties) {
  if (NumNewProperties == 0) return;
  
  if (PropertyDecl) {
    ObjCPropertyDecl **newPropertyDecl =  
      new ObjCPropertyDecl*[NumNewProperties + NumPropertyDecl];
    ObjCPropertyDecl **buf = newPropertyDecl;
    // put back original properties in buffer.
    memcpy(buf, PropertyDecl, NumPropertyDecl*sizeof(ObjCPropertyDecl*));
    // Add new properties to this buffer.
    memcpy(buf+NumPropertyDecl, Properties, 
           NumNewProperties*sizeof(ObjCPropertyDecl*));
    delete[] PropertyDecl;
    PropertyDecl = newPropertyDecl;
    NumPropertyDecl += NumNewProperties;
  }
  else {
    addProperties(Properties, NumNewProperties);
  }
}

/// addProperties - Insert property declaration AST nodes into
/// ObjCContainerDecl's PropertyDecl field.
///
void ObjCContainerDecl::addProperties(ObjCPropertyDecl **Properties, 
                                      unsigned NumProperties) {
  if (NumProperties == 0) return;
  
  NumPropertyDecl = NumProperties;
  PropertyDecl = new ObjCPropertyDecl*[NumProperties];
  memcpy(PropertyDecl, Properties, NumProperties*sizeof(ObjCPropertyDecl*));
}

/// FindPropertyDeclaration - Finds declaration of the property given its name
/// in 'PropertyId' and returns it. It returns 0, if not found.
///
ObjCPropertyDecl *
ObjCContainerDecl::FindPropertyDeclaration(IdentifierInfo *PropertyId) const {
  for (prop_iterator I = prop_begin(), E = prop_end(); I != E; ++I) {
    ObjCPropertyDecl *property = *I;
    if (property->getIdentifier() == PropertyId)
      return property;
  }
  const ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(this);
  if (OID) {
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
/// properties implemented in this category @implementation block and returns the 
/// implemented property that uses it.
///
ObjCPropertyImplDecl *ObjCCategoryImplDecl::FindPropertyImplIvarDecl(IdentifierInfo *ivarId) const {
  for (propimpl_iterator i = propimpl_begin(), e = propimpl_end(); i != e; ++i) {
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
ObjCPropertyImplDecl *ObjCCategoryImplDecl::FindPropertyImplDecl(IdentifierInfo *Id) const {
  for (propimpl_iterator i = propimpl_begin(), e = propimpl_end(); i != e; ++i) {
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
  void *Mem = C.getAllocator().Allocate<ObjCPropertyImplDecl>();
  return new (Mem) ObjCPropertyImplDecl(DC, atLoc, L, property, PK, ivar);
}


