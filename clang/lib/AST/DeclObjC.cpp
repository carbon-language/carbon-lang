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
using namespace clang;

//===----------------------------------------------------------------------===//
// ObjC Decl Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//

ObjCMethodDecl *ObjCMethodDecl::Create(ASTContext &C,
                                       SourceLocation beginLoc, 
                                       SourceLocation endLoc,
                                       Selector SelInfo, QualType T,
                                       Decl *contextDecl,
                                       AttributeList *M, bool isInstance,
                                       bool isVariadic,
                                       bool isSynthesized,
                                       ImplementationControl impControl) {
  void *Mem = C.getAllocator().Allocate<ObjCMethodDecl>();
  return new (Mem) ObjCMethodDecl(beginLoc, endLoc,
                                  SelInfo, T, contextDecl,
                                  M, isInstance, 
                                  isVariadic, isSynthesized, impControl);
}

ObjCInterfaceDecl *ObjCInterfaceDecl::Create(ASTContext &C,
                                             SourceLocation atLoc,
                                             unsigned numRefProtos,
                                             IdentifierInfo *Id, 
                                             SourceLocation ClassLoc,
                                             bool ForwardDecl, bool isInternal){
  void *Mem = C.getAllocator().Allocate<ObjCInterfaceDecl>();
  return new (Mem) ObjCInterfaceDecl(atLoc, numRefProtos,
                                     Id, ClassLoc, ForwardDecl,
                                     isInternal);
}

ObjCIvarDecl *ObjCIvarDecl::Create(ASTContext &C, SourceLocation L,
                                   IdentifierInfo *Id, QualType T) {
  void *Mem = C.getAllocator().Allocate<ObjCIvarDecl>();
  return new (Mem) ObjCIvarDecl(L, Id, T);
}

ObjCProtocolDecl *ObjCProtocolDecl::Create(ASTContext &C,
                                           SourceLocation L, 
                                           unsigned numRefProtos,
                                           IdentifierInfo *Id) {
  void *Mem = C.getAllocator().Allocate<ObjCProtocolDecl>();
  return new (Mem) ObjCProtocolDecl(L, numRefProtos, Id);
}

ObjCClassDecl *ObjCClassDecl::Create(ASTContext &C,
                                     SourceLocation L,
                                     ObjCInterfaceDecl **Elts, unsigned nElts) {
  void *Mem = C.getAllocator().Allocate<ObjCClassDecl>();
  return new (Mem) ObjCClassDecl(L, Elts, nElts);
}

ObjCForwardProtocolDecl *
ObjCForwardProtocolDecl::Create(ASTContext &C,
                                SourceLocation L, 
                                ObjCProtocolDecl **Elts, unsigned NumElts) {
  void *Mem = C.getAllocator().Allocate<ObjCForwardProtocolDecl>();
  return new (Mem) ObjCForwardProtocolDecl(L, Elts, NumElts);
}

ObjCCategoryDecl *ObjCCategoryDecl::Create(ASTContext &C,
                                           SourceLocation L,
                                           IdentifierInfo *Id) {
  void *Mem = C.getAllocator().Allocate<ObjCCategoryDecl>();
  return new (Mem) ObjCCategoryDecl(L, Id);
}

ObjCCategoryImplDecl *
ObjCCategoryImplDecl::Create(ASTContext &C,
                             SourceLocation L,IdentifierInfo *Id,
                             ObjCInterfaceDecl *ClassInterface) {
  void *Mem = C.getAllocator().Allocate<ObjCCategoryImplDecl>();
  return new (Mem) ObjCCategoryImplDecl(L, Id, ClassInterface);
}

ObjCImplementationDecl *
ObjCImplementationDecl::Create(ASTContext &C,
                               SourceLocation L,
                               IdentifierInfo *Id,
                               ObjCInterfaceDecl *ClassInterface,
                               ObjCInterfaceDecl *SuperDecl) {
  void *Mem = C.getAllocator().Allocate<ObjCImplementationDecl>();
  return new (Mem) ObjCImplementationDecl(L, Id, ClassInterface, SuperDecl);
}

ObjCCompatibleAliasDecl *
ObjCCompatibleAliasDecl::Create(ASTContext &C,
                                SourceLocation L,
                                IdentifierInfo *Id, 
                                ObjCInterfaceDecl* AliasedClass) {
  void *Mem = C.getAllocator().Allocate<ObjCCompatibleAliasDecl>();
  return new (Mem) ObjCCompatibleAliasDecl(L, Id, AliasedClass);
}

ObjCPropertyDecl *ObjCPropertyDecl::Create(ASTContext &C,
                                           SourceLocation L,
                                           IdentifierInfo *Id,
                                           QualType T,
                                           PropertyControl propControl) {
  void *Mem = C.getAllocator().Allocate<ObjCPropertyDecl>();
  return new (Mem) ObjCPropertyDecl(L, Id, T);
}

//===----------------------------------------------------------------------===//
// Objective-C Decl Implementation
//===----------------------------------------------------------------------===//

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

ObjCMethodDecl::~ObjCMethodDecl() {
  delete[] ParamInfo;
}

/// FindPropertyDeclaration - Finds declaration of the property given its name
/// in 'PropertyId' and returns it. It returns 0, if not found.
///
ObjCPropertyDecl *
  ObjCInterfaceDecl::FindPropertyDeclaration(IdentifierInfo *PropertyId) const {
  for (ObjCInterfaceDecl::classprop_iterator I = classprop_begin(),
       E = classprop_end(); I != E; ++I) {
    ObjCPropertyDecl *property = *I;
    if (property->getIdentifier() == PropertyId)
      return property;
  }
  // Look through categories.
  for (ObjCCategoryDecl *Category = getCategoryList();
       Category; Category = Category->getNextClassCategory()) {
    ObjCPropertyDecl *property = Category->FindPropertyDeclaration(PropertyId);
    if (property)
      return property;
  }
  if (getSuperClass())
    return getSuperClass()->FindPropertyDeclaration(PropertyId);
  return 0;
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

/// addMethods - Insert instance and methods declarations into
/// ObjCInterfaceDecl's InsMethods and ClsMethods fields.
///
void ObjCInterfaceDecl::addMethods(ObjCMethodDecl **insMethods, 
                                   unsigned numInsMembers,
                                   ObjCMethodDecl **clsMethods,
                                   unsigned numClsMembers,
                                   SourceLocation endLoc) {
  NumInstanceMethods = numInsMembers;
  if (numInsMembers) {
    InstanceMethods = new ObjCMethodDecl*[numInsMembers];
    memcpy(InstanceMethods, insMethods, numInsMembers*sizeof(ObjCMethodDecl*));
  }
  NumClassMethods = numClsMembers;
  if (numClsMembers) {
    ClassMethods = new ObjCMethodDecl*[numClsMembers];
    memcpy(ClassMethods, clsMethods, numClsMembers*sizeof(ObjCMethodDecl*));
  }
  AtEndLoc = endLoc;
}

/// addProperties - Insert property declaration AST nodes into
/// ObjCInterfaceDecl's PropertyDecl field.
///
void ObjCInterfaceDecl::addProperties(ObjCPropertyDecl **Properties, 
                                      unsigned NumProperties) {
  if (NumProperties == 0) return;
  
  NumPropertyDecl = NumProperties;
  PropertyDecl = new ObjCPropertyDecl*[NumProperties];
  memcpy(PropertyDecl, Properties, NumProperties*sizeof(ObjCPropertyDecl*));
}                                   

/// mergeProperties - Adds properties to the end of list of current properties
/// for this class.

void ObjCInterfaceDecl::mergeProperties(ObjCPropertyDecl **Properties, 
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

/// addPropertyMethods - Goes through list of properties declared in this class
/// and builds setter/getter method declartions depending on the setter/getter
/// attributes of the property.
///
void ObjCInterfaceDecl::addPropertyMethods(
       ASTContext &Context,
       ObjCPropertyDecl *property,
       llvm::SmallVector<ObjCMethodDecl*, 32> &insMethods) {
  // Find the default getter and if one not found, add one.
  ObjCMethodDecl *GetterDecl = getInstanceMethod(property->getGetterName());
  if (GetterDecl) {
    // An instance method with same name as property getter name found.
    property->setGetterMethodDecl(GetterDecl);
  }
  else {
    // No instance method of same name as property getter name was found.
    // Declare a getter method and add it to the list of methods 
    // for this class.
    QualType resultDeclType = property->getType();
    ObjCMethodDecl* ObjCMethod =
    ObjCMethodDecl::Create(Context, property->getLocation(), 
                           property->getLocation(), 
                           property->getGetterName(), resultDeclType,
                           this, 0,
                           true, false, true, ObjCMethodDecl::Required);
    property->setGetterMethodDecl(ObjCMethod);
    insMethods.push_back(ObjCMethod);
  }
}

/// addProperties - Insert property declaration AST nodes into
/// ObjCProtocolDecl's PropertyDecl field.
///
void ObjCProtocolDecl::addProperties(ObjCPropertyDecl **Properties, 
                                     unsigned NumProperties) {
  if (NumProperties == 0) return;
  
  NumPropertyDecl = NumProperties;
  PropertyDecl = new ObjCPropertyDecl*[NumProperties];
  memcpy(PropertyDecl, Properties, NumProperties*sizeof(ObjCPropertyDecl*));
}

/// addProperties - Insert property declaration AST nodes into
/// ObjCCategoryDecl's PropertyDecl field.
///
void ObjCCategoryDecl::addProperties(ObjCPropertyDecl **Properties, 
                                     unsigned NumProperties) {
  if (NumProperties == 0) return;
  
  NumPropertyDecl = NumProperties;
  PropertyDecl = new ObjCPropertyDecl*[NumProperties];
  memcpy(PropertyDecl, Properties, NumProperties*sizeof(ObjCPropertyDecl*));
}

/// addMethods - Insert instance and methods declarations into
/// ObjCProtocolDecl's ProtoInsMethods and ProtoClsMethods fields.
///
void ObjCProtocolDecl::addMethods(ObjCMethodDecl **insMethods, 
                                  unsigned numInsMembers,
                                  ObjCMethodDecl **clsMethods,
                                  unsigned numClsMembers,
                                  SourceLocation endLoc) {
  NumInstanceMethods = numInsMembers;
  if (numInsMembers) {
    InstanceMethods = new ObjCMethodDecl*[numInsMembers];
    memcpy(InstanceMethods, insMethods, numInsMembers*sizeof(ObjCMethodDecl*));
  }
  NumClassMethods = numClsMembers;
  if (numClsMembers) {
    ClassMethods = new ObjCMethodDecl*[numClsMembers];
    memcpy(ClassMethods, clsMethods, numClsMembers*sizeof(ObjCMethodDecl*));
  }
  AtEndLoc = endLoc;
}

void ObjCCategoryDecl::setReferencedProtocolList(ObjCProtocolDecl **List,
                                                 unsigned NumRPs) {
  assert(NumReferencedProtocols == 0 && "Protocol list already set");
  if (NumRPs == 0) return;
  
  ReferencedProtocols = new ObjCProtocolDecl*[NumRPs];
  memcpy(ReferencedProtocols, List, NumRPs*sizeof(ObjCProtocolDecl*));
  NumReferencedProtocols = NumRPs;
}


/// addMethods - Insert instance and methods declarations into
/// ObjCCategoryDecl's CatInsMethods and CatClsMethods fields.
///
void ObjCCategoryDecl::addMethods(ObjCMethodDecl **insMethods, 
                                  unsigned numInsMembers,
                                  ObjCMethodDecl **clsMethods,
                                  unsigned numClsMembers,
                                  SourceLocation endLoc) {
  NumInstanceMethods = numInsMembers;
  if (numInsMembers) {
    InstanceMethods = new ObjCMethodDecl*[numInsMembers];
    memcpy(InstanceMethods, insMethods, numInsMembers*sizeof(ObjCMethodDecl*));
  }
  NumClassMethods = numClsMembers;
  if (numClsMembers) {
    ClassMethods = new ObjCMethodDecl*[numClsMembers];
    memcpy(ClassMethods, clsMethods, numClsMembers*sizeof(ObjCMethodDecl*));
  }
  AtEndLoc = endLoc;
}

/// FindPropertyDeclaration - Finds declaration of the property given its name
/// in 'PropertyId' and returns it. It returns 0, if not found.
///
ObjCPropertyDecl *
ObjCCategoryDecl::FindPropertyDeclaration(IdentifierInfo *PropertyId) const {
  for (ObjCCategoryDecl::classprop_iterator I = classprop_begin(),
       E = classprop_end(); I != E; ++I) {
    ObjCPropertyDecl *property = *I;
    if (property->getIdentifier() == PropertyId)
      return property;
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
    ObjCProtocolDecl **protocols = ClassDecl->getReferencedProtocols();
    int numProtocols = ClassDecl->getNumIntfRefProtocols();
    for (int pIdx = 0; pIdx < numProtocols; pIdx++) {
      if ((MethodDecl = protocols[pIdx]->getInstanceMethod(Sel)))
        return MethodDecl;
    }
    // Didn't find one yet - now look through categories.
    ObjCCategoryDecl *CatDecl = ClassDecl->getCategoryList();
    while (CatDecl) {
      if ((MethodDecl = CatDecl->getInstanceMethod(Sel)))
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
    ObjCProtocolDecl **protocols = ClassDecl->getReferencedProtocols();
    int numProtocols = ClassDecl->getNumIntfRefProtocols();
    for (int pIdx = 0; pIdx < numProtocols; pIdx++) {
      if ((MethodDecl = protocols[pIdx]->getClassMethod(Sel)))
        return MethodDecl;
    }
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

/// lookupInstanceMethod - This method returns an instance method by looking in
/// the class implementation. Unlike interfaces, we don't look outside the
/// implementation.
ObjCMethodDecl *ObjCImplementationDecl::getInstanceMethod(Selector Sel) {
  for (instmeth_iterator I = instmeth_begin(), E = instmeth_end(); I != E; ++I)
    if ((*I)->getSelector() == Sel)
      return *I;
  return NULL;
}

/// lookupClassMethod - This method returns a class method by looking in
/// the class implementation. Unlike interfaces, we don't look outside the
/// implementation.
ObjCMethodDecl *ObjCImplementationDecl::getClassMethod(Selector Sel) {
  for (classmeth_iterator I = classmeth_begin(), E = classmeth_end();
       I != E; ++I)
    if ((*I)->getSelector() == Sel)
      return *I;
  return NULL;
}

// lookupInstanceMethod - This method returns an instance method by looking in
// the class implementation. Unlike interfaces, we don't look outside the
// implementation.
ObjCMethodDecl *ObjCCategoryImplDecl::getInstanceMethod(Selector Sel) {
  for (instmeth_iterator I = instmeth_begin(), E = instmeth_end(); I != E; ++I)
    if ((*I)->getSelector() == Sel)
      return *I;
  return NULL;
}

// lookupClassMethod - This method returns an instance method by looking in
// the class implementation. Unlike interfaces, we don't look outside the
// implementation.
ObjCMethodDecl *ObjCCategoryImplDecl::getClassMethod(Selector Sel) {
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
    
  if (getNumReferencedProtocols() > 0) {
    ObjCProtocolDecl **RefPDecl = getReferencedProtocols();
    
    for (unsigned i = 0; i < getNumReferencedProtocols(); i++) {
      if ((MethodDecl = RefPDecl[i]->getInstanceMethod(Sel)))
        return MethodDecl;
    }
  }
  return NULL;
}

// lookupInstanceMethod - Lookup a class method in the protocol and protocols
// it inherited.
ObjCMethodDecl *ObjCProtocolDecl::lookupClassMethod(Selector Sel) {
  ObjCMethodDecl *MethodDecl = NULL;

  if ((MethodDecl = getClassMethod(Sel)))
    return MethodDecl;
    
  if (getNumReferencedProtocols() > 0) {
    ObjCProtocolDecl **RefPDecl = getReferencedProtocols();
    
    for(unsigned i = 0; i < getNumReferencedProtocols(); i++) {
      if ((MethodDecl = RefPDecl[i]->getClassMethod(Sel)))
        return MethodDecl;
    }
  }
  return NULL;
}

/// getSynthesizedMethodSize - Compute size of synthesized method name
/// as done be the rewrite.
///
unsigned ObjCMethodDecl::getSynthesizedMethodSize() const {
  // syntesized method name is a concatenation of -/+[class-name selector]
  // Get length of this name.
  unsigned length = 3;  // _I_ or _C_
  length += strlen(getClassInterface()->getName()) +1; // extra for _
  NamedDecl *MethodContext = getMethodContext();
  if (ObjCCategoryImplDecl *CID = 
      dyn_cast<ObjCCategoryImplDecl>(MethodContext))
    length += strlen(CID->getName()) +1;
  length += getSelector().getName().size(); // selector name
  return length; 
}

ObjCInterfaceDecl *ObjCMethodDecl::getClassInterface() {
  if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(MethodContext))
    return ID;
  if (ObjCCategoryDecl *CD = dyn_cast<ObjCCategoryDecl>(MethodContext))
    return CD->getClassInterface();
  if (ObjCImplementationDecl *IMD = 
        dyn_cast<ObjCImplementationDecl>(MethodContext))
    return IMD->getClassInterface();
  if (ObjCCategoryImplDecl *CID = dyn_cast<ObjCCategoryImplDecl>(MethodContext))
    return CID->getClassInterface();
  assert(false && "unknown method context");
  return 0;
}

ObjCPropertyImplDecl *ObjCPropertyImplDecl::Create(ASTContext &C,
                                                   SourceLocation atLoc,
                                                   SourceLocation L,
                                                   ObjCPropertyDecl *property,
                                                   PropertyImplKind kind,
                                                   ObjCIvarDecl *ivar) {
  void *Mem = C.getAllocator().Allocate<ObjCPropertyImplDecl>();
  return new (Mem) ObjCPropertyImplDecl(atLoc, L, property, kind, ivar);
}


