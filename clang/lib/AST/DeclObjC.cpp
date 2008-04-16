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
                                       ImplementationControl impControl) {
  void *Mem = C.getAllocator().Allocate<ObjCMethodDecl>();
  return new (Mem) ObjCMethodDecl(beginLoc, endLoc,
                                  SelInfo, T, contextDecl,
                                  M, isInstance, 
                                  isVariadic, impControl);
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
                                           QualType T) {
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

/// addProperties - Insert property declaration AST nodes into
/// ObjCProtocolDecl's PropertyDecl field.
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



