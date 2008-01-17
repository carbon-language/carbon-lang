//===--- Decl.cpp - Declaration AST Node Implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Decl class and subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/IdentifierTable.h"
using namespace clang;

// temporary statistics gathering
static unsigned nFuncs = 0;
static unsigned nBlockVars = 0;
static unsigned nFileVars = 0;
static unsigned nParmVars = 0;
static unsigned nSUC = 0;
static unsigned nEnumConst = 0;
static unsigned nEnumDecls = 0;
static unsigned nTypedef = 0;
static unsigned nFieldDecls = 0;
static unsigned nInterfaceDecls = 0;
static unsigned nClassDecls = 0;
static unsigned nMethodDecls = 0;
static unsigned nProtocolDecls = 0;
static unsigned nForwardProtocolDecls = 0;
static unsigned nCategoryDecls = 0;
static unsigned nIvarDecls = 0;
static unsigned nObjCImplementationDecls = 0;
static unsigned nObjCCategoryImpl = 0;
static unsigned nObjCCompatibleAlias = 0;
static unsigned nObjCPropertyDecl = 0;
static unsigned nLinkageSpecDecl = 0;

static bool StatSwitch = false;

const char *Decl::getDeclKindName() const {
  switch (DeclKind) {
  default: assert(0 && "Unknown decl kind!");
  case Typedef:
    return "Typedef";
  case Function:
    return "Function";
  case BlockVar:
    return "BlockVar";
  case FileVar:
    return "FileVar";
  case ParmVar:
    return "ParmVar";
  case EnumConstant:
    return "EnumConstant";
  case ObjCInterface:
    return "ObjCInterface";
  case ObjCClass:
    return "ObjCClass";
  case ObjCMethod:
    return "ObjCMethod";
  case ObjCProtocol:
    return "ObjCProtocol";
  case ObjCForwardProtocol:
    return "ObjCForwardProtocol"; 
  case Struct:
    return "Struct";
  case Union:
    return "Union";
  case Class:
    return "Class";
  case Enum:
    return "Enum";
  }
}

bool Decl::CollectingStats(bool enable) {
  if (enable) StatSwitch = true;
    return StatSwitch;
}

void Decl::PrintStats() {
  fprintf(stderr, "*** Decl Stats:\n");
  fprintf(stderr, "  %d decls total.\n", 
	  int(nFuncs+nBlockVars+nFileVars+nParmVars+nFieldDecls+nSUC+
	      nEnumDecls+nEnumConst+nTypedef+nInterfaceDecls+nClassDecls+
	      nMethodDecls+nProtocolDecls+nCategoryDecls+nIvarDecls));
  fprintf(stderr, "    %d function decls, %d each (%d bytes)\n", 
	  nFuncs, (int)sizeof(FunctionDecl), int(nFuncs*sizeof(FunctionDecl)));
  fprintf(stderr, "    %d block variable decls, %d each (%d bytes)\n", 
	  nBlockVars, (int)sizeof(BlockVarDecl), 
	  int(nBlockVars*sizeof(BlockVarDecl)));
  fprintf(stderr, "    %d file variable decls, %d each (%d bytes)\n", 
	  nFileVars, (int)sizeof(FileVarDecl), 
	  int(nFileVars*sizeof(FileVarDecl)));
  fprintf(stderr, "    %d parameter variable decls, %d each (%d bytes)\n", 
	  nParmVars, (int)sizeof(ParmVarDecl),
	  int(nParmVars*sizeof(ParmVarDecl)));
  fprintf(stderr, "    %d field decls, %d each (%d bytes)\n", 
	  nFieldDecls, (int)sizeof(FieldDecl),
	  int(nFieldDecls*sizeof(FieldDecl)));
  fprintf(stderr, "    %d struct/union/class decls, %d each (%d bytes)\n", 
	  nSUC, (int)sizeof(RecordDecl),
	  int(nSUC*sizeof(RecordDecl)));
  fprintf(stderr, "    %d enum decls, %d each (%d bytes)\n", 
	  nEnumDecls, (int)sizeof(EnumDecl), 
	  int(nEnumDecls*sizeof(EnumDecl)));
  fprintf(stderr, "    %d enum constant decls, %d each (%d bytes)\n", 
	  nEnumConst, (int)sizeof(EnumConstantDecl),
	  int(nEnumConst*sizeof(EnumConstantDecl)));
  fprintf(stderr, "    %d typedef decls, %d each (%d bytes)\n", 
	  nTypedef, (int)sizeof(TypedefDecl),int(nTypedef*sizeof(TypedefDecl)));
  // Objective-C decls...
  fprintf(stderr, "    %d interface decls, %d each (%d bytes)\n", 
	  nInterfaceDecls, (int)sizeof(ObjCInterfaceDecl),
	  int(nInterfaceDecls*sizeof(ObjCInterfaceDecl)));
  fprintf(stderr, "    %d instance variable decls, %d each (%d bytes)\n", 
	  nIvarDecls, (int)sizeof(ObjCIvarDecl),
	  int(nIvarDecls*sizeof(ObjCIvarDecl)));
  fprintf(stderr, "    %d class decls, %d each (%d bytes)\n", 
	  nClassDecls, (int)sizeof(ObjCClassDecl),
	  int(nClassDecls*sizeof(ObjCClassDecl)));
  fprintf(stderr, "    %d method decls, %d each (%d bytes)\n", 
	  nMethodDecls, (int)sizeof(ObjCMethodDecl),
	  int(nMethodDecls*sizeof(ObjCMethodDecl)));
  fprintf(stderr, "    %d protocol decls, %d each (%d bytes)\n", 
	  nProtocolDecls, (int)sizeof(ObjCProtocolDecl),
	  int(nProtocolDecls*sizeof(ObjCProtocolDecl)));
  fprintf(stderr, "    %d forward protocol decls, %d each (%d bytes)\n", 
	  nForwardProtocolDecls, (int)sizeof(ObjCForwardProtocolDecl),
	  int(nForwardProtocolDecls*sizeof(ObjCForwardProtocolDecl)));
  fprintf(stderr, "    %d category decls, %d each (%d bytes)\n", 
	  nCategoryDecls, (int)sizeof(ObjCCategoryDecl),
	  int(nCategoryDecls*sizeof(ObjCCategoryDecl)));

  fprintf(stderr, "    %d class implementation decls, %d each (%d bytes)\n", 
	  nObjCImplementationDecls, (int)sizeof(ObjCImplementationDecl),
	  int(nObjCImplementationDecls*sizeof(ObjCImplementationDecl)));

  fprintf(stderr, "    %d class implementation decls, %d each (%d bytes)\n", 
	  nObjCCategoryImpl, (int)sizeof(ObjCCategoryImplDecl),
	  int(nObjCCategoryImpl*sizeof(ObjCCategoryImplDecl)));

  fprintf(stderr, "    %d compatibility alias decls, %d each (%d bytes)\n", 
	  nObjCCompatibleAlias, (int)sizeof(ObjCCompatibleAliasDecl),
	  int(nObjCCompatibleAlias*sizeof(ObjCCompatibleAliasDecl)));
  
  fprintf(stderr, "    %d property decls, %d each (%d bytes)\n", 
	  nObjCPropertyDecl, (int)sizeof(ObjCPropertyDecl),
	  int(nObjCPropertyDecl*sizeof(ObjCPropertyDecl)));
  
  fprintf(stderr, "Total bytes = %d\n", 
	  int(nFuncs*sizeof(FunctionDecl)+nBlockVars*sizeof(BlockVarDecl)+
	      nFileVars*sizeof(FileVarDecl)+nParmVars*sizeof(ParmVarDecl)+
	      nFieldDecls*sizeof(FieldDecl)+nSUC*sizeof(RecordDecl)+
	      nEnumDecls*sizeof(EnumDecl)+nEnumConst*sizeof(EnumConstantDecl)+
	      nTypedef*sizeof(TypedefDecl)+
	      nLinkageSpecDecl*sizeof(LinkageSpecDecl))
	  /* FIXME: add ObjC decls */);
}

void Decl::addDeclKind(const Kind k) {
  switch (k) {
    case Typedef:
      nTypedef++;
      break;
    case Function:
      nFuncs++;
      break;
    case BlockVar:
      nBlockVars++;
      break;
    case FileVar:
      nFileVars++;
      break;
    case ParmVar:
      nParmVars++;
      break;
    case EnumConstant:
      nEnumConst++;
      break;
    case Field:
      nFieldDecls++;
      break;
    case Struct:
    case Union:
    case Class:
      nSUC++;
      break;
    case Enum:
      nEnumDecls++;
      break;
    case ObjCInterface:
      nInterfaceDecls++;
      break;
    case ObjCClass:
      nClassDecls++;
      break;
    case ObjCMethod:
      nMethodDecls++;
      break;
    case ObjCProtocol:
      nProtocolDecls++;
      break;
    case ObjCForwardProtocol:
      nForwardProtocolDecls++;
      break;
    case ObjCCategory:
     nCategoryDecls++;
     break;
    case ObjCIvar:
      nIvarDecls++;
      break;
    case ObjCImplementation: 
      nObjCImplementationDecls++;
      break;
    case ObjCCategoryImpl:
      nObjCCategoryImpl++;
      break;
    case CompatibleAlias:
      nObjCCompatibleAlias++;
      break;
    case PropertyDecl:
      nObjCPropertyDecl++;
      break;
    case LinkageSpec:
      nLinkageSpecDecl++;
      break;
  }
}

// Out-of-line virtual method providing a home for Decl.
Decl::~Decl() {
}

const char *NamedDecl::getName() const {
  if (const IdentifierInfo *II = getIdentifier())
    return II->getName();
  return "";
}


FunctionDecl::~FunctionDecl() {
  delete[] ParamInfo;
}

unsigned FunctionDecl::getNumParams() const {
  if (isa<FunctionTypeNoProto>(getCanonicalType())) return 0;
  return cast<FunctionTypeProto>(getCanonicalType())->getNumArgs();
}

void FunctionDecl::setParams(ParmVarDecl **NewParamInfo, unsigned NumParams) {
  assert(ParamInfo == 0 && "Already has param info!");
  assert(NumParams == getNumParams() && "Parameter count mismatch!");
  
  // Zero params -> null pointer.
  if (NumParams) {
    ParamInfo = new ParmVarDecl*[NumParams];
    memcpy(ParamInfo, NewParamInfo, sizeof(ParmVarDecl*)*NumParams);
  }
}


/// defineBody - When created, RecordDecl's correspond to a forward declared
/// record.  This method is used to mark the decl as being defined, with the
/// specified contents.
void RecordDecl::defineBody(FieldDecl **members, unsigned numMembers) {
  assert(!isDefinition() && "Cannot redefine record!");
  setDefinition(true);
  NumMembers = numMembers;
  if (numMembers) {
    Members = new FieldDecl*[numMembers];
    memcpy(Members, members, numMembers*sizeof(Decl*));
  }
}

FieldDecl* RecordDecl::getMember(IdentifierInfo *name) {
  if (Members == 0 || NumMembers < 0)
    return 0;
  
  // linear search. When C++ classes come along, will likely need to revisit.
  for (int i = 0; i < NumMembers; ++i) {
    if (Members[i]->getIdentifier() == name)
      return Members[i];
  }
  return 0;
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

int ObjCMethodDecl::getSynthesizedSelectorSize() const {
  // syntesized method name is a concatenation of -/+[class-name selector]
  // Get length of this name.
  int length = 4;  // for '+' or '-', '[', space in between and ']'
  length += getSelector().getName().size(); // for selector name.
  length += strlen(getMethodContext()->getName()); // for its class name
  return length; 
}

ObjCInterfaceDecl *const ObjCMethodDecl::getClassInterface() const {
  if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(MethodContext))
    return ID;
  if (ObjCCategoryDecl *CD = dyn_cast<ObjCCategoryDecl>(MethodContext))
    return CD->getClassInterface();
  if (ObjCImplementationDecl *IMD = 
      dyn_cast<ObjCImplementationDecl>(MethodContext))
    return IMD->getClassInterface();
  if (ObjCCategoryImplDecl *CID = 
      dyn_cast<ObjCCategoryImplDecl>(MethodContext))
    return CID->getClassInterface();
  assert(false && "unknown method context");
  return 0;
}
