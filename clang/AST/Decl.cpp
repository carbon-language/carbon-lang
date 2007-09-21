//===--- Decl.cpp - Declaration AST Node Implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Decl class and subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/Lex/IdentifierTable.h"
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

static bool StatSwitch = false;

const char *Decl::getDeclKindName() const {
  switch (DeclKind) {
  default: assert(0 && "Unknown decl kind!");
  case Typedef:
    return "Typedef";
  case Function:
    return "Function";
  case BlockVariable:
    return "BlockVariable";
  case FileVariable:
    return "FileVariable";
  case ParmVariable:
    return "ParmVariable";
  case EnumConstant:
    return "EnumConstant";
  case ObjcInterface:
    return "ObjcInterface";
  case ObjcClass:
    return "ObjcClass";
  case ObjcMethod:
    return "ObjcMethod";
  case ObjcProtoMethod:
    return "ObjcProtoMethod";
  case ObjcProtocol:
    return "ObjcProtocol";
  case ObjcForwardProtocol:
    return "ObjcForwardProtocol"; 
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
	  nInterfaceDecls, (int)sizeof(ObjcInterfaceDecl),
	  int(nInterfaceDecls*sizeof(ObjcInterfaceDecl)));
  fprintf(stderr, "    %d instance variable decls, %d each (%d bytes)\n", 
	  nIvarDecls, (int)sizeof(ObjcIvarDecl),
	  int(nIvarDecls*sizeof(ObjcIvarDecl)));
  fprintf(stderr, "    %d class decls, %d each (%d bytes)\n", 
	  nClassDecls, (int)sizeof(ObjcClassDecl),
	  int(nClassDecls*sizeof(ObjcClassDecl)));
  fprintf(stderr, "    %d method decls, %d each (%d bytes)\n", 
	  nMethodDecls, (int)sizeof(ObjcMethodDecl),
	  int(nMethodDecls*sizeof(ObjcMethodDecl)));
  fprintf(stderr, "    %d protocol decls, %d each (%d bytes)\n", 
	  nProtocolDecls, (int)sizeof(ObjcProtocolDecl),
	  int(nProtocolDecls*sizeof(ObjcProtocolDecl)));
  fprintf(stderr, "    %d forward protocol decls, %d each (%d bytes)\n", 
	  nForwardProtocolDecls, (int)sizeof(ObjcForwardProtocolDecl),
	  int(nForwardProtocolDecls*sizeof(ObjcForwardProtocolDecl)));
  fprintf(stderr, "    %d category decls, %d each (%d bytes)\n", 
	  nCategoryDecls, (int)sizeof(ObjcCategoryDecl),
	  int(nCategoryDecls*sizeof(ObjcCategoryDecl)));

  fprintf(stderr, "Total bytes = %d\n", 
	  int(nFuncs*sizeof(FunctionDecl)+nBlockVars*sizeof(BlockVarDecl)+
	      nFileVars*sizeof(FileVarDecl)+nParmVars*sizeof(ParmVarDecl)+
	      nFieldDecls*sizeof(FieldDecl)+nSUC*sizeof(RecordDecl)+
	      nEnumDecls*sizeof(EnumDecl)+nEnumConst*sizeof(EnumConstantDecl)+
	      nTypedef*sizeof(TypedefDecl)) /* FIXME: add Objc decls */);
}

void Decl::addDeclKind(const Kind k) {
  switch (k) {
    case Typedef:
      nTypedef++;
      break;
    case Function:
      nFuncs++;
      break;
    case BlockVariable:
      nBlockVars++;
      break;
    case FileVariable:
      nFileVars++;
      break;
    case ParmVariable:
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
    case ObjcInterface:
      nInterfaceDecls++;
      break;
    case ObjcClass:
      nClassDecls++;
      break;
    case ObjcMethod:
    case ObjcProtoMethod:
      nMethodDecls++;
      break;
    case ObjcProtocol:
      nProtocolDecls++;
      break;
    case ObjcForwardProtocol:
      nForwardProtocolDecls++;
      break;
    case ObjcCategory:
     nCategoryDecls++;
     break;
    case ObjcIvar:
      nIvarDecls++;
      break;
  }
}

// Out-of-line virtual method providing a home for Decl.
Decl::~Decl() {
}

const char *FieldDecl::getName() const {
  if (const IdentifierInfo *II = getIdentifier())
    return II->getName();
  return "";
}

const char *ScopedDecl::getName() const {
  if (const IdentifierInfo *II = getIdentifier())
    return II->getName();
  return "";
}


FunctionDecl::~FunctionDecl() {
  delete[] ParamInfo;
}

unsigned FunctionDecl::getNumParams() const {
  return cast<FunctionTypeProto>(getType().getTypePtr())->getNumArgs();
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

void ObjcMethodDecl::setMethodParams(ParmVarDecl **NewParamInfo, 
		       unsigned NumParams) {
  assert(ParamInfo == 0 && "Already has param info!");

  // Zero params -> null pointer.
  if (NumParams) {
    ParamInfo = new ParmVarDecl*[NumParams];
    memcpy(ParamInfo, NewParamInfo, sizeof(ParmVarDecl*)*NumParams);
    NumMethodParams = NumParams;
  }
}

ObjcMethodDecl::~ObjcMethodDecl() {
  delete[] ParamInfo;
}

/// ObjcAddInstanceVariablesToClass - Inserts instance variables
/// into ObjcInterfaceDecl's fields.
///
void ObjcInterfaceDecl::ObjcAddInstanceVariablesToClass(ObjcIvarDecl **ivars,
					   		unsigned numIvars) {
  NumIvars = numIvars;
  if (numIvars) {
    Ivars = new ObjcIvarDecl*[numIvars];
    memcpy(Ivars, ivars, numIvars*sizeof(ObjcIvarDecl*));
  }
}

/// addObjcMethods - Insert instance and methods declarations into
/// ObjcInterfaceDecl's InsMethods and ClsMethods fields.
///
void ObjcInterfaceDecl::ObjcAddMethods(ObjcMethodDecl **insMethods, 
				       unsigned numInsMembers,
                                       ObjcMethodDecl **clsMethods,
                                       unsigned numClsMembers) {
  NumInsMethods = numInsMembers;
  if (numInsMembers) {
    InsMethods = new ObjcMethodDecl*[numInsMembers];
    memcpy(InsMethods, insMethods, numInsMembers*sizeof(ObjcMethodDecl*));
  }
  NumClsMethods = numClsMembers;
  if (numClsMembers) {
    ClsMethods = new ObjcMethodDecl*[numClsMembers];
    memcpy(ClsMethods, clsMethods, numClsMembers*sizeof(ObjcMethodDecl*));
  }
}

/// ObjcAddProtoMethods - Insert instance and methods declarations into
/// ObjcProtocolDecl's ProtoInsMethods and ProtoClsMethods fields.
///
void ObjcProtocolDecl::ObjcAddProtoMethods(ObjcMethodDecl **insMethods, 
					   unsigned numInsMembers,
					   ObjcMethodDecl **clsMethods,
					   unsigned numClsMembers) {
  NumProtoInsMethods = numInsMembers;
  if (numInsMembers) {
    ProtoInsMethods = new ObjcMethodDecl*[numInsMembers];
    memcpy(ProtoInsMethods, insMethods, numInsMembers*sizeof(ObjcMethodDecl*));
  }
  NumProtoClsMethods = numClsMembers;
  if (numClsMembers) {
    ProtoClsMethods = new ObjcMethodDecl*[numClsMembers];
    memcpy(ProtoClsMethods, clsMethods, numClsMembers*sizeof(ObjcMethodDecl*));
  }
}

/// ObjcAddCat - Insert instance and methods declarations into
/// ObjcProtocolDecl's CatInsMethods and CatClsMethods fields.
///
void ObjcCategoryDecl::ObjcAddCatMethods(ObjcMethodDecl **insMethods, 
					 unsigned numInsMembers,
					 ObjcMethodDecl **clsMethods,
					 unsigned numClsMembers) {
  NumCatInsMethods = numInsMembers;
  if (numInsMembers) {
    CatInsMethods = new ObjcMethodDecl*[numInsMembers];
    memcpy(CatInsMethods, insMethods, numInsMembers*sizeof(ObjcMethodDecl*));
  }
  NumCatClsMethods = numClsMembers;
  if (numClsMembers) {
    CatClsMethods = new ObjcMethodDecl*[numClsMembers];
    memcpy(CatClsMethods, clsMethods, numClsMembers*sizeof(ObjcMethodDecl*));
  }
}


