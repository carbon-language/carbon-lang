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
static bool StatSwitch = false;

bool Decl::CollectingStats(bool enable) {
  if (enable) StatSwitch = true;
	return StatSwitch;
}

void Decl::PrintStats() {
  fprintf(stderr, "*** Decl Stats:\n");
  fprintf(stderr, "  %d decls total.\n", 
	  int(nFuncs+nBlockVars+nFileVars+nParmVars+nFieldDecls+nSUC+
	      nEnumDecls+nEnumConst+nTypedef));
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
  fprintf(stderr, "Total bytes = %d\n", 
	  int(nFuncs*sizeof(FunctionDecl)+nBlockVars*sizeof(BlockVarDecl)+
	      nFileVars*sizeof(FileVarDecl)+nParmVars*sizeof(ParmVarDecl)+
	      nFieldDecls*sizeof(FieldDecl)+nSUC*sizeof(RecordDecl)+
	      nEnumDecls*sizeof(EnumDecl)+nEnumConst*sizeof(EnumConstantDecl)+
	      nTypedef*sizeof(TypedefDecl)));
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
  }
}

// Out-of-line virtual method providing a home for Decl.
Decl::~Decl() {
}

const char *Decl::getName() const {
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
