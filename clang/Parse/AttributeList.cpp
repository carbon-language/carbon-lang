//===--- AttributeList.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AttributeList class implementation
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/AttributeList.h"
using namespace clang;

AttributeList::AttributeList(IdentifierInfo *aName, SourceLocation aLoc,
                             IdentifierInfo *pName, SourceLocation pLoc,
                             Action::ExprTy **elist, unsigned numargs, 
                             AttributeList *n)
  : AttrName(aName), AttrLoc(aLoc), ParmName(pName), ParmLoc(pLoc),
    NumArgs(numargs), Next(n) {
  Args = new Action::ExprTy*[numargs];
  for (unsigned i = 0; i != numargs; ++i)
    Args[i] = elist[i];
}

AttributeList::~AttributeList() {
  if (Args) {
    // FIXME: before we delete the vector, we need to make sure the Expr's 
    // have been deleted. Since Action::ExprTy is "void", we are dependent
    // on the actions module for actually freeing the memory. The specific
    // hooks are ActOnDeclarator, ActOnTypeName, ActOnParamDeclaratorType, 
    // ParseField, ParseTag. Once these routines have freed the expression, 
    // they should zero out the Args slot (to indicate the memory has been 
    // freed). If any element of the vector is non-null, we should assert.
    delete [] Args;
  }
  delete Next;
}

AttributeList::Kind AttributeList::getKind(const IdentifierInfo *Name) {
  const char *Str = Name->getName();
  unsigned Len = Name->getLength();

  // Normalize the attribute name, __foo__ becomes foo.
  if (Len > 4 && Str[0] == '_' && Str[1] == '_' &&
      Str[Len - 2] == '_' && Str[Len - 1] == '_') {
    Str += 2;
    Len -= 4;
  }
  
  switch (Len) {
  case 6: 
    if (!memcmp(Str, "packed", 6)) return AT_packed;
    break;
  case 7:
    if (!memcmp(Str, "aligned", 7)) return AT_aligned;
    break;
  case 11:   
    if (!memcmp(Str, "vector_size", 11)) return AT_vector_size;
    break;
  case 13:
    if (!memcmp(Str, "address_space", 13)) return AT_address_space;
    break;
  case 15:
    if (!memcmp(Str, "ocu_vector_type", 15)) return AT_ocu_vector_type;
    break;
  }      
  return UnknownAttribute;
}
