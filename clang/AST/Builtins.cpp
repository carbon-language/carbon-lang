//===--- Builtins.cpp - Builtin function implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements various things for builtin functions.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Builtins.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/IdentifierTable.h"
using namespace llvm;
using namespace clang;

static const Builtin::Info BuiltinInfo[] = {
  { "not a builtin function", 0, 0 },
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS },
#include "clang/AST/Builtins.def"
};

/// Builtin::GetName - Return the identifier name for the specified builtin,
/// e.g. "__builtin_abs".
const char *Builtin::GetName(ID id) {
  if (id >= Builtin::FirstTargetSpecificBuiltin)
    return "target-builtin";
  return BuiltinInfo[id].Name;
}


/// InitializeBuiltins - Mark the identifiers for all the builtins with their
/// appropriate builtin ID # and mark any non-portable builtin identifiers as
/// such.
void Builtin::InitializeBuiltins(IdentifierTable &Table,
                                 const TargetInfo &Target) {
  // Step #1: mark all target-independent builtins with their ID's.
  for (unsigned i = Builtin::NotBuiltin+1;
       i != Builtin::FirstTargetSpecificBuiltin; ++i)
    Table.get(BuiltinInfo[i].Name).setBuiltinID(i);
  
  // Step #2: handle target builtins.
  // FIXME: implement.
}

/// DecodeTypeFromStr - This decodes one type descriptor from Str, advancing the
/// pointer over the consumed characters.  This returns the resultant type.
static TypeRef DecodeTypeFromStr(const char *&Str, ASTContext &Context) {
  // Modifiers.
  bool Long = false, LongLong = false, Signed = false, Unsigned = false;
  
  // Read the modifiers first.
  bool Done = false;
  while (!Done) {
    switch (*Str++) {
    default: Done = true; --Str; break; 
    case 'S':
      assert(!Unsigned && "Can't use both 'S' and 'U' modifiers!");
      assert(!Signed && "Can't use 'S' modifier multiple times!");
      Signed = true;
      break;
    case 'U':
      assert(!Signed && "Can't use both 'S' and 'U' modifiers!");
      assert(!Unsigned && "Can't use 'S' modifier multiple times!");
      Unsigned = true;
      break;
    case 'L':
      assert(!LongLong && "Can't have LLL modifier");
      if (Long) 
        LongLong = true;
      else
        Long = true;
      break;
    }
  }

  // Read the base type.
  switch (*Str++) {
  default: assert(0 && "Unknown builtin type letter!");
  case 'f':
    assert(!Long && !Signed && !Unsigned && "Bad modifiers used with 'f'!");
    return Context.FloatTy;
  case 'd':
    assert(!LongLong && !Signed && !Unsigned && "Bad modifiers used with 'd'!");
    if (Long)
      return Context.LongDoubleTy;
    return Context.DoubleTy;
  //case 'i':
  }
}

/// GetBuiltinType - Return the type for the specified builtin.
TypeRef Builtin::GetBuiltinType(ID id, ASTContext &Context) {
  assert(id < Builtin::FirstTargetSpecificBuiltin &&
         "Can't handle target builtins yet!");
  const char *TypeStr = BuiltinInfo[id].Type;
  
  SmallVector<TypeRef, 8> ArgTypes;
  
  TypeRef ResType = DecodeTypeFromStr(TypeStr, Context);
  while (TypeStr[0] && TypeStr[0] != '.')
    ArgTypes.push_back(DecodeTypeFromStr(TypeStr, Context));
  
  assert((TypeStr[0] != '.' || TypeStr[1] == 0) &&
         "'.' should only occur at end of builtin type list!");
  
  return Context.getFunctionType(ResType, &ArgTypes[0], ArgTypes.size(),
                                 TypeStr[0] == '.');
}
