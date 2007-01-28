//===--- Builtins.h - Builtin function header -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines enum values for all the target-independent builtin
// functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_BUILTINS_H
#define LLVM_CLANG_AST_BUILTINS_H

namespace llvm {
namespace clang {
  class TargetInfo;
  class IdentifierTable;
  class ASTContext;
  class TypeRef;

namespace Builtin {
enum ID {
  NotBuiltin  = 0,      // This is not a builtin function.
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/AST/Builtins.def"
  FirstTargetSpecificBuiltin
};

struct Info {
  const char *Name, *Type, *Attributes;
};

/// Builtin::GetName - Return the identifier name for the specified builtin,
/// e.g. "__builtin_abs".
const char *GetName(ID id);

/// InitializeBuiltins - Mark the identifiers for all the builtins with their
/// appropriate builtin ID # and mark any non-portable builtin identifiers as
/// such.
void InitializeBuiltins(IdentifierTable &Table, const TargetInfo &Target);

/// GetBuiltinType - Return the type for the specified builtin.
TypeRef GetBuiltinType(ID id, ASTContext &Context);

}
} // end namespace clang
} // end namespace llvm
#endif
