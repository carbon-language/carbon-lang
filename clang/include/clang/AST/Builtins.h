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

#include <cstring>

namespace clang {
  class TargetInfo;
  class IdentifierTable;
  class ASTContext;
  class QualType;

namespace Builtin {
enum ID {
  NotBuiltin  = 0,      // This is not a builtin function.
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/AST/Builtins.def"
  FirstTSBuiltin
};

struct Info {
  const char *Name, *Type, *Attributes;
  
  bool operator==(const Info &RHS) const {
    return !strcmp(Name, RHS.Name) &&
           !strcmp(Type, RHS.Type) &&
           !strcmp(Attributes, RHS.Attributes);
  }
  bool operator!=(const Info &RHS) const { return !(*this == RHS); }
};

/// Builtin::Context - This holds information about target-independent and
/// target-specific builtins, allowing easy queries by clients.
class Context {
  const Info *TSRecords;
  unsigned NumTSRecords;
public:
  Context() : TSRecords(0), NumTSRecords(0) {}
  
  /// InitializeBuiltins - Mark the identifiers for all the builtins with their
  /// appropriate builtin ID # and mark any non-portable builtin identifiers as
  /// such.
  void InitializeBuiltins(IdentifierTable &Table, const TargetInfo &Target);
  
  /// Builtin::GetName - Return the identifier name for the specified builtin,
  /// e.g. "__builtin_abs".
  const char *GetName(unsigned ID) const {
    return GetRecord(ID).Name;
  }
  
  /// GetBuiltinType - Return the type for the specified builtin.
  QualType GetBuiltinType(unsigned ID, ASTContext &Context) const;
private:
  const Info &GetRecord(unsigned ID) const;
};

}
} // end namespace clang
#endif
