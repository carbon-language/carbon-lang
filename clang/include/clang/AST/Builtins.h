//===--- Builtins.h - Builtin function header -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  
  /// isConst - Return true if this function has no side effects and doesn't
  /// read memory.
  bool isConst(unsigned ID) const {
    return strchr(GetRecord(ID).Attributes, 'c') != 0;
  }
  
  /// isNoThrow - Return true if we know this builtin never throws an exception.
  bool isNoThrow(unsigned ID) const {
    return strchr(GetRecord(ID).Attributes, 'n') != 0;
  }
  
  /// isLibFunction - Return true if this is a builtin for a libc/libm function,
  /// with a "__builtin_" prefix (e.g. __builtin_abs).
  bool isLibFunction(unsigned ID) const {
    return strchr(GetRecord(ID).Attributes, 'F') != 0;
  }
  
  /// isConstantExpr - Return true if this builtin can be used where a
  /// constant expression is required.
  bool isConstantExpr(unsigned ID) const {
    return strchr(GetRecord(ID).Attributes, 'C') != 0;
  }
  
  /// hasVAListUse - Return true of the specified builtin uses __builtin_va_list
  /// as an operand or return type.
  bool hasVAListUse(unsigned ID) const {
    return strchr(GetRecord(ID).Type, 'a') != 0;
  }
  
  /// GetBuiltinType - Return the type for the specified builtin.
  QualType GetBuiltinType(unsigned ID, ASTContext &Context) const;
private:
  const Info &GetRecord(unsigned ID) const;
};

}
} // end namespace clang
#endif
