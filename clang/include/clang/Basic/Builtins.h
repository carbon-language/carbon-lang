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

#ifndef LLVM_CLANG_BASIC_BUILTINS_H
#define LLVM_CLANG_BASIC_BUILTINS_H

#include <cstring>

// VC++ defines 'alloca' as an object-like macro, which interferes with our
// builtins.
#undef alloca

namespace llvm {
  template <typename T> class SmallVectorImpl;
}

namespace clang {
  class TargetInfo;
  class IdentifierTable;
  class ASTContext;
  class QualType;

namespace Builtin {
enum ID {
  NotBuiltin  = 0,      // This is not a builtin function.
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/Builtins.def"
  FirstTSBuiltin
};

struct Info {
  const char *Name, *Type, *Attributes, *HeaderName;
  bool Suppressed;

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
  Context(const TargetInfo &Target);

  /// InitializeBuiltins - Mark the identifiers for all the builtins with their
  /// appropriate builtin ID # and mark any non-portable builtin identifiers as
  /// such.
  void InitializeBuiltins(IdentifierTable &Table, bool NoBuiltins = false);

  /// \brief Popular the vector with the names of all of the builtins.
  void GetBuiltinNames(llvm::SmallVectorImpl<const char *> &Names,
                       bool NoBuiltins);

  /// Builtin::GetName - Return the identifier name for the specified builtin,
  /// e.g. "__builtin_abs".
  const char *GetName(unsigned ID) const {
    return GetRecord(ID).Name;
  }

  /// GetTypeString - Get the type descriptor string for the specified builtin.
  const char *GetTypeString(unsigned ID) const {
    return GetRecord(ID).Type;
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

  /// isNoReturn - Return true if we know this builtin never returns.
  bool isNoReturn(unsigned ID) const {
    return strchr(GetRecord(ID).Attributes, 'r') != 0;
  }

  /// isLibFunction - Return true if this is a builtin for a libc/libm function,
  /// with a "__builtin_" prefix (e.g. __builtin_abs).
  bool isLibFunction(unsigned ID) const {
    return strchr(GetRecord(ID).Attributes, 'F') != 0;
  }

  /// \brief Determines whether this builtin is a predefined libc/libm
  /// function, such as "malloc", where we know the signature a
  /// priori.
  bool isPredefinedLibFunction(unsigned ID) const {
    return strchr(GetRecord(ID).Attributes, 'f') != 0;
  }

  /// \brief If this is a library function that comes from a specific
  /// header, retrieve that header name.
  const char *getHeaderName(unsigned ID) const {
    return GetRecord(ID).HeaderName;
  }

  /// \brief Determine whether this builtin is like printf in its
  /// formatting rules and, if so, set the index to the format string
  /// argument and whether this function as a va_list argument.
  bool isPrintfLike(unsigned ID, unsigned &FormatIdx, bool &HasVAListArg);

  /// \brief Determine whether this builtin is like scanf in its
  /// formatting rules and, if so, set the index to the format string
  /// argument and whether this function as a va_list argument.
  bool isScanfLike(unsigned ID, unsigned &FormatIdx, bool &HasVAListArg);

  /// isConstWithoutErrno - Return true if this function has no side
  /// effects and doesn't read memory, except for possibly errno. Such
  /// functions can be const when the MathErrno lang option is
  /// disabled.
  bool isConstWithoutErrno(unsigned ID) const {
    return strchr(GetRecord(ID).Attributes, 'e') != 0;
  }

private:
  const Info &GetRecord(unsigned ID) const;
};

}
} // end namespace clang
#endif
