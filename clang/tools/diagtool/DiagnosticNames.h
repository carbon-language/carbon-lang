//===- DiagnosticNames.h - Defines a table of all builtin diagnostics ------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"

namespace diagtool {
  struct DiagnosticRecord {
    const char *NameStr;
    unsigned short DiagID;
    uint8_t NameLen;
    
    llvm::StringRef getName() const {
      return llvm::StringRef(NameStr, NameLen);
    }
  };

  extern const DiagnosticRecord BuiltinDiagnostics[];
  extern const size_t BuiltinDiagnosticsCount;

} // end namespace diagtool

