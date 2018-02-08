//===- TableGenBackends.h - Declarations for Clang TableGen Backends ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for all of the Clang TableGen
// backends. A "TableGen backend" is just a function. See
// "$LLVM_ROOT/utils/TableGen/TableGenBackends.h" for more info.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_UTILS_TABLEGEN_TABLEGENBACKENDS_H
#define LLVM_FLANG_UTILS_TABLEGEN_TABLEGENBACKENDS_H

#include <string>

namespace llvm {
class raw_ostream;
class RecordKeeper;
}  // namespace llvm

using llvm::RecordKeeper;
using llvm::raw_ostream;

namespace flang {

// Used by FlangDiagnosticsEmitter.cpp
void EmitFlangDiagsDefs(
    RecordKeeper &Records, raw_ostream &OS, const std::string &Component);
void EmitFlangDiagGroups(RecordKeeper &Records, raw_ostream &OS);
void EmitFlangDiagsIndexName(RecordKeeper &Records, raw_ostream &OS);
void EmitFlangDiagDocs(RecordKeeper &Records, raw_ostream &OS);

// Used by FlangOptionDocEmitter.cpp
void EmitFlangOptDocs(RecordKeeper &Records, raw_ostream &OS);

}  // end namespace flang

#endif
