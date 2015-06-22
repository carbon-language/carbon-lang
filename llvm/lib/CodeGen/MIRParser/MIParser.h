//===- MIParser.h - Machine Instructions Parser ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the function that parses the machine instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_MIRPARSER_MIPARSER_H
#define LLVM_LIB_CODEGEN_MIRPARSER_MIPARSER_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class MachineInstr;
class MachineFunction;
class SMDiagnostic;
class SourceMgr;

MachineInstr *parseMachineInstr(SourceMgr &SM, MachineFunction &MF,
                                StringRef Src, SMDiagnostic &Error);

} // end namespace llvm

#endif
