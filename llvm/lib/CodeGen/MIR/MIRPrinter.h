//===- MIRPrinter.h - MIR serialization format printer --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the function that prints out the LLVM IR using the MIR
// serialization format.
// TODO: Print out machine functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_MIR_MIRPRINTER_H
#define LLVM_LIB_CODEGEN_MIR_MIRPRINTER_H

namespace llvm {

class Module;
class raw_ostream;

/// Print LLVM IR using the MIR serialization format to the given output stream.
void printMIR(raw_ostream &OS, const Module &Mod);

} // end namespace llvm

#endif
