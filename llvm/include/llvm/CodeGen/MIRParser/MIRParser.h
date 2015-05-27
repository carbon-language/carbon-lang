//===- MIRParser.h - MIR serialization format parser ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This MIR serialization library is currently a work in progress. It can't
// serialize machine functions at this time.
//
// This file declares the functions that parse the MIR serialization format
// files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MIRPARSER_MIRPARSER_H
#define LLVM_CODEGEN_MIRPARSER_MIRPARSER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace llvm {

class SMDiagnostic;

/// This function is the main interface to the MIR serialization format parser.
///
/// It reads a YAML file that has an optional LLVM IR and returns an LLVM
/// module.
/// \param Filename - The name of the file to parse.
/// \param Error - Error result info.
/// \param Context - Context in which to allocate globals info.
std::unique_ptr<Module> parseMIRFile(StringRef Filename, SMDiagnostic &Error,
                                     LLVMContext &Context);

/// This function is another interface to the MIR serialization format parser.
///
/// It parses the optional LLVM IR in the given buffer, and returns an LLVM
/// module.
/// \param Contents - The MemoryBuffer containing the machine level IR.
/// \param Error - Error result info.
/// \param Context - Context in which to allocate globals info.
std::unique_ptr<Module> parseMIR(std::unique_ptr<MemoryBuffer> Contents,
                                 SMDiagnostic &Error, LLVMContext &Context);

} // end namespace llvm

#endif
