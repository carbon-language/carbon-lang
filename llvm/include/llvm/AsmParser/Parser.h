//===-- Parser.h - Parser for LLVM IR text assembly files -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  These classes are implemented by the lib/AsmParser library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASMPARSER_PARSER_H
#define LLVM_ASMPARSER_PARSER_H

#include "llvm/ADT/StringRef.h"
#include <memory>

namespace llvm {

class Module;
class MemoryBuffer;
class SMDiagnostic;
class LLVMContext;

/// This function is the main interface to the LLVM Assembly Parser. It parses
/// an ASCII file that (presumably) contains LLVM Assembly code. It returns a
/// Module (intermediate representation) with the corresponding features. Note
/// that this does not verify that the generated Module is valid, so you should
/// run the verifier after parsing the file to check that it is okay.
/// @brief Parse LLVM Assembly from a file
/// @param Filename The name of the file to parse
/// @param Error Error result info.
/// @param Context Context in which to allocate globals info.
std::unique_ptr<Module> parseAssemblyFile(StringRef Filename,
                                          SMDiagnostic &Error,
                                          LLVMContext &Context);

/// The function is a secondary interface to the LLVM Assembly Parser. It parses
/// an ASCII string that (presumably) contains LLVM Assembly code. It returns a
/// Module (intermediate representation) with the corresponding features. Note
/// that this does not verify that the generated Module is valid, so you should
/// run the verifier after parsing the file to check that it is okay.
/// @brief Parse LLVM Assembly from a string
/// @param AsmString The string containing assembly
/// @param Error Error result info.
/// @param Context Context in which to allocate globals info.
std::unique_ptr<Module> parseAssemblyString(StringRef AsmString,
                                            SMDiagnostic &Error,
                                            LLVMContext &Context);

/// This function is the low-level interface to the LLVM Assembly Parser.
/// ParseAssemblyFile and ParseAssemblyString are wrappers around this function.
/// @brief Parse LLVM Assembly from a MemoryBuffer.
/// @param F The MemoryBuffer containing assembly
/// @param Err Error result info.
/// @param Context Context in which to allocate globals info.
std::unique_ptr<Module> parseAssembly(std::unique_ptr<MemoryBuffer> F,
                                      SMDiagnostic &Err, LLVMContext &Context);

} // End llvm namespace

#endif
