//===- IRPrintingPasses.h - Passes to print out IR constructs ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines passes to print out IR in various granularities. The
/// PrintModulePass pass simply prints out the entire module when it is
/// executed. The PrintFunctionPass class is designed to be pipelined with
/// other FunctionPass's, and prints out the functions of the module as they
/// are processed.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PRINTMODULEPASS_H
#define LLVM_IR_PRINTMODULEPASS_H

#include <string>

namespace llvm {
class FunctionPass;
class ModulePass;
class BasicBlockPass;
class raw_ostream;

/// \brief Create and return a pass that writes the module to the specified
/// \c raw_ostream.
ModulePass *createPrintModulePass(raw_ostream &OS,
                                  const std::string &Banner = "");

/// \brief Create and return a pass that prints functions to the specified
/// \c raw_ostream as they are processed.
FunctionPass *createPrintFunctionPass(raw_ostream &OS,
                                      const std::string &Banner = "");

/// \brief Create and return a pass that writes the BB to the specified
/// \c raw_ostream.
BasicBlockPass *createPrintBasicBlockPass(raw_ostream &OS,
                                          const std::string &Banner = "");

} // End llvm namespace

#endif
