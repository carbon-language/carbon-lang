//===- llvm/Transforms/Utils/Linker.h - Module Linker Interface -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the interface to the module linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMATIONS_UTILS_LINKER_H
#define LLVM_TRANSFORMATIONS_UTILS_LINKER_H

#include <string>

namespace llvm {

class Module;

// LinkModules - This function links two modules together, with the resulting
// left module modified to be the composite of the two input modules.  If an
// error occurs, true is returned and ErrorMsg (if not null) is set to indicate
// the problem.
//
bool LinkModules(Module *Dest, const Module *Src, std::string *ErrorMsg = 0);

} // End llvm namespace

#endif

