//===--- CppWriter.h - Generate C++ IR to C++ Source Interface ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares a function, WriteModuleToCppFile that will convert a 
// Module into the corresponding C++ code to construct the same module.
//
//===------------------------------------------------------------------------===
#include <ostream>
namespace llvm {
class Module;
void WriteModuleToCppFile(Module* mod, std::ostream& out);
}
