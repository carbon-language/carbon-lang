//===-- llvm/CGen/Writer.h - Printer for VM assembly files -------*- C++ -*--=//
//
// This functionality is implemented by the lib/CWriter library.
// This library is used to print C language files to an iostream. 
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_C_WRITER_H
#define LLVM_C_WRITER_H

class Module;

#include <iosfwd>

void WriteToC(const Module  *Module, std::ostream &o);

#endif



