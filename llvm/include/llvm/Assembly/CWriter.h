//===-- llvm/Assembly/CWriter.h - C Printer for LLVM programs ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This functionality is implemented by the lib/CWriter library.  This library
// is used to print C language files to an iostream.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_CWRITER_H
#define LLVM_ASSEMBLY_CWRITER_H

#include <iosfwd>
class Pass;
Pass *createWriteToCPass(std::ostream &o);

#endif
