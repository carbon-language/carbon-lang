//===-- llvm/Bytecode/Writer.h - Writer for VM bytecode files ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This functionality is implemented by the lib/BytecodeWriter library.
// This library is used to write VM bytecode files to an iostream.  First, you
// have to make a BytecodeStream object, which you can then put a class into
// by using operator <<.
//
// This library uses the Analysis library to figure out offsets for
// variables in the method tables...
//
// Note that performance of this library is not as crucial as performance of the
// bytecode reader (which is to be used in JIT type applications), so we have
// designed the bytecode format to support quick reading.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BYTECODE_WRITER_H
#define LLVM_BYTECODE_WRITER_H

#include <iosfwd>

namespace llvm {

class Module;
void WriteBytecodeToFile(const Module *C, std::ostream &Out);

} // End llvm namespace

#endif
