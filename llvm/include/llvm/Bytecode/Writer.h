//===-- llvm/Bytecode/Writer.h - Writer for VM bytecode files ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This functionality is implemented by the lib/Bytecode/Writer library.
// This library is used to write bytecode files to an iostream.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BYTECODE_WRITER_H
#define LLVM_BYTECODE_WRITER_H

#include <iosfwd>

namespace llvm {
  class Module;
  /// WriteBytecodeToFile - Write the specified module to the specified output
  /// stream.  If compress is set to true, try to use compression when writing
  /// out the file.  This can never fail if M is a well-formed module.
  void WriteBytecodeToFile(const Module *M, std::ostream &Out,
                           bool compress = true);
} // End llvm namespace

#endif
