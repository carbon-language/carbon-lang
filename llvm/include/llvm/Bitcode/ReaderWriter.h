//===-- llvm/Bitcode/ReaderWriter.h - Bitcode reader/writers ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces to read and write LLVM bitcode files/streams.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_H
#define LLVM_BITCODE_H

#include <iosfwd>
#include <string>

namespace llvm {
  class Module;
  class ModuleProvider;
  
  ModuleProvider *getBitcodeModuleProvider(const std::string &Filename,
                                           std::string *ErrMsg = 0);

  
  /// ParseBitcodeFile - Read the specified bitcode file, returning the module.
  /// If an error occurs, return null and fill in *ErrMsg if non-null.
  Module *ParseBitcodeFile(const std::string &Filename,
                           std::string *ErrMsg = 0);
  
  /// WriteBitcodeToFile - Write the specified module to the specified output
  /// stream.
  void WriteBitcodeToFile(const Module *M, std::ostream &Out);
} // End llvm namespace

#endif
