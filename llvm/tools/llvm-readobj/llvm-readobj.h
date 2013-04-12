//===-- llvm-readobj.h ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_READ_OBJ_H
#define LLVM_TOOLS_READ_OBJ_H

#include "llvm/Support/CommandLine.h"
#include <string>

namespace llvm {
  namespace object {
    class RelocationRef;
  }

  class error_code;

  // Various helper functions.
  bool error(error_code ec);
  bool relocAddressLess(object::RelocationRef A,
                        object::RelocationRef B);
} // namespace llvm

namespace opts {
  extern llvm::cl::list<std::string> InputFilenames;
  extern llvm::cl::opt<bool> FileHeaders;
  extern llvm::cl::opt<bool> Sections;
  extern llvm::cl::opt<bool> SectionRelocations;
  extern llvm::cl::opt<bool> SectionSymbols;
  extern llvm::cl::opt<bool> SectionData;
  extern llvm::cl::opt<bool> Relocations;
  extern llvm::cl::opt<bool> Symbols;
  extern llvm::cl::opt<bool> DynamicSymbols;
  extern llvm::cl::opt<bool> UnwindInfo;
  extern llvm::cl::opt<bool> ExpandRelocs;
} // namespace opts

#define LLVM_READOBJ_ENUM_ENT(ns, enum) \
  { #enum, ns::enum }

#endif
