//===- Driver.h -------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_DRIVER_H
#define LLD_ELF_DRIVER_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"

namespace lld {
namespace elf2 {

class SymbolTable;

// The owner of all opened files.
extern std::vector<std::unique_ptr<MemoryBuffer>> *MemoryBufferPool;
MemoryBufferRef openFile(StringRef Path);

// Entry point of the ELF linker.
void link(ArrayRef<const char *> Args);

class ArgParser {
public:
  // Parses command line options.
  llvm::opt::InputArgList parse(ArrayRef<const char *> Args);

private:
  llvm::BumpPtrAllocator Alloc;
};

class LinkerDriver {
public:
  void link(ArrayRef<const char *> Args);

private:
  ArgParser Parser;
};

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

// Parses a linker script. Calling this function updates the Symtab and Config.
void readLinkerScript(SymbolTable *Symtab, MemoryBufferRef MB);

} // namespace elf2
} // namespace lld

#endif
