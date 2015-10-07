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

#include "SymbolTable.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"

namespace lld {
namespace elf2 {

extern class LinkerDriver *Driver;

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
  void addFile(StringRef Path);

private:
  template <template <class> class T>
  std::unique_ptr<ELFFileBase> createELFInputFile(MemoryBufferRef MB);

  SymbolTable Symtab;
  ArgParser Parser;
  std::vector<std::unique_ptr<MemoryBuffer>> OwningMBs;
};

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

// Parses a linker script. Calling this function updates the Symtab and Config.
void readLinkerScript(MemoryBufferRef MB);

} // namespace elf2
} // namespace lld

#endif
