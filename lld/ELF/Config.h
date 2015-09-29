//===- Config.h -------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_CONFIG_H
#define LLD_ELF_CONFIG_H

#include "llvm/ADT/StringRef.h"

#include <vector>

namespace lld {
namespace elf2 {

struct Configuration {
  llvm::StringRef DynamicLinker;
  llvm::StringRef Entry;
  llvm::StringRef OutputFile = "a.out";
  llvm::StringRef Sysroot;
  std::string RPath;
  std::vector<llvm::StringRef> InputSearchPaths;
  bool AllowMultipleDefinition = false;
  bool DiscardAll = false;
  bool DiscardLocals = false;
  bool DiscardNone = false;
  bool ExportDynamic = false;
  bool NoInhibitExec = false;
  bool Shared = false;
};

extern Configuration *Config;

} // namespace elf2
} // namespace lld

#endif
