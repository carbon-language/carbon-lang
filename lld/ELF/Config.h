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
  llvm::StringRef SoName;
  llvm::StringRef Sysroot;
  std::string RPath;
  std::vector<llvm::StringRef> InputSearchPaths;
  bool AllowMultipleDefinition;
  bool DiscardAll;
  bool DiscardLocals;
  bool DiscardNone;
  bool ExportDynamic;
  bool NoInhibitExec;
  bool NoUndefined;
  bool Shared;
  bool Static = false;
  bool WholeArchive = false;
};

extern Configuration *Config;

} // namespace elf2
} // namespace lld

#endif
