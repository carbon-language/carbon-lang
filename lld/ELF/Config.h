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
#include "llvm/Support/ELF.h"

#include <vector>

namespace lld {
namespace elf2 {

enum ELFKind {
  ELFNoneKind,
  ELF32LEKind,
  ELF32BEKind,
  ELF64LEKind,
  ELF64BEKind
};

struct Configuration {
  llvm::StringRef DynamicLinker;
  llvm::StringRef Entry;
  llvm::StringRef Fini;
  llvm::StringRef Init;
  llvm::StringRef OutputFile;
  llvm::StringRef SoName;
  llvm::StringRef Sysroot;
  std::string RPath;
  std::vector<llvm::StringRef> InputSearchPaths;
  bool AllowMultipleDefinition;
  bool DiscardAll;
  bool DiscardLocals;
  bool DiscardNone;
  bool EnableNewDtags;
  bool ExportDynamic;
  bool NoInhibitExec;
  bool NoUndefined;
  bool ZNow = false;
  bool Shared;
  bool Static = false;
  bool WholeArchive = false;
  ELFKind ElfKind = ELFNoneKind;
  uint16_t EMachine = llvm::ELF::EM_NONE;
};

extern Configuration *Config;

} // namespace elf2
} // namespace lld

#endif
