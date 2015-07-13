//===- Config.h -----------------------------------------------------------===//
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
#include "llvm/Object/ELF.h"
#include <set>
#include <string>

namespace lld {
namespace elfv2 {

using llvm::StringRef;

class Configuration {
public:
  int MachineArchitecture = llvm::ELF::EM_X86_64;
  bool Verbose = false;
  StringRef EntryName;
  std::string OutputFile;
  bool DoGC = true;

  // Symbols in this set are considered as live by the garbage collector.
  std::set<StringRef> GCRoots;
};

extern Configuration *Config;

} // namespace elfv2
} // namespace lld

#endif
