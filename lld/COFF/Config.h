//===- Config.h -----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_CONFIG_H
#define LLD_COFF_CONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/COFF.h"
#include <cstdint>
#include <set>
#include <string>

namespace lld {
namespace coff {

class Configuration {
public:
  llvm::COFF::MachineTypes MachineType = llvm::COFF::IMAGE_FILE_MACHINE_AMD64;
  bool Verbose = false;
  std::string EntryName = "mainCRTStartup";

  uint64_t ImageBase = 0x140000000;
  uint64_t StackReserve = 1024 * 1024;
  uint64_t StackCommit = 4096;
  uint64_t HeapReserve = 1024 * 1024;
  uint64_t HeapCommit = 4096;

  bool insertFile(llvm::StringRef Path) {
    return VisitedFiles.insert(Path.lower()).second;
  }

private:
  std::set<std::string> VisitedFiles;
};

extern Configuration *Config;

} // namespace coff
} // namespace lld

#endif
