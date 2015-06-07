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
#include <map>
#include <set>
#include <string>

namespace lld {
namespace coff {

using llvm::COFF::WindowsSubsystem;
using llvm::StringRef;

class Configuration {
public:
  llvm::COFF::MachineTypes MachineType = llvm::COFF::IMAGE_FILE_MACHINE_AMD64;
  bool Verbose = false;
  WindowsSubsystem Subsystem = llvm::COFF::IMAGE_SUBSYSTEM_UNKNOWN;
  StringRef EntryName;
  std::string OutputFile;
  bool DoGC = true;

  // Symbols in this set are considered as live by the garbage collector.
  std::set<StringRef> GCRoots;

  std::set<StringRef> NoDefaultLibs;
  bool NoDefaultLibAll = false;

  // Used by /failifmismatch option.
  std::map<StringRef, StringRef> MustMatch;

  uint64_t ImageBase = 0x140000000;
  uint64_t StackReserve = 1024 * 1024;
  uint64_t StackCommit = 4096;
  uint64_t HeapReserve = 1024 * 1024;
  uint64_t HeapCommit = 4096;
  uint32_t MajorImageVersion = 0;
  uint32_t MinorImageVersion = 0;
  uint32_t MajorOSVersion = 6;
  uint32_t MinorOSVersion = 0;
};

extern Configuration *Config;

} // namespace coff
} // namespace lld

#endif
