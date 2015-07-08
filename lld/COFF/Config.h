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

using llvm::COFF::IMAGE_FILE_MACHINE_UNKNOWN;
using llvm::COFF::WindowsSubsystem;
using llvm::StringRef;
class Undefined;

// Represents an /export option.
struct Export {
  StringRef Name;
  StringRef ExtName;
  Undefined *Sym = nullptr;
  uint16_t Ordinal = 0;
  bool Noname = false;
  bool Data = false;
  bool Private = false;

  bool operator==(const Export &E) {
    return (Name == E.Name && ExtName == E.ExtName &&
            Ordinal == E.Ordinal && Noname == E.Noname &&
            Data == E.Data && Private == E.Private);
  }
};

// Global configuration.
struct Configuration {
  enum ManifestKind { SideBySide, Embed, No };

  llvm::COFF::MachineTypes MachineType = IMAGE_FILE_MACHINE_UNKNOWN;
  bool Verbose = false;
  WindowsSubsystem Subsystem = llvm::COFF::IMAGE_SUBSYSTEM_UNKNOWN;
  Undefined *Entry = nullptr;
  bool NoEntry = false;
  std::string OutputFile;
  bool DoGC = true;
  bool Relocatable = true;
  bool Force = false;
  bool Debug = false;

  // Symbols in this set are considered as live by the garbage collector.
  std::set<Undefined *> GCRoot;

  std::set<StringRef> NoDefaultLibs;
  bool NoDefaultLibAll = false;

  // True if we are creating a DLL.
  bool DLL = false;
  StringRef Implib;
  std::vector<Export> Exports;
  std::set<std::string> DelayLoads;

  // Used for /opt:icf
  bool ICF = false;

  // Used for /merge:from=to (e.g. /merge:.rdata=.text)
  std::map<StringRef, StringRef> Merge;

  // Options for manifest files.
  ManifestKind Manifest = SideBySide;
  int ManifestID = 1;
  StringRef ManifestDependency;
  bool ManifestUAC = true;
  StringRef ManifestLevel = "'asInvoker'";
  StringRef ManifestUIAccess = "'false'";
  StringRef ManifestFile;

  // Used for /failifmismatch.
  std::map<StringRef, StringRef> MustMatch;

  // Used for /alternatename.
  std::map<StringRef, StringRef> AlternateNames;

  uint64_t ImageBase = 0x140000000U;
  uint64_t StackReserve = 1024 * 1024;
  uint64_t StackCommit = 4096;
  uint64_t HeapReserve = 1024 * 1024;
  uint64_t HeapCommit = 4096;
  uint32_t MajorImageVersion = 0;
  uint32_t MinorImageVersion = 0;
  uint32_t MajorOSVersion = 6;
  uint32_t MinorOSVersion = 0;
  bool DynamicBase = true;
  bool HighEntropyVA = true;
  bool AllowBind = true;
  bool NxCompat = true;
  bool AllowIsolation = true;
  bool TerminalServerAware = true;
};

extern Configuration *Config;

} // namespace coff
} // namespace lld

#endif
