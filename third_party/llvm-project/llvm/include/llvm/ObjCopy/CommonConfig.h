//===- CommonConfig.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJCOPY_COMMONCONFIG_H
#define LLVM_OBJCOPY_COMMONCONFIG_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/Regex.h"
// Necessary for llvm::DebugCompressionType::None
#include "llvm/Target/TargetOptions.h"
#include <vector>

namespace llvm {
namespace objcopy {

enum class FileFormat {
  Unspecified,
  ELF,
  Binary,
  IHex,
};

// This type keeps track of the machine info for various architectures. This
// lets us map architecture names to ELF types and the e_machine value of the
// ELF file.
struct MachineInfo {
  MachineInfo(uint16_t EM, uint8_t ABI, bool Is64, bool IsLittle)
      : EMachine(EM), OSABI(ABI), Is64Bit(Is64), IsLittleEndian(IsLittle) {}
  // Alternative constructor that defaults to NONE for OSABI.
  MachineInfo(uint16_t EM, bool Is64, bool IsLittle)
      : MachineInfo(EM, ELF::ELFOSABI_NONE, Is64, IsLittle) {}
  // Default constructor for unset fields.
  MachineInfo() : MachineInfo(0, 0, false, false) {}
  uint16_t EMachine;
  uint8_t OSABI;
  bool Is64Bit;
  bool IsLittleEndian;
};

// Flags set by --set-section-flags or --rename-section. Interpretation of these
// is format-specific and not all flags are meaningful for all object file
// formats. This is a bitmask; many section flags may be set.
enum SectionFlag {
  SecNone = 0,
  SecAlloc = 1 << 0,
  SecLoad = 1 << 1,
  SecNoload = 1 << 2,
  SecReadonly = 1 << 3,
  SecDebug = 1 << 4,
  SecCode = 1 << 5,
  SecData = 1 << 6,
  SecRom = 1 << 7,
  SecMerge = 1 << 8,
  SecStrings = 1 << 9,
  SecContents = 1 << 10,
  SecShare = 1 << 11,
  SecExclude = 1 << 12,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/SecExclude)
};

struct SectionRename {
  StringRef OriginalName;
  StringRef NewName;
  Optional<SectionFlag> NewFlags;
};

struct SectionFlagsUpdate {
  StringRef Name;
  SectionFlag NewFlags;
};

enum class DiscardType {
  None,   // Default
  All,    // --discard-all (-x)
  Locals, // --discard-locals (-X)
};

enum class MatchStyle {
  Literal,  // Default for symbols.
  Wildcard, // Default for sections, or enabled with --wildcard (-w).
  Regex,    // Enabled with --regex.
};

class NameOrPattern {
  StringRef Name;
  // Regex is shared between multiple CommonConfig instances.
  std::shared_ptr<Regex> R;
  std::shared_ptr<GlobPattern> G;
  bool IsPositiveMatch = true;

  NameOrPattern(StringRef N) : Name(N) {}
  NameOrPattern(std::shared_ptr<Regex> R) : R(R) {}
  NameOrPattern(std::shared_ptr<GlobPattern> G, bool IsPositiveMatch)
      : G(G), IsPositiveMatch(IsPositiveMatch) {}

public:
  // ErrorCallback is used to handle recoverable errors. An Error returned
  // by the callback aborts the parsing and is then returned by this function.
  static Expected<NameOrPattern>
  create(StringRef Pattern, MatchStyle MS,
         llvm::function_ref<Error(Error)> ErrorCallback);

  bool isPositiveMatch() const { return IsPositiveMatch; }
  Optional<StringRef> getName() const {
    if (!R && !G)
      return Name;
    return None;
  }
  bool operator==(StringRef S) const {
    return R ? R->match(S) : G ? G->match(S) : Name == S;
  }
  bool operator!=(StringRef S) const { return !operator==(S); }
};

// Matcher that checks symbol or section names against the command line flags
// provided for that option.
class NameMatcher {
  DenseSet<CachedHashStringRef> PosNames;
  std::vector<NameOrPattern> PosPatterns;
  std::vector<NameOrPattern> NegMatchers;

public:
  Error addMatcher(Expected<NameOrPattern> Matcher) {
    if (!Matcher)
      return Matcher.takeError();
    if (Matcher->isPositiveMatch()) {
      if (Optional<StringRef> MaybeName = Matcher->getName())
        PosNames.insert(CachedHashStringRef(*MaybeName));
      else
        PosPatterns.push_back(std::move(*Matcher));
    } else {
      NegMatchers.push_back(std::move(*Matcher));
    }
    return Error::success();
  }
  bool matches(StringRef S) const {
    return (PosNames.contains(CachedHashStringRef(S)) ||
            is_contained(PosPatterns, S)) &&
           !is_contained(NegMatchers, S);
  }
  bool empty() const {
    return PosNames.empty() && PosPatterns.empty() && NegMatchers.empty();
  }
};

enum class SymbolFlag {
  Global,
  Local,
  Weak,
  Default,
  Hidden,
  Protected,
  File,
  Section,
  Object,
  Function,
  IndirectFunction,
  Debug,
  Constructor,
  Warning,
  Indirect,
  Synthetic,
  UniqueObject,
};

// Symbol info specified by --add-symbol option. Symbol flags not supported
// by a concrete format should be ignored.
struct NewSymbolInfo {
  StringRef SymbolName;
  StringRef SectionName;
  uint64_t Value = 0;
  std::vector<SymbolFlag> Flags;
  std::vector<StringRef> BeforeSyms;
};

// Configuration for copying/stripping a single file.
struct CommonConfig {
  // Main input/output options
  StringRef InputFilename;
  FileFormat InputFormat = FileFormat::Unspecified;
  StringRef OutputFilename;
  FileFormat OutputFormat = FileFormat::Unspecified;

  // Only applicable when --output-format!=binary (e.g. elf64-x86-64).
  Optional<MachineInfo> OutputArch;

  // Advanced options
  StringRef AddGnuDebugLink;
  // Cached gnu_debuglink's target CRC
  uint32_t GnuDebugLinkCRC32;
  Optional<StringRef> ExtractPartition;
  StringRef SplitDWO;
  StringRef SymbolsPrefix;
  StringRef AllocSectionsPrefix;
  DiscardType DiscardMode = DiscardType::None;

  // Repeated options
  std::vector<StringRef> AddSection;
  std::vector<StringRef> DumpSection;
  std::vector<StringRef> UpdateSection;

  // Section matchers
  NameMatcher KeepSection;
  NameMatcher OnlySection;
  NameMatcher ToRemove;

  // Symbol matchers
  NameMatcher SymbolsToGlobalize;
  NameMatcher SymbolsToKeep;
  NameMatcher SymbolsToLocalize;
  NameMatcher SymbolsToRemove;
  NameMatcher UnneededSymbolsToRemove;
  NameMatcher SymbolsToWeaken;
  NameMatcher SymbolsToKeepGlobal;

  // Map options
  StringMap<SectionRename> SectionsToRename;
  StringMap<uint64_t> SetSectionAlignment;
  StringMap<SectionFlagsUpdate> SetSectionFlags;
  StringMap<StringRef> SymbolsToRename;

  // Symbol info specified by --add-symbol option.
  std::vector<NewSymbolInfo> SymbolsToAdd;

  // Boolean options
  bool DeterministicArchives = true;
  bool ExtractDWO = false;
  bool ExtractMainPartition = false;
  bool OnlyKeepDebug = false;
  bool PreserveDates = false;
  bool StripAll = false;
  bool StripAllGNU = false;
  bool StripDWO = false;
  bool StripDebug = false;
  bool StripNonAlloc = false;
  bool StripSections = false;
  bool StripUnneeded = false;
  bool Weaken = false;
  bool DecompressDebugSections = false;

  DebugCompressionType CompressionType = DebugCompressionType::None;
};

} // namespace objcopy
} // namespace llvm

#endif // LLVM_OBJCOPY_COMMONCONFIG_H
