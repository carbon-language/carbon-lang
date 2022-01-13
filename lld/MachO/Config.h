//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_CONFIG_H
#define LLD_MACHO_CONFIG_H

#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/CachePruning.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/TextAPI/Architecture.h"
#include "llvm/TextAPI/Platform.h"
#include "llvm/TextAPI/Target.h"

#include <vector>

namespace lld {
namespace macho {

class Symbol;
struct SymbolPriorityEntry;

using NamePair = std::pair<llvm::StringRef, llvm::StringRef>;
using SectionRenameMap = llvm::DenseMap<NamePair, NamePair>;
using SegmentRenameMap = llvm::DenseMap<llvm::StringRef, llvm::StringRef>;

struct PlatformInfo {
  llvm::MachO::Target target;
  llvm::VersionTuple minimum;
  llvm::VersionTuple sdk;
};

inline uint32_t encodeVersion(const llvm::VersionTuple &version) {
  return ((version.getMajor() << 020) |
          (version.getMinor().getValueOr(0) << 010) |
          version.getSubminor().getValueOr(0));
}

enum class NamespaceKind {
  twolevel,
  flat,
};

enum class UndefinedSymbolTreatment {
  unknown,
  error,
  warning,
  suppress,
  dynamic_lookup,
};

enum class ICFLevel {
  unknown,
  none,
  safe,
  all,
};

struct SectionAlign {
  llvm::StringRef segName;
  llvm::StringRef sectName;
  uint32_t align;
};

struct SegmentProtection {
  llvm::StringRef name;
  uint32_t maxProt;
  uint32_t initProt;
};

class SymbolPatterns {
public:
  // GlobPattern can also match literals,
  // but we prefer the O(1) lookup of DenseSet.
  llvm::DenseSet<llvm::CachedHashStringRef> literals;
  std::vector<llvm::GlobPattern> globs;

  bool empty() const { return literals.empty() && globs.empty(); }
  void clear();
  void insert(llvm::StringRef symbolName);
  bool matchLiteral(llvm::StringRef symbolName) const;
  bool matchGlob(llvm::StringRef symbolName) const;
  bool match(llvm::StringRef symbolName) const;
};

struct Configuration {
  Symbol *entry = nullptr;
  bool hasReexports = false;
  bool allLoad = false;
  bool applicationExtension = false;
  bool archMultiple = false;
  bool exportDynamic = false;
  bool forceLoadObjC = false;
  bool forceLoadSwift = false;
  bool staticLink = false;
  bool implicitDylibs = false;
  bool isPic = false;
  bool headerPadMaxInstallNames = false;
  bool ltoNewPassManager = LLVM_ENABLE_NEW_PASS_MANAGER;
  bool markDeadStrippableDylib = false;
  bool printDylibSearch = false;
  bool printEachFile = false;
  bool printWhyLoad = false;
  bool searchDylibsFirst = false;
  bool saveTemps = false;
  bool adhocCodesign = false;
  bool emitFunctionStarts = false;
  bool emitBitcodeBundle = false;
  bool emitDataInCodeInfo = false;
  bool emitEncryptionInfo = false;
  bool timeTraceEnabled = false;
  bool dataConst = false;
  bool dedupLiterals = true;
  bool omitDebugInfo = false;
  bool warnDylibInstallName = false;
  uint32_t headerPad;
  uint32_t dylibCompatibilityVersion = 0;
  uint32_t dylibCurrentVersion = 0;
  uint32_t timeTraceGranularity = 500;
  unsigned optimize;
  std::string progName;

  // For `clang -arch arm64 -arch x86_64`, clang will:
  // 1. invoke the linker twice, to write one temporary output per arch
  // 2. invoke `lipo` to merge the two outputs into a single file
  // `outputFile` is the name of the temporary file the linker writes to.
  // `finalOutput `is the name of the file lipo writes to after the link.
  llvm::StringRef outputFile;
  llvm::StringRef finalOutput;

  llvm::StringRef installName;
  llvm::StringRef mapFile;
  llvm::StringRef ltoObjPath;
  llvm::StringRef thinLTOJobs;
  llvm::StringRef umbrella;
  uint32_t ltoo = 2;
  llvm::CachePruningPolicy thinLTOCachePolicy;
  llvm::StringRef thinLTOCacheDir;
  bool deadStripDylibs = false;
  bool demangle = false;
  bool deadStrip = false;
  bool errorForArchMismatch = false;
  PlatformInfo platformInfo;
  NamespaceKind namespaceKind = NamespaceKind::twolevel;
  UndefinedSymbolTreatment undefinedSymbolTreatment =
      UndefinedSymbolTreatment::error;
  ICFLevel icfLevel = ICFLevel::none;
  llvm::MachO::HeaderFileType outputType;
  std::vector<llvm::StringRef> systemLibraryRoots;
  std::vector<llvm::StringRef> librarySearchPaths;
  std::vector<llvm::StringRef> frameworkSearchPaths;
  std::vector<llvm::StringRef> runtimePaths;
  std::vector<std::string> astPaths;
  std::vector<Symbol *> explicitUndefineds;
  llvm::StringSet<> explicitDynamicLookups;
  // There are typically few custom sectionAlignments or segmentProtections,
  // so use a vector instead of a map.
  std::vector<SectionAlign> sectionAlignments;
  std::vector<SegmentProtection> segmentProtections;

  llvm::DenseMap<llvm::StringRef, SymbolPriorityEntry> priorities;
  SectionRenameMap sectionRenameMap;
  SegmentRenameMap segmentRenameMap;

  SymbolPatterns exportedSymbols;
  SymbolPatterns unexportedSymbols;

  bool zeroModTime = false;

  llvm::StringRef osoPrefix;

  llvm::MachO::Architecture arch() const { return platformInfo.target.Arch; }

  llvm::MachO::PlatformKind platform() const {
    return platformInfo.target.Platform;
  }
};

// The symbol with the highest priority should be ordered first in the output
// section (modulo input section contiguity constraints). Using priority
// (highest first) instead of order (lowest first) has the convenient property
// that the default-constructed zero priority -- for symbols/sections without a
// user-defined order -- naturally ends up putting them at the end of the
// output.
struct SymbolPriorityEntry {
  // The priority given to a matching symbol, regardless of which object file
  // it originated from.
  size_t anyObjectFile = 0;
  // The priority given to a matching symbol from a particular object file.
  llvm::DenseMap<llvm::StringRef, size_t> objectFiles;
};

// Whether to force-load an archive.
enum class ForceLoad {
  Default, // Apply -all_load or -ObjC behaviors if those flags are enabled
  Yes,     // Always load the archive, regardless of other flags
  No,      // Never load the archive, regardless of other flags
};

extern std::unique_ptr<Configuration> config;

} // namespace macho
} // namespace lld

#endif
