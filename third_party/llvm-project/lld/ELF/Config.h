//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_CONFIG_H
#define LLD_ELF_CONFIG_H

#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/CachePruning.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/PrettyStackTrace.h"
#include <atomic>
#include <memory>
#include <vector>

namespace lld {
namespace elf {

class InputFile;
class InputSectionBase;

enum ELFKind : uint8_t {
  ELFNoneKind,
  ELF32LEKind,
  ELF32BEKind,
  ELF64LEKind,
  ELF64BEKind
};

// For -Bno-symbolic, -Bsymbolic-non-weak-functions, -Bsymbolic-functions,
// -Bsymbolic.
enum class BsymbolicKind { None, NonWeakFunctions, Functions, All };

// For --build-id.
enum class BuildIdKind { None, Fast, Md5, Sha1, Hexstring, Uuid };

// For --discard-{all,locals,none}.
enum class DiscardPolicy { Default, All, Locals, None };

// For --icf={none,safe,all}.
enum class ICFLevel { None, Safe, All };

// For --strip-{all,debug}.
enum class StripPolicy { None, All, Debug };

// For --unresolved-symbols.
enum class UnresolvedPolicy { ReportError, Warn, Ignore };

// For --orphan-handling.
enum class OrphanHandlingPolicy { Place, Warn, Error };

// For --sort-section and linkerscript sorting rules.
enum class SortSectionPolicy { Default, None, Alignment, Name, Priority };

// For --target2
enum class Target2Policy { Abs, Rel, GotRel };

// For tracking ARM Float Argument PCS
enum class ARMVFPArgKind { Default, Base, VFP, ToolChain };

// For -z noseparate-code, -z separate-code and -z separate-loadable-segments.
enum class SeparateSegmentKind { None, Code, Loadable };

// For -z *stack
enum class GnuStackKind { None, Exec, NoExec };

struct SymbolVersion {
  llvm::StringRef name;
  bool isExternCpp;
  bool hasWildcard;
};

// This struct contains symbols version definition that
// can be found in version script if it is used for link.
struct VersionDefinition {
  llvm::StringRef name;
  uint16_t id;
  SmallVector<SymbolVersion, 0> nonLocalPatterns;
  SmallVector<SymbolVersion, 0> localPatterns;
};

// This struct contains the global configuration for the linker.
// Most fields are direct mapping from the command line options
// and such fields have the same name as the corresponding options.
// Most fields are initialized by the driver.
struct Configuration {
  uint8_t osabi = 0;
  uint32_t andFeatures = 0;
  llvm::CachePruningPolicy thinLTOCachePolicy;
  llvm::SetVector<llvm::CachedHashString> dependencyFiles; // for --dependency-file
  llvm::StringMap<uint64_t> sectionStartMap;
  llvm::StringRef bfdname;
  llvm::StringRef chroot;
  llvm::StringRef dependencyFile;
  llvm::StringRef dwoDir;
  llvm::StringRef dynamicLinker;
  llvm::StringRef entry;
  llvm::StringRef emulation;
  llvm::StringRef fini;
  llvm::StringRef init;
  llvm::StringRef ltoAAPipeline;
  llvm::StringRef ltoCSProfileFile;
  llvm::StringRef ltoNewPmPasses;
  llvm::StringRef ltoObjPath;
  llvm::StringRef ltoSampleProfile;
  llvm::StringRef mapFile;
  llvm::StringRef outputFile;
  llvm::StringRef optRemarksFilename;
  llvm::Optional<uint64_t> optRemarksHotnessThreshold = 0;
  llvm::StringRef optRemarksPasses;
  llvm::StringRef optRemarksFormat;
  llvm::StringRef progName;
  llvm::StringRef printArchiveStats;
  llvm::StringRef printSymbolOrder;
  llvm::StringRef soName;
  llvm::StringRef sysroot;
  llvm::StringRef thinLTOCacheDir;
  llvm::StringRef thinLTOIndexOnlyArg;
  llvm::StringRef whyExtract;
  StringRef zBtiReport = "none";
  StringRef zCetReport = "none";
  llvm::StringRef ltoBasicBlockSections;
  std::pair<llvm::StringRef, llvm::StringRef> thinLTOObjectSuffixReplace;
  std::pair<llvm::StringRef, llvm::StringRef> thinLTOPrefixReplace;
  std::string rpath;
  std::vector<VersionDefinition> versionDefinitions;
  std::vector<llvm::StringRef> auxiliaryList;
  std::vector<llvm::StringRef> filterList;
  std::vector<llvm::StringRef> searchPaths;
  std::vector<llvm::StringRef> symbolOrderingFile;
  std::vector<llvm::StringRef> thinLTOModulesToCompile;
  std::vector<llvm::StringRef> undefined;
  std::vector<SymbolVersion> dynamicList;
  std::vector<uint8_t> buildIdVector;
  llvm::MapVector<std::pair<const InputSectionBase *, const InputSectionBase *>,
                  uint64_t>
      callGraphProfile;
  bool allowMultipleDefinition;
  bool androidPackDynRelocs;
  bool armHasBlx = false;
  bool armHasMovtMovw = false;
  bool armJ1J2BranchEncoding = false;
  bool asNeeded = false;
  BsymbolicKind bsymbolic = BsymbolicKind::None;
  bool callGraphProfileSort;
  bool checkSections;
  bool checkDynamicRelocs;
  bool compressDebugSections;
  bool cref;
  std::vector<std::pair<llvm::GlobPattern, uint64_t>> deadRelocInNonAlloc;
  bool demangle = true;
  bool dependentLibraries;
  bool disableVerify;
  bool ehFrameHdr;
  bool emitLLVM;
  bool emitRelocs;
  bool enableNewDtags;
  bool executeOnly;
  bool exportDynamic;
  bool fixCortexA53Errata843419;
  bool fixCortexA8;
  bool formatBinary = false;
  bool fortranCommon;
  bool gcSections;
  bool gdbIndex;
  bool gnuHash = false;
  bool gnuUnique;
  bool hasDynSymTab;
  bool ignoreDataAddressEquality;
  bool ignoreFunctionAddressEquality;
  bool ltoCSProfileGenerate;
  bool ltoPGOWarnMismatch;
  bool ltoDebugPassManager;
  bool ltoEmitAsm;
  bool ltoNewPassManager;
  bool ltoUniqueBasicBlockSectionNames;
  bool ltoWholeProgramVisibility;
  bool mergeArmExidx;
  bool mipsN32Abi = false;
  bool mmapOutputFile;
  bool nmagic;
  bool noDynamicLinker = false;
  bool noinhibitExec;
  bool nostdlib;
  bool oFormatBinary;
  bool omagic;
  bool optEB = false;
  bool optEL = false;
  bool optimizeBBJumps;
  bool optRemarksWithHotness;
  bool picThunk;
  bool pie;
  bool printGcSections;
  bool printIcfSections;
  bool relax;
  bool relocatable;
  bool relrPackDynRelocs;
  bool saveTemps;
  std::vector<std::pair<llvm::GlobPattern, uint32_t>> shuffleSections;
  bool singleRoRx;
  bool shared;
  bool symbolic;
  bool isStatic = false;
  bool sysvHash = false;
  bool target1Rel;
  bool trace;
  bool thinLTOEmitImportsFiles;
  bool thinLTOIndexOnly;
  bool timeTraceEnabled;
  bool tocOptimize;
  bool pcRelOptimize;
  bool undefinedVersion;
  bool unique;
  bool useAndroidRelrTags = false;
  bool warnBackrefs;
  std::vector<llvm::GlobPattern> warnBackrefsExclude;
  bool warnCommon;
  bool warnMissingEntry;
  bool warnSymbolOrdering;
  bool writeAddends;
  bool zCombreloc;
  bool zCopyreloc;
  bool zForceBti;
  bool zForceIbt;
  bool zGlobal;
  bool zHazardplt;
  bool zIfuncNoplt;
  bool zInitfirst;
  bool zInterpose;
  bool zKeepTextSectionPrefix;
  bool zNodefaultlib;
  bool zNodelete;
  bool zNodlopen;
  bool zNow;
  bool zOrigin;
  bool zPacPlt;
  bool zRelro;
  bool zRodynamic;
  bool zShstk;
  bool zStartStopGC;
  uint8_t zStartStopVisibility;
  bool zText;
  bool zRetpolineplt;
  bool zWxneeded;
  DiscardPolicy discard;
  GnuStackKind zGnustack;
  ICFLevel icf;
  OrphanHandlingPolicy orphanHandling;
  SortSectionPolicy sortSection;
  StripPolicy strip;
  UnresolvedPolicy unresolvedSymbols;
  UnresolvedPolicy unresolvedSymbolsInShlib;
  Target2Policy target2;
  bool power10Stubs;
  ARMVFPArgKind armVFPArgs = ARMVFPArgKind::Default;
  BuildIdKind buildId = BuildIdKind::None;
  SeparateSegmentKind zSeparate;
  ELFKind ekind = ELFNoneKind;
  uint16_t emachine = llvm::ELF::EM_NONE;
  llvm::Optional<uint64_t> imageBase;
  uint64_t commonPageSize;
  uint64_t maxPageSize;
  uint64_t mipsGotSize;
  uint64_t zStackSize;
  unsigned ltoPartitions;
  unsigned ltoo;
  unsigned optimize;
  StringRef thinLTOJobs;
  unsigned timeTraceGranularity;
  int32_t splitStackAdjustSize;

  // The following config options do not directly correspond to any
  // particular command line options.

  // True if we need to pass through relocations in input files to the
  // output file. Usually false because we consume relocations.
  bool copyRelocs;

  // True if the target is ELF64. False if ELF32.
  bool is64;

  // True if the target is little-endian. False if big-endian.
  bool isLE;

  // endianness::little if isLE is true. endianness::big otherwise.
  llvm::support::endianness endianness;

  // True if the target is the little-endian MIPS64.
  //
  // The reason why we have this variable only for the MIPS is because
  // we use this often.  Some ELF headers for MIPS64EL are in a
  // mixed-endian (which is horrible and I'd say that's a serious spec
  // bug), and we need to know whether we are reading MIPS ELF files or
  // not in various places.
  //
  // (Note that MIPS64EL is not a typo for MIPS64LE. This is the official
  // name whatever that means. A fun hypothesis is that "EL" is short for
  // little-endian written in the little-endian order, but I don't know
  // if that's true.)
  bool isMips64EL;

  // True if we need to reserve two .got entries for local-dynamic TLS model.
  bool needsTlsLd = false;

  // True if we need to set the DF_STATIC_TLS flag to an output file, which
  // works as a hint to the dynamic loader that the shared object contains code
  // compiled with the initial-exec TLS model.
  bool hasTlsIe = false;

  // Holds set of ELF header flags for the target.
  uint32_t eflags = 0;

  // The ELF spec defines two types of relocation table entries, RELA and
  // REL. RELA is a triplet of (offset, info, addend) while REL is a
  // tuple of (offset, info). Addends for REL are implicit and read from
  // the location where the relocations are applied. So, REL is more
  // compact than RELA but requires a bit of more work to process.
  //
  // (From the linker writer's view, this distinction is not necessary.
  // If the ELF had chosen whichever and sticked with it, it would have
  // been easier to write code to process relocations, but it's too late
  // to change the spec.)
  //
  // Each ABI defines its relocation type. IsRela is true if target
  // uses RELA. As far as we know, all 64-bit ABIs are using RELA. A
  // few 32-bit ABIs are using RELA too.
  bool isRela;

  // True if we are creating position-independent code.
  bool isPic;

  // 4 for ELF32, 8 for ELF64.
  int wordsize;
};

// The only instance of Configuration struct.
extern std::unique_ptr<Configuration> config;

// The first two elements of versionDefinitions represent VER_NDX_LOCAL and
// VER_NDX_GLOBAL. This helper returns other elements.
static inline ArrayRef<VersionDefinition> namedVersionDefs() {
  return llvm::makeArrayRef(config->versionDefinitions).slice(2);
}

void errorOrWarn(const Twine &msg);

static inline void internalLinkerError(StringRef loc, const Twine &msg) {
  errorOrWarn(loc + "internal linker error: " + msg + "\n" +
              llvm::getBugReportMsg());
}

} // namespace elf
} // namespace lld

#endif
