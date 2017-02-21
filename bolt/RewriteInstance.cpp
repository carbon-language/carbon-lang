//===--- RewriteInstance.cpp - Interface for machine-level function -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//


#include "BinaryBasicBlock.h"
#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "BinaryPassManager.h"
#include "DataReader.h"
#include "Exceptions.h"
#include "RewriteInstance.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <fstream>
#include <stack>
#include <system_error>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace object;
using namespace bolt;

namespace opts {

extern cl::opt<JumpTableSupportLevel> JumpTables;
extern cl::opt<BinaryFunction::ReorderType> ReorderFunctions;

static cl::opt<std::string>
OutputFilename("o", cl::desc("<output file>"), cl::Required);

// The default verbosity level (0) is pretty terse, level 1 is fairly
// verbose and usually prints some informational message for every
// function processed.  Level 2 is for the noisiest of messages and
// often prints a message per basic block.
// Error messages should never be suppressed by the verbosity level.
// Only warnings and info messages should be affected.
//
// The rational behind stream usage is as follows:
// outs() for info and debugging controlled by command line flags.
// errs() for errors and warnings.
// dbgs() for output within DEBUG().
cl::opt<unsigned>
Verbosity("v",
          cl::desc("set verbosity level for diagnostic output"),
          cl::init(0),
          cl::ZeroOrMore);

static cl::opt<unsigned>
AlignFunctions("align-functions",
               cl::desc("align functions at a given value"),
               cl::init(64),
               cl::ZeroOrMore);

static cl::opt<unsigned>
AlignFunctionsMaxBytes("align-functions-max-bytes",
               cl::desc("maximum number of bytes to use to align functions"),
               cl::init(7),
               cl::ZeroOrMore);

static cl::list<std::string>
BreakFunctionNames("break-funcs",
                   cl::CommaSeparated,
                   cl::desc("list of functions to core dump on (debugging)"),
                   cl::value_desc("func1,func2,func3,..."),
                   cl::Hidden);

cl::opt<bool>
UseOldText("use-old-text",
           cl::desc("re-use space in old .text if possible"),
           cl::Hidden);

cl::opt<bool>
TrapOldCode("trap-old-code",
            cl::desc("insert traps in old function bodies"),
            cl::Hidden);

cl::opt<bool>
PrintDynoStats("dyno-stats",
               cl::desc("print execution info based on profile"));

cl::opt<bool>
DynoStatsAll("dyno-stats-all", cl::desc("print dyno stats after each stage"),
             cl::ZeroOrMore,
             cl::Hidden);

static cl::opt<unsigned>
TopCalledLimit("top-called-limit",
               cl::desc("maximum number of functions to print in top called "
                        "functions section"),
               cl::init(100),
               cl::ZeroOrMore,
               cl::Hidden);

cl::opt<bool>
HotText("hot-text",
        cl::desc("hot text symbols support"),
        cl::ZeroOrMore);

static cl::list<std::string>
FunctionNames("funcs",
              cl::CommaSeparated,
              cl::desc("list of functions to optimize"),
              cl::value_desc("func1,func2,func3,..."));

static cl::opt<std::string>
FunctionNamesFile("funcs-file",
                  cl::desc("file with list of functions to optimize"));

cl::opt<bool>
Relocs("relocs",
       cl::desc("relocation support (experimental)"),
       cl::ZeroOrMore);

static cl::list<std::string>
FunctionPadSpec("pad-funcs",
                cl::CommaSeparated,
                cl::desc("list of functions to pad with amount of bytes"),
                cl::value_desc("func1:pad1,func2:pad2,func3:pad3,..."));

static cl::list<std::string>
SkipFunctionNames("skip-funcs",
                  cl::CommaSeparated,
                  cl::desc("list of functions to skip"),
                  cl::value_desc("func1,func2,func3,..."));

static cl::opt<std::string>
SkipFunctionNamesFile("skip-funcs-file",
                      cl::desc("file with list of functions to skip"));
static cl::opt<bool>
MarkFuncs("mark-funcs",
          cl::desc("mark function boundaries with break instruction to make "
                   "sure we accidentally don't cross them"),
          cl::ReallyHidden,
          cl::ZeroOrMore);

static cl::opt<unsigned>
MaxFunctions("max-funcs",
             cl::desc("maximum # of functions to overwrite"),
             cl::ZeroOrMore);

cl::opt<BinaryFunction::SplittingType>
SplitFunctions("split-functions",
               cl::desc("split functions into hot and cold regions"),
               cl::init(BinaryFunction::ST_NONE),
               cl::values(clEnumValN(BinaryFunction::ST_NONE, "0",
                                     "do not split any function"),
                          clEnumValN(BinaryFunction::ST_EH, "1",
                                     "split all landing pads"),
                          clEnumValN(BinaryFunction::ST_LARGE, "2",
                                     "also split if function too large to fit"),
                          clEnumValN(BinaryFunction::ST_ALL, "3",
                                     "split all functions"),
                          clEnumValEnd),
               cl::ZeroOrMore);

cl::opt<bool>
UpdateDebugSections("update-debug-sections",
                    cl::desc("update DWARF debug sections of the executable"),
                    cl::ZeroOrMore);

static cl::opt<bool>
FixDebugInfoLargeFunctions("fix-debuginfo-large-functions",
                           cl::init(true),
                           cl::desc("do another pass if we encounter large "
                                    "functions, to correct their debug info."),
                           cl::ZeroOrMore,
                           cl::ReallyHidden);

static cl::opt<bool>
AlignBlocks("align-blocks",
            cl::desc("try to align BBs inserting nops"),
            cl::ZeroOrMore);

static cl::opt<bool>
UseGnuStack("use-gnu-stack",
            cl::desc("use GNU_STACK program header for new segment"),
            cl::ZeroOrMore);

static cl::opt<bool>
DumpEHFrame("dump-eh-frame", cl::desc("dump parsed .eh_frame (debugging)"),
            cl::ZeroOrMore,
            cl::Hidden);

cl::opt<bool>
PrintAll("print-all", cl::desc("print functions after each stage"),
         cl::ZeroOrMore,
         cl::Hidden);

cl::opt<bool>
DumpDotAll("dump-dot-all",
           cl::desc("dump function CFGs to graphviz format after each stage"),
           cl::ZeroOrMore,
           cl::Hidden);

static cl::opt<bool>
PrintCFG("print-cfg", cl::desc("print functions after CFG construction"),
         cl::ZeroOrMore,
         cl::Hidden);

static cl::opt<bool>
PrintLoopInfo("print-loops", cl::desc("print loop related information"),
              cl::ZeroOrMore,
              cl::Hidden);

static cl::opt<bool>
PrintDisasm("print-disasm", cl::desc("print function after disassembly"),
            cl::ZeroOrMore,
            cl::Hidden);

static cl::opt<bool>
KeepTmp("keep-tmp",
        cl::desc("preserve intermediate .o file"),
        cl::Hidden);

cl::opt<bool>
AllowStripped("allow-stripped",
              cl::desc("allow processing of stripped binaries"),
              cl::Hidden);

// Check against lists of functions from options if we should
// optimize the function with a given name.
bool shouldProcess(const BinaryFunction &Function) {
  if (opts::MaxFunctions && Function.getFunctionNumber() > opts::MaxFunctions)
    return false;

  auto populateFunctionNames = [](cl::opt<std::string> &FunctionNamesFile,
                                  cl::list<std::string> &FunctionNames) {
    assert(!FunctionNamesFile.empty() && "unexpected empty file name");
    std::ifstream FuncsFile(FunctionNamesFile, std::ios::in);
    std::string FuncName;
    while (std::getline(FuncsFile, FuncName)) {
      FunctionNames.push_back(FuncName);
    }
    FunctionNamesFile = "";
  };

  if (!FunctionNamesFile.empty())
    populateFunctionNames(FunctionNamesFile, FunctionNames);

  if (!SkipFunctionNamesFile.empty())
    populateFunctionNames(SkipFunctionNamesFile, SkipFunctionNames);

  bool IsValid = true;
  if (!FunctionNames.empty()) {
    IsValid = false;
    for (auto &Name : FunctionNames) {
      if (Function.hasName(Name)) {
        IsValid = true;
        break;
      }
    }
  }
  if (!IsValid)
    return false;

  if (!SkipFunctionNames.empty()) {
    for (auto &Name : SkipFunctionNames) {
      if (Function.hasName(Name)) {
        IsValid = false;
        break;
      }
    }
  }

  return IsValid;
}

size_t padFunction(const BinaryFunction &Function) {
  static std::map<std::string, size_t> FunctionPadding;

  if (FunctionPadding.empty() && !FunctionPadSpec.empty()) {
    for (auto &Spec : FunctionPadSpec) {
      auto N = Spec.find(':');
      if (N == std::string::npos)
        continue;
      auto Name = Spec.substr(0, N);
      auto Padding = std::stoull(Spec.substr(N+1));
      FunctionPadding[Name] = Padding;
    }
  }

  for (auto &FPI : FunctionPadding) {
    auto Name = FPI.first;
    auto Padding = FPI.second;
    if (Function.hasName(Name)) {
      return Padding;
    }
  }

  return 0;
}

} // namespace opts

constexpr const char *RewriteInstance::DebugSectionsToOverwrite[];

const std::string RewriteInstance::OrgSecPrefix = ".bolt.org";

const std::string RewriteInstance::BOLTSecPrefix = ".bolt";

static void report_error(StringRef Message, std::error_code EC) {
  assert(EC);
  errs() << "BOLT-ERROR: '" << Message << "': " << EC.message() << ".\n";
  exit(1);
}

static void check_error(std::error_code EC, StringRef Message) {
  if (!EC)
    return;
  report_error(Message, EC);
}

uint8_t *ExecutableFileMemoryManager::allocateSection(intptr_t Size,
                                                      unsigned Alignment,
                                                      unsigned SectionID,
                                                      StringRef SectionName,
                                                      bool IsCode,
                                                      bool IsReadOnly) {
  uint8_t *ret;
  if (IsCode) {
    ret = SectionMemoryManager::allocateCodeSection(Size, Alignment,
                                                    SectionID, SectionName);
  } else {
    ret = SectionMemoryManager::allocateDataSection(Size, Alignment,
                                                    SectionID, SectionName,
                                                    IsReadOnly);
  }

  bool IsLocal = false;
  if (SectionName.startswith(".local."))
    IsLocal = true;

  DEBUG(dbgs() << "BOLT: allocating " << (IsLocal ? "local " : "")
               << (IsCode ? "code" : (IsReadOnly ? "read-only data" : "data"))
               << " section : " << SectionName
               << " with size " << Size << ", alignment " << Alignment
               << " at 0x" << ret << "\n");

  SectionMapInfo[SectionName] = SectionInfo(reinterpret_cast<uint64_t>(ret),
                                            Size,
                                            Alignment,
                                            IsCode,
                                            IsReadOnly,
                                            IsLocal,
                                            0,
                                            0,
                                            SectionID);

  return ret;
}

/// Notifier for non-allocatable (note) section.
uint8_t *ExecutableFileMemoryManager::recordNoteSection(
    const uint8_t *Data,
    uintptr_t Size,
    unsigned Alignment,
    unsigned SectionID,
    StringRef SectionName) {
  DEBUG(dbgs() << "BOLT: note section "
               << SectionName
               << " with size " << Size << ", alignment " << Alignment
               << " at 0x"
               << Twine::utohexstr(reinterpret_cast<uint64_t>(Data)) << '\n');
  if (SectionName == ".debug_line") {
    // We need to make a copy of the section contents if we'll need it for
    // a future reference.
    uint8_t *DataCopy = new uint8_t[Size];
    memcpy(DataCopy, Data, Size);
    NoteSectionInfo[SectionName] =
      SectionInfo(reinterpret_cast<uint64_t>(DataCopy),
                  Size,
                  Alignment,
                  /*IsCode=*/false,
                  /*IsReadOnly=*/true,
                  /*IsLocal=*/false,
                  0,
                  0,
                  SectionID);
    return DataCopy;
  } else {
    DEBUG(dbgs() << "BOLT-DEBUG: ignoring section " << SectionName
                 << " in recordNoteSection()\n");
    return nullptr;
  }
}

bool ExecutableFileMemoryManager::finalizeMemory(std::string *ErrMsg) {
  DEBUG(dbgs() << "BOLT: finalizeMemory()\n");
  return SectionMemoryManager::finalizeMemory(ErrMsg);
}

ExecutableFileMemoryManager::~ExecutableFileMemoryManager() {
  for (auto &SII : NoteSectionInfo) {
    delete[] reinterpret_cast<uint8_t *>(SII.second.AllocAddress);
  }
}

namespace {

/// Create BinaryContext for a given architecture \p ArchName and
/// triple \p TripleName.
std::unique_ptr<BinaryContext> createBinaryContext(
    std::string ArchName,
    std::string TripleName,
    const DataReader &DR,
    std::unique_ptr<DWARFContext> DwCtx) {

  std::string Error;

  std::unique_ptr<Triple> TheTriple = llvm::make_unique<Triple>(TripleName);
  const Target *TheTarget = TargetRegistry::lookupTarget(ArchName,
                                                         *TheTriple,
                                                         Error);
  if (!TheTarget) {
    errs() << "BOLT-ERROR: " << Error;
    return nullptr;
  }

  std::unique_ptr<const MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TripleName));
  if (!MRI) {
    errs() << "BOLT-ERROR: no register info for target " << TripleName << "\n";
    return nullptr;
  }

  // Set up disassembler.
  std::unique_ptr<const MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName));
  if (!AsmInfo) {
    errs() << "BOLT-ERROR: no assembly info for target " << TripleName << "\n";
    return nullptr;
  }

  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!STI) {
    errs() << "BOLT-ERROR: no subtarget info for target " << TripleName << "\n";
    return nullptr;
  }

  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII) {
    errs() << "BOLT-ERROR: no instruction info for target " << TripleName << "\n";
    return nullptr;
  }

  std::unique_ptr<MCObjectFileInfo> MOFI =
    llvm::make_unique<MCObjectFileInfo>();
  std::unique_ptr<MCContext> Ctx =
    llvm::make_unique<MCContext>(AsmInfo.get(), MRI.get(), MOFI.get());
  MOFI->InitMCObjectFileInfo(*TheTriple, Reloc::Default,
                             CodeModel::Default, *Ctx);

  std::unique_ptr<MCDisassembler> DisAsm(
    TheTarget->createMCDisassembler(*STI, *Ctx));

  if (!DisAsm) {
    errs() << "BOLT-ERROR: no disassembler for target " << TripleName << "\n";
    return nullptr;
  }

  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));
  if (!MIA) {
    errs() << "BOLT-ERROR: failed to create instruction analysis for target"
           << TripleName << "\n";
    return nullptr;
  }

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> InstructionPrinter(
      TheTarget->createMCInstPrinter(Triple(TripleName), AsmPrinterVariant,
                                     *AsmInfo, *MII, *MRI));
  if (!InstructionPrinter) {
    errs() << "BOLT-ERROR: no instruction printer for target " << TripleName
           << '\n';
    return nullptr;
  }
  InstructionPrinter->setPrintImmHex(true);

  std::unique_ptr<MCCodeEmitter> MCE(
      TheTarget->createMCCodeEmitter(*MII, *MRI, *Ctx));

  // Make sure we don't miss any output on core dumps.
  outs().SetUnbuffered();
  errs().SetUnbuffered();
  dbgs().SetUnbuffered();

  auto BC =
      llvm::make_unique<BinaryContext>(std::move(Ctx),
                                       std::move(DwCtx),
                                       std::move(TheTriple),
                                       TheTarget,
                                       TripleName,
                                       std::move(MCE),
                                       std::move(MOFI),
                                       std::move(AsmInfo),
                                       std::move(MII),
                                       std::move(STI),
                                       std::move(InstructionPrinter),
                                       std::move(MIA),
                                       std::move(MRI),
                                       std::move(DisAsm),
                                       DR);

  return BC;
}

} // namespace

RewriteInstance::RewriteInstance(ELFObjectFileBase *File,
                                 const DataReader &DR)
    : InputFile(File),
      BC(createBinaryContext("x86-64", "x86_64-unknown-linux", DR,
         std::unique_ptr<DWARFContext>(
           new DWARFContextInMemory(*InputFile, nullptr, true)))) {
}

RewriteInstance::~RewriteInstance() {}

void RewriteInstance::reset() {
  BinaryFunctions.clear();
  FileSymRefs.clear();
  auto &DR = BC->DR;
  BC = createBinaryContext("x86-64", "x86_64-unknown-linux", DR,
           std::unique_ptr<DWARFContext>(
             new DWARFContextInMemory(*InputFile, nullptr, true)));
  CFIRdWrt.reset(nullptr);
  EFMM.reset(nullptr);
  Out.reset(nullptr);
  EHFrame = nullptr;
  FailedAddresses.clear();
  RangesSectionsWriter.reset();
  TotalScore = 0;
}

void RewriteInstance::discoverStorage() {

  EFMM.reset(new ExecutableFileMemoryManager());

  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(InputFile);
  if (!ELF64LEFile) {
    errs() << "BOLT-ERROR: only 64-bit LE ELF binaries are supported\n";
    exit(1);
  }
  auto Obj = ELF64LEFile->getELFFile();

  EntryPoint = Obj->getHeader()->e_entry;

  // This is where the first segment and ELF header were allocated.
  uint64_t FirstAllocAddress = std::numeric_limits<uint64_t>::max();

  NextAvailableAddress = 0;
  uint64_t NextAvailableOffset = 0;
  for (const auto &Phdr : Obj->program_headers()) {
    if (Phdr.p_type == ELF::PT_LOAD) {
      FirstAllocAddress = std::min(FirstAllocAddress,
                                   static_cast<uint64_t>(Phdr.p_vaddr));
      NextAvailableAddress = std::max(NextAvailableAddress,
                                      Phdr.p_vaddr + Phdr.p_memsz);
      NextAvailableOffset = std::max(NextAvailableOffset,
                                     Phdr.p_offset + Phdr.p_filesz);

      EFMM->SegmentMapInfo[Phdr.p_vaddr] = SegmentInfo{Phdr.p_vaddr,
                                                       Phdr.p_memsz,
                                                       Phdr.p_offset,
                                                       Phdr.p_filesz};
    }
  }

  for (const auto &Section : InputFile->sections()) {
    StringRef SectionName;
    Section.getName(SectionName);
    StringRef SectionContents;
    Section.getContents(SectionContents);
    if (SectionName == ".text") {
      OldTextSectionAddress = Section.getAddress();
      OldTextSectionSize = Section.getSize();
      OldTextSectionOffset =
        SectionContents.data() - InputFile->getData().data();
    }

    if (SectionName.startswith(OrgSecPrefix) ||
        SectionName.startswith(BOLTSecPrefix)) {
      errs() << "BOLT-ERROR: input file was processed by BOLT. "
                "Cannot re-optimize.\n";
      exit(1);
    }
  }

  assert(NextAvailableAddress && NextAvailableOffset &&
         "no PT_LOAD pheader seen");

  outs() << "BOLT-INFO: first alloc address is 0x"
         << Twine::utohexstr(FirstAllocAddress) << '\n';

  FirstNonAllocatableOffset = NextAvailableOffset;

  NextAvailableAddress = RoundUpToAlignment(NextAvailableAddress, PageAlign);
  NextAvailableOffset = RoundUpToAlignment(NextAvailableOffset, PageAlign);

  if (!opts::UseGnuStack) {
    // This is where the black magic happens. Creating PHDR table in a segment
    // other than that containing ELF header is tricky. Some loaders and/or
    // parts of loaders will apply e_phoff from ELF header assuming both are in
    // the same segment, while others will do the proper calculation.
    // We create the new PHDR table in such a way that both of the methods
    // of loading and locating the table work. There's a slight file size
    // overhead because of that.
    //
    // NB: bfd's strip command cannot do the above and will corrupt the
    //     binary during the process of stripping non-allocatable sections.
    if (NextAvailableOffset <= NextAvailableAddress - FirstAllocAddress) {
      NextAvailableOffset = NextAvailableAddress - FirstAllocAddress;
    } else {
      NextAvailableAddress = NextAvailableOffset + FirstAllocAddress;
    }
    assert(NextAvailableOffset == NextAvailableAddress - FirstAllocAddress &&
           "PHDR table address calculation error");

    outs() << "BOLT-INFO: creating new program header table at address 0x"
           << Twine::utohexstr(NextAvailableAddress) << ", offset 0x"
           << Twine::utohexstr(NextAvailableOffset) << '\n';

    PHDRTableAddress = NextAvailableAddress;
    PHDRTableOffset = NextAvailableOffset;

    // Reserve space for 3 extra pheaders.
    unsigned Phnum = Obj->getHeader()->e_phnum;
    Phnum += 3;

    NextAvailableAddress += Phnum * sizeof(ELFFile<ELF64LE>::Elf_Phdr);
    NextAvailableOffset  += Phnum * sizeof(ELFFile<ELF64LE>::Elf_Phdr);
  }

  // Align at cache line.
  NextAvailableAddress = RoundUpToAlignment(NextAvailableAddress, 64);
  NextAvailableOffset = RoundUpToAlignment(NextAvailableOffset, 64);

  NewTextSegmentAddress = NextAvailableAddress;
  NewTextSegmentOffset = NextAvailableOffset;
}

void RewriteInstance::run() {
  if (!BC) {
    errs() << "BOLT-ERROR: failed to create a binary context\n";
    return;
  }

  unsigned PassNumber = 1;

  // Main "loop".
  discoverStorage();
  readSpecialSections();
  discoverFileObjects();
  readDebugInfo();
  disassembleFunctions();
  readFunctionDebugInfo();
  runOptimizationPasses();
  emitFunctions();

  if (opts::SplitFunctions == BinaryFunction::ST_LARGE &&
      checkLargeFunctions()) {
    ++PassNumber;
    // Emit again because now some functions have been split
    outs() << "BOLT: split-functions: starting pass " << PassNumber << "...\n";
    reset();
    discoverStorage();
    readSpecialSections();
    discoverFileObjects();
    readDebugInfo();
    disassembleFunctions();
    readFunctionDebugInfo();
    runOptimizationPasses();
    emitFunctions();
  }

  // Emit functions again ignoring functions which still didn't fit in their
  // original space, so that we don't generate incorrect debugging information
  // for them (information that would reflect the optimized version).
  if (opts::UpdateDebugSections && opts::FixDebugInfoLargeFunctions &&
      checkLargeFunctions()) {
    ++PassNumber;
    outs() << "BOLT: starting pass (ignoring large functions) "
           << PassNumber << "...\n";
    reset();
    discoverStorage();
    readSpecialSections();
    discoverFileObjects();
    readDebugInfo();
    disassembleFunctions();

    for (uint64_t Address : LargeFunctions) {
      auto FunctionIt = BinaryFunctions.find(Address);
      assert(FunctionIt != BinaryFunctions.end() &&
             "Invalid large function address.");
      errs() << "BOLT-WARNING: Function " << FunctionIt->second
             << " is larger than its orginal size: emitting again marking it "
             << "as not simple.\n";
      FunctionIt->second.setSimple(false);
    }

    readFunctionDebugInfo();
    runOptimizationPasses();
    emitFunctions();
  }

  if (opts::UpdateDebugSections)
    updateDebugInfo();

  // Copy allocatable part of the input.
  std::error_code EC;
  Out = llvm::make_unique<tool_output_file>(opts::OutputFilename, EC,
                                            sys::fs::F_None, 0777);
  check_error(EC, "cannot create output executable file");
  Out->os() << InputFile->getData().substr(0, FirstNonAllocatableOffset);

  // Rewrite allocatable contents and copy non-allocatable parts with mods.
  rewriteFile();
}

void RewriteInstance::discoverFileObjects() {
  FileSymRefs.clear();
  BinaryFunctions.clear();
  BC->GlobalAddresses.clear();

  // For local symbols we want to keep track of associated FILE symbol name for
  // disambiguation by combined name.
  StringRef  FileSymbolName;
  bool SeenFileName = false;
  struct SymbolRefHash {
    std::size_t operator()(SymbolRef const &S) const {
      return std::hash<decltype(DataRefImpl::p)>{}(S.getRawDataRefImpl().p);
    }
  };
  std::unordered_map<SymbolRef, StringRef, SymbolRefHash> SymbolToFileName;
  for (const auto &Symbol : InputFile->symbols()) {
    ErrorOr<StringRef> NameOrError = Symbol.getName();
    if (NameOrError && NameOrError->startswith("__asan_init")) {
      errs() << "BOLT-ERROR: input file was compiled or linked with sanitizer "
                "support. Cannot optimize.\n";
      exit(1);
    }

    if (Symbol.getFlags() & SymbolRef::SF_Undefined)
      continue;

    if (Symbol.getType() == SymbolRef::ST_File) {
      check_error(NameOrError.getError(), "cannot get symbol name for file");
      FileSymbolName = *NameOrError;
      SeenFileName = true;
      continue;
    }
    if (!FileSymbolName.empty() &&
        !(Symbol.getFlags() & SymbolRef::SF_Global)) {
      SymbolToFileName[Symbol] = FileSymbolName;
    }
  }

  // Sort symbols in the file by value.
  std::vector<SymbolRef> SortedFileSymbols(InputFile->symbol_begin(),
                                           InputFile->symbol_end());
  std::stable_sort(SortedFileSymbols.begin(), SortedFileSymbols.end(),
                   [](const SymbolRef &A, const SymbolRef &B) {
                     // FUNC symbols have higher precedence.
                     if (*(A.getAddress()) == *(B.getAddress())) {
                       return A.getType() == SymbolRef::ST_Function &&
                              B.getType() != SymbolRef::ST_Function;
                     }
                     return *(A.getAddress()) < *(B.getAddress());
                   });

  BinaryFunction *PreviousFunction = nullptr;
  for (const auto &Symbol : SortedFileSymbols) {
    // Keep undefined symbols for pretty printing?
    if (Symbol.getFlags() & SymbolRef::SF_Undefined)
      continue;

    if (Symbol.getType() == SymbolRef::ST_File)
      continue;

    ErrorOr<StringRef> NameOrError = Symbol.getName();
    check_error(NameOrError.getError(), "cannot get symbol name");

    ErrorOr<uint64_t> AddressOrErr = Symbol.getAddress();
    check_error(AddressOrErr.getError(), "cannot get symbol address");
    uint64_t Address = *AddressOrErr;
    if (Address == 0) {
      if (opts::Verbosity >= 1 && Symbol.getType() == SymbolRef::ST_Function)
        errs() << "BOLT-WARNING: function with 0 address seen\n";
      continue;
    }

    FileSymRefs[Address] = Symbol;

    // There's nothing horribly wrong with anonymous symbols, but let's
    // ignore them for now.
    if (NameOrError->empty())
      continue;

    /// It is possible we are seeing a globalized local. LLVM might treat it as
    /// a local if it has a "private global" prefix, e.g. ".L". Thus we have to
    /// change the prefix to enforce global scope of the symbol.
    std::string Name =
      NameOrError->startswith(BC->AsmInfo->getPrivateGlobalPrefix())
        ? "PG." + std::string(*NameOrError)
        : std::string(*NameOrError);

    // Disambiguate all local symbols before adding to symbol table.
    // Since we don't know if we will see a global with the same name,
    // always modify the local name.
    //
    // NOTE: the naming convention for local symbols should match
    //       the one we use for profile data.
    std::string UniqueName;
    std::string AlternativeName;
    if (Symbol.getFlags() & SymbolRef::SF_Global) {
      assert(BC->GlobalSymbols.find(Name) == BC->GlobalSymbols.end() &&
             "global name not unique");
      UniqueName = Name;
    } else {
      // If we have a local file name, we should create 2 variants for the
      // function name. The reason is that perf profile might have been
      // collected on a binary that did not have the local file name (e.g. as
      // a side effect of stripping debug info from the binary):
      //
      //   primary:     <function>/<id>
      //   alternative: <function>/<file>/<id2>
      //
      // The <id> field is used for disambiguation of local symbols since there
      // could be identical function names coming from identical file names
      // (e.g. from different directories).
      std::string Prefix = Name + "/";
      std::string AltPrefix;
      auto SFI = SymbolToFileName.find(Symbol);
      if (SFI != SymbolToFileName.end()) {
        AltPrefix = Prefix + std::string(SFI->second) + "/";
      }

      auto uniquifyName = [&] (std::string NamePrefix) {
        unsigned LocalID = 1;
        while (BC->GlobalSymbols.find(NamePrefix + std::to_string(LocalID))
               != BC->GlobalSymbols.end())
          ++LocalID;
        return NamePrefix + std::to_string(LocalID);
      };
      UniqueName = uniquifyName(Prefix);
      if (!AltPrefix.empty())
        AlternativeName = uniquifyName(AltPrefix);
    }

    BC->registerNameAtAddress(UniqueName, Address);
    if (!AlternativeName.empty())
      BC->registerNameAtAddress(AlternativeName, Address);

    ErrorOr<section_iterator> SectionOrErr = Symbol.getSection();
    check_error(SectionOrErr.getError(), "cannot get symbol section");
    section_iterator Section = *SectionOrErr;
    if (Section == InputFile->section_end()) {
      // Could be an absolute symbol. Could record for pretty printing.
      continue;
    }

    DEBUG(dbgs() << "BOLT-DEBUG: considering symbol " << UniqueName
                 << " for function\n");

    if (!Section->isText()) {
      assert(Symbol.getType() != SymbolRef::ST_Function &&
             "unexpected function inside non-code section");
      DEBUG(dbgs() << "BOLT-DEBUG: rejecting as symbol is not in code\n");
      continue;
    }

    auto SymbolSize = ELFSymbolRef(Symbol).getSize();

    // Assembly functions could be ST_NONE with 0 size. Check that the
    // corresponding section is a code section and they are not inside any
    // other known function to consider them.
    //
    // Sometimes assembly functions are not marked as functions and neither are
    // their local labels. The only way to tell them apart is to look at
    // symbol scope - global vs local.
    if (Symbol.getType() != SymbolRef::ST_Function) {
      if (PreviousFunction) {
        if (PreviousFunction->getSize() == 0) {
          if (PreviousFunction->isSymbolValidInScope(Symbol, SymbolSize)) {
            DEBUG(dbgs() << "BOLT-DEBUG: symbol is a function local symbol\n");
            continue;
          }
        } else if (PreviousFunction->containsAddress(Address)) {
          if (PreviousFunction->isSymbolValidInScope(Symbol, SymbolSize)) {
            DEBUG(dbgs() << "BOLT-DEBUG: symbol is a function local symbol\n");
            continue;
          } else {
            if (Address == PreviousFunction->getAddress() && SymbolSize == 0) {
              DEBUG(dbgs() << "BOLT-DEBUG: ignoring symbol as a marker\n");
              continue;
            }
            if (opts::Verbosity > 1) {
              errs() << "BOLT-WARNING: symbol " << UniqueName
                     << " seen in the middle of function "
                     << *PreviousFunction << ". Could be a new entry.\n";
            }
            continue;
          }
        }
      }
    }

    if (PreviousFunction &&
        PreviousFunction->containsAddress(Address) &&
        PreviousFunction->getAddress() != Address) {
      if (PreviousFunction->isSymbolValidInScope(Symbol, SymbolSize)) {
        if (opts::Verbosity >= 1) {
          outs() << "BOLT-DEBUG: possibly another entry for function "
                 << *PreviousFunction << " : " << UniqueName << '\n';
        }
      } else {
        outs() << "BOLT-INFO: using " << UniqueName << " as another entry to "
               << "function " << *PreviousFunction << '\n';

        PreviousFunction->
          addEntryPointAtOffset(Address - PreviousFunction->getAddress());

        if (!opts::Relocs)
          PreviousFunction->setSimple(false);

        // Remove the symbol from FileSymRefs so that we can skip it from
        // in the future.
        auto SI = FileSymRefs.find(Address);
        assert(SI != FileSymRefs.end() && "symbol expected to be present");
        assert(SI->second == Symbol && "wrong symbol found");
        FileSymRefs.erase(SI);
      }
      continue;
    }

    // TODO: populate address map with PLT entries for better readability.

    // Checkout for conflicts with function data from FDEs.
    bool IsSimple = true;
    auto FDEI = CFIRdWrt->getFDEs().lower_bound(Address);
    if (FDEI != CFIRdWrt->getFDEs().end()) {
      auto &FDE = *FDEI->second;
      if (FDEI->first != Address) {
        // There's no matching starting address in FDE. Make sure the previous
        // FDE does not contain this address.
        if (FDEI != CFIRdWrt->getFDEs().begin()) {
          --FDEI;
          auto &PrevFDE = *FDEI->second;
          auto PrevStart = PrevFDE.getInitialLocation();
          auto PrevLength = PrevFDE.getAddressRange();
          if (Address > PrevStart && Address < PrevStart + PrevLength) {
            errs() << "BOLT-ERROR: function " << UniqueName
                   << " is in conflict with FDE ["
                   << Twine::utohexstr(PrevStart) << ", "
                   << Twine::utohexstr(PrevStart + PrevLength)
                   << "). Skipping.\n";
            IsSimple = false;
          }
        }
      } else if (FDE.getAddressRange() != SymbolSize) {
        if (SymbolSize) {
          // Function addresses match but sizes differ.
          errs() << "BOLT-ERROR: sizes differ for function " << UniqueName
                 << ". FDE : " << FDE.getAddressRange()
                 << "; symbol table : " << SymbolSize << ". Skipping.\n";

          // Create maximum size non-simple function.
          IsSimple = false;
        }
        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: adjusting size of function " << UniqueName
                 << " using FDE data.\n";
        }
        SymbolSize = std::max(SymbolSize, FDE.getAddressRange());
      }
    }

    BinaryFunction *BF{nullptr};
    auto BFI = BinaryFunctions.find(Address);
    if (BFI != BinaryFunctions.end()) {
      BF = &BFI->second;
      // Duplicate function name. Make sure everything matches before we add
      // an alternative name.
      if (SymbolSize != BF->getSize()) {
        if (opts::Verbosity >= 1) {
          if (SymbolSize && BF->getSize()) {
            errs() << "BOLT-WARNING: size mismatch for duplicate entries "
                   << *BF << " and " << UniqueName << '\n';
          }
          outs() << "BOLT-INFO: adjusting size of function " << *BF
                 << " old " << BF->getSize() << " new " << SymbolSize << "\n";
        }
        BF->setSize(std::max(SymbolSize, BF->getSize()));
      }
      BF->addAlternativeName(UniqueName);
    } else {
      BF = createBinaryFunction(UniqueName, *Section, Address, SymbolSize,
                                IsSimple);
    }
    if (!AlternativeName.empty())
      BF->addAlternativeName(AlternativeName);

    PreviousFunction = BF;
  }

  // See if we missed any functions marked by FDE.
  for (const auto &FDEI : CFIRdWrt->getFDEs()) {
    const auto Address = FDEI.first;
    const auto *FDE = FDEI.second;
    auto *BF = getBinaryFunctionContainingAddress(Address);
    if (!BF) {
      errs() << "BOLT-WARNING: FDE [0x" << Twine::utohexstr(Address) << ", 0x"
             << Twine::utohexstr(Address + FDE->getAddressRange())
             << ") has no corresponding symbol table entry\n";
      auto Section = BC->getSectionForAddress(Address);
      assert(Section && "cannot get section for address from FDE");
      StringRef SectionName;
      Section->getName(SectionName);
      // PLT has a special FDE.
      if (SectionName == ".plt") {
        // Set the size to 0 to prevent PLT from being disassembled.
        createBinaryFunction("__BOLT_PLT_PSEUDO" , *Section, Address, 0, false);
      } else {
        std::string FunctionName =
          "__BOLT_FDE_FUNCat" + Twine::utohexstr(Address).str();
        BC->registerNameAtAddress(FunctionName, Address);
        createBinaryFunction(FunctionName, *Section, Address,
                             FDE->getAddressRange(), true);
      }
    } else if (BF->getAddress() != Address) {
      errs() << "BOLT-WARNING: FDE [0x" << Twine::utohexstr(Address) << ", 0x"
             << Twine::utohexstr(Address + FDE->getAddressRange())
             << ") conflicts with function " << *BF << '\n';
    }
  }

  if (!SeenFileName && BC->DR.hasLocalsWithFileName() && !opts::AllowStripped) {
    errs() << "BOLT-ERROR: input binary does not have local file symbols "
              "but profile data includes function names with embedded file "
              "names. It appears that the input binary was stripped while a "
              "profiled binary was not. If you know what you are doing and "
              "wish to proceed, use -allow-stripped option.\n";
    exit(1);
  }

  // Now that all the functions were created - adjust their boundaries.
  adjustFunctionBoundaries();

  if (!opts::Relocs)
    return;

  // Read all relocations now that we have binary functions mapped.
  for (const auto &Section : InputFile->sections()) {
    if (Section.relocation_begin() != Section.relocation_end()) {
      readRelocations(Section);
    }
  }
}

void RewriteInstance::adjustFunctionBoundaries() {
  for (auto &BFI : BinaryFunctions) {
    auto &Function = BFI.second;

    // Check if there's a symbol with a larger address in the same section.
    // If there is - it determines the maximum size for the current function,
    // otherwise, it is the size of containing section the defines it.
    //
    // NOTE: ignore some symbols that could be tolerated inside the body
    //       of a function.
    auto NextSymRefI = FileSymRefs.upper_bound(Function.getAddress());
    while (NextSymRefI != FileSymRefs.end()) {
      auto &Symbol = NextSymRefI->second;
      auto SymbolSize = ELFSymbolRef(Symbol).getSize();

      if (!Function.isSymbolValidInScope(Symbol, SymbolSize))
        break;

      // This is potentially another entry point into the function.
      auto EntryOffset = NextSymRefI->first - Function.getAddress();
      DEBUG(dbgs() << "BOLT-DEBUG: adding entry point to function "
                   << Function << " at offset 0x"
                   << Twine::utohexstr(EntryOffset) << '\n');
      Function.addEntryPointAtOffset(EntryOffset);

      // In non-relocation mode there's potentially an external undetectable
      // reference to the entry point and hence we cannot move this entry point.
      // Optimizing without moving could be difficult.
      if (!opts::Relocs)
        Function.setSimple(false);

      ++NextSymRefI;
    }
    auto NextSymRefSectionI = (NextSymRefI == FileSymRefs.end())
      ? InputFile->section_end()
      : NextSymRefI->second.getSection();

    uint64_t MaxSize;
    if (NextSymRefI != FileSymRefs.end() &&
        NextSymRefI->second.getSection() &&
        *NextSymRefI->second.getSection() != InputFile->section_end() &&
        **NextSymRefI->second.getSection() == Function.getSection()) {
      MaxSize = NextSymRefI->first - Function.getAddress();
    } else {
      // Function runs till the end of the containing section.
      uint64_t SectionEnd = Function.getSection().getAddress() +
                            Function.getSection().getSize();
      assert((NextSymRefI == FileSymRefs.end() ||
              NextSymRefI->first >= SectionEnd) &&
             "different sections should not overlap");
      MaxSize = SectionEnd - Function.getAddress();
    }

    if (MaxSize < Function.getSize()) {
      errs() << "BOLT-ERROR: symbol seen in the middle of the function "
             << Function << ". Skipping.\n";
      Function.setSimple(false);
      Function.setMaxSize(Function.getSize());
      continue;
    }
    Function.setMaxSize(MaxSize);
    if (!Function.getSize() && Function.isSimple()) {
      // Some assembly functions have their size set to 0, use the max
      // size as their real size.
      if (opts::Verbosity >= 1) {
        outs() << "BOLT-INFO: setting size of function " << Function
               << " to " << Function.getMaxSize() << " (was 0)\n";
      }
      Function.setSize(Function.getMaxSize());
    }
  }
}

void RewriteInstance::relocateEHFrameSection() {
  assert(EHFrameSection.getObject() != nullptr &&
         "non-empty .eh_frame section expected");

  DWARFFrame EHFrame(EHFrameSection.getAddress());
  StringRef EHFrameSectionContents;
  EHFrameSection.getContents(EHFrameSectionContents);
  DataExtractor DE(EHFrameSectionContents,
                   BC->AsmInfo->isLittleEndian(),
                   BC->AsmInfo->getPointerSize());
  auto createReloc = [&](uint64_t Value, uint64_t Offset, uint64_t DwarfType) {
    if (DwarfType == dwarf::DW_EH_PE_omit)
      return;

    if (!(DwarfType & dwarf::DW_EH_PE_pcrel) &&
        !(DwarfType & dwarf::DW_EH_PE_textrel) &&
        !(DwarfType & dwarf::DW_EH_PE_funcrel) &&
        !(DwarfType & dwarf::DW_EH_PE_datarel)) {
      return;
    }

    if (!(DwarfType & dwarf::DW_EH_PE_sdata4))
      return;

    uint64_t RelType;
    switch (DwarfType & 0x0f) {
    default:
      llvm_unreachable("unsupported DWARF encoding type");
    case dwarf::DW_EH_PE_sdata4:
    case dwarf::DW_EH_PE_udata4:
      RelType = ELF::R_X86_64_PC32;
      break;
    case dwarf::DW_EH_PE_sdata8:
    case dwarf::DW_EH_PE_udata8:
      RelType = ELF::R_X86_64_PC64;
      break;
    }

    auto *Symbol = BC->getGlobalSymbolAtAddress(Value);
    if (!Symbol) {
      DEBUG(dbgs() << "BOLT-DEBUG: creating symbol for DWARF reference at 0x"
                   << Twine::utohexstr(Value) << '\n');
      Symbol = BC->getOrCreateGlobalSymbol(Value, "FUNCat");
    }

    DEBUG(dbgs() << "BOLT-DEBUG: adding DWARF reference against symbol "
                 << Symbol->getName() << '\n');

    BC->addSectionRelocation(EHFrameSection, Offset, Symbol, RelType);
  };

  EHFrame.parse(DE, createReloc);

  if (!EHFrame.ParseError.empty()) {
    errs() << "BOLT-ERROR: EHFrame reader failed with message \""
           << EHFrame.ParseError << '\n';
    exit(1);
  }
}

BinaryFunction *RewriteInstance::createBinaryFunction(
    const std::string &Name, SectionRef Section, uint64_t Address,
    uint64_t Size, bool IsSimple) {
  auto Result = BinaryFunctions.emplace(
      Address, BinaryFunction(Name, Section, Address, Size, *BC, IsSimple));
  assert(Result.second == true && "unexpected duplicate function");
  auto *BF = &Result.first->second;
  BC->SymbolToFunctionMap[BF->getSymbol()] = BF;
  return BF;
}

void RewriteInstance::readSpecialSections() {
  // Process special sections.
  for (const auto &Section : InputFile->sections()) {
    StringRef SectionName;
    check_error(Section.getName(SectionName), "cannot get section name");
    StringRef SectionContents;
    check_error(Section.getContents(SectionContents),
                "cannot get section contents");
    ArrayRef<uint8_t> SectionData(
        reinterpret_cast<const uint8_t *>(SectionContents.data()),
        Section.getSize());

    if (SectionName == ".gcc_except_table") {
      LSDAData = SectionData;
      LSDAAddress = Section.getAddress();
    } else if (SectionName == ".debug_loc") {
      DebugLocSize = Section.getSize();
    } else if (SectionName == ".eh_frame") {
      EHFrameSection = Section;
    }

    // Ignore zero-size allocatable sections as they present no interest to us.
    if ((ELFSectionRef(Section).getFlags() & ELF::SHF_ALLOC) &&
        Section.getSize() > 0) {
      BC->AllocatableSections.emplace(std::make_pair(Section.getAddress(),
                                                     Section));
    }
  }

  // Process debug sections.
  EHFrame = BC->DwCtx->getEHFrame();
  if (opts::DumpEHFrame) {
    outs() << "BOLT-INFO: Dumping original binary .eh_frame\n";
    EHFrame->dump(outs());
  }
  CFIRdWrt.reset(new CFIReaderWriter(*EHFrame));
  if (!EHFrame->ParseError.empty()) {
    errs() << "BOLT-ERROR: EHFrame reader failed with message \""
           << EHFrame->ParseError << '\n';
    exit(1);
  }
}

namespace {
template <typename ELFT>
int64_t getRelocationAddend(const ELFObjectFile<ELFT> *Obj,
                            const RelocationRef &RelRef) {
  int64_t Addend = 0;
  const ELFFile<ELFT> &EF = *Obj->getELFFile();
  DataRefImpl Rel = RelRef.getRawDataRefImpl();
  const auto *RelocationSection = *(EF.getSection(Rel.d.a));
  switch (RelocationSection->sh_type) {
  default: llvm_unreachable("unexpected relocation section type");
  case ELF::SHT_REL:
    break;
  case ELF::SHT_RELA: {
    const auto *RelA = Obj->getRela(Rel);
    Addend = RelA->r_addend;
    break;
  }
  }

  return Addend;
}

int64_t getRelocationAddend(const ELFObjectFileBase *Obj,
                         const RelocationRef &Rel) {
  if (auto *ELF32LE = dyn_cast<ELF32LEObjectFile>(Obj))
    return getRelocationAddend(ELF32LE, Rel);
  if (auto *ELF64LE = dyn_cast<ELF64LEObjectFile>(Obj))
    return getRelocationAddend(ELF64LE, Rel);
  if (auto *ELF32BE = dyn_cast<ELF32BEObjectFile>(Obj))
    return getRelocationAddend(ELF32BE, Rel);
  auto *ELF64BE = cast<ELF64BEObjectFile>(Obj);
  return getRelocationAddend(ELF64BE, Rel);
}
} // anonymous namespace

void RewriteInstance::readRelocations(const SectionRef &Section) {
  StringRef SectionName;
  Section.getName(SectionName);
  DEBUG(dbgs() << "BOLT-DEBUG: relocations for section "
               << SectionName << ":\n");
  if (ELFSectionRef(Section).getFlags() & ELF::SHF_ALLOC) {
    DEBUG(dbgs() << "BOLT-DEBUG: ignoring runtime relocations\n");
    return;
  }
  auto SecIter = Section.getRelocatedSection();
  assert(SecIter != InputFile->section_end() && "relocated section expected");
  auto RelocatedSection = *SecIter;

  StringRef RelocatedSectionName;
  RelocatedSection.getName(RelocatedSectionName);
  DEBUG(dbgs() << "BOLT-DEBUG: relocated section is "
               << RelocatedSectionName << '\n');

  if (!(ELFSectionRef(RelocatedSection).getFlags() & ELF::SHF_ALLOC)) {
    DEBUG(dbgs() << "BOLT-DEBUG: ignoring relocations against "
                 << "non-allocatable section\n");
    return;
  }
  const bool SkipRelocs = StringSwitch<bool>(RelocatedSectionName)
    .Cases(".plt", ".rela.plt", ".got.plt", ".eh_frame", true)
    .Default(false);
  if (SkipRelocs) {
    DEBUG(dbgs() << "BOLT-DEBUG: ignoring relocations against known section\n");
    return;
  }

  // For value extraction.
  StringRef RelocatedSectionContents;
  RelocatedSection.getContents(RelocatedSectionContents);
  DataExtractor DE(RelocatedSectionContents,
                   BC->AsmInfo->isLittleEndian(),
                   BC->AsmInfo->getPointerSize());

  bool IsFromCode = RelocatedSection.isText();
  for (const auto &Rel : Section.relocations()) {
    SmallString<16> TypeName;
    Rel.getTypeName(TypeName);
    DEBUG(dbgs() << "BOLT-DEBUG: offset = 0x"
                 << Twine::utohexstr(Rel.getOffset())
                 << "; type name = " << TypeName
                 << '\n');

    // Extract value.
    uint32_t RelocationOffset =
        Rel.getOffset() - RelocatedSection.getAddress();
    auto ExtractedValue = static_cast<uint64_t>(
      DE.getSigned(&RelocationOffset,
                   Relocation::getSizeForType(Rel.getType())));

    bool IsPCRelative = Relocation::isPCRelative(Rel.getType());
    auto Addend = getRelocationAddend(InputFile, Rel);
    uint64_t Address = 0;
    uint64_t SymbolAddress = 0;
    auto SymbolIter = Rel.getSymbol();
    std::string SymbolName = "<no symbol>";
    SymbolAddress = *SymbolIter->getAddress();
    if (!SymbolAddress) {
      Address = ExtractedValue;
      if (IsPCRelative) {
        Address += Rel.getOffset();
      }
    } else {
      Address = SymbolAddress + Addend;
    }
    bool SymbolIsSection = false;
    if (SymbolIter != InputFile->symbol_end()) {
      SymbolName = (*(*SymbolIter).getName());
      if (SymbolIter->getType() == SymbolRef::ST_Debug) {
        // Weird stuff - section symbols are marked as ST_Debug.
        SymbolIsSection = true;
        auto SymbolSection = SymbolIter->getSection();
        if (SymbolSection && *SymbolSection != InputFile->section_end()) {
          StringRef SymbolSectionName;
          (*SymbolSection)->getName(SymbolSectionName);
          SymbolName = "section " + std::string(SymbolSectionName);
          Address = Addend;
        }
      }
    }

    bool ForceRelocation = false;
    if (opts::HotText &&
        (SymbolName == "__hot_start" || SymbolName == "__hot_end")) {
      ForceRelocation = true;
    }

    if (!IsPCRelative && Addend != 0 && IsFromCode && !SymbolIsSection) {
      auto RefSection = BC->getSectionForAddress(SymbolAddress);
      if (RefSection && RefSection->isText()) {
        errs() << "BOLT-WARNING: detected absolute reference from code into a "
               << "middle of a function:\n"
               << " offset = 0x" << Twine::utohexstr(Rel.getOffset())
               << "; symbol = " << SymbolName
               << "; symbol address = 0x" << Twine::utohexstr(SymbolAddress)
               << "; addend = 0x" << Twine::utohexstr(Addend)
               << "; address = 0x" << Twine::utohexstr(Address)
               << "; type = " << Rel.getType()
               << "; type name = " << TypeName
               << '\n';
        assert(ExtractedValue == SymbolAddress + Addend && "value mismatch");
      }
    }

    if (Addend < 0 && IsPCRelative) {
      Address -= Addend;
    } else {
      Addend = 0;
    }

    DEBUG(dbgs() << "BOLT-DEBUG: offset = 0x"
                 << Twine::utohexstr(Rel.getOffset())
                 << "; symbol = " << SymbolName
                 << "; symbol address = 0x" << Twine::utohexstr(SymbolAddress)
                 << "; addend = 0x" << Twine::utohexstr(Addend)
                 << "; address = 0x" << Twine::utohexstr(Address)
                 << "; type = " << Rel.getType()
                 << "; type name = " << TypeName
                 << '\n');

    if (Rel.getType() != ELF::R_X86_64_TPOFF32 &&
        Rel.getType() != ELF::R_X86_64_GOTTPOFF &&
        Rel.getType() != ELF::R_X86_64_GOTPCREL) {
      if (!IsPCRelative) {
        if (opts::Verbosity > 1 &&
            ExtractedValue != Address) {
          errs() << "BOLT-WARNING: mismatch ExtractedValue = 0x"
                 << Twine::utohexstr(ExtractedValue) << '\n';
        }
        Address = ExtractedValue;
      } else {
        if (opts::Verbosity > 1 &&
            ExtractedValue != Address - Rel.getOffset() + Addend) {
          errs() << "BOLT-WARNING: PC-relative mismatch ExtractedValue = 0x"
                 << Twine::utohexstr(ExtractedValue) << '\n';
        }
        Address = ExtractedValue - Addend;
      }
    }

    BinaryFunction *ContainingBF = nullptr;
    if (IsFromCode) {
      ContainingBF = getBinaryFunctionContainingAddress(Rel.getOffset());
      assert(ContainingBF && "cannot find function for address in code");
      DEBUG(dbgs() << "BOLT-DEBUG: relocation belongs to " << ContainingBF
                   << '\n');
    }

    // PC-relative relocations from data to code are tricky since the original
    // information is typically lost after linking even with '--emit-relocs'.
    // They are normally used by PIC-style jump tables and reference both
    // the jump table and jump destination by computing the difference
    // between the two. If we blindly apply the relocation it will appear
    // that it references an arbitrary location in the code, possibly even
    // in a different function from that containing the jump table.
    if (IsPCRelative) {
      // Just register the fact that we have PC-relative relocation at a given
      // address. The actual referenced label/address cannot be determined
      // from linker data alone.
      if (IsFromCode) {
        ContainingBF->addPCRelativeRelocationAddress(Rel.getOffset());
      }
      DEBUG(dbgs() << "BOLT-DEBUG: not creating PC-relative relocation\n");
      continue;
    }

    auto RefSection = BC->getSectionForAddress(Address);
    if (!RefSection && !ForceRelocation) {
      DEBUG(dbgs() << "BOLT-DEBUG: cannot determine referenced section.\n");
      continue;
    }

    bool ToCode = RefSection && RefSection->isText();

    // Occasionally we may see a reference past the last byte of the function
    // typically as a result of __builtin_unreachable(). Check it here.
    auto *ReferencedBF =
      getBinaryFunctionContainingAddress(Address, /*CheckPastEnd*/ true);
    uint64_t RefFunctionOffset = 0;
    MCSymbol *ReferencedSymbol = nullptr;
    if (ForceRelocation) {
      ReferencedSymbol = BC->registerNameAtAddress(SymbolName, 0);
      Addend = Address;
      DEBUG(dbgs() << "BOLT-DEBUG: creating relocations for huge pages against"
                      " symbol " << SymbolName << " with addend " << Addend
                   << '\n');
    } else if (ReferencedBF) {
      RefFunctionOffset = Address - ReferencedBF->getAddress();
      DEBUG(dbgs() << "  referenced function " << *ReferencedBF;
            if (Address != ReferencedBF->getAddress())
              dbgs() << " at offset 0x"
                     << Twine::utohexstr(RefFunctionOffset);
            dbgs() << '\n');
      if (RefFunctionOffset) {
        ReferencedSymbol =
          ReferencedBF->getOrCreateLocalLabel(Address, /*CreatePastEnd*/ true);
      } else {
        ReferencedSymbol = ReferencedBF->getSymbol();
      }
    } else {
      if (RefSection && RefSection->isText() && SymbolAddress) {
        // This can happen e.g. with PIC-style jump tables.
        DEBUG(dbgs() << "BOLT-DEBUG: no corresponding function for "
                        "relocation against code\n");
      }
      ReferencedSymbol = BC->getOrCreateGlobalSymbol(Address, "SYMBOLat");
    }

    if (IsFromCode) {
      if (ReferencedBF || ForceRelocation) {
        ContainingBF->addRelocation(Rel.getOffset(), ReferencedSymbol,
                                    Rel.getType(), Addend, Address);
      } else {
        DEBUG(dbgs() << "BOLT-DEBUG: ignoring relocation from code to data\n");
      }
    } else if (ToCode) {
      assert(Addend == 0 && "did not expect addend");
      BC->addRelocation(Rel.getOffset(), ReferencedSymbol, Rel.getType());
    } else {
      DEBUG(dbgs() << "BOLT-DEBUG: ignoring relocation from data to data\n");
    }
  }
}

void RewriteInstance::readDebugInfo() {
  if (!opts::UpdateDebugSections)
    return;

  BC->preprocessDebugInfo(BinaryFunctions);
}

void RewriteInstance::readFunctionDebugInfo() {
  if (!opts::UpdateDebugSections)
    return;

  BC->preprocessFunctionDebugInfo(BinaryFunctions);
}

void RewriteInstance::disassembleFunctions() {
  // Disassemble every function and build it's control flow graph.
  TotalScore = 0;
  for (auto &BFI : BinaryFunctions) {
    BinaryFunction &Function = BFI.second;

    // If we have to relocate the code we have to disassemble all functions.
    if (!opts::Relocs && !opts::shouldProcess(Function)) {
      DEBUG(dbgs() << "BOLT: skipping processing function "
                   << Function << " per user request.\n");
      continue;
    }

    SectionRef Section = Function.getSection();
    assert(Section.getAddress() <= Function.getAddress() &&
           Section.getAddress() + Section.getSize()
             >= Function.getAddress() + Function.getSize() &&
          "wrong section for function");
    if (!Section.isText() || Section.isVirtual() || !Section.getSize()) {
      // When could it happen?
      errs() << "BOLT-ERROR: corresponding section is non-executable or "
             << "empty for function " << Function << '\n';
      continue;
    }

    // Treat zero-sized functions as non-simple ones.
    if (Function.getSize() == 0) {
      Function.setSimple(false);
      continue;
    }

    StringRef SectionContents;
    check_error(Section.getContents(SectionContents),
                "cannot get section contents");

    assert(SectionContents.size() == Section.getSize() &&
           "section size mismatch");

    // Function offset from the section start.
    auto FunctionOffset = Function.getAddress() - Section.getAddress();

    // Offset of the function in the file.
    Function.setFileOffset(
        SectionContents.data() - InputFile->getData().data() + FunctionOffset);

    ArrayRef<uint8_t> FunctionData(
        reinterpret_cast<const uint8_t *>
          (SectionContents.data()) + FunctionOffset,
        Function.getSize());

    Function.disassemble(FunctionData);

    if (!Function.isSimple() && opts::Relocs) {
      errs() << "BOLT-ERROR: function " << Function << " cannot be properly "
             << "disassembled. Unable to continue in relocation mode.\n";
      abort();
    }


    if (opts::PrintAll || opts::PrintDisasm)
      Function.print(outs(), "after disassembly", true);

    // Post-process inter-procedural references ASAP as it may affect
    // functions we are about to disassemble next.
    for (const auto Addr : BC->InterproceduralReferences) {
      auto *ContainingFunction = getBinaryFunctionContainingAddress(Addr);
      if (ContainingFunction && ContainingFunction->getAddress() != Addr) {
        ContainingFunction->addEntryPoint(Addr);
        if (!opts::Relocs) {
          if (opts::Verbosity >= 1) {
            errs() << "BOLT-WARNING: Function " << *ContainingFunction
                   << " has internal BBs that are target of a reference located"
                   << " in another function. Skipping the function.\n";
          }
          ContainingFunction->setSimple(false);
        }
      } else if (!ContainingFunction && Addr) {
        // Check if address falls in function padding space - this could be
        // unmarked data in code. In this case adjust the padding space size.
        auto Section = BC->getSectionForAddress(Addr);
        assert(Section && "cannot get section for referenced address");

        if (!Section->isText())
          continue;

        // PLT requires special handling and could be ignored in this context.
        StringRef SectionName;
        Section->getName(SectionName);
        if (SectionName == ".plt")
          continue;

        if (opts::Relocs) {
          errs() << "BOLT-ERROR: cannot process binaries with unmarked "
                 << "object in code at address 0x"
                 << Twine::utohexstr(Addr) << " belonging to section "
                 << SectionName << " in relocation mode.\n";
          exit(1);
        }

        ContainingFunction =
          getBinaryFunctionContainingAddress(Addr,
                                             /*CheckPastEnd=*/false,
                                             /*UseMaxSize=*/true);
        if (ContainingFunction) {
          errs() << "BOLT-WARNING: function " << *ContainingFunction
                 << " has an object detected in a padding region at address 0x"
                 << Twine::utohexstr(Addr) << '\n';
          ContainingFunction->setMaxSize(
              Addr - ContainingFunction->getAddress());
        }
      }
    }
    BC->InterproceduralReferences.clear();

    // Fill in CFI information for this function
    if (Function.isSimple()) {
      if (!CFIRdWrt->fillCFIInfoFor(Function)) {
        errs() << "BOLT-ERROR: unable to fill CFI for function "
               << Function << ".\n";
        if (opts::Relocs)
          abort();
        Function.setSimple(false);
        continue;
      }
    }

    // Parse LSDA.
    if (Function.isSimple() && Function.getLSDAAddress() != 0)
      Function.parseLSDA(LSDAData, LSDAAddress);

    if (!Function.buildCFG())
      continue;

    if (opts::PrintAll || opts::PrintCFG)
      Function.print(outs(), "after building cfg", true);

    if (opts::DumpDotAll)
      Function.dumpGraphForPass("build-cfg");

    if (opts::PrintLoopInfo) {
      Function.calculateLoopInfo();
      Function.printLoopInfo(outs());
    }

    TotalScore += Function.getFunctionScore();

  } // Iterate over all functions

  uint64_t NumSimpleFunctions{0};
  uint64_t NumStaleProfileFunctions{0};
  std::vector<BinaryFunction *> ProfiledFunctions;
  for (auto &BFI : BinaryFunctions) {
    auto &Function = BFI.second;
    if (!Function.isSimple())
      continue;
    ++NumSimpleFunctions;
    if (Function.getExecutionCount() == BinaryFunction::COUNT_NO_PROFILE)
      continue;
    if (Function.hasValidProfile())
      ProfiledFunctions.push_back(&Function);
    else
      ++NumStaleProfileFunctions;
  }

  const auto NumAllProfiledFunctions =
                            ProfiledFunctions.size() + NumStaleProfileFunctions;
  outs() << "BOLT-INFO: "
         << NumAllProfiledFunctions
         << " functions out of " << NumSimpleFunctions << " simple functions ("
         << format("%.1f", NumAllProfiledFunctions /
                                            (float) NumSimpleFunctions * 100.0f)
         << "%) have non-empty execution profile.\n";
  if (NumStaleProfileFunctions) {
    outs() << "BOLT-INFO: " << NumStaleProfileFunctions
           << format(" (%.1f%% of all profiled)",
                     NumStaleProfileFunctions /
                                      (float) NumAllProfiledFunctions * 100.0f)
           << " function" << (NumStaleProfileFunctions == 1 ? "" : "s")
           << " have invalid (possibly stale) profile.\n";
  }

  if (ProfiledFunctions.size() > 10) {
    if (opts::Verbosity >= 1) {
      outs() << "BOLT-INFO: top called functions are:\n";
      std::sort(ProfiledFunctions.begin(), ProfiledFunctions.end(),
                [](BinaryFunction *A, BinaryFunction *B) {
                  return B->getExecutionCount() < A->getExecutionCount();
                }
                );
      auto SFI = ProfiledFunctions.begin();
      auto SFIend = ProfiledFunctions.end();
      for (auto i = 0u; i < opts::TopCalledLimit && SFI != SFIend; ++SFI, ++i) {
        outs() << "  " << **SFI << " : "
               << (*SFI)->getExecutionCount() << '\n';
      }
    }
  }
}

void RewriteInstance::runOptimizationPasses() {
  callWithDynoStats(
    [this] {
      BinaryFunctionPassManager::runAllPasses(*BC,
                                              BinaryFunctions,
                                              LargeFunctions);
    },
    BinaryFunctions,
    "optimizations",
    opts::PrintDynoStats || opts::DynoStatsAll);
}

namespace {

// Helper function to emit the contents of a function via a MCStreamer object.
void emitFunction(MCStreamer &Streamer, BinaryFunction &Function,
                  BinaryContext &BC, bool EmitColdPart) {
  if (Function.getSize() == 0)
    return;

  if (Function.getState() == BinaryFunction::State::Empty)
    return;

  MCSection *Section;
  if (opts::Relocs) {
    Section = BC.MOFI->getTextSection();
  } else {
    // Each fuction is emmitted into its own section.
    Section =
        BC.Ctx->getELFSection(EmitColdPart ? Function.getColdCodeSectionName()
                                           : Function.getCodeSectionName(),
                              ELF::SHT_PROGBITS,
                              ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);
  }

  Section->setHasInstructions(true);
  BC.Ctx->addGenDwarfSection(Section);

  Streamer.SwitchSection(Section);

  if (!opts::Relocs)
    Streamer.setCodeSkew(EmitColdPart ? 0 : Function.getAddress());

  if (opts::Relocs) {
    // We have to use at least 2-byte alignment because of C++ ABI.
    Streamer.EmitCodeAlignment(2);
    Streamer.EmitCodeAlignment(opts::AlignFunctions,
                               opts::AlignFunctionsMaxBytes);
  } else {
    Streamer.EmitCodeAlignment(Function.getAlignment());
  }

  MCContext &Context = Streamer.getContext();
  const MCAsmInfo *MAI = Context.getAsmInfo();

  // Emit all names the function is known under.
  for (const auto &Name : Function.getNames()) {
    Twine EmitName = EmitColdPart ? Twine(Name).concat(".cold") : Name;
    auto *EmitSymbol = BC.Ctx->getOrCreateSymbol(EmitName);
    Streamer.EmitSymbolAttribute(EmitSymbol, MCSA_ELF_TypeFunction);
    DEBUG(dbgs() << "emitting symbol " << EmitSymbol->getName()
                 << " for function " << Function << '\n');
    Streamer.EmitLabel(EmitSymbol);
  }

  // Emit CFI start
  if (Function.hasCFI() && (opts::Relocs || Function.isSimple())) {
    Streamer.EmitCFIStartProc(/*IsSimple=*/false);
    if (Function.getPersonalityFunction() != nullptr) {
      Streamer.EmitCFIPersonality(Function.getPersonalityFunction(),
                                  Function.getPersonalityEncoding());
    }
    auto *LSDASymbol = EmitColdPart ? Function.getColdLSDASymbol()
                                    : Function.getLSDASymbol();
    if (LSDASymbol) {
      Streamer.EmitCFILsda(LSDASymbol, BC.MOFI->getLSDAEncoding());
    } else {
      Streamer.EmitCFILsda(0, dwarf::DW_EH_PE_omit);
    }
    // Emit CFI instructions relative to the CIE
    for (const auto &CFIInstr : Function.cie()) {
      // Only write CIE CFI insns that LLVM will not already emit
      const std::vector<MCCFIInstruction> &FrameInstrs =
          MAI->getInitialFrameState();
      if (std::find(FrameInstrs.begin(), FrameInstrs.end(), CFIInstr) ==
          FrameInstrs.end())
        Streamer.EmitCFIInstruction(CFIInstr);
    }
  }

  assert((Function.empty() || !(*Function.begin()).isCold()) &&
         "first basic block should never be cold");

  // Emit UD2 at the beginning if requested by user.
  if (!opts::BreakFunctionNames.empty()) {
    for (auto &Name : opts::BreakFunctionNames) {
      if (Function.hasName(Name)) {
        Streamer.EmitIntValue(0x0B0F, 2); // UD2: 0F 0B
        break;
      }
    }
  }

  // Emit code.
  Function.emitBody(Streamer, EmitColdPart);

  // Emit padding if requested.
  if (auto Padding = opts::padFunction(Function)) {
    DEBUG(dbgs() << "BOLT-DEBUG: padding function " << Function << " with "
                 << Padding << " bytes\n");
    Streamer.EmitFill(Padding, MAI->getTextAlignFillValue());
  }

  if (opts::MarkFuncs) {
    Streamer.EmitIntValue(MAI->getTrapFillValue(), 1);
  }

  // Emit CFI end
  if (Function.hasCFI() && (opts::Relocs || Function.isSimple()))
    Streamer.EmitCFIEndProc();

  Streamer.EmitLabel(EmitColdPart ? Function.getFunctionColdEndLabel()
                                  : Function.getFunctionEndLabel());

  if (!Function.isSimple() && !opts::Relocs)
    return;

  // Exception handling info for the function.
  Function.emitLSDA(&Streamer, EmitColdPart);

  if (!EmitColdPart && opts::JumpTables > JTS_NONE)
    Function.emitJumpTables(&Streamer);
}

template <typename T>
std::vector<T> singletonSet(T t) {
  std::vector<T> Vec;
  Vec.push_back(std::move(t));
  return Vec;
}

} // anonymous namespace

void RewriteInstance::emitFunctions() {
  std::error_code EC;

  // This is an object file, which we keep for debugging purposes.
  // Once we decide it's useless, we should create it in memory.
  std::unique_ptr<tool_output_file> TempOut =
    llvm::make_unique<tool_output_file>(opts::OutputFilename + ".bolt.o",
                                        EC, sys::fs::F_None);
  check_error(EC, "cannot create output object file");

  std::unique_ptr<buffer_ostream> BOS =
      make_unique<buffer_ostream>(TempOut->os());
  raw_pwrite_stream *OS = BOS.get();

  // Implicitly MCObjectStreamer takes ownership of MCAsmBackend (MAB)
  // and MCCodeEmitter (MCE). ~MCObjectStreamer() will delete these
  // two instances.
  auto MCE = BC->TheTarget->createMCCodeEmitter(*BC->MII, *BC->MRI, *BC->Ctx);
  auto MAB = BC->TheTarget->createMCAsmBackend(*BC->MRI, BC->TripleName, "");
  std::unique_ptr<MCStreamer> Streamer(
    BC->TheTarget->createMCObjectStreamer(*BC->TheTriple,
                                          *BC->Ctx,
                                          *MAB,
                                          *OS,
                                          MCE,
                                          *BC->STI,
                                          /* RelaxAll */ false,
                                          /* DWARFMustBeAtTheEnd */ false));

  Streamer->InitSections(false);

  // Mark beginning of "hot text".
  if (opts::Relocs && opts::HotText)
    Streamer->EmitLabel(BC->Ctx->getOrCreateSymbol("__hot_start"));

  // Sort functions for the output.
  std::vector<BinaryFunction *> SortedFunctions(BinaryFunctions.size());
  std::transform(BinaryFunctions.begin(),
                 BinaryFunctions.end(),
                 SortedFunctions.begin(),
                 [](std::pair<const uint64_t, BinaryFunction> &BFI) {
                   return &BFI.second;
                 });

  if (opts::ReorderFunctions != BinaryFunction::RT_NONE) {
    std::stable_sort(SortedFunctions.begin(), SortedFunctions.end(),
                     [](const BinaryFunction *A, const BinaryFunction *B) {
                       if (A->hasValidIndex() && B->hasValidIndex()) {
                         return A->getIndex() < B->getIndex();
                       } else {
                         return A->hasValidIndex();
                       }
                     });
  }

  DEBUG(
    if (!opts::Relocs) {
      auto SortedIt = SortedFunctions.begin();
      for (auto &It : BinaryFunctions) {
        assert(&It.second == *SortedIt);
        ++SortedIt;
      }
    });

  uint32_t LastHotIndex = -1u;
  uint32_t CurrentIndex = 0;
  for (auto *BF : SortedFunctions) {
    if (!BF->hasValidIndex() && LastHotIndex == -1u) {
      LastHotIndex = CurrentIndex;
    }
    assert(LastHotIndex == -1u || !BF->hasValidIndex());
    assert(!BF->hasValidIndex() || CurrentIndex == BF->getIndex());
    ++CurrentIndex;
  }
  CurrentIndex = 0;
  DEBUG(dbgs() << "BOLT-DEBUG: LastHotIndex = " << LastHotIndex << "\n");

  bool ColdFunctionSeen = false;

  // Output functions one by one.
  for (auto *FunctionPtr : SortedFunctions) {
    auto &Function = *FunctionPtr;

    // Emit all cold function split parts at the border of hot and
    // cold functions.
    if (opts::Relocs && !ColdFunctionSeen && CurrentIndex >= LastHotIndex) {
      // Mark the end of "hot" stuff.
      if (opts::HotText) {
        Streamer->SwitchSection(BC->MOFI->getTextSection());
        Streamer->EmitLabel(BC->Ctx->getOrCreateSymbol("__hot_end"));
      }

      ColdFunctionSeen = true;
      if (opts::SplitFunctions != BinaryFunction::ST_NONE) {
        DEBUG(dbgs() << "BOLT-DEBUG: generating code for split functions\n");
        for (auto *FPtr : SortedFunctions) {
          if (!FPtr->isSplit() || !FPtr->isSimple())
            continue;
          emitFunction(*Streamer, *FPtr, *BC.get(), /*EmitColdPart=*/true);
        }
      }
      DEBUG(dbgs() << "BOLT-DEBUG: first cold function: " << Function << '\n');
    }

    if (!opts::Relocs &&
        (!Function.isSimple() || !opts::shouldProcess(Function))) {
      ++CurrentIndex;
      continue;
    }

    DEBUG(dbgs() << "BOLT: generating code for function \""
                 << Function << "\" : "
                 << Function.getFunctionNumber() << '\n');

    emitFunction(*Streamer, Function, *BC.get(), /*EmitColdPart=*/false);

    if (!opts::Relocs && Function.isSplit())
      emitFunction(*Streamer, Function, *BC.get(), /*EmitColdPart=*/true);

    ++CurrentIndex;
  }

  if (!ColdFunctionSeen && opts::HotText) {
    Streamer->SwitchSection(BC->MOFI->getTextSection());
    Streamer->EmitLabel(BC->Ctx->getOrCreateSymbol("__hot_end"));
  }

  if (opts::Relocs) {
    emitDataSections(Streamer.get());
  }


  if (opts::UpdateDebugSections)
    updateDebugLineInfoForNonSimpleFunctions();

  // Relocate .eh_frame to .eh_frame_old.
  if (EHFrameSection.getObject() != nullptr) {
    relocateEHFrameSection();
    emitDataSection(Streamer.get(), EHFrameSection, ".eh_frame_old");
  }

  Streamer->Finish();

  //////////////////////////////////////////////////////////////////////////////
  // Assign addresses to new functions/sections.
  //////////////////////////////////////////////////////////////////////////////

  if (opts::UpdateDebugSections) {
    // Compute offsets of tables in .debug_line for each compile unit.
    updateLineTableOffsets();
  }

  // Get output object as ObjectFile.
  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(BOS->str(), "in-memory object file", false);
  ErrorOr<std::unique_ptr<object::ObjectFile>> ObjOrErr =
    object::ObjectFile::createObjectFile(ObjectMemBuffer->getMemBufferRef());
  check_error(ObjOrErr.getError(), "error creating in-memory object");

  auto Resolver = orc::createLambdaResolver(
          [&](const std::string &Name) {
            DEBUG(dbgs() << "BOLT: looking for " << Name << "\n");
            auto I = BC->GlobalSymbols.find(Name);
            if (I == BC->GlobalSymbols.end())
              return RuntimeDyld::SymbolInfo(nullptr);
            return RuntimeDyld::SymbolInfo(I->second,
                                           JITSymbolFlags::None);
          },
          [](const std::string &S) {
            DEBUG(dbgs() << "BOLT: resolving " << S << "\n");
            return nullptr;
          }
      );
  Resolver->setAllowsZeroSymbols(true);

  auto ObjectsHandle = OLT.addObjectSet(
        singletonSet(std::move(ObjOrErr.get())),
        EFMM.get(),
        std::move(Resolver),
        /* ProcessAllSections = */true);

  // Is there benefit in using notifyObjectLoaded() to remap sections?
  mapFileSections(ObjectsHandle);

  if (opts::UpdateDebugSections) {
    MCAsmLayout Layout(
        static_cast<MCObjectStreamer *>(Streamer.get())->getAssembler());

    for (auto &BFI : BinaryFunctions) {
      auto &Function = BFI.second;
      for (auto &BB : Function) {
        if (!(BB.getLabel()->isDefined(false) &&
              BB.getEndLabel() && BB.getEndLabel()->isDefined(false))) {
          continue;
        }
        uint64_t BaseAddress = (BB.isCold() ? Function.cold().getAddress()
                                            : Function.getAddress());
        uint64_t BeginAddress =
            BaseAddress + Layout.getSymbolOffset(*BB.getLabel());
        uint64_t EndAddress =
            BaseAddress + Layout.getSymbolOffset(*BB.getEndLabel());
        BB.setOutputAddressRange(std::make_pair(BeginAddress, EndAddress));
      }
    }
  }

  OLT.emitAndFinalize(ObjectsHandle);

  const auto *EntryFunction = getBinaryFunctionContainingAddress(EntryPoint);
  assert(EntryFunction && "cannot find function for entry point");
  auto JITS = OLT.findSymbol(EntryFunction->getSymbol()->getName(), false);
  EntryPoint = JITS.getAddress();

  if (opts::KeepTmp)
    TempOut->keep();
}

void RewriteInstance::mapFileSections(
    orc::ObjectLinkingLayer<>::ObjSetHandleT &ObjectsHandle) {
  NewTextSectionStartAddress = NextAvailableAddress;
  if (opts::Relocs) {
    auto SMII = EFMM->SectionMapInfo.find(".text");
    assert(SMII != EFMM->SectionMapInfo.end() &&
           ".text not found in output");
    auto &SI = SMII->second;

    uint64_t NewTextSectionOffset = 0;
    if (opts::UseOldText && SI.Size <= OldTextSectionSize) {
      outs() << "BOLT-INFO: using original .text for new code\n";
      // Utilize the original .text for storage.
      NewTextSectionStartAddress = OldTextSectionAddress;
      NewTextSectionOffset = OldTextSectionOffset;
      auto Padding = OffsetToAlignment(NewTextSectionStartAddress, PageAlign);
      if (Padding + SI.Size <= OldTextSectionSize) {
        outs() << "BOLT-INFO: using 0x200000 alignment\n";
        NewTextSectionStartAddress += Padding;
        NewTextSectionOffset += Padding;
      }
    } else {
      if (opts::UseOldText) {
        errs() << "BOLT-ERROR: original .text too small to fit the new code. "
               << SI.Size << " bytes needed, have " << OldTextSectionSize
               << " bytes available.\n";
      }
      auto Padding = OffsetToAlignment(NewTextSectionStartAddress, PageAlign);
      NextAvailableAddress += Padding;
      NewTextSectionStartAddress = NextAvailableAddress;
      NewTextSectionOffset = getFileOffsetForAddress(NextAvailableAddress);
      NextAvailableAddress += Padding + SI.Size;
    }
    SI.FileAddress = NewTextSectionStartAddress;
    SI.FileOffset = NewTextSectionOffset;

    DEBUG(dbgs() << "BOLT: mapping .text 0x"
                 << Twine::utohexstr(SMII->second.AllocAddress)
                 << " to 0x" << Twine::utohexstr(NewTextSectionStartAddress)
                 << '\n');
    OLT.mapSectionAddress(ObjectsHandle,
                          SI.SectionID,
                          NewTextSectionStartAddress);
  } else {
    for (auto &BFI : BinaryFunctions) {
      auto &Function = BFI.second;
      if (!Function.isSimple() || !opts::shouldProcess(Function))
        continue;

      auto TooLarge = false;
      auto SMII = EFMM->SectionMapInfo.find(Function.getCodeSectionName());
      assert(SMII != EFMM->SectionMapInfo.end() &&
             "cannot find section for function");
      DEBUG(dbgs() << "BOLT: mapping 0x"
                   << Twine::utohexstr(SMII->second.AllocAddress)
                   << " to 0x" << Twine::utohexstr(Function.getAddress())
                   << '\n');
      OLT.mapSectionAddress(ObjectsHandle,
                            SMII->second.SectionID,
                            Function.getAddress());
      Function.setImageAddress(SMII->second.AllocAddress);
      Function.setImageSize(SMII->second.Size);
      if (Function.getImageSize() > Function.getMaxSize()) {
        TooLarge = true;
        FailedAddresses.emplace_back(Function.getAddress());
      }

      // Map jump tables if updating in-place.
      if (opts::JumpTables == JTS_BASIC) {
        for (auto &JTI : Function.JumpTables) {
          auto &JT = JTI.second;
          auto SMII = EFMM->SectionMapInfo.find(JT.SectionName);
          assert(SMII != EFMM->SectionMapInfo.end() &&
                 "cannot find section for jump table");
          JT.SecInfo = &SMII->second;
          JT.SecInfo->FileAddress = JT.Address;
          DEBUG(dbgs() << "BOLT-DEBUG: mapping " << JT.SectionName << " to 0x"
                       << Twine::utohexstr(JT.Address) << '\n');
          OLT.mapSectionAddress(ObjectsHandle,
                                JT.SecInfo->SectionID,
                                JT.Address);
        }
      }

      if (!Function.isSplit())
        continue;

      SMII = EFMM->SectionMapInfo.find(Function.getColdCodeSectionName());
      assert(SMII != EFMM->SectionMapInfo.end() &&
             "cannot find section for cold part");
      // Cold fragments are aligned at 16 bytes.
      NextAvailableAddress = RoundUpToAlignment(NextAvailableAddress, 16);
      auto &ColdPart = Function.cold();
      if (TooLarge) {
        // The corresponding FDE will refer to address 0.
        ColdPart.setAddress(0);
        ColdPart.setImageAddress(0);
        ColdPart.setImageSize(0);
        ColdPart.setFileOffset(0);
      } else {
        ColdPart.setAddress(NextAvailableAddress);
        ColdPart.setImageAddress(SMII->second.AllocAddress);
        ColdPart.setImageSize(SMII->second.Size);
        ColdPart.setFileOffset(getFileOffsetForAddress(NextAvailableAddress));
      }

      DEBUG(dbgs() << "BOLT: mapping cold fragment 0x"
                   << Twine::utohexstr(ColdPart.getImageAddress())
                   << " to 0x"
                   << Twine::utohexstr(ColdPart.getAddress())
                   << " with size "
                   << Twine::utohexstr(ColdPart.getImageSize()) << '\n');
      OLT.mapSectionAddress(ObjectsHandle,
                            SMII->second.SectionID,
                            ColdPart.getAddress());

      NextAvailableAddress += ColdPart.getImageSize();
    }

    // Add the new text section aggregating all existing code sections.
    // This is pseudo-section that serves a purpose of creating a corresponding
    // entry in section header table.
    auto NewTextSectionSize = NextAvailableAddress - NewTextSectionStartAddress;
    if (NewTextSectionSize) {
      EFMM->SectionMapInfo[BOLTSecPrefix + ".text"] =
          SectionInfo(0,
                      NewTextSectionSize,
                      16,
                      true /*IsCode*/,
                      true /*IsReadOnly*/,
                      true /*IsLocal*/,
                      NewTextSectionStartAddress,
                      getFileOffsetForAddress(NewTextSectionStartAddress));
    }
  }

  // Map special sections to their addresses in the output image.
  // These are the sections that we generate via MCStreamer.
  // The order is important.
  std::vector<std::string> Sections = { ".eh_frame", ".eh_frame_old",
                                        ".gcc_except_table",
                                        ".rodata", ".rodata.cold" };
  for (auto &SectionName : Sections) {
    auto SMII = EFMM->SectionMapInfo.find(SectionName);
    if (SMII == EFMM->SectionMapInfo.end())
      continue;
    SectionInfo &SI = SMII->second;
    NextAvailableAddress = RoundUpToAlignment(NextAvailableAddress,
                                              SI.Alignment);
    DEBUG(dbgs() << "BOLT: mapping section " << SectionName << " (0x"
                 << Twine::utohexstr(SI.AllocAddress)
                 << ") to 0x" << Twine::utohexstr(NextAvailableAddress)
                 << '\n');

    OLT.mapSectionAddress(ObjectsHandle,
                          SI.SectionID,
                          NextAvailableAddress);
    SI.FileAddress = NextAvailableAddress;
    SI.FileOffset = getFileOffsetForAddress(NextAvailableAddress);

    NextAvailableAddress += SI.Size;
  }

  // Handling for sections with relocations.
  for (auto &SRI : BC->SectionRelocations) {
    auto &Section = SRI.first;
    StringRef SectionName;
    Section.getName(SectionName);
    auto SMII = EFMM->SectionMapInfo.find(OrgSecPrefix +
                                          std::string(SectionName));
    if (SMII == EFMM->SectionMapInfo.end())
      continue;
    SectionInfo &SI = SMII->second;

    if (SI.FileAddress) {
      DEBUG(dbgs() << "BOLT-DEBUG: section " << SectionName
                   << " is already mapped at 0x"
                   << Twine::utohexstr(SI.FileAddress) << '\n');
      continue;
    }
    DEBUG(dbgs() << "BOLT: mapping original section " << SectionName << " (0x"
                 << Twine::utohexstr(SI.AllocAddress)
                 << ") to 0x" << Twine::utohexstr(Section.getAddress())
                 << '\n');

    OLT.mapSectionAddress(ObjectsHandle,
                          SI.SectionID,
                          Section.getAddress());
    SI.FileAddress = Section.getAddress();

    StringRef SectionContents;
    Section.getContents(SectionContents);
    SI.FileOffset = SectionContents.data() - InputFile->getData().data();
  }
}

void RewriteInstance::emitDataSection(MCStreamer *Streamer, SectionRef Section,
                                      std::string Name) {
  StringRef SectionName;
  if (!Name.empty())
    SectionName = Name;
  else
    Section.getName(SectionName);
  auto *ELFSection = BC->Ctx->getELFSection(SectionName,
                                            ELF::SHT_PROGBITS,
                                            ELF::SHF_WRITE | ELF::SHF_ALLOC);
  StringRef SectionContents;
  Section.getContents(SectionContents);

  Streamer->SwitchSection(ELFSection);
  Streamer->EmitValueToAlignment(Section.getAlignment());

  DEBUG(dbgs() << "BOLT-DEBUG: emitting section " << SectionName << '\n');

  auto SRI = BC->SectionRelocations.find(Section);
  if (SRI == BC->SectionRelocations.end()) {
    Streamer->EmitBytes(SectionContents);
    return;
  }

  auto &Relocations = SRI->second;
  uint64_t SectionOffset = 0;
  for (auto &Relocation : Relocations) {
    assert(Relocation.Offset < Section.getSize() && "overflow detected");
    if (SectionOffset < Relocation.Offset) {
      Streamer->EmitBytes(
          SectionContents.substr(SectionOffset,
            Relocation.Offset - SectionOffset));
      SectionOffset = Relocation.Offset;
    }
    DEBUG(dbgs() << "BOLT-DEBUG: emitting relocation for symbol "
                 << Relocation.Symbol->getName() << " at offset 0x"
                 << Twine::utohexstr(Relocation.Offset)
                 << " with size "
                 << Relocation::getSizeForType(Relocation.Type) << '\n');
    auto RelocationSize = Relocation.emit(Streamer);
    SectionOffset += RelocationSize;
  }
  assert(SectionOffset <= SectionContents.size() && "overflow error");
  if (SectionOffset < SectionContents.size()) {
    Streamer->EmitBytes(SectionContents.substr(SectionOffset));
  }
}

void RewriteInstance::emitDataSections(MCStreamer *Streamer) {
  for (auto &SRI : BC->SectionRelocations) {
    auto &Section = SRI.first;

    StringRef SectionName;
    Section.getName(SectionName);

    assert(SectionName != ".eh_frame" && "should not emit .eh_frame as data");

    auto EmitName = OrgSecPrefix + std::string(SectionName);
    emitDataSection(Streamer, Section, EmitName);
  }
}

bool RewriteInstance::checkLargeFunctions() {
  if (opts::Relocs)
    return false;

  LargeFunctions.clear();
  for (auto &BFI : BinaryFunctions) {
    auto &Function = BFI.second;

    // Ignore this function if we failed to map it to the output binary
    if (Function.getImageAddress() == 0 || Function.getImageSize() == 0)
      continue;

    if (Function.getImageSize() <= Function.getMaxSize())
      continue;

    LargeFunctions.insert(BFI.first);
  }
  return !LargeFunctions.empty();
}

void RewriteInstance::patchELFPHDRTable() {
  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(InputFile);
  if (!ELF64LEFile) {
    errs() << "BOLT-ERROR: only 64-bit LE ELF binaries are supported\n";
    exit(1);
  }
  auto Obj = ELF64LEFile->getELFFile();
  auto &OS = Out->os();

  // Write/re-write program headers.
  Phnum = Obj->getHeader()->e_phnum;
  if (PHDRTableOffset) {
    // Writing new pheader table.
    Phnum += 1; // only adding one new segment
    // Segment size includes the size of the PHDR area.
    NewTextSegmentSize = NextAvailableAddress - PHDRTableAddress;
  } else {
    assert(!PHDRTableAddress && "unexpected address for program header table");
    // Update existing table.
    PHDRTableOffset = Obj->getHeader()->e_phoff;
    NewTextSegmentSize = NextAvailableAddress - NewTextSegmentAddress;
  }
  OS.seek(PHDRTableOffset);

  bool ModdedGnuStack = false;
  bool AddedSegment = false;

  // Copy existing program headers with modifications.
  for (auto &Phdr : Obj->program_headers()) {
    auto NewPhdr = Phdr;
    if (PHDRTableAddress && Phdr.p_type == ELF::PT_PHDR) {
      NewPhdr.p_offset = PHDRTableOffset;
      NewPhdr.p_vaddr = PHDRTableAddress;
      NewPhdr.p_paddr = PHDRTableAddress;
      NewPhdr.p_filesz = sizeof(NewPhdr) * Phnum;
      NewPhdr.p_memsz = sizeof(NewPhdr) * Phnum;
    } else if (Phdr.p_type == ELF::PT_GNU_EH_FRAME) {
      auto SMII = EFMM->SectionMapInfo.find(".eh_frame_hdr");
      if (SMII != EFMM->SectionMapInfo.end()) {
        auto &EHFrameHdrSecInfo = SMII->second;
        NewPhdr.p_offset = EHFrameHdrSecInfo.FileOffset;
        NewPhdr.p_vaddr = EHFrameHdrSecInfo.FileAddress;
        NewPhdr.p_paddr = EHFrameHdrSecInfo.FileAddress;
        NewPhdr.p_filesz = EHFrameHdrSecInfo.Size;
        NewPhdr.p_memsz = EHFrameHdrSecInfo.Size;
      }
    } else if (opts::UseGnuStack && Phdr.p_type == ELF::PT_GNU_STACK) {
      NewPhdr.p_type = ELF::PT_LOAD;
      NewPhdr.p_offset = NewTextSegmentOffset;
      NewPhdr.p_vaddr = NewTextSegmentAddress;
      NewPhdr.p_paddr = NewTextSegmentAddress;
      NewPhdr.p_filesz = NewTextSegmentSize;
      NewPhdr.p_memsz = NewTextSegmentSize;
      NewPhdr.p_flags = ELF::PF_X | ELF::PF_R;
      NewPhdr.p_align = PageAlign;
      ModdedGnuStack = true;
    } else if (!opts::UseGnuStack && Phdr.p_type == ELF::PT_DYNAMIC) {
      // Insert new pheader
      ELFFile<ELF64LE>::Elf_Phdr NewTextPhdr;
      NewTextPhdr.p_type = ELF::PT_LOAD;
      NewTextPhdr.p_offset = PHDRTableOffset;
      NewTextPhdr.p_vaddr = PHDRTableAddress;
      NewTextPhdr.p_paddr = PHDRTableAddress;
      NewTextPhdr.p_filesz = NewTextSegmentSize;
      NewTextPhdr.p_memsz = NewTextSegmentSize;
      NewTextPhdr.p_flags = ELF::PF_X | ELF::PF_R;
      NewTextPhdr.p_align = PageAlign;
      OS.write(reinterpret_cast<const char *>(&NewTextPhdr),
               sizeof(NewTextPhdr));
      AddedSegment = true;
    }
    OS.write(reinterpret_cast<const char *>(&NewPhdr), sizeof(NewPhdr));
  }

  assert((!opts::UseGnuStack || ModdedGnuStack) &&
         "could not find GNU_STACK program header to modify");

  assert((opts::UseGnuStack || AddedSegment) &&
         "could not add program header for the new segment");
}

namespace {
void writePadding(raw_pwrite_stream &OS, unsigned BytesToWrite) {
  for (unsigned I = 0; I < BytesToWrite; ++I)
    OS.write((unsigned char)0);
}
}

void RewriteInstance::rewriteNoteSections() {
  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(InputFile);
  if (!ELF64LEFile) {
    errs() << "BOLT-ERROR: only 64-bit LE ELF binaries are supported\n";
    exit(1);
  }
  auto Obj = ELF64LEFile->getELFFile();
  auto &OS = Out->os();

  uint64_t NextAvailableOffset = getFileOffsetForAddress(NextAvailableAddress);
  assert(NextAvailableOffset >= FirstNonAllocatableOffset &&
         "next available offset calculation failure");
  OS.seek(NextAvailableOffset);

  // Copy over non-allocatable section contents and update file offsets.
  for (auto &Section : Obj->sections()) {
    if (Section.sh_type == ELF::SHT_NULL)
      continue;
    if (Section.sh_flags & ELF::SHF_ALLOC)
      continue;

    if (Section.sh_type == ELF::SHT_RELA)
      continue;

    // Insert padding as needed.
    if (Section.sh_addralign > 1) {
      auto PaddingSize = OffsetToAlignment(NextAvailableOffset,
                                           Section.sh_addralign);
      writePadding(OS, PaddingSize);
      NextAvailableOffset += PaddingSize;

      assert(Section.sh_size % Section.sh_addralign == 0 &&
             "section size does not match section alignment");
    }

    ErrorOr<StringRef> SectionName = Obj->getSectionName(&Section);
    check_error(SectionName.getError(), "cannot get section name");

    // New section size.
    uint64_t Size = 0;

    // Copy over section contents unless it's one of the sections we overwrite.
    if (!willOverwriteSection(*SectionName)) {
      Size = Section.sh_size;
      std::string Data = InputFile->getData().substr(Section.sh_offset, Size);
      auto SectionPatchersIt = SectionPatchers.find(*SectionName);
      if (SectionPatchersIt != SectionPatchers.end()) {
        (*SectionPatchersIt->second).patchBinary(Data);
      }
      OS << Data;
    }

    if (Section.sh_type == ELF::SHT_SYMTAB) {
      NewSymTabOffset = NextAvailableOffset;
    }

    // Address of extension to the section.
    uint64_t Address{0};

    // Perform section post-processing.

    auto SII = EFMM->NoteSectionInfo.find(*SectionName);
    if (SII != EFMM->NoteSectionInfo.end()) {
      auto &SI = SII->second;
      assert(SI.Alignment <= Section.sh_addralign &&
             "alignment exceeds value in file");

      // Write section extension.
      Address = SI.AllocAddress;
      if (Address) {
        DEBUG(dbgs() << "BOLT: " << (Size ? "appending" : "writing")
                     << " contents to section "
                     << *SectionName << '\n');
        OS.write(reinterpret_cast<const char *>(Address), SI.Size);
        Size += SI.Size;
      }

      if (!SI.PendingRelocs.empty()) {
        DEBUG(dbgs() << "BOLT-DEBUG: processing relocs for section "
                     << *SectionName << '\n');
        for (auto &Reloc : SI.PendingRelocs) {
          DEBUG(dbgs() << "BOLT-DEBUG: writing value "
                       << Twine::utohexstr(Reloc.Value)
                       << " of size " << (unsigned)Reloc.Size
                       << " at offset "
                       << Twine::utohexstr(Reloc.Offset) << '\n');
          assert(Reloc.Size == 4 &&
                 "only relocations of size 4 are supported at the moment");
          OS.pwrite(reinterpret_cast<const char*>(&Reloc.Value),
                    Reloc.Size,
                    NextAvailableOffset + Reloc.Offset);
        }
      }
    }

    // Set/modify section info.
    EFMM->NoteSectionInfo[*SectionName] =
      SectionInfo(Address,
                  Size,
                  Section.sh_addralign,
                  /*IsCode=*/false,
                  /*IsReadOnly=*/false,
                  /*IsLocal=*/false,
                  /*FileAddress=*/0,
                  NextAvailableOffset);

    NextAvailableOffset += Size;
  }
}

template <typename ELFT>
void RewriteInstance::writeStringTable(ELFObjectFile<ELFT> *File) {
  auto *Obj = File->getELFFile();
  auto &OS = Out->os();

  // Pre-populate section header string table.
  for (auto &Section : Obj->sections()) {
    ErrorOr<StringRef> SectionName = Obj->getSectionName(&Section);
    check_error(SectionName.getError(), "cannot get section name");
    SHStrTab.add(*SectionName);
    if (willOverwriteSection(*SectionName))
      SHStrTab.add(OrgSecPrefix + SectionName->str());
  }
  for (auto &SMII : EFMM->SectionMapInfo) {
    SHStrTab.add(SMII.first);
  }
  SHStrTab.finalize(StringTableBuilder::ELF);

  auto SII = EFMM->NoteSectionInfo.find(".shstrtab");
  assert(SII != EFMM->NoteSectionInfo.end() && "cannot find .shstrtab");
  auto &SI = SII->second;
  SI.FileOffset = OS.tell();
  SI.Size = SHStrTab.data().size();

  // Write data for the table.
  OS << SHStrTab.data();
}

// Rewrite section header table inserting new entries as needed. The sections
// header table size itself may affect the offsets of other sections,
// so we are placing it at the end of the binary.
//
// As we rewrite entries we need to track how many sections were inserted
// as it changes the sh_link value.
//
// The following are assumptions about file modifications:
//    * There are no modifications done to existing allocatable sections.
//    * All new allocatable sections are written immediately after existing
//      allocatable sections.
//    * There could be modifications done to non-allocatable sections, e.g.
//      size could be increased.
//    * New non-allocatable sections are added to the end of the file.
template <typename ELFT>
void RewriteInstance::patchELFSectionHeaderTable(ELFObjectFile<ELFT> *File) {
  using Elf_Shdr = typename ELFObjectFile<ELFT>::Elf_Shdr;

  auto *Obj = File->getELFFile();
  auto &OS = Out->os();
  auto SHTOffset = OS.tell();
  uint64_t CurrentSectionIndex = 0;

  NewSectionIndex.resize(Obj->getNumSections());

  auto PaddingSize = OffsetToAlignment(SHTOffset, sizeof(Elf_Shdr));
  writePadding(OS, PaddingSize);
  SHTOffset += PaddingSize;

  // Copy over entries for original allocatable sections with minor
  // modifications (e.g. name).
  for (auto &Section : Obj->sections()) {
    // Always ignore this section.
    if (Section.sh_type == ELF::SHT_NULL) {
      OS.write(reinterpret_cast<const char *>(&Section), sizeof(Section));
      NewSectionIndex[0] = CurrentSectionIndex++;
      continue;
    }

    // Skip non-allocatable sections.
    if (!(Section.sh_flags & ELF::SHF_ALLOC))
      continue;

    ErrorOr<StringRef> SectionName = Obj->getSectionName(&Section);
    check_error(SectionName.getError(), "cannot get section name");

    auto NewSection = Section;
    if (*SectionName == ".bss") {
      // .bss section offset matches that of the next section.
      NewSection.sh_offset = NewTextSegmentOffset;
    }

    if (willOverwriteSection(*SectionName)) {
      NewSection.sh_name = SHStrTab.getOffset(OrgSecPrefix +
                                              SectionName->str());
    } else {
      NewSection.sh_name = SHStrTab.getOffset(*SectionName);
    }

    if (Section.sh_addr <= NewTextSectionStartAddress &&
        Section.sh_addr + Section.sh_size > NewTextSectionStartAddress) {
      NewTextSectionIndex = CurrentSectionIndex;
    }

    OS.write(reinterpret_cast<const char *>(&NewSection), sizeof(NewSection));
    NewSectionIndex[std::distance(Obj->section_begin(), &Section)] =
      CurrentSectionIndex++;

  }

  // Create entries for new allocatable sections.
  //
  // Skip sections we overwrite in-place (like data sections).
  std::vector<Elf_Shdr> SectionsToRewrite;
  for (auto &SMII : EFMM->SectionMapInfo) {
    const auto &SectionName = SMII.first;
    const auto &SI = SMII.second;
    // Ignore function sections.
    if (SI.FileAddress < NewTextSegmentAddress) {
      if (opts::Verbosity)
        outs() << "BOLT-INFO: not writing section header for existing section "
               << SMII.first << '\n';
      continue;
    }
    if (opts::Verbosity >= 1)
      outs() << "BOLT-INFO: writing section header for " << SMII.first << '\n';
    Elf_Shdr NewSection;
    NewSection.sh_name = SHStrTab.getOffset(SectionName);
    NewSection.sh_type = ELF::SHT_PROGBITS;
    NewSection.sh_addr = SI.FileAddress;
    NewSection.sh_offset = SI.FileOffset;
    NewSection.sh_size = SI.Size;
    NewSection.sh_entsize = 0;
    NewSection.sh_flags = ELF::SHF_ALLOC | ELF::SHF_EXECINSTR;
    NewSection.sh_link = 0;
    NewSection.sh_info = 0;
    NewSection.sh_addralign = SI.Alignment;
    SectionsToRewrite.emplace_back(NewSection);
  }

  // Write section header entries for new allocatable sections in offset order.
  std::stable_sort(SectionsToRewrite.begin(), SectionsToRewrite.end(),
      [] (Elf_Shdr A, Elf_Shdr B) {
        return A.sh_offset < B.sh_offset;
      });
  for (auto &SI : SectionsToRewrite) {
    if (SI.sh_addr <= NewTextSectionStartAddress &&
        SI.sh_addr + SI.sh_size > NewTextSectionStartAddress) {
      NewTextSectionIndex = CurrentSectionIndex;
    }
    OS.write(reinterpret_cast<const char *>(&SI),
             sizeof(SI));
    ++CurrentSectionIndex;
  }

  int64_t NumNewSections = SectionsToRewrite.size();

  // Copy over entries for non-allocatable sections performing necessary
  // adjustments.
  for (auto &Section : Obj->sections()) {
    if (Section.sh_type == ELF::SHT_NULL)
      continue;
    if (Section.sh_flags & ELF::SHF_ALLOC)
      continue;

    ErrorOr<StringRef> SectionName = Obj->getSectionName(&Section);
    check_error(SectionName.getError(), "cannot get section name");

    if (Section.sh_type == ELF::SHT_RELA) {
      if (opts::Verbosity)
        outs() << "BOLT-INFO: omitting section header for relocation section "
               << *SectionName << '\n';
      --NumNewSections;
      continue;
    }

    auto SII = EFMM->NoteSectionInfo.find(*SectionName);
    assert(SII != EFMM->NoteSectionInfo.end() &&
           "missing section info for non-allocatable section");

    auto NewSection = Section;
    NewSection.sh_offset = SII->second.FileOffset;
    NewSection.sh_size = SII->second.Size;
    NewSection.sh_name = SHStrTab.getOffset(*SectionName);

    // Adjust sh_link for sections that use it.
    if (Section.sh_link)
      NewSection.sh_link = Section.sh_link + NumNewSections;

    // Adjust sh_info for relocation sections.
    if (Section.sh_type == ELF::SHT_REL || Section.sh_type == ELF::SHT_RELA) {
      if (Section.sh_info)
        NewSection.sh_info = Section.sh_info + NumNewSections;
    }

    OS.write(reinterpret_cast<const char *>(&NewSection), sizeof(NewSection));
    NewSectionIndex[std::distance(Obj->section_begin(), &Section)] =
      CurrentSectionIndex++;
  }

  // Using new section indices map updates sh_link and sh_info where needed.
  //

  // New section header string table goes last.

  // Fix ELF header.
  auto NewEhdr = *Obj->getHeader();
  NewEhdr.e_entry = EntryPoint;
  NewEhdr.e_phoff = PHDRTableOffset;
  NewEhdr.e_phnum = Phnum;
  NewEhdr.e_shoff = SHTOffset;
  NewEhdr.e_shnum = NewEhdr.e_shnum + NumNewSections;
  NewEhdr.e_shstrndx = NewEhdr.e_shstrndx + NumNewSections;
  OS.pwrite(reinterpret_cast<const char *>(&NewEhdr), sizeof(NewEhdr), 0);

  assert(NewEhdr.e_shnum == CurrentSectionIndex &&
         "internal calculation error");
}

// FIXME: proper size for symbols based on output. Current method doesn't
// work well with split functions.
template <typename ELFT>
void RewriteInstance::patchELFSymTabs(ELFObjectFile<ELFT> *File) {
  if (!opts::Relocs)
    return;

  auto *Obj = File->getELFFile();
  auto &OS = Out->os();

  using Elf_Shdr = typename ELFObjectFile<ELFT>::Elf_Shdr;
  using Elf_Sym  = typename ELFObjectFile<ELFT>::Elf_Sym;

  auto updateSymbolTable = [&](uint64_t SymTabOffset, const Elf_Shdr *Section) {
    auto StringSectionOrError = Obj->getStringTableForSymtab(*Section);
    for (const Elf_Sym &Symbol : Obj->symbols(Section)) {
      auto NewSymbol = Symbol;
      if (auto NewAddress = getNewFunctionAddress(Symbol.st_value)) {
        std::size_t Size = 0;
        auto BFI = BinaryFunctions.upper_bound(NewAddress);
        if (BFI != BinaryFunctions.end()) {
          Size = BFI->first - NewAddress;
        } else {
          Size = BFI->second.getSize();
        }
        DEBUG(dbgs() << "BOLT-DEBUG: patching symbol address 0x"
                     << Twine::utohexstr(Symbol.st_value) << " with 0x"
                     << Twine::utohexstr(NewAddress)
                     << " size " << Size << '\n');
        NewSymbol.st_value = NewAddress;
        NewSymbol.st_shndx = NewTextSectionIndex;
      } else {
        if (NewSymbol.st_shndx < ELF::SHN_LORESERVE) {
          NewSymbol.st_shndx = NewSectionIndex[NewSymbol.st_shndx];
        }
      }

      if (opts::HotText) {
        auto updateSymbolValue = [&](const StringRef Name) {
          NewSymbol.st_value = getNewValueForSymbol(Name);
          NewSymbol.st_shndx = ELF::SHN_ABS;
          outs() << "BOLT-INFO: setting " << Name << " to 0x"
                 << Twine::utohexstr(NewSymbol.st_value) << '\n';
          return true;
        };

        auto SymbolName = Symbol.getName(*StringSectionOrError);
        assert(SymbolName && "cannot get symbol name");
        if (*SymbolName == "__hot_start" || *SymbolName == "__hot_end")
          updateSymbolValue(*SymbolName);
      }

      OS.pwrite(reinterpret_cast<const char *>(&NewSymbol),
                sizeof(NewSymbol),
                SymTabOffset +
                  (&Symbol - Obj->symbol_begin(Section)) * sizeof(Elf_Sym));
    }
  };

  // Update dynamic symbol table.
  const Elf_Shdr *DynSymSection = nullptr;
  for (const Elf_Shdr &Section : Obj->sections()) {
    if (Section.sh_type == ELF::SHT_DYNSYM) {
      DynSymSection = &Section;
      break;
    }
  }
  assert(DynSymSection && "no dynamic symbol table found");
  updateSymbolTable(DynSymSection->sh_offset, DynSymSection);

  // Update regular symbol table.
  const Elf_Shdr *SymTabSection = nullptr;
  for (const auto &Section : Obj->sections()) {
    if (Section.sh_type == ELF::SHT_SYMTAB) {
      SymTabSection = &Section;
      break;
    }
  }
  if (!SymTabSection) {
    errs() << "BOLT-WARNING: no symbol table found\n";
    return;
  }
  assert(NewSymTabOffset && "expected symbol table offset to be set");
  updateSymbolTable(NewSymTabOffset, SymTabSection);
}

template <typename ELFT>
void RewriteInstance::patchELFRelaPLT(ELFObjectFile<ELFT> *File) {
  auto &OS = Out->os();

  SectionRef RelaPLTSection;
  for (const auto &Section : File->sections()) {
    StringRef SectionName;
    Section.getName(SectionName);
    if (SectionName == ".rela.plt") {
      RelaPLTSection = Section;
      break;
    }
  }
  if (!RelaPLTSection.getObject()) {
    errs() << "BOLT-INFO: no .rela.plt section found\n";
    return;
  }

  for (const auto &Rel : RelaPLTSection.relocations()) {
    if (Rel.getType() == ELF::R_X86_64_IRELATIVE) {
      DataRefImpl DRI = Rel.getRawDataRefImpl();
      const auto *RelA = File->getRela(DRI);
      auto Address = RelA->r_addend;
      auto NewAddress = getNewFunctionAddress(Address);
      DEBUG(dbgs() << "BOLT-DEBUG: patching IRELATIVE .rela.plt entry 0x"
                   << Twine::utohexstr(Address) << " with 0x"
                   << Twine::utohexstr(NewAddress) << '\n');
      auto NewRelA = *RelA;
      NewRelA.r_addend = NewAddress;
      OS.pwrite(reinterpret_cast<const char *>(&NewRelA), sizeof(NewRelA),
        reinterpret_cast<const char *>(RelA) - File->getData().data());
    }
  }
}

template <typename ELFT>
void RewriteInstance::patchELFGOT(ELFObjectFile<ELFT> *File) {
  auto &OS = Out->os();

  SectionRef GOTSection;
  for (const auto &Section : File->sections()) {
    StringRef SectionName;
    Section.getName(SectionName);
    if (SectionName == ".got") {
      GOTSection = Section;
      break;
    }
  }
  if (!GOTSection.getObject()) {
    errs() << "BOLT-INFO: no .got section found\n";
    return;
  }

  StringRef GOTContents;
  GOTSection.getContents(GOTContents);
  for (const uint64_t *GOTEntry =
        reinterpret_cast<const uint64_t *>(GOTContents.data());
       GOTEntry < reinterpret_cast<const uint64_t *>(GOTContents.data() +
                                                            GOTContents.size());
       ++GOTEntry) {
    if (auto NewAddress = getNewFunctionAddress(*GOTEntry)) {
      DEBUG(dbgs() << "BOLT-DEBUG: patching GOT entry 0x"
                   << Twine::utohexstr(*GOTEntry) << " with 0x"
                   << Twine::utohexstr(NewAddress) << '\n');
      OS.pwrite(reinterpret_cast<const char *>(&NewAddress), sizeof(NewAddress),
        reinterpret_cast<const char *>(GOTEntry) - File->getData().data());
    }
  }
}

template <typename ELFT>
void RewriteInstance::patchELFDynamic(ELFObjectFile<ELFT> *File) {
  auto *Obj = File->getELFFile();
  auto &OS = Out->os();

  using Elf_Phdr = typename ELFFile<ELFT>::Elf_Phdr;
  using Elf_Dyn  = typename ELFFile<ELFT>::Elf_Dyn;

  // Locate DYNAMIC by looking through program headers.
  uint64_t DynamicOffset = 0;
  const Elf_Phdr *DynamicPhdr = 0;
  for (auto &Phdr : Obj->program_headers()) {
    if (Phdr.p_type == ELF::PT_DYNAMIC) {
      DynamicOffset = Phdr.p_offset;
      DynamicPhdr = &Phdr;
      assert(Phdr.p_memsz == Phdr.p_filesz && "dynamic sizes should match");
      break;
    }
  }
  assert(DynamicPhdr && "missing dynamic in ELF binary");

  // Go through all dynamic entries and patch functions addresses with
  // new ones.
  ErrorOr<const Elf_Dyn *> DTB = Obj->dynamic_table_begin(DynamicPhdr);
  ErrorOr<const Elf_Dyn *> DTE = Obj->dynamic_table_end(DynamicPhdr);
  assert(DTB && DTE && "error accessing dynamic table");
  for (auto *DE = *DTB; DE != *DTE; ++DE) {
    auto NewDE = *DE;
    bool ShouldPatch = true;
    switch (DE->getTag()) {
    default:
      ShouldPatch = false;
      break;
    case ELF::DT_INIT:
    case ELF::DT_FINI:
      if (auto NewAddress = getNewFunctionAddress(DE->getPtr())) {
        DEBUG(dbgs() << "BOLT-DEBUG: patching dynamic entry of type "
                     << DE->getTag() << '\n');
        NewDE.d_un.d_ptr = NewAddress;
      }
      break;
    }
    if (ShouldPatch) {
      OS.pwrite(reinterpret_cast<const char *>(&NewDE), sizeof(NewDE),
                DynamicOffset + (DE - *DTB) * sizeof(*DE));
    }
  }
}

uint64_t RewriteInstance::getNewFunctionAddress(uint64_t OldAddress) {
  const auto *Function = getBinaryFunctionAtAddress(OldAddress);
  if (!Function)
    return 0;
  auto JITS = OLT.findSymbol(Function->getSymbol()->getName(), false);
  return JITS.getAddress();
}

void RewriteInstance::rewriteFile() {
  auto &OS = Out->os();

  // We obtain an asm-specific writer so that we can emit nops in an
  // architecture-specific way at the end of the function.
  auto MCE = BC->TheTarget->createMCCodeEmitter(*BC->MII, *BC->MRI, *BC->Ctx);
  auto MAB = BC->TheTarget->createMCAsmBackend(*BC->MRI, BC->TripleName, "");
  std::unique_ptr<MCStreamer> Streamer(
    BC->TheTarget->createMCObjectStreamer(*BC->TheTriple,
                                          *BC->Ctx,
                                          *MAB,
                                          OS,
                                          MCE,
                                          *BC->STI,
                                          /* RelaxAll */ false,
                                          /* DWARFMustBeAtTheEnd */ false));

  auto &Writer = static_cast<MCObjectStreamer *>(Streamer.get())
                     ->getAssembler()
                     .getWriter();

  // Make sure output stream has enough reserved space, otherwise
  // pwrite() will fail.
  auto Offset = OS.seek(getFileOffsetForAddress(NextAvailableAddress));
  assert(Offset == getFileOffsetForAddress(NextAvailableAddress) &&
         "error resizing output file");

  if (!opts::Relocs) {
    // Overwrite functions in the output file.
    uint64_t CountOverwrittenFunctions = 0;
    uint64_t OverwrittenScore = 0;
    for (auto &BFI : BinaryFunctions) {
      auto &Function = BFI.second;

      if (Function.getImageAddress() == 0 || Function.getImageSize() == 0)
        continue;

      if (Function.getImageSize() > Function.getMaxSize()) {
        if (opts::Verbosity >= 1) {
          errs() << "BOLT-WARNING: new function size (0x"
                 << Twine::utohexstr(Function.getImageSize())
                 << ") is larger than maximum allowed size (0x"
                 << Twine::utohexstr(Function.getMaxSize())
                 << ") for function " << Function << '\n';
        }
        FailedAddresses.emplace_back(Function.getAddress());
        continue;
      }

      if (Function.isSplit() && (Function.cold().getImageAddress() == 0 ||
                                 Function.cold().getImageSize() == 0))
        continue;

      OverwrittenScore += Function.getFunctionScore();
      // Overwrite function in the output file.
      if (opts::Verbosity >= 2) {
        outs() << "BOLT: rewriting function \"" << Function << "\"\n";
      }
      OS.pwrite(reinterpret_cast<char *>(Function.getImageAddress()),
                       Function.getImageSize(), Function.getFileOffset());

      // Write nops at the end of the function.
      auto Pos = OS.tell();
      OS.seek(Function.getFileOffset() + Function.getImageSize());
      MAB->writeNopData(Function.getMaxSize() - Function.getImageSize(),
                        &Writer);
      OS.seek(Pos);

      // Write jump tables if updating in-place.
      if (opts::JumpTables == JTS_BASIC) {
        for (auto &JTI : Function.JumpTables) {
          auto &JT = JTI.second;
          assert(JT.SecInfo && "section info for jump table expected");
          JT.SecInfo->FileOffset =
            getFileOffsetForAddress(JT.Address);
          assert(JT.SecInfo->FileOffset && "no matching offset in file");
          Out->os().pwrite(reinterpret_cast<char *>(JT.SecInfo->AllocAddress),
                           JT.SecInfo->Size,
                           JT.SecInfo->FileOffset);
        }
      }

      if (!Function.isSplit()) {
        ++CountOverwrittenFunctions;
        if (opts::MaxFunctions &&
            CountOverwrittenFunctions == opts::MaxFunctions) {
          outs() << "BOLT: maximum number of functions reached\n";
          break;
        }
        continue;
      }

      // Write cold part
      if (opts::Verbosity >= 2) {
        outs() << "BOLT: rewriting function \"" << Function
               << "\" (cold part)\n";
      }
      OS.pwrite(reinterpret_cast<char*>
                                            (Function.cold().getImageAddress()),
                       Function.cold().getImageSize(),
                       Function.cold().getFileOffset());

      // FIXME: write nops after cold part too.

      ++CountOverwrittenFunctions;
      if (opts::MaxFunctions &&
          CountOverwrittenFunctions == opts::MaxFunctions) {
        outs() << "BOLT: maximum number of functions reached\n";
        break;
      }
    }

    // Print function statistics.
    outs() << "BOLT: " << CountOverwrittenFunctions
           << " out of " << BinaryFunctions.size()
           << " functions were overwritten.\n";
    if (TotalScore != 0) {
      double Coverage = OverwrittenScore / (double)TotalScore * 100.0;
      outs() << format("BOLT: Rewritten functions cover %.2lf", Coverage)
             << "% of the execution count of simple functions of "
                "this binary.\n";
    }
  }

  if (opts::Relocs && opts::TrapOldCode) {
    auto SavedPos = OS.tell();
    // Overwrite function body to make sure we never execute these instructions.
    for (auto &BFI : BinaryFunctions) {
      auto &BF = BFI.second;
      if (!BF.getFileOffset())
        continue;
      OS.seek(BF.getFileOffset());
      for (unsigned I = 0; I < BF.getMaxSize(); ++I)
        OS.write((unsigned char)
            Streamer->getContext().getAsmInfo()->getTrapFillValue());
    }
    OS.seek(SavedPos);
  }

  // Write all non-local sections, i.e. those not emitted with the function.
  for (auto &SMII : EFMM->SectionMapInfo) {
    SectionInfo &SI = SMII.second;
    if (SI.IsLocal)
      continue;
    if (opts::Verbosity >= 1) {
      outs() << "BOLT: writing new section " << SMII.first << '\n';
      outs() << " data at 0x" << Twine::utohexstr(SI.AllocAddress) << '\n';
      outs() << " of size " << SI.Size << '\n';
      outs() << " at offset " << SI.FileOffset << '\n';
    }
    OS.pwrite(reinterpret_cast<const char *>(SI.AllocAddress),
                     SI.Size,
                     SI.FileOffset);
  }

  // If .eh_frame is present create .eh_frame_hdr.
  auto SMII = EFMM->SectionMapInfo.find(".eh_frame");
  if (SMII != EFMM->SectionMapInfo.end()) {
    writeEHFrameHeader(SMII->second);
  }

  // Patch program header table.
  patchELFPHDRTable();

  // Copy non-allocatable sections once allocatable part is finished.
  rewriteNoteSections();

  // Write string.
  writeStringTable();

  if (opts::Relocs) {
    // Patch dynamic section/segment.
    patchELFDynamic();

    patchELFRelaPLT();

    patchELFGOT();
  }

  // Update ELF book-keeping info.
  patchELFSectionHeaderTable();

  // Update symbol tables.
  if (opts::Relocs)
    patchELFSymTabs();

  // TODO: we should find a way to mark the binary as optimized by us.
  Out->keep();

  // If requested, open again the binary we just wrote to dump its EH Frame
  if (opts::DumpEHFrame) {
    ErrorOr<OwningBinary<Binary>> BinaryOrErr =
        createBinary(opts::OutputFilename);
    if (std::error_code EC = BinaryOrErr.getError())
      report_error(opts::OutputFilename, EC);
    Binary &Binary = *BinaryOrErr.get().getBinary();

    if (auto *E = dyn_cast<ELFObjectFileBase>(&Binary)) {
      DWARFContextInMemory DwCtx(*E, nullptr, true);
      const auto &EHFrame = DwCtx.getEHFrame();
      outs() << "BOLT-INFO: Dumping rewritten .eh_frame\n";
      EHFrame->dump(outs());
    }
  }
}

void RewriteInstance::writeEHFrameHeader(SectionInfo &EHFrameSecInfo) {
  DWARFFrame NewEHFrame(EHFrameSecInfo.FileAddress);
  NewEHFrame.parse(
    DataExtractor(StringRef(reinterpret_cast<const char *>(
                                                 EHFrameSecInfo.AllocAddress),
                            EHFrameSecInfo.Size),
                  BC->AsmInfo->isLittleEndian(),
                  BC->AsmInfo->getPointerSize()));
  if (!NewEHFrame.ParseError.empty()) {
    errs() << "BOLT-ERROR: EHFrame reader failed with message \""
           << NewEHFrame.ParseError << '\n';
    exit(1);
  }

  auto OldSMII = EFMM->SectionMapInfo.find(".eh_frame_old");
  assert(OldSMII != EFMM->SectionMapInfo.end() &&
         "expected .eh_frame_old to be present");
  auto &OldEHFrameSecInfo = OldSMII->second;
  DWARFFrame OldEHFrame(OldEHFrameSecInfo.FileAddress);
  OldEHFrame.parse(
    DataExtractor(StringRef(reinterpret_cast<const char *>(
                                              OldEHFrameSecInfo.AllocAddress),
                            OldEHFrameSecInfo.Size),
                  BC->AsmInfo->isLittleEndian(),
                  BC->AsmInfo->getPointerSize()));
  if (!OldEHFrame.ParseError.empty()) {
    errs() << "BOLT-ERROR: EHFrame reader failed with message \""
           << OldEHFrame.ParseError << '\n';
    exit(1);
  }

  DEBUG(dbgs() << "BOLT: writing a new .eh_frame_hdr\n");

  auto PaddingSize = OffsetToAlignment(NextAvailableAddress, EHFrameHdrAlign);
  writePadding(Out->os(), PaddingSize);
  NextAvailableAddress += PaddingSize;

  SectionInfo EHFrameHdrSecInfo;
  EHFrameHdrSecInfo.FileAddress = NextAvailableAddress;
  EHFrameHdrSecInfo.FileOffset = getFileOffsetForAddress(NextAvailableAddress);

  auto NewEHFrameHdr =
      CFIRdWrt->generateEHFrameHeader(OldEHFrame,
                                      NewEHFrame,
                                      EHFrameHdrSecInfo.FileAddress,
                                      FailedAddresses);

  EHFrameHdrSecInfo.Size = NewEHFrameHdr.size();

  assert(Out->os().tell() == EHFrameHdrSecInfo.FileOffset &&
         "offset mismatch");
  Out->os().write(NewEHFrameHdr.data(), EHFrameHdrSecInfo.Size);

  EFMM->SectionMapInfo[".eh_frame_hdr"] = EHFrameHdrSecInfo;

  NextAvailableAddress += EHFrameHdrSecInfo.Size;

  // Merge .eh_frame and .eh_frame_old so that gdb can locate all FDEs.
  EHFrameSecInfo.Size = OldEHFrameSecInfo.FileAddress + OldEHFrameSecInfo.Size
                        - EHFrameSecInfo.FileAddress;
  EFMM->SectionMapInfo.erase(OldSMII);
  DEBUG(dbgs() << "BOLT-DEBUG: size of .eh_frame after merge is "
               << EHFrameSecInfo.Size << '\n');
}

uint64_t RewriteInstance::getFileOffsetForAddress(uint64_t Address) const {
  // Check if it's possibly part of the new segment.
  if (Address >= NewTextSegmentAddress) {
    return Address - NewTextSegmentAddress + NewTextSegmentOffset;
  }

  // Find an existing segment that matches the address.
  const auto SegmentInfoI = EFMM->SegmentMapInfo.upper_bound(Address);
  if (SegmentInfoI == EFMM->SegmentMapInfo.begin())
    return 0;

  const auto &SegmentInfo = std::prev(SegmentInfoI)->second;
  if (Address < SegmentInfo.Address ||
      Address >= SegmentInfo.Address + SegmentInfo.FileSize)
    return 0;

  return  SegmentInfo.FileOffset + Address - SegmentInfo.Address;
}

bool RewriteInstance::willOverwriteSection(StringRef SectionName) {
  if (opts::UpdateDebugSections) {
    for (auto &OverwriteName : DebugSectionsToOverwrite) {
      if (SectionName == OverwriteName)
        return true;
    }
  }

  auto SMII = EFMM->SectionMapInfo.find(SectionName);
  if (SMII != EFMM->SectionMapInfo.end())
    return true;

  return false;
}

BinaryFunction *
RewriteInstance::getBinaryFunctionContainingAddress(uint64_t Address,
                                                    bool CheckPastEnd,
                                                    bool UseMaxSize) {
  auto FI = BinaryFunctions.upper_bound(Address);
  if (FI == BinaryFunctions.begin())
    return nullptr;
  --FI;

  const auto UsedSize = UseMaxSize ? FI->second.getMaxSize()
                                   : FI->second.getSize();

  if (Address >= FI->first + UsedSize + (CheckPastEnd ? 1 : 0))
    return nullptr;
  return &FI->second;
}

const BinaryFunction *
RewriteInstance::getBinaryFunctionAtAddress(uint64_t Address) const {
  const auto *Symbol = BC->getGlobalSymbolAtAddress(Address);
  if (!Symbol)
    return nullptr;

  return BC->getFunctionForSymbol(Symbol);
}
