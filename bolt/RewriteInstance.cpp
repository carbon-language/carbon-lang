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
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
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
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
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

static cl::opt<std::string>
OutputFilename("o", cl::desc("<output file>"), cl::Required);

static cl::list<std::string>
BreakFunctionNames("break-funcs",
                   cl::CommaSeparated,
                   cl::desc("list of functions to core dump on (debugging)"),
                   cl::value_desc("func1,func2,func3,..."),
                   cl::Hidden);

static cl::list<std::string>
FunctionNames("funcs",
              cl::CommaSeparated,
              cl::desc("list of functions to optimize"),
              cl::value_desc("func1,func2,func3,..."));

static cl::opt<std::string>
FunctionNamesFile("funcs-file",
                  cl::desc("file with list of functions to optimize"));

static cl::list<std::string>
SkipFunctionNames("skip-funcs",
                  cl::CommaSeparated,
                  cl::desc("list of functions to skip"),
                  cl::value_desc("func1,func2,func3,..."));

static cl::opt<std::string>
SkipFunctionNamesFile("skip-funcs-file",
                      cl::desc("file with list of functions to skip"));

static cl::opt<unsigned>
MaxFunctions("max-funcs",
             cl::desc("maximum # of functions to overwrite"),
             cl::Optional);

static cl::opt<bool>
EliminateUnreachable("eliminate-unreachable",
                     cl::desc("eliminate unreachable code"),
                     cl::Optional);

static cl::opt<BinaryFunction::SplittingType>
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
               cl::Optional);

static cl::opt<bool>
UpdateDebugSections("update-debug-sections",
                    cl::desc("update DWARF debug sections of the executable"),
                    cl::Optional);

static cl::opt<bool>
FixDebugInfoLargeFunctions("fix-debuginfo-large-functions",
                           cl::desc("do another pass if we encounter large "
                                    "functions, to correct their debug info."),
                           cl::Optional);

static cl::opt<BinaryFunction::LayoutType>
ReorderBlocks(
    "reorder-blocks",
    cl::desc("change layout of basic blocks in a function"),
    cl::init(BinaryFunction::LT_NONE),
    cl::values(clEnumValN(BinaryFunction::LT_NONE,
                          "none",
                          "do not reorder basic blocks"),
               clEnumValN(BinaryFunction::LT_REVERSE,
                          "reverse",
                          "layout blocks in reverse order"),
               clEnumValN(BinaryFunction::LT_OPTIMIZE,
                          "normal",
                          "perform optimal layout based on profile"),
               clEnumValN(BinaryFunction::LT_OPTIMIZE_BRANCH,
                          "branch-predictor",
                          "perform optimal layout prioritizing branch "
                            "predictions"),
               clEnumValN(BinaryFunction::LT_OPTIMIZE_CACHE,
                          "cache",
                          "perform optimal layout prioritizing I-cache "
                            "behavior"),
               clEnumValEnd));


static cl::opt<bool>
AlignBlocks("align-blocks",
            cl::desc("try to align BBs inserting nops"),
            cl::Optional);

static cl::opt<bool>
UseGnuStack("use-gnu-stack",
            cl::desc("use GNU_STACK program header for new segment"));

static cl::opt<bool>
DumpEHFrame("dump-eh-frame", cl::desc("dump parsed .eh_frame (debugging)"),
            cl::Hidden);

static cl::opt<bool>
PrintAll("print-all", cl::desc("print functions after each stage"),
         cl::Hidden);

static cl::opt<bool>
PrintCFG("print-cfg", cl::desc("print functions after CFG construction"),
         cl::Hidden);

static cl::opt<bool>
PrintUCE("print-uce",
         cl::desc("print functions after unreachable code elimination"),
         cl::Hidden);

static cl::opt<bool>
PrintDisasm("print-disasm", cl::desc("print function after disassembly"),
            cl::Hidden);

static cl::opt<bool>
PrintEHRanges("print-eh-ranges",
              cl::desc("print function with updated exception ranges"),
              cl::Hidden);

static cl::opt<bool>
PrintReordered("print-reordered",
               cl::desc("print functions after layout optimization"),
               cl::Hidden);

static cl::opt<bool>
KeepTmp("keep-tmp",
        cl::desc("preserve intermediate .o file"),
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
      if (Function.getName() == Name) {
        IsValid = true;
        break;
      }
    }
  }
  if (!IsValid)
    return false;

  if (!SkipFunctionNames.empty()) {
    for (auto &Name : SkipFunctionNames) {
      if (Function.getName() == Name) {
        IsValid = false;
        break;
      }
    }
  }

  return IsValid;
}

} // namespace opts


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

  DEBUG(dbgs() << "BOLT: allocating " << (IsCode ? "code" : "data")
               << " section : " << SectionName
               << " with size " << Size << ", alignment " << Alignment
               << " at 0x" << ret << "\n");

  SectionMapInfo[SectionName] = SectionInfo(reinterpret_cast<uint64_t>(ret),
                                            Size,
                                            Alignment,
                                            IsCode,
                                            IsReadOnly,
                                            0,
                                            0,
                                            SectionID);

  return ret;
}

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
                  /*IsReadOnly*/true,
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

/// Create BinaryContext for a given architecture \p ArchName and
/// triple \p TripleName.
static std::unique_ptr<BinaryContext> CreateBinaryContext(
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
    errs() << "BOLT: " << Error;
    return nullptr;
  }

  std::unique_ptr<const MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TripleName));
  if (!MRI) {
    errs() << "error: no register info for target " << TripleName << "\n";
    return nullptr;
  }

  // Set up disassembler.
  std::unique_ptr<const MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName));
  if (!AsmInfo) {
    errs() << "error: no assembly info for target " << TripleName << "\n";
    return nullptr;
  }

  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!STI) {
    errs() << "error: no subtarget info for target " << TripleName << "\n";
    return nullptr;
  }

  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII) {
    errs() << "error: no instruction info for target " << TripleName << "\n";
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
    errs() << "error: no disassembler for target " << TripleName << "\n";
    return nullptr;
  }

  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));
  if (!MIA) {
    errs() << "error: failed to create instruction analysis for target"
           << TripleName << "\n";
    return nullptr;
  }

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> InstructionPrinter(
      TheTarget->createMCInstPrinter(Triple(TripleName), AsmPrinterVariant,
                                     *AsmInfo, *MII, *MRI));
  if (!InstructionPrinter) {
    errs() << "error: no instruction printer for target " << TripleName
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

RewriteInstance::RewriteInstance(ELFObjectFileBase *File,
                                 const DataReader &DR)
    : InputFile(File),
      BC(CreateBinaryContext("x86-64", "x86_64-unknown-linux", DR,
         std::unique_ptr<DWARFContext>(new DWARFContextInMemory(*InputFile)))) {
}

RewriteInstance::~RewriteInstance() {}

void RewriteInstance::reset() {
  BinaryFunctions.clear();
  FileSymRefs.clear();
  auto &DR = BC->DR;
  BC = CreateBinaryContext("x86-64", "x86_64-unknown-linux", DR,
           std::unique_ptr<DWARFContext>(new DWARFContextInMemory(*InputFile)));
  CFIRdWrt.reset(nullptr);
  SectionMM.reset(nullptr);
  Out.reset(nullptr);
  EHFrame = nullptr;
  FailedAddresses.clear();
  RangesSectionsWriter.reset();
  TotalScore = 0;
}

void RewriteInstance::discoverStorage() {
  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(InputFile);
  if (!ELF64LEFile) {
    errs() << "BOLT-ERROR: only 64-bit LE ELF binaries are supported\n";
    exit(1);
  }
  auto Obj = ELF64LEFile->getELFFile();

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
    }
  }

  assert(NextAvailableAddress && NextAvailableOffset &&
         "no PT_LOAD pheader seen");

  errs() << "BOLT-INFO: first alloc address is 0x"
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

    errs() << "BOLT-INFO: creating new program header table at address 0x"
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
    errs() << "failed to create a binary context\n";
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
    outs() << "BOLT: starting pass (ignoring large functions)"
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
      errs() << "BOLT-WARNING: Function " << FunctionIt->second.getName()
             << " is larger than it's  orginal size: emitting again marking it "
             << "as not simple.\n";
      FunctionIt->second.setSimple(false);
    }

    readFunctionDebugInfo();
    runOptimizationPasses();
    emitFunctions();
  }

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
  std::string FileSymbolName;

  FileSymRefs.clear();
  BinaryFunctions.clear();
  BC->GlobalAddresses.clear();

  // For local symbols we want to keep track of associated FILE symbol for
  // disambiguation by name.
  for (const SymbolRef &Symbol : InputFile->symbols()) {
    // Keep undefined symbols for pretty printing?
    if (Symbol.getFlags() & SymbolRef::SF_Undefined)
      continue;

    ErrorOr<StringRef> Name = Symbol.getName();
    check_error(Name.getError(), "cannot get symbol name");

    if (Symbol.getType() == SymbolRef::ST_File) {
      // Could be used for local symbol disambiguation.
      FileSymbolName = *Name;
      continue;
    }

    ErrorOr<uint64_t> AddressOrErr = Symbol.getAddress();
    check_error(AddressOrErr.getError(), "cannot get symbol address");
    uint64_t Address = *AddressOrErr;
    if (Address == 0) {
      if (Symbol.getType() == SymbolRef::ST_Function)
        errs() << "BOLT-WARNING: function with 0 address seen\n";
      continue;
    }

    FileSymRefs[Address] = Symbol;

    // There's nothing horribly wrong with anonymous symbols, but let's
    // ignore them for now.
    if (Name->empty())
      continue;

    // Disambiguate all local symbols before adding to symbol table.
    // Since we don't know if we'll see a global with the same name,
    // always modify the local name.
    std::string UniqueName;
    if (Symbol.getFlags() & SymbolRef::SF_Global) {
      assert(BC->GlobalSymbols.find(*Name) == BC->GlobalSymbols.end() &&
             "global name not unique");
      UniqueName = *Name;
      /// It's possible we are seeing a globalized local. LLVM might treat it as
      /// local if it has a "private global" prefix, e.g. ".L". Thus we have to
      /// change the prefix to enforce global scope of the symbol.
      if (StringRef(UniqueName)
              .startswith(BC->AsmInfo->getPrivateGlobalPrefix()))
        UniqueName = "PG." + UniqueName;
    } else {
      unsigned LocalCount = 1;
      std::string LocalName = (*Name).str() + "/" + FileSymbolName + "/";

      if ((*Name).startswith(BC->AsmInfo->getPrivateGlobalPrefix())) {
        LocalName = "PG." + LocalName;
      }

      while (BC->GlobalSymbols.find(LocalName + std::to_string(LocalCount)) !=
             BC->GlobalSymbols.end()) {
        ++LocalCount;
      }
      UniqueName = LocalName + std::to_string(LocalCount);
    }

    // Add the name to global symbols map.
    BC->GlobalSymbols[UniqueName] = Address;

    // Add to the reverse map. There could multiple names at the same address.
    BC->GlobalAddresses.emplace(std::make_pair(Address, UniqueName));

    // Only consider ST_Function symbols for functions. Although this
    // assumption  could be broken by assembly functions for which the type
    // could be wrong, we skip such entries till the support for
    // assembly is implemented.
    if (Symbol.getType() != SymbolRef::ST_Function)
      continue;

    // TODO: populate address map with PLT entries for better readability.

    // Ignore function with 0 size for now (possibly coming from assembly).
    auto SymbolSize = ELFSymbolRef(Symbol).getSize();
    if (SymbolSize == 0)
      continue;

    ErrorOr<section_iterator> SectionOrErr = Symbol.getSection();
    check_error(SectionOrErr.getError(), "cannot get symbol section");
    section_iterator Section = *SectionOrErr;
    if (Section == InputFile->section_end()) {
      // Could be an absolute symbol. Could record for pretty printing.
      continue;
    }

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
            errs() << "BOLT-WARNING: function " << UniqueName
                   << " is in conflict with FDE ["
                   << Twine::utohexstr(PrevStart) << ", "
                   << Twine::utohexstr(PrevStart + PrevLength)
                   << "). Skipping.\n";
            IsSimple = false;
          }
        }
      } else if (FDE.getAddressRange() != SymbolSize) {
        // Function addresses match but sizes differ.
        errs() << "BOLT-WARNING: sizes differ for function " << UniqueName
               << ". FDE : " << FDE.getAddressRange()
               << "; symbol table : " << SymbolSize << ". Skipping.\n";

        // Create maximum size non-simple function.
        IsSimple = false;
        SymbolSize = std::max(SymbolSize, FDE.getAddressRange());
      }
    }

    // Create the function and add to the map.
    BinaryFunctions.emplace(
        Address,
        BinaryFunction(UniqueName, Symbol, *Section, Address,
                       SymbolSize, *BC, IsSimple)
    );
  }
}

void RewriteInstance::readSpecialSections() {
  // Process special sections.
  StringRef FrameHdrContents;
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
    } else if (SectionName == ".eh_frame_hdr") {
      FrameHdrAddress = Section.getAddress();
      FrameHdrContents = SectionContents;
      FrameHdrAlign = Section.getAlignment();
    } else if (SectionName == ".debug_line") {
      DebugLineSize = Section.getSize();
    } else if (SectionName == ".debug_ranges") {
      DebugRangesSize = Section.getSize();
    } else if (SectionName == ".debug_loc") {
      DebugLocSize = Section.getSize();
    }
  }

  FrameHdrCopy =
      std::vector<char>(FrameHdrContents.begin(), FrameHdrContents.end());
  // Process debug sections.
  EHFrame = BC->DwCtx->getEHFrame();
  if (opts::DumpEHFrame) {
    EHFrame->dump(outs());
  }
  CFIRdWrt.reset(new CFIReaderWriter(*EHFrame, FrameHdrAddress, FrameHdrCopy));
  if (!EHFrame->ParseError.empty()) {
    errs() << "BOLT-ERROR: EHFrame reader failed with message \""
           << EHFrame->ParseError << "\"\n";
    exit(1);
  }
}

void RewriteInstance::readDebugInfo() {
  if (!opts::UpdateDebugSections)
    return;

  BC->preprocessDebugInfo();
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

    if (!opts::shouldProcess(Function)) {
      DEBUG(dbgs() << "BOLT: skipping processing function "
                   << Function.getName() << " per user request.\n");
      continue;
    }

    SectionRef Section = Function.getSection();
    assert(Section.getAddress() <= Function.getAddress() &&
           Section.getAddress() + Section.getSize()
             >= Function.getAddress() + Function.getSize() &&
          "wrong section for function");
    if (!Section.isText() || Section.isVirtual() || !Section.getSize()) {
      // When could it happen?
      errs() << "BOLT: corresponding section is non-executable or empty "
             << "for function " << Function.getName();
      continue;
    }

    // Set the proper maximum size value after the whole symbol table
    // has been processed.
    auto SymRefI = FileSymRefs.upper_bound(Function.getAddress());
    if (SymRefI != FileSymRefs.end()) {
      uint64_t MaxSize;
      auto SectionIter = *SymRefI->second.getSection();
      if (SectionIter != InputFile->section_end() &&
          *SectionIter == Function.getSection()) {
        MaxSize = SymRefI->first - Function.getAddress();
      } else {
        // Function runs till the end of the containing section assuming
        // the section does not run over the next symbol.
        uint64_t SectionEnd = Function.getSection().getAddress() +
                              Function.getSection().getSize();
        if (SectionEnd > SymRefI->first) {
          errs() << "BOLT-WARNING: symbol after " << Function.getName()
                 << " should not be in the same section.\n";
          MaxSize = 0;
        } else {
          MaxSize = SectionEnd - Function.getAddress();
        }
      }

      if (MaxSize < Function.getSize()) {
        errs() << "BOLT-WARNING: symbol seen in the middle of the function "
               << Function.getName() << ". Skipping.\n";
        Function.setSimple(false);
        continue;
      }
      Function.setMaxSize(MaxSize);
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

    if (!Function.disassemble(FunctionData, opts::UpdateDebugSections))
      continue;

    if (opts::PrintAll || opts::PrintDisasm)
      Function.print(errs(), "after disassembly", true);

    if (!Function.isSimple())
      continue;

    // Fill in CFI information for this function
    if (EHFrame->ParseError.empty()) {
      if (!CFIRdWrt->fillCFIInfoFor(Function)) {
        errs() << "BOLT-WARNING: unable to fill CFI for function "
               << Function.getName() << '\n';
        Function.setSimple(false);
        continue;
      }
    }

    // Parse LSDA.
    if (Function.getLSDAAddress() != 0)
      Function.parseLSDA(LSDAData, LSDAAddress);

    if (!Function.buildCFG())
      continue;

    if (opts::PrintAll || opts::PrintCFG)
      Function.print(errs(), "after building cfg", true);

    TotalScore += Function.getFunctionScore();

  } // Iterate over all functions

  // Mark all functions with internal addresses serving as interprocedural
  // branch targets as not simple --  pretty rare but can happen in code
  // written in assembly.
  // TODO: #9301815
  for (auto Addr : BC->InterproceduralBranchTargets) {
    // Check if this address is internal to some function we are reordering
    auto I = BinaryFunctions.upper_bound(Addr);
    if (I == BinaryFunctions.begin())
      continue;
    BinaryFunction &Func = (--I)->second;
    uint64_t Offset = Addr - I->first;
    if (Offset == 0 || Offset >= Func.getSize())
      continue;
    errs() << "BOLT-WARNING: Function " << Func.getName()
           << " has internal BBs that are target of a branch located in "
              "another function. We will not process this function.\n";
    Func.setSimple(false);
  }

  uint64_t NumSimpleFunctions{0};
  std::vector<BinaryFunction *> ProfiledFunctions;
  for (auto &BFI : BinaryFunctions) {
    if (!BFI.second.isSimple())
      continue;
    ++NumSimpleFunctions;
    if (BFI.second.getExecutionCount() != BinaryFunction::COUNT_NO_PROFILE)
      ProfiledFunctions.push_back(&BFI.second);
  }

  errs() << "BOLT-INFO: " << ProfiledFunctions.size() << " functions out of "
         << NumSimpleFunctions
         << " simple functions ("
         << format("%.1f",
                   ProfiledFunctions.size() /
                   (float) NumSimpleFunctions * 100.0)
         << "%) have non-empty execution profile.\n";

  if (ProfiledFunctions.size() > 10) {
    errs() << "BOLT-INFO: top called functions are:\n";
    std::sort(ProfiledFunctions.begin(), ProfiledFunctions.end(),
              [](BinaryFunction *A, BinaryFunction *B) {
                return B->getExecutionCount() < A->getExecutionCount();
              }
    );
    auto SFI = ProfiledFunctions.begin();
    for (int i = 0; i < 100 && SFI != ProfiledFunctions.end(); ++SFI, ++i) {
      errs() << "  " << (*SFI)->getName() << " : "
             << (*SFI)->getExecutionCount() << '\n';
    }
  }
}

void RewriteInstance::runOptimizationPasses() {
  // Run optimization passes.
  //
  // FIXME: use real optimization passes.
  bool NagUser = true;
  for (auto &BFI : BinaryFunctions) {
    auto &Function = BFI.second;

    if (!opts::shouldProcess(Function))
      continue;

    if (!Function.isSimple())
      continue;

    // Detect and eliminate unreachable basic blocks. We could have those
    // filled with nops and they are used for alignment.
    //
    // FIXME: this wouldn't work with C++ exceptions until we implement
    //        support for those as there will be "invisible" edges
    //        in the graph.
    if (opts::EliminateUnreachable && Function.layout_size() > 0) {
      if (NagUser) {
        outs()
            << "BOLT-WARNING: Using -eliminate-unreachable is experimental and "
               "unsafe for exceptions\n";
        NagUser = false;
      }

      std::stack<BinaryBasicBlock*> Stack;
      std::map<BinaryBasicBlock *, bool> Reachable;
      BinaryBasicBlock *Entry = *Function.layout_begin();
      Stack.push(Entry);
      Reachable[Entry] = true;
      // Determine reachable BBs from the entry point
      while (!Stack.empty()) {
        auto BB = Stack.top();
        Stack.pop();
        for (auto Succ : BB->successors()) {
          if (Reachable[Succ])
            continue;
          Reachable[Succ] = true;
          Stack.push(Succ);
        }
      }

      auto Count = Function.eraseDeadBBs(Reachable);
      if (Count) {
        DEBUG(dbgs() << "BOLT: Removed " << Count
                     << " dead basic block(s) in function "
                     << Function.getName() << '\n');
      }

      if (opts::PrintAll || opts::PrintUCE)
        Function.print(errs(), "after unreachable code elimination", true);
    }

    if (opts::ReorderBlocks != BinaryFunction::LT_NONE) {
      bool ShouldSplit =
        (opts::SplitFunctions == BinaryFunction::ST_ALL) ||
        (opts::SplitFunctions == BinaryFunction::ST_EH &&
         Function.hasEHRanges()) ||
        (LargeFunctions.find(BFI.first) != LargeFunctions.end());
      BFI.second.modifyLayout(opts::ReorderBlocks, ShouldSplit);
      if (opts::PrintAll || opts::PrintReordered)
        Function.print(errs(), "after reordering blocks", true);
    }

    // Post-processing passes.
    BinaryFunctionPassManager::runAllPasses(*BC, BinaryFunctions);

    // Fix the CFI state.
    if (!Function.fixCFIState()) {
      errs() << "BOLT-WARNING: unable to fix CFI state for function "
             << Function.getName() << ". Skipping.\n";
      Function.setSimple(false);
      continue;
    }

    // Update exception handling information.
    Function.updateEHRanges();
    if (opts::PrintAll || opts::PrintEHRanges)
      Function.print(errs(), "after updating EH ranges", true);
  }
}

namespace {

// Helper function to emit the contents of a function via a MCStreamer object.
void emitFunction(MCStreamer &Streamer, BinaryFunction &Function,
                  BinaryContext &BC, bool EmitColdPart) {
  // Define a helper to decode and emit CFI instructions at a given point in a
  // BB
  auto emitCFIInstr = [&Streamer](MCCFIInstruction &CFIInstr) {
    switch (CFIInstr.getOperation()) {
    default:
      llvm_unreachable("Unexpected instruction");
    case MCCFIInstruction::OpDefCfaOffset:
      Streamer.EmitCFIDefCfaOffset(CFIInstr.getOffset());
      break;
    case MCCFIInstruction::OpAdjustCfaOffset:
      Streamer.EmitCFIAdjustCfaOffset(CFIInstr.getOffset());
      break;
    case MCCFIInstruction::OpDefCfa:
      Streamer.EmitCFIDefCfa(CFIInstr.getRegister(), CFIInstr.getOffset());
      break;
    case MCCFIInstruction::OpDefCfaRegister:
      Streamer.EmitCFIDefCfaRegister(CFIInstr.getRegister());
      break;
    case MCCFIInstruction::OpOffset:
      Streamer.EmitCFIOffset(CFIInstr.getRegister(), CFIInstr.getOffset());
      break;
    case MCCFIInstruction::OpRegister:
      Streamer.EmitCFIRegister(CFIInstr.getRegister(),
                                CFIInstr.getRegister2());
      break;
    case MCCFIInstruction::OpRelOffset:
      Streamer.EmitCFIRelOffset(CFIInstr.getRegister(), CFIInstr.getOffset());
      break;
    case MCCFIInstruction::OpUndefined:
      Streamer.EmitCFIUndefined(CFIInstr.getRegister());
      break;
    case MCCFIInstruction::OpRememberState:
      Streamer.EmitCFIRememberState();
      break;
    case MCCFIInstruction::OpRestoreState:
      Streamer.EmitCFIRestoreState();
      break;
    case MCCFIInstruction::OpRestore:
      Streamer.EmitCFIRestore(CFIInstr.getRegister());
      break;
    case MCCFIInstruction::OpSameValue:
      Streamer.EmitCFISameValue(CFIInstr.getRegister());
      break;
    case MCCFIInstruction::OpGnuArgsSize:
      Streamer.EmitCFIGnuArgsSize(CFIInstr.getOffset());
      break;
    }
  };

  // No need for human readability?
  // FIXME: what difference does it make in reality?
  // Ctx.setUseNamesOnTempLabels(false);

  // Emit function start

  // Each fuction is emmitted into its own section.
  MCSectionELF *FunctionSection =
      EmitColdPart
          ? BC.Ctx->getELFSection(
                Function.getCodeSectionName().str().append(".cold"),
                ELF::SHT_PROGBITS, ELF::SHF_EXECINSTR | ELF::SHF_ALLOC)
          : BC.Ctx->getELFSection(Function.getCodeSectionName(),
                                  ELF::SHT_PROGBITS,
                                  ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);

  MCSection *Section = FunctionSection;

  Section->setHasInstructions(true);
  BC.Ctx->addGenDwarfSection(Section);

  Streamer.SwitchSection(Section);

  Streamer.EmitCodeAlignment(Function.getAlignment());

  if (!EmitColdPart) {
    MCSymbol *FunctionSymbol = BC.Ctx->getOrCreateSymbol(Function.getName());
    Streamer.EmitSymbolAttribute(FunctionSymbol, MCSA_ELF_TypeFunction);
    Streamer.EmitLabel(FunctionSymbol);
    Function.setOutputSymbol(FunctionSymbol);
  } else {
    MCSymbol *FunctionSymbol =
      BC.Ctx->getOrCreateSymbol(Twine(Function.getName()).concat(".cold"));
    Streamer.EmitSymbolAttribute(FunctionSymbol, MCSA_ELF_TypeFunction);
    Streamer.EmitLabel(FunctionSymbol);
    Function.cold().setOutputSymbol(FunctionSymbol);
  }

  // Emit CFI start
  if (Function.hasCFI()) {
    Streamer.EmitCFIStartProc(/*IsSimple=*/false);
    if (Function.getPersonalityFunction() != nullptr) {
      Streamer.EmitCFIPersonality(Function.getPersonalityFunction(),
                                  Function.getPersonalityEncoding());
    }
    if (!EmitColdPart && Function.getLSDASymbol()) {
      Streamer.EmitCFILsda(Function.getLSDASymbol(),
                           BC.MOFI->getLSDAEncoding());
    } else {
      Streamer.EmitCFILsda(0, dwarf::DW_EH_PE_omit);
    }
    // Emit CFI instructions relative to the CIE
    for (auto &CFIInstr : Function.cie()) {
      // Ignore these CIE CFI insns because LLVM will already emit this.
      switch (CFIInstr.getOperation()) {
      default:
        break;
      case MCCFIInstruction::OpDefCfa:
        if (CFIInstr.getRegister() == 7 && CFIInstr.getOffset() == 8)
          continue;
        break;
      case MCCFIInstruction::OpOffset:
        if (CFIInstr.getRegister() == 16 && CFIInstr.getOffset() == -8)
          continue;
        break;
      }
      emitCFIInstr(CFIInstr);
    }
  }

  assert(!Function.begin()->isCold() &&
         "first basic block should never be cold");

  // Emit UD2 at the beginning if requested by user.
  if (!opts::BreakFunctionNames.empty()) {
    for (auto &Name : opts::BreakFunctionNames) {
      if (Function.getName() == Name) {
        Streamer.EmitIntValue(0x0B0F, 2); // UD2: 0F 0B
        break;
      }
    }
  }

  // Emit code.
  int64_t CurrentGnuArgsSize = 0;
  for (auto BB : Function.layout()) {
    if (EmitColdPart != BB->isCold())
      continue;
    if (opts::AlignBlocks && BB->getAlignment() > 1)
      Streamer.EmitCodeAlignment(BB->getAlignment());
    Streamer.EmitLabel(BB->getLabel());
    // Remember last .debug_line entry emitted so that we don't repeat them in
    // subsequent instructions, as gdb can figure it out by looking at the
    // previous instruction with available line number info.
    SMLoc LastLocSeen;

    for (const auto &Instr : *BB) {
      // Handle pseudo instructions.
      if (BC.MIA->isEHLabel(Instr)) {
        assert(Instr.getNumOperands() == 1 && Instr.getOperand(0).isExpr() &&
               "bad EH_LABEL instruction");
        auto Label = &(cast<MCSymbolRefExpr>(Instr.getOperand(0).getExpr())
                           ->getSymbol());
        Streamer.EmitLabel(const_cast<MCSymbol *>(Label));
        continue;
      }
      if (BC.MIA->isCFI(Instr)) {
        emitCFIInstr(*Function.getCFIFor(Instr));
        continue;
      }
      if (opts::UpdateDebugSections) {
        auto RowReference = DebugLineTableRowRef::fromSMLoc(Instr.getLoc());
        if (RowReference != DebugLineTableRowRef::NULL_ROW &&
            Instr.getLoc().getPointer() != LastLocSeen.getPointer()) {
          auto CompileUnit =
              BC.OffsetToDwarfCU[RowReference.DwCompileUnitIndex];
          assert(CompileUnit &&
                 "Invalid CU offset set in instruction debug info.");

          auto OriginalLineTable =
            BC.DwCtx->getLineTableForUnit(
                CompileUnit);
          const auto &OriginalRow =
              OriginalLineTable->Rows[RowReference.RowIndex - 1];

          BC.Ctx->setCurrentDwarfLoc(
              OriginalRow.File,
              OriginalRow.Line,
              OriginalRow.Column,
              (DWARF2_FLAG_IS_STMT * OriginalRow.IsStmt) |
              (DWARF2_FLAG_BASIC_BLOCK * OriginalRow.BasicBlock) |
              (DWARF2_FLAG_PROLOGUE_END * OriginalRow.PrologueEnd) |
              (DWARF2_FLAG_EPILOGUE_BEGIN * OriginalRow.EpilogueBegin),
              OriginalRow.Isa,
              OriginalRow.Discriminator);
          BC.Ctx->setDwarfCompileUnitID(CompileUnit->getOffset());
          LastLocSeen = Instr.getLoc();
        }
      }

      // Emit GNU_args_size CFIs as necessary.
      if (Function.usesGnuArgsSize() && BC.MIA->isInvoke(Instr)) {
        auto NewGnuArgsSize = BC.MIA->getGnuArgsSize(Instr);
        if (NewGnuArgsSize >= 0 && NewGnuArgsSize != CurrentGnuArgsSize) {
          CurrentGnuArgsSize = NewGnuArgsSize;
          Streamer.EmitCFIGnuArgsSize(CurrentGnuArgsSize);
        }
      }

      Streamer.EmitInstruction(Instr, *BC.STI);
    }

    MCSymbol *BBEndLabel = BC.Ctx->createTempSymbol();
    BB->setEndLabel(BBEndLabel);
    Streamer.EmitLabel(BBEndLabel);
  }

  // Emit CFI end
  if (Function.hasCFI())
    Streamer.EmitCFIEndProc();

  if (!EmitColdPart && Function.getFunctionEndLabel())
    Streamer.EmitLabel(Function.getFunctionEndLabel());

  // Emit LSDA before anything else?
  if (!EmitColdPart)
    Function.emitLSDA(&Streamer);

  // TODO: is there any use in emiting end of function?
  //       Perhaps once we have a support for C++ exceptions.
  // auto FunctionEndLabel = Ctx.createTempSymbol("func_end");
  // Streamer.EmitLabel(FunctionEndLabel);
  // Streamer.emitELFSize(FunctionSymbol, MCExpr());
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

  // Output functions one by one.
  for (auto &BFI : BinaryFunctions) {
    auto &Function = BFI.second;

    if (!Function.isSimple())
      continue;

    if (!opts::shouldProcess(Function))
      continue;

    DEBUG(dbgs() << "BOLT: generating code for function \""
                 << Function.getName() << "\" : "
                 << Function.getFunctionNumber() << '\n');

    emitFunction(*Streamer, Function, *BC.get(), /*EmitColdPart=*/false);

    if (Function.isSplit())
      emitFunction(*Streamer, Function, *BC.get(), /*EmitColdPart=*/true);
  }

  updateDebugLineInfoForNonSimpleFunctions();

  Streamer->Finish();

  //////////////////////////////////////////////////////////////////////////////
  // Assign addresses to new functions/sections.
  //////////////////////////////////////////////////////////////////////////////

  auto EFMM = new ExecutableFileMemoryManager();
  SectionMM.reset(EFMM);

  if (opts::UpdateDebugSections) {
    // Compute offsets of tables in .debug_line for each compile unit.
    computeLineTableOffsets();
  }

  // Get output object as ObjectFile.
  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(BOS->str(), "in-memory object file", false);
  ErrorOr<std::unique_ptr<object::ObjectFile>> ObjOrErr =
    object::ObjectFile::createObjectFile(ObjectMemBuffer->getMemBufferRef());
  check_error(ObjOrErr.getError(), "error creating in-memory object");

  // Run ObjectLinkingLayer() with custom memory manager and symbol resolver.
  orc::ObjectLinkingLayer<> OLT;

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
  auto ObjectsHandle = OLT.addObjectSet(
        singletonSet(std::move(ObjOrErr.get())),
        SectionMM.get(),
        std::move(Resolver),
        /* ProcessAllSections = */true);

  // FIXME: use notifyObjectLoaded() to remap sections.

  // Map every function/section current address in memory to that in
  // the output binary.
  uint64_t NewTextSectionStartAddress = NextAvailableAddress;
  for (auto &BFI : BinaryFunctions) {
    auto &Function = BFI.second;
    if (!Function.isSimple())
      continue;

    auto SMII = EFMM->SectionMapInfo.find(Function.getCodeSectionName());
    if (SMII != EFMM->SectionMapInfo.end()) {
      DEBUG(dbgs() << "BOLT: mapping 0x"
                   << Twine::utohexstr(SMII->second.AllocAddress)
                   << " to 0x" << Twine::utohexstr(Function.getAddress())
                   << '\n');
      OLT.mapSectionAddress(ObjectsHandle,
                            SMII->second.SectionID,
                            Function.getAddress());
      Function.setImageAddress(SMII->second.AllocAddress);
      Function.setImageSize(SMII->second.Size);
    } else {
      errs() << "BOLT: cannot remap function " << Function.getName() << "\n";
      FailedAddresses.emplace_back(Function.getAddress());
    }

    if (!Function.isSplit())
      continue;

    SMII = EFMM->SectionMapInfo.find(
        Function.getCodeSectionName().str().append(".cold"));
    if (SMII != EFMM->SectionMapInfo.end()) {
      // Cold fragments are aligned at 16 bytes.
      NextAvailableAddress = RoundUpToAlignment(NextAvailableAddress, 16);
      DEBUG(dbgs() << "BOLT: mapping 0x"
                   << Twine::utohexstr(SMII->second.AllocAddress)
                   << " to 0x" << Twine::utohexstr(NextAvailableAddress)
                   << " with size " << Twine::utohexstr(SMII->second.Size)
                   << '\n');
      OLT.mapSectionAddress(ObjectsHandle,
                            SMII->second.SectionID,
                            NextAvailableAddress);
      Function.cold().setAddress(NextAvailableAddress);
      Function.cold().setImageAddress(SMII->second.AllocAddress);
      Function.cold().setImageSize(SMII->second.Size);
      Function.cold().setFileOffset(getFileOffsetFor(NextAvailableAddress));

      NextAvailableAddress += SMII->second.Size;
    } else {
      errs() << "BOLT: cannot remap function " << Function.getName() << "\n";
      FailedAddresses.emplace_back(Function.getAddress());
    }
  }

  // Add the new text section aggregating all existing code sections.
  auto NewTextSectionSize = NextAvailableAddress - NewTextSectionStartAddress;
  if (NewTextSectionSize) {
    SectionMM->SectionMapInfo[".bolt.text"] =
        SectionInfo(0,
                    NewTextSectionSize,
                    16,
                    true /*IsCode*/,
                    true /*IsReadOnly*/,
                    NewTextSectionStartAddress,
                    getFileOffsetFor(NewTextSectionStartAddress));
  }

  // Map special sections to their addresses in the output image.
  //
  // TODO: perhaps we should process all the allocated sections here?
  std::vector<std::string> Sections = { ".eh_frame", ".gcc_except_table" };
  for (auto &SectionName : Sections) {
    auto SMII = EFMM->SectionMapInfo.find(SectionName);
    if (SMII != EFMM->SectionMapInfo.end()) {
      SectionInfo &SI = SMII->second;
      NextAvailableAddress = RoundUpToAlignment(NextAvailableAddress,
                                                SI.Alignment);
      DEBUG(dbgs() << "BOLT: mapping 0x"
                   << Twine::utohexstr(SI.AllocAddress)
                   << " to 0x" << Twine::utohexstr(NextAvailableAddress)
                   << '\n');

      OLT.mapSectionAddress(ObjectsHandle,
                            SI.SectionID,
                            NextAvailableAddress);
      SI.FileAddress = NextAvailableAddress;
      SI.FileOffset = getFileOffsetFor(NextAvailableAddress);

      NextAvailableAddress += SI.Size;
    } else {
      errs() << "BOLT: cannot remap " << SectionName << '\n';
    }
  }

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

  if (opts::KeepTmp)
    TempOut->keep();
}

bool RewriteInstance::checkLargeFunctions() {
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

void RewriteInstance::updateFunctionRanges() {
  auto addDebugArangesEntry = [&](uint64_t OriginalFunctionAddress,
                                  uint64_t RangeBegin,
                                  uint64_t RangeSize) {
    if (auto DebugAranges = BC->DwCtx->getDebugAranges()) {
      uint32_t CUOffset = DebugAranges->findAddress(OriginalFunctionAddress);
      if (CUOffset != -1U)
        RangesSectionsWriter.AddRange(CUOffset, RangeBegin, RangeSize);
    }
  };

  for (auto &BFI : BinaryFunctions) {
    auto &Function = BFI.second;
    // Use either new (image) or original size for the function range.
    auto Size = Function.isSimple() ? Function.getImageSize()
                                    : Function.getSize();
    addDebugArangesEntry(Function.getAddress(),
                         Function.getAddress(),
                         Size);
    RangesSectionsWriter.AddRange(&Function, Function.getAddress(), Size);
    if (Function.isSimple() && Function.cold().getImageSize()) {
      addDebugArangesEntry(Function.getAddress(),
                           Function.cold().getAddress(),
                           Function.cold().getImageSize());
      RangesSectionsWriter.AddRange(&Function,
                                    Function.cold().getAddress(),
                                    Function.cold().getImageSize());
    }
  }
}

void RewriteInstance::generateDebugRanges() {
  using RangeType = enum { RANGES, ARANGES };
  for (int IntRT = RANGES; IntRT <= ARANGES; ++IntRT) {
    RangeType RT = static_cast<RangeType>(IntRT);
    const char *SectionName = (RT == RANGES) ? ".debug_ranges"
                                             : ".debug_aranges";
    SmallVector<char, 16> RangesBuffer;
    raw_svector_ostream OS(RangesBuffer);

    auto MAB = BC->TheTarget->createMCAsmBackend(*BC->MRI, BC->TripleName, "");
    auto Writer = MAB->createObjectWriter(OS);

    if (RT == RANGES) {
      RangesSectionsWriter.WriteRangesSection(Writer);
    } else {
      RangesSectionsWriter.WriteArangesSection(Writer);
    }
    const auto &DebugRangesContents = OS.str();

    // Free'd by SectionMM.
    uint8_t *SectionData = new uint8_t[DebugRangesContents.size()];
    memcpy(SectionData, DebugRangesContents.data(), DebugRangesContents.size());

    SectionMM->NoteSectionInfo[SectionName] = SectionInfo(
        reinterpret_cast<uint64_t>(SectionData),
        DebugRangesContents.size(),
        /*Alignment=*/0,
        /*IsCode=*/false,
        /*IsReadOnly=*/true);
  }
}

void RewriteInstance::updateLocationLists() {
  // Write new contents to .debug_loc.
  SmallVector<char, 16> DebugLocBuffer;
  raw_svector_ostream OS(DebugLocBuffer);

  auto MAB = BC->TheTarget->createMCAsmBackend(*BC->MRI, BC->TripleName, "");
  auto Writer = MAB->createObjectWriter(OS);

  DebugLocWriter LocationListsWriter;

  for (const auto &Loc : BC->LocationLists) {
    LocationListsWriter.write(Loc, Writer);
  }

  const auto &DebugLocContents = OS.str();

  // Free'd by SectionMM.
  uint8_t *SectionData = new uint8_t[DebugLocContents.size()];
  memcpy(SectionData, DebugLocContents.data(), DebugLocContents.size());

  SectionMM->NoteSectionInfo[".debug_loc"] = SectionInfo(
      reinterpret_cast<uint64_t>(SectionData),
      DebugLocContents.size(),
      /*Alignment=*/0,
      /*IsCode=*/false,
      /*IsReadOnly=*/true);

  // For each CU, update pointers into .debug_loc.
  for (const auto &CU : BC->DwCtx->compile_units()) {
    updateLocationListPointers(
        CU.get(),
        CU->getUnitDIE(false),
        LocationListsWriter.getUpdatedLocationListOffsets());
  }
}

void RewriteInstance::updateLocationListPointers(
    const DWARFUnit *Unit,
    const DWARFDebugInfoEntryMinimal *DIE,
    const std::map<uint32_t, uint32_t> &UpdatedOffsets) {
  // Stop if we're in a non-simple function, which will not be rewritten.
  auto Tag = DIE->getTag();
  if (Tag == dwarf::DW_TAG_subprogram) {
    uint64_t LowPC = -1ULL, HighPC = -1ULL;
    DIE->getLowAndHighPC(Unit, LowPC, HighPC);
    if (LowPC != -1ULL) {
      auto It = BinaryFunctions.find(LowPC);
      if (It != BinaryFunctions.end() && !It->second.isSimple())
        return;
    }
  }
  // If the DIE has a DW_AT_location attribute with a section offset, update it.
  DWARFFormValue Value;
  uint32_t AttrOffset;
  if (DIE->getAttributeValue(Unit, dwarf::DW_AT_location, Value, &AttrOffset) &&
      (Value.isFormClass(DWARFFormValue::FC_Constant) ||
       Value.isFormClass(DWARFFormValue::FC_SectionOffset))) {
    uint64_t DebugLocOffset = -1ULL;
    if (Value.isFormClass(DWARFFormValue::FC_SectionOffset)) {
      DebugLocOffset = Value.getAsSectionOffset().getValue();
    } else if (Value.isFormClass(DWARFFormValue::FC_Constant)) {  // DWARF 3
      DebugLocOffset = Value.getAsUnsignedConstant().getValue();
    }

    auto It = UpdatedOffsets.find(DebugLocOffset);
    if (It != UpdatedOffsets.end()) {
      auto DebugInfoPatcher =
          static_cast<SimpleBinaryPatcher *>(
              SectionPatchers[".debug_info"].get());
      DebugInfoPatcher->addLE32Patch(AttrOffset, It->second + DebugLocSize);
    }
  }

  // Recursively visit children.
  for (auto Child = DIE->getFirstChild(); Child; Child = Child->getSibling()) {
    updateLocationListPointers(Unit, Child, UpdatedOffsets);
  }
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
      auto SMII = SectionMM->SectionMapInfo.find(".eh_frame_hdr");
      assert(SMII != SectionMM->SectionMapInfo.end() &&
             ".eh_frame_hdr could not be found for PT_GNU_EH_FRAME");
      auto &EHFrameHdrSecInfo = SMII->second;
      NewPhdr.p_offset = EHFrameHdrSecInfo.FileOffset;
      NewPhdr.p_vaddr = EHFrameHdrSecInfo.FileAddress;
      NewPhdr.p_paddr = EHFrameHdrSecInfo.FileAddress;
      NewPhdr.p_filesz = EHFrameHdrSecInfo.Size;
      NewPhdr.p_memsz = EHFrameHdrSecInfo.Size;
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

void RewriteInstance::rewriteNoteSections() {
  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(InputFile);
  if (!ELF64LEFile) {
    errs() << "BOLT-ERROR: only 64-bit LE ELF binaries are supported\n";
    exit(1);
  }
  auto Obj = ELF64LEFile->getELFFile();
  auto &OS = Out->os();

  uint64_t NextAvailableOffset = getFileOffsetFor(NextAvailableAddress);
  assert(NextAvailableOffset >= FirstNonAllocatableOffset &&
         "next available offset calculation failure");
  OS.seek(NextAvailableOffset);

  // Copy over non-allocatable section contents and update file offsets.
  for (auto &Section : Obj->sections()) {
    if (Section.sh_type == ELF::SHT_NULL)
      continue;
    if (Section.sh_flags & ELF::SHF_ALLOC)
      continue;

    // Insert padding as needed.
    if (Section.sh_addralign > 1) {
      auto Padding = OffsetToAlignment(NextAvailableOffset,
                                       Section.sh_addralign);
      const unsigned char ZeroByte{0};
      for (unsigned I = 0; I < Padding; ++I)
        OS.write(ZeroByte);

      NextAvailableOffset += Padding;

      assert(Section.sh_size % Section.sh_addralign == 0 &&
             "section size does not match section alignment");
    }

    ErrorOr<StringRef> SectionName = Obj->getSectionName(&Section);
    check_error(SectionName.getError(), "cannot get section name");

    // Copy over section contents unless it's .debug_aranges, which shall be
    // overwritten if -update-debug-sections is passed.
    uint64_t Size = 0;

    if (*SectionName != ".debug_aranges" || !opts::UpdateDebugSections) {
      Size = Section.sh_size;
      std::string Data = InputFile->getData().substr(Section.sh_offset, Size);
      auto SectionPatchersIt = SectionPatchers.find(*SectionName);
      if (SectionPatchersIt != SectionPatchers.end()) {
        (*SectionPatchersIt->second).patchBinary(Data);
      }
      OS << Data;
    }

    // Address of extension to the section.
    uint64_t Address{0};

    // Perform section post-processing.

    auto SII = SectionMM->NoteSectionInfo.find(*SectionName);
    if (SII != SectionMM->NoteSectionInfo.end()) {
      auto &SI = SII->second;
      assert(SI.Alignment <= Section.sh_addralign &&
             "alignment exceeds value in file");

      // Write section extension.
      Address = SI.AllocAddress;
      if (Address) {
        DEBUG(dbgs() << "BOLT: appending contents to section "
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
    SectionMM->NoteSectionInfo[*SectionName] =
      SectionInfo(Address,
                  Size,
                  Section.sh_addralign,
                  /*IsCode=*/false,
                  /*IsReadOnly=*/false,
                  /*FileAddress=*/0,
                  NextAvailableOffset);

    NextAvailableOffset += Size;
  }
}

// Rewrite section header table inserting new entries as needed. The sections
// header table size itself may affect the offsets of other sections,
// so we are placing it at the end of the binary.
//
// As we rewrite entries we need to track how many sections were inserted
// as it changes the sh_link value.
//
// The following are assumptoins about file modifications:
//    * There are no modifications done to existing allocatable sections.
//    * All new allocatable sections are written emmediately after existing
//      allocatable sections.
//    * There could be modifications done to non-allocatable sections, e.g.
//      size could be increased.
//    * New non-allocatable sections are added to the end of the file.
void RewriteInstance::patchELFSectionHeaderTable() {
  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(InputFile);
  if (!ELF64LEFile) {
    errs() << "BOLT-ERROR: only 64-bit LE ELF binaries are supported\n";
    exit(1);
  }
  auto Obj = ELF64LEFile->getELFFile();
  using Elf_Shdr = std::remove_pointer<decltype(Obj)>::type::Elf_Shdr;

  auto &OS = Out->os();

  auto SHTOffset = OS.tell();

  // Copy over entries for original allocatable sections with minor
  // modifications (e.g. name).
  for (auto &Section : Obj->sections()) {
    // Always ignore this section.
    if (Section.sh_type == ELF::SHT_NULL) {
      OS.write(reinterpret_cast<const char *>(&Section), sizeof(Section));
      continue;
    }

    // Break at first non-allocatable section.
    if (!(Section.sh_flags & ELF::SHF_ALLOC))
      break;

    ErrorOr<StringRef> SectionName = Obj->getSectionName(&Section);
    check_error(SectionName.getError(), "cannot get section name");

    auto NewSection = Section;
    if (*SectionName == ".bss") {
      // .bss section offset matches that of the next section.
      NewSection.sh_offset = NewTextSegmentOffset;
    }

    auto SMII = SectionMM->SectionMapInfo.find(*SectionName);
    if (SMII != SectionMM->SectionMapInfo.end()) {
      auto &SecInfo = SMII->second;
      SecInfo.ShName = Section.sh_name;
    }

    OS.write(reinterpret_cast<const char *>(&NewSection), sizeof(NewSection));
  }

  // Create entries for new allocatable sections.
  std::vector<Elf_Shdr> SectionsToRewrite;
  for (auto &SMII : SectionMM->SectionMapInfo) {
    SectionInfo &SI = SMII.second;
    // Ignore function sections.
    if (SI.IsCode && SMII.first != ".bolt.text")
      continue;
    errs() << "BOLT-INFO: writing section header for "
           << SMII.first << '\n';
    Elf_Shdr NewSection;
    NewSection.sh_name = SI.ShName;
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
    OS.write(reinterpret_cast<const char *>(&SI),
             sizeof(SI));
  }

  auto NumNewSections = SectionsToRewrite.size();

  // Copy over entries for non-allocatable sections performing necessary
  // adjustements.
  for (auto &Section : Obj->sections()) {
    if (Section.sh_type == ELF::SHT_NULL)
      continue;
    if (Section.sh_flags & ELF::SHF_ALLOC)
      continue;

    ErrorOr<StringRef> SectionName = Obj->getSectionName(&Section);
    check_error(SectionName.getError(), "cannot get section name");

    auto SII = SectionMM->NoteSectionInfo.find(*SectionName);
    assert(SII != SectionMM->NoteSectionInfo.end() &&
           "missing section info for non-allocatable section");

    auto NewSection = Section;
    NewSection.sh_offset = SII->second.FileOffset;
    NewSection.sh_size = SII->second.Size;

    // Adjust sh_link for sections that use it.
    if (Section.sh_link)
      NewSection.sh_link = Section.sh_link + NumNewSections;

    // Adjust sh_info for relocation sections.
    if (Section.sh_type == ELF::SHT_REL || Section.sh_type == ELF::SHT_RELA) {
      if (Section.sh_info)
        NewSection.sh_info = Section.sh_info + NumNewSections;
    }

    OS.write(reinterpret_cast<const char *>(&NewSection), sizeof(NewSection));
  }

  // FIXME: Update _end in .dynamic

  // Fix ELF header.
  auto NewEhdr = *Obj->getHeader();
  NewEhdr.e_phoff = PHDRTableOffset;
  NewEhdr.e_phnum = Phnum;
  NewEhdr.e_shoff = SHTOffset;
  NewEhdr.e_shnum = NewEhdr.e_shnum + NumNewSections;
  NewEhdr.e_shstrndx = NewEhdr.e_shstrndx + NumNewSections;
  OS.pwrite(reinterpret_cast<const char *>(&NewEhdr), sizeof(NewEhdr), 0);
}

void RewriteInstance::rewriteFile() {
  // We obtain an asm-specific writer so that we can emit nops in an
  // architecture-specific way at the end of the function.
  auto MCE = BC->TheTarget->createMCCodeEmitter(*BC->MII, *BC->MRI, *BC->Ctx);
  auto MAB = BC->TheTarget->createMCAsmBackend(*BC->MRI, BC->TripleName, "");
  std::unique_ptr<MCStreamer> Streamer(
    BC->TheTarget->createMCObjectStreamer(*BC->TheTriple,
                                          *BC->Ctx,
                                          *MAB,
                                          Out->os(),
                                          MCE,
                                          *BC->STI,
                                          /* RelaxAll */ false,
                                          /* DWARFMustBeAtTheEnd */ false));

  auto &Writer = static_cast<MCObjectStreamer *>(Streamer.get())
                     ->getAssembler()
                     .getWriter();

  // Make sure output stream has enough reserved space, otherwise
  // pwrite() will fail.
  auto Offset = Out->os().seek(getFileOffsetFor(NextAvailableAddress));
  assert(Offset == getFileOffsetFor(NextAvailableAddress) &&
         "error resizing output file");

  // Overwrite function in the output file.
  uint64_t CountOverwrittenFunctions = 0;
  uint64_t OverwrittenScore = 0;
  for (auto &BFI : BinaryFunctions) {
    auto &Function = BFI.second;

    if (Function.getImageAddress() == 0 || Function.getImageSize() == 0)
      continue;

    if (Function.isSplit() && (Function.cold().getImageAddress() == 0 ||
                               Function.cold().getImageSize() == 0))
      continue;

    if (Function.getImageSize() > Function.getMaxSize()) {
      errs() << "BOLT-WARNING: new function size (0x"
             << Twine::utohexstr(Function.getImageSize())
             << ") is larger than maximum allowed size (0x"
             << Twine::utohexstr(Function.getMaxSize())
             << ") for function " << Function.getName() << '\n';
      FailedAddresses.emplace_back(Function.getAddress());
      continue;
    }

    OverwrittenScore += Function.getFunctionScore();
    // Overwrite function in the output file.
    outs() << "BOLT: rewriting function \"" << Function.getName() << "\"\n";
    Out->os().pwrite(reinterpret_cast<char *>(Function.getImageAddress()),
                     Function.getImageSize(), Function.getFileOffset());

    // Write nops at the end of the function.
    auto Pos = Out->os().tell();
    Out->os().seek(Function.getFileOffset() + Function.getImageSize());
    MAB->writeNopData(Function.getMaxSize() - Function.getImageSize(),
                      &Writer);
    Out->os().seek(Pos);

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
    outs() << "BOLT: rewriting function \"" << Function.getName()
           << "\" (cold part)\n";
    Out->os().pwrite(reinterpret_cast<char*>(Function.cold().getImageAddress()),
                     Function.cold().getImageSize(),
                     Function.cold().getFileOffset());

    // FIXME: write nops after cold part too.

    ++CountOverwrittenFunctions;
    if (opts::MaxFunctions && CountOverwrittenFunctions == opts::MaxFunctions) {
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
           << "% of the execution count of simple functions of this binary.\n";
  }

  // Write all non-code sections.
  for (auto &SMII : SectionMM->SectionMapInfo) {
    SectionInfo &SI = SMII.second;
    if (SI.IsCode)
      continue;
    outs() << "BOLT: writing new section " << SMII.first << '\n';
    Out->os().pwrite(reinterpret_cast<const char *>(SI.AllocAddress),
                     SI.Size,
                     SI.FileOffset);
  }

  // If .eh_frame is present it requires special handling.
  auto SMII = SectionMM->SectionMapInfo.find(".eh_frame");
  if (SMII != SectionMM->SectionMapInfo.end()) {
    auto &EHFrameSecInfo = SMII->second;
    outs() << "BOLT: writing a new .eh_frame_hdr\n";
    if (FrameHdrAlign > 1) {
      NextAvailableAddress =
        RoundUpToAlignment(NextAvailableAddress, FrameHdrAlign);
    }

    SectionInfo EHFrameHdrSecInfo;
    EHFrameHdrSecInfo.FileAddress = NextAvailableAddress;
    EHFrameHdrSecInfo.FileOffset = getFileOffsetFor(NextAvailableAddress);

    std::sort(FailedAddresses.begin(), FailedAddresses.end());
    CFIRdWrt->rewriteHeaderFor(
        StringRef(reinterpret_cast<const char *>(EHFrameSecInfo.AllocAddress),
                  EHFrameSecInfo.Size),
        EHFrameSecInfo.FileAddress,
        EHFrameHdrSecInfo.FileAddress,
        FailedAddresses);

    EHFrameHdrSecInfo.Size = FrameHdrCopy.size();

    assert(Out->os().tell() == EHFrameHdrSecInfo.FileOffset &&
           "offset mismatch");
    Out->os().write(FrameHdrCopy.data(), EHFrameHdrSecInfo.Size);

    SectionMM->SectionMapInfo[".eh_frame_hdr"] = EHFrameHdrSecInfo;

    NextAvailableAddress += EHFrameHdrSecInfo.Size;
  }

  // Patch program header table.
  patchELFPHDRTable();

  // Copy non-allocatable sections once allocatable part is finished.
  rewriteNoteSections();

  // Update ELF book-keeping info.
  patchELFSectionHeaderTable();

  // TODO: we should find a way to mark the binary as optimized by us.
  Out->keep();
}

void RewriteInstance::updateAddressRangesObjects() {
  for (auto &Obj : BC->AddressRangesObjects) {
    for (const auto &Range : Obj.getAbsoluteAddressRanges()) {
      RangesSectionsWriter.AddRange(&Obj, Range.first,
                                    Range.second - Range.first);
    }
  }
}

void RewriteInstance::computeLineTableOffsets() {
  const auto LineSection =
    BC->Ctx->getObjectFileInfo()->getDwarfLineSection();
  auto CurrentFragment = LineSection->begin();
  uint32_t CurrentOffset = 0;
  uint32_t Offset = 0;

  // Line tables are stored in MCContext in ascending order of offset in the
  // output file, thus we can compute all table's offset by passing through
  // each fragment at most once, continuing from the last CU's beginning
  // instead of from the first fragment.
  for (const auto &CUIDLineTablePair : BC->Ctx->getMCDwarfLineTables()) {
    auto Label = CUIDLineTablePair.second.getLabel();

    if (!Label)
      continue;

    auto Fragment = Label->getFragment();

    while (&*CurrentFragment != Fragment) {
      switch (CurrentFragment->getKind()) {
      case MCFragment::FT_Dwarf:
        Offset += cast<MCDwarfLineAddrFragment>(*CurrentFragment)
          .getContents().size() - CurrentOffset;
        break;
      case MCFragment::FT_Data:
        Offset += cast<MCDataFragment>(*CurrentFragment)
          .getContents().size() - CurrentOffset;
        break;
      default:
        llvm_unreachable(".debug_line section shouldn't contain other types "
                         "of fragments.");
      }

      ++CurrentFragment;
      CurrentOffset = 0;
    }

    Offset += Label->getOffset() - CurrentOffset;
    CurrentOffset = Label->getOffset();

    auto CompileUnit = BC->OffsetToDwarfCU[CUIDLineTablePair.first];
    BC->CompileUnitLineTableOffset[CompileUnit] = Offset;

    auto LTOI = BC->LineTableOffsetCUMap.find(CUIDLineTablePair.first);
    if (LTOI != BC->LineTableOffsetCUMap.end()) {
      DEBUG(dbgs() << "BOLT-DEBUG: adding relocation for stmt_list "
                   << "in .debug_info\n");
      auto &SI = SectionMM->NoteSectionInfo[".debug_info"];
      SI.PendingRelocs.emplace_back(
          SectionInfo::Reloc{LTOI->second, 4, 0, Offset + DebugLineSize});
    }
    DEBUG(dbgs() << "BOLT-DEBUG: CU " << CUIDLineTablePair.first
                << " has line table at " << Offset << "\n");
  }
}

void RewriteInstance::updateDebugInfo() {
  if (!opts::UpdateDebugSections)
    return;

  SectionPatchers[".debug_abbrev"] = llvm::make_unique<DebugAbbrevPatcher>();
  SectionPatchers[".debug_info"]  = llvm::make_unique<SimpleBinaryPatcher>();

  updateFunctionRanges();

  updateAddressRangesObjects();

  generateDebugRanges();

  updateLocationLists();

  auto &DebugInfoSI = SectionMM->NoteSectionInfo[".debug_info"];
  for (const auto &CU : BC->DwCtx->compile_units()) {
    const auto CUID = CU->getOffset();

    // Update DW_AT_ranges
    auto RangesFieldOffset =
      BC->DwCtx->getAttrFieldOffsetForUnit(CU.get(), dwarf::DW_AT_ranges);
    if (RangesFieldOffset) {
      DEBUG(dbgs() << "BOLT-DEBUG: adding relocation for DW_AT_ranges for "
                   << "compile unit in .debug_info\n");
      const auto RSOI = RangesSectionsWriter.getRangesOffsetCUMap().find(CUID);
      if (RSOI != RangesSectionsWriter.getRangesOffsetCUMap().end()) {
        auto Offset = RSOI->second;
        DebugInfoSI.PendingRelocs.emplace_back(
            SectionInfo::Reloc{RangesFieldOffset, 4, 0,
                               Offset + DebugRangesSize});
      } else {
        DEBUG(dbgs() << "BOLT-DEBUG: no .debug_ranges entry found for CU "
                     << CUID << '\n');
      }
    }
  }

  updateDWARFAddressRanges();
}

void RewriteInstance::updateDWARFAddressRanges() {
  // Update address ranges of functions.
  for (const auto &BFI : BinaryFunctions) {
    const auto &Function = BFI.second;
    for (const auto DIECompileUnitPair : Function.getSubprocedureDIEs()) {
      updateDWARFObjectAddressRanges(
          Function.getAddressRangesOffset() + DebugRangesSize,
          DIECompileUnitPair.second,
          DIECompileUnitPair.first);
    }
  }

  // Update address ranges of DIEs with addresses that don't match functions.
  for (auto &DIECompileUnitPair : BC->UnknownFunctions) {
    updateDWARFObjectAddressRanges(
        RangesSectionsWriter.getEmptyRangesListOffset(),
        DIECompileUnitPair.second,
        DIECompileUnitPair.first);
  }

  // Update address ranges of DWARF block objects (lexical/try/catch blocks,
  // inlined subroutine instances, etc).
  for (const auto &Obj : BC->AddressRangesObjects) {
    updateDWARFObjectAddressRanges(
        Obj.getAddressRangesOffset() + DebugRangesSize,
        Obj.getCompileUnit(),
        Obj.getDIE());
  }
}

void RewriteInstance::updateDWARFObjectAddressRanges(
    uint32_t DebugRangesOffset,
    const DWARFUnit *Unit,
    const DWARFDebugInfoEntryMinimal *DIE) {

  // Some objects don't have an associated DIE and cannot be updated (such as
  // compiler-generated functions).
  if (!DIE) {
    return;
  }

  auto DebugInfoPatcher =
      static_cast<SimpleBinaryPatcher *>(SectionPatchers[".debug_info"].get());
  auto AbbrevPatcher =
      static_cast<DebugAbbrevPatcher*>(SectionPatchers[".debug_abbrev"].get());

  assert(DebugInfoPatcher && AbbrevPatcher && "Patchers not initialized.");

  const auto *AbbreviationDecl = DIE->getAbbreviationDeclarationPtr();
  assert(AbbreviationDecl &&
         "Object's DIE doesn't have an abbreviation: not supported yet.");
  auto AbbrevCode = AbbreviationDecl->getCode();

  if (AbbreviationDecl->findAttributeIndex(dwarf::DW_AT_ranges) != -1U) {
    // Case 1: The object was already non-contiguous and had DW_AT_ranges.
    // In this case we simply need to update the value of DW_AT_ranges.
    DWARFFormValue FormValue;
    uint32_t AttrOffset = -1U;
    DIE->getAttributeValue(Unit, dwarf::DW_AT_ranges, FormValue, &AttrOffset);
    DebugInfoPatcher->addLE32Patch(AttrOffset, DebugRangesOffset);
  } else {
    // Case 2: The object has both DW_AT_low_pc and DW_AT_high_pc.
    // We require the compiler to put both attributes one after the other
    // for our approach to work. low_pc and high_pc both occupy 8 bytes
    // as we're dealing with a 64-bit ELF. We basically change low_pc to
    // DW_AT_ranges and high_pc to DW_AT_producer. ranges spans only 4 bytes
    // in 32-bit DWARF, which we assume to be used, which leaves us with 12
    // more bytes. We then set the value of DW_AT_producer as an arbitrary
    // 12-byte string that fills the remaining space and leaves the rest of
    // the abbreviation layout unchanged.
    if (AbbreviationDecl->findAttributeIndex(dwarf::DW_AT_low_pc) != -1U &&
        AbbreviationDecl->findAttributeIndex(dwarf::DW_AT_high_pc) != -1U) {
      uint32_t LowPCOffset = -1U;
      uint32_t HighPCOffset = -1U;
      DWARFFormValue FormValue;
      DIE->getAttributeValue(Unit, dwarf::DW_AT_low_pc, FormValue,
                             &LowPCOffset);
      DIE->getAttributeValue(Unit, dwarf::DW_AT_high_pc, FormValue,
                             &HighPCOffset);

      AbbrevPatcher->addAttributePatch(Unit,
                                       AbbrevCode,
                                       dwarf::DW_AT_low_pc,
                                       dwarf::DW_AT_ranges,
                                       dwarf::DW_FORM_sec_offset);
      AbbrevPatcher->addAttributePatch(Unit,
                                       AbbrevCode,
                                       dwarf::DW_AT_high_pc,
                                       dwarf::DW_AT_producer,
                                       dwarf::DW_FORM_string);
      assert(LowPCOffset != -1U && LowPCOffset + 8 == HighPCOffset &&
             "We depend on the compiler putting high_pc right after low_pc.");
      DebugInfoPatcher->addLE32Patch(LowPCOffset, DebugRangesOffset);
      std::string ProducerString{"LLVM-BOLT"};
      ProducerString.resize(12, ' ');
      ProducerString.back() = '\0';

      DebugInfoPatcher->addBinaryPatch(LowPCOffset + 4, ProducerString);
    } else {
      DEBUG(errs() << "BOLT-WARNING: Cannot update ranges for DIE at offset 0x"
                   << Twine::utohexstr(DIE->getOffset()) << "\n");
    }
  }
}

void RewriteInstance::updateDebugLineInfoForNonSimpleFunctions() {
  if (!opts::UpdateDebugSections)
    return;

  auto DebugAranges = BC->DwCtx->getDebugAranges();
  assert(DebugAranges && "Need .debug_aranges in the input file.");

  for (auto It : BinaryFunctions) {
    const auto &Function = It.second;

    if (Function.isSimple())
      continue;

    uint64_t Address = It.first;
    uint32_t CUOffset = DebugAranges->findAddress(Address);
    if (CUOffset == -1U) {
      DEBUG(errs() << "BOLT-DEBUG: Function does not belong to any compile unit"
                   << "in .debug_aranges: " << Function.getName() << "\n");
      continue;
    }
    auto Unit = BC->OffsetToDwarfCU[CUOffset];
    auto LineTable = BC->DwCtx->getLineTableForUnit(Unit);
    assert(LineTable && "CU without .debug_line info.");

    std::vector<uint32_t> Results;
    MCSectionELF *FunctionSection =
        BC->Ctx->getELFSection(Function.getCodeSectionName(),
                               ELF::SHT_PROGBITS,
                               ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);

    if (LineTable->lookupAddressRange(Address, Function.getSize(), Results)) {
      for (auto RowIndex : Results) {
        const auto &Row = LineTable->Rows[RowIndex];
        BC->Ctx->setCurrentDwarfLoc(
            Row.File,
            Row.Line,
            Row.Column,
            (DWARF2_FLAG_IS_STMT * Row.IsStmt) |
            (DWARF2_FLAG_BASIC_BLOCK * Row.BasicBlock) |
            (DWARF2_FLAG_PROLOGUE_END * Row.PrologueEnd) |
            (DWARF2_FLAG_EPILOGUE_BEGIN * Row.EpilogueBegin),
            Row.Isa,
            Row.Discriminator,
            Row.Address);

        auto Loc = BC->Ctx->getCurrentDwarfLoc();
        BC->Ctx->clearDwarfLocSeen();

        auto &OutputLineTable =
            BC->Ctx->getMCDwarfLineTable(CUOffset).getMCLineSections();
        OutputLineTable.addLineEntry(MCLineEntry{nullptr, Loc},
                                     FunctionSection);
      }
    } else {
      DEBUG(errs() << "BOLT-DEBUG: Function " << Function.getName()
                   << " has no associated line number information.\n");
    }
  }
}
