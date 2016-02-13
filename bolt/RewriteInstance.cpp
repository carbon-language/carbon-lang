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
#include "DataReader.h"
#include "Exceptions.h"
#include "RewriteInstance.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
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
FunctionNames("funcs",
              cl::CommaSeparated,
              cl::desc("list of functions to optimize"),
              cl::value_desc("func1,func2,func3,..."));

static cl::opt<std::string>
FunctionNamesFile("funcs_file",
                  cl::desc("file with list of functions to optimize"));

static cl::list<std::string>
SkipFunctionNames("skip_funcs",
                  cl::CommaSeparated,
                  cl::desc("list of functions to skip"),
                  cl::value_desc("func1,func2,func3,..."));

static cl::opt<unsigned>
MaxFunctions("max_funcs",
             cl::desc("maximum # of functions to overwrite"),
             cl::Optional);

static cl::opt<bool>
EliminateUnreachable("eliminate-unreachable",
                     cl::desc("eliminate unreachable code"),
                     cl::Optional);

static cl::opt<bool>
SplitFunctions("split-functions",
               cl::desc("split functions into hot and cold distinct regions"),
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

  if (!FunctionNamesFile.empty()) {
    std::ifstream FuncsFile(FunctionNamesFile, std::ios::in);
    std::string FuncName;
    while (std::getline(FuncsFile, FuncName)) {
      FunctionNames.push_back(FuncName);
    }
    FunctionNamesFile = "";
  }

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
                                            IsReadOnly);

  return ret;
}

bool ExecutableFileMemoryManager::finalizeMemory(std::string *ErrMsg) {
  DEBUG(dbgs() << "BOLT: finalizeMemory()\n");
  return SectionMemoryManager::finalizeMemory(ErrMsg);
}

/// Create BinaryContext for a given architecture \p ArchName and
/// triple \p TripleName.
static std::unique_ptr<BinaryContext> CreateBinaryContext(
    std::string ArchName,
    std::string TripleName, const DataReader &DR) {

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
    : File(File), BC(CreateBinaryContext("x86-64", "x86_64-unknown-linux", DR)),
      DwCtx(new DWARFContextInMemory(*File)) {}

RewriteInstance::~RewriteInstance() {}

void RewriteInstance::reset() {
  BinaryFunctions.clear();
  FileSymRefs.clear();
  auto &DR = BC->DR;
  BC = CreateBinaryContext("x86-64", "x86_64-unknown-linux", DR);
  DwCtx.reset(new DWARFContextInMemory(*File));
  CFIRdWrt.reset(nullptr);
  SectionMM.reset(nullptr);
  Out.reset(nullptr);
  EHFrame = nullptr;
  FailedAddresses.clear();
  TotalScore = 0;
}

void RewriteInstance::discoverStorage() {
  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(File);
  if (!ELF64LEFile) {
    errs() << "BOLT-ERROR: only 64-bit LE ELF binaries are supported\n";
    exit(1);
  }

  auto Obj = ELF64LEFile->getELFFile();


  // Discover important addresses in the binary.

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

  discoverStorage();
  readSymbolTable();
  readSpecialSections();
  disassembleFunctions();
  runOptimizationPasses();
  emitFunctions();

  if (opts::SplitFunctions && splitLargeFunctions()) {
    // Emit again because now some functions have been split
    outs() << "BOLT: split-functions: starting pass 2...\n";
    reset();
    discoverStorage();
    readSymbolTable();
    readSpecialSections();
    disassembleFunctions();
    runOptimizationPasses();
    emitFunctions();
  }

  // Copy input file to output
  std::error_code EC;
  Out = llvm::make_unique<tool_output_file>(opts::OutputFilename, EC,
                                            sys::fs::F_None, 0777);
  check_error(EC, "cannot create output executable file");
  Out->os() << File->getData();

  // Rewrite optimized functions back to this output
  rewriteFile();
}

void RewriteInstance::readSymbolTable() {
  std::string FileSymbolName;

  FileSymRefs.clear();
  BinaryFunctions.clear();
  BC->GlobalAddresses.clear();

  // For local symbols we want to keep track of associated FILE symbol for
  // disambiguation by name.
  for (const SymbolRef &Symbol : File->symbols()) {
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
    if (Section == File->section_end()) {
      // Could be an absolute symbol. Could record for pretty printing.
      continue;
    }

    // Create the function and add to the map.
    BinaryFunctions.emplace(
        Address,
        BinaryFunction(UniqueName, Symbol, *Section, Address,
                       SymbolSize, *BC)
    );
  }
}

void RewriteInstance::readSpecialSections() {
  // Process special sections.
  StringRef FrameHdrContents;
  for (const auto &Section : File->sections()) {
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
    }
    if (SectionName == ".eh_frame_hdr") {
      FrameHdrAddress = Section.getAddress();
      FrameHdrContents = SectionContents;
      FrameHdrAlign = Section.getAlignment();
    }
  }

  FrameHdrCopy =
      std::vector<char>(FrameHdrContents.begin(), FrameHdrContents.end());
  // Process debug sections.
  EHFrame = DwCtx->getEHFrame();
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

void RewriteInstance::disassembleFunctions() {
  // Disassemble every function and build it's control flow graph.
  TotalScore = 0;
  for (auto &BFI : BinaryFunctions) {
    BinaryFunction &Function = BFI.second;

    if (!opts::shouldProcess(Function)) {
      DEBUG(dbgs() << "BOLT: skipping processing function " << Function.getName()
                   << " per user request.\n");
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
      auto MaxSize = SymRefI->first - Function.getAddress();
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
        SectionContents.data() - File->getData().data() + FunctionOffset);

    ArrayRef<uint8_t> FunctionData(
        reinterpret_cast<const uint8_t *>
          (SectionContents.data()) + FunctionOffset,
        Function.getSize());

    if (!Function.disassemble(FunctionData))
      continue;

    if (opts::PrintAll || opts::PrintDisasm)
      Function.print(errs(), "after disassembly");

    if (!Function.isSimple())
      continue;

    // Fill in CFI information for this function
    if (EHFrame->ParseError.empty() && Function.isSimple()) {
      CFIRdWrt->fillCFIInfoFor(Function);
    }

    // Parse LSDA.
    if (Function.getLSDAAddress() != 0)
      Function.parseLSDA(LSDAData, LSDAAddress);

    if (!Function.buildCFG())
      continue;

    if (opts::PrintAll || opts::PrintCFG)
      Function.print(errs(), "after building cfg");

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
        Function.print(errs(), "after unreachable code elimination");
    }

    if (opts::ReorderBlocks != BinaryFunction::LT_NONE) {
      bool ShouldSplit = ToSplit.find(BFI.first) != ToSplit.end();
      BFI.second.modifyLayout(opts::ReorderBlocks, ShouldSplit);
      if (opts::PrintAll || opts::PrintReordered)
        Function.print(errs(), "after reordering blocks");
    }

    // Post-processing passes.

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
      Function.print(errs(), "after updating EH ranges");

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

  // Emit code.
  for (auto BB : Function.layout()) {
    if (EmitColdPart != BB->isCold())
      continue;
    if (opts::AlignBlocks && BB->getAlignment() > 1)
      Streamer.EmitCodeAlignment(BB->getAlignment());
    Streamer.EmitLabel(BB->getLabel());
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
      if (!BC.MIA->isCFI(Instr)) {
        Streamer.EmitInstruction(Instr, *BC.STI);
        continue;
      }
      emitCFIInstr(*Function.getCFIFor(Instr));
    }
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

  Streamer->Finish();

  //////////////////////////////////////////////////////////////////////////////
  // Assign addresses to new functions/sections.
  //////////////////////////////////////////////////////////////////////////////

  // Get output object as ObjectFile.
  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(BOS->str(), "in-memory object file", false);
  ErrorOr<std::unique_ptr<object::ObjectFile>> ObjOrErr =
    object::ObjectFile::createObjectFile(ObjectMemBuffer->getMemBufferRef());
  check_error(ObjOrErr.getError(), "error creating in-memory object");

  auto EFMM = new ExecutableFileMemoryManager();
  SectionMM.reset(EFMM);


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
        std::move(Resolver));

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
          reinterpret_cast<const void*>(SMII->second.AllocAddress),
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
          reinterpret_cast<const void*>(SMII->second.AllocAddress),
          NextAvailableAddress);
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
                            reinterpret_cast<const void *>(SI.AllocAddress),
                            NextAvailableAddress);

      SI.FileAddress = NextAvailableAddress;
      SI.FileOffset = getFileOffsetFor(NextAvailableAddress);

      NextAvailableAddress += SI.Size;
    } else {
      errs() << "BOLT: cannot remap " << SectionName << '\n';
    }
  }

  OLT.emitAndFinalize(ObjectsHandle);

  if (opts::KeepTmp)
    TempOut->keep();
}

bool RewriteInstance::splitLargeFunctions() {
  bool Changed = false;
  for (auto &BFI : BinaryFunctions) {
    auto &Function = BFI.second;

    // Ignore this function if we failed to map it to the output binary
    if (Function.getImageAddress() == 0 || Function.getImageSize() == 0)
      continue;

    if (Function.getImageSize() <= Function.getMaxSize())
      continue;

    ToSplit.insert(BFI.first);
    Changed = true;
  }
  return Changed;
}

void RewriteInstance::patchELF() {
  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(File);
  if (!ELF64LEFile) {
    errs() << "BOLT-ERROR: only 64-bit LE ELF binaries are supported\n";
    exit(1);
  }
  auto Obj = ELF64LEFile->getELFFile();
  auto &OS = Out->os();

  // Write/re-write program headers.
  unsigned Phnum = Obj->getHeader()->e_phnum;
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

  // Copy original non-allocatable contents and update section offsets.
  uint64_t NextAvailableOffset = getFileOffsetFor(NextAvailableAddress);
  assert(NextAvailableOffset >= FirstNonAllocatableOffset &&
         "next available offset calculation failure");

  // Re-write using this offset delta.
  uint64_t OffsetDelta = NextAvailableOffset - FirstNonAllocatableOffset;

  // Make sure offset delta is a multiple of alignment;
  OffsetDelta = RoundUpToAlignment(OffsetDelta, MaxNonAllocAlign);
  NextAvailableOffset = FirstNonAllocatableOffset + OffsetDelta;

  // FIXME: only write up to SHDR table.
  OS.seek(NextAvailableOffset);
  OS << File->getData().drop_front(FirstNonAllocatableOffset);

  bool SeenNonAlloc = false;
  uint64_t ExtraDelta = 0;  // for dynamically adjusting delta
  unsigned NumNewSections = 0;

  // Update section table. Note that the section table itself has shifted.
  OS.seek(Obj->getHeader()->e_shoff + OffsetDelta);
  for (auto &Section : Obj->sections()) {
    // Always ignore this section.
    if (Section.sh_type == ELF::SHT_NULL) {
      OS.write(reinterpret_cast<const char *>(&Section), sizeof(Section));
      continue;
    }

    auto NewSection = Section;
    uint64_t SectionLoc = (uintptr_t)&Section - (uintptr_t)Obj->base();

    ErrorOr<StringRef> SectionName = Obj->getSectionName(&Section);
    check_error(SectionName.getError(), "cannot get section name");

    if (!(Section.sh_flags & ELF::SHF_ALLOC)) {
      if (!SeenNonAlloc) {

        // This is where we place all our new sections.

        std::vector<decltype(NewSection)> SectionsToRewrite;
        for (auto &SMII : SectionMM->SectionMapInfo) {
          SectionInfo &SI = SMII.second;
          if (SI.IsCode && SMII.first != ".bolt.text")
            continue;
          errs() << "BOLT-INFO: re-writing section header for "
                 << SMII.first << '\n';
          auto NewSection = Section;
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

        // Do actual writing after sorting out.
        OS.seek(SectionLoc + OffsetDelta);
        std::stable_sort(SectionsToRewrite.begin(), SectionsToRewrite.end(),
            [] (decltype(Section) A, decltype(Section) B) {
              return A.sh_offset < B.sh_offset;
            });
        for (auto &SI : SectionsToRewrite) {
          OS.write(reinterpret_cast<const char *>(&SI),
                   sizeof(SI));
        }

        NumNewSections = SectionsToRewrite.size();
        ExtraDelta += sizeof(Section) * NumNewSections;

        SeenNonAlloc = true;
      }

      assert(Section.sh_addralign <= MaxNonAllocAlign &&
             "unexpected alignment for non-allocatable section");
      assert(Section.sh_offset >= FirstNonAllocatableOffset &&
             "bad offset for non-allocatable section");

      NewSection.sh_offset = Section.sh_offset + OffsetDelta;

      if (Section.sh_offset > Obj->getHeader()->e_shoff) {
        // The section is going to be shifted.
        NewSection.sh_offset = NewSection.sh_offset + ExtraDelta;
      }

      if (Section.sh_link)
        NewSection.sh_link = Section.sh_link + NumNewSections;

    } else if (*SectionName == ".bss") {
      NewSection.sh_offset = NewTextSegmentOffset;
    }

    auto SMII = SectionMM->SectionMapInfo.find(*SectionName);
    if (SMII != SectionMM->SectionMapInfo.end()) {
      auto &SecInfo = SMII->second;
      SecInfo.ShName = Section.sh_name;
    }

    OS.write(reinterpret_cast<const char *>(&NewSection), sizeof(NewSection));
  }

  // Write all the sections past the section table again as they are shifted.
  auto OffsetPastShdrTable = Obj->getHeader()->e_shoff +
      Obj->getHeader()->e_shnum * sizeof(ELFFile<ELF64LE>::Elf_Shdr);
  OS.seek(OffsetPastShdrTable + OffsetDelta + ExtraDelta);
  OS << File->getData().drop_front(OffsetPastShdrTable);

  // FIXME: Update _end in .dynamic

  // Fix ELF header.
  auto NewEhdr = *Obj->getHeader();
  NewEhdr.e_phoff = PHDRTableOffset;
  NewEhdr.e_phnum = Phnum;
  NewEhdr.e_shoff = NewEhdr.e_shoff + OffsetDelta;
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

  outs() << "BOLT: " << CountOverwrittenFunctions
         << " out of " << BinaryFunctions.size()
         << " functions were overwritten.\n";

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

  // Update ELF book-keeping info.
  patchELF();

  if (TotalScore != 0) {
    double Coverage = OverwrittenScore / (double)TotalScore * 100.0;
    outs() << format("BOLT: Rewritten functions cover %.2lf", Coverage)
           << "% of the execution count of simple functions of this binary.\n";
  }

  // TODO: we should find a way to mark the binary as optimized by us.
  Out->keep();
}
