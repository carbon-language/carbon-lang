//===--- BinaryContext.cpp  - Interface for machine-level context ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryContext.h"
#include "BinaryEmitter.h"
#include "BinaryFunction.h"
#include "ParallelUtilities.h"
#include "Utils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include <functional>
#include <iterator>

using namespace llvm;

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

namespace opts {

extern cl::OptionCategory BoltCategory;

extern cl::opt<bool> AggregateOnly;
extern cl::opt<bool> StrictMode;
extern cl::opt<bool> UseOldText;
extern cl::opt<unsigned> Verbosity;

extern bool processAllFunctions();

cl::opt<bool>
NoHugePages("no-huge-pages",
  cl::desc("use regular size pages for code alignment"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<bool>
PrintDebugInfo("print-debug-info",
  cl::desc("print debug info when printing functions"),
  cl::Hidden,
  cl::ZeroOrMore,
  cl::cat(BoltCategory));

cl::opt<bool>
PrintRelocations("print-relocations",
  cl::desc("print relocations when printing functions/objects"),
  cl::Hidden,
  cl::ZeroOrMore,
  cl::cat(BoltCategory));

static cl::opt<bool>
PrintMemData("print-mem-data",
  cl::desc("print memory data annotations when printing functions"),
  cl::Hidden,
  cl::ZeroOrMore,
  cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

BinaryContext::BinaryContext(std::unique_ptr<MCContext> Ctx,
                             std::unique_ptr<DWARFContext> DwCtx,
                             std::unique_ptr<Triple> TheTriple,
                             const Target *TheTarget,
                             std::string TripleName,
                             std::unique_ptr<MCCodeEmitter> MCE,
                             std::unique_ptr<MCObjectFileInfo> MOFI,
                             std::unique_ptr<const MCAsmInfo> AsmInfo,
                             std::unique_ptr<const MCInstrInfo> MII,
                             std::unique_ptr<const MCSubtargetInfo> STI,
                             std::unique_ptr<MCInstPrinter> InstPrinter,
                             std::unique_ptr<const MCInstrAnalysis> MIA,
                             std::unique_ptr<MCPlusBuilder> MIB,
                             std::unique_ptr<const MCRegisterInfo> MRI,
                             std::unique_ptr<MCDisassembler> DisAsm)
    : Ctx(std::move(Ctx)),
      DwCtx(std::move(DwCtx)),
      TheTriple(std::move(TheTriple)),
      TheTarget(TheTarget),
      TripleName(TripleName),
      MCE(std::move(MCE)),
      MOFI(std::move(MOFI)),
      AsmInfo(std::move(AsmInfo)),
      MII(std::move(MII)),
      STI(std::move(STI)),
      InstPrinter(std::move(InstPrinter)),
      MIA(std::move(MIA)),
      MIB(std::move(MIB)),
      MRI(std::move(MRI)),
      DisAsm(std::move(DisAsm)) {
  Relocation::Arch = this->TheTriple->getArch();
  PageAlign = opts::NoHugePages ? RegularPageSize : HugePageSize;
}

BinaryContext::~BinaryContext() {
  for (auto *Section : Sections) {
    delete Section;
  }
  for (auto *InjectedFunction : InjectedBinaryFunctions) {
    delete InjectedFunction;
  }
  for (auto JTI : JumpTables) {
    delete JTI.second;
  }
  clearBinaryData();
}

extern MCPlusBuilder *createX86MCPlusBuilder(const MCInstrAnalysis *,
                                             const MCInstrInfo *,
                                             const MCRegisterInfo *);
extern MCPlusBuilder *createAArch64MCPlusBuilder(const MCInstrAnalysis *,
                                                 const MCInstrInfo *,
                                                 const MCRegisterInfo *);

namespace {

MCPlusBuilder *createMCPlusBuilder(const Triple::ArchType Arch,
                                   const MCInstrAnalysis *Analysis,
                                   const MCInstrInfo *Info,
                                   const MCRegisterInfo *RegInfo) {
#ifdef X86_AVAILABLE
  if (Arch == Triple::x86_64)
    return createX86MCPlusBuilder(Analysis, Info, RegInfo);
#endif

#ifdef AARCH64_AVAILABLE
  if (Arch == Triple::aarch64)
    return createAArch64MCPlusBuilder(Analysis, Info, RegInfo);
#endif

  llvm_unreachable("architecture unsupport by MCPlusBuilder");
}

} // anonymous namespace

/// Create BinaryContext for a given architecture \p ArchName and
/// triple \p TripleName.
std::unique_ptr<BinaryContext>
BinaryContext::createBinaryContext(ObjectFile *File,
                                   std::unique_ptr<DWARFContext> DwCtx) {
  StringRef ArchName = "";
  StringRef FeaturesStr = "";
  switch (File->getArch()) {
  case llvm::Triple::x86_64:
    ArchName = "x86-64";
    FeaturesStr = "+nopl";
    break;
  case llvm::Triple::aarch64:
    ArchName = "aarch64";
    FeaturesStr = "+fp-armv8,+neon,+crypto,+dotprod,+crc,+lse,+ras,+rdm,"
                  "+fullfp16,+spe,+fuse-aes,+rcpc";
    break;
  default:
    errs() << "BOLT-ERROR: Unrecognized machine in ELF file.\n";
    return nullptr;
  }

  auto TheTriple = llvm::make_unique<Triple>(File->makeTriple());
  const llvm::StringRef TripleName = TheTriple->str();

  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(ArchName, *TheTriple, Error);
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
      TheTarget->createMCSubtargetInfo(TripleName, "", FeaturesStr));
  if (!STI) {
    errs() << "BOLT-ERROR: no subtarget info for target " << TripleName << "\n";
    return nullptr;
  }

  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII) {
    errs() << "BOLT-ERROR: no instruction info for target " << TripleName
           << "\n";
    return nullptr;
  }

  std::unique_ptr<MCObjectFileInfo> MOFI =
      llvm::make_unique<MCObjectFileInfo>();
  std::unique_ptr<MCContext> Ctx =
      llvm::make_unique<MCContext>(AsmInfo.get(), MRI.get(), MOFI.get());
  MOFI->InitMCObjectFileInfo(*TheTriple, /*PIC=*/false, *Ctx);

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

  std::unique_ptr<MCPlusBuilder> MIB(createMCPlusBuilder(
      TheTriple->getArch(), MIA.get(), MII.get(), MRI.get()));
  if (!MIB) {
    errs() << "BOLT-ERROR: failed to create instruction builder for target"
           << TripleName << "\n";
    return nullptr;
  }

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> InstructionPrinter(
      TheTarget->createMCInstPrinter(*TheTriple, AsmPrinterVariant, *AsmInfo,
                                     *MII, *MRI));
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

  auto BC = llvm::make_unique<BinaryContext>(
      std::move(Ctx), std::move(DwCtx), std::move(TheTriple), TheTarget,
      TripleName, std::move(MCE), std::move(MOFI), std::move(AsmInfo),
      std::move(MII), std::move(STI), std::move(InstructionPrinter),
      std::move(MIA), std::move(MIB), std::move(MRI), std::move(DisAsm));

  BC->MAB = std::unique_ptr<MCAsmBackend>(
      BC->TheTarget->createMCAsmBackend(*BC->STI, *BC->MRI, MCTargetOptions()));

  BC->setFilename(File->getFileName());

  return BC;
}

std::unique_ptr<MCObjectWriter>
BinaryContext::createObjectWriter(raw_pwrite_stream &OS) {
  return MAB->createObjectWriter(OS);
}

bool BinaryContext::validateObjectNesting() const {
  auto Itr = BinaryDataMap.begin();
  auto End = BinaryDataMap.end();
  bool Valid = true;
  while (Itr != End) {
    auto Next = std::next(Itr);
    while (Next != End &&
           Itr->second->getSection() == Next->second->getSection() &&
           Itr->second->containsRange(Next->second->getAddress(),
                                      Next->second->getSize())) {
      if (Next->second->Parent != Itr->second) {
        errs() << "BOLT-WARNING: object nesting incorrect for:\n"
               << "BOLT-WARNING:  " << *Itr->second << "\n"
               << "BOLT-WARNING:  " << *Next->second << "\n";
        Valid = false;
      }
      ++Next;
    }
    Itr = Next;
  }
  return Valid;
}

bool BinaryContext::validateHoles() const {
  bool Valid = true;
  for (auto &Section : sections()) {
    for (const auto &Rel : Section.relocations()) {
      auto RelAddr = Rel.Offset + Section.getAddress();
      auto *BD = getBinaryDataContainingAddress(RelAddr);
      if (!BD) {
        errs() << "BOLT-WARNING: no BinaryData found for relocation at address"
               << " 0x" << Twine::utohexstr(RelAddr) << " in "
               << Section.getName() << "\n";
        Valid = false;
      } else if (!BD->getAtomicRoot()) {
        errs() << "BOLT-WARNING: no atomic BinaryData found for relocation at "
               << "address 0x" << Twine::utohexstr(RelAddr) << " in "
               << Section.getName() << "\n";
        Valid = false;
      }
    }
  }
  return Valid;
}

void BinaryContext::updateObjectNesting(BinaryDataMapType::iterator GAI) {
  const auto Address = GAI->second->getAddress();
  const auto Size = GAI->second->getSize();

  auto fixParents =
    [&](BinaryDataMapType::iterator Itr, BinaryData *NewParent) {
    auto *OldParent = Itr->second->Parent;
    Itr->second->Parent = NewParent;
    ++Itr;
    while (Itr != BinaryDataMap.end() && OldParent &&
           Itr->second->Parent == OldParent) {
      Itr->second->Parent = NewParent;
      ++Itr;
    }
  };

  // Check if the previous symbol contains the newly added symbol.
  if (GAI != BinaryDataMap.begin()) {
    auto *Prev = std::prev(GAI)->second;
    while (Prev) {
      if (Prev->getSection() == GAI->second->getSection() &&
          Prev->containsRange(Address, Size)) {
        fixParents(GAI, Prev);
      } else {
        fixParents(GAI, nullptr);
      }
      Prev = Prev->Parent;
    }
  }

  // Check if the newly added symbol contains any subsequent symbols.
  if (Size != 0) {
    auto *BD = GAI->second->Parent ? GAI->second->Parent : GAI->second;
    auto Itr = std::next(GAI);
    while (Itr != BinaryDataMap.end() &&
           BD->containsRange(Itr->second->getAddress(),
                             Itr->second->getSize())) {
      Itr->second->Parent = BD;
      ++Itr;
    }
  }
}

iterator_range<BinaryContext::binary_data_iterator>
BinaryContext::getSubBinaryData(BinaryData *BD) {
  auto Start = std::next(BinaryDataMap.find(BD->getAddress()));
  auto End = Start;
  while (End != BinaryDataMap.end() &&
         BD->isAncestorOf(End->second)) {
    ++End;
  }
  return make_range(Start, End);
}

std::pair<const MCSymbol *, uint64_t>
BinaryContext::handleAddressRef(uint64_t Address, BinaryFunction &BF,
                                bool IsPCRel) {
  uint64_t Addend{0};

  if (isAArch64()) {
    // Check if this is an access to a constant island and create bookkeeping
    // to keep track of it and emit it later as part of this function.
    if (MCSymbol *IslandSym = BF.getOrCreateIslandAccess(Address))
      return std::make_pair(IslandSym, Addend);

    // Detect custom code written in assembly that refers to arbitrary
    // constant islands from other functions. Write this reference so we
    // can pull this constant island and emit it as part of this function
    // too.
    auto IslandIter = AddressToConstantIslandMap.lower_bound(Address);
    if (IslandIter != AddressToConstantIslandMap.end()) {
      if (auto *IslandSym =
              IslandIter->second->getOrCreateProxyIslandAccess(Address, BF)) {
        /// Make this function depend on IslandIter->second because we have
        /// a reference to its constant island. When emitting this function,
        /// we will also emit IslandIter->second's constants. This only
        /// happens in custom AArch64 assembly code.
        BF.Islands.Dependency.insert(IslandIter->second);
        BF.Islands.ProxySymbols[IslandSym] = IslandIter->second;
        return std::make_pair(IslandSym, Addend);
      }
    }
  }

  // Note that the address does not necessarily have to reside inside
  // a section, it could be an absolute address too.
  auto Section = getSectionForAddress(Address);
  if (Section && Section->isText()) {
    if (BF.containsAddress(Address, /*UseMaxSize=*/ isAArch64())) {
      if (Address != BF.getAddress()) {
        // The address could potentially escape. Mark it as another entry
        // point into the function.
        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: potentially escaped address 0x"
                 << Twine::utohexstr(Address) << " in function "
                 << BF << '\n';
        }
        BF.HasInternalLabelReference = true;
        return std::make_pair(
                  BF.addEntryPointAtOffset(Address - BF.getAddress()),
                  Addend);
      }
    } else {
      BF.InterproceduralReferences.insert(Address);
    }
  }

  // With relocations, catch jump table references outside of the basic block
  // containing the indirect jump.
  if (HasRelocations) {
    const auto MemType = analyzeMemoryAt(Address, BF);
    if (MemType == MemoryContentsType::POSSIBLE_PIC_JUMP_TABLE && IsPCRel) {
      const MCSymbol *Symbol =
        getOrCreateJumpTable(BF, Address, JumpTable::JTT_PIC);

      return std::make_pair(Symbol, Addend);
    }
  }

  if (auto *BD = getBinaryDataContainingAddress(Address)) {
    return std::make_pair(BD->getSymbol(), Address - BD->getAddress());
  }

  // TODO: use DWARF info to get size/alignment here?
  auto *TargetSymbol = getOrCreateGlobalSymbol(Address, "DATAat");
  DEBUG(dbgs() << "Created symbol " << TargetSymbol->getName());
  return std::make_pair(TargetSymbol, Addend);
}

MemoryContentsType
BinaryContext::analyzeMemoryAt(uint64_t Address, BinaryFunction &BF) {
  if (!isX86())
    return MemoryContentsType::UNKNOWN;

  auto Section = getSectionForAddress(Address);
  if (!Section) {
    // No section - possibly an absolute address. Since we don't allow
    // internal function addresses to escape the function scope - we
    // consider it a tail call.
    if (opts::Verbosity > 1) {
      errs() << "BOLT-WARNING: no section for address 0x"
             << Twine::utohexstr(Address) << " referenced from function "
             << BF << '\n';
    }
    return MemoryContentsType::UNKNOWN;
  }

  if (Section->isVirtual()) {
    // The contents are filled at runtime.
    return MemoryContentsType::UNKNOWN;
  }

  // No support for jump tables in code yet.
  if (Section->isText())
    return MemoryContentsType::UNKNOWN;

  // Start with checking for PIC jump table. We expect non-PIC jump tables
  // to have high 32 bits set to 0.
  if (analyzeJumpTable(Address, JumpTable::JTT_PIC, BF))
    return MemoryContentsType::POSSIBLE_PIC_JUMP_TABLE;

  if (analyzeJumpTable(Address, JumpTable::JTT_NORMAL, BF))
    return MemoryContentsType::POSSIBLE_JUMP_TABLE;

  return MemoryContentsType::UNKNOWN;
}

bool BinaryContext::analyzeJumpTable(const uint64_t Address,
                                     const JumpTable::JumpTableType Type,
                                     const BinaryFunction &BF,
                                     const uint64_t NextJTAddress,
                                     JumpTable::OffsetsType *Offsets) {
  // Is one of the targets __builtin_unreachable?
  bool HasUnreachable{false};

  // Number of targets other than __builtin_unreachable.
  uint64_t NumRealEntries{0};

  auto addOffset = [&](uint64_t Offset) {
    if (Offsets)
      Offsets->emplace_back(Offset);
  };

  auto Section = getSectionForAddress(Address);
  if (!Section)
    return false;

  // The upper bound is defined by containing object, section limits, and
  // the next jump table in memory.
  auto UpperBound = Section->getEndAddress();
  const auto *JumpTableBD = getBinaryDataAtAddress(Address);
  if (JumpTableBD && JumpTableBD->getSize()) {
    assert(JumpTableBD->getEndAddress() <= UpperBound &&
           "data object cannot cross a section boundary");
    UpperBound = JumpTableBD->getEndAddress();
  }
  if (NextJTAddress) {
    UpperBound = std::min(NextJTAddress, UpperBound);
  }

  const auto EntrySize = getJumpTableEntrySize(Type);
  for (auto EntryAddress = Address; EntryAddress <= UpperBound - EntrySize;
       EntryAddress += EntrySize) {
    // Check if there's a proper relocation against the jump table entry.
    if (HasRelocations) {
      if (Type == JumpTable::JTT_PIC && !DataPCRelocations.count(EntryAddress))
        break;
      if (Type == JumpTable::JTT_NORMAL && !getRelocationAt(EntryAddress))
        break;
    }

    const uint64_t Value = (Type == JumpTable::JTT_PIC)
      ? Address + *getSignedValueAtAddress(EntryAddress, EntrySize)
      : *getPointerAtAddress(EntryAddress);

    // __builtin_unreachable() case.
    if (Value == BF.getAddress() + BF.getSize()) {
      addOffset(Value - BF.getAddress());
      HasUnreachable = true;
      continue;
    }

    // We assume that a jump table cannot have function start as an entry.
    if (!BF.containsAddress(Value) || Value == BF.getAddress())
      break;

    // Check there's an instruction at this offset.
    if (BF.getState() == BinaryFunction::State::Disassembled &&
        !BF.getInstructionAtOffset(Value - BF.getAddress()))
      break;

    addOffset(Value - BF.getAddress());
    ++NumRealEntries;
  }

  // It's a jump table if the number of real entries is more than 1, or there's
  // one real entry and "unreachable" targets. If there are only multiple
  // "unreachable" targets, then it's not a jump table.
  return NumRealEntries + HasUnreachable >= 2;
}

void BinaryContext::populateJumpTables() {
  for (auto JTI = JumpTables.begin(), JTE = JumpTables.end(); JTI != JTE;
       ++JTI) {
    auto *JT = JTI->second;
    auto &BF = *JT->Parent;

    if (!BF.isSimple())
      continue;

    uint64_t NextJTAddress{0};
    auto NextJTI = std::next(JTI);
    if (NextJTI != JTE) {
      NextJTAddress = NextJTI->second->getAddress();
    }

    const auto Success = analyzeJumpTable(JT->getAddress(),
                                          JT->Type,
                                          BF,
                                          NextJTAddress,
                                          &JT->OffsetEntries);
    if (!Success) {
      dbgs() << "failed to analyze jump table in function " << BF << '\n';
      JT->print(dbgs());
      if (NextJTI != JTE) {
        dbgs() << "next jump table at 0x"
               << Twine::utohexstr(NextJTI->second->getAddress())
               << " belongs to function " << *NextJTI->second->Parent << '\n';
        NextJTI->second->print(dbgs());
      }
      llvm_unreachable("jump table heuristic failure");
    }

    for (auto EntryOffset : JT->OffsetEntries) {
      if (EntryOffset == BF.getSize())
        BF.IgnoredBranches.emplace_back(EntryOffset, BF.getSize());
      else
        BF.registerReferencedOffset(EntryOffset);
    }

    // In strict mode, erase PC-relative relocation record. Later we check that
    // all such records are erased and thus have been accounted for.
    if (opts::StrictMode && JT->Type == JumpTable::JTT_PIC) {
      for (auto Address = JT->getAddress();
           Address < JT->getAddress() + JT->getSize();
           Address += JT->EntrySize) {
        DataPCRelocations.erase(DataPCRelocations.find(Address));
      }
    }
  }

  assert((!opts::StrictMode || !DataPCRelocations.size()) &&
         "unclaimed PC-relative relocations left in data\n");
  clearList(DataPCRelocations);
}

MCSymbol *BinaryContext::getOrCreateGlobalSymbol(uint64_t Address,
                                                 Twine Prefix,
                                                 uint64_t Size,
                                                 uint16_t Alignment,
                                                 unsigned Flags) {
  auto Itr = BinaryDataMap.find(Address);
  if (Itr != BinaryDataMap.end()) {
    assert(Itr->second->getSize() == Size || !Size);
    return Itr->second->getSymbol();
  }

  std::string Name = (Prefix + "0x" + Twine::utohexstr(Address)).str();
  assert(!GlobalSymbols.count(Name) && "created name is not unique");
  return registerNameAtAddress(Name, Address, Size, Alignment, Flags);
}

BinaryFunction *BinaryContext::createBinaryFunction(
    const std::string &Name, BinarySection &Section, uint64_t Address,
    uint64_t Size, uint64_t SymbolSize, uint16_t Alignment) {
  auto Result = BinaryFunctions.emplace(
      Address, BinaryFunction(Name, Section, Address, Size, *this));
  assert(Result.second == true && "unexpected duplicate function");
  auto *BF = &Result.first->second;
  registerNameAtAddress(Name, Address, SymbolSize ? SymbolSize : Size,
                        Alignment);
  setSymbolToFunctionMap(BF->getSymbol(), BF);
  return BF;
}

const MCSymbol *
BinaryContext::getOrCreateJumpTable(BinaryFunction &Function, uint64_t Address,
                                    JumpTable::JumpTableType Type) {
  if (auto *JT = getJumpTableContainingAddress(Address)) {
    assert(JT->Type == Type && "jump table types have to match");
    assert(JT->Parent == &Function &&
           "cannot re-use jump table of a different function");
    assert(Address == JT->getAddress() && "unexpected non-empty jump table");

    return JT->getFirstLabel();
  }

  // Re-use the existing symbol if possible.
  MCSymbol *JTLabel{nullptr};
  if (auto *Object = getBinaryDataAtAddress(Address)) {
    if (!isInternalSymbolName(Object->getSymbol()->getName()))
      JTLabel = Object->getSymbol();
  }

  const auto EntrySize = getJumpTableEntrySize(Type);
  if (!JTLabel) {
    const auto JumpTableName = generateJumpTableName(Function, Address);
    JTLabel = registerNameAtAddress(JumpTableName, Address, 0, EntrySize);
  }

  DEBUG(dbgs() << "BOLT-DEBUG: creating jump table "
               << JTLabel->getName()
               << " in function " << Function << 'n');

  auto *JT = new JumpTable(*JTLabel,
                           Address,
                           EntrySize,
                           Type,
                           JumpTable::LabelMapType{{0, JTLabel}},
                           Function,
                           *getSectionForAddress(Address));
  JumpTables.emplace(Address, JT);

  // Duplicate the entry for the parent function for easy access.
  Function.JumpTables.emplace(Address, JT);

  return JTLabel;
}

std::pair<uint64_t, const MCSymbol *>
BinaryContext::duplicateJumpTable(BinaryFunction &Function, JumpTable *JT,
                                  const MCSymbol *OldLabel) {
  auto L = scopeLock();
  unsigned Offset = 0;
  bool Found = false;
  for (auto Elmt : JT->Labels) {
    if (Elmt.second != OldLabel)
      continue;
    Offset = Elmt.first;
    Found = true;
    break;
  }
  assert(Found && "Label not found");
  auto *NewLabel = Ctx->createTempSymbol("duplicatedJT", true);
  auto *NewJT = new JumpTable(*NewLabel,
                              JT->getAddress(),
                              JT->EntrySize,
                              JT->Type,
                              JumpTable::LabelMapType{{Offset, NewLabel}},
                              Function,
                              *getSectionForAddress(JT->getAddress()));
  NewJT->Entries = JT->Entries;
  NewJT->Counts = JT->Counts;
  uint64_t JumpTableID = ++DuplicatedJumpTables;
  // Invert it to differentiate from regular jump tables whose IDs are their
  // addresses in the input binary memory space
  JumpTableID = ~JumpTableID;
  JumpTables.emplace(JumpTableID, NewJT);
  Function.JumpTables.emplace(JumpTableID, NewJT);
  return std::make_pair(JumpTableID, NewLabel);
}

std::string BinaryContext::generateJumpTableName(const BinaryFunction &BF,
                                                 uint64_t Address) {
  size_t Id;
  uint64_t Offset = 0;
  if (const auto *JT = BF.getJumpTableContainingAddress(Address)) {
    Offset = Address - JT->getAddress();
    auto Itr = JT->Labels.find(Offset);
    if (Itr != JT->Labels.end()) {
      return Itr->second->getName();
    }
    Id = JumpTableIds.at(JT->getAddress());
  } else {
    Id = JumpTableIds[Address] = BF.JumpTables.size();
  }
  return ("JUMP_TABLE/" + BF.getOneName().str() + "." + std::to_string(Id) +
          (Offset ? ("." + std::to_string(Offset)) : ""));
}

bool BinaryContext::hasValidCodePadding(const BinaryFunction &BF) {
  // FIXME: aarch64 support is missing.
  if (!isX86())
    return true;

  if (BF.getSize() == BF.getMaxSize())
    return true;

  auto FunctionData = BF.getData();
  assert(FunctionData && "cannot get function as data");

  uint64_t Offset = BF.getSize();
  MCInst Instr;
  uint64_t InstrSize{0};
  uint64_t InstrAddress = BF.getAddress() + Offset;
  using std::placeholders::_1;

  // Skip instructions that satisfy the predicate condition.
  auto skipInstructions = [&](std::function<bool(const MCInst &)> Predicate) {
    const auto StartOffset = Offset;
    for (; Offset < BF.getMaxSize();
         Offset += InstrSize, InstrAddress += InstrSize) {
      if (!DisAsm->getInstruction(Instr,
                                  InstrSize,
                                  FunctionData->slice(Offset),
                                  InstrAddress,
                                  nulls(),
                                  nulls()))
        break;
      if (!Predicate(Instr))
        break;
    }

    return Offset - StartOffset;
  };

  // Skip a sequence of zero bytes.
  auto skipZeros = [&]() {
    const auto StartOffset = Offset;
    for (; Offset < BF.getMaxSize(); ++Offset)
      if ((*FunctionData)[Offset] != 0)
        break;

    return Offset - StartOffset;
  };

  // Accept the whole padding area filled with breakpoints.
  auto isBreakpoint = std::bind(&MCPlusBuilder::isBreakpoint, MIB.get(), _1);
  if (skipInstructions(isBreakpoint) && Offset == BF.getMaxSize())
    return true;

  auto isNoop = std::bind(&MCPlusBuilder::isNoop, MIB.get(), _1);

  // Some functions have a jump to the next function or to the padding area
  // inserted after the body.
  auto isSkipJump = [&](const MCInst &Instr) {
    uint64_t TargetAddress{0};
    if (MIB->isUnconditionalBranch(Instr) &&
        MIB->evaluateBranch(Instr, InstrAddress, InstrSize, TargetAddress)) {
      if (TargetAddress >= InstrAddress + InstrSize &&
          TargetAddress <= BF.getAddress() + BF.getMaxSize()) {
        return true;
      }
    }
    return false;
  };

  // Skip over nops, jumps, and zero padding. Allow interleaving (this happens).
  while (skipInstructions(isNoop) ||
         skipInstructions(isSkipJump) ||
         skipZeros())
    ;

  if (Offset == BF.getMaxSize())
    return true;

  if (opts::Verbosity >= 1) {
    errs() << "BOLT-WARNING: bad padding at address 0x"
           << Twine::utohexstr(BF.getAddress() + BF.getSize())
           << " starting at offset "
           << (Offset - BF.getSize()) << " in function "
           << BF << '\n'
           << FunctionData->slice(BF.getSize(), BF.getMaxSize() - BF.getSize())
           << '\n';
  }

  return false;
}

void BinaryContext::adjustCodePadding() {
  for (auto &BFI : BinaryFunctions) {
    auto &BF = BFI.second;
    if (!shouldEmit(BF))
      continue;

    if (!hasValidCodePadding(BF)) {
      if (HasRelocations) {
        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: function " << BF
                 << " has invalid padding. Ignoring the function.\n";
        }
        BF.setIgnored();
      } else {
        BF.setMaxSize(BF.getSize());
      }
    }
  }
}

MCSymbol *BinaryContext::registerNameAtAddress(StringRef Name,
                                               uint64_t Address,
                                               uint64_t Size,
                                               uint16_t Alignment,
                                               unsigned Flags) {
  // Register the name with MCContext.
  auto *Symbol = Ctx->getOrCreateSymbol(Name);

  auto GAI = BinaryDataMap.find(Address);
  BinaryData *BD;
  if (GAI == BinaryDataMap.end()) {
    auto SectionOrErr = getSectionForAddress(Address);
    auto &Section = SectionOrErr ? SectionOrErr.get() : absoluteSection();
    BD = new BinaryData(*Symbol,
                        Address,
                        Size,
                        Alignment ? Alignment : 1,
                        Section,
                        Flags);
    GAI = BinaryDataMap.emplace(Address, BD).first;
    GlobalSymbols[Name] = BD;
    updateObjectNesting(GAI);
  } else {
    BD = GAI->second;
    if (!BD->hasName(Name)) {
      GlobalSymbols[Name] = BD;
      BD->Symbols.push_back(Symbol);
    }
  }

  return Symbol;
}

const BinaryData *
BinaryContext::getBinaryDataContainingAddressImpl(uint64_t Address) const {
  auto NI = BinaryDataMap.lower_bound(Address);
  auto End = BinaryDataMap.end();
  if ((NI != End && Address == NI->first) ||
      ((NI != BinaryDataMap.begin()) && (NI-- != BinaryDataMap.begin()))) {
    if (NI->second->containsAddress(Address)) {
      return NI->second;
    }

    // If this is a sub-symbol, see if a parent data contains the address.
    auto *BD = NI->second->getParent();
    while (BD) {
      if (BD->containsAddress(Address))
        return BD;
      BD = BD->getParent();
    }
  }
  return nullptr;
}

bool BinaryContext::setBinaryDataSize(uint64_t Address, uint64_t Size) {
  auto NI = BinaryDataMap.find(Address);
  assert(NI != BinaryDataMap.end());
  if (NI == BinaryDataMap.end())
    return false;
  // TODO: it's possible that a jump table starts at the same address
  // as a larger blob of private data.  When we set the size of the
  // jump table, it might be smaller than the total blob size.  In this
  // case we just leave the original size since (currently) it won't really
  // affect anything.  See T26915981.
  assert((!NI->second->Size || NI->second->Size == Size ||
          (NI->second->isJumpTable() && NI->second->Size > Size)) &&
         "can't change the size of a symbol that has already had its "
         "size set");
  if (!NI->second->Size) {
    NI->second->Size = Size;
    updateObjectNesting(NI);
    return true;
  }
  return false;
}

void BinaryContext::generateSymbolHashes() {
  auto isPadding = [](const BinaryData &BD) {
    auto Contents = BD.getSection().getContents();
    auto SymData = Contents.substr(BD.getOffset(), BD.getSize());
    return (BD.getName().startswith("HOLEat") ||
            SymData.find_first_not_of(0) == StringRef::npos);
  };

  uint64_t NumCollisions = 0;
  for (auto &Entry : BinaryDataMap) {
    auto &BD = *Entry.second;
    auto Name = BD.getName();

    if (!isInternalSymbolName(Name))
      continue;

    // First check if a non-anonymous alias exists and move it to the front.
    if (BD.getSymbols().size() > 1) {
      auto Itr = std::find_if(BD.getSymbols().begin(),
                              BD.getSymbols().end(),
                              [&](const MCSymbol *Symbol) {
                                return !isInternalSymbolName(Symbol->getName());
                              });
      if (Itr != BD.getSymbols().end()) {
        auto Idx = std::distance(BD.getSymbols().begin(), Itr);
        std::swap(BD.getSymbols()[0], BD.getSymbols()[Idx]);
        continue;
      }
    }

    // We have to skip 0 size symbols since they will all collide.
    if (BD.getSize() == 0) {
      continue;
    }

    const auto Hash = BD.getSection().hash(BD);
    const auto Idx = Name.find("0x");
    std::string NewName = (Twine(Name.substr(0, Idx)) +
                 "_" + Twine::utohexstr(Hash)).str();
    if (getBinaryDataByName(NewName)) {
      // Ignore collisions for symbols that appear to be padding
      // (i.e. all zeros or a "hole")
      if (!isPadding(BD)) {
        if (opts::Verbosity) {
          errs() << "BOLT-WARNING: collision detected when hashing " << BD
                 << " with new name (" << NewName << "), skipping.\n";
        }
        ++NumCollisions;
      }
      continue;
    }
    BD.Symbols.insert(BD.Symbols.begin(),
                       Ctx->getOrCreateSymbol(NewName));
    GlobalSymbols[NewName] = &BD;
  }
  if (NumCollisions) {
    errs() << "BOLT-WARNING: " << NumCollisions
           << " collisions detected while hashing binary objects";
    if (!opts::Verbosity)
      errs() << ". Use -v=1 to see the list.";
    errs() << '\n';
  }
}

void BinaryContext::processInterproceduralReferences(BinaryFunction &Function) {
  for (auto Address : Function.InterproceduralReferences) {
    auto *ContainingFunction = getBinaryFunctionContainingAddress(Address);
    if (&Function == ContainingFunction)
      continue;

    if (ContainingFunction) {
      // Only a parent function (or a sibling) can reach its fragment.
      if (ContainingFunction->IsFragment) {
        assert(!Function.IsFragment &&
               "only one cold fragment is supported at this time");
        ContainingFunction->setParentFunction(&Function);
        Function.addFragment(ContainingFunction);
        if (!HasRelocations) {
          ContainingFunction->setSimple(false);
          Function.setSimple(false);
        }
        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: marking " << *ContainingFunction
                 << " as a fragment of " << Function << '\n';
        }
        continue;
      }

      if (ContainingFunction->getAddress() != Address) {
        ContainingFunction->
          addEntryPointAtOffset(Address - ContainingFunction->getAddress());
      }
    } else if (Address) {
      // Check if address falls in function padding space - this could be
      // unmarked data in code. In this case adjust the padding space size.
      auto Section = getSectionForAddress(Address);
      assert(Section && "cannot get section for referenced address");

      if (!Section->isText())
        continue;

      // PLT requires special handling and could be ignored in this context.
      StringRef SectionName = Section->getName();
      if (SectionName == ".plt" || SectionName == ".plt.got")
        continue;

      if (opts::UseOldText) {
        errs() << "BOLT-ERROR: cannot process binaries with unmarked "
               << "object in code at address 0x"
               << Twine::utohexstr(Address) << " belonging to section "
               << SectionName << " in relocation mode.\n";
        exit(1);
      }

      ContainingFunction =
        getBinaryFunctionContainingAddress(Address,
                                           /*CheckPastEnd=*/false,
                                           /*UseMaxSize=*/true);
      // We are not going to overwrite non-simple functions, but for simple
      // ones - adjust the padding size.
      if (ContainingFunction && ContainingFunction->isSimple()) {
        errs() << "BOLT-WARNING: function " << *ContainingFunction
               << " has an object detected in a padding region at address 0x"
               << Twine::utohexstr(Address) << '\n';
        ContainingFunction->setMaxSize(ContainingFunction->getSize());
      }
    }
  }

  clearList(Function.InterproceduralReferences);
}

void BinaryContext::postProcessSymbolTable() {
  fixBinaryDataHoles();
  bool Valid = true;
  for (auto &Entry : BinaryDataMap) {
    auto *BD = Entry.second;
    if ((BD->getName().startswith("SYMBOLat") ||
         BD->getName().startswith("DATAat")) &&
        !BD->getParent() &&
        !BD->getSize() &&
        !BD->isAbsolute() &&
        BD->getSection()) {
      errs() << "BOLT-WARNING: zero-sized top level symbol: " << *BD << "\n";
      Valid = false;
    }
  }
  assert(Valid);
  generateSymbolHashes();
}

void BinaryContext::foldFunction(BinaryFunction &ChildBF,
                                 BinaryFunction &ParentBF) {
  assert(!ChildBF.isMultiEntry() && !ParentBF.isMultiEntry() &&
         "cannot merge functions with multiple entry points");

  std::unique_lock<std::shared_timed_mutex> WriteCtxLock(CtxMutex,
                                                         std::defer_lock);
  std::unique_lock<std::shared_timed_mutex> WriteSymbolMapLock(
      SymbolToFunctionMapMutex, std::defer_lock);

  const auto ChildName = ChildBF.getOneName();

  // Move symbols over and update bookkeeping info.
  for (auto *Symbol : ChildBF.getSymbols()) {
    ParentBF.getSymbols().push_back(Symbol);
    WriteSymbolMapLock.lock();
    SymbolToFunctionMap[Symbol] = &ParentBF;
    WriteSymbolMapLock.unlock();
    // NB: there's no need to update BinaryDataMap and GlobalSymbols.
  }
  ChildBF.getSymbols().clear();

  // Move other names the child function is known under.
  std::move(ChildBF.Aliases.begin(), ChildBF.Aliases.end(),
            std::back_inserter(ParentBF.Aliases));
  ChildBF.Aliases.clear();

  if (HasRelocations) {
    // Merge execution counts of ChildBF into those of ParentBF.
    // Without relocations, we cannot reliably merge profiles as both functions
    // continue to exist and either one can be executed.
    ChildBF.mergeProfileDataInto(ParentBF);

    std::shared_lock<std::shared_timed_mutex> ReadBfsLock(BinaryFunctionsMutex,
                                                          std::defer_lock);
    std::unique_lock<std::shared_timed_mutex> WriteBfsLock(BinaryFunctionsMutex,
                                                           std::defer_lock);
    // Remove ChildBF from the global set of functions in relocs mode.
    ReadBfsLock.lock();
    auto FI = BinaryFunctions.find(ChildBF.getAddress());
    ReadBfsLock.unlock();

    assert(FI != BinaryFunctions.end() && "function not found");
    assert(&ChildBF == &FI->second && "function mismatch");

    WriteBfsLock.lock();
    FI = BinaryFunctions.erase(FI);
    WriteBfsLock.unlock();

  } else {
    // In non-relocation mode we keep the function, but rename it.
    std::string NewName = "__ICF_" + ChildName.str();

    WriteCtxLock.lock();
    ChildBF.getSymbols().push_back(Ctx->getOrCreateSymbol(NewName));
    WriteCtxLock.unlock();

    ChildBF.setFolded(&ParentBF);
  }
}

void BinaryContext::fixBinaryDataHoles() {
  assert(validateObjectNesting() && "object nesting inconsitency detected");

  for (auto &Section : allocatableSections()) {
    std::vector<std::pair<uint64_t, uint64_t>> Holes;

    auto isNotHole = [&Section](const binary_data_iterator &Itr) {
      auto *BD = Itr->second;
      bool isHole = (!BD->getParent() &&
                     !BD->getSize() &&
                     BD->isObject() &&
                     (BD->getName().startswith("SYMBOLat0x") ||
                      BD->getName().startswith("DATAat0x") ||
                      BD->getName().startswith("ANONYMOUS")));
      return !isHole && BD->getSection() == Section && !BD->getParent();
    };

    auto BDStart = BinaryDataMap.begin();
    auto BDEnd = BinaryDataMap.end();
    auto Itr = FilteredBinaryDataIterator(isNotHole, BDStart, BDEnd);
    auto End = FilteredBinaryDataIterator(isNotHole, BDEnd, BDEnd);

    uint64_t EndAddress = Section.getAddress();

    while (Itr != End) {
      if (Itr->second->getAddress() > EndAddress) {
        auto Gap = Itr->second->getAddress() - EndAddress;
        Holes.push_back(std::make_pair(EndAddress, Gap));
      }
      EndAddress = Itr->second->getEndAddress();
      ++Itr;
    }

    if (EndAddress < Section.getEndAddress()) {
      Holes.push_back(std::make_pair(EndAddress,
                                     Section.getEndAddress() - EndAddress));
    }

    // If there is already a symbol at the start of the hole, grow that symbol
    // to cover the rest.  Otherwise, create a new symbol to cover the hole.
    for (auto &Hole : Holes) {
      auto *BD = getBinaryDataAtAddress(Hole.first);
      if (BD) {
        // BD->getSection() can be != Section if there are sections that
        // overlap.  In this case it is probably safe to just skip the holes
        // since the overlapping section will not(?) have any symbols in it.
        if (BD->getSection() == Section)
          setBinaryDataSize(Hole.first, Hole.second);
      } else {
        getOrCreateGlobalSymbol(Hole.first, "HOLEat", Hole.second, 1);
      }
    }
  }

  assert(validateObjectNesting() && "object nesting inconsitency detected");
  assert(validateHoles() && "top level hole detected in object map");
}

void BinaryContext::printGlobalSymbols(raw_ostream& OS) const {
  const BinarySection* CurrentSection = nullptr;
  bool FirstSection = true;

  for (auto &Entry : BinaryDataMap) {
    const auto *BD = Entry.second;
    const auto &Section = BD->getSection();
    if (FirstSection || Section != *CurrentSection) {
      uint64_t Address, Size;
      StringRef Name = Section.getName();
      if (Section) {
        Address = Section.getAddress();
        Size = Section.getSize();
      } else {
        Address = BD->getAddress();
        Size = BD->getSize();
      }
      OS << "BOLT-INFO: Section " << Name << ", "
         << "0x" + Twine::utohexstr(Address) << ":"
         << "0x" + Twine::utohexstr(Address + Size) << "/"
         << Size << "\n";
      CurrentSection = &Section;
      FirstSection = false;
    }

    OS << "BOLT-INFO: ";
    auto *P = BD->getParent();
    while (P) {
      OS << "  ";
      P = P->getParent();
    }
    OS << *BD << "\n";
  }
}

namespace {

/// Recursively finds DWARF DW_TAG_subprogram DIEs and match them with
/// BinaryFunctions.
void findSubprograms(const DWARFDie DIE,
                     std::map<uint64_t, BinaryFunction> &BinaryFunctions) {
  if (DIE.isSubprogramDIE()) {
    uint64_t LowPC, HighPC, SectionIndex;
    if (DIE.getLowAndHighPC(LowPC, HighPC, SectionIndex)) {
      auto It = BinaryFunctions.find(LowPC);
      if (It != BinaryFunctions.end()) {
          It->second.addSubprogramDIE(DIE);
      } else {
        // The function must have been optimized away by GC.
      }
    } else {
      const auto RangesVector = DIE.getAddressRanges();
      for (const auto Range : DIE.getAddressRanges()) {
        auto It = BinaryFunctions.find(Range.LowPC);
        if (It != BinaryFunctions.end()) {
            It->second.addSubprogramDIE(DIE);
        }
      }
    }
  }

  for (auto ChildDIE = DIE.getFirstChild(); ChildDIE && !ChildDIE.isNULL();
       ChildDIE = ChildDIE.getSibling()) {
    findSubprograms(ChildDIE, BinaryFunctions);
  }
}

} // namespace

unsigned BinaryContext::addDebugFilenameToUnit(const uint32_t DestCUID,
                                               const uint32_t SrcCUID,
                                               unsigned FileIndex) {
  auto SrcUnit = DwCtx->getCompileUnitForOffset(SrcCUID);
  auto LineTable = DwCtx->getLineTableForUnit(SrcUnit);
  const auto &FileNames = LineTable->Prologue.FileNames;
  // Dir indexes start at 1, as DWARF file numbers, and a dir index 0
  // means empty dir.
  assert(FileIndex > 0 && FileIndex <= FileNames.size() &&
         "FileIndex out of range for the compilation unit.");
  StringRef Dir = "";
  if (FileNames[FileIndex - 1].DirIdx != 0) {
    if (auto DirName =
            LineTable->Prologue
                .IncludeDirectories[FileNames[FileIndex - 1].DirIdx - 1]
                .getAsCString()) {
      Dir = *DirName;
    }
  }
  StringRef FileName = "";
  if (auto FName = FileNames[FileIndex - 1].Name.getAsCString())
    FileName = *FName;
  assert(FileName != "");
  return cantFail(Ctx->getDwarfFile(Dir, FileName, 0, nullptr, None, DestCUID));
}

std::vector<BinaryFunction *> BinaryContext::getSortedFunctions() {
  std::vector<BinaryFunction *> SortedFunctions(BinaryFunctions.size());
  std::transform(BinaryFunctions.begin(), BinaryFunctions.end(),
                 SortedFunctions.begin(),
                 [](std::pair<const uint64_t, BinaryFunction> &BFI) {
                   return &BFI.second;
                 });

  std::stable_sort(SortedFunctions.begin(), SortedFunctions.end(),
                   [] (const BinaryFunction *A, const BinaryFunction *B) {
                     if (A->hasValidIndex() && B->hasValidIndex()) {
                       return A->getIndex() < B->getIndex();
                     }
                     return A->hasValidIndex();
                   });
  return SortedFunctions;
}

void BinaryContext::preprocessDebugInfo() {
  // Populate MCContext with DWARF files.
  for (const auto &CU : DwCtx->compile_units()) {
    const auto CUID = CU->getOffset();
    auto *LineTable = DwCtx->getLineTableForUnit(CU.get());
    const auto &FileNames = LineTable->Prologue.FileNames;
    // Make sure empty debug line tables are registered too.
    if (FileNames.empty()) {
      cantFail(Ctx->getDwarfFile("", "<unknown>", 0, nullptr, None, CUID));
      continue;
    }
    for (size_t I = 0, Size = FileNames.size(); I != Size; ++I) {
      // Dir indexes start at 1, as DWARF file numbers, and a dir index 0
      // means empty dir.
      StringRef Dir = "";
      if (FileNames[I].DirIdx != 0)
        if (auto DirName =
                LineTable->Prologue.IncludeDirectories[FileNames[I].DirIdx - 1]
                    .getAsCString())
          Dir = *DirName;
      StringRef FileName = "";
      if (auto FName = FileNames[I].Name.getAsCString())
        FileName = *FName;
      assert(FileName != "");
      cantFail(Ctx->getDwarfFile(Dir, FileName, 0, nullptr, None, CUID));
    }
  }

  // For each CU, iterate over its children DIEs and match subprogram DIEs to
  // BinaryFunctions.

  // Run findSubprograms on a range of compilation units
  auto processBlock = [&](auto BlockBegin, auto BlockEnd) {
    for (auto It = BlockBegin; It != BlockEnd; ++It) {
      findSubprograms((*It)->getUnitDIE(false), BinaryFunctions);
    }
  };

  if (opts::NoThreads) {
    processBlock(DwCtx->compile_units().begin(), DwCtx->compile_units().end());
  } else {
    auto &ThreadPool = ParallelUtilities::getThreadPool();

    // Divide compilation units uniformally into tasks.
    unsigned BlockCost =
        DwCtx->getNumCompileUnits() / (opts::TaskCount * opts::ThreadCount);
    if (BlockCost == 0)
      BlockCost = 1;

    auto BlockBegin = DwCtx->compile_units().begin();
    unsigned CurrentCost = 0;
    for (auto It = DwCtx->compile_units().begin();
         It != DwCtx->compile_units().end(); It++) {
      CurrentCost++;
      if (CurrentCost >= BlockCost) {
        ThreadPool.async(processBlock, BlockBegin, std::next(It));
        BlockBegin = std::next(It);
        CurrentCost = 0;
      }
    }

    ThreadPool.async(processBlock, BlockBegin, DwCtx->compile_units().end());
    ThreadPool.wait();
  }

  for (auto &KV : BinaryFunctions) {
    const auto FunctionAddress = KV.first;
    auto &Function = KV.second;

    // Sort associated CUs for deterministic update.
    std::sort(Function.getSubprogramDIEs().begin(),
              Function.getSubprogramDIEs().end(),
              [](const DWARFDie &A, const DWARFDie &B) {
                return A.getDwarfUnit()->getOffset() <
                       B.getDwarfUnit()->getOffset();
              });

    // Some functions may not have a corresponding subprogram DIE
    // yet they will be included in some CU and will have line number
    // information. Hence we need to associate them with the CU and include
    // in CU ranges.
    if (Function.getSubprogramDIEs().empty()) {
      if (auto DebugAranges = DwCtx->getDebugAranges()) {
        auto CUOffset = DebugAranges->findAddress(FunctionAddress);
        if (CUOffset != -1U) {
          Function.addSubprogramDIE(
              DWARFDie(DwCtx->getCompileUnitForOffset(CUOffset), nullptr));
        }
      }
    }

#ifdef DWARF_LOOKUP_ALL_RANGES
    if (Function.getSubprogramDIEs().empty()) {
      // Last resort - iterate over all compile units. This should not happen
      // very often. If it does, we need to create a separate lookup table
      // similar to .debug_aranges internally. This slows down processing
      // considerably.
      for (const auto &CU : DwCtx->compile_units()) {
        const auto *CUDie = CU->getUnitDIE();
        for (const auto &Range : CUDie->getAddressRanges(CU.get())) {
          if (FunctionAddress >= Range.first &&
              FunctionAddress < Range.second) {
            Function.addSubprogramDIE(DWARFDie(CU.get(), nullptr));
            break;
          }
        }
      }
    }
#endif

    // Set line table for function to the first CU with such table.
    for (const auto &DIE : Function.getSubprogramDIEs()) {
      if (const auto *LineTable =
              DwCtx->getLineTableForUnit(DIE.getDwarfUnit())) {
        Function.setDWARFUnitLineTable(DIE.getDwarfUnit(), LineTable);
        break;
      }
    }

  }
}

bool BinaryContext::shouldEmit(const BinaryFunction &Function) const {
  if (opts::processAllFunctions())
    return true;

  if (Function.isIgnored())
    return false;

  // In relocation mode we will emit non-simple functions with CFG.
  // If the function does not have a CFG it should be marked as ignored.
  return HasRelocations || Function.isSimple();
}

void BinaryContext::printCFI(raw_ostream &OS, const MCCFIInstruction &Inst) {
  uint32_t Operation = Inst.getOperation();
  switch (Operation) {
  case MCCFIInstruction::OpSameValue:
    OS << "OpSameValue Reg" << Inst.getRegister();
    break;
  case MCCFIInstruction::OpRememberState:
    OS << "OpRememberState";
    break;
  case MCCFIInstruction::OpRestoreState:
    OS << "OpRestoreState";
    break;
  case MCCFIInstruction::OpOffset:
    OS << "OpOffset Reg" << Inst.getRegister() << " " << Inst.getOffset();
    break;
  case MCCFIInstruction::OpDefCfaRegister:
    OS << "OpDefCfaRegister Reg" << Inst.getRegister();
    break;
  case MCCFIInstruction::OpDefCfaOffset:
    OS << "OpDefCfaOffset " << Inst.getOffset();
    break;
  case MCCFIInstruction::OpDefCfa:
    OS << "OpDefCfa Reg" << Inst.getRegister() << " " << Inst.getOffset();
    break;
  case MCCFIInstruction::OpRelOffset:
    OS << "OpRelOffset Reg" << Inst.getRegister() << " " << Inst.getOffset();
    break;
  case MCCFIInstruction::OpAdjustCfaOffset:
    OS << "OfAdjustCfaOffset " << Inst.getOffset();
    break;
  case MCCFIInstruction::OpEscape:
    OS << "OpEscape";
    break;
  case MCCFIInstruction::OpRestore:
    OS << "OpRestore Reg" << Inst.getRegister();
    break;
  case MCCFIInstruction::OpUndefined:
    OS << "OpUndefined Reg" << Inst.getRegister();
    break;
  case MCCFIInstruction::OpRegister:
    OS << "OpRegister Reg" << Inst.getRegister() << " Reg"
       << Inst.getRegister2();
    break;
  case MCCFIInstruction::OpWindowSave:
    OS << "OpWindowSave";
    break;
  case MCCFIInstruction::OpGnuArgsSize:
    OS << "OpGnuArgsSize";
    break;
  default:
    OS << "Op#" << Operation;
    break;
  }
}

void BinaryContext::printInstruction(raw_ostream &OS,
                                     const MCInst &Instruction,
                                     uint64_t Offset,
                                     const BinaryFunction* Function,
                                     bool PrintMCInst,
                                     bool PrintMemData,
                                     bool PrintRelocations) const {
  if (MIB->isEHLabel(Instruction)) {
    OS << "  EH_LABEL: " << *MIB->getTargetSymbol(Instruction) << '\n';
    return;
  }
  OS << format("    %08" PRIx64 ": ", Offset);
  if (MIB->isCFI(Instruction)) {
    uint32_t Offset = Instruction.getOperand(0).getImm();
    OS << "\t!CFI\t$" << Offset << "\t; ";
    if (Function)
      printCFI(OS, *Function->getCFIFor(Instruction));
    OS << "\n";
    return;
  }
  InstPrinter->printInst(&Instruction, OS, "", *STI);
  if (MIB->isCall(Instruction)) {
    if (MIB->isTailCall(Instruction))
      OS << " # TAILCALL ";
    if (MIB->isInvoke(Instruction)) {
      const auto EHInfo = MIB->getEHInfo(Instruction);
      OS << " # handler: ";
      if (EHInfo->first)
        OS << *EHInfo->first;
      else
        OS << '0';
      OS << "; action: " << EHInfo->second;
      const auto GnuArgsSize = MIB->getGnuArgsSize(Instruction);
      if (GnuArgsSize >= 0)
        OS << "; GNU_args_size = " << GnuArgsSize;
    }
  } else if (MIB->isIndirectBranch(Instruction)) {
    if (auto JTAddress = MIB->getJumpTable(Instruction)) {
      OS << " # JUMPTABLE @0x" << Twine::utohexstr(JTAddress);
    } else {
      OS << " # UNKNOWN CONTROL FLOW";
    }
  }

  MIB->printAnnotations(Instruction, OS);

  const DWARFDebugLine::LineTable *LineTable =
    Function && opts::PrintDebugInfo ? Function->getDWARFUnitLineTable().second
                                     : nullptr;

  if (LineTable) {
    auto RowRef = DebugLineTableRowRef::fromSMLoc(Instruction.getLoc());

    if (RowRef != DebugLineTableRowRef::NULL_ROW) {
      const auto &Row = LineTable->Rows[RowRef.RowIndex - 1];
      StringRef FileName = "";
      if (auto FName =
              LineTable->Prologue.FileNames[Row.File - 1].Name.getAsCString())
        FileName = *FName;
      OS << " # debug line " << FileName << ":" << Row.Line;

      if (Row.Column) {
        OS << ":" << Row.Column;
      }
    }
  }

  if ((opts::PrintRelocations || PrintRelocations) && Function) {
    const auto Size = computeCodeSize(&Instruction, &Instruction + 1);
    Function->printRelocations(OS, Offset, Size);
  }

  OS << "\n";

  if (PrintMCInst) {
    Instruction.dump_pretty(OS, InstPrinter.get());
    OS << "\n";
  }
}

ErrorOr<BinarySection&> BinaryContext::getSectionForAddress(uint64_t Address) {
  auto SI = AddressToSection.upper_bound(Address);
  if (SI != AddressToSection.begin()) {
    --SI;
    auto UpperBound = SI->first + SI->second->getSize();
    if (!SI->second->getSize())
      UpperBound += 1;
    if (UpperBound > Address)
      return *SI->second;
  }
  return std::make_error_code(std::errc::bad_address);
}

ErrorOr<StringRef>
BinaryContext::getSectionNameForAddress(uint64_t Address) const {
  if (auto Section = getSectionForAddress(Address)) {
    return Section->getName();
  }
  return std::make_error_code(std::errc::bad_address);
}

BinarySection &BinaryContext::registerSection(BinarySection *Section) {
  assert(!Section->getName().empty() &&
         "can't register sections without a name");
  auto Res = Sections.insert(Section);
  assert(Res.second && "can't register the same section twice.");
  // Only register sections with addresses in the AddressToSection map.
  if (Section->getAddress())
    AddressToSection.insert(std::make_pair(Section->getAddress(), Section));
  NameToSection.insert(std::make_pair(Section->getName(), Section));
  DEBUG(dbgs() << "BOLT-DEBUG: registering " << *Section << "\n");
  return *Section;
}

BinarySection &BinaryContext::registerSection(SectionRef Section) {
  return registerSection(new BinarySection(*this, Section));
}

BinarySection &
BinaryContext::registerSection(StringRef SectionName,
                               const BinarySection &OriginalSection) {
  return registerSection(new BinarySection(*this,
                                           SectionName,
                                           OriginalSection));
}

BinarySection &BinaryContext::registerOrUpdateSection(StringRef Name,
                                                      unsigned ELFType,
                                                      unsigned ELFFlags,
                                                      uint8_t *Data,
                                                      uint64_t Size,
                                                      unsigned Alignment) {
  auto NamedSections = getSectionByName(Name);
  if (NamedSections.begin() != NamedSections.end()) {
    assert(std::next(NamedSections.begin()) == NamedSections.end() &&
           "can only update unique sections");
    auto *Section = NamedSections.begin()->second;

    DEBUG(dbgs() << "BOLT-DEBUG: updating " << *Section << " -> ");
    const auto Flag = Section->isAllocatable();
    Section->update(Data, Size, Alignment, ELFType, ELFFlags);
    DEBUG(dbgs() << *Section << "\n");
    // FIXME: Fix section flags/attributes for MachO.
    if (isELF())
      assert(Flag == Section->isAllocatable() &&
             "can't change section allocation status");
    return *Section;
  }

  return registerSection(new BinarySection(*this, Name, Data, Size, Alignment,
                                           ELFType, ELFFlags));
}

bool BinaryContext::deregisterSection(BinarySection &Section) {
  auto *SectionPtr = &Section;
  auto Itr = Sections.find(SectionPtr);
  if (Itr != Sections.end()) {
    auto Range = AddressToSection.equal_range(SectionPtr->getAddress());
    while (Range.first != Range.second) {
      if (Range.first->second == SectionPtr) {
        AddressToSection.erase(Range.first);
        break;
      }
      ++Range.first;
    }

    auto NameRange = NameToSection.equal_range(SectionPtr->getName());
    while (NameRange.first != NameRange.second) {
      if (NameRange.first->second == SectionPtr) {
        NameToSection.erase(NameRange.first);
        break;
      }
      ++NameRange.first;
    }

    Sections.erase(Itr);
    delete SectionPtr;
    return true;
  }
  return false;
}

void BinaryContext::printSections(raw_ostream &OS) const {
  for (auto &Section : Sections) {
    OS << "BOLT-INFO: " << *Section << "\n";
  }
}

BinarySection &BinaryContext::absoluteSection() {
  if (auto Section = getUniqueSectionByName("<absolute>"))
    return *Section;
  return registerOrUpdateSection("<absolute>", ELF::SHT_NULL, 0u);
}

ErrorOr<uint64_t>
BinaryContext::getUnsignedValueAtAddress(uint64_t Address,
                                         size_t Size) const {
  const auto Section = getSectionForAddress(Address);
  if (!Section)
    return std::make_error_code(std::errc::bad_address);

  if (Section->isVirtual())
    return 0;

  DataExtractor DE(Section->getContents(), AsmInfo->isLittleEndian(),
                   AsmInfo->getCodePointerSize());
  auto ValueOffset = static_cast<uint32_t>(Address - Section->getAddress());
  return DE.getUnsigned(&ValueOffset, Size);
}

ErrorOr<uint64_t>
BinaryContext::getSignedValueAtAddress(uint64_t Address,
                                       size_t Size) const {
  const auto Section = getSectionForAddress(Address);
  if (!Section)
    return std::make_error_code(std::errc::bad_address);

  if (Section->isVirtual())
    return 0;

  DataExtractor DE(Section->getContents(), AsmInfo->isLittleEndian(),
                   AsmInfo->getCodePointerSize());
  auto ValueOffset = static_cast<uint32_t>(Address - Section->getAddress());
  return DE.getSigned(&ValueOffset, Size);
}

void BinaryContext::addRelocation(uint64_t Address,
                                  MCSymbol *Symbol,
                                  uint64_t Type,
                                  uint64_t Addend,
                                  uint64_t Value) {
  auto Section = getSectionForAddress(Address);
  assert(Section && "cannot find section for address");
  Section->addRelocation(Address - Section->getAddress(),
                         Symbol,
                         Type,
                         Addend,
                         Value);
}

bool BinaryContext::removeRelocationAt(uint64_t Address) {
  auto Section = getSectionForAddress(Address);
  assert(Section && "cannot find section for address");
  return Section->removeRelocationAt(Address - Section->getAddress());
}

const Relocation *BinaryContext::getRelocationAt(uint64_t Address) {
  auto Section = getSectionForAddress(Address);
  if (!Section)
    return nullptr;

  return Section->getRelocationAt(Address - Section->getAddress());
}

void BinaryContext::markAmbiguousRelocations(BinaryData &BD,
                                             const uint64_t Address) {
  auto setImmovable = [&](BinaryData &BD) {
    auto *Root = BD.getAtomicRoot();
    DEBUG(if (Root->isMoveable()) {
      dbgs() << "BOLT-DEBUG: setting " << *Root << " as immovable "
             << "due to ambiguous relocation referencing 0x"
             << Twine::utohexstr(Address) << '\n';
    });
    Root->setIsMoveable(false);
  };

  if (Address == BD.getAddress()) {
    setImmovable(BD);

    // Set previous symbol as immovable
    auto *Prev = getBinaryDataContainingAddress(Address-1);
    if (Prev && Prev->getEndAddress() == BD.getAddress())
      setImmovable(*Prev);
  }

  if (Address == BD.getEndAddress()) {
    setImmovable(BD);

    // Set next symbol as immovable
    auto *Next = getBinaryDataContainingAddress(BD.getEndAddress());
    if (Next && Next->getAddress() == BD.getEndAddress())
      setImmovable(*Next);
  }
}

BinaryFunction *BinaryContext::getFunctionForSymbol(const MCSymbol *Symbol,
                                                    uint64_t *EntryDesc) {
  std::shared_lock<std::shared_timed_mutex> Lock(SymbolToFunctionMapMutex);
  auto BFI = SymbolToFunctionMap.find(Symbol);
  if (BFI == SymbolToFunctionMap.end())
    return nullptr;

  auto *BF = BFI->second;
  if (EntryDesc)
    *EntryDesc = BF->getEntryIDForSymbol(Symbol);

  return BF;
}

void BinaryContext::exitWithBugReport(StringRef Message,
                                      const BinaryFunction &Function) const {
  errs() << "=======================================\n";
  errs() << "BOLT is unable to proceed because it couldn't properly understand "
            "this function.\n";
  errs() << "If you are running the most recent version of BOLT, you may "
            "want to "
            "report this and paste this dump.\nPlease check that there is no "
            "sensitive contents being shared in this dump.\n";
  errs() << "\nOffending function: " << Function.getPrintName() << "\n\n";
  ScopedPrinter SP(errs());
  SP.printBinaryBlock("Function contents", *Function.getData());
  errs() << "\n";
  Function.dump();
  errs() << "ERROR: " << Message;
  errs() << "\n=======================================\n";
  exit(1);
}

BinaryFunction *
BinaryContext::createInjectedBinaryFunction(const std::string &Name,
                                            bool IsSimple) {
  InjectedBinaryFunctions.push_back(new BinaryFunction(Name, *this, IsSimple));
  auto *BF = InjectedBinaryFunctions.back();
  setSymbolToFunctionMap(BF->getSymbol(), BF);
  BF->CurrentState = BinaryFunction::State::CFG;
  return BF;
}

std::pair<size_t, size_t>
BinaryContext::calculateEmittedSize(BinaryFunction &BF, bool FixBranches) {
  // Adjust branch instruction to match the current layout.
  if (FixBranches)
    BF.fixBranches();

  // Create local MC context to isolate the effect of ephemeral code emission.
  auto MCEInstance = createIndependentMCCodeEmitter();
  auto *LocalCtx = MCEInstance.LocalCtx.get();
  auto *MAB = TheTarget->createMCAsmBackend(*STI, *MRI, MCTargetOptions());

  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);

  std::unique_ptr<MCStreamer> Streamer(TheTarget->createMCObjectStreamer(
      *TheTriple, *LocalCtx, std::unique_ptr<MCAsmBackend>(MAB), VecOS,
      std::unique_ptr<MCCodeEmitter>(MCEInstance.MCE.release()), *STI,
      /*RelaxAll=*/false,
      /*IncrementalLinkerCompatible=*/false,
      /*DWARFMustBeAtTheEnd=*/false));

  Streamer->InitSections(false);

  auto *Section = MCEInstance.LocalMOFI->getTextSection();
  Section->setHasInstructions(true);

  auto *StartLabel = LocalCtx->getOrCreateSymbol("__hstart");
  auto *EndLabel = LocalCtx->getOrCreateSymbol("__hend");
  auto *ColdStartLabel = LocalCtx->getOrCreateSymbol("__cstart");
  auto *ColdEndLabel = LocalCtx->getOrCreateSymbol("__cend");

  Streamer->SwitchSection(Section);
  Streamer->EmitLabel(StartLabel);
  emitFunctionBody(*Streamer, BF, /*EmitColdPart=*/false,
                   /*EmitCodeOnly=*/true);
  Streamer->EmitLabel(EndLabel);

  if (BF.isSplit()) {
    auto *ColdSection =
      LocalCtx->getELFSection(BF.getColdCodeSectionName(),
                              ELF::SHT_PROGBITS,
                              ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);
    ColdSection->setHasInstructions(true);

    Streamer->SwitchSection(ColdSection);
    Streamer->EmitLabel(ColdStartLabel);
    emitFunctionBody(*Streamer, BF, /*EmitColdPart=*/true,
                     /*EmitCodeOnly=*/true);
    Streamer->EmitLabel(ColdEndLabel);
  }

  // To avoid calling MCObjectStreamer::flushPendingLabels() which is private.
  Streamer->EmitBytes(StringRef(""));

  auto &Assembler =
      static_cast<MCObjectStreamer *>(Streamer.get())->getAssembler();
  MCAsmLayout Layout(Assembler);
  Assembler.layout(Layout);

  const auto HotSize = Layout.getSymbolOffset(*EndLabel) -
                       Layout.getSymbolOffset(*StartLabel);
  const auto ColdSize = BF.isSplit() ? Layout.getSymbolOffset(*ColdEndLabel) -
                                       Layout.getSymbolOffset(*ColdStartLabel)
                                     : 0ULL;

  // Clean-up the effect of the code emission.
  for (const auto &Symbol : Assembler.symbols()) {
    auto *MutableSymbol = const_cast<MCSymbol *>(&Symbol);
    MutableSymbol->setUndefined();
    MutableSymbol->setIsRegistered(false);
  }

  return std::make_pair(HotSize, ColdSize);
}

bool BinaryContext::validateEncoding(const MCInst &Inst,
                                     ArrayRef<uint8_t> InputEncoding) const {
  SmallString<256> Code;
  SmallVector<MCFixup, 4> Fixups;
  raw_svector_ostream VecOS(Code);

  MCE->encodeInstruction(Inst, VecOS, Fixups, *STI);
  auto EncodedData = ArrayRef<uint8_t>((uint8_t *)Code.data(), Code.size());
  if (InputEncoding != EncodedData) {
    if (opts::Verbosity > 1) {
      errs() << "BOLT-WARNING: mismatched encoding detected\n"
             << "      input: " << InputEncoding << '\n'
             << "     output: " << EncodedData << '\n';
    }
    return false;
  }

  return true;
}

BinaryFunction *
BinaryContext::getBinaryFunctionContainingAddress(uint64_t Address,
                                                  bool CheckPastEnd,
                                                  bool UseMaxSize,
                                                  bool Shallow) {
  auto FI = BinaryFunctions.upper_bound(Address);
  if (FI == BinaryFunctions.begin())
    return nullptr;
  --FI;

  const auto UsedSize = UseMaxSize ? FI->second.getMaxSize()
                                   : FI->second.getSize();

  if (Address >= FI->first + UsedSize + (CheckPastEnd ? 1 : 0))
    return nullptr;

  auto *BF = &FI->second;
  if (Shallow)
    return BF;

  while (BF->getParentFunction())
    BF = BF->getParentFunction();

  return BF;
}

BinaryFunction *
BinaryContext::getBinaryFunctionAtAddress(uint64_t Address, bool Shallow) {
  // First, try to find a function starting at the given address. If the
  // function was folded, this will get us the original folded function if it
  // wasn't removed from the list, e.g. in non-relocation mode.
  auto BFI = BinaryFunctions.find(Address);
  if (BFI != BinaryFunctions.end()) {
    auto *BF = &BFI->second;
    while (!Shallow && BF->getParentFunction() && !Shallow) {
      BF = BF->getParentFunction();
    }
    return BF;
  }

  // We might have folded the function matching the object at the given
  // address. In such case, we look for a function matching the symbol
  // registered at the original address. The new function (the one that the
  // original was folded into) will hold the symbol.
  if (const auto *BD = getBinaryDataAtAddress(Address)) {
    uint64_t EntryID{0};
    auto *BF = getFunctionForSymbol(BD->getSymbol(), &EntryID);
    if (BF && EntryID == 0) {
      while (BF->getParentFunction() && !Shallow) {
        BF = BF->getParentFunction();
      }
      return BF;
    }
  }
  return nullptr;
}

DebugAddressRangesVector BinaryContext::translateModuleAddressRanges(
      const DWARFAddressRangesVector &InputRanges) const {
  DebugAddressRangesVector OutputRanges;

  for (const auto Range : InputRanges) {
    auto BFI = BinaryFunctions.lower_bound(Range.LowPC);
    while (BFI != BinaryFunctions.end()) {
      const auto &Function = BFI->second;
      if (Function.getAddress() >= Range.HighPC)
        break;
      const auto FunctionRanges = Function.getOutputAddressRanges();
      std::move(std::begin(FunctionRanges),
                std::end(FunctionRanges),
                std::back_inserter(OutputRanges));
      std::advance(BFI, 1);
    }
  }

  return OutputRanges;
}

} // namespace bolt
} // namespace llvm
