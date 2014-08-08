//===-- MachODump.cpp - Object file dumping utility for llvm --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MachO-specific dumper for llvm-objdump.
//
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <system_error>
using namespace llvm;
using namespace object;

static cl::opt<bool>
  UseDbg("g", cl::desc("Print line information from debug info if available"));

static cl::opt<std::string>
  DSYMFile("dsym", cl::desc("Use .dSYM file for debug info"));

static const Target *GetTarget(const MachOObjectFile *MachOObj) {
  // Figure out the target triple.
  if (TripleName.empty()) {
    llvm::Triple TT("unknown-unknown-unknown");
    TT.setArch(Triple::ArchType(MachOObj->getArch()));
    TripleName = TT.str();
  }

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  if (TheTarget)
    return TheTarget;

  errs() << "llvm-objdump: error: unable to get target for '" << TripleName
         << "', see --version and --triple.\n";
  return nullptr;
}

struct SymbolSorter {
  bool operator()(const SymbolRef &A, const SymbolRef &B) {
    SymbolRef::Type AType, BType;
    A.getType(AType);
    B.getType(BType);

    uint64_t AAddr, BAddr;
    if (AType != SymbolRef::ST_Function)
      AAddr = 0;
    else
      A.getAddress(AAddr);
    if (BType != SymbolRef::ST_Function)
      BAddr = 0;
    else
      B.getAddress(BAddr);
    return AAddr < BAddr;
  }
};

// Types for the storted data in code table that is built before disassembly
// and the predicate function to sort them.
typedef std::pair<uint64_t, DiceRef> DiceTableEntry;
typedef std::vector<DiceTableEntry> DiceTable;
typedef DiceTable::iterator dice_table_iterator;

static bool
compareDiceTableEntries(const DiceTableEntry i,
                        const DiceTableEntry j) {
  return i.first == j.first;
}

static void DumpDataInCode(const char *bytes, uint64_t Size,
                           unsigned short Kind) {
  uint64_t Value;

  switch (Kind) {
  case MachO::DICE_KIND_DATA:
    switch (Size) {
    case 4:
      Value = bytes[3] << 24 |
              bytes[2] << 16 |
              bytes[1] << 8 |
              bytes[0];
      outs() << "\t.long " << Value;
      break;
    case 2:
      Value = bytes[1] << 8 |
              bytes[0];
      outs() << "\t.short " << Value;
      break;
    case 1:
      Value = bytes[0];
      outs() << "\t.byte " << Value;
      break;
    }
    outs() << "\t@ KIND_DATA\n";
    break;
  case MachO::DICE_KIND_JUMP_TABLE8:
    Value = bytes[0];
    outs() << "\t.byte " << Value << "\t@ KIND_JUMP_TABLE8";
    break;
  case MachO::DICE_KIND_JUMP_TABLE16:
    Value = bytes[1] << 8 |
            bytes[0];
    outs() << "\t.short " << Value << "\t@ KIND_JUMP_TABLE16";
    break;
  case MachO::DICE_KIND_JUMP_TABLE32:
    Value = bytes[3] << 24 |
            bytes[2] << 16 |
            bytes[1] << 8 |
            bytes[0];
    outs() << "\t.long " << Value << "\t@ KIND_JUMP_TABLE32";
    break;
  default:
    outs() << "\t@ data in code kind = " << Kind << "\n";
    break;
  }
}

static void getSectionsAndSymbols(const MachO::mach_header Header,
                                  MachOObjectFile *MachOObj,
                                  std::vector<SectionRef> &Sections,
                                  std::vector<SymbolRef> &Symbols,
                                  SmallVectorImpl<uint64_t> &FoundFns,
                                  uint64_t &BaseSegmentAddress) {
  for (const SymbolRef &Symbol : MachOObj->symbols())
    Symbols.push_back(Symbol);

  for (const SectionRef &Section : MachOObj->sections()) {
    StringRef SectName;
    Section.getName(SectName);
    Sections.push_back(Section);
  }

  MachOObjectFile::LoadCommandInfo Command =
      MachOObj->getFirstLoadCommandInfo();
  bool BaseSegmentAddressSet = false;
  for (unsigned i = 0; ; ++i) {
    if (Command.C.cmd == MachO::LC_FUNCTION_STARTS) {
      // We found a function starts segment, parse the addresses for later
      // consumption.
      MachO::linkedit_data_command LLC =
        MachOObj->getLinkeditDataLoadCommand(Command);

      MachOObj->ReadULEB128s(LLC.dataoff, FoundFns);
    }
    else if (Command.C.cmd == MachO::LC_SEGMENT) {
      MachO::segment_command SLC =
        MachOObj->getSegmentLoadCommand(Command);
      StringRef SegName = SLC.segname;
      if(!BaseSegmentAddressSet && SegName != "__PAGEZERO") {
        BaseSegmentAddressSet = true;
        BaseSegmentAddress = SLC.vmaddr;
      }
    }

    if (i == Header.ncmds - 1)
      break;
    else
      Command = MachOObj->getNextLoadCommandInfo(Command);
  }
}

static void DisassembleInputMachO2(StringRef Filename,
                                   MachOObjectFile *MachOOF);

void llvm::DisassembleInputMachO(StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buff =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = Buff.getError()) {
    errs() << "llvm-objdump: " << Filename << ": " << EC.message() << "\n";
    return;
  }

  std::unique_ptr<MachOObjectFile> MachOOF =
    std::move(ObjectFile::createMachOObjectFile(Buff.get()).get());

  DisassembleInputMachO2(Filename, MachOOF.get());
}

static void DisassembleInputMachO2(StringRef Filename,
                                   MachOObjectFile *MachOOF) {
  const Target *TheTarget = GetTarget(MachOOF);
  if (!TheTarget) {
    // GetTarget prints out stuff.
    return;
  }
  std::unique_ptr<const MCInstrInfo> InstrInfo(TheTarget->createMCInstrInfo());
  std::unique_ptr<MCInstrAnalysis> InstrAnalysis(
      TheTarget->createMCInstrAnalysis(InstrInfo.get()));

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (MAttrs.size()) {
    SubtargetFeatures Features;
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  // Set up disassembler.
  std::unique_ptr<const MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TripleName));
  std::unique_ptr<const MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName));
  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, MCPU, FeaturesStr));
  MCContext Ctx(AsmInfo.get(), MRI.get(), nullptr);
  std::unique_ptr<const MCDisassembler> DisAsm(
    TheTarget->createMCDisassembler(*STI, Ctx));
  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      AsmPrinterVariant, *AsmInfo, *InstrInfo, *MRI, *STI));

  if (!InstrAnalysis || !AsmInfo || !STI || !DisAsm || !IP) {
    errs() << "error: couldn't initialize disassembler for target "
           << TripleName << '\n';
    return;
  }

  outs() << '\n' << Filename << ":\n\n";

  MachO::mach_header Header = MachOOF->getHeader();

  // FIXME: FoundFns isn't used anymore. Using symbols/LC_FUNCTION_STARTS to
  // determine function locations will eventually go in MCObjectDisassembler.
  // FIXME: Using the -cfg command line option, this code used to be able to
  // annotate relocations with the referenced symbol's name, and if this was
  // inside a __[cf]string section, the data it points to. This is now replaced
  // by the upcoming MCSymbolizer, which needs the appropriate setup done above.
  std::vector<SectionRef> Sections;
  std::vector<SymbolRef> Symbols;
  SmallVector<uint64_t, 8> FoundFns;
  uint64_t BaseSegmentAddress;

  getSectionsAndSymbols(Header, MachOOF, Sections, Symbols, FoundFns,
                        BaseSegmentAddress);

  // Sort the symbols by address, just in case they didn't come in that way.
  std::sort(Symbols.begin(), Symbols.end(), SymbolSorter());

  // Build a data in code table that is sorted on by the address of each entry.
  uint64_t BaseAddress = 0;
  if (Header.filetype == MachO::MH_OBJECT)
    Sections[0].getAddress(BaseAddress);
  else
    BaseAddress = BaseSegmentAddress;
  DiceTable Dices;
  for (dice_iterator DI = MachOOF->begin_dices(), DE = MachOOF->end_dices();
       DI != DE; ++DI) {
    uint32_t Offset;
    DI->getOffset(Offset);
    Dices.push_back(std::make_pair(BaseAddress + Offset, *DI));
  }
  array_pod_sort(Dices.begin(), Dices.end());

#ifndef NDEBUG
  raw_ostream &DebugOut = DebugFlag ? dbgs() : nulls();
#else
  raw_ostream &DebugOut = nulls();
#endif

  std::unique_ptr<DIContext> diContext;
  ObjectFile *DbgObj = MachOOF;
  // Try to find debug info and set up the DIContext for it.
  if (UseDbg) {
    // A separate DSym file path was specified, parse it as a macho file,
    // get the sections and supply it to the section name parsing machinery.
    if (!DSYMFile.empty()) {
      ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
          MemoryBuffer::getFileOrSTDIN(DSYMFile);
      if (std::error_code EC = Buf.getError()) {
        errs() << "llvm-objdump: " << Filename << ": " << EC.message() << '\n';
        return;
      }
      DbgObj = ObjectFile::createMachOObjectFile(Buf.get()).get().release();
    }

    // Setup the DIContext
    diContext.reset(DIContext::getDWARFContext(*DbgObj));
  }

  for (unsigned SectIdx = 0; SectIdx != Sections.size(); SectIdx++) {

    bool SectIsText = false;
    Sections[SectIdx].isText(SectIsText);
    if (SectIsText == false)
      continue;

    StringRef SectName;
    if (Sections[SectIdx].getName(SectName) ||
        SectName != "__text")
      continue; // Skip non-text sections

    DataRefImpl DR = Sections[SectIdx].getRawDataRefImpl();

    StringRef SegmentName = MachOOF->getSectionFinalSegmentName(DR);
    if (SegmentName != "__TEXT")
      continue;

    StringRef Bytes;
    Sections[SectIdx].getContents(Bytes);
    StringRefMemoryObject memoryObject(Bytes);
    bool symbolTableWorked = false;

    // Parse relocations.
    std::vector<std::pair<uint64_t, SymbolRef>> Relocs;
    for (const RelocationRef &Reloc : Sections[SectIdx].relocations()) {
      uint64_t RelocOffset, SectionAddress;
      Reloc.getOffset(RelocOffset);
      Sections[SectIdx].getAddress(SectionAddress);
      RelocOffset -= SectionAddress;

      symbol_iterator RelocSym = Reloc.getSymbol();

      Relocs.push_back(std::make_pair(RelocOffset, *RelocSym));
    }
    array_pod_sort(Relocs.begin(), Relocs.end());

    // Disassemble symbol by symbol.
    for (unsigned SymIdx = 0; SymIdx != Symbols.size(); SymIdx++) {
      StringRef SymName;
      Symbols[SymIdx].getName(SymName);

      SymbolRef::Type ST;
      Symbols[SymIdx].getType(ST);
      if (ST != SymbolRef::ST_Function)
        continue;

      // Make sure the symbol is defined in this section.
      bool containsSym = false;
      Sections[SectIdx].containsSymbol(Symbols[SymIdx], containsSym);
      if (!containsSym)
        continue;

      // Start at the address of the symbol relative to the section's address.
      uint64_t SectionAddress = 0;
      uint64_t Start = 0;
      Sections[SectIdx].getAddress(SectionAddress);
      Symbols[SymIdx].getAddress(Start);
      Start -= SectionAddress;

      // Stop disassembling either at the beginning of the next symbol or at
      // the end of the section.
      bool containsNextSym = false;
      uint64_t NextSym = 0;
      uint64_t NextSymIdx = SymIdx+1;
      while (Symbols.size() > NextSymIdx) {
        SymbolRef::Type NextSymType;
        Symbols[NextSymIdx].getType(NextSymType);
        if (NextSymType == SymbolRef::ST_Function) {
          Sections[SectIdx].containsSymbol(Symbols[NextSymIdx],
                                           containsNextSym);
          Symbols[NextSymIdx].getAddress(NextSym);
          NextSym -= SectionAddress;
          break;
        }
        ++NextSymIdx;
      }

      uint64_t SectSize;
      Sections[SectIdx].getSize(SectSize);
      uint64_t End = containsNextSym ?  NextSym : SectSize;
      uint64_t Size;

      symbolTableWorked = true;

      outs() << SymName << ":\n";
      DILineInfo lastLine;
      for (uint64_t Index = Start; Index < End; Index += Size) {
        MCInst Inst;

        uint64_t SectAddress = 0;
        Sections[SectIdx].getAddress(SectAddress);
        outs() << format("%8" PRIx64 ":\t", SectAddress + Index);

        // Check the data in code table here to see if this is data not an
        // instruction to be disassembled.
        DiceTable Dice;
        Dice.push_back(std::make_pair(SectAddress + Index, DiceRef()));
        dice_table_iterator DTI = std::search(Dices.begin(), Dices.end(),
                                              Dice.begin(), Dice.end(),
                                              compareDiceTableEntries);
        if (DTI != Dices.end()){
          uint16_t Length;
          DTI->second.getLength(Length);
          DumpBytes(StringRef(Bytes.data() + Index, Length));
          uint16_t Kind;
          DTI->second.getKind(Kind);
          DumpDataInCode(Bytes.data() + Index, Length, Kind);
          continue;
        }

        if (DisAsm->getInstruction(Inst, Size, memoryObject, Index,
                                   DebugOut, nulls())) {
          DumpBytes(StringRef(Bytes.data() + Index, Size));
          IP->printInst(&Inst, outs(), "");

          // Print debug info.
          if (diContext) {
            DILineInfo dli =
              diContext->getLineInfoForAddress(SectAddress + Index);
            // Print valid line info if it changed.
            if (dli != lastLine && dli.Line != 0)
              outs() << "\t## " << dli.FileName << ':' << dli.Line << ':'
                     << dli.Column;
            lastLine = dli;
          }
          outs() << "\n";
        } else {
          errs() << "llvm-objdump: warning: invalid instruction encoding\n";
          if (Size == 0)
            Size = 1; // skip illegible bytes
        }
      }
    }
    if (!symbolTableWorked) {
      // Reading the symbol table didn't work, disassemble the whole section. 
      uint64_t SectAddress;
      Sections[SectIdx].getAddress(SectAddress);
      uint64_t SectSize;
      Sections[SectIdx].getSize(SectSize);
      uint64_t InstSize;
      for (uint64_t Index = 0; Index < SectSize; Index += InstSize) {
        MCInst Inst;

        if (DisAsm->getInstruction(Inst, InstSize, memoryObject, Index,
                                   DebugOut, nulls())) {
          outs() << format("%8" PRIx64 ":\t", SectAddress + Index);
          DumpBytes(StringRef(Bytes.data() + Index, InstSize));
          IP->printInst(&Inst, outs(), "");
          outs() << "\n";
        } else {
          errs() << "llvm-objdump: warning: invalid instruction encoding\n";
          if (InstSize == 0)
            InstSize = 1; // skip illegible bytes
        }
      }
    }
  }
}

namespace {
struct CompactUnwindEntry {
  uint32_t OffsetInSection;

  uint64_t FunctionAddr;
  uint32_t Length;
  uint32_t CompactEncoding;
  uint64_t PersonalityAddr;
  uint64_t LSDAAddr;

  RelocationRef FunctionReloc;
  RelocationRef PersonalityReloc;
  RelocationRef LSDAReloc;

  CompactUnwindEntry(StringRef Contents, unsigned Offset, bool Is64)
    : OffsetInSection(Offset) {
    if (Is64)
      read<uint64_t>(Contents.data() + Offset);
    else
      read<uint32_t>(Contents.data() + Offset);
  }

private:
  template<typename T>
  static uint64_t readNext(const char *&Buf) {
    using llvm::support::little;
    using llvm::support::unaligned;

    uint64_t Val = support::endian::read<T, little, unaligned>(Buf);
    Buf += sizeof(T);
    return Val;
  }

  template<typename UIntPtr>
  void read(const char *Buf) {
    FunctionAddr = readNext<UIntPtr>(Buf);
    Length = readNext<uint32_t>(Buf);
    CompactEncoding = readNext<uint32_t>(Buf);
    PersonalityAddr = readNext<UIntPtr>(Buf);
    LSDAAddr = readNext<UIntPtr>(Buf);
  }
};
}

/// Given a relocation from __compact_unwind, consisting of the RelocationRef
/// and data being relocated, determine the best base Name and Addend to use for
/// display purposes.
///
/// 1. An Extern relocation will directly reference a symbol (and the data is
///    then already an addend), so use that.
/// 2. Otherwise the data is an offset in the object file's layout; try to find
//     a symbol before it in the same section, and use the offset from there.
/// 3. Finally, if all that fails, fall back to an offset from the start of the
///    referenced section.
static void findUnwindRelocNameAddend(const MachOObjectFile *Obj,
                                      std::map<uint64_t, SymbolRef> &Symbols,
                                      const RelocationRef &Reloc,
                                      uint64_t Addr,
                                      StringRef &Name, uint64_t &Addend) {
  if (Reloc.getSymbol() != Obj->symbol_end()) {
    Reloc.getSymbol()->getName(Name);
    Addend = Addr;
    return;
  }

  auto RE = Obj->getRelocation(Reloc.getRawDataRefImpl());
  SectionRef RelocSection = Obj->getRelocationSection(RE);

  uint64_t SectionAddr;
  RelocSection.getAddress(SectionAddr);

  auto Sym = Symbols.upper_bound(Addr);
  if (Sym == Symbols.begin()) {
    // The first symbol in the object is after this reference, the best we can
    // do is section-relative notation.
    RelocSection.getName(Name);
    Addend = Addr - SectionAddr;
    return;
  }

  // Go back one so that SymbolAddress <= Addr.
  --Sym;

  section_iterator SymSection = Obj->section_end();
  Sym->second.getSection(SymSection);
  if (RelocSection == *SymSection) {
    // There's a valid symbol in the same section before this reference.
    Sym->second.getName(Name);
    Addend = Addr - Sym->first;
    return;
  }

  // There is a symbol before this reference, but it's in a different
  // section. Probably not helpful to mention it, so use the section name.
  RelocSection.getName(Name);
  Addend = Addr - SectionAddr;
}

static void printUnwindRelocDest(const MachOObjectFile *Obj,
                                 std::map<uint64_t, SymbolRef> &Symbols,
                                 const RelocationRef &Reloc,
                                 uint64_t Addr) {
  StringRef Name;
  uint64_t Addend;

  findUnwindRelocNameAddend(Obj, Symbols, Reloc, Addr, Name, Addend);

  outs() << Name;
  if (Addend)
    outs() << " + " << format("0x%x", Addend);
}

static void
printMachOCompactUnwindSection(const MachOObjectFile *Obj,
                               std::map<uint64_t, SymbolRef> &Symbols,
                               const SectionRef &CompactUnwind) {

  assert(Obj->isLittleEndian() &&
         "There should not be a big-endian .o with __compact_unwind");

  bool Is64 = Obj->is64Bit();
  uint32_t PointerSize = Is64 ? sizeof(uint64_t) : sizeof(uint32_t);
  uint32_t EntrySize = 3 * PointerSize + 2 * sizeof(uint32_t);

  StringRef Contents;
  CompactUnwind.getContents(Contents);

  SmallVector<CompactUnwindEntry, 4> CompactUnwinds;

  // First populate the initial raw offsets, encodings and so on from the entry.
  for (unsigned Offset = 0; Offset < Contents.size(); Offset += EntrySize) {
    CompactUnwindEntry Entry(Contents.data(), Offset, Is64);
    CompactUnwinds.push_back(Entry);
  }

  // Next we need to look at the relocations to find out what objects are
  // actually being referred to.
  for (const RelocationRef &Reloc : CompactUnwind.relocations()) {
    uint64_t RelocAddress;
    Reloc.getOffset(RelocAddress);

    uint32_t EntryIdx = RelocAddress / EntrySize;
    uint32_t OffsetInEntry = RelocAddress - EntryIdx * EntrySize;
    CompactUnwindEntry &Entry = CompactUnwinds[EntryIdx];

    if (OffsetInEntry == 0)
      Entry.FunctionReloc = Reloc;
    else if (OffsetInEntry == PointerSize + 2 * sizeof(uint32_t))
      Entry.PersonalityReloc = Reloc;
    else if (OffsetInEntry == 2 * PointerSize + 2 * sizeof(uint32_t))
      Entry.LSDAReloc = Reloc;
    else
      llvm_unreachable("Unexpected relocation in __compact_unwind section");
  }

  // Finally, we're ready to print the data we've gathered.
  outs() << "Contents of __compact_unwind section:\n";
  for (auto &Entry : CompactUnwinds) {
    outs() << "  Entry at offset "
           << format("0x%" PRIx32, Entry.OffsetInSection) << ":\n";

    // 1. Start of the region this entry applies to.
    outs() << "    start:                "
           << format("0x%" PRIx64, Entry.FunctionAddr) << ' ';
    printUnwindRelocDest(Obj, Symbols, Entry.FunctionReloc,
                         Entry.FunctionAddr);
    outs() << '\n';

    // 2. Length of the region this entry applies to.
    outs() << "    length:               "
           << format("0x%" PRIx32, Entry.Length) << '\n';
    // 3. The 32-bit compact encoding.
    outs() << "    compact encoding:     "
           << format("0x%08" PRIx32, Entry.CompactEncoding) << '\n';

    // 4. The personality function, if present.
    if (Entry.PersonalityReloc.getObjectFile()) {
      outs() << "    personality function: "
             << format("0x%" PRIx64, Entry.PersonalityAddr) << ' ';
      printUnwindRelocDest(Obj, Symbols, Entry.PersonalityReloc,
                           Entry.PersonalityAddr);
      outs() << '\n';
    }

    // 5. This entry's language-specific data area.
    if (Entry.LSDAReloc.getObjectFile()) {
      outs() << "    LSDA:                 "
             << format("0x%" PRIx64, Entry.LSDAAddr) << ' ';
      printUnwindRelocDest(Obj, Symbols, Entry.LSDAReloc, Entry.LSDAAddr);
      outs() << '\n';
    }
  }
}

void llvm::printMachOUnwindInfo(const MachOObjectFile *Obj) {
  std::map<uint64_t, SymbolRef> Symbols;
  for (const SymbolRef &SymRef : Obj->symbols()) {
    // Discard any undefined or absolute symbols. They're not going to take part
    // in the convenience lookup for unwind info and just take up resources.
    section_iterator Section = Obj->section_end();
    SymRef.getSection(Section);
    if (Section == Obj->section_end())
      continue;

    uint64_t Addr;
    SymRef.getAddress(Addr);
    Symbols.insert(std::make_pair(Addr, SymRef));
  }

  for (const SectionRef &Section : Obj->sections()) {
    StringRef SectName;
    Section.getName(SectName);
    if (SectName == "__compact_unwind")
      printMachOCompactUnwindSection(Obj, Symbols, Section);
    else if (SectName == "__unwind_info")
      outs() << "llvm-objdump: warning: unhandled __unwind_info section\n";
    else if (SectName == "__eh_frame")
      outs() << "llvm-objdump: warning: unhandled __eh_frame section\n";

  }
}
