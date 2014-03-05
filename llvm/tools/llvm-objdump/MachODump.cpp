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
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/MC/MCAsmInfo.h"
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
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <cstring>
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
  return 0;
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

static void
getSectionsAndSymbols(const MachO::mach_header Header,
                      MachOObjectFile *MachOObj,
                      std::vector<SectionRef> &Sections,
                      std::vector<SymbolRef> &Symbols,
                      SmallVectorImpl<uint64_t> &FoundFns,
                      uint64_t &BaseSegmentAddress) {
  for (symbol_iterator SI = MachOObj->symbol_begin(),
                       SE = MachOObj->symbol_end();
       SI != SE; ++SI)
    Symbols.push_back(*SI);

  for (section_iterator SI = MachOObj->section_begin(),
                        SE = MachOObj->section_end();
       SI != SE; ++SI) {
    SectionRef SR = *SI;
    StringRef SectName;
    SR.getName(SectName);
    Sections.push_back(*SI);
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
  OwningPtr<MemoryBuffer> Buff;

  if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buff)) {
    errs() << "llvm-objdump: " << Filename << ": " << ec.message() << "\n";
    return;
  }

  OwningPtr<MachOObjectFile> MachOOF(static_cast<MachOObjectFile *>(
      ObjectFile::createMachOObjectFile(Buff.release()).get()));

  DisassembleInputMachO2(Filename, MachOOF.get());
}

static void DisassembleInputMachO2(StringRef Filename,
                                   MachOObjectFile *MachOOF) {
  const Target *TheTarget = GetTarget(MachOOF);
  if (!TheTarget) {
    // GetTarget prints out stuff.
    return;
  }
  OwningPtr<const MCInstrInfo> InstrInfo(TheTarget->createMCInstrInfo());
  OwningPtr<MCInstrAnalysis>
    InstrAnalysis(TheTarget->createMCInstrAnalysis(InstrInfo.get()));

  // Set up disassembler.
  OwningPtr<const MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  OwningPtr<const MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName));
  OwningPtr<const MCSubtargetInfo>
    STI(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  OwningPtr<const MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI));
  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  OwningPtr<MCInstPrinter>
    IP(TheTarget->createMCInstPrinter(AsmPrinterVariant, *AsmInfo, *InstrInfo,
                                      *MRI, *STI));

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

  OwningPtr<DIContext> diContext;
  ObjectFile *DbgObj = MachOOF;
  // Try to find debug info and set up the DIContext for it.
  if (UseDbg) {
    // A separate DSym file path was specified, parse it as a macho file,
    // get the sections and supply it to the section name parsing machinery.
    if (!DSYMFile.empty()) {
      OwningPtr<MemoryBuffer> Buf;
      if (error_code ec = MemoryBuffer::getFileOrSTDIN(DSYMFile, Buf)) {
        errs() << "llvm-objdump: " << Filename << ": " << ec.message() << '\n';
        return;
      }
      DbgObj = ObjectFile::createMachOObjectFile(Buf.release()).get();
    }

    // Setup the DIContext
    diContext.reset(DIContext::getDWARFContext(DbgObj));
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
    std::vector<std::pair<uint64_t, SymbolRef> > Relocs;
    for (relocation_iterator RI = Sections[SectIdx].relocation_begin(),
                             RE = Sections[SectIdx].relocation_end();
         RI != RE; ++RI) {
      uint64_t RelocOffset, SectionAddress;
      RI->getOffset(RelocOffset);
      Sections[SectIdx].getAddress(SectionAddress);
      RelocOffset -= SectionAddress;

      symbol_iterator RelocSym = RI->getSymbol();

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
            if (dli != lastLine && dli.getLine() != 0)
              outs() << "\t## " << dli.getFileName() << ':'
                << dli.getLine() << ':' << dli.getColumn();
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
