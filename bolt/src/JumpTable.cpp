//===--- JumpTable.h - Representation of a jump table ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "JumpTable.h"
#include "BinarySection.h"
#include "Relocation.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::opt<JumpTableSupportLevel> JumpTables;
extern cl::opt<unsigned> Verbosity;
}

JumpTable::JumpTable(StringRef Name,
                     uint64_t Address,
                     std::size_t EntrySize,
                     JumpTableType Type,
                     OffsetEntriesType &&OffsetEntries,
                     LabelMapType &&Labels,
                     BinaryFunction &BF,
                     BinarySection &Section)
  : BinaryData(Name, Address, 0, EntrySize, Section),
    EntrySize(EntrySize),
    OutputEntrySize(EntrySize),
    Type(Type),
    OffsetEntries(OffsetEntries),
    Labels(Labels),
    Parent(&BF) {
}

std::pair<size_t, size_t>
JumpTable::getEntriesForAddress(const uint64_t Addr) const {
  const uint64_t InstOffset = Addr - getAddress();
  size_t StartIndex = 0, EndIndex = 0;
  uint64_t Offset = 0;

  for (size_t I = 0; I < Entries.size(); ++I) {
    auto LI = Labels.find(Offset);
    if (LI != Labels.end()) {
      const auto NextLI = std::next(LI);
      const auto NextOffset =
        NextLI == Labels.end() ? getSize() : NextLI->first;
      if (InstOffset >= LI->first && InstOffset < NextOffset) {
        StartIndex = I;
        EndIndex = I;
        while (Offset < NextOffset) {
          ++EndIndex;
          Offset += EntrySize;
        }
        break;
      }
    }
    Offset += EntrySize;
  }

  return std::make_pair(StartIndex, EndIndex);
}

bool JumpTable::replaceDestination(uint64_t JTAddress,
                                   const MCSymbol *OldDest,
                                   MCSymbol *NewDest) {
  bool Patched{false};
  const auto Range = getEntriesForAddress(JTAddress);
  for (auto I = &Entries[Range.first], E = &Entries[Range.second];
       I != E; ++I) {
    auto &Entry = *I;
    if (Entry == OldDest) {
      Patched = true;
      Entry = NewDest;
    }
  }
  return Patched;
}

void JumpTable::updateOriginal() {
  // In non-relocation mode we have to emit jump tables in local sections.
  // This way we only overwrite them when a corresponding function is
  // overwritten.
  const uint64_t BaseOffset = getAddress() - getSection().getAddress();
  uint64_t Offset = BaseOffset;
  for (auto *Entry : Entries) {
    const auto RelType =
      Type == JTT_NORMAL ? ELF::R_X86_64_64 : ELF::R_X86_64_PC32;
    const uint64_t RelAddend = (Type == JTT_NORMAL ? 0 : Offset - BaseOffset);
    DEBUG(dbgs() << "BOLT-DEBUG: adding relocation to section "
                 << getSectionName() << " at offset 0x"
                 << Twine::utohexstr(Offset) << " for symbol "
                 << Entry->getName() << " with addend "
                 << Twine::utohexstr(RelAddend) << '\n');
    getOutputSection().addRelocation(Offset, Entry, RelType, RelAddend);
    Offset += EntrySize;
  }
}

uint64_t JumpTable::emit(MCStreamer *Streamer,
                         MCSection *HotSection,
                         MCSection *ColdSection) {
  // Pre-process entries for aggressive splitting.
  // Each label represents a separate switch table and gets its own count
  // determining its destination.
  std::map<MCSymbol *, uint64_t> LabelCounts;
  if (opts::JumpTables > JTS_SPLIT && !Counts.empty()) {
    MCSymbol *CurrentLabel = Labels[0];
    uint64_t CurrentLabelCount = 0;
    for (unsigned Index = 0; Index < Entries.size(); ++Index) {
      auto LI = Labels.find(Index * EntrySize);
      if (LI != Labels.end()) {
        LabelCounts[CurrentLabel] = CurrentLabelCount;
        CurrentLabel = LI->second;
        CurrentLabelCount = 0;
      }
      CurrentLabelCount += Counts[Index].Count;
    }
    LabelCounts[CurrentLabel] = CurrentLabelCount;
  } else {
    Streamer->SwitchSection(Count > 0 ? HotSection : ColdSection);
    Streamer->EmitValueToAlignment(EntrySize);
  }
  MCSymbol *LastLabel = nullptr;
  uint64_t Offset = 0;
  for (auto *Entry : Entries) {
    auto LI = Labels.find(Offset);
    if (LI != Labels.end()) {
      DEBUG(dbgs() << "BOLT-DEBUG: emitting jump table "
                   << LI->second->getName() << " (originally was at address 0x"
                   << Twine::utohexstr(getAddress() + Offset)
                   << (Offset ? "as part of larger jump table\n" : "\n"));
      if (!LabelCounts.empty()) {
        DEBUG(dbgs() << "BOLT-DEBUG: jump table count: "
                     << LabelCounts[LI->second] << '\n');
        if (LabelCounts[LI->second] > 0) {
          Streamer->SwitchSection(HotSection);
        } else {
          Streamer->SwitchSection(ColdSection);
        }
        Streamer->EmitValueToAlignment(EntrySize);
      }
      Streamer->EmitLabel(LI->second);
      LastLabel = LI->second;
    }
    if (Type == JTT_NORMAL) {
      Streamer->EmitSymbolValue(Entry, OutputEntrySize);
    } else { // JTT_PIC
      auto JT = MCSymbolRefExpr::create(LastLabel, Streamer->getContext());
      auto E = MCSymbolRefExpr::create(Entry, Streamer->getContext());
      auto Value = MCBinaryExpr::createSub(E, JT, Streamer->getContext());
      Streamer->EmitValue(Value, EntrySize);
    }
    Offset += EntrySize;
  }

  return Offset;
}

void JumpTable::print(raw_ostream &OS) const {
  uint64_t Offset = 0;
  for (const auto *Entry : Entries) {
    auto LI = Labels.find(Offset);
    if (LI != Labels.end()) {
      OS << "Jump Table " << LI->second->getName() << " at @0x"
         << Twine::utohexstr(getAddress()+Offset);
      if (Offset) {
        OS << " (possibly part of larger jump table):\n";
      } else {
        OS << " with total count of " << Count << ":\n";
      }
    }
    OS << format("  0x%04" PRIx64 " : ", Offset) << Entry->getName();
    if (!Counts.empty()) {
      OS << " : " << Counts[Offset / EntrySize].Mispreds
         << "/" << Counts[Offset / EntrySize].Count;
    }
    OS << '\n';
    Offset += EntrySize;
  }
  OS << "\n\n";
}
