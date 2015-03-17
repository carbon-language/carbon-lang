//===-- llvm/CodeGen/DwarfStringPool.cpp - Dwarf Debug Framework ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DwarfStringPool.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

static std::pair<MCSymbol *, unsigned> &
getEntry(AsmPrinter &Asm,
         StringMap<std::pair<MCSymbol *, unsigned>, BumpPtrAllocator &> &Pool,
         StringRef Prefix, StringRef Str) {
  std::pair<MCSymbol *, unsigned> &Entry = Pool[Str];
  if (!Entry.first) {
    Entry.second = Pool.size() - 1;
    Entry.first = Asm.createTempSymbol(Prefix, Entry.second);
  }
  return Entry;
}

MCSymbol *DwarfStringPool::getSymbol(AsmPrinter &Asm, StringRef Str) {
  return getEntry(Asm, Pool, Prefix, Str).first;
}

unsigned DwarfStringPool::getIndex(AsmPrinter &Asm, StringRef Str) {
  return getEntry(Asm, Pool, Prefix, Str).second;
}

void DwarfStringPool::emit(AsmPrinter &Asm, const MCSection *StrSection,
                           const MCSection *OffsetSection) {
  if (Pool.empty())
    return;

  // Start the dwarf str section.
  Asm.OutStreamer.SwitchSection(StrSection);

  // Get all of the string pool entries and put them in an array by their ID so
  // we can sort them.
  SmallVector<const StringMapEntry<std::pair<MCSymbol *, unsigned>> *, 64>
  Entries(Pool.size());

  for (const auto &E : Pool)
    Entries[E.getValue().second] = &E;

  for (const auto &Entry : Entries) {
    // Emit a label for reference from debug information entries.
    Asm.OutStreamer.EmitLabel(Entry->getValue().first);

    // Emit the string itself with a terminating null byte.
    Asm.OutStreamer.EmitBytes(
        StringRef(Entry->getKeyData(), Entry->getKeyLength() + 1));
  }

  // If we've got an offset section go ahead and emit that now as well.
  if (OffsetSection) {
    Asm.OutStreamer.SwitchSection(OffsetSection);
    unsigned offset = 0;
    unsigned size = 4; // FIXME: DWARF64 is 8.
    for (const auto &Entry : Entries) {
      Asm.OutStreamer.EmitIntValue(offset, size);
      offset += Entry->getKeyLength() + 1;
    }
  }
}
