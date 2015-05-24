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

DwarfStringPool::EntryTy &DwarfStringPool::getEntry(AsmPrinter &Asm,
                                                    StringRef Str) {
  auto &Entry = Pool[Str];
  if (!Entry.Symbol) {
    Entry.Index = Pool.size() - 1;
    Entry.Offset = NumBytes;
    Entry.Symbol = Asm.createTempSymbol(Prefix);

    NumBytes += Str.size() + 1;
    assert(NumBytes > Entry.Offset && "Unexpected overflow");
  }
  return Entry;
}

void DwarfStringPool::emit(AsmPrinter &Asm, MCSection *StrSection,
                           MCSection *OffsetSection) {
  if (Pool.empty())
    return;

  // Start the dwarf str section.
  Asm.OutStreamer->SwitchSection(StrSection);

  // Get all of the string pool entries and put them in an array by their ID so
  // we can sort them.
  SmallVector<const StringMapEntry<EntryTy> *, 64> Entries(Pool.size());

  for (const auto &E : Pool)
    Entries[E.getValue().Index] = &E;

  for (const auto &Entry : Entries) {
    // Emit a label for reference from debug information entries.
    Asm.OutStreamer->EmitLabel(Entry->getValue().Symbol);

    // Emit the string itself with a terminating null byte.
    Asm.OutStreamer->AddComment("string offset=" +
                                Twine(Entry->getValue().Offset));
    Asm.OutStreamer->EmitBytes(
        StringRef(Entry->getKeyData(), Entry->getKeyLength() + 1));
  }

  // If we've got an offset section go ahead and emit that now as well.
  if (OffsetSection) {
    Asm.OutStreamer->SwitchSection(OffsetSection);
    unsigned size = 4; // FIXME: DWARF64 is 8.
    for (const auto &Entry : Entries)
      Asm.OutStreamer->EmitIntValue(Entry->getValue().Offset, size);
  }
}
