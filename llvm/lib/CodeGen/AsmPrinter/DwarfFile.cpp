//===-- llvm/CodeGen/DwarfFile.cpp - Dwarf Debug Framework ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DwarfFile.h"

#include "DwarfDebug.h"
#include "DwarfUnit.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/LEB128.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {
DwarfFile::DwarfFile(AsmPrinter *AP, const char *Pref, BumpPtrAllocator &DA)
    : Asm(AP), StringPool(DA), NextStringPoolNumber(0), StringPref(Pref) {}

DwarfFile::~DwarfFile() {}

MCSymbol *DwarfFile::getStringPoolSym() {
  return Asm->GetTempSymbol(StringPref);
}

MCSymbol *DwarfFile::getStringPoolEntry(StringRef Str) {
  std::pair<MCSymbol *, unsigned> &Entry =
      StringPool.GetOrCreateValue(Str).getValue();
  if (Entry.first)
    return Entry.first;

  Entry.second = NextStringPoolNumber++;
  return Entry.first = Asm->GetTempSymbol(StringPref, Entry.second);
}

unsigned DwarfFile::getStringPoolIndex(StringRef Str) {
  std::pair<MCSymbol *, unsigned> &Entry =
      StringPool.GetOrCreateValue(Str).getValue();
  if (Entry.first)
    return Entry.second;

  Entry.second = NextStringPoolNumber++;
  Entry.first = Asm->GetTempSymbol(StringPref, Entry.second);
  return Entry.second;
}

// Define a unique number for the abbreviation.
//
void DwarfFile::assignAbbrevNumber(DIEAbbrev &Abbrev) {
  // Check the set for priors.
  DIEAbbrev *InSet = AbbreviationsSet.GetOrInsertNode(&Abbrev);

  // If it's newly added.
  if (InSet == &Abbrev) {
    // Add to abbreviation list.
    Abbreviations.push_back(&Abbrev);

    // Assign the vector position + 1 as its number.
    Abbrev.setNumber(Abbreviations.size());
  } else {
    // Assign existing abbreviation number.
    Abbrev.setNumber(InSet->getNumber());
  }
}

void DwarfFile::addUnit(std::unique_ptr<DwarfUnit> U) {
  CUs.push_back(std::move(U));
}

// Emit the various dwarf units to the unit section USection with
// the abbreviations going into ASection.
void DwarfFile::emitUnits(DwarfDebug *DD, const MCSymbol *ASectionSym) {
  for (const auto &TheU : CUs) {
    DIE &Die = TheU->getUnitDie();
    const MCSection *USection = TheU->getSection();
    Asm->OutStreamer.SwitchSection(USection);

    // Emit the compile units header.
    Asm->OutStreamer.EmitLabel(TheU->getLabelBegin());

    // Emit size of content not including length itself
    Asm->OutStreamer.AddComment("Length of Unit");
    Asm->EmitInt32(TheU->getHeaderSize() + Die.getSize());

    TheU->emitHeader(ASectionSym);

    DD->emitDIE(Die);
    Asm->OutStreamer.EmitLabel(TheU->getLabelEnd());
  }
}
// Compute the size and offset for each DIE.
void DwarfFile::computeSizeAndOffsets() {
  // Offset from the first CU in the debug info section is 0 initially.
  unsigned SecOffset = 0;

  // Iterate over each compile unit and set the size and offsets for each
  // DIE within each compile unit. All offsets are CU relative.
  for (const auto &TheU : CUs) {
    TheU->setDebugInfoOffset(SecOffset);

    // CU-relative offset is reset to 0 here.
    unsigned Offset = sizeof(int32_t) +      // Length of Unit Info
                      TheU->getHeaderSize(); // Unit-specific headers

    // EndOffset here is CU-relative, after laying out
    // all of the CU DIE.
    unsigned EndOffset = computeSizeAndOffset(TheU->getUnitDie(), Offset);
    SecOffset += EndOffset;
  }
}
// Compute the size and offset of a DIE. The offset is relative to start of the
// CU. It returns the offset after laying out the DIE.
unsigned DwarfFile::computeSizeAndOffset(DIE &Die, unsigned Offset) {
  // Record the abbreviation.
  assignAbbrevNumber(Die.getAbbrev());

  // Get the abbreviation for this DIE.
  const DIEAbbrev &Abbrev = Die.getAbbrev();

  // Set DIE offset
  Die.setOffset(Offset);

  // Start the size with the size of abbreviation code.
  Offset += getULEB128Size(Die.getAbbrevNumber());

  const SmallVectorImpl<DIEValue *> &Values = Die.getValues();
  const SmallVectorImpl<DIEAbbrevData> &AbbrevData = Abbrev.getData();

  // Size the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; ++i)
    // Size attribute value.
    Offset += Values[i]->SizeOf(Asm, AbbrevData[i].getForm());

  // Get the children.
  const auto &Children = Die.getChildren();

  // Size the DIE children if any.
  if (!Children.empty()) {
    assert(Abbrev.hasChildren() && "Children flag not set");

    for (auto &Child : Children)
      Offset = computeSizeAndOffset(*Child, Offset);

    // End of children marker.
    Offset += sizeof(int8_t);
  }

  Die.setSize(Offset - Die.getOffset());
  return Offset;
}
void DwarfFile::emitAbbrevs(const MCSection *Section) {
  // Check to see if it is worth the effort.
  if (!Abbreviations.empty()) {
    // Start the debug abbrev section.
    Asm->OutStreamer.SwitchSection(Section);

    // For each abbrevation.
    for (const DIEAbbrev *Abbrev : Abbreviations) {
      // Emit the abbrevations code (base 1 index.)
      Asm->EmitULEB128(Abbrev->getNumber(), "Abbreviation Code");

      // Emit the abbreviations data.
      Abbrev->Emit(Asm);
    }

    // Mark end of abbreviations.
    Asm->EmitULEB128(0, "EOM(3)");
  }
}

// Emit strings into a string section.
void DwarfFile::emitStrings(const MCSection *StrSection,
                            const MCSection *OffsetSection,
                            const MCSymbol *StrSecSym) {

  if (StringPool.empty())
    return;

  // Start the dwarf str section.
  Asm->OutStreamer.SwitchSection(StrSection);

  // Get all of the string pool entries and put them in an array by their ID so
  // we can sort them.
  SmallVector<std::pair<unsigned, const StrPool::value_type *>, 64> Entries;

  for (const auto &I : StringPool)
    Entries.push_back(std::make_pair(I.second.second, &I));

  array_pod_sort(Entries.begin(), Entries.end());

  for (const auto &Entry : Entries) {
    // Emit a label for reference from debug information entries.
    Asm->OutStreamer.EmitLabel(Entry.second->getValue().first);

    // Emit the string itself with a terminating null byte.
    Asm->OutStreamer.EmitBytes(StringRef(Entry.second->getKeyData(),
                                         Entry.second->getKeyLength() + 1));
  }

  // If we've got an offset section go ahead and emit that now as well.
  if (OffsetSection) {
    Asm->OutStreamer.SwitchSection(OffsetSection);
    unsigned offset = 0;
    unsigned size = 4; // FIXME: DWARF64 is 8.
    for (const auto &Entry : Entries) {
      Asm->OutStreamer.EmitIntValue(offset, size);
      offset += Entry.second->getKeyLength() + 1;
    }
  }
}
}
