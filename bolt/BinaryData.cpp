//===--- BinaryData.cpp - Representation of section data objects ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryData.h"
#include "BinarySection.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"

using namespace llvm;
using namespace bolt;

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

namespace opts {
extern cl::OptionCategory BoltCategory;
extern cl::opt<unsigned> Verbosity;

cl::opt<bool>
PrintSymbolAliases("print-aliases",
  cl::desc("print aliases when printing objects"),
  cl::Hidden,
  cl::ZeroOrMore,
  cl::cat(BoltCategory));
}

bool BinaryData::isMoveable() const {
  return (!isAbsolute() &&
          (IsMoveable &&
           (!Parent || isTopLevelJumpTable())));
}

void BinaryData::merge(const BinaryData *Other) {
  assert(!Size || !Other->Size || Size == Other->Size);
  assert(Address == Other->Address);
  assert(*Section == *Other->Section);
  assert(OutputOffset == Other->OutputOffset);
  assert(OutputSection == Other->OutputSection);
  Names.insert(Names.end(), Other->Names.begin(), Other->Names.end());
  Symbols.insert(Symbols.end(), Other->Symbols.begin(), Other->Symbols.end());
  MemData.insert(MemData.end(), Other->MemData.begin(), Other->MemData.end());
  if (!Size)
    Size = Other->Size;
}

bool BinaryData::hasNameRegex(StringRef NameRegex) const {
  Regex MatchName(NameRegex);
  for (auto &Name : Names)
    if (MatchName.match(Name))
      return true;
  return false;
}

StringRef BinaryData::getSectionName() const {
  return getSection().getName();
}

uint64_t BinaryData::computeOutputOffset() const {
  return Address - getSection().getAddress();
}

void BinaryData::setSection(BinarySection &NewSection) {
  Section = &NewSection;
  if (OutputSection.empty())
    OutputSection = getSection().getName();
}

bool BinaryData::isMoved() const {
  return (computeOutputOffset() != OutputOffset ||
          OutputSection != getSectionName());
}

void BinaryData::print(raw_ostream &OS) const {
  printBrief(OS);
}

void BinaryData::printBrief(raw_ostream &OS) const {
  OS << "(";

  if (isJumpTable())
    OS << "jump-table: ";
  else
    OS << "object: ";

  OS << getName();

  if ((opts::PrintSymbolAliases || opts::Verbosity > 1) && Names.size() > 1) {
    OS << ", aliases:";
    for (unsigned I = 1u; I < Names.size(); ++I) {
      OS << (I == 1 ? " (" : ", ") << Names[I];
    }
    OS << ")";
  }

  if (opts::Verbosity > 1 && Parent) {
    OS << " (" << Parent->getName() << "/" << Parent->getSize() << ")";
  }

  OS << ", 0x" << Twine::utohexstr(getAddress())
     << ":0x" << Twine::utohexstr(getEndAddress())
     << "/" << getSize();

  if (opts::Verbosity > 1) {
    for (auto &MI : memData()) {
      OS << ", " << MI;
    }
  }

  OS << ")";
}

BinaryData::BinaryData(StringRef Name,
                       uint64_t Address,
                       uint64_t Size,
                       uint16_t Alignment,
                       BinarySection &Section)
: Names({Name}),
  Section(&Section),
  Address(Address),
  Size(Size),
  Alignment(Alignment),
  OutputSection(Section.getName()),
  OutputOffset(computeOutputOffset())
{ }
