//===--- BinarySection.cpp  - Interface for object file section -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinarySection.h"
#include "BinaryContext.h"
#include "llvm/Support/CommandLine.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::opt<bool> PrintRelocations;
extern cl::opt<bool> HotData;
}

bool BinarySection::isELF() const {
  return BC.Ctx->getObjectFileInfo()->getTargetTriple().isOSBinFormatELF();
}

uint64_t
BinarySection::hash(const BinaryData &BD,
                    std::map<const BinaryData *, uint64_t> &Cache) const {
  auto Itr = Cache.find(&BD);
  if (Itr != Cache.end())
    return Itr->second;

  Cache[&BD] = 0;

  auto Offset = BD.getAddress() - getAddress();
  const auto EndOffset = BD.getEndAddress() - getAddress();
  auto Begin = Relocations.lower_bound(Relocation{Offset, 0, 0, 0, 0});
  auto End = Relocations.upper_bound(Relocation{EndOffset, 0, 0, 0, 0});
  const auto Contents = getContents();

  hash_code Hash = hash_combine(hash_value(BD.getSize()),
                                hash_value(BD.getSectionName()));

  while (Begin != End) {
    const auto &Rel = *Begin++;
    Hash = hash_combine(
      Hash,
      hash_value(Contents.substr(Offset, Begin->Offset - Offset)));
    if (auto *RelBD = BC.getBinaryDataByName(Rel.Symbol->getName())) {
      Hash = hash_combine(Hash, hash(*RelBD, Cache));
    }
    Offset = Rel.Offset + Rel.getSize();
  }

  Hash = hash_combine(
    Hash,
    hash_value(Contents.substr(Offset, EndOffset - Offset)));

  Cache[&BD] = Hash;

  return Hash;
}

void BinarySection::emitAsData(MCStreamer &Streamer, StringRef NewName) const {
  StringRef SectionName = !NewName.empty() ? NewName : getName();
  StringRef SectionContents = getContents();
  auto *ELFSection = BC.Ctx->getELFSection(SectionName,
                                           getELFType(),
                                           getELFFlags());

  Streamer.SwitchSection(ELFSection);
  Streamer.EmitValueToAlignment(getAlignment());

  if (BC.HasRelocations && opts::HotData && isReordered())
    Streamer.EmitLabel(BC.Ctx->getOrCreateSymbol("__hot_data_start"));

  DEBUG(dbgs() << "BOLT-DEBUG: emitting "
               << (isAllocatable() ? "" : "non-")
               << "allocatable data section " << SectionName << '\n');

  if (!hasRelocations()) {
    Streamer.EmitBytes(SectionContents);
  } else {
    uint64_t SectionOffset = 0;
    for (auto &Relocation : relocations()) {
      assert(Relocation.Offset < SectionContents.size() && "overflow detected");
      if (SectionOffset < Relocation.Offset) {
        Streamer.EmitBytes(
            SectionContents.substr(SectionOffset,
                                   Relocation.Offset - SectionOffset));
        SectionOffset = Relocation.Offset;
      }
      DEBUG(dbgs() << "BOLT-DEBUG: emitting relocation for symbol "
            << (Relocation.Symbol ? Relocation.Symbol->getName()
                                  : StringRef("<none>"))
            << " at offset 0x" << Twine::utohexstr(Relocation.Offset)
            << " with size "
            << Relocation::getSizeForType(Relocation.Type) << '\n');
      auto RelocationSize = Relocation.emit(&Streamer);
      SectionOffset += RelocationSize;
    }
    assert(SectionOffset <= SectionContents.size() && "overflow error");
    if (SectionOffset < SectionContents.size()) {
      Streamer.EmitBytes(SectionContents.substr(SectionOffset));
    }
  }

  if (BC.HasRelocations && opts::HotData && isReordered())
    Streamer.EmitLabel(BC.Ctx->getOrCreateSymbol("__hot_data_end"));
}

void BinarySection::flushPendingRelocations(raw_pwrite_stream &OS) {
  DEBUG(dbgs() << "BOLT-DEBUG: flushing pending relocs for section "
               << getName() << '\n');
  for (auto &Reloc : PendingRelocations) {
    DEBUG(dbgs() << "BOLT-DEBUG: writing value 0x"
                 << Twine::utohexstr(Reloc.Addend)
                 << " of size " << Relocation::getSizeForType(Reloc.Type)
                 << " at offset 0x"
                 << Twine::utohexstr(Reloc.Offset) << '\n');
    assert(Reloc.Type == ELF::R_X86_64_32 &&
           "only R_X86_64_32 relocations are supported at the moment");
    const uint32_t Value = Reloc.Addend;
    OS.pwrite(reinterpret_cast<const char*>(&Value),
              Relocation::getSizeForType(Reloc.Type),
              FileOffset + Reloc.Offset);
  }

  clearList(PendingRelocations);
}

BinarySection::~BinarySection() {
  if (isReordered()) {
    delete[] getData();
    return;
  }

  if (!isAllocatable() &&
      (!hasSectionRef() ||
       OutputContents.data() != getContents(Section).data())) {
    delete[] getOutputData();
  }
}

void BinarySection::print(raw_ostream &OS) const {
  OS << getName() << ", "
     << "0x" << Twine::utohexstr(getAddress()) << ", "
     << getSize()
     << " (0x" << Twine::utohexstr(getOutputAddress()) << ", "
     << getOutputSize() << ")"
     << ", data = " << getData()
     << ", output data = " << getOutputData();

  if (isAllocatable())
    OS << " (allocatable)";

  if (isVirtual())
    OS << " (virtual)";

  if (isTLS())
    OS << " (tls)";

  if (opts::PrintRelocations) {
    for (auto &R : relocations())
      OS << "\n  " << R;
  }
}

std::set<Relocation> BinarySection::reorderRelocations(bool Inplace) const {
  assert(PendingRelocations.empty() &&
         "reodering pending relocations not supported");
  std::set<Relocation> NewRelocations;
  for (const auto &Rel : relocations()) {
    auto RelAddr = Rel.Offset + getAddress();
    auto *BD = BC.getBinaryDataContainingAddress(RelAddr);
    BD = BD->getAtomicRoot();
    assert(BD);

    if ((!BD->isMoved() && !Inplace) || BD->isJumpTable())
      continue;

    auto NewRel(Rel);
    auto RelOffset = RelAddr - BD->getAddress();
    NewRel.Offset = BD->getOutputOffset() + RelOffset;
    assert(NewRel.Offset < getSize());
    DEBUG(dbgs() << "BOLT-DEBUG: moving " << Rel << " -> " << NewRel << "\n");
    auto Res = NewRelocations.emplace(std::move(NewRel));
    (void)Res;
    assert(Res.second && "Can't overwrite existing relocation");
  }
  return NewRelocations;
}

void BinarySection::reorderContents(const std::vector<BinaryData *> &Order,
                                    bool Inplace) {
  IsReordered = true;

  Relocations = reorderRelocations(Inplace);

  std::string Str;
  raw_string_ostream OS(Str);
  auto *Src = Contents.data();
  DEBUG(dbgs() << "BOLT-DEBUG: reorderContents for " << Name << "\n");
  for (auto *BD : Order) {
    assert((BD->isMoved() || !Inplace) && !BD->isJumpTable());
    assert(BD->isAtomic() && BD->isMoveable());
    const auto SrcOffset = BD->getAddress() - getAddress();
    assert(SrcOffset < Contents.size());
    assert(SrcOffset == BD->getOffset());
    while (OS.tell() < BD->getOutputOffset()) {
      OS.write((unsigned char)0);
    }
    DEBUG(dbgs() << "BOLT-DEBUG: " << BD->getName()
                 << " @ " << OS.tell() << "\n");
    OS.write(&Src[SrcOffset], BD->getOutputSize());
  }
  if (Relocations.empty()) {
    // If there are no existing relocations, tack a phony one at the end
    // of the reordered segment to force LLVM to recognize and map this
    // section.
    auto *ZeroSym = BC.registerNameAtAddress("Zero", 0, 0, 0);
    addRelocation(OS.tell(), ZeroSym, ELF::R_X86_64_64, 0xdeadbeef);

    uint64_t Zero = 0;
    OS.write(reinterpret_cast<const char *>(&Zero), sizeof(Zero));
  }
  auto *NewData = reinterpret_cast<char *>(copyByteArray(OS.str()));
  Contents = OutputContents = StringRef(NewData, OS.str().size());
  OutputSize = Contents.size();
}

std::string BinarySection::encodeELFNote(StringRef NameStr, StringRef DescStr,
                                         uint32_t Type) {
  std::string Str;
  raw_string_ostream OS(Str);
  const uint32_t NameSz = NameStr.size() + 1;
  const uint32_t DescSz = DescStr.size();
  OS.write(reinterpret_cast<const char *>(&(NameSz)), 4);
  OS.write(reinterpret_cast<const char *>(&(DescSz)), 4);
  OS.write(reinterpret_cast<const char *>(&(Type)), 4);
  OS << NameStr << '\0';
  for (uint64_t I = NameSz; I < alignTo(NameSz, 4); ++I) {
    OS << '\0';
  }
  OS << DescStr;
  for (uint64_t I = DescStr.size(); I < alignTo(DescStr.size(), 4); ++I) {
    OS << '\0';
  }
  return OS.str();
}
