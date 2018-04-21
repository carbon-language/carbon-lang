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
#define DEBUG_TYPE "binary-section"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::opt<bool> PrintRelocations;
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
     << " (0x" << Twine::utohexstr(getFileAddress()) << ", "
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
