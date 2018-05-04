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
#include "llvm/Support/CommandLine.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::opt<bool> PrintRelocations;
}

BinarySection::~BinarySection() {
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
