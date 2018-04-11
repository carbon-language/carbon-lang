//===--------------------- BackendStatistics.cpp ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Functionalities used by the BackendPrinter to print out histograms
/// related to number of {issue/retire} per number of cycles.
///
//===----------------------------------------------------------------------===//

#include "BackendStatistics.h"
#include "llvm/Support/Format.h"

using namespace llvm;

namespace mca {

void BackendStatistics::onInstructionEvent(const HWInstructionEvent &Event) {
  if (Event.Type == HWInstructionEvent::Retired)
    ++NumRetired;
}

void BackendStatistics::printView(llvm::raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nRetire Control Unit - "
             << "number of cycles where we saw N instructions retired:\n";
  TempStream << "[# retired], [# cycles]\n";

  for (const std::pair<unsigned, unsigned> &Entry : RetiredPerCycle) {
    TempStream << " " << Entry.first;
    if (Entry.first < 10)
      TempStream << ",           ";
    else
      TempStream << ",          ";
    TempStream << Entry.second << "  ("
               << format("%.1f", ((double)Entry.second / NumCycles) * 100.0)
               << "%)\n";
  }

  TempStream.flush();
  OS << Buffer;
}

} // namespace mca
