//===-- Exceptions.h - Helpers for processing C++ exceptions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_FLO_EXCEPTIONS_H
#define LLVM_TOOLS_LLVM_FLO_EXCEPTIONS_H

#include "BinaryContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/DWARF/DWARFFrame.h"
#include "llvm/Support/Casting.h"
#include <map>

namespace llvm {
namespace flo {

class BinaryFunction;

void readLSDA(ArrayRef<uint8_t> LSDAData, BinaryContext &BC);

/// \brief Wraps up information to read all CFI instructions and feed them to a
/// BinaryFunction, as well as rewriting CFI sections.
class CFIReaderWriter {
public:
  explicit CFIReaderWriter(const DWARFFrame &EHFrame, uint64_t FrameHdrAddress,
                           std::vector<char> &FrameHdrContents)
      : EHFrame(EHFrame), FrameHdrAddress(FrameHdrAddress),
        FrameHdrContents(FrameHdrContents) {
    // Prepare FDEs for fast lookup
    for (const auto &Entry : EHFrame.Entries) {
      const dwarf::FrameEntry *FE = Entry.get();
      if (const auto *CurFDE = dyn_cast<dwarf::FDE>(FE)) {
        FDEs[CurFDE->getInitialLocation()] = CurFDE;
      }
    }
  }

  using FDEsMap = std::map<uint64_t, const dwarf::FDE *>;

  void fillCFIInfoFor(BinaryFunction &Function) const;

  // Include a new EHFrame, updating the .eh_frame_hdr
  void rewriteHeaderFor(StringRef EHFrame, uint64_t EHFrameAddress,
                        uint64_t NewFrameHdrAddress,
                        ArrayRef<uint64_t> FailedAddresses);

private:
  const DWARFFrame &EHFrame;
  uint64_t FrameHdrAddress;
  std::vector<char> &FrameHdrContents;
  FDEsMap FDEs;
};

} // namespace flo
} // namespace llvm

#endif
