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

#ifndef LLVM_TOOLS_LLVM_BOLT_EXCEPTIONS_H
#define LLVM_TOOLS_LLVM_BOLT_EXCEPTIONS_H

#include "BinaryContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/Support/Casting.h"
#include <map>

namespace llvm {
namespace bolt {

class BinaryFunction;
class RewriteInstance;

/// \brief Wraps up information to read all CFI instructions and feed them to a
/// BinaryFunction, as well as rewriting CFI sections.
class CFIReaderWriter {
public:
  explicit CFIReaderWriter(const DWARFDebugFrame &EHFrame);

  bool fillCFIInfoFor(BinaryFunction &Function) const;

  /// Generate .eh_frame_hdr from old and new .eh_frame sections.
  ///
  /// Take FDEs from the \p NewEHFrame unless their initial_pc is listed
  /// in \p FailedAddresses. All other entries are taken from the
  /// \p OldEHFrame.
  ///
  /// \p EHFrameHeaderAddress specifies location of .eh_frame_hdr,
  /// and is required for relative addressing used in the section.
  std::vector<char> generateEHFrameHeader(
      const DWARFDebugFrame &OldEHFrame,
      const DWARFDebugFrame &NewEHFrame,
      uint64_t EHFrameHeaderAddress,
      std::vector<uint64_t> &FailedAddresses) const;

  using FDEsMap = std::map<uint64_t, const dwarf::FDE *>;
  using fde_iterator = FDEsMap::const_iterator;

  /// Get all FDEs discovered by this reader.
  iterator_range<fde_iterator> fdes() const {
    return iterator_range<fde_iterator>(FDEs.begin(), FDEs.end());
  }

  const FDEsMap &getFDEs() const {
    return FDEs;
  }

private:
  FDEsMap FDEs;
};

} // namespace bolt
} // namespace llvm

#endif
