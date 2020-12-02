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
#include "llvm/ADT/DenseMap.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
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

/// Parse an existing .eh_frame and invoke the callback for each
/// address that needs to be fixed if we want to preserve the original
/// .eh_frame while changing code location.
/// This code is based on DWARFDebugFrame::parse(), but trimmed down to
/// parse only the structures that have address references.
class EHFrameParser {
public:
  using PatcherCallbackTy = std::function<void(uint64_t, uint64_t, uint64_t)>;

  /// Call PatcherCallback for every encountered external reference in frame
  /// data. The expected signature is:
  ///
  ///   void PatcherCallback(uint64_t Value, uint64_t Offset, uint64_t Type);
  ///
  /// where Value is a value of the reference, Offset - is an offset into the
  /// frame data at which the reference occured, and Type is a DWARF encoding
  /// type of the reference.
  static Error parse(DWARFDataExtractor Data, uint64_t EHFrameAddress,
                     PatcherCallbackTy PatcherCallback);

private:
  EHFrameParser(DWARFDataExtractor D, uint64_t E, PatcherCallbackTy P)
      : Data(D), EHFrameAddress(E), PatcherCallback(P), Offset(0) {}

  struct CIEInfo {
    uint64_t FDEPtrEncoding;
    uint64_t LSDAPtrEncoding;
    StringRef AugmentationString;

    CIEInfo(uint64_t F, uint64_t L, StringRef A)
        : FDEPtrEncoding(F), LSDAPtrEncoding(L), AugmentationString(A) {}
  };

  Error parseCIE(uint64_t StartOffset);
  Error parseFDE(uint64_t CIEPointer, uint64_t StartStructureOffset);
  Error parse();

  DWARFDataExtractor Data;
  uint64_t EHFrameAddress;
  PatcherCallbackTy PatcherCallback;
  uint64_t Offset;
  DenseMap<uint64_t, CIEInfo *> CIEs;
  std::vector<std::unique_ptr<CIEInfo>> Entries;
};

} // namespace bolt
} // namespace llvm

#endif
