//===--- DWARFRewriter.h --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_DWARF_REWRITER_H
#define LLVM_TOOLS_LLVM_BOLT_DWARF_REWRITER_H

#include "DebugData.h"
#include "RewriteInstance.h"
#include <map>
#include <mutex>
#include <vector>

namespace llvm {

namespace bolt {

class BinaryFunction;

class DWARFRewriter {
  DWARFRewriter() = delete;

  BinaryContext &BC;

  using SectionPatchersType = RewriteInstance::SectionPatchersType;

  SectionPatchersType &SectionPatchers;

  SimpleBinaryPatcher *DebugInfoPatcher{nullptr};

  std::mutex DebugInfoPatcherMutex;

  DebugAbbrevPatcher *AbbrevPatcher{nullptr};

  std::mutex AbbrevPatcherMutex;

  /// Stores and serializes information that will be put into the
  /// .debug_ranges DWARF section.
  std::unique_ptr<DebugRangesSectionWriter> RangesSectionWriter;

  /// Stores and serializes information that will be put into the
  /// .debug_aranges DWARF section.
  std::unique_ptr<DebugARangesSectionWriter> ARangesSectionWriter;

  /// Use a separate location list writer for each compilation unit
  std::vector<std::unique_ptr<DebugLocWriter>> LocListWritersByCU;

  struct LocListDebugInfoPatchType {
    uint32_t DebugInfoOffset;
    size_t CUIndex;
    uint64_t CUWriterOffset;
  };

  /// The list of debug info patches to be made once individual
  /// location list writers have been filled
  std::vector<LocListDebugInfoPatchType> LocListDebugInfoPatches;

  std::mutex LocListDebugInfoPatchesMutex;

  /// Update debug info for all DIEs in \p Unit.
  void updateUnitDebugInfo(size_t CUIndex, DWARFUnit *Unit);

  /// Patches the binary for an object's address ranges to be updated.
  /// The object can be a anything that has associated address ranges via either
  /// DW_AT_low/high_pc or DW_AT_ranges (i.e. functions, lexical blocks, etc).
  /// \p DebugRangesOffset is the offset in .debug_ranges of the object's
  /// new address ranges in the output binary.
  /// \p Unit Compile unit the object belongs to.
  /// \p DIE is the object's DIE in the input binary.
  void updateDWARFObjectAddressRanges(const DWARFDie DIE,
                                      uint64_t DebugRangesOffset);

  std::unique_ptr<LocBufferVector> makeFinalLocListsSection();

  /// Generate new contents for .debug_ranges and .debug_aranges section.
  void finalizeDebugSections();

  /// Patches the binary for DWARF address ranges (e.g. in functions and lexical
  /// blocks) to be updated.
  void updateDebugAddressRanges();

  /// Rewrite .gdb_index section if present.
  void updateGdbIndexSection();

  /// Abbreviations that were converted to use DW_AT_ranges.
  std::set<const DWARFAbbreviationDeclaration *> ConvertedRangesAbbrevs;

  /// DWARFDie contains a pointer to a DIE and hence gets invalidated once the
  /// embedded DIE is destroyed. This wrapper class stores a DIE internally and
  /// could be cast to a DWARFDie that is valid even after the initial DIE is
  /// destroyed.
  struct DWARFDieWrapper {
    DWARFUnit *Unit;
    DWARFDebugInfoEntry DIE;

    DWARFDieWrapper(DWARFUnit *Unit, DWARFDebugInfoEntry DIE) :
      Unit(Unit),
      DIE(DIE) {}

    DWARFDieWrapper(DWARFDie &Die) :
      Unit(Die.getDwarfUnit()),
      DIE(*Die.getDebugInfoEntry()) {}

    operator DWARFDie() {
      return DWARFDie(Unit, &DIE);
    }
  };

  /// DIEs with abbrevs that were not converted to DW_AT_ranges.
  /// We only update those when all DIEs have been processed to guarantee that
  /// the abbrev (which is shared) is intact.
  using PendingRangesType = std::unordered_map<
    const DWARFAbbreviationDeclaration *,
    std::vector<std::pair<DWARFDieWrapper, DebugAddressRange>>>;

  PendingRangesType PendingRanges;

  /// Convert \p Abbrev from using a simple DW_AT_(low|high)_pc range to
  /// DW_AT_ranges.
  void convertToRanges(const DWARFAbbreviationDeclaration *Abbrev);

  /// Update \p DIE that was using DW_AT_(low|high)_pc with DW_AT_ranges offset.
  void convertToRanges(DWARFDie DIE, uint64_t RangesSectionOffset);

  /// Same as above, but takes a vector of \p Ranges as a parameter.
  void convertToRanges(DWARFDie DIE, const DebugAddressRangesVector &Ranges);

  /// Patch DW_AT_(low|high)_pc values for the \p DIE based on \p Range.
  void patchLowHigh(DWARFDie DIE, DebugAddressRange Range);

  /// Convert pending ranges associated with the given \p Abbrev.
  void convertPending(const DWARFAbbreviationDeclaration *Abbrev);

  /// Once all DIEs were seen, update DW_AT_(low|high)_pc values.
  void flushPendingRanges();

public:
  DWARFRewriter(BinaryContext &BC,
                SectionPatchersType &SectionPatchers)
    : BC(BC), SectionPatchers(SectionPatchers) {}

  /// Main function for updating the DWARF debug info.
  void updateDebugInfo();

  /// Computes output .debug_line line table offsets for each compile unit,
  /// and updates stmt_list for a corresponding compile unit.
  void updateLineTableOffsets();
};

} // namespace bolt
} // namespace llvm

#endif
