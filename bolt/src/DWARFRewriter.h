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

namespace llvm {

namespace bolt {

class BinaryFunction;

class DWARFRewriter {
  DWARFRewriter() = delete;

  BinaryContext &BC;

  using SectionPatchersType = RewriteInstance::SectionPatchersType;

  SectionPatchersType &SectionPatchers;

  SimpleBinaryPatcher *DebugInfoPatcher{nullptr};
  DebugAbbrevPatcher *AbbrevPatcher{nullptr};

  /// Stores and serializes information that will be put into the .debug_ranges
  /// and .debug_aranges DWARF sections.
  std::unique_ptr<DebugRangesSectionsWriter> RangesSectionsWriter;

  std::unique_ptr<DebugLocWriter> LocationListWriter;

  /// Recursively update debug info for all DIEs in \p Unit.
  /// If \p Function is not empty, it points to a function corresponding
  /// to a parent DW_TAG_subprogram node of the current \p DIE.
  void updateUnitDebugInfo(const DWARFDie DIE,
                           std::vector<const BinaryFunction *> FunctionStack);

  /// Patches the binary for an object's address ranges to be updated.
  /// The object can be a anything that has associated address ranges via either
  /// DW_AT_low/high_pc or DW_AT_ranges (i.e. functions, lexical blocks, etc).
  /// \p DebugRangesOffset is the offset in .debug_ranges of the object's
  /// new address ranges in the output binary.
  /// \p Unit Compile unit the object belongs to.
  /// \p DIE is the object's DIE in the input binary.
  void updateDWARFObjectAddressRanges(const DWARFDie DIE,
                                      uint64_t DebugRangesOffset);

  /// Generate new contents for .debug_ranges and .debug_aranges section.
  void finalizeDebugSections();

  /// Patches the binary for DWARF address ranges (e.g. in functions and lexical
  /// blocks) to be updated.
  void updateDebugAddressRanges();

  /// Rewrite .gdb_index section if present.
  void updateGdbIndexSection();

  /// Abbreviations that were converted to use DW_AT_ranges.
  std::set<const DWARFAbbreviationDeclaration *> ConvertedRangesAbbrevs;

  /// DIEs with abbrevs that were not converted to DW_AT_ranges.
  /// We only update those when all DIEs have been processed to guarantee that
  /// the abbrev (which is shared) is intact.
  std::map<const DWARFAbbreviationDeclaration *,
           std::vector<std::pair<DWARFDie, DebugAddressRange>>> PendingRanges;

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

  /// Updates debug line information for non-simple functions, which are not
  /// rewritten.
  void updateDebugLineInfoForNonSimpleFunctions();
};

} // namespace bolt
} // namespace llvm

#endif
