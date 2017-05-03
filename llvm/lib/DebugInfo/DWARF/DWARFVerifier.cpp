//===- DWARFVerifier.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFVerifier.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <vector>

using namespace llvm;
using namespace dwarf;
using namespace object;

bool DWARFVerifier::handleDebugInfo() {

  NumDebugInfoErrors = 0;
  // A map that tracks all references (converted absolute references) so we
  // can verify each reference points to a valid DIE and not an offset that
  // lies between to valid DIEs.
  std::map<uint64_t, std::set<uint32_t>> ReferenceToDIEOffsets;

  OS << "Verifying .debug_info...\n";
  for (const auto &CU : DCtx.compile_units()) {
    unsigned NumDies = CU->getNumDIEs();
    for (unsigned I = 0; I < NumDies; ++I) {
      auto Die = CU->getDIEAtIndex(I);
      const auto Tag = Die.getTag();
      if (Tag == DW_TAG_null)
        continue;
      for (auto AttrValue : Die.attributes()) {
        const auto Attr = AttrValue.Attr;
        const auto Form = AttrValue.Value.getForm();
        switch (Attr) {
        case DW_AT_ranges:
          // Make sure the offset in the DW_AT_ranges attribute is valid.
          if (auto SectionOffset = AttrValue.Value.getAsSectionOffset()) {
            if (*SectionOffset >= DCtx.getRangeSection().Data.size()) {
              ++NumDebugInfoErrors;
              OS << "error: DW_AT_ranges offset is beyond .debug_ranges "
                    "bounds:\n";
              Die.dump(OS, 0);
              OS << "\n";
            }
          } else {
            ++NumDebugInfoErrors;
            OS << "error: DIE has invalid DW_AT_ranges encoding:\n";
            Die.dump(OS, 0);
            OS << "\n";
          }
          break;
        case DW_AT_stmt_list:
          // Make sure the offset in the DW_AT_stmt_list attribute is valid.
          if (auto SectionOffset = AttrValue.Value.getAsSectionOffset()) {
            if (*SectionOffset >= DCtx.getLineSection().Data.size()) {
              ++NumDebugInfoErrors;
              OS << "error: DW_AT_stmt_list offset is beyond .debug_line "
                    "bounds: "
                 << format("0x%08" PRIx32, *SectionOffset) << "\n";
              CU->getUnitDIE().dump(OS, 0);
              OS << "\n";
            }
          } else {
            ++NumDebugInfoErrors;
            OS << "error: DIE has invalid DW_AT_stmt_list encoding:\n";
            Die.dump(OS, 0);
            OS << "\n";
          }
          break;

        default:
          break;
        }
        switch (Form) {
        case DW_FORM_ref1:
        case DW_FORM_ref2:
        case DW_FORM_ref4:
        case DW_FORM_ref8:
        case DW_FORM_ref_udata: {
          // Verify all CU relative references are valid CU offsets.
          Optional<uint64_t> RefVal = AttrValue.Value.getAsReference();
          assert(RefVal);
          if (RefVal) {
            auto DieCU = Die.getDwarfUnit();
            auto CUSize = DieCU->getNextUnitOffset() - DieCU->getOffset();
            auto CUOffset = AttrValue.Value.getRawUValue();
            if (CUOffset >= CUSize) {
              ++NumDebugInfoErrors;
              OS << "error: " << FormEncodingString(Form) << " CU offset "
                 << format("0x%08" PRIx32, CUOffset)
                 << " is invalid (must be less than CU size of "
                 << format("0x%08" PRIx32, CUSize) << "):\n";
              Die.dump(OS, 0);
              OS << "\n";
            } else {
              // Valid reference, but we will verify it points to an actual
              // DIE later.
              ReferenceToDIEOffsets[*RefVal].insert(Die.getOffset());
            }
          }
          break;
        }
        case DW_FORM_ref_addr: {
          // Verify all absolute DIE references have valid offsets in the
          // .debug_info section.
          Optional<uint64_t> RefVal = AttrValue.Value.getAsReference();
          assert(RefVal);
          if (RefVal) {
            if (*RefVal >= DCtx.getInfoSection().Data.size()) {
              ++NumDebugInfoErrors;
              OS << "error: DW_FORM_ref_addr offset beyond .debug_info "
                    "bounds:\n";
              Die.dump(OS, 0);
              OS << "\n";
            } else {
              // Valid reference, but we will verify it points to an actual
              // DIE later.
              ReferenceToDIEOffsets[*RefVal].insert(Die.getOffset());
            }
          }
          break;
        }
        case DW_FORM_strp: {
          auto SecOffset = AttrValue.Value.getAsSectionOffset();
          assert(SecOffset); // DW_FORM_strp is a section offset.
          if (SecOffset && *SecOffset >= DCtx.getStringSection().size()) {
            ++NumDebugInfoErrors;
            OS << "error: DW_FORM_strp offset beyond .debug_str bounds:\n";
            Die.dump(OS, 0);
            OS << "\n";
          }
          break;
        }
        default:
          break;
        }
      }
    }
  }

  // Take all references and make sure they point to an actual DIE by
  // getting the DIE by offset and emitting an error
  OS << "Verifying .debug_info references...\n";
  for (auto Pair : ReferenceToDIEOffsets) {
    auto Die = DCtx.getDIEForOffset(Pair.first);
    if (Die)
      continue;
    ++NumDebugInfoErrors;
    OS << "error: invalid DIE reference " << format("0x%08" PRIx64, Pair.first)
       << ". Offset is in between DIEs:\n";
    for (auto Offset : Pair.second) {
      auto ReferencingDie = DCtx.getDIEForOffset(Offset);
      ReferencingDie.dump(OS, 0);
      OS << "\n";
    }
    OS << "\n";
  }
  return NumDebugInfoErrors == 0;
}

bool DWARFVerifier::handleDebugLine() {
  std::map<uint64_t, DWARFDie> StmtListToDie;
  NumDebugLineErrors = 0;
  OS << "Verifying .debug_line...\n";
  for (const auto &CU : DCtx.compile_units()) {
    auto CUDie = CU->getUnitDIE();
    // Get the attribute value as a section offset. No need to produce an
    // error here if the encoding isn't correct because we validate this in
    // the .debug_info verifier.
    auto StmtSectionOffset = toSectionOffset(CUDie.find(DW_AT_stmt_list));
    if (!StmtSectionOffset)
      continue;
    const uint32_t LineTableOffset = *StmtSectionOffset;
    if (LineTableOffset >= DCtx.getLineSection().Data.size()) {
      // Make sure we don't get a valid line table back if the offset
      // is wrong.
      assert(DCtx.getLineTableForUnit(CU.get()) == nullptr);
      // Skip this line table as it isn't valid. No need to create an error
      // here because we validate this in the .debug_info verifier.
      continue;
    }

    auto Iter = StmtListToDie.find(LineTableOffset);
    if (Iter != StmtListToDie.end()) {
      ++NumDebugLineErrors;
      OS << "error: two compile unit DIEs, "
         << format("0x%08" PRIx32, Iter->second.getOffset()) << " and "
         << format("0x%08" PRIx32, CUDie.getOffset())
         << ", have the same DW_AT_stmt_list section offset:\n";
      Iter->second.dump(OS, 0);
      CUDie.dump(OS, 0);
      OS << '\n';
      // Already verified this line table before, no need to do it again.
      continue;
    }
    StmtListToDie[LineTableOffset] = CUDie;

    auto LineTable = DCtx.getLineTableForUnit(CU.get());
    if (!LineTable) {
      ++NumDebugLineErrors;
      OS << "error: .debug_line[" << format("0x%08" PRIx32, LineTableOffset)
         << "] was not able to be parsed for CU:\n";
      CUDie.dump(OS, 0);
      OS << '\n';
      continue;
    }
    uint32_t MaxFileIndex = LineTable->Prologue.FileNames.size();
    uint64_t PrevAddress = 0;
    uint32_t RowIndex = 0;
    for (const auto &Row : LineTable->Rows) {
      if (Row.Address < PrevAddress) {
        ++NumDebugLineErrors;
        OS << "error: .debug_line[" << format("0x%08" PRIx32, LineTableOffset)
           << "] row[" << RowIndex
           << "] decreases in address from previous row:\n";

        DWARFDebugLine::Row::dumpTableHeader(OS);
        if (RowIndex > 0)
          LineTable->Rows[RowIndex - 1].dump(OS);
        Row.dump(OS);
        OS << '\n';
      }

      if (Row.File > MaxFileIndex) {
        ++NumDebugLineErrors;
        OS << "error: .debug_line[" << format("0x%08" PRIx32, LineTableOffset)
           << "][" << RowIndex << "] has invalid file index " << Row.File
           << " (valid values are [1," << MaxFileIndex << "]):\n";
        DWARFDebugLine::Row::dumpTableHeader(OS);
        Row.dump(OS);
        OS << '\n';
      }
      if (Row.EndSequence)
        PrevAddress = 0;
      else
        PrevAddress = Row.Address;
      ++RowIndex;
    }
  }
  return NumDebugLineErrors == 0;
}
