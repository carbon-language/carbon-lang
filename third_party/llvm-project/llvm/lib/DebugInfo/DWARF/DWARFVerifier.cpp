//===- DWARFVerifier.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/DebugInfo/DWARF/DWARFVerifier.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFAttribute.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFLocationExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFObject.h"
#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/DJB.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <vector>

using namespace llvm;
using namespace dwarf;
using namespace object;

namespace llvm {
class DWARFDebugInfoEntry;
}

Optional<DWARFAddressRange>
DWARFVerifier::DieRangeInfo::insert(const DWARFAddressRange &R) {
  auto Begin = Ranges.begin();
  auto End = Ranges.end();
  auto Pos = std::lower_bound(Begin, End, R);

  if (Pos != End) {
    DWARFAddressRange Range(*Pos);
    if (Pos->merge(R))
      return Range;
  }
  if (Pos != Begin) {
    auto Iter = Pos - 1;
    DWARFAddressRange Range(*Iter);
    if (Iter->merge(R))
      return Range;
  }

  Ranges.insert(Pos, R);
  return None;
}

DWARFVerifier::DieRangeInfo::die_range_info_iterator
DWARFVerifier::DieRangeInfo::insert(const DieRangeInfo &RI) {
  if (RI.Ranges.empty())
    return Children.end();

  auto End = Children.end();
  auto Iter = Children.begin();
  while (Iter != End) {
    if (Iter->intersects(RI))
      return Iter;
    ++Iter;
  }
  Children.insert(RI);
  return Children.end();
}

bool DWARFVerifier::DieRangeInfo::contains(const DieRangeInfo &RHS) const {
  auto I1 = Ranges.begin(), E1 = Ranges.end();
  auto I2 = RHS.Ranges.begin(), E2 = RHS.Ranges.end();
  if (I2 == E2)
    return true;

  DWARFAddressRange R = *I2;
  while (I1 != E1) {
    bool Covered = I1->LowPC <= R.LowPC;
    if (R.LowPC == R.HighPC || (Covered && R.HighPC <= I1->HighPC)) {
      if (++I2 == E2)
        return true;
      R = *I2;
      continue;
    }
    if (!Covered)
      return false;
    if (R.LowPC < I1->HighPC)
      R.LowPC = I1->HighPC;
    ++I1;
  }
  return false;
}

bool DWARFVerifier::DieRangeInfo::intersects(const DieRangeInfo &RHS) const {
  auto I1 = Ranges.begin(), E1 = Ranges.end();
  auto I2 = RHS.Ranges.begin(), E2 = RHS.Ranges.end();
  while (I1 != E1 && I2 != E2) {
    if (I1->intersects(*I2))
      return true;
    if (I1->LowPC < I2->LowPC)
      ++I1;
    else
      ++I2;
  }
  return false;
}

bool DWARFVerifier::verifyUnitHeader(const DWARFDataExtractor DebugInfoData,
                                     uint64_t *Offset, unsigned UnitIndex,
                                     uint8_t &UnitType, bool &isUnitDWARF64) {
  uint64_t AbbrOffset, Length;
  uint8_t AddrSize = 0;
  uint16_t Version;
  bool Success = true;

  bool ValidLength = false;
  bool ValidVersion = false;
  bool ValidAddrSize = false;
  bool ValidType = true;
  bool ValidAbbrevOffset = true;

  uint64_t OffsetStart = *Offset;
  DwarfFormat Format;
  std::tie(Length, Format) = DebugInfoData.getInitialLength(Offset);
  isUnitDWARF64 = Format == DWARF64;
  Version = DebugInfoData.getU16(Offset);

  if (Version >= 5) {
    UnitType = DebugInfoData.getU8(Offset);
    AddrSize = DebugInfoData.getU8(Offset);
    AbbrOffset = isUnitDWARF64 ? DebugInfoData.getU64(Offset) : DebugInfoData.getU32(Offset);
    ValidType = dwarf::isUnitType(UnitType);
  } else {
    UnitType = 0;
    AbbrOffset = isUnitDWARF64 ? DebugInfoData.getU64(Offset) : DebugInfoData.getU32(Offset);
    AddrSize = DebugInfoData.getU8(Offset);
  }

  if (!DCtx.getDebugAbbrev()->getAbbreviationDeclarationSet(AbbrOffset))
    ValidAbbrevOffset = false;

  ValidLength = DebugInfoData.isValidOffset(OffsetStart + Length + 3);
  ValidVersion = DWARFContext::isSupportedVersion(Version);
  ValidAddrSize = DWARFContext::isAddressSizeSupported(AddrSize);
  if (!ValidLength || !ValidVersion || !ValidAddrSize || !ValidAbbrevOffset ||
      !ValidType) {
    Success = false;
    error() << format("Units[%d] - start offset: 0x%08" PRIx64 " \n", UnitIndex,
                      OffsetStart);
    if (!ValidLength)
      note() << "The length for this unit is too "
                "large for the .debug_info provided.\n";
    if (!ValidVersion)
      note() << "The 16 bit unit header version is not valid.\n";
    if (!ValidType)
      note() << "The unit type encoding is not valid.\n";
    if (!ValidAbbrevOffset)
      note() << "The offset into the .debug_abbrev section is "
                "not valid.\n";
    if (!ValidAddrSize)
      note() << "The address size is unsupported.\n";
  }
  *Offset = OffsetStart + Length + (isUnitDWARF64 ? 12 : 4);
  return Success;
}

bool DWARFVerifier::verifyName(const DWARFDie &Die) {
  // FIXME Add some kind of record of which DIE names have already failed and
  // don't bother checking a DIE that uses an already failed DIE.

  std::string ReconstructedName;
  raw_string_ostream OS(ReconstructedName);
  std::string OriginalFullName;
  Die.getFullName(OS, &OriginalFullName);
  OS.flush();
  if (OriginalFullName.empty() || OriginalFullName == ReconstructedName)
    return false;

  error() << "Simplified template DW_AT_name could not be reconstituted:\n"
          << formatv("         original: {0}\n"
                     "    reconstituted: {1}\n",
                     OriginalFullName, ReconstructedName);
  dump(Die) << '\n';
  dump(Die.getDwarfUnit()->getUnitDIE()) << '\n';
  return true;
}

unsigned DWARFVerifier::verifyUnitContents(DWARFUnit &Unit,
                                           ReferenceMap &UnitLocalReferences,
                                           ReferenceMap &CrossUnitReferences) {
  unsigned NumUnitErrors = 0;
  unsigned NumDies = Unit.getNumDIEs();
  for (unsigned I = 0; I < NumDies; ++I) {
    auto Die = Unit.getDIEAtIndex(I);

    if (Die.getTag() == DW_TAG_null)
      continue;

    for (auto AttrValue : Die.attributes()) {
      NumUnitErrors += verifyDebugInfoAttribute(Die, AttrValue);
      NumUnitErrors += verifyDebugInfoForm(Die, AttrValue, UnitLocalReferences,
                                           CrossUnitReferences);
    }

    NumUnitErrors += verifyName(Die);

    if (Die.hasChildren()) {
      if (Die.getFirstChild().isValid() &&
          Die.getFirstChild().getTag() == DW_TAG_null) {
        warn() << dwarf::TagString(Die.getTag())
               << " has DW_CHILDREN_yes but DIE has no children: ";
        Die.dump(OS);
      }
    }

    NumUnitErrors += verifyDebugInfoCallSite(Die);
  }

  DWARFDie Die = Unit.getUnitDIE(/* ExtractUnitDIEOnly = */ false);
  if (!Die) {
    error() << "Compilation unit without DIE.\n";
    NumUnitErrors++;
    return NumUnitErrors;
  }

  if (!dwarf::isUnitType(Die.getTag())) {
    error() << "Compilation unit root DIE is not a unit DIE: "
            << dwarf::TagString(Die.getTag()) << ".\n";
    NumUnitErrors++;
  }

  uint8_t UnitType = Unit.getUnitType();
  if (!DWARFUnit::isMatchingUnitTypeAndTag(UnitType, Die.getTag())) {
    error() << "Compilation unit type (" << dwarf::UnitTypeString(UnitType)
            << ") and root DIE (" << dwarf::TagString(Die.getTag())
            << ") do not match.\n";
    NumUnitErrors++;
  }

  //  According to DWARF Debugging Information Format Version 5,
  //  3.1.2 Skeleton Compilation Unit Entries:
  //  "A skeleton compilation unit has no children."
  if (Die.getTag() == dwarf::DW_TAG_skeleton_unit && Die.hasChildren()) {
    error() << "Skeleton compilation unit has children.\n";
    NumUnitErrors++;
  }

  DieRangeInfo RI;
  NumUnitErrors += verifyDieRanges(Die, RI);

  return NumUnitErrors;
}

unsigned DWARFVerifier::verifyDebugInfoCallSite(const DWARFDie &Die) {
  if (Die.getTag() != DW_TAG_call_site && Die.getTag() != DW_TAG_GNU_call_site)
    return 0;

  DWARFDie Curr = Die.getParent();
  for (; Curr.isValid() && !Curr.isSubprogramDIE(); Curr = Die.getParent()) {
    if (Curr.getTag() == DW_TAG_inlined_subroutine) {
      error() << "Call site entry nested within inlined subroutine:";
      Curr.dump(OS);
      return 1;
    }
  }

  if (!Curr.isValid()) {
    error() << "Call site entry not nested within a valid subprogram:";
    Die.dump(OS);
    return 1;
  }

  Optional<DWARFFormValue> CallAttr =
      Curr.find({DW_AT_call_all_calls, DW_AT_call_all_source_calls,
                 DW_AT_call_all_tail_calls, DW_AT_GNU_all_call_sites,
                 DW_AT_GNU_all_source_call_sites,
                 DW_AT_GNU_all_tail_call_sites});
  if (!CallAttr) {
    error() << "Subprogram with call site entry has no DW_AT_call attribute:";
    Curr.dump(OS);
    Die.dump(OS, /*indent*/ 1);
    return 1;
  }

  return 0;
}

unsigned DWARFVerifier::verifyAbbrevSection(const DWARFDebugAbbrev *Abbrev) {
  unsigned NumErrors = 0;
  if (Abbrev) {
    const DWARFAbbreviationDeclarationSet *AbbrDecls =
        Abbrev->getAbbreviationDeclarationSet(0);
    for (auto AbbrDecl : *AbbrDecls) {
      SmallDenseSet<uint16_t> AttributeSet;
      for (auto Attribute : AbbrDecl.attributes()) {
        auto Result = AttributeSet.insert(Attribute.Attr);
        if (!Result.second) {
          error() << "Abbreviation declaration contains multiple "
                  << AttributeString(Attribute.Attr) << " attributes.\n";
          AbbrDecl.dump(OS);
          ++NumErrors;
        }
      }
    }
  }
  return NumErrors;
}

bool DWARFVerifier::handleDebugAbbrev() {
  OS << "Verifying .debug_abbrev...\n";

  const DWARFObject &DObj = DCtx.getDWARFObj();
  unsigned NumErrors = 0;
  if (!DObj.getAbbrevSection().empty())
    NumErrors += verifyAbbrevSection(DCtx.getDebugAbbrev());
  if (!DObj.getAbbrevDWOSection().empty())
    NumErrors += verifyAbbrevSection(DCtx.getDebugAbbrevDWO());

  return NumErrors == 0;
}

unsigned DWARFVerifier::verifyUnits(const DWARFUnitVector &Units) {
  unsigned NumDebugInfoErrors = 0;
  ReferenceMap CrossUnitReferences;

  unsigned Index = 1;
  for (const auto &Unit : Units) {
    OS << "Verifying unit: " << Index << " / " << Units.getNumUnits();
    if (const char* Name = Unit->getUnitDIE(true).getShortName())
      OS << ", \"" << Name << '\"';
    OS << '\n';
    OS.flush();
    ReferenceMap UnitLocalReferences;
    NumDebugInfoErrors +=
        verifyUnitContents(*Unit, UnitLocalReferences, CrossUnitReferences);
    NumDebugInfoErrors += verifyDebugInfoReferences(
        UnitLocalReferences, [&](uint64_t Offset) { return Unit.get(); });
    ++Index;
  }

  NumDebugInfoErrors += verifyDebugInfoReferences(
      CrossUnitReferences, [&](uint64_t Offset) -> DWARFUnit * {
        if (DWARFUnit *U = Units.getUnitForOffset(Offset))
          return U;
        return nullptr;
      });

  return NumDebugInfoErrors;
}

unsigned DWARFVerifier::verifyUnitSection(const DWARFSection &S) {
  const DWARFObject &DObj = DCtx.getDWARFObj();
  DWARFDataExtractor DebugInfoData(DObj, S, DCtx.isLittleEndian(), 0);
  unsigned NumDebugInfoErrors = 0;
  uint64_t Offset = 0, UnitIdx = 0;
  uint8_t UnitType = 0;
  bool isUnitDWARF64 = false;
  bool isHeaderChainValid = true;
  bool hasDIE = DebugInfoData.isValidOffset(Offset);
  DWARFUnitVector TypeUnitVector;
  DWARFUnitVector CompileUnitVector;
  /// A map that tracks all references (converted absolute references) so we
  /// can verify each reference points to a valid DIE and not an offset that
  /// lies between to valid DIEs.
  ReferenceMap CrossUnitReferences;
  while (hasDIE) {
    if (!verifyUnitHeader(DebugInfoData, &Offset, UnitIdx, UnitType,
                          isUnitDWARF64)) {
      isHeaderChainValid = false;
      if (isUnitDWARF64)
        break;
    }
    hasDIE = DebugInfoData.isValidOffset(Offset);
    ++UnitIdx;
  }
  if (UnitIdx == 0 && !hasDIE) {
    warn() << "Section is empty.\n";
    isHeaderChainValid = true;
  }
  if (!isHeaderChainValid)
    ++NumDebugInfoErrors;
  return NumDebugInfoErrors;
}

unsigned DWARFVerifier::verifyIndex(StringRef Name,
                                    DWARFSectionKind InfoColumnKind,
                                    StringRef IndexStr) {
  if (IndexStr.empty())
    return 0;
  OS << "Verifying " << Name << "...\n";
  DWARFUnitIndex Index(InfoColumnKind);
  DataExtractor D(IndexStr, DCtx.isLittleEndian(), 0);
  if (!Index.parse(D))
    return 1;
  using MapType = IntervalMap<uint32_t, uint64_t>;
  MapType::Allocator Alloc;
  std::vector<std::unique_ptr<MapType>> Sections(Index.getColumnKinds().size());
  for (const DWARFUnitIndex::Entry &E : Index.getRows()) {
    uint64_t Sig = E.getSignature();
    if (!E.getContributions())
      continue;
    for (auto E : enumerate(InfoColumnKind == DW_SECT_INFO
                                ? makeArrayRef(E.getContributions(),
                                               Index.getColumnKinds().size())
                                : makeArrayRef(E.getContribution(), 1))) {
      const DWARFUnitIndex::Entry::SectionContribution &SC = E.value();
      int Col = E.index();
      if (SC.Length == 0)
        continue;
      if (!Sections[Col])
        Sections[Col] = std::make_unique<MapType>(Alloc);
      auto &M = *Sections[Col];
      auto I = M.find(SC.Offset);
      if (I != M.end() && I.start() < (SC.Offset + SC.Length)) {
        error() << llvm::formatv(
            "overlapping index entries for entries {0:x16} "
            "and {1:x16} for column {2}\n",
            *I, Sig, toString(Index.getColumnKinds()[Col]));
        return 1;
      }
      M.insert(SC.Offset, SC.Offset + SC.Length - 1, Sig);
    }
  }

  return 0;
}

bool DWARFVerifier::handleDebugCUIndex() {
  return verifyIndex(".debug_cu_index", DWARFSectionKind::DW_SECT_INFO,
                     DCtx.getDWARFObj().getCUIndexSection()) == 0;
}

bool DWARFVerifier::handleDebugTUIndex() {
  return verifyIndex(".debug_tu_index", DWARFSectionKind::DW_SECT_EXT_TYPES,
                     DCtx.getDWARFObj().getTUIndexSection()) == 0;
}

bool DWARFVerifier::handleDebugInfo() {
  const DWARFObject &DObj = DCtx.getDWARFObj();
  unsigned NumErrors = 0;

  OS << "Verifying .debug_info Unit Header Chain...\n";
  DObj.forEachInfoSections([&](const DWARFSection &S) {
    NumErrors += verifyUnitSection(S);
  });

  OS << "Verifying .debug_types Unit Header Chain...\n";
  DObj.forEachTypesSections([&](const DWARFSection &S) {
    NumErrors += verifyUnitSection(S);
  });

  OS << "Verifying non-dwo Units...\n";
  NumErrors += verifyUnits(DCtx.getNormalUnitsVector());

  OS << "Verifying dwo Units...\n";
  NumErrors += verifyUnits(DCtx.getDWOUnitsVector());
  return NumErrors == 0;
}

unsigned DWARFVerifier::verifyDieRanges(const DWARFDie &Die,
                                        DieRangeInfo &ParentRI) {
  unsigned NumErrors = 0;

  if (!Die.isValid())
    return NumErrors;

  DWARFUnit *Unit = Die.getDwarfUnit();

  auto RangesOrError = Die.getAddressRanges();
  if (!RangesOrError) {
    // FIXME: Report the error.
    if (!Unit->isDWOUnit())
      ++NumErrors;
    llvm::consumeError(RangesOrError.takeError());
    return NumErrors;
  }

  const DWARFAddressRangesVector &Ranges = RangesOrError.get();
  // Build RI for this DIE and check that ranges within this DIE do not
  // overlap.
  DieRangeInfo RI(Die);

  // TODO support object files better
  //
  // Some object file formats (i.e. non-MachO) support COMDAT.  ELF in
  // particular does so by placing each function into a section.  The DWARF data
  // for the function at that point uses a section relative DW_FORM_addrp for
  // the DW_AT_low_pc and a DW_FORM_data4 for the offset as the DW_AT_high_pc.
  // In such a case, when the Die is the CU, the ranges will overlap, and we
  // will flag valid conflicting ranges as invalid.
  //
  // For such targets, we should read the ranges from the CU and partition them
  // by the section id.  The ranges within a particular section should be
  // disjoint, although the ranges across sections may overlap.  We would map
  // the child die to the entity that it references and the section with which
  // it is associated.  The child would then be checked against the range
  // information for the associated section.
  //
  // For now, simply elide the range verification for the CU DIEs if we are
  // processing an object file.

  if (!IsObjectFile || IsMachOObject || Die.getTag() != DW_TAG_compile_unit) {
    bool DumpDieAfterError = false;
    for (const auto &Range : Ranges) {
      if (!Range.valid()) {
        ++NumErrors;
        error() << "Invalid address range " << Range << "\n";
        DumpDieAfterError = true;
        continue;
      }

      // Verify that ranges don't intersect and also build up the DieRangeInfo
      // address ranges. Don't break out of the loop below early, or we will
      // think this DIE doesn't have all of the address ranges it is supposed
      // to have. Compile units often have DW_AT_ranges that can contain one or
      // more dead stripped address ranges which tend to all be at the same
      // address: 0 or -1.
      if (auto PrevRange = RI.insert(Range)) {
        ++NumErrors;
        error() << "DIE has overlapping ranges in DW_AT_ranges attribute: "
                << *PrevRange << " and " << Range << '\n';
        DumpDieAfterError = true;
      }
    }
    if (DumpDieAfterError)
      dump(Die, 2) << '\n';
  }

  // Verify that children don't intersect.
  const auto IntersectingChild = ParentRI.insert(RI);
  if (IntersectingChild != ParentRI.Children.end()) {
    ++NumErrors;
    error() << "DIEs have overlapping address ranges:";
    dump(Die);
    dump(IntersectingChild->Die) << '\n';
  }

  // Verify that ranges are contained within their parent.
  bool ShouldBeContained = !RI.Ranges.empty() && !ParentRI.Ranges.empty() &&
                           !(Die.getTag() == DW_TAG_subprogram &&
                             ParentRI.Die.getTag() == DW_TAG_subprogram);
  if (ShouldBeContained && !ParentRI.contains(RI)) {
    ++NumErrors;
    error() << "DIE address ranges are not contained in its parent's ranges:";
    dump(ParentRI.Die);
    dump(Die, 2) << '\n';
  }

  // Recursively check children.
  for (DWARFDie Child : Die)
    NumErrors += verifyDieRanges(Child, RI);

  return NumErrors;
}

unsigned DWARFVerifier::verifyDebugInfoAttribute(const DWARFDie &Die,
                                                 DWARFAttribute &AttrValue) {
  unsigned NumErrors = 0;
  auto ReportError = [&](const Twine &TitleMsg) {
    ++NumErrors;
    error() << TitleMsg << '\n';
    dump(Die) << '\n';
  };

  const DWARFObject &DObj = DCtx.getDWARFObj();
  DWARFUnit *U = Die.getDwarfUnit();
  const auto Attr = AttrValue.Attr;
  switch (Attr) {
  case DW_AT_ranges:
    // Make sure the offset in the DW_AT_ranges attribute is valid.
    if (auto SectionOffset = AttrValue.Value.getAsSectionOffset()) {
      unsigned DwarfVersion = U->getVersion();
      const DWARFSection &RangeSection = DwarfVersion < 5
                                             ? DObj.getRangesSection()
                                             : DObj.getRnglistsSection();
      if (U->isDWOUnit() && RangeSection.Data.empty())
        break;
      if (*SectionOffset >= RangeSection.Data.size())
        ReportError(
            "DW_AT_ranges offset is beyond " +
            StringRef(DwarfVersion < 5 ? ".debug_ranges" : ".debug_rnglists") +
            " bounds: " + llvm::formatv("{0:x8}", *SectionOffset));
      break;
    }
    ReportError("DIE has invalid DW_AT_ranges encoding:");
    break;
  case DW_AT_stmt_list:
    // Make sure the offset in the DW_AT_stmt_list attribute is valid.
    if (auto SectionOffset = AttrValue.Value.getAsSectionOffset()) {
      if (*SectionOffset >= U->getLineSection().Data.size())
        ReportError("DW_AT_stmt_list offset is beyond .debug_line bounds: " +
                    llvm::formatv("{0:x8}", *SectionOffset));
      break;
    }
    ReportError("DIE has invalid DW_AT_stmt_list encoding:");
    break;
  case DW_AT_location: {
    // FIXME: It might be nice if there's a way to walk location expressions
    // without trying to resolve the address ranges - it'd be a more efficient
    // API (since the API is currently unnecessarily resolving addresses for
    // this use case which only wants to validate the expressions themselves) &
    // then the expressions could be validated even if the addresses can't be
    // resolved.
    // That sort of API would probably look like a callback "for each
    // expression" with some way to lazily resolve the address ranges when
    // needed (& then the existing API used here could be built on top of that -
    // using the callback API to build the data structure and return it).
    if (Expected<std::vector<DWARFLocationExpression>> Loc =
            Die.getLocations(DW_AT_location)) {
      for (const auto &Entry : *Loc) {
        DataExtractor Data(toStringRef(Entry.Expr), DCtx.isLittleEndian(), 0);
        DWARFExpression Expression(Data, U->getAddressByteSize(),
                                   U->getFormParams().Format);
        bool Error =
            any_of(Expression, [](const DWARFExpression::Operation &Op) {
              return Op.isError();
            });
        if (Error || !Expression.verify(U))
          ReportError("DIE contains invalid DWARF expression:");
      }
    } else if (Error Err = handleErrors(
                   Loc.takeError(), [&](std::unique_ptr<ResolverError> E) {
                     return U->isDWOUnit() ? Error::success()
                                           : Error(std::move(E));
                   }))
      ReportError(toString(std::move(Err)));
    break;
  }
  case DW_AT_specification:
  case DW_AT_abstract_origin: {
    if (auto ReferencedDie = Die.getAttributeValueAsReferencedDie(Attr)) {
      auto DieTag = Die.getTag();
      auto RefTag = ReferencedDie.getTag();
      if (DieTag == RefTag)
        break;
      if (DieTag == DW_TAG_inlined_subroutine && RefTag == DW_TAG_subprogram)
        break;
      if (DieTag == DW_TAG_variable && RefTag == DW_TAG_member)
        break;
      // This might be reference to a function declaration.
      if (DieTag == DW_TAG_GNU_call_site && RefTag == DW_TAG_subprogram)
        break;
      ReportError("DIE with tag " + TagString(DieTag) + " has " +
                  AttributeString(Attr) +
                  " that points to DIE with "
                  "incompatible tag " +
                  TagString(RefTag));
    }
    break;
  }
  case DW_AT_type: {
    DWARFDie TypeDie = Die.getAttributeValueAsReferencedDie(DW_AT_type);
    if (TypeDie && !isType(TypeDie.getTag())) {
      ReportError("DIE has " + AttributeString(Attr) +
                  " with incompatible tag " + TagString(TypeDie.getTag()));
    }
    break;
  }
  case DW_AT_call_file:
  case DW_AT_decl_file: {
    if (auto FileIdx = AttrValue.Value.getAsUnsignedConstant()) {
      if (U->isDWOUnit() && !U->isTypeUnit())
        break;
      const auto *LT = U->getContext().getLineTableForUnit(U);
      if (LT) {
        if (!LT->hasFileAtIndex(*FileIdx)) {
          bool IsZeroIndexed = LT->Prologue.getVersion() >= 5;
          if (Optional<uint64_t> LastFileIdx = LT->getLastValidFileIndex()) {
            ReportError("DIE has " + AttributeString(Attr) +
                        " with an invalid file index " +
                        llvm::formatv("{0}", *FileIdx) +
                        " (valid values are [" + (IsZeroIndexed ? "0-" : "1-") +
                        llvm::formatv("{0}", *LastFileIdx) + "])");
          } else {
            ReportError("DIE has " + AttributeString(Attr) +
                        " with an invalid file index " +
                        llvm::formatv("{0}", *FileIdx) +
                        " (the file table in the prologue is empty)");
          }
        }
      } else {
        ReportError("DIE has " + AttributeString(Attr) +
                    " that references a file with index " +
                    llvm::formatv("{0}", *FileIdx) +
                    " and the compile unit has no line table");
      }
    } else {
      ReportError("DIE has " + AttributeString(Attr) +
                  " with invalid encoding");
    }
    break;
  }
  default:
    break;
  }
  return NumErrors;
}

unsigned DWARFVerifier::verifyDebugInfoForm(const DWARFDie &Die,
                                            DWARFAttribute &AttrValue,
                                            ReferenceMap &LocalReferences,
                                            ReferenceMap &CrossUnitReferences) {
  auto DieCU = Die.getDwarfUnit();
  unsigned NumErrors = 0;
  const auto Form = AttrValue.Value.getForm();
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
      auto CUSize = DieCU->getNextUnitOffset() - DieCU->getOffset();
      auto CUOffset = AttrValue.Value.getRawUValue();
      if (CUOffset >= CUSize) {
        ++NumErrors;
        error() << FormEncodingString(Form) << " CU offset "
                << format("0x%08" PRIx64, CUOffset)
                << " is invalid (must be less than CU size of "
                << format("0x%08" PRIx64, CUSize) << "):\n";
        Die.dump(OS, 0, DumpOpts);
        dump(Die) << '\n';
      } else {
        // Valid reference, but we will verify it points to an actual
        // DIE later.
        LocalReferences[*RefVal].insert(Die.getOffset());
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
      if (*RefVal >= DieCU->getInfoSection().Data.size()) {
        ++NumErrors;
        error() << "DW_FORM_ref_addr offset beyond .debug_info "
                   "bounds:\n";
        dump(Die) << '\n';
      } else {
        // Valid reference, but we will verify it points to an actual
        // DIE later.
        CrossUnitReferences[*RefVal].insert(Die.getOffset());
      }
    }
    break;
  }
  case DW_FORM_strp:
  case DW_FORM_strx:
  case DW_FORM_strx1:
  case DW_FORM_strx2:
  case DW_FORM_strx3:
  case DW_FORM_strx4: {
    if (Error E = AttrValue.Value.getAsCString().takeError()) {
      ++NumErrors;
      error() << toString(std::move(E)) << ":\n";
      dump(Die) << '\n';
    }
    break;
  }
  default:
    break;
  }
  return NumErrors;
}

unsigned DWARFVerifier::verifyDebugInfoReferences(
    const ReferenceMap &References,
    llvm::function_ref<DWARFUnit *(uint64_t)> GetUnitForOffset) {
  auto GetDIEForOffset = [&](uint64_t Offset) {
    if (DWARFUnit *U = GetUnitForOffset(Offset))
      return U->getDIEForOffset(Offset);
    return DWARFDie();
  };
  unsigned NumErrors = 0;
  for (const std::pair<const uint64_t, std::set<uint64_t>> &Pair :
       References) {
    if (GetDIEForOffset(Pair.first))
      continue;
    ++NumErrors;
    error() << "invalid DIE reference " << format("0x%08" PRIx64, Pair.first)
            << ". Offset is in between DIEs:\n";
    for (auto Offset : Pair.second)
      dump(GetDIEForOffset(Offset)) << '\n';
    OS << "\n";
  }
  return NumErrors;
}

void DWARFVerifier::verifyDebugLineStmtOffsets() {
  std::map<uint64_t, DWARFDie> StmtListToDie;
  for (const auto &CU : DCtx.compile_units()) {
    auto Die = CU->getUnitDIE();
    // Get the attribute value as a section offset. No need to produce an
    // error here if the encoding isn't correct because we validate this in
    // the .debug_info verifier.
    auto StmtSectionOffset = toSectionOffset(Die.find(DW_AT_stmt_list));
    if (!StmtSectionOffset)
      continue;
    const uint64_t LineTableOffset = *StmtSectionOffset;
    auto LineTable = DCtx.getLineTableForUnit(CU.get());
    if (LineTableOffset < DCtx.getDWARFObj().getLineSection().Data.size()) {
      if (!LineTable) {
        ++NumDebugLineErrors;
        error() << ".debug_line[" << format("0x%08" PRIx64, LineTableOffset)
                << "] was not able to be parsed for CU:\n";
        dump(Die) << '\n';
        continue;
      }
    } else {
      // Make sure we don't get a valid line table back if the offset is wrong.
      assert(LineTable == nullptr);
      // Skip this line table as it isn't valid. No need to create an error
      // here because we validate this in the .debug_info verifier.
      continue;
    }
    auto Iter = StmtListToDie.find(LineTableOffset);
    if (Iter != StmtListToDie.end()) {
      ++NumDebugLineErrors;
      error() << "two compile unit DIEs, "
              << format("0x%08" PRIx64, Iter->second.getOffset()) << " and "
              << format("0x%08" PRIx64, Die.getOffset())
              << ", have the same DW_AT_stmt_list section offset:\n";
      dump(Iter->second);
      dump(Die) << '\n';
      // Already verified this line table before, no need to do it again.
      continue;
    }
    StmtListToDie[LineTableOffset] = Die;
  }
}

void DWARFVerifier::verifyDebugLineRows() {
  for (const auto &CU : DCtx.compile_units()) {
    auto Die = CU->getUnitDIE();
    auto LineTable = DCtx.getLineTableForUnit(CU.get());
    // If there is no line table we will have created an error in the
    // .debug_info verifier or in verifyDebugLineStmtOffsets().
    if (!LineTable)
      continue;

    // Verify prologue.
    uint32_t MaxDirIndex = LineTable->Prologue.IncludeDirectories.size();
    uint32_t FileIndex = 1;
    StringMap<uint16_t> FullPathMap;
    for (const auto &FileName : LineTable->Prologue.FileNames) {
      // Verify directory index.
      if (FileName.DirIdx > MaxDirIndex) {
        ++NumDebugLineErrors;
        error() << ".debug_line["
                << format("0x%08" PRIx64,
                          *toSectionOffset(Die.find(DW_AT_stmt_list)))
                << "].prologue.file_names[" << FileIndex
                << "].dir_idx contains an invalid index: " << FileName.DirIdx
                << "\n";
      }

      // Check file paths for duplicates.
      std::string FullPath;
      const bool HasFullPath = LineTable->getFileNameByIndex(
          FileIndex, CU->getCompilationDir(),
          DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath, FullPath);
      assert(HasFullPath && "Invalid index?");
      (void)HasFullPath;
      auto It = FullPathMap.find(FullPath);
      if (It == FullPathMap.end())
        FullPathMap[FullPath] = FileIndex;
      else if (It->second != FileIndex) {
        warn() << ".debug_line["
               << format("0x%08" PRIx64,
                         *toSectionOffset(Die.find(DW_AT_stmt_list)))
               << "].prologue.file_names[" << FileIndex
               << "] is a duplicate of file_names[" << It->second << "]\n";
      }

      FileIndex++;
    }

    // Verify rows.
    uint64_t PrevAddress = 0;
    uint32_t RowIndex = 0;
    for (const auto &Row : LineTable->Rows) {
      // Verify row address.
      if (Row.Address.Address < PrevAddress) {
        ++NumDebugLineErrors;
        error() << ".debug_line["
                << format("0x%08" PRIx64,
                          *toSectionOffset(Die.find(DW_AT_stmt_list)))
                << "] row[" << RowIndex
                << "] decreases in address from previous row:\n";

        DWARFDebugLine::Row::dumpTableHeader(OS, 0);
        if (RowIndex > 0)
          LineTable->Rows[RowIndex - 1].dump(OS);
        Row.dump(OS);
        OS << '\n';
      }

      // Verify file index.
      if (!LineTable->hasFileAtIndex(Row.File)) {
        ++NumDebugLineErrors;
        bool isDWARF5 = LineTable->Prologue.getVersion() >= 5;
        error() << ".debug_line["
                << format("0x%08" PRIx64,
                          *toSectionOffset(Die.find(DW_AT_stmt_list)))
                << "][" << RowIndex << "] has invalid file index " << Row.File
                << " (valid values are [" << (isDWARF5 ? "0," : "1,")
                << LineTable->Prologue.FileNames.size()
                << (isDWARF5 ? ")" : "]") << "):\n";
        DWARFDebugLine::Row::dumpTableHeader(OS, 0);
        Row.dump(OS);
        OS << '\n';
      }
      if (Row.EndSequence)
        PrevAddress = 0;
      else
        PrevAddress = Row.Address.Address;
      ++RowIndex;
    }
  }
}

DWARFVerifier::DWARFVerifier(raw_ostream &S, DWARFContext &D,
                             DIDumpOptions DumpOpts)
    : OS(S), DCtx(D), DumpOpts(std::move(DumpOpts)), IsObjectFile(false),
      IsMachOObject(false) {
  if (const auto *F = DCtx.getDWARFObj().getFile()) {
    IsObjectFile = F->isRelocatableObject();
    IsMachOObject = F->isMachO();
  }
}

bool DWARFVerifier::handleDebugLine() {
  NumDebugLineErrors = 0;
  OS << "Verifying .debug_line...\n";
  verifyDebugLineStmtOffsets();
  verifyDebugLineRows();
  return NumDebugLineErrors == 0;
}

unsigned DWARFVerifier::verifyAppleAccelTable(const DWARFSection *AccelSection,
                                              DataExtractor *StrData,
                                              const char *SectionName) {
  unsigned NumErrors = 0;
  DWARFDataExtractor AccelSectionData(DCtx.getDWARFObj(), *AccelSection,
                                      DCtx.isLittleEndian(), 0);
  AppleAcceleratorTable AccelTable(AccelSectionData, *StrData);

  OS << "Verifying " << SectionName << "...\n";

  // Verify that the fixed part of the header is not too short.
  if (!AccelSectionData.isValidOffset(AccelTable.getSizeHdr())) {
    error() << "Section is too small to fit a section header.\n";
    return 1;
  }

  // Verify that the section is not too short.
  if (Error E = AccelTable.extract()) {
    error() << toString(std::move(E)) << '\n';
    return 1;
  }

  // Verify that all buckets have a valid hash index or are empty.
  uint32_t NumBuckets = AccelTable.getNumBuckets();
  uint32_t NumHashes = AccelTable.getNumHashes();

  uint64_t BucketsOffset =
      AccelTable.getSizeHdr() + AccelTable.getHeaderDataLength();
  uint64_t HashesBase = BucketsOffset + NumBuckets * 4;
  uint64_t OffsetsBase = HashesBase + NumHashes * 4;
  for (uint32_t BucketIdx = 0; BucketIdx < NumBuckets; ++BucketIdx) {
    uint32_t HashIdx = AccelSectionData.getU32(&BucketsOffset);
    if (HashIdx >= NumHashes && HashIdx != UINT32_MAX) {
      error() << format("Bucket[%d] has invalid hash index: %u.\n", BucketIdx,
                        HashIdx);
      ++NumErrors;
    }
  }
  uint32_t NumAtoms = AccelTable.getAtomsDesc().size();
  if (NumAtoms == 0) {
    error() << "No atoms: failed to read HashData.\n";
    return 1;
  }
  if (!AccelTable.validateForms()) {
    error() << "Unsupported form: failed to read HashData.\n";
    return 1;
  }

  for (uint32_t HashIdx = 0; HashIdx < NumHashes; ++HashIdx) {
    uint64_t HashOffset = HashesBase + 4 * HashIdx;
    uint64_t DataOffset = OffsetsBase + 4 * HashIdx;
    uint32_t Hash = AccelSectionData.getU32(&HashOffset);
    uint64_t HashDataOffset = AccelSectionData.getU32(&DataOffset);
    if (!AccelSectionData.isValidOffsetForDataOfSize(HashDataOffset,
                                                     sizeof(uint64_t))) {
      error() << format("Hash[%d] has invalid HashData offset: "
                        "0x%08" PRIx64 ".\n",
                        HashIdx, HashDataOffset);
      ++NumErrors;
    }

    uint64_t StrpOffset;
    uint64_t StringOffset;
    uint32_t StringCount = 0;
    uint64_t Offset;
    unsigned Tag;
    while ((StrpOffset = AccelSectionData.getU32(&HashDataOffset)) != 0) {
      const uint32_t NumHashDataObjects =
          AccelSectionData.getU32(&HashDataOffset);
      for (uint32_t HashDataIdx = 0; HashDataIdx < NumHashDataObjects;
           ++HashDataIdx) {
        std::tie(Offset, Tag) = AccelTable.readAtoms(&HashDataOffset);
        auto Die = DCtx.getDIEForOffset(Offset);
        if (!Die) {
          const uint32_t BucketIdx =
              NumBuckets ? (Hash % NumBuckets) : UINT32_MAX;
          StringOffset = StrpOffset;
          const char *Name = StrData->getCStr(&StringOffset);
          if (!Name)
            Name = "<NULL>";

          error() << format(
              "%s Bucket[%d] Hash[%d] = 0x%08x "
              "Str[%u] = 0x%08" PRIx64 " DIE[%d] = 0x%08" PRIx64 " "
              "is not a valid DIE offset for \"%s\".\n",
              SectionName, BucketIdx, HashIdx, Hash, StringCount, StrpOffset,
              HashDataIdx, Offset, Name);

          ++NumErrors;
          continue;
        }
        if ((Tag != dwarf::DW_TAG_null) && (Die.getTag() != Tag)) {
          error() << "Tag " << dwarf::TagString(Tag)
                  << " in accelerator table does not match Tag "
                  << dwarf::TagString(Die.getTag()) << " of DIE[" << HashDataIdx
                  << "].\n";
          ++NumErrors;
        }
      }
      ++StringCount;
    }
  }
  return NumErrors;
}

unsigned
DWARFVerifier::verifyDebugNamesCULists(const DWARFDebugNames &AccelTable) {
  // A map from CU offset to the (first) Name Index offset which claims to index
  // this CU.
  DenseMap<uint64_t, uint64_t> CUMap;
  const uint64_t NotIndexed = std::numeric_limits<uint64_t>::max();

  CUMap.reserve(DCtx.getNumCompileUnits());
  for (const auto &CU : DCtx.compile_units())
    CUMap[CU->getOffset()] = NotIndexed;

  unsigned NumErrors = 0;
  for (const DWARFDebugNames::NameIndex &NI : AccelTable) {
    if (NI.getCUCount() == 0) {
      error() << formatv("Name Index @ {0:x} does not index any CU\n",
                         NI.getUnitOffset());
      ++NumErrors;
      continue;
    }
    for (uint32_t CU = 0, End = NI.getCUCount(); CU < End; ++CU) {
      uint64_t Offset = NI.getCUOffset(CU);
      auto Iter = CUMap.find(Offset);

      if (Iter == CUMap.end()) {
        error() << formatv(
            "Name Index @ {0:x} references a non-existing CU @ {1:x}\n",
            NI.getUnitOffset(), Offset);
        ++NumErrors;
        continue;
      }

      if (Iter->second != NotIndexed) {
        error() << formatv("Name Index @ {0:x} references a CU @ {1:x}, but "
                           "this CU is already indexed by Name Index @ {2:x}\n",
                           NI.getUnitOffset(), Offset, Iter->second);
        continue;
      }
      Iter->second = NI.getUnitOffset();
    }
  }

  for (const auto &KV : CUMap) {
    if (KV.second == NotIndexed)
      warn() << formatv("CU @ {0:x} not covered by any Name Index\n", KV.first);
  }

  return NumErrors;
}

unsigned
DWARFVerifier::verifyNameIndexBuckets(const DWARFDebugNames::NameIndex &NI,
                                      const DataExtractor &StrData) {
  struct BucketInfo {
    uint32_t Bucket;
    uint32_t Index;

    constexpr BucketInfo(uint32_t Bucket, uint32_t Index)
        : Bucket(Bucket), Index(Index) {}
    bool operator<(const BucketInfo &RHS) const { return Index < RHS.Index; }
  };

  uint32_t NumErrors = 0;
  if (NI.getBucketCount() == 0) {
    warn() << formatv("Name Index @ {0:x} does not contain a hash table.\n",
                      NI.getUnitOffset());
    return NumErrors;
  }

  // Build up a list of (Bucket, Index) pairs. We use this later to verify that
  // each Name is reachable from the appropriate bucket.
  std::vector<BucketInfo> BucketStarts;
  BucketStarts.reserve(NI.getBucketCount() + 1);
  for (uint32_t Bucket = 0, End = NI.getBucketCount(); Bucket < End; ++Bucket) {
    uint32_t Index = NI.getBucketArrayEntry(Bucket);
    if (Index > NI.getNameCount()) {
      error() << formatv("Bucket {0} of Name Index @ {1:x} contains invalid "
                         "value {2}. Valid range is [0, {3}].\n",
                         Bucket, NI.getUnitOffset(), Index, NI.getNameCount());
      ++NumErrors;
      continue;
    }
    if (Index > 0)
      BucketStarts.emplace_back(Bucket, Index);
  }

  // If there were any buckets with invalid values, skip further checks as they
  // will likely produce many errors which will only confuse the actual root
  // problem.
  if (NumErrors > 0)
    return NumErrors;

  // Sort the list in the order of increasing "Index" entries.
  array_pod_sort(BucketStarts.begin(), BucketStarts.end());

  // Insert a sentinel entry at the end, so we can check that the end of the
  // table is covered in the loop below.
  BucketStarts.emplace_back(NI.getBucketCount(), NI.getNameCount() + 1);

  // Loop invariant: NextUncovered is the (1-based) index of the first Name
  // which is not reachable by any of the buckets we processed so far (and
  // hasn't been reported as uncovered).
  uint32_t NextUncovered = 1;
  for (const BucketInfo &B : BucketStarts) {
    // Under normal circumstances B.Index be equal to NextUncovered, but it can
    // be less if a bucket points to names which are already known to be in some
    // bucket we processed earlier. In that case, we won't trigger this error,
    // but report the mismatched hash value error instead. (We know the hash
    // will not match because we have already verified that the name's hash
    // puts it into the previous bucket.)
    if (B.Index > NextUncovered) {
      error() << formatv("Name Index @ {0:x}: Name table entries [{1}, {2}] "
                         "are not covered by the hash table.\n",
                         NI.getUnitOffset(), NextUncovered, B.Index - 1);
      ++NumErrors;
    }
    uint32_t Idx = B.Index;

    // The rest of the checks apply only to non-sentinel entries.
    if (B.Bucket == NI.getBucketCount())
      break;

    // This triggers if a non-empty bucket points to a name with a mismatched
    // hash. Clients are likely to interpret this as an empty bucket, because a
    // mismatched hash signals the end of a bucket, but if this is indeed an
    // empty bucket, the producer should have signalled this by marking the
    // bucket as empty.
    uint32_t FirstHash = NI.getHashArrayEntry(Idx);
    if (FirstHash % NI.getBucketCount() != B.Bucket) {
      error() << formatv(
          "Name Index @ {0:x}: Bucket {1} is not empty but points to a "
          "mismatched hash value {2:x} (belonging to bucket {3}).\n",
          NI.getUnitOffset(), B.Bucket, FirstHash,
          FirstHash % NI.getBucketCount());
      ++NumErrors;
    }

    // This find the end of this bucket and also verifies that all the hashes in
    // this bucket are correct by comparing the stored hashes to the ones we
    // compute ourselves.
    while (Idx <= NI.getNameCount()) {
      uint32_t Hash = NI.getHashArrayEntry(Idx);
      if (Hash % NI.getBucketCount() != B.Bucket)
        break;

      const char *Str = NI.getNameTableEntry(Idx).getString();
      if (caseFoldingDjbHash(Str) != Hash) {
        error() << formatv("Name Index @ {0:x}: String ({1}) at index {2} "
                           "hashes to {3:x}, but "
                           "the Name Index hash is {4:x}\n",
                           NI.getUnitOffset(), Str, Idx,
                           caseFoldingDjbHash(Str), Hash);
        ++NumErrors;
      }

      ++Idx;
    }
    NextUncovered = std::max(NextUncovered, Idx);
  }
  return NumErrors;
}

unsigned DWARFVerifier::verifyNameIndexAttribute(
    const DWARFDebugNames::NameIndex &NI, const DWARFDebugNames::Abbrev &Abbr,
    DWARFDebugNames::AttributeEncoding AttrEnc) {
  StringRef FormName = dwarf::FormEncodingString(AttrEnc.Form);
  if (FormName.empty()) {
    error() << formatv("NameIndex @ {0:x}: Abbreviation {1:x}: {2} uses an "
                       "unknown form: {3}.\n",
                       NI.getUnitOffset(), Abbr.Code, AttrEnc.Index,
                       AttrEnc.Form);
    return 1;
  }

  if (AttrEnc.Index == DW_IDX_type_hash) {
    if (AttrEnc.Form != dwarf::DW_FORM_data8) {
      error() << formatv(
          "NameIndex @ {0:x}: Abbreviation {1:x}: DW_IDX_type_hash "
          "uses an unexpected form {2} (should be {3}).\n",
          NI.getUnitOffset(), Abbr.Code, AttrEnc.Form, dwarf::DW_FORM_data8);
      return 1;
    }
  }

  // A list of known index attributes and their expected form classes.
  // DW_IDX_type_hash is handled specially in the check above, as it has a
  // specific form (not just a form class) we should expect.
  struct FormClassTable {
    dwarf::Index Index;
    DWARFFormValue::FormClass Class;
    StringLiteral ClassName;
  };
  static constexpr FormClassTable Table[] = {
      {dwarf::DW_IDX_compile_unit, DWARFFormValue::FC_Constant, {"constant"}},
      {dwarf::DW_IDX_type_unit, DWARFFormValue::FC_Constant, {"constant"}},
      {dwarf::DW_IDX_die_offset, DWARFFormValue::FC_Reference, {"reference"}},
      {dwarf::DW_IDX_parent, DWARFFormValue::FC_Constant, {"constant"}},
  };

  ArrayRef<FormClassTable> TableRef(Table);
  auto Iter = find_if(TableRef, [AttrEnc](const FormClassTable &T) {
    return T.Index == AttrEnc.Index;
  });
  if (Iter == TableRef.end()) {
    warn() << formatv("NameIndex @ {0:x}: Abbreviation {1:x} contains an "
                      "unknown index attribute: {2}.\n",
                      NI.getUnitOffset(), Abbr.Code, AttrEnc.Index);
    return 0;
  }

  if (!DWARFFormValue(AttrEnc.Form).isFormClass(Iter->Class)) {
    error() << formatv("NameIndex @ {0:x}: Abbreviation {1:x}: {2} uses an "
                       "unexpected form {3} (expected form class {4}).\n",
                       NI.getUnitOffset(), Abbr.Code, AttrEnc.Index,
                       AttrEnc.Form, Iter->ClassName);
    return 1;
  }
  return 0;
}

unsigned
DWARFVerifier::verifyNameIndexAbbrevs(const DWARFDebugNames::NameIndex &NI) {
  if (NI.getLocalTUCount() + NI.getForeignTUCount() > 0) {
    warn() << formatv("Name Index @ {0:x}: Verifying indexes of type units is "
                      "not currently supported.\n",
                      NI.getUnitOffset());
    return 0;
  }

  unsigned NumErrors = 0;
  for (const auto &Abbrev : NI.getAbbrevs()) {
    StringRef TagName = dwarf::TagString(Abbrev.Tag);
    if (TagName.empty()) {
      warn() << formatv("NameIndex @ {0:x}: Abbreviation {1:x} references an "
                        "unknown tag: {2}.\n",
                        NI.getUnitOffset(), Abbrev.Code, Abbrev.Tag);
    }
    SmallSet<unsigned, 5> Attributes;
    for (const auto &AttrEnc : Abbrev.Attributes) {
      if (!Attributes.insert(AttrEnc.Index).second) {
        error() << formatv("NameIndex @ {0:x}: Abbreviation {1:x} contains "
                           "multiple {2} attributes.\n",
                           NI.getUnitOffset(), Abbrev.Code, AttrEnc.Index);
        ++NumErrors;
        continue;
      }
      NumErrors += verifyNameIndexAttribute(NI, Abbrev, AttrEnc);
    }

    if (NI.getCUCount() > 1 && !Attributes.count(dwarf::DW_IDX_compile_unit)) {
      error() << formatv("NameIndex @ {0:x}: Indexing multiple compile units "
                         "and abbreviation {1:x} has no {2} attribute.\n",
                         NI.getUnitOffset(), Abbrev.Code,
                         dwarf::DW_IDX_compile_unit);
      ++NumErrors;
    }
    if (!Attributes.count(dwarf::DW_IDX_die_offset)) {
      error() << formatv(
          "NameIndex @ {0:x}: Abbreviation {1:x} has no {2} attribute.\n",
          NI.getUnitOffset(), Abbrev.Code, dwarf::DW_IDX_die_offset);
      ++NumErrors;
    }
  }
  return NumErrors;
}

static SmallVector<StringRef, 2> getNames(const DWARFDie &DIE,
                                          bool IncludeLinkageName = true) {
  SmallVector<StringRef, 2> Result;
  if (const char *Str = DIE.getShortName())
    Result.emplace_back(Str);
  else if (DIE.getTag() == dwarf::DW_TAG_namespace)
    Result.emplace_back("(anonymous namespace)");

  if (IncludeLinkageName) {
    if (const char *Str = DIE.getLinkageName())
      Result.emplace_back(Str);
  }

  return Result;
}

unsigned DWARFVerifier::verifyNameIndexEntries(
    const DWARFDebugNames::NameIndex &NI,
    const DWARFDebugNames::NameTableEntry &NTE) {
  // Verifying type unit indexes not supported.
  if (NI.getLocalTUCount() + NI.getForeignTUCount() > 0)
    return 0;

  const char *CStr = NTE.getString();
  if (!CStr) {
    error() << formatv(
        "Name Index @ {0:x}: Unable to get string associated with name {1}.\n",
        NI.getUnitOffset(), NTE.getIndex());
    return 1;
  }
  StringRef Str(CStr);

  unsigned NumErrors = 0;
  unsigned NumEntries = 0;
  uint64_t EntryID = NTE.getEntryOffset();
  uint64_t NextEntryID = EntryID;
  Expected<DWARFDebugNames::Entry> EntryOr = NI.getEntry(&NextEntryID);
  for (; EntryOr; ++NumEntries, EntryID = NextEntryID,
                                EntryOr = NI.getEntry(&NextEntryID)) {
    uint32_t CUIndex = *EntryOr->getCUIndex();
    if (CUIndex > NI.getCUCount()) {
      error() << formatv("Name Index @ {0:x}: Entry @ {1:x} contains an "
                         "invalid CU index ({2}).\n",
                         NI.getUnitOffset(), EntryID, CUIndex);
      ++NumErrors;
      continue;
    }
    uint64_t CUOffset = NI.getCUOffset(CUIndex);
    uint64_t DIEOffset = CUOffset + *EntryOr->getDIEUnitOffset();
    DWARFDie DIE = DCtx.getDIEForOffset(DIEOffset);
    if (!DIE) {
      error() << formatv("Name Index @ {0:x}: Entry @ {1:x} references a "
                         "non-existing DIE @ {2:x}.\n",
                         NI.getUnitOffset(), EntryID, DIEOffset);
      ++NumErrors;
      continue;
    }
    if (DIE.getDwarfUnit()->getOffset() != CUOffset) {
      error() << formatv("Name Index @ {0:x}: Entry @ {1:x}: mismatched CU of "
                         "DIE @ {2:x}: index - {3:x}; debug_info - {4:x}.\n",
                         NI.getUnitOffset(), EntryID, DIEOffset, CUOffset,
                         DIE.getDwarfUnit()->getOffset());
      ++NumErrors;
    }
    if (DIE.getTag() != EntryOr->tag()) {
      error() << formatv("Name Index @ {0:x}: Entry @ {1:x}: mismatched Tag of "
                         "DIE @ {2:x}: index - {3}; debug_info - {4}.\n",
                         NI.getUnitOffset(), EntryID, DIEOffset, EntryOr->tag(),
                         DIE.getTag());
      ++NumErrors;
    }

    auto EntryNames = getNames(DIE);
    if (!is_contained(EntryNames, Str)) {
      error() << formatv("Name Index @ {0:x}: Entry @ {1:x}: mismatched Name "
                         "of DIE @ {2:x}: index - {3}; debug_info - {4}.\n",
                         NI.getUnitOffset(), EntryID, DIEOffset, Str,
                         make_range(EntryNames.begin(), EntryNames.end()));
      ++NumErrors;
    }
  }
  handleAllErrors(EntryOr.takeError(),
                  [&](const DWARFDebugNames::SentinelError &) {
                    if (NumEntries > 0)
                      return;
                    error() << formatv("Name Index @ {0:x}: Name {1} ({2}) is "
                                       "not associated with any entries.\n",
                                       NI.getUnitOffset(), NTE.getIndex(), Str);
                    ++NumErrors;
                  },
                  [&](const ErrorInfoBase &Info) {
                    error()
                        << formatv("Name Index @ {0:x}: Name {1} ({2}): {3}\n",
                                   NI.getUnitOffset(), NTE.getIndex(), Str,
                                   Info.message());
                    ++NumErrors;
                  });
  return NumErrors;
}

static bool isVariableIndexable(const DWARFDie &Die, DWARFContext &DCtx) {
  Expected<std::vector<DWARFLocationExpression>> Loc =
      Die.getLocations(DW_AT_location);
  if (!Loc) {
    consumeError(Loc.takeError());
    return false;
  }
  DWARFUnit *U = Die.getDwarfUnit();
  for (const auto &Entry : *Loc) {
    DataExtractor Data(toStringRef(Entry.Expr), DCtx.isLittleEndian(),
                       U->getAddressByteSize());
    DWARFExpression Expression(Data, U->getAddressByteSize(),
                               U->getFormParams().Format);
    bool IsInteresting =
        any_of(Expression, [](const DWARFExpression::Operation &Op) {
          return !Op.isError() && (Op.getCode() == DW_OP_addr ||
                                   Op.getCode() == DW_OP_form_tls_address ||
                                   Op.getCode() == DW_OP_GNU_push_tls_address);
        });
    if (IsInteresting)
      return true;
  }
  return false;
}

unsigned DWARFVerifier::verifyNameIndexCompleteness(
    const DWARFDie &Die, const DWARFDebugNames::NameIndex &NI) {

  // First check, if the Die should be indexed. The code follows the DWARF v5
  // wording as closely as possible.

  // "All non-defining declarations (that is, debugging information entries
  // with a DW_AT_declaration attribute) are excluded."
  if (Die.find(DW_AT_declaration))
    return 0;

  // "DW_TAG_namespace debugging information entries without a DW_AT_name
  // attribute are included with the name “(anonymous namespace)”.
  // All other debugging information entries without a DW_AT_name attribute
  // are excluded."
  // "If a subprogram or inlined subroutine is included, and has a
  // DW_AT_linkage_name attribute, there will be an additional index entry for
  // the linkage name."
  auto IncludeLinkageName = Die.getTag() == DW_TAG_subprogram ||
                            Die.getTag() == DW_TAG_inlined_subroutine;
  auto EntryNames = getNames(Die, IncludeLinkageName);
  if (EntryNames.empty())
    return 0;

  // We deviate from the specification here, which says:
  // "The name index must contain an entry for each debugging information entry
  // that defines a named subprogram, label, variable, type, or namespace,
  // subject to ..."
  // Explicitly exclude all TAGs that we know shouldn't be indexed.
  switch (Die.getTag()) {
  // Compile units and modules have names but shouldn't be indexed.
  case DW_TAG_compile_unit:
  case DW_TAG_module:
    return 0;

  // Function and template parameters are not globally visible, so we shouldn't
  // index them.
  case DW_TAG_formal_parameter:
  case DW_TAG_template_value_parameter:
  case DW_TAG_template_type_parameter:
  case DW_TAG_GNU_template_parameter_pack:
  case DW_TAG_GNU_template_template_param:
    return 0;

  // Object members aren't globally visible.
  case DW_TAG_member:
    return 0;

  // According to a strict reading of the specification, enumerators should not
  // be indexed (and LLVM currently does not do that). However, this causes
  // problems for the debuggers, so we may need to reconsider this.
  case DW_TAG_enumerator:
    return 0;

  // Imported declarations should not be indexed according to the specification
  // and LLVM currently does not do that.
  case DW_TAG_imported_declaration:
    return 0;

  // "DW_TAG_subprogram, DW_TAG_inlined_subroutine, and DW_TAG_label debugging
  // information entries without an address attribute (DW_AT_low_pc,
  // DW_AT_high_pc, DW_AT_ranges, or DW_AT_entry_pc) are excluded."
  case DW_TAG_subprogram:
  case DW_TAG_inlined_subroutine:
  case DW_TAG_label:
    if (Die.findRecursively(
            {DW_AT_low_pc, DW_AT_high_pc, DW_AT_ranges, DW_AT_entry_pc}))
      break;
    return 0;

  // "DW_TAG_variable debugging information entries with a DW_AT_location
  // attribute that includes a DW_OP_addr or DW_OP_form_tls_address operator are
  // included; otherwise, they are excluded."
  //
  // LLVM extension: We also add DW_OP_GNU_push_tls_address to this list.
  case DW_TAG_variable:
    if (isVariableIndexable(Die, DCtx))
      break;
    return 0;

  default:
    break;
  }

  // Now we know that our Die should be present in the Index. Let's check if
  // that's the case.
  unsigned NumErrors = 0;
  uint64_t DieUnitOffset = Die.getOffset() - Die.getDwarfUnit()->getOffset();
  for (StringRef Name : EntryNames) {
    if (none_of(NI.equal_range(Name), [&](const DWARFDebugNames::Entry &E) {
          return E.getDIEUnitOffset() == DieUnitOffset;
        })) {
      error() << formatv("Name Index @ {0:x}: Entry for DIE @ {1:x} ({2}) with "
                         "name {3} missing.\n",
                         NI.getUnitOffset(), Die.getOffset(), Die.getTag(),
                         Name);
      ++NumErrors;
    }
  }
  return NumErrors;
}

unsigned DWARFVerifier::verifyDebugNames(const DWARFSection &AccelSection,
                                         const DataExtractor &StrData) {
  unsigned NumErrors = 0;
  DWARFDataExtractor AccelSectionData(DCtx.getDWARFObj(), AccelSection,
                                      DCtx.isLittleEndian(), 0);
  DWARFDebugNames AccelTable(AccelSectionData, StrData);

  OS << "Verifying .debug_names...\n";

  // This verifies that we can read individual name indices and their
  // abbreviation tables.
  if (Error E = AccelTable.extract()) {
    error() << toString(std::move(E)) << '\n';
    return 1;
  }

  NumErrors += verifyDebugNamesCULists(AccelTable);
  for (const auto &NI : AccelTable)
    NumErrors += verifyNameIndexBuckets(NI, StrData);
  for (const auto &NI : AccelTable)
    NumErrors += verifyNameIndexAbbrevs(NI);

  // Don't attempt Entry validation if any of the previous checks found errors
  if (NumErrors > 0)
    return NumErrors;
  for (const auto &NI : AccelTable)
    for (const DWARFDebugNames::NameTableEntry &NTE : NI)
      NumErrors += verifyNameIndexEntries(NI, NTE);

  if (NumErrors > 0)
    return NumErrors;

  for (const std::unique_ptr<DWARFUnit> &U : DCtx.compile_units()) {
    if (const DWARFDebugNames::NameIndex *NI =
            AccelTable.getCUNameIndex(U->getOffset())) {
      auto *CU = cast<DWARFCompileUnit>(U.get());
      for (const DWARFDebugInfoEntry &Die : CU->dies())
        NumErrors += verifyNameIndexCompleteness(DWARFDie(CU, &Die), *NI);
    }
  }
  return NumErrors;
}

bool DWARFVerifier::handleAccelTables() {
  const DWARFObject &D = DCtx.getDWARFObj();
  DataExtractor StrData(D.getStrSection(), DCtx.isLittleEndian(), 0);
  unsigned NumErrors = 0;
  if (!D.getAppleNamesSection().Data.empty())
    NumErrors += verifyAppleAccelTable(&D.getAppleNamesSection(), &StrData,
                                       ".apple_names");
  if (!D.getAppleTypesSection().Data.empty())
    NumErrors += verifyAppleAccelTable(&D.getAppleTypesSection(), &StrData,
                                       ".apple_types");
  if (!D.getAppleNamespacesSection().Data.empty())
    NumErrors += verifyAppleAccelTable(&D.getAppleNamespacesSection(), &StrData,
                                       ".apple_namespaces");
  if (!D.getAppleObjCSection().Data.empty())
    NumErrors += verifyAppleAccelTable(&D.getAppleObjCSection(), &StrData,
                                       ".apple_objc");

  if (!D.getNamesSection().Data.empty())
    NumErrors += verifyDebugNames(D.getNamesSection(), StrData);
  return NumErrors == 0;
}

raw_ostream &DWARFVerifier::error() const { return WithColor::error(OS); }

raw_ostream &DWARFVerifier::warn() const { return WithColor::warning(OS); }

raw_ostream &DWARFVerifier::note() const { return WithColor::note(OS); }

raw_ostream &DWARFVerifier::dump(const DWARFDie &Die, unsigned indent) const {
  Die.dump(OS, indent, DumpOpts);
  return OS;
}
