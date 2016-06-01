//===--- DWARFRewriter.cpp ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//


#include "BinaryBasicBlock.h"
#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "RewriteInstance.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TimeValue.h"
#include "llvm/Support/Timer.h"
#include <algorithm>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace object;
using namespace bolt;

void RewriteInstance::updateDebugInfo() {
  SectionPatchers[".debug_abbrev"] = llvm::make_unique<DebugAbbrevPatcher>();
  SectionPatchers[".debug_info"]  = llvm::make_unique<SimpleBinaryPatcher>();

  updateFunctionRanges();

  updateAddressRangesObjects();

  updateEmptyModuleRanges();

  generateDebugRanges();

  updateLocationLists();

  updateDWARFAddressRanges();
}

void RewriteInstance::updateEmptyModuleRanges() {
  const auto &CUAddressRanges = RangesSectionsWriter.getCUAddressRanges();
  for (const auto &CU : BC->DwCtx->compile_units()) {
    if (CUAddressRanges.find(CU->getOffset()) != CUAddressRanges.end())
      continue;
    auto const &Ranges = CU->getUnitDIE(true)->getAddressRanges(CU.get());
    for (auto const &Range : Ranges) {
      RangesSectionsWriter.AddRange(CU->getOffset(),
                                    Range.first,
                                    Range.second - Range.first);
    }
  }
}

void RewriteInstance::updateDWARFAddressRanges() {
  // Update DW_AT_ranges for all compilation units.
  for (const auto &CU : BC->DwCtx->compile_units()) {
    const auto CUID = CU->getOffset();
    const auto RSOI = RangesSectionsWriter.getRangesOffsetCUMap().find(CUID);
    if (RSOI == RangesSectionsWriter.getRangesOffsetCUMap().end())
      continue;
    updateDWARFObjectAddressRanges(RSOI->second, CU.get(), CU->getUnitDIE());
  }

  // Update address ranges of functions.
  for (const auto &BFI : BinaryFunctions) {
    const auto &Function = BFI.second;
    for (const auto DIECompileUnitPair : Function.getSubprogramDIEs()) {
      updateDWARFObjectAddressRanges(
          Function.getAddressRangesOffset(),
          DIECompileUnitPair.second,
          DIECompileUnitPair.first);
    }
  }

  // Update address ranges of DIEs with addresses that don't match functions.
  for (auto &DIECompileUnitPair : BC->UnknownFunctions) {
    updateDWARFObjectAddressRanges(
        RangesSectionsWriter.getEmptyRangesListOffset(),
        DIECompileUnitPair.second,
        DIECompileUnitPair.first);
  }

  // Update address ranges of DWARF block objects (lexical/try/catch blocks,
  // inlined subroutine instances, etc).
  for (const auto &Obj : BC->AddressRangesObjects) {
    updateDWARFObjectAddressRanges(
        Obj.getAddressRangesOffset(),
        Obj.getCompileUnit(),
        Obj.getDIE());
  }
}

void RewriteInstance::updateDWARFObjectAddressRanges(
    uint32_t DebugRangesOffset,
    const DWARFUnit *Unit,
    const DWARFDebugInfoEntryMinimal *DIE) {

  // Some objects don't have an associated DIE and cannot be updated (such as
  // compiler-generated functions).
  if (!DIE) {
    return;
  }

  if (DebugRangesOffset == -1U) {
    errs() << "BOLT-WARNING: using invalid DW_AT_range for DIE at offset 0x"
           << Twine::utohexstr(DIE->getOffset()) << '\n';
  }

  auto DebugInfoPatcher =
      static_cast<SimpleBinaryPatcher *>(SectionPatchers[".debug_info"].get());
  auto AbbrevPatcher =
      static_cast<DebugAbbrevPatcher*>(SectionPatchers[".debug_abbrev"].get());

  assert(DebugInfoPatcher && AbbrevPatcher && "Patchers not initialized.");

  const auto *AbbreviationDecl = DIE->getAbbreviationDeclarationPtr();
  if (!AbbreviationDecl) {
    errs() << "BOLT-WARNING: object's DIE doesn't have an abbreviation: "
           << "skipping update. DIE at offset 0x"
           << Twine::utohexstr(DIE->getOffset()) << '\n';
    return;
  }

  auto AbbrevCode = AbbreviationDecl->getCode();

  if (AbbreviationDecl->findAttributeIndex(dwarf::DW_AT_ranges) != -1U) {
    // Case 1: The object was already non-contiguous and had DW_AT_ranges.
    // In this case we simply need to update the value of DW_AT_ranges.
    DWARFFormValue FormValue;
    uint32_t AttrOffset = -1U;
    DIE->getAttributeValue(Unit, dwarf::DW_AT_ranges, FormValue, &AttrOffset);
    DebugInfoPatcher->addLE32Patch(AttrOffset, DebugRangesOffset);
  } else {
    // Case 2: The object has both DW_AT_low_pc and DW_AT_high_pc.
    // We require the compiler to put both attributes one after the other
    // for our approach to work. low_pc and high_pc both occupy 8 bytes
    // as we're dealing with a 64-bit ELF. We basically change low_pc to
    // DW_AT_ranges and high_pc to DW_AT_producer. ranges spans only 4 bytes
    // in 32-bit DWARF, which we assume to be used, which leaves us with 12
    // more bytes. We then set the value of DW_AT_producer as an arbitrary
    // 12-byte string that fills the remaining space and leaves the rest of
    // the abbreviation layout unchanged.
    if (AbbreviationDecl->findAttributeIndex(dwarf::DW_AT_low_pc) != -1U &&
        AbbreviationDecl->findAttributeIndex(dwarf::DW_AT_high_pc) != -1U) {
      uint32_t LowPCOffset = -1U;
      uint32_t HighPCOffset = -1U;
      DWARFFormValue LowPCFormValue;
      DWARFFormValue HighPCFormValue;
      DIE->getAttributeValue(Unit, dwarf::DW_AT_low_pc, LowPCFormValue,
                             &LowPCOffset);
      DIE->getAttributeValue(Unit, dwarf::DW_AT_high_pc, HighPCFormValue,
                             &HighPCOffset);
      if (LowPCFormValue.getForm() != dwarf::DW_FORM_addr ||
          (HighPCFormValue.getForm() != dwarf::DW_FORM_addr &&
           HighPCFormValue.getForm() != dwarf::DW_FORM_data8 &&
           HighPCFormValue.getForm() != dwarf::DW_FORM_data4)) {
        errs() << "BOLT-WARNING: unexpected form value. Cannot update DIE "
                  "at offset 0x" << Twine::utohexstr(DIE->getOffset()) << '\n';
        return;
      }
      if (LowPCOffset == -1U || (LowPCOffset + 8 != HighPCOffset)) {
        errs() << "BOLT-WARNING: high_pc expected immediately after low_pc. "
                  "Cannot update DIE at offset 0x"
               << Twine::utohexstr(DIE->getOffset()) << '\n';
        return;
      }

      AbbrevPatcher->addAttributePatch(Unit,
                                       AbbrevCode,
                                       dwarf::DW_AT_low_pc,
                                       dwarf::DW_AT_ranges,
                                       dwarf::DW_FORM_sec_offset);
      AbbrevPatcher->addAttributePatch(Unit,
                                       AbbrevCode,
                                       dwarf::DW_AT_high_pc,
                                       dwarf::DW_AT_producer,
                                       dwarf::DW_FORM_string);
      unsigned StringSize = 0;
      if (HighPCFormValue.getForm() == dwarf::DW_FORM_addr ||
          HighPCFormValue.getForm() == dwarf::DW_FORM_data8) {
        StringSize = 12;
      } else if (HighPCFormValue.getForm() == dwarf::DW_FORM_data4) {
        StringSize = 8;
      } else {
        assert(0 && "unexpected form");
      }

      DebugInfoPatcher->addLE32Patch(LowPCOffset, DebugRangesOffset);
      std::string ProducerString{"LLVM-BOLT"};
      ProducerString.resize(StringSize, ' ');
      ProducerString.back() = '\0';
      DebugInfoPatcher->addBinaryPatch(LowPCOffset + 4, ProducerString);
    } else {
      errs() << "BOLT-WARNING: Cannot update ranges for DIE at offset 0x"
             << Twine::utohexstr(DIE->getOffset()) << '\n';
    }
  }
}

void RewriteInstance::updateDebugLineInfoForNonSimpleFunctions() {
  for (auto &It : BinaryFunctions) {
    const auto &Function = It.second;

    if (Function.isSimple())
      continue;

    auto ULT = Function.getDWARFUnitLineTable();
    auto Unit = ULT.first;
    auto LineTable = ULT.second;

    if (!LineTable)
      continue; // nothing to update for this function

    std::vector<uint32_t> Results;
    MCSectionELF *FunctionSection =
        BC->Ctx->getELFSection(Function.getCodeSectionName(),
                               ELF::SHT_PROGBITS,
                               ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);

    uint64_t Address = It.first;
    if (LineTable->lookupAddressRange(Address, Function.getMaxSize(),
                                      Results)) {
      auto &OutputLineTable =
          BC->Ctx->getMCDwarfLineTable(Unit->getOffset()).getMCLineSections();
      for (auto RowIndex : Results) {
        const auto &Row = LineTable->Rows[RowIndex];
        BC->Ctx->setCurrentDwarfLoc(
            Row.File,
            Row.Line,
            Row.Column,
            (DWARF2_FLAG_IS_STMT * Row.IsStmt) |
            (DWARF2_FLAG_BASIC_BLOCK * Row.BasicBlock) |
            (DWARF2_FLAG_PROLOGUE_END * Row.PrologueEnd) |
            (DWARF2_FLAG_EPILOGUE_BEGIN * Row.EpilogueBegin),
            Row.Isa,
            Row.Discriminator,
            Row.Address);
        auto Loc = BC->Ctx->getCurrentDwarfLoc();
        BC->Ctx->clearDwarfLocSeen();
        OutputLineTable.addLineEntry(MCLineEntry{nullptr, Loc},
                                     FunctionSection);
      }
      // Add an empty entry past the end of the function
      // for end_sequence mark.
      BC->Ctx->setCurrentDwarfLoc(0, 0, 0, 0, 0, 0,
                                  Address + Function.getMaxSize());
      auto Loc = BC->Ctx->getCurrentDwarfLoc();
      BC->Ctx->clearDwarfLocSeen();
      OutputLineTable.addLineEntry(MCLineEntry{nullptr, Loc},
                                   FunctionSection);
    } else {
      DEBUG(errs() << "BOLT-DEBUG: Function " << Function.getName()
                   << " has no associated line number information.\n");
    }
  }
}

void RewriteInstance::updateAddressRangesObjects() {
  for (auto &Obj : BC->AddressRangesObjects) {
    for (const auto &Range : Obj.getAbsoluteAddressRanges()) {
      RangesSectionsWriter.AddRange(&Obj, Range.first,
                                    Range.second - Range.first);
    }
  }
}

void RewriteInstance::updateLineTableOffsets() {
  const auto LineSection =
    BC->Ctx->getObjectFileInfo()->getDwarfLineSection();
  auto CurrentFragment = LineSection->begin();
  uint32_t CurrentOffset = 0;
  uint32_t Offset = 0;

  // Line tables are stored in MCContext in ascending order of offset in the
  // output file, thus we can compute all table's offset by passing through
  // each fragment at most once, continuing from the last CU's beginning
  // instead of from the first fragment.
  for (const auto &CUIDLineTablePair : BC->Ctx->getMCDwarfLineTables()) {
    auto Label = CUIDLineTablePair.second.getLabel();
    if (!Label)
      continue;

    auto CUOffset = CUIDLineTablePair.first;
    if (CUOffset == -1U)
      continue;

    auto *CU = BC->DwCtx->getCompileUnitForOffset(CUOffset);
    assert(CU && "expected non-null CU");
    auto LTOffset =
      BC->DwCtx->getAttrFieldOffsetForUnit(CU, dwarf::DW_AT_stmt_list);
    if (!LTOffset)
      continue;

    auto Fragment = Label->getFragment();
    while (&*CurrentFragment != Fragment) {
      switch (CurrentFragment->getKind()) {
      case MCFragment::FT_Dwarf:
        Offset += cast<MCDwarfLineAddrFragment>(*CurrentFragment)
          .getContents().size() - CurrentOffset;
        break;
      case MCFragment::FT_Data:
        Offset += cast<MCDataFragment>(*CurrentFragment)
          .getContents().size() - CurrentOffset;
        break;
      default:
        llvm_unreachable(".debug_line section shouldn't contain other types "
                         "of fragments.");
      }
      ++CurrentFragment;
      CurrentOffset = 0;
    }

    Offset += Label->getOffset() - CurrentOffset;
    CurrentOffset = Label->getOffset();

    auto &SI = SectionMM->NoteSectionInfo[".debug_info"];
    SI.PendingRelocs.emplace_back(
        SectionInfo::Reloc{LTOffset, 4, 0, Offset});

    DEBUG(dbgs() << "BOLT-DEBUG: CU " << CUIDLineTablePair.first
                << " has line table at " << Offset << "\n");
  }
}

void RewriteInstance::updateFunctionRanges() {
  auto addDebugArangesEntry = [&](const BinaryFunction &Function,
                                  uint64_t RangeBegin,
                                  uint64_t RangeSize) {
    // The function potentially has multiple associated CUs because of
    // the identical code folding optimization. Update all of them with
    // the range.
    for (const auto DIECompileUnitPair : Function.getSubprogramDIEs()) {
      auto CUOffset = DIECompileUnitPair.second->getOffset();
      if (CUOffset != -1U)
        RangesSectionsWriter.AddRange(CUOffset, RangeBegin, RangeSize);
    }
  };

  for (auto &BFI : BinaryFunctions) {
    auto &Function = BFI.second;
    // If function doesn't have registered DIEs - there's nothting to update.
    if (Function.getSubprogramDIEs().empty())
      continue;
    // Use either new (image) or original size for the function range.
    auto Size = Function.isSimple() ? Function.getImageSize()
                                    : Function.getSize();
    addDebugArangesEntry(Function,
                         Function.getAddress(),
                         Size);
    RangesSectionsWriter.AddRange(&Function, Function.getAddress(), Size);
    if (Function.isSimple() && Function.cold().getImageSize()) {
      addDebugArangesEntry(Function,
                           Function.cold().getAddress(),
                           Function.cold().getImageSize());
      RangesSectionsWriter.AddRange(&Function,
                                    Function.cold().getAddress(),
                                    Function.cold().getImageSize());
    }
  }
}

void RewriteInstance::generateDebugRanges() {
  using RangeType = enum { RANGES, ARANGES };
  for (int IntRT = RANGES; IntRT <= ARANGES; ++IntRT) {
    RangeType RT = static_cast<RangeType>(IntRT);
    const char *SectionName = (RT == RANGES) ? ".debug_ranges"
                                             : ".debug_aranges";
    SmallVector<char, 16> RangesBuffer;
    raw_svector_ostream OS(RangesBuffer);

    auto MAB = BC->TheTarget->createMCAsmBackend(*BC->MRI, BC->TripleName, "");
    auto Writer = MAB->createObjectWriter(OS);

    if (RT == RANGES) {
      RangesSectionsWriter.WriteRangesSection(Writer);
    } else {
      RangesSectionsWriter.WriteArangesSection(Writer);
    }
    const auto &DebugRangesContents = OS.str();

    // Free'd by SectionMM.
    uint8_t *SectionData = new uint8_t[DebugRangesContents.size()];
    memcpy(SectionData, DebugRangesContents.data(), DebugRangesContents.size());

    SectionMM->NoteSectionInfo[SectionName] = SectionInfo(
        reinterpret_cast<uint64_t>(SectionData),
        DebugRangesContents.size(),
        /*Alignment=*/0,
        /*IsCode=*/false,
        /*IsReadOnly=*/true);
  }
}

void RewriteInstance::updateLocationLists() {
  // Write new contents to .debug_loc.
  SmallVector<char, 16> DebugLocBuffer;
  raw_svector_ostream OS(DebugLocBuffer);

  auto MAB = BC->TheTarget->createMCAsmBackend(*BC->MRI, BC->TripleName, "");
  auto Writer = MAB->createObjectWriter(OS);

  DebugLocWriter LocationListsWriter;

  for (const auto &Loc : BC->LocationLists) {
    LocationListsWriter.write(Loc, Writer);
  }

  const auto &DebugLocContents = OS.str();

  // Free'd by SectionMM.
  uint8_t *SectionData = new uint8_t[DebugLocContents.size()];
  memcpy(SectionData, DebugLocContents.data(), DebugLocContents.size());

  SectionMM->NoteSectionInfo[".debug_loc"] = SectionInfo(
      reinterpret_cast<uint64_t>(SectionData),
      DebugLocContents.size(),
      /*Alignment=*/0,
      /*IsCode=*/false,
      /*IsReadOnly=*/true);

  // For each CU, update pointers into .debug_loc.
  for (const auto &CU : BC->DwCtx->compile_units()) {
    updateLocationListPointers(
        CU.get(),
        CU->getUnitDIE(false),
        LocationListsWriter.getUpdatedLocationListOffsets());
  }
}

void RewriteInstance::updateLocationListPointers(
    const DWARFUnit *Unit,
    const DWARFDebugInfoEntryMinimal *DIE,
    const std::map<uint32_t, uint32_t> &UpdatedOffsets) {
  // Stop if we're in a non-simple function, which will not be rewritten.
  auto Tag = DIE->getTag();
  if (Tag == dwarf::DW_TAG_subprogram) {
    uint64_t LowPC = -1ULL, HighPC = -1ULL;
    DIE->getLowAndHighPC(Unit, LowPC, HighPC);
    if (LowPC != -1ULL) {
      auto It = BinaryFunctions.find(LowPC);
      if (It != BinaryFunctions.end() && !It->second.isSimple())
        return;
    }
  }
  // If the DIE has a DW_AT_location attribute with a section offset, update it.
  DWARFFormValue Value;
  uint32_t AttrOffset;
  if (DIE->getAttributeValue(Unit, dwarf::DW_AT_location, Value, &AttrOffset) &&
      (Value.isFormClass(DWARFFormValue::FC_Constant) ||
       Value.isFormClass(DWARFFormValue::FC_SectionOffset))) {
    uint64_t DebugLocOffset = -1ULL;
    if (Value.isFormClass(DWARFFormValue::FC_SectionOffset)) {
      DebugLocOffset = Value.getAsSectionOffset().getValue();
    } else if (Value.isFormClass(DWARFFormValue::FC_Constant)) {  // DWARF 3
      DebugLocOffset = Value.getAsUnsignedConstant().getValue();
    }

    auto It = UpdatedOffsets.find(DebugLocOffset);
    if (It != UpdatedOffsets.end()) {
      auto DebugInfoPatcher =
          static_cast<SimpleBinaryPatcher *>(
              SectionPatchers[".debug_info"].get());
      DebugInfoPatcher->addLE32Patch(AttrOffset, It->second + DebugLocSize);
    }
  }

  // Recursively visit children.
  for (auto Child = DIE->getFirstChild(); Child; Child = Child->getSibling()) {
    updateLocationListPointers(Unit, Child, UpdatedOffsets);
  }
}
