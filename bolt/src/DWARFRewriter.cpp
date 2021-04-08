//===--- DWARFRewriter.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "DWARFRewriter.h"
#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "DebugData.h"
#include "ParallelUtilities.h"
#include "Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Timer.h"
#include <algorithm>
#include <llvm/Support/Error.h>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace llvm::support::endian;
using namespace object;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltCategory;
extern cl::opt<unsigned> Verbosity;

static cl::opt<bool>
KeepARanges("keep-aranges",
  cl::desc("keep or generate .debug_aranges section if .gdb_index is written"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<bool>
DeterministicDebugInfo("deterministic-debuginfo",
  cl::desc("disables parallel execution of tasks that may produce"
           "nondeterministic debug info"),
  cl::init(true),
  cl::cat(BoltCategory));

} // namespace opts

void DWARFRewriter::updateDebugInfo() {
  ErrorOr<BinarySection &> DebugAbbrev =
      BC.getUniqueSectionByName(".debug_abbrev");
  ErrorOr<BinarySection &> DebugInfo = BC.getUniqueSectionByName(".debug_info");
  if (DebugAbbrev) {
    DebugAbbrev->registerPatcher(std::make_unique<DebugAbbrevPatcher>());
    AbbrevPatcher =
        static_cast<DebugAbbrevPatcher *>(DebugAbbrev->getPatcher());
  }
  if (DebugInfo) {
    DebugInfo->registerPatcher(std::make_unique<SimpleBinaryPatcher>());
    DebugInfoPatcher =
        static_cast<SimpleBinaryPatcher *>(DebugInfo->getPatcher());
  }

  ARangesSectionWriter = std::make_unique<DebugARangesSectionWriter>();
  RangesSectionWriter = std::make_unique<DebugRangesSectionWriter>(&BC);

  size_t NumCUs = BC.DwCtx->getNumCompileUnits();
  if (opts::NoThreads || opts::DeterministicDebugInfo) {
    // Use single entry for efficiency when running single-threaded
    NumCUs = 1;
  }

  LocListWritersByCU.resize(NumCUs);

  for (size_t CUIndex = 0; CUIndex < NumCUs; ++CUIndex) {
    LocListWritersByCU[CUIndex] = std::make_unique<DebugLocWriter>(&BC);
  }

  auto processUnitDIE = [&](size_t CUIndex, DWARFUnit *Unit) {
    updateUnitDebugInfo(CUIndex, Unit);
  };

  if (opts::NoThreads || opts::DeterministicDebugInfo) {
    for (std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
      processUnitDIE(0, CU.get());
    }
  } else {
    // Update unit debug info in parallel
    ThreadPool &ThreadPool = ParallelUtilities::getThreadPool();
    size_t CUIndex = 0;
    for (std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
      ThreadPool.async(processUnitDIE, CUIndex, CU.get());
      CUIndex++;
    }

    ThreadPool.wait();
  }

  flushPendingRanges();

  finalizeDebugSections();

  updateGdbIndexSection();
}

void DWARFRewriter::updateUnitDebugInfo(size_t CUIndex, DWARFUnit *Unit) {
  // Cache debug ranges so that the offset for identical ranges could be reused.
  std::map<DebugAddressRangesVector, uint64_t> CachedRanges;

  uint64_t DIEOffset = Unit->getOffset() + Unit->getHeaderSize();
  uint64_t NextCUOffset = Unit->getNextUnitOffset();
  DWARFDebugInfoEntry Die;
  DWARFDataExtractor DebugInfoData = Unit->getDebugInfoExtractor();
  uint32_t Depth = 0;
  while (Die.extractFast(*Unit, &DIEOffset, DebugInfoData, NextCUOffset,
                         Depth)) {
    if (const DWARFAbbreviationDeclaration *AbbrDecl =
            Die.getAbbreviationDeclarationPtr()) {
      if (AbbrDecl->hasChildren())
        ++Depth;
    } else {
      // NULL entry.
      if (Depth > 0)
        --Depth;
      if (Depth == 0)
        break;
    }

    DWARFDie DIE(Unit, &Die);
    switch (DIE.getTag()) {
    case dwarf::DW_TAG_compile_unit: {
      auto ModuleRangesOrError = DIE.getAddressRanges();
      if (!ModuleRangesOrError) {
        consumeError(ModuleRangesOrError.takeError());
        break;
      }
      DWARFAddressRangesVector ModuleRanges = *ModuleRangesOrError;
      DebugAddressRangesVector OutputRanges =
          BC.translateModuleAddressRanges(ModuleRanges);
      const uint64_t RangesSectionOffset =
          RangesSectionWriter->addRanges(OutputRanges);
      ARangesSectionWriter->addCURanges(Unit->getOffset(),
                                        std::move(OutputRanges));
      updateDWARFObjectAddressRanges(DIE, RangesSectionOffset);
      break;
    }
    case dwarf::DW_TAG_subprogram: {
      // Get function address either from ranges or [LowPC, HighPC) pair.
      bool UsesRanges = false;
      uint64_t Address;
      uint64_t SectionIndex, HighPC;
      if (!DIE.getLowAndHighPC(Address, HighPC, SectionIndex)) {
        Expected<DWARFAddressRangesVector> RangesOrError =
            DIE.getAddressRanges();
        if (!RangesOrError) {
          consumeError(RangesOrError.takeError());
          break;
        }
        DWARFAddressRangesVector Ranges = *RangesOrError;
        // Not a function definition.
        if (Ranges.empty())
          break;

        Address = Ranges.front().LowPC;
        UsesRanges = true;
      }

      // Clear cached ranges as the new function will have its own set.
      CachedRanges.clear();

      DebugAddressRangesVector FunctionRanges;
      if (const BinaryFunction *Function =
              BC.getBinaryFunctionAtAddress(Address))
        FunctionRanges = Function->getOutputAddressRanges();

      // Update ranges.
      if (UsesRanges) {
        updateDWARFObjectAddressRanges(DIE,
            RangesSectionWriter->addRanges(FunctionRanges));
      } else {
        // Delay conversion of [LowPC, HighPC) into DW_AT_ranges if possible.
        const DWARFAbbreviationDeclaration *Abbrev =
            DIE.getAbbreviationDeclarationPtr();
        assert(Abbrev && "abbrev expected");

        // Create a critical section.
        static std::shared_timed_mutex CriticalSectionMutex;
        std::unique_lock<std::shared_timed_mutex> Lock(CriticalSectionMutex);

        if (FunctionRanges.size() > 1) {
          convertPending(Abbrev);
          // Exit critical section early.
          Lock.unlock();
          convertToRanges(DIE, FunctionRanges);
        } else if (ConvertedRangesAbbrevs.find(Abbrev) !=
                   ConvertedRangesAbbrevs.end()) {
          // Exit critical section early.
          Lock.unlock();
          convertToRanges(DIE, FunctionRanges);
        } else {
          if (FunctionRanges.empty())
            FunctionRanges.emplace_back(DebugAddressRange());
          PendingRanges[Abbrev].emplace_back(
              std::make_pair(DWARFDieWrapper(DIE), FunctionRanges.front()));
        }
      }
      break;
    }
    case dwarf::DW_TAG_lexical_block:
    case dwarf::DW_TAG_inlined_subroutine:
    case dwarf::DW_TAG_try_block:
    case dwarf::DW_TAG_catch_block: {
      uint64_t RangesSectionOffset =
          RangesSectionWriter->getEmptyRangesOffset();
      Expected<DWARFAddressRangesVector> RangesOrError = DIE.getAddressRanges();
      const BinaryFunction *Function = RangesOrError && !RangesOrError->empty()
          ? BC.getBinaryFunctionContainingAddress(RangesOrError->front().LowPC)
          : nullptr;
      if (Function) {
        DebugAddressRangesVector OutputRanges =
            Function->translateInputToOutputRanges(*RangesOrError);
        LLVM_DEBUG(
          if (OutputRanges.empty() != RangesOrError->empty()) {
            dbgs() << "BOLT-DEBUG: problem with DIE at 0x"
                   << Twine::utohexstr(DIE.getOffset()) << " in CU at 0x"
                   << Twine::utohexstr(Unit->getOffset())
                   << '\n';
          }
        );
        RangesSectionOffset = RangesSectionWriter->addRanges(
            std::move(OutputRanges), CachedRanges);
      } else if (!RangesOrError) {
        consumeError(RangesOrError.takeError());
      }
      updateDWARFObjectAddressRanges(DIE, RangesSectionOffset);
      break;
    }
    default: {
      // Handle any tag that can have DW_AT_location attribute.
      DWARFFormValue Value;
      uint64_t AttrOffset;
      if (Optional<DWARFFormValue> V =
              DIE.find(dwarf::DW_AT_location, &AttrOffset)) {
        Value = *V;
        if (Value.isFormClass(DWARFFormValue::FC_Constant) ||
            Value.isFormClass(DWARFFormValue::FC_SectionOffset)) {
          uint64_t Offset = Value.isFormClass(DWARFFormValue::FC_Constant)
                                      ? Value.getAsUnsignedConstant().getValue()
                                      : Value.getAsSectionOffset().getValue();
          DebugLocationsVector InputLL;
          uint64_t BaseAddress = Unit->getBaseAddress()->Address;
          Error E = Unit->getLocationTable().visitLocationList(
              &Offset,
              [&](const DWARFLocationEntry &Entry) {
                switch (Entry.Kind) {
                case dwarf::DW_LLE_end_of_list:
                  return false;
                case dwarf::DW_LLE_base_address:
                  assert(Entry.SectionIndex == SectionedAddress::UndefSection &&
                         "absolute address expected");
                  BaseAddress = Entry.Value0;
                  break;
                case dwarf::DW_LLE_offset_pair:
                  assert(Entry.SectionIndex == SectionedAddress::UndefSection &&
                         "absolute address expected");
                  InputLL.emplace_back(DebugLocationEntry{
                      BaseAddress + Entry.Value0,
                      BaseAddress + Entry.Value1,
                      Entry.Loc});
                  break;
                }
                return true;
              });
          uint64_t OutputLocListOffset = DebugLocWriter::EmptyListTag;
          if (E || InputLL.empty()) {
            errs() << "BOLT-WARNING: empty location list detected at 0x"
                   << Twine::utohexstr(Offset) << " for DIE at 0x"
                   << Twine::utohexstr(DIE.getOffset()) << " in CU at 0x"
                   << Twine::utohexstr(Unit->getOffset())
                   << '\n';
          } else {
            const uint64_t Address = InputLL.front().LowPC;
            if (const BinaryFunction *Function =
                    BC.getBinaryFunctionContainingAddress(Address)) {
              const DebugLocationsVector OutputLL = Function
                  ->translateInputToOutputLocationList(InputLL);
              LLVM_DEBUG(if (OutputLL.empty()) {
                dbgs() << "BOLT-DEBUG: location list translated to an empty "
                          "one at 0x"
                       << Twine::utohexstr(DIE.getOffset()) << " in CU at 0x"
                       << Twine::utohexstr(Unit->getOffset())
                       << '\n';
              });
              OutputLocListOffset =
                LocListWritersByCU[CUIndex]->addList(OutputLL);
            }
          }

          if (OutputLocListOffset != DebugLocWriter::EmptyListTag) {
            std::lock_guard<std::mutex> Lock(LocListDebugInfoPatchesMutex);
            LocListDebugInfoPatches.push_back(
                {AttrOffset, CUIndex, OutputLocListOffset});
          } else {
            std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
            DebugInfoPatcher->addLE32Patch(AttrOffset,
                                           DebugLocWriter::EmptyListOffset);
          }
        } else {
          assert((Value.isFormClass(DWARFFormValue::FC_Exprloc) ||
                  Value.isFormClass(DWARFFormValue::FC_Block)) &&
                 "unexpected DW_AT_location form");
        }
      } else if (Optional<DWARFFormValue> V =
                     DIE.find(dwarf::DW_AT_low_pc, &AttrOffset)) {
        Value = *V;
        const Optional<uint64_t> Result = Value.getAsAddress();
        if (Result.hasValue()) {
          const uint64_t Address = Result.getValue();
          uint64_t NewAddress = 0;
          if (const BinaryFunction *Function =
                  BC.getBinaryFunctionContainingAddress(Address)) {
            NewAddress = Function->translateInputToOutputAddress(Address);
            LLVM_DEBUG(dbgs()
                       << "BOLT-DEBUG: Fixing low_pc 0x"
                       << Twine::utohexstr(Address) << " for DIE with tag "
                       << DIE.getTag() << " to 0x"
                       << Twine::utohexstr(NewAddress) << '\n');
          }

          std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
          DebugInfoPatcher->addLE64Patch(AttrOffset, NewAddress);
        } else if (opts::Verbosity >= 1) {
          errs() << "BOLT-WARNING: unexpected form value for attribute at 0x"
                 << Twine::utohexstr(AttrOffset);
        }
      }
    }
    }
  }

  if (DIEOffset > NextCUOffset) {
    errs() << "BOLT-WARNING: corrupt DWARF detected at 0x"
           << Twine::utohexstr(Unit->getOffset()) << '\n';
  }
}

void DWARFRewriter::updateDWARFObjectAddressRanges(
    const DWARFDie DIE, uint64_t DebugRangesOffset) {

  // Some objects don't have an associated DIE and cannot be updated (such as
  // compiler-generated functions).
  if (!DIE) {
    return;
  }

  const DWARFAbbreviationDeclaration *AbbreviationDecl =
      DIE.getAbbreviationDeclarationPtr();
  if (!AbbreviationDecl) {
    if (opts::Verbosity >= 1) {
      errs() << "BOLT-WARNING: object's DIE doesn't have an abbreviation: "
             << "skipping update. DIE at offset 0x"
             << Twine::utohexstr(DIE.getOffset()) << '\n';
    }
    return;
  }

  if (AbbreviationDecl->findAttributeIndex(dwarf::DW_AT_ranges)) {
    // Case 1: The object was already non-contiguous and had DW_AT_ranges.
    // In this case we simply need to update the value of DW_AT_ranges.
    uint64_t AttrOffset = -1U;
    DIE.find(dwarf::DW_AT_ranges, &AttrOffset);
    assert(AttrOffset != -1U &&  "failed to locate DWARF attribute");

    std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
    DebugInfoPatcher->addLE32Patch(AttrOffset, DebugRangesOffset);
  } else {
    // Case 2: The object has both DW_AT_low_pc and DW_AT_high_pc emitted back
    // to back. We replace the attributes with DW_AT_ranges and DW_AT_low_pc.
    // The low_pc attribute is required for DW_TAG_compile_units to set a base
    // address.
    //
    // Since DW_AT_ranges takes 4-byte DW_FROM_sec_offset value, we have to fill
    // in up to 12-bytes left after removal of low/high pc field from
    // .debug_info.
    //
    // To fill in the gap we use a variable length DW_FORM_udata encoding for
    // DW_AT_low_pc. We exploit the fact that the encoding can take an arbitrary
    // large size.
    if (AbbreviationDecl->findAttributeIndex(dwarf::DW_AT_low_pc) &&
        AbbreviationDecl->findAttributeIndex(dwarf::DW_AT_high_pc)) {
      convertToRanges(AbbreviationDecl);
      convertToRanges(DIE, DebugRangesOffset);
    } else {
      if (opts::Verbosity >= 1) {
        errs() << "BOLT-WARNING: Cannot update ranges for DIE at offset 0x"
               << Twine::utohexstr(DIE.getOffset()) << '\n';
      }
    }
  }
}

void DWARFRewriter::updateLineTableOffsets() {
  const MCSection *LineSection =
    BC.Ctx->getObjectFileInfo()->getDwarfLineSection();
  auto CurrentFragment = LineSection->begin();
  uint64_t CurrentOffset = 0;
  uint64_t Offset = 0;

  ErrorOr<BinarySection &> DbgInfoSection =
      BC.getUniqueSectionByName(".debug_info");
  ErrorOr<BinarySection &> TypeInfoSection =
      BC.getUniqueSectionByName(".debug_types");
  assert(((BC.DwCtx->getNumTypeUnits() > 0 && TypeInfoSection) ||
          BC.DwCtx->getNumTypeUnits() == 0) &&
         "Was not able to retrieve Debug Types section.");

  // There is no direct connection between CU and TU, but same offsets,
  // encoded in DW_AT_stmt_list, into .debug_line get modified.
  // We take advantage of that to map original CU line table offsets to new
  // ones.
  std::unordered_map<uint64_t, uint64_t> DebugLineOffsetMap;

  auto GetStatementListValue = [](DWARFUnit *Unit) {
    Optional<DWARFFormValue> StmtList =
        Unit->getUnitDIE().find(dwarf::DW_AT_stmt_list);
    Optional<uint64_t> Offset = dwarf::toSectionOffset(StmtList);
    assert(Offset && "Was not able to retreive value of DW_AT_stmt_list.");
    return *Offset;
  };

  for (const std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
    const unsigned CUID = CU->getOffset();
    MCSymbol *Label = BC.Ctx->getMCDwarfLineTable(CUID).getLabel();
    if (!Label)
      continue;

    const uint64_t LTOffset =
      BC.DwCtx->getAttrFieldOffsetForUnit(CU.get(), dwarf::DW_AT_stmt_list);
    if (!LTOffset)
      continue;

    // Line tables are stored in MCContext in ascending order of offset in the
    // output file, thus we can compute all table's offset by passing through
    // each fragment at most once, continuing from the last CU's beginning
    // instead of from the first fragment.
    MCFragment *Fragment = Label->getFragment();
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

    DebugLineOffsetMap[GetStatementListValue(CU.get())] = Offset;
    assert(DbgInfoSection && ".debug_info section must exist");
    DbgInfoSection->addRelocation(LTOffset,
                                  nullptr,
                                  ELF::R_X86_64_32,
                                  Offset,
                                  0,
                                  /*Pending=*/true);

    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: CU " << CUID
                      << " has line table at " << Offset << "\n");
  }

  for (const std::unique_ptr<DWARFUnit> &TU : BC.DwCtx->types_section_units()) {
    DWARFUnit *Unit = TU.get();
    const uint64_t LTOffset =
        BC.DwCtx->getAttrFieldOffsetForUnit(Unit, dwarf::DW_AT_stmt_list);
    if (!LTOffset)
      continue;
    auto Iter = DebugLineOffsetMap.find(GetStatementListValue(Unit));
    assert(Iter != DebugLineOffsetMap.end() &&
           "Type Unit Updated Line Number Entry does not exist.");
    TypeInfoSection->addRelocation(LTOffset, nullptr, ELF::R_X86_64_32,
                                   Iter->second, 0, /*Pending=*/true);
  }

  // Set .debug_info as finalized so it won't be skipped over when
  // we process sections while writing out the new binary.  This ensures
  // that the pending relocations will be processed and not ignored.
  if(DbgInfoSection)
    DbgInfoSection->setIsFinalized();

  if (TypeInfoSection)
    TypeInfoSection->setIsFinalized();
}

void DWARFRewriter::finalizeDebugSections() {
  // Skip .debug_aranges if we are re-generating .gdb_index.
  if (opts::KeepARanges || !BC.getGdbIndexSection()) {
    SmallVector<char, 16> ARangesBuffer;
    raw_svector_ostream OS(ARangesBuffer);

    auto MAB = std::unique_ptr<MCAsmBackend>(BC.TheTarget->createMCAsmBackend(
        *BC.STI, *BC.MRI, MCTargetOptions()));

    ARangesSectionWriter->writeARangesSection(OS);
    const StringRef &ARangesContents = OS.str();

    BC.registerOrUpdateNoteSection(".debug_aranges",
                                    copyByteArray(ARangesContents),
                                    ARangesContents.size());
  }

  std::unique_ptr<RangesBufferVector> RangesSectionContents =
      RangesSectionWriter->finalize();
  BC.registerOrUpdateNoteSection(".debug_ranges",
                                  copyByteArray(*RangesSectionContents),
                                  RangesSectionContents->size());

  std::unique_ptr<LocBufferVector> LocationListSectionContents =
      makeFinalLocListsSection();
  BC.registerOrUpdateNoteSection(".debug_loc",
                                  copyByteArray(*LocationListSectionContents),
                                  LocationListSectionContents->size());
}

void DWARFRewriter::updateGdbIndexSection() {
  if (!BC.getGdbIndexSection())
    return;

  // See https://sourceware.org/gdb/onlinedocs/gdb/Index-Section-Format.html for
  // .gdb_index section format.

  StringRef GdbIndexContents = BC.getGdbIndexSection()->getContents();

  const char *Data = GdbIndexContents.data();

  // Parse the header.
  const uint32_t Version = read32le(Data);
  if (Version != 7 && Version != 8) {
    errs() << "BOLT-ERROR: can only process .gdb_index versions 7 and 8\n";
    exit(1);
  }

  // Some .gdb_index generators use file offsets while others use section
  // offsets. Hence we can only rely on offsets relative to each other,
  // and ignore their absolute values.
  const uint32_t CUListOffset = read32le(Data + 4);
  const uint32_t CUTypesOffset = read32le(Data + 8);
  const uint32_t AddressTableOffset = read32le(Data + 12);
  const uint32_t SymbolTableOffset = read32le(Data + 16);
  const uint32_t ConstantPoolOffset = read32le(Data + 20);
  Data += 24;

  // Map CUs offsets to indices and verify existing index table.
  std::map<uint32_t, uint32_t> OffsetToIndexMap;
  const uint32_t CUListSize = CUTypesOffset - CUListOffset;
  const unsigned NumCUs = BC.DwCtx->getNumCompileUnits();
  if (CUListSize != NumCUs * 16) {
    errs() << "BOLT-ERROR: .gdb_index: CU count mismatch\n";
    exit(1);
  }
  for (unsigned Index = 0; Index < NumCUs; ++Index, Data += 16) {
    const DWARFUnit *CU = BC.DwCtx->getUnitAtIndex(Index);
    const uint64_t Offset = read64le(Data);
    if (CU->getOffset() != Offset) {
      errs() << "BOLT-ERROR: .gdb_index CU offset mismatch\n";
      exit(1);
    }

    OffsetToIndexMap[Offset] = Index;
  }

  // Ignore old address table.
  const uint32_t OldAddressTableSize = SymbolTableOffset - AddressTableOffset;
  // Move Data to the beginning of symbol table.
  Data += SymbolTableOffset - CUTypesOffset;

  // Calculate the size of the new address table.
  uint32_t NewAddressTableSize = 0;
  for (const auto &CURangesPair : ARangesSectionWriter->getCUAddressRanges()) {
    const SmallVector<DebugAddressRange, 2> &Ranges = CURangesPair.second;
    NewAddressTableSize += Ranges.size() * 20;
  }

  // Difference between old and new table (and section) sizes.
  // Could be negative.
  int32_t Delta = NewAddressTableSize - OldAddressTableSize;

  size_t NewGdbIndexSize = GdbIndexContents.size() + Delta;

  // Free'd by ExecutableFileMemoryManager.
  auto *NewGdbIndexContents = new uint8_t[NewGdbIndexSize];
  uint8_t *Buffer = NewGdbIndexContents;

  write32le(Buffer, Version);
  write32le(Buffer + 4, CUListOffset);
  write32le(Buffer + 8, CUTypesOffset);
  write32le(Buffer + 12, AddressTableOffset);
  write32le(Buffer + 16, SymbolTableOffset + Delta);
  write32le(Buffer + 20, ConstantPoolOffset + Delta);
  Buffer += 24;

  // Copy over CU list and types CU list.
  memcpy(Buffer, GdbIndexContents.data() + 24,
         AddressTableOffset - CUListOffset);
  Buffer += AddressTableOffset - CUListOffset;

  // Generate new address table.
  for (const std::pair<const uint64_t, DebugAddressRangesVector> &CURangesPair :
       ARangesSectionWriter->getCUAddressRanges()) {
    const uint32_t CUIndex = OffsetToIndexMap[CURangesPair.first];
    const DebugAddressRangesVector &Ranges = CURangesPair.second;
    for (const DebugAddressRange &Range : Ranges) {
      write64le(Buffer, Range.LowPC);
      write64le(Buffer + 8, Range.HighPC);
      write32le(Buffer + 16, CUIndex);
      Buffer += 20;
    }
  }

  const size_t TrailingSize =
      GdbIndexContents.data() + GdbIndexContents.size() - Data;
  assert(Buffer + TrailingSize == NewGdbIndexContents + NewGdbIndexSize &&
         "size calculation error");

  // Copy over the rest of the original data.
  memcpy(Buffer, Data, TrailingSize);

  // Register the new section.
  BC.registerOrUpdateNoteSection(".gdb_index",
                                  NewGdbIndexContents,
                                  NewGdbIndexSize);
}

void
DWARFRewriter::convertToRanges(const DWARFAbbreviationDeclaration *Abbrev) {
  if (!AbbrevPatcher)
    return;
  dwarf::Form HighPCForm = Abbrev->findAttribute(dwarf::DW_AT_high_pc)->Form;
  std::lock_guard<std::mutex> Lock(AbbrevPatcherMutex);
  AbbrevPatcher->addAttributePatch(Abbrev,
                                   dwarf::DW_AT_low_pc,
                                   dwarf::DW_AT_ranges,
                                   dwarf::DW_FORM_sec_offset);
  if (HighPCForm == dwarf::DW_FORM_addr ||
      HighPCForm == dwarf::DW_FORM_data8) {
    // LowPC must have 12 bytes, use indirect
    AbbrevPatcher->addAttributePatch(Abbrev,
                                     dwarf::DW_AT_high_pc,
                                     dwarf::DW_AT_low_pc,
                                     dwarf::DW_FORM_indirect);
  } else if (HighPCForm == dwarf::DW_FORM_data4) {
    // LowPC must have 8 bytes, use addr
    AbbrevPatcher->addAttributePatch(Abbrev,
                                     dwarf::DW_AT_high_pc,
                                     dwarf::DW_AT_low_pc,
                                     dwarf::DW_FORM_addr);
  } else {
    llvm_unreachable("unexpected form");
  }
}

void DWARFRewriter::convertToRanges(DWARFDie DIE,
                                    const DebugAddressRangesVector &Ranges) {
  uint64_t RangesSectionOffset;
  if (Ranges.empty()) {
    RangesSectionOffset = RangesSectionWriter->getEmptyRangesOffset();
  } else {
    RangesSectionOffset = RangesSectionWriter->addRanges(Ranges);
  }

  convertToRanges(DIE, RangesSectionOffset);
}

void DWARFRewriter::convertPending(const DWARFAbbreviationDeclaration *Abbrev) {
  if (ConvertedRangesAbbrevs.count(Abbrev))
    return;

  convertToRanges(Abbrev);

  auto I = PendingRanges.find(Abbrev);
  if (I != PendingRanges.end()) {
    for (std::pair<DWARFDieWrapper, DebugAddressRange> &Pair : I->second) {
      convertToRanges(Pair.first, {Pair.second});
    }
    PendingRanges.erase(I);
  }

  ConvertedRangesAbbrevs.emplace(Abbrev);
}

std::unique_ptr<LocBufferVector> DWARFRewriter::makeFinalLocListsSection() {
  auto LocBuffer = std::make_unique<LocBufferVector>();
  auto LocStream = std::make_unique<raw_svector_ostream>(*LocBuffer);
  auto Writer =
    std::unique_ptr<MCObjectWriter>(BC.createObjectWriter(*LocStream));

  uint64_t SectionOffset = 0;

  // Add an empty list as the first entry;
  const char Zeroes[16] = {0};
  *LocStream << StringRef(Zeroes, 16);
  SectionOffset += 2 * 8;

  std::vector<uint64_t> SectionOffsetByCU(LocListWritersByCU.size());

  for (size_t CUIndex = 0; CUIndex < LocListWritersByCU.size(); ++CUIndex) {
    SectionOffsetByCU[CUIndex] = SectionOffset;
    std::unique_ptr<LocBufferVector> CurrCULocationLists =
        LocListWritersByCU[CUIndex]->finalize();
    *LocStream << *CurrCULocationLists;
    SectionOffset += CurrCULocationLists->size();
  }

  for (LocListDebugInfoPatchType &Patch : LocListDebugInfoPatches) {
    DebugInfoPatcher
      ->addLE32Patch(Patch.DebugInfoOffset,
                     SectionOffsetByCU[Patch.CUIndex] + Patch.CUWriterOffset);
  }

  return LocBuffer;
}

void DWARFRewriter::flushPendingRanges() {
  for (std::pair<const DWARFAbbreviationDeclaration *const,
                 std::vector<std::pair<DWARFDieWrapper, DebugAddressRange>>>
           &I : PendingRanges) {
    for (std::pair<DWARFDieWrapper, DebugAddressRange> &RangePair : I.second) {
      patchLowHigh(RangePair.first, RangePair.second);
    }
  }
  clearList(PendingRanges);
}

namespace {

void getRangeAttrData(
    DWARFDie DIE,
    uint64_t &LowPCOffset, uint64_t &HighPCOffset,
    DWARFFormValue &LowPCFormValue, DWARFFormValue &HighPCFormValue) {
  LowPCOffset = -1U;
  HighPCOffset = -1U;
  LowPCFormValue = *DIE.find(dwarf::DW_AT_low_pc, &LowPCOffset);
  HighPCFormValue = *DIE.find(dwarf::DW_AT_high_pc, &HighPCOffset);

  if (LowPCFormValue.getForm() != dwarf::DW_FORM_addr ||
      (HighPCFormValue.getForm() != dwarf::DW_FORM_addr &&
       HighPCFormValue.getForm() != dwarf::DW_FORM_data8 &&
       HighPCFormValue.getForm() != dwarf::DW_FORM_data4)) {
    errs() << "BOLT-WARNING: unexpected form value. Cannot update DIE "
             << "at offset 0x" << Twine::utohexstr(DIE.getOffset()) << "\n";
    return;
  }
  if (LowPCOffset == -1U || (LowPCOffset + 8 != HighPCOffset)) {
    errs() << "BOLT-WARNING: high_pc expected immediately after low_pc. "
           << "Cannot update DIE at offset 0x"
           << Twine::utohexstr(DIE.getOffset()) << '\n';
    return;
  }
}

}

void DWARFRewriter::patchLowHigh(DWARFDie DIE, DebugAddressRange Range) {
  if (!DebugInfoPatcher)
    return;
  uint64_t LowPCOffset, HighPCOffset;
  DWARFFormValue LowPCFormValue, HighPCFormValue;
  getRangeAttrData(
      DIE, LowPCOffset, HighPCOffset, LowPCFormValue, HighPCFormValue);
  DebugInfoPatcher->addLE64Patch(LowPCOffset, Range.LowPC);
  if (HighPCFormValue.getForm() == dwarf::DW_FORM_addr ||
      HighPCFormValue.getForm() == dwarf::DW_FORM_data8) {
    DebugInfoPatcher->addLE64Patch(HighPCOffset, Range.HighPC - Range.LowPC);
  } else {
    DebugInfoPatcher->addLE32Patch(HighPCOffset, Range.HighPC - Range.LowPC);
  }
}

void DWARFRewriter::convertToRanges(DWARFDie DIE,
                                    uint64_t RangesSectionOffset) {
  if (!DebugInfoPatcher)
    return;
  uint64_t LowPCOffset, HighPCOffset;
  DWARFFormValue LowPCFormValue, HighPCFormValue;
  getRangeAttrData(
      DIE, LowPCOffset, HighPCOffset, LowPCFormValue, HighPCFormValue);

  unsigned LowPCSize = 0;
  assert(DIE.getDwarfUnit()->getAddressByteSize() == 8);
  if (HighPCFormValue.getForm() == dwarf::DW_FORM_addr ||
      HighPCFormValue.getForm() == dwarf::DW_FORM_data8) {
    LowPCSize = 12;
  } else if (HighPCFormValue.getForm() == dwarf::DW_FORM_data4) {
    LowPCSize = 8;
  } else {
    llvm_unreachable("unexpected form");
  }

  std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
  DebugInfoPatcher->addLE32Patch(LowPCOffset, RangesSectionOffset);
  if (LowPCSize == 12) {
    // Write an indirect 0 value for DW_AT_low_pc so that we can fill
    // 12 bytes of space (see T56239836 for more details)
    DebugInfoPatcher->addUDataPatch(LowPCOffset + 4, dwarf::DW_FORM_addr, 4);
    DebugInfoPatcher->addLE64Patch(LowPCOffset + 8, 0);
  } else {
    DebugInfoPatcher->addLE64Patch(LowPCOffset + 4, 0);
  }
}
