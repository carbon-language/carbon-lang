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
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

LLVM_ATTRIBUTE_UNUSED
static void printDie(const DWARFDie &DIE) {
  DIDumpOptions DumpOpts;
  DumpOpts.ShowForm = true;
  DumpOpts.Verbose = true;
  DumpOpts.ChildRecurseDepth = 0;
  DumpOpts.ShowChildren = 0;
  DIE.dump(dbgs(), 0, DumpOpts);
}

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

static cl::opt<std::string>
    DwoOutputPath("dwo-output-path",
                  cl::desc("Path to where .dwo files will be written out to."),
                  cl::init(""), cl::cat(BoltCategory));

static cl::opt<bool>
    DebugSkeletonCu("debug-skeleton-cu",
                    cl::desc("Prints out offsetrs for abbrev and debu_info of "
                             "Skeleton CUs that get patched."),
                    cl::ZeroOrMore, cl::Hidden, cl::init(false),
                    cl::cat(BoltCategory));
} // namespace opts

/// Returns DWO Name to be used. Handles case where user specifies output DWO
/// directory, and there are duplicate names. Assumes DWO ID is unique.
static std::string
getDWOName(llvm::DWARFUnit &CU,
           std::unordered_map<std::string, uint32_t> *NameToIndexMap,
           std::unordered_map<uint64_t, std::string> &DWOIdToName) {
  llvm::Optional<uint64_t> DWOId = CU.getDWOId();
  assert(DWOId && "DWO ID not found.");
  (void)DWOId;
  auto NameIter = DWOIdToName.find(*DWOId);
  if (NameIter != DWOIdToName.end())
    return NameIter->second;

  std::string DWOName = dwarf::toString(
      CU.getUnitDIE().find({dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}),
      "");
  assert(!DWOName.empty() &&
         "DW_AT_dwo_name/DW_AT_GNU_dwo_name does not exists.");
  if (NameToIndexMap && !opts::DwoOutputPath.empty()) {
    auto Iter = NameToIndexMap->find(DWOName);
    if (Iter == NameToIndexMap->end()) {
      Iter = NameToIndexMap->insert({DWOName, 0}).first;
    }
    DWOName.append(std::to_string(Iter->second));
    ++Iter->second;
  }
  DWOName.append(".dwo");
  DWOIdToName[*DWOId] = DWOName;
  return DWOName;
}

static bool isHighPcFormEightBytes(dwarf::Form DwarfForm) {
  return DwarfForm == dwarf::DW_FORM_addr || DwarfForm == dwarf::DW_FORM_data8;
}

void DWARFRewriter::updateDebugInfo() {
  ErrorOr<BinarySection &> DebugAbbrev =
      BC.getUniqueSectionByName(".debug_abbrev");
  ErrorOr<BinarySection &> DebugInfo = BC.getUniqueSectionByName(".debug_info");

  if (!DebugAbbrev || !DebugInfo)
    return;

  DebugAbbrev->registerPatcher(std::make_unique<DebugAbbrevPatcher>());
  auto *AbbrevPatcher =
      static_cast<DebugAbbrevPatcher *>(DebugAbbrev->getPatcher());

  DebugInfo->registerPatcher(std::make_unique<SimpleBinaryPatcher>());
  auto *DebugInfoPatcher =
      static_cast<SimpleBinaryPatcher *>(DebugInfo->getPatcher());

  ARangesSectionWriter = std::make_unique<DebugARangesSectionWriter>();
  RangesSectionWriter = std::make_unique<DebugRangesSectionWriter>();
  StrWriter = std::make_unique<DebugStrWriter>(&BC);

  AddrWriter = std::make_unique<DebugAddrWriter>(&BC);
  DebugLoclistWriter::setAddressWriter(AddrWriter.get());

  uint64_t NumCUs = BC.DwCtx->getNumCompileUnits();
  if ((opts::NoThreads || opts::DeterministicDebugInfo) &&
      BC.getNumDWOCUs() == 0) {
    // Use single entry for efficiency when running single-threaded
    NumCUs = 1;
  }

  LocListWritersByCU.reserve(NumCUs);

  for (size_t CUIndex = 0; CUIndex < NumCUs; ++CUIndex) {
    LocListWritersByCU[CUIndex] = std::make_unique<DebugLocWriter>(&BC);
  }
  // Unordered maps to handle name collision if output DWO directory is
  // specified.
  std::unordered_map<std::string, uint32_t> NameToIndexMap;
  std::unordered_map<uint64_t, std::string> DWOIdToName;

  auto updateDWONameCompDir = [&](DWARFUnit &Unit) -> void {
    const DWARFDie &DIE = Unit.getUnitDIE();
    uint64_t AttrOffset = 0;
    Optional<DWARFFormValue> ValDwoName =
        DIE.find(dwarf::DW_AT_GNU_dwo_name, &AttrOffset);
    assert(ValDwoName && "Skeleton CU doesn't have dwo_name.");

    std::string ObjectName = getDWOName(Unit, &NameToIndexMap, DWOIdToName);
    uint32_t NewOffset = StrWriter->addString(ObjectName.c_str());
    DebugInfoPatcher->addLE32Patch(AttrOffset, NewOffset);

    Optional<DWARFFormValue> ValCompDir =
        DIE.find(dwarf::DW_AT_comp_dir, &AttrOffset);
    assert(ValCompDir && "DW_AT_comp_dir is not in Skeleton CU.");
    if (!opts::DwoOutputPath.empty()) {
      uint32_t NewOffset = StrWriter->addString(opts::DwoOutputPath.c_str());
      DebugInfoPatcher->addLE32Patch(AttrOffset, NewOffset);
    }
  };

  uint32_t AbbrevOffsetModifier = 0;
  // Case 1) Range_base found: patch .debug_info
  // Case 2) Range_base not found, but Ranges will be used: patch
  // .debug_info/.debug_abbrev
  auto updateRangeBase = [&](DWARFUnit &Unit, uint64_t RangeBase,
                             bool WasRangeBaseUsed) -> void {
    uint64_t AttrOffset = 0;
    DWARFDie DIE = Unit.getUnitDIE();
    Optional<DWARFFormValue> ValRangeBase =
        DIE.find(dwarf::DW_AT_GNU_ranges_base, &AttrOffset);
    bool NeedToPatch = ValRangeBase.hasValue();
    uint32_t PrevAbbrevOffsetModifier = AbbrevOffsetModifier;
    // Case where Skeleton CU doesn't have DW_AT_GNU_ranges_base
    if (!NeedToPatch && WasRangeBaseUsed) {
      const DWARFAbbreviationDeclaration *AbbreviationDecl =
          DIE.getAbbreviationDeclarationPtr();
      if (Optional<DWARFFormValue> ValLowPC =
              DIE.find(dwarf::DW_AT_low_pc, &AttrOffset)) {

        Optional<DWARFFormValue> ValHighPC = DIE.find(dwarf::DW_AT_high_pc);
        uint32_t NumBytesToFill = 7;

        AbbrevPatcher->addAttributePatch(AbbreviationDecl, dwarf::DW_AT_low_pc,
                                         dwarf::DW_AT_GNU_ranges_base,
                                         dwarf::DW_FORM_indirect);
        // Bolt converts DW_AT_low_pc/DW_AT_high_pc to DW_AT_low_pc/DW_at_ranges
        // DW_AT_high_pc can be 4 or 8 bytes. If it's 8 bytes need to use first
        // 4 bytes.
        if (ValHighPC && isHighPcFormEightBytes(ValHighPC->getForm())) {
          NumBytesToFill += 4;
        }
        LLVM_DEBUG(if (opts::DebugSkeletonCu) dbgs()
                       << "AttrOffset: " << Twine::utohexstr(AttrOffset) << "\n"
                       << "Die Offset: " << Twine::utohexstr(DIE.getOffset())
                       << "\n"
                       << "AbbrDecl offfset: "
                       << Twine::utohexstr(Unit.getAbbrOffset()) << "\n"
                       << "Unit Offset: " << Twine::utohexstr(Unit.getOffset())
                       << "\n\n";);
        DebugInfoPatcher->addUDataPatch(AttrOffset, dwarf::DW_FORM_udata, 1);
        DebugInfoPatcher->addUDataPatch(AttrOffset + 1, RangeBase,
                                        NumBytesToFill);

        // 1 Byte for DW_AT_GNU_ranges_base (since it's 2 bytes vs DW_AT_low_pc)
        AbbrevOffsetModifier += 1;
      } else {
        errs() << "BOLT-WARNING: Skeleton CU at 0x"
               << Twine::utohexstr(DIE.getOffset())
               << " doesn't have DW_AT_GNU_ranges_base, or "
                  "DW_AT_low_pc to convert\n";
        return;
      }
    }
    if (NeedToPatch)
      DebugInfoPatcher->addLE32Patch(AttrOffset,
                                     static_cast<uint32_t>(RangeBase));

    // DWARF4
    // unit_length - 4 bytes
    // version - 2 bytes
    // So + 6 to patch debug_abbrev_offset
    if (PrevAbbrevOffsetModifier)
      DebugInfoPatcher->addLE32Patch(
          Unit.getOffset() + 6, static_cast<uint32_t>(Unit.getAbbrOffset()) +
                                    PrevAbbrevOffsetModifier);
  };

  auto processUnitDIE = [&](size_t CUIndex, DWARFUnit *Unit) {
    uint64_t RangeBase = RangesSectionWriter->getSectionOffset();
    updateUnitDebugInfo(CUIndex, *Unit, *DebugInfoPatcher, *AbbrevPatcher);
    if (llvm::Optional<uint64_t> DWOId = Unit->getDWOId()) {
      Optional<DWARFUnit *> CU = BC.getDWOCU(*DWOId);
      if (CU) {
        updateDWONameCompDir(*Unit);
        // Assuming there is unique DWOID per binary. i.e. Two or more CUs don't
        // have same DWO ID.
        assert(LocListWritersByCU.count(*DWOId) == 0 &&
               "LocList writer for DWO unit already exists.");
        LocListWritersByCU[*DWOId] =
            std::make_unique<DebugLoclistWriter>(&BC, *DWOId);
        SimpleBinaryPatcher *DwoDebugInfoPatcher =
            getBinaryDWODebugInfoPatcher(*DWOId);
        DwoDebugInfoPatcher->setRangeBase(RangeBase);
        updateUnitDebugInfo(*DWOId, *(*CU), *DwoDebugInfoPatcher,
                            *getBinaryDWOAbbrevPatcher(*DWOId));
        static_cast<DebugLoclistWriter *>(LocListWritersByCU[*DWOId].get())
            ->finalizePatches();
        updateRangeBase(*Unit, RangeBase,
                        DwoDebugInfoPatcher->getWasRangBasedUsed());
      }
    }
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

  flushPendingRanges(*DebugInfoPatcher);

  finalizeDebugSections(*DebugInfoPatcher);

  writeOutDWOFiles(DWOIdToName);

  updateGdbIndexSection();
}

void DWARFRewriter::updateUnitDebugInfo(uint64_t CUIndex, DWARFUnit &Unit,
                                        SimpleBinaryPatcher &DebugInfoPatcher,
                                        DebugAbbrevPatcher &AbbrevPatcher) {
  // Cache debug ranges so that the offset for identical ranges could be reused.
  std::map<DebugAddressRangesVector, uint64_t> CachedRanges;

  auto &DebugLocWriter = *LocListWritersByCU[CUIndex].get();

  uint64_t DIEOffset = Unit.getOffset() + Unit.getHeaderSize();
  uint64_t NextCUOffset = Unit.getNextUnitOffset();
  DWARFDebugInfoEntry Die;
  DWARFDataExtractor DebugInfoData = Unit.getDebugInfoExtractor();
  uint32_t Depth = 0;

  while (
      Die.extractFast(Unit, &DIEOffset, DebugInfoData, NextCUOffset, Depth)) {
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

    DWARFDie DIE(&Unit, &Die);

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
      ARangesSectionWriter->addCURanges(Unit.getOffset(),
                                        std::move(OutputRanges));
      updateDWARFObjectAddressRanges(DIE, RangesSectionOffset, DebugInfoPatcher,
                                     AbbrevPatcher);
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
        updateDWARFObjectAddressRanges(
            DIE, RangesSectionWriter->addRanges(FunctionRanges),
            DebugInfoPatcher, AbbrevPatcher);
      } else {
        // Delay conversion of [LowPC, HighPC) into DW_AT_ranges if possible.
        const DWARFAbbreviationDeclaration *Abbrev =
            DIE.getAbbreviationDeclarationPtr();
        assert(Abbrev && "abbrev expected");

        // Create a critical section.
        static std::shared_timed_mutex CriticalSectionMutex;
        std::unique_lock<std::shared_timed_mutex> Lock(CriticalSectionMutex);

        if (FunctionRanges.size() > 1) {
          convertPending(Abbrev, DebugInfoPatcher, AbbrevPatcher);
          // Exit critical section early.
          Lock.unlock();
          convertToRanges(DIE, FunctionRanges, DebugInfoPatcher);
        } else if (ConvertedRangesAbbrevs.find(Abbrev) !=
                   ConvertedRangesAbbrevs.end()) {
          // Exit critical section early.
          Lock.unlock();
          convertToRanges(DIE, FunctionRanges, DebugInfoPatcher);
        } else {
          if (FunctionRanges.empty())
            FunctionRanges.emplace_back(DebugAddressRange());
          addToPendingRanges(Abbrev, DIE, FunctionRanges, Unit.getDWOId());
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
        LLVM_DEBUG(if (OutputRanges.empty() != RangesOrError->empty()) {
          dbgs() << "BOLT-DEBUG: problem with DIE at 0x"
                 << Twine::utohexstr(DIE.getOffset()) << " in CU at 0x"
                 << Twine::utohexstr(Unit.getOffset()) << '\n';
        });
        RangesSectionOffset = RangesSectionWriter->addRanges(
            std::move(OutputRanges), CachedRanges);
      } else if (!RangesOrError) {
        consumeError(RangesOrError.takeError());
      }
      updateDWARFObjectAddressRanges(DIE, RangesSectionOffset, DebugInfoPatcher,
                                     AbbrevPatcher);
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

          Optional<object::SectionedAddress> SectionAddress =
              Unit.getBaseAddress();
          uint64_t BaseAddress = 0;
          if (SectionAddress)
            BaseAddress = SectionAddress->Address;

          Error E = Unit.getLocationTable().visitLocationList(
              &Offset, [&](const DWARFLocationEntry &Entry) {
                switch (Entry.Kind) {
                default:
                  llvm_unreachable("Unsupported DWARFLocationEntry Kind.");
                case dwarf::DW_LLE_end_of_list:
                  return false;
                case dwarf::DW_LLE_base_address:
                  assert(Entry.SectionIndex == SectionedAddress::UndefSection &&
                         "absolute address expected");
                  BaseAddress = Entry.Value0;
                  break;
                case dwarf::DW_LLE_offset_pair:
                  assert(
                      (Entry.SectionIndex == SectionedAddress::UndefSection &&
                       !Unit.isDWOUnit()) &&
                      "absolute address expected");
                  InputLL.emplace_back(DebugLocationEntry{
                      BaseAddress + Entry.Value0, BaseAddress + Entry.Value1,
                      Entry.Loc});
                  break;
                case dwarf::DW_LLE_startx_length:
                  assert(Unit.isDWOUnit() &&
                         "None DWO Unit with DW_LLE_startx_length encoding.");
                  Optional<object::SectionedAddress> EntryAddress =
                      Unit.getAddrOffsetSectionItem(Entry.Value0);
                  assert(EntryAddress && "Address does not exist.");
                  InputLL.emplace_back(DebugLocationEntry{
                      EntryAddress->Address,
                      EntryAddress->Address + Entry.Value1, Entry.Loc});
                  break;
                }
                return true;
              });

          uint64_t OutputLocListOffset = DebugLocWriter::EmptyListTag;
          if (E || InputLL.empty()) {
            errs() << "BOLT-WARNING: empty location list detected at 0x"
                   << Twine::utohexstr(Offset) << " for DIE at 0x"
                   << Twine::utohexstr(DIE.getOffset()) << " in CU at 0x"
                   << Twine::utohexstr(Unit.getOffset()) << '\n';
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
                       << Twine::utohexstr(Unit.getOffset()) << '\n';
              });
              OutputLocListOffset = DebugLocWriter.addList(OutputLL);
            }
          }

          if (OutputLocListOffset != DebugLocWriter::EmptyListTag) {
            std::lock_guard<std::mutex> Lock(LocListDebugInfoPatchesMutex);
            if (Unit.isDWOUnit()) {
              // Not sure if better approach is to hide all of this away in a
              // class. Also re-using LocListDebugInfoPatchType. Wasting some
              // space for DWOID/CUIndex.
              DwoLocListDebugInfoPatches[CUIndex].push_back(
                  {AttrOffset, CUIndex, OutputLocListOffset});
            } else {
              LocListDebugInfoPatches.push_back(
                  {AttrOffset, CUIndex, OutputLocListOffset});
            }
          } else {
            std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
            DebugInfoPatcher.addLE32Patch(AttrOffset,
                                          DebugLocWriter::EmptyListOffset);
          }
        } else {
          assert((Value.isFormClass(DWARFFormValue::FC_Exprloc) ||
                  Value.isFormClass(DWARFFormValue::FC_Block)) &&
                 "unexpected DW_AT_location form");
          if (Unit.isDWOUnit()) {
            ArrayRef<uint8_t> Expr = *Value.getAsBlock();
            DataExtractor Data(
                StringRef((const char *)Expr.data(), Expr.size()),
                Unit.getContext().isLittleEndian(), 0);
            DWARFExpression LocExpr(Data, Unit.getAddressByteSize(),
                                    Unit.getFormParams().Format);
            for (auto &Expr : LocExpr) {
              if (Expr.getCode() != dwarf::DW_OP_GNU_addr_index)
                continue;
              uint64_t Index = Expr.getRawOperand(0);
              Optional<object::SectionedAddress> EntryAddress =
                  Unit.getAddrOffsetSectionItem(Index);
              assert(EntryAddress && "Address is not found.");
              assert(Index <= std::numeric_limits<uint32_t>::max() &&
                     "Invalid Operand Index.");
              AddrWriter->addIndexAddress(EntryAddress->Address,
                                          static_cast<uint32_t>(Index),
                                          *Unit.getDWOId());
            }
          }
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

          dwarf::Form Form = Value.getForm();
          assert(Form != dwarf::DW_FORM_LLVM_addrx_offset &&
                 "DW_FORM_LLVM_addrx_offset is not supported");
          std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
          if (Form == dwarf::DW_FORM_GNU_addr_index) {
            assert(Unit.isDWOUnit() &&
                   "DW_FORM_GNU_addr_index in Non DWO unit.");
            uint64_t Index = Value.getRawUValue();
            // If there is no new address, storing old address.
            // Re-using Index to make implementation easier.
            // DW_FORM_GNU_addr_index is variable lenght encoding so we either
            // have to create indices of same sizes, or use same index.
            AddrWriter->addIndexAddress(NewAddress ? NewAddress : Address,
                                        Index, *Unit.getDWOId());
          } else {
            DebugInfoPatcher.addLE64Patch(AttrOffset, NewAddress);
          }
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
           << Twine::utohexstr(Unit.getOffset()) << '\n';
  }
}

void DWARFRewriter::updateDWARFObjectAddressRanges(
    const DWARFDie DIE, uint64_t DebugRangesOffset,
    SimpleBinaryPatcher &DebugInfoPatcher, DebugAbbrevPatcher &AbbrevPatcher) {

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
    assert(AttrOffset != -1U && "failed to locate DWARF attribute");

    std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
    DebugInfoPatcher.addLE32Patch(
        AttrOffset, DebugRangesOffset - DebugInfoPatcher.getRangeBase());
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
      convertToRanges(AbbreviationDecl, AbbrevPatcher);
      convertToRanges(DIE, DebugRangesOffset, DebugInfoPatcher);
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

void DWARFRewriter::finalizeDebugSections(
    SimpleBinaryPatcher &DebugInfoPatcher) {
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

  if (StrWriter->isInitialized()) {
    RewriteInstance::addToDebugSectionsToOverwrite(".debug_str");
    std::unique_ptr<DebugStrBufferVector> DebugStrSectionContents =
        StrWriter->finalize();
    BC.registerOrUpdateNoteSection(".debug_str",
                                   copyByteArray(*DebugStrSectionContents),
                                   DebugStrSectionContents->size());
  }

  if (AddrWriter->isInitialized()) {
    AddressSectionBuffer AddressSectionContents = AddrWriter->finalize();
    BC.registerOrUpdateNoteSection(".debug_addr",
                                   copyByteArray(AddressSectionContents),
                                   AddressSectionContents.size());
    for (auto &CU : BC.DwCtx->compile_units()) {
      DWARFDie DIE = CU->getUnitDIE();
      uint64_t AttrOffset = 0;
      if (Optional<DWARFFormValue> Val =
              DIE.find(dwarf::DW_AT_GNU_addr_base, &AttrOffset)) {
        uint64_t Offset = AddrWriter->getOffset(*CU->getDWOId());
        DebugInfoPatcher.addLE32Patch(AttrOffset, static_cast<int32_t>(Offset));
      }
    }
  }

  std::unique_ptr<RangesBufferVector> RangesSectionContents =
      RangesSectionWriter->finalize();
  BC.registerOrUpdateNoteSection(".debug_ranges",
                                  copyByteArray(*RangesSectionContents),
                                  RangesSectionContents->size());

  std::unique_ptr<LocBufferVector> LocationListSectionContents =
      makeFinalLocListsSection(DebugInfoPatcher);
  BC.registerOrUpdateNoteSection(".debug_loc",
                                  copyByteArray(*LocationListSectionContents),
                                  LocationListSectionContents->size());
}

void DWARFRewriter::writeOutDWOFiles(
    std::unordered_map<uint64_t, std::string> &DWOIdToName) {
  std::string DebugData = "";
  auto ApplyPatch = [&](BinaryPatcher *Patcher, StringRef Data) -> StringRef {
    DebugData = Data.str();
    Patcher->patchBinary(DebugData);
    return StringRef(DebugData.c_str(), DebugData.size());
  };

  for (const std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
    Optional<uint64_t> DWOId = CU->getDWOId();
    if (!DWOId)
      continue;

    Optional<DWARFUnit *> DWOCU = BC.getDWOCU(*DWOId);
    if (!DWOCU)
      continue;

    const object::ObjectFile *File =
        (*DWOCU)->getContext().getDWARFObj().getFile();
    std::string CompDir = opts::DwoOutputPath.empty()
                              ? CU->getCompilationDir()
                              : opts::DwoOutputPath.c_str();
    std::string ObjectName = getDWOName(*CU.get(), nullptr, DWOIdToName);
    auto FullPath = CompDir.append("/").append(ObjectName);

    std::error_code EC;
    std::unique_ptr<ToolOutputFile> TempOut =
        std::make_unique<ToolOutputFile>(FullPath, EC, sys::fs::OF_None);

    std::unique_ptr<BinaryContext> TmpBC = BinaryContext::createBinaryContext(
        File, false,
        DWARFContext::create(*File, nullptr, "", WithColor::defaultErrorHandler,
                             WithColor::defaultWarningHandler,
                             /*UsesRelocs=*/false));
    std::unique_ptr<MCStreamer> Streamer = TmpBC->createStreamer(TempOut->os());
    const MCObjectFileInfo &MCOFI = *Streamer->getContext().getObjectFileInfo();

    const StringMap<MCSection *> KnownSections = {
        {".debug_info.dwo", MCOFI.getDwarfInfoDWOSection()},
        {".debug_types.dwo", MCOFI.getDwarfTypesDWOSection()},
        {".debug_str.dwo", MCOFI.getDwarfStrDWOSection()},
        {".debug_str_offsets.dwo", MCOFI.getDwarfStrOffDWOSection()},
        {".debug_abbrev.dwo", MCOFI.getDwarfAbbrevDWOSection()},
        {".debug_loc.dwo", MCOFI.getDwarfLocDWOSection()},
        {".debug_line.dwo", MCOFI.getDwarfLineDWOSection()}};

    for (const SectionRef &Section : File->sections()) {
      Expected<StringRef> SectionName = Section.getName();
      assert(SectionName && "Invalid section name.");
      auto SectionIter = KnownSections.find(*SectionName);
      if (SectionIter == KnownSections.end())
        continue;
      Streamer->SwitchSection(SectionIter->second);
      Expected<StringRef> Contents = Section.getContents();
      assert(Contents && "Invalid contents.");
      StringRef OutData(*Contents);
      std::unique_ptr<LocBufferVector> Data;
      if (SectionName->equals(".debug_loc.dwo")) {
        DebugLocWriter *LocWriter = LocListWritersByCU[*DWOId].get();
        Data = LocWriter->finalize();
        // Creating explicit with creating of StringRef here, otherwise
        // with impicit conversion it will take null byte as end of
        // string.
        OutData = StringRef(reinterpret_cast<const char *>(Data->data()),
                            Data->size());
      } else if (SectionName->equals(".debug_info.dwo")) {
        SimpleBinaryPatcher *Patcher = getBinaryDWODebugInfoPatcher(*DWOId);
        OutData = ApplyPatch(Patcher, OutData);
      } else if (SectionName->equals(".debug_abbrev.dwo")) {
        DebugAbbrevPatcher *Patcher = getBinaryDWOAbbrevPatcher(*DWOId);
        OutData = ApplyPatch(Patcher, OutData);
      }

      Streamer->emitBytes(OutData);
    }
    Streamer->Finish();
    TempOut->keep();
  }
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

void DWARFRewriter::convertToRanges(const DWARFAbbreviationDeclaration *Abbrev,
                                    DebugAbbrevPatcher &AbbrevPatcher) {
  dwarf::Form HighPCForm = Abbrev->findAttribute(dwarf::DW_AT_high_pc)->Form;
  dwarf::Form LowPCForm = Abbrev->findAttribute(dwarf::DW_AT_low_pc)->Form;

  std::lock_guard<std::mutex> Lock(AbbrevPatcherMutex);
  // DW_FORM_GNU_addr_index is already variable encoding so nothing to do there.
  // If HighForm is 8 bytes need to change low_pc to be variable encoding to
  // consume extra bytes from high_pc, since DW_FORM_sec_offset is 4 bytes for
  // DWARF32.
  if (LowPCForm != dwarf::DW_FORM_GNU_addr_index &&
      isHighPcFormEightBytes(HighPCForm))
    AbbrevPatcher.addAttributePatch(Abbrev, dwarf::DW_AT_low_pc,
                                    dwarf::DW_AT_low_pc,
                                    dwarf::DW_FORM_indirect);

  AbbrevPatcher.addAttributePatch(Abbrev, dwarf::DW_AT_high_pc,
                                  dwarf::DW_AT_ranges,
                                  dwarf::DW_FORM_sec_offset);
}

void DWARFRewriter::convertToRanges(DWARFDie DIE,
                                    const DebugAddressRangesVector &Ranges,
                                    SimpleBinaryPatcher &DebugInfoPatcher) {
  uint64_t RangesSectionOffset;
  if (Ranges.empty()) {
    RangesSectionOffset = RangesSectionWriter->getEmptyRangesOffset();
  } else {
    RangesSectionOffset = RangesSectionWriter->addRanges(Ranges);
  }

  convertToRanges(DIE, RangesSectionOffset, DebugInfoPatcher);
}

void DWARFRewriter::convertPending(const DWARFAbbreviationDeclaration *Abbrev,
                                   SimpleBinaryPatcher &DebugInfoPatcher,
                                   DebugAbbrevPatcher &AbbrevPatcher) {
  if (ConvertedRangesAbbrevs.count(Abbrev))
    return;

  convertToRanges(Abbrev, AbbrevPatcher);

  auto I = PendingRanges.find(Abbrev);
  if (I != PendingRanges.end()) {
    for (std::pair<DWARFDieWrapper, DebugAddressRange> &Pair : I->second) {
      convertToRanges(Pair.first, {Pair.second}, DebugInfoPatcher);
    }
    PendingRanges.erase(I);
  }

  ConvertedRangesAbbrevs.emplace(Abbrev);
}

void DWARFRewriter::addToPendingRanges(
    const DWARFAbbreviationDeclaration *Abbrev, DWARFDie DIE,
    DebugAddressRangesVector &FunctionRanges, Optional<uint64_t> DWOId) {
  Optional<DWARFFormValue> LowPcValue = DIE.find(dwarf::DW_AT_low_pc);
  Optional<DWARFFormValue> HighPcValue = DIE.find(dwarf::DW_AT_high_pc);
  if (LowPcValue &&
      LowPcValue->getForm() == dwarf::Form::DW_FORM_GNU_addr_index) {
    assert(DWOId && "Invalid DWO ID.");
    (void)DWOId;
    assert(HighPcValue && "Low PC exists, but not High PC.");
    (void)HighPcValue;
    uint64_t IndexL = LowPcValue->getRawUValue();
    uint64_t IndexH = HighPcValue->getRawUValue();
    for (auto Address : FunctionRanges) {
      AddrWriter->addIndexAddress(Address.LowPC, IndexL, *DWOId);
      // 2.17.2
      // If the value of the DW_AT_high_pc is of class address, it is the
      // relocated address of the first location past the last instruction
      // associated with the entity; if it is of class constant, the value is
      // an unsigned integer offset which when added to the low PC gives the
      // address of the first location past the last instruction associated
      // with the entity.
      if (!HighPcValue->isFormClass(DWARFFormValue::FC_Constant))
        AddrWriter->addIndexAddress(Address.HighPC, IndexH, *DWOId);
    }
  }
  PendingRanges[Abbrev].emplace_back(
      std::make_pair(DWARFDieWrapper(DIE), FunctionRanges.front()));
}

std::unique_ptr<LocBufferVector>
DWARFRewriter::makeFinalLocListsSection(SimpleBinaryPatcher &DebugInfoPatcher) {
  auto LocBuffer = std::make_unique<LocBufferVector>();
  auto LocStream = std::make_unique<raw_svector_ostream>(*LocBuffer);
  auto Writer =
    std::unique_ptr<MCObjectWriter>(BC.createObjectWriter(*LocStream));

  uint64_t SectionOffset = 0;

  // Add an empty list as the first entry;
  const char Zeroes[16] = {0};
  *LocStream << StringRef(Zeroes, 16);
  SectionOffset += 2 * 8;

  std::unordered_map<uint64_t, uint64_t> SectionOffsetByCU(
      LocListWritersByCU.size());

  for (std::pair<const uint64_t, std::unique_ptr<DebugLocWriter>> &Loc :
       LocListWritersByCU) {
    uint64_t CUIndex = Loc.first;
    DebugLocWriter *LocWriter = Loc.second.get();
    if (llvm::isa<DebugLoclistWriter>(*LocWriter))
      continue;
    SectionOffsetByCU[CUIndex] = SectionOffset;
    std::unique_ptr<LocBufferVector> CurrCULocationLists =
        LocWriter->finalize();
    *LocStream << *CurrCULocationLists;
    SectionOffset += CurrCULocationLists->size();
  }

  for (std::pair<const uint64_t, VectorLocListDebugInfoPatchType> &Iter :
       DwoLocListDebugInfoPatches) {
    uint64_t DWOId = Iter.first;
    SimpleBinaryPatcher *Patcher = getBinaryDWODebugInfoPatcher(DWOId);
    for (LocListDebugInfoPatchType &Patch : Iter.second) {
      Patcher->addLE32Patch(Patch.DebugInfoOffset,
                            SectionOffsetByCU[Patch.CUIndex] +
                                Patch.CUWriterOffset);
    }
  }

  for (LocListDebugInfoPatchType &Patch : LocListDebugInfoPatches) {
    DebugInfoPatcher.addLE32Patch(Patch.DebugInfoOffset,
                                  SectionOffsetByCU[Patch.CUIndex] +
                                      Patch.CUWriterOffset);
  }

  return LocBuffer;
}

void DWARFRewriter::flushPendingRanges(SimpleBinaryPatcher &DebugInfoPatcher) {
  for (std::pair<const DWARFAbbreviationDeclaration *const,
                 std::vector<std::pair<DWARFDieWrapper, DebugAddressRange>>>
           &I : PendingRanges) {
    for (std::pair<DWARFDieWrapper, DebugAddressRange> &RangePair : I.second) {
      patchLowHigh(RangePair.first, RangePair.second, DebugInfoPatcher);
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

  if ((LowPCFormValue.getForm() != dwarf::DW_FORM_addr &&
       LowPCFormValue.getForm() != dwarf::DW_FORM_GNU_addr_index) ||
      (HighPCFormValue.getForm() != dwarf::DW_FORM_addr &&
       HighPCFormValue.getForm() != dwarf::DW_FORM_data8 &&
       HighPCFormValue.getForm() != dwarf::DW_FORM_data4)) {
    errs() << "BOLT-WARNING: unexpected form value. Cannot update DIE "
           << "at offset 0x" << Twine::utohexstr(DIE.getOffset()) << "\n";
    return;
  }
  if ((LowPCOffset == -1U || (LowPCOffset + 8 != HighPCOffset)) &&
      LowPCFormValue.getForm() != dwarf::DW_FORM_GNU_addr_index) {
    errs() << "BOLT-WARNING: high_pc expected immediately after low_pc. "
           << "Cannot update DIE at offset 0x"
           << Twine::utohexstr(DIE.getOffset()) << '\n';
    return;
  }
}

}

void DWARFRewriter::patchLowHigh(DWARFDie DIE, DebugAddressRange Range,
                                 SimpleBinaryPatcher &DebugInfoPatcher) {
  uint64_t LowPCOffset, HighPCOffset;
  DWARFFormValue LowPCFormValue, HighPCFormValue;
  getRangeAttrData(
      DIE, LowPCOffset, HighPCOffset, LowPCFormValue, HighPCFormValue);
  auto *TempDebugPatcher = &DebugInfoPatcher;
  if (LowPCFormValue.getForm() == dwarf::DW_FORM_GNU_addr_index) {
    DWARFUnit *Unit = DIE.getDwarfUnit();
    assert(Unit->isDWOUnit() && "DW_FORM_GNU_addr_index not part of DWO.");
    uint32_t AddressIndex =
        AddrWriter->getIndexFromAddress(Range.LowPC, *Unit->getDWOId());
    TempDebugPatcher = getBinaryDWODebugInfoPatcher(*Unit->getDWOId());
    TempDebugPatcher->addUDataPatch(LowPCOffset, AddressIndex,
                                    std::abs(int(HighPCOffset - LowPCOffset)));
    // TODO: In DWARF5 support ULEB128 for high_pc
  } else {
    TempDebugPatcher->addLE64Patch(LowPCOffset, Range.LowPC);
  }

  if (isHighPcFormEightBytes(HighPCFormValue.getForm())) {
    TempDebugPatcher->addLE64Patch(HighPCOffset, Range.HighPC - Range.LowPC);
  } else {
    TempDebugPatcher->addLE32Patch(HighPCOffset, Range.HighPC - Range.LowPC);
  }
}

void DWARFRewriter::convertToRanges(DWARFDie DIE, uint64_t RangesSectionOffset,
                                    SimpleBinaryPatcher &DebugInfoPatcher) {
  uint64_t LowPCOffset, HighPCOffset;
  DWARFFormValue LowPCFormValue, HighPCFormValue;
  getRangeAttrData(DIE, LowPCOffset, HighPCOffset, LowPCFormValue,
                   HighPCFormValue);

  unsigned LowPCSize = 0;
  assert(DIE.getDwarfUnit()->getAddressByteSize() == 8);
  if (isHighPcFormEightBytes(HighPCFormValue.getForm())) {
    LowPCSize = 12;
  } else if (HighPCFormValue.getForm() == dwarf::DW_FORM_data4) {
    LowPCSize = 8;
  } else {
    llvm_unreachable("unexpected form");
  }

  std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
  uint32_t BaseOffset = 0;
  if (LowPCFormValue.getForm() == dwarf::DW_FORM_GNU_addr_index) {
    // Add Indexer is already variable length encoding.
    DebugInfoPatcher.addUDataPatch(LowPCOffset, 0,
                                   std::abs(int(HighPCOffset - LowPCOffset)) +
                                       LowPCSize - 8);
    // Ranges are relative to DW_AT_GNU_ranges_base.
    BaseOffset = DebugInfoPatcher.getRangeBase();
  } else if (LowPCSize == 12) {
    // Creatively encoding dwarf::DW_FORM_addr in to 4 bytes.
    // Write an indirect 0 value for DW_AT_low_pc so that we can fill
    // 12 bytes of space.
    // The Abbrev wa already changed.
    DebugInfoPatcher.addUDataPatch(LowPCOffset, dwarf::DW_FORM_addr, 4);
    DebugInfoPatcher.addLE64Patch(LowPCOffset + 4, 0);
  } else {
    DebugInfoPatcher.addLE64Patch(LowPCOffset, 0);
  }
  DebugInfoPatcher.addLE32Patch(HighPCOffset + LowPCSize - 8,
                                RangesSectionOffset - BaseOffset);
}
