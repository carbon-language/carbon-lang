//===--- BinaryContext.cpp  - Interface for machine-level context ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {
namespace bolt {

BinaryContext::~BinaryContext() { }

MCSymbol *BinaryContext::getOrCreateGlobalSymbol(uint64_t Address,
                                                 Twine Prefix) {
  MCSymbol *Symbol{nullptr};
  std::string Name;
  auto NI = GlobalAddresses.find(Address);
  if (NI != GlobalAddresses.end()) {
    // Even though there could be multiple names registered at the address,
    // we only use the first one.
    Name = NI->second;
  } else {
    Name = (Prefix + "0x" + Twine::utohexstr(Address)).str();
    assert(GlobalSymbols.find(Name) == GlobalSymbols.end() &&
           "created name is not unique");
    GlobalAddresses.emplace(std::make_pair(Address, Name));
  }

  Symbol = Ctx->lookupSymbol(Name);
  if (Symbol)
    return Symbol;

  Symbol = Ctx->getOrCreateSymbol(Name);
  GlobalSymbols[Name] = Address;

  return Symbol;
}

} // namespace bolt
} // namespace llvm

namespace {

using namespace llvm;
using namespace bolt;

/// Returns the binary function that contains a given address in the input
/// binary, or nullptr if none does.
BinaryFunction *getBinaryFunctionContainingAddress(
    uint64_t Address,
    std::map<uint64_t, BinaryFunction> &BinaryFunctions) {
  auto It = BinaryFunctions.upper_bound(Address);
  if (It != BinaryFunctions.begin()) {
    --It;
    if (It->first + It->second.getSize() > Address) {
      return &It->second;
    }
  }
  return nullptr;
}

// Traverses the DIE tree in a recursive depth-first search and finds lexical
// blocks and instances of inlined subroutines, saving them in
// AddressRangesObjects.
void findAddressRangesObjects(
    const DWARFCompileUnit *Unit,
    const DWARFDebugInfoEntryMinimal *DIE,
    std::map<uint64_t, BinaryFunction> &Functions,
    std::vector<llvm::bolt::AddressRangesDWARFObject> &AddressRangesObjects) {
  auto Tag = DIE->getTag();
  if (Tag == dwarf::DW_TAG_lexical_block ||
      Tag == dwarf::DW_TAG_inlined_subroutine ||
      Tag == dwarf::DW_TAG_try_block ||
      Tag == dwarf::DW_TAG_catch_block) {
    auto const &Ranges = DIE->getAddressRanges(Unit);
    if (!Ranges.empty()) {
      // We have to process all ranges, even for functions that we are not
      // updating. The primary reason is that abbrev entries are shared
      // and if we convert one DIE, it may affect the rest. Thus
      // the conservative approach that does not involve expanding
      // .debug_abbrev, is to switch all DIEs to use .debug_ranges, even if
      // they use a single [a,b) range. The secondary reason is that it allows
      // us to get rid of the original portion of .debug_ranges to save
      // space in the binary.
      auto Function = getBinaryFunctionContainingAddress(Ranges.front().first,
                                                         Functions);
      AddressRangesObjects.emplace_back(Unit, DIE);
      auto &Object = AddressRangesObjects.back();
      for (const auto &Range : Ranges) {
        if (Function && Function->isSimple()) {
          Object.addAddressRange(*Function, Range.first, Range.second);
        } else {
          Object.addAbsoluteRange(Range.first, Range.second);
        }
      }
    }
  }

  // Recursively visit each child.
  for (auto Child = DIE->getFirstChild(); Child; Child = Child->getSibling()) {
    findAddressRangesObjects(Unit, Child, Functions, AddressRangesObjects);
  }
}

/// Recursively finds DWARF DW_TAG_subprogram DIEs and match them with
/// BinaryFunctions. Record DIEs for unknown subprograms (mostly functions that
/// are never called and removed from the binary) in Unknown.
void findSubprograms(DWARFCompileUnit *Unit,
                     const DWARFDebugInfoEntryMinimal *DIE,
                     std::map<uint64_t, BinaryFunction> &BinaryFunctions,
                     BinaryContext::DIECompileUnitVector &Unknown) {
  if (DIE->isSubprogramDIE()) {
    // TODO: handle DW_AT_ranges.
    uint64_t LowPC, HighPC;
    if (DIE->getLowAndHighPC(Unit, LowPC, HighPC)) {
      auto It = BinaryFunctions.find(LowPC);
      if (It != BinaryFunctions.end()) {
        It->second.addSubprogramDIE(Unit, DIE);
      } else {
        Unknown.emplace_back(DIE, Unit);
      }
    }
  }

  for (auto ChildDIE = DIE->getFirstChild();
       ChildDIE != nullptr && !ChildDIE->isNULL();
       ChildDIE = ChildDIE->getSibling()) {
    findSubprograms(Unit, ChildDIE, BinaryFunctions, Unknown);
  }
}

} // namespace

namespace llvm {
namespace bolt {

void BinaryContext::preprocessDebugInfo(
    std::map<uint64_t, BinaryFunction> &BinaryFunctions) {
  // Populate MCContext with DWARF files.
  for (const auto &CU : DwCtx->compile_units()) {
    const auto CUID = CU->getOffset();
    auto LineTable = DwCtx->getLineTableForUnit(CU.get());
    const auto &FileNames = LineTable->Prologue.FileNames;
    for (size_t I = 0, Size = FileNames.size(); I != Size; ++I) {
      // Dir indexes start at 1, as DWARF file numbers, and a dir index 0
      // means empty dir.
      const char *Dir = FileNames[I].DirIdx ?
          LineTable->Prologue.IncludeDirectories[FileNames[I].DirIdx - 1] :
          "";
      Ctx->getDwarfFile(Dir, FileNames[I].Name, I + 1, CUID);
    }
  }

  // For each CU, iterate over its children DIEs and match subprogram DIEs to
  // BinaryFunctions.
  for (auto &CU : DwCtx->compile_units()) {
    findSubprograms(CU.get(), CU->getUnitDIE(false), BinaryFunctions,
                    UnknownFunctions);
  }

  // Some functions may not have a corresponding subprogram DIE
  // yet they will be included in some CU and will have line number information.
  // Hence we need to associate them with the CU and include in CU ranges.
  for (auto &AddrFunctionPair : BinaryFunctions) {
    auto FunctionAddress = AddrFunctionPair.first;
    auto &Function = AddrFunctionPair.second;
    if (!Function.getSubprogramDIEs().empty())
      continue;
    if (auto DebugAranges = DwCtx->getDebugAranges()) {
      auto CUOffset = DebugAranges->findAddress(FunctionAddress);
      if (CUOffset != -1U) {
        Function.addSubprogramDIE(DwCtx->getCompileUnitForOffset(CUOffset),
                                  nullptr);
        continue;
      }
    }

#ifdef DWARF_LOOKUP_ALL_RANGES
    // Last resort - iterate over all compile units. This should not happen
    // very often. If it does, we need to create a separate lookup table
    // similar to .debug_aranges internally. This slows down processing
    // considerably.
    for (const auto &CU : DwCtx->compile_units()) {
      const auto *CUDie = CU->getUnitDIE();
      for (const auto &Range : CUDie->getAddressRanges(CU.get())) {
        if (FunctionAddress >= Range.first &&
            FunctionAddress < Range.second) {
          Function.addSubprogramDIE(CU.get(), nullptr);
          break;
        }
      }
    }
#endif
  }
}

void BinaryContext::preprocessFunctionDebugInfo(
    std::map<uint64_t, BinaryFunction> &BinaryFunctions) {
  // Iterate over DIE trees finding objects that contain address ranges.
  for (const auto &CU : DwCtx->compile_units()) {
    findAddressRangesObjects(CU.get(), CU->getUnitDIE(false), BinaryFunctions,
                             AddressRangesObjects);
  }

  // Iterate over location lists and save them in LocationLists.
  auto DebugLoc = DwCtx->getDebugLoc();
  for (const auto &DebugLocEntry : DebugLoc->getLocationLists()) {
    if (DebugLocEntry.Entries.empty())
      continue;
    auto StartAddress = DebugLocEntry.Entries.front().Begin;
    auto *Function = getBinaryFunctionContainingAddress(StartAddress,
                                                        BinaryFunctions);
    if (!Function || !Function->isSimple())
      continue;
    LocationLists.emplace_back(DebugLocEntry.Offset);
    auto &LocationList = LocationLists.back();
    for (const auto &Location : DebugLocEntry.Entries) {
      LocationList.addLocation(&Location.Loc, *Function, Location.Begin,
                               Location.End);
    }
  }
}

} // namespace bolt
} // namespace llvm
