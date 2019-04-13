//===--- BoltAddressTranslation.cpp ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
#include "BoltAddressTranslation.h"
#include "BinaryFunction.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/Support/DataExtractor.h"

#define DEBUG_TYPE "bolt-bat"

namespace llvm {
namespace bolt {

const char* BoltAddressTranslation::SECTION_NAME = ".note.bolt_bat";

void BoltAddressTranslation::writeEntriesForBB(MapTy &Map,
                                               const BinaryBasicBlock &BB,
                                               uint64_t FuncAddress) {
  const uint64_t Key = BB.getOutputAddressRange().first - FuncAddress;
  const uint64_t Val = BB.getInputOffset();

  assert(Val != BinaryBasicBlock::INVALID_OFFSET &&
         "Every output BB must track back to an input BB for profile "
         "collection in bolted binaries");

  DEBUG(dbgs() << "BB " << BB.getName() <<"\n");
  DEBUG(dbgs() << "  Key: " << Twine::utohexstr(Key)
               << " Val: " << Twine::utohexstr(Val) << "\n");
  Map.insert(std::pair<uint32_t, uint32_t>(Key, Val));

  // Look for special instructions we are interested in mapping offsets. These
  // are key instructions for the profile identified by
  // BC.keepOffsetForInstruction(Inst) and are instructions that cause control
  // flow change. We also record offsets for the last instruction in the BB in
  // some cases. These are harmless for BAT writing purposes, besides increasing
  // the size of the table unnecessarily.
  for (const auto &Inst : BB) {
    if (!BC.MIB->hasAnnotation(Inst, "LocSym"))
      continue;
    const auto OutputOffset =
        BC.MIB->getAnnotationAs<uint32_t>(Inst, "LocSym") - FuncAddress;

    auto InputOffsetOrErr = BC.MIB->tryGetAnnotationAs<uint32_t>(Inst, "Offset");
    DEBUG(if (!InputOffsetOrErr) {
      auto *Function = BB.getFunction();
      dbgs() << "Function: " << Function->getPrintName()
             << " BB: " << BB.getName() << " lacking annotation for: ";
      BC.printInstruction(dbgs(), Inst);
      dbgs() << "\n";
    });
    assert(InputOffsetOrErr && "Expected annotation with input offset");
    const auto InputOffset = *InputOffsetOrErr;

    // Is this the first instruction in the BB? No need to duplicate the entry
    if (Key == OutputOffset)
      continue;

    DEBUG(dbgs() << "  Key: " << Twine::utohexstr(OutputOffset)
                 << " Val: " << Twine::utohexstr(InputOffset)
                 << " (branch)\n");
    Map.insert(
        std::pair<uint32_t, uint32_t>(OutputOffset, InputOffset | BRANCHENTRY));
  }
}

void BoltAddressTranslation::write(raw_ostream &OS) {
  DEBUG(dbgs() << "BOLT-DEBUG: Writing BOLT Address Translation Tables\n");
  for (auto &BFI : BC.getBinaryFunctions()) {
    auto &Function = BFI.second;

    DEBUG(dbgs() << "Function name: " << Function.getPrintName() << "\n");
    DEBUG(dbgs() << " Address reference: 0x"
                 << Twine::utohexstr(Function.getOutputAddress()) << "\n");
    MapTy Map;
    const bool IsSplit = Function.isSplit();
    for (const auto &BB : Function.layout()) {
      if (IsSplit && BB->isCold())
        break;
      writeEntriesForBB(Map, *BB, Function.getOutputAddress());
    }
    Maps.insert(std::pair<uint64_t, MapTy>(Function.getOutputAddress(), Map));

    if (!IsSplit)
      continue;

    // Cold map
    Map.clear();
    DEBUG(dbgs() << " Cold part\n");
    for (const auto &BB : Function.layout()) {
      if (!BB->isCold())
        continue;
      writeEntriesForBB(Map, *BB, Function.cold().getAddress());
    }
    Maps.insert(std::pair<uint64_t, MapTy>(Function.cold().getAddress(), Map));
    ColdPartSource.insert(std::pair<uint64_t, uint64_t>(
        Function.cold().getAddress(), Function.getOutputAddress()));
  }

  const uint32_t NumFuncs = Maps.size();
  OS.write(reinterpret_cast<const char *>(&NumFuncs), 4);
  DEBUG(dbgs() << "Writing " << NumFuncs << " functions for BAT.\n");
  for (auto &MapEntry : Maps) {
    const uint64_t Address = MapEntry.first;
    MapTy &Map = MapEntry.second;
    const uint32_t NumEntries = Map.size();
    DEBUG(dbgs() << "Writing " << NumEntries << " entries for 0x"
                 << Twine::utohexstr(Address) << ".\n");
    OS.write(reinterpret_cast<const char *>(&Address), 8);
    OS.write(reinterpret_cast<const char *>(&NumEntries), 4);
    for (auto &KeyVal : Map) {
      OS.write(reinterpret_cast<const char *>(&KeyVal.first), 4);
      OS.write(reinterpret_cast<const char *>(&KeyVal.second), 4);
    }
  }
  const uint32_t NumColdEntries = ColdPartSource.size();
  DEBUG(dbgs() << "Writing " << NumColdEntries << " cold part mappings.\n");
  OS.write(reinterpret_cast<const char *>(&NumColdEntries), 4);
  for (auto &ColdEntry : ColdPartSource) {
    OS.write(reinterpret_cast<const char *>(&ColdEntry.first), 8);
    OS.write(reinterpret_cast<const char *>(&ColdEntry.second), 8);
    DEBUG(dbgs() << " " << Twine::utohexstr(ColdEntry.first) << " -> "
          << Twine::utohexstr(ColdEntry.second) << "\n");
  }

  outs() << "BOLT-INFO: Wrote " << Maps.size() << " BAT maps\n";
  outs() << "BOLT-INFO: Wrote " << NumColdEntries
         << " BAT cold-to-hot entries\n";
}

std::error_code BoltAddressTranslation::parse(StringRef Buf) {
  DataExtractor DE = DataExtractor(Buf, true, 8);
  uint32_t Offset = 0;
  if (Buf.size() < 12)
    return make_error_code(llvm::errc::io_error);

  const uint32_t NameSz = DE.getU32(&Offset);
  const uint32_t DescSz = DE.getU32(&Offset);
  const uint32_t Type = DE.getU32(&Offset);

  // Note type reserved for BAT
  if (Type != 1 ||
      Buf.size() + Offset < alignTo(NameSz, 4) + DescSz)
    return make_error_code(llvm::errc::io_error);

  StringRef Name = Buf.slice(Offset, Offset + NameSz);
  Offset = alignTo(Offset + NameSz, 4);
  if (Name.substr(0, 4) != "BOLT")
    return make_error_code(llvm::errc::io_error);

  if (Buf.size() - Offset < 4)
    return make_error_code(llvm::errc::io_error);

  const uint32_t NumFunctions = DE.getU32(&Offset);
  DEBUG(dbgs() << "Parsing " << NumFunctions << " functions\n");
  for (uint32_t I = 0; I < NumFunctions; ++I) {
    if (Buf.size() - Offset < 12)
      return make_error_code(llvm::errc::io_error);

    const uint64_t Address = DE.getU64(&Offset);
    const uint32_t NumEntries = DE.getU32(&Offset);
    MapTy Map;

    DEBUG(dbgs() << "Parsing " << NumEntries << " entries for 0x"
                 << Twine::utohexstr(Address) << "\n");
    if (Buf.size() - Offset < 8 * NumEntries)
      return make_error_code(llvm::errc::io_error);
    for (uint32_t J = 0; J < NumEntries; ++J) {
      const uint32_t OutputAddr = DE.getU32(&Offset);
      const uint32_t InputAddr = DE.getU32(&Offset);
      Map.insert(std::pair<uint32_t, uint32_t>(OutputAddr, InputAddr));
      DEBUG(dbgs() << Twine::utohexstr(OutputAddr) << " -> "
                   << Twine::utohexstr(InputAddr) << "\n");
    }
    Maps.insert(std::pair<uint64_t, MapTy>(Address, Map));
  }

  if (Buf.size() - Offset < 4)
    return make_error_code(llvm::errc::io_error);

  const uint32_t NumColdEntries = DE.getU32(&Offset);
  DEBUG(dbgs() << "Parsing " << NumColdEntries << " cold part mappings\n");
  for (uint32_t I = 0; I < NumColdEntries; ++I) {
    if (Buf.size() - Offset < 16)
      return make_error_code(llvm::errc::io_error);
    const uint32_t ColdAddress = DE.getU64(&Offset);
    const uint32_t HotAddress = DE.getU64(&Offset);
    ColdPartSource.insert(
        std::pair<uint64_t, uint64_t>(ColdAddress, HotAddress));
    DEBUG(dbgs() << Twine::utohexstr(ColdAddress) << " -> "
                 << Twine::utohexstr(HotAddress) << "\n");
  }
  outs() << "BOLT-INFO: Parsed " << Maps.size() << " BAT entries\n";
  outs() << "BOLT-INFO: Parsed " << NumColdEntries
         << " BAT cold-to-hot entries\n";

  return std::error_code();
}

uint64_t BoltAddressTranslation::translate(const BinaryFunction &Func,
                                           uint64_t Offset,
                                           bool IsBranchSrc) const {
  auto Iter = Maps.find(Func.getAddress());
  if (Iter == Maps.end())
    return Offset;

  const MapTy &Map = Iter->second;
  auto KeyVal = Map.upper_bound(Offset);
  if (KeyVal == Map.begin())
    return Offset;

  --KeyVal;

  const uint32_t Val = KeyVal->second & ~BRANCHENTRY;
  // Branch source addresses are translated to the first instruction of the
  // source BB to avoid accounting for modifications BOLT may have made in the
  // BB regarding deletion/addition of instructions.
  if (IsBranchSrc)
    return Val;
  return Offset - KeyVal->first + Val;
}

Optional<SmallVector<std::pair<uint64_t, uint64_t>, 16>>
BoltAddressTranslation::getFallthroughsInTrace(
    const BinaryFunction &Func,
    const LBREntry &FirstLBR, const LBREntry &SecondLBR) const {
  SmallVector<std::pair<uint64_t, uint64_t>, 16> Res;

  // Filter out trivial case
  if (FirstLBR.To >= SecondLBR.From)
    return Res;

  const auto From = FirstLBR.To - Func.getAddress();
  const auto To = SecondLBR.From - Func.getAddress();

  auto Iter = Maps.find(Func.getAddress());
  if (Iter == Maps.end()) {
    return NoneType();
  }

  const MapTy &Map = Iter->second;
  auto FromIter = Map.upper_bound(From);
  if (FromIter == Map.begin())
    return Res;
  // Skip instruction entries, to create fallthroughs we are only interested in
  // BB boundaries
  do {
    if (FromIter == Map.begin())
      return Res;
    --FromIter;
  } while (FromIter->second & BRANCHENTRY);

  auto ToIter = Map.upper_bound(To);
  if (ToIter == Map.begin())
    return Res;
  --ToIter;
  if (FromIter->first >= ToIter->first)
    return Res;

  for (auto Iter = FromIter; Iter != ToIter; ) {
    const auto Src = Iter->first;
    if (Iter->second & BRANCHENTRY) {
      ++Iter;
      continue;
    }

    ++Iter;
    while (Iter->second & BRANCHENTRY && Iter != ToIter) {
      ++Iter;
    }
    if (Iter->second & BRANCHENTRY)
      break;
    Res.emplace_back(std::make_pair(Src, Iter->first));
  }

  return Res;
}

uint64_t BoltAddressTranslation::fetchParentAddress(uint64_t Address) const {
  auto Iter = ColdPartSource.find(Address);
  if (Iter == ColdPartSource.end())
    return 0;
  return Iter->second;
}

bool BoltAddressTranslation::enabledFor(
    llvm::object::ELFObjectFileBase *InputFile) const {
  for (const auto &Section : InputFile->sections()) {
    StringRef SectionName;
    if (std::error_code EC = Section.getName(SectionName))
      continue;

    if (SectionName == SECTION_NAME)
      return true;
  }
  return false;
}
}
}
