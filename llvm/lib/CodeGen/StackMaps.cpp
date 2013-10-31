//===---------------------------- StackMaps.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "stackmaps"

#include "llvm/CodeGen/StackMaps.h"

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOpcodes.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

#include <iterator>

using namespace llvm;

void StackMaps::recordStackMap(const MachineInstr &MI, uint32_t ID,
                               MachineInstr::const_mop_iterator MOI,
                               MachineInstr::const_mop_iterator MOE) {

  MCContext &OutContext = AP.OutStreamer.getContext();
  MCSymbol *MILabel = OutContext.CreateTempSymbol();
  AP.OutStreamer.EmitLabel(MILabel);

  LocationVec CallsiteLocs;

  while (MOI != MOE) {
    std::pair<Location, MachineInstr::const_mop_iterator> ParseResult =
      OpParser(MOI, MOE);

    Location &Loc = ParseResult.first;

    // Move large constants into the constant pool.
    if (Loc.LocType == Location::Constant && (Loc.Offset & ~0xFFFFFFFFULL)) {
      Loc.LocType = Location::ConstantIndex;
      Loc.Offset = ConstPool.getConstantIndex(Loc.Offset);
    }

    CallsiteLocs.push_back(Loc);
    MOI = ParseResult.second;
  }

  const MCExpr *CSOffsetExpr = MCBinaryExpr::CreateSub(
    MCSymbolRefExpr::Create(MILabel, OutContext),
    MCSymbolRefExpr::Create(AP.CurrentFnSym, OutContext),
    OutContext);

  CSInfos.push_back(CallsiteInfo(CSOffsetExpr, ID, CallsiteLocs));
}

/// serializeToStackMapSection conceptually populates the following fields:
///
/// uint32 : Reserved (header)
/// uint32 : NumConstants
/// int64  : Constants[NumConstants]
/// uint32 : NumRecords
/// StkMapRecord[NumRecords] {
///   uint32 : PatchPoint ID
///   uint32 : Instruction Offset
///   uint16 : Reserved (record flags)
///   uint16 : NumLocations
///   Location[NumLocations] {
///     uint8  : Register | Direct | Indirect | Constant | ConstantIndex
///     uint8  : Reserved (location flags)
///     uint16 : Dwarf RegNum
///     int32  : Offset
///   }
/// }
///
/// Location Encoding, Type, Value:
///   0x1, Register, Reg                 (value in register)
///   0x2, Direct, Reg + Offset          (frame index)
///   0x3, Indirect, [Reg + Offset]      (spilled value)
///   0x4, Constant, Offset              (small constant)
///   0x5, ConstIndex, Constants[Offset] (large constant)
///
void StackMaps::serializeToStackMapSection() {
  // Bail out if there's no stack map data.
  if (CSInfos.empty())
    return;

  MCContext &OutContext = AP.OutStreamer.getContext();
  const TargetRegisterInfo *TRI = AP.TM.getRegisterInfo();

  // Create the section.
  const MCSection *StackMapSection =
    OutContext.getMachOSection("__LLVM_STACKMAPS", "__llvm_stackmaps", 0,
                               SectionKind::getMetadata());
  AP.OutStreamer.SwitchSection(StackMapSection);

  // Emit a dummy symbol to force section inclusion.
  AP.OutStreamer.EmitLabel(
    OutContext.GetOrCreateSymbol(Twine("__LLVM_StackMaps")));

  // Serialize data.
  const char *WSMP = "Stack Maps: ";
  const MCRegisterInfo &MCRI = *OutContext.getRegisterInfo();

  DEBUG(dbgs() << "********** Stack Map Output **********\n");

  // Header.
  AP.OutStreamer.EmitIntValue(0, 4);

  // Num constants.
  AP.OutStreamer.EmitIntValue(ConstPool.getNumConstants(), 4);

  // Constant pool entries.
  for (unsigned i = 0; i < ConstPool.getNumConstants(); ++i)
    AP.OutStreamer.EmitIntValue(ConstPool.getConstant(i), 8);

  DEBUG(dbgs() << WSMP << "#callsites = " << CSInfos.size() << "\n");
  AP.OutStreamer.EmitIntValue(CSInfos.size(), 4);

  for (CallsiteInfoList::const_iterator CSII = CSInfos.begin(),
                                        CSIE = CSInfos.end();
       CSII != CSIE; ++CSII) {

    unsigned CallsiteID = CSII->ID;
    const LocationVec &CSLocs = CSII->Locations;

    DEBUG(dbgs() << WSMP << "callsite " << CallsiteID << "\n");

    // Verify stack map entry. It's better to communicate a problem to the
    // runtime than crash in case of in-process compilation. Currently, we do
    // simple overflow checks, but we may eventually communicate other
    // compilation errors this way.
    if (CSLocs.size() > UINT16_MAX) {
      AP.OutStreamer.EmitIntValue(UINT32_MAX, 4); // Invalid ID.
      AP.OutStreamer.EmitValue(CSII->CSOffsetExpr, 4);
      AP.OutStreamer.EmitIntValue(0, 2); // Reserved.
      AP.OutStreamer.EmitIntValue(0, 2); // 0 locations.
      continue;
    }

    AP.OutStreamer.EmitIntValue(CallsiteID, 4);
    AP.OutStreamer.EmitValue(CSII->CSOffsetExpr, 4);

    // Reserved for flags.
    AP.OutStreamer.EmitIntValue(0, 2);

    DEBUG(dbgs() << WSMP << "  has " << CSLocs.size() << " locations\n");

    AP.OutStreamer.EmitIntValue(CSLocs.size(), 2);

    unsigned operIdx = 0;
    for (LocationVec::const_iterator LocI = CSLocs.begin(), LocE = CSLocs.end();
         LocI != LocE; ++LocI, ++operIdx) {
      const Location &Loc = *LocI;
      DEBUG(
        dbgs() << WSMP << "  Loc " << operIdx << ": ";
        switch (Loc.LocType) {
        case Location::Unprocessed:
          dbgs() << "<Unprocessed operand>";
          break;
        case Location::Register:
          dbgs() << "Register " << MCRI.getName(Loc.Reg);
          break;
        case Location::Direct:
          dbgs() << "Direct " << MCRI.getName(Loc.Reg);
          if (Loc.Offset)
            dbgs() << " + " << Loc.Offset;
          break;
        case Location::Indirect:
          dbgs() << "Indirect " << MCRI.getName(Loc.Reg)
                 << " + " << Loc.Offset;
          break;
        case Location::Constant:
          dbgs() << "Constant " << Loc.Offset;
          break;
        case Location::ConstantIndex:
          dbgs() << "Constant Index " << Loc.Offset;
          break;
        }
        dbgs() << "\n";
      );

      unsigned RegNo = 0;
      if(Loc.Reg) {
        RegNo = MCRI.getDwarfRegNum(Loc.Reg, false);
        for (MCSuperRegIterator SR(Loc.Reg, TRI);
             SR.isValid() && (int)RegNo < 0; ++SR) {
          RegNo = TRI->getDwarfRegNum(*SR, false);
        }
      }
      else {
        assert((Loc.LocType != Location::Register
                && Loc.LocType != Location::Register) &&
               "Missing location register");
      }
      AP.OutStreamer.EmitIntValue(Loc.LocType, 1);
      AP.OutStreamer.EmitIntValue(0, 1); // Reserved location flags.
      AP.OutStreamer.EmitIntValue(RegNo, 2);
      AP.OutStreamer.EmitIntValue(Loc.Offset, 4);
    }
  }

  AP.OutStreamer.AddBlankLine();

  CSInfos.clear();
}
