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
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOpcodes.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <iterator>

using namespace llvm;

PatchPointOpers::PatchPointOpers(const MachineInstr *MI)
  : MI(MI),
    HasDef(MI->getOperand(0).isReg() && MI->getOperand(0).isDef() &&
           !MI->getOperand(0).isImplicit()),
    IsAnyReg(MI->getOperand(getMetaIdx(CCPos)).getImm() == CallingConv::AnyReg)
{
#ifndef NDEBUG
  unsigned CheckStartIdx = 0, e = MI->getNumOperands();
  while (CheckStartIdx < e && MI->getOperand(CheckStartIdx).isReg() &&
         MI->getOperand(CheckStartIdx).isDef() &&
         !MI->getOperand(CheckStartIdx).isImplicit())
    ++CheckStartIdx;

  assert(getMetaIdx() == CheckStartIdx &&
         "Unexpected additional definition in Patchpoint intrinsic.");
#endif
}

unsigned PatchPointOpers::getNextScratchIdx(unsigned StartIdx) const {
  if (!StartIdx)
    StartIdx = getVarIdx();

  // Find the next scratch register (implicit def and early clobber)
  unsigned ScratchIdx = StartIdx, e = MI->getNumOperands();
  while (ScratchIdx < e &&
         !(MI->getOperand(ScratchIdx).isReg() &&
           MI->getOperand(ScratchIdx).isDef() &&
           MI->getOperand(ScratchIdx).isImplicit() &&
           MI->getOperand(ScratchIdx).isEarlyClobber()))
    ++ScratchIdx;

  assert(ScratchIdx != e && "No scratch register available");
  return ScratchIdx;
}

MachineInstr::const_mop_iterator
StackMaps::parseOperand(MachineInstr::const_mop_iterator MOI,
                        MachineInstr::const_mop_iterator MOE,
                        LocationVec &Locs, LiveOutVec &LiveOuts) const {
  if (MOI->isImm()) {
    switch (MOI->getImm()) {
    default: llvm_unreachable("Unrecognized operand type.");
    case StackMaps::DirectMemRefOp: {
      unsigned Size = AP.TM.getDataLayout()->getPointerSizeInBits();
      assert((Size % 8) == 0 && "Need pointer size in bytes.");
      Size /= 8;
      unsigned Reg = (++MOI)->getReg();
      int64_t Imm = (++MOI)->getImm();
      Locs.push_back(Location(StackMaps::Location::Direct, Size, Reg, Imm));
      break;
    }
    case StackMaps::IndirectMemRefOp: {
      int64_t Size = (++MOI)->getImm();
      assert(Size > 0 && "Need a valid size for indirect memory locations.");
      unsigned Reg = (++MOI)->getReg();
      int64_t Imm = (++MOI)->getImm();
      Locs.push_back(Location(StackMaps::Location::Indirect, Size, Reg, Imm));
      break;
    }
    case StackMaps::ConstantOp: {
      ++MOI;
      assert(MOI->isImm() && "Expected constant operand.");
      int64_t Imm = MOI->getImm();
      Locs.push_back(Location(Location::Constant, sizeof(int64_t), 0, Imm));
      break;
    }
    }
    return ++MOI;
  }

  // The physical register number will ultimately be encoded as a DWARF regno.
  // The stack map also records the size of a spill slot that can hold the
  // register content. (The runtime can track the actual size of the data type
  // if it needs to.)
  if (MOI->isReg()) {
    // Skip implicit registers (this includes our scratch registers)
    if (MOI->isImplicit())
      return ++MOI;

    assert(TargetRegisterInfo::isPhysicalRegister(MOI->getReg()) &&
           "Virtreg operands should have been rewritten before now.");
    const TargetRegisterClass *RC =
      AP.TM.getRegisterInfo()->getMinimalPhysRegClass(MOI->getReg());
    assert(!MOI->getSubReg() && "Physical subreg still around.");
    Locs.push_back(
      Location(Location::Register, RC->getSize(), MOI->getReg(), 0));
    return ++MOI;
  }

  if (MOI->isRegLiveOut())
    LiveOuts = parseRegisterLiveOutMask(MOI->getRegLiveOut());

  return ++MOI;
}

/// Go up the super-register chain until we hit a valid dwarf register number.
static unsigned getDwarfRegNum(unsigned Reg, const TargetRegisterInfo *TRI) {
  int RegNo = TRI->getDwarfRegNum(Reg, false);
  for (MCSuperRegIterator SR(Reg, TRI); SR.isValid() && RegNo < 0; ++SR)
    RegNo = TRI->getDwarfRegNum(*SR, false);

  assert(RegNo >= 0 && "Invalid Dwarf register number.");
  return (unsigned) RegNo;
}

/// Create a live-out register record for the given register Reg.
StackMaps::LiveOutReg
StackMaps::createLiveOutReg(unsigned Reg, const TargetRegisterInfo *TRI) const {
  unsigned RegNo = getDwarfRegNum(Reg, TRI);
  unsigned Size = TRI->getMinimalPhysRegClass(Reg)->getSize();
  return LiveOutReg(Reg, RegNo, Size);
}

/// Parse the register live-out mask and return a vector of live-out registers
/// that need to be recorded in the stackmap.
StackMaps::LiveOutVec
StackMaps::parseRegisterLiveOutMask(const uint32_t *Mask) const {
  assert(Mask && "No register mask specified");
  const TargetRegisterInfo *TRI = AP.TM.getRegisterInfo();
  LiveOutVec LiveOuts;

  // Create a LiveOutReg for each bit that is set in the register mask.
  for (unsigned Reg = 0, NumRegs = TRI->getNumRegs(); Reg != NumRegs; ++Reg)
    if ((Mask[Reg / 32] >> Reg % 32) & 1)
      LiveOuts.push_back(createLiveOutReg(Reg, TRI));

  // We don't need to keep track of a register if its super-register is already
  // in the list. Merge entries that refer to the same dwarf register and use
  // the maximum size that needs to be spilled.
  std::sort(LiveOuts.begin(), LiveOuts.end());
  for (LiveOutVec::iterator I = LiveOuts.begin(), E = LiveOuts.end();
       I != E; ++I) {
    for (LiveOutVec::iterator II = std::next(I); II != E; ++II) {
      if (I->RegNo != II->RegNo) {
        // Skip all the now invalid entries.
        I = --II;
        break;
      }
      I->Size = std::max(I->Size, II->Size);
      if (TRI->isSuperRegister(I->Reg, II->Reg))
        I->Reg = II->Reg;
      II->MarkInvalid();
    }
  }
  LiveOuts.erase(std::remove_if(LiveOuts.begin(), LiveOuts.end(),
                                LiveOutReg::IsInvalid), LiveOuts.end());
  return LiveOuts;
}

void StackMaps::recordStackMapOpers(const MachineInstr &MI, uint64_t ID,
                                    MachineInstr::const_mop_iterator MOI,
                                    MachineInstr::const_mop_iterator MOE,
                                    bool recordResult) {

  MCContext &OutContext = AP.OutStreamer.getContext();
  MCSymbol *MILabel = OutContext.CreateTempSymbol();
  AP.OutStreamer.EmitLabel(MILabel);

  LocationVec Locations;
  LiveOutVec LiveOuts;

  if (recordResult) {
    assert(PatchPointOpers(&MI).hasDef() && "Stackmap has no return value.");
    parseOperand(MI.operands_begin(), std::next(MI.operands_begin()),
                 Locations, LiveOuts);
  }

  // Parse operands.
  while (MOI != MOE) {
    MOI = parseOperand(MOI, MOE, Locations, LiveOuts);
  }

  // Move large constants into the constant pool.
  for (LocationVec::iterator I = Locations.begin(), E = Locations.end();
       I != E; ++I) {
    // Constants are encoded as sign-extended integers.
    // -1 is directly encoded as .long 0xFFFFFFFF with no constant pool.
    if (I->LocType == Location::Constant &&
        ((I->Offset + (int64_t(1)<<31)) >> 32) != 0) {
      I->LocType = Location::ConstantIndex;
      I->Offset = ConstPool.getConstantIndex(I->Offset);
    }
  }

  // Create an expression to calculate the offset of the callsite from function
  // entry.
  const MCExpr *CSOffsetExpr = MCBinaryExpr::CreateSub(
    MCSymbolRefExpr::Create(MILabel, OutContext),
    MCSymbolRefExpr::Create(AP.CurrentFnSym, OutContext),
    OutContext);

  CSInfos.push_back(CallsiteInfo(CSOffsetExpr, ID, Locations, LiveOuts));

  // Record the stack size of the current function.
  const MachineFrameInfo *MFI = AP.MF->getFrameInfo();
  FnStackSize[AP.CurrentFnSym] =
    MFI->hasVarSizedObjects() ? UINT64_MAX : MFI->getStackSize();
}

void StackMaps::recordStackMap(const MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::STACKMAP && "expected stackmap");

  int64_t ID = MI.getOperand(0).getImm();
  recordStackMapOpers(MI, ID, std::next(MI.operands_begin(), 2),
                      MI.operands_end());
}

void StackMaps::recordPatchPoint(const MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::PATCHPOINT && "expected patchpoint");

  PatchPointOpers opers(&MI);
  int64_t ID = opers.getMetaOper(PatchPointOpers::IDPos).getImm();

  MachineInstr::const_mop_iterator MOI =
    std::next(MI.operands_begin(), opers.getStackMapStartIdx());
  recordStackMapOpers(MI, ID, MOI, MI.operands_end(),
                      opers.isAnyReg() && opers.hasDef());

#ifndef NDEBUG
  // verify anyregcc
  LocationVec &Locations = CSInfos.back().Locations;
  if (opers.isAnyReg()) {
    unsigned NArgs = opers.getMetaOper(PatchPointOpers::NArgPos).getImm();
    for (unsigned i = 0, e = (opers.hasDef() ? NArgs+1 : NArgs); i != e; ++i)
      assert(Locations[i].LocType == Location::Register &&
             "anyreg arg must be in reg.");
  }
#endif
}

/// serializeToStackMapSection conceptually populates the following fields:
///
/// Header {
///   uint8  : Stack Map Version (currently 1)
///   uint8  : Reserved (expected to be 0)
///   uint16 : Reserved (expected to be 0)
/// }
/// uint32 : NumFunctions
/// uint32 : NumConstants
/// uint32 : NumRecords
/// StkSizeRecord[NumFunctions] {
///   uint64 : Function Address
///   uint64 : Stack Size
/// }
/// int64  : Constants[NumConstants]
/// StkMapRecord[NumRecords] {
///   uint64 : PatchPoint ID
///   uint32 : Instruction Offset
///   uint16 : Reserved (record flags)
///   uint16 : NumLocations
///   Location[NumLocations] {
///     uint8  : Register | Direct | Indirect | Constant | ConstantIndex
///     uint8  : Size in Bytes
///     uint16 : Dwarf RegNum
///     int32  : Offset
///   }
///   uint16 : Padding
///   uint16 : NumLiveOuts
///   LiveOuts[NumLiveOuts] {
///     uint16 : Dwarf RegNum
///     uint8  : Reserved
///     uint8  : Size in Bytes
///   }
///   uint32 : Padding (only if required to align to 8 byte)
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
    OutContext.getObjectFileInfo()->getStackMapSection();
  AP.OutStreamer.SwitchSection(StackMapSection);

  // Emit a dummy symbol to force section inclusion.
  AP.OutStreamer.EmitLabel(
    OutContext.GetOrCreateSymbol(Twine("__LLVM_StackMaps")));

  // Serialize data.
  const char *WSMP = "Stack Maps: ";
  (void)WSMP;

  DEBUG(dbgs() << "********** Stack Map Output **********\n");

  // Header.
  AP.OutStreamer.EmitIntValue(1, 1); // Version.
  AP.OutStreamer.EmitIntValue(0, 1); // Reserved.
  AP.OutStreamer.EmitIntValue(0, 2); // Reserved.

  // Num functions.
  DEBUG(dbgs() << WSMP << "#functions = " << FnStackSize.size() << '\n');
  AP.OutStreamer.EmitIntValue(FnStackSize.size(), 4);
  // Num constants.
  DEBUG(dbgs() << WSMP << "#constants = " << ConstPool.getNumConstants()
               << '\n');
  AP.OutStreamer.EmitIntValue(ConstPool.getNumConstants(), 4);
  // Num callsites.
  DEBUG(dbgs() << WSMP << "#callsites = " << CSInfos.size() << '\n');
  AP.OutStreamer.EmitIntValue(CSInfos.size(), 4);

  // Function stack size entries.
  for (FnStackSizeMap::iterator I = FnStackSize.begin(), E = FnStackSize.end();
       I != E; ++I) {
    AP.OutStreamer.EmitSymbolValue(I->first, 8);
    AP.OutStreamer.EmitIntValue(I->second, 8);
  }

  // Constant pool entries.
  for (unsigned i = 0; i < ConstPool.getNumConstants(); ++i)
    AP.OutStreamer.EmitIntValue(ConstPool.getConstant(i), 8);

  // Callsite entries.
  for (CallsiteInfoList::const_iterator CSII = CSInfos.begin(),
       CSIE = CSInfos.end(); CSII != CSIE; ++CSII) {
    uint64_t CallsiteID = CSII->ID;
    const LocationVec &CSLocs = CSII->Locations;
    const LiveOutVec &LiveOuts = CSII->LiveOuts;

    DEBUG(dbgs() << WSMP << "callsite " << CallsiteID << "\n");

    // Verify stack map entry. It's better to communicate a problem to the
    // runtime than crash in case of in-process compilation. Currently, we do
    // simple overflow checks, but we may eventually communicate other
    // compilation errors this way.
    if (CSLocs.size() > UINT16_MAX || LiveOuts.size() > UINT16_MAX) {
      AP.OutStreamer.EmitIntValue(UINT64_MAX, 8); // Invalid ID.
      AP.OutStreamer.EmitValue(CSII->CSOffsetExpr, 4);
      AP.OutStreamer.EmitIntValue(0, 2); // Reserved.
      AP.OutStreamer.EmitIntValue(0, 2); // 0 locations.
      AP.OutStreamer.EmitIntValue(0, 2); // padding.
      AP.OutStreamer.EmitIntValue(0, 2); // 0 live-out registers.
      AP.OutStreamer.EmitIntValue(0, 4); // padding.
      continue;
    }

    AP.OutStreamer.EmitIntValue(CallsiteID, 8);
    AP.OutStreamer.EmitValue(CSII->CSOffsetExpr, 4);

    // Reserved for flags.
    AP.OutStreamer.EmitIntValue(0, 2);

    DEBUG(dbgs() << WSMP << "  has " << CSLocs.size() << " locations\n");

    AP.OutStreamer.EmitIntValue(CSLocs.size(), 2);

    unsigned operIdx = 0;
    for (LocationVec::const_iterator LocI = CSLocs.begin(), LocE = CSLocs.end();
         LocI != LocE; ++LocI, ++operIdx) {
      const Location &Loc = *LocI;
      unsigned RegNo = 0;
      int Offset = Loc.Offset;
      if(Loc.Reg) {
        RegNo = getDwarfRegNum(Loc.Reg, TRI);

        // If this is a register location, put the subregister byte offset in
        // the location offset.
        if (Loc.LocType == Location::Register) {
          assert(!Loc.Offset && "Register location should have zero offset");
          unsigned LLVMRegNo = TRI->getLLVMRegNum(RegNo, false);
          unsigned SubRegIdx = TRI->getSubRegIndex(LLVMRegNo, Loc.Reg);
          if (SubRegIdx)
            Offset = TRI->getSubRegIdxOffset(SubRegIdx);
        }
      }
      else {
        assert(Loc.LocType != Location::Register &&
               "Missing location register");
      }

      DEBUG(
        dbgs() << WSMP << "  Loc " << operIdx << ": ";
        switch (Loc.LocType) {
        case Location::Unprocessed:
          dbgs() << "<Unprocessed operand>";
          break;
        case Location::Register:
          dbgs() << "Register " << TRI->getName(Loc.Reg);
          break;
        case Location::Direct:
          dbgs() << "Direct " << TRI->getName(Loc.Reg);
          if (Loc.Offset)
            dbgs() << " + " << Loc.Offset;
          break;
        case Location::Indirect:
          dbgs() << "Indirect " << TRI->getName(Loc.Reg)
                 << " + " << Loc.Offset;
          break;
        case Location::Constant:
          dbgs() << "Constant " << Loc.Offset;
          break;
        case Location::ConstantIndex:
          dbgs() << "Constant Index " << Loc.Offset;
          break;
        }
        dbgs() << "     [encoding: .byte " << Loc.LocType
               << ", .byte " << Loc.Size
               << ", .short " << RegNo
               << ", .int " << Offset << "]\n";
      );

      AP.OutStreamer.EmitIntValue(Loc.LocType, 1);
      AP.OutStreamer.EmitIntValue(Loc.Size, 1);
      AP.OutStreamer.EmitIntValue(RegNo, 2);
      AP.OutStreamer.EmitIntValue(Offset, 4);
    }

    DEBUG(dbgs() << WSMP << "  has " << LiveOuts.size()
                 << " live-out registers\n");

    // Num live-out registers and padding to align to 4 byte.
    AP.OutStreamer.EmitIntValue(0, 2);
    AP.OutStreamer.EmitIntValue(LiveOuts.size(), 2);

    operIdx = 0;
    for (LiveOutVec::const_iterator LI = LiveOuts.begin(), LE = LiveOuts.end();
         LI != LE; ++LI, ++operIdx) {
      DEBUG(dbgs() << WSMP << "  LO " << operIdx << ": "
                   << TRI->getName(LI->Reg)
                   << "     [encoding: .short " << LI->RegNo
                   << ", .byte 0, .byte " << LI->Size << "]\n");

      AP.OutStreamer.EmitIntValue(LI->RegNo, 2);
      AP.OutStreamer.EmitIntValue(0, 1);
      AP.OutStreamer.EmitIntValue(LI->Size, 1);
    }
    // Emit alignment to 8 byte.
    AP.OutStreamer.EmitValueToAlignment(8);
  }

  AP.OutStreamer.AddBlankLine();

  CSInfos.clear();
}
