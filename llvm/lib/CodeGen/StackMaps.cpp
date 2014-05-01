//===---------------------------- StackMaps.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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

#define DEBUG_TYPE "stackmaps"

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
      auto Result = ConstPool.insert(std::make_pair(I->Offset, I->Offset));
      I->Offset = Result.first - ConstPool.begin();
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

/// Emit the stackmap header.
///
/// Header {
///   uint8  : Stack Map Version (currently 1)
///   uint8  : Reserved (expected to be 0)
///   uint16 : Reserved (expected to be 0)
/// }
/// uint32 : NumFunctions
/// uint32 : NumConstants
/// uint32 : NumRecords
void StackMaps::emitStackmapHeader(MCStreamer &OS) {
  // Header.
  OS.EmitIntValue(1, 1); // Version.
  OS.EmitIntValue(0, 1); // Reserved.
  OS.EmitIntValue(0, 2); // Reserved.

  // Num functions.
  DEBUG(dbgs() << WSMP << "#functions = " << FnStackSize.size() << '\n');
  OS.EmitIntValue(FnStackSize.size(), 4);
  // Num constants.
  DEBUG(dbgs() << WSMP << "#constants = " << ConstPool.size() << '\n');
  OS.EmitIntValue(ConstPool.size(), 4);
  // Num callsites.
  DEBUG(dbgs() << WSMP << "#callsites = " << CSInfos.size() << '\n');
  OS.EmitIntValue(CSInfos.size(), 4);
}

/// Emit the function frame record for each function.
///
/// StkSizeRecord[NumFunctions] {
///   uint64 : Function Address
///   uint64 : Stack Size
/// }
void StackMaps::emitFunctionFrameRecords(MCStreamer &OS) {
  // Function Frame records.
  DEBUG(dbgs() << WSMP << "functions:\n");
  for (auto const &FR : FnStackSize) {
    DEBUG(dbgs() << WSMP << "function addr: " << FR.first
                         << " frame size: " << FR.second);
    OS.EmitSymbolValue(FR.first, 8);
    OS.EmitIntValue(FR.second, 8);
  }
}

/// Emit the constant pool.
///
/// int64  : Constants[NumConstants]
void StackMaps::emitConstantPoolEntries(MCStreamer &OS) {
  // Constant pool entries.
  DEBUG(dbgs() << WSMP << "constants:\n");
  for (auto ConstEntry : ConstPool) {
    DEBUG(dbgs() << WSMP << ConstEntry.second << '\n');
    OS.EmitIntValue(ConstEntry.second, 8);
  }
}

/// Emit the callsite info for each callsite.
///
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
void StackMaps::emitCallsiteEntries(MCStreamer &OS,
                                    const TargetRegisterInfo *TRI) {
  // Callsite entries.
  DEBUG(dbgs() << WSMP << "callsites:\n");
  for (const auto &CSI : CSInfos) {
    const LocationVec &CSLocs = CSI.Locations;
    const LiveOutVec &LiveOuts = CSI.LiveOuts;

    DEBUG(dbgs() << WSMP << "callsite " << CSI.ID << "\n");

    // Verify stack map entry. It's better to communicate a problem to the
    // runtime than crash in case of in-process compilation. Currently, we do
    // simple overflow checks, but we may eventually communicate other
    // compilation errors this way.
    if (CSLocs.size() > UINT16_MAX || LiveOuts.size() > UINT16_MAX) {
      OS.EmitIntValue(UINT64_MAX, 8); // Invalid ID.
      OS.EmitValue(CSI.CSOffsetExpr, 4);
      OS.EmitIntValue(0, 2); // Reserved.
      OS.EmitIntValue(0, 2); // 0 locations.
      OS.EmitIntValue(0, 2); // padding.
      OS.EmitIntValue(0, 2); // 0 live-out registers.
      OS.EmitIntValue(0, 4); // padding.
      continue;
    }

    OS.EmitIntValue(CSI.ID, 8);
    OS.EmitValue(CSI.CSOffsetExpr, 4);

    // Reserved for flags.
    OS.EmitIntValue(0, 2);

    DEBUG(dbgs() << WSMP << "  has " << CSLocs.size() << " locations\n");

    OS.EmitIntValue(CSLocs.size(), 2);

    unsigned OperIdx = 0;
    for (const auto &Loc : CSLocs) {
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

      DEBUG(dbgs() << WSMP << "  Loc " << OperIdx << ": ";
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

      OS.EmitIntValue(Loc.LocType, 1);
      OS.EmitIntValue(Loc.Size, 1);
      OS.EmitIntValue(RegNo, 2);
      OS.EmitIntValue(Offset, 4);
      OperIdx++;
    }

    DEBUG(dbgs() << WSMP << "  has " << LiveOuts.size()
                         << " live-out registers\n");

    // Num live-out registers and padding to align to 4 byte.
    OS.EmitIntValue(0, 2);
    OS.EmitIntValue(LiveOuts.size(), 2);

    OperIdx = 0;
    for (const auto &LO : LiveOuts) {
      DEBUG(dbgs() << WSMP << "  LO " << OperIdx << ": "
                           << TRI->getName(LO.Reg)
                           << "     [encoding: .short " << LO.RegNo
                           << ", .byte 0, .byte " << LO.Size << "]\n");
      OS.EmitIntValue(LO.RegNo, 2);
      OS.EmitIntValue(0, 1);
      OS.EmitIntValue(LO.Size, 1);
    }
    // Emit alignment to 8 byte.
    OS.EmitValueToAlignment(8);
  }
}

/// Serialize the stackmap data.
void StackMaps::serializeToStackMapSection() {
  // Bail out if there's no stack map data.
  assert((!CSInfos.empty() || (CSInfos.empty() && ConstPool.empty())) &&
         "Expected empty constant pool too!");
  assert((!CSInfos.empty() || (CSInfos.empty() && FnStackSize.empty())) &&
         "Expected empty function record too!");
  if (CSInfos.empty())
    return;

  MCContext &OutContext = AP.OutStreamer.getContext();
  MCStreamer &OS = AP.OutStreamer;
  const TargetRegisterInfo *TRI = AP.TM.getRegisterInfo();

  // Create the section.
  const MCSection *StackMapSection =
    OutContext.getObjectFileInfo()->getStackMapSection();
  OS.SwitchSection(StackMapSection);

  // Emit a dummy symbol to force section inclusion.
  OS.EmitLabel(OutContext.GetOrCreateSymbol(Twine("__LLVM_StackMaps")));

  // Serialize data.
  DEBUG(dbgs() << "********** Stack Map Output **********\n");
  emitStackmapHeader(OS);
  emitFunctionFrameRecords(OS);
  emitConstantPoolEntries(OS);
  emitCallsiteEntries(OS, TRI);
  OS.AddBlankLine();

  // Clean up.
  CSInfos.clear();
  ConstPool.clear();
}
