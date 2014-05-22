//===-- ARM64ISelDAGToDAG.cpp - A dag to dag inst selector for ARM64 ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the ARM64 target.
//
//===----------------------------------------------------------------------===//

#include "ARM64TargetMachine.h"
#include "MCTargetDesc/ARM64AddressingModes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Function.h" // To access function attributes.
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "arm64-isel"

//===--------------------------------------------------------------------===//
/// ARM64DAGToDAGISel - ARM64 specific code to select ARM64 machine
/// instructions for SelectionDAG operations.
///
namespace {

class ARM64DAGToDAGISel : public SelectionDAGISel {
  ARM64TargetMachine &TM;

  /// Subtarget - Keep a pointer to the ARM64Subtarget around so that we can
  /// make the right decision when generating code for different targets.
  const ARM64Subtarget *Subtarget;

  bool ForCodeSize;

public:
  explicit ARM64DAGToDAGISel(ARM64TargetMachine &tm, CodeGenOpt::Level OptLevel)
      : SelectionDAGISel(tm, OptLevel), TM(tm),
        Subtarget(nullptr), ForCodeSize(false) {}

  const char *getPassName() const override {
    return "ARM64 Instruction Selection";
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    AttributeSet FnAttrs = MF.getFunction()->getAttributes();
    ForCodeSize =
        FnAttrs.hasAttribute(AttributeSet::FunctionIndex,
                             Attribute::OptimizeForSize) ||
        FnAttrs.hasAttribute(AttributeSet::FunctionIndex, Attribute::MinSize);
    Subtarget = &TM.getSubtarget<ARM64Subtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

  SDNode *Select(SDNode *Node) override;

  /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
  /// inline asm expressions.
  bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                    char ConstraintCode,
                                    std::vector<SDValue> &OutOps) override;

  SDNode *SelectMLAV64LaneV128(SDNode *N);
  SDNode *SelectMULLV64LaneV128(unsigned IntNo, SDNode *N);
  bool SelectArithExtendedRegister(SDValue N, SDValue &Reg, SDValue &Shift);
  bool SelectArithImmed(SDValue N, SDValue &Val, SDValue &Shift);
  bool SelectNegArithImmed(SDValue N, SDValue &Val, SDValue &Shift);
  bool SelectArithShiftedRegister(SDValue N, SDValue &Reg, SDValue &Shift) {
    return SelectShiftedRegister(N, false, Reg, Shift);
  }
  bool SelectLogicalShiftedRegister(SDValue N, SDValue &Reg, SDValue &Shift) {
    return SelectShiftedRegister(N, true, Reg, Shift);
  }
  bool SelectAddrModeIndexed8(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed(N, 1, Base, OffImm);
  }
  bool SelectAddrModeIndexed16(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed(N, 2, Base, OffImm);
  }
  bool SelectAddrModeIndexed32(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed(N, 4, Base, OffImm);
  }
  bool SelectAddrModeIndexed64(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed(N, 8, Base, OffImm);
  }
  bool SelectAddrModeIndexed128(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed(N, 16, Base, OffImm);
  }
  bool SelectAddrModeUnscaled8(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeUnscaled(N, 1, Base, OffImm);
  }
  bool SelectAddrModeUnscaled16(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeUnscaled(N, 2, Base, OffImm);
  }
  bool SelectAddrModeUnscaled32(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeUnscaled(N, 4, Base, OffImm);
  }
  bool SelectAddrModeUnscaled64(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeUnscaled(N, 8, Base, OffImm);
  }
  bool SelectAddrModeUnscaled128(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeUnscaled(N, 16, Base, OffImm);
  }

  bool SelectAddrModeRO8(SDValue N, SDValue &Base, SDValue &Offset,
                         SDValue &Imm) {
    return SelectAddrModeRO(N, 1, Base, Offset, Imm);
  }
  bool SelectAddrModeRO16(SDValue N, SDValue &Base, SDValue &Offset,
                          SDValue &Imm) {
    return SelectAddrModeRO(N, 2, Base, Offset, Imm);
  }
  bool SelectAddrModeRO32(SDValue N, SDValue &Base, SDValue &Offset,
                          SDValue &Imm) {
    return SelectAddrModeRO(N, 4, Base, Offset, Imm);
  }
  bool SelectAddrModeRO64(SDValue N, SDValue &Base, SDValue &Offset,
                          SDValue &Imm) {
    return SelectAddrModeRO(N, 8, Base, Offset, Imm);
  }
  bool SelectAddrModeRO128(SDValue N, SDValue &Base, SDValue &Offset,
                           SDValue &Imm) {
    return SelectAddrModeRO(N, 16, Base, Offset, Imm);
  }
  bool SelectAddrModeNoIndex(SDValue N, SDValue &Val);

  /// Form sequences of consecutive 64/128-bit registers for use in NEON
  /// instructions making use of a vector-list (e.g. ldN, tbl). Vecs must have
  /// between 1 and 4 elements. If it contains a single element that is returned
  /// unchanged; otherwise a REG_SEQUENCE value is returned.
  SDValue createDTuple(ArrayRef<SDValue> Vecs);
  SDValue createQTuple(ArrayRef<SDValue> Vecs);

  /// Generic helper for the createDTuple/createQTuple
  /// functions. Those should almost always be called instead.
  SDValue createTuple(ArrayRef<SDValue> Vecs, unsigned RegClassIDs[],
                      unsigned SubRegs[]);

  SDNode *SelectTable(SDNode *N, unsigned NumVecs, unsigned Opc, bool isExt);

  SDNode *SelectIndexedLoad(SDNode *N, bool &Done);

  SDNode *SelectLoad(SDNode *N, unsigned NumVecs, unsigned Opc,
                     unsigned SubRegIdx);
  SDNode *SelectPostLoad(SDNode *N, unsigned NumVecs, unsigned Opc,
                         unsigned SubRegIdx);
  SDNode *SelectLoadLane(SDNode *N, unsigned NumVecs, unsigned Opc);
  SDNode *SelectPostLoadLane(SDNode *N, unsigned NumVecs, unsigned Opc);

  SDNode *SelectStore(SDNode *N, unsigned NumVecs, unsigned Opc);
  SDNode *SelectPostStore(SDNode *N, unsigned NumVecs, unsigned Opc);
  SDNode *SelectStoreLane(SDNode *N, unsigned NumVecs, unsigned Opc);
  SDNode *SelectPostStoreLane(SDNode *N, unsigned NumVecs, unsigned Opc);

  SDNode *SelectSIMDAddSubNarrowing(unsigned IntNo, SDNode *Node);
  SDNode *SelectSIMDXtnNarrowing(unsigned IntNo, SDNode *Node);

  SDNode *SelectBitfieldExtractOp(SDNode *N);
  SDNode *SelectBitfieldInsertOp(SDNode *N);

  SDNode *SelectLIBM(SDNode *N);

// Include the pieces autogenerated from the target description.
#include "ARM64GenDAGISel.inc"

private:
  bool SelectShiftedRegister(SDValue N, bool AllowROR, SDValue &Reg,
                             SDValue &Shift);
  bool SelectAddrModeIndexed(SDValue N, unsigned Size, SDValue &Base,
                             SDValue &OffImm);
  bool SelectAddrModeUnscaled(SDValue N, unsigned Size, SDValue &Base,
                              SDValue &OffImm);
  bool SelectAddrModeRO(SDValue N, unsigned Size, SDValue &Base,
                        SDValue &Offset, SDValue &Imm);
  bool isWorthFolding(SDValue V) const;
  bool SelectExtendedSHL(SDValue N, unsigned Size, SDValue &Offset,
                         SDValue &Imm);

  template<unsigned RegWidth>
  bool SelectCVTFixedPosOperand(SDValue N, SDValue &FixedPos) {
    return SelectCVTFixedPosOperand(N, FixedPos, RegWidth);
  }

  bool SelectCVTFixedPosOperand(SDValue N, SDValue &FixedPos, unsigned Width);
};
} // end anonymous namespace

/// isIntImmediate - This method tests to see if the node is a constant
/// operand. If so Imm will receive the 32-bit value.
static bool isIntImmediate(const SDNode *N, uint64_t &Imm) {
  if (const ConstantSDNode *C = dyn_cast<const ConstantSDNode>(N)) {
    Imm = C->getZExtValue();
    return true;
  }
  return false;
}

// isIntImmediate - This method tests to see if a constant operand.
// If so Imm will receive the value.
static bool isIntImmediate(SDValue N, uint64_t &Imm) {
  return isIntImmediate(N.getNode(), Imm);
}

// isOpcWithIntImmediate - This method tests to see if the node is a specific
// opcode and that it has a immediate integer right operand.
// If so Imm will receive the 32 bit value.
static bool isOpcWithIntImmediate(const SDNode *N, unsigned Opc,
                                  uint64_t &Imm) {
  return N->getOpcode() == Opc &&
         isIntImmediate(N->getOperand(1).getNode(), Imm);
}

bool ARM64DAGToDAGISel::SelectAddrModeNoIndex(SDValue N, SDValue &Val) {
  EVT ValTy = N.getValueType();
  if (ValTy != MVT::i64)
    return false;
  Val = N;
  return true;
}

bool ARM64DAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, char ConstraintCode, std::vector<SDValue> &OutOps) {
  assert(ConstraintCode == 'm' && "unexpected asm memory constraint");
  // Require the address to be in a register.  That is safe for all ARM64
  // variants and it is hard to do anything much smarter without knowing
  // how the operand is used.
  OutOps.push_back(Op);
  return false;
}

/// SelectArithImmed - Select an immediate value that can be represented as
/// a 12-bit value shifted left by either 0 or 12.  If so, return true with
/// Val set to the 12-bit value and Shift set to the shifter operand.
bool ARM64DAGToDAGISel::SelectArithImmed(SDValue N, SDValue &Val,
                                         SDValue &Shift) {
  // This function is called from the addsub_shifted_imm ComplexPattern,
  // which lists [imm] as the list of opcode it's interested in, however
  // we still need to check whether the operand is actually an immediate
  // here because the ComplexPattern opcode list is only used in
  // root-level opcode matching.
  if (!isa<ConstantSDNode>(N.getNode()))
    return false;

  uint64_t Immed = cast<ConstantSDNode>(N.getNode())->getZExtValue();
  unsigned ShiftAmt;

  if (Immed >> 12 == 0) {
    ShiftAmt = 0;
  } else if ((Immed & 0xfff) == 0 && Immed >> 24 == 0) {
    ShiftAmt = 12;
    Immed = Immed >> 12;
  } else
    return false;

  unsigned ShVal = ARM64_AM::getShifterImm(ARM64_AM::LSL, ShiftAmt);
  Val = CurDAG->getTargetConstant(Immed, MVT::i32);
  Shift = CurDAG->getTargetConstant(ShVal, MVT::i32);
  return true;
}

/// SelectNegArithImmed - As above, but negates the value before trying to
/// select it.
bool ARM64DAGToDAGISel::SelectNegArithImmed(SDValue N, SDValue &Val,
                                            SDValue &Shift) {
  // This function is called from the addsub_shifted_imm ComplexPattern,
  // which lists [imm] as the list of opcode it's interested in, however
  // we still need to check whether the operand is actually an immediate
  // here because the ComplexPattern opcode list is only used in
  // root-level opcode matching.
  if (!isa<ConstantSDNode>(N.getNode()))
    return false;

  // The immediate operand must be a 24-bit zero-extended immediate.
  uint64_t Immed = cast<ConstantSDNode>(N.getNode())->getZExtValue();

  // This negation is almost always valid, but "cmp wN, #0" and "cmn wN, #0"
  // have the opposite effect on the C flag, so this pattern mustn't match under
  // those circumstances.
  if (Immed == 0)
    return false;

  if (N.getValueType() == MVT::i32)
    Immed = ~((uint32_t)Immed) + 1;
  else
    Immed = ~Immed + 1ULL;
  if (Immed & 0xFFFFFFFFFF000000ULL)
    return false;

  Immed &= 0xFFFFFFULL;
  return SelectArithImmed(CurDAG->getConstant(Immed, MVT::i32), Val, Shift);
}

/// getShiftTypeForNode - Translate a shift node to the corresponding
/// ShiftType value.
static ARM64_AM::ShiftExtendType getShiftTypeForNode(SDValue N) {
  switch (N.getOpcode()) {
  default:
    return ARM64_AM::InvalidShiftExtend;
  case ISD::SHL:
    return ARM64_AM::LSL;
  case ISD::SRL:
    return ARM64_AM::LSR;
  case ISD::SRA:
    return ARM64_AM::ASR;
  case ISD::ROTR:
    return ARM64_AM::ROR;
  }
}

/// \brief Determine wether it is worth to fold V into an extended register.
bool ARM64DAGToDAGISel::isWorthFolding(SDValue V) const {
  // it hurts if the a value is used at least twice, unless we are optimizing
  // for code size.
  if (ForCodeSize || V.hasOneUse())
    return true;
  return false;
}

/// SelectShiftedRegister - Select a "shifted register" operand.  If the value
/// is not shifted, set the Shift operand to default of "LSL 0".  The logical
/// instructions allow the shifted register to be rotated, but the arithmetic
/// instructions do not.  The AllowROR parameter specifies whether ROR is
/// supported.
bool ARM64DAGToDAGISel::SelectShiftedRegister(SDValue N, bool AllowROR,
                                              SDValue &Reg, SDValue &Shift) {
  ARM64_AM::ShiftExtendType ShType = getShiftTypeForNode(N);
  if (ShType == ARM64_AM::InvalidShiftExtend)
    return false;
  if (!AllowROR && ShType == ARM64_AM::ROR)
    return false;

  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    unsigned BitSize = N.getValueType().getSizeInBits();
    unsigned Val = RHS->getZExtValue() & (BitSize - 1);
    unsigned ShVal = ARM64_AM::getShifterImm(ShType, Val);

    Reg = N.getOperand(0);
    Shift = CurDAG->getTargetConstant(ShVal, MVT::i32);
    return isWorthFolding(N);
  }

  return false;
}

/// getExtendTypeForNode - Translate an extend node to the corresponding
/// ExtendType value.
static ARM64_AM::ShiftExtendType
getExtendTypeForNode(SDValue N, bool IsLoadStore = false) {
  if (N.getOpcode() == ISD::SIGN_EXTEND ||
      N.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    EVT SrcVT;
    if (N.getOpcode() == ISD::SIGN_EXTEND_INREG)
      SrcVT = cast<VTSDNode>(N.getOperand(1))->getVT();
    else
      SrcVT = N.getOperand(0).getValueType();

    if (!IsLoadStore && SrcVT == MVT::i8)
      return ARM64_AM::SXTB;
    else if (!IsLoadStore && SrcVT == MVT::i16)
      return ARM64_AM::SXTH;
    else if (SrcVT == MVT::i32)
      return ARM64_AM::SXTW;
    assert(SrcVT != MVT::i64 && "extend from 64-bits?");

    return ARM64_AM::InvalidShiftExtend;
  } else if (N.getOpcode() == ISD::ZERO_EXTEND ||
             N.getOpcode() == ISD::ANY_EXTEND) {
    EVT SrcVT = N.getOperand(0).getValueType();
    if (!IsLoadStore && SrcVT == MVT::i8)
      return ARM64_AM::UXTB;
    else if (!IsLoadStore && SrcVT == MVT::i16)
      return ARM64_AM::UXTH;
    else if (SrcVT == MVT::i32)
      return ARM64_AM::UXTW;
    assert(SrcVT != MVT::i64 && "extend from 64-bits?");

    return ARM64_AM::InvalidShiftExtend;
  } else if (N.getOpcode() == ISD::AND) {
    ConstantSDNode *CSD = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (!CSD)
      return ARM64_AM::InvalidShiftExtend;
    uint64_t AndMask = CSD->getZExtValue();

    switch (AndMask) {
    default:
      return ARM64_AM::InvalidShiftExtend;
    case 0xFF:
      return !IsLoadStore ? ARM64_AM::UXTB : ARM64_AM::InvalidShiftExtend;
    case 0xFFFF:
      return !IsLoadStore ? ARM64_AM::UXTH : ARM64_AM::InvalidShiftExtend;
    case 0xFFFFFFFF:
      return ARM64_AM::UXTW;
    }
  }

  return ARM64_AM::InvalidShiftExtend;
}

// Helper for SelectMLAV64LaneV128 - Recognize high lane extracts.
static bool checkHighLaneIndex(SDNode *DL, SDValue &LaneOp, int &LaneIdx) {
  if (DL->getOpcode() != ARM64ISD::DUPLANE16 &&
      DL->getOpcode() != ARM64ISD::DUPLANE32)
    return false;

  SDValue SV = DL->getOperand(0);
  if (SV.getOpcode() != ISD::INSERT_SUBVECTOR)
    return false;

  SDValue EV = SV.getOperand(1);
  if (EV.getOpcode() != ISD::EXTRACT_SUBVECTOR)
    return false;

  ConstantSDNode *DLidx = cast<ConstantSDNode>(DL->getOperand(1).getNode());
  ConstantSDNode *EVidx = cast<ConstantSDNode>(EV.getOperand(1).getNode());
  LaneIdx = DLidx->getSExtValue() + EVidx->getSExtValue();
  LaneOp = EV.getOperand(0);

  return true;
}

// Helper for SelectOpcV64LaneV128 - Recogzine operatinos where one operand is a
// high lane extract.
static bool checkV64LaneV128(SDValue Op0, SDValue Op1, SDValue &StdOp,
                             SDValue &LaneOp, int &LaneIdx) {

  if (!checkHighLaneIndex(Op0.getNode(), LaneOp, LaneIdx)) {
    std::swap(Op0, Op1);
    if (!checkHighLaneIndex(Op0.getNode(), LaneOp, LaneIdx))
      return false;
  }
  StdOp = Op1;
  return true;
}

/// SelectMLAV64LaneV128 - ARM64 supports vector MLAs where one multiplicand is
/// a lane in the upper half of a 128-bit vector.  Recognize and select this so
/// that we don't emit unnecessary lane extracts.
SDNode *ARM64DAGToDAGISel::SelectMLAV64LaneV128(SDNode *N) {
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  SDValue MLAOp1;   // Will hold ordinary multiplicand for MLA.
  SDValue MLAOp2;   // Will hold lane-accessed multiplicand for MLA.
  int LaneIdx = -1; // Will hold the lane index.

  if (Op1.getOpcode() != ISD::MUL ||
      !checkV64LaneV128(Op1.getOperand(0), Op1.getOperand(1), MLAOp1, MLAOp2,
                        LaneIdx)) {
    std::swap(Op0, Op1);
    if (Op1.getOpcode() != ISD::MUL ||
        !checkV64LaneV128(Op1.getOperand(0), Op1.getOperand(1), MLAOp1, MLAOp2,
                          LaneIdx))
      return nullptr;
  }

  SDValue LaneIdxVal = CurDAG->getTargetConstant(LaneIdx, MVT::i64);

  SDValue Ops[] = { Op0, MLAOp1, MLAOp2, LaneIdxVal };

  unsigned MLAOpc = ~0U;

  switch (N->getSimpleValueType(0).SimpleTy) {
  default:
    llvm_unreachable("Unrecognized MLA.");
  case MVT::v4i16:
    MLAOpc = ARM64::MLAv4i16_indexed;
    break;
  case MVT::v8i16:
    MLAOpc = ARM64::MLAv8i16_indexed;
    break;
  case MVT::v2i32:
    MLAOpc = ARM64::MLAv2i32_indexed;
    break;
  case MVT::v4i32:
    MLAOpc = ARM64::MLAv4i32_indexed;
    break;
  }

  return CurDAG->getMachineNode(MLAOpc, SDLoc(N), N->getValueType(0), Ops);
}

SDNode *ARM64DAGToDAGISel::SelectMULLV64LaneV128(unsigned IntNo, SDNode *N) {
  SDValue SMULLOp0;
  SDValue SMULLOp1;
  int LaneIdx;

  if (!checkV64LaneV128(N->getOperand(1), N->getOperand(2), SMULLOp0, SMULLOp1,
                        LaneIdx))
    return nullptr;

  SDValue LaneIdxVal = CurDAG->getTargetConstant(LaneIdx, MVT::i64);

  SDValue Ops[] = { SMULLOp0, SMULLOp1, LaneIdxVal };

  unsigned SMULLOpc = ~0U;

  if (IntNo == Intrinsic::arm64_neon_smull) {
    switch (N->getSimpleValueType(0).SimpleTy) {
    default:
      llvm_unreachable("Unrecognized SMULL.");
    case MVT::v4i32:
      SMULLOpc = ARM64::SMULLv4i16_indexed;
      break;
    case MVT::v2i64:
      SMULLOpc = ARM64::SMULLv2i32_indexed;
      break;
    }
  } else if (IntNo == Intrinsic::arm64_neon_umull) {
    switch (N->getSimpleValueType(0).SimpleTy) {
    default:
      llvm_unreachable("Unrecognized SMULL.");
    case MVT::v4i32:
      SMULLOpc = ARM64::UMULLv4i16_indexed;
      break;
    case MVT::v2i64:
      SMULLOpc = ARM64::UMULLv2i32_indexed;
      break;
    }
  } else
    llvm_unreachable("Unrecognized intrinsic.");

  return CurDAG->getMachineNode(SMULLOpc, SDLoc(N), N->getValueType(0), Ops);
}

/// SelectArithExtendedRegister - Select a "extended register" operand.  This
/// operand folds in an extend followed by an optional left shift.
bool ARM64DAGToDAGISel::SelectArithExtendedRegister(SDValue N, SDValue &Reg,
                                                    SDValue &Shift) {
  unsigned ShiftVal = 0;
  ARM64_AM::ShiftExtendType Ext;

  if (N.getOpcode() == ISD::SHL) {
    ConstantSDNode *CSD = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (!CSD)
      return false;
    ShiftVal = CSD->getZExtValue();
    if (ShiftVal > 4)
      return false;

    Ext = getExtendTypeForNode(N.getOperand(0));
    if (Ext == ARM64_AM::InvalidShiftExtend)
      return false;

    Reg = N.getOperand(0).getOperand(0);
  } else {
    Ext = getExtendTypeForNode(N);
    if (Ext == ARM64_AM::InvalidShiftExtend)
      return false;

    Reg = N.getOperand(0);
  }

  // ARM64 mandates that the RHS of the operation must use the smallest
  // register classs that could contain the size being extended from.  Thus,
  // if we're folding a (sext i8), we need the RHS to be a GPR32, even though
  // there might not be an actual 32-bit value in the program.  We can
  // (harmlessly) synthesize one by injected an EXTRACT_SUBREG here.
  if (Reg.getValueType() == MVT::i64 && Ext != ARM64_AM::UXTX &&
      Ext != ARM64_AM::SXTX) {
    SDValue SubReg = CurDAG->getTargetConstant(ARM64::sub_32, MVT::i32);
    MachineSDNode *Node = CurDAG->getMachineNode(
        TargetOpcode::EXTRACT_SUBREG, SDLoc(N), MVT::i32, Reg, SubReg);
    Reg = SDValue(Node, 0);
  }

  Shift = CurDAG->getTargetConstant(getArithExtendImm(Ext, ShiftVal), MVT::i32);
  return isWorthFolding(N);
}

/// SelectAddrModeIndexed - Select a "register plus scaled unsigned 12-bit
/// immediate" address.  The "Size" argument is the size in bytes of the memory
/// reference, which determines the scale.
bool ARM64DAGToDAGISel::SelectAddrModeIndexed(SDValue N, unsigned Size,
                                              SDValue &Base, SDValue &OffImm) {
  const TargetLowering *TLI = getTargetLowering();
  if (N.getOpcode() == ISD::FrameIndex) {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    Base = CurDAG->getTargetFrameIndex(FI, TLI->getPointerTy());
    OffImm = CurDAG->getTargetConstant(0, MVT::i64);
    return true;
  }

  if (N.getOpcode() == ARM64ISD::ADDlow) {
    GlobalAddressSDNode *GAN =
        dyn_cast<GlobalAddressSDNode>(N.getOperand(1).getNode());
    Base = N.getOperand(0);
    OffImm = N.getOperand(1);
    if (!GAN)
      return true;

    const GlobalValue *GV = GAN->getGlobal();
    unsigned Alignment = GV->getAlignment();
    const DataLayout *DL = TLI->getDataLayout();
    if (Alignment == 0 && !Subtarget->isTargetDarwin())
      Alignment = DL->getABITypeAlignment(GV->getType()->getElementType());

    if (Alignment >= Size)
      return true;
  }

  if (CurDAG->isBaseWithConstantOffset(N)) {
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      int64_t RHSC = (int64_t)RHS->getZExtValue();
      unsigned Scale = Log2_32(Size);
      if ((RHSC & (Size - 1)) == 0 && RHSC >= 0 && RHSC < (0x1000 << Scale)) {
        Base = N.getOperand(0);
        if (Base.getOpcode() == ISD::FrameIndex) {
          int FI = cast<FrameIndexSDNode>(Base)->getIndex();
          Base = CurDAG->getTargetFrameIndex(FI, TLI->getPointerTy());
        }
        OffImm = CurDAG->getTargetConstant(RHSC >> Scale, MVT::i64);
        return true;
      }
    }
  }

  // Before falling back to our general case, check if the unscaled
  // instructions can handle this. If so, that's preferable.
  if (SelectAddrModeUnscaled(N, Size, Base, OffImm))
    return false;

  // Base only. The address will be materialized into a register before
  // the memory is accessed.
  //    add x0, Xbase, #offset
  //    ldr x0, [x0]
  Base = N;
  OffImm = CurDAG->getTargetConstant(0, MVT::i64);
  return true;
}

/// SelectAddrModeUnscaled - Select a "register plus unscaled signed 9-bit
/// immediate" address.  This should only match when there is an offset that
/// is not valid for a scaled immediate addressing mode.  The "Size" argument
/// is the size in bytes of the memory reference, which is needed here to know
/// what is valid for a scaled immediate.
bool ARM64DAGToDAGISel::SelectAddrModeUnscaled(SDValue N, unsigned Size,
                                               SDValue &Base, SDValue &OffImm) {
  if (!CurDAG->isBaseWithConstantOffset(N))
    return false;
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    int64_t RHSC = RHS->getSExtValue();
    // If the offset is valid as a scaled immediate, don't match here.
    if ((RHSC & (Size - 1)) == 0 && RHSC >= 0 &&
        RHSC < (0x1000 << Log2_32(Size)))
      return false;
    if (RHSC >= -256 && RHSC < 256) {
      Base = N.getOperand(0);
      if (Base.getOpcode() == ISD::FrameIndex) {
        int FI = cast<FrameIndexSDNode>(Base)->getIndex();
        const TargetLowering *TLI = getTargetLowering();
        Base = CurDAG->getTargetFrameIndex(FI, TLI->getPointerTy());
      }
      OffImm = CurDAG->getTargetConstant(RHSC, MVT::i64);
      return true;
    }
  }
  return false;
}

static SDValue Widen(SelectionDAG *CurDAG, SDValue N) {
  SDValue SubReg = CurDAG->getTargetConstant(ARM64::sub_32, MVT::i32);
  SDValue ImpDef = SDValue(
      CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, SDLoc(N), MVT::i64),
      0);
  MachineSDNode *Node = CurDAG->getMachineNode(
      TargetOpcode::INSERT_SUBREG, SDLoc(N), MVT::i64, ImpDef, N, SubReg);
  return SDValue(Node, 0);
}

static SDValue WidenIfNeeded(SelectionDAG *CurDAG, SDValue N) {
  if (N.getValueType() == MVT::i32) {
    return Widen(CurDAG, N);
  }

  return N;
}

/// \brief Check if the given SHL node (\p N), can be used to form an
/// extended register for an addressing mode.
bool ARM64DAGToDAGISel::SelectExtendedSHL(SDValue N, unsigned Size,
                                          SDValue &Offset, SDValue &Imm) {
  assert(N.getOpcode() == ISD::SHL && "Invalid opcode.");
  ConstantSDNode *CSD = dyn_cast<ConstantSDNode>(N.getOperand(1));
  if (CSD && (CSD->getZExtValue() & 0x7) == CSD->getZExtValue()) {

    ARM64_AM::ShiftExtendType Ext = getExtendTypeForNode(N.getOperand(0), true);
    if (Ext == ARM64_AM::InvalidShiftExtend) {
      Ext = ARM64_AM::UXTX;
      Offset = WidenIfNeeded(CurDAG, N.getOperand(0));
    } else {
      Offset = WidenIfNeeded(CurDAG, N.getOperand(0).getOperand(0));
    }

    unsigned LegalShiftVal = Log2_32(Size);
    unsigned ShiftVal = CSD->getZExtValue();

    if (ShiftVal != 0 && ShiftVal != LegalShiftVal)
      return false;

    Imm = CurDAG->getTargetConstant(
        ARM64_AM::getMemExtendImm(Ext, ShiftVal != 0), MVT::i32);
    if (isWorthFolding(N))
      return true;
  }
  return false;
}

bool ARM64DAGToDAGISel::SelectAddrModeRO(SDValue N, unsigned Size,
                                         SDValue &Base, SDValue &Offset,
                                         SDValue &Imm) {
  if (N.getOpcode() != ISD::ADD)
    return false;
  SDValue LHS = N.getOperand(0);
  SDValue RHS = N.getOperand(1);

  // We don't want to match immediate adds here, because they are better lowered
  // to the register-immediate addressing modes.
  if (isa<ConstantSDNode>(LHS) || isa<ConstantSDNode>(RHS))
    return false;

  // Check if this particular node is reused in any non-memory related
  // operation.  If yes, do not try to fold this node into the address
  // computation, since the computation will be kept.
  const SDNode *Node = N.getNode();
  for (SDNode *UI : Node->uses()) {
    if (!isa<MemSDNode>(*UI))
      return false;
  }

  // Remember if it is worth folding N when it produces extended register.
  bool IsExtendedRegisterWorthFolding = isWorthFolding(N);

  // Try to match a shifted extend on the RHS.
  if (IsExtendedRegisterWorthFolding && RHS.getOpcode() == ISD::SHL &&
      SelectExtendedSHL(RHS, Size, Offset, Imm)) {
    Base = LHS;
    return true;
  }

  // Try to match a shifted extend on the LHS.
  if (IsExtendedRegisterWorthFolding && LHS.getOpcode() == ISD::SHL &&
      SelectExtendedSHL(LHS, Size, Offset, Imm)) {
    Base = RHS;
    return true;
  }

  ARM64_AM::ShiftExtendType Ext = ARM64_AM::UXTX;
  // Try to match an unshifted extend on the LHS.
  if (IsExtendedRegisterWorthFolding &&
      (Ext = getExtendTypeForNode(LHS, true)) != ARM64_AM::InvalidShiftExtend) {
    Base = RHS;
    Offset = WidenIfNeeded(CurDAG, LHS.getOperand(0));
    Imm = CurDAG->getTargetConstant(ARM64_AM::getMemExtendImm(Ext, false),
                                    MVT::i32);
    if (isWorthFolding(LHS))
      return true;
  }

  // Try to match an unshifted extend on the RHS.
  if (IsExtendedRegisterWorthFolding &&
      (Ext = getExtendTypeForNode(RHS, true)) != ARM64_AM::InvalidShiftExtend) {
    Base = LHS;
    Offset = WidenIfNeeded(CurDAG, RHS.getOperand(0));
    Imm = CurDAG->getTargetConstant(ARM64_AM::getMemExtendImm(Ext, false),
                                    MVT::i32);
    if (isWorthFolding(RHS))
      return true;
  }

  // Match any non-shifted, non-extend, non-immediate add expression.
  Base = LHS;
  Offset = WidenIfNeeded(CurDAG, RHS);
  Ext = ARM64_AM::UXTX;
  Imm = CurDAG->getTargetConstant(ARM64_AM::getMemExtendImm(Ext, false),
                                  MVT::i32);
  // Reg1 + Reg2 is free: no check needed.
  return true;
}

SDValue ARM64DAGToDAGISel::createDTuple(ArrayRef<SDValue> Regs) {
  static unsigned RegClassIDs[] = { ARM64::DDRegClassID, ARM64::DDDRegClassID,
                                    ARM64::DDDDRegClassID };
  static unsigned SubRegs[] = { ARM64::dsub0, ARM64::dsub1,
                                ARM64::dsub2, ARM64::dsub3 };

  return createTuple(Regs, RegClassIDs, SubRegs);
}

SDValue ARM64DAGToDAGISel::createQTuple(ArrayRef<SDValue> Regs) {
  static unsigned RegClassIDs[] = { ARM64::QQRegClassID, ARM64::QQQRegClassID,
                                    ARM64::QQQQRegClassID };
  static unsigned SubRegs[] = { ARM64::qsub0, ARM64::qsub1,
                                ARM64::qsub2, ARM64::qsub3 };

  return createTuple(Regs, RegClassIDs, SubRegs);
}

SDValue ARM64DAGToDAGISel::createTuple(ArrayRef<SDValue> Regs,
                                       unsigned RegClassIDs[],
                                       unsigned SubRegs[]) {
  // There's no special register-class for a vector-list of 1 element: it's just
  // a vector.
  if (Regs.size() == 1)
    return Regs[0];

  assert(Regs.size() >= 2 && Regs.size() <= 4);

  SDLoc DL(Regs[0].getNode());

  SmallVector<SDValue, 4> Ops;

  // First operand of REG_SEQUENCE is the desired RegClass.
  Ops.push_back(
      CurDAG->getTargetConstant(RegClassIDs[Regs.size() - 2], MVT::i32));

  // Then we get pairs of source & subregister-position for the components.
  for (unsigned i = 0; i < Regs.size(); ++i) {
    Ops.push_back(Regs[i]);
    Ops.push_back(CurDAG->getTargetConstant(SubRegs[i], MVT::i32));
  }

  SDNode *N =
      CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL, MVT::Untyped, Ops);
  return SDValue(N, 0);
}

SDNode *ARM64DAGToDAGISel::SelectTable(SDNode *N, unsigned NumVecs,
                                       unsigned Opc, bool isExt) {
  SDLoc dl(N);
  EVT VT = N->getValueType(0);

  unsigned ExtOff = isExt;

  // Form a REG_SEQUENCE to force register allocation.
  unsigned Vec0Off = ExtOff + 1;
  SmallVector<SDValue, 4> Regs(N->op_begin() + Vec0Off,
                               N->op_begin() + Vec0Off + NumVecs);
  SDValue RegSeq = createQTuple(Regs);

  SmallVector<SDValue, 6> Ops;
  if (isExt)
    Ops.push_back(N->getOperand(1));
  Ops.push_back(RegSeq);
  Ops.push_back(N->getOperand(NumVecs + ExtOff + 1));
  return CurDAG->getMachineNode(Opc, dl, VT, Ops);
}

SDNode *ARM64DAGToDAGISel::SelectIndexedLoad(SDNode *N, bool &Done) {
  LoadSDNode *LD = cast<LoadSDNode>(N);
  if (LD->isUnindexed())
    return nullptr;
  EVT VT = LD->getMemoryVT();
  EVT DstVT = N->getValueType(0);
  ISD::MemIndexedMode AM = LD->getAddressingMode();
  bool IsPre = AM == ISD::PRE_INC || AM == ISD::PRE_DEC;

  // We're not doing validity checking here. That was done when checking
  // if we should mark the load as indexed or not. We're just selecting
  // the right instruction.
  unsigned Opcode = 0;

  ISD::LoadExtType ExtType = LD->getExtensionType();
  bool InsertTo64 = false;
  if (VT == MVT::i64)
    Opcode = IsPre ? ARM64::LDRXpre_isel : ARM64::LDRXpost_isel;
  else if (VT == MVT::i32) {
    if (ExtType == ISD::NON_EXTLOAD)
      Opcode = IsPre ? ARM64::LDRWpre_isel : ARM64::LDRWpost_isel;
    else if (ExtType == ISD::SEXTLOAD)
      Opcode = IsPre ? ARM64::LDRSWpre_isel : ARM64::LDRSWpost_isel;
    else {
      Opcode = IsPre ? ARM64::LDRWpre_isel : ARM64::LDRWpost_isel;
      InsertTo64 = true;
      // The result of the load is only i32. It's the subreg_to_reg that makes
      // it into an i64.
      DstVT = MVT::i32;
    }
  } else if (VT == MVT::i16) {
    if (ExtType == ISD::SEXTLOAD) {
      if (DstVT == MVT::i64)
        Opcode = IsPre ? ARM64::LDRSHXpre_isel : ARM64::LDRSHXpost_isel;
      else
        Opcode = IsPre ? ARM64::LDRSHWpre_isel : ARM64::LDRSHWpost_isel;
    } else {
      Opcode = IsPre ? ARM64::LDRHHpre_isel : ARM64::LDRHHpost_isel;
      InsertTo64 = DstVT == MVT::i64;
      // The result of the load is only i32. It's the subreg_to_reg that makes
      // it into an i64.
      DstVT = MVT::i32;
    }
  } else if (VT == MVT::i8) {
    if (ExtType == ISD::SEXTLOAD) {
      if (DstVT == MVT::i64)
        Opcode = IsPre ? ARM64::LDRSBXpre_isel : ARM64::LDRSBXpost_isel;
      else
        Opcode = IsPre ? ARM64::LDRSBWpre_isel : ARM64::LDRSBWpost_isel;
    } else {
      Opcode = IsPre ? ARM64::LDRBBpre_isel : ARM64::LDRBBpost_isel;
      InsertTo64 = DstVT == MVT::i64;
      // The result of the load is only i32. It's the subreg_to_reg that makes
      // it into an i64.
      DstVT = MVT::i32;
    }
  } else if (VT == MVT::f32) {
    Opcode = IsPre ? ARM64::LDRSpre_isel : ARM64::LDRSpost_isel;
  } else if (VT == MVT::f64 || VT.is64BitVector()) {
    Opcode = IsPre ? ARM64::LDRDpre_isel : ARM64::LDRDpost_isel;
  } else if (VT.is128BitVector()) {
    Opcode = IsPre ? ARM64::LDRQpre_isel : ARM64::LDRQpost_isel;
  } else
    return nullptr;
  SDValue Chain = LD->getChain();
  SDValue Base = LD->getBasePtr();
  ConstantSDNode *OffsetOp = cast<ConstantSDNode>(LD->getOffset());
  int OffsetVal = (int)OffsetOp->getZExtValue();
  SDValue Offset = CurDAG->getTargetConstant(OffsetVal, MVT::i64);
  SDValue Ops[] = { Base, Offset, Chain };
  SDNode *Res = CurDAG->getMachineNode(Opcode, SDLoc(N), DstVT, MVT::i64,
                                       MVT::Other, Ops);
  // Either way, we're replacing the node, so tell the caller that.
  Done = true;
  if (InsertTo64) {
    SDValue SubReg = CurDAG->getTargetConstant(ARM64::sub_32, MVT::i32);
    SDNode *Sub = CurDAG->getMachineNode(
        ARM64::SUBREG_TO_REG, SDLoc(N), MVT::i64,
        CurDAG->getTargetConstant(0, MVT::i64), SDValue(Res, 0), SubReg);
    ReplaceUses(SDValue(N, 0), SDValue(Sub, 0));
    ReplaceUses(SDValue(N, 1), SDValue(Res, 1));
    ReplaceUses(SDValue(N, 2), SDValue(Res, 2));
    return nullptr;
  }
  return Res;
}

SDNode *ARM64DAGToDAGISel::SelectLoad(SDNode *N, unsigned NumVecs, unsigned Opc,
                                      unsigned SubRegIdx) {
  SDLoc dl(N);
  EVT VT = N->getValueType(0);
  SDValue Chain = N->getOperand(0);

  SmallVector<SDValue, 6> Ops;
  Ops.push_back(N->getOperand(2)); // Mem operand;
  Ops.push_back(Chain);

  std::vector<EVT> ResTys;
  ResTys.push_back(MVT::Untyped);
  ResTys.push_back(MVT::Other);

  SDNode *Ld = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);
  SDValue SuperReg = SDValue(Ld, 0);
  for (unsigned i = 0; i < NumVecs; ++i)
    ReplaceUses(SDValue(N, i),
        CurDAG->getTargetExtractSubreg(SubRegIdx + i, dl, VT, SuperReg));

  ReplaceUses(SDValue(N, NumVecs), SDValue(Ld, 1));
  return nullptr;
}

SDNode *ARM64DAGToDAGISel::SelectPostLoad(SDNode *N, unsigned NumVecs,
                                          unsigned Opc, unsigned SubRegIdx) {
  SDLoc dl(N);
  EVT VT = N->getValueType(0);
  SDValue Chain = N->getOperand(0);

  SmallVector<SDValue, 6> Ops;
  Ops.push_back(N->getOperand(1)); // Mem operand
  Ops.push_back(N->getOperand(2)); // Incremental
  Ops.push_back(Chain);

  std::vector<EVT> ResTys;
  ResTys.push_back(MVT::i64); // Type of the write back register
  ResTys.push_back(MVT::Untyped);
  ResTys.push_back(MVT::Other);

  SDNode *Ld = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);

  // Update uses of write back register
  ReplaceUses(SDValue(N, NumVecs), SDValue(Ld, 0));

  // Update uses of vector list
  SDValue SuperReg = SDValue(Ld, 1);
  if (NumVecs == 1)
    ReplaceUses(SDValue(N, 0), SuperReg);
  else
    for (unsigned i = 0; i < NumVecs; ++i)
      ReplaceUses(SDValue(N, i),
          CurDAG->getTargetExtractSubreg(SubRegIdx + i, dl, VT, SuperReg));

  // Update the chain
  ReplaceUses(SDValue(N, NumVecs + 1), SDValue(Ld, 2));
  return nullptr;
}

SDNode *ARM64DAGToDAGISel::SelectStore(SDNode *N, unsigned NumVecs,
                                       unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getOperand(2)->getValueType(0);

  // Form a REG_SEQUENCE to force register allocation.
  bool Is128Bit = VT.getSizeInBits() == 128;
  SmallVector<SDValue, 4> Regs(N->op_begin() + 2, N->op_begin() + 2 + NumVecs);
  SDValue RegSeq = Is128Bit ? createQTuple(Regs) : createDTuple(Regs);

  SmallVector<SDValue, 6> Ops;
  Ops.push_back(RegSeq);
  Ops.push_back(N->getOperand(NumVecs + 2));
  Ops.push_back(N->getOperand(0));
  SDNode *St = CurDAG->getMachineNode(Opc, dl, N->getValueType(0), Ops);

  return St;
}

SDNode *ARM64DAGToDAGISel::SelectPostStore(SDNode *N, unsigned NumVecs,
                                               unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getOperand(2)->getValueType(0);
  SmallVector<EVT, 2> ResTys;
  ResTys.push_back(MVT::i64);   // Type of the write back register
  ResTys.push_back(MVT::Other); // Type for the Chain

  // Form a REG_SEQUENCE to force register allocation.
  bool Is128Bit = VT.getSizeInBits() == 128;
  SmallVector<SDValue, 4> Regs(N->op_begin() + 1, N->op_begin() + 1 + NumVecs);
  SDValue RegSeq = Is128Bit ? createQTuple(Regs) : createDTuple(Regs);

  SmallVector<SDValue, 6> Ops;
  Ops.push_back(RegSeq);
  Ops.push_back(N->getOperand(NumVecs + 1)); // base register
  Ops.push_back(N->getOperand(NumVecs + 2)); // Incremental
  Ops.push_back(N->getOperand(0)); // Chain
  SDNode *St = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);

  return St;
}

/// WidenVector - Given a value in the V64 register class, produce the
/// equivalent value in the V128 register class.
class WidenVector {
  SelectionDAG &DAG;

public:
  WidenVector(SelectionDAG &DAG) : DAG(DAG) {}

  SDValue operator()(SDValue V64Reg) {
    EVT VT = V64Reg.getValueType();
    unsigned NarrowSize = VT.getVectorNumElements();
    MVT EltTy = VT.getVectorElementType().getSimpleVT();
    MVT WideTy = MVT::getVectorVT(EltTy, 2 * NarrowSize);
    SDLoc DL(V64Reg);

    SDValue Undef =
        SDValue(DAG.getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, WideTy), 0);
    return DAG.getTargetInsertSubreg(ARM64::dsub, DL, WideTy, Undef, V64Reg);
  }
};

/// NarrowVector - Given a value in the V128 register class, produce the
/// equivalent value in the V64 register class.
static SDValue NarrowVector(SDValue V128Reg, SelectionDAG &DAG) {
  EVT VT = V128Reg.getValueType();
  unsigned WideSize = VT.getVectorNumElements();
  MVT EltTy = VT.getVectorElementType().getSimpleVT();
  MVT NarrowTy = MVT::getVectorVT(EltTy, WideSize / 2);

  return DAG.getTargetExtractSubreg(ARM64::dsub, SDLoc(V128Reg), NarrowTy,
                                    V128Reg);
}

SDNode *ARM64DAGToDAGISel::SelectLoadLane(SDNode *N, unsigned NumVecs,
                                          unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getValueType(0);
  bool Narrow = VT.getSizeInBits() == 64;

  // Form a REG_SEQUENCE to force register allocation.
  SmallVector<SDValue, 4> Regs(N->op_begin() + 2, N->op_begin() + 2 + NumVecs);

  if (Narrow)
    std::transform(Regs.begin(), Regs.end(), Regs.begin(),
                   WidenVector(*CurDAG));

  SDValue RegSeq = createQTuple(Regs);

  std::vector<EVT> ResTys;
  ResTys.push_back(MVT::Untyped);
  ResTys.push_back(MVT::Other);

  unsigned LaneNo =
      cast<ConstantSDNode>(N->getOperand(NumVecs + 2))->getZExtValue();

  SmallVector<SDValue, 6> Ops;
  Ops.push_back(RegSeq);
  Ops.push_back(CurDAG->getTargetConstant(LaneNo, MVT::i64));
  Ops.push_back(N->getOperand(NumVecs + 3));
  Ops.push_back(N->getOperand(0));
  SDNode *Ld = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);
  SDValue SuperReg = SDValue(Ld, 0);

  EVT WideVT = RegSeq.getOperand(1)->getValueType(0);
  static unsigned QSubs[] = { ARM64::qsub0, ARM64::qsub1, ARM64::qsub2,
                              ARM64::qsub3 };
  for (unsigned i = 0; i < NumVecs; ++i) {
    SDValue NV = CurDAG->getTargetExtractSubreg(QSubs[i], dl, WideVT, SuperReg);
    if (Narrow)
      NV = NarrowVector(NV, *CurDAG);
    ReplaceUses(SDValue(N, i), NV);
  }

  ReplaceUses(SDValue(N, NumVecs), SDValue(Ld, 1));

  return Ld;
}

SDNode *ARM64DAGToDAGISel::SelectPostLoadLane(SDNode *N, unsigned NumVecs,
                                              unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getValueType(0);
  bool Narrow = VT.getSizeInBits() == 64;

  // Form a REG_SEQUENCE to force register allocation.
  SmallVector<SDValue, 4> Regs(N->op_begin() + 1, N->op_begin() + 1 + NumVecs);

  if (Narrow)
    std::transform(Regs.begin(), Regs.end(), Regs.begin(),
                   WidenVector(*CurDAG));

  SDValue RegSeq = createQTuple(Regs);

  std::vector<EVT> ResTys;
  ResTys.push_back(MVT::i64); // Type of the write back register
  ResTys.push_back(MVT::Untyped);
  ResTys.push_back(MVT::Other);

  unsigned LaneNo =
      cast<ConstantSDNode>(N->getOperand(NumVecs + 1))->getZExtValue();

  SmallVector<SDValue, 6> Ops;
  Ops.push_back(RegSeq);
  Ops.push_back(CurDAG->getTargetConstant(LaneNo, MVT::i64)); // Lane Number
  Ops.push_back(N->getOperand(NumVecs + 2)); // Base register
  Ops.push_back(N->getOperand(NumVecs + 3)); // Incremental
  Ops.push_back(N->getOperand(0));
  SDNode *Ld = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);

  // Update uses of the write back register
  ReplaceUses(SDValue(N, NumVecs), SDValue(Ld, 0));

  // Update uses of the vector list
  SDValue SuperReg = SDValue(Ld, 1);
  if (NumVecs == 1) {
    ReplaceUses(SDValue(N, 0),
                Narrow ? NarrowVector(SuperReg, *CurDAG) : SuperReg);
  } else {
    EVT WideVT = RegSeq.getOperand(1)->getValueType(0);
    static unsigned QSubs[] = { ARM64::qsub0, ARM64::qsub1, ARM64::qsub2,
                                ARM64::qsub3 };
    for (unsigned i = 0; i < NumVecs; ++i) {
      SDValue NV = CurDAG->getTargetExtractSubreg(QSubs[i], dl, WideVT,
                                                  SuperReg);
      if (Narrow)
        NV = NarrowVector(NV, *CurDAG);
      ReplaceUses(SDValue(N, i), NV);
    }
  }

  // Update the Chain
  ReplaceUses(SDValue(N, NumVecs + 1), SDValue(Ld, 2));

  return Ld;
}

SDNode *ARM64DAGToDAGISel::SelectStoreLane(SDNode *N, unsigned NumVecs,
                                           unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getOperand(2)->getValueType(0);
  bool Narrow = VT.getSizeInBits() == 64;

  // Form a REG_SEQUENCE to force register allocation.
  SmallVector<SDValue, 4> Regs(N->op_begin() + 2, N->op_begin() + 2 + NumVecs);

  if (Narrow)
    std::transform(Regs.begin(), Regs.end(), Regs.begin(),
                   WidenVector(*CurDAG));

  SDValue RegSeq = createQTuple(Regs);

  unsigned LaneNo =
      cast<ConstantSDNode>(N->getOperand(NumVecs + 2))->getZExtValue();

  SmallVector<SDValue, 6> Ops;
  Ops.push_back(RegSeq);
  Ops.push_back(CurDAG->getTargetConstant(LaneNo, MVT::i64));
  Ops.push_back(N->getOperand(NumVecs + 3));
  Ops.push_back(N->getOperand(0));
  SDNode *St = CurDAG->getMachineNode(Opc, dl, MVT::Other, Ops);

  // Transfer memoperands.
  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemIntrinsicSDNode>(N)->getMemOperand();
  cast<MachineSDNode>(St)->setMemRefs(MemOp, MemOp + 1);

  return St;
}

SDNode *ARM64DAGToDAGISel::SelectPostStoreLane(SDNode *N, unsigned NumVecs,
                                               unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getOperand(2)->getValueType(0);
  bool Narrow = VT.getSizeInBits() == 64;

  // Form a REG_SEQUENCE to force register allocation.
  SmallVector<SDValue, 4> Regs(N->op_begin() + 1, N->op_begin() + 1 + NumVecs);

  if (Narrow)
    std::transform(Regs.begin(), Regs.end(), Regs.begin(),
                   WidenVector(*CurDAG));

  SDValue RegSeq = createQTuple(Regs);

  SmallVector<EVT, 2> ResTys;
  ResTys.push_back(MVT::i64);   // Type of the write back register
  ResTys.push_back(MVT::Other);

  unsigned LaneNo =
      cast<ConstantSDNode>(N->getOperand(NumVecs + 1))->getZExtValue();

  SmallVector<SDValue, 6> Ops;
  Ops.push_back(RegSeq);
  Ops.push_back(CurDAG->getTargetConstant(LaneNo, MVT::i64));
  Ops.push_back(N->getOperand(NumVecs + 2)); // Base Register
  Ops.push_back(N->getOperand(NumVecs + 3)); // Incremental
  Ops.push_back(N->getOperand(0));
  SDNode *St = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);

  // Transfer memoperands.
  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemIntrinsicSDNode>(N)->getMemOperand();
  cast<MachineSDNode>(St)->setMemRefs(MemOp, MemOp + 1);

  return St;
}

static bool isBitfieldExtractOpFromAnd(SelectionDAG *CurDAG, SDNode *N,
                                       unsigned &Opc, SDValue &Opd0,
                                       unsigned &LSB, unsigned &MSB,
                                       unsigned NumberOfIgnoredLowBits,
                                       bool BiggerPattern) {
  assert(N->getOpcode() == ISD::AND &&
         "N must be a AND operation to call this function");

  EVT VT = N->getValueType(0);

  // Here we can test the type of VT and return false when the type does not
  // match, but since it is done prior to that call in the current context
  // we turned that into an assert to avoid redundant code.
  assert((VT == MVT::i32 || VT == MVT::i64) &&
         "Type checking must have been done before calling this function");

  // FIXME: simplify-demanded-bits in DAGCombine will probably have
  // changed the AND node to a 32-bit mask operation. We'll have to
  // undo that as part of the transform here if we want to catch all
  // the opportunities.
  // Currently the NumberOfIgnoredLowBits argument helps to recover
  // form these situations when matching bigger pattern (bitfield insert).

  // For unsigned extracts, check for a shift right and mask
  uint64_t And_imm = 0;
  if (!isOpcWithIntImmediate(N, ISD::AND, And_imm))
    return false;

  const SDNode *Op0 = N->getOperand(0).getNode();

  // Because of simplify-demanded-bits in DAGCombine, the mask may have been
  // simplified. Try to undo that
  And_imm |= (1 << NumberOfIgnoredLowBits) - 1;

  // The immediate is a mask of the low bits iff imm & (imm+1) == 0
  if (And_imm & (And_imm + 1))
    return false;

  bool ClampMSB = false;
  uint64_t Srl_imm = 0;
  // Handle the SRL + ANY_EXTEND case.
  if (VT == MVT::i64 && Op0->getOpcode() == ISD::ANY_EXTEND &&
      isOpcWithIntImmediate(Op0->getOperand(0).getNode(), ISD::SRL, Srl_imm)) {
    // Extend the incoming operand of the SRL to 64-bit.
    Opd0 = Widen(CurDAG, Op0->getOperand(0).getOperand(0));
    // Make sure to clamp the MSB so that we preserve the semantics of the
    // original operations.
    ClampMSB = true;
  } else if (VT == MVT::i32 && Op0->getOpcode() == ISD::TRUNCATE &&
             isOpcWithIntImmediate(Op0->getOperand(0).getNode(), ISD::SRL,
                                   Srl_imm)) {
    // If the shift result was truncated, we can still combine them.
    Opd0 = Op0->getOperand(0).getOperand(0);

    // Use the type of SRL node.
    VT = Opd0->getValueType(0);
  } else if (isOpcWithIntImmediate(Op0, ISD::SRL, Srl_imm)) {
    Opd0 = Op0->getOperand(0);
  } else if (BiggerPattern) {
    // Let's pretend a 0 shift right has been performed.
    // The resulting code will be at least as good as the original one
    // plus it may expose more opportunities for bitfield insert pattern.
    // FIXME: Currently we limit this to the bigger pattern, because
    // some optimizations expect AND and not UBFM
    Opd0 = N->getOperand(0);
  } else
    return false;

  assert((BiggerPattern || (Srl_imm > 0 && Srl_imm < VT.getSizeInBits())) &&
         "bad amount in shift node!");

  LSB = Srl_imm;
  MSB = Srl_imm + (VT == MVT::i32 ? CountTrailingOnes_32(And_imm)
                                  : CountTrailingOnes_64(And_imm)) -
        1;
  if (ClampMSB)
    // Since we're moving the extend before the right shift operation, we need
    // to clamp the MSB to make sure we don't shift in undefined bits instead of
    // the zeros which would get shifted in with the original right shift
    // operation.
    MSB = MSB > 31 ? 31 : MSB;

  Opc = VT == MVT::i32 ? ARM64::UBFMWri : ARM64::UBFMXri;
  return true;
}

static bool isOneBitExtractOpFromShr(SDNode *N, unsigned &Opc, SDValue &Opd0,
                                     unsigned &LSB, unsigned &MSB) {
  // We are looking for the following pattern which basically extracts a single
  // bit from the source value and places it in the LSB of the destination
  // value, all other bits of the destination value or set to zero:
  //
  // Value2 = AND Value, MaskImm
  // SRL Value2, ShiftImm
  //
  // with MaskImm >> ShiftImm == 1.
  //
  // This gets selected into a single UBFM:
  //
  // UBFM Value, ShiftImm, ShiftImm
  //

  if (N->getOpcode() != ISD::SRL)
    return false;

  uint64_t And_mask = 0;
  if (!isOpcWithIntImmediate(N->getOperand(0).getNode(), ISD::AND, And_mask))
    return false;

  Opd0 = N->getOperand(0).getOperand(0);

  uint64_t Srl_imm = 0;
  if (!isIntImmediate(N->getOperand(1), Srl_imm))
    return false;

  // Check whether we really have a one bit extract here.
  if (And_mask >> Srl_imm == 0x1) {
    if (N->getValueType(0) == MVT::i32)
      Opc = ARM64::UBFMWri;
    else
      Opc = ARM64::UBFMXri;

    LSB = MSB = Srl_imm;

    return true;
  }

  return false;
}

static bool isBitfieldExtractOpFromShr(SDNode *N, unsigned &Opc, SDValue &Opd0,
                                       unsigned &LSB, unsigned &MSB,
                                       bool BiggerPattern) {
  assert((N->getOpcode() == ISD::SRA || N->getOpcode() == ISD::SRL) &&
         "N must be a SHR/SRA operation to call this function");

  EVT VT = N->getValueType(0);

  // Here we can test the type of VT and return false when the type does not
  // match, but since it is done prior to that call in the current context
  // we turned that into an assert to avoid redundant code.
  assert((VT == MVT::i32 || VT == MVT::i64) &&
         "Type checking must have been done before calling this function");

  // Check for AND + SRL doing a one bit extract.
  if (isOneBitExtractOpFromShr(N, Opc, Opd0, LSB, MSB))
    return true;

  // we're looking for a shift of a shift
  uint64_t Shl_imm = 0;
  uint64_t Trunc_bits = 0;
  if (isOpcWithIntImmediate(N->getOperand(0).getNode(), ISD::SHL, Shl_imm)) {
    Opd0 = N->getOperand(0).getOperand(0);
  } else if (VT == MVT::i32 && N->getOpcode() == ISD::SRL &&
             N->getOperand(0).getNode()->getOpcode() == ISD::TRUNCATE) {
    // We are looking for a shift of truncate. Truncate from i64 to i32 could
    // be considered as setting high 32 bits as zero. Our strategy here is to
    // always generate 64bit UBFM. This consistency will help the CSE pass
    // later find more redundancy.
    Opd0 = N->getOperand(0).getOperand(0);
    Trunc_bits = Opd0->getValueType(0).getSizeInBits() - VT.getSizeInBits();
    VT = Opd0->getValueType(0);
    assert(VT == MVT::i64 && "the promoted type should be i64");
  } else if (BiggerPattern) {
    // Let's pretend a 0 shift left has been performed.
    // FIXME: Currently we limit this to the bigger pattern case,
    // because some optimizations expect AND and not UBFM
    Opd0 = N->getOperand(0);
  } else
    return false;

  assert(Shl_imm < VT.getSizeInBits() && "bad amount in shift node!");
  uint64_t Srl_imm = 0;
  if (!isIntImmediate(N->getOperand(1), Srl_imm))
    return false;

  assert(Srl_imm > 0 && Srl_imm < VT.getSizeInBits() &&
         "bad amount in shift node!");
  // Note: The width operand is encoded as width-1.
  unsigned Width = VT.getSizeInBits() - Trunc_bits - Srl_imm - 1;
  int sLSB = Srl_imm - Shl_imm;
  if (sLSB < 0)
    return false;
  LSB = sLSB;
  MSB = LSB + Width;
  // SRA requires a signed extraction
  if (VT == MVT::i32)
    Opc = N->getOpcode() == ISD::SRA ? ARM64::SBFMWri : ARM64::UBFMWri;
  else
    Opc = N->getOpcode() == ISD::SRA ? ARM64::SBFMXri : ARM64::UBFMXri;
  return true;
}

static bool isBitfieldExtractOp(SelectionDAG *CurDAG, SDNode *N, unsigned &Opc,
                                SDValue &Opd0, unsigned &LSB, unsigned &MSB,
                                unsigned NumberOfIgnoredLowBits = 0,
                                bool BiggerPattern = false) {
  if (N->getValueType(0) != MVT::i32 && N->getValueType(0) != MVT::i64)
    return false;

  switch (N->getOpcode()) {
  default:
    if (!N->isMachineOpcode())
      return false;
    break;
  case ISD::AND:
    return isBitfieldExtractOpFromAnd(CurDAG, N, Opc, Opd0, LSB, MSB,
                                      NumberOfIgnoredLowBits, BiggerPattern);
  case ISD::SRL:
  case ISD::SRA:
    return isBitfieldExtractOpFromShr(N, Opc, Opd0, LSB, MSB, BiggerPattern);
  }

  unsigned NOpc = N->getMachineOpcode();
  switch (NOpc) {
  default:
    return false;
  case ARM64::SBFMWri:
  case ARM64::UBFMWri:
  case ARM64::SBFMXri:
  case ARM64::UBFMXri:
    Opc = NOpc;
    Opd0 = N->getOperand(0);
    LSB = cast<ConstantSDNode>(N->getOperand(1).getNode())->getZExtValue();
    MSB = cast<ConstantSDNode>(N->getOperand(2).getNode())->getZExtValue();
    return true;
  }
  // Unreachable
  return false;
}

SDNode *ARM64DAGToDAGISel::SelectBitfieldExtractOp(SDNode *N) {
  unsigned Opc, LSB, MSB;
  SDValue Opd0;
  if (!isBitfieldExtractOp(CurDAG, N, Opc, Opd0, LSB, MSB))
    return nullptr;

  EVT VT = N->getValueType(0);

  // If the bit extract operation is 64bit but the original type is 32bit, we
  // need to add one EXTRACT_SUBREG.
  if ((Opc == ARM64::SBFMXri || Opc == ARM64::UBFMXri) && VT == MVT::i32) {
    SDValue Ops64[] = {Opd0, CurDAG->getTargetConstant(LSB, MVT::i64),
                       CurDAG->getTargetConstant(MSB, MVT::i64)};

    SDNode *BFM = CurDAG->getMachineNode(Opc, SDLoc(N), MVT::i64, Ops64);
    SDValue SubReg = CurDAG->getTargetConstant(ARM64::sub_32, MVT::i32);
    MachineSDNode *Node =
        CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG, SDLoc(N), MVT::i32,
                               SDValue(BFM, 0), SubReg);
    return Node;
  }

  SDValue Ops[] = {Opd0, CurDAG->getTargetConstant(LSB, VT),
                   CurDAG->getTargetConstant(MSB, VT)};
  return CurDAG->SelectNodeTo(N, Opc, VT, Ops);
}

/// Does DstMask form a complementary pair with the mask provided by
/// BitsToBeInserted, suitable for use in a BFI instruction. Roughly speaking,
/// this asks whether DstMask zeroes precisely those bits that will be set by
/// the other half.
static bool isBitfieldDstMask(uint64_t DstMask, APInt BitsToBeInserted,
                              unsigned NumberOfIgnoredHighBits, EVT VT) {
  assert((VT == MVT::i32 || VT == MVT::i64) &&
         "i32 or i64 mask type expected!");
  unsigned BitWidth = VT.getSizeInBits() - NumberOfIgnoredHighBits;

  APInt SignificantDstMask = APInt(BitWidth, DstMask);
  APInt SignificantBitsToBeInserted = BitsToBeInserted.zextOrTrunc(BitWidth);

  return (SignificantDstMask & SignificantBitsToBeInserted) == 0 &&
         (SignificantDstMask | SignificantBitsToBeInserted).isAllOnesValue();
}

// Look for bits that will be useful for later uses.
// A bit is consider useless as soon as it is dropped and never used
// before it as been dropped.
// E.g., looking for useful bit of x
// 1. y = x & 0x7
// 2. z = y >> 2
// After #1, x useful bits are 0x7, then the useful bits of x, live through
// y.
// After #2, the useful bits of x are 0x4.
// However, if x is used on an unpredicatable instruction, then all its bits
// are useful.
// E.g.
// 1. y = x & 0x7
// 2. z = y >> 2
// 3. str x, [@x]
static void getUsefulBits(SDValue Op, APInt &UsefulBits, unsigned Depth = 0);

static void getUsefulBitsFromAndWithImmediate(SDValue Op, APInt &UsefulBits,
                                              unsigned Depth) {
  uint64_t Imm =
      cast<const ConstantSDNode>(Op.getOperand(1).getNode())->getZExtValue();
  Imm = ARM64_AM::decodeLogicalImmediate(Imm, UsefulBits.getBitWidth());
  UsefulBits &= APInt(UsefulBits.getBitWidth(), Imm);
  getUsefulBits(Op, UsefulBits, Depth + 1);
}

static void getUsefulBitsFromBitfieldMoveOpd(SDValue Op, APInt &UsefulBits,
                                             uint64_t Imm, uint64_t MSB,
                                             unsigned Depth) {
  // inherit the bitwidth value
  APInt OpUsefulBits(UsefulBits);
  OpUsefulBits = 1;

  if (MSB >= Imm) {
    OpUsefulBits = OpUsefulBits.shl(MSB - Imm + 1);
    --OpUsefulBits;
    // The interesting part will be in the lower part of the result
    getUsefulBits(Op, OpUsefulBits, Depth + 1);
    // The interesting part was starting at Imm in the argument
    OpUsefulBits = OpUsefulBits.shl(Imm);
  } else {
    OpUsefulBits = OpUsefulBits.shl(MSB + 1);
    --OpUsefulBits;
    // The interesting part will be shifted in the result
    OpUsefulBits = OpUsefulBits.shl(OpUsefulBits.getBitWidth() - Imm);
    getUsefulBits(Op, OpUsefulBits, Depth + 1);
    // The interesting part was at zero in the argument
    OpUsefulBits = OpUsefulBits.lshr(OpUsefulBits.getBitWidth() - Imm);
  }

  UsefulBits &= OpUsefulBits;
}

static void getUsefulBitsFromUBFM(SDValue Op, APInt &UsefulBits,
                                  unsigned Depth) {
  uint64_t Imm =
      cast<const ConstantSDNode>(Op.getOperand(1).getNode())->getZExtValue();
  uint64_t MSB =
      cast<const ConstantSDNode>(Op.getOperand(2).getNode())->getZExtValue();

  getUsefulBitsFromBitfieldMoveOpd(Op, UsefulBits, Imm, MSB, Depth);
}

static void getUsefulBitsFromOrWithShiftedReg(SDValue Op, APInt &UsefulBits,
                                              unsigned Depth) {
  uint64_t ShiftTypeAndValue =
      cast<const ConstantSDNode>(Op.getOperand(2).getNode())->getZExtValue();
  APInt Mask(UsefulBits);
  Mask.clearAllBits();
  Mask.flipAllBits();

  if (ARM64_AM::getShiftType(ShiftTypeAndValue) == ARM64_AM::LSL) {
    // Shift Left
    uint64_t ShiftAmt = ARM64_AM::getShiftValue(ShiftTypeAndValue);
    Mask = Mask.shl(ShiftAmt);
    getUsefulBits(Op, Mask, Depth + 1);
    Mask = Mask.lshr(ShiftAmt);
  } else if (ARM64_AM::getShiftType(ShiftTypeAndValue) == ARM64_AM::LSR) {
    // Shift Right
    // We do not handle ARM64_AM::ASR, because the sign will change the
    // number of useful bits
    uint64_t ShiftAmt = ARM64_AM::getShiftValue(ShiftTypeAndValue);
    Mask = Mask.lshr(ShiftAmt);
    getUsefulBits(Op, Mask, Depth + 1);
    Mask = Mask.shl(ShiftAmt);
  } else
    return;

  UsefulBits &= Mask;
}

static void getUsefulBitsFromBFM(SDValue Op, SDValue Orig, APInt &UsefulBits,
                                 unsigned Depth) {
  uint64_t Imm =
      cast<const ConstantSDNode>(Op.getOperand(2).getNode())->getZExtValue();
  uint64_t MSB =
      cast<const ConstantSDNode>(Op.getOperand(3).getNode())->getZExtValue();

  if (Op.getOperand(1) == Orig)
    return getUsefulBitsFromBitfieldMoveOpd(Op, UsefulBits, Imm, MSB, Depth);

  APInt OpUsefulBits(UsefulBits);
  OpUsefulBits = 1;

  if (MSB >= Imm) {
    OpUsefulBits = OpUsefulBits.shl(MSB - Imm + 1);
    --OpUsefulBits;
    UsefulBits &= ~OpUsefulBits;
    getUsefulBits(Op, UsefulBits, Depth + 1);
  } else {
    OpUsefulBits = OpUsefulBits.shl(MSB + 1);
    --OpUsefulBits;
    UsefulBits = ~(OpUsefulBits.shl(OpUsefulBits.getBitWidth() - Imm));
    getUsefulBits(Op, UsefulBits, Depth + 1);
  }
}

static void getUsefulBitsForUse(SDNode *UserNode, APInt &UsefulBits,
                                SDValue Orig, unsigned Depth) {

  // Users of this node should have already been instruction selected
  // FIXME: Can we turn that into an assert?
  if (!UserNode->isMachineOpcode())
    return;

  switch (UserNode->getMachineOpcode()) {
  default:
    return;
  case ARM64::ANDSWri:
  case ARM64::ANDSXri:
  case ARM64::ANDWri:
  case ARM64::ANDXri:
    // We increment Depth only when we call the getUsefulBits
    return getUsefulBitsFromAndWithImmediate(SDValue(UserNode, 0), UsefulBits,
                                             Depth);
  case ARM64::UBFMWri:
  case ARM64::UBFMXri:
    return getUsefulBitsFromUBFM(SDValue(UserNode, 0), UsefulBits, Depth);

  case ARM64::ORRWrs:
  case ARM64::ORRXrs:
    if (UserNode->getOperand(1) != Orig)
      return;
    return getUsefulBitsFromOrWithShiftedReg(SDValue(UserNode, 0), UsefulBits,
                                             Depth);
  case ARM64::BFMWri:
  case ARM64::BFMXri:
    return getUsefulBitsFromBFM(SDValue(UserNode, 0), Orig, UsefulBits, Depth);
  }
}

static void getUsefulBits(SDValue Op, APInt &UsefulBits, unsigned Depth) {
  if (Depth >= 6)
    return;
  // Initialize UsefulBits
  if (!Depth) {
    unsigned Bitwidth = Op.getValueType().getScalarType().getSizeInBits();
    // At the beginning, assume every produced bits is useful
    UsefulBits = APInt(Bitwidth, 0);
    UsefulBits.flipAllBits();
  }
  APInt UsersUsefulBits(UsefulBits.getBitWidth(), 0);

  for (SDNode *Node : Op.getNode()->uses()) {
    // A use cannot produce useful bits
    APInt UsefulBitsForUse = APInt(UsefulBits);
    getUsefulBitsForUse(Node, UsefulBitsForUse, Op, Depth);
    UsersUsefulBits |= UsefulBitsForUse;
  }
  // UsefulBits contains the produced bits that are meaningful for the
  // current definition, thus a user cannot make a bit meaningful at
  // this point
  UsefulBits &= UsersUsefulBits;
}

/// Create a machine node performing a notional SHL of Op by ShlAmount. If
/// ShlAmount is negative, do a (logical) right-shift instead. If ShlAmount is
/// 0, return Op unchanged.
static SDValue getLeftShift(SelectionDAG *CurDAG, SDValue Op, int ShlAmount) {
  if (ShlAmount == 0)
    return Op;

  EVT VT = Op.getValueType();
  unsigned BitWidth = VT.getSizeInBits();
  unsigned UBFMOpc = BitWidth == 32 ? ARM64::UBFMWri : ARM64::UBFMXri;

  SDNode *ShiftNode;
  if (ShlAmount > 0) {
    // LSL wD, wN, #Amt == UBFM wD, wN, #32-Amt, #31-Amt
    ShiftNode = CurDAG->getMachineNode(
        UBFMOpc, SDLoc(Op), VT, Op,
        CurDAG->getTargetConstant(BitWidth - ShlAmount, VT),
        CurDAG->getTargetConstant(BitWidth - 1 - ShlAmount, VT));
  } else {
    // LSR wD, wN, #Amt == UBFM wD, wN, #Amt, #32-1
    assert(ShlAmount < 0 && "expected right shift");
    int ShrAmount = -ShlAmount;
    ShiftNode = CurDAG->getMachineNode(
        UBFMOpc, SDLoc(Op), VT, Op, CurDAG->getTargetConstant(ShrAmount, VT),
        CurDAG->getTargetConstant(BitWidth - 1, VT));
  }

  return SDValue(ShiftNode, 0);
}

/// Does this tree qualify as an attempt to move a bitfield into position,
/// essentially "(and (shl VAL, N), Mask)".
static bool isBitfieldPositioningOp(SelectionDAG *CurDAG, SDValue Op,
                                    SDValue &Src, int &ShiftAmount,
                                    int &MaskWidth) {
  EVT VT = Op.getValueType();
  unsigned BitWidth = VT.getSizeInBits();
  (void)BitWidth;
  assert(BitWidth == 32 || BitWidth == 64);

  APInt KnownZero, KnownOne;
  CurDAG->computeKnownBits(Op, KnownZero, KnownOne);

  // Non-zero in the sense that they're not provably zero, which is the key
  // point if we want to use this value
  uint64_t NonZeroBits = (~KnownZero).getZExtValue();

  // Discard a constant AND mask if present. It's safe because the node will
  // already have been factored into the computeKnownBits calculation above.
  uint64_t AndImm;
  if (isOpcWithIntImmediate(Op.getNode(), ISD::AND, AndImm)) {
    assert((~APInt(BitWidth, AndImm) & ~KnownZero) == 0);
    Op = Op.getOperand(0);
  }

  uint64_t ShlImm;
  if (!isOpcWithIntImmediate(Op.getNode(), ISD::SHL, ShlImm))
    return false;
  Op = Op.getOperand(0);

  if (!isShiftedMask_64(NonZeroBits))
    return false;

  ShiftAmount = countTrailingZeros(NonZeroBits);
  MaskWidth = CountTrailingOnes_64(NonZeroBits >> ShiftAmount);

  // BFI encompasses sufficiently many nodes that it's worth inserting an extra
  // LSL/LSR if the mask in NonZeroBits doesn't quite match up with the ISD::SHL
  // amount.
  Src = getLeftShift(CurDAG, Op, ShlImm - ShiftAmount);

  return true;
}

// Given a OR operation, check if we have the following pattern
// ubfm c, b, imm, imm2 (or something that does the same jobs, see
//                       isBitfieldExtractOp)
// d = e & mask2 ; where mask is a binary sequence of 1..10..0 and
//                 countTrailingZeros(mask2) == imm2 - imm + 1
// f = d | c
// if yes, given reference arguments will be update so that one can replace
// the OR instruction with:
// f = Opc Opd0, Opd1, LSB, MSB ; where Opc is a BFM, LSB = imm, and MSB = imm2
static bool isBitfieldInsertOpFromOr(SDNode *N, unsigned &Opc, SDValue &Dst,
                                     SDValue &Src, unsigned &ImmR,
                                     unsigned &ImmS, SelectionDAG *CurDAG) {
  assert(N->getOpcode() == ISD::OR && "Expect a OR operation");

  // Set Opc
  EVT VT = N->getValueType(0);
  if (VT == MVT::i32)
    Opc = ARM64::BFMWri;
  else if (VT == MVT::i64)
    Opc = ARM64::BFMXri;
  else
    return false;

  // Because of simplify-demanded-bits in DAGCombine, involved masks may not
  // have the expected shape. Try to undo that.
  APInt UsefulBits;
  getUsefulBits(SDValue(N, 0), UsefulBits);

  unsigned NumberOfIgnoredLowBits = UsefulBits.countTrailingZeros();
  unsigned NumberOfIgnoredHighBits = UsefulBits.countLeadingZeros();

  // OR is commutative, check both possibilities (does llvm provide a
  // way to do that directely, e.g., via code matcher?)
  SDValue OrOpd1Val = N->getOperand(1);
  SDNode *OrOpd0 = N->getOperand(0).getNode();
  SDNode *OrOpd1 = N->getOperand(1).getNode();
  for (int i = 0; i < 2;
       ++i, std::swap(OrOpd0, OrOpd1), OrOpd1Val = N->getOperand(0)) {
    unsigned BFXOpc;
    int DstLSB, Width;
    if (isBitfieldExtractOp(CurDAG, OrOpd0, BFXOpc, Src, ImmR, ImmS,
                            NumberOfIgnoredLowBits, true)) {
      // Check that the returned opcode is compatible with the pattern,
      // i.e., same type and zero extended (U and not S)
      if ((BFXOpc != ARM64::UBFMXri && VT == MVT::i64) ||
          (BFXOpc != ARM64::UBFMWri && VT == MVT::i32))
        continue;

      // Compute the width of the bitfield insertion
      DstLSB = 0;
      Width = ImmS - ImmR + 1;
      // FIXME: This constraint is to catch bitfield insertion we may
      // want to widen the pattern if we want to grab general bitfied
      // move case
      if (Width <= 0)
        continue;

      // If the mask on the insertee is correct, we have a BFXIL operation. We
      // can share the ImmR and ImmS values from the already-computed UBFM.
    } else if (isBitfieldPositioningOp(CurDAG, SDValue(OrOpd0, 0), Src,
                                       DstLSB, Width)) {
      ImmR = (VT.getSizeInBits() - DstLSB) % VT.getSizeInBits();
      ImmS = Width - 1;
    } else
      continue;

    // Check the second part of the pattern
    EVT VT = OrOpd1->getValueType(0);
    assert((VT == MVT::i32 || VT == MVT::i64) && "unexpected OR operand");

    // Compute the Known Zero for the candidate of the first operand.
    // This allows to catch more general case than just looking for
    // AND with imm. Indeed, simplify-demanded-bits may have removed
    // the AND instruction because it proves it was useless.
    APInt KnownZero, KnownOne;
    CurDAG->computeKnownBits(OrOpd1Val, KnownZero, KnownOne);

    // Check if there is enough room for the second operand to appear
    // in the first one
    APInt BitsToBeInserted =
        APInt::getBitsSet(KnownZero.getBitWidth(), DstLSB, DstLSB + Width);

    if ((BitsToBeInserted & ~KnownZero) != 0)
      continue;

    // Set the first operand
    uint64_t Imm;
    if (isOpcWithIntImmediate(OrOpd1, ISD::AND, Imm) &&
        isBitfieldDstMask(Imm, BitsToBeInserted, NumberOfIgnoredHighBits, VT))
      // In that case, we can eliminate the AND
      Dst = OrOpd1->getOperand(0);
    else
      // Maybe the AND has been removed by simplify-demanded-bits
      // or is useful because it discards more bits
      Dst = OrOpd1Val;

    // both parts match
    return true;
  }

  return false;
}

SDNode *ARM64DAGToDAGISel::SelectBitfieldInsertOp(SDNode *N) {
  if (N->getOpcode() != ISD::OR)
    return nullptr;

  unsigned Opc;
  unsigned LSB, MSB;
  SDValue Opd0, Opd1;

  if (!isBitfieldInsertOpFromOr(N, Opc, Opd0, Opd1, LSB, MSB, CurDAG))
    return nullptr;

  EVT VT = N->getValueType(0);
  SDValue Ops[] = { Opd0,
                    Opd1,
                    CurDAG->getTargetConstant(LSB, VT),
                    CurDAG->getTargetConstant(MSB, VT) };
  return CurDAG->SelectNodeTo(N, Opc, VT, Ops);
}

SDNode *ARM64DAGToDAGISel::SelectLIBM(SDNode *N) {
  EVT VT = N->getValueType(0);
  unsigned Variant;
  unsigned Opc;
  unsigned FRINTXOpcs[] = { ARM64::FRINTXSr, ARM64::FRINTXDr };

  if (VT == MVT::f32) {
    Variant = 0;
  } else if (VT == MVT::f64) {
    Variant = 1;
  } else
    return nullptr; // Unrecognized argument type. Fall back on default codegen.

  // Pick the FRINTX variant needed to set the flags.
  unsigned FRINTXOpc = FRINTXOpcs[Variant];

  switch (N->getOpcode()) {
  default:
    return nullptr; // Unrecognized libm ISD node. Fall back on default codegen.
  case ISD::FCEIL: {
    unsigned FRINTPOpcs[] = { ARM64::FRINTPSr, ARM64::FRINTPDr };
    Opc = FRINTPOpcs[Variant];
    break;
  }
  case ISD::FFLOOR: {
    unsigned FRINTMOpcs[] = { ARM64::FRINTMSr, ARM64::FRINTMDr };
    Opc = FRINTMOpcs[Variant];
    break;
  }
  case ISD::FTRUNC: {
    unsigned FRINTZOpcs[] = { ARM64::FRINTZSr, ARM64::FRINTZDr };
    Opc = FRINTZOpcs[Variant];
    break;
  }
  case ISD::FROUND: {
    unsigned FRINTAOpcs[] = { ARM64::FRINTASr, ARM64::FRINTADr };
    Opc = FRINTAOpcs[Variant];
    break;
  }
  }

  SDLoc dl(N);
  SDValue In = N->getOperand(0);
  SmallVector<SDValue, 2> Ops;
  Ops.push_back(In);

  if (!TM.Options.UnsafeFPMath) {
    SDNode *FRINTX = CurDAG->getMachineNode(FRINTXOpc, dl, VT, MVT::Glue, In);
    Ops.push_back(SDValue(FRINTX, 1));
  }

  return CurDAG->getMachineNode(Opc, dl, VT, Ops);
}

bool
ARM64DAGToDAGISel::SelectCVTFixedPosOperand(SDValue N, SDValue &FixedPos,
                                              unsigned RegWidth) {
  APFloat FVal(0.0);
  if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(N))
    FVal = CN->getValueAPF();
  else if (LoadSDNode *LN = dyn_cast<LoadSDNode>(N)) {
    // Some otherwise illegal constants are allowed in this case.
    if (LN->getOperand(1).getOpcode() != ARM64ISD::ADDlow ||
        !isa<ConstantPoolSDNode>(LN->getOperand(1)->getOperand(1)))
      return false;

    ConstantPoolSDNode *CN =
        dyn_cast<ConstantPoolSDNode>(LN->getOperand(1)->getOperand(1));
    FVal = cast<ConstantFP>(CN->getConstVal())->getValueAPF();
  } else
    return false;

  // An FCVT[SU] instruction performs: convertToInt(Val * 2^fbits) where fbits
  // is between 1 and 32 for a destination w-register, or 1 and 64 for an
  // x-register.
  //
  // By this stage, we've detected (fp_to_[su]int (fmul Val, THIS_NODE)) so we
  // want THIS_NODE to be 2^fbits. This is much easier to deal with using
  // integers.
  bool IsExact;

  // fbits is between 1 and 64 in the worst-case, which means the fmul
  // could have 2^64 as an actual operand. Need 65 bits of precision.
  APSInt IntVal(65, true);
  FVal.convertToInteger(IntVal, APFloat::rmTowardZero, &IsExact);

  // N.b. isPowerOf2 also checks for > 0.
  if (!IsExact || !IntVal.isPowerOf2()) return false;
  unsigned FBits = IntVal.logBase2();

  // Checks above should have guaranteed that we haven't lost information in
  // finding FBits, but it must still be in range.
  if (FBits == 0 || FBits > RegWidth) return false;

  FixedPos = CurDAG->getTargetConstant(FBits, MVT::i32);
  return true;
}

SDNode *ARM64DAGToDAGISel::Select(SDNode *Node) {
  // Dump information about the Node being selected
  DEBUG(errs() << "Selecting: ");
  DEBUG(Node->dump(CurDAG));
  DEBUG(errs() << "\n");

  // If we have a custom node, we already have selected!
  if (Node->isMachineOpcode()) {
    DEBUG(errs() << "== "; Node->dump(CurDAG); errs() << "\n");
    Node->setNodeId(-1);
    return nullptr;
  }

  // Few custom selection stuff.
  SDNode *ResNode = nullptr;
  EVT VT = Node->getValueType(0);

  switch (Node->getOpcode()) {
  default:
    break;

  case ISD::ADD:
    if (SDNode *I = SelectMLAV64LaneV128(Node))
      return I;
    break;

  case ISD::LOAD: {
    // Try to select as an indexed load. Fall through to normal processing
    // if we can't.
    bool Done = false;
    SDNode *I = SelectIndexedLoad(Node, Done);
    if (Done)
      return I;
    break;
  }

  case ISD::SRL:
  case ISD::AND:
  case ISD::SRA:
    if (SDNode *I = SelectBitfieldExtractOp(Node))
      return I;
    break;

  case ISD::OR:
    if (SDNode *I = SelectBitfieldInsertOp(Node))
      return I;
    break;

  case ISD::EXTRACT_VECTOR_ELT: {
    // Extracting lane zero is a special case where we can just use a plain
    // EXTRACT_SUBREG instruction, which will become FMOV. This is easier for
    // the rest of the compiler, especially the register allocator and copyi
    // propagation, to reason about, so is preferred when it's possible to
    // use it.
    ConstantSDNode *LaneNode = cast<ConstantSDNode>(Node->getOperand(1));
    // Bail and use the default Select() for non-zero lanes.
    if (LaneNode->getZExtValue() != 0)
      break;
    // If the element type is not the same as the result type, likewise
    // bail and use the default Select(), as there's more to do than just
    // a cross-class COPY. This catches extracts of i8 and i16 elements
    // since they will need an explicit zext.
    if (VT != Node->getOperand(0).getValueType().getVectorElementType())
      break;
    unsigned SubReg;
    switch (Node->getOperand(0)
                .getValueType()
                .getVectorElementType()
                .getSizeInBits()) {
    default:
      assert(0 && "Unexpected vector element type!");
    case 64:
      SubReg = ARM64::dsub;
      break;
    case 32:
      SubReg = ARM64::ssub;
      break;
    case 16: // FALLTHROUGH
    case 8:
      llvm_unreachable("unexpected zext-requiring extract element!");
    }
    SDValue Extract = CurDAG->getTargetExtractSubreg(SubReg, SDLoc(Node), VT,
                                                     Node->getOperand(0));
    DEBUG(dbgs() << "ISEL: Custom selection!\n=> ");
    DEBUG(Extract->dumpr(CurDAG));
    DEBUG(dbgs() << "\n");
    return Extract.getNode();
  }
  case ISD::Constant: {
    // Materialize zero constants as copies from WZR/XZR.  This allows
    // the coalescer to propagate these into other instructions.
    ConstantSDNode *ConstNode = cast<ConstantSDNode>(Node);
    if (ConstNode->isNullValue()) {
      if (VT == MVT::i32)
        return CurDAG->getCopyFromReg(CurDAG->getEntryNode(), SDLoc(Node),
                                      ARM64::WZR, MVT::i32).getNode();
      else if (VT == MVT::i64)
        return CurDAG->getCopyFromReg(CurDAG->getEntryNode(), SDLoc(Node),
                                      ARM64::XZR, MVT::i64).getNode();
    }
    break;
  }

  case ISD::FrameIndex: {
    // Selects to ADDXri FI, 0 which in turn will become ADDXri SP, imm.
    int FI = cast<FrameIndexSDNode>(Node)->getIndex();
    unsigned Shifter = ARM64_AM::getShifterImm(ARM64_AM::LSL, 0);
    const TargetLowering *TLI = getTargetLowering();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, TLI->getPointerTy());
    SDValue Ops[] = { TFI, CurDAG->getTargetConstant(0, MVT::i32),
                      CurDAG->getTargetConstant(Shifter, MVT::i32) };
    return CurDAG->SelectNodeTo(Node, ARM64::ADDXri, MVT::i64, Ops);
  }
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    switch (IntNo) {
    default:
      break;
    case Intrinsic::arm64_ldaxp:
    case Intrinsic::arm64_ldxp: {
      unsigned Op =
          IntNo == Intrinsic::arm64_ldaxp ? ARM64::LDAXPX : ARM64::LDXPX;
      SDValue MemAddr = Node->getOperand(2);
      SDLoc DL(Node);
      SDValue Chain = Node->getOperand(0);

      SDNode *Ld = CurDAG->getMachineNode(Op, DL, MVT::i64, MVT::i64,
                                          MVT::Other, MemAddr, Chain);

      // Transfer memoperands.
      MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
      MemOp[0] = cast<MemIntrinsicSDNode>(Node)->getMemOperand();
      cast<MachineSDNode>(Ld)->setMemRefs(MemOp, MemOp + 1);
      return Ld;
    }
    case Intrinsic::arm64_stlxp:
    case Intrinsic::arm64_stxp: {
      unsigned Op =
          IntNo == Intrinsic::arm64_stlxp ? ARM64::STLXPX : ARM64::STXPX;
      SDLoc DL(Node);
      SDValue Chain = Node->getOperand(0);
      SDValue ValLo = Node->getOperand(2);
      SDValue ValHi = Node->getOperand(3);
      SDValue MemAddr = Node->getOperand(4);

      // Place arguments in the right order.
      SmallVector<SDValue, 7> Ops;
      Ops.push_back(ValLo);
      Ops.push_back(ValHi);
      Ops.push_back(MemAddr);
      Ops.push_back(Chain);

      SDNode *St = CurDAG->getMachineNode(Op, DL, MVT::i32, MVT::Other, Ops);
      // Transfer memoperands.
      MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
      MemOp[0] = cast<MemIntrinsicSDNode>(Node)->getMemOperand();
      cast<MachineSDNode>(St)->setMemRefs(MemOp, MemOp + 1);

      return St;
    }
    case Intrinsic::arm64_neon_ld1x2:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 2, ARM64::LD1Twov8b, ARM64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 2, ARM64::LD1Twov16b, ARM64::qsub0);
      else if (VT == MVT::v4i16)
        return SelectLoad(Node, 2, ARM64::LD1Twov4h, ARM64::dsub0);
      else if (VT == MVT::v8i16)
        return SelectLoad(Node, 2, ARM64::LD1Twov8h, ARM64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 2, ARM64::LD1Twov2s, ARM64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 2, ARM64::LD1Twov4s, ARM64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 2, ARM64::LD1Twov1d, ARM64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 2, ARM64::LD1Twov2d, ARM64::qsub0);
      break;
    case Intrinsic::arm64_neon_ld1x3:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 3, ARM64::LD1Threev8b, ARM64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 3, ARM64::LD1Threev16b, ARM64::qsub0);
      else if (VT == MVT::v4i16)
        return SelectLoad(Node, 3, ARM64::LD1Threev4h, ARM64::dsub0);
      else if (VT == MVT::v8i16)
        return SelectLoad(Node, 3, ARM64::LD1Threev8h, ARM64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 3, ARM64::LD1Threev2s, ARM64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 3, ARM64::LD1Threev4s, ARM64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 3, ARM64::LD1Threev1d, ARM64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 3, ARM64::LD1Threev2d, ARM64::qsub0);
      break;
    case Intrinsic::arm64_neon_ld1x4:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 4, ARM64::LD1Fourv8b, ARM64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 4, ARM64::LD1Fourv16b, ARM64::qsub0);
      else if (VT == MVT::v4i16)
        return SelectLoad(Node, 4, ARM64::LD1Fourv4h, ARM64::dsub0);
      else if (VT == MVT::v8i16)
        return SelectLoad(Node, 4, ARM64::LD1Fourv8h, ARM64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 4, ARM64::LD1Fourv2s, ARM64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 4, ARM64::LD1Fourv4s, ARM64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 4, ARM64::LD1Fourv1d, ARM64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 4, ARM64::LD1Fourv2d, ARM64::qsub0);
      break;
    case Intrinsic::arm64_neon_ld2:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 2, ARM64::LD2Twov8b, ARM64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 2, ARM64::LD2Twov16b, ARM64::qsub0);
      else if (VT == MVT::v4i16)
        return SelectLoad(Node, 2, ARM64::LD2Twov4h, ARM64::dsub0);
      else if (VT == MVT::v8i16)
        return SelectLoad(Node, 2, ARM64::LD2Twov8h, ARM64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 2, ARM64::LD2Twov2s, ARM64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 2, ARM64::LD2Twov4s, ARM64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 2, ARM64::LD1Twov1d, ARM64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 2, ARM64::LD2Twov2d, ARM64::qsub0);
      break;
    case Intrinsic::arm64_neon_ld3:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 3, ARM64::LD3Threev8b, ARM64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 3, ARM64::LD3Threev16b, ARM64::qsub0);
      else if (VT == MVT::v4i16)
        return SelectLoad(Node, 3, ARM64::LD3Threev4h, ARM64::dsub0);
      else if (VT == MVT::v8i16)
        return SelectLoad(Node, 3, ARM64::LD3Threev8h, ARM64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 3, ARM64::LD3Threev2s, ARM64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 3, ARM64::LD3Threev4s, ARM64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 3, ARM64::LD1Threev1d, ARM64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 3, ARM64::LD3Threev2d, ARM64::qsub0);
      break;
    case Intrinsic::arm64_neon_ld4:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 4, ARM64::LD4Fourv8b, ARM64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 4, ARM64::LD4Fourv16b, ARM64::qsub0);
      else if (VT == MVT::v4i16)
        return SelectLoad(Node, 4, ARM64::LD4Fourv4h, ARM64::dsub0);
      else if (VT == MVT::v8i16)
        return SelectLoad(Node, 4, ARM64::LD4Fourv8h, ARM64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 4, ARM64::LD4Fourv2s, ARM64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 4, ARM64::LD4Fourv4s, ARM64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 4, ARM64::LD1Fourv1d, ARM64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 4, ARM64::LD4Fourv2d, ARM64::qsub0);
      break;
    case Intrinsic::arm64_neon_ld2r:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 2, ARM64::LD2Rv8b, ARM64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 2, ARM64::LD2Rv16b, ARM64::qsub0);
      else if (VT == MVT::v4i16)
        return SelectLoad(Node, 2, ARM64::LD2Rv4h, ARM64::dsub0);
      else if (VT == MVT::v8i16)
        return SelectLoad(Node, 2, ARM64::LD2Rv8h, ARM64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 2, ARM64::LD2Rv2s, ARM64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 2, ARM64::LD2Rv4s, ARM64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 2, ARM64::LD2Rv1d, ARM64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 2, ARM64::LD2Rv2d, ARM64::qsub0);
      break;
    case Intrinsic::arm64_neon_ld3r:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 3, ARM64::LD3Rv8b, ARM64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 3, ARM64::LD3Rv16b, ARM64::qsub0);
      else if (VT == MVT::v4i16)
        return SelectLoad(Node, 3, ARM64::LD3Rv4h, ARM64::dsub0);
      else if (VT == MVT::v8i16)
        return SelectLoad(Node, 3, ARM64::LD3Rv8h, ARM64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 3, ARM64::LD3Rv2s, ARM64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 3, ARM64::LD3Rv4s, ARM64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 3, ARM64::LD3Rv1d, ARM64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 3, ARM64::LD3Rv2d, ARM64::qsub0);
      break;
    case Intrinsic::arm64_neon_ld4r:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 4, ARM64::LD4Rv8b, ARM64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 4, ARM64::LD4Rv16b, ARM64::qsub0);
      else if (VT == MVT::v4i16)
        return SelectLoad(Node, 4, ARM64::LD4Rv4h, ARM64::dsub0);
      else if (VT == MVT::v8i16)
        return SelectLoad(Node, 4, ARM64::LD4Rv8h, ARM64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 4, ARM64::LD4Rv2s, ARM64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 4, ARM64::LD4Rv4s, ARM64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 4, ARM64::LD4Rv1d, ARM64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 4, ARM64::LD4Rv2d, ARM64::qsub0);
      break;
    case Intrinsic::arm64_neon_ld2lane:
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectLoadLane(Node, 2, ARM64::LD2i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16)
        return SelectLoadLane(Node, 2, ARM64::LD2i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectLoadLane(Node, 2, ARM64::LD2i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectLoadLane(Node, 2, ARM64::LD2i64);
      break;
    case Intrinsic::arm64_neon_ld3lane:
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectLoadLane(Node, 3, ARM64::LD3i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16)
        return SelectLoadLane(Node, 3, ARM64::LD3i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectLoadLane(Node, 3, ARM64::LD3i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectLoadLane(Node, 3, ARM64::LD3i64);
      break;
    case Intrinsic::arm64_neon_ld4lane:
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectLoadLane(Node, 4, ARM64::LD4i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16)
        return SelectLoadLane(Node, 4, ARM64::LD4i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectLoadLane(Node, 4, ARM64::LD4i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectLoadLane(Node, 4, ARM64::LD4i64);
      break;
    }
  } break;
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(0))->getZExtValue();
    switch (IntNo) {
    default:
      break;
    case Intrinsic::arm64_neon_tbl2:
      return SelectTable(Node, 2, VT == MVT::v8i8 ? ARM64::TBLv8i8Two
                                                  : ARM64::TBLv16i8Two,
                         false);
    case Intrinsic::arm64_neon_tbl3:
      return SelectTable(Node, 3, VT == MVT::v8i8 ? ARM64::TBLv8i8Three
                                                  : ARM64::TBLv16i8Three,
                         false);
    case Intrinsic::arm64_neon_tbl4:
      return SelectTable(Node, 4, VT == MVT::v8i8 ? ARM64::TBLv8i8Four
                                                  : ARM64::TBLv16i8Four,
                         false);
    case Intrinsic::arm64_neon_tbx2:
      return SelectTable(Node, 2, VT == MVT::v8i8 ? ARM64::TBXv8i8Two
                                                  : ARM64::TBXv16i8Two,
                         true);
    case Intrinsic::arm64_neon_tbx3:
      return SelectTable(Node, 3, VT == MVT::v8i8 ? ARM64::TBXv8i8Three
                                                  : ARM64::TBXv16i8Three,
                         true);
    case Intrinsic::arm64_neon_tbx4:
      return SelectTable(Node, 4, VT == MVT::v8i8 ? ARM64::TBXv8i8Four
                                                  : ARM64::TBXv16i8Four,
                         true);
    case Intrinsic::arm64_neon_smull:
    case Intrinsic::arm64_neon_umull:
      if (SDNode *N = SelectMULLV64LaneV128(IntNo, Node))
        return N;
      break;
    }
    break;
  }
  case ISD::INTRINSIC_VOID: {
    unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    if (Node->getNumOperands() >= 3)
      VT = Node->getOperand(2)->getValueType(0);
    switch (IntNo) {
    default:
      break;
    case Intrinsic::arm64_neon_st1x2: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 2, ARM64::ST1Twov8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 2, ARM64::ST1Twov16b);
      else if (VT == MVT::v4i16)
        return SelectStore(Node, 2, ARM64::ST1Twov4h);
      else if (VT == MVT::v8i16)
        return SelectStore(Node, 2, ARM64::ST1Twov8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 2, ARM64::ST1Twov2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 2, ARM64::ST1Twov4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 2, ARM64::ST1Twov2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 2, ARM64::ST1Twov1d);
      break;
    }
    case Intrinsic::arm64_neon_st1x3: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 3, ARM64::ST1Threev8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 3, ARM64::ST1Threev16b);
      else if (VT == MVT::v4i16)
        return SelectStore(Node, 3, ARM64::ST1Threev4h);
      else if (VT == MVT::v8i16)
        return SelectStore(Node, 3, ARM64::ST1Threev8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 3, ARM64::ST1Threev2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 3, ARM64::ST1Threev4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 3, ARM64::ST1Threev2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 3, ARM64::ST1Threev1d);
      break;
    }
    case Intrinsic::arm64_neon_st1x4: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 4, ARM64::ST1Fourv8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 4, ARM64::ST1Fourv16b);
      else if (VT == MVT::v4i16)
        return SelectStore(Node, 4, ARM64::ST1Fourv4h);
      else if (VT == MVT::v8i16)
        return SelectStore(Node, 4, ARM64::ST1Fourv8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 4, ARM64::ST1Fourv2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 4, ARM64::ST1Fourv4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 4, ARM64::ST1Fourv2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 4, ARM64::ST1Fourv1d);
      break;
    }
    case Intrinsic::arm64_neon_st2: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 2, ARM64::ST2Twov8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 2, ARM64::ST2Twov16b);
      else if (VT == MVT::v4i16)
        return SelectStore(Node, 2, ARM64::ST2Twov4h);
      else if (VT == MVT::v8i16)
        return SelectStore(Node, 2, ARM64::ST2Twov8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 2, ARM64::ST2Twov2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 2, ARM64::ST2Twov4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 2, ARM64::ST2Twov2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 2, ARM64::ST1Twov1d);
      break;
    }
    case Intrinsic::arm64_neon_st3: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 3, ARM64::ST3Threev8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 3, ARM64::ST3Threev16b);
      else if (VT == MVT::v4i16)
        return SelectStore(Node, 3, ARM64::ST3Threev4h);
      else if (VT == MVT::v8i16)
        return SelectStore(Node, 3, ARM64::ST3Threev8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 3, ARM64::ST3Threev2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 3, ARM64::ST3Threev4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 3, ARM64::ST3Threev2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 3, ARM64::ST1Threev1d);
      break;
    }
    case Intrinsic::arm64_neon_st4: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 4, ARM64::ST4Fourv8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 4, ARM64::ST4Fourv16b);
      else if (VT == MVT::v4i16)
        return SelectStore(Node, 4, ARM64::ST4Fourv4h);
      else if (VT == MVT::v8i16)
        return SelectStore(Node, 4, ARM64::ST4Fourv8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 4, ARM64::ST4Fourv2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 4, ARM64::ST4Fourv4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 4, ARM64::ST4Fourv2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 4, ARM64::ST1Fourv1d);
      break;
    }
    case Intrinsic::arm64_neon_st2lane: {
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectStoreLane(Node, 2, ARM64::ST2i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16)
        return SelectStoreLane(Node, 2, ARM64::ST2i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectStoreLane(Node, 2, ARM64::ST2i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectStoreLane(Node, 2, ARM64::ST2i64);
      break;
    }
    case Intrinsic::arm64_neon_st3lane: {
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectStoreLane(Node, 3, ARM64::ST3i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16)
        return SelectStoreLane(Node, 3, ARM64::ST3i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectStoreLane(Node, 3, ARM64::ST3i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectStoreLane(Node, 3, ARM64::ST3i64);
      break;
    }
    case Intrinsic::arm64_neon_st4lane: {
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectStoreLane(Node, 4, ARM64::ST4i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16)
        return SelectStoreLane(Node, 4, ARM64::ST4i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectStoreLane(Node, 4, ARM64::ST4i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectStoreLane(Node, 4, ARM64::ST4i64);
      break;
    }
    }
  }
  case ARM64ISD::LD2post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 2, ARM64::LD2Twov8b_POST, ARM64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 2, ARM64::LD2Twov16b_POST, ARM64::qsub0);
    else if (VT == MVT::v4i16)
      return SelectPostLoad(Node, 2, ARM64::LD2Twov4h_POST, ARM64::dsub0);
    else if (VT == MVT::v8i16)
      return SelectPostLoad(Node, 2, ARM64::LD2Twov8h_POST, ARM64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 2, ARM64::LD2Twov2s_POST, ARM64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 2, ARM64::LD2Twov4s_POST, ARM64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 2, ARM64::LD1Twov1d_POST, ARM64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 2, ARM64::LD2Twov2d_POST, ARM64::qsub0);
    break;
  }
  case ARM64ISD::LD3post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 3, ARM64::LD3Threev8b_POST, ARM64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 3, ARM64::LD3Threev16b_POST, ARM64::qsub0);
    else if (VT == MVT::v4i16)
      return SelectPostLoad(Node, 3, ARM64::LD3Threev4h_POST, ARM64::dsub0);
    else if (VT == MVT::v8i16)
      return SelectPostLoad(Node, 3, ARM64::LD3Threev8h_POST, ARM64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 3, ARM64::LD3Threev2s_POST, ARM64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 3, ARM64::LD3Threev4s_POST, ARM64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 3, ARM64::LD1Threev1d_POST, ARM64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 3, ARM64::LD3Threev2d_POST, ARM64::qsub0);
    break;
  }
  case ARM64ISD::LD4post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 4, ARM64::LD4Fourv8b_POST, ARM64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 4, ARM64::LD4Fourv16b_POST, ARM64::qsub0);
    else if (VT == MVT::v4i16)
      return SelectPostLoad(Node, 4, ARM64::LD4Fourv4h_POST, ARM64::dsub0);
    else if (VT == MVT::v8i16)
      return SelectPostLoad(Node, 4, ARM64::LD4Fourv8h_POST, ARM64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 4, ARM64::LD4Fourv2s_POST, ARM64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 4, ARM64::LD4Fourv4s_POST, ARM64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 4, ARM64::LD1Fourv1d_POST, ARM64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 4, ARM64::LD4Fourv2d_POST, ARM64::qsub0);
    break;
  }
  case ARM64ISD::LD1x2post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 2, ARM64::LD1Twov8b_POST, ARM64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 2, ARM64::LD1Twov16b_POST, ARM64::qsub0);
    else if (VT == MVT::v4i16)
      return SelectPostLoad(Node, 2, ARM64::LD1Twov4h_POST, ARM64::dsub0);
    else if (VT == MVT::v8i16)
      return SelectPostLoad(Node, 2, ARM64::LD1Twov8h_POST, ARM64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 2, ARM64::LD1Twov2s_POST, ARM64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 2, ARM64::LD1Twov4s_POST, ARM64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 2, ARM64::LD1Twov1d_POST, ARM64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 2, ARM64::LD1Twov2d_POST, ARM64::qsub0);
    break;
  }
  case ARM64ISD::LD1x3post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 3, ARM64::LD1Threev8b_POST, ARM64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 3, ARM64::LD1Threev16b_POST, ARM64::qsub0);
    else if (VT == MVT::v4i16)
      return SelectPostLoad(Node, 3, ARM64::LD1Threev4h_POST, ARM64::dsub0);
    else if (VT == MVT::v8i16)
      return SelectPostLoad(Node, 3, ARM64::LD1Threev8h_POST, ARM64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 3, ARM64::LD1Threev2s_POST, ARM64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 3, ARM64::LD1Threev4s_POST, ARM64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 3, ARM64::LD1Threev1d_POST, ARM64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 3, ARM64::LD1Threev2d_POST, ARM64::qsub0);
    break;
  }
  case ARM64ISD::LD1x4post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 4, ARM64::LD1Fourv8b_POST, ARM64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 4, ARM64::LD1Fourv16b_POST, ARM64::qsub0);
    else if (VT == MVT::v4i16)
      return SelectPostLoad(Node, 4, ARM64::LD1Fourv4h_POST, ARM64::dsub0);
    else if (VT == MVT::v8i16)
      return SelectPostLoad(Node, 4, ARM64::LD1Fourv8h_POST, ARM64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 4, ARM64::LD1Fourv2s_POST, ARM64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 4, ARM64::LD1Fourv4s_POST, ARM64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 4, ARM64::LD1Fourv1d_POST, ARM64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 4, ARM64::LD1Fourv2d_POST, ARM64::qsub0);
    break;
  }
  case ARM64ISD::LD1DUPpost: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 1, ARM64::LD1Rv8b_POST, ARM64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 1, ARM64::LD1Rv16b_POST, ARM64::qsub0);
    else if (VT == MVT::v4i16)
      return SelectPostLoad(Node, 1, ARM64::LD1Rv4h_POST, ARM64::dsub0);
    else if (VT == MVT::v8i16)
      return SelectPostLoad(Node, 1, ARM64::LD1Rv8h_POST, ARM64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 1, ARM64::LD1Rv2s_POST, ARM64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 1, ARM64::LD1Rv4s_POST, ARM64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 1, ARM64::LD1Rv1d_POST, ARM64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 1, ARM64::LD1Rv2d_POST, ARM64::qsub0);
    break;
  }
  case ARM64ISD::LD2DUPpost: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 2, ARM64::LD2Rv8b_POST, ARM64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 2, ARM64::LD2Rv16b_POST, ARM64::qsub0);
    else if (VT == MVT::v4i16)
      return SelectPostLoad(Node, 2, ARM64::LD2Rv4h_POST, ARM64::dsub0);
    else if (VT == MVT::v8i16)
      return SelectPostLoad(Node, 2, ARM64::LD2Rv8h_POST, ARM64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 2, ARM64::LD2Rv2s_POST, ARM64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 2, ARM64::LD2Rv4s_POST, ARM64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 2, ARM64::LD2Rv1d_POST, ARM64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 2, ARM64::LD2Rv2d_POST, ARM64::qsub0);
    break;
  }
  case ARM64ISD::LD3DUPpost: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 3, ARM64::LD3Rv8b_POST, ARM64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 3, ARM64::LD3Rv16b_POST, ARM64::qsub0);
    else if (VT == MVT::v4i16)
      return SelectPostLoad(Node, 3, ARM64::LD3Rv4h_POST, ARM64::dsub0);
    else if (VT == MVT::v8i16)
      return SelectPostLoad(Node, 3, ARM64::LD3Rv8h_POST, ARM64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 3, ARM64::LD3Rv2s_POST, ARM64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 3, ARM64::LD3Rv4s_POST, ARM64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 3, ARM64::LD3Rv1d_POST, ARM64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 3, ARM64::LD3Rv2d_POST, ARM64::qsub0);
    break;
  }
  case ARM64ISD::LD4DUPpost: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 4, ARM64::LD4Rv8b_POST, ARM64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 4, ARM64::LD4Rv16b_POST, ARM64::qsub0);
    else if (VT == MVT::v4i16)
      return SelectPostLoad(Node, 4, ARM64::LD4Rv4h_POST, ARM64::dsub0);
    else if (VT == MVT::v8i16)
      return SelectPostLoad(Node, 4, ARM64::LD4Rv8h_POST, ARM64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 4, ARM64::LD4Rv2s_POST, ARM64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 4, ARM64::LD4Rv4s_POST, ARM64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 4, ARM64::LD4Rv1d_POST, ARM64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 4, ARM64::LD4Rv2d_POST, ARM64::qsub0);
    break;
  }
  case ARM64ISD::LD1LANEpost: {
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostLoadLane(Node, 1, ARM64::LD1i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16)
      return SelectPostLoadLane(Node, 1, ARM64::LD1i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostLoadLane(Node, 1, ARM64::LD1i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostLoadLane(Node, 1, ARM64::LD1i64_POST);
    break;
  }
  case ARM64ISD::LD2LANEpost: {
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostLoadLane(Node, 2, ARM64::LD2i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16)
      return SelectPostLoadLane(Node, 2, ARM64::LD2i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostLoadLane(Node, 2, ARM64::LD2i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostLoadLane(Node, 2, ARM64::LD2i64_POST);
    break;
  }
  case ARM64ISD::LD3LANEpost: {
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostLoadLane(Node, 3, ARM64::LD3i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16)
      return SelectPostLoadLane(Node, 3, ARM64::LD3i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostLoadLane(Node, 3, ARM64::LD3i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostLoadLane(Node, 3, ARM64::LD3i64_POST);
    break;
  }
  case ARM64ISD::LD4LANEpost: {
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostLoadLane(Node, 4, ARM64::LD4i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16)
      return SelectPostLoadLane(Node, 4, ARM64::LD4i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostLoadLane(Node, 4, ARM64::LD4i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostLoadLane(Node, 4, ARM64::LD4i64_POST);
    break;
  }
  case ARM64ISD::ST2post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 2, ARM64::ST2Twov8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 2, ARM64::ST2Twov16b_POST);
    else if (VT == MVT::v4i16)
      return SelectPostStore(Node, 2, ARM64::ST2Twov4h_POST);
    else if (VT == MVT::v8i16)
      return SelectPostStore(Node, 2, ARM64::ST2Twov8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 2, ARM64::ST2Twov2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 2, ARM64::ST2Twov4s_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 2, ARM64::ST2Twov2d_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 2, ARM64::ST1Twov1d_POST);
    break;
  }
  case ARM64ISD::ST3post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 3, ARM64::ST3Threev8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 3, ARM64::ST3Threev16b_POST);
    else if (VT == MVT::v4i16)
      return SelectPostStore(Node, 3, ARM64::ST3Threev4h_POST);
    else if (VT == MVT::v8i16)
      return SelectPostStore(Node, 3, ARM64::ST3Threev8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 3, ARM64::ST3Threev2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 3, ARM64::ST3Threev4s_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 3, ARM64::ST3Threev2d_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 3, ARM64::ST1Threev1d_POST);
    break;
  }
  case ARM64ISD::ST4post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 4, ARM64::ST4Fourv8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 4, ARM64::ST4Fourv16b_POST);
    else if (VT == MVT::v4i16)
      return SelectPostStore(Node, 4, ARM64::ST4Fourv4h_POST);
    else if (VT == MVT::v8i16)
      return SelectPostStore(Node, 4, ARM64::ST4Fourv8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 4, ARM64::ST4Fourv2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 4, ARM64::ST4Fourv4s_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 4, ARM64::ST4Fourv2d_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 4, ARM64::ST1Fourv1d_POST);
    break;
  }
  case ARM64ISD::ST1x2post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 2, ARM64::ST1Twov8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 2, ARM64::ST1Twov16b_POST);
    else if (VT == MVT::v4i16)
      return SelectPostStore(Node, 2, ARM64::ST1Twov4h_POST);
    else if (VT == MVT::v8i16)
      return SelectPostStore(Node, 2, ARM64::ST1Twov8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 2, ARM64::ST1Twov2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 2, ARM64::ST1Twov4s_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 2, ARM64::ST1Twov1d_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 2, ARM64::ST1Twov2d_POST);
    break;
  }
  case ARM64ISD::ST1x3post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 3, ARM64::ST1Threev8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 3, ARM64::ST1Threev16b_POST);
    else if (VT == MVT::v4i16)
      return SelectPostStore(Node, 3, ARM64::ST1Threev4h_POST);
    else if (VT == MVT::v8i16)
      return SelectPostStore(Node, 3, ARM64::ST1Threev8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 3, ARM64::ST1Threev2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 3, ARM64::ST1Threev4s_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 3, ARM64::ST1Threev1d_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 3, ARM64::ST1Threev2d_POST);
    break;
  }
  case ARM64ISD::ST1x4post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 4, ARM64::ST1Fourv8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 4, ARM64::ST1Fourv16b_POST);
    else if (VT == MVT::v4i16)
      return SelectPostStore(Node, 4, ARM64::ST1Fourv4h_POST);
    else if (VT == MVT::v8i16)
      return SelectPostStore(Node, 4, ARM64::ST1Fourv8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 4, ARM64::ST1Fourv2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 4, ARM64::ST1Fourv4s_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 4, ARM64::ST1Fourv1d_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 4, ARM64::ST1Fourv2d_POST);
    break;
  }
  case ARM64ISD::ST2LANEpost: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostStoreLane(Node, 2, ARM64::ST2i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16)
      return SelectPostStoreLane(Node, 2, ARM64::ST2i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostStoreLane(Node, 2, ARM64::ST2i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostStoreLane(Node, 2, ARM64::ST2i64_POST);
    break;
  }
  case ARM64ISD::ST3LANEpost: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostStoreLane(Node, 3, ARM64::ST3i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16)
      return SelectPostStoreLane(Node, 3, ARM64::ST3i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostStoreLane(Node, 3, ARM64::ST3i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostStoreLane(Node, 3, ARM64::ST3i64_POST);
    break;
  }
  case ARM64ISD::ST4LANEpost: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostStoreLane(Node, 4, ARM64::ST4i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16)
      return SelectPostStoreLane(Node, 4, ARM64::ST4i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostStoreLane(Node, 4, ARM64::ST4i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostStoreLane(Node, 4, ARM64::ST4i64_POST);
    break;
  }

  case ISD::FCEIL:
  case ISD::FFLOOR:
  case ISD::FTRUNC:
  case ISD::FROUND:
    if (SDNode *I = SelectLIBM(Node))
      return I;
    break;
  }

  // Select the default instruction
  ResNode = SelectCode(Node);

  DEBUG(errs() << "=> ");
  if (ResNode == nullptr || ResNode == Node)
    DEBUG(Node->dump(CurDAG));
  else
    DEBUG(ResNode->dump(CurDAG));
  DEBUG(errs() << "\n");

  return ResNode;
}

/// createARM64ISelDag - This pass converts a legalized DAG into a
/// ARM64-specific DAG, ready for instruction scheduling.
FunctionPass *llvm::createARM64ISelDag(ARM64TargetMachine &TM,
                                       CodeGenOpt::Level OptLevel) {
  return new ARM64DAGToDAGISel(TM, OptLevel);
}
