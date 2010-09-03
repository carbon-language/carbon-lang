//===-- ARMISelDAGToDAG.cpp - A dag to dag inst selector for ARM ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the ARM target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-isel"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMTargetMachine.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool>
DisableShifterOp("disable-shifter-op", cl::Hidden,
  cl::desc("Disable isel of shifter-op"),
  cl::init(false));

//===--------------------------------------------------------------------===//
/// ARMDAGToDAGISel - ARM specific code to select ARM machine
/// instructions for SelectionDAG operations.
///
namespace {
class ARMDAGToDAGISel : public SelectionDAGISel {
  ARMBaseTargetMachine &TM;

  /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
  /// make the right decision when generating code for different targets.
  const ARMSubtarget *Subtarget;

public:
  explicit ARMDAGToDAGISel(ARMBaseTargetMachine &tm,
                           CodeGenOpt::Level OptLevel)
    : SelectionDAGISel(tm, OptLevel), TM(tm),
    Subtarget(&TM.getSubtarget<ARMSubtarget>()) {
  }

  virtual const char *getPassName() const {
    return "ARM Instruction Selection";
  }

  /// getI32Imm - Return a target constant of type i32 with the specified
  /// value.
  inline SDValue getI32Imm(unsigned Imm) {
    return CurDAG->getTargetConstant(Imm, MVT::i32);
  }

  SDNode *Select(SDNode *N);

  bool SelectShifterOperandReg(SDNode *Op, SDValue N, SDValue &A,
                               SDValue &B, SDValue &C);
  bool SelectAddrMode2(SDNode *Op, SDValue N, SDValue &Base,
                       SDValue &Offset, SDValue &Opc);
  bool SelectAddrMode2Offset(SDNode *Op, SDValue N,
                             SDValue &Offset, SDValue &Opc);
  bool SelectAddrMode3(SDNode *Op, SDValue N, SDValue &Base,
                       SDValue &Offset, SDValue &Opc);
  bool SelectAddrMode3Offset(SDNode *Op, SDValue N,
                             SDValue &Offset, SDValue &Opc);
  bool SelectAddrMode4(SDNode *Op, SDValue N, SDValue &Addr,
                       SDValue &Mode);
  bool SelectAddrMode5(SDNode *Op, SDValue N, SDValue &Base,
                       SDValue &Offset);
  bool SelectAddrMode6(SDNode *Op, SDValue N, SDValue &Addr, SDValue &Align);

  bool SelectAddrModePC(SDNode *Op, SDValue N, SDValue &Offset,
                        SDValue &Label);

  bool SelectThumbAddrModeRR(SDNode *Op, SDValue N, SDValue &Base,
                             SDValue &Offset);
  bool SelectThumbAddrModeRI5(SDNode *Op, SDValue N, unsigned Scale,
                              SDValue &Base, SDValue &OffImm,
                              SDValue &Offset);
  bool SelectThumbAddrModeS1(SDNode *Op, SDValue N, SDValue &Base,
                             SDValue &OffImm, SDValue &Offset);
  bool SelectThumbAddrModeS2(SDNode *Op, SDValue N, SDValue &Base,
                             SDValue &OffImm, SDValue &Offset);
  bool SelectThumbAddrModeS4(SDNode *Op, SDValue N, SDValue &Base,
                             SDValue &OffImm, SDValue &Offset);
  bool SelectThumbAddrModeSP(SDNode *Op, SDValue N, SDValue &Base,
                             SDValue &OffImm);

  bool SelectT2ShifterOperandReg(SDNode *Op, SDValue N,
                                 SDValue &BaseReg, SDValue &Opc);
  bool SelectT2AddrModeImm12(SDNode *Op, SDValue N, SDValue &Base,
                             SDValue &OffImm);
  bool SelectT2AddrModeImm8(SDNode *Op, SDValue N, SDValue &Base,
                            SDValue &OffImm);
  bool SelectT2AddrModeImm8Offset(SDNode *Op, SDValue N,
                                 SDValue &OffImm);
  bool SelectT2AddrModeImm8s4(SDNode *Op, SDValue N, SDValue &Base,
                              SDValue &OffImm);
  bool SelectT2AddrModeSoReg(SDNode *Op, SDValue N, SDValue &Base,
                             SDValue &OffReg, SDValue &ShImm);

  inline bool Pred_so_imm(SDNode *inN) const {
    ConstantSDNode *N = cast<ConstantSDNode>(inN);
    return ARM_AM::getSOImmVal(N->getZExtValue()) != -1;
  }

  inline bool Pred_t2_so_imm(SDNode *inN) const {
    ConstantSDNode *N = cast<ConstantSDNode>(inN);
    return ARM_AM::getT2SOImmVal(N->getZExtValue()) != -1;
  }

  // Include the pieces autogenerated from the target description.
#include "ARMGenDAGISel.inc"

private:
  /// SelectARMIndexedLoad - Indexed (pre/post inc/dec) load matching code for
  /// ARM.
  SDNode *SelectARMIndexedLoad(SDNode *N);
  SDNode *SelectT2IndexedLoad(SDNode *N);

  /// SelectVLD - Select NEON load intrinsics.  NumVecs should be
  /// 1, 2, 3 or 4.  The opcode arrays specify the instructions used for
  /// loads of D registers and even subregs and odd subregs of Q registers.
  /// For NumVecs <= 2, QOpcodes1 is not used.
  SDNode *SelectVLD(SDNode *N, unsigned NumVecs, unsigned *DOpcodes,
                    unsigned *QOpcodes0, unsigned *QOpcodes1);

  /// SelectVST - Select NEON store intrinsics.  NumVecs should
  /// be 1, 2, 3 or 4.  The opcode arrays specify the instructions used for
  /// stores of D registers and even subregs and odd subregs of Q registers.
  /// For NumVecs <= 2, QOpcodes1 is not used.
  SDNode *SelectVST(SDNode *N, unsigned NumVecs, unsigned *DOpcodes,
                    unsigned *QOpcodes0, unsigned *QOpcodes1);

  /// SelectVLDSTLane - Select NEON load/store lane intrinsics.  NumVecs should
  /// be 2, 3 or 4.  The opcode arrays specify the instructions used for
  /// load/store of D registers and even subregs and odd subregs of Q registers.
  SDNode *SelectVLDSTLane(SDNode *N, bool IsLoad, unsigned NumVecs,
                          unsigned *DOpcodes, unsigned *QOpcodes0,
                          unsigned *QOpcodes1);

  /// SelectVTBL - Select NEON VTBL and VTBX intrinsics.  NumVecs should be 2,
  /// 3 or 4.  These are custom-selected so that a REG_SEQUENCE can be
  /// generated to force the table registers to be consecutive.
  SDNode *SelectVTBL(SDNode *N, bool IsExt, unsigned NumVecs, unsigned Opc);

  /// SelectV6T2BitfieldExtractOp - Select SBFX/UBFX instructions for ARM.
  SDNode *SelectV6T2BitfieldExtractOp(SDNode *N, bool isSigned);

  /// SelectCMOVOp - Select CMOV instructions for ARM.
  SDNode *SelectCMOVOp(SDNode *N);
  SDNode *SelectT2CMOVShiftOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                              ARMCC::CondCodes CCVal, SDValue CCR,
                              SDValue InFlag);
  SDNode *SelectARMCMOVShiftOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                               ARMCC::CondCodes CCVal, SDValue CCR,
                               SDValue InFlag);
  SDNode *SelectT2CMOVSoImmOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                              ARMCC::CondCodes CCVal, SDValue CCR,
                              SDValue InFlag);
  SDNode *SelectARMCMOVSoImmOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                               ARMCC::CondCodes CCVal, SDValue CCR,
                               SDValue InFlag);

  SDNode *SelectConcatVector(SDNode *N);

  /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
  /// inline asm expressions.
  virtual bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                            char ConstraintCode,
                                            std::vector<SDValue> &OutOps);

  // Form pairs of consecutive S, D, or Q registers.
  SDNode *PairSRegs(EVT VT, SDValue V0, SDValue V1);
  SDNode *PairDRegs(EVT VT, SDValue V0, SDValue V1);
  SDNode *PairQRegs(EVT VT, SDValue V0, SDValue V1);

  // Form sequences of 4 consecutive S, D, or Q registers.
  SDNode *QuadSRegs(EVT VT, SDValue V0, SDValue V1, SDValue V2, SDValue V3);
  SDNode *QuadDRegs(EVT VT, SDValue V0, SDValue V1, SDValue V2, SDValue V3);
  SDNode *QuadQRegs(EVT VT, SDValue V0, SDValue V1, SDValue V2, SDValue V3);

  // Form sequences of 8 consecutive D registers.
  SDNode *OctoDRegs(EVT VT, SDValue V0, SDValue V1, SDValue V2, SDValue V3,
                    SDValue V4, SDValue V5, SDValue V6, SDValue V7);
};
}

/// isInt32Immediate - This method tests to see if the node is a 32-bit constant
/// operand. If so Imm will receive the 32-bit value.
static bool isInt32Immediate(SDNode *N, unsigned &Imm) {
  if (N->getOpcode() == ISD::Constant && N->getValueType(0) == MVT::i32) {
    Imm = cast<ConstantSDNode>(N)->getZExtValue();
    return true;
  }
  return false;
}

// isInt32Immediate - This method tests to see if a constant operand.
// If so Imm will receive the 32 bit value.
static bool isInt32Immediate(SDValue N, unsigned &Imm) {
  return isInt32Immediate(N.getNode(), Imm);
}

// isOpcWithIntImmediate - This method tests to see if the node is a specific
// opcode and that it has a immediate integer right operand.
// If so Imm will receive the 32 bit value.
static bool isOpcWithIntImmediate(SDNode *N, unsigned Opc, unsigned& Imm) {
  return N->getOpcode() == Opc &&
         isInt32Immediate(N->getOperand(1).getNode(), Imm);
}


bool ARMDAGToDAGISel::SelectShifterOperandReg(SDNode *Op,
                                              SDValue N,
                                              SDValue &BaseReg,
                                              SDValue &ShReg,
                                              SDValue &Opc) {
  if (DisableShifterOp)
    return false;

  ARM_AM::ShiftOpc ShOpcVal = ARM_AM::getShiftOpcForNode(N);

  // Don't match base register only case. That is matched to a separate
  // lower complexity pattern with explicit register operand.
  if (ShOpcVal == ARM_AM::no_shift) return false;

  BaseReg = N.getOperand(0);
  unsigned ShImmVal = 0;
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    ShReg = CurDAG->getRegister(0, MVT::i32);
    ShImmVal = RHS->getZExtValue() & 31;
  } else {
    ShReg = N.getOperand(1);
  }
  Opc = CurDAG->getTargetConstant(ARM_AM::getSORegOpc(ShOpcVal, ShImmVal),
                                  MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode2(SDNode *Op, SDValue N,
                                      SDValue &Base, SDValue &Offset,
                                      SDValue &Opc) {
  if (N.getOpcode() == ISD::MUL) {
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      // X * [3,5,9] -> X + X * [2,4,8] etc.
      int RHSC = (int)RHS->getZExtValue();
      if (RHSC & 1) {
        RHSC = RHSC & ~1;
        ARM_AM::AddrOpc AddSub = ARM_AM::add;
        if (RHSC < 0) {
          AddSub = ARM_AM::sub;
          RHSC = - RHSC;
        }
        if (isPowerOf2_32(RHSC)) {
          unsigned ShAmt = Log2_32(RHSC);
          Base = Offset = N.getOperand(0);
          Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, ShAmt,
                                                            ARM_AM::lsl),
                                          MVT::i32);
          return true;
        }
      }
    }
  }

  if (N.getOpcode() != ISD::ADD && N.getOpcode() != ISD::SUB) {
    Base = N;
    if (N.getOpcode() == ISD::FrameIndex) {
      int FI = cast<FrameIndexSDNode>(N)->getIndex();
      Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    } else if (N.getOpcode() == ARMISD::Wrapper &&
               !(Subtarget->useMovt() &&
                 N.getOperand(0).getOpcode() == ISD::TargetGlobalAddress)) {
      Base = N.getOperand(0);
    }
    Offset = CurDAG->getRegister(0, MVT::i32);
    Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(ARM_AM::add, 0,
                                                      ARM_AM::no_shift),
                                    MVT::i32);
    return true;
  }

  // Match simple R +/- imm12 operands.
  if (N.getOpcode() == ISD::ADD)
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      int RHSC = (int)RHS->getZExtValue();
      if ((RHSC >= 0 && RHSC < 0x1000) ||
          (RHSC < 0 && RHSC > -0x1000)) { // 12 bits.
        Base = N.getOperand(0);
        if (Base.getOpcode() == ISD::FrameIndex) {
          int FI = cast<FrameIndexSDNode>(Base)->getIndex();
          Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
        }
        Offset = CurDAG->getRegister(0, MVT::i32);

        ARM_AM::AddrOpc AddSub = ARM_AM::add;
        if (RHSC < 0) {
          AddSub = ARM_AM::sub;
          RHSC = - RHSC;
        }
        Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, RHSC,
                                                          ARM_AM::no_shift),
                                        MVT::i32);
        return true;
      }
    }

  // Otherwise this is R +/- [possibly shifted] R.
  ARM_AM::AddrOpc AddSub = N.getOpcode() == ISD::ADD ? ARM_AM::add:ARM_AM::sub;
  ARM_AM::ShiftOpc ShOpcVal = ARM_AM::getShiftOpcForNode(N.getOperand(1));
  unsigned ShAmt = 0;

  Base   = N.getOperand(0);
  Offset = N.getOperand(1);

  if (ShOpcVal != ARM_AM::no_shift) {
    // Check to see if the RHS of the shift is a constant, if not, we can't fold
    // it.
    if (ConstantSDNode *Sh =
           dyn_cast<ConstantSDNode>(N.getOperand(1).getOperand(1))) {
      ShAmt = Sh->getZExtValue();
      Offset = N.getOperand(1).getOperand(0);
    } else {
      ShOpcVal = ARM_AM::no_shift;
    }
  }

  // Try matching (R shl C) + (R).
  if (N.getOpcode() == ISD::ADD && ShOpcVal == ARM_AM::no_shift) {
    ShOpcVal = ARM_AM::getShiftOpcForNode(N.getOperand(0));
    if (ShOpcVal != ARM_AM::no_shift) {
      // Check to see if the RHS of the shift is a constant, if not, we can't
      // fold it.
      if (ConstantSDNode *Sh =
          dyn_cast<ConstantSDNode>(N.getOperand(0).getOperand(1))) {
        ShAmt = Sh->getZExtValue();
        Offset = N.getOperand(0).getOperand(0);
        Base = N.getOperand(1);
      } else {
        ShOpcVal = ARM_AM::no_shift;
      }
    }
  }

  Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, ShAmt, ShOpcVal),
                                  MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode2Offset(SDNode *Op, SDValue N,
                                            SDValue &Offset, SDValue &Opc) {
  unsigned Opcode = Op->getOpcode();
  ISD::MemIndexedMode AM = (Opcode == ISD::LOAD)
    ? cast<LoadSDNode>(Op)->getAddressingMode()
    : cast<StoreSDNode>(Op)->getAddressingMode();
  ARM_AM::AddrOpc AddSub = (AM == ISD::PRE_INC || AM == ISD::POST_INC)
    ? ARM_AM::add : ARM_AM::sub;
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N)) {
    int Val = (int)C->getZExtValue();
    if (Val >= 0 && Val < 0x1000) { // 12 bits.
      Offset = CurDAG->getRegister(0, MVT::i32);
      Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, Val,
                                                        ARM_AM::no_shift),
                                      MVT::i32);
      return true;
    }
  }

  Offset = N;
  ARM_AM::ShiftOpc ShOpcVal = ARM_AM::getShiftOpcForNode(N);
  unsigned ShAmt = 0;
  if (ShOpcVal != ARM_AM::no_shift) {
    // Check to see if the RHS of the shift is a constant, if not, we can't fold
    // it.
    if (ConstantSDNode *Sh = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      ShAmt = Sh->getZExtValue();
      Offset = N.getOperand(0);
    } else {
      ShOpcVal = ARM_AM::no_shift;
    }
  }

  Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, ShAmt, ShOpcVal),
                                  MVT::i32);
  return true;
}


bool ARMDAGToDAGISel::SelectAddrMode3(SDNode *Op, SDValue N,
                                      SDValue &Base, SDValue &Offset,
                                      SDValue &Opc) {
  if (N.getOpcode() == ISD::SUB) {
    // X - C  is canonicalize to X + -C, no need to handle it here.
    Base = N.getOperand(0);
    Offset = N.getOperand(1);
    Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(ARM_AM::sub, 0),MVT::i32);
    return true;
  }

  if (N.getOpcode() != ISD::ADD) {
    Base = N;
    if (N.getOpcode() == ISD::FrameIndex) {
      int FI = cast<FrameIndexSDNode>(N)->getIndex();
      Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    }
    Offset = CurDAG->getRegister(0, MVT::i32);
    Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(ARM_AM::add, 0),MVT::i32);
    return true;
  }

  // If the RHS is +/- imm8, fold into addr mode.
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    int RHSC = (int)RHS->getZExtValue();
    if ((RHSC >= 0 && RHSC < 256) ||
        (RHSC < 0 && RHSC > -256)) { // note -256 itself isn't allowed.
      Base = N.getOperand(0);
      if (Base.getOpcode() == ISD::FrameIndex) {
        int FI = cast<FrameIndexSDNode>(Base)->getIndex();
        Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
      }
      Offset = CurDAG->getRegister(0, MVT::i32);

      ARM_AM::AddrOpc AddSub = ARM_AM::add;
      if (RHSC < 0) {
        AddSub = ARM_AM::sub;
        RHSC = - RHSC;
      }
      Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(AddSub, RHSC),MVT::i32);
      return true;
    }
  }

  Base = N.getOperand(0);
  Offset = N.getOperand(1);
  Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(ARM_AM::add, 0), MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode3Offset(SDNode *Op, SDValue N,
                                            SDValue &Offset, SDValue &Opc) {
  unsigned Opcode = Op->getOpcode();
  ISD::MemIndexedMode AM = (Opcode == ISD::LOAD)
    ? cast<LoadSDNode>(Op)->getAddressingMode()
    : cast<StoreSDNode>(Op)->getAddressingMode();
  ARM_AM::AddrOpc AddSub = (AM == ISD::PRE_INC || AM == ISD::POST_INC)
    ? ARM_AM::add : ARM_AM::sub;
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N)) {
    int Val = (int)C->getZExtValue();
    if (Val >= 0 && Val < 256) {
      Offset = CurDAG->getRegister(0, MVT::i32);
      Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(AddSub, Val), MVT::i32);
      return true;
    }
  }

  Offset = N;
  Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(AddSub, 0), MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode4(SDNode *Op, SDValue N,
                                      SDValue &Addr, SDValue &Mode) {
  Addr = N;
  Mode = CurDAG->getTargetConstant(ARM_AM::getAM4ModeImm(ARM_AM::ia), MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode5(SDNode *Op, SDValue N,
                                      SDValue &Base, SDValue &Offset) {
  if (N.getOpcode() != ISD::ADD) {
    Base = N;
    if (N.getOpcode() == ISD::FrameIndex) {
      int FI = cast<FrameIndexSDNode>(N)->getIndex();
      Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    } else if (N.getOpcode() == ARMISD::Wrapper &&
               !(Subtarget->useMovt() &&
                 N.getOperand(0).getOpcode() == ISD::TargetGlobalAddress)) {
      Base = N.getOperand(0);
    }
    Offset = CurDAG->getTargetConstant(ARM_AM::getAM5Opc(ARM_AM::add, 0),
                                       MVT::i32);
    return true;
  }

  // If the RHS is +/- imm8, fold into addr mode.
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    int RHSC = (int)RHS->getZExtValue();
    if ((RHSC & 3) == 0) {  // The constant is implicitly multiplied by 4.
      RHSC >>= 2;
      if ((RHSC >= 0 && RHSC < 256) ||
          (RHSC < 0 && RHSC > -256)) { // note -256 itself isn't allowed.
        Base = N.getOperand(0);
        if (Base.getOpcode() == ISD::FrameIndex) {
          int FI = cast<FrameIndexSDNode>(Base)->getIndex();
          Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
        }

        ARM_AM::AddrOpc AddSub = ARM_AM::add;
        if (RHSC < 0) {
          AddSub = ARM_AM::sub;
          RHSC = - RHSC;
        }
        Offset = CurDAG->getTargetConstant(ARM_AM::getAM5Opc(AddSub, RHSC),
                                           MVT::i32);
        return true;
      }
    }
  }

  Base = N;
  Offset = CurDAG->getTargetConstant(ARM_AM::getAM5Opc(ARM_AM::add, 0),
                                     MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode6(SDNode *Op, SDValue N,
                                      SDValue &Addr, SDValue &Align) {
  Addr = N;
  // Default to no alignment.
  Align = CurDAG->getTargetConstant(0, MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrModePC(SDNode *Op, SDValue N,
                                       SDValue &Offset, SDValue &Label) {
  if (N.getOpcode() == ARMISD::PIC_ADD && N.hasOneUse()) {
    Offset = N.getOperand(0);
    SDValue N1 = N.getOperand(1);
    Label  = CurDAG->getTargetConstant(cast<ConstantSDNode>(N1)->getZExtValue(),
                                       MVT::i32);
    return true;
  }
  return false;
}

bool ARMDAGToDAGISel::SelectThumbAddrModeRR(SDNode *Op, SDValue N,
                                            SDValue &Base, SDValue &Offset){
  // FIXME dl should come from the parent load or store, not the address
  if (N.getOpcode() != ISD::ADD) {
    ConstantSDNode *NC = dyn_cast<ConstantSDNode>(N);
    if (!NC || !NC->isNullValue())
      return false;

    Base = Offset = N;
    return true;
  }

  Base = N.getOperand(0);
  Offset = N.getOperand(1);
  return true;
}

bool
ARMDAGToDAGISel::SelectThumbAddrModeRI5(SDNode *Op, SDValue N,
                                        unsigned Scale, SDValue &Base,
                                        SDValue &OffImm, SDValue &Offset) {
  if (Scale == 4) {
    SDValue TmpBase, TmpOffImm;
    if (SelectThumbAddrModeSP(Op, N, TmpBase, TmpOffImm))
      return false;  // We want to select tLDRspi / tSTRspi instead.
    if (N.getOpcode() == ARMISD::Wrapper &&
        N.getOperand(0).getOpcode() == ISD::TargetConstantPool)
      return false;  // We want to select tLDRpci instead.
  }

  if (N.getOpcode() != ISD::ADD) {
    if (N.getOpcode() == ARMISD::Wrapper &&
        !(Subtarget->useMovt() &&
          N.getOperand(0).getOpcode() == ISD::TargetGlobalAddress)) {
      Base = N.getOperand(0);
    } else
      Base = N;

    Offset = CurDAG->getRegister(0, MVT::i32);
    OffImm = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }

  // Thumb does not have [sp, r] address mode.
  RegisterSDNode *LHSR = dyn_cast<RegisterSDNode>(N.getOperand(0));
  RegisterSDNode *RHSR = dyn_cast<RegisterSDNode>(N.getOperand(1));
  if ((LHSR && LHSR->getReg() == ARM::SP) ||
      (RHSR && RHSR->getReg() == ARM::SP)) {
    Base = N;
    Offset = CurDAG->getRegister(0, MVT::i32);
    OffImm = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }

  // If the RHS is + imm5 * scale, fold into addr mode.
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    int RHSC = (int)RHS->getZExtValue();
    if ((RHSC & (Scale-1)) == 0) {  // The constant is implicitly multiplied.
      RHSC /= Scale;
      if (RHSC >= 0 && RHSC < 32) {
        Base = N.getOperand(0);
        Offset = CurDAG->getRegister(0, MVT::i32);
        OffImm = CurDAG->getTargetConstant(RHSC, MVT::i32);
        return true;
      }
    }
  }

  Base = N.getOperand(0);
  Offset = N.getOperand(1);
  OffImm = CurDAG->getTargetConstant(0, MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectThumbAddrModeS1(SDNode *Op, SDValue N,
                                            SDValue &Base, SDValue &OffImm,
                                            SDValue &Offset) {
  return SelectThumbAddrModeRI5(Op, N, 1, Base, OffImm, Offset);
}

bool ARMDAGToDAGISel::SelectThumbAddrModeS2(SDNode *Op, SDValue N,
                                            SDValue &Base, SDValue &OffImm,
                                            SDValue &Offset) {
  return SelectThumbAddrModeRI5(Op, N, 2, Base, OffImm, Offset);
}

bool ARMDAGToDAGISel::SelectThumbAddrModeS4(SDNode *Op, SDValue N,
                                            SDValue &Base, SDValue &OffImm,
                                            SDValue &Offset) {
  return SelectThumbAddrModeRI5(Op, N, 4, Base, OffImm, Offset);
}

bool ARMDAGToDAGISel::SelectThumbAddrModeSP(SDNode *Op, SDValue N,
                                           SDValue &Base, SDValue &OffImm) {
  if (N.getOpcode() == ISD::FrameIndex) {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    OffImm = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }

  if (N.getOpcode() != ISD::ADD)
    return false;

  RegisterSDNode *LHSR = dyn_cast<RegisterSDNode>(N.getOperand(0));
  if (N.getOperand(0).getOpcode() == ISD::FrameIndex ||
      (LHSR && LHSR->getReg() == ARM::SP)) {
    // If the RHS is + imm8 * scale, fold into addr mode.
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      int RHSC = (int)RHS->getZExtValue();
      if ((RHSC & 3) == 0) {  // The constant is implicitly multiplied.
        RHSC >>= 2;
        if (RHSC >= 0 && RHSC < 256) {
          Base = N.getOperand(0);
          if (Base.getOpcode() == ISD::FrameIndex) {
            int FI = cast<FrameIndexSDNode>(Base)->getIndex();
            Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
          }
          OffImm = CurDAG->getTargetConstant(RHSC, MVT::i32);
          return true;
        }
      }
    }
  }

  return false;
}

bool ARMDAGToDAGISel::SelectT2ShifterOperandReg(SDNode *Op, SDValue N,
                                                SDValue &BaseReg,
                                                SDValue &Opc) {
  if (DisableShifterOp)
    return false;

  ARM_AM::ShiftOpc ShOpcVal = ARM_AM::getShiftOpcForNode(N);

  // Don't match base register only case. That is matched to a separate
  // lower complexity pattern with explicit register operand.
  if (ShOpcVal == ARM_AM::no_shift) return false;

  BaseReg = N.getOperand(0);
  unsigned ShImmVal = 0;
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    ShImmVal = RHS->getZExtValue() & 31;
    Opc = getI32Imm(ARM_AM::getSORegOpc(ShOpcVal, ShImmVal));
    return true;
  }

  return false;
}

bool ARMDAGToDAGISel::SelectT2AddrModeImm12(SDNode *Op, SDValue N,
                                            SDValue &Base, SDValue &OffImm) {
  // Match simple R + imm12 operands.

  // Base only.
  if (N.getOpcode() != ISD::ADD && N.getOpcode() != ISD::SUB) {
    if (N.getOpcode() == ISD::FrameIndex) {
      // Match frame index...
      int FI = cast<FrameIndexSDNode>(N)->getIndex();
      Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
      OffImm  = CurDAG->getTargetConstant(0, MVT::i32);
      return true;
    } else if (N.getOpcode() == ARMISD::Wrapper &&
               !(Subtarget->useMovt() &&
                 N.getOperand(0).getOpcode() == ISD::TargetGlobalAddress)) {
      Base = N.getOperand(0);
      if (Base.getOpcode() == ISD::TargetConstantPool)
        return false;  // We want to select t2LDRpci instead.
    } else
      Base = N;
    OffImm  = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }

  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    if (SelectT2AddrModeImm8(Op, N, Base, OffImm))
      // Let t2LDRi8 handle (R - imm8).
      return false;

    int RHSC = (int)RHS->getZExtValue();
    if (N.getOpcode() == ISD::SUB)
      RHSC = -RHSC;

    if (RHSC >= 0 && RHSC < 0x1000) { // 12 bits (unsigned)
      Base   = N.getOperand(0);
      if (Base.getOpcode() == ISD::FrameIndex) {
        int FI = cast<FrameIndexSDNode>(Base)->getIndex();
        Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
      }
      OffImm = CurDAG->getTargetConstant(RHSC, MVT::i32);
      return true;
    }
  }

  // Base only.
  Base = N;
  OffImm  = CurDAG->getTargetConstant(0, MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectT2AddrModeImm8(SDNode *Op, SDValue N,
                                           SDValue &Base, SDValue &OffImm) {
  // Match simple R - imm8 operands.
  if (N.getOpcode() == ISD::ADD || N.getOpcode() == ISD::SUB) {
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      int RHSC = (int)RHS->getSExtValue();
      if (N.getOpcode() == ISD::SUB)
        RHSC = -RHSC;

      if ((RHSC >= -255) && (RHSC < 0)) { // 8 bits (always negative)
        Base = N.getOperand(0);
        if (Base.getOpcode() == ISD::FrameIndex) {
          int FI = cast<FrameIndexSDNode>(Base)->getIndex();
          Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
        }
        OffImm = CurDAG->getTargetConstant(RHSC, MVT::i32);
        return true;
      }
    }
  }

  return false;
}

bool ARMDAGToDAGISel::SelectT2AddrModeImm8Offset(SDNode *Op, SDValue N,
                                                 SDValue &OffImm){
  unsigned Opcode = Op->getOpcode();
  ISD::MemIndexedMode AM = (Opcode == ISD::LOAD)
    ? cast<LoadSDNode>(Op)->getAddressingMode()
    : cast<StoreSDNode>(Op)->getAddressingMode();
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N)) {
    int RHSC = (int)RHS->getZExtValue();
    if (RHSC >= 0 && RHSC < 0x100) { // 8 bits.
      OffImm = ((AM == ISD::PRE_INC) || (AM == ISD::POST_INC))
        ? CurDAG->getTargetConstant(RHSC, MVT::i32)
        : CurDAG->getTargetConstant(-RHSC, MVT::i32);
      return true;
    }
  }

  return false;
}

bool ARMDAGToDAGISel::SelectT2AddrModeImm8s4(SDNode *Op, SDValue N,
                                             SDValue &Base, SDValue &OffImm) {
  if (N.getOpcode() == ISD::ADD) {
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      int RHSC = (int)RHS->getZExtValue();
      // 8 bits.
      if (((RHSC & 0x3) == 0) &&
          ((RHSC >= 0 && RHSC < 0x400) || (RHSC < 0 && RHSC > -0x400))) {
        Base   = N.getOperand(0);
        OffImm = CurDAG->getTargetConstant(RHSC, MVT::i32);
        return true;
      }
    }
  } else if (N.getOpcode() == ISD::SUB) {
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      int RHSC = (int)RHS->getZExtValue();
      // 8 bits.
      if (((RHSC & 0x3) == 0) && (RHSC >= 0 && RHSC < 0x400)) {
        Base   = N.getOperand(0);
        OffImm = CurDAG->getTargetConstant(-RHSC, MVT::i32);
        return true;
      }
    }
  }

  return false;
}

bool ARMDAGToDAGISel::SelectT2AddrModeSoReg(SDNode *Op, SDValue N,
                                            SDValue &Base,
                                            SDValue &OffReg, SDValue &ShImm) {
  // (R - imm8) should be handled by t2LDRi8. The rest are handled by t2LDRi12.
  if (N.getOpcode() != ISD::ADD)
    return false;

  // Leave (R + imm12) for t2LDRi12, (R - imm8) for t2LDRi8.
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    int RHSC = (int)RHS->getZExtValue();
    if (RHSC >= 0 && RHSC < 0x1000) // 12 bits (unsigned)
      return false;
    else if (RHSC < 0 && RHSC >= -255) // 8 bits
      return false;
  }

  // Look for (R + R) or (R + (R << [1,2,3])).
  unsigned ShAmt = 0;
  Base   = N.getOperand(0);
  OffReg = N.getOperand(1);

  // Swap if it is ((R << c) + R).
  ARM_AM::ShiftOpc ShOpcVal = ARM_AM::getShiftOpcForNode(OffReg);
  if (ShOpcVal != ARM_AM::lsl) {
    ShOpcVal = ARM_AM::getShiftOpcForNode(Base);
    if (ShOpcVal == ARM_AM::lsl)
      std::swap(Base, OffReg);
  }

  if (ShOpcVal == ARM_AM::lsl) {
    // Check to see if the RHS of the shift is a constant, if not, we can't fold
    // it.
    if (ConstantSDNode *Sh = dyn_cast<ConstantSDNode>(OffReg.getOperand(1))) {
      ShAmt = Sh->getZExtValue();
      if (ShAmt >= 4) {
        ShAmt = 0;
        ShOpcVal = ARM_AM::no_shift;
      } else
        OffReg = OffReg.getOperand(0);
    } else {
      ShOpcVal = ARM_AM::no_shift;
    }
  }

  ShImm = CurDAG->getTargetConstant(ShAmt, MVT::i32);

  return true;
}

//===--------------------------------------------------------------------===//

/// getAL - Returns a ARMCC::AL immediate node.
static inline SDValue getAL(SelectionDAG *CurDAG) {
  return CurDAG->getTargetConstant((uint64_t)ARMCC::AL, MVT::i32);
}

SDNode *ARMDAGToDAGISel::SelectARMIndexedLoad(SDNode *N) {
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::MemIndexedMode AM = LD->getAddressingMode();
  if (AM == ISD::UNINDEXED)
    return NULL;

  EVT LoadedVT = LD->getMemoryVT();
  SDValue Offset, AMOpc;
  bool isPre = (AM == ISD::PRE_INC) || (AM == ISD::PRE_DEC);
  unsigned Opcode = 0;
  bool Match = false;
  if (LoadedVT == MVT::i32 &&
      SelectAddrMode2Offset(N, LD->getOffset(), Offset, AMOpc)) {
    Opcode = isPre ? ARM::LDR_PRE : ARM::LDR_POST;
    Match = true;
  } else if (LoadedVT == MVT::i16 &&
             SelectAddrMode3Offset(N, LD->getOffset(), Offset, AMOpc)) {
    Match = true;
    Opcode = (LD->getExtensionType() == ISD::SEXTLOAD)
      ? (isPre ? ARM::LDRSH_PRE : ARM::LDRSH_POST)
      : (isPre ? ARM::LDRH_PRE : ARM::LDRH_POST);
  } else if (LoadedVT == MVT::i8 || LoadedVT == MVT::i1) {
    if (LD->getExtensionType() == ISD::SEXTLOAD) {
      if (SelectAddrMode3Offset(N, LD->getOffset(), Offset, AMOpc)) {
        Match = true;
        Opcode = isPre ? ARM::LDRSB_PRE : ARM::LDRSB_POST;
      }
    } else {
      if (SelectAddrMode2Offset(N, LD->getOffset(), Offset, AMOpc)) {
        Match = true;
        Opcode = isPre ? ARM::LDRB_PRE : ARM::LDRB_POST;
      }
    }
  }

  if (Match) {
    SDValue Chain = LD->getChain();
    SDValue Base = LD->getBasePtr();
    SDValue Ops[]= { Base, Offset, AMOpc, getAL(CurDAG),
                     CurDAG->getRegister(0, MVT::i32), Chain };
    return CurDAG->getMachineNode(Opcode, N->getDebugLoc(), MVT::i32, MVT::i32,
                                  MVT::Other, Ops, 6);
  }

  return NULL;
}

SDNode *ARMDAGToDAGISel::SelectT2IndexedLoad(SDNode *N) {
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::MemIndexedMode AM = LD->getAddressingMode();
  if (AM == ISD::UNINDEXED)
    return NULL;

  EVT LoadedVT = LD->getMemoryVT();
  bool isSExtLd = LD->getExtensionType() == ISD::SEXTLOAD;
  SDValue Offset;
  bool isPre = (AM == ISD::PRE_INC) || (AM == ISD::PRE_DEC);
  unsigned Opcode = 0;
  bool Match = false;
  if (SelectT2AddrModeImm8Offset(N, LD->getOffset(), Offset)) {
    switch (LoadedVT.getSimpleVT().SimpleTy) {
    case MVT::i32:
      Opcode = isPre ? ARM::t2LDR_PRE : ARM::t2LDR_POST;
      break;
    case MVT::i16:
      if (isSExtLd)
        Opcode = isPre ? ARM::t2LDRSH_PRE : ARM::t2LDRSH_POST;
      else
        Opcode = isPre ? ARM::t2LDRH_PRE : ARM::t2LDRH_POST;
      break;
    case MVT::i8:
    case MVT::i1:
      if (isSExtLd)
        Opcode = isPre ? ARM::t2LDRSB_PRE : ARM::t2LDRSB_POST;
      else
        Opcode = isPre ? ARM::t2LDRB_PRE : ARM::t2LDRB_POST;
      break;
    default:
      return NULL;
    }
    Match = true;
  }

  if (Match) {
    SDValue Chain = LD->getChain();
    SDValue Base = LD->getBasePtr();
    SDValue Ops[]= { Base, Offset, getAL(CurDAG),
                     CurDAG->getRegister(0, MVT::i32), Chain };
    return CurDAG->getMachineNode(Opcode, N->getDebugLoc(), MVT::i32, MVT::i32,
                                  MVT::Other, Ops, 5);
  }

  return NULL;
}

/// PairSRegs - Form a D register from a pair of S registers.
///
SDNode *ARMDAGToDAGISel::PairSRegs(EVT VT, SDValue V0, SDValue V1) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::ssub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::ssub_1, MVT::i32);
  const SDValue Ops[] = { V0, SubReg0, V1, SubReg1 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 4);
}

/// PairDRegs - Form a quad register from a pair of D registers.
///
SDNode *ARMDAGToDAGISel::PairDRegs(EVT VT, SDValue V0, SDValue V1) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::dsub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::dsub_1, MVT::i32);
  const SDValue Ops[] = { V0, SubReg0, V1, SubReg1 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 4);
}

/// PairQRegs - Form 4 consecutive D registers from a pair of Q registers.
///
SDNode *ARMDAGToDAGISel::PairQRegs(EVT VT, SDValue V0, SDValue V1) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::qsub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::qsub_1, MVT::i32);
  const SDValue Ops[] = { V0, SubReg0, V1, SubReg1 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 4);
}

/// QuadSRegs - Form 4 consecutive S registers.
///
SDNode *ARMDAGToDAGISel::QuadSRegs(EVT VT, SDValue V0, SDValue V1,
                                   SDValue V2, SDValue V3) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::ssub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::ssub_1, MVT::i32);
  SDValue SubReg2 = CurDAG->getTargetConstant(ARM::ssub_2, MVT::i32);
  SDValue SubReg3 = CurDAG->getTargetConstant(ARM::ssub_3, MVT::i32);
  const SDValue Ops[] = { V0, SubReg0, V1, SubReg1, V2, SubReg2, V3, SubReg3 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 8);
}

/// QuadDRegs - Form 4 consecutive D registers.
///
SDNode *ARMDAGToDAGISel::QuadDRegs(EVT VT, SDValue V0, SDValue V1,
                                   SDValue V2, SDValue V3) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::dsub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::dsub_1, MVT::i32);
  SDValue SubReg2 = CurDAG->getTargetConstant(ARM::dsub_2, MVT::i32);
  SDValue SubReg3 = CurDAG->getTargetConstant(ARM::dsub_3, MVT::i32);
  const SDValue Ops[] = { V0, SubReg0, V1, SubReg1, V2, SubReg2, V3, SubReg3 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 8);
}

/// QuadQRegs - Form 4 consecutive Q registers.
///
SDNode *ARMDAGToDAGISel::QuadQRegs(EVT VT, SDValue V0, SDValue V1,
                                   SDValue V2, SDValue V3) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::qsub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::qsub_1, MVT::i32);
  SDValue SubReg2 = CurDAG->getTargetConstant(ARM::qsub_2, MVT::i32);
  SDValue SubReg3 = CurDAG->getTargetConstant(ARM::qsub_3, MVT::i32);
  const SDValue Ops[] = { V0, SubReg0, V1, SubReg1, V2, SubReg2, V3, SubReg3 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 8);
}

/// OctoDRegs - Form 8 consecutive D registers.
///
SDNode *ARMDAGToDAGISel::OctoDRegs(EVT VT, SDValue V0, SDValue V1,
                                   SDValue V2, SDValue V3,
                                   SDValue V4, SDValue V5,
                                   SDValue V6, SDValue V7) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::dsub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::dsub_1, MVT::i32);
  SDValue SubReg2 = CurDAG->getTargetConstant(ARM::dsub_2, MVT::i32);
  SDValue SubReg3 = CurDAG->getTargetConstant(ARM::dsub_3, MVT::i32);
  SDValue SubReg4 = CurDAG->getTargetConstant(ARM::dsub_4, MVT::i32);
  SDValue SubReg5 = CurDAG->getTargetConstant(ARM::dsub_5, MVT::i32);
  SDValue SubReg6 = CurDAG->getTargetConstant(ARM::dsub_6, MVT::i32);
  SDValue SubReg7 = CurDAG->getTargetConstant(ARM::dsub_7, MVT::i32);
  const SDValue Ops[] ={ V0, SubReg0, V1, SubReg1, V2, SubReg2, V3, SubReg3,
                         V4, SubReg4, V5, SubReg5, V6, SubReg6, V7, SubReg7 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 16);
}

/// GetNEONSubregVT - Given a type for a 128-bit NEON vector, return the type
/// for a 64-bit subregister of the vector.
static EVT GetNEONSubregVT(EVT VT) {
  switch (VT.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("unhandled NEON type");
  case MVT::v16i8: return MVT::v8i8;
  case MVT::v8i16: return MVT::v4i16;
  case MVT::v4f32: return MVT::v2f32;
  case MVT::v4i32: return MVT::v2i32;
  case MVT::v2i64: return MVT::v1i64;
  }
}

SDNode *ARMDAGToDAGISel::SelectVLD(SDNode *N, unsigned NumVecs,
                                   unsigned *DOpcodes, unsigned *QOpcodes0,
                                   unsigned *QOpcodes1) {
  assert(NumVecs >= 1 && NumVecs <= 4 && "VLD NumVecs out-of-range");
  DebugLoc dl = N->getDebugLoc();

  SDValue MemAddr, Align;
  if (!SelectAddrMode6(N, N->getOperand(2), MemAddr, Align))
    return NULL;

  SDValue Chain = N->getOperand(0);
  EVT VT = N->getValueType(0);
  bool is64BitVector = VT.is64BitVector();

  unsigned OpcodeIndex;
  switch (VT.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("unhandled vld type");
    // Double-register operations:
  case MVT::v8i8:  OpcodeIndex = 0; break;
  case MVT::v4i16: OpcodeIndex = 1; break;
  case MVT::v2f32:
  case MVT::v2i32: OpcodeIndex = 2; break;
  case MVT::v1i64: OpcodeIndex = 3; break;
    // Quad-register operations:
  case MVT::v16i8: OpcodeIndex = 0; break;
  case MVT::v8i16: OpcodeIndex = 1; break;
  case MVT::v4f32:
  case MVT::v4i32: OpcodeIndex = 2; break;
  case MVT::v2i64: OpcodeIndex = 3;
    assert(NumVecs == 1 && "v2i64 type only supported for VLD1");
    break;
  }

  EVT ResTy;
  if (NumVecs == 1)
    ResTy = VT;
  else {
    unsigned ResTyElts = (NumVecs == 3) ? 4 : NumVecs;
    if (!is64BitVector)
      ResTyElts *= 2;
    ResTy = EVT::getVectorVT(*CurDAG->getContext(), MVT::i64, ResTyElts);
  }

  SDValue Pred = getAL(CurDAG);
  SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
  SDValue SuperReg;
  if (is64BitVector) {
    unsigned Opc = DOpcodes[OpcodeIndex];
    const SDValue Ops[] = { MemAddr, Align, Pred, Reg0, Chain };
    SDNode *VLd = CurDAG->getMachineNode(Opc, dl, ResTy, MVT::Other, Ops, 5);
    if (NumVecs == 1)
      return VLd;

    SuperReg = SDValue(VLd, 0);
    assert(ARM::dsub_7 == ARM::dsub_0+7 && "Unexpected subreg numbering");
    for (unsigned Vec = 0; Vec < NumVecs; ++Vec) {
      SDValue D = CurDAG->getTargetExtractSubreg(ARM::dsub_0+Vec,
                                                 dl, VT, SuperReg);
      ReplaceUses(SDValue(N, Vec), D);
    }
    ReplaceUses(SDValue(N, NumVecs), SDValue(VLd, 1));
    return NULL;
  }

  if (NumVecs <= 2) {
    // Quad registers are directly supported for VLD1 and VLD2,
    // loading pairs of D regs.
    unsigned Opc = QOpcodes0[OpcodeIndex];
    const SDValue Ops[] = { MemAddr, Align, Pred, Reg0, Chain };
    SDNode *VLd = CurDAG->getMachineNode(Opc, dl, ResTy, MVT::Other, Ops, 5);
    if (NumVecs == 1)
      return VLd;

    SuperReg = SDValue(VLd, 0);
    Chain = SDValue(VLd, 1);

  } else {
    // Otherwise, quad registers are loaded with two separate instructions,
    // where one loads the even registers and the other loads the odd registers.
    EVT AddrTy = MemAddr.getValueType();

    // Load the even subregs.
    unsigned Opc = QOpcodes0[OpcodeIndex];
    SDValue ImplDef =
      SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, dl, ResTy), 0);
    const SDValue OpsA[] = { MemAddr, Align, Reg0, ImplDef, Pred, Reg0, Chain };
    SDNode *VLdA =
      CurDAG->getMachineNode(Opc, dl, ResTy, AddrTy, MVT::Other, OpsA, 7);
    Chain = SDValue(VLdA, 2);

    // Load the odd subregs.
    Opc = QOpcodes1[OpcodeIndex];
    const SDValue OpsB[] = { SDValue(VLdA, 1), Align, Reg0, SDValue(VLdA, 0),
                             Pred, Reg0, Chain };
    SDNode *VLdB =
      CurDAG->getMachineNode(Opc, dl, ResTy, AddrTy, MVT::Other, OpsB, 7);
    SuperReg = SDValue(VLdB, 0);
    Chain = SDValue(VLdB, 2);
  }

  // Extract out the Q registers.
  assert(ARM::qsub_3 == ARM::qsub_0+3 && "Unexpected subreg numbering");
  for (unsigned Vec = 0; Vec < NumVecs; ++Vec) {
    SDValue Q = CurDAG->getTargetExtractSubreg(ARM::qsub_0+Vec,
                                               dl, VT, SuperReg);
    ReplaceUses(SDValue(N, Vec), Q);
  }
  ReplaceUses(SDValue(N, NumVecs), Chain);
  return NULL;
}

SDNode *ARMDAGToDAGISel::SelectVST(SDNode *N, unsigned NumVecs,
                                   unsigned *DOpcodes, unsigned *QOpcodes0,
                                   unsigned *QOpcodes1) {
  assert(NumVecs >= 1 && NumVecs <= 4 && "VST NumVecs out-of-range");
  DebugLoc dl = N->getDebugLoc();

  SDValue MemAddr, Align;
  if (!SelectAddrMode6(N, N->getOperand(2), MemAddr, Align))
    return NULL;

  SDValue Chain = N->getOperand(0);
  EVT VT = N->getOperand(3).getValueType();
  bool is64BitVector = VT.is64BitVector();

  unsigned OpcodeIndex;
  switch (VT.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("unhandled vst type");
    // Double-register operations:
  case MVT::v8i8:  OpcodeIndex = 0; break;
  case MVT::v4i16: OpcodeIndex = 1; break;
  case MVT::v2f32:
  case MVT::v2i32: OpcodeIndex = 2; break;
  case MVT::v1i64: OpcodeIndex = 3; break;
    // Quad-register operations:
  case MVT::v16i8: OpcodeIndex = 0; break;
  case MVT::v8i16: OpcodeIndex = 1; break;
  case MVT::v4f32:
  case MVT::v4i32: OpcodeIndex = 2; break;
  case MVT::v2i64: OpcodeIndex = 3;
    assert(NumVecs == 1 && "v2i64 type only supported for VST1");
    break;
  }

  SDValue Pred = getAL(CurDAG);
  SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);

  SmallVector<SDValue, 7> Ops;
  Ops.push_back(MemAddr);
  Ops.push_back(Align);

  if (is64BitVector) {
    if (NumVecs == 1) {
      Ops.push_back(N->getOperand(3));
    } else {
      SDValue RegSeq;
      SDValue V0 = N->getOperand(0+3);
      SDValue V1 = N->getOperand(1+3);

      // Form a REG_SEQUENCE to force register allocation.
      if (NumVecs == 2)
        RegSeq = SDValue(PairDRegs(MVT::v2i64, V0, V1), 0);
      else {
        SDValue V2 = N->getOperand(2+3);
        // If it's a vld3, form a quad D-register and leave the last part as 
        // an undef.
        SDValue V3 = (NumVecs == 3)
          ? SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,dl,VT), 0)
          : N->getOperand(3+3);
        RegSeq = SDValue(QuadDRegs(MVT::v4i64, V0, V1, V2, V3), 0);
      }
      Ops.push_back(RegSeq);
    }
    Ops.push_back(Pred);
    Ops.push_back(Reg0); // predicate register
    Ops.push_back(Chain);
    unsigned Opc = DOpcodes[OpcodeIndex];
    return CurDAG->getMachineNode(Opc, dl, MVT::Other, Ops.data(), 6);
  }

  if (NumVecs <= 2) {
    // Quad registers are directly supported for VST1 and VST2.
    unsigned Opc = QOpcodes0[OpcodeIndex];
    if (NumVecs == 1) {
      Ops.push_back(N->getOperand(3));
    } else {
      // Form a QQ register.
      SDValue Q0 = N->getOperand(3);
      SDValue Q1 = N->getOperand(4);
      Ops.push_back(SDValue(PairQRegs(MVT::v4i64, Q0, Q1), 0));
    }
    Ops.push_back(Pred);
    Ops.push_back(Reg0); // predicate register
    Ops.push_back(Chain);
    return CurDAG->getMachineNode(Opc, dl, MVT::Other, Ops.data(), 6);
  }

  // Otherwise, quad registers are stored with two separate instructions,
  // where one stores the even registers and the other stores the odd registers.

  // Form the QQQQ REG_SEQUENCE.
  SDValue V0 = N->getOperand(0+3);
  SDValue V1 = N->getOperand(1+3);
  SDValue V2 = N->getOperand(2+3);
  SDValue V3 = (NumVecs == 3)
    ? SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, dl, VT), 0)
    : N->getOperand(3+3);
  SDValue RegSeq = SDValue(QuadQRegs(MVT::v8i64, V0, V1, V2, V3), 0);

  // Store the even D registers.
  Ops.push_back(Reg0); // post-access address offset
  Ops.push_back(RegSeq);
  Ops.push_back(Pred);
  Ops.push_back(Reg0); // predicate register
  Ops.push_back(Chain);
  unsigned Opc = QOpcodes0[OpcodeIndex];
  SDNode *VStA = CurDAG->getMachineNode(Opc, dl, MemAddr.getValueType(),
                                        MVT::Other, Ops.data(), 7);
  Chain = SDValue(VStA, 1);

  // Store the odd D registers.
  Ops[0] = SDValue(VStA, 0); // MemAddr
  Ops[6] = Chain;
  Opc = QOpcodes1[OpcodeIndex];
  SDNode *VStB = CurDAG->getMachineNode(Opc, dl, MemAddr.getValueType(),
                                        MVT::Other, Ops.data(), 7);
  Chain = SDValue(VStB, 1);
  ReplaceUses(SDValue(N, 0), Chain);
  return NULL;
}

SDNode *ARMDAGToDAGISel::SelectVLDSTLane(SDNode *N, bool IsLoad,
                                         unsigned NumVecs, unsigned *DOpcodes,
                                         unsigned *QOpcodes0,
                                         unsigned *QOpcodes1) {
  assert(NumVecs >=2 && NumVecs <= 4 && "VLDSTLane NumVecs out-of-range");
  DebugLoc dl = N->getDebugLoc();

  SDValue MemAddr, Align;
  if (!SelectAddrMode6(N, N->getOperand(2), MemAddr, Align))
    return NULL;

  SDValue Chain = N->getOperand(0);
  unsigned Lane =
    cast<ConstantSDNode>(N->getOperand(NumVecs+3))->getZExtValue();
  EVT VT = IsLoad ? N->getValueType(0) : N->getOperand(3).getValueType();
  bool is64BitVector = VT.is64BitVector();

  // Quad registers are handled by load/store of subregs. Find the subreg info.
  unsigned NumElts = 0;
  bool Even = false;
  EVT RegVT = VT;
  if (!is64BitVector) {
    RegVT = GetNEONSubregVT(VT);
    NumElts = RegVT.getVectorNumElements();
    Even = Lane < NumElts;
  }

  unsigned OpcodeIndex;
  switch (VT.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("unhandled vld/vst lane type");
    // Double-register operations:
  case MVT::v8i8:  OpcodeIndex = 0; break;
  case MVT::v4i16: OpcodeIndex = 1; break;
  case MVT::v2f32:
  case MVT::v2i32: OpcodeIndex = 2; break;
    // Quad-register operations:
  case MVT::v8i16: OpcodeIndex = 0; break;
  case MVT::v4f32:
  case MVT::v4i32: OpcodeIndex = 1; break;
  }

  SDValue Pred = getAL(CurDAG);
  SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);

  SmallVector<SDValue, 10> Ops;
  Ops.push_back(MemAddr);
  Ops.push_back(Align);

  unsigned Opc = 0;
  if (is64BitVector) {
    Opc = DOpcodes[OpcodeIndex];
    SDValue RegSeq;
    SDValue V0 = N->getOperand(0+3);
    SDValue V1 = N->getOperand(1+3);
    if (NumVecs == 2) {
      RegSeq = SDValue(PairDRegs(MVT::v2i64, V0, V1), 0);
    } else {
      SDValue V2 = N->getOperand(2+3);
      SDValue V3 = (NumVecs == 3)
        ? SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,dl,VT), 0)
        : N->getOperand(3+3);
      RegSeq = SDValue(QuadDRegs(MVT::v4i64, V0, V1, V2, V3), 0);
    }

    // Now extract the D registers back out.
    Ops.push_back(CurDAG->getTargetExtractSubreg(ARM::dsub_0, dl, VT, RegSeq));
    Ops.push_back(CurDAG->getTargetExtractSubreg(ARM::dsub_1, dl, VT, RegSeq));
    if (NumVecs > 2)
      Ops.push_back(CurDAG->getTargetExtractSubreg(ARM::dsub_2, dl, VT,RegSeq));
    if (NumVecs > 3)
      Ops.push_back(CurDAG->getTargetExtractSubreg(ARM::dsub_3, dl, VT,RegSeq));
  } else {
    // Check if this is loading the even or odd subreg of a Q register.
    if (Lane < NumElts) {
      Opc = QOpcodes0[OpcodeIndex];
    } else {
      Lane -= NumElts;
      Opc = QOpcodes1[OpcodeIndex];
    }

    SDValue RegSeq;
    SDValue V0 = N->getOperand(0+3);
    SDValue V1 = N->getOperand(1+3);
    if (NumVecs == 2) {
      RegSeq = SDValue(PairQRegs(MVT::v4i64, V0, V1), 0);
    } else {
      SDValue V2 = N->getOperand(2+3);
      SDValue V3 = (NumVecs == 3)
        ? SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,dl,VT), 0)
        : N->getOperand(3+3);
      RegSeq = SDValue(QuadQRegs(MVT::v8i64, V0, V1, V2, V3), 0);
    }

    // Extract the subregs of the input vector.
    unsigned SubIdx = Even ? ARM::dsub_0 : ARM::dsub_1;
    for (unsigned Vec = 0; Vec < NumVecs; ++Vec)
      Ops.push_back(CurDAG->getTargetExtractSubreg(SubIdx+Vec*2, dl, RegVT,
                                                   RegSeq));
  }
  Ops.push_back(getI32Imm(Lane));
  Ops.push_back(Pred);
  Ops.push_back(Reg0);
  Ops.push_back(Chain);

  if (!IsLoad)
    return CurDAG->getMachineNode(Opc, dl, MVT::Other, Ops.data(), NumVecs+6);

  std::vector<EVT> ResTys(NumVecs, RegVT);
  ResTys.push_back(MVT::Other);
  SDNode *VLdLn = CurDAG->getMachineNode(Opc, dl, ResTys, Ops.data(),NumVecs+6);

  // Form a REG_SEQUENCE to force register allocation.
  SDValue RegSeq;
  if (is64BitVector) {
    SDValue V0 = SDValue(VLdLn, 0);
    SDValue V1 = SDValue(VLdLn, 1);
    if (NumVecs == 2) {
      RegSeq = SDValue(PairDRegs(MVT::v2i64, V0, V1), 0);
    } else {
      SDValue V2 = SDValue(VLdLn, 2);
      // If it's a vld3, form a quad D-register but discard the last part.
      SDValue V3 = (NumVecs == 3)
        ? SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,dl,VT), 0)
        : SDValue(VLdLn, 3);
      RegSeq = SDValue(QuadDRegs(MVT::v4i64, V0, V1, V2, V3), 0);
    }
  } else {
    // For 128-bit vectors, take the 64-bit results of the load and insert
    // them as subregs into the result.
    SDValue V[8];
    for (unsigned Vec = 0, i = 0; Vec < NumVecs; ++Vec, i+=2) {
      if (Even) {
        V[i]   = SDValue(VLdLn, Vec);
        V[i+1] = SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,
                                                dl, RegVT), 0);
      } else {
        V[i]   = SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,
                                                dl, RegVT), 0);
        V[i+1] = SDValue(VLdLn, Vec);
      }
    }
    if (NumVecs == 3)
      V[6] = V[7] = SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,
                                                   dl, RegVT), 0);

    if (NumVecs == 2)
      RegSeq = SDValue(QuadDRegs(MVT::v4i64, V[0], V[1], V[2], V[3]), 0);
    else
      RegSeq = SDValue(OctoDRegs(MVT::v8i64, V[0], V[1], V[2], V[3],
                                 V[4], V[5], V[6], V[7]), 0);
  }

  assert(ARM::dsub_7 == ARM::dsub_0+7 && "Unexpected subreg numbering");
  assert(ARM::qsub_3 == ARM::qsub_0+3 && "Unexpected subreg numbering");
  unsigned SubIdx = is64BitVector ? ARM::dsub_0 : ARM::qsub_0;
  for (unsigned Vec = 0; Vec < NumVecs; ++Vec)
    ReplaceUses(SDValue(N, Vec),
                CurDAG->getTargetExtractSubreg(SubIdx+Vec, dl, VT, RegSeq));
  ReplaceUses(SDValue(N, NumVecs), SDValue(VLdLn, NumVecs));
  return NULL;
}

SDNode *ARMDAGToDAGISel::SelectVTBL(SDNode *N, bool IsExt, unsigned NumVecs,
                                    unsigned Opc) {
  assert(NumVecs >= 2 && NumVecs <= 4 && "VTBL NumVecs out-of-range");
  DebugLoc dl = N->getDebugLoc();
  EVT VT = N->getValueType(0);
  unsigned FirstTblReg = IsExt ? 2 : 1;

  // Form a REG_SEQUENCE to force register allocation.
  SDValue RegSeq;
  SDValue V0 = N->getOperand(FirstTblReg + 0);
  SDValue V1 = N->getOperand(FirstTblReg + 1);
  if (NumVecs == 2)
    RegSeq = SDValue(PairDRegs(MVT::v16i8, V0, V1), 0);
  else {
    SDValue V2 = N->getOperand(FirstTblReg + 2);
    // If it's a vtbl3, form a quad D-register and leave the last part as 
    // an undef.
    SDValue V3 = (NumVecs == 3)
      ? SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, dl, VT), 0)
      : N->getOperand(FirstTblReg + 3);
    RegSeq = SDValue(QuadDRegs(MVT::v4i64, V0, V1, V2, V3), 0);
  }

  // Now extract the D registers back out.
  SmallVector<SDValue, 6> Ops;
  if (IsExt)
    Ops.push_back(N->getOperand(1));
  Ops.push_back(CurDAG->getTargetExtractSubreg(ARM::dsub_0, dl, VT, RegSeq));
  Ops.push_back(CurDAG->getTargetExtractSubreg(ARM::dsub_1, dl, VT, RegSeq));
  if (NumVecs > 2)
    Ops.push_back(CurDAG->getTargetExtractSubreg(ARM::dsub_2, dl, VT, RegSeq));
  if (NumVecs > 3)
    Ops.push_back(CurDAG->getTargetExtractSubreg(ARM::dsub_3, dl, VT, RegSeq));

  Ops.push_back(N->getOperand(FirstTblReg + NumVecs));
  Ops.push_back(getAL(CurDAG)); // predicate
  Ops.push_back(CurDAG->getRegister(0, MVT::i32)); // predicate register
  return CurDAG->getMachineNode(Opc, dl, VT, Ops.data(), Ops.size());
}

SDNode *ARMDAGToDAGISel::SelectV6T2BitfieldExtractOp(SDNode *N,
                                                     bool isSigned) {
  if (!Subtarget->hasV6T2Ops())
    return NULL;

  unsigned Opc = isSigned ? (Subtarget->isThumb() ? ARM::t2SBFX : ARM::SBFX)
    : (Subtarget->isThumb() ? ARM::t2UBFX : ARM::UBFX);


  // For unsigned extracts, check for a shift right and mask
  unsigned And_imm = 0;
  if (N->getOpcode() == ISD::AND) {
    if (isOpcWithIntImmediate(N, ISD::AND, And_imm)) {

      // The immediate is a mask of the low bits iff imm & (imm+1) == 0
      if (And_imm & (And_imm + 1))
        return NULL;

      unsigned Srl_imm = 0;
      if (isOpcWithIntImmediate(N->getOperand(0).getNode(), ISD::SRL,
                                Srl_imm)) {
        assert(Srl_imm > 0 && Srl_imm < 32 && "bad amount in shift node!");

        unsigned Width = CountTrailingOnes_32(And_imm);
        unsigned LSB = Srl_imm;
        SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
        SDValue Ops[] = { N->getOperand(0).getOperand(0),
                          CurDAG->getTargetConstant(LSB, MVT::i32),
                          CurDAG->getTargetConstant(Width, MVT::i32),
          getAL(CurDAG), Reg0 };
        return CurDAG->SelectNodeTo(N, Opc, MVT::i32, Ops, 5);
      }
    }
    return NULL;
  }

  // Otherwise, we're looking for a shift of a shift
  unsigned Shl_imm = 0;
  if (isOpcWithIntImmediate(N->getOperand(0).getNode(), ISD::SHL, Shl_imm)) {
    assert(Shl_imm > 0 && Shl_imm < 32 && "bad amount in shift node!");
    unsigned Srl_imm = 0;
    if (isInt32Immediate(N->getOperand(1), Srl_imm)) {
      assert(Srl_imm > 0 && Srl_imm < 32 && "bad amount in shift node!");
      unsigned Width = 32 - Srl_imm;
      int LSB = Srl_imm - Shl_imm;
      if (LSB < 0)
        return NULL;
      SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
      SDValue Ops[] = { N->getOperand(0).getOperand(0),
                        CurDAG->getTargetConstant(LSB, MVT::i32),
                        CurDAG->getTargetConstant(Width, MVT::i32),
                        getAL(CurDAG), Reg0 };
      return CurDAG->SelectNodeTo(N, Opc, MVT::i32, Ops, 5);
    }
  }
  return NULL;
}

SDNode *ARMDAGToDAGISel::
SelectT2CMOVShiftOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                    ARMCC::CondCodes CCVal, SDValue CCR, SDValue InFlag) {
  SDValue CPTmp0;
  SDValue CPTmp1;
  if (SelectT2ShifterOperandReg(N, TrueVal, CPTmp0, CPTmp1)) {
    unsigned SOVal = cast<ConstantSDNode>(CPTmp1)->getZExtValue();
    unsigned SOShOp = ARM_AM::getSORegShOp(SOVal);
    unsigned Opc = 0;
    switch (SOShOp) {
    case ARM_AM::lsl: Opc = ARM::t2MOVCClsl; break;
    case ARM_AM::lsr: Opc = ARM::t2MOVCClsr; break;
    case ARM_AM::asr: Opc = ARM::t2MOVCCasr; break;
    case ARM_AM::ror: Opc = ARM::t2MOVCCror; break;
    default:
      llvm_unreachable("Unknown so_reg opcode!");
      break;
    }
    SDValue SOShImm =
      CurDAG->getTargetConstant(ARM_AM::getSORegOffset(SOVal), MVT::i32);
    SDValue CC = CurDAG->getTargetConstant(CCVal, MVT::i32);
    SDValue Ops[] = { FalseVal, CPTmp0, SOShImm, CC, CCR, InFlag };
    return CurDAG->SelectNodeTo(N, Opc, MVT::i32,Ops, 6);
  }
  return 0;
}

SDNode *ARMDAGToDAGISel::
SelectARMCMOVShiftOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                     ARMCC::CondCodes CCVal, SDValue CCR, SDValue InFlag) {
  SDValue CPTmp0;
  SDValue CPTmp1;
  SDValue CPTmp2;
  if (SelectShifterOperandReg(N, TrueVal, CPTmp0, CPTmp1, CPTmp2)) {
    SDValue CC = CurDAG->getTargetConstant(CCVal, MVT::i32);
    SDValue Ops[] = { FalseVal, CPTmp0, CPTmp1, CPTmp2, CC, CCR, InFlag };
    return CurDAG->SelectNodeTo(N, ARM::MOVCCs, MVT::i32, Ops, 7);
  }
  return 0;
}

SDNode *ARMDAGToDAGISel::
SelectT2CMOVSoImmOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                    ARMCC::CondCodes CCVal, SDValue CCR, SDValue InFlag) {
  ConstantSDNode *T = dyn_cast<ConstantSDNode>(TrueVal);
  if (!T)
    return 0;

  if (Pred_t2_so_imm(TrueVal.getNode())) {
    SDValue True = CurDAG->getTargetConstant(T->getZExtValue(), MVT::i32);
    SDValue CC = CurDAG->getTargetConstant(CCVal, MVT::i32);
    SDValue Ops[] = { FalseVal, True, CC, CCR, InFlag };
    return CurDAG->SelectNodeTo(N,
                                ARM::t2MOVCCi, MVT::i32, Ops, 5);
  }
  return 0;
}

SDNode *ARMDAGToDAGISel::
SelectARMCMOVSoImmOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                     ARMCC::CondCodes CCVal, SDValue CCR, SDValue InFlag) {
  ConstantSDNode *T = dyn_cast<ConstantSDNode>(TrueVal);
  if (!T)
    return 0;

  if (Pred_so_imm(TrueVal.getNode())) {
    SDValue True = CurDAG->getTargetConstant(T->getZExtValue(), MVT::i32);
    SDValue CC = CurDAG->getTargetConstant(CCVal, MVT::i32);
    SDValue Ops[] = { FalseVal, True, CC, CCR, InFlag };
    return CurDAG->SelectNodeTo(N,
                                ARM::MOVCCi, MVT::i32, Ops, 5);
  }
  return 0;
}

SDNode *ARMDAGToDAGISel::SelectCMOVOp(SDNode *N) {
  EVT VT = N->getValueType(0);
  SDValue FalseVal = N->getOperand(0);
  SDValue TrueVal  = N->getOperand(1);
  SDValue CC = N->getOperand(2);
  SDValue CCR = N->getOperand(3);
  SDValue InFlag = N->getOperand(4);
  assert(CC.getOpcode() == ISD::Constant);
  assert(CCR.getOpcode() == ISD::Register);
  ARMCC::CondCodes CCVal =
    (ARMCC::CondCodes)cast<ConstantSDNode>(CC)->getZExtValue();

  if (!Subtarget->isThumb1Only() && VT == MVT::i32) {
    // Pattern: (ARMcmov:i32 GPR:i32:$false, so_reg:i32:$true, (imm:i32):$cc)
    // Emits: (MOVCCs:i32 GPR:i32:$false, so_reg:i32:$true, (imm:i32):$cc)
    // Pattern complexity = 18  cost = 1  size = 0
    SDValue CPTmp0;
    SDValue CPTmp1;
    SDValue CPTmp2;
    if (Subtarget->isThumb()) {
      SDNode *Res = SelectT2CMOVShiftOp(N, FalseVal, TrueVal,
                                        CCVal, CCR, InFlag);
      if (!Res)
        Res = SelectT2CMOVShiftOp(N, TrueVal, FalseVal,
                               ARMCC::getOppositeCondition(CCVal), CCR, InFlag);
      if (Res)
        return Res;
    } else {
      SDNode *Res = SelectARMCMOVShiftOp(N, FalseVal, TrueVal,
                                         CCVal, CCR, InFlag);
      if (!Res)
        Res = SelectARMCMOVShiftOp(N, TrueVal, FalseVal,
                               ARMCC::getOppositeCondition(CCVal), CCR, InFlag);
      if (Res)
        return Res;
    }

    // Pattern: (ARMcmov:i32 GPR:i32:$false,
    //             (imm:i32)<<P:Pred_so_imm>>:$true,
    //             (imm:i32):$cc)
    // Emits: (MOVCCi:i32 GPR:i32:$false,
    //           (so_imm:i32 (imm:i32):$true), (imm:i32):$cc)
    // Pattern complexity = 10  cost = 1  size = 0
    if (Subtarget->isThumb()) {
      SDNode *Res = SelectT2CMOVSoImmOp(N, FalseVal, TrueVal,
                                        CCVal, CCR, InFlag);
      if (!Res)
        Res = SelectT2CMOVSoImmOp(N, TrueVal, FalseVal,
                               ARMCC::getOppositeCondition(CCVal), CCR, InFlag);
      if (Res)
        return Res;
    } else {
      SDNode *Res = SelectARMCMOVSoImmOp(N, FalseVal, TrueVal,
                                         CCVal, CCR, InFlag);
      if (!Res)
        Res = SelectARMCMOVSoImmOp(N, TrueVal, FalseVal,
                               ARMCC::getOppositeCondition(CCVal), CCR, InFlag);
      if (Res)
        return Res;
    }
  }

  // Pattern: (ARMcmov:i32 GPR:i32:$false, GPR:i32:$true, (imm:i32):$cc)
  // Emits: (MOVCCr:i32 GPR:i32:$false, GPR:i32:$true, (imm:i32):$cc)
  // Pattern complexity = 6  cost = 1  size = 0
  //
  // Pattern: (ARMcmov:i32 GPR:i32:$false, GPR:i32:$true, (imm:i32):$cc)
  // Emits: (tMOVCCr:i32 GPR:i32:$false, GPR:i32:$true, (imm:i32):$cc)
  // Pattern complexity = 6  cost = 11  size = 0
  //
  // Also FCPYScc and FCPYDcc.
  SDValue Tmp2 = CurDAG->getTargetConstant(CCVal, MVT::i32);
  SDValue Ops[] = { FalseVal, TrueVal, Tmp2, CCR, InFlag };
  unsigned Opc = 0;
  switch (VT.getSimpleVT().SimpleTy) {
  default: assert(false && "Illegal conditional move type!");
    break;
  case MVT::i32:
    Opc = Subtarget->isThumb()
      ? (Subtarget->hasThumb2() ? ARM::t2MOVCCr : ARM::tMOVCCr_pseudo)
      : ARM::MOVCCr;
    break;
  case MVT::f32:
    Opc = ARM::VMOVScc;
    break;
  case MVT::f64:
    Opc = ARM::VMOVDcc;
    break;
  }
  return CurDAG->SelectNodeTo(N, Opc, VT, Ops, 5);
}

SDNode *ARMDAGToDAGISel::SelectConcatVector(SDNode *N) {
  // The only time a CONCAT_VECTORS operation can have legal types is when
  // two 64-bit vectors are concatenated to a 128-bit vector.
  EVT VT = N->getValueType(0);
  if (!VT.is128BitVector() || N->getNumOperands() != 2)
    llvm_unreachable("unexpected CONCAT_VECTORS");
  DebugLoc dl = N->getDebugLoc();
  SDValue V0 = N->getOperand(0);
  SDValue V1 = N->getOperand(1);
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::dsub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::dsub_1, MVT::i32);
  const SDValue Ops[] = { V0, SubReg0, V1, SubReg1 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 4);
}

SDNode *ARMDAGToDAGISel::Select(SDNode *N) {
  DebugLoc dl = N->getDebugLoc();

  if (N->isMachineOpcode())
    return NULL;   // Already selected.

  switch (N->getOpcode()) {
  default: break;
  case ISD::Constant: {
    unsigned Val = cast<ConstantSDNode>(N)->getZExtValue();
    bool UseCP = true;
    if (Subtarget->hasThumb2())
      // Thumb2-aware targets have the MOVT instruction, so all immediates can
      // be done with MOV + MOVT, at worst.
      UseCP = 0;
    else {
      if (Subtarget->isThumb()) {
        UseCP = (Val > 255 &&                          // MOV
                 ~Val > 255 &&                         // MOV + MVN
                 !ARM_AM::isThumbImmShiftedVal(Val));  // MOV + LSL
      } else
        UseCP = (ARM_AM::getSOImmVal(Val) == -1 &&     // MOV
                 ARM_AM::getSOImmVal(~Val) == -1 &&    // MVN
                 !ARM_AM::isSOImmTwoPartVal(Val));     // two instrs.
    }

    if (UseCP) {
      SDValue CPIdx =
        CurDAG->getTargetConstantPool(ConstantInt::get(
                                  Type::getInt32Ty(*CurDAG->getContext()), Val),
                                      TLI.getPointerTy());

      SDNode *ResNode;
      if (Subtarget->isThumb1Only()) {
        SDValue Pred = getAL(CurDAG);
        SDValue PredReg = CurDAG->getRegister(0, MVT::i32);
        SDValue Ops[] = { CPIdx, Pred, PredReg, CurDAG->getEntryNode() };
        ResNode = CurDAG->getMachineNode(ARM::tLDRcp, dl, MVT::i32, MVT::Other,
                                         Ops, 4);
      } else {
        SDValue Ops[] = {
          CPIdx,
          CurDAG->getRegister(0, MVT::i32),
          CurDAG->getTargetConstant(0, MVT::i32),
          getAL(CurDAG),
          CurDAG->getRegister(0, MVT::i32),
          CurDAG->getEntryNode()
        };
        ResNode=CurDAG->getMachineNode(ARM::LDRcp, dl, MVT::i32, MVT::Other,
                                       Ops, 6);
      }
      ReplaceUses(SDValue(N, 0), SDValue(ResNode, 0));
      return NULL;
    }

    // Other cases are autogenerated.
    break;
  }
  case ISD::FrameIndex: {
    // Selects to ADDri FI, 0 which in turn will become ADDri SP, imm.
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    if (Subtarget->isThumb1Only()) {
      return CurDAG->SelectNodeTo(N, ARM::tADDrSPi, MVT::i32, TFI,
                                  CurDAG->getTargetConstant(0, MVT::i32));
    } else {
      unsigned Opc = ((Subtarget->isThumb() && Subtarget->hasThumb2()) ?
                      ARM::t2ADDri : ARM::ADDri);
      SDValue Ops[] = { TFI, CurDAG->getTargetConstant(0, MVT::i32),
                        getAL(CurDAG), CurDAG->getRegister(0, MVT::i32),
                        CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->SelectNodeTo(N, Opc, MVT::i32, Ops, 5);
    }
  }
  case ISD::SRL:
    if (SDNode *I = SelectV6T2BitfieldExtractOp(N, false))
      return I;
    break;
  case ISD::SRA:
    if (SDNode *I = SelectV6T2BitfieldExtractOp(N, true))
      return I;
    break;
  case ISD::MUL:
    if (Subtarget->isThumb1Only())
      break;
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(1))) {
      unsigned RHSV = C->getZExtValue();
      if (!RHSV) break;
      if (isPowerOf2_32(RHSV-1)) {  // 2^n+1?
        unsigned ShImm = Log2_32(RHSV-1);
        if (ShImm >= 32)
          break;
        SDValue V = N->getOperand(0);
        ShImm = ARM_AM::getSORegOpc(ARM_AM::lsl, ShImm);
        SDValue ShImmOp = CurDAG->getTargetConstant(ShImm, MVT::i32);
        SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
        if (Subtarget->isThumb()) {
          SDValue Ops[] = { V, V, ShImmOp, getAL(CurDAG), Reg0, Reg0 };
          return CurDAG->SelectNodeTo(N, ARM::t2ADDrs, MVT::i32, Ops, 6);
        } else {
          SDValue Ops[] = { V, V, Reg0, ShImmOp, getAL(CurDAG), Reg0, Reg0 };
          return CurDAG->SelectNodeTo(N, ARM::ADDrs, MVT::i32, Ops, 7);
        }
      }
      if (isPowerOf2_32(RHSV+1)) {  // 2^n-1?
        unsigned ShImm = Log2_32(RHSV+1);
        if (ShImm >= 32)
          break;
        SDValue V = N->getOperand(0);
        ShImm = ARM_AM::getSORegOpc(ARM_AM::lsl, ShImm);
        SDValue ShImmOp = CurDAG->getTargetConstant(ShImm, MVT::i32);
        SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
        if (Subtarget->isThumb()) {
          SDValue Ops[] = { V, V, ShImmOp, getAL(CurDAG), Reg0, Reg0 };
          return CurDAG->SelectNodeTo(N, ARM::t2RSBrs, MVT::i32, Ops, 6);
        } else {
          SDValue Ops[] = { V, V, Reg0, ShImmOp, getAL(CurDAG), Reg0, Reg0 };
          return CurDAG->SelectNodeTo(N, ARM::RSBrs, MVT::i32, Ops, 7);
        }
      }
    }
    break;
  case ISD::AND: {
    // Check for unsigned bitfield extract
    if (SDNode *I = SelectV6T2BitfieldExtractOp(N, false))
      return I;

    // (and (or x, c2), c1) and top 16-bits of c1 and c2 match, lower 16-bits
    // of c1 are 0xffff, and lower 16-bit of c2 are 0. That is, the top 16-bits
    // are entirely contributed by c2 and lower 16-bits are entirely contributed
    // by x. That's equal to (or (and x, 0xffff), (and c1, 0xffff0000)).
    // Select it to: "movt x, ((c1 & 0xffff) >> 16)
    EVT VT = N->getValueType(0);
    if (VT != MVT::i32)
      break;
    unsigned Opc = (Subtarget->isThumb() && Subtarget->hasThumb2())
      ? ARM::t2MOVTi16
      : (Subtarget->hasV6T2Ops() ? ARM::MOVTi16 : 0);
    if (!Opc)
      break;
    SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);
    ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
    if (!N1C)
      break;
    if (N0.getOpcode() == ISD::OR && N0.getNode()->hasOneUse()) {
      SDValue N2 = N0.getOperand(1);
      ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2);
      if (!N2C)
        break;
      unsigned N1CVal = N1C->getZExtValue();
      unsigned N2CVal = N2C->getZExtValue();
      if ((N1CVal & 0xffff0000U) == (N2CVal & 0xffff0000U) &&
          (N1CVal & 0xffffU) == 0xffffU &&
          (N2CVal & 0xffffU) == 0x0U) {
        SDValue Imm16 = CurDAG->getTargetConstant((N2CVal & 0xFFFF0000U) >> 16,
                                                  MVT::i32);
        SDValue Ops[] = { N0.getOperand(0), Imm16,
                          getAL(CurDAG), CurDAG->getRegister(0, MVT::i32) };
        return CurDAG->getMachineNode(Opc, dl, VT, Ops, 4);
      }
    }
    break;
  }
  case ARMISD::VMOVRRD:
    return CurDAG->getMachineNode(ARM::VMOVRRD, dl, MVT::i32, MVT::i32,
                                  N->getOperand(0), getAL(CurDAG),
                                  CurDAG->getRegister(0, MVT::i32));
  case ISD::UMUL_LOHI: {
    if (Subtarget->isThumb1Only())
      break;
    if (Subtarget->isThumb()) {
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                        getAL(CurDAG), CurDAG->getRegister(0, MVT::i32),
                        CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->getMachineNode(ARM::t2UMULL, dl, MVT::i32, MVT::i32,Ops,4);
    } else {
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                        getAL(CurDAG), CurDAG->getRegister(0, MVT::i32),
                        CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->getMachineNode(ARM::UMULL, dl, MVT::i32, MVT::i32, Ops, 5);
    }
  }
  case ISD::SMUL_LOHI: {
    if (Subtarget->isThumb1Only())
      break;
    if (Subtarget->isThumb()) {
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                        getAL(CurDAG), CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->getMachineNode(ARM::t2SMULL, dl, MVT::i32, MVT::i32,Ops,4);
    } else {
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                        getAL(CurDAG), CurDAG->getRegister(0, MVT::i32),
                        CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->getMachineNode(ARM::SMULL, dl, MVT::i32, MVT::i32, Ops, 5);
    }
  }
  case ISD::LOAD: {
    SDNode *ResNode = 0;
    if (Subtarget->isThumb() && Subtarget->hasThumb2())
      ResNode = SelectT2IndexedLoad(N);
    else
      ResNode = SelectARMIndexedLoad(N);
    if (ResNode)
      return ResNode;
    // Other cases are autogenerated.
    break;
  }
  case ARMISD::BRCOND: {
    // Pattern: (ARMbrcond:void (bb:Other):$dst, (imm:i32):$cc)
    // Emits: (Bcc:void (bb:Other):$dst, (imm:i32):$cc)
    // Pattern complexity = 6  cost = 1  size = 0

    // Pattern: (ARMbrcond:void (bb:Other):$dst, (imm:i32):$cc)
    // Emits: (tBcc:void (bb:Other):$dst, (imm:i32):$cc)
    // Pattern complexity = 6  cost = 1  size = 0

    // Pattern: (ARMbrcond:void (bb:Other):$dst, (imm:i32):$cc)
    // Emits: (t2Bcc:void (bb:Other):$dst, (imm:i32):$cc)
    // Pattern complexity = 6  cost = 1  size = 0

    unsigned Opc = Subtarget->isThumb() ?
      ((Subtarget->hasThumb2()) ? ARM::t2Bcc : ARM::tBcc) : ARM::Bcc;
    SDValue Chain = N->getOperand(0);
    SDValue N1 = N->getOperand(1);
    SDValue N2 = N->getOperand(2);
    SDValue N3 = N->getOperand(3);
    SDValue InFlag = N->getOperand(4);
    assert(N1.getOpcode() == ISD::BasicBlock);
    assert(N2.getOpcode() == ISD::Constant);
    assert(N3.getOpcode() == ISD::Register);

    SDValue Tmp2 = CurDAG->getTargetConstant(((unsigned)
                               cast<ConstantSDNode>(N2)->getZExtValue()),
                               MVT::i32);
    SDValue Ops[] = { N1, Tmp2, N3, Chain, InFlag };
    SDNode *ResNode = CurDAG->getMachineNode(Opc, dl, MVT::Other,
                                             MVT::Flag, Ops, 5);
    Chain = SDValue(ResNode, 0);
    if (N->getNumValues() == 2) {
      InFlag = SDValue(ResNode, 1);
      ReplaceUses(SDValue(N, 1), InFlag);
    }
    ReplaceUses(SDValue(N, 0),
                SDValue(Chain.getNode(), Chain.getResNo()));
    return NULL;
  }
  case ARMISD::CMOV:
    return SelectCMOVOp(N);
  case ARMISD::CNEG: {
    EVT VT = N->getValueType(0);
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);
    SDValue N2 = N->getOperand(2);
    SDValue N3 = N->getOperand(3);
    SDValue InFlag = N->getOperand(4);
    assert(N2.getOpcode() == ISD::Constant);
    assert(N3.getOpcode() == ISD::Register);

    SDValue Tmp2 = CurDAG->getTargetConstant(((unsigned)
                               cast<ConstantSDNode>(N2)->getZExtValue()),
                               MVT::i32);
    SDValue Ops[] = { N0, N1, Tmp2, N3, InFlag };
    unsigned Opc = 0;
    switch (VT.getSimpleVT().SimpleTy) {
    default: assert(false && "Illegal conditional move type!");
      break;
    case MVT::f32:
      Opc = ARM::VNEGScc;
      break;
    case MVT::f64:
      Opc = ARM::VNEGDcc;
      break;
    }
    return CurDAG->SelectNodeTo(N, Opc, VT, Ops, 5);
  }

  case ARMISD::VZIP: {
    unsigned Opc = 0;
    EVT VT = N->getValueType(0);
    switch (VT.getSimpleVT().SimpleTy) {
    default: return NULL;
    case MVT::v8i8:  Opc = ARM::VZIPd8; break;
    case MVT::v4i16: Opc = ARM::VZIPd16; break;
    case MVT::v2f32:
    case MVT::v2i32: Opc = ARM::VZIPd32; break;
    case MVT::v16i8: Opc = ARM::VZIPq8; break;
    case MVT::v8i16: Opc = ARM::VZIPq16; break;
    case MVT::v4f32:
    case MVT::v4i32: Opc = ARM::VZIPq32; break;
    }
    SDValue Pred = getAL(CurDAG);
    SDValue PredReg = CurDAG->getRegister(0, MVT::i32);
    SDValue Ops[] = { N->getOperand(0), N->getOperand(1), Pred, PredReg };
    return CurDAG->getMachineNode(Opc, dl, VT, VT, Ops, 4);
  }
  case ARMISD::VUZP: {
    unsigned Opc = 0;
    EVT VT = N->getValueType(0);
    switch (VT.getSimpleVT().SimpleTy) {
    default: return NULL;
    case MVT::v8i8:  Opc = ARM::VUZPd8; break;
    case MVT::v4i16: Opc = ARM::VUZPd16; break;
    case MVT::v2f32:
    case MVT::v2i32: Opc = ARM::VUZPd32; break;
    case MVT::v16i8: Opc = ARM::VUZPq8; break;
    case MVT::v8i16: Opc = ARM::VUZPq16; break;
    case MVT::v4f32:
    case MVT::v4i32: Opc = ARM::VUZPq32; break;
    }
    SDValue Pred = getAL(CurDAG);
    SDValue PredReg = CurDAG->getRegister(0, MVT::i32);
    SDValue Ops[] = { N->getOperand(0), N->getOperand(1), Pred, PredReg };
    return CurDAG->getMachineNode(Opc, dl, VT, VT, Ops, 4);
  }
  case ARMISD::VTRN: {
    unsigned Opc = 0;
    EVT VT = N->getValueType(0);
    switch (VT.getSimpleVT().SimpleTy) {
    default: return NULL;
    case MVT::v8i8:  Opc = ARM::VTRNd8; break;
    case MVT::v4i16: Opc = ARM::VTRNd16; break;
    case MVT::v2f32:
    case MVT::v2i32: Opc = ARM::VTRNd32; break;
    case MVT::v16i8: Opc = ARM::VTRNq8; break;
    case MVT::v8i16: Opc = ARM::VTRNq16; break;
    case MVT::v4f32:
    case MVT::v4i32: Opc = ARM::VTRNq32; break;
    }
    SDValue Pred = getAL(CurDAG);
    SDValue PredReg = CurDAG->getRegister(0, MVT::i32);
    SDValue Ops[] = { N->getOperand(0), N->getOperand(1), Pred, PredReg };
    return CurDAG->getMachineNode(Opc, dl, VT, VT, Ops, 4);
  }
  case ARMISD::BUILD_VECTOR: {
    EVT VecVT = N->getValueType(0);
    EVT EltVT = VecVT.getVectorElementType();
    unsigned NumElts = VecVT.getVectorNumElements();
    if (EltVT.getSimpleVT() == MVT::f64) {
      assert(NumElts == 2 && "unexpected type for BUILD_VECTOR");
      return PairDRegs(VecVT, N->getOperand(0), N->getOperand(1));
    }
    assert(EltVT.getSimpleVT() == MVT::f32 &&
           "unexpected type for BUILD_VECTOR");
    if (NumElts == 2)
      return PairSRegs(VecVT, N->getOperand(0), N->getOperand(1));
    assert(NumElts == 4 && "unexpected type for BUILD_VECTOR");
    return QuadSRegs(VecVT, N->getOperand(0), N->getOperand(1),
                     N->getOperand(2), N->getOperand(3));
  }

  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
    switch (IntNo) {
    default:
      break;

    case Intrinsic::arm_neon_vld1: {
      unsigned DOpcodes[] = { ARM::VLD1d8, ARM::VLD1d16,
                              ARM::VLD1d32, ARM::VLD1d64 };
      unsigned QOpcodes[] = { ARM::VLD1q8Pseudo, ARM::VLD1q16Pseudo,
                              ARM::VLD1q32Pseudo, ARM::VLD1q64Pseudo };
      return SelectVLD(N, 1, DOpcodes, QOpcodes, 0);
    }

    case Intrinsic::arm_neon_vld2: {
      unsigned DOpcodes[] = { ARM::VLD2d8Pseudo, ARM::VLD2d16Pseudo,
                              ARM::VLD2d32Pseudo, ARM::VLD1q64Pseudo };
      unsigned QOpcodes[] = { ARM::VLD2q8Pseudo, ARM::VLD2q16Pseudo,
                              ARM::VLD2q32Pseudo };
      return SelectVLD(N, 2, DOpcodes, QOpcodes, 0);
    }

    case Intrinsic::arm_neon_vld3: {
      unsigned DOpcodes[] = { ARM::VLD3d8Pseudo, ARM::VLD3d16Pseudo,
                              ARM::VLD3d32Pseudo, ARM::VLD1d64TPseudo };
      unsigned QOpcodes0[] = { ARM::VLD3q8Pseudo_UPD,
                               ARM::VLD3q16Pseudo_UPD,
                               ARM::VLD3q32Pseudo_UPD };
      unsigned QOpcodes1[] = { ARM::VLD3q8oddPseudo_UPD,
                               ARM::VLD3q16oddPseudo_UPD,
                               ARM::VLD3q32oddPseudo_UPD };
      return SelectVLD(N, 3, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vld4: {
      unsigned DOpcodes[] = { ARM::VLD4d8Pseudo, ARM::VLD4d16Pseudo,
                              ARM::VLD4d32Pseudo, ARM::VLD1d64QPseudo };
      unsigned QOpcodes0[] = { ARM::VLD4q8Pseudo_UPD,
                               ARM::VLD4q16Pseudo_UPD,
                               ARM::VLD4q32Pseudo_UPD };
      unsigned QOpcodes1[] = { ARM::VLD4q8oddPseudo_UPD,
                               ARM::VLD4q16oddPseudo_UPD,
                               ARM::VLD4q32oddPseudo_UPD };
      return SelectVLD(N, 4, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vld2lane: {
      unsigned DOpcodes[] = { ARM::VLD2LNd8, ARM::VLD2LNd16, ARM::VLD2LNd32 };
      unsigned QOpcodes0[] = { ARM::VLD2LNq16, ARM::VLD2LNq32 };
      unsigned QOpcodes1[] = { ARM::VLD2LNq16odd, ARM::VLD2LNq32odd };
      return SelectVLDSTLane(N, true, 2, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vld3lane: {
      unsigned DOpcodes[] = { ARM::VLD3LNd8, ARM::VLD3LNd16, ARM::VLD3LNd32 };
      unsigned QOpcodes0[] = { ARM::VLD3LNq16, ARM::VLD3LNq32 };
      unsigned QOpcodes1[] = { ARM::VLD3LNq16odd, ARM::VLD3LNq32odd };
      return SelectVLDSTLane(N, true, 3, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vld4lane: {
      unsigned DOpcodes[] = { ARM::VLD4LNd8, ARM::VLD4LNd16, ARM::VLD4LNd32 };
      unsigned QOpcodes0[] = { ARM::VLD4LNq16, ARM::VLD4LNq32 };
      unsigned QOpcodes1[] = { ARM::VLD4LNq16odd, ARM::VLD4LNq32odd };
      return SelectVLDSTLane(N, true, 4, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vst1: {
      unsigned DOpcodes[] = { ARM::VST1d8, ARM::VST1d16,
                              ARM::VST1d32, ARM::VST1d64 };
      unsigned QOpcodes[] = { ARM::VST1q8Pseudo, ARM::VST1q16Pseudo,
                              ARM::VST1q32Pseudo, ARM::VST1q64Pseudo };
      return SelectVST(N, 1, DOpcodes, QOpcodes, 0);
    }

    case Intrinsic::arm_neon_vst2: {
      unsigned DOpcodes[] = { ARM::VST2d8Pseudo, ARM::VST2d16Pseudo,
                              ARM::VST2d32Pseudo, ARM::VST1q64Pseudo };
      unsigned QOpcodes[] = { ARM::VST2q8Pseudo, ARM::VST2q16Pseudo,
                              ARM::VST2q32Pseudo };
      return SelectVST(N, 2, DOpcodes, QOpcodes, 0);
    }

    case Intrinsic::arm_neon_vst3: {
      unsigned DOpcodes[] = { ARM::VST3d8Pseudo, ARM::VST3d16Pseudo,
                              ARM::VST3d32Pseudo, ARM::VST1d64TPseudo };
      unsigned QOpcodes0[] = { ARM::VST3q8Pseudo_UPD,
                               ARM::VST3q16Pseudo_UPD,
                               ARM::VST3q32Pseudo_UPD };
      unsigned QOpcodes1[] = { ARM::VST3q8oddPseudo_UPD,
                               ARM::VST3q16oddPseudo_UPD,
                               ARM::VST3q32oddPseudo_UPD };
      return SelectVST(N, 3, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vst4: {
      unsigned DOpcodes[] = { ARM::VST4d8Pseudo, ARM::VST4d16Pseudo,
                              ARM::VST4d32Pseudo, ARM::VST1d64QPseudo };
      unsigned QOpcodes0[] = { ARM::VST4q8Pseudo_UPD,
                               ARM::VST4q16Pseudo_UPD,
                               ARM::VST4q32Pseudo_UPD };
      unsigned QOpcodes1[] = { ARM::VST4q8oddPseudo_UPD,
                               ARM::VST4q16oddPseudo_UPD,
                               ARM::VST4q32oddPseudo_UPD };
      return SelectVST(N, 4, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vst2lane: {
      unsigned DOpcodes[] = { ARM::VST2LNd8, ARM::VST2LNd16, ARM::VST2LNd32 };
      unsigned QOpcodes0[] = { ARM::VST2LNq16, ARM::VST2LNq32 };
      unsigned QOpcodes1[] = { ARM::VST2LNq16odd, ARM::VST2LNq32odd };
      return SelectVLDSTLane(N, false, 2, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vst3lane: {
      unsigned DOpcodes[] = { ARM::VST3LNd8, ARM::VST3LNd16, ARM::VST3LNd32 };
      unsigned QOpcodes0[] = { ARM::VST3LNq16, ARM::VST3LNq32 };
      unsigned QOpcodes1[] = { ARM::VST3LNq16odd, ARM::VST3LNq32odd };
      return SelectVLDSTLane(N, false, 3, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vst4lane: {
      unsigned DOpcodes[] = { ARM::VST4LNd8, ARM::VST4LNd16, ARM::VST4LNd32 };
      unsigned QOpcodes0[] = { ARM::VST4LNq16, ARM::VST4LNq32 };
      unsigned QOpcodes1[] = { ARM::VST4LNq16odd, ARM::VST4LNq32odd };
      return SelectVLDSTLane(N, false, 4, DOpcodes, QOpcodes0, QOpcodes1);
    }
    }
    break;
  }

  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    switch (IntNo) {
    default:
      break;

    case Intrinsic::arm_neon_vtbl2:
      return SelectVTBL(N, false, 2, ARM::VTBL2);
    case Intrinsic::arm_neon_vtbl3:
      return SelectVTBL(N, false, 3, ARM::VTBL3);
    case Intrinsic::arm_neon_vtbl4:
      return SelectVTBL(N, false, 4, ARM::VTBL4);

    case Intrinsic::arm_neon_vtbx2:
      return SelectVTBL(N, true, 2, ARM::VTBX2);
    case Intrinsic::arm_neon_vtbx3:
      return SelectVTBL(N, true, 3, ARM::VTBX3);
    case Intrinsic::arm_neon_vtbx4:
      return SelectVTBL(N, true, 4, ARM::VTBX4);
    }
    break;
  }

  case ISD::CONCAT_VECTORS:
    return SelectConcatVector(N);
  }

  return SelectCode(N);
}

bool ARMDAGToDAGISel::
SelectInlineAsmMemoryOperand(const SDValue &Op, char ConstraintCode,
                             std::vector<SDValue> &OutOps) {
  assert(ConstraintCode == 'm' && "unexpected asm memory constraint");
  // Require the address to be in a register.  That is safe for all ARM
  // variants and it is hard to do anything much smarter without knowing
  // how the operand is used.
  OutOps.push_back(Op);
  return false;
}

/// createARMISelDag - This pass converts a legalized DAG into a
/// ARM-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createARMISelDag(ARMBaseTargetMachine &TM,
                                     CodeGenOpt::Level OptLevel) {
  return new ARMDAGToDAGISel(TM, OptLevel);
}
