//===-- MipsSEISelDAGToDAG.cpp - A Dag to Dag Inst Selector for MipsSE ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Subclass of MipsDAGToDAGISel specialized for mips32/64.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-isel"
#include "MipsSEISelDAGToDAG.h"
#include "Mips.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MipsAnalyzeImmediate.h"
#include "MipsMachineFunction.h"
#include "MipsRegisterInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;


bool MipsSEDAGToDAGISel::replaceUsesWithZeroReg(MachineRegisterInfo *MRI,
                                                const MachineInstr& MI) {
  unsigned DstReg = 0, ZeroReg = 0;

  // Check if MI is "addiu $dst, $zero, 0" or "daddiu $dst, $zero, 0".
  if ((MI.getOpcode() == Mips::ADDiu) &&
      (MI.getOperand(1).getReg() == Mips::ZERO) &&
      (MI.getOperand(2).getImm() == 0)) {
    DstReg = MI.getOperand(0).getReg();
    ZeroReg = Mips::ZERO;
  } else if ((MI.getOpcode() == Mips::DADDiu) &&
             (MI.getOperand(1).getReg() == Mips::ZERO_64) &&
             (MI.getOperand(2).getImm() == 0)) {
    DstReg = MI.getOperand(0).getReg();
    ZeroReg = Mips::ZERO_64;
  }

  if (!DstReg)
    return false;

  // Replace uses with ZeroReg.
  for (MachineRegisterInfo::use_iterator U = MRI->use_begin(DstReg),
       E = MRI->use_end(); U != E;) {
    MachineOperand &MO = U.getOperand();
    unsigned OpNo = U.getOperandNo();
    MachineInstr *MI = MO.getParent();
    ++U;

    // Do not replace if it is a phi's operand or is tied to def operand.
    if (MI->isPHI() || MI->isRegTiedToDefOperand(OpNo) || MI->isPseudo())
      continue;

    MO.setReg(ZeroReg);
  }

  return true;
}

void MipsSEDAGToDAGISel::initGlobalBaseReg(MachineFunction &MF) {
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();

  if (!MipsFI->globalBaseRegSet())
    return;

  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator I = MBB.begin();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  DebugLoc DL = I != MBB.end() ? I->getDebugLoc() : DebugLoc();
  unsigned V0, V1, GlobalBaseReg = MipsFI->getGlobalBaseReg();
  const TargetRegisterClass *RC;

  if (Subtarget.isABI_N64())
    RC = (const TargetRegisterClass*)&Mips::CPU64RegsRegClass;
  else
    RC = (const TargetRegisterClass*)&Mips::CPURegsRegClass;

  V0 = RegInfo.createVirtualRegister(RC);
  V1 = RegInfo.createVirtualRegister(RC);

  if (Subtarget.isABI_N64()) {
    MF.getRegInfo().addLiveIn(Mips::T9_64);
    MBB.addLiveIn(Mips::T9_64);

    // lui $v0, %hi(%neg(%gp_rel(fname)))
    // daddu $v1, $v0, $t9
    // daddiu $globalbasereg, $v1, %lo(%neg(%gp_rel(fname)))
    const GlobalValue *FName = MF.getFunction();
    BuildMI(MBB, I, DL, TII.get(Mips::LUi64), V0)
      .addGlobalAddress(FName, 0, MipsII::MO_GPOFF_HI);
    BuildMI(MBB, I, DL, TII.get(Mips::DADDu), V1).addReg(V0)
      .addReg(Mips::T9_64);
    BuildMI(MBB, I, DL, TII.get(Mips::DADDiu), GlobalBaseReg).addReg(V1)
      .addGlobalAddress(FName, 0, MipsII::MO_GPOFF_LO);
    return;
  }

  if (MF.getTarget().getRelocationModel() == Reloc::Static) {
    // Set global register to __gnu_local_gp.
    //
    // lui   $v0, %hi(__gnu_local_gp)
    // addiu $globalbasereg, $v0, %lo(__gnu_local_gp)
    BuildMI(MBB, I, DL, TII.get(Mips::LUi), V0)
      .addExternalSymbol("__gnu_local_gp", MipsII::MO_ABS_HI);
    BuildMI(MBB, I, DL, TII.get(Mips::ADDiu), GlobalBaseReg).addReg(V0)
      .addExternalSymbol("__gnu_local_gp", MipsII::MO_ABS_LO);
    return;
  }

  MF.getRegInfo().addLiveIn(Mips::T9);
  MBB.addLiveIn(Mips::T9);

  if (Subtarget.isABI_N32()) {
    // lui $v0, %hi(%neg(%gp_rel(fname)))
    // addu $v1, $v0, $t9
    // addiu $globalbasereg, $v1, %lo(%neg(%gp_rel(fname)))
    const GlobalValue *FName = MF.getFunction();
    BuildMI(MBB, I, DL, TII.get(Mips::LUi), V0)
      .addGlobalAddress(FName, 0, MipsII::MO_GPOFF_HI);
    BuildMI(MBB, I, DL, TII.get(Mips::ADDu), V1).addReg(V0).addReg(Mips::T9);
    BuildMI(MBB, I, DL, TII.get(Mips::ADDiu), GlobalBaseReg).addReg(V1)
      .addGlobalAddress(FName, 0, MipsII::MO_GPOFF_LO);
    return;
  }

  assert(Subtarget.isABI_O32());

  // For O32 ABI, the following instruction sequence is emitted to initialize
  // the global base register:
  //
  //  0. lui   $2, %hi(_gp_disp)
  //  1. addiu $2, $2, %lo(_gp_disp)
  //  2. addu  $globalbasereg, $2, $t9
  //
  // We emit only the last instruction here.
  //
  // GNU linker requires that the first two instructions appear at the beginning
  // of a function and no instructions be inserted before or between them.
  // The two instructions are emitted during lowering to MC layer in order to
  // avoid any reordering.
  //
  // Register $2 (Mips::V0) is added to the list of live-in registers to ensure
  // the value instruction 1 (addiu) defines is valid when instruction 2 (addu)
  // reads it.
  MF.getRegInfo().addLiveIn(Mips::V0);
  MBB.addLiveIn(Mips::V0);
  BuildMI(MBB, I, DL, TII.get(Mips::ADDu), GlobalBaseReg)
    .addReg(Mips::V0).addReg(Mips::T9);
}

void MipsSEDAGToDAGISel::processFunctionAfterISel(MachineFunction &MF) {
  initGlobalBaseReg(MF);

  MachineRegisterInfo *MRI = &MF.getRegInfo();

  for (MachineFunction::iterator MFI = MF.begin(), MFE = MF.end(); MFI != MFE;
       ++MFI)
    for (MachineBasicBlock::iterator I = MFI->begin(); I != MFI->end(); ++I)
      replaceUsesWithZeroReg(MRI, *I);
}

/// Select multiply instructions.
std::pair<SDNode*, SDNode*>
MipsSEDAGToDAGISel::selectMULT(SDNode *N, unsigned Opc, DebugLoc DL, EVT Ty,
                               bool HasLo, bool HasHi) {
  SDNode *Lo = 0, *Hi = 0;
  SDNode *Mul = CurDAG->getMachineNode(Opc, DL, MVT::Glue, N->getOperand(0),
                                       N->getOperand(1));
  SDValue InFlag = SDValue(Mul, 0);

  if (HasLo) {
    unsigned Opcode = (Ty == MVT::i32 ? Mips::MFLO : Mips::MFLO64);
    Lo = CurDAG->getMachineNode(Opcode, DL, Ty, MVT::Glue, InFlag);
    InFlag = SDValue(Lo, 1);
  }
  if (HasHi) {
    unsigned Opcode = (Ty == MVT::i32 ? Mips::MFHI : Mips::MFHI64);
    Hi = CurDAG->getMachineNode(Opcode, DL, Ty, InFlag);
  }
  return std::make_pair(Lo, Hi);
}

/// ComplexPattern used on MipsInstrInfo
/// Used on Mips Load/Store instructions
bool MipsSEDAGToDAGISel::selectAddrRegImm(SDValue Addr, SDValue &Base,
                                          SDValue &Offset) const {
  EVT ValTy = Addr.getValueType();

  // if Address is FI, get the TargetFrameIndex.
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base   = CurDAG->getTargetFrameIndex(FIN->getIndex(), ValTy);
    Offset = CurDAG->getTargetConstant(0, ValTy);
    return true;
  }

  // on PIC code Load GA
  if (Addr.getOpcode() == MipsISD::Wrapper) {
    Base   = Addr.getOperand(0);
    Offset = Addr.getOperand(1);
    return true;
  }

  if (TM.getRelocationModel() != Reloc::PIC_) {
    if ((Addr.getOpcode() == ISD::TargetExternalSymbol ||
        Addr.getOpcode() == ISD::TargetGlobalAddress))
      return false;
  }

  // Addresses of the form FI+const or FI|const
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1));
    if (isInt<16>(CN->getSExtValue())) {

      // If the first operand is a FI, get the TargetFI Node
      if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>
                                  (Addr.getOperand(0)))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), ValTy);
      else
        Base = Addr.getOperand(0);

      Offset = CurDAG->getTargetConstant(CN->getZExtValue(), ValTy);
      return true;
    }
  }

  // Operand is a result from an ADD.
  if (Addr.getOpcode() == ISD::ADD) {
    // When loading from constant pools, load the lower address part in
    // the instruction itself. Example, instead of:
    //  lui $2, %hi($CPI1_0)
    //  addiu $2, $2, %lo($CPI1_0)
    //  lwc1 $f0, 0($2)
    // Generate:
    //  lui $2, %hi($CPI1_0)
    //  lwc1 $f0, %lo($CPI1_0)($2)
    if (Addr.getOperand(1).getOpcode() == MipsISD::Lo ||
        Addr.getOperand(1).getOpcode() == MipsISD::GPRel) {
      SDValue Opnd0 = Addr.getOperand(1).getOperand(0);
      if (isa<ConstantPoolSDNode>(Opnd0) || isa<GlobalAddressSDNode>(Opnd0) ||
          isa<JumpTableSDNode>(Opnd0)) {
        Base = Addr.getOperand(0);
        Offset = Opnd0;
        return true;
      }
    }
  }

  return false;
}

bool MipsSEDAGToDAGISel::selectAddrDefault(SDValue Addr, SDValue &Base,
                                           SDValue &Offset) const {
  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, Addr.getValueType());
  return true;
}

bool MipsSEDAGToDAGISel::selectIntAddr(SDValue Addr, SDValue &Base,
                                       SDValue &Offset) const {
  return selectAddrRegImm(Addr, Base, Offset) ||
    selectAddrDefault(Addr, Base, Offset);
}

std::pair<bool, SDNode*> MipsSEDAGToDAGISel::selectNode(SDNode *Node) {
  unsigned Opcode = Node->getOpcode();
  DebugLoc DL = Node->getDebugLoc();

  ///
  // Instruction Selection not handled by the auto-generated
  // tablegen selection should be handled here.
  ///
  EVT NodeTy = Node->getValueType(0);
  SDNode *Result;
  unsigned MultOpc;

  switch(Opcode) {
  default: break;

  case ISD::SUBE:
  case ISD::ADDE: {
    SDValue InFlag = Node->getOperand(2), CmpLHS;
    unsigned Opc = InFlag.getOpcode(); (void)Opc;
    assert(((Opc == ISD::ADDC || Opc == ISD::ADDE) ||
            (Opc == ISD::SUBC || Opc == ISD::SUBE)) &&
           "(ADD|SUB)E flag operand must come from (ADD|SUB)C/E insn");

    unsigned MOp;
    if (Opcode == ISD::ADDE) {
      CmpLHS = InFlag.getValue(0);
      MOp = Mips::ADDu;
    } else {
      CmpLHS = InFlag.getOperand(0);
      MOp = Mips::SUBu;
    }

    SDValue Ops[] = { CmpLHS, InFlag.getOperand(1) };

    SDValue LHS = Node->getOperand(0);
    SDValue RHS = Node->getOperand(1);

    EVT VT = LHS.getValueType();

    unsigned Sltu_op = Mips::SLTu;
    SDNode *Carry = CurDAG->getMachineNode(Sltu_op, DL, VT, Ops, 2);
    unsigned Addu_op = Mips::ADDu;
    SDNode *AddCarry = CurDAG->getMachineNode(Addu_op, DL, VT,
                                              SDValue(Carry,0), RHS);

    Result = CurDAG->SelectNodeTo(Node, MOp, VT, MVT::Glue, LHS,
                                  SDValue(AddCarry,0));
    return std::make_pair(true, Result);
  }

  /// Mul with two results
  case ISD::SMUL_LOHI:
  case ISD::UMUL_LOHI: {
    if (NodeTy == MVT::i32)
      MultOpc = (Opcode == ISD::UMUL_LOHI ? Mips::MULTu : Mips::MULT);
    else
      MultOpc = (Opcode == ISD::UMUL_LOHI ? Mips::DMULTu : Mips::DMULT);

    std::pair<SDNode*, SDNode*> LoHi = selectMULT(Node, MultOpc, DL, NodeTy,
                                                  true, true);

    if (!SDValue(Node, 0).use_empty())
      ReplaceUses(SDValue(Node, 0), SDValue(LoHi.first, 0));

    if (!SDValue(Node, 1).use_empty())
      ReplaceUses(SDValue(Node, 1), SDValue(LoHi.second, 0));

    return std::make_pair(true, (SDNode*)NULL);
  }

  /// Special Muls
  case ISD::MUL: {
    // Mips32 has a 32-bit three operand mul instruction.
    if (Subtarget.hasMips32() && NodeTy == MVT::i32)
      break;
    MultOpc = NodeTy == MVT::i32 ? Mips::MULT : Mips::DMULT;
    Result = selectMULT(Node, MultOpc, DL, NodeTy, true, false).first;
    return std::make_pair(true, Result);
  }
  case ISD::MULHS:
  case ISD::MULHU: {
    if (NodeTy == MVT::i32)
      MultOpc = (Opcode == ISD::MULHU ? Mips::MULTu : Mips::MULT);
    else
      MultOpc = (Opcode == ISD::MULHU ? Mips::DMULTu : Mips::DMULT);

    Result = selectMULT(Node, MultOpc, DL, NodeTy, false, true).second;
    return std::make_pair(true, Result);
  }

  case ISD::ConstantFP: {
    ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(Node);
    if (Node->getValueType(0) == MVT::f64 && CN->isExactlyValue(+0.0)) {
      if (Subtarget.hasMips64()) {
        SDValue Zero = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL,
                                              Mips::ZERO_64, MVT::i64);
        Result = CurDAG->getMachineNode(Mips::DMTC1, DL, MVT::f64, Zero);
      } else {
        SDValue Zero = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL,
                                              Mips::ZERO, MVT::i32);
        Result = CurDAG->getMachineNode(Mips::BuildPairF64, DL, MVT::f64, Zero,
                                        Zero);
      }

      return std::make_pair(true, Result);
    }
    break;
  }

  case ISD::Constant: {
    const ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Node);
    unsigned Size = CN->getValueSizeInBits(0);

    if (Size == 32)
      break;

    MipsAnalyzeImmediate AnalyzeImm;
    int64_t Imm = CN->getSExtValue();

    const MipsAnalyzeImmediate::InstSeq &Seq =
      AnalyzeImm.Analyze(Imm, Size, false);

    MipsAnalyzeImmediate::InstSeq::const_iterator Inst = Seq.begin();
    DebugLoc DL = CN->getDebugLoc();
    SDNode *RegOpnd;
    SDValue ImmOpnd = CurDAG->getTargetConstant(SignExtend64<16>(Inst->ImmOpnd),
                                                MVT::i64);

    // The first instruction can be a LUi which is different from other
    // instructions (ADDiu, ORI and SLL) in that it does not have a register
    // operand.
    if (Inst->Opc == Mips::LUi64)
      RegOpnd = CurDAG->getMachineNode(Inst->Opc, DL, MVT::i64, ImmOpnd);
    else
      RegOpnd =
        CurDAG->getMachineNode(Inst->Opc, DL, MVT::i64,
                               CurDAG->getRegister(Mips::ZERO_64, MVT::i64),
                               ImmOpnd);

    // The remaining instructions in the sequence are handled here.
    for (++Inst; Inst != Seq.end(); ++Inst) {
      ImmOpnd = CurDAG->getTargetConstant(SignExtend64<16>(Inst->ImmOpnd),
                                          MVT::i64);
      RegOpnd = CurDAG->getMachineNode(Inst->Opc, DL, MVT::i64,
                                       SDValue(RegOpnd, 0), ImmOpnd);
    }

    return std::make_pair(true, RegOpnd);
  }

  case MipsISD::ThreadPointer: {
    EVT PtrVT = TLI.getPointerTy();
    unsigned RdhwrOpc, SrcReg, DestReg;

    if (PtrVT == MVT::i32) {
      RdhwrOpc = Mips::RDHWR;
      SrcReg = Mips::HWR29;
      DestReg = Mips::V1;
    } else {
      RdhwrOpc = Mips::RDHWR64;
      SrcReg = Mips::HWR29_64;
      DestReg = Mips::V1_64;
    }

    SDNode *Rdhwr =
      CurDAG->getMachineNode(RdhwrOpc, Node->getDebugLoc(),
                             Node->getValueType(0),
                             CurDAG->getRegister(SrcReg, PtrVT));
    SDValue Chain = CurDAG->getCopyToReg(CurDAG->getEntryNode(), DL, DestReg,
                                         SDValue(Rdhwr, 0));
    SDValue ResNode = CurDAG->getCopyFromReg(Chain, DL, DestReg, PtrVT);
    ReplaceUses(SDValue(Node, 0), ResNode);
    return std::make_pair(true, ResNode.getNode());
  }
  }

  return std::make_pair(false, (SDNode*)NULL);
}

FunctionPass *llvm::createMipsSEISelDag(MipsTargetMachine &TM) {
  return new MipsSEDAGToDAGISel(TM);
}
