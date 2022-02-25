//===-- RISCVISelDAGToDAG.cpp - A dag to dag inst selector for RISCV ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the RISCV target.
//
//===----------------------------------------------------------------------===//

#include "RISCVISelDAGToDAG.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "RISCVISelLowering.h"
#include "RISCVMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-isel"

namespace llvm {
namespace RISCV {
#define GET_RISCVVSSEGTable_IMPL
#define GET_RISCVVLSEGTable_IMPL
#define GET_RISCVVLXSEGTable_IMPL
#define GET_RISCVVSXSEGTable_IMPL
#define GET_RISCVVLETable_IMPL
#define GET_RISCVVSETable_IMPL
#define GET_RISCVVLXTable_IMPL
#define GET_RISCVVSXTable_IMPL
#include "RISCVGenSearchableTables.inc"
} // namespace RISCV
} // namespace llvm

void RISCVDAGToDAGISel::PreprocessISelDAG() {
  for (SelectionDAG::allnodes_iterator I = CurDAG->allnodes_begin(),
                                       E = CurDAG->allnodes_end();
       I != E;) {
    SDNode *N = &*I++; // Preincrement iterator to avoid invalidation issues.

    // Lower SPLAT_VECTOR_SPLIT_I64 to two scalar stores and a stride 0 vector
    // load. Done after lowering and combining so that we have a chance to
    // optimize this to VMV_V_X_VL when the upper bits aren't needed.
    if (N->getOpcode() != RISCVISD::SPLAT_VECTOR_SPLIT_I64_VL)
      continue;

    assert(N->getNumOperands() == 3 && "Unexpected number of operands");
    MVT VT = N->getSimpleValueType(0);
    SDValue Lo = N->getOperand(0);
    SDValue Hi = N->getOperand(1);
    SDValue VL = N->getOperand(2);
    assert(VT.getVectorElementType() == MVT::i64 && VT.isScalableVector() &&
           Lo.getValueType() == MVT::i32 && Hi.getValueType() == MVT::i32 &&
           "Unexpected VTs!");
    MachineFunction &MF = CurDAG->getMachineFunction();
    RISCVMachineFunctionInfo *FuncInfo = MF.getInfo<RISCVMachineFunctionInfo>();
    SDLoc DL(N);

    // We use the same frame index we use for moving two i32s into 64-bit FPR.
    // This is an analogous operation.
    int FI = FuncInfo->getMoveF64FrameIndex(MF);
    MachinePointerInfo MPI = MachinePointerInfo::getFixedStack(MF, FI);
    const TargetLowering &TLI = CurDAG->getTargetLoweringInfo();
    SDValue StackSlot =
        CurDAG->getFrameIndex(FI, TLI.getPointerTy(CurDAG->getDataLayout()));

    SDValue Chain = CurDAG->getEntryNode();
    Lo = CurDAG->getStore(Chain, DL, Lo, StackSlot, MPI, Align(8));

    SDValue OffsetSlot =
        CurDAG->getMemBasePlusOffset(StackSlot, TypeSize::Fixed(4), DL);
    Hi = CurDAG->getStore(Chain, DL, Hi, OffsetSlot, MPI.getWithOffset(4),
                          Align(8));

    Chain = CurDAG->getNode(ISD::TokenFactor, DL, MVT::Other, Lo, Hi);

    SDVTList VTs = CurDAG->getVTList({VT, MVT::Other});
    SDValue IntID =
        CurDAG->getTargetConstant(Intrinsic::riscv_vlse, DL, MVT::i64);
    SDValue Ops[] = {Chain, IntID, StackSlot,
                     CurDAG->getRegister(RISCV::X0, MVT::i64), VL};

    SDValue Result = CurDAG->getMemIntrinsicNode(
        ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops, MVT::i64, MPI, Align(8),
        MachineMemOperand::MOLoad);

    // We're about to replace all uses of the SPLAT_VECTOR_SPLIT_I64 with the
    // vlse we created.  This will cause general havok on the dag because
    // anything below the conversion could be folded into other existing nodes.
    // To avoid invalidating 'I', back it up to the convert node.
    --I;
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), Result);

    // Now that we did that, the node is dead.  Increment the iterator to the
    // next node to process, then delete N.
    ++I;
    CurDAG->DeleteNode(N);
  }
}

void RISCVDAGToDAGISel::PostprocessISelDAG() {
  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  bool MadeChange = false;
  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    // Skip dead nodes and any non-machine opcodes.
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    MadeChange |= doPeepholeSExtW(N);
    MadeChange |= doPeepholeLoadStoreADDI(N);
  }

  if (MadeChange)
    CurDAG->RemoveDeadNodes();
}

static SDNode *selectImm(SelectionDAG *CurDAG, const SDLoc &DL, int64_t Imm,
                         const RISCVSubtarget &Subtarget) {
  MVT XLenVT = Subtarget.getXLenVT();
  RISCVMatInt::InstSeq Seq =
      RISCVMatInt::generateInstSeq(Imm, Subtarget.getFeatureBits());

  SDNode *Result = nullptr;
  SDValue SrcReg = CurDAG->getRegister(RISCV::X0, XLenVT);
  for (RISCVMatInt::Inst &Inst : Seq) {
    SDValue SDImm = CurDAG->getTargetConstant(Inst.Imm, DL, XLenVT);
    if (Inst.Opc == RISCV::LUI)
      Result = CurDAG->getMachineNode(RISCV::LUI, DL, XLenVT, SDImm);
    else if (Inst.Opc == RISCV::ADDUW)
      Result = CurDAG->getMachineNode(RISCV::ADDUW, DL, XLenVT, SrcReg,
                                      CurDAG->getRegister(RISCV::X0, XLenVT));
    else
      Result = CurDAG->getMachineNode(Inst.Opc, DL, XLenVT, SrcReg, SDImm);

    // Only the first instruction has X0 as its source.
    SrcReg = SDValue(Result, 0);
  }

  return Result;
}

static SDValue createTupleImpl(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                               unsigned RegClassID, unsigned SubReg0) {
  assert(Regs.size() >= 2 && Regs.size() <= 8);

  SDLoc DL(Regs[0]);
  SmallVector<SDValue, 8> Ops;

  Ops.push_back(CurDAG.getTargetConstant(RegClassID, DL, MVT::i32));

  for (unsigned I = 0; I < Regs.size(); ++I) {
    Ops.push_back(Regs[I]);
    Ops.push_back(CurDAG.getTargetConstant(SubReg0 + I, DL, MVT::i32));
  }
  SDNode *N =
      CurDAG.getMachineNode(TargetOpcode::REG_SEQUENCE, DL, MVT::Untyped, Ops);
  return SDValue(N, 0);
}

static SDValue createM1Tuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                             unsigned NF) {
  static const unsigned RegClassIDs[] = {
      RISCV::VRN2M1RegClassID, RISCV::VRN3M1RegClassID, RISCV::VRN4M1RegClassID,
      RISCV::VRN5M1RegClassID, RISCV::VRN6M1RegClassID, RISCV::VRN7M1RegClassID,
      RISCV::VRN8M1RegClassID};

  return createTupleImpl(CurDAG, Regs, RegClassIDs[NF - 2], RISCV::sub_vrm1_0);
}

static SDValue createM2Tuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                             unsigned NF) {
  static const unsigned RegClassIDs[] = {RISCV::VRN2M2RegClassID,
                                         RISCV::VRN3M2RegClassID,
                                         RISCV::VRN4M2RegClassID};

  return createTupleImpl(CurDAG, Regs, RegClassIDs[NF - 2], RISCV::sub_vrm2_0);
}

static SDValue createM4Tuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                             unsigned NF) {
  return createTupleImpl(CurDAG, Regs, RISCV::VRN2M4RegClassID,
                         RISCV::sub_vrm4_0);
}

static SDValue createTuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                           unsigned NF, RISCVII::VLMUL LMUL) {
  switch (LMUL) {
  default:
    llvm_unreachable("Invalid LMUL.");
  case RISCVII::VLMUL::LMUL_F8:
  case RISCVII::VLMUL::LMUL_F4:
  case RISCVII::VLMUL::LMUL_F2:
  case RISCVII::VLMUL::LMUL_1:
    return createM1Tuple(CurDAG, Regs, NF);
  case RISCVII::VLMUL::LMUL_2:
    return createM2Tuple(CurDAG, Regs, NF);
  case RISCVII::VLMUL::LMUL_4:
    return createM4Tuple(CurDAG, Regs, NF);
  }
}

void RISCVDAGToDAGISel::addVectorLoadStoreOperands(
    SDNode *Node, unsigned Log2SEW, const SDLoc &DL, unsigned CurOp,
    bool IsMasked, bool IsStridedOrIndexed, SmallVectorImpl<SDValue> &Operands,
    MVT *IndexVT) {
  SDValue Chain = Node->getOperand(0);
  SDValue Glue;

  SDValue Base;
  SelectBaseAddr(Node->getOperand(CurOp++), Base);
  Operands.push_back(Base); // Base pointer.

  if (IsStridedOrIndexed) {
    Operands.push_back(Node->getOperand(CurOp++)); // Index.
    if (IndexVT)
      *IndexVT = Operands.back()->getSimpleValueType(0);
  }

  if (IsMasked) {
    // Mask needs to be copied to V0.
    SDValue Mask = Node->getOperand(CurOp++);
    Chain = CurDAG->getCopyToReg(Chain, DL, RISCV::V0, Mask, SDValue());
    Glue = Chain.getValue(1);
    Operands.push_back(CurDAG->getRegister(RISCV::V0, Mask.getValueType()));
  }
  SDValue VL;
  selectVLOp(Node->getOperand(CurOp++), VL);
  Operands.push_back(VL);

  MVT XLenVT = Subtarget->getXLenVT();
  SDValue SEWOp = CurDAG->getTargetConstant(Log2SEW, DL, XLenVT);
  Operands.push_back(SEWOp);

  Operands.push_back(Chain); // Chain.
  if (Glue)
    Operands.push_back(Glue);
}

void RISCVDAGToDAGISel::selectVLSEG(SDNode *Node, bool IsMasked,
                                    bool IsStrided) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 1;
  MVT VT = Node->getSimpleValueType(0);
  unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
  RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;
  if (IsMasked) {
    SmallVector<SDValue, 8> Regs(Node->op_begin() + CurOp,
                                 Node->op_begin() + CurOp + NF);
    SDValue MaskedOff = createTuple(*CurDAG, Regs, NF, LMUL);
    Operands.push_back(MaskedOff);
    CurOp += NF;
  }

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                             Operands);

  const RISCV::VLSEGPseudo *P =
      RISCV::getVLSEGPseudo(NF, IsMasked, IsStrided, /*FF*/ false, Log2SEW,
                            static_cast<unsigned>(LMUL));
  MachineSDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I) {
    unsigned SubRegIdx = RISCVTargetLowering::getSubregIndexByMVT(VT, I);
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, SuperReg));
  }

  ReplaceUses(SDValue(Node, NF), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVLSEGFF(SDNode *Node, bool IsMasked) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 2; // Do not count VL and Chain.
  MVT VT = Node->getSimpleValueType(0);
  MVT XLenVT = Subtarget->getXLenVT();
  unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
  RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 7> Operands;
  if (IsMasked) {
    SmallVector<SDValue, 8> Regs(Node->op_begin() + CurOp,
                                 Node->op_begin() + CurOp + NF);
    SDValue MaskedOff = createTuple(*CurDAG, Regs, NF, LMUL);
    Operands.push_back(MaskedOff);
    CurOp += NF;
  }

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ false, Operands);

  const RISCV::VLSEGPseudo *P =
      RISCV::getVLSEGPseudo(NF, IsMasked, /*Strided*/ false, /*FF*/ true,
                            Log2SEW, static_cast<unsigned>(LMUL));
  MachineSDNode *Load = CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped,
                                               MVT::Other, MVT::Glue, Operands);
  SDNode *ReadVL = CurDAG->getMachineNode(RISCV::PseudoReadVL, DL, XLenVT,
                                          /*Glue*/ SDValue(Load, 2));

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I) {
    unsigned SubRegIdx = RISCVTargetLowering::getSubregIndexByMVT(VT, I);
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, SuperReg));
  }

  ReplaceUses(SDValue(Node, NF), SDValue(ReadVL, 0));   // VL
  ReplaceUses(SDValue(Node, NF + 1), SDValue(Load, 1)); // Chain
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVLXSEG(SDNode *Node, bool IsMasked,
                                     bool IsOrdered) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 1;
  MVT VT = Node->getSimpleValueType(0);
  unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
  RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;
  if (IsMasked) {
    SmallVector<SDValue, 8> Regs(Node->op_begin() + CurOp,
                                 Node->op_begin() + CurOp + NF);
    SDValue MaskedOff = createTuple(*CurDAG, Regs, NF, LMUL);
    Operands.push_back(MaskedOff);
    CurOp += NF;
  }

  MVT IndexVT;
  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ true, Operands, &IndexVT);

  assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
         "Element count mismatch");

  RISCVII::VLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
  unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
  const RISCV::VLXSEGPseudo *P = RISCV::getVLXSEGPseudo(
      NF, IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  MachineSDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I) {
    unsigned SubRegIdx = RISCVTargetLowering::getSubregIndexByMVT(VT, I);
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, SuperReg));
  }

  ReplaceUses(SDValue(Node, NF), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVSSEG(SDNode *Node, bool IsMasked,
                                    bool IsStrided) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumOperands() - 4;
  if (IsStrided)
    NF--;
  if (IsMasked)
    NF--;
  MVT VT = Node->getOperand(2)->getSimpleValueType(0);
  unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
  RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue StoreVal = createTuple(*CurDAG, Regs, NF, LMUL);

  SmallVector<SDValue, 8> Operands;
  Operands.push_back(StoreVal);
  unsigned CurOp = 2 + NF;

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                             Operands);

  const RISCV::VSSEGPseudo *P = RISCV::getVSSEGPseudo(
      NF, IsMasked, IsStrided, Log2SEW, static_cast<unsigned>(LMUL));
  MachineSDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

  ReplaceNode(Node, Store);
}

void RISCVDAGToDAGISel::selectVSXSEG(SDNode *Node, bool IsMasked,
                                     bool IsOrdered) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumOperands() - 5;
  if (IsMasked)
    --NF;
  MVT VT = Node->getOperand(2)->getSimpleValueType(0);
  unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
  RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue StoreVal = createTuple(*CurDAG, Regs, NF, LMUL);

  SmallVector<SDValue, 8> Operands;
  Operands.push_back(StoreVal);
  unsigned CurOp = 2 + NF;

  MVT IndexVT;
  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ true, Operands, &IndexVT);

  assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
         "Element count mismatch");

  RISCVII::VLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
  unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
  const RISCV::VSXSEGPseudo *P = RISCV::getVSXSEGPseudo(
      NF, IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  MachineSDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

  ReplaceNode(Node, Store);
}


void RISCVDAGToDAGISel::Select(SDNode *Node) {
  // If we have a custom node, we have already selected.
  if (Node->isMachineOpcode()) {
    LLVM_DEBUG(dbgs() << "== "; Node->dump(CurDAG); dbgs() << "\n");
    Node->setNodeId(-1);
    return;
  }

  // Instruction Selection not handled by the auto-generated tablegen selection
  // should be handled here.
  unsigned Opcode = Node->getOpcode();
  MVT XLenVT = Subtarget->getXLenVT();
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);

  switch (Opcode) {
  case ISD::Constant: {
    auto *ConstNode = cast<ConstantSDNode>(Node);
    if (VT == XLenVT && ConstNode->isNullValue()) {
      SDValue New =
          CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL, RISCV::X0, XLenVT);
      ReplaceNode(Node, New.getNode());
      return;
    }
    int64_t Imm = ConstNode->getSExtValue();
    // If the upper XLen-16 bits are not used, try to convert this to a simm12
    // by sign extending bit 15.
    if (isUInt<16>(Imm) && isInt<12>(SignExtend64(Imm, 16)) &&
        hasAllHUsers(Node))
      Imm = SignExtend64(Imm, 16);
    // If the upper 32-bits are not used try to convert this into a simm32 by
    // sign extending bit 32.
    if (!isInt<32>(Imm) && isUInt<32>(Imm) && hasAllWUsers(Node))
      Imm = SignExtend64(Imm, 32);

    ReplaceNode(Node, selectImm(CurDAG, DL, Imm, *Subtarget));
    return;
  }
  case ISD::FrameIndex: {
    SDValue Imm = CurDAG->getTargetConstant(0, DL, XLenVT);
    int FI = cast<FrameIndexSDNode>(Node)->getIndex();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, VT);
    ReplaceNode(Node, CurDAG->getMachineNode(RISCV::ADDI, DL, VT, TFI, Imm));
    return;
  }
  case ISD::SRL: {
    // We don't need this transform if zext.h is supported.
    if (Subtarget->hasStdExtZbb() || Subtarget->hasStdExtZbp())
      break;
    // Optimize (srl (and X, 0xffff), C) ->
    //          (srli (slli X, (XLen-16), (XLen-16) + C)
    // Taking into account that the 0xffff may have had lower bits unset by
    // SimplifyDemandedBits. This avoids materializing the 0xffff immediate.
    // This pattern occurs when type legalizing i16 right shifts.
    // FIXME: This could be extended to other AND masks.
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (N1C) {
      uint64_t ShAmt = N1C->getZExtValue();
      SDValue N0 = Node->getOperand(0);
      if (ShAmt < 16 && N0.getOpcode() == ISD::AND && N0.hasOneUse() &&
          isa<ConstantSDNode>(N0.getOperand(1))) {
        uint64_t Mask = N0.getConstantOperandVal(1);
        Mask |= maskTrailingOnes<uint64_t>(ShAmt);
        if (Mask == 0xffff) {
          unsigned LShAmt = Subtarget->getXLen() - 16;
          SDNode *SLLI =
              CurDAG->getMachineNode(RISCV::SLLI, DL, VT, N0->getOperand(0),
                                     CurDAG->getTargetConstant(LShAmt, DL, VT));
          SDNode *SRLI = CurDAG->getMachineNode(
              RISCV::SRLI, DL, VT, SDValue(SLLI, 0),
              CurDAG->getTargetConstant(LShAmt + ShAmt, DL, VT));
          ReplaceNode(Node, SRLI);
          return;
        }
      }
    }

    break;
  }
  case ISD::AND: {
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;

    SDValue N0 = Node->getOperand(0);

    bool LeftShift = N0.getOpcode() == ISD::SHL;
    if (!LeftShift && N0.getOpcode() != ISD::SRL)
      break;

    auto *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (!C)
      break;
    uint64_t C2 = C->getZExtValue();
    unsigned XLen = Subtarget->getXLen();
    if (!C2 || C2 >= XLen)
      break;

    uint64_t C1 = N1C->getZExtValue();

    // Keep track of whether this is a andi, zext.h, or zext.w.
    bool ZExtOrANDI = isInt<12>(N1C->getSExtValue());
    if (C1 == UINT64_C(0xFFFF) &&
        (Subtarget->hasStdExtZbb() || Subtarget->hasStdExtZbp()))
      ZExtOrANDI = true;
    if (C1 == UINT64_C(0xFFFFFFFF) && Subtarget->hasStdExtZba())
      ZExtOrANDI = true;

    // Clear irrelevant bits in the mask.
    if (LeftShift)
      C1 &= maskTrailingZeros<uint64_t>(C2);
    else
      C1 &= maskTrailingOnes<uint64_t>(XLen - C2);

    // Some transforms should only be done if the shift has a single use or
    // the AND would become (srli (slli X, 32), 32)
    bool OneUseOrZExtW = N0.hasOneUse() || C1 == UINT64_C(0xFFFFFFFF);

    SDValue X = N0.getOperand(0);

    // Turn (and (srl x, c2) c1) -> (srli (slli x, c3-c2), c3) if c1 is a mask
    // with c3 leading zeros.
    if (!LeftShift && isMask_64(C1)) {
      uint64_t C3 = XLen - (64 - countLeadingZeros(C1));
      if (C2 < C3) {
        // If the number of leading zeros is C2+32 this can be SRLIW.
        if (C2 + 32 == C3) {
          SDNode *SRLIW =
              CurDAG->getMachineNode(RISCV::SRLIW, DL, XLenVT, X,
                                     CurDAG->getTargetConstant(C2, DL, XLenVT));
          ReplaceNode(Node, SRLIW);
          return;
        }

        // (and (srl (sexti32 Y), c2), c1) -> (srliw (sraiw Y, 31), c3 - 32) if
        // c1 is a mask with c3 leading zeros and c2 >= 32 and c3-c2==1.
        //
        // This pattern occurs when (i32 (srl (sra 31), c3 - 32)) is type
        // legalized and goes through DAG combine.
        SDValue Y;
        if (C2 >= 32 && (C3 - C2) == 1 && N0.hasOneUse() &&
            selectSExti32(X, Y)) {
          SDNode *SRAIW =
              CurDAG->getMachineNode(RISCV::SRAIW, DL, XLenVT, Y,
                                     CurDAG->getTargetConstant(31, DL, XLenVT));
          SDNode *SRLIW = CurDAG->getMachineNode(
              RISCV::SRLIW, DL, XLenVT, SDValue(SRAIW, 0),
              CurDAG->getTargetConstant(C3 - 32, DL, XLenVT));
          ReplaceNode(Node, SRLIW);
          return;
        }

        // (srli (slli x, c3-c2), c3).
        if (OneUseOrZExtW && !ZExtOrANDI) {
          SDNode *SLLI = CurDAG->getMachineNode(
              RISCV::SLLI, DL, XLenVT, X,
              CurDAG->getTargetConstant(C3 - C2, DL, XLenVT));
          SDNode *SRLI =
              CurDAG->getMachineNode(RISCV::SRLI, DL, XLenVT, SDValue(SLLI, 0),
                                     CurDAG->getTargetConstant(C3, DL, XLenVT));
          ReplaceNode(Node, SRLI);
          return;
        }
      }
    }

    // Turn (and (shl x, c2) c1) -> (srli (slli c2+c3), c3) if c1 is a mask
    // shifted by c2 bits with c3 leading zeros.
    if (LeftShift && isShiftedMask_64(C1)) {
      uint64_t C3 = XLen - (64 - countLeadingZeros(C1));

      if (C2 + C3 < XLen &&
          C1 == (maskTrailingOnes<uint64_t>(XLen - (C2 + C3)) << C2)) {
        // Use slli.uw when possible.
        if ((XLen - (C2 + C3)) == 32 && Subtarget->hasStdExtZba()) {
          SDNode *SLLIUW =
              CurDAG->getMachineNode(RISCV::SLLIUW, DL, XLenVT, X,
                                     CurDAG->getTargetConstant(C2, DL, XLenVT));
          ReplaceNode(Node, SLLIUW);
          return;
        }

        // (srli (slli c2+c3), c3)
        if (OneUseOrZExtW && !ZExtOrANDI) {
          SDNode *SLLI = CurDAG->getMachineNode(
              RISCV::SLLI, DL, XLenVT, X,
              CurDAG->getTargetConstant(C2 + C3, DL, XLenVT));
          SDNode *SRLI =
              CurDAG->getMachineNode(RISCV::SRLI, DL, XLenVT, SDValue(SLLI, 0),
                                     CurDAG->getTargetConstant(C3, DL, XLenVT));
          ReplaceNode(Node, SRLI);
          return;
        }
      }
    }

    break;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo = Node->getConstantOperandVal(0);
    switch (IntNo) {
      // By default we do not custom select any intrinsic.
    default:
      break;
    case Intrinsic::riscv_vmsgeu:
    case Intrinsic::riscv_vmsge: {
      SDValue Src1 = Node->getOperand(1);
      SDValue Src2 = Node->getOperand(2);
      // Only custom select scalar second operand.
      if (Src2.getValueType() != XLenVT)
        break;
      // Small constants are handled with patterns.
      if (auto *C = dyn_cast<ConstantSDNode>(Src2)) {
        int64_t CVal = C->getSExtValue();
        if (CVal >= -15 && CVal <= 16)
          break;
      }
      bool IsUnsigned = IntNo == Intrinsic::riscv_vmsgeu;
      MVT Src1VT = Src1.getSimpleValueType();
      unsigned VMSLTOpcode, VMNANDOpcode;
      switch (RISCVTargetLowering::getLMUL(Src1VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
      case RISCVII::VLMUL::LMUL_F8:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_MF8 : RISCV::PseudoVMSLT_VX_MF8;
        VMNANDOpcode = RISCV::PseudoVMNAND_MM_MF8;
        break;
      case RISCVII::VLMUL::LMUL_F4:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_MF4 : RISCV::PseudoVMSLT_VX_MF4;
        VMNANDOpcode = RISCV::PseudoVMNAND_MM_MF4;
        break;
      case RISCVII::VLMUL::LMUL_F2:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_MF2 : RISCV::PseudoVMSLT_VX_MF2;
        VMNANDOpcode = RISCV::PseudoVMNAND_MM_MF2;
        break;
      case RISCVII::VLMUL::LMUL_1:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_M1 : RISCV::PseudoVMSLT_VX_M1;
        VMNANDOpcode = RISCV::PseudoVMNAND_MM_M1;
        break;
      case RISCVII::VLMUL::LMUL_2:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_M2 : RISCV::PseudoVMSLT_VX_M2;
        VMNANDOpcode = RISCV::PseudoVMNAND_MM_M2;
        break;
      case RISCVII::VLMUL::LMUL_4:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_M4 : RISCV::PseudoVMSLT_VX_M4;
        VMNANDOpcode = RISCV::PseudoVMNAND_MM_M4;
        break;
      case RISCVII::VLMUL::LMUL_8:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_M8 : RISCV::PseudoVMSLT_VX_M8;
        VMNANDOpcode = RISCV::PseudoVMNAND_MM_M8;
        break;
      }
      SDValue SEW = CurDAG->getTargetConstant(
          Log2_32(Src1VT.getScalarSizeInBits()), DL, XLenVT);
      SDValue VL;
      selectVLOp(Node->getOperand(3), VL);

      // Expand to
      // vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd
      SDValue Cmp = SDValue(
          CurDAG->getMachineNode(VMSLTOpcode, DL, VT, {Src1, Src2, VL, SEW}),
          0);
      ReplaceNode(Node, CurDAG->getMachineNode(VMNANDOpcode, DL, VT,
                                               {Cmp, Cmp, VL, SEW}));
      return;
    }
    case Intrinsic::riscv_vmsgeu_mask:
    case Intrinsic::riscv_vmsge_mask: {
      SDValue Src1 = Node->getOperand(2);
      SDValue Src2 = Node->getOperand(3);
      // Only custom select scalar second operand.
      if (Src2.getValueType() != XLenVT)
        break;
      // Small constants are handled with patterns.
      if (auto *C = dyn_cast<ConstantSDNode>(Src2)) {
        int64_t CVal = C->getSExtValue();
        if (CVal >= -15 && CVal <= 16)
          break;
      }
      bool IsUnsigned = IntNo == Intrinsic::riscv_vmsgeu_mask;
      MVT Src1VT = Src1.getSimpleValueType();
      unsigned VMSLTOpcode, VMSLTMaskOpcode, VMXOROpcode, VMANDNOTOpcode;
      switch (RISCVTargetLowering::getLMUL(Src1VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
      case RISCVII::VLMUL::LMUL_F8:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_MF8 : RISCV::PseudoVMSLT_VX_MF8;
        VMSLTMaskOpcode = IsUnsigned ? RISCV::PseudoVMSLTU_VX_MF8_MASK
                                     : RISCV::PseudoVMSLT_VX_MF8_MASK;
        break;
      case RISCVII::VLMUL::LMUL_F4:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_MF4 : RISCV::PseudoVMSLT_VX_MF4;
        VMSLTMaskOpcode = IsUnsigned ? RISCV::PseudoVMSLTU_VX_MF4_MASK
                                     : RISCV::PseudoVMSLT_VX_MF4_MASK;
        break;
      case RISCVII::VLMUL::LMUL_F2:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_MF2 : RISCV::PseudoVMSLT_VX_MF2;
        VMSLTMaskOpcode = IsUnsigned ? RISCV::PseudoVMSLTU_VX_MF2_MASK
                                     : RISCV::PseudoVMSLT_VX_MF2_MASK;
        break;
      case RISCVII::VLMUL::LMUL_1:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_M1 : RISCV::PseudoVMSLT_VX_M1;
        VMSLTMaskOpcode = IsUnsigned ? RISCV::PseudoVMSLTU_VX_M1_MASK
                                     : RISCV::PseudoVMSLT_VX_M1_MASK;
        break;
      case RISCVII::VLMUL::LMUL_2:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_M2 : RISCV::PseudoVMSLT_VX_M2;
        VMSLTMaskOpcode = IsUnsigned ? RISCV::PseudoVMSLTU_VX_M2_MASK
                                     : RISCV::PseudoVMSLT_VX_M2_MASK;
        break;
      case RISCVII::VLMUL::LMUL_4:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_M4 : RISCV::PseudoVMSLT_VX_M4;
        VMSLTMaskOpcode = IsUnsigned ? RISCV::PseudoVMSLTU_VX_M4_MASK
                                     : RISCV::PseudoVMSLT_VX_M4_MASK;
        break;
      case RISCVII::VLMUL::LMUL_8:
        VMSLTOpcode =
            IsUnsigned ? RISCV::PseudoVMSLTU_VX_M8 : RISCV::PseudoVMSLT_VX_M8;
        VMSLTMaskOpcode = IsUnsigned ? RISCV::PseudoVMSLTU_VX_M8_MASK
                                     : RISCV::PseudoVMSLT_VX_M8_MASK;
        break;
      }
      // Mask operations use the LMUL from the mask type.
      switch (RISCVTargetLowering::getLMUL(VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
      case RISCVII::VLMUL::LMUL_F8:
        VMXOROpcode = RISCV::PseudoVMXOR_MM_MF8;
        VMANDNOTOpcode = RISCV::PseudoVMANDNOT_MM_MF8;
        break;
      case RISCVII::VLMUL::LMUL_F4:
        VMXOROpcode = RISCV::PseudoVMXOR_MM_MF4;
        VMANDNOTOpcode = RISCV::PseudoVMANDNOT_MM_MF4;
        break;
      case RISCVII::VLMUL::LMUL_F2:
        VMXOROpcode = RISCV::PseudoVMXOR_MM_MF2;
        VMANDNOTOpcode = RISCV::PseudoVMANDNOT_MM_MF2;
        break;
      case RISCVII::VLMUL::LMUL_1:
        VMXOROpcode = RISCV::PseudoVMXOR_MM_M1;
        VMANDNOTOpcode = RISCV::PseudoVMANDNOT_MM_M1;
        break;
      case RISCVII::VLMUL::LMUL_2:
        VMXOROpcode = RISCV::PseudoVMXOR_MM_M2;
        VMANDNOTOpcode = RISCV::PseudoVMANDNOT_MM_M2;
        break;
      case RISCVII::VLMUL::LMUL_4:
        VMXOROpcode = RISCV::PseudoVMXOR_MM_M4;
        VMANDNOTOpcode = RISCV::PseudoVMANDNOT_MM_M4;
        break;
      case RISCVII::VLMUL::LMUL_8:
        VMXOROpcode = RISCV::PseudoVMXOR_MM_M8;
        VMANDNOTOpcode = RISCV::PseudoVMANDNOT_MM_M8;
        break;
      }
      SDValue SEW = CurDAG->getTargetConstant(
          Log2_32(Src1VT.getScalarSizeInBits()), DL, XLenVT);
      SDValue MaskSEW = CurDAG->getTargetConstant(0, DL, XLenVT);
      SDValue VL;
      selectVLOp(Node->getOperand(5), VL);
      SDValue MaskedOff = Node->getOperand(1);
      SDValue Mask = Node->getOperand(4);
      // If the MaskedOff value and the Mask are the same value use
      // vmslt{u}.vx vt, va, x;  vmandnot.mm vd, vd, vt
      // This avoids needing to copy v0 to vd before starting the next sequence.
      if (Mask == MaskedOff) {
        SDValue Cmp = SDValue(
            CurDAG->getMachineNode(VMSLTOpcode, DL, VT, {Src1, Src2, VL, SEW}),
            0);
        ReplaceNode(Node, CurDAG->getMachineNode(VMANDNOTOpcode, DL, VT,
                                                 {Mask, Cmp, VL, MaskSEW}));
        return;
      }

      // Mask needs to be copied to V0.
      SDValue Chain = CurDAG->getCopyToReg(CurDAG->getEntryNode(), DL,
                                           RISCV::V0, Mask, SDValue());
      SDValue Glue = Chain.getValue(1);
      SDValue V0 = CurDAG->getRegister(RISCV::V0, VT);

      // Otherwise use
      // vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0
      SDValue Cmp = SDValue(
          CurDAG->getMachineNode(VMSLTMaskOpcode, DL, VT,
                                 {MaskedOff, Src1, Src2, V0, VL, SEW, Glue}),
          0);
      ReplaceNode(Node, CurDAG->getMachineNode(VMXOROpcode, DL, VT,
                                               {Cmp, Mask, VL, MaskSEW}));
      return;
    }
    }
    break;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    switch (IntNo) {
      // By default we do not custom select any intrinsic.
    default:
      break;

    case Intrinsic::riscv_vsetvli:
    case Intrinsic::riscv_vsetvlimax: {
      if (!Subtarget->hasStdExtV())
        break;

      bool VLMax = IntNo == Intrinsic::riscv_vsetvlimax;
      unsigned Offset = VLMax ? 2 : 3;

      assert(Node->getNumOperands() == Offset + 2 &&
             "Unexpected number of operands");

      unsigned SEW =
          RISCVVType::decodeVSEW(Node->getConstantOperandVal(Offset) & 0x7);
      RISCVII::VLMUL VLMul = static_cast<RISCVII::VLMUL>(
          Node->getConstantOperandVal(Offset + 1) & 0x7);

      unsigned VTypeI = RISCVVType::encodeVTYPE(
          VLMul, SEW, /*TailAgnostic*/ true, /*MaskAgnostic*/ false);
      SDValue VTypeIOp = CurDAG->getTargetConstant(VTypeI, DL, XLenVT);

      SDValue VLOperand;
      if (VLMax) {
        VLOperand = CurDAG->getRegister(RISCV::X0, XLenVT);
      } else {
        VLOperand = Node->getOperand(2);

        if (auto *C = dyn_cast<ConstantSDNode>(VLOperand)) {
          uint64_t AVL = C->getZExtValue();
          if (isUInt<5>(AVL)) {
            SDValue VLImm = CurDAG->getTargetConstant(AVL, DL, XLenVT);
            ReplaceNode(
                Node, CurDAG->getMachineNode(RISCV::PseudoVSETIVLI, DL, XLenVT,
                                             MVT::Other, VLImm, VTypeIOp,
                                             /* Chain */ Node->getOperand(0)));
            return;
          }
        }
      }

      ReplaceNode(Node,
                  CurDAG->getMachineNode(RISCV::PseudoVSETVLI, DL, XLenVT,
                                         MVT::Other, VLOperand, VTypeIOp,
                                         /* Chain */ Node->getOperand(0)));
      return;
    }
    case Intrinsic::riscv_vlseg2:
    case Intrinsic::riscv_vlseg3:
    case Intrinsic::riscv_vlseg4:
    case Intrinsic::riscv_vlseg5:
    case Intrinsic::riscv_vlseg6:
    case Intrinsic::riscv_vlseg7:
    case Intrinsic::riscv_vlseg8: {
      selectVLSEG(Node, /*IsMasked*/ false, /*IsStrided*/ false);
      return;
    }
    case Intrinsic::riscv_vlseg2_mask:
    case Intrinsic::riscv_vlseg3_mask:
    case Intrinsic::riscv_vlseg4_mask:
    case Intrinsic::riscv_vlseg5_mask:
    case Intrinsic::riscv_vlseg6_mask:
    case Intrinsic::riscv_vlseg7_mask:
    case Intrinsic::riscv_vlseg8_mask: {
      selectVLSEG(Node, /*IsMasked*/ true, /*IsStrided*/ false);
      return;
    }
    case Intrinsic::riscv_vlsseg2:
    case Intrinsic::riscv_vlsseg3:
    case Intrinsic::riscv_vlsseg4:
    case Intrinsic::riscv_vlsseg5:
    case Intrinsic::riscv_vlsseg6:
    case Intrinsic::riscv_vlsseg7:
    case Intrinsic::riscv_vlsseg8: {
      selectVLSEG(Node, /*IsMasked*/ false, /*IsStrided*/ true);
      return;
    }
    case Intrinsic::riscv_vlsseg2_mask:
    case Intrinsic::riscv_vlsseg3_mask:
    case Intrinsic::riscv_vlsseg4_mask:
    case Intrinsic::riscv_vlsseg5_mask:
    case Intrinsic::riscv_vlsseg6_mask:
    case Intrinsic::riscv_vlsseg7_mask:
    case Intrinsic::riscv_vlsseg8_mask: {
      selectVLSEG(Node, /*IsMasked*/ true, /*IsStrided*/ true);
      return;
    }
    case Intrinsic::riscv_vloxseg2:
    case Intrinsic::riscv_vloxseg3:
    case Intrinsic::riscv_vloxseg4:
    case Intrinsic::riscv_vloxseg5:
    case Intrinsic::riscv_vloxseg6:
    case Intrinsic::riscv_vloxseg7:
    case Intrinsic::riscv_vloxseg8:
      selectVLXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ true);
      return;
    case Intrinsic::riscv_vluxseg2:
    case Intrinsic::riscv_vluxseg3:
    case Intrinsic::riscv_vluxseg4:
    case Intrinsic::riscv_vluxseg5:
    case Intrinsic::riscv_vluxseg6:
    case Intrinsic::riscv_vluxseg7:
    case Intrinsic::riscv_vluxseg8:
      selectVLXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ false);
      return;
    case Intrinsic::riscv_vloxseg2_mask:
    case Intrinsic::riscv_vloxseg3_mask:
    case Intrinsic::riscv_vloxseg4_mask:
    case Intrinsic::riscv_vloxseg5_mask:
    case Intrinsic::riscv_vloxseg6_mask:
    case Intrinsic::riscv_vloxseg7_mask:
    case Intrinsic::riscv_vloxseg8_mask:
      selectVLXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ true);
      return;
    case Intrinsic::riscv_vluxseg2_mask:
    case Intrinsic::riscv_vluxseg3_mask:
    case Intrinsic::riscv_vluxseg4_mask:
    case Intrinsic::riscv_vluxseg5_mask:
    case Intrinsic::riscv_vluxseg6_mask:
    case Intrinsic::riscv_vluxseg7_mask:
    case Intrinsic::riscv_vluxseg8_mask:
      selectVLXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ false);
      return;
    case Intrinsic::riscv_vlseg8ff:
    case Intrinsic::riscv_vlseg7ff:
    case Intrinsic::riscv_vlseg6ff:
    case Intrinsic::riscv_vlseg5ff:
    case Intrinsic::riscv_vlseg4ff:
    case Intrinsic::riscv_vlseg3ff:
    case Intrinsic::riscv_vlseg2ff: {
      selectVLSEGFF(Node, /*IsMasked*/ false);
      return;
    }
    case Intrinsic::riscv_vlseg8ff_mask:
    case Intrinsic::riscv_vlseg7ff_mask:
    case Intrinsic::riscv_vlseg6ff_mask:
    case Intrinsic::riscv_vlseg5ff_mask:
    case Intrinsic::riscv_vlseg4ff_mask:
    case Intrinsic::riscv_vlseg3ff_mask:
    case Intrinsic::riscv_vlseg2ff_mask: {
      selectVLSEGFF(Node, /*IsMasked*/ true);
      return;
    }
    case Intrinsic::riscv_vloxei:
    case Intrinsic::riscv_vloxei_mask:
    case Intrinsic::riscv_vluxei:
    case Intrinsic::riscv_vluxei_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vloxei_mask ||
                      IntNo == Intrinsic::riscv_vluxei_mask;
      bool IsOrdered = IntNo == Intrinsic::riscv_vloxei ||
                       IntNo == Intrinsic::riscv_vloxei_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      if (IsMasked)
        Operands.push_back(Node->getOperand(CurOp++));

      MVT IndexVT;
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ true, Operands,
                                 &IndexVT);

      assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
             "Element count mismatch");

      RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      RISCVII::VLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
      unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
      const RISCV::VLX_VSXPseudo *P = RISCV::getVLXPseudo(
          IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
          static_cast<unsigned>(IndexLMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    case Intrinsic::riscv_vle1:
    case Intrinsic::riscv_vle:
    case Intrinsic::riscv_vle_mask:
    case Intrinsic::riscv_vlse:
    case Intrinsic::riscv_vlse_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vle_mask ||
                      IntNo == Intrinsic::riscv_vlse_mask;
      bool IsStrided =
          IntNo == Intrinsic::riscv_vlse || IntNo == Intrinsic::riscv_vlse_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      if (IsMasked)
        Operands.push_back(Node->getOperand(CurOp++));

      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                                 Operands);

      RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      const RISCV::VLEPseudo *P =
          RISCV::getVLEPseudo(IsMasked, IsStrided, /*FF*/ false, Log2SEW,
                              static_cast<unsigned>(LMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    case Intrinsic::riscv_vleff:
    case Intrinsic::riscv_vleff_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vleff_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 7> Operands;
      if (IsMasked)
        Operands.push_back(Node->getOperand(CurOp++));

      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ false, Operands);

      RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      const RISCV::VLEPseudo *P =
          RISCV::getVLEPseudo(IsMasked, /*Strided*/ false, /*FF*/ true, Log2SEW,
                              static_cast<unsigned>(LMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0),
                                 MVT::Other, MVT::Glue, Operands);
      SDNode *ReadVL = CurDAG->getMachineNode(RISCV::PseudoReadVL, DL, XLenVT,
                                              /*Glue*/ SDValue(Load, 2));

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceUses(SDValue(Node, 0), SDValue(Load, 0));
      ReplaceUses(SDValue(Node, 1), SDValue(ReadVL, 0)); // VL
      ReplaceUses(SDValue(Node, 2), SDValue(Load, 1));   // Chain
      CurDAG->RemoveDeadNode(Node);
      return;
    }
    }
    break;
  }
  case ISD::INTRINSIC_VOID: {
    unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    switch (IntNo) {
    case Intrinsic::riscv_vsseg2:
    case Intrinsic::riscv_vsseg3:
    case Intrinsic::riscv_vsseg4:
    case Intrinsic::riscv_vsseg5:
    case Intrinsic::riscv_vsseg6:
    case Intrinsic::riscv_vsseg7:
    case Intrinsic::riscv_vsseg8: {
      selectVSSEG(Node, /*IsMasked*/ false, /*IsStrided*/ false);
      return;
    }
    case Intrinsic::riscv_vsseg2_mask:
    case Intrinsic::riscv_vsseg3_mask:
    case Intrinsic::riscv_vsseg4_mask:
    case Intrinsic::riscv_vsseg5_mask:
    case Intrinsic::riscv_vsseg6_mask:
    case Intrinsic::riscv_vsseg7_mask:
    case Intrinsic::riscv_vsseg8_mask: {
      selectVSSEG(Node, /*IsMasked*/ true, /*IsStrided*/ false);
      return;
    }
    case Intrinsic::riscv_vssseg2:
    case Intrinsic::riscv_vssseg3:
    case Intrinsic::riscv_vssseg4:
    case Intrinsic::riscv_vssseg5:
    case Intrinsic::riscv_vssseg6:
    case Intrinsic::riscv_vssseg7:
    case Intrinsic::riscv_vssseg8: {
      selectVSSEG(Node, /*IsMasked*/ false, /*IsStrided*/ true);
      return;
    }
    case Intrinsic::riscv_vssseg2_mask:
    case Intrinsic::riscv_vssseg3_mask:
    case Intrinsic::riscv_vssseg4_mask:
    case Intrinsic::riscv_vssseg5_mask:
    case Intrinsic::riscv_vssseg6_mask:
    case Intrinsic::riscv_vssseg7_mask:
    case Intrinsic::riscv_vssseg8_mask: {
      selectVSSEG(Node, /*IsMasked*/ true, /*IsStrided*/ true);
      return;
    }
    case Intrinsic::riscv_vsoxseg2:
    case Intrinsic::riscv_vsoxseg3:
    case Intrinsic::riscv_vsoxseg4:
    case Intrinsic::riscv_vsoxseg5:
    case Intrinsic::riscv_vsoxseg6:
    case Intrinsic::riscv_vsoxseg7:
    case Intrinsic::riscv_vsoxseg8:
      selectVSXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ true);
      return;
    case Intrinsic::riscv_vsuxseg2:
    case Intrinsic::riscv_vsuxseg3:
    case Intrinsic::riscv_vsuxseg4:
    case Intrinsic::riscv_vsuxseg5:
    case Intrinsic::riscv_vsuxseg6:
    case Intrinsic::riscv_vsuxseg7:
    case Intrinsic::riscv_vsuxseg8:
      selectVSXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ false);
      return;
    case Intrinsic::riscv_vsoxseg2_mask:
    case Intrinsic::riscv_vsoxseg3_mask:
    case Intrinsic::riscv_vsoxseg4_mask:
    case Intrinsic::riscv_vsoxseg5_mask:
    case Intrinsic::riscv_vsoxseg6_mask:
    case Intrinsic::riscv_vsoxseg7_mask:
    case Intrinsic::riscv_vsoxseg8_mask:
      selectVSXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ true);
      return;
    case Intrinsic::riscv_vsuxseg2_mask:
    case Intrinsic::riscv_vsuxseg3_mask:
    case Intrinsic::riscv_vsuxseg4_mask:
    case Intrinsic::riscv_vsuxseg5_mask:
    case Intrinsic::riscv_vsuxseg6_mask:
    case Intrinsic::riscv_vsuxseg7_mask:
    case Intrinsic::riscv_vsuxseg8_mask:
      selectVSXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ false);
      return;
    case Intrinsic::riscv_vsoxei:
    case Intrinsic::riscv_vsoxei_mask:
    case Intrinsic::riscv_vsuxei:
    case Intrinsic::riscv_vsuxei_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vsoxei_mask ||
                      IntNo == Intrinsic::riscv_vsuxei_mask;
      bool IsOrdered = IntNo == Intrinsic::riscv_vsoxei ||
                       IntNo == Intrinsic::riscv_vsoxei_mask;

      MVT VT = Node->getOperand(2)->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      Operands.push_back(Node->getOperand(CurOp++)); // Store value.

      MVT IndexVT;
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ true, Operands,
                                 &IndexVT);

      assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
             "Element count mismatch");

      RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      RISCVII::VLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
      unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
      const RISCV::VLX_VSXPseudo *P = RISCV::getVSXPseudo(
          IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
          static_cast<unsigned>(IndexLMUL));
      MachineSDNode *Store =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

      ReplaceNode(Node, Store);
      return;
    }
    case Intrinsic::riscv_vse1:
    case Intrinsic::riscv_vse:
    case Intrinsic::riscv_vse_mask:
    case Intrinsic::riscv_vsse:
    case Intrinsic::riscv_vsse_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vse_mask ||
                      IntNo == Intrinsic::riscv_vsse_mask;
      bool IsStrided =
          IntNo == Intrinsic::riscv_vsse || IntNo == Intrinsic::riscv_vsse_mask;

      MVT VT = Node->getOperand(2)->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      Operands.push_back(Node->getOperand(CurOp++)); // Store value.

      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                                 Operands);

      RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      const RISCV::VSEPseudo *P = RISCV::getVSEPseudo(
          IsMasked, IsStrided, Log2SEW, static_cast<unsigned>(LMUL));
      MachineSDNode *Store =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);
      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

      ReplaceNode(Node, Store);
      return;
    }
    }
    break;
  }
  case ISD::BITCAST: {
    MVT SrcVT = Node->getOperand(0).getSimpleValueType();
    // Just drop bitcasts between vectors if both are fixed or both are
    // scalable.
    if ((VT.isScalableVector() && SrcVT.isScalableVector()) ||
        (VT.isFixedLengthVector() && SrcVT.isFixedLengthVector())) {
      ReplaceUses(SDValue(Node, 0), Node->getOperand(0));
      CurDAG->RemoveDeadNode(Node);
      return;
    }
    break;
  }
  case ISD::INSERT_SUBVECTOR: {
    SDValue V = Node->getOperand(0);
    SDValue SubV = Node->getOperand(1);
    SDLoc DL(SubV);
    auto Idx = Node->getConstantOperandVal(2);
    MVT SubVecVT = SubV.getSimpleValueType();

    const RISCVTargetLowering &TLI = *Subtarget->getTargetLowering();
    MVT SubVecContainerVT = SubVecVT;
    // Establish the correct scalable-vector types for any fixed-length type.
    if (SubVecVT.isFixedLengthVector())
      SubVecContainerVT = TLI.getContainerForFixedLengthVector(SubVecVT);
    if (VT.isFixedLengthVector())
      VT = TLI.getContainerForFixedLengthVector(VT);

    const auto *TRI = Subtarget->getRegisterInfo();
    unsigned SubRegIdx;
    std::tie(SubRegIdx, Idx) =
        RISCVTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
            VT, SubVecContainerVT, Idx, TRI);

    // If the Idx hasn't been completely eliminated then this is a subvector
    // insert which doesn't naturally align to a vector register. These must
    // be handled using instructions to manipulate the vector registers.
    if (Idx != 0)
      break;

    RISCVII::VLMUL SubVecLMUL = RISCVTargetLowering::getLMUL(SubVecContainerVT);
    bool IsSubVecPartReg = SubVecLMUL == RISCVII::VLMUL::LMUL_F2 ||
                           SubVecLMUL == RISCVII::VLMUL::LMUL_F4 ||
                           SubVecLMUL == RISCVII::VLMUL::LMUL_F8;
    (void)IsSubVecPartReg; // Silence unused variable warning without asserts.
    assert((!IsSubVecPartReg || V.isUndef()) &&
           "Expecting lowering to have created legal INSERT_SUBVECTORs when "
           "the subvector is smaller than a full-sized register");

    // If we haven't set a SubRegIdx, then we must be going between
    // equally-sized LMUL groups (e.g. VR -> VR). This can be done as a copy.
    if (SubRegIdx == RISCV::NoSubRegister) {
      unsigned InRegClassID = RISCVTargetLowering::getRegClassIDForVecVT(VT);
      assert(RISCVTargetLowering::getRegClassIDForVecVT(SubVecContainerVT) ==
                 InRegClassID &&
             "Unexpected subvector extraction");
      SDValue RC = CurDAG->getTargetConstant(InRegClassID, DL, XLenVT);
      SDNode *NewNode = CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS,
                                               DL, VT, SubV, RC);
      ReplaceNode(Node, NewNode);
      return;
    }

    SDValue Insert = CurDAG->getTargetInsertSubreg(SubRegIdx, DL, VT, V, SubV);
    ReplaceNode(Node, Insert.getNode());
    return;
  }
  case ISD::EXTRACT_SUBVECTOR: {
    SDValue V = Node->getOperand(0);
    auto Idx = Node->getConstantOperandVal(1);
    MVT InVT = V.getSimpleValueType();
    SDLoc DL(V);

    const RISCVTargetLowering &TLI = *Subtarget->getTargetLowering();
    MVT SubVecContainerVT = VT;
    // Establish the correct scalable-vector types for any fixed-length type.
    if (VT.isFixedLengthVector())
      SubVecContainerVT = TLI.getContainerForFixedLengthVector(VT);
    if (InVT.isFixedLengthVector())
      InVT = TLI.getContainerForFixedLengthVector(InVT);

    const auto *TRI = Subtarget->getRegisterInfo();
    unsigned SubRegIdx;
    std::tie(SubRegIdx, Idx) =
        RISCVTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
            InVT, SubVecContainerVT, Idx, TRI);

    // If the Idx hasn't been completely eliminated then this is a subvector
    // extract which doesn't naturally align to a vector register. These must
    // be handled using instructions to manipulate the vector registers.
    if (Idx != 0)
      break;

    // If we haven't set a SubRegIdx, then we must be going between
    // equally-sized LMUL types (e.g. VR -> VR). This can be done as a copy.
    if (SubRegIdx == RISCV::NoSubRegister) {
      unsigned InRegClassID = RISCVTargetLowering::getRegClassIDForVecVT(InVT);
      assert(RISCVTargetLowering::getRegClassIDForVecVT(SubVecContainerVT) ==
                 InRegClassID &&
             "Unexpected subvector extraction");
      SDValue RC = CurDAG->getTargetConstant(InRegClassID, DL, XLenVT);
      SDNode *NewNode =
          CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, DL, VT, V, RC);
      ReplaceNode(Node, NewNode);
      return;
    }

    SDValue Extract = CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, V);
    ReplaceNode(Node, Extract.getNode());
    return;
  }
  case RISCVISD::VMV_V_X_VL:
  case RISCVISD::VFMV_V_F_VL: {
    // Try to match splat of a scalar load to a strided load with stride of x0.
    SDValue Src = Node->getOperand(0);
    auto *Ld = dyn_cast<LoadSDNode>(Src);
    if (!Ld)
      break;
    EVT MemVT = Ld->getMemoryVT();
    // The memory VT should be the same size as the element type.
    if (MemVT.getStoreSize() != VT.getVectorElementType().getStoreSize())
      break;
    if (!IsProfitableToFold(Src, Node, Node) ||
        !IsLegalToFold(Src, Node, Node, TM.getOptLevel()))
      break;

    SDValue VL;
    selectVLOp(Node->getOperand(1), VL);

    unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
    SDValue SEW = CurDAG->getTargetConstant(Log2SEW, DL, XLenVT);

    SDValue Operands[] = {Ld->getBasePtr(),
                          CurDAG->getRegister(RISCV::X0, XLenVT), VL, SEW,
                          Ld->getChain()};

    RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
    const RISCV::VLEPseudo *P = RISCV::getVLEPseudo(
        /*IsMasked*/ false, /*IsStrided*/ true, /*FF*/ false, Log2SEW,
        static_cast<unsigned>(LMUL));
    MachineSDNode *Load =
        CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

    if (auto *MemOp = dyn_cast<MemSDNode>(Node))
      CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

    ReplaceNode(Node, Load);
    return;
  }
  }

  // Select the default instruction.
  SelectCode(Node);
}

bool RISCVDAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, unsigned ConstraintID, std::vector<SDValue> &OutOps) {
  switch (ConstraintID) {
  case InlineAsm::Constraint_m:
    // We just support simple memory operands that have a single address
    // operand and need no special handling.
    OutOps.push_back(Op);
    return false;
  case InlineAsm::Constraint_A:
    OutOps.push_back(Op);
    return false;
  default:
    break;
  }

  return true;
}

bool RISCVDAGToDAGISel::SelectAddrFI(SDValue Addr, SDValue &Base) {
  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
    return true;
  }
  return false;
}

bool RISCVDAGToDAGISel::SelectBaseAddr(SDValue Addr, SDValue &Base) {
  // If this is FrameIndex, select it directly. Otherwise just let it get
  // selected to a register independently.
  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr))
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
  else
    Base = Addr;
  return true;
}

bool RISCVDAGToDAGISel::selectShiftMask(SDValue N, unsigned ShiftWidth,
                                        SDValue &ShAmt) {
  // Shift instructions on RISCV only read the lower 5 or 6 bits of the shift
  // amount. If there is an AND on the shift amount, we can bypass it if it
  // doesn't affect any of those bits.
  if (N.getOpcode() == ISD::AND && isa<ConstantSDNode>(N.getOperand(1))) {
    const APInt &AndMask = N->getConstantOperandAPInt(1);

    // Since the max shift amount is a power of 2 we can subtract 1 to make a
    // mask that covers the bits needed to represent all shift amounts.
    assert(isPowerOf2_32(ShiftWidth) && "Unexpected max shift amount!");
    APInt ShMask(AndMask.getBitWidth(), ShiftWidth - 1);

    if (ShMask.isSubsetOf(AndMask)) {
      ShAmt = N.getOperand(0);
      return true;
    }

    // SimplifyDemandedBits may have optimized the mask so try restoring any
    // bits that are known zero.
    KnownBits Known = CurDAG->computeKnownBits(N->getOperand(0));
    if (ShMask.isSubsetOf(AndMask | Known.Zero)) {
      ShAmt = N.getOperand(0);
      return true;
    }
  }

  ShAmt = N;
  return true;
}

bool RISCVDAGToDAGISel::selectSExti32(SDValue N, SDValue &Val) {
  if (N.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      cast<VTSDNode>(N.getOperand(1))->getVT() == MVT::i32) {
    Val = N.getOperand(0);
    return true;
  }
  MVT VT = N.getSimpleValueType();
  if (CurDAG->ComputeNumSignBits(N) > (VT.getSizeInBits() - 32)) {
    Val = N;
    return true;
  }

  return false;
}

bool RISCVDAGToDAGISel::selectZExti32(SDValue N, SDValue &Val) {
  if (N.getOpcode() == ISD::AND) {
    auto *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (C && C->getZExtValue() == UINT64_C(0xFFFFFFFF)) {
      Val = N.getOperand(0);
      return true;
    }
  }
  MVT VT = N.getSimpleValueType();
  APInt Mask = APInt::getHighBitsSet(VT.getSizeInBits(), 32);
  if (CurDAG->MaskedValueIsZero(N, Mask)) {
    Val = N;
    return true;
  }

  return false;
}

// Return true if all users of this SDNode* only consume the lower \p Bits.
// This can be used to form W instructions for add/sub/mul/shl even when the
// root isn't a sext_inreg. This can allow the ADDW/SUBW/MULW/SLLIW to CSE if
// SimplifyDemandedBits has made it so some users see a sext_inreg and some
// don't. The sext_inreg+add/sub/mul/shl will get selected, but still leave
// the add/sub/mul/shl to become non-W instructions. By checking the users we
// may be able to use a W instruction and CSE with the other instruction if
// this has happened. We could try to detect that the CSE opportunity exists
// before doing this, but that would be more complicated.
// TODO: Does this need to look through AND/OR/XOR to their users to find more
// opportunities.
bool RISCVDAGToDAGISel::hasAllNBitUsers(SDNode *Node, unsigned Bits) const {
  assert((Node->getOpcode() == ISD::ADD || Node->getOpcode() == ISD::SUB ||
          Node->getOpcode() == ISD::MUL || Node->getOpcode() == ISD::SHL ||
          isa<ConstantSDNode>(Node)) &&
         "Unexpected opcode");

  for (auto UI = Node->use_begin(), UE = Node->use_end(); UI != UE; ++UI) {
    SDNode *User = *UI;
    // Users of this node should have already been instruction selected
    if (!User->isMachineOpcode())
      return false;

    // TODO: Add more opcodes?
    switch (User->getMachineOpcode()) {
    default:
      return false;
    case RISCV::ADDW:
    case RISCV::ADDIW:
    case RISCV::SUBW:
    case RISCV::MULW:
    case RISCV::SLLW:
    case RISCV::SLLIW:
    case RISCV::SRAW:
    case RISCV::SRAIW:
    case RISCV::SRLW:
    case RISCV::SRLIW:
    case RISCV::DIVW:
    case RISCV::DIVUW:
    case RISCV::REMW:
    case RISCV::REMUW:
    case RISCV::ROLW:
    case RISCV::RORW:
    case RISCV::RORIW:
    case RISCV::CLZW:
    case RISCV::CTZW:
    case RISCV::CPOPW:
    case RISCV::SLLIUW:
      if (Bits < 32)
        return false;
      break;
    case RISCV::SLLI:
      // SLLI only uses the lower (XLen - ShAmt) bits.
      if (Bits < Subtarget->getXLen() - User->getConstantOperandVal(1))
        return false;
      break;
    case RISCV::ADDUW:
    case RISCV::SH1ADDUW:
    case RISCV::SH2ADDUW:
    case RISCV::SH3ADDUW:
      // The first operand to add.uw/shXadd.uw is implicitly zero extended from
      // 32 bits.
      if (UI.getOperandNo() != 0 || Bits < 32)
        return false;
      break;
    case RISCV::SB:
      if (UI.getOperandNo() != 0 || Bits < 8)
        return false;
      break;
    case RISCV::SH:
      if (UI.getOperandNo() != 0 || Bits < 16)
        return false;
      break;
    case RISCV::SW:
      if (UI.getOperandNo() != 0 || Bits < 32)
        return false;
      break;
    }
  }

  return true;
}

// Select VL as a 5 bit immediate or a value that will become a register. This
// allows us to choose betwen VSETIVLI or VSETVLI later.
bool RISCVDAGToDAGISel::selectVLOp(SDValue N, SDValue &VL) {
  auto *C = dyn_cast<ConstantSDNode>(N);
  if (C && isUInt<5>(C->getZExtValue()))
    VL = CurDAG->getTargetConstant(C->getZExtValue(), SDLoc(N),
                                   N->getValueType(0));
  else
    VL = N;

  return true;
}

bool RISCVDAGToDAGISel::selectVSplat(SDValue N, SDValue &SplatVal) {
  if (N.getOpcode() != ISD::SPLAT_VECTOR &&
      N.getOpcode() != RISCVISD::SPLAT_VECTOR_I64 &&
      N.getOpcode() != RISCVISD::VMV_V_X_VL)
    return false;
  SplatVal = N.getOperand(0);
  return true;
}

using ValidateFn = bool (*)(int64_t);

static bool selectVSplatSimmHelper(SDValue N, SDValue &SplatVal,
                                   SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   ValidateFn ValidateImm) {
  if ((N.getOpcode() != ISD::SPLAT_VECTOR &&
       N.getOpcode() != RISCVISD::SPLAT_VECTOR_I64 &&
       N.getOpcode() != RISCVISD::VMV_V_X_VL) ||
      !isa<ConstantSDNode>(N.getOperand(0)))
    return false;

  int64_t SplatImm = cast<ConstantSDNode>(N.getOperand(0))->getSExtValue();

  // ISD::SPLAT_VECTOR, RISCVISD::SPLAT_VECTOR_I64 and RISCVISD::VMV_V_X_VL
  // share semantics when the operand type is wider than the resulting vector
  // element type: an implicit truncation first takes place. Therefore, perform
  // a manual truncation/sign-extension in order to ignore any truncated bits
  // and catch any zero-extended immediate.
  // For example, we wish to match (i8 -1) -> (XLenVT 255) as a simm5 by first
  // sign-extending to (XLenVT -1).
  MVT XLenVT = Subtarget.getXLenVT();
  assert(XLenVT == N.getOperand(0).getSimpleValueType() &&
         "Unexpected splat operand type");
  MVT EltVT = N.getSimpleValueType().getVectorElementType();
  if (EltVT.bitsLT(XLenVT))
    SplatImm = SignExtend64(SplatImm, EltVT.getSizeInBits());

  if (!ValidateImm(SplatImm))
    return false;

  SplatVal = DAG.getTargetConstant(SplatImm, SDLoc(N), XLenVT);
  return true;
}

bool RISCVDAGToDAGISel::selectVSplatSimm5(SDValue N, SDValue &SplatVal) {
  return selectVSplatSimmHelper(N, SplatVal, *CurDAG, *Subtarget,
                                [](int64_t Imm) { return isInt<5>(Imm); });
}

bool RISCVDAGToDAGISel::selectVSplatSimm5Plus1(SDValue N, SDValue &SplatVal) {
  return selectVSplatSimmHelper(
      N, SplatVal, *CurDAG, *Subtarget,
      [](int64_t Imm) { return (isInt<5>(Imm) && Imm != -16) || Imm == 16; });
}

bool RISCVDAGToDAGISel::selectVSplatSimm5Plus1NonZero(SDValue N,
                                                      SDValue &SplatVal) {
  return selectVSplatSimmHelper(
      N, SplatVal, *CurDAG, *Subtarget, [](int64_t Imm) {
        return Imm != 0 && ((isInt<5>(Imm) && Imm != -16) || Imm == 16);
      });
}

bool RISCVDAGToDAGISel::selectVSplatUimm5(SDValue N, SDValue &SplatVal) {
  if ((N.getOpcode() != ISD::SPLAT_VECTOR &&
       N.getOpcode() != RISCVISD::SPLAT_VECTOR_I64 &&
       N.getOpcode() != RISCVISD::VMV_V_X_VL) ||
      !isa<ConstantSDNode>(N.getOperand(0)))
    return false;

  int64_t SplatImm = cast<ConstantSDNode>(N.getOperand(0))->getSExtValue();

  if (!isUInt<5>(SplatImm))
    return false;

  SplatVal =
      CurDAG->getTargetConstant(SplatImm, SDLoc(N), Subtarget->getXLenVT());

  return true;
}

bool RISCVDAGToDAGISel::selectRVVSimm5(SDValue N, unsigned Width,
                                       SDValue &Imm) {
  if (auto *C = dyn_cast<ConstantSDNode>(N)) {
    int64_t ImmVal = SignExtend64(C->getSExtValue(), Width);

    if (!isInt<5>(ImmVal))
      return false;

    Imm = CurDAG->getTargetConstant(ImmVal, SDLoc(N), Subtarget->getXLenVT());
    return true;
  }

  return false;
}

// Merge an ADDI into the offset of a load/store instruction where possible.
// (load (addi base, off1), off2) -> (load base, off1+off2)
// (store val, (addi base, off1), off2) -> (store val, base, off1+off2)
// This is possible when off1+off2 fits a 12-bit immediate.
bool RISCVDAGToDAGISel::doPeepholeLoadStoreADDI(SDNode *N) {
  int OffsetOpIdx;
  int BaseOpIdx;

  // Only attempt this optimisation for I-type loads and S-type stores.
  switch (N->getMachineOpcode()) {
  default:
    return false;
  case RISCV::LB:
  case RISCV::LH:
  case RISCV::LW:
  case RISCV::LBU:
  case RISCV::LHU:
  case RISCV::LWU:
  case RISCV::LD:
  case RISCV::FLH:
  case RISCV::FLW:
  case RISCV::FLD:
    BaseOpIdx = 0;
    OffsetOpIdx = 1;
    break;
  case RISCV::SB:
  case RISCV::SH:
  case RISCV::SW:
  case RISCV::SD:
  case RISCV::FSH:
  case RISCV::FSW:
  case RISCV::FSD:
    BaseOpIdx = 1;
    OffsetOpIdx = 2;
    break;
  }

  if (!isa<ConstantSDNode>(N->getOperand(OffsetOpIdx)))
    return false;

  SDValue Base = N->getOperand(BaseOpIdx);

  // If the base is an ADDI, we can merge it in to the load/store.
  if (!Base.isMachineOpcode() || Base.getMachineOpcode() != RISCV::ADDI)
    return false;

  SDValue ImmOperand = Base.getOperand(1);
  uint64_t Offset2 = N->getConstantOperandVal(OffsetOpIdx);

  if (auto *Const = dyn_cast<ConstantSDNode>(ImmOperand)) {
    int64_t Offset1 = Const->getSExtValue();
    int64_t CombinedOffset = Offset1 + Offset2;
    if (!isInt<12>(CombinedOffset))
      return false;
    ImmOperand = CurDAG->getTargetConstant(CombinedOffset, SDLoc(ImmOperand),
                                           ImmOperand.getValueType());
  } else if (auto *GA = dyn_cast<GlobalAddressSDNode>(ImmOperand)) {
    // If the off1 in (addi base, off1) is a global variable's address (its
    // low part, really), then we can rely on the alignment of that variable
    // to provide a margin of safety before off1 can overflow the 12 bits.
    // Check if off2 falls within that margin; if so off1+off2 can't overflow.
    const DataLayout &DL = CurDAG->getDataLayout();
    Align Alignment = GA->getGlobal()->getPointerAlignment(DL);
    if (Offset2 != 0 && Alignment <= Offset2)
      return false;
    int64_t Offset1 = GA->getOffset();
    int64_t CombinedOffset = Offset1 + Offset2;
    ImmOperand = CurDAG->getTargetGlobalAddress(
        GA->getGlobal(), SDLoc(ImmOperand), ImmOperand.getValueType(),
        CombinedOffset, GA->getTargetFlags());
  } else if (auto *CP = dyn_cast<ConstantPoolSDNode>(ImmOperand)) {
    // Ditto.
    Align Alignment = CP->getAlign();
    if (Offset2 != 0 && Alignment <= Offset2)
      return false;
    int64_t Offset1 = CP->getOffset();
    int64_t CombinedOffset = Offset1 + Offset2;
    ImmOperand = CurDAG->getTargetConstantPool(
        CP->getConstVal(), ImmOperand.getValueType(), CP->getAlign(),
        CombinedOffset, CP->getTargetFlags());
  } else {
    return false;
  }

  LLVM_DEBUG(dbgs() << "Folding add-immediate into mem-op:\nBase:    ");
  LLVM_DEBUG(Base->dump(CurDAG));
  LLVM_DEBUG(dbgs() << "\nN: ");
  LLVM_DEBUG(N->dump(CurDAG));
  LLVM_DEBUG(dbgs() << "\n");

  // Modify the offset operand of the load/store.
  if (BaseOpIdx == 0) // Load
    CurDAG->UpdateNodeOperands(N, Base.getOperand(0), ImmOperand,
                               N->getOperand(2));
  else // Store
    CurDAG->UpdateNodeOperands(N, N->getOperand(0), Base.getOperand(0),
                               ImmOperand, N->getOperand(3));

  return true;
}

// Try to remove sext.w if the input is a W instruction or can be made into
// a W instruction cheaply.
bool RISCVDAGToDAGISel::doPeepholeSExtW(SDNode *N) {
  // Look for the sext.w pattern, addiw rd, rs1, 0.
  if (N->getMachineOpcode() != RISCV::ADDIW ||
      !isNullConstant(N->getOperand(1)))
    return false;

  SDValue N0 = N->getOperand(0);
  if (!N0.isMachineOpcode())
    return false;

  switch (N0.getMachineOpcode()) {
  default:
    break;
  case RISCV::ADD:
  case RISCV::ADDI:
  case RISCV::SUB:
  case RISCV::MUL:
  case RISCV::SLLI: {
    // Convert sext.w+add/sub/mul to their W instructions. This will create
    // a new independent instruction. This improves latency.
    unsigned Opc;
    switch (N0.getMachineOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case RISCV::ADD:  Opc = RISCV::ADDW;  break;
    case RISCV::ADDI: Opc = RISCV::ADDIW; break;
    case RISCV::SUB:  Opc = RISCV::SUBW;  break;
    case RISCV::MUL:  Opc = RISCV::MULW;  break;
    case RISCV::SLLI: Opc = RISCV::SLLIW; break;
    }

    SDValue N00 = N0.getOperand(0);
    SDValue N01 = N0.getOperand(1);

    // Shift amount needs to be uimm5.
    if (N0.getMachineOpcode() == RISCV::SLLI &&
        !isUInt<5>(cast<ConstantSDNode>(N01)->getSExtValue()))
      break;

    SDNode *Result =
        CurDAG->getMachineNode(Opc, SDLoc(N), N->getValueType(0),
                               N00, N01);
    ReplaceUses(N, Result);
    return true;
  }
  case RISCV::ADDW:
  case RISCV::ADDIW:
  case RISCV::SUBW:
  case RISCV::MULW:
  case RISCV::SLLIW:
    // Result is already sign extended just remove the sext.w.
    // NOTE: We only handle the nodes that are selected with hasAllWUsers.
    ReplaceUses(N, N0.getNode());
    return true;
  }

  return false;
}

// This pass converts a legalized DAG into a RISCV-specific DAG, ready
// for instruction scheduling.
FunctionPass *llvm::createRISCVISelDag(RISCVTargetMachine &TM) {
  return new RISCVDAGToDAGISel(TM);
}
