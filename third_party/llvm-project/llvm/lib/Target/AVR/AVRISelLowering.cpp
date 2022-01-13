//===-- AVRISelLowering.cpp - AVR DAG Lowering Implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that AVR uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "AVRISelLowering.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorHandling.h"

#include "AVR.h"
#include "AVRMachineFunctionInfo.h"
#include "AVRSubtarget.h"
#include "AVRTargetMachine.h"
#include "MCTargetDesc/AVRMCTargetDesc.h"

namespace llvm {

AVRTargetLowering::AVRTargetLowering(const AVRTargetMachine &TM,
                                     const AVRSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
  // Set up the register classes.
  addRegisterClass(MVT::i8, &AVR::GPR8RegClass);
  addRegisterClass(MVT::i16, &AVR::DREGSRegClass);

  // Compute derived properties from the register classes.
  computeRegisterProperties(Subtarget.getRegisterInfo());

  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(ZeroOrOneBooleanContent);
  setSchedulingPreference(Sched::RegPressure);
  setStackPointerRegisterToSaveRestore(AVR::SP);
  setSupportsUnalignedAtomics(true);

  setOperationAction(ISD::GlobalAddress, MVT::i16, Custom);
  setOperationAction(ISD::BlockAddress, MVT::i16, Custom);

  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i8, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i16, Expand);

  for (MVT VT : MVT::integer_valuetypes()) {
    for (auto N : {ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD}) {
      setLoadExtAction(N, VT, MVT::i1, Promote);
      setLoadExtAction(N, VT, MVT::i8, Expand);
    }
  }

  setTruncStoreAction(MVT::i16, MVT::i8, Expand);

  for (MVT VT : MVT::integer_valuetypes()) {
    setOperationAction(ISD::ADDC, VT, Legal);
    setOperationAction(ISD::SUBC, VT, Legal);
    setOperationAction(ISD::ADDE, VT, Legal);
    setOperationAction(ISD::SUBE, VT, Legal);
  }

  // sub (x, imm) gets canonicalized to add (x, -imm), so for illegal types
  // revert into a sub since we don't have an add with immediate instruction.
  setOperationAction(ISD::ADD, MVT::i32, Custom);
  setOperationAction(ISD::ADD, MVT::i64, Custom);

  // our shift instructions are only able to shift 1 bit at a time, so handle
  // this in a custom way.
  setOperationAction(ISD::SRA, MVT::i8, Custom);
  setOperationAction(ISD::SHL, MVT::i8, Custom);
  setOperationAction(ISD::SRL, MVT::i8, Custom);
  setOperationAction(ISD::SRA, MVT::i16, Custom);
  setOperationAction(ISD::SHL, MVT::i16, Custom);
  setOperationAction(ISD::SRL, MVT::i16, Custom);
  setOperationAction(ISD::SHL_PARTS, MVT::i16, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i16, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i16, Expand);

  setOperationAction(ISD::ROTL, MVT::i8, Custom);
  setOperationAction(ISD::ROTL, MVT::i16, Expand);
  setOperationAction(ISD::ROTR, MVT::i8, Custom);
  setOperationAction(ISD::ROTR, MVT::i16, Expand);

  setOperationAction(ISD::BR_CC, MVT::i8, Custom);
  setOperationAction(ISD::BR_CC, MVT::i16, Custom);
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::i64, Custom);
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);

  setOperationAction(ISD::SELECT_CC, MVT::i8, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i16, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Expand);
  setOperationAction(ISD::SETCC, MVT::i8, Custom);
  setOperationAction(ISD::SETCC, MVT::i16, Custom);
  setOperationAction(ISD::SETCC, MVT::i32, Custom);
  setOperationAction(ISD::SETCC, MVT::i64, Custom);
  setOperationAction(ISD::SELECT, MVT::i8, Expand);
  setOperationAction(ISD::SELECT, MVT::i16, Expand);

  setOperationAction(ISD::BSWAP, MVT::i16, Expand);

  // Add support for postincrement and predecrement load/stores.
  setIndexedLoadAction(ISD::POST_INC, MVT::i8, Legal);
  setIndexedLoadAction(ISD::POST_INC, MVT::i16, Legal);
  setIndexedLoadAction(ISD::PRE_DEC, MVT::i8, Legal);
  setIndexedLoadAction(ISD::PRE_DEC, MVT::i16, Legal);
  setIndexedStoreAction(ISD::POST_INC, MVT::i8, Legal);
  setIndexedStoreAction(ISD::POST_INC, MVT::i16, Legal);
  setIndexedStoreAction(ISD::PRE_DEC, MVT::i8, Legal);
  setIndexedStoreAction(ISD::PRE_DEC, MVT::i16, Legal);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);

  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);
  setOperationAction(ISD::VAARG, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);

  // Atomic operations which must be lowered to rtlib calls
  for (MVT VT : MVT::integer_valuetypes()) {
    setOperationAction(ISD::ATOMIC_SWAP, VT, Expand);
    setOperationAction(ISD::ATOMIC_CMP_SWAP, VT, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_NAND, VT, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_MAX, VT, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_MIN, VT, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_UMAX, VT, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_UMIN, VT, Expand);
  }

  // Division/remainder
  setOperationAction(ISD::UDIV, MVT::i8, Expand);
  setOperationAction(ISD::UDIV, MVT::i16, Expand);
  setOperationAction(ISD::UREM, MVT::i8, Expand);
  setOperationAction(ISD::UREM, MVT::i16, Expand);
  setOperationAction(ISD::SDIV, MVT::i8, Expand);
  setOperationAction(ISD::SDIV, MVT::i16, Expand);
  setOperationAction(ISD::SREM, MVT::i8, Expand);
  setOperationAction(ISD::SREM, MVT::i16, Expand);

  // Make division and modulus custom
  setOperationAction(ISD::UDIVREM, MVT::i8, Custom);
  setOperationAction(ISD::UDIVREM, MVT::i16, Custom);
  setOperationAction(ISD::UDIVREM, MVT::i32, Custom);
  setOperationAction(ISD::SDIVREM, MVT::i8, Custom);
  setOperationAction(ISD::SDIVREM, MVT::i16, Custom);
  setOperationAction(ISD::SDIVREM, MVT::i32, Custom);

  // Do not use MUL. The AVR instructions are closer to SMUL_LOHI &co.
  setOperationAction(ISD::MUL, MVT::i8, Expand);
  setOperationAction(ISD::MUL, MVT::i16, Expand);

  // Expand 16 bit multiplications.
  setOperationAction(ISD::SMUL_LOHI, MVT::i16, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i16, Expand);

  // Expand multiplications to libcalls when there is
  // no hardware MUL.
  if (!Subtarget.supportsMultiplication()) {
    setOperationAction(ISD::SMUL_LOHI, MVT::i8, Expand);
    setOperationAction(ISD::UMUL_LOHI, MVT::i8, Expand);
  }

  for (MVT VT : MVT::integer_valuetypes()) {
    setOperationAction(ISD::MULHS, VT, Expand);
    setOperationAction(ISD::MULHU, VT, Expand);
  }

  for (MVT VT : MVT::integer_valuetypes()) {
    setOperationAction(ISD::CTPOP, VT, Expand);
    setOperationAction(ISD::CTLZ, VT, Expand);
    setOperationAction(ISD::CTTZ, VT, Expand);
  }

  for (MVT VT : MVT::integer_valuetypes()) {
    setOperationAction(ISD::SIGN_EXTEND_INREG, VT, Expand);
    // TODO: The generated code is pretty poor. Investigate using the
    // same "shift and subtract with carry" trick that we do for
    // extending 8-bit to 16-bit. This may require infrastructure
    // improvements in how we treat 16-bit "registers" to be feasible.
  }

  // Division rtlib functions (not supported), use divmod functions instead
  setLibcallName(RTLIB::SDIV_I8, nullptr);
  setLibcallName(RTLIB::SDIV_I16, nullptr);
  setLibcallName(RTLIB::SDIV_I32, nullptr);
  setLibcallName(RTLIB::UDIV_I8, nullptr);
  setLibcallName(RTLIB::UDIV_I16, nullptr);
  setLibcallName(RTLIB::UDIV_I32, nullptr);

  // Modulus rtlib functions (not supported), use divmod functions instead
  setLibcallName(RTLIB::SREM_I8, nullptr);
  setLibcallName(RTLIB::SREM_I16, nullptr);
  setLibcallName(RTLIB::SREM_I32, nullptr);
  setLibcallName(RTLIB::UREM_I8, nullptr);
  setLibcallName(RTLIB::UREM_I16, nullptr);
  setLibcallName(RTLIB::UREM_I32, nullptr);

  // Division and modulus rtlib functions
  setLibcallName(RTLIB::SDIVREM_I8, "__divmodqi4");
  setLibcallName(RTLIB::SDIVREM_I16, "__divmodhi4");
  setLibcallName(RTLIB::SDIVREM_I32, "__divmodsi4");
  setLibcallName(RTLIB::UDIVREM_I8, "__udivmodqi4");
  setLibcallName(RTLIB::UDIVREM_I16, "__udivmodhi4");
  setLibcallName(RTLIB::UDIVREM_I32, "__udivmodsi4");

  // Several of the runtime library functions use a special calling conv
  setLibcallCallingConv(RTLIB::SDIVREM_I8, CallingConv::AVR_BUILTIN);
  setLibcallCallingConv(RTLIB::SDIVREM_I16, CallingConv::AVR_BUILTIN);
  setLibcallCallingConv(RTLIB::UDIVREM_I8, CallingConv::AVR_BUILTIN);
  setLibcallCallingConv(RTLIB::UDIVREM_I16, CallingConv::AVR_BUILTIN);

  // Trigonometric rtlib functions
  setLibcallName(RTLIB::SIN_F32, "sin");
  setLibcallName(RTLIB::COS_F32, "cos");

  setMinFunctionAlignment(Align(2));
  setMinimumJumpTableEntries(UINT_MAX);
}

const char *AVRTargetLowering::getTargetNodeName(unsigned Opcode) const {
#define NODE(name)                                                             \
  case AVRISD::name:                                                           \
    return #name

  switch (Opcode) {
  default:
    return nullptr;
    NODE(RET_FLAG);
    NODE(RETI_FLAG);
    NODE(CALL);
    NODE(WRAPPER);
    NODE(LSL);
    NODE(LSR);
    NODE(ROL);
    NODE(ROR);
    NODE(ASR);
    NODE(LSLLOOP);
    NODE(LSRLOOP);
    NODE(ROLLOOP);
    NODE(RORLOOP);
    NODE(ASRLOOP);
    NODE(BRCOND);
    NODE(CMP);
    NODE(CMPC);
    NODE(TST);
    NODE(SELECT_CC);
#undef NODE
  }
}

EVT AVRTargetLowering::getSetCCResultType(const DataLayout &DL, LLVMContext &,
                                          EVT VT) const {
  assert(!VT.isVector() && "No AVR SetCC type for vectors!");
  return MVT::i8;
}

SDValue AVRTargetLowering::LowerShifts(SDValue Op, SelectionDAG &DAG) const {
  //: TODO: this function has to be completely rewritten to produce optimal
  // code, for now it's producing very long but correct code.
  unsigned Opc8;
  const SDNode *N = Op.getNode();
  EVT VT = Op.getValueType();
  SDLoc dl(N);
  assert(isPowerOf2_32(VT.getSizeInBits()) &&
         "Expected power-of-2 shift amount");

  // Expand non-constant shifts to loops.
  if (!isa<ConstantSDNode>(N->getOperand(1))) {
    switch (Op.getOpcode()) {
    default:
      llvm_unreachable("Invalid shift opcode!");
    case ISD::SHL:
      return DAG.getNode(AVRISD::LSLLOOP, dl, VT, N->getOperand(0),
                         N->getOperand(1));
    case ISD::SRL:
      return DAG.getNode(AVRISD::LSRLOOP, dl, VT, N->getOperand(0),
                         N->getOperand(1));
    case ISD::ROTL: {
      SDValue Amt = N->getOperand(1);
      EVT AmtVT = Amt.getValueType();
      Amt = DAG.getNode(ISD::AND, dl, AmtVT, Amt,
                        DAG.getConstant(VT.getSizeInBits() - 1, dl, AmtVT));
      return DAG.getNode(AVRISD::ROLLOOP, dl, VT, N->getOperand(0), Amt);
    }
    case ISD::ROTR: {
      SDValue Amt = N->getOperand(1);
      EVT AmtVT = Amt.getValueType();
      Amt = DAG.getNode(ISD::AND, dl, AmtVT, Amt,
                        DAG.getConstant(VT.getSizeInBits() - 1, dl, AmtVT));
      return DAG.getNode(AVRISD::RORLOOP, dl, VT, N->getOperand(0), Amt);
    }
    case ISD::SRA:
      return DAG.getNode(AVRISD::ASRLOOP, dl, VT, N->getOperand(0),
                         N->getOperand(1));
    }
  }

  uint64_t ShiftAmount = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
  SDValue Victim = N->getOperand(0);

  switch (Op.getOpcode()) {
  case ISD::SRA:
    Opc8 = AVRISD::ASR;
    break;
  case ISD::ROTL:
    Opc8 = AVRISD::ROL;
    ShiftAmount = ShiftAmount % VT.getSizeInBits();
    break;
  case ISD::ROTR:
    Opc8 = AVRISD::ROR;
    ShiftAmount = ShiftAmount % VT.getSizeInBits();
    break;
  case ISD::SRL:
    Opc8 = AVRISD::LSR;
    break;
  case ISD::SHL:
    Opc8 = AVRISD::LSL;
    break;
  default:
    llvm_unreachable("Invalid shift opcode");
  }

  // Optimize int8/int16 shifts.
  if (VT.getSizeInBits() == 8) {
    if (Op.getOpcode() == ISD::SHL && 4 <= ShiftAmount && ShiftAmount < 7) {
      // Optimize LSL when 4 <= ShiftAmount <= 6.
      Victim = DAG.getNode(AVRISD::SWAP, dl, VT, Victim);
      Victim =
          DAG.getNode(ISD::AND, dl, VT, Victim, DAG.getConstant(0xf0, dl, VT));
      ShiftAmount -= 4;
    } else if (Op.getOpcode() == ISD::SRL && 4 <= ShiftAmount &&
               ShiftAmount < 7) {
      // Optimize LSR when 4 <= ShiftAmount <= 6.
      Victim = DAG.getNode(AVRISD::SWAP, dl, VT, Victim);
      Victim =
          DAG.getNode(ISD::AND, dl, VT, Victim, DAG.getConstant(0x0f, dl, VT));
      ShiftAmount -= 4;
    } else if (Op.getOpcode() == ISD::SHL && ShiftAmount == 7) {
      // Optimize LSL when ShiftAmount == 7.
      Victim = DAG.getNode(AVRISD::LSLBN, dl, VT, Victim,
                           DAG.getConstant(7, dl, VT));
      ShiftAmount = 0;
    } else if (Op.getOpcode() == ISD::SRL && ShiftAmount == 7) {
      // Optimize LSR when ShiftAmount == 7.
      Victim = DAG.getNode(AVRISD::LSRBN, dl, VT, Victim,
                           DAG.getConstant(7, dl, VT));
      ShiftAmount = 0;
    } else if (Op.getOpcode() == ISD::SRA && ShiftAmount == 6) {
      // Optimize ASR when ShiftAmount == 6.
      Victim = DAG.getNode(AVRISD::ASRBN, dl, VT, Victim,
                           DAG.getConstant(6, dl, VT));
      ShiftAmount = 0;
    } else if (Op.getOpcode() == ISD::SRA && ShiftAmount == 7) {
      // Optimize ASR when ShiftAmount == 7.
      Victim = DAG.getNode(AVRISD::ASRBN, dl, VT, Victim,
                           DAG.getConstant(7, dl, VT));
      ShiftAmount = 0;
    }
  } else if (VT.getSizeInBits() == 16) {
    if (4 <= ShiftAmount && ShiftAmount < 8)
      switch (Op.getOpcode()) {
      case ISD::SHL:
        Victim = DAG.getNode(AVRISD::LSLWN, dl, VT, Victim,
                             DAG.getConstant(4, dl, VT));
        ShiftAmount -= 4;
        break;
      case ISD::SRL:
        Victim = DAG.getNode(AVRISD::LSRWN, dl, VT, Victim,
                             DAG.getConstant(4, dl, VT));
        ShiftAmount -= 4;
        break;
      default:
        break;
      }
    else if (8 <= ShiftAmount && ShiftAmount < 12)
      switch (Op.getOpcode()) {
      case ISD::SHL:
        Victim = DAG.getNode(AVRISD::LSLWN, dl, VT, Victim,
                             DAG.getConstant(8, dl, VT));
        ShiftAmount -= 8;
        // Only operate on the higher byte for remaining shift bits.
        Opc8 = AVRISD::LSLHI;
        break;
      case ISD::SRL:
        Victim = DAG.getNode(AVRISD::LSRWN, dl, VT, Victim,
                             DAG.getConstant(8, dl, VT));
        ShiftAmount -= 8;
        // Only operate on the lower byte for remaining shift bits.
        Opc8 = AVRISD::LSRLO;
        break;
      case ISD::SRA:
        Victim = DAG.getNode(AVRISD::ASRWN, dl, VT, Victim,
                             DAG.getConstant(8, dl, VT));
        ShiftAmount -= 8;
        // Only operate on the lower byte for remaining shift bits.
        Opc8 = AVRISD::ASRLO;
        break;
      default:
        break;
      }
    else if (12 <= ShiftAmount)
      switch (Op.getOpcode()) {
      case ISD::SHL:
        Victim = DAG.getNode(AVRISD::LSLWN, dl, VT, Victim,
                             DAG.getConstant(12, dl, VT));
        ShiftAmount -= 12;
        // Only operate on the higher byte for remaining shift bits.
        Opc8 = AVRISD::LSLHI;
        break;
      case ISD::SRL:
        Victim = DAG.getNode(AVRISD::LSRWN, dl, VT, Victim,
                             DAG.getConstant(12, dl, VT));
        ShiftAmount -= 12;
        // Only operate on the lower byte for remaining shift bits.
        Opc8 = AVRISD::LSRLO;
        break;
      case ISD::SRA:
        Victim = DAG.getNode(AVRISD::ASRWN, dl, VT, Victim,
                             DAG.getConstant(8, dl, VT));
        ShiftAmount -= 8;
        // Only operate on the lower byte for remaining shift bits.
        Opc8 = AVRISD::ASRLO;
        break;
      default:
        break;
      }
  }

  while (ShiftAmount--) {
    Victim = DAG.getNode(Opc8, dl, VT, Victim);
  }

  return Victim;
}

SDValue AVRTargetLowering::LowerDivRem(SDValue Op, SelectionDAG &DAG) const {
  unsigned Opcode = Op->getOpcode();
  assert((Opcode == ISD::SDIVREM || Opcode == ISD::UDIVREM) &&
         "Invalid opcode for Div/Rem lowering");
  bool IsSigned = (Opcode == ISD::SDIVREM);
  EVT VT = Op->getValueType(0);
  Type *Ty = VT.getTypeForEVT(*DAG.getContext());

  RTLIB::Libcall LC;
  switch (VT.getSimpleVT().SimpleTy) {
  default:
    llvm_unreachable("Unexpected request for libcall!");
  case MVT::i8:
    LC = IsSigned ? RTLIB::SDIVREM_I8 : RTLIB::UDIVREM_I8;
    break;
  case MVT::i16:
    LC = IsSigned ? RTLIB::SDIVREM_I16 : RTLIB::UDIVREM_I16;
    break;
  case MVT::i32:
    LC = IsSigned ? RTLIB::SDIVREM_I32 : RTLIB::UDIVREM_I32;
    break;
  }

  SDValue InChain = DAG.getEntryNode();

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  for (SDValue const &Value : Op->op_values()) {
    Entry.Node = Value;
    Entry.Ty = Value.getValueType().getTypeForEVT(*DAG.getContext());
    Entry.IsSExt = IsSigned;
    Entry.IsZExt = !IsSigned;
    Args.push_back(Entry);
  }

  SDValue Callee = DAG.getExternalSymbol(getLibcallName(LC),
                                         getPointerTy(DAG.getDataLayout()));

  Type *RetTy = (Type *)StructType::get(Ty, Ty);

  SDLoc dl(Op);
  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(dl)
      .setChain(InChain)
      .setLibCallee(getLibcallCallingConv(LC), RetTy, Callee, std::move(Args))
      .setInRegister()
      .setSExtResult(IsSigned)
      .setZExtResult(!IsSigned);

  std::pair<SDValue, SDValue> CallInfo = LowerCallTo(CLI);
  return CallInfo.first;
}

SDValue AVRTargetLowering::LowerGlobalAddress(SDValue Op,
                                              SelectionDAG &DAG) const {
  auto DL = DAG.getDataLayout();

  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  int64_t Offset = cast<GlobalAddressSDNode>(Op)->getOffset();

  // Create the TargetGlobalAddress node, folding in the constant offset.
  SDValue Result =
      DAG.getTargetGlobalAddress(GV, SDLoc(Op), getPointerTy(DL), Offset);
  return DAG.getNode(AVRISD::WRAPPER, SDLoc(Op), getPointerTy(DL), Result);
}

SDValue AVRTargetLowering::LowerBlockAddress(SDValue Op,
                                             SelectionDAG &DAG) const {
  auto DL = DAG.getDataLayout();
  const BlockAddress *BA = cast<BlockAddressSDNode>(Op)->getBlockAddress();

  SDValue Result = DAG.getTargetBlockAddress(BA, getPointerTy(DL));

  return DAG.getNode(AVRISD::WRAPPER, SDLoc(Op), getPointerTy(DL), Result);
}

/// IntCCToAVRCC - Convert a DAG integer condition code to an AVR CC.
static AVRCC::CondCodes intCCToAVRCC(ISD::CondCode CC) {
  switch (CC) {
  default:
    llvm_unreachable("Unknown condition code!");
  case ISD::SETEQ:
    return AVRCC::COND_EQ;
  case ISD::SETNE:
    return AVRCC::COND_NE;
  case ISD::SETGE:
    return AVRCC::COND_GE;
  case ISD::SETLT:
    return AVRCC::COND_LT;
  case ISD::SETUGE:
    return AVRCC::COND_SH;
  case ISD::SETULT:
    return AVRCC::COND_LO;
  }
}

/// Returns appropriate CP/CPI/CPC nodes code for the given 8/16-bit operands.
SDValue AVRTargetLowering::getAVRCmp(SDValue LHS, SDValue RHS,
                                     SelectionDAG &DAG, SDLoc DL) const {
  assert((LHS.getSimpleValueType() == RHS.getSimpleValueType()) &&
         "LHS and RHS have different types");
  assert(((LHS.getSimpleValueType() == MVT::i16) ||
          (LHS.getSimpleValueType() == MVT::i8)) &&
         "invalid comparison type");

  SDValue Cmp;

  if (LHS.getSimpleValueType() == MVT::i16 && isa<ConstantSDNode>(RHS)) {
    // Generate a CPI/CPC pair if RHS is a 16-bit constant.
    SDValue LHSlo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i8, LHS,
                                DAG.getIntPtrConstant(0, DL));
    SDValue LHShi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i8, LHS,
                                DAG.getIntPtrConstant(1, DL));
    SDValue RHSlo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i8, RHS,
                                DAG.getIntPtrConstant(0, DL));
    SDValue RHShi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i8, RHS,
                                DAG.getIntPtrConstant(1, DL));
    Cmp = DAG.getNode(AVRISD::CMP, DL, MVT::Glue, LHSlo, RHSlo);
    Cmp = DAG.getNode(AVRISD::CMPC, DL, MVT::Glue, LHShi, RHShi, Cmp);
  } else {
    // Generate ordinary 16-bit comparison.
    Cmp = DAG.getNode(AVRISD::CMP, DL, MVT::Glue, LHS, RHS);
  }

  return Cmp;
}

/// Returns appropriate AVR CMP/CMPC nodes and corresponding condition code for
/// the given operands.
SDValue AVRTargetLowering::getAVRCmp(SDValue LHS, SDValue RHS, ISD::CondCode CC,
                                     SDValue &AVRcc, SelectionDAG &DAG,
                                     SDLoc DL) const {
  SDValue Cmp;
  EVT VT = LHS.getValueType();
  bool UseTest = false;

  switch (CC) {
  default:
    break;
  case ISD::SETLE: {
    // Swap operands and reverse the branching condition.
    std::swap(LHS, RHS);
    CC = ISD::SETGE;
    break;
  }
  case ISD::SETGT: {
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(RHS)) {
      switch (C->getSExtValue()) {
      case -1: {
        // When doing lhs > -1 use a tst instruction on the top part of lhs
        // and use brpl instead of using a chain of cp/cpc.
        UseTest = true;
        AVRcc = DAG.getConstant(AVRCC::COND_PL, DL, MVT::i8);
        break;
      }
      case 0: {
        // Turn lhs > 0 into 0 < lhs since 0 can be materialized with
        // __zero_reg__ in lhs.
        RHS = LHS;
        LHS = DAG.getConstant(0, DL, VT);
        CC = ISD::SETLT;
        break;
      }
      default: {
        // Turn lhs < rhs with lhs constant into rhs >= lhs+1, this allows
        // us to  fold the constant into the cmp instruction.
        RHS = DAG.getConstant(C->getSExtValue() + 1, DL, VT);
        CC = ISD::SETGE;
        break;
      }
      }
      break;
    }
    // Swap operands and reverse the branching condition.
    std::swap(LHS, RHS);
    CC = ISD::SETLT;
    break;
  }
  case ISD::SETLT: {
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(RHS)) {
      switch (C->getSExtValue()) {
      case 1: {
        // Turn lhs < 1 into 0 >= lhs since 0 can be materialized with
        // __zero_reg__ in lhs.
        RHS = LHS;
        LHS = DAG.getConstant(0, DL, VT);
        CC = ISD::SETGE;
        break;
      }
      case 0: {
        // When doing lhs < 0 use a tst instruction on the top part of lhs
        // and use brmi instead of using a chain of cp/cpc.
        UseTest = true;
        AVRcc = DAG.getConstant(AVRCC::COND_MI, DL, MVT::i8);
        break;
      }
      }
    }
    break;
  }
  case ISD::SETULE: {
    // Swap operands and reverse the branching condition.
    std::swap(LHS, RHS);
    CC = ISD::SETUGE;
    break;
  }
  case ISD::SETUGT: {
    // Turn lhs < rhs with lhs constant into rhs >= lhs+1, this allows us to
    // fold the constant into the cmp instruction.
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(RHS)) {
      RHS = DAG.getConstant(C->getSExtValue() + 1, DL, VT);
      CC = ISD::SETUGE;
      break;
    }
    // Swap operands and reverse the branching condition.
    std::swap(LHS, RHS);
    CC = ISD::SETULT;
    break;
  }
  }

  // Expand 32 and 64 bit comparisons with custom CMP and CMPC nodes instead of
  // using the default and/or/xor expansion code which is much longer.
  if (VT == MVT::i32) {
    SDValue LHSlo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, LHS,
                                DAG.getIntPtrConstant(0, DL));
    SDValue LHShi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, LHS,
                                DAG.getIntPtrConstant(1, DL));
    SDValue RHSlo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, RHS,
                                DAG.getIntPtrConstant(0, DL));
    SDValue RHShi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, RHS,
                                DAG.getIntPtrConstant(1, DL));

    if (UseTest) {
      // When using tst we only care about the highest part.
      SDValue Top = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i8, LHShi,
                                DAG.getIntPtrConstant(1, DL));
      Cmp = DAG.getNode(AVRISD::TST, DL, MVT::Glue, Top);
    } else {
      Cmp = getAVRCmp(LHSlo, RHSlo, DAG, DL);
      Cmp = DAG.getNode(AVRISD::CMPC, DL, MVT::Glue, LHShi, RHShi, Cmp);
    }
  } else if (VT == MVT::i64) {
    SDValue LHS_0 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, LHS,
                                DAG.getIntPtrConstant(0, DL));
    SDValue LHS_1 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, LHS,
                                DAG.getIntPtrConstant(1, DL));

    SDValue LHS0 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, LHS_0,
                               DAG.getIntPtrConstant(0, DL));
    SDValue LHS1 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, LHS_0,
                               DAG.getIntPtrConstant(1, DL));
    SDValue LHS2 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, LHS_1,
                               DAG.getIntPtrConstant(0, DL));
    SDValue LHS3 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, LHS_1,
                               DAG.getIntPtrConstant(1, DL));

    SDValue RHS_0 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, RHS,
                                DAG.getIntPtrConstant(0, DL));
    SDValue RHS_1 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, RHS,
                                DAG.getIntPtrConstant(1, DL));

    SDValue RHS0 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, RHS_0,
                               DAG.getIntPtrConstant(0, DL));
    SDValue RHS1 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, RHS_0,
                               DAG.getIntPtrConstant(1, DL));
    SDValue RHS2 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, RHS_1,
                               DAG.getIntPtrConstant(0, DL));
    SDValue RHS3 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i16, RHS_1,
                               DAG.getIntPtrConstant(1, DL));

    if (UseTest) {
      // When using tst we only care about the highest part.
      SDValue Top = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i8, LHS3,
                                DAG.getIntPtrConstant(1, DL));
      Cmp = DAG.getNode(AVRISD::TST, DL, MVT::Glue, Top);
    } else {
      Cmp = getAVRCmp(LHS0, RHS0, DAG, DL);
      Cmp = DAG.getNode(AVRISD::CMPC, DL, MVT::Glue, LHS1, RHS1, Cmp);
      Cmp = DAG.getNode(AVRISD::CMPC, DL, MVT::Glue, LHS2, RHS2, Cmp);
      Cmp = DAG.getNode(AVRISD::CMPC, DL, MVT::Glue, LHS3, RHS3, Cmp);
    }
  } else if (VT == MVT::i8 || VT == MVT::i16) {
    if (UseTest) {
      // When using tst we only care about the highest part.
      Cmp = DAG.getNode(AVRISD::TST, DL, MVT::Glue,
                        (VT == MVT::i8)
                            ? LHS
                            : DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i8,
                                          LHS, DAG.getIntPtrConstant(1, DL)));
    } else {
      Cmp = getAVRCmp(LHS, RHS, DAG, DL);
    }
  } else {
    llvm_unreachable("Invalid comparison size");
  }

  // When using a test instruction AVRcc is already set.
  if (!UseTest) {
    AVRcc = DAG.getConstant(intCCToAVRCC(CC), DL, MVT::i8);
  }

  return Cmp;
}

SDValue AVRTargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);
  SDLoc dl(Op);

  SDValue TargetCC;
  SDValue Cmp = getAVRCmp(LHS, RHS, CC, TargetCC, DAG, dl);

  return DAG.getNode(AVRISD::BRCOND, dl, MVT::Other, Chain, Dest, TargetCC,
                     Cmp);
}

SDValue AVRTargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue TrueV = Op.getOperand(2);
  SDValue FalseV = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDLoc dl(Op);

  SDValue TargetCC;
  SDValue Cmp = getAVRCmp(LHS, RHS, CC, TargetCC, DAG, dl);

  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
  SDValue Ops[] = {TrueV, FalseV, TargetCC, Cmp};

  return DAG.getNode(AVRISD::SELECT_CC, dl, VTs, Ops);
}

SDValue AVRTargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  SDLoc DL(Op);

  SDValue TargetCC;
  SDValue Cmp = getAVRCmp(LHS, RHS, CC, TargetCC, DAG, DL);

  SDValue TrueV = DAG.getConstant(1, DL, Op.getValueType());
  SDValue FalseV = DAG.getConstant(0, DL, Op.getValueType());
  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
  SDValue Ops[] = {TrueV, FalseV, TargetCC, Cmp};

  return DAG.getNode(AVRISD::SELECT_CC, DL, VTs, Ops);
}

SDValue AVRTargetLowering::LowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  const MachineFunction &MF = DAG.getMachineFunction();
  const AVRMachineFunctionInfo *AFI = MF.getInfo<AVRMachineFunctionInfo>();
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  auto DL = DAG.getDataLayout();
  SDLoc dl(Op);

  // Vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  SDValue FI = DAG.getFrameIndex(AFI->getVarArgsFrameIndex(), getPointerTy(DL));

  return DAG.getStore(Op.getOperand(0), dl, FI, Op.getOperand(1),
                      MachinePointerInfo(SV));
}

SDValue AVRTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("Don't know how to custom lower this!");
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTL:
  case ISD::ROTR:
    return LowerShifts(Op, DAG);
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::BlockAddress:
    return LowerBlockAddress(Op, DAG);
  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  case ISD::SETCC:
    return LowerSETCC(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG);
  case ISD::SDIVREM:
  case ISD::UDIVREM:
    return LowerDivRem(Op, DAG);
  }

  return SDValue();
}

/// Replace a node with an illegal result type
/// with a new node built out of custom code.
void AVRTargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue> &Results,
                                           SelectionDAG &DAG) const {
  SDLoc DL(N);

  switch (N->getOpcode()) {
  case ISD::ADD: {
    // Convert add (x, imm) into sub (x, -imm).
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(1))) {
      SDValue Sub = DAG.getNode(
          ISD::SUB, DL, N->getValueType(0), N->getOperand(0),
          DAG.getConstant(-C->getAPIntValue(), DL, C->getValueType(0)));
      Results.push_back(Sub);
    }
    break;
  }
  default: {
    SDValue Res = LowerOperation(SDValue(N, 0), DAG);

    for (unsigned I = 0, E = Res->getNumValues(); I != E; ++I)
      Results.push_back(Res.getValue(I));

    break;
  }
  }
}

/// Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
bool AVRTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                              const AddrMode &AM, Type *Ty,
                                              unsigned AS,
                                              Instruction *I) const {
  int64_t Offs = AM.BaseOffs;

  // Allow absolute addresses.
  if (AM.BaseGV && !AM.HasBaseReg && AM.Scale == 0 && Offs == 0) {
    return true;
  }

  // Flash memory instructions only allow zero offsets.
  if (isa<PointerType>(Ty) && AS == AVR::ProgramMemory) {
    return false;
  }

  // Allow reg+<6bit> offset.
  if (Offs < 0)
    Offs = -Offs;
  if (AM.BaseGV == nullptr && AM.HasBaseReg && AM.Scale == 0 &&
      isUInt<6>(Offs)) {
    return true;
  }

  return false;
}

/// Returns true by value, base pointer and
/// offset pointer and addressing mode by reference if the node's address
/// can be legally represented as pre-indexed load / store address.
bool AVRTargetLowering::getPreIndexedAddressParts(SDNode *N, SDValue &Base,
                                                  SDValue &Offset,
                                                  ISD::MemIndexedMode &AM,
                                                  SelectionDAG &DAG) const {
  EVT VT;
  const SDNode *Op;
  SDLoc DL(N);

  if (const LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    VT = LD->getMemoryVT();
    Op = LD->getBasePtr().getNode();
    if (LD->getExtensionType() != ISD::NON_EXTLOAD)
      return false;
    if (AVR::isProgramMemoryAccess(LD)) {
      return false;
    }
  } else if (const StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    VT = ST->getMemoryVT();
    Op = ST->getBasePtr().getNode();
    if (AVR::isProgramMemoryAccess(ST)) {
      return false;
    }
  } else {
    return false;
  }

  if (VT != MVT::i8 && VT != MVT::i16) {
    return false;
  }

  if (Op->getOpcode() != ISD::ADD && Op->getOpcode() != ISD::SUB) {
    return false;
  }

  if (const ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(Op->getOperand(1))) {
    int RHSC = RHS->getSExtValue();
    if (Op->getOpcode() == ISD::SUB)
      RHSC = -RHSC;

    if ((VT == MVT::i16 && RHSC != -2) || (VT == MVT::i8 && RHSC != -1)) {
      return false;
    }

    Base = Op->getOperand(0);
    Offset = DAG.getConstant(RHSC, DL, MVT::i8);
    AM = ISD::PRE_DEC;

    return true;
  }

  return false;
}

/// Returns true by value, base pointer and
/// offset pointer and addressing mode by reference if this node can be
/// combined with a load / store to form a post-indexed load / store.
bool AVRTargetLowering::getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                                   SDValue &Base,
                                                   SDValue &Offset,
                                                   ISD::MemIndexedMode &AM,
                                                   SelectionDAG &DAG) const {
  EVT VT;
  SDLoc DL(N);

  if (const LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    VT = LD->getMemoryVT();
    if (LD->getExtensionType() != ISD::NON_EXTLOAD)
      return false;
  } else if (const StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    VT = ST->getMemoryVT();
    if (AVR::isProgramMemoryAccess(ST)) {
      return false;
    }
  } else {
    return false;
  }

  if (VT != MVT::i8 && VT != MVT::i16) {
    return false;
  }

  if (Op->getOpcode() != ISD::ADD && Op->getOpcode() != ISD::SUB) {
    return false;
  }

  if (const ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(Op->getOperand(1))) {
    int RHSC = RHS->getSExtValue();
    if (Op->getOpcode() == ISD::SUB)
      RHSC = -RHSC;
    if ((VT == MVT::i16 && RHSC != 2) || (VT == MVT::i8 && RHSC != 1)) {
      return false;
    }

    Base = Op->getOperand(0);
    Offset = DAG.getConstant(RHSC, DL, MVT::i8);
    AM = ISD::POST_INC;

    return true;
  }

  return false;
}

bool AVRTargetLowering::isOffsetFoldingLegal(
    const GlobalAddressSDNode *GA) const {
  return true;
}

//===----------------------------------------------------------------------===//
//             Formal Arguments Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "AVRGenCallingConv.inc"

/// Registers for calling conventions, ordered in reverse as required by ABI.
/// Both arrays must be of the same length.
static const MCPhysReg RegList8[] = {
    AVR::R25, AVR::R24, AVR::R23, AVR::R22, AVR::R21, AVR::R20,
    AVR::R19, AVR::R18, AVR::R17, AVR::R16, AVR::R15, AVR::R14,
    AVR::R13, AVR::R12, AVR::R11, AVR::R10, AVR::R9,  AVR::R8};
static const MCPhysReg RegList16[] = {
    AVR::R26R25, AVR::R25R24, AVR::R24R23, AVR::R23R22, AVR::R22R21,
    AVR::R21R20, AVR::R20R19, AVR::R19R18, AVR::R18R17, AVR::R17R16,
    AVR::R16R15, AVR::R15R14, AVR::R14R13, AVR::R13R12, AVR::R12R11,
    AVR::R11R10, AVR::R10R9,  AVR::R9R8};

static_assert(array_lengthof(RegList8) == array_lengthof(RegList16),
              "8-bit and 16-bit register arrays must be of equal length");

/// Analyze incoming and outgoing function arguments. We need custom C++ code
/// to handle special constraints in the ABI.
/// In addition, all pieces of a certain argument have to be passed either
/// using registers or the stack but never mixing both.
template <typename ArgT>
static void
analyzeArguments(TargetLowering::CallLoweringInfo *CLI, const Function *F,
                 const DataLayout *TD, const SmallVectorImpl<ArgT> &Args,
                 SmallVectorImpl<CCValAssign> &ArgLocs, CCState &CCInfo) {
  unsigned NumArgs = Args.size();
  // This is the index of the last used register, in RegList*.
  // -1 means R26 (R26 is never actually used in CC).
  int RegLastIdx = -1;
  // Once a value is passed to the stack it will always be used
  bool UseStack = false;
  for (unsigned i = 0; i != NumArgs;) {
    MVT VT = Args[i].VT;
    // We have to count the number of bytes for each function argument, that is
    // those Args with the same OrigArgIndex. This is important in case the
    // function takes an aggregate type.
    // Current argument will be between [i..j).
    unsigned ArgIndex = Args[i].OrigArgIndex;
    unsigned TotalBytes = VT.getStoreSize();
    unsigned j = i + 1;
    for (; j != NumArgs; ++j) {
      if (Args[j].OrigArgIndex != ArgIndex)
        break;
      TotalBytes += Args[j].VT.getStoreSize();
    }
    // Round up to even number of bytes.
    TotalBytes = alignTo(TotalBytes, 2);
    // Skip zero sized arguments
    if (TotalBytes == 0)
      continue;
    // The index of the first register to be used
    unsigned RegIdx = RegLastIdx + TotalBytes;
    RegLastIdx = RegIdx;
    // If there are not enough registers, use the stack
    if (RegIdx >= array_lengthof(RegList8)) {
      UseStack = true;
    }
    for (; i != j; ++i) {
      MVT VT = Args[i].VT;

      if (UseStack) {
        auto evt = EVT(VT).getTypeForEVT(CCInfo.getContext());
        unsigned Offset = CCInfo.AllocateStack(TD->getTypeAllocSize(evt),
                                               TD->getABITypeAlign(evt));
        CCInfo.addLoc(
            CCValAssign::getMem(i, VT, Offset, VT, CCValAssign::Full));
      } else {
        unsigned Reg;
        if (VT == MVT::i8) {
          Reg = CCInfo.AllocateReg(RegList8[RegIdx]);
        } else if (VT == MVT::i16) {
          Reg = CCInfo.AllocateReg(RegList16[RegIdx]);
        } else {
          llvm_unreachable(
              "calling convention can only manage i8 and i16 types");
        }
        assert(Reg && "register not available in calling convention");
        CCInfo.addLoc(CCValAssign::getReg(i, VT, Reg, VT, CCValAssign::Full));
        // Registers inside a particular argument are sorted in increasing order
        // (remember the array is reversed).
        RegIdx -= VT.getStoreSize();
      }
    }
  }
}

/// Count the total number of bytes needed to pass or return these arguments.
template <typename ArgT>
static unsigned
getTotalArgumentsSizeInBytes(const SmallVectorImpl<ArgT> &Args) {
  unsigned TotalBytes = 0;

  for (const ArgT &Arg : Args) {
    TotalBytes += Arg.VT.getStoreSize();
  }
  return TotalBytes;
}

/// Analyze incoming and outgoing value of returning from a function.
/// The algorithm is similar to analyzeArguments, but there can only be
/// one value, possibly an aggregate, and it is limited to 8 bytes.
template <typename ArgT>
static void analyzeReturnValues(const SmallVectorImpl<ArgT> &Args,
                                CCState &CCInfo) {
  unsigned NumArgs = Args.size();
  unsigned TotalBytes = getTotalArgumentsSizeInBytes(Args);
  // CanLowerReturn() guarantees this assertion.
  assert(TotalBytes <= 8 &&
         "return values greater than 8 bytes cannot be lowered");

  // GCC-ABI says that the size is rounded up to the next even number,
  // but actually once it is more than 4 it will always round up to 8.
  if (TotalBytes > 4) {
    TotalBytes = 8;
  } else {
    TotalBytes = alignTo(TotalBytes, 2);
  }

  // The index of the first register to use.
  int RegIdx = TotalBytes - 1;
  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT VT = Args[i].VT;
    unsigned Reg;
    if (VT == MVT::i8) {
      Reg = CCInfo.AllocateReg(RegList8[RegIdx]);
    } else if (VT == MVT::i16) {
      Reg = CCInfo.AllocateReg(RegList16[RegIdx]);
    } else {
      llvm_unreachable("calling convention can only manage i8 and i16 types");
    }
    assert(Reg && "register not available in calling convention");
    CCInfo.addLoc(CCValAssign::getReg(i, VT, Reg, VT, CCValAssign::Full));
    // Registers sort in increasing order
    RegIdx -= VT.getStoreSize();
  }
}

SDValue AVRTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto DL = DAG.getDataLayout();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  // Variadic functions do not need all the analysis below.
  if (isVarArg) {
    CCInfo.AnalyzeFormalArguments(Ins, ArgCC_AVR_Vararg);
  } else {
    analyzeArguments(nullptr, &MF.getFunction(), &DL, Ins, ArgLocs, CCInfo);
  }

  SDValue ArgValue;
  for (CCValAssign &VA : ArgLocs) {

    // Arguments stored on registers.
    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();
      const TargetRegisterClass *RC;
      if (RegVT == MVT::i8) {
        RC = &AVR::GPR8RegClass;
      } else if (RegVT == MVT::i16) {
        RC = &AVR::DREGSRegClass;
      } else {
        llvm_unreachable("Unknown argument type!");
      }

      unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);
      ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, RegVT);

      // :NOTE: Clang should not promote any i8 into i16 but for safety the
      // following code will handle zexts or sexts generated by other
      // front ends. Otherwise:
      // If this is an 8 bit value, it is really passed promoted
      // to 16 bits. Insert an assert[sz]ext to capture this, then
      // truncate to the right size.
      switch (VA.getLocInfo()) {
      default:
        llvm_unreachable("Unknown loc info!");
      case CCValAssign::Full:
        break;
      case CCValAssign::BCvt:
        ArgValue = DAG.getNode(ISD::BITCAST, dl, VA.getValVT(), ArgValue);
        break;
      case CCValAssign::SExt:
        ArgValue = DAG.getNode(ISD::AssertSext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
        ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
        break;
      case CCValAssign::ZExt:
        ArgValue = DAG.getNode(ISD::AssertZext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
        ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
        break;
      }

      InVals.push_back(ArgValue);
    } else {
      // Only arguments passed on the stack should make it here.
      assert(VA.isMemLoc());

      EVT LocVT = VA.getLocVT();

      // Create the frame index object for this incoming parameter.
      int FI = MFI.CreateFixedObject(LocVT.getSizeInBits() / 8,
                                     VA.getLocMemOffset(), true);

      // Create the SelectionDAG nodes corresponding to a load
      // from this parameter.
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DL));
      InVals.push_back(DAG.getLoad(LocVT, dl, Chain, FIN,
                                   MachinePointerInfo::getFixedStack(MF, FI)));
    }
  }

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (isVarArg) {
    unsigned StackSize = CCInfo.getNextStackOffset();
    AVRMachineFunctionInfo *AFI = MF.getInfo<AVRMachineFunctionInfo>();

    AFI->setVarArgsFrameIndex(MFI.CreateFixedObject(2, StackSize, true));
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
//                  Call Calling Convention Implementation
//===----------------------------------------------------------------------===//

SDValue AVRTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                                     SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &isTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool isVarArg = CLI.IsVarArg;

  MachineFunction &MF = DAG.getMachineFunction();

  // AVR does not yet support tail call optimization.
  isTailCall = false;

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  const Function *F = nullptr;
  if (const GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();

    F = cast<Function>(GV);
    Callee =
        DAG.getTargetGlobalAddress(GV, DL, getPointerTy(DAG.getDataLayout()));
  } else if (const ExternalSymbolSDNode *ES =
                 dyn_cast<ExternalSymbolSDNode>(Callee)) {
    Callee = DAG.getTargetExternalSymbol(ES->getSymbol(),
                                         getPointerTy(DAG.getDataLayout()));
  }

  // Variadic functions do not need all the analysis below.
  if (isVarArg) {
    CCInfo.AnalyzeCallOperands(Outs, ArgCC_AVR_Vararg);
  } else {
    analyzeArguments(&CLI, F, &DAG.getDataLayout(), Outs, ArgLocs, CCInfo);
  }

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, DL);

  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;

  // First, walk the register assignments, inserting copies.
  unsigned AI, AE;
  bool HasStackArgs = false;
  for (AI = 0, AE = ArgLocs.size(); AI != AE; ++AI) {
    CCValAssign &VA = ArgLocs[AI];
    EVT RegVT = VA.getLocVT();
    SDValue Arg = OutVals[AI];

    // Promote the value if needed. With Clang this should not happen.
    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full:
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, RegVT, Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, RegVT, Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, DL, RegVT, Arg);
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, DL, RegVT, Arg);
      break;
    }

    // Stop when we encounter a stack argument, we need to process them
    // in reverse order in the loop below.
    if (VA.isMemLoc()) {
      HasStackArgs = true;
      break;
    }

    // Arguments that can be passed on registers must be kept in the RegsToPass
    // vector.
    RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
  }

  // Second, stack arguments have to walked.
  // Previously this code created chained stores but those chained stores appear
  // to be unchained in the legalization phase. Therefore, do not attempt to
  // chain them here. In fact, chaining them here somehow causes the first and
  // second store to be reversed which is the exact opposite of the intended
  // effect.
  if (HasStackArgs) {
    SmallVector<SDValue, 8> MemOpChains;
    for (; AI != AE; AI++) {
      CCValAssign &VA = ArgLocs[AI];
      SDValue Arg = OutVals[AI];

      assert(VA.isMemLoc());

      // SP points to one stack slot further so add one to adjust it.
      SDValue PtrOff = DAG.getNode(
          ISD::ADD, DL, getPointerTy(DAG.getDataLayout()),
          DAG.getRegister(AVR::SP, getPointerTy(DAG.getDataLayout())),
          DAG.getIntPtrConstant(VA.getLocMemOffset() + 1, DL));

      MemOpChains.push_back(
          DAG.getStore(Chain, DL, Arg, PtrOff,
                       MachinePointerInfo::getStack(MF, VA.getLocMemOffset())));
    }

    if (!MemOpChains.empty())
      Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);
  }

  // Build a sequence of copy-to-reg nodes chained together with token chain and
  // flag operands which copy the outgoing args into registers.  The InFlag in
  // necessary since all emited instructions must be stuck together.
  SDValue InFlag;
  for (auto Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, Reg.first, Reg.second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (auto Reg : RegsToPass) {
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));
  }

  // Add a register mask operand representing the call-preserved registers.
  const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *Mask =
      TRI->getCallPreservedMask(DAG.getMachineFunction(), CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (InFlag.getNode()) {
    Ops.push_back(InFlag);
  }

  Chain = DAG.getNode(AVRISD::CALL, DL, NodeTys, Ops);
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, DL, true),
                             DAG.getIntPtrConstant(0, DL, true), InFlag, DL);

  if (!Ins.empty()) {
    InFlag = Chain.getValue(1);
  }

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg, Ins, DL, DAG,
                         InVals);
}

/// Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
///
SDValue AVRTargetLowering::LowerCallResult(
    SDValue Chain, SDValue InFlag, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  // Handle runtime calling convs.
  if (CallConv == CallingConv::AVR_BUILTIN) {
    CCInfo.AnalyzeCallResult(Ins, RetCC_AVR_BUILTIN);
  } else {
    analyzeReturnValues(Ins, CCInfo);
  }

  // Copy all of the result registers out of their specified physreg.
  for (CCValAssign const &RVLoc : RVLocs) {
    Chain = DAG.getCopyFromReg(Chain, dl, RVLoc.getLocReg(), RVLoc.getValVT(),
                               InFlag)
                .getValue(1);
    InFlag = Chain.getValue(2);
    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
//               Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

bool AVRTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool isVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  if (CallConv == CallingConv::AVR_BUILTIN) {
    SmallVector<CCValAssign, 16> RVLocs;
    CCState CCInfo(CallConv, isVarArg, MF, RVLocs, Context);
    return CCInfo.CheckReturn(Outs, RetCC_AVR_BUILTIN);
  }

  unsigned TotalBytes = getTotalArgumentsSizeInBytes(Outs);
  return TotalBytes <= 8;
}

SDValue
AVRTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                               bool isVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               const SDLoc &dl, SelectionDAG &DAG) const {
  // CCValAssign - represent the assignment of the return value to locations.
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  MachineFunction &MF = DAG.getMachineFunction();

  // Analyze return values.
  if (CallConv == CallingConv::AVR_BUILTIN) {
    CCInfo.AnalyzeReturn(Outs, RetCC_AVR_BUILTIN);
  } else {
    analyzeReturnValues(Outs, CCInfo);
  }

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps(1, Chain);
  // Copy the result values into the output registers.
  for (unsigned i = 0, e = RVLocs.size(); i != e; ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), OutVals[i], Flag);

    // Guarantee that all emitted copies are stuck together with flags.
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  // Don't emit the ret/reti instruction when the naked attribute is present in
  // the function being compiled.
  if (MF.getFunction().getAttributes().hasFnAttr(Attribute::Naked)) {
    return Chain;
  }

  const AVRMachineFunctionInfo *AFI = MF.getInfo<AVRMachineFunctionInfo>();

  unsigned RetOpc =
      AFI->isInterruptOrSignalHandler() ? AVRISD::RETI_FLAG : AVRISD::RET_FLAG;

  RetOps[0] = Chain; // Update chain.

  if (Flag.getNode()) {
    RetOps.push_back(Flag);
  }

  return DAG.getNode(RetOpc, dl, MVT::Other, RetOps);
}

//===----------------------------------------------------------------------===//
//  Custom Inserters
//===----------------------------------------------------------------------===//

MachineBasicBlock *AVRTargetLowering::insertShift(MachineInstr &MI,
                                                  MachineBasicBlock *BB) const {
  unsigned Opc;
  const TargetRegisterClass *RC;
  bool HasRepeatedOperand = false;
  MachineFunction *F = BB->getParent();
  MachineRegisterInfo &RI = F->getRegInfo();
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Invalid shift opcode!");
  case AVR::Lsl8:
    Opc = AVR::ADDRdRr; // LSL is an alias of ADD Rd, Rd
    RC = &AVR::GPR8RegClass;
    HasRepeatedOperand = true;
    break;
  case AVR::Lsl16:
    Opc = AVR::LSLWRd;
    RC = &AVR::DREGSRegClass;
    break;
  case AVR::Asr8:
    Opc = AVR::ASRRd;
    RC = &AVR::GPR8RegClass;
    break;
  case AVR::Asr16:
    Opc = AVR::ASRWRd;
    RC = &AVR::DREGSRegClass;
    break;
  case AVR::Lsr8:
    Opc = AVR::LSRRd;
    RC = &AVR::GPR8RegClass;
    break;
  case AVR::Lsr16:
    Opc = AVR::LSRWRd;
    RC = &AVR::DREGSRegClass;
    break;
  case AVR::Rol8:
    Opc = AVR::ROLBRd;
    RC = &AVR::GPR8RegClass;
    break;
  case AVR::Rol16:
    Opc = AVR::ROLWRd;
    RC = &AVR::DREGSRegClass;
    break;
  case AVR::Ror8:
    Opc = AVR::RORBRd;
    RC = &AVR::GPR8RegClass;
    break;
  case AVR::Ror16:
    Opc = AVR::RORWRd;
    RC = &AVR::DREGSRegClass;
    break;
  }

  const BasicBlock *LLVM_BB = BB->getBasicBlock();

  MachineFunction::iterator I;
  for (I = BB->getIterator(); I != F->end() && &(*I) != BB; ++I)
    ;
  if (I != F->end())
    ++I;

  // Create loop block.
  MachineBasicBlock *LoopBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *CheckBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *RemBB = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(I, LoopBB);
  F->insert(I, CheckBB);
  F->insert(I, RemBB);

  // Update machine-CFG edges by transferring all successors of the current
  // block to the block containing instructions after shift.
  RemBB->splice(RemBB->begin(), BB, std::next(MachineBasicBlock::iterator(MI)),
                BB->end());
  RemBB->transferSuccessorsAndUpdatePHIs(BB);

  // Add edges BB => LoopBB => CheckBB => RemBB, CheckBB => LoopBB.
  BB->addSuccessor(CheckBB);
  LoopBB->addSuccessor(CheckBB);
  CheckBB->addSuccessor(LoopBB);
  CheckBB->addSuccessor(RemBB);

  Register ShiftAmtReg = RI.createVirtualRegister(&AVR::GPR8RegClass);
  Register ShiftAmtReg2 = RI.createVirtualRegister(&AVR::GPR8RegClass);
  Register ShiftReg = RI.createVirtualRegister(RC);
  Register ShiftReg2 = RI.createVirtualRegister(RC);
  Register ShiftAmtSrcReg = MI.getOperand(2).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  Register DstReg = MI.getOperand(0).getReg();

  // BB:
  // rjmp CheckBB
  BuildMI(BB, dl, TII.get(AVR::RJMPk)).addMBB(CheckBB);

  // LoopBB:
  // ShiftReg2 = shift ShiftReg
  auto ShiftMI = BuildMI(LoopBB, dl, TII.get(Opc), ShiftReg2).addReg(ShiftReg);
  if (HasRepeatedOperand)
    ShiftMI.addReg(ShiftReg);

  // CheckBB:
  // ShiftReg = phi [%SrcReg, BB], [%ShiftReg2, LoopBB]
  // ShiftAmt = phi [%N,      BB], [%ShiftAmt2, LoopBB]
  // DestReg  = phi [%SrcReg, BB], [%ShiftReg,  LoopBB]
  // ShiftAmt2 = ShiftAmt - 1;
  // if (ShiftAmt2 >= 0) goto LoopBB;
  BuildMI(CheckBB, dl, TII.get(AVR::PHI), ShiftReg)
      .addReg(SrcReg)
      .addMBB(BB)
      .addReg(ShiftReg2)
      .addMBB(LoopBB);
  BuildMI(CheckBB, dl, TII.get(AVR::PHI), ShiftAmtReg)
      .addReg(ShiftAmtSrcReg)
      .addMBB(BB)
      .addReg(ShiftAmtReg2)
      .addMBB(LoopBB);
  BuildMI(CheckBB, dl, TII.get(AVR::PHI), DstReg)
      .addReg(SrcReg)
      .addMBB(BB)
      .addReg(ShiftReg2)
      .addMBB(LoopBB);

  BuildMI(CheckBB, dl, TII.get(AVR::DECRd), ShiftAmtReg2).addReg(ShiftAmtReg);
  BuildMI(CheckBB, dl, TII.get(AVR::BRPLk)).addMBB(LoopBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return RemBB;
}

static bool isCopyMulResult(MachineBasicBlock::iterator const &I) {
  if (I->getOpcode() == AVR::COPY) {
    Register SrcReg = I->getOperand(1).getReg();
    return (SrcReg == AVR::R0 || SrcReg == AVR::R1);
  }

  return false;
}

// The mul instructions wreak havock on our zero_reg R1. We need to clear it
// after the result has been evacuated. This is probably not the best way to do
// it, but it works for now.
MachineBasicBlock *AVRTargetLowering::insertMul(MachineInstr &MI,
                                                MachineBasicBlock *BB) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  MachineBasicBlock::iterator I(MI);
  ++I; // in any case insert *after* the mul instruction
  if (isCopyMulResult(I))
    ++I;
  if (isCopyMulResult(I))
    ++I;
  BuildMI(*BB, I, MI.getDebugLoc(), TII.get(AVR::EORRdRr), AVR::R1)
      .addReg(AVR::R1)
      .addReg(AVR::R1);
  return BB;
}

MachineBasicBlock *
AVRTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                               MachineBasicBlock *MBB) const {
  int Opc = MI.getOpcode();

  // Pseudo shift instructions with a non constant shift amount are expanded
  // into a loop.
  switch (Opc) {
  case AVR::Lsl8:
  case AVR::Lsl16:
  case AVR::Lsr8:
  case AVR::Lsr16:
  case AVR::Rol8:
  case AVR::Rol16:
  case AVR::Ror8:
  case AVR::Ror16:
  case AVR::Asr8:
  case AVR::Asr16:
    return insertShift(MI, MBB);
  case AVR::MULRdRr:
  case AVR::MULSRdRr:
    return insertMul(MI, MBB);
  }

  assert((Opc == AVR::Select16 || Opc == AVR::Select8) &&
         "Unexpected instr type to insert");

  const AVRInstrInfo &TII = (const AVRInstrInfo &)*MI.getParent()
                                ->getParent()
                                ->getSubtarget()
                                .getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();

  // To "insert" a SELECT instruction, we insert the diamond
  // control-flow pattern. The incoming instruction knows the
  // destination vreg to set, the condition code register to branch
  // on, the true/false values to select between, and a branch opcode
  // to use.

  MachineFunction *MF = MBB->getParent();
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineBasicBlock *FallThrough = MBB->getFallThrough();

  // If the current basic block falls through to another basic block,
  // we must insert an unconditional branch to the fallthrough destination
  // if we are to insert basic blocks at the prior fallthrough point.
  if (FallThrough != nullptr) {
    BuildMI(MBB, dl, TII.get(AVR::RJMPk)).addMBB(FallThrough);
  }

  MachineBasicBlock *trueMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *falseMBB = MF->CreateMachineBasicBlock(LLVM_BB);

  MachineFunction::iterator I;
  for (I = MF->begin(); I != MF->end() && &(*I) != MBB; ++I)
    ;
  if (I != MF->end())
    ++I;
  MF->insert(I, trueMBB);
  MF->insert(I, falseMBB);

  // Transfer remaining instructions and all successors of the current
  // block to the block which will contain the Phi node for the
  // select.
  trueMBB->splice(trueMBB->begin(), MBB,
                  std::next(MachineBasicBlock::iterator(MI)), MBB->end());
  trueMBB->transferSuccessorsAndUpdatePHIs(MBB);

  AVRCC::CondCodes CC = (AVRCC::CondCodes)MI.getOperand(3).getImm();
  BuildMI(MBB, dl, TII.getBrCond(CC)).addMBB(trueMBB);
  BuildMI(MBB, dl, TII.get(AVR::RJMPk)).addMBB(falseMBB);
  MBB->addSuccessor(falseMBB);
  MBB->addSuccessor(trueMBB);

  // Unconditionally flow back to the true block
  BuildMI(falseMBB, dl, TII.get(AVR::RJMPk)).addMBB(trueMBB);
  falseMBB->addSuccessor(trueMBB);

  // Set up the Phi node to determine where we came from
  BuildMI(*trueMBB, trueMBB->begin(), dl, TII.get(AVR::PHI),
          MI.getOperand(0).getReg())
      .addReg(MI.getOperand(1).getReg())
      .addMBB(MBB)
      .addReg(MI.getOperand(2).getReg())
      .addMBB(falseMBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return trueMBB;
}

//===----------------------------------------------------------------------===//
//  Inline Asm Support
//===----------------------------------------------------------------------===//

AVRTargetLowering::ConstraintType
AVRTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    // See http://www.nongnu.org/avr-libc/user-manual/inline_asm.html
    switch (Constraint[0]) {
    default:
      break;
    case 'a': // Simple upper registers
    case 'b': // Base pointer registers pairs
    case 'd': // Upper register
    case 'l': // Lower registers
    case 'e': // Pointer register pairs
    case 'q': // Stack pointer register
    case 'r': // Any register
    case 'w': // Special upper register pairs
      return C_RegisterClass;
    case 't': // Temporary register
    case 'x':
    case 'X': // Pointer register pair X
    case 'y':
    case 'Y': // Pointer register pair Y
    case 'z':
    case 'Z': // Pointer register pair Z
      return C_Register;
    case 'Q': // A memory address based on Y or Z pointer with displacement.
      return C_Memory;
    case 'G': // Floating point constant
    case 'I': // 6-bit positive integer constant
    case 'J': // 6-bit negative integer constant
    case 'K': // Integer constant (Range: 2)
    case 'L': // Integer constant (Range: 0)
    case 'M': // 8-bit integer constant
    case 'N': // Integer constant (Range: -1)
    case 'O': // Integer constant (Range: 8, 16, 24)
    case 'P': // Integer constant (Range: 1)
    case 'R': // Integer constant (Range: -6 to 5)x
      return C_Immediate;
    }
  }

  return TargetLowering::getConstraintType(Constraint);
}

unsigned
AVRTargetLowering::getInlineAsmMemConstraint(StringRef ConstraintCode) const {
  // Not sure if this is actually the right thing to do, but we got to do
  // *something* [agnat]
  switch (ConstraintCode[0]) {
  case 'Q':
    return InlineAsm::Constraint_Q;
  }
  return TargetLowering::getInlineAsmMemConstraint(ConstraintCode);
}

AVRTargetLowering::ConstraintWeight
AVRTargetLowering::getSingleConstraintMatchWeight(
    AsmOperandInfo &info, const char *constraint) const {
  ConstraintWeight weight = CW_Invalid;
  Value *CallOperandVal = info.CallOperandVal;

  // If we don't have a value, we can't do a match,
  // but allow it at the lowest weight.
  // (this behaviour has been copied from the ARM backend)
  if (!CallOperandVal) {
    return CW_Default;
  }

  // Look at the constraint type.
  switch (*constraint) {
  default:
    weight = TargetLowering::getSingleConstraintMatchWeight(info, constraint);
    break;
  case 'd':
  case 'r':
  case 'l':
    weight = CW_Register;
    break;
  case 'a':
  case 'b':
  case 'e':
  case 'q':
  case 't':
  case 'w':
  case 'x':
  case 'X':
  case 'y':
  case 'Y':
  case 'z':
  case 'Z':
    weight = CW_SpecificReg;
    break;
  case 'G':
    if (const ConstantFP *C = dyn_cast<ConstantFP>(CallOperandVal)) {
      if (C->isZero()) {
        weight = CW_Constant;
      }
    }
    break;
  case 'I':
    if (const ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if (isUInt<6>(C->getZExtValue())) {
        weight = CW_Constant;
      }
    }
    break;
  case 'J':
    if (const ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if ((C->getSExtValue() >= -63) && (C->getSExtValue() <= 0)) {
        weight = CW_Constant;
      }
    }
    break;
  case 'K':
    if (const ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if (C->getZExtValue() == 2) {
        weight = CW_Constant;
      }
    }
    break;
  case 'L':
    if (const ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if (C->getZExtValue() == 0) {
        weight = CW_Constant;
      }
    }
    break;
  case 'M':
    if (const ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if (isUInt<8>(C->getZExtValue())) {
        weight = CW_Constant;
      }
    }
    break;
  case 'N':
    if (const ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if (C->getSExtValue() == -1) {
        weight = CW_Constant;
      }
    }
    break;
  case 'O':
    if (const ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if ((C->getZExtValue() == 8) || (C->getZExtValue() == 16) ||
          (C->getZExtValue() == 24)) {
        weight = CW_Constant;
      }
    }
    break;
  case 'P':
    if (const ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if (C->getZExtValue() == 1) {
        weight = CW_Constant;
      }
    }
    break;
  case 'R':
    if (const ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if ((C->getSExtValue() >= -6) && (C->getSExtValue() <= 5)) {
        weight = CW_Constant;
      }
    }
    break;
  case 'Q':
    weight = CW_Memory;
    break;
  }

  return weight;
}

std::pair<unsigned, const TargetRegisterClass *>
AVRTargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                StringRef Constraint,
                                                MVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'a': // Simple upper registers r16..r23.
      if (VT == MVT::i8)
        return std::make_pair(0U, &AVR::LD8loRegClass);
      else if (VT == MVT::i16)
        return std::make_pair(0U, &AVR::DREGSLD8loRegClass);
      break;
    case 'b': // Base pointer registers: y, z.
      if (VT == MVT::i8 || VT == MVT::i16)
        return std::make_pair(0U, &AVR::PTRDISPREGSRegClass);
      break;
    case 'd': // Upper registers r16..r31.
      if (VT == MVT::i8)
        return std::make_pair(0U, &AVR::LD8RegClass);
      else if (VT == MVT::i16)
        return std::make_pair(0U, &AVR::DLDREGSRegClass);
      break;
    case 'l': // Lower registers r0..r15.
      if (VT == MVT::i8)
        return std::make_pair(0U, &AVR::GPR8loRegClass);
      else if (VT == MVT::i16)
        return std::make_pair(0U, &AVR::DREGSloRegClass);
      break;
    case 'e': // Pointer register pairs: x, y, z.
      if (VT == MVT::i8 || VT == MVT::i16)
        return std::make_pair(0U, &AVR::PTRREGSRegClass);
      break;
    case 'q': // Stack pointer register: SPH:SPL.
      return std::make_pair(0U, &AVR::GPRSPRegClass);
    case 'r': // Any register: r0..r31.
      if (VT == MVT::i8)
        return std::make_pair(0U, &AVR::GPR8RegClass);
      else if (VT == MVT::i16)
        return std::make_pair(0U, &AVR::DREGSRegClass);
      break;
    case 't': // Temporary register: r0.
      if (VT == MVT::i8)
        return std::make_pair(unsigned(AVR::R0), &AVR::GPR8RegClass);
      break;
    case 'w': // Special upper register pairs: r24, r26, r28, r30.
      if (VT == MVT::i8 || VT == MVT::i16)
        return std::make_pair(0U, &AVR::IWREGSRegClass);
      break;
    case 'x': // Pointer register pair X: r27:r26.
    case 'X':
      if (VT == MVT::i8 || VT == MVT::i16)
        return std::make_pair(unsigned(AVR::R27R26), &AVR::PTRREGSRegClass);
      break;
    case 'y': // Pointer register pair Y: r29:r28.
    case 'Y':
      if (VT == MVT::i8 || VT == MVT::i16)
        return std::make_pair(unsigned(AVR::R29R28), &AVR::PTRREGSRegClass);
      break;
    case 'z': // Pointer register pair Z: r31:r30.
    case 'Z':
      if (VT == MVT::i8 || VT == MVT::i16)
        return std::make_pair(unsigned(AVR::R31R30), &AVR::PTRREGSRegClass);
      break;
    default:
      break;
    }
  }

  return TargetLowering::getRegForInlineAsmConstraint(
      Subtarget.getRegisterInfo(), Constraint, VT);
}

void AVRTargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                     std::string &Constraint,
                                                     std::vector<SDValue> &Ops,
                                                     SelectionDAG &DAG) const {
  SDValue Result;
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();

  // Currently only support length 1 constraints.
  if (Constraint.length() != 1) {
    return;
  }

  char ConstraintLetter = Constraint[0];
  switch (ConstraintLetter) {
  default:
    break;
  // Deal with integers first:
  case 'I':
  case 'J':
  case 'K':
  case 'L':
  case 'M':
  case 'N':
  case 'O':
  case 'P':
  case 'R': {
    const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
    if (!C) {
      return;
    }

    int64_t CVal64 = C->getSExtValue();
    uint64_t CUVal64 = C->getZExtValue();
    switch (ConstraintLetter) {
    case 'I': // 0..63
      if (!isUInt<6>(CUVal64))
        return;
      Result = DAG.getTargetConstant(CUVal64, DL, Ty);
      break;
    case 'J': // -63..0
      if (CVal64 < -63 || CVal64 > 0)
        return;
      Result = DAG.getTargetConstant(CVal64, DL, Ty);
      break;
    case 'K': // 2
      if (CUVal64 != 2)
        return;
      Result = DAG.getTargetConstant(CUVal64, DL, Ty);
      break;
    case 'L': // 0
      if (CUVal64 != 0)
        return;
      Result = DAG.getTargetConstant(CUVal64, DL, Ty);
      break;
    case 'M': // 0..255
      if (!isUInt<8>(CUVal64))
        return;
      // i8 type may be printed as a negative number,
      // e.g. 254 would be printed as -2,
      // so we force it to i16 at least.
      if (Ty.getSimpleVT() == MVT::i8) {
        Ty = MVT::i16;
      }
      Result = DAG.getTargetConstant(CUVal64, DL, Ty);
      break;
    case 'N': // -1
      if (CVal64 != -1)
        return;
      Result = DAG.getTargetConstant(CVal64, DL, Ty);
      break;
    case 'O': // 8, 16, 24
      if (CUVal64 != 8 && CUVal64 != 16 && CUVal64 != 24)
        return;
      Result = DAG.getTargetConstant(CUVal64, DL, Ty);
      break;
    case 'P': // 1
      if (CUVal64 != 1)
        return;
      Result = DAG.getTargetConstant(CUVal64, DL, Ty);
      break;
    case 'R': // -6..5
      if (CVal64 < -6 || CVal64 > 5)
        return;
      Result = DAG.getTargetConstant(CVal64, DL, Ty);
      break;
    }

    break;
  }
  case 'G':
    const ConstantFPSDNode *FC = dyn_cast<ConstantFPSDNode>(Op);
    if (!FC || !FC->isZero())
      return;
    // Soften float to i8 0
    Result = DAG.getTargetConstant(0, DL, MVT::i8);
    break;
  }

  if (Result.getNode()) {
    Ops.push_back(Result);
    return;
  }

  return TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

Register AVRTargetLowering::getRegisterByName(const char *RegName, LLT VT,
                                              const MachineFunction &MF) const {
  Register Reg;

  if (VT == LLT::scalar(8)) {
    Reg = StringSwitch<unsigned>(RegName)
              .Case("r0", AVR::R0)
              .Case("r1", AVR::R1)
              .Default(0);
  } else {
    Reg = StringSwitch<unsigned>(RegName)
              .Case("r0", AVR::R1R0)
              .Case("sp", AVR::SP)
              .Default(0);
  }

  if (Reg)
    return Reg;

  report_fatal_error(
      Twine("Invalid register name \"" + StringRef(RegName) + "\"."));
}

} // end of namespace llvm
