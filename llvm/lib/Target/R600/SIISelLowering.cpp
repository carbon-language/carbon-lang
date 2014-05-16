//===-- SIISelLowering.cpp - SI DAG Lowering Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Custom DAG lowering for SI
//
//===----------------------------------------------------------------------===//

#include "SIISelLowering.h"
#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "AMDILIntrinsicInfo.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Function.h"

using namespace llvm;

SITargetLowering::SITargetLowering(TargetMachine &TM) :
    AMDGPUTargetLowering(TM) {
  addRegisterClass(MVT::i1, &AMDGPU::VReg_1RegClass);
  addRegisterClass(MVT::i64, &AMDGPU::SReg_64RegClass);

  addRegisterClass(MVT::v32i8, &AMDGPU::SReg_256RegClass);
  addRegisterClass(MVT::v64i8, &AMDGPU::SReg_512RegClass);

  addRegisterClass(MVT::i32, &AMDGPU::SReg_32RegClass);
  addRegisterClass(MVT::f32, &AMDGPU::VReg_32RegClass);

  addRegisterClass(MVT::f64, &AMDGPU::VReg_64RegClass);
  addRegisterClass(MVT::v2i32, &AMDGPU::SReg_64RegClass);
  addRegisterClass(MVT::v2f32, &AMDGPU::VReg_64RegClass);

  addRegisterClass(MVT::v4i32, &AMDGPU::SReg_128RegClass);
  addRegisterClass(MVT::v4f32, &AMDGPU::VReg_128RegClass);

  addRegisterClass(MVT::v8i32, &AMDGPU::VReg_256RegClass);
  addRegisterClass(MVT::v8f32, &AMDGPU::VReg_256RegClass);

  addRegisterClass(MVT::v16i32, &AMDGPU::VReg_512RegClass);
  addRegisterClass(MVT::v16f32, &AMDGPU::VReg_512RegClass);

  computeRegisterProperties();

  // Condition Codes
  setCondCodeAction(ISD::SETONE, MVT::f32, Expand);
  setCondCodeAction(ISD::SETUEQ, MVT::f32, Expand);
  setCondCodeAction(ISD::SETUGE, MVT::f32, Expand);
  setCondCodeAction(ISD::SETUGT, MVT::f32, Expand);
  setCondCodeAction(ISD::SETULE, MVT::f32, Expand);
  setCondCodeAction(ISD::SETULT, MVT::f32, Expand);

  setCondCodeAction(ISD::SETONE, MVT::f64, Expand);
  setCondCodeAction(ISD::SETUEQ, MVT::f64, Expand);
  setCondCodeAction(ISD::SETUGE, MVT::f64, Expand);
  setCondCodeAction(ISD::SETUGT, MVT::f64, Expand);
  setCondCodeAction(ISD::SETULE, MVT::f64, Expand);
  setCondCodeAction(ISD::SETULT, MVT::f64, Expand);

  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8f32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16f32, Expand);

  setOperationAction(ISD::ADD, MVT::i32, Legal);
  setOperationAction(ISD::ADDC, MVT::i32, Legal);
  setOperationAction(ISD::ADDE, MVT::i32, Legal);

  // We need to custom lower vector stores from local memory
  setOperationAction(ISD::LOAD, MVT::v2i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v4i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v8i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v16i32, Custom);

  setOperationAction(ISD::STORE, MVT::v8i32, Custom);
  setOperationAction(ISD::STORE, MVT::v16i32, Custom);

  // We need to custom lower loads/stores from private memory
  setOperationAction(ISD::LOAD, MVT::i32, Custom);
  setOperationAction(ISD::LOAD, MVT::i64, Custom);
  setOperationAction(ISD::LOAD, MVT::v2i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v4i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v8i32, Custom);

  setOperationAction(ISD::STORE, MVT::i1, Custom);
  setOperationAction(ISD::STORE, MVT::i32, Custom);
  setOperationAction(ISD::STORE, MVT::i64, Custom);
  setOperationAction(ISD::STORE, MVT::v2i32, Custom);
  setOperationAction(ISD::STORE, MVT::v4i32, Custom);

  setOperationAction(ISD::SELECT, MVT::f32, Promote);
  AddPromotedToType(ISD::SELECT, MVT::f32, MVT::i32);
  setOperationAction(ISD::SELECT, MVT::i64, Custom);
  setOperationAction(ISD::SELECT, MVT::f64, Promote);
  AddPromotedToType(ISD::SELECT, MVT::f64, MVT::i64);

  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);

  setOperationAction(ISD::SELECT_CC, MVT::Other, Expand);

  setOperationAction(ISD::SETCC, MVT::v2i1, Expand);
  setOperationAction(ISD::SETCC, MVT::v4i1, Expand);

  setOperationAction(ISD::ANY_EXTEND, MVT::i64, Custom);
  setOperationAction(ISD::SIGN_EXTEND, MVT::i64, Custom);
  setOperationAction(ISD::ZERO_EXTEND, MVT::i64, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i1, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i1, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i8, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i8, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i16, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i16, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::Other, Custom);

  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::f32, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v16i8, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v4f32, Custom);

  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);

  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i8, Custom);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i16, Custom);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i32, Expand);
  setLoadExtAction(ISD::SEXTLOAD, MVT::v8i16, Expand);
  setLoadExtAction(ISD::SEXTLOAD, MVT::v16i16, Expand);

  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i8, Custom);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i16, Custom);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i32, Expand);

  setLoadExtAction(ISD::EXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::EXTLOAD, MVT::i8, Custom);
  setLoadExtAction(ISD::EXTLOAD, MVT::i16, Custom);
  setLoadExtAction(ISD::EXTLOAD, MVT::i32, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, Expand);

  setTruncStoreAction(MVT::i32, MVT::i8, Custom);
  setTruncStoreAction(MVT::i32, MVT::i16, Custom);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  setTruncStoreAction(MVT::i64, MVT::i32, Expand);
  setTruncStoreAction(MVT::v8i32, MVT::v8i16, Expand);
  setTruncStoreAction(MVT::v16i32, MVT::v16i16, Expand);

  setOperationAction(ISD::LOAD, MVT::i1, Custom);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);
  setOperationAction(ISD::FrameIndex, MVT::i32, Custom);

  // These should use UDIVREM, so set them to expand
  setOperationAction(ISD::UDIV, MVT::i64, Expand);
  setOperationAction(ISD::UREM, MVT::i64, Expand);

  // We only support LOAD/STORE and vector manipulation ops for vectors
  // with > 4 elements.
  MVT VecTypes[] = {
    MVT::v8i32, MVT::v8f32, MVT::v16i32, MVT::v16f32
  };

  for (MVT VT : VecTypes) {
    for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op) {
      switch(Op) {
      case ISD::LOAD:
      case ISD::STORE:
      case ISD::BUILD_VECTOR:
      case ISD::BITCAST:
      case ISD::EXTRACT_VECTOR_ELT:
      case ISD::INSERT_VECTOR_ELT:
      case ISD::CONCAT_VECTORS:
      case ISD::INSERT_SUBVECTOR:
      case ISD::EXTRACT_SUBVECTOR:
        break;
      default:
        setOperationAction(Op, VT, Expand);
        break;
      }
    }
  }

  for (int I = MVT::v1f64; I <= MVT::v8f64; ++I) {
    MVT::SimpleValueType VT = static_cast<MVT::SimpleValueType>(I);
    setOperationAction(ISD::FTRUNC, VT, Expand);
    setOperationAction(ISD::FCEIL, VT, Expand);
    setOperationAction(ISD::FFLOOR, VT, Expand);
  }

  if (Subtarget->getGeneration() >= AMDGPUSubtarget::SEA_ISLANDS) {
    setOperationAction(ISD::FTRUNC, MVT::f64, Legal);
    setOperationAction(ISD::FCEIL, MVT::f64, Legal);
    setOperationAction(ISD::FFLOOR, MVT::f64, Legal);
    setOperationAction(ISD::FRINT, MVT::f64, Legal);
  }

  setTargetDAGCombine(ISD::SELECT_CC);
  setTargetDAGCombine(ISD::SETCC);

  setSchedulingPreference(Sched::RegPressure);
}

//===----------------------------------------------------------------------===//
// TargetLowering queries
//===----------------------------------------------------------------------===//

bool SITargetLowering::allowsUnalignedMemoryAccesses(EVT  VT,
                                                     unsigned AddrSpace,
                                                     bool *IsFast) const {
  if (IsFast)
    *IsFast = false;

  // XXX: This depends on the address space and also we may want to revist
  // the alignment values we specify in the DataLayout.

  // TODO: I think v3i32 should allow unaligned accesses on CI with DS_READ_B96,
  // which isn't a simple VT.
  if (!VT.isSimple() || VT == MVT::Other)
    return false;

  // XXX - CI changes say "Support for unaligned memory accesses" but I don't
  // see what for specifically. The wording everywhere else seems to be the
  // same.

  // 3.6.4 - Operations using pairs of VGPRs (for example: double-floats) have
  // no alignment restrictions.
  if (AddrSpace == AMDGPUAS::PRIVATE_ADDRESS) {
    // Using any pair of GPRs should be the same as any other pair.
    if (IsFast)
      *IsFast = true;
    return VT.bitsGE(MVT::i64);
  }

  // XXX - The only mention I see of this in the ISA manual is for LDS direct
  // reads the "byte address and must be dword aligned". Is it also true for the
  // normal loads and stores?
  if (AddrSpace == AMDGPUAS::LOCAL_ADDRESS)
    return false;

  // 8.1.6 - For Dword or larger reads or writes, the two LSBs of the
  // byte-address are ignored, thus forcing Dword alignment.
  if (IsFast)
    *IsFast = true;
  return VT.bitsGT(MVT::i32);
}

bool SITargetLowering::shouldSplitVectorType(EVT VT) const {
  return VT.getScalarType().bitsLE(MVT::i16);
}

bool SITargetLowering::shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                                         Type *Ty) const {
  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo*>(getTargetMachine().getInstrInfo());
  return TII->isInlineConstant(Imm);
}

SDValue SITargetLowering::LowerParameter(SelectionDAG &DAG, EVT VT, EVT MemVT,
                                         SDLoc DL, SDValue Chain,
                                         unsigned Offset, bool Signed) const {
  MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
  PointerType *PtrTy = PointerType::get(VT.getTypeForEVT(*DAG.getContext()),
                                            AMDGPUAS::CONSTANT_ADDRESS);
  SDValue BasePtr =  DAG.getCopyFromReg(Chain, DL,
                           MRI.getLiveInVirtReg(AMDGPU::SGPR0_SGPR1), MVT::i64);
  SDValue Ptr = DAG.getNode(ISD::ADD, DL, MVT::i64, BasePtr,
                                             DAG.getConstant(Offset, MVT::i64));
  return DAG.getExtLoad(Signed ? ISD::SEXTLOAD : ISD::ZEXTLOAD, DL, VT, Chain, Ptr,
                            MachinePointerInfo(UndefValue::get(PtrTy)), MemVT,
                            false, false, MemVT.getSizeInBits() >> 3);

}

SDValue SITargetLowering::LowerFormalArguments(
                                      SDValue Chain,
                                      CallingConv::ID CallConv,
                                      bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      SDLoc DL, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const {

  const TargetRegisterInfo *TRI = getTargetMachine().getRegisterInfo();

  MachineFunction &MF = DAG.getMachineFunction();
  FunctionType *FType = MF.getFunction()->getFunctionType();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  assert(CallConv == CallingConv::C);

  SmallVector<ISD::InputArg, 16> Splits;
  uint32_t Skipped = 0;

  for (unsigned i = 0, e = Ins.size(), PSInputNum = 0; i != e; ++i) {
    const ISD::InputArg &Arg = Ins[i];

    // First check if it's a PS input addr
    if (Info->ShaderType == ShaderType::PIXEL && !Arg.Flags.isInReg() &&
        !Arg.Flags.isByVal()) {

      assert((PSInputNum <= 15) && "Too many PS inputs!");

      if (!Arg.Used) {
        // We can savely skip PS inputs
        Skipped |= 1 << i;
        ++PSInputNum;
        continue;
      }

      Info->PSInputAddr |= 1 << PSInputNum++;
    }

    // Second split vertices into their elements
    if (Info->ShaderType != ShaderType::COMPUTE && Arg.VT.isVector()) {
      ISD::InputArg NewArg = Arg;
      NewArg.Flags.setSplit();
      NewArg.VT = Arg.VT.getVectorElementType();

      // We REALLY want the ORIGINAL number of vertex elements here, e.g. a
      // three or five element vertex only needs three or five registers,
      // NOT four or eigth.
      Type *ParamType = FType->getParamType(Arg.OrigArgIndex);
      unsigned NumElements = ParamType->getVectorNumElements();

      for (unsigned j = 0; j != NumElements; ++j) {
        Splits.push_back(NewArg);
        NewArg.PartOffset += NewArg.VT.getStoreSize();
      }

    } else if (Info->ShaderType != ShaderType::COMPUTE) {
      Splits.push_back(Arg);
    }
  }

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());

  // At least one interpolation mode must be enabled or else the GPU will hang.
  if (Info->ShaderType == ShaderType::PIXEL && (Info->PSInputAddr & 0x7F) == 0) {
    Info->PSInputAddr |= 1;
    CCInfo.AllocateReg(AMDGPU::VGPR0);
    CCInfo.AllocateReg(AMDGPU::VGPR1);
  }

  // The pointer to the list of arguments is stored in SGPR0, SGPR1
  if (Info->ShaderType == ShaderType::COMPUTE) {
    CCInfo.AllocateReg(AMDGPU::SGPR0);
    CCInfo.AllocateReg(AMDGPU::SGPR1);
    MF.addLiveIn(AMDGPU::SGPR0_SGPR1, &AMDGPU::SReg_64RegClass);
  }

  if (Info->ShaderType == ShaderType::COMPUTE) {
    getOriginalFunctionArgs(DAG, DAG.getMachineFunction().getFunction(), Ins,
                            Splits);
  }

  AnalyzeFormalArguments(CCInfo, Splits);

  for (unsigned i = 0, e = Ins.size(), ArgIdx = 0; i != e; ++i) {

    const ISD::InputArg &Arg = Ins[i];
    if (Skipped & (1 << i)) {
      InVals.push_back(DAG.getUNDEF(Arg.VT));
      continue;
    }

    CCValAssign &VA = ArgLocs[ArgIdx++];
    EVT VT = VA.getLocVT();

    if (VA.isMemLoc()) {
      VT = Ins[i].VT;
      EVT MemVT = Splits[i].VT;
      // The first 36 bytes of the input buffer contains information about
      // thread group and global sizes.
      SDValue Arg = LowerParameter(DAG, VT, MemVT,  DL, DAG.getRoot(),
                                   36 + VA.getLocMemOffset(),
                                   Ins[i].Flags.isSExt());
      InVals.push_back(Arg);
      continue;
    }
    assert(VA.isRegLoc() && "Parameter must be in a register!");

    unsigned Reg = VA.getLocReg();

    if (VT == MVT::i64) {
      // For now assume it is a pointer
      Reg = TRI->getMatchingSuperReg(Reg, AMDGPU::sub0,
                                     &AMDGPU::SReg_64RegClass);
      Reg = MF.addLiveIn(Reg, &AMDGPU::SReg_64RegClass);
      InVals.push_back(DAG.getCopyFromReg(Chain, DL, Reg, VT));
      continue;
    }

    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg, VT);

    Reg = MF.addLiveIn(Reg, RC);
    SDValue Val = DAG.getCopyFromReg(Chain, DL, Reg, VT);

    if (Arg.VT.isVector()) {

      // Build a vector from the registers
      Type *ParamType = FType->getParamType(Arg.OrigArgIndex);
      unsigned NumElements = ParamType->getVectorNumElements();

      SmallVector<SDValue, 4> Regs;
      Regs.push_back(Val);
      for (unsigned j = 1; j != NumElements; ++j) {
        Reg = ArgLocs[ArgIdx++].getLocReg();
        Reg = MF.addLiveIn(Reg, RC);
        Regs.push_back(DAG.getCopyFromReg(Chain, DL, Reg, VT));
      }

      // Fill up the missing vector elements
      NumElements = Arg.VT.getVectorNumElements() - NumElements;
      for (unsigned j = 0; j != NumElements; ++j)
        Regs.push_back(DAG.getUNDEF(VT));

      InVals.push_back(DAG.getNode(ISD::BUILD_VECTOR, DL, Arg.VT, Regs));
      continue;
    }

    InVals.push_back(Val);
  }
  return Chain;
}

MachineBasicBlock * SITargetLowering::EmitInstrWithCustomInserter(
    MachineInstr * MI, MachineBasicBlock * BB) const {

  MachineBasicBlock::iterator I = *MI;
  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo*>(getTargetMachine().getInstrInfo());
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  switch (MI->getOpcode()) {
  default:
    return AMDGPUTargetLowering::EmitInstrWithCustomInserter(MI, BB);
  case AMDGPU::BRANCH: return BB;
  case AMDGPU::SI_ADDR64_RSRC: {
    unsigned SuperReg = MI->getOperand(0).getReg();
    unsigned SubRegLo = MRI.createVirtualRegister(&AMDGPU::SGPR_64RegClass);
    unsigned SubRegHi = MRI.createVirtualRegister(&AMDGPU::SGPR_64RegClass);
    unsigned SubRegHiHi = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
    unsigned SubRegHiLo = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::S_MOV_B64), SubRegLo)
            .addOperand(MI->getOperand(1));
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::S_MOV_B32), SubRegHiLo)
            .addImm(0);
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::S_MOV_B32), SubRegHiHi)
            .addImm(AMDGPU::RSRC_DATA_FORMAT >> 32);
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::REG_SEQUENCE), SubRegHi)
            .addReg(SubRegHiLo)
            .addImm(AMDGPU::sub0)
            .addReg(SubRegHiHi)
            .addImm(AMDGPU::sub1);
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::REG_SEQUENCE), SuperReg)
            .addReg(SubRegLo)
            .addImm(AMDGPU::sub0_sub1)
            .addReg(SubRegHi)
            .addImm(AMDGPU::sub2_sub3);
    MI->eraseFromParent();
    break;
  }
  case AMDGPU::V_SUB_F64:
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::V_ADD_F64),
            MI->getOperand(0).getReg())
            .addReg(MI->getOperand(1).getReg())
            .addReg(MI->getOperand(2).getReg())
            .addImm(0)  /* src2 */
            .addImm(0)  /* ABS */
            .addImm(0)  /* CLAMP */
            .addImm(0)  /* OMOD */
            .addImm(2); /* NEG */
    MI->eraseFromParent();
    break;

  case AMDGPU::SI_RegisterStorePseudo: {
    MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
    unsigned Reg = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);
    MachineInstrBuilder MIB =
        BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::SI_RegisterStore),
                Reg);
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
      MIB.addOperand(MI->getOperand(i));

    MI->eraseFromParent();
    break;
  }
  case AMDGPU::FABS_SI: {
    MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
    const SIInstrInfo *TII =
      static_cast<const SIInstrInfo*>(getTargetMachine().getInstrInfo());
    unsigned Reg = MRI.createVirtualRegister(&AMDGPU::VReg_32RegClass);
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::V_MOV_B32_e32),
            Reg)
            .addImm(0x7fffffff);
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::V_AND_B32_e32),
            MI->getOperand(0).getReg())
            .addReg(MI->getOperand(1).getReg())
            .addReg(Reg);
    MI->eraseFromParent();
    break;
  }
  case AMDGPU::FNEG_SI: {
    MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
    const SIInstrInfo *TII =
      static_cast<const SIInstrInfo*>(getTargetMachine().getInstrInfo());
    unsigned Reg = MRI.createVirtualRegister(&AMDGPU::VReg_32RegClass);
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::V_MOV_B32_e32),
            Reg)
            .addImm(0x80000000);
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::V_XOR_B32_e32),
            MI->getOperand(0).getReg())
            .addReg(MI->getOperand(1).getReg())
            .addReg(Reg);
    MI->eraseFromParent();
    break;
  }
  case AMDGPU::FCLAMP_SI: {
    const SIInstrInfo *TII =
      static_cast<const SIInstrInfo*>(getTargetMachine().getInstrInfo());
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::V_ADD_F32_e64),
            MI->getOperand(0).getReg())
            .addImm(0) // SRC0 modifiers
            .addOperand(MI->getOperand(1))
            .addImm(0) // SRC1 modifiers
            .addImm(0) // SRC1
            .addImm(1) // CLAMP
            .addImm(0); // OMOD
    MI->eraseFromParent();
  }
  }
  return BB;
}

EVT SITargetLowering::getSetCCResultType(LLVMContext &, EVT VT) const {
  if (!VT.isVector()) {
    return MVT::i1;
  }
  return MVT::getVectorVT(MVT::i1, VT.getVectorNumElements());
}

MVT SITargetLowering::getScalarShiftAmountTy(EVT VT) const {
  return MVT::i32;
}

bool SITargetLowering::isFMAFasterThanFMulAndFAdd(EVT VT) const {
  VT = VT.getScalarType();

  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32:
    return false; /* There is V_MAD_F32 for f32 */
  case MVT::f64:
    return true;
  default:
    break;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Custom DAG Lowering Operations
//===----------------------------------------------------------------------===//

SDValue SITargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  switch (Op.getOpcode()) {
  default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
  case ISD::BRCOND: return LowerBRCOND(Op, DAG);
  case ISD::LOAD: {
    LoadSDNode *Load = dyn_cast<LoadSDNode>(Op);
    if (Op.getValueType().isVector() &&
        (Load->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS ||
         Load->getAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS ||
         (Load->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS &&
          Op.getValueType().getVectorNumElements() > 4))) {
      SDValue MergedValues[2] = {
        SplitVectorLoad(Op, DAG),
        Load->getChain()
      };
      return DAG.getMergeValues(MergedValues, SDLoc(Op));
    } else {
      return LowerLOAD(Op, DAG);
    }
  }

  case ISD::SELECT: return LowerSELECT(Op, DAG);
  case ISD::SELECT_CC: return LowerSELECT_CC(Op, DAG);
  case ISD::SIGN_EXTEND: return LowerSIGN_EXTEND(Op, DAG);
  case ISD::STORE: return LowerSTORE(Op, DAG);
  case ISD::ANY_EXTEND: // Fall-through
  case ISD::ZERO_EXTEND: return LowerZERO_EXTEND(Op, DAG);
  case ISD::GlobalAddress: return LowerGlobalAddress(MFI, Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntrinsicID =
                         cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
    EVT VT = Op.getValueType();
    SDLoc DL(Op);
    //XXX: Hardcoded we only use two to store the pointer to the parameters.
    unsigned NumUserSGPRs = 2;
    switch (IntrinsicID) {
    default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
    case Intrinsic::r600_read_ngroups_x:
      return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(), 0, false);
    case Intrinsic::r600_read_ngroups_y:
      return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(), 4, false);
    case Intrinsic::r600_read_ngroups_z:
      return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(), 8, false);
    case Intrinsic::r600_read_global_size_x:
      return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(), 12, false);
    case Intrinsic::r600_read_global_size_y:
      return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(), 16, false);
    case Intrinsic::r600_read_global_size_z:
      return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(), 20, false);
    case Intrinsic::r600_read_local_size_x:
      return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(), 24, false);
    case Intrinsic::r600_read_local_size_y:
      return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(), 28, false);
    case Intrinsic::r600_read_local_size_z:
      return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(), 32, false);
    case Intrinsic::r600_read_tgid_x:
      return CreateLiveInRegister(DAG, &AMDGPU::SReg_32RegClass,
                     AMDGPU::SReg_32RegClass.getRegister(NumUserSGPRs + 0), VT);
    case Intrinsic::r600_read_tgid_y:
      return CreateLiveInRegister(DAG, &AMDGPU::SReg_32RegClass,
                     AMDGPU::SReg_32RegClass.getRegister(NumUserSGPRs + 1), VT);
    case Intrinsic::r600_read_tgid_z:
      return CreateLiveInRegister(DAG, &AMDGPU::SReg_32RegClass,
                     AMDGPU::SReg_32RegClass.getRegister(NumUserSGPRs + 2), VT);
    case Intrinsic::r600_read_tidig_x:
      return CreateLiveInRegister(DAG, &AMDGPU::VReg_32RegClass,
                                  AMDGPU::VGPR0, VT);
    case Intrinsic::r600_read_tidig_y:
      return CreateLiveInRegister(DAG, &AMDGPU::VReg_32RegClass,
                                  AMDGPU::VGPR1, VT);
    case Intrinsic::r600_read_tidig_z:
      return CreateLiveInRegister(DAG, &AMDGPU::VReg_32RegClass,
                                  AMDGPU::VGPR2, VT);
    case AMDGPUIntrinsic::SI_load_const: {
      SDValue Ops [] = {
        Op.getOperand(1),
        Op.getOperand(2)
      };

      MachineMemOperand *MMO = MF.getMachineMemOperand(
          MachinePointerInfo(),
          MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant,
          VT.getSizeInBits() / 8, 4);
      return DAG.getMemIntrinsicNode(AMDGPUISD::LOAD_CONSTANT, DL,
                                     Op->getVTList(), Ops, VT, MMO);
    }
    case AMDGPUIntrinsic::SI_sample:
      return LowerSampleIntrinsic(AMDGPUISD::SAMPLE, Op, DAG);
    case AMDGPUIntrinsic::SI_sampleb:
      return LowerSampleIntrinsic(AMDGPUISD::SAMPLEB, Op, DAG);
    case AMDGPUIntrinsic::SI_sampled:
      return LowerSampleIntrinsic(AMDGPUISD::SAMPLED, Op, DAG);
    case AMDGPUIntrinsic::SI_samplel:
      return LowerSampleIntrinsic(AMDGPUISD::SAMPLEL, Op, DAG);
    case AMDGPUIntrinsic::SI_vs_load_input:
      return DAG.getNode(AMDGPUISD::LOAD_INPUT, DL, VT,
                         Op.getOperand(1),
                         Op.getOperand(2),
                         Op.getOperand(3));
    }
  }

  case ISD::INTRINSIC_VOID:
    SDValue Chain = Op.getOperand(0);
    unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();

    switch (IntrinsicID) {
      case AMDGPUIntrinsic::SI_tbuffer_store: {
        SDLoc DL(Op);
        SDValue Ops [] = {
          Chain,
          Op.getOperand(2),
          Op.getOperand(3),
          Op.getOperand(4),
          Op.getOperand(5),
          Op.getOperand(6),
          Op.getOperand(7),
          Op.getOperand(8),
          Op.getOperand(9),
          Op.getOperand(10),
          Op.getOperand(11),
          Op.getOperand(12),
          Op.getOperand(13),
          Op.getOperand(14)
        };
        EVT VT = Op.getOperand(3).getValueType();

        MachineMemOperand *MMO = MF.getMachineMemOperand(
            MachinePointerInfo(),
            MachineMemOperand::MOStore,
            VT.getSizeInBits() / 8, 4);
        return DAG.getMemIntrinsicNode(AMDGPUISD::TBUFFER_STORE_FORMAT, DL,
                                       Op->getVTList(), Ops, VT, MMO);
      }
      default:
        break;
    }
  }
  return SDValue();
}

/// \brief Helper function for LowerBRCOND
static SDNode *findUser(SDValue Value, unsigned Opcode) {

  SDNode *Parent = Value.getNode();
  for (SDNode::use_iterator I = Parent->use_begin(), E = Parent->use_end();
       I != E; ++I) {

    if (I.getUse().get() != Value)
      continue;

    if (I->getOpcode() == Opcode)
      return *I;
  }
  return nullptr;
}

/// This transforms the control flow intrinsics to get the branch destination as
/// last parameter, also switches branch target with BR if the need arise
SDValue SITargetLowering::LowerBRCOND(SDValue BRCOND,
                                      SelectionDAG &DAG) const {

  SDLoc DL(BRCOND);

  SDNode *Intr = BRCOND.getOperand(1).getNode();
  SDValue Target = BRCOND.getOperand(2);
  SDNode *BR = nullptr;

  if (Intr->getOpcode() == ISD::SETCC) {
    // As long as we negate the condition everything is fine
    SDNode *SetCC = Intr;
    assert(SetCC->getConstantOperandVal(1) == 1);
    assert(cast<CondCodeSDNode>(SetCC->getOperand(2).getNode())->get() ==
           ISD::SETNE);
    Intr = SetCC->getOperand(0).getNode();

  } else {
    // Get the target from BR if we don't negate the condition
    BR = findUser(BRCOND, ISD::BR);
    Target = BR->getOperand(1);
  }

  assert(Intr->getOpcode() == ISD::INTRINSIC_W_CHAIN);

  // Build the result and
  SmallVector<EVT, 4> Res;
  for (unsigned i = 1, e = Intr->getNumValues(); i != e; ++i)
    Res.push_back(Intr->getValueType(i));

  // operands of the new intrinsic call
  SmallVector<SDValue, 4> Ops;
  Ops.push_back(BRCOND.getOperand(0));
  for (unsigned i = 1, e = Intr->getNumOperands(); i != e; ++i)
    Ops.push_back(Intr->getOperand(i));
  Ops.push_back(Target);

  // build the new intrinsic call
  SDNode *Result = DAG.getNode(
    Res.size() > 1 ? ISD::INTRINSIC_W_CHAIN : ISD::INTRINSIC_VOID, DL,
    DAG.getVTList(Res), Ops).getNode();

  if (BR) {
    // Give the branch instruction our target
    SDValue Ops[] = {
      BR->getOperand(0),
      BRCOND.getOperand(2)
    };
    DAG.MorphNodeTo(BR, ISD::BR, BR->getVTList(), Ops);
  }

  SDValue Chain = SDValue(Result, Result->getNumValues() - 1);

  // Copy the intrinsic results to registers
  for (unsigned i = 1, e = Intr->getNumValues() - 1; i != e; ++i) {
    SDNode *CopyToReg = findUser(SDValue(Intr, i), ISD::CopyToReg);
    if (!CopyToReg)
      continue;

    Chain = DAG.getCopyToReg(
      Chain, DL,
      CopyToReg->getOperand(1),
      SDValue(Result, i - 1),
      SDValue());

    DAG.ReplaceAllUsesWith(SDValue(CopyToReg, 0), CopyToReg->getOperand(0));
  }

  // Remove the old intrinsic from the chain
  DAG.ReplaceAllUsesOfValueWith(
    SDValue(Intr, Intr->getNumValues() - 1),
    Intr->getOperand(0));

  return Chain;
}

SDValue SITargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  LoadSDNode *Load = cast<LoadSDNode>(Op);
  SDValue Ret = AMDGPUTargetLowering::LowerLOAD(Op, DAG);
  SDValue MergedValues[2];
  MergedValues[1] = Load->getChain();
  if (Ret.getNode()) {
    MergedValues[0] = Ret;
    return DAG.getMergeValues(MergedValues, DL);
  }

  if (Load->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS) {
    return SDValue();
  }

  EVT MemVT = Load->getMemoryVT();

  assert(!MemVT.isVector() && "Private loads should be scalarized");
  assert(!MemVT.isFloatingPoint() && "FP loads should be promoted to int");

  SDValue Ptr = DAG.getNode(ISD::SRL, DL, MVT::i32, Load->getBasePtr(),
                            DAG.getConstant(2, MVT::i32));
  Ret = DAG.getNode(AMDGPUISD::REGISTER_LOAD, DL, MVT::i32,
                    Load->getChain(), Ptr,
                    DAG.getTargetConstant(0, MVT::i32),
                    Op.getOperand(2));
  if (MemVT.getSizeInBits() == 64) {
    SDValue IncPtr = DAG.getNode(ISD::ADD, DL, MVT::i32, Ptr,
                                 DAG.getConstant(1, MVT::i32));

    SDValue LoadUpper = DAG.getNode(AMDGPUISD::REGISTER_LOAD, DL, MVT::i32,
                                    Load->getChain(), IncPtr,
                                    DAG.getTargetConstant(0, MVT::i32),
                                    Op.getOperand(2));

    Ret = DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, Ret, LoadUpper);
  }

  MergedValues[0] = Ret;
  return DAG.getMergeValues(MergedValues, DL);

}

SDValue SITargetLowering::LowerSampleIntrinsic(unsigned Opcode,
                                               const SDValue &Op,
                                               SelectionDAG &DAG) const {
  return DAG.getNode(Opcode, SDLoc(Op), Op.getValueType(), Op.getOperand(1),
                     Op.getOperand(2),
                     Op.getOperand(3),
                     Op.getOperand(4));
}

SDValue SITargetLowering::LowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  if (Op.getValueType() != MVT::i64)
    return SDValue();

  SDLoc DL(Op);
  SDValue Cond = Op.getOperand(0);

  SDValue Zero = DAG.getConstant(0, MVT::i32);
  SDValue One = DAG.getConstant(1, MVT::i32);

  SDValue LHS = DAG.getNode(ISD::BITCAST, DL, MVT::v2i32, Op.getOperand(1));
  SDValue RHS = DAG.getNode(ISD::BITCAST, DL, MVT::v2i32, Op.getOperand(2));

  SDValue Lo0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, LHS, Zero);
  SDValue Lo1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, RHS, Zero);

  SDValue Lo = DAG.getSelect(DL, MVT::i32, Cond, Lo0, Lo1);

  SDValue Hi0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, LHS, One);
  SDValue Hi1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, RHS, One);

  SDValue Hi = DAG.getSelect(DL, MVT::i32, Cond, Hi0, Hi1);

  SDValue Res = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Lo, Hi);
  return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Res);
}

SDValue SITargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue True = Op.getOperand(2);
  SDValue False = Op.getOperand(3);
  SDValue CC = Op.getOperand(4);
  EVT VT = Op.getValueType();
  SDLoc DL(Op);

  SDValue Cond = DAG.getNode(ISD::SETCC, DL, MVT::i1, LHS, RHS, CC);
  return DAG.getNode(ISD::SELECT, DL, VT, Cond, True, False);
}

SDValue SITargetLowering::LowerSIGN_EXTEND(SDValue Op,
                                           SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  SDLoc DL(Op);

  if (VT != MVT::i64) {
    return SDValue();
  }

  SDValue Hi = DAG.getNode(ISD::SRA, DL, MVT::i32, Op.getOperand(0),
                                                 DAG.getConstant(31, MVT::i32));

  return DAG.getNode(ISD::BUILD_PAIR, DL, VT, Op.getOperand(0), Hi);
}

SDValue SITargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  StoreSDNode *Store = cast<StoreSDNode>(Op);
  EVT VT = Store->getMemoryVT();

  SDValue Ret = AMDGPUTargetLowering::LowerSTORE(Op, DAG);
  if (Ret.getNode())
    return Ret;

  if (VT.isVector() && VT.getVectorNumElements() >= 8)
      return SplitVectorStore(Op, DAG);

  if (VT == MVT::i1)
    return DAG.getTruncStore(Store->getChain(), DL,
                        DAG.getSExtOrTrunc(Store->getValue(), DL, MVT::i32),
                        Store->getBasePtr(), MVT::i1, Store->getMemOperand());

  if (Store->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS)
    return SDValue();

  SDValue Ptr = DAG.getNode(ISD::SRL, DL, MVT::i32, Store->getBasePtr(),
                            DAG.getConstant(2, MVT::i32));
  SDValue Chain = Store->getChain();
  SmallVector<SDValue, 8> Values;

  if (Store->isTruncatingStore()) {
    unsigned Mask = 0;
    if (Store->getMemoryVT() == MVT::i8) {
      Mask = 0xff;
    } else if (Store->getMemoryVT() == MVT::i16) {
      Mask = 0xffff;
    }
    SDValue Dst = DAG.getNode(AMDGPUISD::REGISTER_LOAD, DL, MVT::i32,
                              Chain, Store->getBasePtr(),
                              DAG.getConstant(0, MVT::i32));
    SDValue ByteIdx = DAG.getNode(ISD::AND, DL, MVT::i32, Store->getBasePtr(),
                                  DAG.getConstant(0x3, MVT::i32));
    SDValue ShiftAmt = DAG.getNode(ISD::SHL, DL, MVT::i32, ByteIdx,
                                   DAG.getConstant(3, MVT::i32));
    SDValue MaskedValue = DAG.getNode(ISD::AND, DL, MVT::i32, Store->getValue(),
                                      DAG.getConstant(Mask, MVT::i32));
    SDValue ShiftedValue = DAG.getNode(ISD::SHL, DL, MVT::i32,
                                       MaskedValue, ShiftAmt);
    SDValue RotrAmt = DAG.getNode(ISD::SUB, DL, MVT::i32,
                                  DAG.getConstant(32, MVT::i32), ShiftAmt);
    SDValue DstMask = DAG.getNode(ISD::ROTR, DL, MVT::i32,
                                  DAG.getConstant(Mask, MVT::i32),
                                  RotrAmt);
    Dst = DAG.getNode(ISD::AND, DL, MVT::i32, Dst, DstMask);
    Dst = DAG.getNode(ISD::OR, DL, MVT::i32, Dst, ShiftedValue);

    Values.push_back(Dst);
  } else if (VT == MVT::i64) {
    for (unsigned i = 0; i < 2; ++i) {
      Values.push_back(DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32,
                       Store->getValue(), DAG.getConstant(i, MVT::i32)));
    }
  } else if (VT == MVT::i128) {
    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned j = 0; j < 2; ++j) {
        Values.push_back(DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32,
                           DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i64,
                           Store->getValue(), DAG.getConstant(i, MVT::i32)),
                         DAG.getConstant(j, MVT::i32)));
      }
    }
  } else {
    Values.push_back(Store->getValue());
  }

  for (unsigned i = 0; i < Values.size(); ++i) {
    SDValue PartPtr = DAG.getNode(ISD::ADD, DL, MVT::i32,
                                  Ptr, DAG.getConstant(i, MVT::i32));
    Chain = DAG.getNode(AMDGPUISD::REGISTER_STORE, DL, MVT::Other,
                        Chain, Values[i], PartPtr,
                        DAG.getTargetConstant(0, MVT::i32));
  }
  return Chain;
}


SDValue SITargetLowering::LowerZERO_EXTEND(SDValue Op,
                                           SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  SDLoc DL(Op);

  if (VT != MVT::i64) {
    return SDValue();
  }

  SDValue Src = Op.getOperand(0);
  if (Src.getValueType() != MVT::i32)
    Src = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i32, Src);

  SDValue Zero = DAG.getConstant(0, MVT::i32);
  return DAG.getNode(ISD::BUILD_PAIR, DL, VT, Src, Zero);
}

//===----------------------------------------------------------------------===//
// Custom DAG optimizations
//===----------------------------------------------------------------------===//

SDValue SITargetLowering::PerformDAGCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  switch (N->getOpcode()) {
    default: return AMDGPUTargetLowering::PerformDAGCombine(N, DCI);
    case ISD::SELECT_CC: {
      ConstantSDNode *True, *False;
      // i1 selectcc(l, r, -1, 0, cc) -> i1 setcc(l, r, cc)
      if ((True = dyn_cast<ConstantSDNode>(N->getOperand(2)))
          && (False = dyn_cast<ConstantSDNode>(N->getOperand(3)))
          && True->isAllOnesValue()
          && False->isNullValue()
          && VT == MVT::i1) {
        return DAG.getNode(ISD::SETCC, DL, VT, N->getOperand(0),
                           N->getOperand(1), N->getOperand(4));

      }
      break;
    }
    case ISD::SETCC: {
      SDValue Arg0 = N->getOperand(0);
      SDValue Arg1 = N->getOperand(1);
      SDValue CC = N->getOperand(2);
      ConstantSDNode * C = nullptr;
      ISD::CondCode CCOp = dyn_cast<CondCodeSDNode>(CC)->get();

      // i1 setcc (sext(i1), 0, setne) -> i1 setcc(i1, 0, setne)
      if (VT == MVT::i1
          && Arg0.getOpcode() == ISD::SIGN_EXTEND
          && Arg0.getOperand(0).getValueType() == MVT::i1
          && (C = dyn_cast<ConstantSDNode>(Arg1))
          && C->isNullValue()
          && CCOp == ISD::SETNE) {
        return SimplifySetCC(VT, Arg0.getOperand(0),
                             DAG.getConstant(0, MVT::i1), CCOp, true, DCI, DL);
      }
      break;
    }
  }
  return SDValue();
}

/// \brief Test if RegClass is one of the VSrc classes
static bool isVSrc(unsigned RegClass) {
  return AMDGPU::VSrc_32RegClassID == RegClass ||
         AMDGPU::VSrc_64RegClassID == RegClass;
}

/// \brief Test if RegClass is one of the SSrc classes
static bool isSSrc(unsigned RegClass) {
  return AMDGPU::SSrc_32RegClassID == RegClass ||
         AMDGPU::SSrc_64RegClassID == RegClass;
}

/// \brief Analyze the possible immediate value Op
///
/// Returns -1 if it isn't an immediate, 0 if it's and inline immediate
/// and the immediate value if it's a literal immediate
int32_t SITargetLowering::analyzeImmediate(const SDNode *N) const {

  union {
    int32_t I;
    float F;
  } Imm;

  if (const ConstantSDNode *Node = dyn_cast<ConstantSDNode>(N)) {
    if (Node->getZExtValue() >> 32) {
        return -1;
    }
    Imm.I = Node->getSExtValue();
  } else if (const ConstantFPSDNode *Node = dyn_cast<ConstantFPSDNode>(N)) {
    if (N->getValueType(0) != MVT::f32)
      return -1;
    Imm.F = Node->getValueAPF().convertToFloat();
  } else
    return -1; // It isn't an immediate

  if ((Imm.I >= -16 && Imm.I <= 64) ||
      Imm.F == 0.5f || Imm.F == -0.5f ||
      Imm.F == 1.0f || Imm.F == -1.0f ||
      Imm.F == 2.0f || Imm.F == -2.0f ||
      Imm.F == 4.0f || Imm.F == -4.0f)
    return 0; // It's an inline immediate

  return Imm.I; // It's a literal immediate
}

/// \brief Try to fold an immediate directly into an instruction
bool SITargetLowering::foldImm(SDValue &Operand, int32_t &Immediate,
                               bool &ScalarSlotUsed) const {

  MachineSDNode *Mov = dyn_cast<MachineSDNode>(Operand);
  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo*>(getTargetMachine().getInstrInfo());
  if (!Mov || !TII->isMov(Mov->getMachineOpcode()))
    return false;

  const SDValue &Op = Mov->getOperand(0);
  int32_t Value = analyzeImmediate(Op.getNode());
  if (Value == -1) {
    // Not an immediate at all
    return false;

  } else if (Value == 0) {
    // Inline immediates can always be fold
    Operand = Op;
    return true;

  } else if (Value == Immediate) {
    // Already fold literal immediate
    Operand = Op;
    return true;

  } else if (!ScalarSlotUsed && !Immediate) {
    // Fold this literal immediate
    ScalarSlotUsed = true;
    Immediate = Value;
    Operand = Op;
    return true;

  }

  return false;
}

const TargetRegisterClass *SITargetLowering::getRegClassForNode(
                                   SelectionDAG &DAG, const SDValue &Op) const {
  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo*>(getTargetMachine().getInstrInfo());
  const SIRegisterInfo &TRI = TII->getRegisterInfo();

  if (!Op->isMachineOpcode()) {
    switch(Op->getOpcode()) {
    case ISD::CopyFromReg: {
      MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
      unsigned Reg = cast<RegisterSDNode>(Op->getOperand(1))->getReg();
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        return MRI.getRegClass(Reg);
      }
      return TRI.getPhysRegClass(Reg);
    }
    default:  return nullptr;
    }
  }
  const MCInstrDesc &Desc = TII->get(Op->getMachineOpcode());
  int OpClassID = Desc.OpInfo[Op.getResNo()].RegClass;
  if (OpClassID != -1) {
    return TRI.getRegClass(OpClassID);
  }
  switch(Op.getMachineOpcode()) {
  case AMDGPU::COPY_TO_REGCLASS:
    // Operand 1 is the register class id for COPY_TO_REGCLASS instructions.
    OpClassID = cast<ConstantSDNode>(Op->getOperand(1))->getZExtValue();

    // If the COPY_TO_REGCLASS instruction is copying to a VSrc register
    // class, then the register class for the value could be either a
    // VReg or and SReg.  In order to get a more accurate
    if (OpClassID == AMDGPU::VSrc_32RegClassID ||
        OpClassID == AMDGPU::VSrc_64RegClassID) {
      return getRegClassForNode(DAG, Op.getOperand(0));
    }
    return TRI.getRegClass(OpClassID);
  case AMDGPU::EXTRACT_SUBREG: {
    int SubIdx = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    const TargetRegisterClass *SuperClass =
      getRegClassForNode(DAG, Op.getOperand(0));
    return TRI.getSubClassWithSubReg(SuperClass, SubIdx);
  }
  case AMDGPU::REG_SEQUENCE:
    // Operand 0 is the register class id for REG_SEQUENCE instructions.
    return TRI.getRegClass(
      cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue());
  default:
    return getRegClassFor(Op.getSimpleValueType());
  }
}

/// \brief Does "Op" fit into register class "RegClass" ?
bool SITargetLowering::fitsRegClass(SelectionDAG &DAG, const SDValue &Op,
                                    unsigned RegClass) const {
  const TargetRegisterInfo *TRI = getTargetMachine().getRegisterInfo();
  const TargetRegisterClass *RC = getRegClassForNode(DAG, Op);
  if (!RC) {
    return false;
  }
  return TRI->getRegClass(RegClass)->hasSubClassEq(RC);
}

/// \brief Make sure that we don't exeed the number of allowed scalars
void SITargetLowering::ensureSRegLimit(SelectionDAG &DAG, SDValue &Operand,
                                       unsigned RegClass,
                                       bool &ScalarSlotUsed) const {

  // First map the operands register class to a destination class
  if (RegClass == AMDGPU::VSrc_32RegClassID)
    RegClass = AMDGPU::VReg_32RegClassID;
  else if (RegClass == AMDGPU::VSrc_64RegClassID)
    RegClass = AMDGPU::VReg_64RegClassID;
  else
    return;

  // Nothing to do if they fit naturally
  if (fitsRegClass(DAG, Operand, RegClass))
    return;

  // If the scalar slot isn't used yet use it now
  if (!ScalarSlotUsed) {
    ScalarSlotUsed = true;
    return;
  }

  // This is a conservative aproach. It is possible that we can't determine the
  // correct register class and copy too often, but better safe than sorry.
  SDValue RC = DAG.getTargetConstant(RegClass, MVT::i32);
  SDNode *Node = DAG.getMachineNode(TargetOpcode::COPY_TO_REGCLASS, SDLoc(),
                                    Operand.getValueType(), Operand, RC);
  Operand = SDValue(Node, 0);
}

/// \returns true if \p Node's operands are different from the SDValue list
/// \p Ops
static bool isNodeChanged(const SDNode *Node, const std::vector<SDValue> &Ops) {
  for (unsigned i = 0, e = Node->getNumOperands(); i < e; ++i) {
    if (Ops[i].getNode() != Node->getOperand(i).getNode()) {
      return true;
    }
  }
  return false;
}

/// \brief Try to fold the Nodes operands into the Node
SDNode *SITargetLowering::foldOperands(MachineSDNode *Node,
                                       SelectionDAG &DAG) const {

  // Original encoding (either e32 or e64)
  int Opcode = Node->getMachineOpcode();
  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo*>(getTargetMachine().getInstrInfo());
  const MCInstrDesc *Desc = &TII->get(Opcode);

  unsigned NumDefs = Desc->getNumDefs();
  unsigned NumOps = Desc->getNumOperands();

  // Commuted opcode if available
  int OpcodeRev = Desc->isCommutable() ? TII->commuteOpcode(Opcode) : -1;
  const MCInstrDesc *DescRev = OpcodeRev == -1 ? nullptr : &TII->get(OpcodeRev);

  assert(!DescRev || DescRev->getNumDefs() == NumDefs);
  assert(!DescRev || DescRev->getNumOperands() == NumOps);

  // e64 version if available, -1 otherwise
  int OpcodeE64 = AMDGPU::getVOPe64(Opcode);
  const MCInstrDesc *DescE64 = OpcodeE64 == -1 ? nullptr : &TII->get(OpcodeE64);
  int InputModifiers[3] = {0};

  assert(!DescE64 || DescE64->getNumDefs() == NumDefs);

  int32_t Immediate = Desc->getSize() == 4 ? 0 : -1;
  bool HaveVSrc = false, HaveSSrc = false;

  // First figure out what we alread have in this instruction
  for (unsigned i = 0, e = Node->getNumOperands(), Op = NumDefs;
       i != e && Op < NumOps; ++i, ++Op) {

    unsigned RegClass = Desc->OpInfo[Op].RegClass;
    if (isVSrc(RegClass))
      HaveVSrc = true;
    else if (isSSrc(RegClass))
      HaveSSrc = true;
    else
      continue;

    int32_t Imm = analyzeImmediate(Node->getOperand(i).getNode());
    if (Imm != -1 && Imm != 0) {
      // Literal immediate
      Immediate = Imm;
    }
  }

  // If we neither have VSrc nor SSrc it makes no sense to continue
  if (!HaveVSrc && !HaveSSrc)
    return Node;

  // No scalar allowed when we have both VSrc and SSrc
  bool ScalarSlotUsed = HaveVSrc && HaveSSrc;

  // Second go over the operands and try to fold them
  std::vector<SDValue> Ops;
  bool Promote2e64 = false;
  for (unsigned i = 0, e = Node->getNumOperands(), Op = NumDefs;
       i != e && Op < NumOps; ++i, ++Op) {

    const SDValue &Operand = Node->getOperand(i);
    Ops.push_back(Operand);

    // Already folded immediate ?
    if (isa<ConstantSDNode>(Operand.getNode()) ||
        isa<ConstantFPSDNode>(Operand.getNode()))
      continue;

    // Is this a VSrc or SSrc operand ?
    unsigned RegClass = Desc->OpInfo[Op].RegClass;
    if (isVSrc(RegClass) || isSSrc(RegClass)) {
      // Try to fold the immediates
      if (!foldImm(Ops[i], Immediate, ScalarSlotUsed)) {
        // Folding didn't worked, make sure we don't hit the SReg limit
        ensureSRegLimit(DAG, Ops[i], RegClass, ScalarSlotUsed);
      }
      continue;
    }

    if (i == 1 && DescRev && fitsRegClass(DAG, Ops[0], RegClass)) {

      unsigned OtherRegClass = Desc->OpInfo[NumDefs].RegClass;
      assert(isVSrc(OtherRegClass) || isSSrc(OtherRegClass));

      // Test if it makes sense to swap operands
      if (foldImm(Ops[1], Immediate, ScalarSlotUsed) ||
          (!fitsRegClass(DAG, Ops[1], RegClass) &&
           fitsRegClass(DAG, Ops[1], OtherRegClass))) {

        // Swap commutable operands
        std::swap(Ops[0], Ops[1]);

        Desc = DescRev;
        DescRev = nullptr;
        continue;
      }
    }

    if (Immediate)
      continue;

    if (DescE64) {

      // Test if it makes sense to switch to e64 encoding
      unsigned OtherRegClass = DescE64->OpInfo[Op].RegClass;
      if (!isVSrc(OtherRegClass) && !isSSrc(OtherRegClass))
        continue;

      int32_t TmpImm = -1;
      if (foldImm(Ops[i], TmpImm, ScalarSlotUsed) ||
          (!fitsRegClass(DAG, Ops[i], RegClass) &&
           fitsRegClass(DAG, Ops[1], OtherRegClass))) {

        // Switch to e64 encoding
        Immediate = -1;
        Promote2e64 = true;
        Desc = DescE64;
        DescE64 = nullptr;
      }
    }

    if (!DescE64 && !Promote2e64)
      continue;
    if (!Operand.isMachineOpcode())
      continue;
    if (Operand.getMachineOpcode() == AMDGPU::FNEG_SI) {
      Ops.pop_back();
      Ops.push_back(Operand.getOperand(0));
      InputModifiers[i] = 1;
      Promote2e64 = true;
      if (!DescE64)
        continue;
      Desc = DescE64;
      DescE64 = 0;
    }
    else if (Operand.getMachineOpcode() == AMDGPU::FABS_SI) {
      Ops.pop_back();
      Ops.push_back(Operand.getOperand(0));
      InputModifiers[i] = 2;
      Promote2e64 = true;
      if (!DescE64)
        continue;
      Desc = DescE64;
      DescE64 = 0;
    }
  }

  if (Promote2e64) {
    std::vector<SDValue> OldOps(Ops);
    Ops.clear();
    for (unsigned i = 0; i < OldOps.size(); ++i) {
      // src_modifier
      Ops.push_back(DAG.getTargetConstant(InputModifiers[i], MVT::i32));
      Ops.push_back(OldOps[i]);
    }
    // Add the modifier flags while promoting
    for (unsigned i = 0; i < 2; ++i)
      Ops.push_back(DAG.getTargetConstant(0, MVT::i32));
  }

  // Add optional chain and glue
  for (unsigned i = NumOps - NumDefs, e = Node->getNumOperands(); i < e; ++i)
    Ops.push_back(Node->getOperand(i));

  // Nodes that have a glue result are not CSE'd by getMachineNode(), so in
  // this case a brand new node is always be created, even if the operands
  // are the same as before.  So, manually check if anything has been changed.
  if (Desc->Opcode == Opcode && !isNodeChanged(Node, Ops)) {
    return Node;
  }

  // Create a complete new instruction
  return DAG.getMachineNode(Desc->Opcode, SDLoc(Node), Node->getVTList(), Ops);
}

/// \brief Helper function for adjustWritemask
static unsigned SubIdx2Lane(unsigned Idx) {
  switch (Idx) {
  default: return 0;
  case AMDGPU::sub0: return 0;
  case AMDGPU::sub1: return 1;
  case AMDGPU::sub2: return 2;
  case AMDGPU::sub3: return 3;
  }
}

/// \brief Adjust the writemask of MIMG instructions
void SITargetLowering::adjustWritemask(MachineSDNode *&Node,
                                       SelectionDAG &DAG) const {
  SDNode *Users[4] = { };
  unsigned Lane = 0;
  unsigned OldDmask = Node->getConstantOperandVal(0);
  unsigned NewDmask = 0;

  // Try to figure out the used register components
  for (SDNode::use_iterator I = Node->use_begin(), E = Node->use_end();
       I != E; ++I) {

    // Abort if we can't understand the usage
    if (!I->isMachineOpcode() ||
        I->getMachineOpcode() != TargetOpcode::EXTRACT_SUBREG)
      return;

    // Lane means which subreg of %VGPRa_VGPRb_VGPRc_VGPRd is used.
    // Note that subregs are packed, i.e. Lane==0 is the first bit set
    // in OldDmask, so it can be any of X,Y,Z,W; Lane==1 is the second bit
    // set, etc.
    Lane = SubIdx2Lane(I->getConstantOperandVal(1));

    // Set which texture component corresponds to the lane.
    unsigned Comp;
    for (unsigned i = 0, Dmask = OldDmask; i <= Lane; i++) {
      assert(Dmask);
      Comp = countTrailingZeros(Dmask);
      Dmask &= ~(1 << Comp);
    }

    // Abort if we have more than one user per component
    if (Users[Lane])
      return;

    Users[Lane] = *I;
    NewDmask |= 1 << Comp;
  }

  // Abort if there's no change
  if (NewDmask == OldDmask)
    return;

  // Adjust the writemask in the node
  std::vector<SDValue> Ops;
  Ops.push_back(DAG.getTargetConstant(NewDmask, MVT::i32));
  for (unsigned i = 1, e = Node->getNumOperands(); i != e; ++i)
    Ops.push_back(Node->getOperand(i));
  Node = (MachineSDNode*)DAG.UpdateNodeOperands(Node, Ops);

  // If we only got one lane, replace it with a copy
  // (if NewDmask has only one bit set...)
  if (NewDmask && (NewDmask & (NewDmask-1)) == 0) {
    SDValue RC = DAG.getTargetConstant(AMDGPU::VReg_32RegClassID, MVT::i32);
    SDNode *Copy = DAG.getMachineNode(TargetOpcode::COPY_TO_REGCLASS,
                                      SDLoc(), Users[Lane]->getValueType(0),
                                      SDValue(Node, 0), RC);
    DAG.ReplaceAllUsesWith(Users[Lane], Copy);
    return;
  }

  // Update the users of the node with the new indices
  for (unsigned i = 0, Idx = AMDGPU::sub0; i < 4; ++i) {

    SDNode *User = Users[i];
    if (!User)
      continue;

    SDValue Op = DAG.getTargetConstant(Idx, MVT::i32);
    DAG.UpdateNodeOperands(User, User->getOperand(0), Op);

    switch (Idx) {
    default: break;
    case AMDGPU::sub0: Idx = AMDGPU::sub1; break;
    case AMDGPU::sub1: Idx = AMDGPU::sub2; break;
    case AMDGPU::sub2: Idx = AMDGPU::sub3; break;
    }
  }
}

/// \brief Fold the instructions after slecting them
SDNode *SITargetLowering::PostISelFolding(MachineSDNode *Node,
                                          SelectionDAG &DAG) const {
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo*>(getTargetMachine().getInstrInfo());
  Node = AdjustRegClass(Node, DAG);

  if (TII->isMIMG(Node->getMachineOpcode()))
    adjustWritemask(Node, DAG);

  return foldOperands(Node, DAG);
}

/// \brief Assign the register class depending on the number of
/// bits set in the writemask
void SITargetLowering::AdjustInstrPostInstrSelection(MachineInstr *MI,
                                                     SDNode *Node) const {
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo*>(getTargetMachine().getInstrInfo());
  if (!TII->isMIMG(MI->getOpcode()))
    return;

  unsigned VReg = MI->getOperand(0).getReg();
  unsigned Writemask = MI->getOperand(1).getImm();
  unsigned BitsSet = 0;
  for (unsigned i = 0; i < 4; ++i)
    BitsSet += Writemask & (1 << i) ? 1 : 0;

  const TargetRegisterClass *RC;
  switch (BitsSet) {
  default: return;
  case 1:  RC = &AMDGPU::VReg_32RegClass; break;
  case 2:  RC = &AMDGPU::VReg_64RegClass; break;
  case 3:  RC = &AMDGPU::VReg_96RegClass; break;
  }

  unsigned NewOpcode = TII->getMaskedMIMGOp(MI->getOpcode(), BitsSet);
  MI->setDesc(TII->get(NewOpcode));
  MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  MRI.setRegClass(VReg, RC);
}

MachineSDNode *SITargetLowering::AdjustRegClass(MachineSDNode *N,
                                                SelectionDAG &DAG) const {

  SDLoc DL(N);
  unsigned NewOpcode = N->getMachineOpcode();

  switch (N->getMachineOpcode()) {
  default: return N;
  case AMDGPU::S_LOAD_DWORD_IMM:
    NewOpcode = AMDGPU::BUFFER_LOAD_DWORD_ADDR64;
    // Fall-through
  case AMDGPU::S_LOAD_DWORDX2_SGPR:
    if (NewOpcode == N->getMachineOpcode()) {
      NewOpcode = AMDGPU::BUFFER_LOAD_DWORDX2_ADDR64;
    }
    // Fall-through
  case AMDGPU::S_LOAD_DWORDX4_IMM:
  case AMDGPU::S_LOAD_DWORDX4_SGPR: {
    if (NewOpcode == N->getMachineOpcode()) {
      NewOpcode = AMDGPU::BUFFER_LOAD_DWORDX4_ADDR64;
    }
    if (fitsRegClass(DAG, N->getOperand(0), AMDGPU::SReg_64RegClassID)) {
      return N;
    }
    ConstantSDNode *Offset = cast<ConstantSDNode>(N->getOperand(1));
    SDValue Ops[] = {
      SDValue(DAG.getMachineNode(AMDGPU::SI_ADDR64_RSRC, DL, MVT::i128,
                                 DAG.getConstant(0, MVT::i64)), 0),
      N->getOperand(0),
      DAG.getConstant(Offset->getSExtValue() << 2, MVT::i32)
    };
    return DAG.getMachineNode(NewOpcode, DL, N->getVTList(), Ops);
  }
  }
}

SDValue SITargetLowering::CreateLiveInRegister(SelectionDAG &DAG,
                                               const TargetRegisterClass *RC,
                                               unsigned Reg, EVT VT) const {
  SDValue VReg = AMDGPUTargetLowering::CreateLiveInRegister(DAG, RC, Reg, VT);

  return DAG.getCopyFromReg(DAG.getEntryNode(), SDLoc(DAG.getEntryNode()),
                            cast<RegisterSDNode>(VReg)->getReg(), VT);
}
