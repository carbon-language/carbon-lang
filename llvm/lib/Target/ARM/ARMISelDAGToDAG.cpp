//===-- ARMISelDAGToDAG.cpp - A dag to dag inst selector for ARM ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the ARM target.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMTargetMachine.h"
#include "llvm/CallingConv.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Constants.h"
#include "llvm/Intrinsics.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/Debug.h"
#include <vector>
using namespace llvm;

namespace {
  class ARMTargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
  public:
    ARMTargetLowering(TargetMachine &TM);
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    virtual const char *getTargetNodeName(unsigned Opcode) const;
    std::vector<unsigned>
    getRegClassForInlineAsmConstraint(const std::string &Constraint,
				      MVT::ValueType VT) const;
  };

}

ARMTargetLowering::ARMTargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  addRegisterClass(MVT::i32, ARM::IntRegsRegisterClass);
  addRegisterClass(MVT::f32, ARM::FPRegsRegisterClass);
  addRegisterClass(MVT::f64, ARM::DFPRegsRegisterClass);

  setLoadXAction(ISD::EXTLOAD, MVT::f32, Expand);

  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);

  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Custom);

  setOperationAction(ISD::RET,           MVT::Other, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i32,   Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i32,   Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8 , Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1 , Expand);

  setOperationAction(ISD::SELECT, MVT::i32, Expand);

  setOperationAction(ISD::SETCC, MVT::i32, Expand);
  setOperationAction(ISD::SETCC, MVT::f32, Expand);
  setOperationAction(ISD::SETCC, MVT::f64, Expand);

  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);

  setOperationAction(ISD::MEMMOVE, MVT::Other, Expand);
  setOperationAction(ISD::MEMSET, MVT::Other, Expand);
  setOperationAction(ISD::MEMCPY, MVT::Other, Expand);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f64, Custom);

  setOperationAction(ISD::BRCOND,        MVT::Other, Expand);

  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SDIV,      MVT::i32, Expand);
  setOperationAction(ISD::UDIV,      MVT::i32, Expand);
  setOperationAction(ISD::SREM,      MVT::i32, Expand);
  setOperationAction(ISD::UREM,      MVT::i32, Expand);

  setOperationAction(ISD::VASTART,       MVT::Other, Custom);
  setOperationAction(ISD::VACOPY,            MVT::Other, Expand);
  setOperationAction(ISD::VAEND,         MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Expand);

  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f32, Expand);

  setStackPointerRegisterToSaveRestore(ARM::R13);

  setSchedulingPreference(SchedulingForRegPressure);
  computeRegisterProperties();
}

namespace llvm {
  namespace ARMISD {
    enum NodeType {
      // Start the numbering where the builting ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+ARM::INSTRUCTION_LIST_END,
      /// CALL - A direct function call.
      CALL,

      /// Return with a flag operand.
      RET_FLAG,

      CMP,

      SELECT,

      BR,

      FSITOS,
      FTOSIS,

      FSITOD,
      FTOSID,

      FUITOS,
      FTOUIS,

      FUITOD,
      FTOUID,

      FMRRD,

      FMDRR,

      FMSTAT
    };
  }
}

/// DAGFPCCToARMCC - Convert a DAG fp condition code to an ARM CC
// Unordered = !N & !Z & C & V = V
// Ordered   =  N | Z | !C | !V = N | Z | !V
static ARMCC::CondCodes DAGFPCCToARMCC(ISD::CondCode CC) {
  switch (CC) {
  default:
    assert(0 && "Unknown fp condition code!");
// SETOEQ = (N | Z | !V) & Z = Z                               = EQ
  case ISD::SETEQ:
  case ISD::SETOEQ: return ARMCC::EQ;
// SETOGT = (N | Z | !V) & !N & !Z = !V &!N &!Z = (N = V) & !Z = GT
  case ISD::SETGT:
  case ISD::SETOGT: return ARMCC::GT;
// SETOGE = (N | Z | !V) & !N = (Z | !V) & !N = !V & !N        = GE
  case ISD::SETGE:
  case ISD::SETOGE: return ARMCC::GE;
// SETOLT = (N | Z | !V) & N = N                               = MI
  case ISD::SETLT:
  case ISD::SETOLT: return ARMCC::MI;
// SETOLE = (N | Z | !V) & (N | Z) = N | Z = !C | Z            = LS
  case ISD::SETLE:
  case ISD::SETOLE: return ARMCC::LS;
// SETONE = (N | Z | !V) & !Z = (N | !V) & Z = !V & Z = Z      = NE
  case ISD::SETNE:
  case ISD::SETONE: return ARMCC::NE;
// SETO   = N | Z | !V = Z | !V = !V                           = VC
  case ISD::SETO:   return ARMCC::VC;
// SETUO  = V                                                  = VS
  case ISD::SETUO:  return ARMCC::VS;
// SETUEQ = V | Z                                              = ??
// SETUGT = V | (!Z & !N) = !Z & !N = !Z & C                   = HI
  case ISD::SETUGT: return ARMCC::HI;
// SETUGE = V | !N = !N                                        = PL
  case ISD::SETUGE: return ARMCC::PL;
// SETULT = V | N                                              = ??
// SETULE = V | Z | N                                          = ??
// SETUNE = V | !Z = !Z                                        = NE
  case ISD::SETUNE: return ARMCC::NE;
  }
}

/// DAGIntCCToARMCC - Convert a DAG integer condition code to an ARM CC
static ARMCC::CondCodes DAGIntCCToARMCC(ISD::CondCode CC) {
  switch (CC) {
  default:
    assert(0 && "Unknown integer condition code!");
  case ISD::SETEQ:  return ARMCC::EQ;
  case ISD::SETNE:  return ARMCC::NE;
  case ISD::SETLT:  return ARMCC::LT;
  case ISD::SETLE:  return ARMCC::LE;
  case ISD::SETGT:  return ARMCC::GT;
  case ISD::SETGE:  return ARMCC::GE;
  case ISD::SETULT: return ARMCC::CC;
  case ISD::SETULE: return ARMCC::LS;
  case ISD::SETUGT: return ARMCC::HI;
  case ISD::SETUGE: return ARMCC::CS;
  }
}

std::vector<unsigned> ARMTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const {
  if (Constraint.size() == 1) {
    // FIXME: handling only r regs
    switch (Constraint[0]) {
    default: break;  // Unknown constraint letter

    case 'r':   // GENERAL_REGS
    case 'R':   // LEGACY_REGS
      if (VT == MVT::i32)
        return make_vector<unsigned>(ARM::R0,  ARM::R1,  ARM::R2,  ARM::R3,
                                     ARM::R4,  ARM::R5,  ARM::R6,  ARM::R7,
                                     ARM::R8,  ARM::R9,  ARM::R10, ARM::R11,
                                     ARM::R12, ARM::R13, ARM::R14, 0);
      break;

    }
  }

  return std::vector<unsigned>();
}

const char *ARMTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case ARMISD::CALL:          return "ARMISD::CALL";
  case ARMISD::RET_FLAG:      return "ARMISD::RET_FLAG";
  case ARMISD::SELECT:        return "ARMISD::SELECT";
  case ARMISD::CMP:           return "ARMISD::CMP";
  case ARMISD::BR:            return "ARMISD::BR";
  case ARMISD::FSITOS:        return "ARMISD::FSITOS";
  case ARMISD::FTOSIS:        return "ARMISD::FTOSIS";
  case ARMISD::FSITOD:        return "ARMISD::FSITOD";
  case ARMISD::FTOSID:        return "ARMISD::FTOSID";
  case ARMISD::FUITOS:        return "ARMISD::FUITOS";
  case ARMISD::FTOUIS:        return "ARMISD::FTOUIS";
  case ARMISD::FUITOD:        return "ARMISD::FUITOD";
  case ARMISD::FTOUID:        return "ARMISD::FTOUID";
  case ARMISD::FMRRD:         return "ARMISD::FMRRD";
  case ARMISD::FMDRR:         return "ARMISD::FMDRR";
  case ARMISD::FMSTAT:        return "ARMISD::FMSTAT";
  }
}

class ArgumentLayout {
  std::vector<bool>           is_reg;
  std::vector<unsigned>       pos;
  std::vector<MVT::ValueType> types;
public:
  ArgumentLayout(const std::vector<MVT::ValueType> &Types) {
    types = Types;

    unsigned      RegNum = 0;
    unsigned StackOffset = 0;
    for(std::vector<MVT::ValueType>::const_iterator I = Types.begin();
        I != Types.end();
        ++I) {
      MVT::ValueType VT = *I;
      assert(VT == MVT::i32 || VT == MVT::f32 || VT == MVT::f64);
      unsigned     size = MVT::getSizeInBits(VT)/32;

      RegNum = ((RegNum + size - 1) / size) * size;
      if (RegNum < 4) {
        pos.push_back(RegNum);
        is_reg.push_back(true);
        RegNum += size;
      } else {
        unsigned bytes = size * 32/8;
        StackOffset = ((StackOffset + bytes - 1) / bytes) * bytes;
        pos.push_back(StackOffset);
        is_reg.push_back(false);
        StackOffset += bytes;
      }
    }
  }
  unsigned getRegisterNum(unsigned argNum) {
    assert(isRegister(argNum));
    return pos[argNum];
  }
  unsigned getOffset(unsigned argNum) {
    assert(isOffset(argNum));
    return pos[argNum];
  }
  unsigned isRegister(unsigned argNum) {
    assert(argNum < is_reg.size());
    return is_reg[argNum];
  }
  unsigned isOffset(unsigned argNum) {
    return !isRegister(argNum);
  }
  MVT::ValueType getType(unsigned argNum) {
    assert(argNum < types.size());
    return types[argNum];
  }
  unsigned getStackSize(void) {
    int last = is_reg.size() - 1;
    if (last < 0)
      return 0;
    if (isRegister(last))
      return 0;
    return getOffset(last) + MVT::getSizeInBits(getType(last))/8;
  }
  int lastRegArg(void) {
    int size = is_reg.size();
    int last = 0;
    while(last < size && isRegister(last))
      last++;
    last--;
    return last;
  }
  int lastRegNum(void) {
    int            l = lastRegArg();
    if (l < 0)
      return -1;
    unsigned       r = getRegisterNum(l);
    MVT::ValueType t = getType(l);
    assert(t == MVT::i32 || t == MVT::f32 || t == MVT::f64);
    if (t == MVT::f64)
      return r + 1;
    return r;
  }
};

// This transforms a ISD::CALL node into a
// callseq_star <- ARMISD:CALL <- callseq_end
// chain
static SDOperand LowerCALL(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Chain    = Op.getOperand(0);
  unsigned CallConv  = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
  assert((CallConv == CallingConv::C ||
          CallConv == CallingConv::Fast)
         && "unknown calling convention");
  SDOperand Callee   = Op.getOperand(4);
  unsigned NumOps    = (Op.getNumOperands() - 5) / 2;
  SDOperand StackPtr = DAG.getRegister(ARM::R13, MVT::i32);
  static const unsigned regs[] = {
    ARM::R0, ARM::R1, ARM::R2, ARM::R3
  };

  std::vector<MVT::ValueType> Types;
  for (unsigned i = 0; i < NumOps; ++i) {
    MVT::ValueType VT = Op.getOperand(5+2*i).getValueType();
    Types.push_back(VT);
  }
  ArgumentLayout Layout(Types);

  unsigned NumBytes = Layout.getStackSize();

  Chain = DAG.getCALLSEQ_START(Chain,
                               DAG.getConstant(NumBytes, MVT::i32));

  //Build a sequence of stores
  std::vector<SDOperand> MemOpChains;
  for (unsigned i = Layout.lastRegArg() + 1; i < NumOps; ++i) {
    SDOperand      Arg = Op.getOperand(5+2*i);
    unsigned ArgOffset = Layout.getOffset(i);
    SDOperand   PtrOff = DAG.getConstant(ArgOffset, StackPtr.getValueType());
    PtrOff             = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
    MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
  }
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  assert(Callee.getValueType() == MVT::i32);
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i32);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), MVT::i32);

  // If this is a direct call, pass the chain and the callee.
  assert (Callee.Val);
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDOperand InFlag;
  for (int i = 0, e = Layout.lastRegArg(); i <= e; ++i) {
    SDOperand     Arg = Op.getOperand(5+2*i);
    unsigned   RegNum = Layout.getRegisterNum(i);
    unsigned     Reg1 = regs[RegNum];
    MVT::ValueType VT = Layout.getType(i);
    assert(VT == Arg.getValueType());
    assert(VT == MVT::i32 || VT == MVT::f32 || VT == MVT::f64);

    // Add argument register to the end of the list so that it is known live
    // into the call.
    Ops.push_back(DAG.getRegister(Reg1, MVT::i32));
    if (VT == MVT::f64) {
      unsigned    Reg2 = regs[RegNum + 1];
      SDOperand SDReg1 = DAG.getRegister(Reg1, MVT::i32);
      SDOperand SDReg2 = DAG.getRegister(Reg2, MVT::i32);

      Ops.push_back(DAG.getRegister(Reg2, MVT::i32));
      SDVTList    VTs = DAG.getVTList(MVT::Other, MVT::Flag);
      SDOperand Ops[] = {Chain, SDReg1, SDReg2, Arg, InFlag};
      Chain = DAG.getNode(ARMISD::FMRRD, VTs, Ops, InFlag.Val ? 5 : 4);
    } else {
      if (VT == MVT::f32)
        Arg = DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Arg);
      Chain = DAG.getCopyToReg(Chain, Reg1, Arg, InFlag);
    }
    InFlag = Chain.getValue(1);
  }

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.

  unsigned CallOpc = ARMISD::CALL;
  if (InFlag.Val)
    Ops.push_back(InFlag);
  Chain = DAG.getNode(CallOpc, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  std::vector<SDOperand> ResultVals;
  NodeTys.clear();

  // If the call has results, copy the values out of the ret val registers.
  MVT::ValueType VT = Op.Val->getValueType(0);
  if (VT != MVT::Other) {
    assert(VT == MVT::i32 || VT == MVT::f32 || VT == MVT::f64);

    SDOperand Value1 = DAG.getCopyFromReg(Chain, ARM::R0, MVT::i32, InFlag);
    Chain            = Value1.getValue(1);
    InFlag           = Value1.getValue(2);
    NodeTys.push_back(VT);
    if (VT == MVT::i32) {
      ResultVals.push_back(Value1);
      if (Op.Val->getValueType(1) == MVT::i32) {
        SDOperand Value2 = DAG.getCopyFromReg(Chain, ARM::R1, MVT::i32, InFlag);
        Chain            = Value2.getValue(1);
        ResultVals.push_back(Value2);
        NodeTys.push_back(VT);
      }
    }
    if (VT == MVT::f32) {
      SDOperand Value = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, Value1);
      ResultVals.push_back(Value);
    }
    if (VT == MVT::f64) {
      SDOperand Value2 = DAG.getCopyFromReg(Chain, ARM::R1, MVT::i32, InFlag);
      Chain            = Value2.getValue(1);
      SDOperand Value  = DAG.getNode(ARMISD::FMDRR, MVT::f64, Value1, Value2);
      ResultVals.push_back(Value);
    }
  }

  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, MVT::i32));
  NodeTys.push_back(MVT::Other);

  if (ResultVals.empty())
    return Chain;

  ResultVals.push_back(Chain);
  SDOperand Res = DAG.getNode(ISD::MERGE_VALUES, NodeTys, &ResultVals[0],
                              ResultVals.size());
  return Res.getValue(Op.ResNo);
}

static SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Copy;
  SDOperand Chain = Op.getOperand(0);
  SDOperand    R0 = DAG.getRegister(ARM::R0, MVT::i32);
  SDOperand    R1 = DAG.getRegister(ARM::R1, MVT::i32);

  switch(Op.getNumOperands()) {
  default:
    assert(0 && "Do not know how to return this many arguments!");
    abort();
  case 1: {
    SDOperand LR = DAG.getRegister(ARM::R14, MVT::i32);
    return DAG.getNode(ARMISD::RET_FLAG, MVT::Other, Chain);
  }
  case 3: {
    SDOperand Val = Op.getOperand(1);
    assert(Val.getValueType() == MVT::i32 ||
	   Val.getValueType() == MVT::f32 ||
	   Val.getValueType() == MVT::f64);

    if (Val.getValueType() == MVT::f64) {
      SDVTList    VTs = DAG.getVTList(MVT::Other, MVT::Flag);
      SDOperand Ops[] = {Chain, R0, R1, Val};
      Copy  = DAG.getNode(ARMISD::FMRRD, VTs, Ops, 4);
    } else {
      if (Val.getValueType() == MVT::f32)
	Val = DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Val);
      Copy = DAG.getCopyToReg(Chain, R0, Val, SDOperand());
    }

    if (DAG.getMachineFunction().liveout_empty()) {
      DAG.getMachineFunction().addLiveOut(ARM::R0);
      if (Val.getValueType() == MVT::f64)
        DAG.getMachineFunction().addLiveOut(ARM::R1);
    }
    break;
  }
  case 5:
    Copy = DAG.getCopyToReg(Chain, ARM::R1, Op.getOperand(3), SDOperand());
    Copy = DAG.getCopyToReg(Copy, ARM::R0, Op.getOperand(1), Copy.getValue(1));
    // If we haven't noted the R0+R1 are live out, do so now.
    if (DAG.getMachineFunction().liveout_empty()) {
      DAG.getMachineFunction().addLiveOut(ARM::R0);
      DAG.getMachineFunction().addLiveOut(ARM::R1);
    }
    break;
  }

  //We must use RET_FLAG instead of BRIND because BRIND doesn't have a flag
  return DAG.getNode(ARMISD::RET_FLAG, MVT::Other, Copy, Copy.getValue(1));
}

static SDOperand LowerConstantPool(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType PtrVT = Op.getValueType();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  Constant *C = CP->getConstVal();
  SDOperand CPI = DAG.getTargetConstantPool(C, PtrVT, CP->getAlignment());

  return CPI;
}

static SDOperand LowerGlobalAddress(SDOperand Op,
				    SelectionDAG &DAG) {
  GlobalValue  *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  int alignment = 2;
  SDOperand CPAddr = DAG.getConstantPool(GV, MVT::i32, alignment);
  return DAG.getLoad(MVT::i32, DAG.getEntryNode(), CPAddr, NULL, 0);
}

static SDOperand LowerVASTART(SDOperand Op, SelectionDAG &DAG,
                              unsigned VarArgsFrameIndex) {
  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  MVT::ValueType PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, PtrVT);
  SrcValueSDNode *SV = cast<SrcValueSDNode>(Op.getOperand(2));
  return DAG.getStore(Op.getOperand(0), FR, Op.getOperand(1), SV->getValue(),
                      SV->getOffset());
}

static SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG,
				       int &VarArgsFrameIndex) {
  MachineFunction   &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  SSARegMap     *RegMap = MF.getSSARegMap();
  unsigned      NumArgs = Op.Val->getNumValues()-1;
  SDOperand        Root = Op.getOperand(0);
  bool         isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  static const unsigned REGS[] = {
    ARM::R0, ARM::R1, ARM::R2, ARM::R3
  };

  std::vector<MVT::ValueType> Types(Op.Val->value_begin(), Op.Val->value_end() - 1);
  ArgumentLayout Layout(Types);

  std::vector<SDOperand> ArgValues;
  for (unsigned ArgNo = 0; ArgNo < NumArgs; ++ArgNo) {
    MVT::ValueType VT = Types[ArgNo];

    SDOperand Value;
    if (Layout.isRegister(ArgNo)) {
      assert(VT == MVT::i32 || VT == MVT::f32 || VT == MVT::f64);
      unsigned  RegNum = Layout.getRegisterNum(ArgNo);
      unsigned    Reg1 = REGS[RegNum];
      unsigned   VReg1 = RegMap->createVirtualRegister(&ARM::IntRegsRegClass);
      SDOperand Value1 = DAG.getCopyFromReg(Root, VReg1, MVT::i32);
      MF.addLiveIn(Reg1, VReg1);
      if (VT == MVT::f64) {
        unsigned    Reg2 = REGS[RegNum + 1];
        unsigned   VReg2 = RegMap->createVirtualRegister(&ARM::IntRegsRegClass);
        SDOperand Value2 = DAG.getCopyFromReg(Root, VReg2, MVT::i32);
        MF.addLiveIn(Reg2, VReg2);
        Value            = DAG.getNode(ARMISD::FMDRR, MVT::f64, Value1, Value2);
      } else {
        Value = Value1;
        if (VT == MVT::f32)
          Value = DAG.getNode(ISD::BIT_CONVERT, VT, Value);
      }
    } else {
      // If the argument is actually used, emit a load from the right stack
      // slot.
      if (!Op.Val->hasNUsesOfValue(0, ArgNo)) {
        unsigned Offset = Layout.getOffset(ArgNo);
        unsigned   Size = MVT::getSizeInBits(VT)/8;
        int          FI = MFI->CreateFixedObject(Size, Offset);
        SDOperand   FIN = DAG.getFrameIndex(FI, VT);
        Value = DAG.getLoad(VT, Root, FIN, NULL, 0);
      } else {
        Value = DAG.getNode(ISD::UNDEF, VT);
      }
    }
    ArgValues.push_back(Value);
  }

  unsigned NextRegNum = Layout.lastRegNum() + 1;

  if (isVarArg) {
    //If this function is vararg we must store the remaing
    //registers so that they can be acessed with va_start
    VarArgsFrameIndex = MFI->CreateFixedObject(MVT::getSizeInBits(MVT::i32)/8,
                                               -16 + NextRegNum * 4);

    SmallVector<SDOperand, 4> MemOps;
    for (unsigned RegNo = NextRegNum; RegNo < 4; ++RegNo) {
      int RegOffset = - (4 - RegNo) * 4;
      int FI = MFI->CreateFixedObject(MVT::getSizeInBits(MVT::i32)/8,
				      RegOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);

      unsigned VReg = RegMap->createVirtualRegister(&ARM::IntRegsRegClass);
      MF.addLiveIn(REGS[RegNo], VReg);

      SDOperand Val = DAG.getCopyFromReg(Root, VReg, MVT::i32);
      SDOperand Store = DAG.getStore(Val.getValue(1), Val, FIN, NULL, 0);
      MemOps.push_back(Store);
    }
    Root = DAG.getNode(ISD::TokenFactor, MVT::Other,&MemOps[0],MemOps.size());
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  std::vector<MVT::ValueType> RetVT(Op.Val->value_begin(),
                                    Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVT, &ArgValues[0], ArgValues.size());
}

static SDOperand GetCMP(ISD::CondCode CC, SDOperand LHS, SDOperand RHS,
                        SelectionDAG &DAG) {
  MVT::ValueType vt = LHS.getValueType();
  assert(vt == MVT::i32 || vt == MVT::f32 || vt == MVT::f64);

  SDOperand Cmp = DAG.getNode(ARMISD::CMP,  MVT::Flag, LHS, RHS);

  if (vt != MVT::i32)
    Cmp = DAG.getNode(ARMISD::FMSTAT, MVT::Flag, Cmp);
  return Cmp;
}

static SDOperand GetARMCC(ISD::CondCode CC, MVT::ValueType vt,
                          SelectionDAG &DAG) {
  assert(vt == MVT::i32 || vt == MVT::f32 || vt == MVT::f64);
  if (vt == MVT::i32)
    return DAG.getConstant(DAGIntCCToARMCC(CC), MVT::i32);
  else
    return DAG.getConstant(DAGFPCCToARMCC(CC), MVT::i32);
}

static SDOperand LowerSELECT_CC(SDOperand Op, SelectionDAG &DAG) {
  SDOperand LHS = Op.getOperand(0);
  SDOperand RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDOperand TrueVal = Op.getOperand(2);
  SDOperand FalseVal = Op.getOperand(3);
  SDOperand      Cmp = GetCMP(CC, LHS, RHS, DAG);
  SDOperand    ARMCC = GetARMCC(CC, LHS.getValueType(), DAG);
  return DAG.getNode(ARMISD::SELECT, MVT::i32, TrueVal, FalseVal, ARMCC, Cmp);
}

static SDOperand LowerBR_CC(SDOperand Op, SelectionDAG &DAG) {
  SDOperand  Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDOperand    LHS = Op.getOperand(2);
  SDOperand    RHS = Op.getOperand(3);
  SDOperand   Dest = Op.getOperand(4);
  SDOperand    Cmp = GetCMP(CC, LHS, RHS, DAG);
  SDOperand  ARMCC = GetARMCC(CC, LHS.getValueType(), DAG);
  return DAG.getNode(ARMISD::BR, MVT::Other, Chain, Dest, ARMCC, Cmp);
}

static SDOperand LowerSINT_TO_FP(SDOperand Op, SelectionDAG &DAG) {
  SDOperand IntVal  = Op.getOperand(0);
  assert(IntVal.getValueType() == MVT::i32);
  MVT::ValueType vt = Op.getValueType();
  assert(vt == MVT::f32 ||
         vt == MVT::f64);

  SDOperand Tmp = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, IntVal);
  ARMISD::NodeType op = vt == MVT::f32 ? ARMISD::FSITOS : ARMISD::FSITOD;
  return DAG.getNode(op, vt, Tmp);
}

static SDOperand LowerFP_TO_SINT(SDOperand Op, SelectionDAG &DAG) {
  assert(Op.getValueType() == MVT::i32);
  SDOperand FloatVal = Op.getOperand(0);
  MVT::ValueType  vt = FloatVal.getValueType();
  assert(vt == MVT::f32 || vt == MVT::f64);

  ARMISD::NodeType op = vt == MVT::f32 ? ARMISD::FTOSIS : ARMISD::FTOSID;
  SDOperand Tmp = DAG.getNode(op, MVT::f32, FloatVal);
  return DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Tmp);
}

static SDOperand LowerUINT_TO_FP(SDOperand Op, SelectionDAG &DAG) {
  SDOperand IntVal  = Op.getOperand(0);
  assert(IntVal.getValueType() == MVT::i32);
  MVT::ValueType vt = Op.getValueType();
  assert(vt == MVT::f32 ||
         vt == MVT::f64);

  SDOperand Tmp = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, IntVal);
  ARMISD::NodeType op = vt == MVT::f32 ? ARMISD::FUITOS : ARMISD::FUITOD;
  return DAG.getNode(op, vt, Tmp);
}

static SDOperand LowerFP_TO_UINT(SDOperand Op, SelectionDAG &DAG) {
  assert(Op.getValueType() == MVT::i32);
  SDOperand FloatVal = Op.getOperand(0);
  MVT::ValueType  vt = FloatVal.getValueType();
  assert(vt == MVT::f32 || vt == MVT::f64);

  ARMISD::NodeType op = vt == MVT::f32 ? ARMISD::FTOUIS : ARMISD::FTOUID;
  SDOperand Tmp = DAG.getNode(op, MVT::f32, FloatVal);
  return DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Tmp);
}

SDOperand ARMTargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default:
    assert(0 && "Should not custom lower this!");
    abort();
  case ISD::ConstantPool:
    return LowerConstantPool(Op, DAG);
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::FP_TO_SINT:
    return LowerFP_TO_SINT(Op, DAG);
  case ISD::SINT_TO_FP:
    return LowerSINT_TO_FP(Op, DAG);
  case ISD::FP_TO_UINT:
    return LowerFP_TO_UINT(Op, DAG);
  case ISD::UINT_TO_FP:
    return LowerUINT_TO_FP(Op, DAG);
  case ISD::FORMAL_ARGUMENTS:
    return LowerFORMAL_ARGUMENTS(Op, DAG, VarArgsFrameIndex);
  case ISD::CALL:
    return LowerCALL(Op, DAG);
  case ISD::RET:
    return LowerRET(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG, VarArgsFrameIndex);
  }
}

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
/// ARMDAGToDAGISel - ARM specific code to select ARM machine
/// instructions for SelectionDAG operations.
///
namespace {
class ARMDAGToDAGISel : public SelectionDAGISel {
  ARMTargetLowering Lowering;

public:
  ARMDAGToDAGISel(TargetMachine &TM)
    : SelectionDAGISel(Lowering), Lowering(TM) {
  }

  SDNode *Select(SDOperand Op);
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);
  bool SelectAddrMode1(SDOperand Op, SDOperand N, SDOperand &Arg,
                       SDOperand &Shift, SDOperand &ShiftType);
  bool SelectAddrMode1a(SDOperand Op, SDOperand N, SDOperand &Arg,
			SDOperand &Shift, SDOperand &ShiftType);
  bool SelectAddrMode2(SDOperand Op, SDOperand N, SDOperand &Arg,
                       SDOperand &Offset);
  bool SelectAddrMode5(SDOperand Op, SDOperand N, SDOperand &Arg,
                       SDOperand &Offset);

  // Include the pieces autogenerated from the target description.
#include "ARMGenDAGISel.inc"
};

void ARMDAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());

  DAG.setRoot(SelectRoot(DAG.getRoot()));
  DAG.RemoveDeadNodes();

  ScheduleAndEmitDAG(DAG);
}

static bool isInt12Immediate(SDNode *N, short &Imm) {
  if (N->getOpcode() != ISD::Constant)
    return false;

  int32_t t = cast<ConstantSDNode>(N)->getValue();
  int max = 1<<12;
  int min = -max;
  if (t > min && t < max) {
    Imm = t;
    return true;
  }
  else
    return false;
}

static bool isInt12Immediate(SDOperand Op, short &Imm) {
  return isInt12Immediate(Op.Val, Imm);
}

static uint32_t rotateL(uint32_t x) {
  uint32_t bit31 = (x & (1 << 31)) >> 31;
  uint32_t     t = x << 1;
  return t | bit31;
}

static bool isUInt8Immediate(uint32_t x) {
  return x < (1 << 8);
}

static bool isRotInt8Immediate(uint32_t x) {
  int r;
  for (r = 0; r < 16; r++) {
    if (isUInt8Immediate(x))
      return true;
    x = rotateL(rotateL(x));
  }
  return false;
}

bool ARMDAGToDAGISel::SelectAddrMode1(SDOperand Op,
                                      SDOperand N,
				      SDOperand &Arg,
				      SDOperand &Shift,
				      SDOperand &ShiftType) {
  switch(N.getOpcode()) {
  case ISD::Constant: {
    uint32_t val = cast<ConstantSDNode>(N)->getValue();
    if(!isRotInt8Immediate(val)) {
      Constant    *C = ConstantInt::get(Type::UIntTy, val);
      int  alignment = 2;
      SDOperand Addr = CurDAG->getTargetConstantPool(C, MVT::i32, alignment);
      SDOperand    Z = CurDAG->getTargetConstant(0,     MVT::i32);
      SDNode      *n = CurDAG->getTargetNode(ARM::LDR,  MVT::i32, Addr, Z);
      Arg            = SDOperand(n, 0);
    } else
      Arg            = CurDAG->getTargetConstant(val,    MVT::i32);

    Shift     = CurDAG->getTargetConstant(0,             MVT::i32);
    ShiftType = CurDAG->getTargetConstant(ARMShift::LSL, MVT::i32);
    return true;
  }
  case ISD::SRA:
    Arg       = N.getOperand(0);
    Shift     = N.getOperand(1);
    ShiftType = CurDAG->getTargetConstant(ARMShift::ASR, MVT::i32);
    return true;
  case ISD::SRL:
    Arg       = N.getOperand(0);
    Shift     = N.getOperand(1);
    ShiftType = CurDAG->getTargetConstant(ARMShift::LSR, MVT::i32);
    return true;
  case ISD::SHL:
    Arg       = N.getOperand(0);
    Shift     = N.getOperand(1);
    ShiftType = CurDAG->getTargetConstant(ARMShift::LSL, MVT::i32);
    return true;
  }

  Arg       = N;
  Shift     = CurDAG->getTargetConstant(0, MVT::i32);
  ShiftType = CurDAG->getTargetConstant(ARMShift::LSL, MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode1a(SDOperand Op,
				       SDOperand N,
				       SDOperand &Arg,
				       SDOperand &Shift,
				       SDOperand &ShiftType) {
  if (N.getOpcode() != ISD::Constant)
    return false;

  uint32_t val = ~cast<ConstantSDNode>(N)->getValue();
  if(!isRotInt8Immediate(val))
    return false;

  Arg       = CurDAG->getTargetConstant(val,    MVT::i32);
  Shift     = CurDAG->getTargetConstant(0,             MVT::i32);
  ShiftType = CurDAG->getTargetConstant(ARMShift::LSL, MVT::i32);

  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode2(SDOperand Op, SDOperand N,
                                      SDOperand &Arg, SDOperand &Offset) {
  //TODO: complete and cleanup!
  SDOperand Zero = CurDAG->getTargetConstant(0, MVT::i32);
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(N)) {
    Arg    = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i32);
    Offset = Zero;
    return true;
  }
  if (N.getOpcode() == ISD::ADD) {
    short imm = 0;
    if (isInt12Immediate(N.getOperand(1), imm)) {
      Offset = CurDAG->getTargetConstant(imm, MVT::i32);
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N.getOperand(0))) {
	Arg = CurDAG->getTargetFrameIndex(FI->getIndex(), N.getValueType());
      } else {
	Arg = N.getOperand(0);
      }
      return true; // [r+i]
    }
  }
  Offset = Zero;
  if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N))
    Arg = CurDAG->getTargetFrameIndex(FI->getIndex(), N.getValueType());
  else
    Arg = N;
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode5(SDOperand Op,
                                      SDOperand N, SDOperand &Arg,
                                      SDOperand &Offset) {
  //TODO: detect offset
  Offset = CurDAG->getTargetConstant(0, MVT::i32);
  Arg    = N;
  return true;
}

SDNode *ARMDAGToDAGISel::Select(SDOperand Op) {
  SDNode *N = Op.Val;

  switch (N->getOpcode()) {
  default:
    return SelectCode(Op);
    break;
  case ISD::FrameIndex: {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    SDOperand Ops[] = {CurDAG->getTargetFrameIndex(FI, MVT::i32),
                       CurDAG->getTargetConstant(0, MVT::i32),
                       CurDAG->getTargetConstant(0, MVT::i32),
                       CurDAG->getTargetConstant(ARMShift::LSL, MVT::i32)};

    return CurDAG->SelectNodeTo(N, ARM::ADD, MVT::i32, Ops,
                                sizeof(Ops)/sizeof(SDOperand));
    break;
  }
  }
}

}  // end anonymous namespace

/// createARMISelDag - This pass converts a legalized DAG into a
/// ARM-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createARMISelDag(TargetMachine &TM) {
  return new ARMDAGToDAGISel(TM);
}
