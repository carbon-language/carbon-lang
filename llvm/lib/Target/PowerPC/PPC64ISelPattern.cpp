//===-- PPC64ISelPattern.cpp - A pattern matching inst selector for PPC64 -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for 64 bit PowerPC.
//
//===----------------------------------------------------------------------===//

#include "PowerPC.h"
#include "PowerPCInstrBuilder.h"
#include "PowerPCInstrInfo.h"
#include "PPC64RegisterInfo.h"
#include "llvm/Constants.h"                   // FIXME: REMOVE
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineConstantPool.h" // FIXME: REMOVE
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/Statistic.h"
#include <set>
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  PPC32TargetLowering - PPC32 Implementation of the TargetLowering interface
namespace {
  class PPC64TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    int ReturnAddrIndex;              // FrameIndex for return slot.
  public:
    PPC64TargetLowering(TargetMachine &TM) : TargetLowering(TM) {
      // Fold away setcc operations if possible.
      setSetCCIsExpensive();

      // Set up the register classes.
      addRegisterClass(MVT::i64, PPC64::GPRCRegisterClass);
      addRegisterClass(MVT::f32, PPC64::FPRCRegisterClass);
      addRegisterClass(MVT::f64, PPC64::FPRCRegisterClass);

      // PowerPC has no intrinsics for these particular operations
      setOperationAction(ISD::BRCONDTWOWAY, MVT::Other, Expand);
      setOperationAction(ISD::MEMMOVE, MVT::Other, Expand);
      setOperationAction(ISD::MEMSET, MVT::Other, Expand);
      setOperationAction(ISD::MEMCPY, MVT::Other, Expand);

      // We don't support sin/cos/sqrt/fmod
      setOperationAction(ISD::FSIN , MVT::f64, Expand);
      setOperationAction(ISD::FCOS , MVT::f64, Expand);
      setOperationAction(ISD::FSQRT, MVT::f64, Expand);
      setOperationAction(ISD::SREM , MVT::f64, Expand);
      setOperationAction(ISD::FSIN , MVT::f32, Expand);
      setOperationAction(ISD::FCOS , MVT::f32, Expand);
      setOperationAction(ISD::FSQRT, MVT::f32, Expand);
      setOperationAction(ISD::SREM , MVT::f32, Expand);

      // PPC 64 has i16 and i32 but no i8 (or i1) SEXTLOAD
      setOperationAction(ISD::SEXTLOAD, MVT::i1, Expand);
      setOperationAction(ISD::SEXTLOAD, MVT::i8, Expand);

      // PowerPC has no SREM/UREM instructions
      setOperationAction(ISD::SREM, MVT::i64, Expand);
      setOperationAction(ISD::UREM, MVT::i64, Expand);

      // PowerPC has these, but they are not implemented
      setOperationAction(ISD::CTPOP, MVT::i64, Expand);
      setOperationAction(ISD::CTTZ , MVT::i64, Expand);
      setOperationAction(ISD::CTLZ , MVT::i64, Expand);

      setShiftAmountFlavor(Extend);   // shl X, 32 == 0
      addLegalFPImmediate(+0.0); // Necessary for FSEL
      addLegalFPImmediate(-0.0); //

      computeRegisterProperties();
    }

    /// LowerArguments - This hook must be implemented to indicate how we should
    /// lower the arguments for the specified function, into the specified DAG.
    virtual std::vector<SDOperand>
    LowerArguments(Function &F, SelectionDAG &DAG);

    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
    LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg, unsigned CC,
                bool isTailCall, SDOperand Callee, ArgListTy &Args,
                SelectionDAG &DAG);

    virtual std::pair<SDOperand, SDOperand>
    LowerVAStart(SDOperand Chain, SelectionDAG &DAG);

    virtual std::pair<SDOperand,SDOperand>
    LowerVAArgNext(bool isVANext, SDOperand Chain, SDOperand VAList,
                   const Type *ArgTy, SelectionDAG &DAG);

    virtual std::pair<SDOperand, SDOperand>
    LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                            SelectionDAG &DAG);
  };
}


std::vector<SDOperand>
PPC64TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  //
  // add beautiful description of PPC stack frame format, or at least some docs
  //
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock& BB = MF.front();
  std::vector<SDOperand> ArgValues;

  // Due to the rather complicated nature of the PowerPC ABI, rather than a
  // fixed size array of physical args, for the sake of simplicity let the STL
  // handle tracking them for us.
  std::vector<unsigned> argVR, argPR, argOp;
  unsigned ArgOffset = 48;
  unsigned GPR_remaining = 8;
  unsigned FPR_remaining = 13;
  unsigned GPR_idx = 0, FPR_idx = 0;
  static const unsigned GPR[] = {
    PPC::R3, PPC::R4, PPC::R5, PPC::R6,
    PPC::R7, PPC::R8, PPC::R9, PPC::R10,
  };
  static const unsigned FPR[] = {
    PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
    PPC::F8, PPC::F9, PPC::F10, PPC::F11, PPC::F12, PPC::F13
  };

  // Add DAG nodes to load the arguments...  On entry to a function on PPC,
  // the arguments start at offset 48, although they are likely to be passed
  // in registers.
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I) {
    SDOperand newroot, argt;
    bool needsLoad = false;
    MVT::ValueType ObjectVT = getValueType(I->getType());

    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64:
      if (GPR_remaining > 0) {
        BuildMI(&BB, PPC::IMPLICIT_DEF, 0, GPR[GPR_idx]);
        argt = newroot = DAG.getCopyFromReg(GPR[GPR_idx], MVT::i32,
                                            DAG.getRoot());
        if (ObjectVT != MVT::i64)
          argt = DAG.getNode(ISD::TRUNCATE, ObjectVT, newroot);
      } else {
        needsLoad = true;
      }
      break;
    case MVT::f32:
    case MVT::f64:
      if (FPR_remaining > 0) {
        BuildMI(&BB, PPC::IMPLICIT_DEF, 0, FPR[FPR_idx]);
        argt = newroot = DAG.getCopyFromReg(FPR[FPR_idx], ObjectVT,
                                            DAG.getRoot());
        --FPR_remaining;
        ++FPR_idx;
      } else {
        needsLoad = true;
      }
      break;
    }

    // We need to load the argument to a virtual register if we determined above
    // that we ran out of physical registers of the appropriate type
    if (needsLoad) {
      unsigned SubregOffset = 0;
      switch (ObjectVT) {
      default: assert(0 && "Unhandled argument type!");
      case MVT::i1:
      case MVT::i8:   SubregOffset = 7; break;
      case MVT::i16:  SubregOffset = 6; break;
      case MVT::i32:
      case MVT::f32:  SubregOffset = 4; break;
      case MVT::i64:
      case MVT::f64:  SubregOffset = 0; break;
      }
      int FI = MFI->CreateFixedObject(8, ArgOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i64);
      FIN = DAG.getNode(ISD::ADD, MVT::i64, FIN,
                        DAG.getConstant(SubregOffset, MVT::i64));
      argt = newroot = DAG.getLoad(ObjectVT, DAG.getEntryNode(), FIN,
                                   DAG.getSrcValue(NULL));
    }

    // Every 4 bytes of argument space consumes one of the GPRs available for
    // argument passing.
    if (GPR_remaining > 0) {
      --GPR_remaining;
      ++GPR_idx;
    }
    ArgOffset += 8;

    DAG.setRoot(newroot.getValue(1));
    ArgValues.push_back(argt);
  }

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (F.isVarArg()) {
    VarArgsFrameIndex = MFI->CreateFixedObject(8, ArgOffset);
    SDOperand FIN = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i64);
    // If this function is vararg, store any remaining integer argument regs
    // to their spots on the stack so that they may be loaded by deferencing the
    // result of va_next.
    std::vector<SDOperand> MemOps;
    for (; GPR_remaining > 0; --GPR_remaining, ++GPR_idx) {
      BuildMI(&BB, PPC::IMPLICIT_DEF, 0, GPR[GPR_idx]);
      SDOperand Val = DAG.getCopyFromReg(GPR[GPR_idx], MVT::i64, DAG.getRoot());
      SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Val.getValue(1),
                                    Val, FIN, DAG.getSrcValue(NULL));
      MemOps.push_back(Store);
      // Increment the address by eight for the next argument to store
      SDOperand PtrOff = DAG.getConstant(8, getPointerTy());
      FIN = DAG.getNode(ISD::ADD, MVT::i32, FIN, PtrOff);
    }
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other, MemOps));
  }

  return ArgValues;
}

std::pair<SDOperand, SDOperand>
PPC64TargetLowering::LowerCallTo(SDOperand Chain,
                                 const Type *RetTy, bool isVarArg,
                                 unsigned CallingConv, bool isTailCall,
                                 SDOperand Callee, ArgListTy &Args,
                                 SelectionDAG &DAG) {
  // args_to_use will accumulate outgoing args for the ISD::CALL case in
  // SelectExpr to use to put the arguments in the appropriate registers.
  std::vector<SDOperand> args_to_use;

  // Count how many bytes are to be pushed on the stack, including the linkage
  // area, and parameter passing area.
  unsigned NumBytes = 48;

  if (Args.empty()) {
    Chain = DAG.getNode(ISD::CALLSEQ_START, MVT::Other, Chain,
                        DAG.getConstant(NumBytes, getPointerTy()));
  } else {
    NumBytes = 8 * Args.size(); // All arguments are rounded up to 8 bytes

    // Just to be safe, we'll always reserve the full 48 bytes of linkage area
    // plus 64 bytes of argument space in case any called code gets funky on us.
    if (NumBytes < 112) NumBytes = 112;

    // Adjust the stack pointer for the new arguments...
    // These operations are automatically eliminated by the prolog/epilog pass
    Chain = DAG.getNode(ISD::CALLSEQ_START, MVT::Other, Chain,
                        DAG.getConstant(NumBytes, getPointerTy()));

    // Set up a copy of the stack pointer for use loading and storing any
    // arguments that may not fit in the registers available for argument
    // passing.
    SDOperand StackPtr = DAG.getCopyFromReg(PPC::R1, MVT::i32,
                                            DAG.getEntryNode());

    // Figure out which arguments are going to go in registers, and which in
    // memory.  Also, if this is a vararg function, floating point operations
    // must be stored to our stack, and loaded into integer regs as well, if
    // any integer regs are available for argument passing.
    unsigned ArgOffset = 48;
    unsigned GPR_remaining = 8;
    unsigned FPR_remaining = 13;

    std::vector<SDOperand> MemOps;
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
      // PtrOff will be used to store the current argument to the stack if a
      // register cannot be found for it.
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
      MVT::ValueType ArgVT = getValueType(Args[i].second);

      switch (ArgVT) {
      default: assert(0 && "Unexpected ValueType for argument!");
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
        // Promote the integer to 64 bits.  If the input type is signed use a
        // sign extend, otherwise use a zero extend.
        if (Args[i].second->isSigned())
          Args[i].first =DAG.getNode(ISD::SIGN_EXTEND, MVT::i64, Args[i].first);
        else
          Args[i].first =DAG.getNode(ISD::ZERO_EXTEND, MVT::i64, Args[i].first);
        // FALL THROUGH
      case MVT::i64:
        if (GPR_remaining > 0) {
          args_to_use.push_back(Args[i].first);
          --GPR_remaining;
        } else {
          MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                          Args[i].first, PtrOff,
                                       DAG.getSrcValue(NULL)));
        }
        ArgOffset += 8;
        break;
      case MVT::f32:
      case MVT::f64:
        if (FPR_remaining > 0) {
          args_to_use.push_back(Args[i].first);
          --FPR_remaining;
          if (isVarArg) {
            SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                          Args[i].first, PtrOff,
                                          DAG.getSrcValue(NULL));
            MemOps.push_back(Store);
            // Float varargs are always shadowed in available integer registers
            if (GPR_remaining > 0) {
              SDOperand Load = DAG.getLoad(MVT::i64, Store, PtrOff,
                                           DAG.getSrcValue(NULL));
              MemOps.push_back(Load);
              args_to_use.push_back(Load);
              --GPR_remaining;
            }
          } else {
            // If we have any FPRs remaining, we may also have GPRs remaining.
            // Args passed in FPRs also consume an available GPR.
            if (GPR_remaining > 0) {
              args_to_use.push_back(DAG.getNode(ISD::UNDEF, MVT::i64));
              --GPR_remaining;
            }
          }
        } else {
          MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                          Args[i].first, PtrOff,
                                       DAG.getSrcValue(NULL)));
        }
        ArgOffset += 8;
        break;
      }
    }
    if (!MemOps.empty())
      Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, MemOps);
  }

  std::vector<MVT::ValueType> RetVals;
  MVT::ValueType RetTyVT = getValueType(RetTy);
  if (RetTyVT != MVT::isVoid)
    RetVals.push_back(RetTyVT);
  RetVals.push_back(MVT::Other);

  SDOperand TheCall = SDOperand(DAG.getCall(RetVals,
                                            Chain, Callee, args_to_use), 0);
  Chain = TheCall.getValue(RetTyVT != MVT::isVoid);
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  return std::make_pair(TheCall, Chain);
}

std::pair<SDOperand, SDOperand>
PPC64TargetLowering::LowerVAStart(SDOperand Chain, SelectionDAG &DAG) {
  //vastart just returns the address of the VarArgsFrameIndex slot.
  return std::make_pair(DAG.getFrameIndex(VarArgsFrameIndex, MVT::i64), Chain);
}

std::pair<SDOperand,SDOperand> PPC64TargetLowering::
LowerVAArgNext(bool isVANext, SDOperand Chain, SDOperand VAList,
               const Type *ArgTy, SelectionDAG &DAG) {
  MVT::ValueType ArgVT = getValueType(ArgTy);
  SDOperand Result;
  if (!isVANext) {
    Result = DAG.getLoad(ArgVT, DAG.getEntryNode(), VAList,
                         DAG.getSrcValue(NULL));
  } else {
    Result = DAG.getNode(ISD::ADD, VAList.getValueType(), VAList,
                         DAG.getConstant(8, VAList.getValueType()));
  }
  return std::make_pair(Result, Chain);
}


std::pair<SDOperand, SDOperand> PPC64TargetLowering::
LowerFrameReturnAddress(bool isFrameAddress, SDOperand Chain, unsigned Depth,
                        SelectionDAG &DAG) {
  assert(0 && "LowerFrameReturnAddress unimplemented");
  abort();
}

namespace {
Statistic<>NotLogic("ppc-codegen", "Number of inverted logical ops");
Statistic<>FusedFP("ppc-codegen", "Number of fused fp operations");
//===--------------------------------------------------------------------===//
/// ISel - PPC32 specific code to select PPC32 machine instructions for
/// SelectionDAG operations.
//===--------------------------------------------------------------------===//
class ISel : public SelectionDAGISel {

  /// Comment Here.
  PPC64TargetLowering PPC64Lowering;

  /// ExprMap - As shared expressions are codegen'd, we keep track of which
  /// vreg the value is produced in, so we only emit one copy of each compiled
  /// tree.
  std::map<SDOperand, unsigned> ExprMap;

  unsigned GlobalBaseReg;
  bool GlobalBaseInitialized;

public:
  ISel(TargetMachine &TM) : SelectionDAGISel(PPC64Lowering), PPC64Lowering(TM)
  {}

  /// runOnFunction - Override this function in order to reset our per-function
  /// variables.
  virtual bool runOnFunction(Function &Fn) {
    // Make sure we re-emit a set of the global base reg if necessary
    GlobalBaseInitialized = false;
    return SelectionDAGISel::runOnFunction(Fn);
  }

  /// InstructionSelectBasicBlock - This callback is invoked by
  /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG) {
    DEBUG(BB->dump());
    // Codegen the basic block.
    Select(DAG.getRoot());

    // Clear state used for selection.
    ExprMap.clear();
  }

  unsigned getGlobalBaseReg();
  unsigned getConstDouble(double floatVal, unsigned Result);
  unsigned SelectSetCR0(SDOperand CC);
  unsigned SelectExpr(SDOperand N);
  unsigned SelectExprFP(SDOperand N, unsigned Result);
  void Select(SDOperand N);

  bool SelectAddr(SDOperand N, unsigned& Reg, int& offset);
  void SelectBranchCC(SDOperand N);
};

/// ExactLog2 - This function solves for (Val == 1 << (N-1)) and returns N.  It
/// returns zero when the input is not exactly a power of two.
static unsigned ExactLog2(unsigned Val) {
  if (Val == 0 || (Val & (Val-1))) return 0;
  unsigned Count = 0;
  while (Val != 1) {
    Val >>= 1;
    ++Count;
  }
  return Count;
}

/// getImmediateForOpcode - This method returns a value indicating whether
/// the ConstantSDNode N can be used as an immediate to Opcode.  The return
/// values are either 0, 1 or 2.  0 indicates that either N is not a
/// ConstantSDNode, or is not suitable for use by that opcode.  A return value
/// of 1 indicates that the constant may be used in normal immediate form.  A
/// return value of 2 indicates that the constant may be used in shifted
/// immediate form.  A return value of 3 indicates that log base 2 of the
/// constant may be used.
///
static unsigned getImmediateForOpcode(SDOperand N, unsigned Opcode,
                                      unsigned& Imm, bool U = false) {
  if (N.getOpcode() != ISD::Constant) return 0;

  int v = (int)cast<ConstantSDNode>(N)->getSignExtended();

  switch(Opcode) {
  default: return 0;
  case ISD::ADD:
    if (v <= 32767 && v >= -32768) { Imm = v & 0xFFFF; return 1; }
    if ((v & 0x0000FFFF) == 0) { Imm = v >> 16; return 2; }
    break;
  case ISD::AND:
  case ISD::XOR:
  case ISD::OR:
    if (v >= 0 && v <= 65535) { Imm = v & 0xFFFF; return 1; }
    if ((v & 0x0000FFFF) == 0) { Imm = v >> 16; return 2; }
    break;
  case ISD::MUL:
  case ISD::SUB:
    if (v <= 32767 && v >= -32768) { Imm = v & 0xFFFF; return 1; }
    break;
  case ISD::SETCC:
    if (U && (v >= 0 && v <= 65535)) { Imm = v & 0xFFFF; return 1; }
    if (!U && (v <= 32767 && v >= -32768)) { Imm = v & 0xFFFF; return 1; }
    break;
  case ISD::SDIV:
    if ((Imm = ExactLog2(v))) { return 3; }
    break;
  }
  return 0;
}

/// getBCCForSetCC - Returns the PowerPC condition branch mnemonic corresponding
/// to Condition.  If the Condition is unordered or unsigned, the bool argument
/// U is set to true, otherwise it is set to false.
static unsigned getBCCForSetCC(unsigned Condition, bool& U) {
  U = false;
  switch (Condition) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETEQ:  return PPC::BEQ;
  case ISD::SETNE:  return PPC::BNE;
  case ISD::SETULT: U = true;
  case ISD::SETLT:  return PPC::BLT;
  case ISD::SETULE: U = true;
  case ISD::SETLE:  return PPC::BLE;
  case ISD::SETUGT: U = true;
  case ISD::SETGT:  return PPC::BGT;
  case ISD::SETUGE: U = true;
  case ISD::SETGE:  return PPC::BGE;
  }
  return 0;
}

/// IndexedOpForOp - Return the indexed variant for each of the PowerPC load
/// and store immediate instructions.
static unsigned IndexedOpForOp(unsigned Opcode) {
  switch(Opcode) {
  default: assert(0 && "Unknown opcode!"); abort();
  case PPC::LBZ: return PPC::LBZX;  case PPC::STB: return PPC::STBX;
  case PPC::LHZ: return PPC::LHZX;  case PPC::STH: return PPC::STHX;
  case PPC::LHA: return PPC::LHAX;  case PPC::STW: return PPC::STWX;
  case PPC::LWZ: return PPC::LWZX;  case PPC::STD: return PPC::STDX;
  case PPC::LD:  return PPC::LDX;   case PPC::STFS: return PPC::STFSX;
  case PPC::LFS: return PPC::LFSX;  case PPC::STFD: return PPC::STFDX;
  case PPC::LFD: return PPC::LFDX;
  }
  return 0;
}
}

/// getGlobalBaseReg - Output the instructions required to put the
/// base address to use for accessing globals into a register.
///
unsigned ISel::getGlobalBaseReg() {
  if (!GlobalBaseInitialized) {
    // Insert the set of GlobalBaseReg into the first MBB of the function
    MachineBasicBlock &FirstMBB = BB->getParent()->front();
    MachineBasicBlock::iterator MBBI = FirstMBB.begin();
    GlobalBaseReg = MakeReg(MVT::i64);
    BuildMI(FirstMBB, MBBI, PPC::MovePCtoLR, 0, PPC::LR);
    BuildMI(FirstMBB, MBBI, PPC::MFLR, 1, GlobalBaseReg).addReg(PPC::LR);
    GlobalBaseInitialized = true;
  }
  return GlobalBaseReg;
}

/// getConstDouble - Loads a floating point value into a register, via the
/// Constant Pool.  Optionally takes a register in which to load the value.
unsigned ISel::getConstDouble(double doubleVal, unsigned Result=0) {
  unsigned Tmp1 = MakeReg(MVT::i64);
  if (0 == Result) Result = MakeReg(MVT::f64);
  MachineConstantPool *CP = BB->getParent()->getConstantPool();
  ConstantFP *CFP = ConstantFP::get(Type::DoubleTy, doubleVal);
  unsigned CPI = CP->getConstantPoolIndex(CFP);
  BuildMI(BB, PPC::LOADHiAddr, 2, Tmp1).addReg(getGlobalBaseReg())
    .addConstantPoolIndex(CPI);
  BuildMI(BB, PPC::LFD, 2, Result).addConstantPoolIndex(CPI).addReg(Tmp1);
  return Result;
}

unsigned ISel::SelectSetCR0(SDOperand CC) {
  unsigned Opc, Tmp1, Tmp2;
  static const unsigned CompareOpcodes[] =
    { PPC::FCMPU, PPC::FCMPU, PPC::CMPW, PPC::CMPLW };

  // If the first operand to the select is a SETCC node, then we can fold it
  // into the branch that selects which value to return.
  SetCCSDNode* SetCC = dyn_cast<SetCCSDNode>(CC.Val);
  if (SetCC && CC.getOpcode() == ISD::SETCC) {
    bool U;
    Opc = getBCCForSetCC(SetCC->getCondition(), U);
    Tmp1 = SelectExpr(SetCC->getOperand(0));

    // Pass the optional argument U to getImmediateForOpcode for SETCC,
    // so that it knows whether the SETCC immediate range is signed or not.
    if (1 == getImmediateForOpcode(SetCC->getOperand(1), ISD::SETCC,
                                   Tmp2, U)) {
      if (U)
        BuildMI(BB, PPC::CMPLWI, 2, PPC::CR0).addReg(Tmp1).addImm(Tmp2);
      else
        BuildMI(BB, PPC::CMPWI, 2, PPC::CR0).addReg(Tmp1).addSImm(Tmp2);
    } else {
      bool IsInteger = MVT::isInteger(SetCC->getOperand(0).getValueType());
      unsigned CompareOpc = CompareOpcodes[2 * IsInteger + U];
      Tmp2 = SelectExpr(SetCC->getOperand(1));
      BuildMI(BB, CompareOpc, 2, PPC::CR0).addReg(Tmp1).addReg(Tmp2);
    }
  } else {
    Tmp1 = SelectExpr(CC);
    BuildMI(BB, PPC::CMPLWI, 2, PPC::CR0).addReg(Tmp1).addImm(0);
    Opc = PPC::BNE;
  }
  return Opc;
}

/// Check to see if the load is a constant offset from a base register
bool ISel::SelectAddr(SDOperand N, unsigned& Reg, int& offset)
{
  unsigned imm = 0, opcode = N.getOpcode();
  if (N.getOpcode() == ISD::ADD) {
    Reg = SelectExpr(N.getOperand(0));
    if (1 == getImmediateForOpcode(N.getOperand(1), opcode, imm)) {
      offset = imm;
      return false;
    }
    offset = SelectExpr(N.getOperand(1));
    return true;
  }
  Reg = SelectExpr(N);
  offset = 0;
  return false;
}

void ISel::SelectBranchCC(SDOperand N)
{
  assert(N.getOpcode() == ISD::BRCOND && "Not a BranchCC???");
  MachineBasicBlock *Dest =
    cast<BasicBlockSDNode>(N.getOperand(2))->getBasicBlock();

  // Get the MBB we will fall through to so that we can hand it off to the
  // branch selection pass as an argument to the PPC::COND_BRANCH pseudo op.
  //ilist<MachineBasicBlock>::iterator It = BB;
  //MachineBasicBlock *Fallthrough = ++It;

  Select(N.getOperand(0));  //chain
  unsigned Opc = SelectSetCR0(N.getOperand(1));
  // FIXME: Use this once we have something approximating two-way branches
  // We cannot currently use this in case the ISel hands us something like
  // BRcc MBBx
  // BR MBBy
  // since the fallthrough basic block for the conditional branch does not start
  // with the unconditional branch (it is skipped over).
  //BuildMI(BB, PPC::COND_BRANCH, 4).addReg(PPC::CR0).addImm(Opc)
  //  .addMBB(Dest).addMBB(Fallthrough);
  BuildMI(BB, Opc, 2).addReg(PPC::CR0).addMBB(Dest);
  return;
}

unsigned ISel::SelectExprFP(SDOperand N, unsigned Result)
{
  unsigned Tmp1, Tmp2, Tmp3;
  unsigned Opc = 0;
  SDNode *Node = N.Val;
  MVT::ValueType DestType = N.getValueType();
  unsigned opcode = N.getOpcode();

  switch (opcode) {
  default:
    Node->dump();
    assert(0 && "Node not handled!\n");

  case ISD::SELECT: {
    // Attempt to generate FSEL.  We can do this whenever we have an FP result,
    // and an FP comparison in the SetCC node.
    SetCCSDNode* SetCC = dyn_cast<SetCCSDNode>(N.getOperand(0).Val);
    if (SetCC && N.getOperand(0).getOpcode() == ISD::SETCC &&
        !MVT::isInteger(SetCC->getOperand(0).getValueType()) &&
        SetCC->getCondition() != ISD::SETEQ &&
        SetCC->getCondition() != ISD::SETNE) {
      MVT::ValueType VT = SetCC->getOperand(0).getValueType();
      Tmp1 = SelectExpr(SetCC->getOperand(0));   // Val to compare against
      unsigned TV = SelectExpr(N.getOperand(1)); // Use if TRUE
      unsigned FV = SelectExpr(N.getOperand(2)); // Use if FALSE

      ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(SetCC->getOperand(1));
      if (CN && (CN->isExactlyValue(-0.0) || CN->isExactlyValue(0.0))) {
        switch(SetCC->getCondition()) {
        default: assert(0 && "Invalid FSEL condition"); abort();
        case ISD::SETULT:
        case ISD::SETLT:
          BuildMI(BB, PPC::FSEL, 3, Result).addReg(Tmp1).addReg(FV).addReg(TV);
          return Result;
        case ISD::SETUGE:
        case ISD::SETGE:
          BuildMI(BB, PPC::FSEL, 3, Result).addReg(Tmp1).addReg(TV).addReg(FV);
          return Result;
        case ISD::SETUGT:
        case ISD::SETGT: {
          Tmp2 = MakeReg(VT);
          BuildMI(BB, PPC::FNEG, 1, Tmp2).addReg(Tmp1);
          BuildMI(BB, PPC::FSEL, 3, Result).addReg(Tmp2).addReg(FV).addReg(TV);
          return Result;
        }
        case ISD::SETULE:
        case ISD::SETLE: {
          Tmp2 = MakeReg(VT);
          BuildMI(BB, PPC::FNEG, 1, Tmp2).addReg(Tmp1);
          BuildMI(BB, PPC::FSEL, 3, Result).addReg(Tmp2).addReg(TV).addReg(FV);
          return Result;
        }
        }
      } else {
        Opc = (MVT::f64 == VT) ? PPC::FSUB : PPC::FSUBS;
        Tmp2 = SelectExpr(SetCC->getOperand(1));
        Tmp3 =  MakeReg(VT);
        switch(SetCC->getCondition()) {
        default: assert(0 && "Invalid FSEL condition"); abort();
        case ISD::SETULT:
        case ISD::SETLT:
          BuildMI(BB, Opc, 2, Tmp3).addReg(Tmp1).addReg(Tmp2);
          BuildMI(BB, PPC::FSEL, 3, Result).addReg(Tmp3).addReg(FV).addReg(TV);
          return Result;
        case ISD::SETUGE:
        case ISD::SETGE:
          BuildMI(BB, Opc, 2, Tmp3).addReg(Tmp1).addReg(Tmp2);
          BuildMI(BB, PPC::FSEL, 3, Result).addReg(Tmp3).addReg(TV).addReg(FV);
          return Result;
        case ISD::SETUGT:
        case ISD::SETGT:
          BuildMI(BB, Opc, 2, Tmp3).addReg(Tmp2).addReg(Tmp1);
          BuildMI(BB, PPC::FSEL, 3, Result).addReg(Tmp3).addReg(FV).addReg(TV);
          return Result;
        case ISD::SETULE:
        case ISD::SETLE:
          BuildMI(BB, Opc, 2, Tmp3).addReg(Tmp2).addReg(Tmp1);
          BuildMI(BB, PPC::FSEL, 3, Result).addReg(Tmp3).addReg(TV).addReg(FV);
          return Result;
        }
      }
      assert(0 && "Should never get here");
      return 0;
    }

    unsigned TrueValue = SelectExpr(N.getOperand(1)); //Use if TRUE
    unsigned FalseValue = SelectExpr(N.getOperand(2)); //Use if FALSE
    Opc = SelectSetCR0(N.getOperand(0));

    // Create an iterator with which to insert the MBB for copying the false
    // value and the MBB to hold the PHI instruction for this SetCC.
    MachineBasicBlock *thisMBB = BB;
    const BasicBlock *LLVM_BB = BB->getBasicBlock();
    ilist<MachineBasicBlock>::iterator It = BB;
    ++It;

    //  thisMBB:
    //  ...
    //   TrueVal = ...
    //   cmpTY cr0, r1, r2
    //   bCC copy1MBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
    BuildMI(BB, Opc, 2).addReg(PPC::CR0).addMBB(sinkMBB);
    MachineFunction *F = BB->getParent();
    F->getBasicBlockList().insert(It, copy0MBB);
    F->getBasicBlockList().insert(It, sinkMBB);
    // Update machine-CFG edges
    BB->addSuccessor(copy0MBB);
    BB->addSuccessor(sinkMBB);

    //  copy0MBB:
    //   %FalseValue = ...
    //   # fallthrough to sinkMBB
    BB = copy0MBB;
    // Update machine-CFG edges
    BB->addSuccessor(sinkMBB);

    //  sinkMBB:
    //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
    //  ...
    BB = sinkMBB;
    BuildMI(BB, PPC::PHI, 4, Result).addReg(FalseValue)
      .addMBB(copy0MBB).addReg(TrueValue).addMBB(thisMBB);
    return Result;
  }

  case ISD::FNEG:
    if (!NoExcessFPPrecision &&
        ISD::ADD == N.getOperand(0).getOpcode() &&
        N.getOperand(0).Val->hasOneUse() &&
        ISD::MUL == N.getOperand(0).getOperand(0).getOpcode() &&
        N.getOperand(0).getOperand(0).Val->hasOneUse()) {
      ++FusedFP; // Statistic
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(0).getOperand(1));
      Tmp3 = SelectExpr(N.getOperand(0).getOperand(1));
      Opc = DestType == MVT::f64 ? PPC::FNMADD : PPC::FNMADDS;
      BuildMI(BB, Opc, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
    } else if (!NoExcessFPPrecision &&
        ISD::SUB == N.getOperand(0).getOpcode() &&
        N.getOperand(0).Val->hasOneUse() &&
        ISD::MUL == N.getOperand(0).getOperand(0).getOpcode() &&
        N.getOperand(0).getOperand(0).Val->hasOneUse()) {
      ++FusedFP; // Statistic
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(0).getOperand(1));
      Tmp3 = SelectExpr(N.getOperand(0).getOperand(1));
      Opc = DestType == MVT::f64 ? PPC::FNMSUB : PPC::FNMSUBS;
      BuildMI(BB, Opc, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
    } else if (ISD::FABS == N.getOperand(0).getOpcode()) {
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
      BuildMI(BB, PPC::FNABS, 1, Result).addReg(Tmp1);
    } else {
      Tmp1 = SelectExpr(N.getOperand(0));
      BuildMI(BB, PPC::FNEG, 1, Result).addReg(Tmp1);
    }
    return Result;

  case ISD::FABS:
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, PPC::FABS, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::FP_ROUND:
    assert (DestType == MVT::f32 &&
            N.getOperand(0).getValueType() == MVT::f64 &&
            "only f64 to f32 conversion supported here");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, PPC::FRSP, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::FP_EXTEND:
    assert (DestType == MVT::f64 &&
            N.getOperand(0).getValueType() == MVT::f32 &&
            "only f32 to f64 conversion supported here");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, PPC::FMR, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::CopyFromReg:
    if (Result == 1)
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());
    Tmp1 = dyn_cast<RegSDNode>(Node)->getReg();
    BuildMI(BB, PPC::FMR, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::ConstantFP: {
    ConstantFPSDNode *CN = cast<ConstantFPSDNode>(N);
    Result = getConstDouble(CN->getValue(), Result);
    return Result;
  }

  case ISD::ADD:
    if (!NoExcessFPPrecision && N.getOperand(0).getOpcode() == ISD::MUL &&
        N.getOperand(0).Val->hasOneUse()) {
      ++FusedFP; // Statistic
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
      Tmp3 = SelectExpr(N.getOperand(1));
      Opc = DestType == MVT::f64 ? PPC::FMADD : PPC::FMADDS;
      BuildMI(BB, Opc, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
      return Result;
    }
    Opc = DestType == MVT::f64 ? PPC::FADD : PPC::FADDS;
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::SUB:
    if (!NoExcessFPPrecision && N.getOperand(0).getOpcode() == ISD::MUL &&
        N.getOperand(0).Val->hasOneUse()) {
      ++FusedFP; // Statistic
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
      Tmp3 = SelectExpr(N.getOperand(1));
      Opc = DestType == MVT::f64 ? PPC::FMSUB : PPC::FMSUBS;
      BuildMI(BB, Opc, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
      return Result;
    }
    Opc = DestType == MVT::f64 ? PPC::FSUB : PPC::FSUBS;
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::MUL:
  case ISD::SDIV:
    switch( opcode ) {
    case ISD::MUL:  Opc = DestType == MVT::f64 ? PPC::FMUL : PPC::FMULS; break;
    case ISD::SDIV: Opc = DestType == MVT::f64 ? PPC::FDIV : PPC::FDIVS; break;
    };
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::UINT_TO_FP:
  case ISD::SINT_TO_FP: {
    bool IsUnsigned = (ISD::UINT_TO_FP == opcode);
    Tmp1 = SelectExpr(N.getOperand(0));  // Get the operand register
    Tmp2 = MakeReg(MVT::f64); // temp reg to load the integer value into
    Tmp3 = MakeReg(MVT::i64); // temp reg to hold the conversion constant
    unsigned ConstF = MakeReg(MVT::f64); // temp reg to hold the fp constant

    int FrameIdx = BB->getParent()->getFrameInfo()->CreateStackObject(8, 8);
    MachineConstantPool *CP = BB->getParent()->getConstantPool();

    // FIXME: pull this FP constant generation stuff out into something like
    // the simple ISel's getReg.
    if (IsUnsigned) {
      addFrameReference(BuildMI(BB, PPC::STD, 3).addReg(Tmp1), FrameIdx);
      addFrameReference(BuildMI(BB, PPC::LFD, 2, Tmp2), FrameIdx);
      BuildMI(BB, PPC::FCFID, 1, Result).addReg(Tmp2);
    } else {
      ConstantFP *CFP = ConstantFP::get(Type::DoubleTy, 0x1.000008p52);
      unsigned CPI = CP->getConstantPoolIndex(CFP);
      // Load constant fp value
      unsigned Tmp4 = MakeReg(MVT::i32);
      unsigned TmpL = MakeReg(MVT::i32);
      BuildMI(BB, PPC::LOADHiAddr, 2, Tmp4).addReg(getGlobalBaseReg())
        .addConstantPoolIndex(CPI);
      BuildMI(BB, PPC::LFD, 2, ConstF).addConstantPoolIndex(CPI).addReg(Tmp4);
      // Store the hi & low halves of the fp value, currently in int regs
      BuildMI(BB, PPC::LIS, 1, Tmp3).addSImm(0x4330);
      addFrameReference(BuildMI(BB, PPC::STW, 3).addReg(Tmp3), FrameIdx);
      BuildMI(BB, PPC::XORIS, 2, TmpL).addReg(Tmp1).addImm(0x8000);
      addFrameReference(BuildMI(BB, PPC::STW, 3).addReg(TmpL), FrameIdx, 4);
      addFrameReference(BuildMI(BB, PPC::LFD, 2, Tmp2), FrameIdx);
      // Generate the return value with a subtract
      BuildMI(BB, PPC::FSUB, 2, Result).addReg(Tmp2).addReg(ConstF);
    }
    return Result;
  }
  }
  assert(0 && "Should never get here");
  return 0;
}

unsigned ISel::SelectExpr(SDOperand N) {
  unsigned Result;
  unsigned Tmp1, Tmp2, Tmp3;
  unsigned Opc = 0;
  unsigned opcode = N.getOpcode();

  SDNode *Node = N.Val;
  MVT::ValueType DestType = N.getValueType();

  unsigned &Reg = ExprMap[N];
  if (Reg) return Reg;

  switch (N.getOpcode()) {
  default:
    Reg = Result = (N.getValueType() != MVT::Other) ?
                            MakeReg(N.getValueType()) : 1;
    break;
  case ISD::TAILCALL:
  case ISD::CALL:
    // If this is a call instruction, make sure to prepare ALL of the result
    // values as well as the chain.
    if (Node->getNumValues() == 1)
      Reg = Result = 1;  // Void call, just a chain.
    else {
      Result = MakeReg(Node->getValueType(0));
      ExprMap[N.getValue(0)] = Result;
      for (unsigned i = 1, e = N.Val->getNumValues()-1; i != e; ++i)
        ExprMap[N.getValue(i)] = MakeReg(Node->getValueType(i));
      ExprMap[SDOperand(Node, Node->getNumValues()-1)] = 1;
    }
    break;
  }

  if (ISD::CopyFromReg == opcode)
    DestType = N.getValue(0).getValueType();

  if (DestType == MVT::f64 || DestType == MVT::f32)
    if (ISD::LOAD != opcode && ISD::EXTLOAD != opcode && ISD::UNDEF != opcode)
      return SelectExprFP(N, Result);

  switch (opcode) {
  default:
    Node->dump();
    assert(0 && "Node not handled!\n");
  case ISD::UNDEF:
    BuildMI(BB, PPC::IMPLICIT_DEF, 0, Result);
    return Result;
  case ISD::DYNAMIC_STACKALLOC:
    // Generate both result values.  FIXME: Need a better commment here?
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    // FIXME: We are currently ignoring the requested alignment for handling
    // greater than the stack alignment.  This will need to be revisited at some
    // point.  Align = N.getOperand(2);
    if (!isa<ConstantSDNode>(N.getOperand(2)) ||
        cast<ConstantSDNode>(N.getOperand(2))->getValue() != 0) {
      std::cerr << "Cannot allocate stack object with greater alignment than"
                << " the stack alignment yet!";
      abort();
    }
    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    // Subtract size from stack pointer, thereby allocating some space.
    BuildMI(BB, PPC::SUBF, 2, PPC::R1).addReg(Tmp1).addReg(PPC::R1);
    // Put a pointer to the space into the result register by copying the SP
    BuildMI(BB, PPC::OR, 2, Result).addReg(PPC::R1).addReg(PPC::R1);
    return Result;

  case ISD::ConstantPool:
    Tmp1 = cast<ConstantPoolSDNode>(N)->getIndex();
    Tmp2 = MakeReg(MVT::i64);
    BuildMI(BB, PPC::LOADHiAddr, 2, Tmp2).addReg(getGlobalBaseReg())
      .addConstantPoolIndex(Tmp1);
    BuildMI(BB, PPC::LA, 2, Result).addReg(Tmp2).addConstantPoolIndex(Tmp1);
    return Result;

  case ISD::FrameIndex:
    Tmp1 = cast<FrameIndexSDNode>(N)->getIndex();
    addFrameReference(BuildMI(BB, PPC::ADDI, 2, Result), (int)Tmp1, 0, false);
    return Result;

  case ISD::GlobalAddress: {
    GlobalValue *GV = cast<GlobalAddressSDNode>(N)->getGlobal();
    Tmp1 = MakeReg(MVT::i64);
    BuildMI(BB, PPC::LOADHiAddr, 2, Tmp1).addReg(getGlobalBaseReg())
      .addGlobalAddress(GV);
    if (GV->hasWeakLinkage() || GV->isExternal()) {
      BuildMI(BB, PPC::LD, 2, Result).addGlobalAddress(GV).addReg(Tmp1);
    } else {
      BuildMI(BB, PPC::LA, 2, Result).addReg(Tmp1).addGlobalAddress(GV);
    }
    return Result;
  }

  case ISD::LOAD:
  case ISD::EXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::SEXTLOAD: {
    MVT::ValueType TypeBeingLoaded = (ISD::LOAD == opcode) ?
      Node->getValueType(0) : cast<MVTSDNode>(Node)->getExtraValueType();
    bool sext = (ISD::SEXTLOAD == opcode);

    // Make sure we generate both values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    SDOperand Chain   = N.getOperand(0);
    SDOperand Address = N.getOperand(1);
    Select(Chain);

    switch (TypeBeingLoaded) {
    default: Node->dump(); assert(0 && "Cannot load this type!");
    case MVT::i1:  Opc = PPC::LBZ; break;
    case MVT::i8:  Opc = PPC::LBZ; break;
    case MVT::i16: Opc = sext ? PPC::LHA : PPC::LHZ; break;
    case MVT::i32: Opc = sext ? PPC::LWA : PPC::LWZ; break;
    case MVT::i64: Opc = PPC::LD;  break;
    case MVT::f32: Opc = PPC::LFS; break;
    case MVT::f64: Opc = PPC::LFD; break;
    }

    if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Address)) {
      Tmp1 = MakeReg(MVT::i64);
      int CPI = CP->getIndex();
      BuildMI(BB, PPC::LOADHiAddr, 2, Tmp1).addReg(getGlobalBaseReg())
        .addConstantPoolIndex(CPI);
      BuildMI(BB, Opc, 2, Result).addConstantPoolIndex(CPI).addReg(Tmp1);
    }
    else if(Address.getOpcode() == ISD::FrameIndex) {
      Tmp1 = cast<FrameIndexSDNode>(Address)->getIndex();
      addFrameReference(BuildMI(BB, Opc, 2, Result), (int)Tmp1);
    } else {
      int offset;
      bool idx = SelectAddr(Address, Tmp1, offset);
      if (idx) {
        Opc = IndexedOpForOp(Opc);
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(offset);
      } else {
        BuildMI(BB, Opc, 2, Result).addSImm(offset).addReg(Tmp1);
      }
    }
    return Result;
  }

  case ISD::TAILCALL:
  case ISD::CALL: {
    unsigned GPR_idx = 0, FPR_idx = 0;
    static const unsigned GPR[] = {
      PPC::R3, PPC::R4, PPC::R5, PPC::R6,
      PPC::R7, PPC::R8, PPC::R9, PPC::R10,
    };
    static const unsigned FPR[] = {
      PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
      PPC::F8, PPC::F9, PPC::F10, PPC::F11, PPC::F12, PPC::F13
    };

    // Lower the chain for this call.
    Select(N.getOperand(0));
    ExprMap[N.getValue(Node->getNumValues()-1)] = 1;

    MachineInstr *CallMI;
    // Emit the correct call instruction based on the type of symbol called.
    if (GlobalAddressSDNode *GASD =
        dyn_cast<GlobalAddressSDNode>(N.getOperand(1))) {
      CallMI = BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(GASD->getGlobal(),
                                                           true);
    } else if (ExternalSymbolSDNode *ESSDN =
               dyn_cast<ExternalSymbolSDNode>(N.getOperand(1))) {
      CallMI = BuildMI(PPC::CALLpcrel, 1).addExternalSymbol(ESSDN->getSymbol(),
                                                            true);
    } else {
      Tmp1 = SelectExpr(N.getOperand(1));
      BuildMI(BB, PPC::OR, 2, PPC::R12).addReg(Tmp1).addReg(Tmp1);
      BuildMI(BB, PPC::MTCTR, 1).addReg(PPC::R12);
      CallMI = BuildMI(PPC::CALLindirect, 3).addImm(20).addImm(0)
        .addReg(PPC::R12);
    }

    // Load the register args to virtual regs
    std::vector<unsigned> ArgVR;
    for(int i = 2, e = Node->getNumOperands(); i < e; ++i)
      ArgVR.push_back(SelectExpr(N.getOperand(i)));

    // Copy the virtual registers into the appropriate argument register
    for(int i = 0, e = ArgVR.size(); i < e; ++i) {
      switch(N.getOperand(i+2).getValueType()) {
      default: Node->dump(); assert(0 && "Unknown value type for call");
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
      case MVT::i64:
        assert(GPR_idx < 8 && "Too many int args");
        if (N.getOperand(i+2).getOpcode() != ISD::UNDEF) {
          BuildMI(BB, PPC::OR,2,GPR[GPR_idx]).addReg(ArgVR[i]).addReg(ArgVR[i]);
          CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
        }
        ++GPR_idx;
        break;
      case MVT::f64:
      case MVT::f32:
        assert(FPR_idx < 13 && "Too many fp args");
        BuildMI(BB, PPC::FMR, 1, FPR[FPR_idx]).addReg(ArgVR[i]);
        CallMI->addRegOperand(FPR[FPR_idx], MachineOperand::Use);
        ++FPR_idx;
        break;
      }
    }

    // Put the call instruction in the correct place in the MachineBasicBlock
    BB->push_back(CallMI);

    switch (Node->getValueType(0)) {
    default: assert(0 && "Unknown value type for call result!");
    case MVT::Other: return 1;
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64:
      BuildMI(BB, PPC::OR, 2, Result).addReg(PPC::R3).addReg(PPC::R3);
      break;
    case MVT::f32:
    case MVT::f64:
      BuildMI(BB, PPC::FMR, 1, Result).addReg(PPC::F1);
      break;
    }
    return Result+N.ResNo;
  }

  case ISD::SIGN_EXTEND:
  case ISD::SIGN_EXTEND_INREG:
    Tmp1 = SelectExpr(N.getOperand(0));
    switch(cast<MVTSDNode>(Node)->getExtraValueType()) {
    default: Node->dump(); assert(0 && "Unhandled SIGN_EXTEND type"); break;
    case MVT::i32:
      BuildMI(BB, PPC::EXTSW, 1, Result).addReg(Tmp1);
      break;
    case MVT::i16:
      BuildMI(BB, PPC::EXTSH, 1, Result).addReg(Tmp1);
      break;
    case MVT::i8:
      BuildMI(BB, PPC::EXTSB, 1, Result).addReg(Tmp1);
      break;
    case MVT::i1:
      BuildMI(BB, PPC::SUBFIC, 2, Result).addReg(Tmp1).addSImm(0);
      break;
    }
    return Result;

  case ISD::CopyFromReg:
    if (Result == 1)
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());
    Tmp1 = dyn_cast<RegSDNode>(Node)->getReg();
    BuildMI(BB, PPC::OR, 2, Result).addReg(Tmp1).addReg(Tmp1);
    return Result;

  case ISD::SHL:
    Tmp1 = SelectExpr(N.getOperand(0));
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      Tmp2 = CN->getValue() & 0x3F;
      BuildMI(BB, PPC::RLDICR, 3, Result).addReg(Tmp1).addImm(Tmp2)
        .addImm(63-Tmp2);
    } else {
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, PPC::SLD, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::SRL:
    Tmp1 = SelectExpr(N.getOperand(0));
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      Tmp2 = CN->getValue() & 0x3F;
      BuildMI(BB, PPC::RLDICL, 3, Result).addReg(Tmp1).addImm(64-Tmp2)
        .addImm(Tmp2);
    } else {
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, PPC::SRD, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::SRA:
    Tmp1 = SelectExpr(N.getOperand(0));
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      Tmp2 = CN->getValue() & 0x3F;
      BuildMI(BB, PPC::SRADI, 2, Result).addReg(Tmp1).addImm(Tmp2);
    } else {
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, PPC::SRAD, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::ADD:
    Tmp1 = SelectExpr(N.getOperand(0));
    switch(getImmediateForOpcode(N.getOperand(1), opcode, Tmp2)) {
      default: assert(0 && "unhandled result code");
      case 0: // No immediate
        Tmp2 = SelectExpr(N.getOperand(1));
        BuildMI(BB, PPC::ADD, 2, Result).addReg(Tmp1).addReg(Tmp2);
        break;
      case 1: // Low immediate
        BuildMI(BB, PPC::ADDI, 2, Result).addReg(Tmp1).addSImm(Tmp2);
        break;
      case 2: // Shifted immediate
        BuildMI(BB, PPC::ADDIS, 2, Result).addReg(Tmp1).addSImm(Tmp2);
        break;
    }
    return Result;

  case ISD::AND:
  case ISD::OR:
    Tmp1 = SelectExpr(N.getOperand(0));
    switch(getImmediateForOpcode(N.getOperand(1), opcode, Tmp2)) {
      default: assert(0 && "unhandled result code");
      case 0: // No immediate
        Tmp2 = SelectExpr(N.getOperand(1));
        switch (opcode) {
        case ISD::AND: Opc = PPC::AND; break;
        case ISD::OR:  Opc = PPC::OR;  break;
        }
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
        break;
      case 1: // Low immediate
        switch (opcode) {
        case ISD::AND: Opc = PPC::ANDIo; break;
        case ISD::OR:  Opc = PPC::ORI;   break;
        }
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(Tmp2);
        break;
      case 2: // Shifted immediate
        switch (opcode) {
        case ISD::AND: Opc = PPC::ANDISo;  break;
        case ISD::OR:  Opc = PPC::ORIS;    break;
        }
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(Tmp2);
        break;
    }
    return Result;

  case ISD::XOR: {
    // Check for EQV: xor, (xor a, -1), b
    if (N.getOperand(0).getOpcode() == ISD::XOR &&
        N.getOperand(0).getOperand(1).getOpcode() == ISD::Constant &&
        cast<ConstantSDNode>(N.getOperand(0).getOperand(1))->isAllOnesValue()) {
      ++NotLogic;
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, PPC::EQV, 2, Result).addReg(Tmp1).addReg(Tmp2);
      return Result;
    }
    // Check for NOT, NOR, and NAND: xor (copy, or, and), -1
    if (N.getOperand(1).getOpcode() == ISD::Constant &&
        cast<ConstantSDNode>(N.getOperand(1))->isAllOnesValue()) {
      ++NotLogic;
      switch(N.getOperand(0).getOpcode()) {
      case ISD::OR:
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
        BuildMI(BB, PPC::NOR, 2, Result).addReg(Tmp1).addReg(Tmp2);
        break;
      case ISD::AND:
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
        BuildMI(BB, PPC::NAND, 2, Result).addReg(Tmp1).addReg(Tmp2);
        break;
      default:
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, PPC::NOR, 2, Result).addReg(Tmp1).addReg(Tmp1);
        break;
      }
      return Result;
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    switch(getImmediateForOpcode(N.getOperand(1), opcode, Tmp2)) {
      default: assert(0 && "unhandled result code");
      case 0: // No immediate
        Tmp2 = SelectExpr(N.getOperand(1));
        BuildMI(BB, PPC::XOR, 2, Result).addReg(Tmp1).addReg(Tmp2);
        break;
      case 1: // Low immediate
        BuildMI(BB, PPC::XORI, 2, Result).addReg(Tmp1).addImm(Tmp2);
        break;
      case 2: // Shifted immediate
        BuildMI(BB, PPC::XORIS, 2, Result).addReg(Tmp1).addImm(Tmp2);
        break;
    }
    return Result;
  }

  case ISD::SUB:
    Tmp2 = SelectExpr(N.getOperand(1));
    if (1 == getImmediateForOpcode(N.getOperand(0), opcode, Tmp1))
      BuildMI(BB, PPC::SUBFIC, 2, Result).addReg(Tmp2).addSImm(Tmp1);
    else {
      Tmp1 = SelectExpr(N.getOperand(0));
      BuildMI(BB, PPC::SUBF, 2, Result).addReg(Tmp2).addReg(Tmp1);
    }
    return Result;

  case ISD::MUL:
    Tmp1 = SelectExpr(N.getOperand(0));
    if (1 == getImmediateForOpcode(N.getOperand(1), opcode, Tmp2))
      BuildMI(BB, PPC::MULLI, 2, Result).addReg(Tmp1).addSImm(Tmp2);
    else {
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, PPC::MULLD, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::SDIV:
  case ISD::UDIV:
    if (3 == getImmediateForOpcode(N.getOperand(1), opcode, Tmp3)) {
      Tmp1 = MakeReg(MVT::i64);
      Tmp2 = SelectExpr(N.getOperand(0));
      BuildMI(BB, PPC::SRADI, 2, Tmp1).addReg(Tmp2).addImm(Tmp3);
      BuildMI(BB, PPC::ADDZE, 1, Result).addReg(Tmp1);
      return Result;
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    Opc = (ISD::UDIV == opcode) ? PPC::DIVWU : PPC::DIVW;
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::FP_TO_UINT:
  case ISD::FP_TO_SINT: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = MakeReg(MVT::f64);
    BuildMI(BB, PPC::FCTIDZ, 1, Tmp2).addReg(Tmp1);
    int FrameIdx = BB->getParent()->getFrameInfo()->CreateStackObject(8, 8);
    addFrameReference(BuildMI(BB, PPC::STFD, 3).addReg(Tmp2), FrameIdx);
    addFrameReference(BuildMI(BB, PPC::LD, 2, Result), FrameIdx);
    return Result;
  }

  case ISD::SETCC:
    if (SetCCSDNode *SetCC = dyn_cast<SetCCSDNode>(Node)) {
      Opc = SelectSetCR0(N);

      unsigned TrueValue = MakeReg(MVT::i32);
      BuildMI(BB, PPC::LI, 1, TrueValue).addSImm(1);
      unsigned FalseValue = MakeReg(MVT::i32);
      BuildMI(BB, PPC::LI, 1, FalseValue).addSImm(0);

      // Create an iterator with which to insert the MBB for copying the false
      // value and the MBB to hold the PHI instruction for this SetCC.
      MachineBasicBlock *thisMBB = BB;
      const BasicBlock *LLVM_BB = BB->getBasicBlock();
      ilist<MachineBasicBlock>::iterator It = BB;
      ++It;

      //  thisMBB:
      //  ...
      //   cmpTY cr0, r1, r2
      //   %TrueValue = li 1
      //   bCC sinkMBB
      MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
      MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
      BuildMI(BB, Opc, 2).addReg(PPC::CR0).addMBB(sinkMBB);
      MachineFunction *F = BB->getParent();
      F->getBasicBlockList().insert(It, copy0MBB);
      F->getBasicBlockList().insert(It, sinkMBB);
      // Update machine-CFG edges
      BB->addSuccessor(copy0MBB);
      BB->addSuccessor(sinkMBB);

      //  copy0MBB:
      //   %FalseValue = li 0
      //   fallthrough
      BB = copy0MBB;
      // Update machine-CFG edges
      BB->addSuccessor(sinkMBB);

      //  sinkMBB:
      //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
      //  ...
      BB = sinkMBB;
      BuildMI(BB, PPC::PHI, 4, Result).addReg(FalseValue)
        .addMBB(copy0MBB).addReg(TrueValue).addMBB(thisMBB);
      return Result;
    }
    assert(0 && "Is this legal?");
    return 0;

  case ISD::SELECT: {
    unsigned TrueValue = SelectExpr(N.getOperand(1)); //Use if TRUE
    unsigned FalseValue = SelectExpr(N.getOperand(2)); //Use if FALSE
    Opc = SelectSetCR0(N.getOperand(0));

    // Create an iterator with which to insert the MBB for copying the false
    // value and the MBB to hold the PHI instruction for this SetCC.
    MachineBasicBlock *thisMBB = BB;
    const BasicBlock *LLVM_BB = BB->getBasicBlock();
    ilist<MachineBasicBlock>::iterator It = BB;
    ++It;

    //  thisMBB:
    //  ...
    //   TrueVal = ...
    //   cmpTY cr0, r1, r2
    //   bCC copy1MBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
    BuildMI(BB, Opc, 2).addReg(PPC::CR0).addMBB(sinkMBB);
    MachineFunction *F = BB->getParent();
    F->getBasicBlockList().insert(It, copy0MBB);
    F->getBasicBlockList().insert(It, sinkMBB);
    // Update machine-CFG edges
    BB->addSuccessor(copy0MBB);
    BB->addSuccessor(sinkMBB);

    //  copy0MBB:
    //   %FalseValue = ...
    //   # fallthrough to sinkMBB
    BB = copy0MBB;
    // Update machine-CFG edges
    BB->addSuccessor(sinkMBB);

    //  sinkMBB:
    //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
    //  ...
    BB = sinkMBB;
    BuildMI(BB, PPC::PHI, 4, Result).addReg(FalseValue)
      .addMBB(copy0MBB).addReg(TrueValue).addMBB(thisMBB);

    // FIXME: Select i64?
    return Result;
  }

  case ISD::Constant:
    switch (N.getValueType()) {
    default: assert(0 && "Cannot use constants of this type!");
    case MVT::i1:
      BuildMI(BB, PPC::LI, 1, Result)
        .addSImm(!cast<ConstantSDNode>(N)->isNullValue());
      break;
    case MVT::i32:
      {
        int v = (int)cast<ConstantSDNode>(N)->getSignExtended();
        if (v < 32768 && v >= -32768) {
          BuildMI(BB, PPC::LI, 1, Result).addSImm(v);
        } else {
          Tmp1 = MakeReg(MVT::i32);
          BuildMI(BB, PPC::LIS, 1, Tmp1).addSImm(v >> 16);
          BuildMI(BB, PPC::ORI, 2, Result).addReg(Tmp1).addImm(v & 0xFFFF);
        }
      }
    }
    return Result;
  }

  return 0;
}

void ISel::Select(SDOperand N) {
  unsigned Tmp1, Tmp2, Opc;
  unsigned opcode = N.getOpcode();

  if (!ExprMap.insert(std::make_pair(N, 1)).second)
    return;  // Already selected.

  SDNode *Node = N.Val;

  switch (Node->getOpcode()) {
  default:
    Node->dump(); std::cerr << "\n";
    assert(0 && "Node not handled yet!");
  case ISD::EntryToken: return;  // Noop
  case ISD::TokenFactor:
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
      Select(Node->getOperand(i));
    return;
  case ISD::CALLSEQ_START:
  case ISD::CALLSEQ_END:
    Select(N.getOperand(0));
    Tmp1 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
    Opc = N.getOpcode() == ISD::CALLSEQ_START ? PPC::ADJCALLSTACKDOWN :
      PPC::ADJCALLSTACKUP;
    BuildMI(BB, Opc, 1).addImm(Tmp1);
    return;
  case ISD::BR: {
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(1))->getBasicBlock();
    Select(N.getOperand(0));
    BuildMI(BB, PPC::B, 1).addMBB(Dest);
    return;
  }
  case ISD::BRCOND:
    SelectBranchCC(N);
    return;
  case ISD::CopyToReg:
    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    Tmp2 = cast<RegSDNode>(N)->getReg();

    if (Tmp1 != Tmp2) {
      if (N.getOperand(1).getValueType() == MVT::f64 ||
          N.getOperand(1).getValueType() == MVT::f32)
        BuildMI(BB, PPC::FMR, 1, Tmp2).addReg(Tmp1);
      else
        BuildMI(BB, PPC::OR, 2, Tmp2).addReg(Tmp1).addReg(Tmp1);
    }
    return;
  case ISD::ImplicitDef:
    Select(N.getOperand(0));
    BuildMI(BB, PPC::IMPLICIT_DEF, 0, cast<RegSDNode>(N)->getReg());
    return;
  case ISD::RET:
    switch (N.getNumOperands()) {
    default:
      assert(0 && "Unknown return instruction!");
    case 3:
      assert(N.getOperand(1).getValueType() == MVT::i32 &&
             N.getOperand(2).getValueType() == MVT::i32 &&
             "Unknown two-register value!");
      Select(N.getOperand(0));
      Tmp1 = SelectExpr(N.getOperand(1));
      Tmp2 = SelectExpr(N.getOperand(2));
      BuildMI(BB, PPC::OR, 2, PPC::R3).addReg(Tmp2).addReg(Tmp2);
      BuildMI(BB, PPC::OR, 2, PPC::R4).addReg(Tmp1).addReg(Tmp1);
      break;
    case 2:
      Select(N.getOperand(0));
      Tmp1 = SelectExpr(N.getOperand(1));
      switch (N.getOperand(1).getValueType()) {
        default:
          assert(0 && "Unknown return type!");
        case MVT::f64:
        case MVT::f32:
          BuildMI(BB, PPC::FMR, 1, PPC::F1).addReg(Tmp1);
          break;
        case MVT::i32:
          BuildMI(BB, PPC::OR, 2, PPC::R3).addReg(Tmp1).addReg(Tmp1);
          break;
      }
    case 1:
      Select(N.getOperand(0));
      break;
    }
    BuildMI(BB, PPC::BLR, 0); // Just emit a 'ret' instruction
    return;
  case ISD::TRUNCSTORE:
  case ISD::STORE:
    {
      SDOperand Chain   = N.getOperand(0);
      SDOperand Value   = N.getOperand(1);
      SDOperand Address = N.getOperand(2);
      Select(Chain);

      Tmp1 = SelectExpr(Value); //value

      if (opcode == ISD::STORE) {
        switch(Value.getValueType()) {
        default: assert(0 && "unknown Type in store");
        case MVT::i64: Opc = PPC::STD; break;
        case MVT::f64: Opc = PPC::STFD; break;
        case MVT::f32: Opc = PPC::STFS; break;
        }
      } else { //ISD::TRUNCSTORE
        switch(cast<MVTSDNode>(Node)->getExtraValueType()) {
        default: assert(0 && "unknown Type in store");
        case MVT::i1: //FIXME: DAG does not promote this load
        case MVT::i8:  Opc= PPC::STB; break;
        case MVT::i16: Opc = PPC::STH; break;
        case MVT::i32: Opc = PPC::STW; break;
        }
      }

      if(Address.getOpcode() == ISD::FrameIndex)
      {
        Tmp2 = cast<FrameIndexSDNode>(Address)->getIndex();
        addFrameReference(BuildMI(BB, Opc, 3).addReg(Tmp1), (int)Tmp2);
      }
      else
      {
        int offset;
        bool idx = SelectAddr(Address, Tmp2, offset);
        if (idx) {
          Opc = IndexedOpForOp(Opc);
          BuildMI(BB, Opc, 3).addReg(Tmp1).addReg(Tmp2).addReg(offset);
        } else {
          BuildMI(BB, Opc, 3).addReg(Tmp1).addImm(offset).addReg(Tmp2);
        }
      }
      return;
    }
  case ISD::EXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::LOAD:
  case ISD::CopyFromReg:
  case ISD::TAILCALL:
  case ISD::CALL:
  case ISD::DYNAMIC_STACKALLOC:
    ExprMap.erase(N);
    SelectExpr(N);
    return;
  }
  assert(0 && "Should not be reached!");
}


/// createPPC32PatternInstructionSelector - This pass converts an LLVM function
/// into a machine code representation using pattern matching and a machine
/// description file.
///
FunctionPass *llvm::createPPC64ISelPattern(TargetMachine &TM) {
  return new ISel(TM);
}

