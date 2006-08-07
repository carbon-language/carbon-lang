//===-- SparcISelDAGToDAG.cpp - A dag to dag inst selector for Sparc ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the SPARC target.
//
//===----------------------------------------------------------------------===//

#include "Sparc.h"
#include "SparcTargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/Debug.h"
#include <iostream>
#include <queue>
#include <set>
using namespace llvm;

//===----------------------------------------------------------------------===//
// TargetLowering Implementation
//===----------------------------------------------------------------------===//

namespace SPISD {
  enum {
    FIRST_NUMBER = ISD::BUILTIN_OP_END+SP::INSTRUCTION_LIST_END,
    CMPICC,      // Compare two GPR operands, set icc.
    CMPFCC,      // Compare two FP operands, set fcc.
    BRICC,       // Branch to dest on icc condition
    BRFCC,       // Branch to dest on fcc condition
    SELECT_ICC,  // Select between two values using the current ICC flags.
    SELECT_FCC,  // Select between two values using the current FCC flags.
    
    Hi, Lo,      // Hi/Lo operations, typically on a global address.
    
    FTOI,        // FP to Int within a FP register.
    ITOF,        // Int to FP within a FP register.

    CALL,        // A call instruction.
    RET_FLAG     // Return with a flag operand.
  };
}

/// IntCondCCodeToICC - Convert a DAG integer condition code to a SPARC ICC
/// condition.
static SPCC::CondCodes IntCondCCodeToICC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown integer condition code!");
  case ISD::SETEQ:  return SPCC::ICC_E;
  case ISD::SETNE:  return SPCC::ICC_NE;
  case ISD::SETLT:  return SPCC::ICC_L;
  case ISD::SETGT:  return SPCC::ICC_G;
  case ISD::SETLE:  return SPCC::ICC_LE;
  case ISD::SETGE:  return SPCC::ICC_GE;
  case ISD::SETULT: return SPCC::ICC_CS;
  case ISD::SETULE: return SPCC::ICC_LEU;
  case ISD::SETUGT: return SPCC::ICC_GU;
  case ISD::SETUGE: return SPCC::ICC_CC;
  }
}

/// FPCondCCodeToFCC - Convert a DAG floatingp oint condition code to a SPARC
/// FCC condition.
static SPCC::CondCodes FPCondCCodeToFCC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown fp condition code!");
  case ISD::SETEQ:
  case ISD::SETOEQ: return SPCC::FCC_E;
  case ISD::SETNE:
  case ISD::SETUNE: return SPCC::FCC_NE;
  case ISD::SETLT:
  case ISD::SETOLT: return SPCC::FCC_L;
  case ISD::SETGT:
  case ISD::SETOGT: return SPCC::FCC_G;
  case ISD::SETLE:
  case ISD::SETOLE: return SPCC::FCC_LE;
  case ISD::SETGE:
  case ISD::SETOGE: return SPCC::FCC_GE;
  case ISD::SETULT: return SPCC::FCC_UL;
  case ISD::SETULE: return SPCC::FCC_ULE;
  case ISD::SETUGT: return SPCC::FCC_UG;
  case ISD::SETUGE: return SPCC::FCC_UGE;
  case ISD::SETUO:  return SPCC::FCC_U;
  case ISD::SETO:   return SPCC::FCC_O;
  case ISD::SETONE: return SPCC::FCC_LG;
  case ISD::SETUEQ: return SPCC::FCC_UE;
  }
}

namespace {
  class SparcTargetLowering : public TargetLowering {
    int VarArgsFrameOffset;   // Frame offset to start of varargs area.
  public:
    SparcTargetLowering(TargetMachine &TM);
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    
    /// computeMaskedBitsForTargetNode - Determine which of the bits specified 
    /// in Mask are known to be either zero or one and return them in the 
    /// KnownZero/KnownOne bitsets.
    virtual void computeMaskedBitsForTargetNode(const SDOperand Op,
                                                uint64_t Mask,
                                                uint64_t &KnownZero, 
                                                uint64_t &KnownOne,
                                                unsigned Depth = 0) const;
    
    virtual std::vector<SDOperand>
      LowerArguments(Function &F, SelectionDAG &DAG);
    virtual std::pair<SDOperand, SDOperand>
      LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg,
                  unsigned CC,
                  bool isTailCall, SDOperand Callee, ArgListTy &Args,
                  SelectionDAG &DAG);
    virtual MachineBasicBlock *InsertAtEndOfBasicBlock(MachineInstr *MI,
                                                       MachineBasicBlock *MBB);
    
    virtual const char *getTargetNodeName(unsigned Opcode) const;
  };
}

SparcTargetLowering::SparcTargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  
  // Set up the register classes.
  addRegisterClass(MVT::i32, SP::IntRegsRegisterClass);
  addRegisterClass(MVT::f32, SP::FPRegsRegisterClass);
  addRegisterClass(MVT::f64, SP::DFPRegsRegisterClass);

  // Custom legalize GlobalAddress nodes into LO/HI parts.
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool , MVT::i32, Custom);
  
  // Sparc doesn't have sext_inreg, replace them with shl/sra
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8 , Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1 , Expand);

  // Sparc has no REM operation.
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i32, Expand);

  // Custom expand fp<->sint
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);

  // Expand fp<->uint
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);
  
  setOperationAction(ISD::BIT_CONVERT, MVT::f32, Expand);
  setOperationAction(ISD::BIT_CONVERT, MVT::i32, Expand);
  
  // Turn FP extload into load/fextend
  setOperationAction(ISD::EXTLOAD, MVT::f32, Expand);
  
  // Sparc has no select or setcc: expand to SELECT_CC.
  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT, MVT::f32, Expand);
  setOperationAction(ISD::SELECT, MVT::f64, Expand);
  setOperationAction(ISD::SETCC, MVT::i32, Expand);
  setOperationAction(ISD::SETCC, MVT::f32, Expand);
  setOperationAction(ISD::SETCC, MVT::f64, Expand);
  
  // Sparc doesn't have BRCOND either, it has BR_CC.
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::i32, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f64, Custom);
  
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);
  
  // SPARC has no intrinsics for these particular operations.
  setOperationAction(ISD::MEMMOVE, MVT::Other, Expand);
  setOperationAction(ISD::MEMSET, MVT::Other, Expand);
  setOperationAction(ISD::MEMCPY, MVT::Other, Expand);
  
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ , MVT::i32, Expand);
  setOperationAction(ISD::CTLZ , MVT::i32, Expand);
  setOperationAction(ISD::ROTL , MVT::i32, Expand);
  setOperationAction(ISD::ROTR , MVT::i32, Expand);
  setOperationAction(ISD::BSWAP, MVT::i32, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);

  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);

  // We don't have line number support yet.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LABEL, MVT::Other, Expand);

  // RET must be custom lowered, to meet ABI requirements
  setOperationAction(ISD::RET               , MVT::Other, Custom);
  
  // VASTART needs to be custom lowered to use the VarArgsFrameIndex.
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);
  // VAARG needs to be lowered to not do unaligned accesses for doubles.
  setOperationAction(ISD::VAARG             , MVT::Other, Custom);
  
  // Use the default implementation.
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE         , MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE      , MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Custom);

  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f32, Expand);
  
  setStackPointerRegisterToSaveRestore(SP::O6);

  if (TM.getSubtarget<SparcSubtarget>().isV9()) {
    setOperationAction(ISD::CTPOP, MVT::i32, Legal);
  }
  
  computeRegisterProperties();
}

const char *SparcTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case SPISD::CMPICC:     return "SPISD::CMPICC";
  case SPISD::CMPFCC:     return "SPISD::CMPFCC";
  case SPISD::BRICC:      return "SPISD::BRICC";
  case SPISD::BRFCC:      return "SPISD::BRFCC";
  case SPISD::SELECT_ICC: return "SPISD::SELECT_ICC";
  case SPISD::SELECT_FCC: return "SPISD::SELECT_FCC";
  case SPISD::Hi:         return "SPISD::Hi";
  case SPISD::Lo:         return "SPISD::Lo";
  case SPISD::FTOI:       return "SPISD::FTOI";
  case SPISD::ITOF:       return "SPISD::ITOF";
  case SPISD::CALL:       return "SPISD::CALL";
  case SPISD::RET_FLAG:   return "SPISD::RET_FLAG";
  }
}

/// isMaskedValueZeroForTargetNode - Return true if 'Op & Mask' is known to
/// be zero. Op is expected to be a target specific node. Used by DAG
/// combiner.
void SparcTargetLowering::computeMaskedBitsForTargetNode(const SDOperand Op,
                                                         uint64_t Mask,
                                                         uint64_t &KnownZero, 
                                                         uint64_t &KnownOne,
                                                         unsigned Depth) const {
  uint64_t KnownZero2, KnownOne2;
  KnownZero = KnownOne = 0;   // Don't know anything.
  
  switch (Op.getOpcode()) {
  default: break;
  case SPISD::SELECT_ICC:
  case SPISD::SELECT_FCC:
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  }
}

/// LowerArguments - V8 uses a very simple ABI, where all values are passed in
/// either one or two GPRs, including FP values.  TODO: we should pass FP values
/// in FP registers for fastcc functions.
std::vector<SDOperand>
SparcTargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  SSARegMap *RegMap = MF.getSSARegMap();
  std::vector<SDOperand> ArgValues;
  
  static const unsigned ArgRegs[] = {
    SP::I0, SP::I1, SP::I2, SP::I3, SP::I4, SP::I5
  };
  
  const unsigned *CurArgReg = ArgRegs, *ArgRegEnd = ArgRegs+6;
  unsigned ArgOffset = 68;
  
  SDOperand Root = DAG.getRoot();
  std::vector<SDOperand> OutChains;

  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I) {
    MVT::ValueType ObjectVT = getValueType(I->getType());
    
    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      if (I->use_empty()) {                // Argument is dead.
        if (CurArgReg < ArgRegEnd) ++CurArgReg;
        ArgValues.push_back(DAG.getNode(ISD::UNDEF, ObjectVT));
      } else if (CurArgReg < ArgRegEnd) {  // Lives in an incoming GPR
        unsigned VReg = RegMap->createVirtualRegister(&SP::IntRegsRegClass);
        MF.addLiveIn(*CurArgReg++, VReg);
        SDOperand Arg = DAG.getCopyFromReg(Root, VReg, MVT::i32);
        if (ObjectVT != MVT::i32) {
          unsigned AssertOp = I->getType()->isSigned() ? ISD::AssertSext 
                                                       : ISD::AssertZext;
          Arg = DAG.getNode(AssertOp, MVT::i32, Arg, 
                            DAG.getValueType(ObjectVT));
          Arg = DAG.getNode(ISD::TRUNCATE, ObjectVT, Arg);
        }
        ArgValues.push_back(Arg);
      } else {
        int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
        SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
        SDOperand Load;
        if (ObjectVT == MVT::i32) {
          Load = DAG.getLoad(MVT::i32, Root, FIPtr, DAG.getSrcValue(0));
        } else {
          unsigned LoadOp =
            I->getType()->isSigned() ? ISD::SEXTLOAD : ISD::ZEXTLOAD;

          // Sparc is big endian, so add an offset based on the ObjectVT.
          unsigned Offset = 4-std::max(1U, MVT::getSizeInBits(ObjectVT)/8);
          FIPtr = DAG.getNode(ISD::ADD, MVT::i32, FIPtr,
                              DAG.getConstant(Offset, MVT::i32));
          Load = DAG.getExtLoad(LoadOp, MVT::i32, Root, FIPtr,
                                DAG.getSrcValue(0), ObjectVT);
          Load = DAG.getNode(ISD::TRUNCATE, ObjectVT, Load);
        }
        ArgValues.push_back(Load);
      }
      
      ArgOffset += 4;
      break;
    case MVT::f32:
      if (I->use_empty()) {                // Argument is dead.
        if (CurArgReg < ArgRegEnd) ++CurArgReg;
        ArgValues.push_back(DAG.getNode(ISD::UNDEF, ObjectVT));
      } else if (CurArgReg < ArgRegEnd) {  // Lives in an incoming GPR
        // FP value is passed in an integer register.
        unsigned VReg = RegMap->createVirtualRegister(&SP::IntRegsRegClass);
        MF.addLiveIn(*CurArgReg++, VReg);
        SDOperand Arg = DAG.getCopyFromReg(Root, VReg, MVT::i32);

        Arg = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, Arg);
        ArgValues.push_back(Arg);
      } else {
        int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
        SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
        SDOperand Load = DAG.getLoad(MVT::f32, Root, FIPtr, DAG.getSrcValue(0));
        ArgValues.push_back(Load);
      }
      ArgOffset += 4;
      break;

    case MVT::i64:
    case MVT::f64:
      if (I->use_empty()) {                // Argument is dead.
        if (CurArgReg < ArgRegEnd) ++CurArgReg;
        if (CurArgReg < ArgRegEnd) ++CurArgReg;
        ArgValues.push_back(DAG.getNode(ISD::UNDEF, ObjectVT));
      } else if (/* FIXME: Apparently this isn't safe?? */
                 0 && CurArgReg == ArgRegEnd && ObjectVT == MVT::f64 &&
                 ((CurArgReg-ArgRegs) & 1) == 0) {
        // If this is a double argument and the whole thing lives on the stack,
        // and the argument is aligned, load the double straight from the stack.
        // We can't do a load in cases like void foo([6ints], int,double),
        // because the double wouldn't be aligned!
        int FrameIdx = MF.getFrameInfo()->CreateFixedObject(8, ArgOffset);
        SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
        ArgValues.push_back(DAG.getLoad(MVT::f64, Root, FIPtr, 
                                        DAG.getSrcValue(0)));
      } else {
        SDOperand HiVal;
        if (CurArgReg < ArgRegEnd) {  // Lives in an incoming GPR
          unsigned VRegHi = RegMap->createVirtualRegister(&SP::IntRegsRegClass);
          MF.addLiveIn(*CurArgReg++, VRegHi);
          HiVal = DAG.getCopyFromReg(Root, VRegHi, MVT::i32);
        } else {
          int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
          SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
          HiVal = DAG.getLoad(MVT::i32, Root, FIPtr, DAG.getSrcValue(0));
        }
        
        SDOperand LoVal;
        if (CurArgReg < ArgRegEnd) {  // Lives in an incoming GPR
          unsigned VRegLo = RegMap->createVirtualRegister(&SP::IntRegsRegClass);
          MF.addLiveIn(*CurArgReg++, VRegLo);
          LoVal = DAG.getCopyFromReg(Root, VRegLo, MVT::i32);
        } else {
          int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset+4);
          SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
          LoVal = DAG.getLoad(MVT::i32, Root, FIPtr, DAG.getSrcValue(0));
        }
        
        // Compose the two halves together into an i64 unit.
        SDOperand WholeValue = 
          DAG.getNode(ISD::BUILD_PAIR, MVT::i64, LoVal, HiVal);
        
        // If we want a double, do a bit convert.
        if (ObjectVT == MVT::f64)
          WholeValue = DAG.getNode(ISD::BIT_CONVERT, MVT::f64, WholeValue);
        
        ArgValues.push_back(WholeValue);
      }
      ArgOffset += 8;
      break;
    }
  }
  
  // Store remaining ArgRegs to the stack if this is a varargs function.
  if (F.getFunctionType()->isVarArg()) {
    // Remember the vararg offset for the va_start implementation.
    VarArgsFrameOffset = ArgOffset;
    
    for (; CurArgReg != ArgRegEnd; ++CurArgReg) {
      unsigned VReg = RegMap->createVirtualRegister(&SP::IntRegsRegClass);
      MF.addLiveIn(*CurArgReg, VReg);
      SDOperand Arg = DAG.getCopyFromReg(DAG.getRoot(), VReg, MVT::i32);

      int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
      SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);

      OutChains.push_back(DAG.getNode(ISD::STORE, MVT::Other, DAG.getRoot(),
                                      Arg, FIPtr, DAG.getSrcValue(0)));
      ArgOffset += 4;
    }
  }
  
  if (!OutChains.empty())
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other, OutChains));
  
  // Finally, inform the code generator which regs we return values in.
  switch (getValueType(F.getReturnType())) {
  default: assert(0 && "Unknown type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    MF.addLiveOut(SP::I0);
    break;
  case MVT::i64:
    MF.addLiveOut(SP::I0);
    MF.addLiveOut(SP::I1);
    break;
  case MVT::f32:
    MF.addLiveOut(SP::F0);
    break;
  case MVT::f64:
    MF.addLiveOut(SP::D0);
    break;
  }
  
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
SparcTargetLowering::LowerCallTo(SDOperand Chain, const Type *RetTy,
                                 bool isVarArg, unsigned CC,
                                 bool isTailCall, SDOperand Callee, 
                                 ArgListTy &Args, SelectionDAG &DAG) {
  // Count the size of the outgoing arguments.
  unsigned ArgsSize = 0;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    switch (getValueType(Args[i].second)) {
    default: assert(0 && "Unknown value type!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::f32:
      ArgsSize += 4;
      break;
    case MVT::i64:
    case MVT::f64:
      ArgsSize += 8;
      break;
    }
  }
  if (ArgsSize > 4*6)
    ArgsSize -= 4*6;    // Space for first 6 arguments is prereserved.
  else
    ArgsSize = 0;

  // Keep stack frames 8-byte aligned.
  ArgsSize = (ArgsSize+7) & ~7;

  Chain = DAG.getCALLSEQ_START(Chain,DAG.getConstant(ArgsSize, getPointerTy()));
  
  SDOperand StackPtr, NullSV;
  std::vector<SDOperand> Stores;
  std::vector<SDOperand> RegValuesToPass;
  unsigned ArgOffset = 68;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    SDOperand Val = Args[i].first;
    MVT::ValueType ObjectVT = Val.getValueType();
    SDOperand ValToStore(0, 0);
    unsigned ObjSize;
    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
      // Promote the integer to 32-bits.  If the input type is signed, use a
      // sign extend, otherwise use a zero extend.
      if (Args[i].second->isSigned())
        Val = DAG.getNode(ISD::SIGN_EXTEND, MVT::i32, Val);
      else
        Val = DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, Val);
      // FALL THROUGH
    case MVT::i32:
      ObjSize = 4;

      if (RegValuesToPass.size() >= 6) {
        ValToStore = Val;
      } else {
        RegValuesToPass.push_back(Val);
      }
      break;
    case MVT::f32:
      ObjSize = 4;
      if (RegValuesToPass.size() >= 6) {
        ValToStore = Val;
      } else {
        // Convert this to a FP value in an int reg.
        Val = DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Val);
        RegValuesToPass.push_back(Val);
      }
      break;
    case MVT::f64:
      ObjSize = 8;
      // If we can store this directly into the outgoing slot, do so.  We can
      // do this when all ArgRegs are used and if the outgoing slot is aligned.
      // FIXME: McGill/misr fails with this.
      if (0 && RegValuesToPass.size() >= 6 && ((ArgOffset-68) & 7) == 0) {
        ValToStore = Val;
        break;
      }
      
      // Otherwise, convert this to a FP value in int regs.
      Val = DAG.getNode(ISD::BIT_CONVERT, MVT::i64, Val);
      // FALL THROUGH
    case MVT::i64:
      ObjSize = 8;
      if (RegValuesToPass.size() >= 6) {
        ValToStore = Val;    // Whole thing is passed in memory.
        break;
      }
      
      // Split the value into top and bottom part.  Top part goes in a reg.
      SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, getPointerTy(), Val, 
                                 DAG.getConstant(1, MVT::i32));
      SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, getPointerTy(), Val,
                                 DAG.getConstant(0, MVT::i32));
      RegValuesToPass.push_back(Hi);
      
      if (RegValuesToPass.size() >= 6) {
        ValToStore = Lo;
        ArgOffset += 4;
        ObjSize = 4;
      } else {
        RegValuesToPass.push_back(Lo);
      }
      break;
    }
    
    if (ValToStore.Val) {
      if (!StackPtr.Val) {
        StackPtr = DAG.getRegister(SP::O6, MVT::i32);
        NullSV = DAG.getSrcValue(NULL);
      }
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
      Stores.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                   ValToStore, PtrOff, NullSV));
    }
    ArgOffset += ObjSize;
  }
  
  // Emit all stores, make sure the occur before any copies into physregs.
  if (!Stores.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, Stores);
  
  static const unsigned ArgRegs[] = {
    SP::O0, SP::O1, SP::O2, SP::O3, SP::O4, SP::O5
  };
  
  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into O[0-5].
  SDOperand InFlag;
  for (unsigned i = 0, e = RegValuesToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, ArgRegs[i], RegValuesToPass[i], InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i32);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), MVT::i32);

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  if (InFlag.Val)
    Ops.push_back(InFlag);
  Chain = DAG.getNode(SPISD::CALL, NodeTys, Ops);
  InFlag = Chain.getValue(1);
  
  MVT::ValueType RetTyVT = getValueType(RetTy);
  SDOperand RetVal;
  if (RetTyVT != MVT::isVoid) {
    switch (RetTyVT) {
    default: assert(0 && "Unknown value type to return!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
      RetVal = DAG.getCopyFromReg(Chain, SP::O0, MVT::i32, InFlag);
      Chain = RetVal.getValue(1);
      
      // Add a note to keep track of whether it is sign or zero extended.
      RetVal = DAG.getNode(RetTy->isSigned() ? ISD::AssertSext :ISD::AssertZext,
                           MVT::i32, RetVal, DAG.getValueType(RetTyVT));
      RetVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, RetVal);
      break;
    case MVT::i32:
      RetVal = DAG.getCopyFromReg(Chain, SP::O0, MVT::i32, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::f32:
      RetVal = DAG.getCopyFromReg(Chain, SP::F0, MVT::f32, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::f64:
      RetVal = DAG.getCopyFromReg(Chain, SP::D0, MVT::f64, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::i64:
      SDOperand Lo = DAG.getCopyFromReg(Chain, SP::O1, MVT::i32, InFlag);
      SDOperand Hi = DAG.getCopyFromReg(Lo.getValue(1), SP::O0, MVT::i32, 
                                        Lo.getValue(2));
      RetVal = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Lo, Hi);
      Chain = Hi.getValue(1);
      break;
    }
  }
  
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(ArgsSize, getPointerTy()));
  
  return std::make_pair(RetVal, Chain);
}

// Look at LHS/RHS/CC and see if they are a lowered setcc instruction.  If so
// set LHS/RHS and SPCC to the LHS/RHS of the setcc and SPCC to the condition.
static void LookThroughSetCC(SDOperand &LHS, SDOperand &RHS,
                             ISD::CondCode CC, unsigned &SPCC) {
  if (isa<ConstantSDNode>(RHS) && cast<ConstantSDNode>(RHS)->getValue() == 0 &&
      CC == ISD::SETNE && 
      ((LHS.getOpcode() == SPISD::SELECT_ICC &&
        LHS.getOperand(3).getOpcode() == SPISD::CMPICC) ||
       (LHS.getOpcode() == SPISD::SELECT_FCC &&
        LHS.getOperand(3).getOpcode() == SPISD::CMPFCC)) &&
      isa<ConstantSDNode>(LHS.getOperand(0)) &&
      isa<ConstantSDNode>(LHS.getOperand(1)) &&
      cast<ConstantSDNode>(LHS.getOperand(0))->getValue() == 1 &&
      cast<ConstantSDNode>(LHS.getOperand(1))->getValue() == 0) {
    SDOperand CMPCC = LHS.getOperand(3);
    SPCC = cast<ConstantSDNode>(LHS.getOperand(2))->getValue();
    LHS = CMPCC.getOperand(0);
    RHS = CMPCC.getOperand(1);
  }
}


SDOperand SparcTargetLowering::
LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Should not custom lower this!");
  case ISD::GlobalAddress: {
    GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
    SDOperand GA = DAG.getTargetGlobalAddress(GV, MVT::i32);
    SDOperand Hi = DAG.getNode(SPISD::Hi, MVT::i32, GA);
    SDOperand Lo = DAG.getNode(SPISD::Lo, MVT::i32, GA);
    return DAG.getNode(ISD::ADD, MVT::i32, Lo, Hi);
  }
  case ISD::ConstantPool: {
    Constant *C = cast<ConstantPoolSDNode>(Op)->get();
    SDOperand CP = DAG.getTargetConstantPool(C, MVT::i32,
                                  cast<ConstantPoolSDNode>(Op)->getAlignment());
    SDOperand Hi = DAG.getNode(SPISD::Hi, MVT::i32, CP);
    SDOperand Lo = DAG.getNode(SPISD::Lo, MVT::i32, CP);
    return DAG.getNode(ISD::ADD, MVT::i32, Lo, Hi);
  }
  case ISD::FP_TO_SINT:
    // Convert the fp value to integer in an FP register.
    assert(Op.getValueType() == MVT::i32);
    Op = DAG.getNode(SPISD::FTOI, MVT::f32, Op.getOperand(0));
    return DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Op);
  case ISD::SINT_TO_FP: {
    assert(Op.getOperand(0).getValueType() == MVT::i32);
    SDOperand Tmp = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, Op.getOperand(0));
    // Convert the int value to FP in an FP register.
    return DAG.getNode(SPISD::ITOF, Op.getValueType(), Tmp);
  }
  case ISD::BR_CC: {
    SDOperand Chain = Op.getOperand(0);
    ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
    SDOperand LHS = Op.getOperand(2);
    SDOperand RHS = Op.getOperand(3);
    SDOperand Dest = Op.getOperand(4);
    unsigned Opc, SPCC = ~0U;
    
    // If this is a br_cc of a "setcc", and if the setcc got lowered into
    // an CMP[IF]CC/SELECT_[IF]CC pair, find the original compared values.
    LookThroughSetCC(LHS, RHS, CC, SPCC);
    
    // Get the condition flag.
    SDOperand CompareFlag;
    if (LHS.getValueType() == MVT::i32) {
      std::vector<MVT::ValueType> VTs;
      VTs.push_back(MVT::i32);
      VTs.push_back(MVT::Flag);
      std::vector<SDOperand> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      CompareFlag = DAG.getNode(SPISD::CMPICC, VTs, Ops).getValue(1);
      if (SPCC == ~0U) SPCC = IntCondCCodeToICC(CC);
      Opc = SPISD::BRICC;
    } else {
      CompareFlag = DAG.getNode(SPISD::CMPFCC, MVT::Flag, LHS, RHS);
      if (SPCC == ~0U) SPCC = FPCondCCodeToFCC(CC);
      Opc = SPISD::BRFCC;
    }
    return DAG.getNode(Opc, MVT::Other, Chain, Dest,
                       DAG.getConstant(SPCC, MVT::i32), CompareFlag);
  }
  case ISD::SELECT_CC: {
    SDOperand LHS = Op.getOperand(0);
    SDOperand RHS = Op.getOperand(1);
    ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
    SDOperand TrueVal = Op.getOperand(2);
    SDOperand FalseVal = Op.getOperand(3);
    unsigned Opc, SPCC = ~0U;

    // If this is a select_cc of a "setcc", and if the setcc got lowered into
    // an CMP[IF]CC/SELECT_[IF]CC pair, find the original compared values.
    LookThroughSetCC(LHS, RHS, CC, SPCC);
    
    SDOperand CompareFlag;
    if (LHS.getValueType() == MVT::i32) {
      std::vector<MVT::ValueType> VTs;
      VTs.push_back(LHS.getValueType());   // subcc returns a value
      VTs.push_back(MVT::Flag);
      std::vector<SDOperand> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      CompareFlag = DAG.getNode(SPISD::CMPICC, VTs, Ops).getValue(1);
      Opc = SPISD::SELECT_ICC;
      if (SPCC == ~0U) SPCC = IntCondCCodeToICC(CC);
    } else {
      CompareFlag = DAG.getNode(SPISD::CMPFCC, MVT::Flag, LHS, RHS);
      Opc = SPISD::SELECT_FCC;
      if (SPCC == ~0U) SPCC = FPCondCCodeToFCC(CC);
    }
    return DAG.getNode(Opc, TrueVal.getValueType(), TrueVal, FalseVal, 
                       DAG.getConstant(SPCC, MVT::i32), CompareFlag);
  }
  case ISD::VASTART: {
    // vastart just stores the address of the VarArgsFrameIndex slot into the
    // memory location argument.
    SDOperand Offset = DAG.getNode(ISD::ADD, MVT::i32,
                                   DAG.getRegister(SP::I6, MVT::i32),
                                DAG.getConstant(VarArgsFrameOffset, MVT::i32));
    return DAG.getNode(ISD::STORE, MVT::Other, Op.getOperand(0), Offset, 
                       Op.getOperand(1), Op.getOperand(2));
  }
  case ISD::VAARG: {
    SDNode *Node = Op.Val;
    MVT::ValueType VT = Node->getValueType(0);
    SDOperand InChain = Node->getOperand(0);
    SDOperand VAListPtr = Node->getOperand(1);
    SDOperand VAList = DAG.getLoad(getPointerTy(), InChain, VAListPtr,
                                   Node->getOperand(2));
    // Increment the pointer, VAList, to the next vaarg
    SDOperand NextPtr = DAG.getNode(ISD::ADD, getPointerTy(), VAList, 
                                    DAG.getConstant(MVT::getSizeInBits(VT)/8, 
                                                    getPointerTy()));
    // Store the incremented VAList to the legalized pointer
    InChain = DAG.getNode(ISD::STORE, MVT::Other, VAList.getValue(1), NextPtr,
                          VAListPtr, Node->getOperand(2));
    // Load the actual argument out of the pointer VAList, unless this is an
    // f64 load.
    if (VT != MVT::f64) {
      return DAG.getLoad(VT, InChain, VAList, DAG.getSrcValue(0));
    } else {
      // Otherwise, load it as i64, then do a bitconvert.
      SDOperand V = DAG.getLoad(MVT::i64, InChain, VAList, DAG.getSrcValue(0));
      std::vector<MVT::ValueType> Tys;
      Tys.push_back(MVT::f64);
      Tys.push_back(MVT::Other);
      std::vector<SDOperand> Ops;
      // Bit-Convert the value to f64.
      Ops.push_back(DAG.getNode(ISD::BIT_CONVERT, MVT::f64, V));
      Ops.push_back(V.getValue(1));
      return DAG.getNode(ISD::MERGE_VALUES, Tys, Ops);
    }
  }
  case ISD::DYNAMIC_STACKALLOC: {
    SDOperand Chain = Op.getOperand(0);  // Legalize the chain.
    SDOperand Size  = Op.getOperand(1);  // Legalize the size.
    
    unsigned SPReg = SP::O6;
    SDOperand SP = DAG.getCopyFromReg(Chain, SPReg, MVT::i32);
    SDOperand NewSP = DAG.getNode(ISD::SUB, MVT::i32, SP, Size);    // Value
    Chain = DAG.getCopyToReg(SP.getValue(1), SPReg, NewSP);      // Output chain

    // The resultant pointer is actually 16 words from the bottom of the stack,
    // to provide a register spill area.
    SDOperand NewVal = DAG.getNode(ISD::ADD, MVT::i32, NewSP,
                                   DAG.getConstant(96, MVT::i32));
    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::i32);
    Tys.push_back(MVT::Other);
    std::vector<SDOperand> Ops;
    Ops.push_back(NewVal);
    Ops.push_back(Chain);
    return DAG.getNode(ISD::MERGE_VALUES, Tys, Ops);
  }
  case ISD::RET: {
    SDOperand Copy;
    
    switch(Op.getNumOperands()) {
    default:
      assert(0 && "Do not know how to return this many arguments!");
      abort();
    case 1: 
      return SDOperand(); // ret void is legal
    case 3: {
      unsigned ArgReg;
      switch(Op.getOperand(1).getValueType()) {
      default: assert(0 && "Unknown type to return!");
      case MVT::i32: ArgReg = SP::I0; break;
      case MVT::f32: ArgReg = SP::F0; break;
      case MVT::f64: ArgReg = SP::D0; break;
      }
      Copy = DAG.getCopyToReg(Op.getOperand(0), ArgReg, Op.getOperand(1),
                              SDOperand());
      break;
    }
    case 5:
      Copy = DAG.getCopyToReg(Op.getOperand(0), SP::I0, Op.getOperand(3), 
                              SDOperand());
      Copy = DAG.getCopyToReg(Copy, SP::I1, Op.getOperand(1), Copy.getValue(1));
      break;
    }
    return DAG.getNode(SPISD::RET_FLAG, MVT::Other, Copy, Copy.getValue(1));
  }
  }
}

MachineBasicBlock *
SparcTargetLowering::InsertAtEndOfBasicBlock(MachineInstr *MI,
                                             MachineBasicBlock *BB) {
  unsigned BROpcode;
  unsigned CC;
  // Figure out the conditional branch opcode to use for this select_cc.
  switch (MI->getOpcode()) {
  default: assert(0 && "Unknown SELECT_CC!");
  case SP::SELECT_CC_Int_ICC:
  case SP::SELECT_CC_FP_ICC:
  case SP::SELECT_CC_DFP_ICC:
    BROpcode = SP::BCOND;
    break;
  case SP::SELECT_CC_Int_FCC:
  case SP::SELECT_CC_FP_FCC:
  case SP::SELECT_CC_DFP_FCC:
    BROpcode = SP::FBCOND;
    break;
  }

  CC = (SPCC::CondCodes)MI->getOperand(3).getImmedValue();
  
  // To "insert" a SELECT_CC instruction, we actually have to insert the diamond
  // control-flow pattern.  The incoming instruction knows the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  ilist<MachineBasicBlock>::iterator It = BB;
  ++It;
  
  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   [f]bCC copy1MBB
  //   fallthrough --> copy0MBB
  MachineBasicBlock *thisMBB = BB;
  MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
  BuildMI(BB, BROpcode, 2).addMBB(sinkMBB).addImm(CC);
  MachineFunction *F = BB->getParent();
  F->getBasicBlockList().insert(It, copy0MBB);
  F->getBasicBlockList().insert(It, sinkMBB);
  // Update machine-CFG edges by first adding all successors of the current
  // block to the new block which will contain the Phi node for the select.
  for(MachineBasicBlock::succ_iterator i = BB->succ_begin(), 
      e = BB->succ_end(); i != e; ++i)
    sinkMBB->addSuccessor(*i);
  // Next, remove all successors of the current block, and add the true
  // and fallthrough blocks as its successors.
  while(!BB->succ_empty())
    BB->removeSuccessor(BB->succ_begin());
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
  BuildMI(BB, SP::PHI, 4, MI->getOperand(0).getReg())
    .addReg(MI->getOperand(2).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(1).getReg()).addMBB(thisMBB);
  
  delete MI;   // The pseudo instruction is gone now.
  return BB;
}
  
//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
/// SparcDAGToDAGISel - SPARC specific code to select SPARC machine
/// instructions for SelectionDAG operations.
///
namespace {
class SparcDAGToDAGISel : public SelectionDAGISel {
  SparcTargetLowering Lowering;

  /// Subtarget - Keep a pointer to the Sparc Subtarget around so that we can
  /// make the right decision when generating code for different targets.
  const SparcSubtarget &Subtarget;
public:
  SparcDAGToDAGISel(TargetMachine &TM)
    : SelectionDAGISel(Lowering), Lowering(TM),
      Subtarget(TM.getSubtarget<SparcSubtarget>()) {
  }

  void Select(SDOperand &Result, SDOperand Op);

  // Complex Pattern Selectors.
  bool SelectADDRrr(SDOperand N, SDOperand &R1, SDOperand &R2);
  bool SelectADDRri(SDOperand N, SDOperand &Base, SDOperand &Offset);
  
  /// InstructionSelectBasicBlock - This callback is invoked by
  /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);
  
  virtual const char *getPassName() const {
    return "SPARC DAG->DAG Pattern Instruction Selection";
  } 
  
  // Include the pieces autogenerated from the target description.
#include "SparcGenDAGISel.inc"
};
}  // end anonymous namespace

/// InstructionSelectBasicBlock - This callback is invoked by
/// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
void SparcDAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());
  
  // Select target instructions for the DAG.
  DAG.setRoot(SelectRoot(DAG.getRoot()));
  DAG.RemoveDeadNodes();
  
  // Emit machine code to BB. 
  ScheduleAndEmitDAG(DAG);
}

bool SparcDAGToDAGISel::SelectADDRri(SDOperand Addr, SDOperand &Base,
                                     SDOperand &Offset) {
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i32);
    Offset = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }
  if (Addr.getOpcode() == ISD::TargetExternalSymbol ||
      Addr.getOpcode() == ISD::TargetGlobalAddress)
    return false;  // direct calls.
  
  if (Addr.getOpcode() == ISD::ADD) {
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1))) {
      if (Predicate_simm13(CN)) {
        if (FrameIndexSDNode *FIN = 
                dyn_cast<FrameIndexSDNode>(Addr.getOperand(0))) {
          // Constant offset from frame ref.
          Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i32);
        } else {
          Base = Addr.getOperand(0);
        }
        Offset = CurDAG->getTargetConstant(CN->getValue(), MVT::i32);
        return true;
      }
    }
    if (Addr.getOperand(0).getOpcode() == SPISD::Lo) {
      Base = Addr.getOperand(1);
      Offset = Addr.getOperand(0).getOperand(0);
      return true;
    }
    if (Addr.getOperand(1).getOpcode() == SPISD::Lo) {
      Base = Addr.getOperand(0);
      Offset = Addr.getOperand(1).getOperand(0);
      return true;
    }
  }
  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, MVT::i32);
  return true;
}

bool SparcDAGToDAGISel::SelectADDRrr(SDOperand Addr, SDOperand &R1, 
                                     SDOperand &R2) {
  if (Addr.getOpcode() == ISD::FrameIndex) return false;
  if (Addr.getOpcode() == ISD::TargetExternalSymbol ||
      Addr.getOpcode() == ISD::TargetGlobalAddress)
    return false;  // direct calls.
  
  if (Addr.getOpcode() == ISD::ADD) {
    if (isa<ConstantSDNode>(Addr.getOperand(1)) &&
        Predicate_simm13(Addr.getOperand(1).Val))
      return false;  // Let the reg+imm pattern catch this!
    if (Addr.getOperand(0).getOpcode() == SPISD::Lo ||
        Addr.getOperand(1).getOpcode() == SPISD::Lo)
      return false;  // Let the reg+imm pattern catch this!
    R1 = Addr.getOperand(0);
    R2 = Addr.getOperand(1);
    return true;
  }

  R1 = Addr;
  R2 = CurDAG->getRegister(SP::G0, MVT::i32);
  return true;
}

void SparcDAGToDAGISel::Select(SDOperand &Result, SDOperand Op) {
  SDNode *N = Op.Val;
  if (N->getOpcode() >= ISD::BUILTIN_OP_END &&
      N->getOpcode() < SPISD::FIRST_NUMBER) {
    Result = Op;
    return;   // Already selected.
  }

  switch (N->getOpcode()) {
  default: break;
  case ISD::SDIV:
  case ISD::UDIV: {
    // FIXME: should use a custom expander to expose the SRA to the dag.
    SDOperand DivLHS, DivRHS;
    AddToQueue(DivLHS, N->getOperand(0));
    AddToQueue(DivRHS, N->getOperand(1));
    
    // Set the Y register to the high-part.
    SDOperand TopPart;
    if (N->getOpcode() == ISD::SDIV) {
      TopPart = SDOperand(CurDAG->getTargetNode(SP::SRAri, MVT::i32, DivLHS,
                                   CurDAG->getTargetConstant(31, MVT::i32)), 0);
    } else {
      TopPart = CurDAG->getRegister(SP::G0, MVT::i32);
    }
    TopPart = SDOperand(CurDAG->getTargetNode(SP::WRYrr, MVT::Flag, TopPart,
                                     CurDAG->getRegister(SP::G0, MVT::i32)), 0);

    // FIXME: Handle div by immediate.
    unsigned Opcode = N->getOpcode() == ISD::SDIV ? SP::SDIVrr : SP::UDIVrr;
    Result = CurDAG->SelectNodeTo(N, Opcode, MVT::i32, DivLHS, DivRHS, TopPart);
    return;
  }    
  case ISD::MULHU:
  case ISD::MULHS: {
    // FIXME: Handle mul by immediate.
    SDOperand MulLHS, MulRHS;
    AddToQueue(MulLHS, N->getOperand(0));
    AddToQueue(MulRHS, N->getOperand(1));
    unsigned Opcode = N->getOpcode() == ISD::MULHU ? SP::UMULrr : SP::SMULrr;
    SDNode *Mul = CurDAG->getTargetNode(Opcode, MVT::i32, MVT::Flag,
                                        MulLHS, MulRHS);
    // The high part is in the Y register.
    Result = CurDAG->SelectNodeTo(N, SP::RDY, MVT::i32, SDOperand(Mul, 1));
    return;
  }
  }
  
  SelectCode(Result, Op);
}


/// createSparcISelDag - This pass converts a legalized DAG into a 
/// SPARC-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createSparcISelDag(TargetMachine &TM) {
  return new SparcDAGToDAGISel(TM);
}
