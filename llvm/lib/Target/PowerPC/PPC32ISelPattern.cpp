//===-- PPC32ISelPattern.cpp - A pattern matching inst selector for PPC32 -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for 32 bit PowerPC.
// Magic number generation for integer divide from the PowerPC Compiler Writer's
// Guide, section 3.2.3.5
//
//===----------------------------------------------------------------------===//

#include "PowerPC.h"
#include "PowerPCInstrBuilder.h"
#include "PowerPCInstrInfo.h"
#include "PPC32TargetMachine.h"
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
  class PPC32TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    int ReturnAddrIndex;              // FrameIndex for return slot.
  public:
    PPC32TargetLowering(TargetMachine &TM) : TargetLowering(TM) {
      // Set up the register classes.
      addRegisterClass(MVT::i32, PPC32::GPRCRegisterClass);
      addRegisterClass(MVT::f32, PPC32::FPRCRegisterClass);
      addRegisterClass(MVT::f64, PPC32::FPRCRegisterClass);

      // PowerPC has no intrinsics for these particular operations
      setOperationAction(ISD::MEMMOVE, MVT::Other, Expand);
      setOperationAction(ISD::MEMSET, MVT::Other, Expand);
      setOperationAction(ISD::MEMCPY, MVT::Other, Expand);

      // PowerPC has an i16 but no i8 (or i1) SEXTLOAD
      setOperationAction(ISD::SEXTLOAD, MVT::i1, Expand);
      setOperationAction(ISD::SEXTLOAD, MVT::i8, Expand);

      // PowerPC has no SREM/UREM instructions
      setOperationAction(ISD::SREM, MVT::i32, Expand);
      setOperationAction(ISD::UREM, MVT::i32, Expand);

      // We don't support sin/cos/sqrt
      setOperationAction(ISD::FSIN , MVT::f64, Expand);
      setOperationAction(ISD::FCOS , MVT::f64, Expand);
      setOperationAction(ISD::FSQRT, MVT::f64, Expand);
      setOperationAction(ISD::FSIN , MVT::f32, Expand);
      setOperationAction(ISD::FCOS , MVT::f32, Expand);
      setOperationAction(ISD::FSQRT, MVT::f32, Expand);

      //PowerPC has these, but they are not implemented
      setOperationAction(ISD::CTPOP, MVT::i32  , Expand);
      setOperationAction(ISD::CTTZ , MVT::i32  , Expand);
      setOperationAction(ISD::CTTZ , MVT::i32  , Expand);

      setSetCCResultContents(ZeroOrOneSetCCResult);
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
    LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg,
                SDOperand Callee, ArgListTy &Args, SelectionDAG &DAG);

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
PPC32TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
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
  unsigned ArgOffset = 24;
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
  // the arguments start at offset 24, although they are likely to be passed
  // in registers.
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I) {
    SDOperand newroot, argt;
    unsigned ObjSize;
    bool needsLoad = false;
    bool ArgLive = !I->use_empty();
    MVT::ValueType ObjectVT = getValueType(I->getType());

    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      ObjSize = 4;
      if (!ArgLive) break;
      if (GPR_remaining > 0) {
        MF.addLiveIn(GPR[GPR_idx]);
        argt = newroot = DAG.getCopyFromReg(GPR[GPR_idx], MVT::i32,
                                            DAG.getRoot());
        if (ObjectVT != MVT::i32)
          argt = DAG.getNode(ISD::TRUNCATE, ObjectVT, newroot);
      } else {
        needsLoad = true;
      }
      break;
      case MVT::i64: ObjSize = 8;
      if (!ArgLive) break;
      if (GPR_remaining > 0) {
        SDOperand argHi, argLo;
        MF.addLiveIn(GPR[GPR_idx]);
        argHi = DAG.getCopyFromReg(GPR[GPR_idx], MVT::i32, DAG.getRoot());
        // If we have two or more remaining argument registers, then both halves
        // of the i64 can be sourced from there.  Otherwise, the lower half will
        // have to come off the stack.  This can happen when an i64 is preceded
        // by 28 bytes of arguments.
        if (GPR_remaining > 1) {
          MF.addLiveIn(GPR[GPR_idx+1]);
          argLo = DAG.getCopyFromReg(GPR[GPR_idx+1], MVT::i32, argHi);
        } else {
          int FI = MFI->CreateFixedObject(4, ArgOffset+4);
          SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);
          argLo = DAG.getLoad(MVT::i32, DAG.getEntryNode(), FIN, DAG.getSrcValue(NULL));
        }
        // Build the outgoing arg thingy
        argt = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, argLo, argHi);
        newroot = argLo;
      } else {
        needsLoad = true;
      }
      break;
      case MVT::f32:
      case MVT::f64:
      ObjSize = (ObjectVT == MVT::f64) ? 8 : 4;
      if (!ArgLive) break;
      if (FPR_remaining > 0) {
        MF.addLiveIn(FPR[FPR_idx]);
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
      if (ObjectVT == MVT::i8 || ObjectVT == MVT::i1) SubregOffset = 3;
      if (ObjectVT == MVT::i16) SubregOffset = 2;
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);
      FIN = DAG.getNode(ISD::ADD, MVT::i32, FIN,
                        DAG.getConstant(SubregOffset, MVT::i32));
      argt = newroot = DAG.getLoad(ObjectVT, DAG.getEntryNode(), FIN, DAG.getSrcValue(NULL));
    }

    // Every 4 bytes of argument space consumes one of the GPRs available for
    // argument passing.
    if (GPR_remaining > 0) {
      unsigned delta = (GPR_remaining > 1 && ObjSize == 8) ? 2 : 1;
      GPR_remaining -= delta;
      GPR_idx += delta;
    }
    ArgOffset += ObjSize;
    if (newroot.Val)
      DAG.setRoot(newroot.getValue(1));

    ArgValues.push_back(argt);
  }

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (F.isVarArg()) {
    VarArgsFrameIndex = MFI->CreateFixedObject(4, ArgOffset);
    SDOperand FIN = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i32);
    // If this function is vararg, store any remaining integer argument regs
    // to their spots on the stack so that they may be loaded by deferencing the
    // result of va_next.
    std::vector<SDOperand> MemOps;
    for (; GPR_remaining > 0; --GPR_remaining, ++GPR_idx) {
      MF.addLiveIn(GPR[GPR_idx]);
      SDOperand Val = DAG.getCopyFromReg(GPR[GPR_idx], MVT::i32, DAG.getRoot());
      SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Val.getValue(1),
                                    Val, FIN, DAG.getSrcValue(NULL));
      MemOps.push_back(Store);
      // Increment the address by four for the next argument to store
      SDOperand PtrOff = DAG.getConstant(4, getPointerTy());
      FIN = DAG.getNode(ISD::ADD, MVT::i32, FIN, PtrOff);
    }
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other, MemOps));
  }

  // Finally, inform the code generator which regs we return values in.
  switch (getValueType(F.getReturnType())) {
  default: assert(0 && "Unknown type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    MF.addLiveOut(PPC::R3);
    break;
  case MVT::i64:
    MF.addLiveOut(PPC::R3);
    MF.addLiveOut(PPC::R4);
    break;
  case MVT::f32:
  case MVT::f64:
    MF.addLiveOut(PPC::F1);
    break;
  }

  return ArgValues;
}

std::pair<SDOperand, SDOperand>
PPC32TargetLowering::LowerCallTo(SDOperand Chain,
                                 const Type *RetTy, bool isVarArg,
                                 SDOperand Callee, ArgListTy &Args,
                                 SelectionDAG &DAG) {
  // args_to_use will accumulate outgoing args for the ISD::CALL case in
  // SelectExpr to use to put the arguments in the appropriate registers.
  std::vector<SDOperand> args_to_use;

  // Count how many bytes are to be pushed on the stack, including the linkage
  // area, and parameter passing area.
  unsigned NumBytes = 24;

  if (Args.empty()) {
    Chain = DAG.getNode(ISD::ADJCALLSTACKDOWN, MVT::Other, Chain,
                        DAG.getConstant(NumBytes, getPointerTy()));
  } else {
    for (unsigned i = 0, e = Args.size(); i != e; ++i)
      switch (getValueType(Args[i].second)) {
      default: assert(0 && "Unknown value type!");
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
      case MVT::f32:
        NumBytes += 4;
        break;
      case MVT::i64:
      case MVT::f64:
        NumBytes += 8;
        break;
      }

    // Just to be safe, we'll always reserve the full 24 bytes of linkage area
    // plus 32 bytes of argument space in case any called code gets funky on us.
    if (NumBytes < 56) NumBytes = 56;

    // Adjust the stack pointer for the new arguments...
    // These operations are automatically eliminated by the prolog/epilog pass
    Chain = DAG.getNode(ISD::ADJCALLSTACKDOWN, MVT::Other, Chain,
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
    unsigned ArgOffset = 24;
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
        // Promote the integer to 32 bits.  If the input type is signed use a
        // sign extend, otherwise use a zero extend.
        if (Args[i].second->isSigned())
          Args[i].first =DAG.getNode(ISD::SIGN_EXTEND, MVT::i32, Args[i].first);
        else
          Args[i].first =DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, Args[i].first);
        // FALL THROUGH
      case MVT::i32:
        if (GPR_remaining > 0) {
          args_to_use.push_back(Args[i].first);
          --GPR_remaining;
        } else {
          MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                       Args[i].first, PtrOff, DAG.getSrcValue(NULL)));
        }
        ArgOffset += 4;
        break;
      case MVT::i64:
        // If we have one free GPR left, we can place the upper half of the i64
        // in it, and store the other half to the stack.  If we have two or more
        // free GPRs, then we can pass both halves of the i64 in registers.
        if (GPR_remaining > 0) {
          SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32,
            Args[i].first, DAG.getConstant(1, MVT::i32));
          SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32,
            Args[i].first, DAG.getConstant(0, MVT::i32));
          args_to_use.push_back(Hi);
          --GPR_remaining;
          if (GPR_remaining > 0) {
            args_to_use.push_back(Lo);
            --GPR_remaining;
          } else {
            SDOperand ConstFour = DAG.getConstant(4, getPointerTy());
            PtrOff = DAG.getNode(ISD::ADD, MVT::i32, PtrOff, ConstFour);
            MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                         Lo, PtrOff, DAG.getSrcValue(NULL)));
          }
        } else {
          MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                       Args[i].first, PtrOff, DAG.getSrcValue(NULL)));
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
                                          Args[i].first, PtrOff, DAG.getSrcValue(NULL));
            MemOps.push_back(Store);
            // Float varargs are always shadowed in available integer registers
            if (GPR_remaining > 0) {
              SDOperand Load = DAG.getLoad(MVT::i32, Store, PtrOff, DAG.getSrcValue(NULL));
              MemOps.push_back(Load);
              args_to_use.push_back(Load);
              --GPR_remaining;
            }
            if (GPR_remaining > 0 && MVT::f64 == ArgVT) {
              SDOperand ConstFour = DAG.getConstant(4, getPointerTy());
              PtrOff = DAG.getNode(ISD::ADD, MVT::i32, PtrOff, ConstFour);
              SDOperand Load = DAG.getLoad(MVT::i32, Store, PtrOff, DAG.getSrcValue(NULL));
              MemOps.push_back(Load);
              args_to_use.push_back(Load);
              --GPR_remaining;
            }
          } else {
            // If we have any FPRs remaining, we may also have GPRs remaining.
            // Args passed in FPRs consume either 1 (f32) or 2 (f64) available
            // GPRs.
            if (GPR_remaining > 0) {
              args_to_use.push_back(DAG.getNode(ISD::UNDEF, MVT::i32));
              --GPR_remaining;
            }
            if (GPR_remaining > 0 && MVT::f64 == ArgVT) {
              args_to_use.push_back(DAG.getNode(ISD::UNDEF, MVT::i32));
              --GPR_remaining;
            }
          }
        } else {
          MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                       Args[i].first, PtrOff, DAG.getSrcValue(NULL)));
        }
        ArgOffset += (ArgVT == MVT::f32) ? 4 : 8;
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
  Chain = DAG.getNode(ISD::ADJCALLSTACKUP, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  return std::make_pair(TheCall, Chain);
}

std::pair<SDOperand, SDOperand>
PPC32TargetLowering::LowerVAStart(SDOperand Chain, SelectionDAG &DAG) {
  //vastart just returns the address of the VarArgsFrameIndex slot.
  return std::make_pair(DAG.getFrameIndex(VarArgsFrameIndex, MVT::i32), Chain);
}

std::pair<SDOperand,SDOperand> PPC32TargetLowering::
LowerVAArgNext(bool isVANext, SDOperand Chain, SDOperand VAList,
               const Type *ArgTy, SelectionDAG &DAG) {
  MVT::ValueType ArgVT = getValueType(ArgTy);
  SDOperand Result;
  if (!isVANext) {
    Result = DAG.getLoad(ArgVT, DAG.getEntryNode(), VAList, DAG.getSrcValue(NULL));
  } else {
    unsigned Amt;
    if (ArgVT == MVT::i32 || ArgVT == MVT::f32)
      Amt = 4;
    else {
      assert((ArgVT == MVT::i64 || ArgVT == MVT::f64) &&
             "Other types should have been promoted for varargs!");
      Amt = 8;
    }
    Result = DAG.getNode(ISD::ADD, VAList.getValueType(), VAList,
                         DAG.getConstant(Amt, VAList.getValueType()));
  }
  return std::make_pair(Result, Chain);
}


std::pair<SDOperand, SDOperand> PPC32TargetLowering::
LowerFrameReturnAddress(bool isFrameAddress, SDOperand Chain, unsigned Depth,
                        SelectionDAG &DAG) {
  assert(0 && "LowerFrameReturnAddress unimplemented");
  abort();
}

namespace {
Statistic<>Recorded("ppc-codegen", "Number of recording ops emitted");
Statistic<>FusedFP("ppc-codegen", "Number of fused fp operations");
Statistic<>MultiBranch("ppc-codegen", "Number of setcc logical ops collapsed");
//===--------------------------------------------------------------------===//
/// ISel - PPC32 specific code to select PPC32 machine instructions for
/// SelectionDAG operations.
//===--------------------------------------------------------------------===//
class ISel : public SelectionDAGISel {
  PPC32TargetLowering PPC32Lowering;
  SelectionDAG *ISelDAG;  // Hack to support us having a dag->dag transform
                          // for sdiv and udiv until it is put into the future
                          // dag combiner.

  /// ExprMap - As shared expressions are codegen'd, we keep track of which
  /// vreg the value is produced in, so we only emit one copy of each compiled
  /// tree.
  std::map<SDOperand, unsigned> ExprMap;

  unsigned GlobalBaseReg;
  bool GlobalBaseInitialized;
  bool RecordSuccess;
public:
  ISel(TargetMachine &TM) : SelectionDAGISel(PPC32Lowering), PPC32Lowering(TM),
                            ISelDAG(0) {}

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
    ISelDAG = &DAG;
    Select(DAG.getRoot());

    // Clear state used for selection.
    ExprMap.clear();
    ISelDAG = 0;
  }

  // dag -> dag expanders for integer divide by constant
  SDOperand BuildSDIVSequence(SDOperand N);
  SDOperand BuildUDIVSequence(SDOperand N);

  unsigned getGlobalBaseReg();
  unsigned getConstDouble(double floatVal, unsigned Result);
  void MoveCRtoGPR(unsigned CCReg, bool Inv, unsigned Idx, unsigned Result);
  bool SelectBitfieldInsert(SDOperand OR, unsigned Result);
  unsigned FoldIfWideZeroExtend(SDOperand N);
  unsigned SelectCC(SDOperand CC, unsigned &Opc, bool &Inv, unsigned &Idx);
  unsigned SelectCCExpr(SDOperand N, unsigned& Opc, bool &Inv, unsigned &Idx);
  unsigned SelectExpr(SDOperand N, bool Recording=false);
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

// IsRunOfOnes - returns true if Val consists of one contiguous run of 1's with
// any number of 0's on either side.  the 1's are allowed to wrap from LSB to
// MSB.  so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.  0x0F0F0000 is
// not, since all 1's are not contiguous.
static bool IsRunOfOnes(unsigned Val, unsigned &MB, unsigned &ME) {
  bool isRun = true;
  MB = 0;
  ME = 0;

  // look for first set bit
  int i = 0;
  for (; i < 32; i++) {
    if ((Val & (1 << (31 - i))) != 0) {
      MB = i;
      ME = i;
      break;
    }
  }

  // look for last set bit
  for (; i < 32; i++) {
    if ((Val & (1 << (31 - i))) == 0)
      break;
    ME = i;
  }

  // look for next set bit
  for (; i < 32; i++) {
    if ((Val & (1 << (31 - i))) != 0)
      break;
  }

  // if we exhausted all the bits, we found a match at this point for 0*1*0*
  if (i == 32)
    return true;

  // since we just encountered more 1's, if it doesn't wrap around to the
  // most significant bit of the word, then we did not find a match to 1*0*1* so
  // exit.
  if (MB != 0)
    return false;

  // look for last set bit
  for (MB = i; i < 32; i++) {
    if ((Val & (1 << (31 - i))) == 0)
      break;
  }

  // if we exhausted all the bits, then we found a match for 1*0*1*, otherwise,
  // the value is not a run of ones.
  if (i == 32)
    return true;
  return false;
}

/// getImmediateForOpcode - This method returns a value indicating whether
/// the ConstantSDNode N can be used as an immediate to Opcode.  The return
/// values are either 0, 1 or 2.  0 indicates that either N is not a
/// ConstantSDNode, or is not suitable for use by that opcode.
/// Return value codes for turning into an enum someday:
/// 1: constant may be used in normal immediate form.
/// 2: constant may be used in shifted immediate form.
/// 3: log base 2 of the constant may be used.
/// 4: constant is suitable for integer division conversion
/// 5: constant is a bitfield mask
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
  case ISD::AND: {
    unsigned MB, ME;
    if (IsRunOfOnes(v, MB, ME)) { Imm = MB << 16 | ME & 0xFFFF; return 5; }
    if (v >= 0 && v <= 65535) { Imm = v & 0xFFFF; return 1; }
    if ((v & 0x0000FFFF) == 0) { Imm = v >> 16; return 2; }
    break;
  }
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
    if ((Imm = ExactLog2(-v))) { Imm = -Imm; return 3; }
    if (v <= -2 || v >= 2) { return 4; }
    break;
  case ISD::UDIV:
    if (v > 1) { return 4; }
    break;
  }
  return 0;
}

/// NodeHasRecordingVariant - If SelectExpr can always produce code for
/// NodeOpcode that also sets CR0 as a side effect, return true.  Otherwise,
/// return false.
static bool NodeHasRecordingVariant(unsigned NodeOpcode) {
  switch(NodeOpcode) {
  default: return false;
  case ISD::AND:
  case ISD::OR:
    return true;
  }
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

/// getCROpForOp - Return the condition register opcode (or inverted opcode)
/// associated with the SelectionDAG opcode.
static unsigned getCROpForSetCC(unsigned Opcode, bool Inv1, bool Inv2) {
  switch (Opcode) {
  default: assert(0 && "Unknown opcode!"); abort();
  case ISD::AND:
    if (Inv1 && Inv2) return PPC::CRNOR; // De Morgan's Law
    if (!Inv1 && !Inv2) return PPC::CRAND;
    if (Inv1 ^ Inv2) return PPC::CRANDC;
  case ISD::OR:
    if (Inv1 && Inv2) return PPC::CRNAND; // De Morgan's Law
    if (!Inv1 && !Inv2) return PPC::CROR;
    if (Inv1 ^ Inv2) return PPC::CRORC;
  }
  return 0;
}

/// getCRIdxForSetCC - Return the index of the condition register field
/// associated with the SetCC condition, and whether or not the field is
/// treated as inverted.  That is, lt = 0; ge = 0 inverted.
static unsigned getCRIdxForSetCC(unsigned Condition, bool& Inv) {
  switch (Condition) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETULT:
  case ISD::SETLT:  Inv = false;  return 0;
  case ISD::SETUGE:
  case ISD::SETGE:  Inv = true;   return 0;
  case ISD::SETUGT:
  case ISD::SETGT:  Inv = false;  return 1;
  case ISD::SETULE:
  case ISD::SETLE:  Inv = true;   return 1;
  case ISD::SETEQ:  Inv = false;  return 2;
  case ISD::SETNE:  Inv = true;   return 2;
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
  case PPC::LWZ: return PPC::LWZX;  case PPC::STFS: return PPC::STFSX;
  case PPC::LFS: return PPC::LFSX;  case PPC::STFD: return PPC::STFDX;
  case PPC::LFD: return PPC::LFDX;
  }
  return 0;
}

// Structure used to return the necessary information to codegen an SDIV as
// a multiply.
struct ms {
  int m; // magic number
  int s; // shift amount
};

struct mu {
  unsigned int m; // magic number
  int a;          // add indicator
  int s;          // shift amount
};

/// magic - calculate the magic numbers required to codegen an integer sdiv as
/// a sequence of multiply and shifts.  Requires that the divisor not be 0, 1,
/// or -1.
static struct ms magic(int d) {
  int p;
  unsigned int ad, anc, delta, q1, r1, q2, r2, t;
  const unsigned int two31 = 2147483648U; // 2^31
  struct ms mag;

  ad = abs(d);
  t = two31 + ((unsigned int)d >> 31);
  anc = t - 1 - t%ad;   // absolute value of nc
  p = 31;               // initialize p
  q1 = two31/anc;       // initialize q1 = 2p/abs(nc)
  r1 = two31 - q1*anc;  // initialize r1 = rem(2p,abs(nc))
  q2 = two31/ad;        // initialize q2 = 2p/abs(d)
  r2 = two31 - q2*ad;   // initialize r2 = rem(2p,abs(d))
  do {
    p = p + 1;
    q1 = 2*q1;        // update q1 = 2p/abs(nc)
    r1 = 2*r1;        // update r1 = rem(2p/abs(nc))
    if (r1 >= anc) {  // must be unsigned comparison
      q1 = q1 + 1;
      r1 = r1 - anc;
    }
    q2 = 2*q2;        // update q2 = 2p/abs(d)
    r2 = 2*r2;        // update r2 = rem(2p/abs(d))
    if (r2 >= ad) {   // must be unsigned comparison
      q2 = q2 + 1;
      r2 = r2 - ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));

  mag.m = q2 + 1;
  if (d < 0) mag.m = -mag.m; // resulting magic number
  mag.s = p - 32;            // resulting shift
  return mag;
}

/// magicu - calculate the magic numbers required to codegen an integer udiv as
/// a sequence of multiply, add and shifts.  Requires that the divisor not be 0.
static struct mu magicu(unsigned d)
{
  int p;
  unsigned int nc, delta, q1, r1, q2, r2;
  struct mu magu;
  magu.a = 0;               // initialize "add" indicator
  nc = - 1 - (-d)%d;
  p = 31;                   // initialize p
  q1 = 0x80000000/nc;       // initialize q1 = 2p/nc
  r1 = 0x80000000 - q1*nc;  // initialize r1 = rem(2p,nc)
  q2 = 0x7FFFFFFF/d;        // initialize q2 = (2p-1)/d
  r2 = 0x7FFFFFFF - q2*d;   // initialize r2 = rem((2p-1),d)
  do {
    p = p + 1;
    if (r1 >= nc - r1 ) {
      q1 = 2*q1 + 1;  // update q1
      r1 = 2*r1 - nc; // update r1
    }
    else {
      q1 = 2*q1; // update q1
      r1 = 2*r1; // update r1
    }
    if (r2 + 1 >= d - r2) {
      if (q2 >= 0x7FFFFFFF) magu.a = 1;
      q2 = 2*q2 + 1;     // update q2
      r2 = 2*r2 + 1 - d; // update r2
    }
    else {
      if (q2 >= 0x80000000) magu.a = 1;
      q2 = 2*q2;     // update q2
      r2 = 2*r2 + 1; // update r2
    }
    delta = d - 1 - r2;
  } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
  magu.m = q2 + 1; // resulting magic number
  magu.s = p - 32;  // resulting shift
  return magu;
}
}

/// BuildSDIVSequence - Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand ISel::BuildSDIVSequence(SDOperand N) {
  int d = (int)cast<ConstantSDNode>(N.getOperand(1))->getSignExtended();
  ms magics = magic(d);
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = ISelDAG->getNode(ISD::MULHS, MVT::i32, N.getOperand(0),
                                 ISelDAG->getConstant(magics.m, MVT::i32));
  // If d > 0 and m < 0, add the numerator
  if (d > 0 && magics.m < 0)
    Q = ISelDAG->getNode(ISD::ADD, MVT::i32, Q, N.getOperand(0));
  // If d < 0 and m > 0, subtract the numerator.
  if (d < 0 && magics.m > 0)
    Q = ISelDAG->getNode(ISD::SUB, MVT::i32, Q, N.getOperand(0));
  // Shift right algebraic if shift value is nonzero
  if (magics.s > 0)
    Q = ISelDAG->getNode(ISD::SRA, MVT::i32, Q,
                         ISelDAG->getConstant(magics.s, MVT::i32));
  // Extract the sign bit and add it to the quotient
  SDOperand T =
    ISelDAG->getNode(ISD::SRL, MVT::i32, Q, ISelDAG->getConstant(31, MVT::i32));
  return ISelDAG->getNode(ISD::ADD, MVT::i32, Q, T);
}

/// BuildUDIVSequence - Given an ISD::UDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand ISel::BuildUDIVSequence(SDOperand N) {
  unsigned d =
    (unsigned)cast<ConstantSDNode>(N.getOperand(1))->getSignExtended();
  mu magics = magicu(d);
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = ISelDAG->getNode(ISD::MULHU, MVT::i32, N.getOperand(0),
                                 ISelDAG->getConstant(magics.m, MVT::i32));
  if (magics.a == 0) {
    Q = ISelDAG->getNode(ISD::SRL, MVT::i32, Q,
                         ISelDAG->getConstant(magics.s, MVT::i32));
  } else {
    SDOperand NPQ = ISelDAG->getNode(ISD::SUB, MVT::i32, N.getOperand(0), Q);
    NPQ = ISelDAG->getNode(ISD::SRL, MVT::i32, NPQ,
                           ISelDAG->getConstant(1, MVT::i32));
    NPQ = ISelDAG->getNode(ISD::ADD, MVT::i32, NPQ, Q);
    Q = ISelDAG->getNode(ISD::SRL, MVT::i32, NPQ,
                           ISelDAG->getConstant(magics.s-1, MVT::i32));
  }
  return Q;
}

/// getGlobalBaseReg - Output the instructions required to put the
/// base address to use for accessing globals into a register.
///
unsigned ISel::getGlobalBaseReg() {
  if (!GlobalBaseInitialized) {
    // Insert the set of GlobalBaseReg into the first MBB of the function
    MachineBasicBlock &FirstMBB = BB->getParent()->front();
    MachineBasicBlock::iterator MBBI = FirstMBB.begin();
    GlobalBaseReg = MakeReg(MVT::i32);
    BuildMI(FirstMBB, MBBI, PPC::MovePCtoLR, 0, PPC::LR);
    BuildMI(FirstMBB, MBBI, PPC::MFLR, 1, GlobalBaseReg).addReg(PPC::LR);
    GlobalBaseInitialized = true;
  }
  return GlobalBaseReg;
}

/// getConstDouble - Loads a floating point value into a register, via the
/// Constant Pool.  Optionally takes a register in which to load the value.
unsigned ISel::getConstDouble(double doubleVal, unsigned Result=0) {
  unsigned Tmp1 = MakeReg(MVT::i32);
  if (0 == Result) Result = MakeReg(MVT::f64);
  MachineConstantPool *CP = BB->getParent()->getConstantPool();
  ConstantFP *CFP = ConstantFP::get(Type::DoubleTy, doubleVal);
  unsigned CPI = CP->getConstantPoolIndex(CFP);
  BuildMI(BB, PPC::LOADHiAddr, 2, Tmp1).addReg(getGlobalBaseReg())
    .addConstantPoolIndex(CPI);
  BuildMI(BB, PPC::LFD, 2, Result).addConstantPoolIndex(CPI).addReg(Tmp1);
  return Result;
}

/// MoveCRtoGPR - Move CCReg[Idx] to the least significant bit of Result.  If
/// Inv is true, then invert the result.
void ISel::MoveCRtoGPR(unsigned CCReg, bool Inv, unsigned Idx, unsigned Result){
  unsigned IntCR = MakeReg(MVT::i32);
  BuildMI(BB, PPC::MCRF, 1, PPC::CR7).addReg(CCReg);
  BuildMI(BB, PPC::MFCR, 1, IntCR).addReg(PPC::CR7);
  if (Inv) {
    unsigned Tmp1 = MakeReg(MVT::i32);
    BuildMI(BB, PPC::RLWINM, 4, Tmp1).addReg(IntCR).addImm(32-(3-Idx))
      .addImm(31).addImm(31);
    BuildMI(BB, PPC::XORI, 2, Result).addReg(Tmp1).addImm(1);
  } else {
    BuildMI(BB, PPC::RLWINM, 4, Result).addReg(IntCR).addImm(32-(3-Idx))
      .addImm(31).addImm(31);
  }
}

/// SelectBitfieldInsert - turn an or of two masked values into
/// the rotate left word immediate then mask insert (rlwimi) instruction.
/// Returns true on success, false if the caller still needs to select OR.
///
/// Patterns matched:
/// 1. or shl, and   5. or and, and
/// 2. or and, shl   6. or shl, shr
/// 3. or shr, and   7. or shr, shl
/// 4. or and, shr
bool ISel::SelectBitfieldInsert(SDOperand OR, unsigned Result) {
  bool IsRotate = false;
  unsigned TgtMask = 0xFFFFFFFF, InsMask = 0xFFFFFFFF, Amount = 0;
  unsigned Op0Opc = OR.getOperand(0).getOpcode();
  unsigned Op1Opc = OR.getOperand(1).getOpcode();

  // Verify that we have the correct opcodes
  if (ISD::SHL != Op0Opc && ISD::SRL != Op0Opc && ISD::AND != Op0Opc)
    return false;
  if (ISD::SHL != Op1Opc && ISD::SRL != Op1Opc && ISD::AND != Op1Opc)
    return false;

  // Generate Mask value for Target
  if (ConstantSDNode *CN =
      dyn_cast<ConstantSDNode>(OR.getOperand(0).getOperand(1).Val)) {
    switch(Op0Opc) {
    case ISD::SHL: TgtMask <<= (unsigned)CN->getValue(); break;
    case ISD::SRL: TgtMask >>= (unsigned)CN->getValue(); break;
    case ISD::AND: TgtMask &= (unsigned)CN->getValue(); break;
    }
  } else {
    return false;
  }

  // Generate Mask value for Insert
  if (ConstantSDNode *CN =
      dyn_cast<ConstantSDNode>(OR.getOperand(1).getOperand(1).Val)) {
    switch(Op1Opc) {
    case ISD::SHL:
      Amount = CN->getValue();
      InsMask <<= Amount;
      if (Op0Opc == ISD::SRL) IsRotate = true;
      break;
    case ISD::SRL:
      Amount = CN->getValue();
      InsMask >>= Amount;
      Amount = 32-Amount;
      if (Op0Opc == ISD::SHL) IsRotate = true;
      break;
    case ISD::AND:
      InsMask &= (unsigned)CN->getValue();
      break;
    }
  } else {
    return false;
  }

  // Verify that the Target mask and Insert mask together form a full word mask
  // and that the Insert mask is a run of set bits (which implies both are runs
  // of set bits).  Given that, Select the arguments and generate the rlwimi
  // instruction.
  unsigned MB, ME;
  if (((TgtMask ^ InsMask) == 0xFFFFFFFF) && IsRunOfOnes(InsMask, MB, ME)) {
    unsigned Tmp1, Tmp2;
    // Check for rotlwi / rotrwi here, a special case of bitfield insert
    // where both bitfield halves are sourced from the same value.
    if (IsRotate &&
        OR.getOperand(0).getOperand(0) == OR.getOperand(1).getOperand(0)) {
      Tmp1 = SelectExpr(OR.getOperand(0).getOperand(0));
      BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp1).addImm(Amount)
        .addImm(0).addImm(31);
      return true;
    }
    if (Op0Opc == ISD::AND)
      Tmp1 = SelectExpr(OR.getOperand(0).getOperand(0));
    else
      Tmp1 = SelectExpr(OR.getOperand(0));
    Tmp2 = SelectExpr(OR.getOperand(1).getOperand(0));
    BuildMI(BB, PPC::RLWIMI, 5, Result).addReg(Tmp1).addReg(Tmp2)
      .addImm(Amount).addImm(MB).addImm(ME);
    return true;
  }
  return false;
}

/// FoldIfWideZeroExtend - 32 bit PowerPC implicit masks shift amounts to the
/// low six bits.  If the shift amount is an ISD::AND node with a mask that is
/// wider than the implicit mask, then we can get rid of the AND and let the
/// shift do the mask.
unsigned ISel::FoldIfWideZeroExtend(SDOperand N) {
  unsigned C;
  if (N.getOpcode() == ISD::AND &&
      5 == getImmediateForOpcode(N.getOperand(1), ISD::AND, C) && // isMask
      31 == (C & 0xFFFF) && // ME
      26 >= (C >> 16))      // MB
    return SelectExpr(N.getOperand(0));
  else
    return SelectExpr(N);
}

unsigned ISel::SelectCC(SDOperand CC, unsigned& Opc, bool &Inv, unsigned& Idx) {
  unsigned Result, Tmp1, Tmp2;
  bool AlreadySelected = false;
  static const unsigned CompareOpcodes[] =
    { PPC::FCMPU, PPC::FCMPU, PPC::CMPW, PPC::CMPLW };

  // Allocate a condition register for this expression
  Result = RegMap->createVirtualRegister(PPC32::CRRCRegisterClass);

  // If the first operand to the select is a SETCC node, then we can fold it
  // into the branch that selects which value to return.
  if (SetCCSDNode* SetCC = dyn_cast<SetCCSDNode>(CC.Val)) {
    bool U;
    Opc = getBCCForSetCC(SetCC->getCondition(), U);
    Idx = getCRIdxForSetCC(SetCC->getCondition(), Inv);

    // Pass the optional argument U to getImmediateForOpcode for SETCC,
    // so that it knows whether the SETCC immediate range is signed or not.
    if (1 == getImmediateForOpcode(SetCC->getOperand(1), ISD::SETCC,
                                   Tmp2, U)) {
      // For comparisons against zero, we can implicity set CR0 if a recording
      // variant (e.g. 'or.' instead of 'or') of the instruction that defines
      // operand zero of the SetCC node is available.
      if (0 == Tmp2 &&
          NodeHasRecordingVariant(SetCC->getOperand(0).getOpcode()) &&
          SetCC->getOperand(0).Val->hasOneUse()) {
        RecordSuccess = false;
        Tmp1 = SelectExpr(SetCC->getOperand(0), true);
        if (RecordSuccess) {
          ++Recorded;
          BuildMI(BB, PPC::MCRF, 1, Result).addReg(PPC::CR0);
          return Result;
        }
        AlreadySelected = true;
      }
      // If we could not implicitly set CR0, then emit a compare immediate
      // instead.
      if (!AlreadySelected) Tmp1 = SelectExpr(SetCC->getOperand(0));
      if (U)
        BuildMI(BB, PPC::CMPLWI, 2, Result).addReg(Tmp1).addImm(Tmp2);
      else
        BuildMI(BB, PPC::CMPWI, 2, Result).addReg(Tmp1).addSImm(Tmp2);
    } else {
      bool IsInteger = MVT::isInteger(SetCC->getOperand(0).getValueType());
      unsigned CompareOpc = CompareOpcodes[2 * IsInteger + U];
      Tmp1 = SelectExpr(SetCC->getOperand(0));
      Tmp2 = SelectExpr(SetCC->getOperand(1));
      BuildMI(BB, CompareOpc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
  } else {
    if (PPCCRopts)
      return SelectCCExpr(CC, Opc, Inv, Idx);
    // If this isn't a SetCC, then select the value and compare it against zero,
    // treating it as if it were a boolean.
    Opc = PPC::BNE;
    Idx = getCRIdxForSetCC(ISD::SETNE, Inv);
    Tmp1 = SelectExpr(CC);
    BuildMI(BB, PPC::CMPLWI, 2, Result).addReg(Tmp1).addImm(0);
  }
  return Result;
}

unsigned ISel::SelectCCExpr(SDOperand N, unsigned& Opc, bool &Inv,
                            unsigned &Idx) {
  bool Inv0, Inv1;
  unsigned Idx0, Idx1, CROpc, Opc1, Tmp1, Tmp2;

  // Allocate a condition register for this expression
  unsigned Result = RegMap->createVirtualRegister(PPC32::CRRCRegisterClass);

  // Check for the operations we support:
  switch(N.getOpcode()) {
  default:
    Opc = PPC::BNE;
    Idx = getCRIdxForSetCC(ISD::SETNE, Inv);
    Tmp1 = SelectExpr(N);
    BuildMI(BB, PPC::CMPLWI, 2, Result).addReg(Tmp1).addImm(0);
    break;
  case ISD::OR:
  case ISD::AND:
    ++MultiBranch;
    Tmp1 = SelectCCExpr(N.getOperand(0), Opc, Inv0, Idx0);
    Tmp2 = SelectCCExpr(N.getOperand(1), Opc1, Inv1, Idx1);
    CROpc = getCROpForSetCC(N.getOpcode(), Inv0, Inv1);
    if (Inv0 && !Inv1) {
      std::swap(Tmp1, Tmp2);
      std::swap(Idx0, Idx1);
      Opc = Opc1;
    }
    if (Inv0 && Inv1) Opc = PPC32InstrInfo::invertPPCBranchOpcode(Opc);
    BuildMI(BB, CROpc, 5, Result).addImm(Idx0).addReg(Tmp1).addImm(Idx0)
      .addReg(Tmp2).addImm(Idx1);
    Inv = false;
    Idx = Idx0;
    break;
  case ISD::SETCC:
    Tmp1 = SelectCC(N, Opc, Inv, Idx);
    Result = Tmp1;
    break;
  }
  return Result;
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
  MachineBasicBlock *Dest =
    cast<BasicBlockSDNode>(N.getOperand(2))->getBasicBlock();

  bool Inv;
  unsigned Opc, CCReg, Idx;
  Select(N.getOperand(0));  //chain
  CCReg = SelectCC(N.getOperand(1), Opc, Inv, Idx);

  // Iterate to the next basic block, unless we're already at the end of the
  ilist<MachineBasicBlock>::iterator It = BB, E = BB->getParent()->end();
  if (++It == E) It = BB;

  // If this is a two way branch, then grab the fallthrough basic block argument
  // and build a PowerPC branch pseudo-op, suitable for long branch conversion
  // if necessary by the branch selection pass.  Otherwise, emit a standard
  // conditional branch.
  if (N.getOpcode() == ISD::BRCONDTWOWAY) {
    MachineBasicBlock *Fallthrough =
      cast<BasicBlockSDNode>(N.getOperand(3))->getBasicBlock();
    if (Dest != It) {
      BuildMI(BB, PPC::COND_BRANCH, 4).addReg(CCReg).addImm(Opc)
        .addMBB(Dest).addMBB(Fallthrough);
      if (Fallthrough != It)
        BuildMI(BB, PPC::B, 1).addMBB(Fallthrough);
    } else {
      if (Fallthrough != It) {
        Opc = PPC32InstrInfo::invertPPCBranchOpcode(Opc);
        BuildMI(BB, PPC::COND_BRANCH, 4).addReg(CCReg).addImm(Opc)
          .addMBB(Fallthrough).addMBB(Dest);
      }
    }
  } else {
    BuildMI(BB, PPC::COND_BRANCH, 4).addReg(CCReg).addImm(Opc)
      .addMBB(Dest).addMBB(It);
  }
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
      unsigned TV = SelectExpr(N.getOperand(1)); // Use if TRUE
      unsigned FV = SelectExpr(N.getOperand(2)); // Use if FALSE

      ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(SetCC->getOperand(1));
      if (CN && (CN->isExactlyValue(-0.0) || CN->isExactlyValue(0.0))) {
        switch(SetCC->getCondition()) {
        default: assert(0 && "Invalid FSEL condition"); abort();
        case ISD::SETULT:
        case ISD::SETLT:
          std::swap(TV, FV);  // fsel is natively setge, swap operands for setlt
        case ISD::SETUGE:
        case ISD::SETGE:
          Tmp1 = SelectExpr(SetCC->getOperand(0));   // Val to compare against
          BuildMI(BB, PPC::FSEL, 3, Result).addReg(Tmp1).addReg(TV).addReg(FV);
          return Result;
        case ISD::SETUGT:
        case ISD::SETGT:
          std::swap(TV, FV);  // fsel is natively setge, swap operands for setlt
        case ISD::SETULE:
        case ISD::SETLE: {
          if (SetCC->getOperand(0).getOpcode() == ISD::FNEG) {
            Tmp2 = SelectExpr(SetCC->getOperand(0).getOperand(0));
          } else {
            Tmp2 = MakeReg(VT);
            Tmp1 = SelectExpr(SetCC->getOperand(0));   // Val to compare against
            BuildMI(BB, PPC::FNEG, 1, Tmp2).addReg(Tmp1);
          }
          BuildMI(BB, PPC::FSEL, 3, Result).addReg(Tmp2).addReg(TV).addReg(FV);
          return Result;
        }
        }
      } else {
        Opc = (MVT::f64 == VT) ? PPC::FSUB : PPC::FSUBS;
        Tmp1 = SelectExpr(SetCC->getOperand(0));   // Val to compare against
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

    bool Inv;
    unsigned TrueValue = SelectExpr(N.getOperand(1)); //Use if TRUE
    unsigned FalseValue = SelectExpr(N.getOperand(2)); //Use if FALSE
    unsigned CCReg = SelectCC(N.getOperand(0), Opc, Inv, Tmp3);

    // Create an iterator with which to insert the MBB for copying the false
    // value and the MBB to hold the PHI instruction for this SetCC.
    MachineBasicBlock *thisMBB = BB;
    const BasicBlock *LLVM_BB = BB->getBasicBlock();
    ilist<MachineBasicBlock>::iterator It = BB;
    ++It;

    //  thisMBB:
    //  ...
    //   TrueVal = ...
    //   cmpTY ccX, r1, r2
    //   bCC copy1MBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
    BuildMI(BB, Opc, 2).addReg(CCReg).addMBB(sinkMBB);
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
        ISD::ADD == N.getOperand(0).getOpcode() &&
        N.getOperand(0).Val->hasOneUse() &&
        ISD::MUL == N.getOperand(0).getOperand(1).getOpcode() &&
        N.getOperand(0).getOperand(1).Val->hasOneUse()) {
      ++FusedFP; // Statistic
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(1).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(1).getOperand(1));
      Tmp3 = SelectExpr(N.getOperand(0).getOperand(0));
      Opc = DestType == MVT::f64 ? PPC::FNMADD : PPC::FNMADDS;
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
    if (!NoExcessFPPrecision && N.getOperand(1).getOpcode() == ISD::MUL &&
        N.getOperand(1).Val->hasOneUse()) {
      ++FusedFP; // Statistic
      Tmp1 = SelectExpr(N.getOperand(1).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1).getOperand(1));
      Tmp3 = SelectExpr(N.getOperand(0));
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
    if (!NoExcessFPPrecision && N.getOperand(1).getOpcode() == ISD::MUL &&
        N.getOperand(1).Val->hasOneUse()) {
      ++FusedFP; // Statistic
      Tmp1 = SelectExpr(N.getOperand(1).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1).getOperand(1));
      Tmp3 = SelectExpr(N.getOperand(0));
      Opc = DestType == MVT::f64 ? PPC::FNMSUB : PPC::FNMSUBS;
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
    assert (N.getOperand(0).getValueType() == MVT::i32
            && "int to float must operate on i32");
    bool IsUnsigned = (ISD::UINT_TO_FP == opcode);
    Tmp1 = SelectExpr(N.getOperand(0));  // Get the operand register
    Tmp2 = MakeReg(MVT::f64); // temp reg to load the integer value into
    Tmp3 = MakeReg(MVT::i32); // temp reg to hold the conversion constant

    int FrameIdx = BB->getParent()->getFrameInfo()->CreateStackObject(8, 8);
    MachineConstantPool *CP = BB->getParent()->getConstantPool();

    if (IsUnsigned) {
      unsigned ConstF = getConstDouble(0x1.000000p52);
      // Store the hi & low halves of the fp value, currently in int regs
      BuildMI(BB, PPC::LIS, 1, Tmp3).addSImm(0x4330);
      addFrameReference(BuildMI(BB, PPC::STW, 3).addReg(Tmp3), FrameIdx);
      addFrameReference(BuildMI(BB, PPC::STW, 3).addReg(Tmp1), FrameIdx, 4);
      addFrameReference(BuildMI(BB, PPC::LFD, 2, Tmp2), FrameIdx);
      // Generate the return value with a subtract
      BuildMI(BB, PPC::FSUB, 2, Result).addReg(Tmp2).addReg(ConstF);
    } else {
      unsigned ConstF = getConstDouble(0x1.000008p52);
      unsigned TmpL = MakeReg(MVT::i32);
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

unsigned ISel::SelectExpr(SDOperand N, bool Recording) {
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
  case ISD::ADD_PARTS:
  case ISD::SUB_PARTS:
  case ISD::SHL_PARTS:
  case ISD::SRL_PARTS:
  case ISD::SRA_PARTS:
    Result = MakeReg(Node->getValueType(0));
    ExprMap[N.getValue(0)] = Result;
    for (unsigned i = 1, e = N.Val->getNumValues(); i != e; ++i)
      ExprMap[N.getValue(i)] = MakeReg(Node->getValueType(i));
    break;
  }

  if (ISD::CopyFromReg == opcode)
    DestType = N.getValue(0).getValueType();

  if (DestType == MVT::f64 || DestType == MVT::f32)
    if (ISD::LOAD != opcode && ISD::EXTLOAD != opcode &&
        ISD::UNDEF != opcode && ISD::CALL != opcode)
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
    Tmp2 = MakeReg(MVT::i32);
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
    Tmp1 = MakeReg(MVT::i32);
    BuildMI(BB, PPC::LOADHiAddr, 2, Tmp1).addReg(getGlobalBaseReg())
      .addGlobalAddress(GV);
    if (GV->hasWeakLinkage() || GV->isExternal()) {
      BuildMI(BB, PPC::LWZ, 2, Result).addGlobalAddress(GV).addReg(Tmp1);
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
    case MVT::i32: Opc = PPC::LWZ; break;
    case MVT::f32: Opc = PPC::LFS; break;
    case MVT::f64: Opc = PPC::LFD; break;
    }

    if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Address)) {
      Tmp1 = MakeReg(MVT::i32);
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
      if (Node->getValueType(1) == MVT::i32) {
        BuildMI(BB, PPC::OR, 2, Result+1).addReg(PPC::R3).addReg(PPC::R3);
        BuildMI(BB, PPC::OR, 2, Result).addReg(PPC::R4).addReg(PPC::R4);
      } else {
        BuildMI(BB, PPC::OR, 2, Result).addReg(PPC::R3).addReg(PPC::R3);
      }
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
      Tmp2 = CN->getValue() & 0x1F;
      BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp1).addImm(Tmp2).addImm(0)
        .addImm(31-Tmp2);
    } else {
      Tmp2 = FoldIfWideZeroExtend(N.getOperand(1));
      BuildMI(BB, PPC::SLW, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::SRL:
    Tmp1 = SelectExpr(N.getOperand(0));
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      Tmp2 = CN->getValue() & 0x1F;
      BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp1).addImm(32-Tmp2)
        .addImm(Tmp2).addImm(31);
    } else {
      Tmp2 = FoldIfWideZeroExtend(N.getOperand(1));
      BuildMI(BB, PPC::SRW, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::SRA:
    Tmp1 = SelectExpr(N.getOperand(0));
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      Tmp2 = CN->getValue() & 0x1F;
      BuildMI(BB, PPC::SRAWI, 2, Result).addReg(Tmp1).addImm(Tmp2);
    } else {
      Tmp2 = FoldIfWideZeroExtend(N.getOperand(1));
      BuildMI(BB, PPC::SRAW, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::ADD:
    assert (DestType == MVT::i32 && "Only do arithmetic on i32s!");
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
    if (PPCCRopts) {
      if (N.getOperand(0).getOpcode() == ISD::SETCC ||
          N.getOperand(1).getOpcode() == ISD::SETCC) {
        bool Inv;
        Tmp1 = SelectCCExpr(N, Opc, Inv, Tmp2);
        MoveCRtoGPR(Tmp1, Inv, Tmp2, Result);
        return Result;
      }
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    // FIXME: should add check in getImmediateForOpcode to return a value
    // indicating the immediate is a run of set bits so we can emit a bitfield
    // clear with RLWINM instead.
    switch(getImmediateForOpcode(N.getOperand(1), opcode, Tmp2)) {
      default: assert(0 && "unhandled result code");
      case 0: // No immediate
        Tmp2 = SelectExpr(N.getOperand(1));
        Opc = Recording ? PPC::ANDo : PPC::AND;
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
        break;
      case 1: // Low immediate
        BuildMI(BB, PPC::ANDIo, 2, Result).addReg(Tmp1).addImm(Tmp2);
        break;
      case 2: // Shifted immediate
        BuildMI(BB, PPC::ANDISo, 2, Result).addReg(Tmp1).addImm(Tmp2);
        break;
      case 5: // Bitfield mask
        Opc = Recording ? PPC::RLWINMo : PPC::RLWINM;
        Tmp3 = Tmp2 >> 16;  // MB
        Tmp2 &= 0xFFFF;     // ME
        BuildMI(BB, Opc, 4, Result).addReg(Tmp1).addImm(0)
          .addImm(Tmp3).addImm(Tmp2);
        break;
    }
    RecordSuccess = true;
    return Result;

  case ISD::OR:
    if (SelectBitfieldInsert(N, Result))
      return Result;
    if (PPCCRopts) {
      if (N.getOperand(0).getOpcode() == ISD::SETCC ||
          N.getOperand(1).getOpcode() == ISD::SETCC) {
        bool Inv;
        Tmp1 = SelectCCExpr(N, Opc, Inv, Tmp2);
        MoveCRtoGPR(Tmp1, Inv, Tmp2, Result);
        return Result;
      }
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    switch(getImmediateForOpcode(N.getOperand(1), opcode, Tmp2)) {
      default: assert(0 && "unhandled result code");
      case 0: // No immediate
        Tmp2 = SelectExpr(N.getOperand(1));
        Opc = Recording ? PPC::ORo : PPC::OR;
        RecordSuccess = true;
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
        break;
      case 1: // Low immediate
        BuildMI(BB, PPC::ORI, 2, Result).addReg(Tmp1).addImm(Tmp2);
        break;
      case 2: // Shifted immediate
        BuildMI(BB, PPC::ORIS, 2, Result).addReg(Tmp1).addImm(Tmp2);
        break;
    }
    return Result;

  case ISD::XOR: {
    // Check for EQV: xor, (xor a, -1), b
    if (N.getOperand(0).getOpcode() == ISD::XOR &&
        N.getOperand(0).getOperand(1).getOpcode() == ISD::Constant &&
        cast<ConstantSDNode>(N.getOperand(0).getOperand(1))->isAllOnesValue()) {
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, PPC::EQV, 2, Result).addReg(Tmp1).addReg(Tmp2);
      return Result;
    }
    // Check for NOT, NOR, EQV, and NAND: xor (copy, or, xor, and), -1
    if (N.getOperand(1).getOpcode() == ISD::Constant &&
        cast<ConstantSDNode>(N.getOperand(1))->isAllOnesValue()) {
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
      case ISD::XOR:
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
        BuildMI(BB, PPC::EQV, 2, Result).addReg(Tmp1).addReg(Tmp2);
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
      BuildMI(BB, PPC::MULLW, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;

  case ISD::MULHS:
  case ISD::MULHU:
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    Opc = (ISD::MULHU == opcode) ? PPC::MULHWU : PPC::MULHW;
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::SDIV:
  case ISD::UDIV:
    switch (getImmediateForOpcode(N.getOperand(1), opcode, Tmp3)) {
    default: break;
    // If this is an sdiv by a power of two, we can use an srawi/addze pair.
    case 3:
      Tmp1 = MakeReg(MVT::i32);
      Tmp2 = SelectExpr(N.getOperand(0));
      if ((int)Tmp3 < 0) {
        unsigned Tmp4 = MakeReg(MVT::i32);
        BuildMI(BB, PPC::SRAWI, 2, Tmp1).addReg(Tmp2).addImm(-Tmp3);
        BuildMI(BB, PPC::ADDZE, 1, Tmp4).addReg(Tmp1);
        BuildMI(BB, PPC::NEG, 1, Result).addReg(Tmp4);
      } else {
        BuildMI(BB, PPC::SRAWI, 2, Tmp1).addReg(Tmp2).addImm(Tmp3);
        BuildMI(BB, PPC::ADDZE, 1, Result).addReg(Tmp1);
      }
      return Result;
    // If this is a divide by constant, we can emit code using some magic
    // constants to implement it as a multiply instead.
    case 4:
      ExprMap.erase(N);
      if (opcode == ISD::SDIV)
        return SelectExpr(BuildSDIVSequence(N));
      else
        return SelectExpr(BuildUDIVSequence(N));
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    Opc = (ISD::UDIV == opcode) ? PPC::DIVWU : PPC::DIVW;
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::ADD_PARTS:
  case ISD::SUB_PARTS: {
    assert(N.getNumOperands() == 4 && N.getValueType() == MVT::i32 &&
           "Not an i64 add/sub!");
    // Emit all of the operands.
    std::vector<unsigned> InVals;
    for (unsigned i = 0, e = N.getNumOperands(); i != e; ++i)
      InVals.push_back(SelectExpr(N.getOperand(i)));
    if (N.getOpcode() == ISD::ADD_PARTS) {
      BuildMI(BB, PPC::ADDC, 2, Result).addReg(InVals[0]).addReg(InVals[2]);
      BuildMI(BB, PPC::ADDE, 2, Result+1).addReg(InVals[1]).addReg(InVals[3]);
    } else {
      BuildMI(BB, PPC::SUBFC, 2, Result).addReg(InVals[2]).addReg(InVals[0]);
      BuildMI(BB, PPC::SUBFE, 2, Result+1).addReg(InVals[3]).addReg(InVals[1]);
    }
    return Result+N.ResNo;
  }

  case ISD::SHL_PARTS:
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS: {
    assert(N.getNumOperands() == 3 && N.getValueType() == MVT::i32 &&
           "Not an i64 shift!");
    unsigned ShiftOpLo = SelectExpr(N.getOperand(0));
    unsigned ShiftOpHi = SelectExpr(N.getOperand(1));
    unsigned SHReg = FoldIfWideZeroExtend(N.getOperand(2));
    Tmp1 = MakeReg(MVT::i32);
    Tmp2 = MakeReg(MVT::i32);
    Tmp3 = MakeReg(MVT::i32);
    unsigned Tmp4 = MakeReg(MVT::i32);
    unsigned Tmp5 = MakeReg(MVT::i32);
    unsigned Tmp6 = MakeReg(MVT::i32);
    BuildMI(BB, PPC::SUBFIC, 2, Tmp1).addReg(SHReg).addSImm(32);
    if (ISD::SHL_PARTS == opcode) {
      BuildMI(BB, PPC::SLW, 2, Tmp2).addReg(ShiftOpHi).addReg(SHReg);
      BuildMI(BB, PPC::SRW, 2, Tmp3).addReg(ShiftOpLo).addReg(Tmp1);
      BuildMI(BB, PPC::OR, 2, Tmp4).addReg(Tmp2).addReg(Tmp3);
      BuildMI(BB, PPC::ADDI, 2, Tmp5).addReg(SHReg).addSImm(-32);
      BuildMI(BB, PPC::SLW, 2, Tmp6).addReg(ShiftOpLo).addReg(Tmp5);
      BuildMI(BB, PPC::OR, 2, Result+1).addReg(Tmp4).addReg(Tmp6);
      BuildMI(BB, PPC::SLW, 2, Result).addReg(ShiftOpLo).addReg(SHReg);
    } else if (ISD::SRL_PARTS == opcode) {
      BuildMI(BB, PPC::SRW, 2, Tmp2).addReg(ShiftOpLo).addReg(SHReg);
      BuildMI(BB, PPC::SLW, 2, Tmp3).addReg(ShiftOpHi).addReg(Tmp1);
      BuildMI(BB, PPC::OR, 2, Tmp4).addReg(Tmp2).addReg(Tmp3);
      BuildMI(BB, PPC::ADDI, 2, Tmp5).addReg(SHReg).addSImm(-32);
      BuildMI(BB, PPC::SRW, 2, Tmp6).addReg(ShiftOpHi).addReg(Tmp5);
      BuildMI(BB, PPC::OR, 2, Result).addReg(Tmp4).addReg(Tmp6);
      BuildMI(BB, PPC::SRW, 2, Result+1).addReg(ShiftOpHi).addReg(SHReg);
    } else {
      MachineBasicBlock *TmpMBB = new MachineBasicBlock(BB->getBasicBlock());
      MachineBasicBlock *PhiMBB = new MachineBasicBlock(BB->getBasicBlock());
      MachineBasicBlock *OldMBB = BB;
      MachineFunction *F = BB->getParent();
      ilist<MachineBasicBlock>::iterator It = BB; ++It;
      F->getBasicBlockList().insert(It, TmpMBB);
      F->getBasicBlockList().insert(It, PhiMBB);
      BB->addSuccessor(TmpMBB);
      BB->addSuccessor(PhiMBB);
      BuildMI(BB, PPC::SRW, 2, Tmp2).addReg(ShiftOpLo).addReg(SHReg);
      BuildMI(BB, PPC::SLW, 2, Tmp3).addReg(ShiftOpHi).addReg(Tmp1);
      BuildMI(BB, PPC::OR, 2, Tmp4).addReg(Tmp2).addReg(Tmp3);
      BuildMI(BB, PPC::ADDICo, 2, Tmp5).addReg(SHReg).addSImm(-32);
      BuildMI(BB, PPC::SRAW, 2, Tmp6).addReg(ShiftOpHi).addReg(Tmp5);
      BuildMI(BB, PPC::SRAW, 2, Result+1).addReg(ShiftOpHi).addReg(SHReg);
      BuildMI(BB, PPC::BLE, 2).addReg(PPC::CR0).addMBB(PhiMBB);
      // Select correct least significant half if the shift amount > 32
      BB = TmpMBB;
      unsigned Tmp7 = MakeReg(MVT::i32);
      BuildMI(BB, PPC::OR, 2, Tmp7).addReg(Tmp6).addReg(Tmp6);
      TmpMBB->addSuccessor(PhiMBB);
      BB = PhiMBB;
      BuildMI(BB, PPC::PHI, 4, Result).addReg(Tmp4).addMBB(OldMBB)
        .addReg(Tmp7).addMBB(TmpMBB);
    }
    return Result+N.ResNo;
  }

  case ISD::FP_TO_UINT:
  case ISD::FP_TO_SINT: {
    bool U = (ISD::FP_TO_UINT == opcode);
    Tmp1 = SelectExpr(N.getOperand(0));
    if (!U) {
      Tmp2 = MakeReg(MVT::f64);
      BuildMI(BB, PPC::FCTIWZ, 1, Tmp2).addReg(Tmp1);
      int FrameIdx = BB->getParent()->getFrameInfo()->CreateStackObject(8, 8);
      addFrameReference(BuildMI(BB, PPC::STFD, 3).addReg(Tmp2), FrameIdx);
      addFrameReference(BuildMI(BB, PPC::LWZ, 2, Result), FrameIdx, 4);
      return Result;
    } else {
      unsigned Zero = getConstDouble(0.0);
      unsigned MaxInt = getConstDouble((1LL << 32) - 1);
      unsigned Border = getConstDouble(1LL << 31);
      unsigned UseZero = MakeReg(MVT::f64);
      unsigned UseMaxInt = MakeReg(MVT::f64);
      unsigned UseChoice = MakeReg(MVT::f64);
      unsigned TmpReg = MakeReg(MVT::f64);
      unsigned TmpReg2 = MakeReg(MVT::f64);
      unsigned ConvReg = MakeReg(MVT::f64);
      unsigned IntTmp = MakeReg(MVT::i32);
      unsigned XorReg = MakeReg(MVT::i32);
      MachineFunction *F = BB->getParent();
      int FrameIdx = F->getFrameInfo()->CreateStackObject(8, 8);
      // Update machine-CFG edges
      MachineBasicBlock *XorMBB = new MachineBasicBlock(BB->getBasicBlock());
      MachineBasicBlock *PhiMBB = new MachineBasicBlock(BB->getBasicBlock());
      MachineBasicBlock *OldMBB = BB;
      ilist<MachineBasicBlock>::iterator It = BB; ++It;
      F->getBasicBlockList().insert(It, XorMBB);
      F->getBasicBlockList().insert(It, PhiMBB);
      BB->addSuccessor(XorMBB);
      BB->addSuccessor(PhiMBB);
      // Convert from floating point to unsigned 32-bit value
      // Use 0 if incoming value is < 0.0
      BuildMI(BB, PPC::FSEL, 3, UseZero).addReg(Tmp1).addReg(Tmp1).addReg(Zero);
      // Use 2**32 - 1 if incoming value is >= 2**32
      BuildMI(BB, PPC::FSUB, 2, UseMaxInt).addReg(MaxInt).addReg(Tmp1);
      BuildMI(BB, PPC::FSEL, 3, UseChoice).addReg(UseMaxInt).addReg(UseZero)
        .addReg(MaxInt);
      // Subtract 2**31
      BuildMI(BB, PPC::FSUB, 2, TmpReg).addReg(UseChoice).addReg(Border);
      // Use difference if >= 2**31
      BuildMI(BB, PPC::FCMPU, 2, PPC::CR0).addReg(UseChoice).addReg(Border);
      BuildMI(BB, PPC::FSEL, 3, TmpReg2).addReg(TmpReg).addReg(TmpReg)
        .addReg(UseChoice);
      // Convert to integer
      BuildMI(BB, PPC::FCTIWZ, 1, ConvReg).addReg(TmpReg2);
      addFrameReference(BuildMI(BB, PPC::STFD, 3).addReg(ConvReg), FrameIdx);
      addFrameReference(BuildMI(BB, PPC::LWZ, 2, IntTmp), FrameIdx, 4);
      BuildMI(BB, PPC::BLT, 2).addReg(PPC::CR0).addMBB(PhiMBB);
      BuildMI(BB, PPC::B, 1).addMBB(XorMBB);

      // XorMBB:
      //   add 2**31 if input was >= 2**31
      BB = XorMBB;
      BuildMI(BB, PPC::XORIS, 2, XorReg).addReg(IntTmp).addImm(0x8000);
      XorMBB->addSuccessor(PhiMBB);

      // PhiMBB:
      //   DestReg = phi [ IntTmp, OldMBB ], [ XorReg, XorMBB ]
      BB = PhiMBB;
      BuildMI(BB, PPC::PHI, 4, Result).addReg(IntTmp).addMBB(OldMBB)
        .addReg(XorReg).addMBB(XorMBB);
      return Result;
    }
    assert(0 && "Should never get here");
    return 0;
  }

  case ISD::SETCC:
    if (SetCCSDNode *SetCC = dyn_cast<SetCCSDNode>(Node)) {
      if (ConstantSDNode *CN =
          dyn_cast<ConstantSDNode>(SetCC->getOperand(1).Val)) {
        // We can codegen setcc op, imm very efficiently compared to a brcond.
        // Check for those cases here.
        // setcc op, 0
        if (CN->getValue() == 0) {
          Tmp1 = SelectExpr(SetCC->getOperand(0));
          switch (SetCC->getCondition()) {
          default: SetCC->dump(); assert(0 && "Unhandled SetCC condition"); abort();
          case ISD::SETEQ:
            Tmp2 = MakeReg(MVT::i32);
            BuildMI(BB, PPC::CNTLZW, 1, Tmp2).addReg(Tmp1);
            BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp2).addImm(27)
              .addImm(5).addImm(31);
            break;
          case ISD::SETNE:
            Tmp2 = MakeReg(MVT::i32);
            BuildMI(BB, PPC::ADDIC, 2, Tmp2).addReg(Tmp1).addSImm(-1);
            BuildMI(BB, PPC::SUBFE, 2, Result).addReg(Tmp2).addReg(Tmp1);
            break;
          case ISD::SETLT:
            BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp1).addImm(1)
              .addImm(31).addImm(31);
            break;
          case ISD::SETGT:
            Tmp2 = MakeReg(MVT::i32);
            Tmp3 = MakeReg(MVT::i32);
            BuildMI(BB, PPC::NEG, 2, Tmp2).addReg(Tmp1);
            BuildMI(BB, PPC::ANDC, 2, Tmp3).addReg(Tmp2).addReg(Tmp1);
            BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp3).addImm(1)
              .addImm(31).addImm(31);
            break;
          }
          return Result;
        }
        // setcc op, -1
        if (CN->isAllOnesValue()) {
          Tmp1 = SelectExpr(SetCC->getOperand(0));
          switch (SetCC->getCondition()) {
          default: assert(0 && "Unhandled SetCC condition"); abort();
          case ISD::SETEQ:
            Tmp2 = MakeReg(MVT::i32);
            Tmp3 = MakeReg(MVT::i32);
            BuildMI(BB, PPC::ADDIC, 2, Tmp2).addReg(Tmp1).addSImm(1);
            BuildMI(BB, PPC::LI, 1, Tmp3).addSImm(0);
            BuildMI(BB, PPC::ADDZE, 1, Result).addReg(Tmp3);
            break;
          case ISD::SETNE:
            Tmp2 = MakeReg(MVT::i32);
            Tmp3 = MakeReg(MVT::i32);
            BuildMI(BB, PPC::NOR, 2, Tmp2).addReg(Tmp1).addReg(Tmp1);
            BuildMI(BB, PPC::ADDIC, 2, Tmp3).addReg(Tmp2).addSImm(-1);
            BuildMI(BB, PPC::SUBFE, 2, Result).addReg(Tmp3).addReg(Tmp2);
            break;
          case ISD::SETLT:
            Tmp2 = MakeReg(MVT::i32);
            Tmp3 = MakeReg(MVT::i32);
            BuildMI(BB, PPC::ADDI, 2, Tmp2).addReg(Tmp1).addSImm(1);
            BuildMI(BB, PPC::AND, 2, Tmp3).addReg(Tmp2).addReg(Tmp1);
            BuildMI(BB, PPC::RLWINM, 4, Result).addReg(Tmp3).addImm(1)
              .addImm(31).addImm(31);
            break;
          case ISD::SETGT:
            Tmp2 = MakeReg(MVT::i32);
            BuildMI(BB, PPC::RLWINM, 4, Tmp2).addReg(Tmp1).addImm(1)
              .addImm(31).addImm(31);
            BuildMI(BB, PPC::XORI, 2, Result).addReg(Tmp2).addImm(1);
            break;
          }
          return Result;
        }
      }

      bool Inv;
      unsigned CCReg = SelectCC(N, Opc, Inv, Tmp2);
      MoveCRtoGPR(CCReg, Inv, Tmp2, Result);
      return Result;
    }
    assert(0 && "Is this legal?");
    return 0;

  case ISD::SELECT: {
    bool Inv;
    unsigned TrueValue = SelectExpr(N.getOperand(1)); //Use if TRUE
    unsigned FalseValue = SelectExpr(N.getOperand(2)); //Use if FALSE
    unsigned CCReg = SelectCC(N.getOperand(0), Opc, Inv, Tmp3);

    // Create an iterator with which to insert the MBB for copying the false
    // value and the MBB to hold the PHI instruction for this SetCC.
    MachineBasicBlock *thisMBB = BB;
    const BasicBlock *LLVM_BB = BB->getBasicBlock();
    ilist<MachineBasicBlock>::iterator It = BB;
    ++It;

    //  thisMBB:
    //  ...
    //   TrueVal = ...
    //   cmpTY ccX, r1, r2
    //   bCC copy1MBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
    BuildMI(BB, Opc, 2).addReg(CCReg).addMBB(sinkMBB);
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
  case ISD::ADJCALLSTACKDOWN:
  case ISD::ADJCALLSTACKUP:
    Select(N.getOperand(0));
    Tmp1 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
    Opc = N.getOpcode() == ISD::ADJCALLSTACKDOWN ? PPC::ADJCALLSTACKDOWN :
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
  case ISD::BRCONDTWOWAY:
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
        case MVT::i32: Opc = PPC::STW; break;
        case MVT::f64: Opc = PPC::STFD; break;
        case MVT::f32: Opc = PPC::STFS; break;
        }
      } else { //ISD::TRUNCSTORE
        switch(cast<MVTSDNode>(Node)->getExtraValueType()) {
        default: assert(0 && "unknown Type in store");
        case MVT::i1:
        case MVT::i8: Opc  = PPC::STB; break;
        case MVT::i16: Opc = PPC::STH; break;
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
FunctionPass *llvm::createPPC32ISelPattern(TargetMachine &TM) {
  return new ISel(TM);
}

