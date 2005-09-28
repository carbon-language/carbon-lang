//===-- X86ISelPattern.cpp - A pattern matching inst selector for X86 -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/Statistic.h"
#include <set>
#include <algorithm>
using namespace llvm;

// FIXME: temporary.
#include "llvm/Support/CommandLine.h"
static cl::opt<bool> EnableFastCC("enable-x86-fastcc", cl::Hidden,
                                  cl::desc("Enable fastcc on X86"));

namespace {
  // X86 Specific DAG Nodes
  namespace X86ISD {
    enum NodeType {
      // Start the numbering where the builtin ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      /// FILD64m - This instruction implements SINT_TO_FP with a
      /// 64-bit source in memory and a FP reg result.  This corresponds to
      /// the X86::FILD64m instruction.  It has two inputs (token chain and
      /// address) and two outputs (FP value and token chain).
      FILD64m,

      /// FP_TO_INT*_IN_MEM - This instruction implements FP_TO_SINT with the
      /// integer destination in memory and a FP reg source.  This corresponds
      /// to the X86::FIST*m instructions and the rounding mode change stuff. It
      /// has two inputs (token chain and address) and two outputs (FP value and
      /// token chain).
      FP_TO_INT16_IN_MEM,
      FP_TO_INT32_IN_MEM,
      FP_TO_INT64_IN_MEM,

      /// CALL/TAILCALL - These operations represent an abstract X86 call
      /// instruction, which includes a bunch of information.  In particular the
      /// operands of these node are:
      ///
      ///     #0 - The incoming token chain
      ///     #1 - The callee
      ///     #2 - The number of arg bytes the caller pushes on the stack.
      ///     #3 - The number of arg bytes the callee pops off the stack.
      ///     #4 - The value to pass in AL/AX/EAX (optional)
      ///     #5 - The value to pass in DL/DX/EDX (optional)
      ///
      /// The result values of these nodes are:
      ///
      ///     #0 - The outgoing token chain
      ///     #1 - The first register result value (optional)
      ///     #2 - The second register result value (optional)
      ///
      /// The CALL vs TAILCALL distinction boils down to whether the callee is
      /// known not to modify the caller's stack frame, as is standard with
      /// LLVM.
      CALL,
      TAILCALL,
    };
  }
}

//===----------------------------------------------------------------------===//
//  X86TargetLowering - X86 Implementation of the TargetLowering interface
namespace {
  class X86TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    int ReturnAddrIndex;              // FrameIndex for return slot.
    int BytesToPopOnReturn;           // Number of arg bytes ret should pop.
    int BytesCallerReserves;          // Number of arg bytes caller makes.
  public:
    X86TargetLowering(TargetMachine &TM) : TargetLowering(TM) {
      // Set up the TargetLowering object.

      // X86 is weird, it always uses i8 for shift amounts and setcc results.
      setShiftAmountType(MVT::i8);
      setSetCCResultType(MVT::i8);
      setSetCCResultContents(ZeroOrOneSetCCResult);
      setShiftAmountFlavor(Mask);   // shl X, 32 == shl X, 0

      // Set up the register classes.
      // FIXME: Eliminate these two classes when legalize can handle promotions
      // well.
      addRegisterClass(MVT::i1, X86::R8RegisterClass);
      addRegisterClass(MVT::i8, X86::R8RegisterClass);
      addRegisterClass(MVT::i16, X86::R16RegisterClass);
      addRegisterClass(MVT::i32, X86::R32RegisterClass);

      // Promote all UINT_TO_FP to larger SINT_TO_FP's, as X86 doesn't have this
      // operation.
      setOperationAction(ISD::UINT_TO_FP       , MVT::i1   , Promote);
      setOperationAction(ISD::UINT_TO_FP       , MVT::i8   , Promote);
      setOperationAction(ISD::UINT_TO_FP       , MVT::i16  , Promote);
      setOperationAction(ISD::UINT_TO_FP       , MVT::i32  , Promote);

      // Promote i1/i8 SINT_TO_FP to larger SINT_TO_FP's, as X86 doesn't have
      // this operation.
      setOperationAction(ISD::SINT_TO_FP       , MVT::i1   , Promote);
      setOperationAction(ISD::SINT_TO_FP       , MVT::i8   , Promote);

      if (!X86ScalarSSE) {
        // We can handle SINT_TO_FP and FP_TO_SINT from/TO i64 even though i64
        // isn't legal.
        setOperationAction(ISD::SINT_TO_FP     , MVT::i64  , Custom);
        setOperationAction(ISD::FP_TO_SINT     , MVT::i64  , Custom);
        setOperationAction(ISD::FP_TO_SINT     , MVT::i32  , Custom);
        setOperationAction(ISD::FP_TO_SINT     , MVT::i16  , Custom);
      }

      // Handle FP_TO_UINT by promoting the destination to a larger signed
      // conversion.
      setOperationAction(ISD::FP_TO_UINT       , MVT::i1   , Promote);
      setOperationAction(ISD::FP_TO_UINT       , MVT::i8   , Promote);
      setOperationAction(ISD::FP_TO_UINT       , MVT::i16  , Promote);

      if (!X86ScalarSSE)
        setOperationAction(ISD::FP_TO_UINT     , MVT::i32  , Promote);

      // Promote i1/i8 FP_TO_SINT to larger FP_TO_SINTS's, as X86 doesn't have
      // this operation.
      setOperationAction(ISD::FP_TO_SINT       , MVT::i1   , Promote);
      setOperationAction(ISD::FP_TO_SINT       , MVT::i8   , Promote);
      setOperationAction(ISD::FP_TO_SINT       , MVT::i16  , Promote);

      setOperationAction(ISD::BRCONDTWOWAY     , MVT::Other, Expand);
      setOperationAction(ISD::BRTWOWAY_CC      , MVT::Other, Expand);
      setOperationAction(ISD::MEMMOVE          , MVT::Other, Expand);
      setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16  , Expand);
      setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1   , Expand);
      setOperationAction(ISD::FP_ROUND_INREG   , MVT::f32  , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i1   , Expand);
      setOperationAction(ISD::FREM             , MVT::f64  , Expand);
      setOperationAction(ISD::CTPOP            , MVT::i8   , Expand);
      setOperationAction(ISD::CTTZ             , MVT::i8   , Expand);
      setOperationAction(ISD::CTLZ             , MVT::i8   , Expand);
      setOperationAction(ISD::CTPOP            , MVT::i16  , Expand);
      setOperationAction(ISD::CTTZ             , MVT::i16  , Expand);
      setOperationAction(ISD::CTLZ             , MVT::i16  , Expand);
      setOperationAction(ISD::CTPOP            , MVT::i32  , Expand);
      setOperationAction(ISD::CTTZ             , MVT::i32  , Expand);
      setOperationAction(ISD::CTLZ             , MVT::i32  , Expand);

      setOperationAction(ISD::READIO           , MVT::i1   , Expand);
      setOperationAction(ISD::READIO           , MVT::i8   , Expand);
      setOperationAction(ISD::READIO           , MVT::i16  , Expand);
      setOperationAction(ISD::READIO           , MVT::i32  , Expand);
      setOperationAction(ISD::WRITEIO          , MVT::i1   , Expand);
      setOperationAction(ISD::WRITEIO          , MVT::i8   , Expand);
      setOperationAction(ISD::WRITEIO          , MVT::i16  , Expand);
      setOperationAction(ISD::WRITEIO          , MVT::i32  , Expand);

      // These should be promoted to a larger select which is supported.
      setOperationAction(ISD::SELECT           , MVT::i1   , Promote);
      setOperationAction(ISD::SELECT           , MVT::i8   , Promote);

      if (X86ScalarSSE) {
        // Set up the FP register classes.
        addRegisterClass(MVT::f32, X86::RXMMRegisterClass);
        addRegisterClass(MVT::f64, X86::RXMMRegisterClass);

        // SSE has no load+extend ops
        setOperationAction(ISD::EXTLOAD,  MVT::f32, Expand);
        setOperationAction(ISD::ZEXTLOAD, MVT::f32, Expand);

        // SSE has no i16 to fp conversion, only i32
        setOperationAction(ISD::SINT_TO_FP, MVT::i16, Promote);
        setOperationAction(ISD::FP_TO_SINT, MVT::i16, Promote);

        // Expand FP_TO_UINT into a select.
        // FIXME: We would like to use a Custom expander here eventually to do
        // the optimal thing for SSE vs. the default expansion in the legalizer.
        setOperationAction(ISD::FP_TO_UINT       , MVT::i32  , Expand);
        
        // We don't support sin/cos/sqrt/fmod
        setOperationAction(ISD::FSIN , MVT::f64, Expand);
        setOperationAction(ISD::FCOS , MVT::f64, Expand);
        setOperationAction(ISD::FABS , MVT::f64, Expand);
        setOperationAction(ISD::FNEG , MVT::f64, Expand);
        setOperationAction(ISD::FREM , MVT::f64, Expand);
        setOperationAction(ISD::FSIN , MVT::f32, Expand);
        setOperationAction(ISD::FCOS , MVT::f32, Expand);
        setOperationAction(ISD::FABS , MVT::f32, Expand);
        setOperationAction(ISD::FNEG , MVT::f32, Expand);
        setOperationAction(ISD::FREM , MVT::f32, Expand);

        addLegalFPImmediate(+0.0); // xorps / xorpd
      } else {
        // Set up the FP register classes.
        addRegisterClass(MVT::f64, X86::RFPRegisterClass);

        if (!UnsafeFPMath) {
          setOperationAction(ISD::FSIN           , MVT::f64  , Expand);
          setOperationAction(ISD::FCOS           , MVT::f64  , Expand);
        }

        addLegalFPImmediate(+0.0); // FLD0
        addLegalFPImmediate(+1.0); // FLD1
        addLegalFPImmediate(-0.0); // FLD0/FCHS
        addLegalFPImmediate(-1.0); // FLD1/FCHS
      }
      computeRegisterProperties();

      maxStoresPerMemSet = 8; // For %llvm.memset -> sequence of stores
      maxStoresPerMemCpy = 8; // For %llvm.memcpy -> sequence of stores
      maxStoresPerMemMove = 8; // For %llvm.memmove -> sequence of stores
      allowUnalignedMemoryAccesses = true; // x86 supports it!
    }

    // Return the number of bytes that a function should pop when it returns (in
    // addition to the space used by the return address).
    //
    unsigned getBytesToPopOnReturn() const { return BytesToPopOnReturn; }

    // Return the number of bytes that the caller reserves for arguments passed
    // to this function.
    unsigned getBytesCallerReserves() const { return BytesCallerReserves; }

    /// LowerOperation - Provide custom lowering hooks for some operations.
    ///
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);

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

    virtual SDOperand LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                   Value *VAListV, SelectionDAG &DAG);
    virtual std::pair<SDOperand,SDOperand>
      LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
                 const Type *ArgTy, SelectionDAG &DAG);

    virtual std::pair<SDOperand, SDOperand>
    LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                            SelectionDAG &DAG);

    SDOperand getReturnAddressFrameIndex(SelectionDAG &DAG);

  private:
    // C Calling Convention implementation.
    std::vector<SDOperand> LowerCCCArguments(Function &F, SelectionDAG &DAG);
    std::pair<SDOperand, SDOperand>
    LowerCCCCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg,
                   bool isTailCall,
                   SDOperand Callee, ArgListTy &Args, SelectionDAG &DAG);

    // Fast Calling Convention implementation.
    std::vector<SDOperand> LowerFastCCArguments(Function &F, SelectionDAG &DAG);
    std::pair<SDOperand, SDOperand>
    LowerFastCCCallTo(SDOperand Chain, const Type *RetTy, bool isTailCall,
                      SDOperand Callee, ArgListTy &Args, SelectionDAG &DAG);
  };
}

std::vector<SDOperand>
X86TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  if (F.getCallingConv() == CallingConv::Fast && EnableFastCC)
    return LowerFastCCArguments(F, DAG);
  return LowerCCCArguments(F, DAG);
}

std::pair<SDOperand, SDOperand>
X86TargetLowering::LowerCallTo(SDOperand Chain, const Type *RetTy,
                               bool isVarArg, unsigned CallingConv,
                               bool isTailCall,
                               SDOperand Callee, ArgListTy &Args,
                               SelectionDAG &DAG) {
  assert((!isVarArg || CallingConv == CallingConv::C) &&
         "Only C takes varargs!");
  if (CallingConv == CallingConv::Fast && EnableFastCC)
    return LowerFastCCCallTo(Chain, RetTy, isTailCall, Callee, Args, DAG);
  return  LowerCCCCallTo(Chain, RetTy, isVarArg, isTailCall, Callee, Args, DAG);
}

//===----------------------------------------------------------------------===//
//                    C Calling Convention implementation
//===----------------------------------------------------------------------===//

std::vector<SDOperand>
X86TargetLowering::LowerCCCArguments(Function &F, SelectionDAG &DAG) {
  std::vector<SDOperand> ArgValues;

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Add DAG nodes to load the arguments...  On entry to a function on the X86,
  // the stack frame looks like this:
  //
  // [ESP] -- return address
  // [ESP + 4] -- first argument (leftmost lexically)
  // [ESP + 8] -- second argument, if first argument is four bytes in size
  //    ...
  //
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I) {
    MVT::ValueType ObjectVT = getValueType(I->getType());
    unsigned ArgIncrement = 4;
    unsigned ObjSize;
    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i1:
    case MVT::i8:  ObjSize = 1;                break;
    case MVT::i16: ObjSize = 2;                break;
    case MVT::i32: ObjSize = 4;                break;
    case MVT::i64: ObjSize = ArgIncrement = 8; break;
    case MVT::f32: ObjSize = 4;                break;
    case MVT::f64: ObjSize = ArgIncrement = 8; break;
    }
    // Create the frame index object for this incoming parameter...
    int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);

    // Create the SelectionDAG nodes corresponding to a load from this parameter
    SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);

    // Don't codegen dead arguments.  FIXME: remove this check when we can nuke
    // dead loads.
    SDOperand ArgValue;
    if (!I->use_empty())
      ArgValue = DAG.getLoad(ObjectVT, DAG.getEntryNode(), FIN,
                             DAG.getSrcValue(NULL));
    else {
      if (MVT::isInteger(ObjectVT))
        ArgValue = DAG.getConstant(0, ObjectVT);
      else
        ArgValue = DAG.getConstantFP(0, ObjectVT);
    }
    ArgValues.push_back(ArgValue);

    ArgOffset += ArgIncrement;   // Move on to the next argument...
  }

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (F.isVarArg())
    VarArgsFrameIndex = MFI->CreateFixedObject(1, ArgOffset);
  ReturnAddrIndex = 0;     // No return address slot generated yet.
  BytesToPopOnReturn = 0;  // Callee pops nothing.
  BytesCallerReserves = ArgOffset;

  // Finally, inform the code generator which regs we return values in.
  switch (getValueType(F.getReturnType())) {
  default: assert(0 && "Unknown type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    MF.addLiveOut(X86::EAX);
    break;
  case MVT::i64:
    MF.addLiveOut(X86::EAX);
    MF.addLiveOut(X86::EDX);
    break;
  case MVT::f32:
  case MVT::f64:
    MF.addLiveOut(X86::ST0);
    break;
  }
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
X86TargetLowering::LowerCCCCallTo(SDOperand Chain, const Type *RetTy,
                                  bool isVarArg, bool isTailCall,
                                  SDOperand Callee, ArgListTy &Args,
                                  SelectionDAG &DAG) {
  // Count how many bytes are to be pushed on the stack.
  unsigned NumBytes = 0;

  if (Args.empty()) {
    // Save zero bytes.
    Chain = DAG.getNode(ISD::CALLSEQ_START, MVT::Other, Chain,
                        DAG.getConstant(0, getPointerTy()));
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

    Chain = DAG.getNode(ISD::CALLSEQ_START, MVT::Other, Chain,
                        DAG.getConstant(NumBytes, getPointerTy()));

    // Arguments go on the stack in reverse order, as specified by the ABI.
    unsigned ArgOffset = 0;
    SDOperand StackPtr = DAG.getCopyFromReg(DAG.getEntryNode(),
                                            X86::ESP, MVT::i32);
    std::vector<SDOperand> Stores;

    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);

      switch (getValueType(Args[i].second)) {
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
      case MVT::f32:
        Stores.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                     Args[i].first, PtrOff,
                                     DAG.getSrcValue(NULL)));
        ArgOffset += 4;
        break;
      case MVT::i64:
      case MVT::f64:
        Stores.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                     Args[i].first, PtrOff,
                                     DAG.getSrcValue(NULL)));
        ArgOffset += 8;
        break;
      }
    }
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, Stores);
  }

  std::vector<MVT::ValueType> RetVals;
  MVT::ValueType RetTyVT = getValueType(RetTy);
  RetVals.push_back(MVT::Other);

  // The result values produced have to be legal.  Promote the result.
  switch (RetTyVT) {
  case MVT::isVoid: break;
  default:
    RetVals.push_back(RetTyVT);
    break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
    RetVals.push_back(MVT::i32);
    break;
  case MVT::f32:
    if (X86ScalarSSE)
      RetVals.push_back(MVT::f32);
    else
      RetVals.push_back(MVT::f64);
    break;
  case MVT::i64:
    RetVals.push_back(MVT::i32);
    RetVals.push_back(MVT::i32);
    break;
  }
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  Ops.push_back(DAG.getConstant(NumBytes, getPointerTy()));
  Ops.push_back(DAG.getConstant(0, getPointerTy()));
  SDOperand TheCall = DAG.getNode(isTailCall ? X86ISD::TAILCALL : X86ISD::CALL,
                                  RetVals, Ops);
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, TheCall);

  SDOperand ResultVal;
  switch (RetTyVT) {
  case MVT::isVoid: break;
  default:
    ResultVal = TheCall.getValue(1);
    break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
    ResultVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, TheCall.getValue(1));
    break;
  case MVT::f32:
    // FIXME: we would really like to remember that this FP_ROUND operation is
    // okay to eliminate if we allow excess FP precision.
    ResultVal = DAG.getNode(ISD::FP_ROUND, MVT::f32, TheCall.getValue(1));
    break;
  case MVT::i64:
    ResultVal = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, TheCall.getValue(1),
                            TheCall.getValue(2));
    break;
  }

  return std::make_pair(ResultVal, Chain);
}

SDOperand
X86TargetLowering::LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                Value *VAListV, SelectionDAG &DAG) {
  // vastart just stores the address of the VarArgsFrameIndex slot.
  SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i32);
  return DAG.getNode(ISD::STORE, MVT::Other, Chain, FR, VAListP,
                     DAG.getSrcValue(VAListV));
}


std::pair<SDOperand,SDOperand>
X86TargetLowering::LowerVAArg(SDOperand Chain, SDOperand VAListP,
                              Value *VAListV, const Type *ArgTy,
                              SelectionDAG &DAG) {
  MVT::ValueType ArgVT = getValueType(ArgTy);
  SDOperand Val = DAG.getLoad(MVT::i32, Chain,
                              VAListP, DAG.getSrcValue(VAListV));
  SDOperand Result = DAG.getLoad(ArgVT, Chain, Val,
                                 DAG.getSrcValue(NULL));
  unsigned Amt;
  if (ArgVT == MVT::i32)
    Amt = 4;
  else {
    assert((ArgVT == MVT::i64 || ArgVT == MVT::f64) &&
           "Other types should have been promoted for varargs!");
    Amt = 8;
  }
  Val = DAG.getNode(ISD::ADD, Val.getValueType(), Val,
                    DAG.getConstant(Amt, Val.getValueType()));
  Chain = DAG.getNode(ISD::STORE, MVT::Other, Chain,
                      Val, VAListP, DAG.getSrcValue(VAListV));
  return std::make_pair(Result, Chain);
}

//===----------------------------------------------------------------------===//
//                    Fast Calling Convention implementation
//===----------------------------------------------------------------------===//
//
// The X86 'fast' calling convention passes up to two integer arguments in
// registers (an appropriate portion of EAX/EDX), passes arguments in C order,
// and requires that the callee pop its arguments off the stack (allowing proper
// tail calls), and has the same return value conventions as C calling convs.
//
// This calling convention always arranges for the callee pop value to be 8n+4
// bytes, which is needed for tail recursion elimination and stack alignment
// reasons.
//
// Note that this can be enhanced in the future to pass fp vals in registers
// (when we have a global fp allocator) and do other tricks.
//

/// AddLiveIn - This helper function adds the specified physical register to the
/// MachineFunction as a live in value.  It also creates a corresponding virtual
/// register for it.
static unsigned AddLiveIn(MachineFunction &MF, unsigned PReg,
                          TargetRegisterClass *RC) {
  assert(RC->contains(PReg) && "Not the correct regclass!");
  unsigned VReg = MF.getSSARegMap()->createVirtualRegister(RC);
  MF.addLiveIn(PReg, VReg);
  return VReg;
}


std::vector<SDOperand>
X86TargetLowering::LowerFastCCArguments(Function &F, SelectionDAG &DAG) {
  std::vector<SDOperand> ArgValues;

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Add DAG nodes to load the arguments...  On entry to a function the stack
  // frame looks like this:
  //
  // [ESP] -- return address
  // [ESP + 4] -- first nonreg argument (leftmost lexically)
  // [ESP + 8] -- second nonreg argument, if first argument is 4 bytes in size
  //    ...
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot

  // Keep track of the number of integer regs passed so far.  This can be either
  // 0 (neither EAX or EDX used), 1 (EAX is used) or 2 (EAX and EDX are both
  // used).
  unsigned NumIntRegs = 0;

  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I) {
    MVT::ValueType ObjectVT = getValueType(I->getType());
    unsigned ArgIncrement = 4;
    unsigned ObjSize = 0;
    SDOperand ArgValue;

    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i1:
    case MVT::i8:
      if (NumIntRegs < 2) {
        if (!I->use_empty()) {
          unsigned VReg = AddLiveIn(MF, NumIntRegs ? X86::DL : X86::AL,
                                    X86::R8RegisterClass);
          ArgValue = DAG.getCopyFromReg(DAG.getRoot(), VReg, MVT::i8);
          DAG.setRoot(ArgValue.getValue(1));
        }
        ++NumIntRegs;
        break;
      }

      ObjSize = 1;
      break;
    case MVT::i16:
      if (NumIntRegs < 2) {
        if (!I->use_empty()) {
          unsigned VReg = AddLiveIn(MF, NumIntRegs ? X86::DX : X86::AX,
                                    X86::R16RegisterClass);
          ArgValue = DAG.getCopyFromReg(DAG.getRoot(), VReg, MVT::i16);
          DAG.setRoot(ArgValue.getValue(1));
        }
        ++NumIntRegs;
        break;
      }
      ObjSize = 2;
      break;
    case MVT::i32:
      if (NumIntRegs < 2) {
        if (!I->use_empty()) {
          unsigned VReg = AddLiveIn(MF,NumIntRegs ? X86::EDX : X86::EAX,
                                    X86::R32RegisterClass);
          ArgValue = DAG.getCopyFromReg(DAG.getRoot(), VReg, MVT::i32);
          DAG.setRoot(ArgValue.getValue(1));
        }
        ++NumIntRegs;
        break;
      }
      ObjSize = 4;
      break;
    case MVT::i64:
      if (NumIntRegs == 0) {
        if (!I->use_empty()) {
          unsigned BotReg = AddLiveIn(MF, X86::EAX, X86::R32RegisterClass);
          unsigned TopReg = AddLiveIn(MF, X86::EDX, X86::R32RegisterClass);

          SDOperand Low = DAG.getCopyFromReg(DAG.getRoot(), BotReg, MVT::i32);
          SDOperand Hi  = DAG.getCopyFromReg(Low.getValue(1), TopReg, MVT::i32);
          DAG.setRoot(Hi.getValue(1));

          ArgValue = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Low, Hi);
        }
        NumIntRegs = 2;
        break;
      } else if (NumIntRegs == 1) {
        if (!I->use_empty()) {
          unsigned BotReg = AddLiveIn(MF, X86::EDX, X86::R32RegisterClass);
          SDOperand Low = DAG.getCopyFromReg(DAG.getRoot(), BotReg, MVT::i32);
          DAG.setRoot(Low.getValue(1));

          // Load the high part from memory.
          // Create the frame index object for this incoming parameter...
          int FI = MFI->CreateFixedObject(4, ArgOffset);
          SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);
          SDOperand Hi = DAG.getLoad(MVT::i32, DAG.getEntryNode(), FIN,
                                     DAG.getSrcValue(NULL));
          ArgValue = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Low, Hi);
        }
        ArgOffset += 4;
        NumIntRegs = 2;
        break;
      }
      ObjSize = ArgIncrement = 8;
      break;
    case MVT::f32: ObjSize = 4;                break;
    case MVT::f64: ObjSize = ArgIncrement = 8; break;
    }

    // Don't codegen dead arguments.  FIXME: remove this check when we can nuke
    // dead loads.
    if (ObjSize && !I->use_empty()) {
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);

      // Create the SelectionDAG nodes corresponding to a load from this
      // parameter.
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);

      ArgValue = DAG.getLoad(ObjectVT, DAG.getEntryNode(), FIN,
                             DAG.getSrcValue(NULL));
    } else if (ArgValue.Val == 0) {
      if (MVT::isInteger(ObjectVT))
        ArgValue = DAG.getConstant(0, ObjectVT);
      else
        ArgValue = DAG.getConstantFP(0, ObjectVT);
    }
    ArgValues.push_back(ArgValue);

    if (ObjSize)
      ArgOffset += ArgIncrement;   // Move on to the next argument.
  }

  // Make sure the instruction takes 8n+4 bytes to make sure the start of the
  // arguments and the arguments after the retaddr has been pushed are aligned.
  if ((ArgOffset & 7) == 0)
    ArgOffset += 4;

  VarArgsFrameIndex = 0xAAAAAAA;   // fastcc functions can't have varargs.
  ReturnAddrIndex = 0;             // No return address slot generated yet.
  BytesToPopOnReturn = ArgOffset;  // Callee pops all stack arguments.
  BytesCallerReserves = 0;

  // Finally, inform the code generator which regs we return values in.
  switch (getValueType(F.getReturnType())) {
  default: assert(0 && "Unknown type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    MF.addLiveOut(X86::EAX);
    break;
  case MVT::i64:
    MF.addLiveOut(X86::EAX);
    MF.addLiveOut(X86::EDX);
    break;
  case MVT::f32:
  case MVT::f64:
    MF.addLiveOut(X86::ST0);
    break;
  }
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
X86TargetLowering::LowerFastCCCallTo(SDOperand Chain, const Type *RetTy,
                                     bool isTailCall, SDOperand Callee,
                                     ArgListTy &Args, SelectionDAG &DAG) {
  // Count how many bytes are to be pushed on the stack.
  unsigned NumBytes = 0;

  // Keep track of the number of integer regs passed so far.  This can be either
  // 0 (neither EAX or EDX used), 1 (EAX is used) or 2 (EAX and EDX are both
  // used).
  unsigned NumIntRegs = 0;

  for (unsigned i = 0, e = Args.size(); i != e; ++i)
    switch (getValueType(Args[i].second)) {
    default: assert(0 && "Unknown value type!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      if (NumIntRegs < 2) {
        ++NumIntRegs;
        break;
      }
      // fall through
    case MVT::f32:
      NumBytes += 4;
      break;
    case MVT::i64:
      if (NumIntRegs == 0) {
        NumIntRegs = 2;
        break;
      } else if (NumIntRegs == 1) {
        NumIntRegs = 2;
        NumBytes += 4;
        break;
      }

      // fall through
    case MVT::f64:
      NumBytes += 8;
      break;
    }

  // Make sure the instruction takes 8n+4 bytes to make sure the start of the
  // arguments and the arguments after the retaddr has been pushed are aligned.
  if ((NumBytes & 7) == 0)
    NumBytes += 4;

  Chain = DAG.getNode(ISD::CALLSEQ_START, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));

  // Arguments go on the stack in reverse order, as specified by the ABI.
  unsigned ArgOffset = 0;
  SDOperand StackPtr = DAG.getCopyFromReg(DAG.getEntryNode(),
                                          X86::ESP, MVT::i32);
  NumIntRegs = 0;
  std::vector<SDOperand> Stores;
  std::vector<SDOperand> RegValuesToPass;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    switch (getValueType(Args[i].second)) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      if (NumIntRegs < 2) {
        RegValuesToPass.push_back(Args[i].first);
        ++NumIntRegs;
        break;
      }
      // Fall through
    case MVT::f32: {
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
      Stores.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                   Args[i].first, PtrOff,
                                   DAG.getSrcValue(NULL)));
      ArgOffset += 4;
      break;
    }
    case MVT::i64:
      if (NumIntRegs < 2) {    // Can pass part of it in regs?
        SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32,
                                   Args[i].first, DAG.getConstant(1, MVT::i32));
        SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32,
                                   Args[i].first, DAG.getConstant(0, MVT::i32));
        RegValuesToPass.push_back(Lo);
        ++NumIntRegs;
        if (NumIntRegs < 2) {   // Pass both parts in regs?
          RegValuesToPass.push_back(Hi);
          ++NumIntRegs;
        } else {
          // Pass the high part in memory.
          SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
          PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
          Stores.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                       Hi, PtrOff, DAG.getSrcValue(NULL)));
          ArgOffset += 4;
        }
        break;
      }
      // Fall through
    case MVT::f64:
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
      Stores.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                   Args[i].first, PtrOff,
                                   DAG.getSrcValue(NULL)));
      ArgOffset += 8;
      break;
    }
  }
  if (!Stores.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, Stores);

  // Make sure the instruction takes 8n+4 bytes to make sure the start of the
  // arguments and the arguments after the retaddr has been pushed are aligned.
  if ((ArgOffset & 7) == 0)
    ArgOffset += 4;

  std::vector<MVT::ValueType> RetVals;
  MVT::ValueType RetTyVT = getValueType(RetTy);

  RetVals.push_back(MVT::Other);

  // The result values produced have to be legal.  Promote the result.
  switch (RetTyVT) {
  case MVT::isVoid: break;
  default:
    RetVals.push_back(RetTyVT);
    break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
    RetVals.push_back(MVT::i32);
    break;
  case MVT::f32:
    if (X86ScalarSSE)
      RetVals.push_back(MVT::f32);
    else
      RetVals.push_back(MVT::f64);
    break;
  case MVT::i64:
    RetVals.push_back(MVT::i32);
    RetVals.push_back(MVT::i32);
    break;
  }

  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  Ops.push_back(DAG.getConstant(ArgOffset, getPointerTy()));
  // Callee pops all arg values on the stack.
  Ops.push_back(DAG.getConstant(ArgOffset, getPointerTy()));

  // Pass register arguments as needed.
  Ops.insert(Ops.end(), RegValuesToPass.begin(), RegValuesToPass.end());

  SDOperand TheCall = DAG.getNode(isTailCall ? X86ISD::TAILCALL : X86ISD::CALL,
                                  RetVals, Ops);
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, TheCall);

  SDOperand ResultVal;
  switch (RetTyVT) {
  case MVT::isVoid: break;
  default:
    ResultVal = TheCall.getValue(1);
    break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
    ResultVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, TheCall.getValue(1));
    break;
  case MVT::f32:
    // FIXME: we would really like to remember that this FP_ROUND operation is
    // okay to eliminate if we allow excess FP precision.
    ResultVal = DAG.getNode(ISD::FP_ROUND, MVT::f32, TheCall.getValue(1));
    break;
  case MVT::i64:
    ResultVal = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, TheCall.getValue(1),
                            TheCall.getValue(2));
    break;
  }

  return std::make_pair(ResultVal, Chain);
}

SDOperand X86TargetLowering::getReturnAddressFrameIndex(SelectionDAG &DAG) {
  if (ReturnAddrIndex == 0) {
    // Set up a frame object for the return address.
    MachineFunction &MF = DAG.getMachineFunction();
    ReturnAddrIndex = MF.getFrameInfo()->CreateFixedObject(4, -4);
  }

  return DAG.getFrameIndex(ReturnAddrIndex, MVT::i32);
}



std::pair<SDOperand, SDOperand> X86TargetLowering::
LowerFrameReturnAddress(bool isFrameAddress, SDOperand Chain, unsigned Depth,
                        SelectionDAG &DAG) {
  SDOperand Result;
  if (Depth)        // Depths > 0 not supported yet!
    Result = DAG.getConstant(0, getPointerTy());
  else {
    SDOperand RetAddrFI = getReturnAddressFrameIndex(DAG);
    if (!isFrameAddress)
      // Just load the return address
      Result = DAG.getLoad(MVT::i32, DAG.getEntryNode(), RetAddrFI,
                           DAG.getSrcValue(NULL));
    else
      Result = DAG.getNode(ISD::SUB, MVT::i32, RetAddrFI,
                           DAG.getConstant(4, MVT::i32));
  }
  return std::make_pair(Result, Chain);
}

//===----------------------------------------------------------------------===//
//                           X86 Custom Lowering Hooks
//===----------------------------------------------------------------------===//

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDOperand X86TargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Should not custom lower this!");
  case ISD::SINT_TO_FP: {
    assert(Op.getValueType() == MVT::f64 &&
           Op.getOperand(0).getValueType() == MVT::i64 &&
           "Unknown SINT_TO_FP to lower!");
    // We lower sint64->FP into a store to a temporary stack slot, followed by a
    // FILD64m node.
    MachineFunction &MF = DAG.getMachineFunction();
    int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8);
    SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
    SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, DAG.getEntryNode(),
                           Op.getOperand(0), StackSlot, DAG.getSrcValue(NULL));
    std::vector<MVT::ValueType> RTs;
    RTs.push_back(MVT::f64);
    RTs.push_back(MVT::Other);
    std::vector<SDOperand> Ops;
    Ops.push_back(Store);
    Ops.push_back(StackSlot);
    return DAG.getNode(X86ISD::FILD64m, RTs, Ops);
  }
  case ISD::FP_TO_SINT: {
    assert(Op.getValueType() <= MVT::i64 && Op.getValueType() >= MVT::i16 &&
           Op.getOperand(0).getValueType() == MVT::f64 &&
           "Unknown FP_TO_SINT to lower!");
    // We lower FP->sint64 into FISTP64, followed by a load, all to a temporary
    // stack slot.
    MachineFunction &MF = DAG.getMachineFunction();
    unsigned MemSize = MVT::getSizeInBits(Op.getValueType())/8;
    int SSFI = MF.getFrameInfo()->CreateStackObject(MemSize, MemSize);
    SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());

    unsigned Opc;
    switch (Op.getValueType()) {
    default: assert(0 && "Invalid FP_TO_SINT to lower!");
    case MVT::i16: Opc = X86ISD::FP_TO_INT16_IN_MEM; break;
    case MVT::i32: Opc = X86ISD::FP_TO_INT32_IN_MEM; break;
    case MVT::i64: Opc = X86ISD::FP_TO_INT64_IN_MEM; break;
    }

    // Build the FP_TO_INT*_IN_MEM
    std::vector<SDOperand> Ops;
    Ops.push_back(DAG.getEntryNode());
    Ops.push_back(Op.getOperand(0));
    Ops.push_back(StackSlot);
    SDOperand FIST = DAG.getNode(Opc, MVT::Other, Ops);

    // Load the result.
    return DAG.getLoad(Op.getValueType(), FIST, StackSlot,
                       DAG.getSrcValue(NULL));
  }
  }
}


//===----------------------------------------------------------------------===//
//                      Pattern Matcher Implementation
//===----------------------------------------------------------------------===//

namespace {
  /// X86ISelAddressMode - This corresponds to X86AddressMode, but uses
  /// SDOperand's instead of register numbers for the leaves of the matched
  /// tree.
  struct X86ISelAddressMode {
    enum {
      RegBase,
      FrameIndexBase,
    } BaseType;

    struct {            // This is really a union, discriminated by BaseType!
      SDOperand Reg;
      int FrameIndex;
    } Base;

    unsigned Scale;
    SDOperand IndexReg;
    unsigned Disp;
    GlobalValue *GV;

    X86ISelAddressMode()
      : BaseType(RegBase), Scale(1), IndexReg(), Disp(), GV(0) {
    }
  };
}


namespace {
  Statistic<>
  NumFPKill("x86-codegen", "Number of FP_REG_KILL instructions added");

  //===--------------------------------------------------------------------===//
  /// ISel - X86 specific code to select X86 machine instructions for
  /// SelectionDAG operations.
  ///
  class ISel : public SelectionDAGISel {
    /// ContainsFPCode - Every instruction we select that uses or defines a FP
    /// register should set this to true.
    bool ContainsFPCode;

    /// X86Lowering - This object fully describes how to lower LLVM code to an
    /// X86-specific SelectionDAG.
    X86TargetLowering X86Lowering;

    /// RegPressureMap - This keeps an approximate count of the number of
    /// registers required to evaluate each node in the graph.
    std::map<SDNode*, unsigned> RegPressureMap;

    /// ExprMap - As shared expressions are codegen'd, we keep track of which
    /// vreg the value is produced in, so we only emit one copy of each compiled
    /// tree.
    std::map<SDOperand, unsigned> ExprMap;

    /// TheDAG - The DAG being selected during Select* operations.
    SelectionDAG *TheDAG;

    /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget *Subtarget;
  public:
    ISel(TargetMachine &TM) : SelectionDAGISel(X86Lowering), X86Lowering(TM) {
      Subtarget = &TM.getSubtarget<X86Subtarget>();
    }

    virtual const char *getPassName() const {
      return "X86 Pattern Instruction Selection";
    }

    unsigned getRegPressure(SDOperand O) {
      return RegPressureMap[O.Val];
    }
    unsigned ComputeRegPressure(SDOperand O);

    /// InstructionSelectBasicBlock - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);

    virtual void EmitFunctionEntryCode(Function &Fn, MachineFunction &MF);

    bool isFoldableLoad(SDOperand Op, SDOperand OtherOp,
                        bool FloatPromoteOk = false);
    void EmitFoldedLoad(SDOperand Op, X86AddressMode &AM);
    bool TryToFoldLoadOpStore(SDNode *Node);
    bool EmitOrOpOp(SDOperand Op1, SDOperand Op2, unsigned DestReg);
    void EmitCMP(SDOperand LHS, SDOperand RHS, bool isOnlyUse);
    bool EmitBranchCC(MachineBasicBlock *Dest, SDOperand Chain, SDOperand Cond);
    void EmitSelectCC(SDOperand Cond, SDOperand True, SDOperand False, 
                      MVT::ValueType SVT, unsigned RDest);
    unsigned SelectExpr(SDOperand N);

    X86AddressMode SelectAddrExprs(const X86ISelAddressMode &IAM);
    bool MatchAddress(SDOperand N, X86ISelAddressMode &AM);
    void SelectAddress(SDOperand N, X86AddressMode &AM);
    bool EmitPotentialTailCall(SDNode *Node);
    void EmitFastCCToFastCCTailCall(SDNode *TailCallNode);
    void Select(SDOperand N);
  };
}

/// EmitSpecialCodeForMain - Emit any code that needs to be executed only in
/// the main function.
static void EmitSpecialCodeForMain(MachineBasicBlock *BB,
                                   MachineFrameInfo *MFI) {
  // Switch the FPU to 64-bit precision mode for better compatibility and speed.
  int CWFrameIdx = MFI->CreateStackObject(2, 2);
  addFrameReference(BuildMI(BB, X86::FNSTCW16m, 4), CWFrameIdx);

  // Set the high part to be 64-bit precision.
  addFrameReference(BuildMI(BB, X86::MOV8mi, 5),
                    CWFrameIdx, 1).addImm(2);

  // Reload the modified control word now.
  addFrameReference(BuildMI(BB, X86::FLDCW16m, 4), CWFrameIdx);
}

void ISel::EmitFunctionEntryCode(Function &Fn, MachineFunction &MF) {
  // If this is main, emit special code for main.
  MachineBasicBlock *BB = MF.begin();
  if (Fn.hasExternalLinkage() && Fn.getName() == "main")
    EmitSpecialCodeForMain(BB, MF.getFrameInfo());
}


/// InstructionSelectBasicBlock - This callback is invoked by SelectionDAGISel
/// when it has created a SelectionDAG for us to codegen.
void ISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  // While we're doing this, keep track of whether we see any FP code for
  // FP_REG_KILL insertion.
  ContainsFPCode = false;
  MachineFunction *MF = BB->getParent();

  // Scan the PHI nodes that already are inserted into this basic block.  If any
  // of them is a PHI of a floating point value, we need to insert an
  // FP_REG_KILL.
  SSARegMap *RegMap = MF->getSSARegMap();
  if (BB != MF->begin())
    for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
      assert(I->getOpcode() == X86::PHI &&
             "Isn't just PHI nodes?");
      if (RegMap->getRegClass(I->getOperand(0).getReg()) ==
          X86::RFPRegisterClass) {
        ContainsFPCode = true;
        break;
      }
    }

  // Compute the RegPressureMap, which is an approximation for the number of
  // registers required to compute each node.
  ComputeRegPressure(DAG.getRoot());

  TheDAG = &DAG;

  // Codegen the basic block.
  Select(DAG.getRoot());

  TheDAG = 0;

  // Finally, look at all of the successors of this block.  If any contain a PHI
  // node of FP type, we need to insert an FP_REG_KILL in this block.
  for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
         E = BB->succ_end(); SI != E && !ContainsFPCode; ++SI)
    for (MachineBasicBlock::iterator I = (*SI)->begin(), E = (*SI)->end();
         I != E && I->getOpcode() == X86::PHI; ++I) {
      if (RegMap->getRegClass(I->getOperand(0).getReg()) ==
          X86::RFPRegisterClass) {
        ContainsFPCode = true;
        break;
      }
    }

  // Final check, check LLVM BB's that are successors to the LLVM BB
  // corresponding to BB for FP PHI nodes.
  const BasicBlock *LLVMBB = BB->getBasicBlock();
  const PHINode *PN;
  if (!ContainsFPCode)
    for (succ_const_iterator SI = succ_begin(LLVMBB), E = succ_end(LLVMBB);
         SI != E && !ContainsFPCode; ++SI)
      for (BasicBlock::const_iterator II = SI->begin();
           (PN = dyn_cast<PHINode>(II)); ++II)
        if (PN->getType()->isFloatingPoint()) {
          ContainsFPCode = true;
          break;
        }


  // Insert FP_REG_KILL instructions into basic blocks that need them.  This
  // only occurs due to the floating point stackifier not being aggressive
  // enough to handle arbitrary global stackification.
  //
  // Currently we insert an FP_REG_KILL instruction into each block that uses or
  // defines a floating point virtual register.
  //
  // When the global register allocators (like linear scan) finally update live
  // variable analysis, we can keep floating point values in registers across
  // basic blocks.  This will be a huge win, but we are waiting on the global
  // allocators before we can do this.
  //
  if (ContainsFPCode) {
    BuildMI(*BB, BB->getFirstTerminator(), X86::FP_REG_KILL, 0);
    ++NumFPKill;
  }

  // Clear state used for selection.
  ExprMap.clear();
  RegPressureMap.clear();
}


// ComputeRegPressure - Compute the RegPressureMap, which is an approximation
// for the number of registers required to compute each node.  This is basically
// computing a generalized form of the Sethi-Ullman number for each node.
unsigned ISel::ComputeRegPressure(SDOperand O) {
  SDNode *N = O.Val;
  unsigned &Result = RegPressureMap[N];
  if (Result) return Result;

  // FIXME: Should operations like CALL (which clobber lots o regs) have a
  // higher fixed cost??

  if (N->getNumOperands() == 0) {
    Result = 1;
  } else {
    unsigned MaxRegUse = 0;
    unsigned NumExtraMaxRegUsers = 0;
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
      unsigned Regs;
      if (N->getOperand(i).getOpcode() == ISD::Constant)
        Regs = 0;
      else
        Regs = ComputeRegPressure(N->getOperand(i));
      if (Regs > MaxRegUse) {
        MaxRegUse = Regs;
        NumExtraMaxRegUsers = 0;
      } else if (Regs == MaxRegUse &&
                 N->getOperand(i).getValueType() != MVT::Other) {
        ++NumExtraMaxRegUsers;
      }
    }

    if (O.getOpcode() != ISD::TokenFactor)
      Result = MaxRegUse+NumExtraMaxRegUsers;
    else
      Result = MaxRegUse == 1 ? 0 : MaxRegUse-1;
  }

  //std::cerr << " WEIGHT: " << Result << " ";  N->dump(); std::cerr << "\n";
  return Result;
}

/// NodeTransitivelyUsesValue - Return true if N or any of its uses uses Op.
/// The DAG cannot have cycles in it, by definition, so the visited set is not
/// needed to prevent infinite loops.  The DAG CAN, however, have unbounded
/// reuse, so it prevents exponential cases.
///
static bool NodeTransitivelyUsesValue(SDOperand N, SDOperand Op,
                                      std::set<SDNode*> &Visited) {
  if (N == Op) return true;                        // Found it.
  SDNode *Node = N.Val;
  if (Node->getNumOperands() == 0 ||      // Leaf?
      Node->getNodeDepth() <= Op.getNodeDepth()) return false; // Can't find it?
  if (!Visited.insert(Node).second) return false;  // Already visited?

  // Recurse for the first N-1 operands.
  for (unsigned i = 1, e = Node->getNumOperands(); i != e; ++i)
    if (NodeTransitivelyUsesValue(Node->getOperand(i), Op, Visited))
      return true;

  // Tail recurse for the last operand.
  return NodeTransitivelyUsesValue(Node->getOperand(0), Op, Visited);
}

X86AddressMode ISel::SelectAddrExprs(const X86ISelAddressMode &IAM) {
  X86AddressMode Result;

  // If we need to emit two register operands, emit the one with the highest
  // register pressure first.
  if (IAM.BaseType == X86ISelAddressMode::RegBase &&
      IAM.Base.Reg.Val && IAM.IndexReg.Val) {
    bool EmitBaseThenIndex;
    if (getRegPressure(IAM.Base.Reg) > getRegPressure(IAM.IndexReg)) {
      std::set<SDNode*> Visited;
      EmitBaseThenIndex = true;
      // If Base ends up pointing to Index, we must emit index first.  This is
      // because of the way we fold loads, we may end up doing bad things with
      // the folded add.
      if (NodeTransitivelyUsesValue(IAM.Base.Reg, IAM.IndexReg, Visited))
        EmitBaseThenIndex = false;
    } else {
      std::set<SDNode*> Visited;
      EmitBaseThenIndex = false;
      // If Base ends up pointing to Index, we must emit index first.  This is
      // because of the way we fold loads, we may end up doing bad things with
      // the folded add.
      if (NodeTransitivelyUsesValue(IAM.IndexReg, IAM.Base.Reg, Visited))
        EmitBaseThenIndex = true;
    }

    if (EmitBaseThenIndex) {
      Result.Base.Reg = SelectExpr(IAM.Base.Reg);
      Result.IndexReg = SelectExpr(IAM.IndexReg);
    } else {
      Result.IndexReg = SelectExpr(IAM.IndexReg);
      Result.Base.Reg = SelectExpr(IAM.Base.Reg);
    }

  } else if (IAM.BaseType == X86ISelAddressMode::RegBase && IAM.Base.Reg.Val) {
    Result.Base.Reg = SelectExpr(IAM.Base.Reg);
  } else if (IAM.IndexReg.Val) {
    Result.IndexReg = SelectExpr(IAM.IndexReg);
  }

  switch (IAM.BaseType) {
  case X86ISelAddressMode::RegBase:
    Result.BaseType = X86AddressMode::RegBase;
    break;
  case X86ISelAddressMode::FrameIndexBase:
    Result.BaseType = X86AddressMode::FrameIndexBase;
    Result.Base.FrameIndex = IAM.Base.FrameIndex;
    break;
  default:
    assert(0 && "Unknown base type!");
    break;
  }
  Result.Scale = IAM.Scale;
  Result.Disp = IAM.Disp;
  Result.GV = IAM.GV;
  return Result;
}

/// SelectAddress - Pattern match the maximal addressing mode for this node and
/// emit all of the leaf registers.
void ISel::SelectAddress(SDOperand N, X86AddressMode &AM) {
  X86ISelAddressMode IAM;
  MatchAddress(N, IAM);
  AM = SelectAddrExprs(IAM);
}

/// MatchAddress - Add the specified node to the specified addressing mode,
/// returning true if it cannot be done.  This just pattern matches for the
/// addressing mode, it does not cause any code to be emitted.  For that, use
/// SelectAddress.
bool ISel::MatchAddress(SDOperand N, X86ISelAddressMode &AM) {
  switch (N.getOpcode()) {
  default: break;
  case ISD::FrameIndex:
    if (AM.BaseType == X86ISelAddressMode::RegBase && AM.Base.Reg.Val == 0) {
      AM.BaseType = X86ISelAddressMode::FrameIndexBase;
      AM.Base.FrameIndex = cast<FrameIndexSDNode>(N)->getIndex();
      return false;
    }
    break;
  case ISD::GlobalAddress:
    if (AM.GV == 0) {
      GlobalValue *GV = cast<GlobalAddressSDNode>(N)->getGlobal();
      // For Darwin, external and weak symbols are indirect, so we want to load
      // the value at address GV, not the value of GV itself.  This means that
      // the GlobalAddress must be in the base or index register of the address,
      // not the GV offset field.
      if (Subtarget->getIndirectExternAndWeakGlobals() &&
          (GV->hasWeakLinkage() || GV->isExternal())) {
        break;
      } else {
        AM.GV = GV;
        return false;
      }
    }
    break;
  case ISD::Constant:
    AM.Disp += cast<ConstantSDNode>(N)->getValue();
    return false;
  case ISD::SHL:
    // We might have folded the load into this shift, so don't regen the value
    // if so.
    if (ExprMap.count(N)) break;

    if (AM.IndexReg.Val == 0 && AM.Scale == 1)
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.Val->getOperand(1))) {
        unsigned Val = CN->getValue();
        if (Val == 1 || Val == 2 || Val == 3) {
          AM.Scale = 1 << Val;
          SDOperand ShVal = N.Val->getOperand(0);

          // Okay, we know that we have a scale by now.  However, if the scaled
          // value is an add of something and a constant, we can fold the
          // constant into the disp field here.
          if (ShVal.Val->getOpcode() == ISD::ADD && ShVal.hasOneUse() &&
              isa<ConstantSDNode>(ShVal.Val->getOperand(1))) {
            AM.IndexReg = ShVal.Val->getOperand(0);
            ConstantSDNode *AddVal =
              cast<ConstantSDNode>(ShVal.Val->getOperand(1));
            AM.Disp += AddVal->getValue() << Val;
          } else {
            AM.IndexReg = ShVal;
          }
          return false;
        }
      }
    break;
  case ISD::MUL:
    // We might have folded the load into this mul, so don't regen the value if
    // so.
    if (ExprMap.count(N)) break;

    // X*[3,5,9] -> X+X*[2,4,8]
    if (AM.IndexReg.Val == 0 && AM.BaseType == X86ISelAddressMode::RegBase &&
        AM.Base.Reg.Val == 0)
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.Val->getOperand(1)))
        if (CN->getValue() == 3 || CN->getValue() == 5 || CN->getValue() == 9) {
          AM.Scale = unsigned(CN->getValue())-1;

          SDOperand MulVal = N.Val->getOperand(0);
          SDOperand Reg;

          // Okay, we know that we have a scale by now.  However, if the scaled
          // value is an add of something and a constant, we can fold the
          // constant into the disp field here.
          if (MulVal.Val->getOpcode() == ISD::ADD && MulVal.hasOneUse() &&
              isa<ConstantSDNode>(MulVal.Val->getOperand(1))) {
            Reg = MulVal.Val->getOperand(0);
            ConstantSDNode *AddVal =
              cast<ConstantSDNode>(MulVal.Val->getOperand(1));
            AM.Disp += AddVal->getValue() * CN->getValue();
          } else {
            Reg = N.Val->getOperand(0);
          }

          AM.IndexReg = AM.Base.Reg = Reg;
          return false;
        }
    break;

  case ISD::ADD: {
    // We might have folded the load into this mul, so don't regen the value if
    // so.
    if (ExprMap.count(N)) break;

    X86ISelAddressMode Backup = AM;
    if (!MatchAddress(N.Val->getOperand(0), AM) &&
        !MatchAddress(N.Val->getOperand(1), AM))
      return false;
    AM = Backup;
    if (!MatchAddress(N.Val->getOperand(1), AM) &&
        !MatchAddress(N.Val->getOperand(0), AM))
      return false;
    AM = Backup;
    break;
  }
  }

  // Is the base register already occupied?
  if (AM.BaseType != X86ISelAddressMode::RegBase || AM.Base.Reg.Val) {
    // If so, check to see if the scale index register is set.
    if (AM.IndexReg.Val == 0) {
      AM.IndexReg = N;
      AM.Scale = 1;
      return false;
    }

    // Otherwise, we cannot select it.
    return true;
  }

  // Default, generate it as a register.
  AM.BaseType = X86ISelAddressMode::RegBase;
  AM.Base.Reg = N;
  return false;
}

/// Emit2SetCCsAndLogical - Emit the following sequence of instructions,
/// assuming that the temporary registers are in the 8-bit register class.
///
///  Tmp1 = setcc1
///  Tmp2 = setcc2
///  DestReg = logicalop Tmp1, Tmp2
///
static void Emit2SetCCsAndLogical(MachineBasicBlock *BB, unsigned SetCC1,
                                  unsigned SetCC2, unsigned LogicalOp,
                                  unsigned DestReg) {
  SSARegMap *RegMap = BB->getParent()->getSSARegMap();
  unsigned Tmp1 = RegMap->createVirtualRegister(X86::R8RegisterClass);
  unsigned Tmp2 = RegMap->createVirtualRegister(X86::R8RegisterClass);
  BuildMI(BB, SetCC1, 0, Tmp1);
  BuildMI(BB, SetCC2, 0, Tmp2);
  BuildMI(BB, LogicalOp, 2, DestReg).addReg(Tmp1).addReg(Tmp2);
}

/// EmitSetCC - Emit the code to set the specified 8-bit register to 1 if the
/// condition codes match the specified SetCCOpcode.  Note that some conditions
/// require multiple instructions to generate the correct value.
static void EmitSetCC(MachineBasicBlock *BB, unsigned DestReg,
                      ISD::CondCode SetCCOpcode, bool isFP) {
  unsigned Opc;
  if (!isFP) {
    switch (SetCCOpcode) {
    default: assert(0 && "Illegal integer SetCC!");
    case ISD::SETEQ: Opc = X86::SETEr; break;
    case ISD::SETGT: Opc = X86::SETGr; break;
    case ISD::SETGE: Opc = X86::SETGEr; break;
    case ISD::SETLT: Opc = X86::SETLr; break;
    case ISD::SETLE: Opc = X86::SETLEr; break;
    case ISD::SETNE: Opc = X86::SETNEr; break;
    case ISD::SETULT: Opc = X86::SETBr; break;
    case ISD::SETUGT: Opc = X86::SETAr; break;
    case ISD::SETULE: Opc = X86::SETBEr; break;
    case ISD::SETUGE: Opc = X86::SETAEr; break;
    }
  } else {
    // On a floating point condition, the flags are set as follows:
    // ZF  PF  CF   op
    //  0 | 0 | 0 | X > Y
    //  0 | 0 | 1 | X < Y
    //  1 | 0 | 0 | X == Y
    //  1 | 1 | 1 | unordered
    //
    switch (SetCCOpcode) {
    default: assert(0 && "Invalid FP setcc!");
    case ISD::SETUEQ:
    case ISD::SETEQ:
      Opc = X86::SETEr;    // True if ZF = 1
      break;
    case ISD::SETOGT:
    case ISD::SETGT:
      Opc = X86::SETAr;    // True if CF = 0 and ZF = 0
      break;
    case ISD::SETOGE:
    case ISD::SETGE:
      Opc = X86::SETAEr;   // True if CF = 0
      break;
    case ISD::SETULT:
    case ISD::SETLT:
      Opc = X86::SETBr;    // True if CF = 1
      break;
    case ISD::SETULE:
    case ISD::SETLE:
      Opc = X86::SETBEr;   // True if CF = 1 or ZF = 1
      break;
    case ISD::SETONE:
    case ISD::SETNE:
      Opc = X86::SETNEr;   // True if ZF = 0
      break;
    case ISD::SETUO:
      Opc = X86::SETPr;    // True if PF = 1
      break;
    case ISD::SETO:
      Opc = X86::SETNPr;   // True if PF = 0
      break;
    case ISD::SETOEQ:      // !PF & ZF
      Emit2SetCCsAndLogical(BB, X86::SETNPr, X86::SETEr, X86::AND8rr, DestReg);
      return;
    case ISD::SETOLT:      // !PF & CF
      Emit2SetCCsAndLogical(BB, X86::SETNPr, X86::SETBr, X86::AND8rr, DestReg);
      return;
    case ISD::SETOLE:      // !PF & (CF || ZF)
      Emit2SetCCsAndLogical(BB, X86::SETNPr, X86::SETBEr, X86::AND8rr, DestReg);
      return;
    case ISD::SETUGT:      // PF | (!ZF & !CF)
      Emit2SetCCsAndLogical(BB, X86::SETPr, X86::SETAr, X86::OR8rr, DestReg);
      return;
    case ISD::SETUGE:      // PF | !CF
      Emit2SetCCsAndLogical(BB, X86::SETPr, X86::SETAEr, X86::OR8rr, DestReg);
      return;
    case ISD::SETUNE:      // PF | !ZF
      Emit2SetCCsAndLogical(BB, X86::SETPr, X86::SETNEr, X86::OR8rr, DestReg);
      return;
    }
  }
  BuildMI(BB, Opc, 0, DestReg);
}


/// EmitBranchCC - Emit code into BB that arranges for control to transfer to
/// the Dest block if the Cond condition is true.  If we cannot fold this
/// condition into the branch, return true.
///
bool ISel::EmitBranchCC(MachineBasicBlock *Dest, SDOperand Chain,
                        SDOperand Cond) {
  // FIXME: Evaluate whether it would be good to emit code like (X < Y) | (A >
  // B) using two conditional branches instead of one condbr, two setcc's, and
  // an or.
  if ((Cond.getOpcode() == ISD::OR ||
       Cond.getOpcode() == ISD::AND) && Cond.Val->hasOneUse()) {
    // And and or set the flags for us, so there is no need to emit a TST of the
    // result.  It is only safe to do this if there is only a single use of the
    // AND/OR though, otherwise we don't know it will be emitted here.
    Select(Chain);
    SelectExpr(Cond);
    BuildMI(BB, X86::JNE, 1).addMBB(Dest);
    return false;
  }

  // Codegen br not C -> JE.
  if (Cond.getOpcode() == ISD::XOR)
    if (ConstantSDNode *NC = dyn_cast<ConstantSDNode>(Cond.Val->getOperand(1)))
      if (NC->isAllOnesValue()) {
        unsigned CondR;
        if (getRegPressure(Chain) > getRegPressure(Cond)) {
          Select(Chain);
          CondR = SelectExpr(Cond.Val->getOperand(0));
        } else {
          CondR = SelectExpr(Cond.Val->getOperand(0));
          Select(Chain);
        }
        BuildMI(BB, X86::TEST8rr, 2).addReg(CondR).addReg(CondR);
        BuildMI(BB, X86::JE, 1).addMBB(Dest);
        return false;
      }

  if (Cond.getOpcode() != ISD::SETCC)
    return true;                       // Can only handle simple setcc's so far.
  ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();

  unsigned Opc;

  // Handle integer conditions first.
  if (MVT::isInteger(Cond.getOperand(0).getValueType())) {
    switch (CC) {
    default: assert(0 && "Illegal integer SetCC!");
    case ISD::SETEQ: Opc = X86::JE; break;
    case ISD::SETGT: Opc = X86::JG; break;
    case ISD::SETGE: Opc = X86::JGE; break;
    case ISD::SETLT: Opc = X86::JL; break;
    case ISD::SETLE: Opc = X86::JLE; break;
    case ISD::SETNE: Opc = X86::JNE; break;
    case ISD::SETULT: Opc = X86::JB; break;
    case ISD::SETUGT: Opc = X86::JA; break;
    case ISD::SETULE: Opc = X86::JBE; break;
    case ISD::SETUGE: Opc = X86::JAE; break;
    }
    Select(Chain);
    EmitCMP(Cond.getOperand(0), Cond.getOperand(1), Cond.hasOneUse());
    BuildMI(BB, Opc, 1).addMBB(Dest);
    return false;
  }

  unsigned Opc2 = 0;  // Second branch if needed.

  // On a floating point condition, the flags are set as follows:
  // ZF  PF  CF   op
  //  0 | 0 | 0 | X > Y
  //  0 | 0 | 1 | X < Y
  //  1 | 0 | 0 | X == Y
  //  1 | 1 | 1 | unordered
  //
  switch (CC) {
  default: assert(0 && "Invalid FP setcc!");
  case ISD::SETUEQ:
  case ISD::SETEQ:   Opc = X86::JE;  break;     // True if ZF = 1
  case ISD::SETOGT:
  case ISD::SETGT:   Opc = X86::JA;  break;     // True if CF = 0 and ZF = 0
  case ISD::SETOGE:
  case ISD::SETGE:   Opc = X86::JAE; break;     // True if CF = 0
  case ISD::SETULT:
  case ISD::SETLT:   Opc = X86::JB;  break;     // True if CF = 1
  case ISD::SETULE:
  case ISD::SETLE:   Opc = X86::JBE; break;     // True if CF = 1 or ZF = 1
  case ISD::SETONE:
  case ISD::SETNE:   Opc = X86::JNE; break;     // True if ZF = 0
  case ISD::SETUO:   Opc = X86::JP;  break;     // True if PF = 1
  case ISD::SETO:    Opc = X86::JNP; break;     // True if PF = 0
  case ISD::SETUGT:      // PF = 1 | (ZF = 0 & CF = 0)
    Opc = X86::JA;       // ZF = 0 & CF = 0
    Opc2 = X86::JP;      // PF = 1
    break;
  case ISD::SETUGE:      // PF = 1 | CF = 0
    Opc = X86::JAE;      // CF = 0
    Opc2 = X86::JP;      // PF = 1
    break;
  case ISD::SETUNE:      // PF = 1 | ZF = 0
    Opc = X86::JNE;      // ZF = 0
    Opc2 = X86::JP;      // PF = 1
    break;
  case ISD::SETOEQ:      // PF = 0 & ZF = 1
    //X86::JNP, X86::JE
    //X86::AND8rr
    return true;    // FIXME: Emit more efficient code for this branch.
  case ISD::SETOLT:      // PF = 0 & CF = 1
    //X86::JNP, X86::JB
    //X86::AND8rr
    return true;    // FIXME: Emit more efficient code for this branch.
  case ISD::SETOLE:      // PF = 0 & (CF = 1 || ZF = 1)
    //X86::JNP, X86::JBE
    //X86::AND8rr
    return true;    // FIXME: Emit more efficient code for this branch.
  }

  Select(Chain);
  EmitCMP(Cond.getOperand(0), Cond.getOperand(1), Cond.hasOneUse());
  BuildMI(BB, Opc, 1).addMBB(Dest);
  if (Opc2)
    BuildMI(BB, Opc2, 1).addMBB(Dest);
  return false;
}

/// EmitSelectCC - Emit code into BB that performs a select operation between
/// the two registers RTrue and RFalse, generating a result into RDest.
///
void ISel::EmitSelectCC(SDOperand Cond, SDOperand True, SDOperand False,
                        MVT::ValueType SVT, unsigned RDest) {
  unsigned RTrue, RFalse;
  enum Condition {
    EQ, NE, LT, LE, GT, GE, B, BE, A, AE, P, NP,
    NOT_SET
  } CondCode = NOT_SET;

  static const unsigned CMOVTAB16[] = {
    X86::CMOVE16rr,  X86::CMOVNE16rr, X86::CMOVL16rr,  X86::CMOVLE16rr,
    X86::CMOVG16rr,  X86::CMOVGE16rr, X86::CMOVB16rr,  X86::CMOVBE16rr,
    X86::CMOVA16rr,  X86::CMOVAE16rr, X86::CMOVP16rr,  X86::CMOVNP16rr,
  };
  static const unsigned CMOVTAB32[] = {
    X86::CMOVE32rr,  X86::CMOVNE32rr, X86::CMOVL32rr,  X86::CMOVLE32rr,
    X86::CMOVG32rr,  X86::CMOVGE32rr, X86::CMOVB32rr,  X86::CMOVBE32rr,
    X86::CMOVA32rr,  X86::CMOVAE32rr, X86::CMOVP32rr,  X86::CMOVNP32rr,
  };
  static const unsigned CMOVTABFP[] = {
    X86::FCMOVE ,  X86::FCMOVNE, /*missing*/0, /*missing*/0,
    /*missing*/0,  /*missing*/0, X86::FCMOVB , X86::FCMOVBE,
    X86::FCMOVA ,  X86::FCMOVAE, X86::FCMOVP , X86::FCMOVNP
  };
  static const int SSE_CMOVTAB[] = {
    /*CMPEQ*/   0, /*CMPNEQ*/   4, /*missing*/  0, /*missing*/  0,
    /*missing*/ 0, /*missing*/  0, /*CMPLT*/    1, /*CMPLE*/    2,
    /*CMPNLE*/  6, /*CMPNLT*/   5, /*CMPUNORD*/ 3, /*CMPORD*/   7
  };
  
  if (Cond.getOpcode() == ISD::SETCC) {
    ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
    if (MVT::isInteger(Cond.getOperand(0).getValueType())) {
      switch (CC) {
      default: assert(0 && "Unknown integer comparison!");
      case ISD::SETEQ:  CondCode = EQ; break;
      case ISD::SETGT:  CondCode = GT; break;
      case ISD::SETGE:  CondCode = GE; break;
      case ISD::SETLT:  CondCode = LT; break;
      case ISD::SETLE:  CondCode = LE; break;
      case ISD::SETNE:  CondCode = NE; break;
      case ISD::SETULT: CondCode = B; break;
      case ISD::SETUGT: CondCode = A; break;
      case ISD::SETULE: CondCode = BE; break;
      case ISD::SETUGE: CondCode = AE; break;
      }
    } else {
      // On a floating point condition, the flags are set as follows:
      // ZF  PF  CF   op
      //  0 | 0 | 0 | X > Y
      //  0 | 0 | 1 | X < Y
      //  1 | 0 | 0 | X == Y
      //  1 | 1 | 1 | unordered
      //
      switch (CC) {
      default: assert(0 && "Unknown FP comparison!");
      case ISD::SETUEQ:
      case ISD::SETEQ:  CondCode = EQ; break;     // True if ZF = 1
      case ISD::SETOGT:
      case ISD::SETGT:  CondCode = A;  break;     // True if CF = 0 and ZF = 0
      case ISD::SETOGE:
      case ISD::SETGE:  CondCode = AE; break;     // True if CF = 0
      case ISD::SETULT:
      case ISD::SETLT:  CondCode = B;  break;     // True if CF = 1
      case ISD::SETULE:
      case ISD::SETLE:  CondCode = BE; break;     // True if CF = 1 or ZF = 1
      case ISD::SETONE:
      case ISD::SETNE:  CondCode = NE; break;     // True if ZF = 0
      case ISD::SETUO:  CondCode = P;  break;     // True if PF = 1
      case ISD::SETO:   CondCode = NP; break;     // True if PF = 0
      case ISD::SETUGT:      // PF = 1 | (ZF = 0 & CF = 0)
      case ISD::SETUGE:      // PF = 1 | CF = 0
      case ISD::SETUNE:      // PF = 1 | ZF = 0
      case ISD::SETOEQ:      // PF = 0 & ZF = 1
      case ISD::SETOLT:      // PF = 0 & CF = 1
      case ISD::SETOLE:      // PF = 0 & (CF = 1 || ZF = 1)
        // We cannot emit this comparison as a single cmov.
        break;
      }
    }
  

    // There's no SSE equivalent of FCMOVE.  For cases where we set a condition
    // code above and one of the results of the select is +0.0, then we can fake
    // it up through a clever AND with mask.  Otherwise, we will fall through to
    // the code below that will use a PHI node to select the right value.
    if (X86ScalarSSE && (SVT == MVT::f32 || SVT == MVT::f64)) {
      if (Cond.getOperand(0).getValueType() == SVT && 
          NOT_SET != CondCode) {
        ConstantFPSDNode *CT = dyn_cast<ConstantFPSDNode>(True);
        ConstantFPSDNode *CF = dyn_cast<ConstantFPSDNode>(False);
        bool TrueZero = CT && CT->isExactlyValue(0.0);
        bool FalseZero = CF && CF->isExactlyValue(0.0);
        if (TrueZero || FalseZero) {
          SDOperand LHS = Cond.getOperand(0);
          SDOperand RHS = Cond.getOperand(1);
          
          // Select the two halves of the condition
          unsigned RLHS, RRHS;
          if (getRegPressure(LHS) > getRegPressure(RHS)) {
            RLHS = SelectExpr(LHS);
            RRHS = SelectExpr(RHS);
          } else {
            RRHS = SelectExpr(RHS);
            RLHS = SelectExpr(LHS);
          }
          
          // Emit the comparison and generate a mask from it
          unsigned MaskReg = MakeReg(SVT);
          unsigned Opc = (SVT == MVT::f32) ? X86::CMPSSrr : X86::CMPSDrr;
          BuildMI(BB, Opc, 3, MaskReg).addReg(RLHS).addReg(RRHS)
            .addImm(SSE_CMOVTAB[CondCode]);
          
          if (TrueZero) {
            RFalse = SelectExpr(False);
            Opc = (SVT == MVT::f32) ? X86::ANDNPSrr : X86::ANDNPDrr;
            BuildMI(BB, Opc, 2, RDest).addReg(MaskReg).addReg(RFalse);
          } else {
            RTrue = SelectExpr(True);
            Opc = (SVT == MVT::f32) ? X86::ANDPSrr : X86::ANDPDrr;
            BuildMI(BB, Opc, 2, RDest).addReg(MaskReg).addReg(RTrue);
          }
          return;
        }
      }
    }
  }
    
  // Select the true and false values for use in both the SSE PHI case, and the
  // integer or x87 cmov cases below.
  if (getRegPressure(True) > getRegPressure(False)) {
    RTrue = SelectExpr(True);
    RFalse = SelectExpr(False);
  } else {
    RFalse = SelectExpr(False);
    RTrue = SelectExpr(True);
  }

  // Since there's no SSE equivalent of FCMOVE, and we couldn't generate an
  // AND with mask, we'll have to do the normal RISC thing and generate a PHI
  // node to select between the true and false values.
  if (X86ScalarSSE && (SVT == MVT::f32 || SVT == MVT::f64)) {
    // FIXME: emit a direct compare and branch rather than setting a cond reg
    //        and testing it.
    unsigned CondReg = SelectExpr(Cond);
    BuildMI(BB, X86::TEST8rr, 2).addReg(CondReg).addReg(CondReg);
    
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
    //   bCC sinkMBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
    BuildMI(BB, X86::JNE, 1).addMBB(sinkMBB);
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
    BuildMI(BB, X86::PHI, 4, RDest).addReg(RFalse)
      .addMBB(copy0MBB).addReg(RTrue).addMBB(thisMBB);
    return;
  }

  unsigned Opc = 0;
  if (CondCode != NOT_SET) {
    switch (SVT) {
    default: assert(0 && "Cannot select this type!");
    case MVT::i16: Opc = CMOVTAB16[CondCode]; break;
    case MVT::i32: Opc = CMOVTAB32[CondCode]; break;
    case MVT::f64: Opc = CMOVTABFP[CondCode]; break;
    }
  }

  // Finally, if we weren't able to fold this, just emit the condition and test
  // it.
  if (CondCode == NOT_SET || Opc == 0) {
    // Get the condition into the zero flag.
    unsigned CondReg = SelectExpr(Cond);
    BuildMI(BB, X86::TEST8rr, 2).addReg(CondReg).addReg(CondReg);

    switch (SVT) {
    default: assert(0 && "Cannot select this type!");
    case MVT::i16: Opc = X86::CMOVE16rr; break;
    case MVT::i32: Opc = X86::CMOVE32rr; break;
    case MVT::f64: Opc = X86::FCMOVE; break;
    }
  } else {
    // FIXME: CMP R, 0 -> TEST R, R
    EmitCMP(Cond.getOperand(0), Cond.getOperand(1), Cond.Val->hasOneUse());
    std::swap(RTrue, RFalse);
  }
  BuildMI(BB, Opc, 2, RDest).addReg(RTrue).addReg(RFalse);
}

void ISel::EmitCMP(SDOperand LHS, SDOperand RHS, bool HasOneUse) {
  unsigned Opc;
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(RHS)) {
    Opc = 0;
    if (HasOneUse && isFoldableLoad(LHS, RHS)) {
      switch (RHS.getValueType()) {
      default: break;
      case MVT::i1:
      case MVT::i8:  Opc = X86::CMP8mi;  break;
      case MVT::i16: Opc = X86::CMP16mi; break;
      case MVT::i32: Opc = X86::CMP32mi; break;
      }
      if (Opc) {
        X86AddressMode AM;
        EmitFoldedLoad(LHS, AM);
        addFullAddress(BuildMI(BB, Opc, 5), AM).addImm(CN->getValue());
        return;
      }
    }

    switch (RHS.getValueType()) {
    default: break;
    case MVT::i1:
    case MVT::i8:  Opc = X86::CMP8ri;  break;
    case MVT::i16: Opc = X86::CMP16ri; break;
    case MVT::i32: Opc = X86::CMP32ri; break;
    }
    if (Opc) {
      unsigned Tmp1 = SelectExpr(LHS);
      BuildMI(BB, Opc, 2).addReg(Tmp1).addImm(CN->getValue());
      return;
    }
  } else if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(RHS)) {
    if (!X86ScalarSSE && (CN->isExactlyValue(+0.0) ||
                          CN->isExactlyValue(-0.0))) {
      unsigned Reg = SelectExpr(LHS);
      BuildMI(BB, X86::FTST, 1).addReg(Reg);
      BuildMI(BB, X86::FNSTSW8r, 0);
      BuildMI(BB, X86::SAHF, 1);
      return;
    }
  }

  Opc = 0;
  if (HasOneUse && isFoldableLoad(LHS, RHS)) {
    switch (RHS.getValueType()) {
    default: break;
    case MVT::i1:
    case MVT::i8:  Opc = X86::CMP8mr;  break;
    case MVT::i16: Opc = X86::CMP16mr; break;
    case MVT::i32: Opc = X86::CMP32mr; break;
    }
    if (Opc) {
      X86AddressMode AM;
      EmitFoldedLoad(LHS, AM);
      unsigned Reg = SelectExpr(RHS);
      addFullAddress(BuildMI(BB, Opc, 5), AM).addReg(Reg);
      return;
    }
  }

  switch (LHS.getValueType()) {
  default: assert(0 && "Cannot compare this value!");
  case MVT::i1:
  case MVT::i8:  Opc = X86::CMP8rr;  break;
  case MVT::i16: Opc = X86::CMP16rr; break;
  case MVT::i32: Opc = X86::CMP32rr; break;
  case MVT::f32: Opc = X86::UCOMISSrr; break;
  case MVT::f64: Opc = X86ScalarSSE ? X86::UCOMISDrr : X86::FUCOMIr; break;
  }
  unsigned Tmp1, Tmp2;
  if (getRegPressure(LHS) > getRegPressure(RHS)) {
    Tmp1 = SelectExpr(LHS);
    Tmp2 = SelectExpr(RHS);
  } else {
    Tmp2 = SelectExpr(RHS);
    Tmp1 = SelectExpr(LHS);
  }
  BuildMI(BB, Opc, 2).addReg(Tmp1).addReg(Tmp2);
}

/// isFoldableLoad - Return true if this is a load instruction that can safely
/// be folded into an operation that uses it.
bool ISel::isFoldableLoad(SDOperand Op, SDOperand OtherOp, bool FloatPromoteOk){
  if (Op.getOpcode() == ISD::LOAD) {
    // FIXME: currently can't fold constant pool indexes.
    if (isa<ConstantPoolSDNode>(Op.getOperand(1)))
      return false;
  } else if (FloatPromoteOk && Op.getOpcode() == ISD::EXTLOAD &&
             cast<VTSDNode>(Op.getOperand(3))->getVT() == MVT::f32) {
    // FIXME: currently can't fold constant pool indexes.
    if (isa<ConstantPoolSDNode>(Op.getOperand(1)))
      return false;
  } else {
    return false;
  }

  // If this load has already been emitted, we clearly can't fold it.
  assert(Op.ResNo == 0 && "Not a use of the value of the load?");
  if (ExprMap.count(Op.getValue(1))) return false;
  assert(!ExprMap.count(Op.getValue(0)) && "Value in map but not token chain?");
  assert(!ExprMap.count(Op.getValue(1))&&"Token lowered but value not in map?");

  // If there is not just one use of its value, we cannot fold.
  if (!Op.Val->hasNUsesOfValue(1, 0)) return false;

  // Finally, we cannot fold the load into the operation if this would induce a
  // cycle into the resultant dag.  To check for this, see if OtherOp (the other
  // operand of the operation we are folding the load into) can possible use the
  // chain node defined by the load.
  if (OtherOp.Val && !Op.Val->hasNUsesOfValue(0, 1)) { // Has uses of chain?
    std::set<SDNode*> Visited;
    if (NodeTransitivelyUsesValue(OtherOp, Op.getValue(1), Visited))
      return false;
  }
  return true;
}


/// EmitFoldedLoad - Ensure that the arguments of the load are code generated,
/// and compute the address being loaded into AM.
void ISel::EmitFoldedLoad(SDOperand Op, X86AddressMode &AM) {
  SDOperand Chain   = Op.getOperand(0);
  SDOperand Address = Op.getOperand(1);

  if (getRegPressure(Chain) > getRegPressure(Address)) {
    Select(Chain);
    SelectAddress(Address, AM);
  } else {
    SelectAddress(Address, AM);
    Select(Chain);
  }

  // The chain for this load is now lowered.
  assert(ExprMap.count(SDOperand(Op.Val, 1)) == 0 &&
         "Load emitted more than once?");
  if (!ExprMap.insert(std::make_pair(Op.getValue(1), 1)).second)
    assert(0 && "Load emitted more than once!");
}

// EmitOrOpOp - Pattern match the expression (Op1|Op2), where we know that op1
// and op2 are i8/i16/i32 values with one use each (the or).  If we can form a
// SHLD or SHRD, emit the instruction (generating the value into DestReg) and
// return true.
bool ISel::EmitOrOpOp(SDOperand Op1, SDOperand Op2, unsigned DestReg) {
  if (Op1.getOpcode() == ISD::SHL && Op2.getOpcode() == ISD::SRL) {
    // good!
  } else if (Op2.getOpcode() == ISD::SHL && Op1.getOpcode() == ISD::SRL) {
    std::swap(Op1, Op2);  // Op1 is the SHL now.
  } else {
    return false;  // No match
  }

  SDOperand ShlVal = Op1.getOperand(0);
  SDOperand ShlAmt = Op1.getOperand(1);
  SDOperand ShrVal = Op2.getOperand(0);
  SDOperand ShrAmt = Op2.getOperand(1);

  unsigned RegSize = MVT::getSizeInBits(Op1.getValueType());

  // Find out if ShrAmt = 32-ShlAmt  or  ShlAmt = 32-ShrAmt.
  if (ShlAmt.getOpcode() == ISD::SUB && ShlAmt.getOperand(1) == ShrAmt)
    if (ConstantSDNode *SubCST = dyn_cast<ConstantSDNode>(ShlAmt.getOperand(0)))
      if (SubCST->getValue() == RegSize) {
        // (A >> ShrAmt) | (A << (32-ShrAmt)) ==> ROR A, ShrAmt
        // (A >> ShrAmt) | (B << (32-ShrAmt)) ==> SHRD A, B, ShrAmt
        if (ShrVal == ShlVal) {
          unsigned Reg, ShAmt;
          if (getRegPressure(ShrVal) > getRegPressure(ShrAmt)) {
            Reg = SelectExpr(ShrVal);
            ShAmt = SelectExpr(ShrAmt);
          } else {
            ShAmt = SelectExpr(ShrAmt);
            Reg = SelectExpr(ShrVal);
          }
          BuildMI(BB, X86::MOV8rr, 1, X86::CL).addReg(ShAmt);
          unsigned Opc = RegSize == 8 ? X86::ROR8rCL :
                        (RegSize == 16 ? X86::ROR16rCL : X86::ROR32rCL);
          BuildMI(BB, Opc, 1, DestReg).addReg(Reg);
          return true;
        } else if (RegSize != 8) {
          unsigned AReg, BReg;
          if (getRegPressure(ShlVal) > getRegPressure(ShrVal)) {
            BReg = SelectExpr(ShlVal);
            AReg = SelectExpr(ShrVal);
          } else {
            AReg = SelectExpr(ShrVal);
            BReg = SelectExpr(ShlVal);
          }
          unsigned ShAmt = SelectExpr(ShrAmt);
          BuildMI(BB, X86::MOV8rr, 1, X86::CL).addReg(ShAmt);
          unsigned Opc = RegSize == 16 ? X86::SHRD16rrCL : X86::SHRD32rrCL;
          BuildMI(BB, Opc, 2, DestReg).addReg(AReg).addReg(BReg);
          return true;
        }
      }

  if (ShrAmt.getOpcode() == ISD::SUB && ShrAmt.getOperand(1) == ShlAmt)
    if (ConstantSDNode *SubCST = dyn_cast<ConstantSDNode>(ShrAmt.getOperand(0)))
      if (SubCST->getValue() == RegSize) {
        // (A << ShlAmt) | (A >> (32-ShlAmt)) ==> ROL A, ShrAmt
        // (A << ShlAmt) | (B >> (32-ShlAmt)) ==> SHLD A, B, ShrAmt
        if (ShrVal == ShlVal) {
          unsigned Reg, ShAmt;
          if (getRegPressure(ShrVal) > getRegPressure(ShlAmt)) {
            Reg = SelectExpr(ShrVal);
            ShAmt = SelectExpr(ShlAmt);
          } else {
            ShAmt = SelectExpr(ShlAmt);
            Reg = SelectExpr(ShrVal);
          }
          BuildMI(BB, X86::MOV8rr, 1, X86::CL).addReg(ShAmt);
          unsigned Opc = RegSize == 8 ? X86::ROL8rCL :
                        (RegSize == 16 ? X86::ROL16rCL : X86::ROL32rCL);
          BuildMI(BB, Opc, 1, DestReg).addReg(Reg);
          return true;
        } else if (RegSize != 8) {
          unsigned AReg, BReg;
          if (getRegPressure(ShlVal) > getRegPressure(ShrVal)) {
            AReg = SelectExpr(ShlVal);
            BReg = SelectExpr(ShrVal);
          } else {
            BReg = SelectExpr(ShrVal);
            AReg = SelectExpr(ShlVal);
          }
          unsigned ShAmt = SelectExpr(ShlAmt);
          BuildMI(BB, X86::MOV8rr, 1, X86::CL).addReg(ShAmt);
          unsigned Opc = RegSize == 16 ? X86::SHLD16rrCL : X86::SHLD32rrCL;
          BuildMI(BB, Opc, 2, DestReg).addReg(AReg).addReg(BReg);
          return true;
        }
      }

  if (ConstantSDNode *ShrCst = dyn_cast<ConstantSDNode>(ShrAmt))
    if (ConstantSDNode *ShlCst = dyn_cast<ConstantSDNode>(ShlAmt))
      if (ShrCst->getValue() < RegSize && ShlCst->getValue() < RegSize)
        if (ShrCst->getValue() == RegSize-ShlCst->getValue()) {
          // (A >> 5) | (A << 27) --> ROR A, 5
          // (A >> 5) | (B << 27) --> SHRD A, B, 5
          if (ShrVal == ShlVal) {
            unsigned Reg = SelectExpr(ShrVal);
            unsigned Opc = RegSize == 8 ? X86::ROR8ri :
              (RegSize == 16 ? X86::ROR16ri : X86::ROR32ri);
            BuildMI(BB, Opc, 2, DestReg).addReg(Reg).addImm(ShrCst->getValue());
            return true;
          } else if (RegSize != 8) {
            unsigned AReg, BReg;
            if (getRegPressure(ShlVal) > getRegPressure(ShrVal)) {
              BReg = SelectExpr(ShlVal);
              AReg = SelectExpr(ShrVal);
            } else {
              AReg = SelectExpr(ShrVal);
              BReg = SelectExpr(ShlVal);
            }
            unsigned Opc = RegSize == 16 ? X86::SHRD16rri8 : X86::SHRD32rri8;
            BuildMI(BB, Opc, 3, DestReg).addReg(AReg).addReg(BReg)
              .addImm(ShrCst->getValue());
            return true;
          }
        }

  return false;
}

unsigned ISel::SelectExpr(SDOperand N) {
  unsigned Result;
  unsigned Tmp1, Tmp2, Tmp3;
  unsigned Opc = 0;
  SDNode *Node = N.Val;
  SDOperand Op0, Op1;

  if (Node->getOpcode() == ISD::CopyFromReg) {
    unsigned Reg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
    // Just use the specified register as our input if we can.
    if (MRegisterInfo::isVirtualRegister(Reg) || Reg == X86::ESP)
      return Reg;
  }

  unsigned &Reg = ExprMap[N];
  if (Reg) return Reg;

  switch (N.getOpcode()) {
  default:
    Reg = Result = (N.getValueType() != MVT::Other) ?
                            MakeReg(N.getValueType()) : 1;
    break;
  case X86ISD::TAILCALL:
  case X86ISD::CALL:
    // If this is a call instruction, make sure to prepare ALL of the result
    // values as well as the chain.
    ExprMap[N.getValue(0)] = 1;
    if (Node->getNumValues() > 1) {
      Result = MakeReg(Node->getValueType(1));
      ExprMap[N.getValue(1)] = Result;
      for (unsigned i = 2, e = Node->getNumValues(); i != e; ++i)
        ExprMap[N.getValue(i)] = MakeReg(Node->getValueType(i));
    } else {
      Result = 1;
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

  switch (N.getOpcode()) {
  default:
    Node->dump();
    assert(0 && "Node not handled!\n");
  case ISD::FP_EXTEND:
    assert(X86ScalarSSE && "Scalar SSE FP must be enabled to use f32");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, X86::CVTSS2SDrr, 1, Result).addReg(Tmp1);
    return Result;
  case ISD::FP_ROUND:
    assert(X86ScalarSSE && "Scalar SSE FP must be enabled to use f32");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, X86::CVTSD2SSrr, 1, Result).addReg(Tmp1);
    return Result;
  case ISD::CopyFromReg:
    Select(N.getOperand(0));
    if (Result == 1) {
      Reg = Result = ExprMap[N.getValue(0)] =
        MakeReg(N.getValue(0).getValueType());
    }
    Tmp1 = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
    switch (Node->getValueType(0)) {
    default: assert(0 && "Cannot CopyFromReg this!");
    case MVT::i1:
    case MVT::i8:
      BuildMI(BB, X86::MOV8rr, 1, Result).addReg(Tmp1);
      return Result;
    case MVT::i16:
      BuildMI(BB, X86::MOV16rr, 1, Result).addReg(Tmp1);
      return Result;
    case MVT::i32:
      BuildMI(BB, X86::MOV32rr, 1, Result).addReg(Tmp1);
      return Result;
    }

  case ISD::FrameIndex:
    Tmp1 = cast<FrameIndexSDNode>(N)->getIndex();
    addFrameReference(BuildMI(BB, X86::LEA32r, 4, Result), (int)Tmp1);
    return Result;
  case ISD::ConstantPool:
    Tmp1 = BB->getParent()->getConstantPool()->
         getConstantPoolIndex(cast<ConstantPoolSDNode>(N)->get());
    addConstantPoolReference(BuildMI(BB, X86::LEA32r, 4, Result), Tmp1);
    return Result;
  case ISD::ConstantFP:
    if (X86ScalarSSE) {
      assert(cast<ConstantFPSDNode>(N)->isExactlyValue(+0.0) &&
             "SSE only supports +0.0");
      Opc = (N.getValueType() == MVT::f32) ? X86::FLD0SS : X86::FLD0SD;
      BuildMI(BB, Opc, 0, Result);
      return Result;
    }
    ContainsFPCode = true;
    Tmp1 = Result;   // Intermediate Register
    if (cast<ConstantFPSDNode>(N)->getValue() < 0.0 ||
        cast<ConstantFPSDNode>(N)->isExactlyValue(-0.0))
      Tmp1 = MakeReg(MVT::f64);

    if (cast<ConstantFPSDNode>(N)->isExactlyValue(+0.0) ||
        cast<ConstantFPSDNode>(N)->isExactlyValue(-0.0))
      BuildMI(BB, X86::FLD0, 0, Tmp1);
    else if (cast<ConstantFPSDNode>(N)->isExactlyValue(+1.0) ||
             cast<ConstantFPSDNode>(N)->isExactlyValue(-1.0))
      BuildMI(BB, X86::FLD1, 0, Tmp1);
    else
      assert(0 && "Unexpected constant!");
    if (Tmp1 != Result)
      BuildMI(BB, X86::FCHS, 1, Result).addReg(Tmp1);
    return Result;
  case ISD::Constant:
    switch (N.getValueType()) {
    default: assert(0 && "Cannot use constants of this type!");
    case MVT::i1:
    case MVT::i8:  Opc = X86::MOV8ri;  break;
    case MVT::i16: Opc = X86::MOV16ri; break;
    case MVT::i32: Opc = X86::MOV32ri; break;
    }
    BuildMI(BB, Opc, 1,Result).addImm(cast<ConstantSDNode>(N)->getValue());
    return Result;
  case ISD::UNDEF:
    if (Node->getValueType(0) == MVT::f64) {
      // FIXME: SHOULD TEACH STACKIFIER ABOUT UNDEF VALUES!
      BuildMI(BB, X86::FLD0, 0, Result);
    } else {
      BuildMI(BB, X86::IMPLICIT_DEF, 0, Result);
    }
    return Result;
  case ISD::GlobalAddress: {
    GlobalValue *GV = cast<GlobalAddressSDNode>(N)->getGlobal();
    // For Darwin, external and weak symbols are indirect, so we want to load
    // the value at address GV, not the value of GV itself.
    if (Subtarget->getIndirectExternAndWeakGlobals() &&
        (GV->hasWeakLinkage() || GV->isExternal())) {
      BuildMI(BB, X86::MOV32rm, 4, Result).addReg(0).addZImm(1).addReg(0)
        .addGlobalAddress(GV, false, 0);
    } else {
      BuildMI(BB, X86::MOV32ri, 1, Result).addGlobalAddress(GV);
    }
    return Result;
  }
  case ISD::ExternalSymbol: {
    const char *Sym = cast<ExternalSymbolSDNode>(N)->getSymbol();
    BuildMI(BB, X86::MOV32ri, 1, Result).addExternalSymbol(Sym);
    return Result;
  }
  case ISD::ANY_EXTEND:   // treat any extend like zext
  case ISD::ZERO_EXTEND: {
    int DestIs16 = N.getValueType() == MVT::i16;
    int SrcIs16  = N.getOperand(0).getValueType() == MVT::i16;

    // FIXME: This hack is here for zero extension casts from bool to i8.  This
    // would not be needed if bools were promoted by Legalize.
    if (N.getValueType() == MVT::i8) {
      Tmp1 = SelectExpr(N.getOperand(0));
      BuildMI(BB, X86::MOV8rr, 1, Result).addReg(Tmp1);
      return Result;
    }

    if (isFoldableLoad(N.getOperand(0), SDOperand())) {
      static const unsigned Opc[3] = {
        X86::MOVZX32rm8, X86::MOVZX32rm16, X86::MOVZX16rm8
      };

      X86AddressMode AM;
      EmitFoldedLoad(N.getOperand(0), AM);
      addFullAddress(BuildMI(BB, Opc[SrcIs16+DestIs16*2], 4, Result), AM);

      return Result;
    }

    static const unsigned Opc[3] = {
      X86::MOVZX32rr8, X86::MOVZX32rr16, X86::MOVZX16rr8
    };
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, Opc[SrcIs16+DestIs16*2], 1, Result).addReg(Tmp1);
    return Result;
  }
  case ISD::SIGN_EXTEND: {
    int DestIs16 = N.getValueType() == MVT::i16;
    int SrcIs16  = N.getOperand(0).getValueType() == MVT::i16;

    // FIXME: Legalize should promote bools to i8!
    assert(N.getOperand(0).getValueType() != MVT::i1 &&
           "Sign extend from bool not implemented!");

    if (isFoldableLoad(N.getOperand(0), SDOperand())) {
      static const unsigned Opc[3] = {
        X86::MOVSX32rm8, X86::MOVSX32rm16, X86::MOVSX16rm8
      };

      X86AddressMode AM;
      EmitFoldedLoad(N.getOperand(0), AM);
      addFullAddress(BuildMI(BB, Opc[SrcIs16+DestIs16*2], 4, Result), AM);
      return Result;
    }

    static const unsigned Opc[3] = {
      X86::MOVSX32rr8, X86::MOVSX32rr16, X86::MOVSX16rr8
    };
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, Opc[SrcIs16+DestIs16*2], 1, Result).addReg(Tmp1);
    return Result;
  }
  case ISD::TRUNCATE:
    // Fold TRUNCATE (LOAD P) into a smaller load from P.
    // FIXME: This should be performed by the DAGCombiner.
    if (isFoldableLoad(N.getOperand(0), SDOperand())) {
      switch (N.getValueType()) {
      default: assert(0 && "Unknown truncate!");
      case MVT::i1:
      case MVT::i8:  Opc = X86::MOV8rm;  break;
      case MVT::i16: Opc = X86::MOV16rm; break;
      }
      X86AddressMode AM;
      EmitFoldedLoad(N.getOperand(0), AM);
      addFullAddress(BuildMI(BB, Opc, 4, Result), AM);
      return Result;
    }

    // Handle cast of LARGER int to SMALLER int using a move to EAX followed by
    // a move out of AX or AL.
    switch (N.getOperand(0).getValueType()) {
    default: assert(0 && "Unknown truncate!");
    case MVT::i8:  Tmp2 = X86::AL;  Opc = X86::MOV8rr;  break;
    case MVT::i16: Tmp2 = X86::AX;  Opc = X86::MOV16rr; break;
    case MVT::i32: Tmp2 = X86::EAX; Opc = X86::MOV32rr; break;
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, Opc, 1, Tmp2).addReg(Tmp1);

    switch (N.getValueType()) {
    default: assert(0 && "Unknown truncate!");
    case MVT::i1:
    case MVT::i8:  Tmp2 = X86::AL;  Opc = X86::MOV8rr;  break;
    case MVT::i16: Tmp2 = X86::AX;  Opc = X86::MOV16rr; break;
    }
    BuildMI(BB, Opc, 1, Result).addReg(Tmp2);
    return Result;

  case ISD::SINT_TO_FP: {
    Tmp1 = SelectExpr(N.getOperand(0));  // Get the operand register
    unsigned PromoteOpcode = 0;

    // We can handle any sint to fp with the direct sse conversion instructions.
    if (X86ScalarSSE) {
      Opc = (N.getValueType() == MVT::f64) ? X86::CVTSI2SDrr : X86::CVTSI2SSrr;
      BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
      return Result;
    }

    ContainsFPCode = true;

    // Spill the integer to memory and reload it from there.
    MVT::ValueType SrcTy = N.getOperand(0).getValueType();
    unsigned Size = MVT::getSizeInBits(SrcTy)/8;
    MachineFunction *F = BB->getParent();
    int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, Size);

    switch (SrcTy) {
    case MVT::i32:
      addFrameReference(BuildMI(BB, X86::MOV32mr, 5), FrameIdx).addReg(Tmp1);
      addFrameReference(BuildMI(BB, X86::FILD32m, 5, Result), FrameIdx);
      break;
    case MVT::i16:
      addFrameReference(BuildMI(BB, X86::MOV16mr, 5), FrameIdx).addReg(Tmp1);
      addFrameReference(BuildMI(BB, X86::FILD16m, 5, Result), FrameIdx);
      break;
    default: break; // No promotion required.
    }
    return Result;
  }
  case ISD::FP_TO_SINT:
    Tmp1 = SelectExpr(N.getOperand(0));  // Get the operand register

    // If the target supports SSE2 and is performing FP operations in SSE regs
    // instead of the FP stack, then we can use the efficient CVTSS2SI and
    // CVTSD2SI instructions.
    assert(X86ScalarSSE);
    if (MVT::f32 == N.getOperand(0).getValueType()) {
      BuildMI(BB, X86::CVTTSS2SIrr, 1, Result).addReg(Tmp1);
    } else if (MVT::f64 == N.getOperand(0).getValueType()) {
      BuildMI(BB, X86::CVTTSD2SIrr, 1, Result).addReg(Tmp1);
    } else {
      assert(0 && "Not an f32 or f64?");
      abort();
    }
    return Result;

  case ISD::FADD:
  case ISD::ADD:
    Op0 = N.getOperand(0);
    Op1 = N.getOperand(1);

    if (isFoldableLoad(Op0, Op1, true)) {
      std::swap(Op0, Op1);
      goto FoldAdd;
    }

    if (isFoldableLoad(Op1, Op0, true)) {
    FoldAdd:
      switch (N.getValueType()) {
      default: assert(0 && "Cannot add this type!");
      case MVT::i1:
      case MVT::i8:  Opc = X86::ADD8rm;  break;
      case MVT::i16: Opc = X86::ADD16rm; break;
      case MVT::i32: Opc = X86::ADD32rm; break;
      case MVT::f32: Opc = X86::ADDSSrm; break;
      case MVT::f64:
        // For F64, handle promoted load operations (from F32) as well!
        if (X86ScalarSSE) {
          assert(Op1.getOpcode() == ISD::LOAD && "SSE load not promoted");
          Opc = X86::ADDSDrm;
        } else {
          Opc = Op1.getOpcode() == ISD::LOAD ? X86::FADD64m : X86::FADD32m;
        }
        break;
      }
      X86AddressMode AM;
      EmitFoldedLoad(Op1, AM);
      Tmp1 = SelectExpr(Op0);
      addFullAddress(BuildMI(BB, Opc, 5, Result).addReg(Tmp1), AM);
      return Result;
    }

    // See if we can codegen this as an LEA to fold operations together.
    if (N.getValueType() == MVT::i32) {
      ExprMap.erase(N);
      X86ISelAddressMode AM;
      MatchAddress(N, AM);
      ExprMap[N] = Result;

      // If this is not just an add, emit the LEA.  For a simple add (like
      // reg+reg or reg+imm), we just emit an add.  It might be a good idea to
      // leave this as LEA, then peephole it to 'ADD' after two address elim
      // happens.
      if (AM.Scale != 1 || AM.BaseType == X86ISelAddressMode::FrameIndexBase||
          AM.GV || (AM.Base.Reg.Val && AM.IndexReg.Val && AM.Disp)) {
        X86AddressMode XAM = SelectAddrExprs(AM);
        addFullAddress(BuildMI(BB, X86::LEA32r, 4, Result), XAM);
        return Result;
      }
    }

    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op1)) {
      Opc = 0;
      if (CN->getValue() == 1) {   // add X, 1 -> inc X
        switch (N.getValueType()) {
        default: assert(0 && "Cannot integer add this type!");
        case MVT::i8:  Opc = X86::INC8r; break;
        case MVT::i16: Opc = X86::INC16r; break;
        case MVT::i32: Opc = X86::INC32r; break;
        }
      } else if (CN->isAllOnesValue()) { // add X, -1 -> dec X
        switch (N.getValueType()) {
        default: assert(0 && "Cannot integer add this type!");
        case MVT::i8:  Opc = X86::DEC8r; break;
        case MVT::i16: Opc = X86::DEC16r; break;
        case MVT::i32: Opc = X86::DEC32r; break;
        }
      }

      if (Opc) {
        Tmp1 = SelectExpr(Op0);
        BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
        return Result;
      }

      switch (N.getValueType()) {
      default: assert(0 && "Cannot add this type!");
      case MVT::i8:  Opc = X86::ADD8ri; break;
      case MVT::i16: Opc = X86::ADD16ri; break;
      case MVT::i32: Opc = X86::ADD32ri; break;
      }
      if (Opc) {
        Tmp1 = SelectExpr(Op0);
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(CN->getValue());
        return Result;
      }
    }

    switch (N.getValueType()) {
    default: assert(0 && "Cannot add this type!");
    case MVT::i8:  Opc = X86::ADD8rr; break;
    case MVT::i16: Opc = X86::ADD16rr; break;
    case MVT::i32: Opc = X86::ADD32rr; break;
    case MVT::f32: Opc = X86::ADDSSrr; break;
    case MVT::f64: Opc = X86ScalarSSE ? X86::ADDSDrr : X86::FpADD; break;
    }

    if (getRegPressure(Op0) > getRegPressure(Op1)) {
      Tmp1 = SelectExpr(Op0);
      Tmp2 = SelectExpr(Op1);
    } else {
      Tmp2 = SelectExpr(Op1);
      Tmp1 = SelectExpr(Op0);
    }

    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::FSQRT:
    Tmp1 = SelectExpr(Node->getOperand(0));
    if (X86ScalarSSE) {
      Opc = (N.getValueType() == MVT::f32) ? X86::SQRTSSrr : X86::SQRTSDrr;
      BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    } else {
      BuildMI(BB, X86::FSQRT, 1, Result).addReg(Tmp1);
    }
    return Result;

  // FIXME:
  // Once we can spill 16 byte constants into the constant pool, we can
  // implement SSE equivalents of FABS and FCHS.
  case ISD::FABS:
  case ISD::FNEG:
  case ISD::FSIN:
  case ISD::FCOS:
    assert(N.getValueType()==MVT::f64 && "Illegal type for this operation");
    Tmp1 = SelectExpr(Node->getOperand(0));
    switch (N.getOpcode()) {
    default: assert(0 && "Unreachable!");
    case ISD::FABS: BuildMI(BB, X86::FABS, 1, Result).addReg(Tmp1); break;
    case ISD::FNEG: BuildMI(BB, X86::FCHS, 1, Result).addReg(Tmp1); break;
    case ISD::FSIN: BuildMI(BB, X86::FSIN, 1, Result).addReg(Tmp1); break;
    case ISD::FCOS: BuildMI(BB, X86::FCOS, 1, Result).addReg(Tmp1); break;
    }
    return Result;

  case ISD::MULHU:
    switch (N.getValueType()) {
    default: assert(0 && "Unsupported VT!");
    case MVT::i8:  Tmp2 = X86::MUL8r;  break;
    case MVT::i16: Tmp2 = X86::MUL16r;  break;
    case MVT::i32: Tmp2 = X86::MUL32r;  break;
    }
    // FALL THROUGH
  case ISD::MULHS: {
    unsigned MovOpc, LowReg, HiReg;
    switch (N.getValueType()) {
    default: assert(0 && "Unsupported VT!");
    case MVT::i8:
      MovOpc = X86::MOV8rr;
      LowReg = X86::AL;
      HiReg = X86::AH;
      Opc = X86::IMUL8r;
      break;
    case MVT::i16:
      MovOpc = X86::MOV16rr;
      LowReg = X86::AX;
      HiReg = X86::DX;
      Opc = X86::IMUL16r;
      break;
    case MVT::i32:
      MovOpc = X86::MOV32rr;
      LowReg = X86::EAX;
      HiReg = X86::EDX;
      Opc = X86::IMUL32r;
      break;
    }
    if (Node->getOpcode() != ISD::MULHS)
      Opc = Tmp2;  // Get the MULHU opcode.

    Op0 = Node->getOperand(0);
    Op1 = Node->getOperand(1);
    if (getRegPressure(Op0) > getRegPressure(Op1)) {
      Tmp1 = SelectExpr(Op0);
      Tmp2 = SelectExpr(Op1);
    } else {
      Tmp2 = SelectExpr(Op1);
      Tmp1 = SelectExpr(Op0);
    }

    // FIXME: Implement folding of loads into the memory operands here!
    BuildMI(BB, MovOpc, 1, LowReg).addReg(Tmp1);
    BuildMI(BB, Opc, 1).addReg(Tmp2);
    BuildMI(BB, MovOpc, 1, Result).addReg(HiReg);
    return Result;
  }

  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR: {
    static const unsigned SUBTab[] = {
      X86::SUB8ri, X86::SUB16ri, X86::SUB32ri, 0, 0,
      X86::SUB8rm, X86::SUB16rm, X86::SUB32rm, X86::FSUB32m, X86::FSUB64m,
      X86::SUB8rr, X86::SUB16rr, X86::SUB32rr, X86::FpSUB  , X86::FpSUB,
    };
    static const unsigned SSE_SUBTab[] = {
      X86::SUB8ri, X86::SUB16ri, X86::SUB32ri, 0, 0,
      X86::SUB8rm, X86::SUB16rm, X86::SUB32rm, X86::SUBSSrm, X86::SUBSDrm,
      X86::SUB8rr, X86::SUB16rr, X86::SUB32rr, X86::SUBSSrr, X86::SUBSDrr,
    };
    static const unsigned MULTab[] = {
      0, X86::IMUL16rri, X86::IMUL32rri, 0, 0,
      0, X86::IMUL16rm , X86::IMUL32rm, X86::FMUL32m, X86::FMUL64m,
      0, X86::IMUL16rr , X86::IMUL32rr, X86::FpMUL  , X86::FpMUL,
    };
    static const unsigned SSE_MULTab[] = {
      0, X86::IMUL16rri, X86::IMUL32rri, 0, 0,
      0, X86::IMUL16rm , X86::IMUL32rm, X86::MULSSrm, X86::MULSDrm,
      0, X86::IMUL16rr , X86::IMUL32rr, X86::MULSSrr, X86::MULSDrr,
    };
    static const unsigned ANDTab[] = {
      X86::AND8ri, X86::AND16ri, X86::AND32ri, 0, 0,
      X86::AND8rm, X86::AND16rm, X86::AND32rm, 0, 0,
      X86::AND8rr, X86::AND16rr, X86::AND32rr, 0, 0,
    };
    static const unsigned ORTab[] = {
      X86::OR8ri, X86::OR16ri, X86::OR32ri, 0, 0,
      X86::OR8rm, X86::OR16rm, X86::OR32rm, 0, 0,
      X86::OR8rr, X86::OR16rr, X86::OR32rr, 0, 0,
    };
    static const unsigned XORTab[] = {
      X86::XOR8ri, X86::XOR16ri, X86::XOR32ri, 0, 0,
      X86::XOR8rm, X86::XOR16rm, X86::XOR32rm, 0, 0,
      X86::XOR8rr, X86::XOR16rr, X86::XOR32rr, 0, 0,
    };

    Op0 = Node->getOperand(0);
    Op1 = Node->getOperand(1);

    if (Node->getOpcode() == ISD::OR && Op0.hasOneUse() && Op1.hasOneUse())
      if (EmitOrOpOp(Op0, Op1, Result)) // Match SHLD, SHRD, and rotates.
        return Result;

    if (Node->getOpcode() == ISD::SUB)
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(0)))
        if (CN->isNullValue()) {   // 0 - N -> neg N
          switch (N.getValueType()) {
          default: assert(0 && "Cannot sub this type!");
          case MVT::i1:
          case MVT::i8:  Opc = X86::NEG8r;  break;
          case MVT::i16: Opc = X86::NEG16r; break;
          case MVT::i32: Opc = X86::NEG32r; break;
          }
          Tmp1 = SelectExpr(N.getOperand(1));
          BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
          return Result;
        }

    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op1)) {
      if (CN->isAllOnesValue() && Node->getOpcode() == ISD::XOR) {
        Opc = 0;
        switch (N.getValueType()) {
        default: assert(0 && "Cannot add this type!");
        case MVT::i1:  break;  // Not supported, don't invert upper bits!
        case MVT::i8:  Opc = X86::NOT8r;  break;
        case MVT::i16: Opc = X86::NOT16r; break;
        case MVT::i32: Opc = X86::NOT32r; break;
        }
        if (Opc) {
          Tmp1 = SelectExpr(Op0);
          BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
          return Result;
        }
      }

      // Fold common multiplies into LEA instructions.
      if (Node->getOpcode() == ISD::MUL && N.getValueType() == MVT::i32) {
        switch ((int)CN->getValue()) {
        default: break;
        case 3:
        case 5:
        case 9:
          // Remove N from exprmap so SelectAddress doesn't get confused.
          ExprMap.erase(N);
          X86AddressMode AM;
          SelectAddress(N, AM);
          // Restore it to the map.
          ExprMap[N] = Result;
          addFullAddress(BuildMI(BB, X86::LEA32r, 4, Result), AM);
          return Result;
        }
      }

      switch (N.getValueType()) {
      default: assert(0 && "Cannot xor this type!");
      case MVT::i1:
      case MVT::i8:  Opc = 0; break;
      case MVT::i16: Opc = 1; break;
      case MVT::i32: Opc = 2; break;
      }
      switch (Node->getOpcode()) {
      default: assert(0 && "Unreachable!");
      case ISD::FSUB:
      case ISD::SUB: Opc = X86ScalarSSE ? SSE_SUBTab[Opc] : SUBTab[Opc]; break;
      case ISD::FMUL:
      case ISD::MUL: Opc = X86ScalarSSE ? SSE_MULTab[Opc] : MULTab[Opc]; break;
      case ISD::AND: Opc = ANDTab[Opc]; break;
      case ISD::OR:  Opc =  ORTab[Opc]; break;
      case ISD::XOR: Opc = XORTab[Opc]; break;
      }
      if (Opc) {  // Can't fold MUL:i8 R, imm
        Tmp1 = SelectExpr(Op0);
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(CN->getValue());
        return Result;
      }
    }

    if (isFoldableLoad(Op0, Op1, true))
      if (Node->getOpcode() != ISD::SUB && Node->getOpcode() != ISD::FSUB) {
        std::swap(Op0, Op1);
        goto FoldOps;
      } else {
        // For FP, emit 'reverse' subract, with a memory operand.
        if (N.getValueType() == MVT::f64 && !X86ScalarSSE) {
          if (Op0.getOpcode() == ISD::EXTLOAD)
            Opc = X86::FSUBR32m;
          else
            Opc = X86::FSUBR64m;

          X86AddressMode AM;
          EmitFoldedLoad(Op0, AM);
          Tmp1 = SelectExpr(Op1);
          addFullAddress(BuildMI(BB, Opc, 5, Result).addReg(Tmp1), AM);
          return Result;
        }
      }

    if (isFoldableLoad(Op1, Op0, true)) {
    FoldOps:
      switch (N.getValueType()) {
      default: assert(0 && "Cannot operate on this type!");
      case MVT::i1:
      case MVT::i8:  Opc = 5; break;
      case MVT::i16: Opc = 6; break;
      case MVT::i32: Opc = 7; break;
      case MVT::f32: Opc = 8; break;
        // For F64, handle promoted load operations (from F32) as well!
      case MVT::f64:
        assert((!X86ScalarSSE || Op1.getOpcode() == ISD::LOAD) &&
               "SSE load should have been promoted");
        Opc = Op1.getOpcode() == ISD::LOAD ? 9 : 8; break;
      }
      switch (Node->getOpcode()) {
      default: assert(0 && "Unreachable!");
      case ISD::FSUB:
      case ISD::SUB: Opc = X86ScalarSSE ? SSE_SUBTab[Opc] : SUBTab[Opc]; break;
      case ISD::FMUL:
      case ISD::MUL: Opc = X86ScalarSSE ? SSE_MULTab[Opc] : MULTab[Opc]; break;
      case ISD::AND: Opc = ANDTab[Opc]; break;
      case ISD::OR:  Opc =  ORTab[Opc]; break;
      case ISD::XOR: Opc = XORTab[Opc]; break;
      }

      X86AddressMode AM;
      EmitFoldedLoad(Op1, AM);
      Tmp1 = SelectExpr(Op0);
      if (Opc) {
        addFullAddress(BuildMI(BB, Opc, 5, Result).addReg(Tmp1), AM);
      } else {
        assert(Node->getOpcode() == ISD::MUL &&
               N.getValueType() == MVT::i8 && "Unexpected situation!");
        // Must use the MUL instruction, which forces use of AL.
        BuildMI(BB, X86::MOV8rr, 1, X86::AL).addReg(Tmp1);
        addFullAddress(BuildMI(BB, X86::MUL8m, 1), AM);
        BuildMI(BB, X86::MOV8rr, 1, Result).addReg(X86::AL);
      }
      return Result;
    }

    if (getRegPressure(Op0) > getRegPressure(Op1)) {
      Tmp1 = SelectExpr(Op0);
      Tmp2 = SelectExpr(Op1);
    } else {
      Tmp2 = SelectExpr(Op1);
      Tmp1 = SelectExpr(Op0);
    }

    switch (N.getValueType()) {
    default: assert(0 && "Cannot add this type!");
    case MVT::i1:
    case MVT::i8:  Opc = 10; break;
    case MVT::i16: Opc = 11; break;
    case MVT::i32: Opc = 12; break;
    case MVT::f32: Opc = 13; break;
    case MVT::f64: Opc = 14; break;
    }
    switch (Node->getOpcode()) {
    default: assert(0 && "Unreachable!");
    case ISD::FSUB:
    case ISD::SUB: Opc = X86ScalarSSE ? SSE_SUBTab[Opc] : SUBTab[Opc]; break;
    case ISD::FMUL:
    case ISD::MUL: Opc = X86ScalarSSE ? SSE_MULTab[Opc] : MULTab[Opc]; break;
    case ISD::AND: Opc = ANDTab[Opc]; break;
    case ISD::OR:  Opc =  ORTab[Opc]; break;
    case ISD::XOR: Opc = XORTab[Opc]; break;
    }
    if (Opc) {
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    } else {
      assert(Node->getOpcode() == ISD::MUL &&
             N.getValueType() == MVT::i8 && "Unexpected situation!");
      // Must use the MUL instruction, which forces use of AL.
      BuildMI(BB, X86::MOV8rr, 1, X86::AL).addReg(Tmp1);
      BuildMI(BB, X86::MUL8r, 1).addReg(Tmp2);
      BuildMI(BB, X86::MOV8rr, 1, Result).addReg(X86::AL);
    }
    return Result;
  }
  case ISD::ADD_PARTS:
  case ISD::SUB_PARTS: {
    assert(N.getNumOperands() == 4 && N.getValueType() == MVT::i32 &&
           "Not an i64 add/sub!");
    // Emit all of the operands.
    std::vector<unsigned> InVals;
    for (unsigned i = 0, e = N.getNumOperands(); i != e; ++i)
      InVals.push_back(SelectExpr(N.getOperand(i)));
    if (N.getOpcode() == ISD::ADD_PARTS) {
      BuildMI(BB, X86::ADD32rr, 2, Result).addReg(InVals[0]).addReg(InVals[2]);
      BuildMI(BB, X86::ADC32rr,2,Result+1).addReg(InVals[1]).addReg(InVals[3]);
    } else {
      BuildMI(BB, X86::SUB32rr, 2, Result).addReg(InVals[0]).addReg(InVals[2]);
      BuildMI(BB, X86::SBB32rr, 2,Result+1).addReg(InVals[1]).addReg(InVals[3]);
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
    unsigned TmpReg = MakeReg(MVT::i32);
    if (N.getOpcode() == ISD::SRA_PARTS) {
      // If this is a SHR of a Long, then we need to do funny sign extension
      // stuff.  TmpReg gets the value to use as the high-part if we are
      // shifting more than 32 bits.
      BuildMI(BB, X86::SAR32ri, 2, TmpReg).addReg(ShiftOpHi).addImm(31);
    } else {
      // Other shifts use a fixed zero value if the shift is more than 32 bits.
      BuildMI(BB, X86::MOV32ri, 1, TmpReg).addImm(0);
    }

    // Initialize CL with the shift amount.
    unsigned ShiftAmountReg = SelectExpr(N.getOperand(2));
    BuildMI(BB, X86::MOV8rr, 1, X86::CL).addReg(ShiftAmountReg);

    unsigned TmpReg2 = MakeReg(MVT::i32);
    unsigned TmpReg3 = MakeReg(MVT::i32);
    if (N.getOpcode() == ISD::SHL_PARTS) {
      // TmpReg2 = shld inHi, inLo
      BuildMI(BB, X86::SHLD32rrCL, 2,TmpReg2).addReg(ShiftOpHi)
        .addReg(ShiftOpLo);
      // TmpReg3 = shl  inLo, CL
      BuildMI(BB, X86::SHL32rCL, 1, TmpReg3).addReg(ShiftOpLo);

      // Set the flags to indicate whether the shift was by more than 32 bits.
      BuildMI(BB, X86::TEST8ri, 2).addReg(X86::CL).addImm(32);

      // DestHi = (>32) ? TmpReg3 : TmpReg2;
      BuildMI(BB, X86::CMOVNE32rr, 2,
              Result+1).addReg(TmpReg2).addReg(TmpReg3);
      // DestLo = (>32) ? TmpReg : TmpReg3;
      BuildMI(BB, X86::CMOVNE32rr, 2,
              Result).addReg(TmpReg3).addReg(TmpReg);
    } else {
      // TmpReg2 = shrd inLo, inHi
      BuildMI(BB, X86::SHRD32rrCL,2,TmpReg2).addReg(ShiftOpLo)
        .addReg(ShiftOpHi);
      // TmpReg3 = s[ah]r  inHi, CL
      BuildMI(BB, N.getOpcode() == ISD::SRA_PARTS ? X86::SAR32rCL
                                                  : X86::SHR32rCL, 1, TmpReg3)
        .addReg(ShiftOpHi);

      // Set the flags to indicate whether the shift was by more than 32 bits.
      BuildMI(BB, X86::TEST8ri, 2).addReg(X86::CL).addImm(32);

      // DestLo = (>32) ? TmpReg3 : TmpReg2;
      BuildMI(BB, X86::CMOVNE32rr, 2,
              Result).addReg(TmpReg2).addReg(TmpReg3);

      // DestHi = (>32) ? TmpReg : TmpReg3;
      BuildMI(BB, X86::CMOVNE32rr, 2,
              Result+1).addReg(TmpReg3).addReg(TmpReg);
    }
    return Result+N.ResNo;
  }

  case ISD::SELECT:
    EmitSelectCC(N.getOperand(0), N.getOperand(1), N.getOperand(2),
                 N.getValueType(), Result);
    return Result;

  case ISD::FDIV:
  case ISD::FREM:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SREM:
  case ISD::UREM: {
    assert((N.getOpcode() != ISD::SREM || MVT::isInteger(N.getValueType())) &&
           "We don't support this operator!");

    if (N.getOpcode() == ISD::SDIV || N.getOpcode() == ISD::FDIV) {
      // We can fold loads into FpDIVs, but not really into any others.
      if (N.getValueType() == MVT::f64 && !X86ScalarSSE) {
        // Check for reversed and unreversed DIV.
        if (isFoldableLoad(N.getOperand(0), N.getOperand(1), true)) {
          if (N.getOperand(0).getOpcode() == ISD::EXTLOAD)
            Opc = X86::FDIVR32m;
          else
            Opc = X86::FDIVR64m;
          X86AddressMode AM;
          EmitFoldedLoad(N.getOperand(0), AM);
          Tmp1 = SelectExpr(N.getOperand(1));
          addFullAddress(BuildMI(BB, Opc, 5, Result).addReg(Tmp1), AM);
          return Result;
        } else if (isFoldableLoad(N.getOperand(1), N.getOperand(0), true) &&
                   N.getOperand(1).getOpcode() == ISD::LOAD) {
          if (N.getOperand(1).getOpcode() == ISD::EXTLOAD)
            Opc = X86::FDIV32m;
          else
            Opc = X86::FDIV64m;
          X86AddressMode AM;
          EmitFoldedLoad(N.getOperand(1), AM);
          Tmp1 = SelectExpr(N.getOperand(0));
          addFullAddress(BuildMI(BB, Opc, 5, Result).addReg(Tmp1), AM);
          return Result;
        }
      }

      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
        // FIXME: These special cases should be handled by the lowering impl!
        unsigned RHS = CN->getValue();
        bool isNeg = false;
        if ((int)RHS < 0) {
          isNeg = true;
          RHS = -RHS;
        }
        if (RHS && (RHS & (RHS-1)) == 0) {   // Signed division by power of 2?
          unsigned Log = Log2_32(RHS);
          unsigned SAROpc, SHROpc, ADDOpc, NEGOpc;
          switch (N.getValueType()) {
          default: assert("Unknown type to signed divide!");
          case MVT::i8:
            SAROpc = X86::SAR8ri;
            SHROpc = X86::SHR8ri;
            ADDOpc = X86::ADD8rr;
            NEGOpc = X86::NEG8r;
            break;
          case MVT::i16:
            SAROpc = X86::SAR16ri;
            SHROpc = X86::SHR16ri;
            ADDOpc = X86::ADD16rr;
            NEGOpc = X86::NEG16r;
            break;
          case MVT::i32:
            SAROpc = X86::SAR32ri;
            SHROpc = X86::SHR32ri;
            ADDOpc = X86::ADD32rr;
            NEGOpc = X86::NEG32r;
            break;
          }
          unsigned RegSize = MVT::getSizeInBits(N.getValueType());
          Tmp1 = SelectExpr(N.getOperand(0));
          unsigned TmpReg;
          if (Log != 1) {
            TmpReg = MakeReg(N.getValueType());
            BuildMI(BB, SAROpc, 2, TmpReg).addReg(Tmp1).addImm(Log-1);
          } else {
            TmpReg = Tmp1;
          }
          unsigned TmpReg2 = MakeReg(N.getValueType());
          BuildMI(BB, SHROpc, 2, TmpReg2).addReg(TmpReg).addImm(RegSize-Log);
          unsigned TmpReg3 = MakeReg(N.getValueType());
          BuildMI(BB, ADDOpc, 2, TmpReg3).addReg(Tmp1).addReg(TmpReg2);

          unsigned TmpReg4 = isNeg ? MakeReg(N.getValueType()) : Result;
          BuildMI(BB, SAROpc, 2, TmpReg4).addReg(TmpReg3).addImm(Log);
          if (isNeg)
            BuildMI(BB, NEGOpc, 1, Result).addReg(TmpReg4);
          return Result;
        }
      }
    }

    if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(1))) {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
    } else {
      Tmp2 = SelectExpr(N.getOperand(1));
      Tmp1 = SelectExpr(N.getOperand(0));
    }

    bool isSigned = N.getOpcode() == ISD::SDIV || N.getOpcode() == ISD::SREM;
    bool isDiv    = N.getOpcode() == ISD::SDIV || N.getOpcode() == ISD::UDIV;
    unsigned LoReg, HiReg, DivOpcode, MovOpcode, ClrOpcode, SExtOpcode;
    switch (N.getValueType()) {
    default: assert(0 && "Cannot sdiv this type!");
    case MVT::i8:
      DivOpcode = isSigned ? X86::IDIV8r : X86::DIV8r;
      LoReg = X86::AL;
      HiReg = X86::AH;
      MovOpcode = X86::MOV8rr;
      ClrOpcode = X86::MOV8ri;
      SExtOpcode = X86::CBW;
      break;
    case MVT::i16:
      DivOpcode = isSigned ? X86::IDIV16r : X86::DIV16r;
      LoReg = X86::AX;
      HiReg = X86::DX;
      MovOpcode = X86::MOV16rr;
      ClrOpcode = X86::MOV16ri;
      SExtOpcode = X86::CWD;
      break;
    case MVT::i32:
      DivOpcode = isSigned ? X86::IDIV32r : X86::DIV32r;
      LoReg = X86::EAX;
      HiReg = X86::EDX;
      MovOpcode = X86::MOV32rr;
      ClrOpcode = X86::MOV32ri;
      SExtOpcode = X86::CDQ;
      break;
    case MVT::f32:
      BuildMI(BB, X86::DIVSSrr, 2, Result).addReg(Tmp1).addReg(Tmp2);
      return Result;
    case MVT::f64:
      Opc = X86ScalarSSE ? X86::DIVSDrr : X86::FpDIV;
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
      return Result;
    }

    // Set up the low part.
    BuildMI(BB, MovOpcode, 1, LoReg).addReg(Tmp1);

    if (isSigned) {
      // Sign extend the low part into the high part.
      BuildMI(BB, SExtOpcode, 0);
    } else {
      // Zero out the high part, effectively zero extending the input.
      BuildMI(BB, ClrOpcode, 1, HiReg).addImm(0);
    }

    // Emit the DIV/IDIV instruction.
    BuildMI(BB, DivOpcode, 1).addReg(Tmp2);

    // Get the result of the divide or rem.
    BuildMI(BB, MovOpcode, 1, Result).addReg(isDiv ? LoReg : HiReg);
    return Result;
  }

  case ISD::SHL:
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      if (CN->getValue() == 1) {   // X = SHL Y, 1  -> X = ADD Y, Y
        switch (N.getValueType()) {
        default: assert(0 && "Cannot shift this type!");
        case MVT::i8:  Opc = X86::ADD8rr; break;
        case MVT::i16: Opc = X86::ADD16rr; break;
        case MVT::i32: Opc = X86::ADD32rr; break;
        }
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp1);
        return Result;
      }

      switch (N.getValueType()) {
      default: assert(0 && "Cannot shift this type!");
      case MVT::i8:  Opc = X86::SHL8ri; break;
      case MVT::i16: Opc = X86::SHL16ri; break;
      case MVT::i32: Opc = X86::SHL32ri; break;
      }
      Tmp1 = SelectExpr(N.getOperand(0));
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(CN->getValue());
      return Result;
    }

    if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(1))) {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
    } else {
      Tmp2 = SelectExpr(N.getOperand(1));
      Tmp1 = SelectExpr(N.getOperand(0));
    }

    switch (N.getValueType()) {
    default: assert(0 && "Cannot shift this type!");
    case MVT::i8 : Opc = X86::SHL8rCL; break;
    case MVT::i16: Opc = X86::SHL16rCL; break;
    case MVT::i32: Opc = X86::SHL32rCL; break;
    }
    BuildMI(BB, X86::MOV8rr, 1, X86::CL).addReg(Tmp2);
    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;
  case ISD::SRL:
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      switch (N.getValueType()) {
      default: assert(0 && "Cannot shift this type!");
      case MVT::i8:  Opc = X86::SHR8ri; break;
      case MVT::i16: Opc = X86::SHR16ri; break;
      case MVT::i32: Opc = X86::SHR32ri; break;
      }
      Tmp1 = SelectExpr(N.getOperand(0));
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(CN->getValue());
      return Result;
    }

    if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(1))) {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
    } else {
      Tmp2 = SelectExpr(N.getOperand(1));
      Tmp1 = SelectExpr(N.getOperand(0));
    }

    switch (N.getValueType()) {
    default: assert(0 && "Cannot shift this type!");
    case MVT::i8 : Opc = X86::SHR8rCL; break;
    case MVT::i16: Opc = X86::SHR16rCL; break;
    case MVT::i32: Opc = X86::SHR32rCL; break;
    }
    BuildMI(BB, X86::MOV8rr, 1, X86::CL).addReg(Tmp2);
    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;
  case ISD::SRA:
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      switch (N.getValueType()) {
      default: assert(0 && "Cannot shift this type!");
      case MVT::i8:  Opc = X86::SAR8ri; break;
      case MVT::i16: Opc = X86::SAR16ri; break;
      case MVT::i32: Opc = X86::SAR32ri; break;
      }
      Tmp1 = SelectExpr(N.getOperand(0));
      BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(CN->getValue());
      return Result;
    }

    if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(1))) {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
    } else {
      Tmp2 = SelectExpr(N.getOperand(1));
      Tmp1 = SelectExpr(N.getOperand(0));
    }

    switch (N.getValueType()) {
    default: assert(0 && "Cannot shift this type!");
    case MVT::i8 : Opc = X86::SAR8rCL; break;
    case MVT::i16: Opc = X86::SAR16rCL; break;
    case MVT::i32: Opc = X86::SAR32rCL; break;
    }
    BuildMI(BB, X86::MOV8rr, 1, X86::CL).addReg(Tmp2);
    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;

  case ISD::SETCC:
    EmitCMP(N.getOperand(0), N.getOperand(1), Node->hasOneUse());
    EmitSetCC(BB, Result, cast<CondCodeSDNode>(N.getOperand(2))->get(),
              MVT::isFloatingPoint(N.getOperand(1).getValueType()));
    return Result;
  case ISD::LOAD:
    // Make sure we generate both values.
    if (Result != 1) {  // Generate the token
      if (!ExprMap.insert(std::make_pair(N.getValue(1), 1)).second)
        assert(0 && "Load already emitted!?");
    } else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    switch (Node->getValueType(0)) {
    default: assert(0 && "Cannot load this type!");
    case MVT::i1:
    case MVT::i8:  Opc = X86::MOV8rm; break;
    case MVT::i16: Opc = X86::MOV16rm; break;
    case MVT::i32: Opc = X86::MOV32rm; break;
    case MVT::f32: Opc = X86::MOVSSrm; break;
    case MVT::f64:
      if (X86ScalarSSE) {
        Opc = X86::MOVSDrm;
      } else {
        Opc = X86::FLD64m;
        ContainsFPCode = true;
      }
      break;
    }

    if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(N.getOperand(1))){
      unsigned CPIdx = BB->getParent()->getConstantPool()->
         getConstantPoolIndex(CP->get());
      Select(N.getOperand(0));
      addConstantPoolReference(BuildMI(BB, Opc, 4, Result), CPIdx);
    } else {
      X86AddressMode AM;

      SDOperand Chain   = N.getOperand(0);
      SDOperand Address = N.getOperand(1);
      if (getRegPressure(Chain) > getRegPressure(Address)) {
        Select(Chain);
        SelectAddress(Address, AM);
      } else {
        SelectAddress(Address, AM);
        Select(Chain);
      }

      addFullAddress(BuildMI(BB, Opc, 4, Result), AM);
    }
    return Result;
  case X86ISD::FILD64m:
    // Make sure we generate both values.
    assert(Result != 1 && N.getValueType() == MVT::f64);
    if (!ExprMap.insert(std::make_pair(N.getValue(1), 1)).second)
      assert(0 && "Load already emitted!?");

    {
      X86AddressMode AM;

      SDOperand Chain   = N.getOperand(0);
      SDOperand Address = N.getOperand(1);
      if (getRegPressure(Chain) > getRegPressure(Address)) {
        Select(Chain);
        SelectAddress(Address, AM);
      } else {
        SelectAddress(Address, AM);
        Select(Chain);
      }

      addFullAddress(BuildMI(BB, X86::FILD64m, 4, Result), AM);
    }
    return Result;

  case ISD::EXTLOAD:          // Arbitrarily codegen extloads as MOVZX*
  case ISD::ZEXTLOAD: {
    // Make sure we generate both values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(N.getOperand(1)))
      if (Node->getValueType(0) == MVT::f64) {
        assert(cast<VTSDNode>(Node->getOperand(3))->getVT() == MVT::f32 &&
               "Bad EXTLOAD!");
        unsigned CPIdx = BB->getParent()->getConstantPool()->
          getConstantPoolIndex(CP->get());

        addConstantPoolReference(BuildMI(BB, X86::FLD32m, 4, Result), CPIdx);
        return Result;
      }

    X86AddressMode AM;
    if (getRegPressure(Node->getOperand(0)) >
           getRegPressure(Node->getOperand(1))) {
      Select(Node->getOperand(0)); // chain
      SelectAddress(Node->getOperand(1), AM);
    } else {
      SelectAddress(Node->getOperand(1), AM);
      Select(Node->getOperand(0)); // chain
    }

    switch (Node->getValueType(0)) {
    default: assert(0 && "Unknown type to sign extend to.");
    case MVT::f64:
      assert(cast<VTSDNode>(Node->getOperand(3))->getVT() == MVT::f32 &&
             "Bad EXTLOAD!");
      addFullAddress(BuildMI(BB, X86::FLD32m, 5, Result), AM);
      break;
    case MVT::i32:
      switch (cast<VTSDNode>(Node->getOperand(3))->getVT()) {
      default:
        assert(0 && "Bad zero extend!");
      case MVT::i1:
      case MVT::i8:
        addFullAddress(BuildMI(BB, X86::MOVZX32rm8, 5, Result), AM);
        break;
      case MVT::i16:
        addFullAddress(BuildMI(BB, X86::MOVZX32rm16, 5, Result), AM);
        break;
      }
      break;
    case MVT::i16:
      assert(cast<VTSDNode>(Node->getOperand(3))->getVT() <= MVT::i8 &&
             "Bad zero extend!");
      addFullAddress(BuildMI(BB, X86::MOVSX16rm8, 5, Result), AM);
      break;
    case MVT::i8:
      assert(cast<VTSDNode>(Node->getOperand(3))->getVT() == MVT::i1 &&
             "Bad zero extend!");
      addFullAddress(BuildMI(BB, X86::MOV8rm, 5, Result), AM);
      break;
    }
    return Result;
  }
  case ISD::SEXTLOAD: {
    // Make sure we generate both values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    X86AddressMode AM;
    if (getRegPressure(Node->getOperand(0)) >
           getRegPressure(Node->getOperand(1))) {
      Select(Node->getOperand(0)); // chain
      SelectAddress(Node->getOperand(1), AM);
    } else {
      SelectAddress(Node->getOperand(1), AM);
      Select(Node->getOperand(0)); // chain
    }

    switch (Node->getValueType(0)) {
    case MVT::i8: assert(0 && "Cannot sign extend from bool!");
    default: assert(0 && "Unknown type to sign extend to.");
    case MVT::i32:
      switch (cast<VTSDNode>(Node->getOperand(3))->getVT()) {
      default:
      case MVT::i1: assert(0 && "Cannot sign extend from bool!");
      case MVT::i8:
        addFullAddress(BuildMI(BB, X86::MOVSX32rm8, 5, Result), AM);
        break;
      case MVT::i16:
        addFullAddress(BuildMI(BB, X86::MOVSX32rm16, 5, Result), AM);
        break;
      }
      break;
    case MVT::i16:
      assert(cast<VTSDNode>(Node->getOperand(3))->getVT() == MVT::i8 &&
             "Cannot sign extend from bool!");
      addFullAddress(BuildMI(BB, X86::MOVSX16rm8, 5, Result), AM);
      break;
    }
    return Result;
  }

  case ISD::DYNAMIC_STACKALLOC:
    // Generate both result values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
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

    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      Select(N.getOperand(0));
      BuildMI(BB, X86::SUB32ri, 2, X86::ESP).addReg(X86::ESP)
        .addImm(CN->getValue());
    } else {
      if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(1))) {
        Select(N.getOperand(0));
        Tmp1 = SelectExpr(N.getOperand(1));
      } else {
        Tmp1 = SelectExpr(N.getOperand(1));
        Select(N.getOperand(0));
      }

      // Subtract size from stack pointer, thereby allocating some space.
      BuildMI(BB, X86::SUB32rr, 2, X86::ESP).addReg(X86::ESP).addReg(Tmp1);
    }

    // Put a pointer to the space into the result register, by copying the stack
    // pointer.
    BuildMI(BB, X86::MOV32rr, 1, Result).addReg(X86::ESP);
    return Result;

  case X86ISD::TAILCALL:
  case X86ISD::CALL: {
    // The chain for this call is now lowered.
    ExprMap.insert(std::make_pair(N.getValue(0), 1));

    bool isDirect = isa<GlobalAddressSDNode>(N.getOperand(1)) ||
                    isa<ExternalSymbolSDNode>(N.getOperand(1));
    unsigned Callee = 0;
    if (isDirect) {
      Select(N.getOperand(0));
    } else {
      if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(1))) {
        Select(N.getOperand(0));
        Callee = SelectExpr(N.getOperand(1));
      } else {
        Callee = SelectExpr(N.getOperand(1));
        Select(N.getOperand(0));
      }
    }

    // If this call has values to pass in registers, do so now.
    if (Node->getNumOperands() > 4) {
      // The first value is passed in (a part of) EAX, the second in EDX.
      unsigned RegOp1 = SelectExpr(N.getOperand(4));
      unsigned RegOp2 =
        Node->getNumOperands() > 5 ? SelectExpr(N.getOperand(5)) : 0;

      switch (N.getOperand(4).getValueType()) {
      default: assert(0 && "Bad thing to pass in regs");
      case MVT::i1:
      case MVT::i8:  BuildMI(BB, X86::MOV8rr , 1,X86::AL).addReg(RegOp1); break;
      case MVT::i16: BuildMI(BB, X86::MOV16rr, 1,X86::AX).addReg(RegOp1); break;
      case MVT::i32: BuildMI(BB, X86::MOV32rr, 1,X86::EAX).addReg(RegOp1);break;
      }
      if (RegOp2)
        switch (N.getOperand(5).getValueType()) {
        default: assert(0 && "Bad thing to pass in regs");
        case MVT::i1:
        case MVT::i8:
          BuildMI(BB, X86::MOV8rr , 1, X86::DL).addReg(RegOp2);
          break;
        case MVT::i16:
          BuildMI(BB, X86::MOV16rr, 1, X86::DX).addReg(RegOp2);
          break;
        case MVT::i32:
          BuildMI(BB, X86::MOV32rr, 1, X86::EDX).addReg(RegOp2);
          break;
        }
    }

    if (GlobalAddressSDNode *GASD =
               dyn_cast<GlobalAddressSDNode>(N.getOperand(1))) {
      BuildMI(BB, X86::CALLpcrel32, 1).addGlobalAddress(GASD->getGlobal(),true);
    } else if (ExternalSymbolSDNode *ESSDN =
               dyn_cast<ExternalSymbolSDNode>(N.getOperand(1))) {
      BuildMI(BB, X86::CALLpcrel32,
              1).addExternalSymbol(ESSDN->getSymbol(), true);
    } else {
      if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(1))) {
        Select(N.getOperand(0));
        Tmp1 = SelectExpr(N.getOperand(1));
      } else {
        Tmp1 = SelectExpr(N.getOperand(1));
        Select(N.getOperand(0));
      }

      BuildMI(BB, X86::CALL32r, 1).addReg(Tmp1);
    }

    // Get caller stack amount and amount the callee added to the stack pointer.
    Tmp1 = cast<ConstantSDNode>(N.getOperand(2))->getValue();
    Tmp2 = cast<ConstantSDNode>(N.getOperand(3))->getValue();
    BuildMI(BB, X86::ADJCALLSTACKUP, 2).addImm(Tmp1).addImm(Tmp2);

    if (Node->getNumValues() != 1)
      switch (Node->getValueType(1)) {
      default: assert(0 && "Unknown value type for call result!");
      case MVT::Other: return 1;
      case MVT::i1:
      case MVT::i8:
        BuildMI(BB, X86::MOV8rr, 1, Result).addReg(X86::AL);
        break;
      case MVT::i16:
        BuildMI(BB, X86::MOV16rr, 1, Result).addReg(X86::AX);
        break;
      case MVT::i32:
        BuildMI(BB, X86::MOV32rr, 1, Result).addReg(X86::EAX);
        if (Node->getNumValues() == 3 && Node->getValueType(2) == MVT::i32)
          BuildMI(BB, X86::MOV32rr, 1, Result+1).addReg(X86::EDX);
        break;
      case MVT::f64:     // Floating-point return values live in %ST(0)
        if (X86ScalarSSE) {
          ContainsFPCode = true;
          BuildMI(BB, X86::FpGETRESULT, 1, X86::FP0);

          unsigned Size = MVT::getSizeInBits(MVT::f64)/8;
          MachineFunction *F = BB->getParent();
          int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, Size);
          addFrameReference(BuildMI(BB, X86::FST64m, 5), FrameIdx).addReg(X86::FP0);
          addFrameReference(BuildMI(BB, X86::MOVSDrm, 4, Result), FrameIdx);
          break;
        } else {
          ContainsFPCode = true;
          BuildMI(BB, X86::FpGETRESULT, 1, Result);
          break;
        }
      }
    return Result+N.ResNo-1;
  }
  case ISD::READPORT:
    // First, determine that the size of the operand falls within the acceptable
    // range for this architecture.
    //
    if (Node->getOperand(1).getValueType() != MVT::i16) {
      std::cerr << "llvm.readport: Address size is not 16 bits\n";
      exit(1);
    }

    // Make sure we generate both values.
    if (Result != 1) {  // Generate the token
      if (!ExprMap.insert(std::make_pair(N.getValue(1), 1)).second)
        assert(0 && "readport already emitted!?");
    } else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    Select(Node->getOperand(0));  // Select the chain.

    // If the port is a single-byte constant, use the immediate form.
    if (ConstantSDNode *Port = dyn_cast<ConstantSDNode>(Node->getOperand(1)))
      if ((Port->getValue() & 255) == Port->getValue()) {
        switch (Node->getValueType(0)) {
        case MVT::i8:
          BuildMI(BB, X86::IN8ri, 1).addImm(Port->getValue());
          BuildMI(BB, X86::MOV8rr, 1, Result).addReg(X86::AL);
          return Result;
        case MVT::i16:
          BuildMI(BB, X86::IN16ri, 1).addImm(Port->getValue());
          BuildMI(BB, X86::MOV16rr, 1, Result).addReg(X86::AX);
          return Result;
        case MVT::i32:
          BuildMI(BB, X86::IN32ri, 1).addImm(Port->getValue());
          BuildMI(BB, X86::MOV32rr, 1, Result).addReg(X86::EAX);
          return Result;
        default: break;
        }
      }

    // Now, move the I/O port address into the DX register and use the IN
    // instruction to get the input data.
    //
    Tmp1 = SelectExpr(Node->getOperand(1));
    BuildMI(BB, X86::MOV16rr, 1, X86::DX).addReg(Tmp1);
    switch (Node->getValueType(0)) {
    case MVT::i8:
      BuildMI(BB, X86::IN8rr, 0);
      BuildMI(BB, X86::MOV8rr, 1, Result).addReg(X86::AL);
      return Result;
    case MVT::i16:
      BuildMI(BB, X86::IN16rr, 0);
      BuildMI(BB, X86::MOV16rr, 1, Result).addReg(X86::AX);
      return Result;
    case MVT::i32:
      BuildMI(BB, X86::IN32rr, 0);
      BuildMI(BB, X86::MOV32rr, 1, Result).addReg(X86::EAX);
      return Result;
    default:
      std::cerr << "Cannot do input on this data type";
      exit(1);
    }

  }

  return 0;
}

/// TryToFoldLoadOpStore - Given a store node, try to fold together a
/// load/op/store instruction.  If successful return true.
bool ISel::TryToFoldLoadOpStore(SDNode *Node) {
  assert(Node->getOpcode() == ISD::STORE && "Can only do this for stores!");
  SDOperand Chain  = Node->getOperand(0);
  SDOperand StVal  = Node->getOperand(1);
  SDOperand StPtr  = Node->getOperand(2);

  // The chain has to be a load, the stored value must be an integer binary
  // operation with one use.
  if (!StVal.Val->hasOneUse() || StVal.Val->getNumOperands() != 2 ||
      MVT::isFloatingPoint(StVal.getValueType()))
    return false;

  // Token chain must either be a factor node or the load to fold.
  if (Chain.getOpcode() != ISD::LOAD && Chain.getOpcode() != ISD::TokenFactor)
    return false;

  SDOperand TheLoad;

  // Check to see if there is a load from the same pointer that we're storing
  // to in either operand of the binop.
  if (StVal.getOperand(0).getOpcode() == ISD::LOAD &&
      StVal.getOperand(0).getOperand(1) == StPtr)
    TheLoad = StVal.getOperand(0);
  else if (StVal.getOperand(1).getOpcode() == ISD::LOAD &&
           StVal.getOperand(1).getOperand(1) == StPtr)
    TheLoad = StVal.getOperand(1);
  else
    return false;  // No matching load operand.

  // We can only fold the load if there are no intervening side-effecting
  // operations.  This means that the store uses the load as its token chain, or
  // there are only token factor nodes in between the store and load.
  if (Chain != TheLoad.getValue(1)) {
    // Okay, the other option is that we have a store referring to (possibly
    // nested) token factor nodes.  For now, just try peeking through one level
    // of token factors to see if this is the case.
    bool ChainOk = false;
    if (Chain.getOpcode() == ISD::TokenFactor) {
      for (unsigned i = 0, e = Chain.getNumOperands(); i != e; ++i)
        if (Chain.getOperand(i) == TheLoad.getValue(1)) {
          ChainOk = true;
          break;
        }
    }

    if (!ChainOk) return false;
  }

  if (TheLoad.getOperand(1) != StPtr)
    return false;

  // Make sure that one of the operands of the binop is the load, and that the
  // load folds into the binop.
  if (((StVal.getOperand(0) != TheLoad ||
        !isFoldableLoad(TheLoad, StVal.getOperand(1))) &&
       (StVal.getOperand(1) != TheLoad ||
        !isFoldableLoad(TheLoad, StVal.getOperand(0)))))
    return false;

  // Finally, check to see if this is one of the ops we can handle!
  static const unsigned ADDTAB[] = {
    X86::ADD8mi, X86::ADD16mi, X86::ADD32mi,
    X86::ADD8mr, X86::ADD16mr, X86::ADD32mr,
  };
  static const unsigned SUBTAB[] = {
    X86::SUB8mi, X86::SUB16mi, X86::SUB32mi,
    X86::SUB8mr, X86::SUB16mr, X86::SUB32mr,
  };
  static const unsigned ANDTAB[] = {
    X86::AND8mi, X86::AND16mi, X86::AND32mi,
    X86::AND8mr, X86::AND16mr, X86::AND32mr,
  };
  static const unsigned ORTAB[] = {
    X86::OR8mi, X86::OR16mi, X86::OR32mi,
    X86::OR8mr, X86::OR16mr, X86::OR32mr,
  };
  static const unsigned XORTAB[] = {
    X86::XOR8mi, X86::XOR16mi, X86::XOR32mi,
    X86::XOR8mr, X86::XOR16mr, X86::XOR32mr,
  };
  static const unsigned SHLTAB[] = {
    X86::SHL8mi, X86::SHL16mi, X86::SHL32mi,
    /*Have to put the reg in CL*/0, 0, 0,
  };
  static const unsigned SARTAB[] = {
    X86::SAR8mi, X86::SAR16mi, X86::SAR32mi,
    /*Have to put the reg in CL*/0, 0, 0,
  };
  static const unsigned SHRTAB[] = {
    X86::SHR8mi, X86::SHR16mi, X86::SHR32mi,
    /*Have to put the reg in CL*/0, 0, 0,
  };

  const unsigned *TabPtr = 0;
  switch (StVal.getOpcode()) {
  default:
    std::cerr << "CANNOT [mem] op= val: ";
    StVal.Val->dump(); std::cerr << "\n";
  case ISD::FMUL:
  case ISD::MUL:
  case ISD::FDIV:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::FREM:
  case ISD::SREM:
  case ISD::UREM: return false;

  case ISD::ADD: TabPtr = ADDTAB; break;
  case ISD::SUB: TabPtr = SUBTAB; break;
  case ISD::AND: TabPtr = ANDTAB; break;
  case ISD:: OR: TabPtr =  ORTAB; break;
  case ISD::XOR: TabPtr = XORTAB; break;
  case ISD::SHL: TabPtr = SHLTAB; break;
  case ISD::SRA: TabPtr = SARTAB; break;
  case ISD::SRL: TabPtr = SHRTAB; break;
  }

  // Handle: [mem] op= CST
  SDOperand Op0 = StVal.getOperand(0);
  SDOperand Op1 = StVal.getOperand(1);
  unsigned Opc = 0;
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op1)) {
    switch (Op0.getValueType()) { // Use Op0's type because of shifts.
    default: break;
    case MVT::i1:
    case MVT::i8:  Opc = TabPtr[0]; break;
    case MVT::i16: Opc = TabPtr[1]; break;
    case MVT::i32: Opc = TabPtr[2]; break;
    }

    if (Opc) {
      if (!ExprMap.insert(std::make_pair(TheLoad.getValue(1), 1)).second)
        assert(0 && "Already emitted?");
      Select(Chain);

      X86AddressMode AM;
      if (getRegPressure(TheLoad.getOperand(0)) >
          getRegPressure(TheLoad.getOperand(1))) {
        Select(TheLoad.getOperand(0));
        SelectAddress(TheLoad.getOperand(1), AM);
      } else {
        SelectAddress(TheLoad.getOperand(1), AM);
        Select(TheLoad.getOperand(0));
      }

      if (StVal.getOpcode() == ISD::ADD) {
        if (CN->getValue() == 1) {
          switch (Op0.getValueType()) {
          default: break;
          case MVT::i8:
            addFullAddress(BuildMI(BB, X86::INC8m, 4), AM);
            return true;
          case MVT::i16: Opc = TabPtr[1];
            addFullAddress(BuildMI(BB, X86::INC16m, 4), AM);
            return true;
          case MVT::i32: Opc = TabPtr[2];
            addFullAddress(BuildMI(BB, X86::INC32m, 4), AM);
            return true;
          }
        } else if (CN->getValue()+1 == 0) {   // [X] += -1 -> DEC [X]
          switch (Op0.getValueType()) {
          default: break;
          case MVT::i8:
            addFullAddress(BuildMI(BB, X86::DEC8m, 4), AM);
            return true;
          case MVT::i16: Opc = TabPtr[1];
            addFullAddress(BuildMI(BB, X86::DEC16m, 4), AM);
            return true;
          case MVT::i32: Opc = TabPtr[2];
            addFullAddress(BuildMI(BB, X86::DEC32m, 4), AM);
            return true;
          }
        }
      }

      addFullAddress(BuildMI(BB, Opc, 4+1),AM).addImm(CN->getValue());
      return true;
    }
  }

  // If we have [mem] = V op [mem], try to turn it into:
  // [mem] = [mem] op V.
  if (Op1 == TheLoad && 
      StVal.getOpcode() != ISD::SUB && StVal.getOpcode() != ISD::FSUB &&
      StVal.getOpcode() != ISD::SHL && StVal.getOpcode() != ISD::SRA &&
      StVal.getOpcode() != ISD::SRL)
    std::swap(Op0, Op1);

  if (Op0 != TheLoad) return false;

  switch (Op0.getValueType()) {
  default: return false;
  case MVT::i1:
  case MVT::i8:  Opc = TabPtr[3]; break;
  case MVT::i16: Opc = TabPtr[4]; break;
  case MVT::i32: Opc = TabPtr[5]; break;
  }

  // Table entry doesn't exist?
  if (Opc == 0) return false;

  if (!ExprMap.insert(std::make_pair(TheLoad.getValue(1), 1)).second)
    assert(0 && "Already emitted?");
  Select(Chain);
  Select(TheLoad.getOperand(0));

  X86AddressMode AM;
  SelectAddress(TheLoad.getOperand(1), AM);
  unsigned Reg = SelectExpr(Op1);
  addFullAddress(BuildMI(BB, Opc, 4+1), AM).addReg(Reg);
  return true;
}

/// If node is a ret(tailcall) node, emit the specified tail call and return
/// true, otherwise return false.
///
/// FIXME: This whole thing should be a post-legalize optimization pass which
/// recognizes and transforms the dag.  We don't want the selection phase doing
/// this stuff!!
///
bool ISel::EmitPotentialTailCall(SDNode *RetNode) {
  assert(RetNode->getOpcode() == ISD::RET && "Not a return");

  SDOperand Chain = RetNode->getOperand(0);

  // If this is a token factor node where one operand is a call, dig into it.
  SDOperand TokFactor;
  unsigned TokFactorOperand = 0;
  if (Chain.getOpcode() == ISD::TokenFactor) {
    for (unsigned i = 0, e = Chain.getNumOperands(); i != e; ++i)
      if (Chain.getOperand(i).getOpcode() == ISD::CALLSEQ_END ||
          Chain.getOperand(i).getOpcode() == X86ISD::TAILCALL) {
        TokFactorOperand = i;
        TokFactor = Chain;
        Chain = Chain.getOperand(i);
        break;
      }
    if (TokFactor.Val == 0) return false;  // No call operand.
  }

  // Skip the CALLSEQ_END node if present.
  if (Chain.getOpcode() == ISD::CALLSEQ_END)
    Chain = Chain.getOperand(0);

  // Is a tailcall the last control operation that occurs before the return?
  if (Chain.getOpcode() != X86ISD::TAILCALL)
    return false;

  // If we return a value, is it the value produced by the call?
  if (RetNode->getNumOperands() > 1) {
    // Not returning the ret val of the call?
    if (Chain.Val->getNumValues() == 1 ||
        RetNode->getOperand(1) != Chain.getValue(1))
      return false;

    if (RetNode->getNumOperands() > 2) {
      if (Chain.Val->getNumValues() == 2 ||
          RetNode->getOperand(2) != Chain.getValue(2))
        return false;
    }
    assert(RetNode->getNumOperands() <= 3);
  }

  // CalleeCallArgAmt - The total number of bytes used for the callee arg area.
  // For FastCC, this will always be > 0.
  unsigned CalleeCallArgAmt =
    cast<ConstantSDNode>(Chain.getOperand(2))->getValue();

  // CalleeCallArgPopAmt - The number of bytes in the call area popped by the
  // callee.  For FastCC this will always be > 0, for CCC this is always 0.
  unsigned CalleeCallArgPopAmt =
    cast<ConstantSDNode>(Chain.getOperand(3))->getValue();

  // There are several cases we can handle here.  First, if the caller and
  // callee are both CCC functions, we can tailcall if the callee takes <= the
  // number of argument bytes that the caller does.
  if (CalleeCallArgPopAmt == 0 &&                  // Callee is C CallingConv?
      X86Lowering.getBytesToPopOnReturn() == 0) {  // Caller is C CallingConv?
    // Check to see if caller arg area size >= callee arg area size.
    if (X86Lowering.getBytesCallerReserves() >= CalleeCallArgAmt) {
      //std::cerr << "CCC TAILCALL UNIMP!\n";
      // If TokFactor is non-null, emit all operands.

      //EmitCCCToCCCTailCall(Chain.Val);
      //return true;
    }
    return false;
  }

  // Second, if both are FastCC functions, we can always perform the tail call.
  if (CalleeCallArgPopAmt && X86Lowering.getBytesToPopOnReturn()) {
    // If TokFactor is non-null, emit all operands before the call.
    if (TokFactor.Val) {
      for (unsigned i = 0, e = TokFactor.getNumOperands(); i != e; ++i)
        if (i != TokFactorOperand)
          Select(TokFactor.getOperand(i));
    }

    EmitFastCCToFastCCTailCall(Chain.Val);
    return true;
  }

  // We don't support mixed calls, due to issues with alignment.  We could in
  // theory handle some mixed calls from CCC -> FastCC if the stack is properly
  // aligned (which depends on the number of arguments to the callee).  TODO.
  return false;
}

static SDOperand GetAdjustedArgumentStores(SDOperand Chain, int Offset,
                                           SelectionDAG &DAG) {
  MVT::ValueType StoreVT;
  switch (Chain.getOpcode()) {
  default: assert(0 && "Unexpected node!");
  case ISD::CALLSEQ_START:
    // If we found the start of the call sequence, we're done.  We actually
    // strip off the CALLSEQ_START node, to avoid generating the
    // ADJCALLSTACKDOWN marker for the tail call.
    return Chain.getOperand(0);
  case ISD::TokenFactor: {
    std::vector<SDOperand> Ops;
    Ops.reserve(Chain.getNumOperands());
    for (unsigned i = 0, e = Chain.getNumOperands(); i != e; ++i)
      Ops.push_back(GetAdjustedArgumentStores(Chain.getOperand(i), Offset,DAG));
    return DAG.getNode(ISD::TokenFactor, MVT::Other, Ops);
  }
  case ISD::STORE:       // Normal store
    StoreVT = Chain.getOperand(1).getValueType();
    break;
  case ISD::TRUNCSTORE:  // FLOAT store
    StoreVT = cast<VTSDNode>(Chain.getOperand(4))->getVT();
    break;
  }

  SDOperand OrigDest = Chain.getOperand(2);
  unsigned OrigOffset;

  if (OrigDest.getOpcode() == ISD::CopyFromReg) {
    OrigOffset = 0;
    assert(cast<RegisterSDNode>(OrigDest.getOperand(1))->getReg() == X86::ESP);
  } else {
    // We expect only (ESP+C)
    assert(OrigDest.getOpcode() == ISD::ADD &&
           isa<ConstantSDNode>(OrigDest.getOperand(1)) &&
           OrigDest.getOperand(0).getOpcode() == ISD::CopyFromReg &&
           cast<RegisterSDNode>(OrigDest.getOperand(0).getOperand(1))->getReg()
                 == X86::ESP);
    OrigOffset = cast<ConstantSDNode>(OrigDest.getOperand(1))->getValue();
  }

  // Compute the new offset from the incoming ESP value we wish to use.
  unsigned NewOffset = OrigOffset + Offset;

  unsigned OpSize = (MVT::getSizeInBits(StoreVT)+7)/8;  // Bits -> Bytes
  MachineFunction &MF = DAG.getMachineFunction();
  int FI = MF.getFrameInfo()->CreateFixedObject(OpSize, NewOffset);
  SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);

  SDOperand InChain = GetAdjustedArgumentStores(Chain.getOperand(0), Offset,
                                                DAG);
  if (Chain.getOpcode() == ISD::STORE)
    return DAG.getNode(ISD::STORE, MVT::Other, InChain, Chain.getOperand(1),
                       FIN);
  assert(Chain.getOpcode() == ISD::TRUNCSTORE);
  return DAG.getNode(ISD::TRUNCSTORE, MVT::Other, InChain, Chain.getOperand(1),
                     FIN, DAG.getSrcValue(NULL), DAG.getValueType(StoreVT));
}


/// EmitFastCCToFastCCTailCall - Given a tailcall in the tail position to a
/// fastcc function from a fastcc function, emit the code to emit a 'proper'
/// tail call.
void ISel::EmitFastCCToFastCCTailCall(SDNode *TailCallNode) {
  unsigned CalleeCallArgSize =
    cast<ConstantSDNode>(TailCallNode->getOperand(2))->getValue();
  unsigned CallerArgSize = X86Lowering.getBytesToPopOnReturn();

  //std::cerr << "****\n*** EMITTING TAIL CALL!\n****\n";

  // Adjust argument stores.  Instead of storing to [ESP], f.e., store to frame
  // indexes that are relative to the incoming ESP.  If the incoming and
  // outgoing arg sizes are the same we will store to [InESP] instead of
  // [CurESP] and the ESP referenced will be relative to the incoming function
  // ESP.
  int ESPOffset = CallerArgSize-CalleeCallArgSize;
  SDOperand AdjustedArgStores =
    GetAdjustedArgumentStores(TailCallNode->getOperand(0), ESPOffset, *TheDAG);

  // Copy the return address of the caller into a virtual register so we don't
  // clobber it.
  SDOperand RetVal;
  if (ESPOffset) {
    SDOperand RetValAddr = X86Lowering.getReturnAddressFrameIndex(*TheDAG);
    RetVal = TheDAG->getLoad(MVT::i32, TheDAG->getEntryNode(),
                                       RetValAddr, TheDAG->getSrcValue(NULL));
    SelectExpr(RetVal);
  }

  // Codegen all of the argument stores.
  Select(AdjustedArgStores);

  if (RetVal.Val) {
    // Emit a store of the saved ret value to the new location.
    MachineFunction &MF = TheDAG->getMachineFunction();
    int ReturnAddrFI = MF.getFrameInfo()->CreateFixedObject(4, ESPOffset-4);
    SDOperand RetValAddr = TheDAG->getFrameIndex(ReturnAddrFI, MVT::i32);
    Select(TheDAG->getNode(ISD::STORE, MVT::Other, TheDAG->getEntryNode(),
                           RetVal, RetValAddr));
  }

  // Get the destination value.
  SDOperand Callee = TailCallNode->getOperand(1);
  bool isDirect = isa<GlobalAddressSDNode>(Callee) ||
                  isa<ExternalSymbolSDNode>(Callee);
  unsigned CalleeReg = 0;
  if (!isDirect) CalleeReg = SelectExpr(Callee);

  unsigned RegOp1 = 0;
  unsigned RegOp2 = 0;

  if (TailCallNode->getNumOperands() > 4) {
    // The first value is passed in (a part of) EAX, the second in EDX.
    RegOp1 = SelectExpr(TailCallNode->getOperand(4));
    if (TailCallNode->getNumOperands() > 5)
      RegOp2 = SelectExpr(TailCallNode->getOperand(5));

    switch (TailCallNode->getOperand(4).getValueType()) {
    default: assert(0 && "Bad thing to pass in regs");
    case MVT::i1:
    case MVT::i8:
      BuildMI(BB, X86::MOV8rr, 1, X86::AL).addReg(RegOp1);
      RegOp1 = X86::AL;
      break;
    case MVT::i16:
      BuildMI(BB, X86::MOV16rr, 1,X86::AX).addReg(RegOp1);
      RegOp1 = X86::AX;
      break;
    case MVT::i32:
      BuildMI(BB, X86::MOV32rr, 1,X86::EAX).addReg(RegOp1);
      RegOp1 = X86::EAX;
      break;
    }
    if (RegOp2)
      switch (TailCallNode->getOperand(5).getValueType()) {
      default: assert(0 && "Bad thing to pass in regs");
      case MVT::i1:
      case MVT::i8:
        BuildMI(BB, X86::MOV8rr, 1, X86::DL).addReg(RegOp2);
        RegOp2 = X86::DL;
        break;
      case MVT::i16:
        BuildMI(BB, X86::MOV16rr, 1, X86::DX).addReg(RegOp2);
        RegOp2 = X86::DX;
        break;
      case MVT::i32:
        BuildMI(BB, X86::MOV32rr, 1, X86::EDX).addReg(RegOp2);
        RegOp2 = X86::EDX;
        break;
      }
  }

  // Adjust ESP.
  if (ESPOffset)
    BuildMI(BB, X86::ADJSTACKPTRri, 2,
            X86::ESP).addReg(X86::ESP).addImm(ESPOffset);

  // TODO: handle jmp [mem]
  if (!isDirect) {
    BuildMI(BB, X86::TAILJMPr, 1).addReg(CalleeReg);
  } else if (GlobalAddressSDNode *GASD = dyn_cast<GlobalAddressSDNode>(Callee)){
    BuildMI(BB, X86::TAILJMPd, 1).addGlobalAddress(GASD->getGlobal(), true);
  } else {
    ExternalSymbolSDNode *ESSDN = cast<ExternalSymbolSDNode>(Callee);
    BuildMI(BB, X86::TAILJMPd, 1).addExternalSymbol(ESSDN->getSymbol(), true);
  }
  // ADD IMPLICIT USE RegOp1/RegOp2's
}


void ISel::Select(SDOperand N) {
  unsigned Tmp1, Tmp2, Opc;

  if (!ExprMap.insert(std::make_pair(N, 1)).second)
    return;  // Already selected.

  SDNode *Node = N.Val;

  switch (Node->getOpcode()) {
  default:
    Node->dump(); std::cerr << "\n";
    assert(0 && "Node not handled yet!");
  case ISD::EntryToken: return;  // Noop
  case ISD::TokenFactor:
    if (Node->getNumOperands() == 2) {
      bool OneFirst =
        getRegPressure(Node->getOperand(1))>getRegPressure(Node->getOperand(0));
      Select(Node->getOperand(OneFirst));
      Select(Node->getOperand(!OneFirst));
    } else {
      std::vector<std::pair<unsigned, unsigned> > OpsP;
      for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
        OpsP.push_back(std::make_pair(getRegPressure(Node->getOperand(i)), i));
      std::sort(OpsP.begin(), OpsP.end());
      std::reverse(OpsP.begin(), OpsP.end());
      for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
        Select(Node->getOperand(OpsP[i].second));
    }
    return;
  case ISD::CopyToReg:
    if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(2))) {
      Select(N.getOperand(0));
      Tmp1 = SelectExpr(N.getOperand(2));
    } else {
      Tmp1 = SelectExpr(N.getOperand(2));
      Select(N.getOperand(0));
    }
    Tmp2 = cast<RegisterSDNode>(N.getOperand(1))->getReg();

    if (Tmp1 != Tmp2) {
      switch (N.getOperand(2).getValueType()) {
      default: assert(0 && "Invalid type for operation!");
      case MVT::i1:
      case MVT::i8:  Opc = X86::MOV8rr; break;
      case MVT::i16: Opc = X86::MOV16rr; break;
      case MVT::i32: Opc = X86::MOV32rr; break;
      case MVT::f32: Opc = X86::MOVAPSrr; break;
      case MVT::f64:
        if (X86ScalarSSE) {
          Opc = X86::MOVAPDrr;
        } else {
          Opc = X86::FpMOV;
          ContainsFPCode = true;
        }
        break;
      }
      BuildMI(BB, Opc, 1, Tmp2).addReg(Tmp1);
    }
    return;
  case ISD::RET:
    if (N.getOperand(0).getOpcode() == ISD::CALLSEQ_END ||
        N.getOperand(0).getOpcode() == X86ISD::TAILCALL ||
        N.getOperand(0).getOpcode() == ISD::TokenFactor)
      if (EmitPotentialTailCall(Node))
        return;

    switch (N.getNumOperands()) {
    default:
      assert(0 && "Unknown return instruction!");
    case 3:
      assert(N.getOperand(1).getValueType() == MVT::i32 &&
             N.getOperand(2).getValueType() == MVT::i32 &&
             "Unknown two-register value!");
      if (getRegPressure(N.getOperand(1)) > getRegPressure(N.getOperand(2))) {
        Tmp1 = SelectExpr(N.getOperand(1));
        Tmp2 = SelectExpr(N.getOperand(2));
      } else {
        Tmp2 = SelectExpr(N.getOperand(2));
        Tmp1 = SelectExpr(N.getOperand(1));
      }
      Select(N.getOperand(0));

      BuildMI(BB, X86::MOV32rr, 1, X86::EAX).addReg(Tmp1);
      BuildMI(BB, X86::MOV32rr, 1, X86::EDX).addReg(Tmp2);
      break;
    case 2:
      if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(1))) {
        Select(N.getOperand(0));
        Tmp1 = SelectExpr(N.getOperand(1));
      } else {
        Tmp1 = SelectExpr(N.getOperand(1));
        Select(N.getOperand(0));
      }
      switch (N.getOperand(1).getValueType()) {
      default: assert(0 && "All other types should have been promoted!!");
      case MVT::f32:
        if (X86ScalarSSE) {
          // Spill the value to memory and reload it into top of stack.
          unsigned Size = MVT::getSizeInBits(MVT::f32)/8;
          MachineFunction *F = BB->getParent();
          int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, Size);
          addFrameReference(BuildMI(BB, X86::MOVSSmr, 5), FrameIdx).addReg(Tmp1);
          addFrameReference(BuildMI(BB, X86::FLD32m, 4, X86::FP0), FrameIdx);
          BuildMI(BB, X86::FpSETRESULT, 1).addReg(X86::FP0);
          ContainsFPCode = true;
        } else {
          assert(0 && "MVT::f32 only legal with scalar sse fp");
          abort();
        }
        break;
      case MVT::f64:
        if (X86ScalarSSE) {
          // Spill the value to memory and reload it into top of stack.
          unsigned Size = MVT::getSizeInBits(MVT::f64)/8;
          MachineFunction *F = BB->getParent();
          int FrameIdx = F->getFrameInfo()->CreateStackObject(Size, Size);
          addFrameReference(BuildMI(BB, X86::MOVSDmr, 5), FrameIdx).addReg(Tmp1);
          addFrameReference(BuildMI(BB, X86::FLD64m, 4, X86::FP0), FrameIdx);
          BuildMI(BB, X86::FpSETRESULT, 1).addReg(X86::FP0);
          ContainsFPCode = true;
        } else {
          BuildMI(BB, X86::FpSETRESULT, 1).addReg(Tmp1);
        }
        break;
      case MVT::i32:
        BuildMI(BB, X86::MOV32rr, 1, X86::EAX).addReg(Tmp1);
        break;
      }
      break;
    case 1:
      Select(N.getOperand(0));
      break;
    }
    if (X86Lowering.getBytesToPopOnReturn() == 0)
      BuildMI(BB, X86::RET, 0); // Just emit a 'ret' instruction
    else
      BuildMI(BB, X86::RETI, 1).addImm(X86Lowering.getBytesToPopOnReturn());
    return;
  case ISD::BR: {
    Select(N.getOperand(0));
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(1))->getBasicBlock();
    BuildMI(BB, X86::JMP, 1).addMBB(Dest);
    return;
  }

  case ISD::BRCOND: {
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(2))->getBasicBlock();

    // Try to fold a setcc into the branch.  If this fails, emit a test/jne
    // pair.
    if (EmitBranchCC(Dest, N.getOperand(0), N.getOperand(1))) {
      if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(1))) {
        Select(N.getOperand(0));
        Tmp1 = SelectExpr(N.getOperand(1));
      } else {
        Tmp1 = SelectExpr(N.getOperand(1));
        Select(N.getOperand(0));
      }
      BuildMI(BB, X86::TEST8rr, 2).addReg(Tmp1).addReg(Tmp1);
      BuildMI(BB, X86::JNE, 1).addMBB(Dest);
    }

    return;
  }

  case ISD::LOAD:
    // If this load could be folded into the only using instruction, and if it
    // is safe to emit the instruction here, try to do so now.
    if (Node->hasNUsesOfValue(1, 0)) {
      SDOperand TheVal = N.getValue(0);
      SDNode *User = 0;
      for (SDNode::use_iterator UI = Node->use_begin(); ; ++UI) {
        assert(UI != Node->use_end() && "Didn't find use!");
        SDNode *UN = *UI;
        for (unsigned i = 0, e = UN->getNumOperands(); i != e; ++i)
          if (UN->getOperand(i) == TheVal) {
            User = UN;
            goto FoundIt;
          }
      }
    FoundIt:
      // Only handle unary operators right now.
      if (User->getNumOperands() == 1) {
        ExprMap.erase(N);
        SelectExpr(SDOperand(User, 0));
        return;
      }
    }
    ExprMap.erase(N);
    SelectExpr(N);
    return;
  case ISD::READPORT:
  case ISD::EXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::DYNAMIC_STACKALLOC:
  case X86ISD::TAILCALL:
  case X86ISD::CALL:
    ExprMap.erase(N);
    SelectExpr(N);
    return;
  case ISD::CopyFromReg:
  case X86ISD::FILD64m:
    ExprMap.erase(N);
    SelectExpr(N.getValue(0));
    return;

  case X86ISD::FP_TO_INT16_IN_MEM:
  case X86ISD::FP_TO_INT32_IN_MEM:
  case X86ISD::FP_TO_INT64_IN_MEM: {
    assert(N.getOperand(1).getValueType() == MVT::f64);
    X86AddressMode AM;
    Select(N.getOperand(0));   // Select the token chain

    unsigned ValReg;
    if (getRegPressure(N.getOperand(1)) > getRegPressure(N.getOperand(2))) {
      ValReg = SelectExpr(N.getOperand(1));
      SelectAddress(N.getOperand(2), AM);
     } else {
       SelectAddress(N.getOperand(2), AM);
       ValReg = SelectExpr(N.getOperand(1));
     }

    // Change the floating point control register to use "round towards zero"
    // mode when truncating to an integer value.
    //
    MachineFunction *F = BB->getParent();
    int CWFrameIdx = F->getFrameInfo()->CreateStackObject(2, 2);
    addFrameReference(BuildMI(BB, X86::FNSTCW16m, 4), CWFrameIdx);

    // Load the old value of the high byte of the control word...
    unsigned OldCW = MakeReg(MVT::i16);
    addFrameReference(BuildMI(BB, X86::MOV16rm, 4, OldCW), CWFrameIdx);

    // Set the high part to be round to zero...
    addFrameReference(BuildMI(BB, X86::MOV16mi, 5), CWFrameIdx).addImm(0xC7F);

    // Reload the modified control word now...
    addFrameReference(BuildMI(BB, X86::FLDCW16m, 4), CWFrameIdx);

    // Restore the memory image of control word to original value
    addFrameReference(BuildMI(BB, X86::MOV16mr, 5), CWFrameIdx).addReg(OldCW);

    // Get the X86 opcode to use.
    switch (N.getOpcode()) {
    case X86ISD::FP_TO_INT16_IN_MEM: Tmp1 = X86::FIST16m; break;
    case X86ISD::FP_TO_INT32_IN_MEM: Tmp1 = X86::FIST32m; break;
    case X86ISD::FP_TO_INT64_IN_MEM: Tmp1 = X86::FISTP64m; break;
    }

    addFullAddress(BuildMI(BB, Tmp1, 5), AM).addReg(ValReg);

    // Reload the original control word now.
    addFrameReference(BuildMI(BB, X86::FLDCW16m, 4), CWFrameIdx);
    return;
  }

  case ISD::TRUNCSTORE: {  // truncstore chain, val, ptr, SRCVALUE, storety
    X86AddressMode AM;
    MVT::ValueType StoredTy = cast<VTSDNode>(N.getOperand(4))->getVT();
    assert((StoredTy == MVT::i1 || StoredTy == MVT::f32 ||
            StoredTy == MVT::i16 /*FIXME: THIS IS JUST FOR TESTING!*/)
           && "Unsupported TRUNCSTORE for this target!");

    if (StoredTy == MVT::i16) {
      // FIXME: This is here just to allow testing.  X86 doesn't really have a
      // TRUNCSTORE i16 operation, but this is required for targets that do not
      // have 16-bit integer registers.  We occasionally disable 16-bit integer
      // registers to test the promotion code.
      Select(N.getOperand(0));
      Tmp1 = SelectExpr(N.getOperand(1));
      SelectAddress(N.getOperand(2), AM);

      BuildMI(BB, X86::MOV32rr, 1, X86::EAX).addReg(Tmp1);
      addFullAddress(BuildMI(BB, X86::MOV16mr, 5), AM).addReg(X86::AX);
      return;
    }

    // Store of constant bool?
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(2))) {
        Select(N.getOperand(0));
        SelectAddress(N.getOperand(2), AM);
      } else {
        SelectAddress(N.getOperand(2), AM);
        Select(N.getOperand(0));
      }
      addFullAddress(BuildMI(BB, X86::MOV8mi, 5), AM).addImm(CN->getValue());
      return;
    }

    switch (StoredTy) {
    default: assert(0 && "Cannot truncstore this type!");
    case MVT::i1: Opc = X86::MOV8mr; break;
    case MVT::f32:
      assert(!X86ScalarSSE && "Cannot truncstore scalar SSE regs");
      Opc = X86::FST32m; break;
    }

    std::vector<std::pair<unsigned, unsigned> > RP;
    RP.push_back(std::make_pair(getRegPressure(N.getOperand(0)), 0));
    RP.push_back(std::make_pair(getRegPressure(N.getOperand(1)), 1));
    RP.push_back(std::make_pair(getRegPressure(N.getOperand(2)), 2));
    std::sort(RP.begin(), RP.end());

    Tmp1 = 0;   // Silence a warning.
    for (unsigned i = 0; i != 3; ++i)
      switch (RP[2-i].second) {
      default: assert(0 && "Unknown operand number!");
      case 0: Select(N.getOperand(0)); break;
      case 1: Tmp1 = SelectExpr(N.getOperand(1)); break;
      case 2: SelectAddress(N.getOperand(2), AM); break;
      }

    addFullAddress(BuildMI(BB, Opc, 4+1), AM).addReg(Tmp1);
    return;
  }
  case ISD::STORE: {
    X86AddressMode AM;

    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      Opc = 0;
      switch (CN->getValueType(0)) {
      default: assert(0 && "Invalid type for operation!");
      case MVT::i1:
      case MVT::i8:  Opc = X86::MOV8mi; break;
      case MVT::i16: Opc = X86::MOV16mi; break;
      case MVT::i32: Opc = X86::MOV32mi; break;
      }
      if (Opc) {
        if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(2))) {
          Select(N.getOperand(0));
          SelectAddress(N.getOperand(2), AM);
        } else {
          SelectAddress(N.getOperand(2), AM);
          Select(N.getOperand(0));
        }
        addFullAddress(BuildMI(BB, Opc, 4+1), AM).addImm(CN->getValue());
        return;
      }
    } else if (GlobalAddressSDNode *GA =
                      dyn_cast<GlobalAddressSDNode>(N.getOperand(1))) {
      assert(GA->getValueType(0) == MVT::i32 && "Bad pointer operand");

      if (getRegPressure(N.getOperand(0)) > getRegPressure(N.getOperand(2))) {
        Select(N.getOperand(0));
        SelectAddress(N.getOperand(2), AM);
      } else {
        SelectAddress(N.getOperand(2), AM);
        Select(N.getOperand(0));
      }
      GlobalValue *GV = GA->getGlobal();
      // For Darwin, external and weak symbols are indirect, so we want to load
      // the value at address GV, not the value of GV itself.
      if (Subtarget->getIndirectExternAndWeakGlobals() &&
          (GV->hasWeakLinkage() || GV->isExternal())) {
        Tmp1 = MakeReg(MVT::i32);
        BuildMI(BB, X86::MOV32rm, 4, Tmp1).addReg(0).addZImm(1).addReg(0)
          .addGlobalAddress(GV, false, 0);
        addFullAddress(BuildMI(BB, X86::MOV32mr, 4+1),AM).addReg(Tmp1);
      } else {
        addFullAddress(BuildMI(BB, X86::MOV32mi, 4+1),AM).addGlobalAddress(GV);
      }
      return;
    }

    // Check to see if this is a load/op/store combination.
    if (TryToFoldLoadOpStore(Node))
      return;

    switch (N.getOperand(1).getValueType()) {
    default: assert(0 && "Cannot store this type!");
    case MVT::i1:
    case MVT::i8:  Opc = X86::MOV8mr; break;
    case MVT::i16: Opc = X86::MOV16mr; break;
    case MVT::i32: Opc = X86::MOV32mr; break;
    case MVT::f32: Opc = X86::MOVSSmr; break;
    case MVT::f64: Opc = X86ScalarSSE ? X86::MOVSDmr : X86::FST64m; break;
    }

    std::vector<std::pair<unsigned, unsigned> > RP;
    RP.push_back(std::make_pair(getRegPressure(N.getOperand(0)), 0));
    RP.push_back(std::make_pair(getRegPressure(N.getOperand(1)), 1));
    RP.push_back(std::make_pair(getRegPressure(N.getOperand(2)), 2));
    std::sort(RP.begin(), RP.end());

    Tmp1 = 0; // Silence a warning.
    for (unsigned i = 0; i != 3; ++i)
      switch (RP[2-i].second) {
      default: assert(0 && "Unknown operand number!");
      case 0: Select(N.getOperand(0)); break;
      case 1: Tmp1 = SelectExpr(N.getOperand(1)); break;
      case 2: SelectAddress(N.getOperand(2), AM); break;
      }

    addFullAddress(BuildMI(BB, Opc, 4+1), AM).addReg(Tmp1);
    return;
  }
  case ISD::CALLSEQ_START:
    Select(N.getOperand(0));
    // Stack amount
    Tmp1 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
    BuildMI(BB, X86::ADJCALLSTACKDOWN, 1).addImm(Tmp1);
    return;
  case ISD::CALLSEQ_END:
    Select(N.getOperand(0));
    return;
  case ISD::MEMSET: {
    Select(N.getOperand(0));  // Select the chain.
    unsigned Align =
      (unsigned)cast<ConstantSDNode>(Node->getOperand(4))->getValue();
    if (Align == 0) Align = 1;

    // Turn the byte code into # iterations
    unsigned CountReg;
    unsigned Opcode;
    if (ConstantSDNode *ValC = dyn_cast<ConstantSDNode>(Node->getOperand(2))) {
      unsigned Val = ValC->getValue() & 255;

      // If the value is a constant, then we can potentially use larger sets.
      switch (Align & 3) {
      case 2:   // WORD aligned
        CountReg = MakeReg(MVT::i32);
        if (ConstantSDNode *I = dyn_cast<ConstantSDNode>(Node->getOperand(3))) {
          BuildMI(BB, X86::MOV32ri, 1, CountReg).addImm(I->getValue()/2);
        } else {
          unsigned ByteReg = SelectExpr(Node->getOperand(3));
          BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(1);
        }
        BuildMI(BB, X86::MOV16ri, 1, X86::AX).addImm((Val << 8) | Val);
        Opcode = X86::REP_STOSW;
        break;
      case 0:   // DWORD aligned
        CountReg = MakeReg(MVT::i32);
        if (ConstantSDNode *I = dyn_cast<ConstantSDNode>(Node->getOperand(3))) {
          BuildMI(BB, X86::MOV32ri, 1, CountReg).addImm(I->getValue()/4);
        } else {
          unsigned ByteReg = SelectExpr(Node->getOperand(3));
          BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(2);
        }
        Val = (Val << 8) | Val;
        BuildMI(BB, X86::MOV32ri, 1, X86::EAX).addImm((Val << 16) | Val);
        Opcode = X86::REP_STOSD;
        break;
      default:  // BYTE aligned
        CountReg = SelectExpr(Node->getOperand(3));
        BuildMI(BB, X86::MOV8ri, 1, X86::AL).addImm(Val);
        Opcode = X86::REP_STOSB;
        break;
      }
    } else {
      // If it's not a constant value we are storing, just fall back.  We could
      // try to be clever to form 16 bit and 32 bit values, but we don't yet.
      unsigned ValReg = SelectExpr(Node->getOperand(2));
      BuildMI(BB, X86::MOV8rr, 1, X86::AL).addReg(ValReg);
      CountReg = SelectExpr(Node->getOperand(3));
      Opcode = X86::REP_STOSB;
    }

    // No matter what the alignment is, we put the source in ESI, the
    // destination in EDI, and the count in ECX.
    unsigned TmpReg1 = SelectExpr(Node->getOperand(1));
    BuildMI(BB, X86::MOV32rr, 1, X86::ECX).addReg(CountReg);
    BuildMI(BB, X86::MOV32rr, 1, X86::EDI).addReg(TmpReg1);
    BuildMI(BB, Opcode, 0);
    return;
  }
  case ISD::MEMCPY: {
    Select(N.getOperand(0));  // Select the chain.
    unsigned Align =
      (unsigned)cast<ConstantSDNode>(Node->getOperand(4))->getValue();
    if (Align == 0) Align = 1;

    // Turn the byte code into # iterations
    unsigned CountReg;
    unsigned Opcode;
    switch (Align & 3) {
    case 2:   // WORD aligned
      CountReg = MakeReg(MVT::i32);
      if (ConstantSDNode *I = dyn_cast<ConstantSDNode>(Node->getOperand(3))) {
        BuildMI(BB, X86::MOV32ri, 1, CountReg).addImm(I->getValue()/2);
      } else {
        unsigned ByteReg = SelectExpr(Node->getOperand(3));
        BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(1);
      }
      Opcode = X86::REP_MOVSW;
      break;
    case 0:   // DWORD aligned
      CountReg = MakeReg(MVT::i32);
      if (ConstantSDNode *I = dyn_cast<ConstantSDNode>(Node->getOperand(3))) {
        BuildMI(BB, X86::MOV32ri, 1, CountReg).addImm(I->getValue()/4);
      } else {
        unsigned ByteReg = SelectExpr(Node->getOperand(3));
        BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(2);
      }
      Opcode = X86::REP_MOVSD;
      break;
    default:  // BYTE aligned
      CountReg = SelectExpr(Node->getOperand(3));
      Opcode = X86::REP_MOVSB;
      break;
    }

    // No matter what the alignment is, we put the source in ESI, the
    // destination in EDI, and the count in ECX.
    unsigned TmpReg1 = SelectExpr(Node->getOperand(1));
    unsigned TmpReg2 = SelectExpr(Node->getOperand(2));
    BuildMI(BB, X86::MOV32rr, 1, X86::ECX).addReg(CountReg);
    BuildMI(BB, X86::MOV32rr, 1, X86::EDI).addReg(TmpReg1);
    BuildMI(BB, X86::MOV32rr, 1, X86::ESI).addReg(TmpReg2);
    BuildMI(BB, Opcode, 0);
    return;
  }
  case ISD::WRITEPORT:
    if (Node->getOperand(2).getValueType() != MVT::i16) {
      std::cerr << "llvm.writeport: Address size is not 16 bits\n";
      exit(1);
    }
    Select(Node->getOperand(0)); // Emit the chain.

    Tmp1 = SelectExpr(Node->getOperand(1));
    switch (Node->getOperand(1).getValueType()) {
    case MVT::i8:
      BuildMI(BB, X86::MOV8rr, 1, X86::AL).addReg(Tmp1);
      Tmp2 = X86::OUT8ir;  Opc = X86::OUT8rr;
      break;
    case MVT::i16:
      BuildMI(BB, X86::MOV16rr, 1, X86::AX).addReg(Tmp1);
      Tmp2 = X86::OUT16ir; Opc = X86::OUT16rr;
      break;
    case MVT::i32:
      BuildMI(BB, X86::MOV32rr, 1, X86::EAX).addReg(Tmp1);
      Tmp2 = X86::OUT32ir; Opc = X86::OUT32rr;
      break;
    default:
      std::cerr << "llvm.writeport: invalid data type for X86 target";
      exit(1);
    }

    // If the port is a single-byte constant, use the immediate form.
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Node->getOperand(2)))
      if ((CN->getValue() & 255) == CN->getValue()) {
        BuildMI(BB, Tmp2, 1).addImm(CN->getValue());
        return;
      }

    // Otherwise, move the I/O port address into the DX register.
    unsigned Reg = SelectExpr(Node->getOperand(2));
    BuildMI(BB, X86::MOV16rr, 1, X86::DX).addReg(Reg);
    BuildMI(BB, Opc, 0);
    return;
  }
  assert(0 && "Should not be reached!");
}


/// createX86PatternInstructionSelector - This pass converts an LLVM function
/// into a machine code representation using pattern matching and a machine
/// description file.
///
FunctionPass *llvm::createX86PatternInstructionSelector(TargetMachine &TM) {
  return new ISel(TM);
}
