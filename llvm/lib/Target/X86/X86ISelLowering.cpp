//===-- X86ISelLowering.h - X86 DAG Lowering Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that X86 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86ISelLowering.h"
#include "X86TargetMachine.h"
#include "llvm/CallingConv.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

// FIXME: temporary.
#include "llvm/Support/CommandLine.h"
static cl::opt<bool> EnableFastCC("enable-x86-fastcc", cl::Hidden,
                                  cl::desc("Enable fastcc on X86"));

X86TargetLowering::X86TargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  // Set up the TargetLowering object.

  // X86 is weird, it always uses i8 for shift amounts and setcc results.
  setShiftAmountType(MVT::i8);
  setSetCCResultType(MVT::i8);
  setSetCCResultContents(ZeroOrOneSetCCResult);
  setShiftAmountFlavor(Mask);   // shl X, 32 == shl X, 0

  // Set up the register classes.
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

  setOperationAction(ISD::BIT_CONVERT, MVT::f32, Expand);
  setOperationAction(ISD::BIT_CONVERT, MVT::i32, Expand);

  if (X86DAGIsel) {
    setOperationAction(ISD::BRCOND         , MVT::Other, Custom);
  }
  setOperationAction(ISD::BRCONDTWOWAY     , MVT::Other, Expand);
  setOperationAction(ISD::BRTWOWAY_CC      , MVT::Other, Expand);
  setOperationAction(ISD::MEMMOVE          , MVT::Other, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16  , Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8   , Expand);
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
  setOperationAction(ISD::READCYCLECOUNTER , MVT::i64  , Custom);

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
  // X86 wants to expand cmov itself.
  if (X86DAGIsel) {
    setOperationAction(ISD::SELECT         , MVT::i16  , Custom);
    setOperationAction(ISD::SELECT         , MVT::i32  , Custom);
    setOperationAction(ISD::SETCC          , MVT::i8   , Custom);
    setOperationAction(ISD::SETCC          , MVT::i16  , Custom);
    setOperationAction(ISD::SETCC          , MVT::i32  , Custom);
    setOperationAction(ISD::GlobalAddress  , MVT::i32  , Custom);
  }

  // We don't have line number support yet.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);

  if (X86ScalarSSE) {
    // Set up the FP register classes.
    addRegisterClass(MVT::f32, X86::V4F4RegisterClass);
    addRegisterClass(MVT::f64, X86::V2F8RegisterClass);

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

SDOperand X86TargetLowering::LowerReturnTo(SDOperand Chain, SDOperand Op,
                                           SelectionDAG &DAG) {
  if (!X86DAGIsel)
    return DAG.getNode(ISD::RET, MVT::Other, Chain, Op);

  SDOperand Copy;
  MVT::ValueType OpVT = Op.getValueType();
  switch (OpVT) {
    default: assert(0 && "Unknown type to return!");
    case MVT::i32:
      Copy = DAG.getCopyToReg(Chain, X86::EAX, Op, SDOperand());
      break;
    case MVT::i64: {
      SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op, 
                                 DAG.getConstant(1, MVT::i32));
      SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op,
                                 DAG.getConstant(0, MVT::i32));
      Copy = DAG.getCopyToReg(Chain, X86::EAX, Hi, SDOperand());
      Copy = DAG.getCopyToReg(Copy,  X86::EDX, Lo, Copy.getValue(1));
      break;
    }
    case MVT::f32:
      assert(X86ScalarSSE && "MVT::f32 only legal with scalar sse fp");
      // Fallthrough intended
    case MVT::f64:
      if (!X86ScalarSSE) {
        std::vector<MVT::ValueType> Tys;
        Tys.push_back(MVT::Other);
        Tys.push_back(MVT::Flag);
        std::vector<SDOperand> Ops;
        Ops.push_back(Chain);
        Ops.push_back(Op);
        Copy = DAG.getNode(X86ISD::FP_SET_RESULT, Tys, Ops);
      } else {
        // Spill the value to memory and reload it into top of stack.
        unsigned Size = MVT::getSizeInBits(OpVT)/8;
        MachineFunction &MF = DAG.getMachineFunction();
        int SSFI = MF.getFrameInfo()->CreateStackObject(Size, Size);
        SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
        Chain = DAG.getNode(ISD::STORE, MVT::Other, Chain, Op,
                            StackSlot, DAG.getSrcValue(NULL));
        std::vector<MVT::ValueType> Tys;
        Tys.push_back(MVT::f64);
        Tys.push_back(MVT::Other);
        std::vector<SDOperand> Ops;
        Ops.push_back(Chain);
        Ops.push_back(StackSlot);
        Ops.push_back(DAG.getValueType(OpVT));
        Copy = DAG.getNode(X86ISD::FLD, Tys, Ops);
        Tys.clear();
        Tys.push_back(MVT::Other);
        Tys.push_back(MVT::Flag);
        Ops.clear();
        Ops.push_back(Copy.getValue(1));
        Ops.push_back(Copy);
        Copy = DAG.getNode(X86ISD::FP_SET_RESULT, Tys, Ops);
      }
      break;
  }

  return DAG.getNode(X86ISD::RET_FLAG, MVT::Other,
                     Copy, DAG.getConstant(getBytesToPopOnReturn(), MVT::i16),
                     Copy.getValue(1));
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
  case ISD::READCYCLECOUNTER: {
    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::Other);
    Tys.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    Ops.push_back(Op.getOperand(0));
    SDOperand rd = DAG.getNode(X86ISD::RDTSC_DAG, Tys, Ops);
    Ops.clear();
    Ops.push_back(DAG.getCopyFromReg(rd, X86::EAX, MVT::i32, rd.getValue(1)));
    Ops.push_back(DAG.getCopyFromReg(Ops[0].getValue(1), X86::EDX, 
                                     MVT::i32, Ops[0].getValue(2)));
    Ops.push_back(Ops[1].getValue(1));
    Tys[0] = Tys[1] = MVT::i32;
    Tys.push_back(MVT::Other);
    return DAG.getNode(ISD::MERGE_VALUES, Tys, Ops);
  }
  case ISD::SETCC: {
    assert(Op.getValueType() == MVT::i8 && "SetCC type must be 8-bit integer");
    SDOperand CC   = Op.getOperand(2);
    SDOperand Cond = DAG.getNode(X86ISD::CMP, MVT::Flag,
                                 Op.getOperand(0), Op.getOperand(1));
    return DAG.getNode(X86ISD::SETCC, MVT::i8, CC, Cond);
  }
  case ISD::SELECT: {
    SDOperand Cond  = Op.getOperand(0);
    SDOperand CC;
    if (Cond.getOpcode() == X86ISD::SETCC) {
      CC = Cond.getOperand(0);
      Cond = Cond.getOperand(1);
    } else if (Cond.getOpcode() == ISD::SETCC) {
      CC = Cond.getOperand(2);
      Cond = DAG.getNode(X86ISD::CMP, MVT::Flag,
                         Cond.getOperand(0), Cond.getOperand(1));
    } else {
      CC = DAG.getCondCode(ISD::SETEQ);
      Cond = DAG.getNode(X86ISD::TEST, MVT::Flag, Cond, Cond);
    }
    return DAG.getNode(X86ISD::CMOV, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2), CC, Cond);
  }
  case ISD::BRCOND: {
    SDOperand Cond  = Op.getOperand(1);
    SDOperand Dest  = Op.getOperand(2);
    SDOperand CC;
    // TODO: handle Cond == OR / AND / XOR
    if (Cond.getOpcode() == X86ISD::SETCC) {
      CC = Cond.getOperand(0);
      Cond = Cond.getOperand(1);
    } else if (Cond.getOpcode() == ISD::SETCC) {
      CC = Cond.getOperand(2);
      Cond = DAG.getNode(X86ISD::CMP, MVT::Flag,
                         Cond.getOperand(0), Cond.getOperand(1));
    } else {
      CC = DAG.getCondCode(ISD::SETNE);
      Cond = DAG.getNode(X86ISD::TEST, MVT::Flag, Cond, Cond);
    }
    return DAG.getNode(X86ISD::BRCOND, Op.getValueType(),
                       Op.getOperand(0), Op.getOperand(2), CC, Cond);
  }
  case ISD::GlobalAddress:
    GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
    SDOperand GVOp = DAG.getTargetGlobalAddress(GV, getPointerTy());
    // For Darwin, external and weak symbols are indirect, so we want to load
    // the value at address GV, not the value of GV itself.  This means that
    // the GlobalAddress must be in the base or index register of the address,
    // not the GV offset field.
    if (getTargetMachine().
        getSubtarget<X86Subtarget>().getIndirectExternAndWeakGlobals() &&
        (GV->hasWeakLinkage() || GV->isExternal()))
      return DAG.getLoad(MVT::i32, DAG.getEntryNode(),
                         GVOp, DAG.getSrcValue(NULL));
    else
      return GVOp;
    break;
  }
}

const char *X86TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return NULL;
  case X86ISD::FILD64m:            return "X86ISD::FILD64m";
  case X86ISD::FP_TO_INT16_IN_MEM: return "X86ISD::FP_TO_INT16_IN_MEM";
  case X86ISD::FP_TO_INT32_IN_MEM: return "X86ISD::FP_TO_INT32_IN_MEM";
  case X86ISD::FP_TO_INT64_IN_MEM: return "X86ISD::FP_TO_INT64_IN_MEM";
  case X86ISD::FLD:                return "X86ISD::FLD";
  case X86ISD::FP_SET_RESULT:      return "X86ISD::FP_SET_RESULT";
  case X86ISD::CALL:               return "X86ISD::CALL";
  case X86ISD::TAILCALL:           return "X86ISD::TAILCALL";
  case X86ISD::RDTSC_DAG:          return "X86ISD::RDTSC_DAG";
  case X86ISD::CMP:                return "X86ISD::CMP";
  case X86ISD::TEST:               return "X86ISD::TEST";
  case X86ISD::SETCC:              return "X86ISD::SETCC";
  case X86ISD::CMOV:               return "X86ISD::CMOV";
  case X86ISD::BRCOND:             return "X86ISD::BRCOND";
  case X86ISD::RET_FLAG:           return "X86ISD::RET_FLAG";
  }
}

bool X86TargetLowering::isMaskedValueZeroForTargetNode(const SDOperand &Op,
                                                       uint64_t Mask) const {

  unsigned Opc = Op.getOpcode();

  switch (Opc) {
  default:
    assert(Opc >= ISD::BUILTIN_OP_END && "Expected a target specific node");
    break;
  case X86ISD::SETCC: return (Mask & 1) == 0;
  }

  return false;
}
