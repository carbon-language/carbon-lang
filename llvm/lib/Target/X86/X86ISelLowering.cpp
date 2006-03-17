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
#include "X86InstrBuilder.h"
#include "X86ISelLowering.h"
#include "X86TargetMachine.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

// FIXME: temporary.
#include "llvm/Support/CommandLine.h"
static cl::opt<bool> EnableFastCC("enable-x86-fastcc", cl::Hidden,
                                  cl::desc("Enable fastcc on X86"));

X86TargetLowering::X86TargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  Subtarget = &TM.getSubtarget<X86Subtarget>();
  X86ScalarSSE = Subtarget->hasSSE2();

  // Set up the TargetLowering object.

  // X86 is weird, it always uses i8 for shift amounts and setcc results.
  setShiftAmountType(MVT::i8);
  setSetCCResultType(MVT::i8);
  setSetCCResultContents(ZeroOrOneSetCCResult);
  setSchedulingPreference(SchedulingForRegPressure);
  setShiftAmountFlavor(Mask);   // shl X, 32 == shl X, 0
  setStackPointerRegisterToSaveRestore(X86::ESP);

  // Add legal addressing mode scale values.
  addLegalAddressScale(8);
  addLegalAddressScale(4);
  addLegalAddressScale(2);
  // Enter the ones which require both scale + index last. These are more
  // expensive.
  addLegalAddressScale(9);
  addLegalAddressScale(5);
  addLegalAddressScale(3);
  
  // Set up the register classes.
  addRegisterClass(MVT::i8, X86::R8RegisterClass);
  addRegisterClass(MVT::i16, X86::R16RegisterClass);
  addRegisterClass(MVT::i32, X86::R32RegisterClass);

  // Promote all UINT_TO_FP to larger SINT_TO_FP's, as X86 doesn't have this
  // operation.
  setOperationAction(ISD::UINT_TO_FP       , MVT::i1   , Promote);
  setOperationAction(ISD::UINT_TO_FP       , MVT::i8   , Promote);
  setOperationAction(ISD::UINT_TO_FP       , MVT::i16  , Promote);

  if (X86ScalarSSE)
    // No SSE i64 SINT_TO_FP, so expand i32 UINT_TO_FP instead.
    setOperationAction(ISD::UINT_TO_FP     , MVT::i32  , Expand);
  else
    setOperationAction(ISD::UINT_TO_FP     , MVT::i32  , Promote);

  // Promote i1/i8 SINT_TO_FP to larger SINT_TO_FP's, as X86 doesn't have
  // this operation.
  setOperationAction(ISD::SINT_TO_FP       , MVT::i1   , Promote);
  setOperationAction(ISD::SINT_TO_FP       , MVT::i8   , Promote);
  // SSE has no i16 to fp conversion, only i32
  if (X86ScalarSSE)
    setOperationAction(ISD::SINT_TO_FP     , MVT::i16  , Promote);
  else {
    setOperationAction(ISD::SINT_TO_FP     , MVT::i16  , Custom);
    setOperationAction(ISD::SINT_TO_FP     , MVT::i32  , Custom);
  }

  // We can handle SINT_TO_FP and FP_TO_SINT from/to i64 even though i64
  // isn't legal.
  setOperationAction(ISD::SINT_TO_FP       , MVT::i64  , Custom);
  setOperationAction(ISD::FP_TO_SINT       , MVT::i64  , Custom);

  // Promote i1/i8 FP_TO_SINT to larger FP_TO_SINTS's, as X86 doesn't have
  // this operation.
  setOperationAction(ISD::FP_TO_SINT       , MVT::i1   , Promote);
  setOperationAction(ISD::FP_TO_SINT       , MVT::i8   , Promote);

  if (X86ScalarSSE) {
    setOperationAction(ISD::FP_TO_SINT     , MVT::i16  , Promote);
  } else {
    setOperationAction(ISD::FP_TO_SINT     , MVT::i16  , Custom);
    setOperationAction(ISD::FP_TO_SINT     , MVT::i32  , Custom);
  }

  // Handle FP_TO_UINT by promoting the destination to a larger signed
  // conversion.
  setOperationAction(ISD::FP_TO_UINT       , MVT::i1   , Promote);
  setOperationAction(ISD::FP_TO_UINT       , MVT::i8   , Promote);
  setOperationAction(ISD::FP_TO_UINT       , MVT::i16  , Promote);

  if (X86ScalarSSE && !Subtarget->hasSSE3())
    // Expand FP_TO_UINT into a select.
    // FIXME: We would like to use a Custom expander here eventually to do
    // the optimal thing for SSE vs. the default expansion in the legalizer.
    setOperationAction(ISD::FP_TO_UINT     , MVT::i32  , Expand);
  else
    // With SSE3 we can use fisttpll to convert to a signed i64.
    setOperationAction(ISD::FP_TO_UINT     , MVT::i32  , Promote);

  setOperationAction(ISD::BIT_CONVERT      , MVT::f32  , Expand);
  setOperationAction(ISD::BIT_CONVERT      , MVT::i32  , Expand);

  setOperationAction(ISD::BRCOND           , MVT::Other, Custom);
  setOperationAction(ISD::BR_CC            , MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC        , MVT::Other, Expand);
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
  setOperationAction(ISD::BSWAP            , MVT::i16  , Expand);

  // These should be promoted to a larger select which is supported.
  setOperationAction(ISD::SELECT           , MVT::i1   , Promote);
  setOperationAction(ISD::SELECT           , MVT::i8   , Promote);

  // X86 wants to expand cmov itself.
  setOperationAction(ISD::SELECT          , MVT::i16  , Custom);
  setOperationAction(ISD::SELECT          , MVT::i32  , Custom);
  setOperationAction(ISD::SELECT          , MVT::f32  , Custom);
  setOperationAction(ISD::SELECT          , MVT::f64  , Custom);
  setOperationAction(ISD::SETCC           , MVT::i8   , Custom);
  setOperationAction(ISD::SETCC           , MVT::i16  , Custom);
  setOperationAction(ISD::SETCC           , MVT::i32  , Custom);
  setOperationAction(ISD::SETCC           , MVT::f32  , Custom);
  setOperationAction(ISD::SETCC           , MVT::f64  , Custom);
  // X86 ret instruction may pop stack.
  setOperationAction(ISD::RET             , MVT::Other, Custom);
  // Darwin ABI issue.
  setOperationAction(ISD::ConstantPool    , MVT::i32  , Custom);
  setOperationAction(ISD::GlobalAddress   , MVT::i32  , Custom);
  setOperationAction(ISD::ExternalSymbol  , MVT::i32  , Custom);
  // 64-bit addm sub, shl, sra, srl (iff 32-bit x86)
  setOperationAction(ISD::SHL_PARTS       , MVT::i32  , Custom);
  setOperationAction(ISD::SRA_PARTS       , MVT::i32  , Custom);
  setOperationAction(ISD::SRL_PARTS       , MVT::i32  , Custom);
  // X86 wants to expand memset / memcpy itself.
  setOperationAction(ISD::MEMSET          , MVT::Other, Custom);
  setOperationAction(ISD::MEMCPY          , MVT::Other, Custom);

  // We don't have line number support yet.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  // FIXME - use subtarget debug flags
  if (!TM.getSubtarget<X86Subtarget>().isTargetDarwin())
    setOperationAction(ISD::DEBUG_LABEL, MVT::Other, Expand);

  // VASTART needs to be custom lowered to use the VarArgsFrameIndex
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);
  
  // Use the default implementation.
  setOperationAction(ISD::VAARG             , MVT::Other, Expand);
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE,          MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE,       MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Expand);

  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);

  if (X86ScalarSSE) {
    // Set up the FP register classes.
    addRegisterClass(MVT::f32, X86::FR32RegisterClass);
    addRegisterClass(MVT::f64, X86::FR64RegisterClass);

    // SSE has no load+extend ops
    setOperationAction(ISD::EXTLOAD,  MVT::f32, Expand);
    setOperationAction(ISD::ZEXTLOAD, MVT::f32, Expand);

    // Use ANDPD to simulate FABS.
    setOperationAction(ISD::FABS , MVT::f64, Custom);
    setOperationAction(ISD::FABS , MVT::f32, Custom);

    // Use XORP to simulate FNEG.
    setOperationAction(ISD::FNEG , MVT::f64, Custom);
    setOperationAction(ISD::FNEG , MVT::f32, Custom);

    // We don't support sin/cos/fmod
    setOperationAction(ISD::FSIN , MVT::f64, Expand);
    setOperationAction(ISD::FCOS , MVT::f64, Expand);
    setOperationAction(ISD::FREM , MVT::f64, Expand);
    setOperationAction(ISD::FSIN , MVT::f32, Expand);
    setOperationAction(ISD::FCOS , MVT::f32, Expand);
    setOperationAction(ISD::FREM , MVT::f32, Expand);

    // Expand FP immediates into loads from the stack, except for the special
    // cases we handle.
    setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
    setOperationAction(ISD::ConstantFP, MVT::f32, Expand);
    addLegalFPImmediate(+0.0); // xorps / xorpd
  } else {
    // Set up the FP register classes.
    addRegisterClass(MVT::f64, X86::RFPRegisterClass);
    
    setOperationAction(ISD::UNDEF, MVT::f64, Expand);
    
    if (!UnsafeFPMath) {
      setOperationAction(ISD::FSIN           , MVT::f64  , Expand);
      setOperationAction(ISD::FCOS           , MVT::f64  , Expand);
    }

    setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
    addLegalFPImmediate(+0.0); // FLD0
    addLegalFPImmediate(+1.0); // FLD1
    addLegalFPImmediate(-0.0); // FLD0/FCHS
    addLegalFPImmediate(-1.0); // FLD1/FCHS
  }

  // First set operation action for all vector types to expand. Then we
  // will selectively turn on ones that can be effectively codegen'd.
  for (unsigned VT = (unsigned)MVT::Vector + 1;
       VT != (unsigned)MVT::LAST_VALUETYPE; VT++) {
    setOperationAction(ISD::ADD , (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::SUB , (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::MUL , (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::LOAD, (MVT::ValueType)VT, Expand);
  }

  if (TM.getSubtarget<X86Subtarget>().hasMMX()) {
    addRegisterClass(MVT::v8i8,  X86::VR64RegisterClass);
    addRegisterClass(MVT::v4i16, X86::VR64RegisterClass);
    addRegisterClass(MVT::v2i32, X86::VR64RegisterClass);

    // FIXME: add MMX packed arithmetics
    setOperationAction(ISD::ConstantVec, MVT::v8i8,  Expand);
    setOperationAction(ISD::ConstantVec, MVT::v4i16, Expand);
    setOperationAction(ISD::ConstantVec, MVT::v2i32, Expand);
  }

  if (TM.getSubtarget<X86Subtarget>().hasSSE1()) {
    addRegisterClass(MVT::v4f32, X86::VR128RegisterClass);

    setOperationAction(ISD::ADD        , MVT::v4f32, Legal);
    setOperationAction(ISD::SUB        , MVT::v4f32, Legal);
    setOperationAction(ISD::MUL        , MVT::v4f32, Legal);
    setOperationAction(ISD::LOAD       , MVT::v4f32, Legal);
    setOperationAction(ISD::ConstantVec, MVT::v4f32, Expand);
  }

  if (TM.getSubtarget<X86Subtarget>().hasSSE2()) {
    addRegisterClass(MVT::v2f64, X86::VR128RegisterClass);
    addRegisterClass(MVT::v16i8, X86::VR128RegisterClass);
    addRegisterClass(MVT::v8i16, X86::VR128RegisterClass);
    addRegisterClass(MVT::v4i32, X86::VR128RegisterClass);
    addRegisterClass(MVT::v2i64, X86::VR128RegisterClass);


    setOperationAction(ISD::ADD        , MVT::v2f64, Legal);
    setOperationAction(ISD::SUB        , MVT::v2f64, Legal);
    setOperationAction(ISD::MUL        , MVT::v2f64, Legal);
    setOperationAction(ISD::LOAD       , MVT::v2f64, Legal);
    setOperationAction(ISD::ConstantVec, MVT::v2f64, Expand);
    setOperationAction(ISD::ConstantVec, MVT::v16i8, Expand);
    setOperationAction(ISD::ConstantVec, MVT::v8i16, Expand);
    setOperationAction(ISD::ConstantVec, MVT::v4i32, Expand);
    setOperationAction(ISD::ConstantVec, MVT::v2i64, Expand);
  }

  computeRegisterProperties();

  // FIXME: These should be based on subtarget info. Plus, the values should
  // be smaller when we are in optimizing for size mode.
  maxStoresPerMemset = 16; // For %llvm.memset -> sequence of stores
  maxStoresPerMemcpy = 16; // For %llvm.memcpy -> sequence of stores
  maxStoresPerMemmove = 16; // For %llvm.memmove -> sequence of stores
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

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), getPointerTy());
  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy());

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
    Chain = DAG.getCALLSEQ_START(Chain, DAG.getConstant(0, getPointerTy()));
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

    Chain = DAG.getCALLSEQ_START(Chain,
                                 DAG.getConstant(NumBytes, getPointerTy()));

    // Arguments go on the stack in reverse order, as specified by the ABI.
    unsigned ArgOffset = 0;
    SDOperand StackPtr = DAG.getRegister(X86::ESP, MVT::i32);
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

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // FIXME: Do not generate X86ISD::TAILCALL for now.
  Chain = DAG.getNode(X86ISD::CALL, NodeTys, Ops);
  SDOperand InFlag = Chain.getValue(1);

  NodeTys.clear();
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  Ops.clear();
  Ops.push_back(Chain);
  Ops.push_back(DAG.getConstant(NumBytes, getPointerTy()));
  Ops.push_back(DAG.getConstant(0, getPointerTy()));
  Ops.push_back(InFlag);
  Chain = DAG.getNode(ISD::CALLSEQ_END, NodeTys, Ops);
  InFlag = Chain.getValue(1);
  
  SDOperand RetVal;
  if (RetTyVT != MVT::isVoid) {
    switch (RetTyVT) {
    default: assert(0 && "Unknown value type to return!");
    case MVT::i1:
    case MVT::i8:
      RetVal = DAG.getCopyFromReg(Chain, X86::AL, MVT::i8, InFlag);
      Chain = RetVal.getValue(1);
      if (RetTyVT == MVT::i1) 
        RetVal = DAG.getNode(ISD::TRUNCATE, MVT::i1, RetVal);
      break;
    case MVT::i16:
      RetVal = DAG.getCopyFromReg(Chain, X86::AX, MVT::i16, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::i32:
      RetVal = DAG.getCopyFromReg(Chain, X86::EAX, MVT::i32, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::i64: {
      SDOperand Lo = DAG.getCopyFromReg(Chain, X86::EAX, MVT::i32, InFlag);
      SDOperand Hi = DAG.getCopyFromReg(Lo.getValue(1), X86::EDX, MVT::i32, 
                                        Lo.getValue(2));
      RetVal = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Lo, Hi);
      Chain = Hi.getValue(1);
      break;
    }
    case MVT::f32:
    case MVT::f64: {
      std::vector<MVT::ValueType> Tys;
      Tys.push_back(MVT::f64);
      Tys.push_back(MVT::Other);
      Tys.push_back(MVT::Flag);
      std::vector<SDOperand> Ops;
      Ops.push_back(Chain);
      Ops.push_back(InFlag);
      RetVal = DAG.getNode(X86ISD::FP_GET_RESULT, Tys, Ops);
      Chain  = RetVal.getValue(1);
      InFlag = RetVal.getValue(2);
      if (X86ScalarSSE) {
        // FIXME: Currently the FST is flagged to the FP_GET_RESULT. This
        // shouldn't be necessary except that RFP cannot be live across
        // multiple blocks. When stackifier is fixed, they can be uncoupled.
        MachineFunction &MF = DAG.getMachineFunction();
        int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8);
        SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
        Tys.clear();
        Tys.push_back(MVT::Other);
        Ops.clear();
        Ops.push_back(Chain);
        Ops.push_back(RetVal);
        Ops.push_back(StackSlot);
        Ops.push_back(DAG.getValueType(RetTyVT));
        Ops.push_back(InFlag);
        Chain = DAG.getNode(X86ISD::FST, Tys, Ops);
        RetVal = DAG.getLoad(RetTyVT, Chain, StackSlot,
                             DAG.getSrcValue(NULL));
        Chain = RetVal.getValue(1);
      }

      if (RetTyVT == MVT::f32 && !X86ScalarSSE)
        // FIXME: we would really like to remember that this FP_ROUND
        // operation is okay to eliminate if we allow excess FP precision.
        RetVal = DAG.getNode(ISD::FP_ROUND, MVT::f32, RetVal);
      break;
    }
    }
  }

  return std::make_pair(RetVal, Chain);
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

// FASTCC_NUM_INT_ARGS_INREGS - This is the max number of integer arguments
// to pass in registers.  0 is none, 1 is is "use EAX", 2 is "use EAX and
// EDX".  Anything more is illegal.
//
// FIXME: The linscan register allocator currently has problem with
// coallescing.  At the time of this writing, whenever it decides to coallesce
// a physreg with a virtreg, this increases the size of the physreg's live
// range, and the live range cannot ever be reduced.  This causes problems if
// too many physregs are coalleced with virtregs, which can cause the register
// allocator to wedge itself.
//
// This code triggers this problem more often if we pass args in registers,
// so disable it until this is fixed.
//
// NOTE: this isn't marked const, so that GCC doesn't emit annoying warnings
// about code being dead.
//
static unsigned FASTCC_NUM_INT_ARGS_INREGS = 0;


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
      if (NumIntRegs < FASTCC_NUM_INT_ARGS_INREGS) {
        if (!I->use_empty()) {
          unsigned VReg = AddLiveIn(MF, NumIntRegs ? X86::DL : X86::AL,
                                    X86::R8RegisterClass);
          ArgValue = DAG.getCopyFromReg(DAG.getRoot(), VReg, MVT::i8);
          DAG.setRoot(ArgValue.getValue(1));
          if (ObjectVT == MVT::i1)
            // FIXME: Should insert a assertzext here.
            ArgValue = DAG.getNode(ISD::TRUNCATE, MVT::i1, ArgValue);
        }
        ++NumIntRegs;
        break;
      }

      ObjSize = 1;
      break;
    case MVT::i16:
      if (NumIntRegs < FASTCC_NUM_INT_ARGS_INREGS) {
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
      if (NumIntRegs < FASTCC_NUM_INT_ARGS_INREGS) {
        if (!I->use_empty()) {
          unsigned VReg = AddLiveIn(MF, NumIntRegs ? X86::EDX : X86::EAX,
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
      if (NumIntRegs+2 <= FASTCC_NUM_INT_ARGS_INREGS) {
        if (!I->use_empty()) {
          unsigned BotReg = AddLiveIn(MF, X86::EAX, X86::R32RegisterClass);
          unsigned TopReg = AddLiveIn(MF, X86::EDX, X86::R32RegisterClass);

          SDOperand Low = DAG.getCopyFromReg(DAG.getRoot(), BotReg, MVT::i32);
          SDOperand Hi  = DAG.getCopyFromReg(Low.getValue(1), TopReg, MVT::i32);
          DAG.setRoot(Hi.getValue(1));

          ArgValue = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Low, Hi);
        }
        NumIntRegs += 2;
        break;
      } else if (NumIntRegs+1 <= FASTCC_NUM_INT_ARGS_INREGS) {
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
        NumIntRegs = FASTCC_NUM_INT_ARGS_INREGS;
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
      if (NumIntRegs < FASTCC_NUM_INT_ARGS_INREGS) {
        ++NumIntRegs;
        break;
      }
      // fall through
    case MVT::f32:
      NumBytes += 4;
      break;
    case MVT::i64:
      if (NumIntRegs+2 <= FASTCC_NUM_INT_ARGS_INREGS) {
        NumIntRegs += 2;
        break;
      } else if (NumIntRegs+1 <= FASTCC_NUM_INT_ARGS_INREGS) {
        NumIntRegs = FASTCC_NUM_INT_ARGS_INREGS;
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

  Chain = DAG.getCALLSEQ_START(Chain,DAG.getConstant(NumBytes, getPointerTy()));

  // Arguments go on the stack in reverse order, as specified by the ABI.
  unsigned ArgOffset = 0;
  SDOperand StackPtr = DAG.getRegister(X86::ESP, MVT::i32);
  NumIntRegs = 0;
  std::vector<SDOperand> Stores;
  std::vector<SDOperand> RegValuesToPass;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    switch (getValueType(Args[i].second)) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i1:
      Args[i].first = DAG.getNode(ISD::ANY_EXTEND, MVT::i8, Args[i].first);
      // Fall through.
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      if (NumIntRegs < FASTCC_NUM_INT_ARGS_INREGS) {
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
       // Can pass (at least) part of it in regs?
      if (NumIntRegs < FASTCC_NUM_INT_ARGS_INREGS) {
        SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32,
                                   Args[i].first, DAG.getConstant(1, MVT::i32));
        SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32,
                                   Args[i].first, DAG.getConstant(0, MVT::i32));
        RegValuesToPass.push_back(Lo);
        ++NumIntRegs;
        
        // Pass both parts in regs?
        if (NumIntRegs < FASTCC_NUM_INT_ARGS_INREGS) {
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

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into registers.
  SDOperand InFlag;
  for (unsigned i = 0, e = RegValuesToPass.size(); i != e; ++i) {
    unsigned CCReg;
    SDOperand RegToPass = RegValuesToPass[i];
    switch (RegToPass.getValueType()) {
    default: assert(0 && "Bad thing to pass in regs");
    case MVT::i8:
      CCReg = (i == 0) ? X86::AL  : X86::DL;
      break;
    case MVT::i16:
      CCReg = (i == 0) ? X86::AX  : X86::DX;
      break;
    case MVT::i32:
      CCReg = (i == 0) ? X86::EAX : X86::EDX;
      break;
    }

    Chain = DAG.getCopyToReg(Chain, CCReg, RegToPass, InFlag);
    InFlag = Chain.getValue(1);
  }

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  if (InFlag.Val)
    Ops.push_back(InFlag);

  // FIXME: Do not generate X86ISD::TAILCALL for now.
  Chain = DAG.getNode(X86ISD::CALL, NodeTys, Ops);
  InFlag = Chain.getValue(1);

  NodeTys.clear();
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  Ops.clear();
  Ops.push_back(Chain);
  Ops.push_back(DAG.getConstant(ArgOffset, getPointerTy()));
  Ops.push_back(DAG.getConstant(ArgOffset, getPointerTy()));
  Ops.push_back(InFlag);
  Chain = DAG.getNode(ISD::CALLSEQ_END, NodeTys, Ops);
  InFlag = Chain.getValue(1);
  
  SDOperand RetVal;
  if (RetTyVT != MVT::isVoid) {
    switch (RetTyVT) {
    default: assert(0 && "Unknown value type to return!");
    case MVT::i1:
    case MVT::i8:
      RetVal = DAG.getCopyFromReg(Chain, X86::AL, MVT::i8, InFlag);
      Chain = RetVal.getValue(1);
      if (RetTyVT == MVT::i1) 
        RetVal = DAG.getNode(ISD::TRUNCATE, MVT::i1, RetVal);
      break;
    case MVT::i16:
      RetVal = DAG.getCopyFromReg(Chain, X86::AX, MVT::i16, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::i32:
      RetVal = DAG.getCopyFromReg(Chain, X86::EAX, MVT::i32, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::i64: {
      SDOperand Lo = DAG.getCopyFromReg(Chain, X86::EAX, MVT::i32, InFlag);
      SDOperand Hi = DAG.getCopyFromReg(Lo.getValue(1), X86::EDX, MVT::i32, 
                                        Lo.getValue(2));
      RetVal = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Lo, Hi);
      Chain = Hi.getValue(1);
      break;
    }
    case MVT::f32:
    case MVT::f64: {
      std::vector<MVT::ValueType> Tys;
      Tys.push_back(MVT::f64);
      Tys.push_back(MVT::Other);
      Tys.push_back(MVT::Flag);
      std::vector<SDOperand> Ops;
      Ops.push_back(Chain);
      Ops.push_back(InFlag);
      RetVal = DAG.getNode(X86ISD::FP_GET_RESULT, Tys, Ops);
      Chain  = RetVal.getValue(1);
      InFlag = RetVal.getValue(2);
      if (X86ScalarSSE) {
        // FIXME: Currently the FST is flagged to the FP_GET_RESULT. This
        // shouldn't be necessary except that RFP cannot be live across
        // multiple blocks. When stackifier is fixed, they can be uncoupled.
        MachineFunction &MF = DAG.getMachineFunction();
        int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8);
        SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
        Tys.clear();
        Tys.push_back(MVT::Other);
        Ops.clear();
        Ops.push_back(Chain);
        Ops.push_back(RetVal);
        Ops.push_back(StackSlot);
        Ops.push_back(DAG.getValueType(RetTyVT));
        Ops.push_back(InFlag);
        Chain = DAG.getNode(X86ISD::FST, Tys, Ops);
        RetVal = DAG.getLoad(RetTyVT, Chain, StackSlot,
                             DAG.getSrcValue(NULL));
        Chain = RetVal.getValue(1);
      }

      if (RetTyVT == MVT::f32 && !X86ScalarSSE)
        // FIXME: we would really like to remember that this FP_ROUND
        // operation is okay to eliminate if we allow excess FP precision.
        RetVal = DAG.getNode(ISD::FP_ROUND, MVT::f32, RetVal);
      break;
    }
    }
  }

  return std::make_pair(RetVal, Chain);
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

/// getCondBrOpcodeForX86CC - Returns the X86 conditional branch opcode
/// which corresponds to the condition code.
static unsigned getCondBrOpcodeForX86CC(unsigned X86CC) {
  switch (X86CC) {
  default: assert(0 && "Unknown X86 conditional code!");
  case X86ISD::COND_A:  return X86::JA;
  case X86ISD::COND_AE: return X86::JAE;
  case X86ISD::COND_B:  return X86::JB;
  case X86ISD::COND_BE: return X86::JBE;
  case X86ISD::COND_E:  return X86::JE;
  case X86ISD::COND_G:  return X86::JG;
  case X86ISD::COND_GE: return X86::JGE;
  case X86ISD::COND_L:  return X86::JL;
  case X86ISD::COND_LE: return X86::JLE;
  case X86ISD::COND_NE: return X86::JNE;
  case X86ISD::COND_NO: return X86::JNO;
  case X86ISD::COND_NP: return X86::JNP;
  case X86ISD::COND_NS: return X86::JNS;
  case X86ISD::COND_O:  return X86::JO;
  case X86ISD::COND_P:  return X86::JP;
  case X86ISD::COND_S:  return X86::JS;
  }
}

/// translateX86CC - do a one to one translation of a ISD::CondCode to the X86
/// specific condition code. It returns a false if it cannot do a direct
/// translation. X86CC is the translated CondCode. Flip is set to true if the
/// the order of comparison operands should be flipped.
static bool translateX86CC(SDOperand CC, bool isFP, unsigned &X86CC,
                           bool &Flip) {
  ISD::CondCode SetCCOpcode = cast<CondCodeSDNode>(CC)->get();
  Flip = false;
  X86CC = X86ISD::COND_INVALID;
  if (!isFP) {
    switch (SetCCOpcode) {
    default: break;
    case ISD::SETEQ:  X86CC = X86ISD::COND_E;  break;
    case ISD::SETGT:  X86CC = X86ISD::COND_G;  break;
    case ISD::SETGE:  X86CC = X86ISD::COND_GE; break;
    case ISD::SETLT:  X86CC = X86ISD::COND_L;  break;
    case ISD::SETLE:  X86CC = X86ISD::COND_LE; break;
    case ISD::SETNE:  X86CC = X86ISD::COND_NE; break;
    case ISD::SETULT: X86CC = X86ISD::COND_B;  break;
    case ISD::SETUGT: X86CC = X86ISD::COND_A;  break;
    case ISD::SETULE: X86CC = X86ISD::COND_BE; break;
    case ISD::SETUGE: X86CC = X86ISD::COND_AE; break;
    }
  } else {
    // On a floating point condition, the flags are set as follows:
    // ZF  PF  CF   op
    //  0 | 0 | 0 | X > Y
    //  0 | 0 | 1 | X < Y
    //  1 | 0 | 0 | X == Y
    //  1 | 1 | 1 | unordered
    switch (SetCCOpcode) {
    default: break;
    case ISD::SETUEQ:
    case ISD::SETEQ: X86CC = X86ISD::COND_E;  break;
    case ISD::SETOLE: Flip = true; // Fallthrough
    case ISD::SETOGT:
    case ISD::SETGT: X86CC = X86ISD::COND_A;  break;
    case ISD::SETOLT: Flip = true; // Fallthrough
    case ISD::SETOGE:
    case ISD::SETGE: X86CC = X86ISD::COND_AE; break;
    case ISD::SETUGE: Flip = true; // Fallthrough
    case ISD::SETULT:
    case ISD::SETLT: X86CC = X86ISD::COND_B;  break;
    case ISD::SETUGT: Flip = true; // Fallthrough
    case ISD::SETULE:
    case ISD::SETLE: X86CC = X86ISD::COND_BE; break;
    case ISD::SETONE:
    case ISD::SETNE: X86CC = X86ISD::COND_NE; break;
    case ISD::SETUO: X86CC = X86ISD::COND_P;  break;
    case ISD::SETO:  X86CC = X86ISD::COND_NP; break;
    }
  }

  return X86CC != X86ISD::COND_INVALID;
}

/// hasFPCMov - is there a floating point cmov for the specific X86 condition
/// code. Current x86 isa includes the following FP cmov instructions:
/// fcmovb, fcomvbe, fcomve, fcmovu, fcmovae, fcmova, fcmovne, fcmovnu.
static bool hasFPCMov(unsigned X86CC) {
  switch (X86CC) {
  default:
    return false;
  case X86ISD::COND_B:
  case X86ISD::COND_BE:
  case X86ISD::COND_E:
  case X86ISD::COND_P:
  case X86ISD::COND_A:
  case X86ISD::COND_AE:
  case X86ISD::COND_NE:
  case X86ISD::COND_NP:
    return true;
  }
}

MachineBasicBlock *
X86TargetLowering::InsertAtEndOfBasicBlock(MachineInstr *MI,
                                           MachineBasicBlock *BB) {
  switch (MI->getOpcode()) {
  default: assert(false && "Unexpected instr type to insert");
  case X86::CMOV_FR32:
  case X86::CMOV_FR64: {
    // To "insert" a SELECT_CC instruction, we actually have to insert the
    // diamond control-flow pattern.  The incoming instruction knows the
    // destination vreg to set, the condition code register to branch on, the
    // true/false values to select between, and a branch opcode to use.
    const BasicBlock *LLVM_BB = BB->getBasicBlock();
    ilist<MachineBasicBlock>::iterator It = BB;
    ++It;
  
    //  thisMBB:
    //  ...
    //   TrueVal = ...
    //   cmpTY ccX, r1, r2
    //   bCC copy1MBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *thisMBB = BB;
    MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
    unsigned Opc = getCondBrOpcodeForX86CC(MI->getOperand(3).getImmedValue());
    BuildMI(BB, Opc, 1).addMBB(sinkMBB);
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
    BuildMI(BB, X86::PHI, 4, MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg()).addMBB(copy0MBB)
      .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);

    delete MI;   // The pseudo instruction is gone now.
    return BB;
  }

  case X86::FP_TO_INT16_IN_MEM:
  case X86::FP_TO_INT32_IN_MEM:
  case X86::FP_TO_INT64_IN_MEM: {
    // Change the floating point control register to use "round towards zero"
    // mode when truncating to an integer value.
    MachineFunction *F = BB->getParent();
    int CWFrameIdx = F->getFrameInfo()->CreateStackObject(2, 2);
    addFrameReference(BuildMI(BB, X86::FNSTCW16m, 4), CWFrameIdx);

    // Load the old value of the high byte of the control word...
    unsigned OldCW =
      F->getSSARegMap()->createVirtualRegister(X86::R16RegisterClass);
    addFrameReference(BuildMI(BB, X86::MOV16rm, 4, OldCW), CWFrameIdx);

    // Set the high part to be round to zero...
    addFrameReference(BuildMI(BB, X86::MOV16mi, 5), CWFrameIdx).addImm(0xC7F);

    // Reload the modified control word now...
    addFrameReference(BuildMI(BB, X86::FLDCW16m, 4), CWFrameIdx);

    // Restore the memory image of control word to original value
    addFrameReference(BuildMI(BB, X86::MOV16mr, 5), CWFrameIdx).addReg(OldCW);

    // Get the X86 opcode to use.
    unsigned Opc;
    switch (MI->getOpcode()) {
    default: assert(0 && "illegal opcode!");
    case X86::FP_TO_INT16_IN_MEM: Opc = X86::FpIST16m; break;
    case X86::FP_TO_INT32_IN_MEM: Opc = X86::FpIST32m; break;
    case X86::FP_TO_INT64_IN_MEM: Opc = X86::FpIST64m; break;
    }

    X86AddressMode AM;
    MachineOperand &Op = MI->getOperand(0);
    if (Op.isRegister()) {
      AM.BaseType = X86AddressMode::RegBase;
      AM.Base.Reg = Op.getReg();
    } else {
      AM.BaseType = X86AddressMode::FrameIndexBase;
      AM.Base.FrameIndex = Op.getFrameIndex();
    }
    Op = MI->getOperand(1);
    if (Op.isImmediate())
      AM.Scale = Op.getImmedValue();
    Op = MI->getOperand(2);
    if (Op.isImmediate())
      AM.IndexReg = Op.getImmedValue();
    Op = MI->getOperand(3);
    if (Op.isGlobalAddress()) {
      AM.GV = Op.getGlobal();
    } else {
      AM.Disp = Op.getImmedValue();
    }
    addFullAddress(BuildMI(BB, Opc, 5), AM).addReg(MI->getOperand(4).getReg());

    // Reload the original control word now.
    addFrameReference(BuildMI(BB, X86::FLDCW16m, 4), CWFrameIdx);

    delete MI;   // The pseudo instruction is gone now.
    return BB;
  }
  }
}


//===----------------------------------------------------------------------===//
//                           X86 Custom Lowering Hooks
//===----------------------------------------------------------------------===//

/// DarwinGVRequiresExtraLoad - true if accessing the GV requires an extra
/// load. For Darwin, external and weak symbols are indirect, loading the value
/// at address GV rather then the value of GV itself. This means that the
/// GlobalAddress must be in the base or index register of the address, not the
/// GV offset field.
static bool DarwinGVRequiresExtraLoad(GlobalValue *GV) {
  return (GV->hasWeakLinkage() || GV->hasLinkOnceLinkage() ||
          (GV->isExternal() && !GV->hasNotBeenReadFromBytecode()));
}

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDOperand X86TargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Should not custom lower this!");
  case ISD::SHL_PARTS:
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS: {
    assert(Op.getNumOperands() == 3 && Op.getValueType() == MVT::i32 &&
           "Not an i64 shift!");
    bool isSRA = Op.getOpcode() == ISD::SRA_PARTS;
    SDOperand ShOpLo = Op.getOperand(0);
    SDOperand ShOpHi = Op.getOperand(1);
    SDOperand ShAmt  = Op.getOperand(2);
    SDOperand Tmp1 = isSRA ? DAG.getNode(ISD::SRA, MVT::i32, ShOpHi,
                                         DAG.getConstant(31, MVT::i8))
                           : DAG.getConstant(0, MVT::i32);

    SDOperand Tmp2, Tmp3;
    if (Op.getOpcode() == ISD::SHL_PARTS) {
      Tmp2 = DAG.getNode(X86ISD::SHLD, MVT::i32, ShOpHi, ShOpLo, ShAmt);
      Tmp3 = DAG.getNode(ISD::SHL, MVT::i32, ShOpLo, ShAmt);
    } else {
      Tmp2 = DAG.getNode(X86ISD::SHRD, MVT::i32, ShOpLo, ShOpHi, ShAmt);
      Tmp3 = DAG.getNode(isSRA ? ISD::SRA : ISD::SRL, MVT::i32, ShOpHi, ShAmt);
    }

    SDOperand InFlag = DAG.getNode(X86ISD::TEST, MVT::Flag,
                                   ShAmt, DAG.getConstant(32, MVT::i8));

    SDOperand Hi, Lo;
    SDOperand CC = DAG.getConstant(X86ISD::COND_NE, MVT::i8);

    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::i32);
    Tys.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    if (Op.getOpcode() == ISD::SHL_PARTS) {
      Ops.push_back(Tmp2);
      Ops.push_back(Tmp3);
      Ops.push_back(CC);
      Ops.push_back(InFlag);
      Hi = DAG.getNode(X86ISD::CMOV, Tys, Ops);
      InFlag = Hi.getValue(1);

      Ops.clear();
      Ops.push_back(Tmp3);
      Ops.push_back(Tmp1);
      Ops.push_back(CC);
      Ops.push_back(InFlag);
      Lo = DAG.getNode(X86ISD::CMOV, Tys, Ops);
    } else {
      Ops.push_back(Tmp2);
      Ops.push_back(Tmp3);
      Ops.push_back(CC);
      Ops.push_back(InFlag);
      Lo = DAG.getNode(X86ISD::CMOV, Tys, Ops);
      InFlag = Lo.getValue(1);

      Ops.clear();
      Ops.push_back(Tmp3);
      Ops.push_back(Tmp1);
      Ops.push_back(CC);
      Ops.push_back(InFlag);
      Hi = DAG.getNode(X86ISD::CMOV, Tys, Ops);
    }

    Tys.clear();
    Tys.push_back(MVT::i32);
    Tys.push_back(MVT::i32);
    Ops.clear();
    Ops.push_back(Lo);
    Ops.push_back(Hi);
    return DAG.getNode(ISD::MERGE_VALUES, Tys, Ops);
  }
  case ISD::SINT_TO_FP: {
    assert(Op.getOperand(0).getValueType() <= MVT::i64 &&
           Op.getOperand(0).getValueType() >= MVT::i16 &&
           "Unknown SINT_TO_FP to lower!");

    SDOperand Result;
    MVT::ValueType SrcVT = Op.getOperand(0).getValueType();
    unsigned Size = MVT::getSizeInBits(SrcVT)/8;
    MachineFunction &MF = DAG.getMachineFunction();
    int SSFI = MF.getFrameInfo()->CreateStackObject(Size, Size);
    SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
    SDOperand Chain = DAG.getNode(ISD::STORE, MVT::Other,
                                  DAG.getEntryNode(), Op.getOperand(0),
                                  StackSlot, DAG.getSrcValue(NULL));

    // Build the FILD
    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::f64);
    Tys.push_back(MVT::Other);
    if (X86ScalarSSE) Tys.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(StackSlot);
    Ops.push_back(DAG.getValueType(SrcVT));
    Result = DAG.getNode(X86ScalarSSE ? X86ISD::FILD_FLAG :X86ISD::FILD,
                         Tys, Ops);

    if (X86ScalarSSE) {
      Chain = Result.getValue(1);
      SDOperand InFlag = Result.getValue(2);

      // FIXME: Currently the FST is flagged to the FILD_FLAG. This
      // shouldn't be necessary except that RFP cannot be live across
      // multiple blocks. When stackifier is fixed, they can be uncoupled.
      MachineFunction &MF = DAG.getMachineFunction();
      int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8);
      SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
      std::vector<MVT::ValueType> Tys;
      Tys.push_back(MVT::Other);
      std::vector<SDOperand> Ops;
      Ops.push_back(Chain);
      Ops.push_back(Result);
      Ops.push_back(StackSlot);
      Ops.push_back(DAG.getValueType(Op.getValueType()));
      Ops.push_back(InFlag);
      Chain = DAG.getNode(X86ISD::FST, Tys, Ops);
      Result = DAG.getLoad(Op.getValueType(), Chain, StackSlot,
                           DAG.getSrcValue(NULL));
    }

    return Result;
  }
  case ISD::FP_TO_SINT: {
    assert(Op.getValueType() <= MVT::i64 && Op.getValueType() >= MVT::i16 &&
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

    SDOperand Chain = DAG.getEntryNode();
    SDOperand Value = Op.getOperand(0);
    if (X86ScalarSSE) {
      assert(Op.getValueType() == MVT::i64 && "Invalid FP_TO_SINT to lower!");
      Chain = DAG.getNode(ISD::STORE, MVT::Other, Chain, Value, StackSlot, 
                          DAG.getSrcValue(0));
      std::vector<MVT::ValueType> Tys;
      Tys.push_back(MVT::f64);
      Tys.push_back(MVT::Other);
      std::vector<SDOperand> Ops;
      Ops.push_back(Chain);
      Ops.push_back(StackSlot);
      Ops.push_back(DAG.getValueType(Op.getOperand(0).getValueType()));
      Value = DAG.getNode(X86ISD::FLD, Tys, Ops);
      Chain = Value.getValue(1);
      SSFI = MF.getFrameInfo()->CreateStackObject(MemSize, MemSize);
      StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
    }

    // Build the FP_TO_INT*_IN_MEM
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(Value);
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
  case ISD::FABS: {
    MVT::ValueType VT = Op.getValueType();
    const Type *OpNTy =  MVT::getTypeForValueType(VT);
    std::vector<Constant*> CV;
    if (VT == MVT::f64) {
      CV.push_back(ConstantFP::get(OpNTy, BitsToDouble(~(1ULL << 63))));
      CV.push_back(ConstantFP::get(OpNTy, 0.0));
    } else {
      CV.push_back(ConstantFP::get(OpNTy, BitsToFloat(~(1U << 31))));
      CV.push_back(ConstantFP::get(OpNTy, 0.0));
      CV.push_back(ConstantFP::get(OpNTy, 0.0));
      CV.push_back(ConstantFP::get(OpNTy, 0.0));
    }
    Constant *CS = ConstantStruct::get(CV);
    SDOperand CPIdx = DAG.getConstantPool(CS, getPointerTy(), 4);
    SDOperand Mask 
      = DAG.getNode(X86ISD::LOAD_PACK,
                    VT, DAG.getEntryNode(), CPIdx, DAG.getSrcValue(NULL));
    return DAG.getNode(X86ISD::FAND, VT, Op.getOperand(0), Mask);
  }
  case ISD::FNEG: {
    MVT::ValueType VT = Op.getValueType();
    const Type *OpNTy =  MVT::getTypeForValueType(VT);
    std::vector<Constant*> CV;
    if (VT == MVT::f64) {
      CV.push_back(ConstantFP::get(OpNTy, BitsToDouble(1ULL << 63)));
      CV.push_back(ConstantFP::get(OpNTy, 0.0));
    } else {
      CV.push_back(ConstantFP::get(OpNTy, BitsToFloat(1U << 31)));
      CV.push_back(ConstantFP::get(OpNTy, 0.0));
      CV.push_back(ConstantFP::get(OpNTy, 0.0));
      CV.push_back(ConstantFP::get(OpNTy, 0.0));
    }
    Constant *CS = ConstantStruct::get(CV);
    SDOperand CPIdx = DAG.getConstantPool(CS, getPointerTy(), 4);
    SDOperand Mask 
      = DAG.getNode(X86ISD::LOAD_PACK,
                    VT, DAG.getEntryNode(), CPIdx, DAG.getSrcValue(NULL));
    return DAG.getNode(X86ISD::FXOR, VT, Op.getOperand(0), Mask);
  }
  case ISD::SETCC: {
    assert(Op.getValueType() == MVT::i8 && "SetCC type must be 8-bit integer");
    SDOperand Cond;
    SDOperand CC = Op.getOperand(2);
    ISD::CondCode SetCCOpcode = cast<CondCodeSDNode>(CC)->get();
    bool isFP = MVT::isFloatingPoint(Op.getOperand(1).getValueType());
    bool Flip;
    unsigned X86CC;
    if (translateX86CC(CC, isFP, X86CC, Flip)) {
      if (Flip)
        Cond = DAG.getNode(X86ISD::CMP, MVT::Flag,
                           Op.getOperand(1), Op.getOperand(0));
      else
        Cond = DAG.getNode(X86ISD::CMP, MVT::Flag,
                           Op.getOperand(0), Op.getOperand(1));
      return DAG.getNode(X86ISD::SETCC, MVT::i8, 
                         DAG.getConstant(X86CC, MVT::i8), Cond);
    } else {
      assert(isFP && "Illegal integer SetCC!");

      Cond = DAG.getNode(X86ISD::CMP, MVT::Flag,
                         Op.getOperand(0), Op.getOperand(1));
      std::vector<MVT::ValueType> Tys;
      std::vector<SDOperand> Ops;
      switch (SetCCOpcode) {
      default: assert(false && "Illegal floating point SetCC!");
      case ISD::SETOEQ: {  // !PF & ZF
        Tys.push_back(MVT::i8);
        Tys.push_back(MVT::Flag);
        Ops.push_back(DAG.getConstant(X86ISD::COND_NP, MVT::i8));
        Ops.push_back(Cond);
        SDOperand Tmp1 = DAG.getNode(X86ISD::SETCC, Tys, Ops);
        SDOperand Tmp2 = DAG.getNode(X86ISD::SETCC, MVT::i8,
                                     DAG.getConstant(X86ISD::COND_E, MVT::i8),
                                     Tmp1.getValue(1));
        return DAG.getNode(ISD::AND, MVT::i8, Tmp1, Tmp2);
      }
      case ISD::SETUNE: {  // PF | !ZF
        Tys.push_back(MVT::i8);
        Tys.push_back(MVT::Flag);
        Ops.push_back(DAG.getConstant(X86ISD::COND_P, MVT::i8));
        Ops.push_back(Cond);
        SDOperand Tmp1 = DAG.getNode(X86ISD::SETCC, Tys, Ops);
        SDOperand Tmp2 = DAG.getNode(X86ISD::SETCC, MVT::i8,
                                     DAG.getConstant(X86ISD::COND_NE, MVT::i8),
                                     Tmp1.getValue(1));
        return DAG.getNode(ISD::OR, MVT::i8, Tmp1, Tmp2);
      }
      }
    }
  }
  case ISD::SELECT: {
    MVT::ValueType VT = Op.getValueType();
    bool isFP      = MVT::isFloatingPoint(VT);
    bool isFPStack = isFP && !X86ScalarSSE;
    bool isFPSSE   = isFP && X86ScalarSSE;
    bool addTest   = false;
    SDOperand Op0 = Op.getOperand(0);
    SDOperand Cond, CC;
    if (Op0.getOpcode() == ISD::SETCC)
      Op0 = LowerOperation(Op0, DAG);

    if (Op0.getOpcode() == X86ISD::SETCC) {
      // If condition flag is set by a X86ISD::CMP, then make a copy of it
      // (since flag operand cannot be shared). If the X86ISD::SETCC does not
      // have another use it will be eliminated.
      // If the X86ISD::SETCC has more than one use, then it's probably better
      // to use a test instead of duplicating the X86ISD::CMP (for register
      // pressure reason).
      if (Op0.getOperand(1).getOpcode() == X86ISD::CMP) {
        if (!Op0.hasOneUse()) {
          std::vector<MVT::ValueType> Tys;
          for (unsigned i = 0; i < Op0.Val->getNumValues(); ++i)
            Tys.push_back(Op0.Val->getValueType(i));
          std::vector<SDOperand> Ops;
          for (unsigned i = 0; i < Op0.getNumOperands(); ++i)
            Ops.push_back(Op0.getOperand(i));
          Op0 = DAG.getNode(X86ISD::SETCC, Tys, Ops);
        }

        CC   = Op0.getOperand(0);
        Cond = Op0.getOperand(1);
        // Make a copy as flag result cannot be used by more than one.
        Cond = DAG.getNode(X86ISD::CMP, MVT::Flag,
                           Cond.getOperand(0), Cond.getOperand(1));
        addTest =
          isFPStack && !hasFPCMov(cast<ConstantSDNode>(CC)->getSignExtended());
      } else
        addTest = true;
    } else
      addTest = true;

    if (addTest) {
      CC = DAG.getConstant(X86ISD::COND_NE, MVT::i8);
      Cond = DAG.getNode(X86ISD::TEST, MVT::Flag, Op0, Op0);
    }

    std::vector<MVT::ValueType> Tys;
    Tys.push_back(Op.getValueType());
    Tys.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    // X86ISD::CMOV means set the result (which is operand 1) to the RHS if
    // condition is true.
    Ops.push_back(Op.getOperand(2));
    Ops.push_back(Op.getOperand(1));
    Ops.push_back(CC);
    Ops.push_back(Cond);
    return DAG.getNode(X86ISD::CMOV, Tys, Ops);
  }
  case ISD::BRCOND: {
    bool addTest = false;
    SDOperand Cond  = Op.getOperand(1);
    SDOperand Dest  = Op.getOperand(2);
    SDOperand CC;
    if (Cond.getOpcode() == ISD::SETCC)
      Cond = LowerOperation(Cond, DAG);

    if (Cond.getOpcode() == X86ISD::SETCC) {
      // If condition flag is set by a X86ISD::CMP, then make a copy of it
      // (since flag operand cannot be shared). If the X86ISD::SETCC does not
      // have another use it will be eliminated.
      // If the X86ISD::SETCC has more than one use, then it's probably better
      // to use a test instead of duplicating the X86ISD::CMP (for register
      // pressure reason).
      if (Cond.getOperand(1).getOpcode() == X86ISD::CMP) {
        if (!Cond.hasOneUse()) {
          std::vector<MVT::ValueType> Tys;
          for (unsigned i = 0; i < Cond.Val->getNumValues(); ++i)
            Tys.push_back(Cond.Val->getValueType(i));
          std::vector<SDOperand> Ops;
          for (unsigned i = 0; i < Cond.getNumOperands(); ++i)
            Ops.push_back(Cond.getOperand(i));
          Cond = DAG.getNode(X86ISD::SETCC, Tys, Ops);
        }

        CC   = Cond.getOperand(0);
        Cond = Cond.getOperand(1);
        // Make a copy as flag result cannot be used by more than one.
        Cond = DAG.getNode(X86ISD::CMP, MVT::Flag,
                           Cond.getOperand(0), Cond.getOperand(1));
      } else
        addTest = true;
    } else
      addTest = true;

    if (addTest) {
      CC = DAG.getConstant(X86ISD::COND_NE, MVT::i8);
      Cond = DAG.getNode(X86ISD::TEST, MVT::Flag, Cond, Cond);
    }
    return DAG.getNode(X86ISD::BRCOND, Op.getValueType(),
                       Op.getOperand(0), Op.getOperand(2), CC, Cond);
  }
  case ISD::MEMSET: {
    SDOperand InFlag(0, 0);
    SDOperand Chain = Op.getOperand(0);
    unsigned Align =
      (unsigned)cast<ConstantSDNode>(Op.getOperand(4))->getValue();
    if (Align == 0) Align = 1;

    ConstantSDNode *I = dyn_cast<ConstantSDNode>(Op.getOperand(3));
    // If not DWORD aligned, call memset if size is less than the threshold.
    // It knows how to align to the right boundary first.
    if ((Align & 3) != 0 ||
        (I && I->getValue() < Subtarget->getMinRepStrSizeThreshold())) {
      MVT::ValueType IntPtr = getPointerTy();
      const Type *IntPtrTy = getTargetData().getIntPtrType();
      std::vector<std::pair<SDOperand, const Type*> > Args;
      Args.push_back(std::make_pair(Op.getOperand(1), IntPtrTy));
      // Extend the ubyte argument to be an int value for the call.
      SDOperand Val = DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, Op.getOperand(2));
      Args.push_back(std::make_pair(Val, IntPtrTy));
      Args.push_back(std::make_pair(Op.getOperand(3), IntPtrTy));
      std::pair<SDOperand,SDOperand> CallResult =
        LowerCallTo(Chain, Type::VoidTy, false, CallingConv::C, false,
                    DAG.getExternalSymbol("memset", IntPtr), Args, DAG);
      return CallResult.second;
    }

    MVT::ValueType AVT;
    SDOperand Count;
    ConstantSDNode *ValC = dyn_cast<ConstantSDNode>(Op.getOperand(2));
    unsigned BytesLeft = 0;
    bool TwoRepStos = false;
    if (ValC) {
      unsigned ValReg;
      unsigned Val = ValC->getValue() & 255;

      // If the value is a constant, then we can potentially use larger sets.
      switch (Align & 3) {
      case 2:   // WORD aligned
        AVT = MVT::i16;
        Count = DAG.getConstant(I->getValue() / 2, MVT::i32);
        BytesLeft = I->getValue() % 2;
        Val    = (Val << 8) | Val;
        ValReg = X86::AX;
        break;
      case 0:   // DWORD aligned
        AVT = MVT::i32;
        if (I) {
          Count = DAG.getConstant(I->getValue() / 4, MVT::i32);
          BytesLeft = I->getValue() % 4;
        } else {
          Count = DAG.getNode(ISD::SRL, MVT::i32, Op.getOperand(3),
                              DAG.getConstant(2, MVT::i8));
          TwoRepStos = true;
        }
        Val = (Val << 8)  | Val;
        Val = (Val << 16) | Val;
        ValReg = X86::EAX;
        break;
      default:  // Byte aligned
        AVT = MVT::i8;
        Count = Op.getOperand(3);
        ValReg = X86::AL;
        break;
      }

      Chain  = DAG.getCopyToReg(Chain, ValReg, DAG.getConstant(Val, AVT),
                                InFlag);
      InFlag = Chain.getValue(1);
    } else {
      AVT = MVT::i8;
      Count  = Op.getOperand(3);
      Chain  = DAG.getCopyToReg(Chain, X86::AL, Op.getOperand(2), InFlag);
      InFlag = Chain.getValue(1);
    }

    Chain  = DAG.getCopyToReg(Chain, X86::ECX, Count, InFlag);
    InFlag = Chain.getValue(1);
    Chain  = DAG.getCopyToReg(Chain, X86::EDI, Op.getOperand(1), InFlag);
    InFlag = Chain.getValue(1);

    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::Other);
    Tys.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(DAG.getValueType(AVT));
    Ops.push_back(InFlag);
    Chain  = DAG.getNode(X86ISD::REP_STOS, Tys, Ops);

    if (TwoRepStos) {
      InFlag = Chain.getValue(1);
      Count = Op.getOperand(3);
      MVT::ValueType CVT = Count.getValueType();
      SDOperand Left = DAG.getNode(ISD::AND, CVT, Count,
                                   DAG.getConstant(3, CVT));
      Chain  = DAG.getCopyToReg(Chain, X86::ECX, Left, InFlag);
      InFlag = Chain.getValue(1);
      Tys.clear();
      Tys.push_back(MVT::Other);
      Tys.push_back(MVT::Flag);
      Ops.clear();
      Ops.push_back(Chain);
      Ops.push_back(DAG.getValueType(MVT::i8));
      Ops.push_back(InFlag);
      Chain  = DAG.getNode(X86ISD::REP_STOS, Tys, Ops);
    } else if (BytesLeft) {
      // Issue stores for the last 1 - 3 bytes.
      SDOperand Value;
      unsigned Val = ValC->getValue() & 255;
      unsigned Offset = I->getValue() - BytesLeft;
      SDOperand DstAddr = Op.getOperand(1);
      MVT::ValueType AddrVT = DstAddr.getValueType();
      if (BytesLeft >= 2) {
        Value = DAG.getConstant((Val << 8) | Val, MVT::i16);
        Chain = DAG.getNode(ISD::STORE, MVT::Other, Chain, Value,
                            DAG.getNode(ISD::ADD, AddrVT, DstAddr,
                                        DAG.getConstant(Offset, AddrVT)),
                            DAG.getSrcValue(NULL));
        BytesLeft -= 2;
        Offset += 2;
      }

      if (BytesLeft == 1) {
        Value = DAG.getConstant(Val, MVT::i8);
        Chain = DAG.getNode(ISD::STORE, MVT::Other, Chain, Value,
                            DAG.getNode(ISD::ADD, AddrVT, DstAddr,
                                        DAG.getConstant(Offset, AddrVT)),
                            DAG.getSrcValue(NULL));
      }
    }

    return Chain;
  }
  case ISD::MEMCPY: {
    SDOperand Chain = Op.getOperand(0);
    unsigned Align =
      (unsigned)cast<ConstantSDNode>(Op.getOperand(4))->getValue();
    if (Align == 0) Align = 1;

    ConstantSDNode *I = dyn_cast<ConstantSDNode>(Op.getOperand(3));
    // If not DWORD aligned, call memcpy if size is less than the threshold.
    // It knows how to align to the right boundary first.
    if ((Align & 3) != 0 ||
        (I && I->getValue() < Subtarget->getMinRepStrSizeThreshold())) {
      MVT::ValueType IntPtr = getPointerTy();
      const Type *IntPtrTy = getTargetData().getIntPtrType();
      std::vector<std::pair<SDOperand, const Type*> > Args;
      Args.push_back(std::make_pair(Op.getOperand(1), IntPtrTy));
      Args.push_back(std::make_pair(Op.getOperand(2), IntPtrTy));
      Args.push_back(std::make_pair(Op.getOperand(3), IntPtrTy));
      std::pair<SDOperand,SDOperand> CallResult =
        LowerCallTo(Chain, Type::VoidTy, false, CallingConv::C, false,
                    DAG.getExternalSymbol("memcpy", IntPtr), Args, DAG);
      return CallResult.second;
    }

    MVT::ValueType AVT;
    SDOperand Count;
    unsigned BytesLeft = 0;
    bool TwoRepMovs = false;
    switch (Align & 3) {
    case 2:   // WORD aligned
      AVT = MVT::i16;
      Count = DAG.getConstant(I->getValue() / 2, MVT::i32);
      BytesLeft = I->getValue() % 2;
      break;
    case 0:   // DWORD aligned
      AVT = MVT::i32;
      if (I) {
        Count = DAG.getConstant(I->getValue() / 4, MVT::i32);
        BytesLeft = I->getValue() % 4;
      } else {
        Count = DAG.getNode(ISD::SRL, MVT::i32, Op.getOperand(3),
                            DAG.getConstant(2, MVT::i8));
        TwoRepMovs = true;
      }
      break;
    default:  // Byte aligned
      AVT = MVT::i8;
      Count = Op.getOperand(3);
      break;
    }

    SDOperand InFlag(0, 0);
    Chain  = DAG.getCopyToReg(Chain, X86::ECX, Count, InFlag);
    InFlag = Chain.getValue(1);
    Chain  = DAG.getCopyToReg(Chain, X86::EDI, Op.getOperand(1), InFlag);
    InFlag = Chain.getValue(1);
    Chain  = DAG.getCopyToReg(Chain, X86::ESI, Op.getOperand(2), InFlag);
    InFlag = Chain.getValue(1);

    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::Other);
    Tys.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(DAG.getValueType(AVT));
    Ops.push_back(InFlag);
    Chain = DAG.getNode(X86ISD::REP_MOVS, Tys, Ops);

    if (TwoRepMovs) {
      InFlag = Chain.getValue(1);
      Count = Op.getOperand(3);
      MVT::ValueType CVT = Count.getValueType();
      SDOperand Left = DAG.getNode(ISD::AND, CVT, Count,
                                   DAG.getConstant(3, CVT));
      Chain  = DAG.getCopyToReg(Chain, X86::ECX, Left, InFlag);
      InFlag = Chain.getValue(1);
      Tys.clear();
      Tys.push_back(MVT::Other);
      Tys.push_back(MVT::Flag);
      Ops.clear();
      Ops.push_back(Chain);
      Ops.push_back(DAG.getValueType(MVT::i8));
      Ops.push_back(InFlag);
      Chain = DAG.getNode(X86ISD::REP_MOVS, Tys, Ops);
    } else if (BytesLeft) {
      // Issue loads and stores for the last 1 - 3 bytes.
      unsigned Offset = I->getValue() - BytesLeft;
      SDOperand DstAddr = Op.getOperand(1);
      MVT::ValueType DstVT = DstAddr.getValueType();
      SDOperand SrcAddr = Op.getOperand(2);
      MVT::ValueType SrcVT = SrcAddr.getValueType();
      SDOperand Value;
      if (BytesLeft >= 2) {
        Value = DAG.getLoad(MVT::i16, Chain,
                            DAG.getNode(ISD::ADD, SrcVT, SrcAddr,
                                        DAG.getConstant(Offset, SrcVT)),
                            DAG.getSrcValue(NULL));
        Chain = Value.getValue(1);
        Chain = DAG.getNode(ISD::STORE, MVT::Other, Chain, Value,
                            DAG.getNode(ISD::ADD, DstVT, DstAddr,
                                        DAG.getConstant(Offset, DstVT)),
                            DAG.getSrcValue(NULL));
        BytesLeft -= 2;
        Offset += 2;
      }

      if (BytesLeft == 1) {
        Value = DAG.getLoad(MVT::i8, Chain,
                            DAG.getNode(ISD::ADD, SrcVT, SrcAddr,
                                        DAG.getConstant(Offset, SrcVT)),
                            DAG.getSrcValue(NULL));
        Chain = Value.getValue(1);
        Chain = DAG.getNode(ISD::STORE, MVT::Other, Chain, Value,
                            DAG.getNode(ISD::ADD, DstVT, DstAddr,
                                        DAG.getConstant(Offset, DstVT)),
                            DAG.getSrcValue(NULL));
      }
    }

    return Chain;
  }

  // ConstantPool, GlobalAddress, and ExternalSymbol are lowered as their
  // target countpart wrapped in the X86ISD::Wrapper node. Suppose N is
  // one of the above mentioned nodes. It has to be wrapped because otherwise
  // Select(N) returns N. So the raw TargetGlobalAddress nodes, etc. can only
  // be used to form addressing mode. These wrapped nodes will be selected
  // into MOV32ri.
  case ISD::ConstantPool: {
    ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
    SDOperand Result = DAG.getNode(X86ISD::Wrapper, getPointerTy(),
                         DAG.getTargetConstantPool(CP->get(), getPointerTy(),
                                                   CP->getAlignment()));
    if (getTargetMachine().getSubtarget<X86Subtarget>().isTargetDarwin()) {
      // With PIC, the address is actually $g + Offset.
      if (getTargetMachine().getRelocationModel() == Reloc::PIC)
        Result = DAG.getNode(ISD::ADD, getPointerTy(),
                DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()), Result);    
    }

    return Result;
  }
  case ISD::GlobalAddress: {
    GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
    SDOperand Result = DAG.getNode(X86ISD::Wrapper, getPointerTy(),
                         DAG.getTargetGlobalAddress(GV, getPointerTy()));
    if (getTargetMachine().
        getSubtarget<X86Subtarget>().isTargetDarwin()) {
      // With PIC, the address is actually $g + Offset.
      if (getTargetMachine().getRelocationModel() == Reloc::PIC)
        Result = DAG.getNode(ISD::ADD, getPointerTy(),
                    DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()), Result);

      // For Darwin, external and weak symbols are indirect, so we want to load
      // the value at address GV, not the value of GV itself. This means that
      // the GlobalAddress must be in the base or index register of the address,
      // not the GV offset field.
      if (getTargetMachine().getRelocationModel() != Reloc::Static &&
          DarwinGVRequiresExtraLoad(GV))
        Result = DAG.getLoad(MVT::i32, DAG.getEntryNode(),
                             Result, DAG.getSrcValue(NULL));
    }

    return Result;
  }
  case ISD::ExternalSymbol: {
    const char *Sym = cast<ExternalSymbolSDNode>(Op)->getSymbol();
    SDOperand Result = DAG.getNode(X86ISD::Wrapper, getPointerTy(),
                         DAG.getTargetExternalSymbol(Sym, getPointerTy()));
    if (getTargetMachine().
        getSubtarget<X86Subtarget>().isTargetDarwin()) {
      // With PIC, the address is actually $g + Offset.
      if (getTargetMachine().getRelocationModel() == Reloc::PIC)
        Result = DAG.getNode(ISD::ADD, getPointerTy(),
                    DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()), Result);
    }

    return Result;
  }
  case ISD::VASTART: {
    // vastart just stores the address of the VarArgsFrameIndex slot into the
    // memory location argument.
    // FIXME: Replace MVT::i32 with PointerTy
    SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i32);
    return DAG.getNode(ISD::STORE, MVT::Other, Op.getOperand(0), FR, 
                       Op.getOperand(1), Op.getOperand(2));
  }
  case ISD::RET: {
    SDOperand Copy;
    
    switch(Op.getNumOperands()) {
    default:
      assert(0 && "Do not know how to return this many arguments!");
      abort();
    case 1: 
      return DAG.getNode(X86ISD::RET_FLAG, MVT::Other, Op.getOperand(0),
                         DAG.getConstant(getBytesToPopOnReturn(), MVT::i16));
    case 2: {
      MVT::ValueType ArgVT = Op.getOperand(1).getValueType();
      if (MVT::isInteger(ArgVT))
        Copy = DAG.getCopyToReg(Op.getOperand(0), X86::EAX, Op.getOperand(1),
                                SDOperand());
      else if (!X86ScalarSSE) {
        std::vector<MVT::ValueType> Tys;
        Tys.push_back(MVT::Other);
        Tys.push_back(MVT::Flag);
        std::vector<SDOperand> Ops;
        Ops.push_back(Op.getOperand(0));
        Ops.push_back(Op.getOperand(1));
        Copy = DAG.getNode(X86ISD::FP_SET_RESULT, Tys, Ops);
      } else {
        SDOperand MemLoc;
        SDOperand Chain = Op.getOperand(0);
        SDOperand Value = Op.getOperand(1);

        if (Value.getOpcode() == ISD::LOAD &&
            (Chain == Value.getValue(1) || Chain == Value.getOperand(0))) {
          Chain  = Value.getOperand(0);
          MemLoc = Value.getOperand(1);
        } else {
          // Spill the value to memory and reload it into top of stack.
          unsigned Size = MVT::getSizeInBits(ArgVT)/8;
          MachineFunction &MF = DAG.getMachineFunction();
          int SSFI = MF.getFrameInfo()->CreateStackObject(Size, Size);
          MemLoc = DAG.getFrameIndex(SSFI, getPointerTy());
          Chain = DAG.getNode(ISD::STORE, MVT::Other, Op.getOperand(0), 
                              Value, MemLoc, DAG.getSrcValue(0));
        }
        std::vector<MVT::ValueType> Tys;
        Tys.push_back(MVT::f64);
        Tys.push_back(MVT::Other);
        std::vector<SDOperand> Ops;
        Ops.push_back(Chain);
        Ops.push_back(MemLoc);
        Ops.push_back(DAG.getValueType(ArgVT));
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
    case 3:
      Copy = DAG.getCopyToReg(Op.getOperand(0), X86::EDX, Op.getOperand(2), 
                              SDOperand());
      Copy = DAG.getCopyToReg(Copy, X86::EAX,Op.getOperand(1),Copy.getValue(1));
      break;
    }
    return DAG.getNode(X86ISD::RET_FLAG, MVT::Other,
                       Copy, DAG.getConstant(getBytesToPopOnReturn(), MVT::i16),
                       Copy.getValue(1));
  }
  }
}

const char *X86TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return NULL;
  case X86ISD::SHLD:               return "X86ISD::SHLD";
  case X86ISD::SHRD:               return "X86ISD::SHRD";
  case X86ISD::FAND:               return "X86ISD::FAND";
  case X86ISD::FXOR:               return "X86ISD::FXOR";
  case X86ISD::FILD:               return "X86ISD::FILD";
  case X86ISD::FILD_FLAG:          return "X86ISD::FILD_FLAG";
  case X86ISD::FP_TO_INT16_IN_MEM: return "X86ISD::FP_TO_INT16_IN_MEM";
  case X86ISD::FP_TO_INT32_IN_MEM: return "X86ISD::FP_TO_INT32_IN_MEM";
  case X86ISD::FP_TO_INT64_IN_MEM: return "X86ISD::FP_TO_INT64_IN_MEM";
  case X86ISD::FLD:                return "X86ISD::FLD";
  case X86ISD::FST:                return "X86ISD::FST";
  case X86ISD::FP_GET_RESULT:      return "X86ISD::FP_GET_RESULT";
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
  case X86ISD::REP_STOS:           return "X86ISD::REP_STOS";
  case X86ISD::REP_MOVS:           return "X86ISD::REP_MOVS";
  case X86ISD::LOAD_PACK:          return "X86ISD::LOAD_PACK";
  case X86ISD::GlobalBaseReg:      return "X86ISD::GlobalBaseReg";
  case X86ISD::Wrapper:            return "X86ISD::Wrapper";
  }
}

void X86TargetLowering::computeMaskedBitsForTargetNode(const SDOperand Op,
                                                       uint64_t Mask,
                                                       uint64_t &KnownZero, 
                                                       uint64_t &KnownOne,
                                                       unsigned Depth) const {

  unsigned Opc = Op.getOpcode();
  KnownZero = KnownOne = 0;   // Don't know anything.

  switch (Opc) {
  default:
    assert(Opc >= ISD::BUILTIN_OP_END && "Expected a target specific node");
    break;
  case X86ISD::SETCC: 
    KnownZero |= (MVT::getIntVTBitMask(Op.getValueType()) ^ 1ULL);
    break;
  }
}

std::vector<unsigned> X86TargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const {
  if (Constraint.size() == 1) {
    // FIXME: not handling fp-stack yet!
    // FIXME: not handling MMX registers yet ('y' constraint).
    switch (Constraint[0]) {      // GCC X86 Constraint Letters
    default: break;  // Unknown constriant letter
    case 'r':   // GENERAL_REGS
    case 'R':   // LEGACY_REGS
      return make_vector<unsigned>(X86::EAX, X86::EBX, X86::ECX, X86::EDX,
                                   X86::ESI, X86::EDI, X86::EBP, X86::ESP, 0);
    case 'l':   // INDEX_REGS
      return make_vector<unsigned>(X86::EAX, X86::EBX, X86::ECX, X86::EDX,
                                   X86::ESI, X86::EDI, X86::EBP, 0);
    case 'q':   // Q_REGS (GENERAL_REGS in 64-bit mode)
    case 'Q':   // Q_REGS
      return make_vector<unsigned>(X86::EAX, X86::EBX, X86::ECX, X86::EDX, 0);
    case 'x':   // SSE_REGS if SSE1 allowed
      if (Subtarget->hasSSE1())
        return make_vector<unsigned>(X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
                                     X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7,
                                     0);
      return std::vector<unsigned>();
    case 'Y':   // SSE_REGS if SSE2 allowed
      if (Subtarget->hasSSE2())
        return make_vector<unsigned>(X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
                                     X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7,
                                     0);
      return std::vector<unsigned>();
    }
  }
  
  return std::vector<unsigned>();
}

/// isLegalAddressImmediate - Return true if the integer value or
/// GlobalValue can be used as the offset of the target addressing mode.
bool X86TargetLowering::isLegalAddressImmediate(int64_t V) const {
  // X86 allows a sign-extended 32-bit immediate field.
  return (V > -(1LL << 32) && V < (1LL << 32)-1);
}

bool X86TargetLowering::isLegalAddressImmediate(GlobalValue *GV) const {
  if (getTargetMachine().
      getSubtarget<X86Subtarget>().isTargetDarwin()) {
    Reloc::Model RModel = getTargetMachine().getRelocationModel();
    if (RModel == Reloc::Static)
      return true;
    else if (RModel == Reloc::DynamicNoPIC)
      return !DarwinGVRequiresExtraLoad(GV);
    else
      return false;
  } else
    return true;
}
