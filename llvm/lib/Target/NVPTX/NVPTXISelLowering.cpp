//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that NVPTX uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "NVPTXISelLowering.h"
#include "NVPTX.h"
#include "NVPTXTargetMachine.h"
#include "NVPTXTargetObjectFile.h"
#include "NVPTXUtilities.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

#undef DEBUG_TYPE
#define DEBUG_TYPE "nvptx-lower"

using namespace llvm;

static unsigned int uniqueCallSite = 0;

static cl::opt<bool> sched4reg(
    "nvptx-sched4reg",
    cl::desc("NVPTX Specific: schedule for register pressue"), cl::init(false));

static bool IsPTXVectorType(MVT VT) {
  switch (VT.SimpleTy) {
  default:
    return false;
  case MVT::v2i1:
  case MVT::v4i1:
  case MVT::v2i8:
  case MVT::v4i8:
  case MVT::v2i16:
  case MVT::v4i16:
  case MVT::v2i32:
  case MVT::v4i32:
  case MVT::v2i64:
  case MVT::v2f32:
  case MVT::v4f32:
  case MVT::v2f64:
    return true;
  }
}

/// ComputePTXValueVTs - For the given Type \p Ty, returns the set of primitive
/// EVTs that compose it.  Unlike ComputeValueVTs, this will break apart vectors
/// into their primitive components.
/// NOTE: This is a band-aid for code that expects ComputeValueVTs to return the
/// same number of types as the Ins/Outs arrays in LowerFormalArguments,
/// LowerCall, and LowerReturn.
static void ComputePTXValueVTs(const TargetLowering &TLI, Type *Ty,
                               SmallVectorImpl<EVT> &ValueVTs,
                               SmallVectorImpl<uint64_t> *Offsets = 0,
                               uint64_t StartingOffset = 0) {
  SmallVector<EVT, 16> TempVTs;
  SmallVector<uint64_t, 16> TempOffsets;

  ComputeValueVTs(TLI, Ty, TempVTs, &TempOffsets, StartingOffset);
  for (unsigned i = 0, e = TempVTs.size(); i != e; ++i) {
    EVT VT = TempVTs[i];
    uint64_t Off = TempOffsets[i];
    if (VT.isVector())
      for (unsigned j = 0, je = VT.getVectorNumElements(); j != je; ++j) {
        ValueVTs.push_back(VT.getVectorElementType());
        if (Offsets)
          Offsets->push_back(Off+j*VT.getVectorElementType().getStoreSize());
      }
    else {
      ValueVTs.push_back(VT);
      if (Offsets)
        Offsets->push_back(Off);
    }
  }
}

// NVPTXTargetLowering Constructor.
NVPTXTargetLowering::NVPTXTargetLowering(NVPTXTargetMachine &TM)
    : TargetLowering(TM, new NVPTXTargetObjectFile()), nvTM(&TM),
      nvptxSubtarget(TM.getSubtarget<NVPTXSubtarget>()) {

  // always lower memset, memcpy, and memmove intrinsics to load/store
  // instructions, rather
  // then generating calls to memset, mempcy or memmove.
  MaxStoresPerMemset = (unsigned) 0xFFFFFFFF;
  MaxStoresPerMemcpy = (unsigned) 0xFFFFFFFF;
  MaxStoresPerMemmove = (unsigned) 0xFFFFFFFF;

  setBooleanContents(ZeroOrNegativeOneBooleanContent);

  // Jump is Expensive. Don't create extra control flow for 'and', 'or'
  // condition branches.
  setJumpIsExpensive(true);

  // By default, use the Source scheduling
  if (sched4reg)
    setSchedulingPreference(Sched::RegPressure);
  else
    setSchedulingPreference(Sched::Source);

  addRegisterClass(MVT::i1, &NVPTX::Int1RegsRegClass);
  addRegisterClass(MVT::i16, &NVPTX::Int16RegsRegClass);
  addRegisterClass(MVT::i32, &NVPTX::Int32RegsRegClass);
  addRegisterClass(MVT::i64, &NVPTX::Int64RegsRegClass);
  addRegisterClass(MVT::f32, &NVPTX::Float32RegsRegClass);
  addRegisterClass(MVT::f64, &NVPTX::Float64RegsRegClass);

  // Operations not directly supported by NVPTX.
  setOperationAction(ISD::SELECT_CC, MVT::Other, Expand);
  setOperationAction(ISD::BR_CC, MVT::f32, Expand);
  setOperationAction(ISD::BR_CC, MVT::f64, Expand);
  setOperationAction(ISD::BR_CC, MVT::i1, Expand);
  setOperationAction(ISD::BR_CC, MVT::i8, Expand);
  setOperationAction(ISD::BR_CC, MVT::i16, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Expand);
  setOperationAction(ISD::BR_CC, MVT::i64, Expand);
  // Some SIGN_EXTEND_INREG can be done using cvt instruction.
  // For others we will expand to a SHL/SRA pair.
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i64, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8 , Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  if (nvptxSubtarget.hasROT64()) {
    setOperationAction(ISD::ROTL, MVT::i64, Legal);
    setOperationAction(ISD::ROTR, MVT::i64, Legal);
  } else {
    setOperationAction(ISD::ROTL, MVT::i64, Expand);
    setOperationAction(ISD::ROTR, MVT::i64, Expand);
  }
  if (nvptxSubtarget.hasROT32()) {
    setOperationAction(ISD::ROTL, MVT::i32, Legal);
    setOperationAction(ISD::ROTR, MVT::i32, Legal);
  } else {
    setOperationAction(ISD::ROTL, MVT::i32, Expand);
    setOperationAction(ISD::ROTR, MVT::i32, Expand);
  }

  setOperationAction(ISD::ROTL, MVT::i16, Expand);
  setOperationAction(ISD::ROTR, MVT::i16, Expand);
  setOperationAction(ISD::ROTL, MVT::i8, Expand);
  setOperationAction(ISD::ROTR, MVT::i8, Expand);
  setOperationAction(ISD::BSWAP, MVT::i16, Expand);
  setOperationAction(ISD::BSWAP, MVT::i32, Expand);
  setOperationAction(ISD::BSWAP, MVT::i64, Expand);

  // Indirect branch is not supported.
  // This also disables Jump Table creation.
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);

  // We want to legalize constant related memmove and memcopy
  // intrinsics.
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);

  // Turn FP extload into load/fextend
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, Expand);
  // Turn FP truncstore into trunc + store.
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  // PTX does not support load / store predicate registers
  setOperationAction(ISD::LOAD, MVT::i1, Custom);
  setOperationAction(ISD::STORE, MVT::i1, Custom);

  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1, Promote);
  setTruncStoreAction(MVT::i64, MVT::i1, Expand);
  setTruncStoreAction(MVT::i32, MVT::i1, Expand);
  setTruncStoreAction(MVT::i16, MVT::i1, Expand);
  setTruncStoreAction(MVT::i8, MVT::i1, Expand);

  // This is legal in NVPTX
  setOperationAction(ISD::ConstantFP, MVT::f64, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);

  // TRAP can be lowered to PTX trap
  setOperationAction(ISD::TRAP, MVT::Other, Legal);

  setOperationAction(ISD::ADDC, MVT::i64, Expand);
  setOperationAction(ISD::ADDE, MVT::i64, Expand);

  // Register custom handling for vector loads/stores
  for (int i = MVT::FIRST_VECTOR_VALUETYPE; i <= MVT::LAST_VECTOR_VALUETYPE;
       ++i) {
    MVT VT = (MVT::SimpleValueType) i;
    if (IsPTXVectorType(VT)) {
      setOperationAction(ISD::LOAD, VT, Custom);
      setOperationAction(ISD::STORE, VT, Custom);
      setOperationAction(ISD::INTRINSIC_W_CHAIN, VT, Custom);
    }
  }

  // Custom handling for i8 intrinsics
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i8, Custom);

  setOperationAction(ISD::CTLZ, MVT::i16, Legal);
  setOperationAction(ISD::CTLZ, MVT::i32, Legal);
  setOperationAction(ISD::CTLZ, MVT::i64, Legal);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i16, Legal);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Legal);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i64, Legal);
  setOperationAction(ISD::CTTZ, MVT::i16, Expand);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ, MVT::i64, Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i16, Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i64, Expand);
  setOperationAction(ISD::CTPOP, MVT::i16, Legal);
  setOperationAction(ISD::CTPOP, MVT::i32, Legal);
  setOperationAction(ISD::CTPOP, MVT::i64, Legal);

  // Now deduce the information based on the above mentioned
  // actions
  computeRegisterProperties();
}

const char *NVPTXTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default:
    return 0;
  case NVPTXISD::CALL:
    return "NVPTXISD::CALL";
  case NVPTXISD::RET_FLAG:
    return "NVPTXISD::RET_FLAG";
  case NVPTXISD::Wrapper:
    return "NVPTXISD::Wrapper";
  case NVPTXISD::DeclareParam:
    return "NVPTXISD::DeclareParam";
  case NVPTXISD::DeclareScalarParam:
    return "NVPTXISD::DeclareScalarParam";
  case NVPTXISD::DeclareRet:
    return "NVPTXISD::DeclareRet";
  case NVPTXISD::DeclareRetParam:
    return "NVPTXISD::DeclareRetParam";
  case NVPTXISD::PrintCall:
    return "NVPTXISD::PrintCall";
  case NVPTXISD::LoadParam:
    return "NVPTXISD::LoadParam";
  case NVPTXISD::LoadParamV2:
    return "NVPTXISD::LoadParamV2";
  case NVPTXISD::LoadParamV4:
    return "NVPTXISD::LoadParamV4";
  case NVPTXISD::StoreParam:
    return "NVPTXISD::StoreParam";
  case NVPTXISD::StoreParamV2:
    return "NVPTXISD::StoreParamV2";
  case NVPTXISD::StoreParamV4:
    return "NVPTXISD::StoreParamV4";
  case NVPTXISD::StoreParamS32:
    return "NVPTXISD::StoreParamS32";
  case NVPTXISD::StoreParamU32:
    return "NVPTXISD::StoreParamU32";
  case NVPTXISD::CallArgBegin:
    return "NVPTXISD::CallArgBegin";
  case NVPTXISD::CallArg:
    return "NVPTXISD::CallArg";
  case NVPTXISD::LastCallArg:
    return "NVPTXISD::LastCallArg";
  case NVPTXISD::CallArgEnd:
    return "NVPTXISD::CallArgEnd";
  case NVPTXISD::CallVoid:
    return "NVPTXISD::CallVoid";
  case NVPTXISD::CallVal:
    return "NVPTXISD::CallVal";
  case NVPTXISD::CallSymbol:
    return "NVPTXISD::CallSymbol";
  case NVPTXISD::Prototype:
    return "NVPTXISD::Prototype";
  case NVPTXISD::MoveParam:
    return "NVPTXISD::MoveParam";
  case NVPTXISD::StoreRetval:
    return "NVPTXISD::StoreRetval";
  case NVPTXISD::StoreRetvalV2:
    return "NVPTXISD::StoreRetvalV2";
  case NVPTXISD::StoreRetvalV4:
    return "NVPTXISD::StoreRetvalV4";
  case NVPTXISD::PseudoUseParam:
    return "NVPTXISD::PseudoUseParam";
  case NVPTXISD::RETURN:
    return "NVPTXISD::RETURN";
  case NVPTXISD::CallSeqBegin:
    return "NVPTXISD::CallSeqBegin";
  case NVPTXISD::CallSeqEnd:
    return "NVPTXISD::CallSeqEnd";
  case NVPTXISD::LoadV2:
    return "NVPTXISD::LoadV2";
  case NVPTXISD::LoadV4:
    return "NVPTXISD::LoadV4";
  case NVPTXISD::LDGV2:
    return "NVPTXISD::LDGV2";
  case NVPTXISD::LDGV4:
    return "NVPTXISD::LDGV4";
  case NVPTXISD::LDUV2:
    return "NVPTXISD::LDUV2";
  case NVPTXISD::LDUV4:
    return "NVPTXISD::LDUV4";
  case NVPTXISD::StoreV2:
    return "NVPTXISD::StoreV2";
  case NVPTXISD::StoreV4:
    return "NVPTXISD::StoreV4";
  }
}

bool NVPTXTargetLowering::shouldSplitVectorElementType(EVT VT) const {
  return VT == MVT::i1;
}

SDValue
NVPTXTargetLowering::LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const {
  SDLoc dl(Op);
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  Op = DAG.getTargetGlobalAddress(GV, dl, getPointerTy());
  return DAG.getNode(NVPTXISD::Wrapper, dl, getPointerTy(), Op);
}

std::string
NVPTXTargetLowering::getPrototype(Type *retTy, const ArgListTy &Args,
                                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  unsigned retAlignment,
                                  const ImmutableCallSite *CS) const {

  bool isABI = (nvptxSubtarget.getSmVersion() >= 20);
  assert(isABI && "Non-ABI compilation is not supported");
  if (!isABI)
    return "";

  std::stringstream O;
  O << "prototype_" << uniqueCallSite << " : .callprototype ";

  if (retTy->getTypeID() == Type::VoidTyID) {
    O << "()";
  } else {
    O << "(";
    if (retTy->isPrimitiveType() || retTy->isIntegerTy()) {
      unsigned size = 0;
      if (const IntegerType *ITy = dyn_cast<IntegerType>(retTy)) {
        size = ITy->getBitWidth();
        if (size < 32)
          size = 32;
      } else {
        assert(retTy->isFloatingPointTy() &&
               "Floating point type expected here");
        size = retTy->getPrimitiveSizeInBits();
      }

      O << ".param .b" << size << " _";
    } else if (isa<PointerType>(retTy)) {
      O << ".param .b" << getPointerTy().getSizeInBits() << " _";
    } else {
      if ((retTy->getTypeID() == Type::StructTyID) || isa<VectorType>(retTy)) {
        SmallVector<EVT, 16> vtparts;
        ComputeValueVTs(*this, retTy, vtparts);
        unsigned totalsz = 0;
        for (unsigned i = 0, e = vtparts.size(); i != e; ++i) {
          unsigned elems = 1;
          EVT elemtype = vtparts[i];
          if (vtparts[i].isVector()) {
            elems = vtparts[i].getVectorNumElements();
            elemtype = vtparts[i].getVectorElementType();
          }
          // TODO: no need to loop
          for (unsigned j = 0, je = elems; j != je; ++j) {
            unsigned sz = elemtype.getSizeInBits();
            if (elemtype.isInteger() && (sz < 8))
              sz = 8;
            totalsz += sz / 8;
          }
        }
        O << ".param .align " << retAlignment << " .b8 _[" << totalsz << "]";
      } else {
        assert(false && "Unknown return type");
      }
    }
    O << ") ";
  }
  O << "_ (";

  bool first = true;
  MVT thePointerTy = getPointerTy();

  unsigned OIdx = 0;
  for (unsigned i = 0, e = Args.size(); i != e; ++i, ++OIdx) {
    Type *Ty = Args[i].Ty;
    if (!first) {
      O << ", ";
    }
    first = false;

    if (Outs[OIdx].Flags.isByVal() == false) {
      if (Ty->isAggregateType() || Ty->isVectorTy()) {
        unsigned align = 0;
        const CallInst *CallI = cast<CallInst>(CS->getInstruction());
        const DataLayout *TD = getDataLayout();
        // +1 because index 0 is reserved for return type alignment
        if (!llvm::getAlign(*CallI, i + 1, align))
          align = TD->getABITypeAlignment(Ty);
        unsigned sz = TD->getTypeAllocSize(Ty);
        O << ".param .align " << align << " .b8 ";
        O << "_";
        O << "[" << sz << "]";
        // update the index for Outs
        SmallVector<EVT, 16> vtparts;
        ComputeValueVTs(*this, Ty, vtparts);
        if (unsigned len = vtparts.size())
          OIdx += len - 1;
        continue;
      }
       // i8 types in IR will be i16 types in SDAG
      assert((getValueType(Ty) == Outs[OIdx].VT ||
             (getValueType(Ty) == MVT::i8 && Outs[OIdx].VT == MVT::i16)) &&
             "type mismatch between callee prototype and arguments");
      // scalar type
      unsigned sz = 0;
      if (isa<IntegerType>(Ty)) {
        sz = cast<IntegerType>(Ty)->getBitWidth();
        if (sz < 32)
          sz = 32;
      } else if (isa<PointerType>(Ty))
        sz = thePointerTy.getSizeInBits();
      else
        sz = Ty->getPrimitiveSizeInBits();
      O << ".param .b" << sz << " ";
      O << "_";
      continue;
    }
    const PointerType *PTy = dyn_cast<PointerType>(Ty);
    assert(PTy && "Param with byval attribute should be a pointer type");
    Type *ETy = PTy->getElementType();

    unsigned align = Outs[OIdx].Flags.getByValAlign();
    unsigned sz = getDataLayout()->getTypeAllocSize(ETy);
    O << ".param .align " << align << " .b8 ";
    O << "_";
    O << "[" << sz << "]";
  }
  O << ");";
  return O.str();
}

unsigned
NVPTXTargetLowering::getArgumentAlignment(SDValue Callee,
                                          const ImmutableCallSite *CS,
                                          Type *Ty,
                                          unsigned Idx) const {
  const DataLayout *TD = getDataLayout();
  unsigned align = 0;
  GlobalAddressSDNode *Func = dyn_cast<GlobalAddressSDNode>(Callee.getNode());

  if (Func) { // direct call
    assert(CS->getCalledFunction() &&
           "direct call cannot find callee");
    if (!llvm::getAlign(*(CS->getCalledFunction()), Idx, align))
      align = TD->getABITypeAlignment(Ty);
  }
  else { // indirect call
    const CallInst *CallI = dyn_cast<CallInst>(CS->getInstruction());
    if (!llvm::getAlign(*CallI, Idx, align))
      align = TD->getABITypeAlignment(Ty);
  }

  return align;
}

SDValue NVPTXTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                                       SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc dl = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &isTailCall = CLI.IsTailCall;
  ArgListTy &Args = CLI.Args;
  Type *retTy = CLI.RetTy;
  ImmutableCallSite *CS = CLI.CS;

  bool isABI = (nvptxSubtarget.getSmVersion() >= 20);
  assert(isABI && "Non-ABI compilation is not supported");
  if (!isABI)
    return Chain;
  const DataLayout *TD = getDataLayout();
  MachineFunction &MF = DAG.getMachineFunction();
  const Function *F = MF.getFunction();

  SDValue tempChain = Chain;
  Chain =
      DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(uniqueCallSite, true),
                           dl);
  SDValue InFlag = Chain.getValue(1);

  unsigned paramCount = 0;
  // Args.size() and Outs.size() need not match.
  // Outs.size() will be larger
  //   * if there is an aggregate argument with multiple fields (each field
  //     showing up separately in Outs)
  //   * if there is a vector argument with more than typical vector-length
  //     elements (generally if more than 4) where each vector element is
  //     individually present in Outs.
  // So a different index should be used for indexing into Outs/OutVals.
  // See similar issue in LowerFormalArguments.
  unsigned OIdx = 0;
  // Declare the .params or .reg need to pass values
  // to the function
  for (unsigned i = 0, e = Args.size(); i != e; ++i, ++OIdx) {
    EVT VT = Outs[OIdx].VT;
    Type *Ty = Args[i].Ty;

    if (Outs[OIdx].Flags.isByVal() == false) {
      if (Ty->isAggregateType()) {
        // aggregate
        SmallVector<EVT, 16> vtparts;
        ComputeValueVTs(*this, Ty, vtparts);

        unsigned align = getArgumentAlignment(Callee, CS, Ty, paramCount + 1);
        // declare .param .align <align> .b8 .param<n>[<size>];
        unsigned sz = TD->getTypeAllocSize(Ty);
        SDVTList DeclareParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
        SDValue DeclareParamOps[] = { Chain, DAG.getConstant(align, MVT::i32),
                                      DAG.getConstant(paramCount, MVT::i32),
                                      DAG.getConstant(sz, MVT::i32), InFlag };
        Chain = DAG.getNode(NVPTXISD::DeclareParam, dl, DeclareParamVTs,
                            DeclareParamOps, 5);
        InFlag = Chain.getValue(1);
        unsigned curOffset = 0;
        for (unsigned j = 0, je = vtparts.size(); j != je; ++j) {
          unsigned elems = 1;
          EVT elemtype = vtparts[j];
          if (vtparts[j].isVector()) {
            elems = vtparts[j].getVectorNumElements();
            elemtype = vtparts[j].getVectorElementType();
          }
          for (unsigned k = 0, ke = elems; k != ke; ++k) {
            unsigned sz = elemtype.getSizeInBits();
            if (elemtype.isInteger() && (sz < 8))
              sz = 8;
            SDValue StVal = OutVals[OIdx];
            if (elemtype.getSizeInBits() < 16) {
              StVal = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i16, StVal);
            }
            SDVTList CopyParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
            SDValue CopyParamOps[] = { Chain,
                                       DAG.getConstant(paramCount, MVT::i32),
                                       DAG.getConstant(curOffset, MVT::i32),
                                       StVal, InFlag };
            Chain = DAG.getMemIntrinsicNode(NVPTXISD::StoreParam, dl,
                                            CopyParamVTs, &CopyParamOps[0], 5,
                                            elemtype, MachinePointerInfo());
            InFlag = Chain.getValue(1);
            curOffset += sz / 8;
            ++OIdx;
          }
        }
        if (vtparts.size() > 0)
          --OIdx;
        ++paramCount;
        continue;
      }
      if (Ty->isVectorTy()) {
        EVT ObjectVT = getValueType(Ty);
        unsigned align = getArgumentAlignment(Callee, CS, Ty, paramCount + 1);
        // declare .param .align <align> .b8 .param<n>[<size>];
        unsigned sz = TD->getTypeAllocSize(Ty);
        SDVTList DeclareParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
        SDValue DeclareParamOps[] = { Chain, DAG.getConstant(align, MVT::i32),
                                      DAG.getConstant(paramCount, MVT::i32),
                                      DAG.getConstant(sz, MVT::i32), InFlag };
        Chain = DAG.getNode(NVPTXISD::DeclareParam, dl, DeclareParamVTs,
                            DeclareParamOps, 5);
        InFlag = Chain.getValue(1);
        unsigned NumElts = ObjectVT.getVectorNumElements();
        EVT EltVT = ObjectVT.getVectorElementType();
        EVT MemVT = EltVT;
        bool NeedExtend = false;
        if (EltVT.getSizeInBits() < 16) {
          NeedExtend = true;
          EltVT = MVT::i16;
        }

        // V1 store
        if (NumElts == 1) {
          SDValue Elt = OutVals[OIdx++];
          if (NeedExtend)
            Elt = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, Elt);

          SDVTList CopyParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
          SDValue CopyParamOps[] = { Chain,
                                     DAG.getConstant(paramCount, MVT::i32),
                                     DAG.getConstant(0, MVT::i32), Elt,
                                     InFlag };
          Chain = DAG.getMemIntrinsicNode(NVPTXISD::StoreParam, dl,
                                          CopyParamVTs, &CopyParamOps[0], 5,
                                          MemVT, MachinePointerInfo());
          InFlag = Chain.getValue(1);
        } else if (NumElts == 2) {
          SDValue Elt0 = OutVals[OIdx++];
          SDValue Elt1 = OutVals[OIdx++];
          if (NeedExtend) {
            Elt0 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, Elt0);
            Elt1 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, Elt1);
          }

          SDVTList CopyParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
          SDValue CopyParamOps[] = { Chain,
                                     DAG.getConstant(paramCount, MVT::i32),
                                     DAG.getConstant(0, MVT::i32), Elt0, Elt1,
                                     InFlag };
          Chain = DAG.getMemIntrinsicNode(NVPTXISD::StoreParamV2, dl,
                                          CopyParamVTs, &CopyParamOps[0], 6,
                                          MemVT, MachinePointerInfo());
          InFlag = Chain.getValue(1);
        } else {
          unsigned curOffset = 0;
          // V4 stores
          // We have at least 4 elements (<3 x Ty> expands to 4 elements) and
          // the
          // vector will be expanded to a power of 2 elements, so we know we can
          // always round up to the next multiple of 4 when creating the vector
          // stores.
          // e.g.  4 elem => 1 st.v4
          //       6 elem => 2 st.v4
          //       8 elem => 2 st.v4
          //      11 elem => 3 st.v4
          unsigned VecSize = 4;
          if (EltVT.getSizeInBits() == 64)
            VecSize = 2;

          // This is potentially only part of a vector, so assume all elements
          // are packed together.
          unsigned PerStoreOffset = MemVT.getStoreSizeInBits() / 8 * VecSize;

          for (unsigned i = 0; i < NumElts; i += VecSize) {
            // Get values
            SDValue StoreVal;
            SmallVector<SDValue, 8> Ops;
            Ops.push_back(Chain);
            Ops.push_back(DAG.getConstant(paramCount, MVT::i32));
            Ops.push_back(DAG.getConstant(curOffset, MVT::i32));

            unsigned Opc = NVPTXISD::StoreParamV2;

            StoreVal = OutVals[OIdx++];
            if (NeedExtend)
              StoreVal = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, StoreVal);
            Ops.push_back(StoreVal);

            if (i + 1 < NumElts) {
              StoreVal = OutVals[OIdx++];
              if (NeedExtend)
                StoreVal =
                    DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, StoreVal);
            } else {
              StoreVal = DAG.getUNDEF(EltVT);
            }
            Ops.push_back(StoreVal);

            if (VecSize == 4) {
              Opc = NVPTXISD::StoreParamV4;
              if (i + 2 < NumElts) {
                StoreVal = OutVals[OIdx++];
                if (NeedExtend)
                  StoreVal =
                      DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, StoreVal);
              } else {
                StoreVal = DAG.getUNDEF(EltVT);
              }
              Ops.push_back(StoreVal);

              if (i + 3 < NumElts) {
                StoreVal = OutVals[OIdx++];
                if (NeedExtend)
                  StoreVal =
                      DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, StoreVal);
              } else {
                StoreVal = DAG.getUNDEF(EltVT);
              }
              Ops.push_back(StoreVal);
            }

            Ops.push_back(InFlag);

            SDVTList CopyParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
            Chain = DAG.getMemIntrinsicNode(Opc, dl, CopyParamVTs, &Ops[0],
                                            Ops.size(), MemVT,
                                            MachinePointerInfo());
            InFlag = Chain.getValue(1);
            curOffset += PerStoreOffset;
          }
        }
        ++paramCount;
        --OIdx;
        continue;
      }
      // Plain scalar
      // for ABI,    declare .param .b<size> .param<n>;
      unsigned sz = VT.getSizeInBits();
      bool needExtend = false;
      if (VT.isInteger()) {
        if (sz < 16)
          needExtend = true;
        if (sz < 32)
          sz = 32;
      }
      SDVTList DeclareParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
      SDValue DeclareParamOps[] = { Chain,
                                    DAG.getConstant(paramCount, MVT::i32),
                                    DAG.getConstant(sz, MVT::i32),
                                    DAG.getConstant(0, MVT::i32), InFlag };
      Chain = DAG.getNode(NVPTXISD::DeclareScalarParam, dl, DeclareParamVTs,
                          DeclareParamOps, 5);
      InFlag = Chain.getValue(1);
      SDValue OutV = OutVals[OIdx];
      if (needExtend) {
        // zext/sext i1 to i16
        unsigned opc = ISD::ZERO_EXTEND;
        if (Outs[OIdx].Flags.isSExt())
          opc = ISD::SIGN_EXTEND;
        OutV = DAG.getNode(opc, dl, MVT::i16, OutV);
      }
      SDVTList CopyParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
      SDValue CopyParamOps[] = { Chain, DAG.getConstant(paramCount, MVT::i32),
                                 DAG.getConstant(0, MVT::i32), OutV, InFlag };

      unsigned opcode = NVPTXISD::StoreParam;
      if (Outs[OIdx].Flags.isZExt())
        opcode = NVPTXISD::StoreParamU32;
      else if (Outs[OIdx].Flags.isSExt())
        opcode = NVPTXISD::StoreParamS32;
      Chain = DAG.getMemIntrinsicNode(opcode, dl, CopyParamVTs, CopyParamOps, 5,
                                      VT, MachinePointerInfo());

      InFlag = Chain.getValue(1);
      ++paramCount;
      continue;
    }
    // struct or vector
    SmallVector<EVT, 16> vtparts;
    const PointerType *PTy = dyn_cast<PointerType>(Args[i].Ty);
    assert(PTy && "Type of a byval parameter should be pointer");
    ComputeValueVTs(*this, PTy->getElementType(), vtparts);

    // declare .param .align <align> .b8 .param<n>[<size>];
    unsigned sz = Outs[OIdx].Flags.getByValSize();
    SDVTList DeclareParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
    // The ByValAlign in the Outs[OIdx].Flags is alway set at this point,
    // so we don't need to worry about natural alignment or not.
    // See TargetLowering::LowerCallTo().
    SDValue DeclareParamOps[] = {
      Chain, DAG.getConstant(Outs[OIdx].Flags.getByValAlign(), MVT::i32),
      DAG.getConstant(paramCount, MVT::i32), DAG.getConstant(sz, MVT::i32),
      InFlag
    };
    Chain = DAG.getNode(NVPTXISD::DeclareParam, dl, DeclareParamVTs,
                        DeclareParamOps, 5);
    InFlag = Chain.getValue(1);
    unsigned curOffset = 0;
    for (unsigned j = 0, je = vtparts.size(); j != je; ++j) {
      unsigned elems = 1;
      EVT elemtype = vtparts[j];
      if (vtparts[j].isVector()) {
        elems = vtparts[j].getVectorNumElements();
        elemtype = vtparts[j].getVectorElementType();
      }
      for (unsigned k = 0, ke = elems; k != ke; ++k) {
        unsigned sz = elemtype.getSizeInBits();
        if (elemtype.isInteger() && (sz < 8))
          sz = 8;
        SDValue srcAddr =
            DAG.getNode(ISD::ADD, dl, getPointerTy(), OutVals[OIdx],
                        DAG.getConstant(curOffset, getPointerTy()));
        SDValue theVal = DAG.getLoad(elemtype, dl, tempChain, srcAddr,
                                     MachinePointerInfo(), false, false, false,
                                     0);
        if (elemtype.getSizeInBits() < 16) {
          theVal = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i16, theVal);
        }
        SDVTList CopyParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
        SDValue CopyParamOps[] = { Chain, DAG.getConstant(paramCount, MVT::i32),
                                   DAG.getConstant(curOffset, MVT::i32), theVal,
                                   InFlag };
        Chain = DAG.getMemIntrinsicNode(NVPTXISD::StoreParam, dl, CopyParamVTs,
                                        CopyParamOps, 5, elemtype,
                                        MachinePointerInfo());

        InFlag = Chain.getValue(1);
        curOffset += sz / 8;
      }
    }
    ++paramCount;
  }

  GlobalAddressSDNode *Func = dyn_cast<GlobalAddressSDNode>(Callee.getNode());
  unsigned retAlignment = 0;

  // Handle Result
  if (Ins.size() > 0) {
    SmallVector<EVT, 16> resvtparts;
    ComputeValueVTs(*this, retTy, resvtparts);

    // Declare
    //  .param .align 16 .b8 retval0[<size-in-bytes>], or
    //  .param .b<size-in-bits> retval0
    unsigned resultsz = TD->getTypeAllocSizeInBits(retTy);
    if (retTy->isPrimitiveType() || retTy->isIntegerTy() ||
        retTy->isPointerTy()) {
      // Scalar needs to be at least 32bit wide
      if (resultsz < 32)
        resultsz = 32;
      SDVTList DeclareRetVTs = DAG.getVTList(MVT::Other, MVT::Glue);
      SDValue DeclareRetOps[] = { Chain, DAG.getConstant(1, MVT::i32),
                                  DAG.getConstant(resultsz, MVT::i32),
                                  DAG.getConstant(0, MVT::i32), InFlag };
      Chain = DAG.getNode(NVPTXISD::DeclareRet, dl, DeclareRetVTs,
                          DeclareRetOps, 5);
      InFlag = Chain.getValue(1);
    } else {
      retAlignment = getArgumentAlignment(Callee, CS, retTy, 0);
      SDVTList DeclareRetVTs = DAG.getVTList(MVT::Other, MVT::Glue);
      SDValue DeclareRetOps[] = { Chain,
                                  DAG.getConstant(retAlignment, MVT::i32),
                                  DAG.getConstant(resultsz / 8, MVT::i32),
                                  DAG.getConstant(0, MVT::i32), InFlag };
      Chain = DAG.getNode(NVPTXISD::DeclareRetParam, dl, DeclareRetVTs,
                          DeclareRetOps, 5);
      InFlag = Chain.getValue(1);
    }
  }

  if (!Func) {
    // This is indirect function call case : PTX requires a prototype of the
    // form
    // proto_0 : .callprototype(.param .b32 _) _ (.param .b32 _);
    // to be emitted, and the label has to used as the last arg of call
    // instruction.
    // The prototype is embedded in a string and put as the operand for an
    // INLINEASM SDNode.
    SDVTList InlineAsmVTs = DAG.getVTList(MVT::Other, MVT::Glue);
    std::string proto_string =
        getPrototype(retTy, Args, Outs, retAlignment, CS);
    const char *asmstr = nvTM->getManagedStrPool()
        ->getManagedString(proto_string.c_str())->c_str();
    SDValue InlineAsmOps[] = {
      Chain, DAG.getTargetExternalSymbol(asmstr, getPointerTy()),
      DAG.getMDNode(0), DAG.getTargetConstant(0, MVT::i32), InFlag
    };
    Chain = DAG.getNode(ISD::INLINEASM, dl, InlineAsmVTs, InlineAsmOps, 5);
    InFlag = Chain.getValue(1);
  }
  // Op to just print "call"
  SDVTList PrintCallVTs = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue PrintCallOps[] = {
    Chain, DAG.getConstant((Ins.size() == 0) ? 0 : 1, MVT::i32), InFlag
  };
  Chain = DAG.getNode(Func ? (NVPTXISD::PrintCallUni) : (NVPTXISD::PrintCall),
                      dl, PrintCallVTs, PrintCallOps, 3);
  InFlag = Chain.getValue(1);

  // Ops to print out the function name
  SDVTList CallVoidVTs = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue CallVoidOps[] = { Chain, Callee, InFlag };
  Chain = DAG.getNode(NVPTXISD::CallVoid, dl, CallVoidVTs, CallVoidOps, 3);
  InFlag = Chain.getValue(1);

  // Ops to print out the param list
  SDVTList CallArgBeginVTs = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue CallArgBeginOps[] = { Chain, InFlag };
  Chain = DAG.getNode(NVPTXISD::CallArgBegin, dl, CallArgBeginVTs,
                      CallArgBeginOps, 2);
  InFlag = Chain.getValue(1);

  for (unsigned i = 0, e = paramCount; i != e; ++i) {
    unsigned opcode;
    if (i == (e - 1))
      opcode = NVPTXISD::LastCallArg;
    else
      opcode = NVPTXISD::CallArg;
    SDVTList CallArgVTs = DAG.getVTList(MVT::Other, MVT::Glue);
    SDValue CallArgOps[] = { Chain, DAG.getConstant(1, MVT::i32),
                             DAG.getConstant(i, MVT::i32), InFlag };
    Chain = DAG.getNode(opcode, dl, CallArgVTs, CallArgOps, 4);
    InFlag = Chain.getValue(1);
  }
  SDVTList CallArgEndVTs = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue CallArgEndOps[] = { Chain, DAG.getConstant(Func ? 1 : 0, MVT::i32),
                              InFlag };
  Chain =
      DAG.getNode(NVPTXISD::CallArgEnd, dl, CallArgEndVTs, CallArgEndOps, 3);
  InFlag = Chain.getValue(1);

  if (!Func) {
    SDVTList PrototypeVTs = DAG.getVTList(MVT::Other, MVT::Glue);
    SDValue PrototypeOps[] = { Chain, DAG.getConstant(uniqueCallSite, MVT::i32),
                               InFlag };
    Chain = DAG.getNode(NVPTXISD::Prototype, dl, PrototypeVTs, PrototypeOps, 3);
    InFlag = Chain.getValue(1);
  }

  // Generate loads from param memory/moves from registers for result
  if (Ins.size() > 0) {
    unsigned resoffset = 0;
    if (retTy && retTy->isVectorTy()) {
      EVT ObjectVT = getValueType(retTy);
      unsigned NumElts = ObjectVT.getVectorNumElements();
      EVT EltVT = ObjectVT.getVectorElementType();
      assert(nvTM->getTargetLowering()->getNumRegisters(F->getContext(),
                                                        ObjectVT) == NumElts &&
             "Vector was not scalarized");
      unsigned sz = EltVT.getSizeInBits();
      bool needTruncate = sz < 16 ? true : false;

      if (NumElts == 1) {
        // Just a simple load
        std::vector<EVT> LoadRetVTs;
        if (needTruncate) {
          // If loading i1 result, generate
          //   load i16
          //   trunc i16 to i1
          LoadRetVTs.push_back(MVT::i16);
        } else
          LoadRetVTs.push_back(EltVT);
        LoadRetVTs.push_back(MVT::Other);
        LoadRetVTs.push_back(MVT::Glue);
        std::vector<SDValue> LoadRetOps;
        LoadRetOps.push_back(Chain);
        LoadRetOps.push_back(DAG.getConstant(1, MVT::i32));
        LoadRetOps.push_back(DAG.getConstant(0, MVT::i32));
        LoadRetOps.push_back(InFlag);
        SDValue retval = DAG.getMemIntrinsicNode(
            NVPTXISD::LoadParam, dl,
            DAG.getVTList(&LoadRetVTs[0], LoadRetVTs.size()), &LoadRetOps[0],
            LoadRetOps.size(), EltVT, MachinePointerInfo());
        Chain = retval.getValue(1);
        InFlag = retval.getValue(2);
        SDValue Ret0 = retval;
        if (needTruncate)
          Ret0 = DAG.getNode(ISD::TRUNCATE, dl, EltVT, Ret0);
        InVals.push_back(Ret0);
      } else if (NumElts == 2) {
        // LoadV2
        std::vector<EVT> LoadRetVTs;
        if (needTruncate) {
          // If loading i1 result, generate
          //   load i16
          //   trunc i16 to i1
          LoadRetVTs.push_back(MVT::i16);
          LoadRetVTs.push_back(MVT::i16);
        } else {
          LoadRetVTs.push_back(EltVT);
          LoadRetVTs.push_back(EltVT);
        }
        LoadRetVTs.push_back(MVT::Other);
        LoadRetVTs.push_back(MVT::Glue);
        std::vector<SDValue> LoadRetOps;
        LoadRetOps.push_back(Chain);
        LoadRetOps.push_back(DAG.getConstant(1, MVT::i32));
        LoadRetOps.push_back(DAG.getConstant(0, MVT::i32));
        LoadRetOps.push_back(InFlag);
        SDValue retval = DAG.getMemIntrinsicNode(
            NVPTXISD::LoadParamV2, dl,
            DAG.getVTList(&LoadRetVTs[0], LoadRetVTs.size()), &LoadRetOps[0],
            LoadRetOps.size(), EltVT, MachinePointerInfo());
        Chain = retval.getValue(2);
        InFlag = retval.getValue(3);
        SDValue Ret0 = retval.getValue(0);
        SDValue Ret1 = retval.getValue(1);
        if (needTruncate) {
          Ret0 = DAG.getNode(ISD::TRUNCATE, dl, MVT::i1, Ret0);
          InVals.push_back(Ret0);
          Ret1 = DAG.getNode(ISD::TRUNCATE, dl, MVT::i1, Ret1);
          InVals.push_back(Ret1);
        } else {
          InVals.push_back(Ret0);
          InVals.push_back(Ret1);
        }
      } else {
        // Split into N LoadV4
        unsigned Ofst = 0;
        unsigned VecSize = 4;
        unsigned Opc = NVPTXISD::LoadParamV4;
        if (EltVT.getSizeInBits() == 64) {
          VecSize = 2;
          Opc = NVPTXISD::LoadParamV2;
        }
        EVT VecVT = EVT::getVectorVT(F->getContext(), EltVT, VecSize);
        for (unsigned i = 0; i < NumElts; i += VecSize) {
          SmallVector<EVT, 8> LoadRetVTs;
          if (needTruncate) {
            // If loading i1 result, generate
            //   load i16
            //   trunc i16 to i1
            for (unsigned j = 0; j < VecSize; ++j)
              LoadRetVTs.push_back(MVT::i16);
          } else {
            for (unsigned j = 0; j < VecSize; ++j)
              LoadRetVTs.push_back(EltVT);
          }
          LoadRetVTs.push_back(MVT::Other);
          LoadRetVTs.push_back(MVT::Glue);
          SmallVector<SDValue, 4> LoadRetOps;
          LoadRetOps.push_back(Chain);
          LoadRetOps.push_back(DAG.getConstant(1, MVT::i32));
          LoadRetOps.push_back(DAG.getConstant(Ofst, MVT::i32));
          LoadRetOps.push_back(InFlag);
          SDValue retval = DAG.getMemIntrinsicNode(
              Opc, dl, DAG.getVTList(&LoadRetVTs[0], LoadRetVTs.size()),
              &LoadRetOps[0], LoadRetOps.size(), EltVT, MachinePointerInfo());
          if (VecSize == 2) {
            Chain = retval.getValue(2);
            InFlag = retval.getValue(3);
          } else {
            Chain = retval.getValue(4);
            InFlag = retval.getValue(5);
          }

          for (unsigned j = 0; j < VecSize; ++j) {
            if (i + j >= NumElts)
              break;
            SDValue Elt = retval.getValue(j);
            if (needTruncate)
              Elt = DAG.getNode(ISD::TRUNCATE, dl, EltVT, Elt);
            InVals.push_back(Elt);
          }
          Ofst += TD->getTypeAllocSize(VecVT.getTypeForEVT(F->getContext()));
        }
      }
    } else {
      SmallVector<EVT, 16> VTs;
      ComputePTXValueVTs(*this, retTy, VTs);
      assert(VTs.size() == Ins.size() && "Bad value decomposition");
      for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
        unsigned sz = VTs[i].getSizeInBits();
        bool needTruncate = sz < 8 ? true : false;
        if (VTs[i].isInteger() && (sz < 8))
          sz = 8;

        SmallVector<EVT, 4> LoadRetVTs;
        EVT TheLoadType = VTs[i];
        if (retTy->isIntegerTy() &&
            TD->getTypeAllocSizeInBits(retTy) < 32) {
          // This is for integer types only, and specifically not for
          // aggregates.
          LoadRetVTs.push_back(MVT::i32);
          TheLoadType = MVT::i32;
        } else if (sz < 16) {
          // If loading i1/i8 result, generate
          //   load i8 (-> i16)
          //   trunc i16 to i1/i8
          LoadRetVTs.push_back(MVT::i16);
        } else
          LoadRetVTs.push_back(Ins[i].VT);
        LoadRetVTs.push_back(MVT::Other);
        LoadRetVTs.push_back(MVT::Glue);

        SmallVector<SDValue, 4> LoadRetOps;
        LoadRetOps.push_back(Chain);
        LoadRetOps.push_back(DAG.getConstant(1, MVT::i32));
        LoadRetOps.push_back(DAG.getConstant(resoffset, MVT::i32));
        LoadRetOps.push_back(InFlag);
        SDValue retval = DAG.getMemIntrinsicNode(
            NVPTXISD::LoadParam, dl,
            DAG.getVTList(&LoadRetVTs[0], LoadRetVTs.size()), &LoadRetOps[0],
            LoadRetOps.size(), TheLoadType, MachinePointerInfo());
        Chain = retval.getValue(1);
        InFlag = retval.getValue(2);
        SDValue Ret0 = retval.getValue(0);
        if (needTruncate)
          Ret0 = DAG.getNode(ISD::TRUNCATE, dl, Ins[i].VT, Ret0);
        InVals.push_back(Ret0);
        resoffset += sz / 8;
      }
    }
  }

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(uniqueCallSite, true),
                             DAG.getIntPtrConstant(uniqueCallSite + 1, true),
                             InFlag, dl);
  uniqueCallSite++;

  // set isTailCall to false for now, until we figure out how to express
  // tail call optimization in PTX
  isTailCall = false;
  return Chain;
}

// By default CONCAT_VECTORS is lowered by ExpandVectorBuildThroughStack()
// (see LegalizeDAG.cpp). This is slow and uses local memory.
// We use extract/insert/build vector just as what LegalizeOp() does in llvm 2.5
SDValue
NVPTXTargetLowering::LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  SDLoc dl(Node);
  SmallVector<SDValue, 8> Ops;
  unsigned NumOperands = Node->getNumOperands();
  for (unsigned i = 0; i < NumOperands; ++i) {
    SDValue SubOp = Node->getOperand(i);
    EVT VVT = SubOp.getNode()->getValueType(0);
    EVT EltVT = VVT.getVectorElementType();
    unsigned NumSubElem = VVT.getVectorNumElements();
    for (unsigned j = 0; j < NumSubElem; ++j) {
      Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, SubOp,
                                DAG.getIntPtrConstant(j)));
    }
  }
  return DAG.getNode(ISD::BUILD_VECTOR, dl, Node->getValueType(0), &Ops[0],
                     Ops.size());
}

SDValue
NVPTXTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::RETURNADDR:
    return SDValue();
  case ISD::FRAMEADDR:
    return SDValue();
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::INTRINSIC_W_CHAIN:
    return Op;
  case ISD::BUILD_VECTOR:
  case ISD::EXTRACT_SUBVECTOR:
    return Op;
  case ISD::CONCAT_VECTORS:
    return LowerCONCAT_VECTORS(Op, DAG);
  case ISD::STORE:
    return LowerSTORE(Op, DAG);
  case ISD::LOAD:
    return LowerLOAD(Op, DAG);
  default:
    llvm_unreachable("Custom lowering not defined for operation");
  }
}

SDValue NVPTXTargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  if (Op.getValueType() == MVT::i1)
    return LowerLOADi1(Op, DAG);
  else
    return SDValue();
}

// v = ld i1* addr
//   =>
// v1 = ld i8* addr (-> i16)
// v = trunc i16 to i1
SDValue NVPTXTargetLowering::LowerLOADi1(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  LoadSDNode *LD = cast<LoadSDNode>(Node);
  SDLoc dl(Node);
  assert(LD->getExtensionType() == ISD::NON_EXTLOAD);
  assert(Node->getValueType(0) == MVT::i1 &&
         "Custom lowering for i1 load only");
  SDValue newLD =
      DAG.getLoad(MVT::i16, dl, LD->getChain(), LD->getBasePtr(),
                  LD->getPointerInfo(), LD->isVolatile(), LD->isNonTemporal(),
                  LD->isInvariant(), LD->getAlignment());
  SDValue result = DAG.getNode(ISD::TRUNCATE, dl, MVT::i1, newLD);
  // The legalizer (the caller) is expecting two values from the legalized
  // load, so we build a MergeValues node for it. See ExpandUnalignedLoad()
  // in LegalizeDAG.cpp which also uses MergeValues.
  SDValue Ops[] = { result, LD->getChain() };
  return DAG.getMergeValues(Ops, 2, dl);
}

SDValue NVPTXTargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  EVT ValVT = Op.getOperand(1).getValueType();
  if (ValVT == MVT::i1)
    return LowerSTOREi1(Op, DAG);
  else if (ValVT.isVector())
    return LowerSTOREVector(Op, DAG);
  else
    return SDValue();
}

SDValue
NVPTXTargetLowering::LowerSTOREVector(SDValue Op, SelectionDAG &DAG) const {
  SDNode *N = Op.getNode();
  SDValue Val = N->getOperand(1);
  SDLoc DL(N);
  EVT ValVT = Val.getValueType();

  if (ValVT.isVector()) {
    // We only handle "native" vector sizes for now, e.g. <4 x double> is not
    // legal.  We can (and should) split that into 2 stores of <2 x double> here
    // but I'm leaving that as a TODO for now.
    if (!ValVT.isSimple())
      return SDValue();
    switch (ValVT.getSimpleVT().SimpleTy) {
    default:
      return SDValue();
    case MVT::v2i8:
    case MVT::v2i16:
    case MVT::v2i32:
    case MVT::v2i64:
    case MVT::v2f32:
    case MVT::v2f64:
    case MVT::v4i8:
    case MVT::v4i16:
    case MVT::v4i32:
    case MVT::v4f32:
      // This is a "native" vector type
      break;
    }

    unsigned Opcode = 0;
    EVT EltVT = ValVT.getVectorElementType();
    unsigned NumElts = ValVT.getVectorNumElements();

    // Since StoreV2 is a target node, we cannot rely on DAG type legalization.
    // Therefore, we must ensure the type is legal.  For i1 and i8, we set the
    // stored type to i16 and propogate the "real" type as the memory type.
    bool NeedExt = false;
    if (EltVT.getSizeInBits() < 16)
      NeedExt = true;

    switch (NumElts) {
    default:
      return SDValue();
    case 2:
      Opcode = NVPTXISD::StoreV2;
      break;
    case 4: {
      Opcode = NVPTXISD::StoreV4;
      break;
    }
    }

    SmallVector<SDValue, 8> Ops;

    // First is the chain
    Ops.push_back(N->getOperand(0));

    // Then the split values
    for (unsigned i = 0; i < NumElts; ++i) {
      SDValue ExtVal = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, Val,
                                   DAG.getIntPtrConstant(i));
      if (NeedExt)
        ExtVal = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, ExtVal);
      Ops.push_back(ExtVal);
    }

    // Then any remaining arguments
    for (unsigned i = 2, e = N->getNumOperands(); i != e; ++i) {
      Ops.push_back(N->getOperand(i));
    }

    MemSDNode *MemSD = cast<MemSDNode>(N);

    SDValue NewSt = DAG.getMemIntrinsicNode(
        Opcode, DL, DAG.getVTList(MVT::Other), &Ops[0], Ops.size(),
        MemSD->getMemoryVT(), MemSD->getMemOperand());

    //return DCI.CombineTo(N, NewSt, true);
    return NewSt;
  }

  return SDValue();
}

// st i1 v, addr
//    =>
// v1 = zxt v to i16
// st.u8 i16, addr
SDValue NVPTXTargetLowering::LowerSTOREi1(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  SDLoc dl(Node);
  StoreSDNode *ST = cast<StoreSDNode>(Node);
  SDValue Tmp1 = ST->getChain();
  SDValue Tmp2 = ST->getBasePtr();
  SDValue Tmp3 = ST->getValue();
  assert(Tmp3.getValueType() == MVT::i1 && "Custom lowering for i1 store only");
  unsigned Alignment = ST->getAlignment();
  bool isVolatile = ST->isVolatile();
  bool isNonTemporal = ST->isNonTemporal();
  Tmp3 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, Tmp3);
  SDValue Result = DAG.getTruncStore(Tmp1, dl, Tmp3, Tmp2,
                                     ST->getPointerInfo(), MVT::i8, isNonTemporal,
                                     isVolatile, Alignment);
  return Result;
}

SDValue NVPTXTargetLowering::getExtSymb(SelectionDAG &DAG, const char *inname,
                                        int idx, EVT v) const {
  std::string *name = nvTM->getManagedStrPool()->getManagedString(inname);
  std::stringstream suffix;
  suffix << idx;
  *name += suffix.str();
  return DAG.getTargetExternalSymbol(name->c_str(), v);
}

SDValue
NVPTXTargetLowering::getParamSymbol(SelectionDAG &DAG, int idx, EVT v) const {
  return getExtSymb(DAG, ".PARAM", idx, v);
}

SDValue NVPTXTargetLowering::getParamHelpSymbol(SelectionDAG &DAG, int idx) {
  return getExtSymb(DAG, ".HLPPARAM", idx);
}

// Check to see if the kernel argument is image*_t or sampler_t

bool llvm::isImageOrSamplerVal(const Value *arg, const Module *context) {
  static const char *const specialTypes[] = { "struct._image2d_t",
                                              "struct._image3d_t",
                                              "struct._sampler_t" };

  const Type *Ty = arg->getType();
  const PointerType *PTy = dyn_cast<PointerType>(Ty);

  if (!PTy)
    return false;

  if (!context)
    return false;

  const StructType *STy = dyn_cast<StructType>(PTy->getElementType());
  const std::string TypeName = STy && !STy->isLiteral() ? STy->getName() : "";

  for (int i = 0, e = array_lengthof(specialTypes); i != e; ++i)
    if (TypeName == specialTypes[i])
      return true;

  return false;
}

SDValue NVPTXTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, SDLoc dl, SelectionDAG &DAG,
    SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  const DataLayout *TD = getDataLayout();

  const Function *F = MF.getFunction();
  const AttributeSet &PAL = F->getAttributes();
  const TargetLowering *TLI = nvTM->getTargetLowering();

  SDValue Root = DAG.getRoot();
  std::vector<SDValue> OutChains;

  bool isKernel = llvm::isKernelFunction(*F);
  bool isABI = (nvptxSubtarget.getSmVersion() >= 20);
  assert(isABI && "Non-ABI compilation is not supported");
  if (!isABI)
    return Chain;

  std::vector<Type *> argTypes;
  std::vector<const Argument *> theArgs;
  for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; ++I) {
    theArgs.push_back(I);
    argTypes.push_back(I->getType());
  }
  // argTypes.size() (or theArgs.size()) and Ins.size() need not match.
  // Ins.size() will be larger
  //   * if there is an aggregate argument with multiple fields (each field
  //     showing up separately in Ins)
  //   * if there is a vector argument with more than typical vector-length
  //     elements (generally if more than 4) where each vector element is
  //     individually present in Ins.
  // So a different index should be used for indexing into Ins.
  // See similar issue in LowerCall.
  unsigned InsIdx = 0;

  int idx = 0;
  for (unsigned i = 0, e = theArgs.size(); i != e; ++i, ++idx, ++InsIdx) {
    Type *Ty = argTypes[i];

    // If the kernel argument is image*_t or sampler_t, convert it to
    // a i32 constant holding the parameter position. This can later
    // matched in the AsmPrinter to output the correct mangled name.
    if (isImageOrSamplerVal(
            theArgs[i],
            (theArgs[i]->getParent() ? theArgs[i]->getParent()->getParent()
                                     : 0))) {
      assert(isKernel && "Only kernels can have image/sampler params");
      InVals.push_back(DAG.getConstant(i + 1, MVT::i32));
      continue;
    }

    if (theArgs[i]->use_empty()) {
      // argument is dead
      if (Ty->isAggregateType()) {
        SmallVector<EVT, 16> vtparts;

        ComputePTXValueVTs(*this, Ty, vtparts);
        assert(vtparts.size() > 0 && "empty aggregate type not expected");
        for (unsigned parti = 0, parte = vtparts.size(); parti != parte;
             ++parti) {
          EVT partVT = vtparts[parti];
          InVals.push_back(DAG.getNode(ISD::UNDEF, dl, partVT));
          ++InsIdx;
        }
        if (vtparts.size() > 0)
          --InsIdx;
        continue;
      }
      if (Ty->isVectorTy()) {
        EVT ObjectVT = getValueType(Ty);
        unsigned NumRegs = TLI->getNumRegisters(F->getContext(), ObjectVT);
        for (unsigned parti = 0; parti < NumRegs; ++parti) {
          InVals.push_back(DAG.getNode(ISD::UNDEF, dl, Ins[InsIdx].VT));
          ++InsIdx;
        }
        if (NumRegs > 0)
          --InsIdx;
        continue;
      }
      InVals.push_back(DAG.getNode(ISD::UNDEF, dl, Ins[InsIdx].VT));
      continue;
    }

    // In the following cases, assign a node order of "idx+1"
    // to newly created nodes. The SDNodes for params have to
    // appear in the same order as their order of appearance
    // in the original function. "idx+1" holds that order.
    if (PAL.hasAttribute(i + 1, Attribute::ByVal) == false) {
      if (Ty->isAggregateType()) {
        SmallVector<EVT, 16> vtparts;
        SmallVector<uint64_t, 16> offsets;

        // NOTE: Here, we lose the ability to issue vector loads for vectors
        // that are a part of a struct.  This should be investigated in the
        // future.
        ComputePTXValueVTs(*this, Ty, vtparts, &offsets, 0);
        assert(vtparts.size() > 0 && "empty aggregate type not expected");
        bool aggregateIsPacked = false;
        if (StructType *STy = llvm::dyn_cast<StructType>(Ty))
          aggregateIsPacked = STy->isPacked();

        SDValue Arg = getParamSymbol(DAG, idx, getPointerTy());
        for (unsigned parti = 0, parte = vtparts.size(); parti != parte;
             ++parti) {
          EVT partVT = vtparts[parti];
          Value *srcValue = Constant::getNullValue(
              PointerType::get(partVT.getTypeForEVT(F->getContext()),
                               llvm::ADDRESS_SPACE_PARAM));
          SDValue srcAddr =
              DAG.getNode(ISD::ADD, dl, getPointerTy(), Arg,
                          DAG.getConstant(offsets[parti], getPointerTy()));
          unsigned partAlign =
              aggregateIsPacked ? 1
                                : TD->getABITypeAlignment(
                                      partVT.getTypeForEVT(F->getContext()));
          SDValue p;
          if (Ins[InsIdx].VT.getSizeInBits() > partVT.getSizeInBits()) {
            ISD::LoadExtType ExtOp = Ins[InsIdx].Flags.isSExt() ? 
                                     ISD::SEXTLOAD : ISD::ZEXTLOAD;
            p = DAG.getExtLoad(ExtOp, dl, Ins[InsIdx].VT, Root, srcAddr,
                               MachinePointerInfo(srcValue), partVT, false,
                               false, partAlign);
          } else {
            p = DAG.getLoad(partVT, dl, Root, srcAddr,
                            MachinePointerInfo(srcValue), false, false, false,
                            partAlign);
          }
          if (p.getNode())
            p.getNode()->setIROrder(idx + 1);
          InVals.push_back(p);
          ++InsIdx;
        }
        if (vtparts.size() > 0)
          --InsIdx;
        continue;
      }
      if (Ty->isVectorTy()) {
        EVT ObjectVT = getValueType(Ty);
        SDValue Arg = getParamSymbol(DAG, idx, getPointerTy());
        unsigned NumElts = ObjectVT.getVectorNumElements();
        assert(TLI->getNumRegisters(F->getContext(), ObjectVT) == NumElts &&
               "Vector was not scalarized");
        unsigned Ofst = 0;
        EVT EltVT = ObjectVT.getVectorElementType();

        // V1 load
        // f32 = load ...
        if (NumElts == 1) {
          // We only have one element, so just directly load it
          Value *SrcValue = Constant::getNullValue(PointerType::get(
              EltVT.getTypeForEVT(F->getContext()), llvm::ADDRESS_SPACE_PARAM));
          SDValue SrcAddr = DAG.getNode(ISD::ADD, dl, getPointerTy(), Arg,
                                        DAG.getConstant(Ofst, getPointerTy()));
          SDValue P = DAG.getLoad(
              EltVT, dl, Root, SrcAddr, MachinePointerInfo(SrcValue), false,
              false, true,
              TD->getABITypeAlignment(EltVT.getTypeForEVT(F->getContext())));
          if (P.getNode())
            P.getNode()->setIROrder(idx + 1);

          if (Ins[InsIdx].VT.getSizeInBits() > EltVT.getSizeInBits())
            P = DAG.getNode(ISD::ANY_EXTEND, dl, Ins[InsIdx].VT, P);
          InVals.push_back(P);
          Ofst += TD->getTypeAllocSize(EltVT.getTypeForEVT(F->getContext()));
          ++InsIdx;
        } else if (NumElts == 2) {
          // V2 load
          // f32,f32 = load ...
          EVT VecVT = EVT::getVectorVT(F->getContext(), EltVT, 2);
          Value *SrcValue = Constant::getNullValue(PointerType::get(
              VecVT.getTypeForEVT(F->getContext()), llvm::ADDRESS_SPACE_PARAM));
          SDValue SrcAddr = DAG.getNode(ISD::ADD, dl, getPointerTy(), Arg,
                                        DAG.getConstant(Ofst, getPointerTy()));
          SDValue P = DAG.getLoad(
              VecVT, dl, Root, SrcAddr, MachinePointerInfo(SrcValue), false,
              false, true,
              TD->getABITypeAlignment(VecVT.getTypeForEVT(F->getContext())));
          if (P.getNode())
            P.getNode()->setIROrder(idx + 1);

          SDValue Elt0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, P,
                                     DAG.getIntPtrConstant(0));
          SDValue Elt1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, P,
                                     DAG.getIntPtrConstant(1));

          if (Ins[InsIdx].VT.getSizeInBits() > EltVT.getSizeInBits()) {
            Elt0 = DAG.getNode(ISD::ANY_EXTEND, dl, Ins[InsIdx].VT, Elt0);
            Elt1 = DAG.getNode(ISD::ANY_EXTEND, dl, Ins[InsIdx].VT, Elt1);
          }

          InVals.push_back(Elt0);
          InVals.push_back(Elt1);
          Ofst += TD->getTypeAllocSize(VecVT.getTypeForEVT(F->getContext()));
          InsIdx += 2;
        } else {
          // V4 loads
          // We have at least 4 elements (<3 x Ty> expands to 4 elements) and
          // the
          // vector will be expanded to a power of 2 elements, so we know we can
          // always round up to the next multiple of 4 when creating the vector
          // loads.
          // e.g.  4 elem => 1 ld.v4
          //       6 elem => 2 ld.v4
          //       8 elem => 2 ld.v4
          //      11 elem => 3 ld.v4
          unsigned VecSize = 4;
          if (EltVT.getSizeInBits() == 64) {
            VecSize = 2;
          }
          EVT VecVT = EVT::getVectorVT(F->getContext(), EltVT, VecSize);
          for (unsigned i = 0; i < NumElts; i += VecSize) {
            Value *SrcValue = Constant::getNullValue(
                PointerType::get(VecVT.getTypeForEVT(F->getContext()),
                                 llvm::ADDRESS_SPACE_PARAM));
            SDValue SrcAddr =
                DAG.getNode(ISD::ADD, dl, getPointerTy(), Arg,
                            DAG.getConstant(Ofst, getPointerTy()));
            SDValue P = DAG.getLoad(
                VecVT, dl, Root, SrcAddr, MachinePointerInfo(SrcValue), false,
                false, true,
                TD->getABITypeAlignment(VecVT.getTypeForEVT(F->getContext())));
            if (P.getNode())
              P.getNode()->setIROrder(idx + 1);

            for (unsigned j = 0; j < VecSize; ++j) {
              if (i + j >= NumElts)
                break;
              SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, P,
                                        DAG.getIntPtrConstant(j));
              if (Ins[InsIdx].VT.getSizeInBits() > EltVT.getSizeInBits())
                Elt = DAG.getNode(ISD::ANY_EXTEND, dl, Ins[InsIdx].VT, Elt);
              InVals.push_back(Elt);
            }
            Ofst += TD->getTypeAllocSize(VecVT.getTypeForEVT(F->getContext()));
          }
          InsIdx += VecSize;
        }

        if (NumElts > 0)
          --InsIdx;
        continue;
      }
      // A plain scalar.
      EVT ObjectVT = getValueType(Ty);
      // If ABI, load from the param symbol
      SDValue Arg = getParamSymbol(DAG, idx, getPointerTy());
      Value *srcValue = Constant::getNullValue(PointerType::get(
          ObjectVT.getTypeForEVT(F->getContext()), llvm::ADDRESS_SPACE_PARAM));
      SDValue p;
       if (ObjectVT.getSizeInBits() < Ins[InsIdx].VT.getSizeInBits()) {
        ISD::LoadExtType ExtOp = Ins[InsIdx].Flags.isSExt() ? 
                                       ISD::SEXTLOAD : ISD::ZEXTLOAD;
        p = DAG.getExtLoad(ExtOp, dl, Ins[InsIdx].VT, Root, Arg,
                           MachinePointerInfo(srcValue), ObjectVT, false, false,
        TD->getABITypeAlignment(ObjectVT.getTypeForEVT(F->getContext())));
      } else {
        p = DAG.getLoad(Ins[InsIdx].VT, dl, Root, Arg,
                        MachinePointerInfo(srcValue), false, false, false,
        TD->getABITypeAlignment(ObjectVT.getTypeForEVT(F->getContext())));
      }
      if (p.getNode())
        p.getNode()->setIROrder(idx + 1);
      InVals.push_back(p);
      continue;
    }

    // Param has ByVal attribute
    // Return MoveParam(param symbol).
    // Ideally, the param symbol can be returned directly,
    // but when SDNode builder decides to use it in a CopyToReg(),
    // machine instruction fails because TargetExternalSymbol
    // (not lowered) is target dependent, and CopyToReg assumes
    // the source is lowered.
    EVT ObjectVT = getValueType(Ty);
    assert(ObjectVT == Ins[InsIdx].VT &&
           "Ins type did not match function type");
    SDValue Arg = getParamSymbol(DAG, idx, getPointerTy());
    SDValue p = DAG.getNode(NVPTXISD::MoveParam, dl, ObjectVT, Arg);
    if (p.getNode())
      p.getNode()->setIROrder(idx + 1);
    if (isKernel)
      InVals.push_back(p);
    else {
      SDValue p2 = DAG.getNode(
          ISD::INTRINSIC_WO_CHAIN, dl, ObjectVT,
          DAG.getConstant(Intrinsic::nvvm_ptr_local_to_gen, MVT::i32), p);
      InVals.push_back(p2);
    }
  }

  // Clang will check explicit VarArg and issue error if any. However, Clang
  // will let code with
  // implicit var arg like f() pass. See bug 617733.
  // We treat this case as if the arg list is empty.
  // if (F.isVarArg()) {
  // assert(0 && "VarArg not supported yet!");
  //}

  if (!OutChains.empty())
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &OutChains[0],
                            OutChains.size()));

  return Chain;
}


SDValue
NVPTXTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                 bool isVarArg,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 SDLoc dl, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  const Function *F = MF.getFunction();
  Type *RetTy = F->getReturnType();
  const DataLayout *TD = getDataLayout();

  bool isABI = (nvptxSubtarget.getSmVersion() >= 20);
  assert(isABI && "Non-ABI compilation is not supported");
  if (!isABI)
    return Chain;

  if (VectorType *VTy = dyn_cast<VectorType>(RetTy)) {
    // If we have a vector type, the OutVals array will be the scalarized
    // components and we have combine them into 1 or more vector stores.
    unsigned NumElts = VTy->getNumElements();
    assert(NumElts == Outs.size() && "Bad scalarization of return value");

    // const_cast can be removed in later LLVM versions
    EVT EltVT = getValueType(RetTy).getVectorElementType();
    bool NeedExtend = false;
    if (EltVT.getSizeInBits() < 16)
      NeedExtend = true;

    // V1 store
    if (NumElts == 1) {
      SDValue StoreVal = OutVals[0];
      // We only have one element, so just directly store it
      if (NeedExtend)
        StoreVal = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, StoreVal);
      SDValue Ops[] = { Chain, DAG.getConstant(0, MVT::i32), StoreVal };
      Chain = DAG.getMemIntrinsicNode(NVPTXISD::StoreRetval, dl,
                                      DAG.getVTList(MVT::Other), &Ops[0], 3,
                                      EltVT, MachinePointerInfo());

    } else if (NumElts == 2) {
      // V2 store
      SDValue StoreVal0 = OutVals[0];
      SDValue StoreVal1 = OutVals[1];

      if (NeedExtend) {
        StoreVal0 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, StoreVal0);
        StoreVal1 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, StoreVal1);
      }

      SDValue Ops[] = { Chain, DAG.getConstant(0, MVT::i32), StoreVal0,
                        StoreVal1 };
      Chain = DAG.getMemIntrinsicNode(NVPTXISD::StoreRetvalV2, dl,
                                      DAG.getVTList(MVT::Other), &Ops[0], 4,
                                      EltVT, MachinePointerInfo());
    } else {
      // V4 stores
      // We have at least 4 elements (<3 x Ty> expands to 4 elements) and the
      // vector will be expanded to a power of 2 elements, so we know we can
      // always round up to the next multiple of 4 when creating the vector
      // stores.
      // e.g.  4 elem => 1 st.v4
      //       6 elem => 2 st.v4
      //       8 elem => 2 st.v4
      //      11 elem => 3 st.v4

      unsigned VecSize = 4;
      if (OutVals[0].getValueType().getSizeInBits() == 64)
        VecSize = 2;

      unsigned Offset = 0;

      EVT VecVT =
          EVT::getVectorVT(F->getContext(), OutVals[0].getValueType(), VecSize);
      unsigned PerStoreOffset =
          TD->getTypeAllocSize(VecVT.getTypeForEVT(F->getContext()));

      for (unsigned i = 0; i < NumElts; i += VecSize) {
        // Get values
        SDValue StoreVal;
        SmallVector<SDValue, 8> Ops;
        Ops.push_back(Chain);
        Ops.push_back(DAG.getConstant(Offset, MVT::i32));
        unsigned Opc = NVPTXISD::StoreRetvalV2;
        EVT ExtendedVT = (NeedExtend) ? MVT::i16 : OutVals[0].getValueType();

        StoreVal = OutVals[i];
        if (NeedExtend)
          StoreVal = DAG.getNode(ISD::ZERO_EXTEND, dl, ExtendedVT, StoreVal);
        Ops.push_back(StoreVal);

        if (i + 1 < NumElts) {
          StoreVal = OutVals[i + 1];
          if (NeedExtend)
            StoreVal = DAG.getNode(ISD::ZERO_EXTEND, dl, ExtendedVT, StoreVal);
        } else {
          StoreVal = DAG.getUNDEF(ExtendedVT);
        }
        Ops.push_back(StoreVal);

        if (VecSize == 4) {
          Opc = NVPTXISD::StoreRetvalV4;
          if (i + 2 < NumElts) {
            StoreVal = OutVals[i + 2];
            if (NeedExtend)
              StoreVal =
                  DAG.getNode(ISD::ZERO_EXTEND, dl, ExtendedVT, StoreVal);
          } else {
            StoreVal = DAG.getUNDEF(ExtendedVT);
          }
          Ops.push_back(StoreVal);

          if (i + 3 < NumElts) {
            StoreVal = OutVals[i + 3];
            if (NeedExtend)
              StoreVal =
                  DAG.getNode(ISD::ZERO_EXTEND, dl, ExtendedVT, StoreVal);
          } else {
            StoreVal = DAG.getUNDEF(ExtendedVT);
          }
          Ops.push_back(StoreVal);
        }

        // Chain = DAG.getNode(Opc, dl, MVT::Other, &Ops[0], Ops.size());
        Chain =
            DAG.getMemIntrinsicNode(Opc, dl, DAG.getVTList(MVT::Other), &Ops[0],
                                    Ops.size(), EltVT, MachinePointerInfo());
        Offset += PerStoreOffset;
      }
    }
  } else {
    SmallVector<EVT, 16> ValVTs;
    // const_cast is necessary since we are still using an LLVM version from
    // before the type system re-write.
    ComputePTXValueVTs(*this, RetTy, ValVTs);
    assert(ValVTs.size() == OutVals.size() && "Bad return value decomposition");

    unsigned SizeSoFar = 0;
    for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
      SDValue theVal = OutVals[i];
      EVT TheValType = theVal.getValueType();
      unsigned numElems = 1;
      if (TheValType.isVector())
        numElems = TheValType.getVectorNumElements();
      for (unsigned j = 0, je = numElems; j != je; ++j) {
        SDValue TmpVal = theVal;
        if (TheValType.isVector())
          TmpVal = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl,
                               TheValType.getVectorElementType(), TmpVal,
                               DAG.getIntPtrConstant(j));
        EVT TheStoreType = ValVTs[i];
        if (RetTy->isIntegerTy() &&
            TD->getTypeAllocSizeInBits(RetTy) < 32) {
          // The following zero-extension is for integer types only, and
          // specifically not for aggregates.
          TmpVal = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, TmpVal);
          TheStoreType = MVT::i32;
        }
        else if (TmpVal.getValueType().getSizeInBits() < 16)
          TmpVal = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i16, TmpVal);

        SDValue Ops[] = { Chain, DAG.getConstant(SizeSoFar, MVT::i32), TmpVal };
        Chain = DAG.getMemIntrinsicNode(NVPTXISD::StoreRetval, dl,
                                        DAG.getVTList(MVT::Other), &Ops[0],
                                        3, TheStoreType,
                                        MachinePointerInfo());
        if(TheValType.isVector())
          SizeSoFar += 
            TheStoreType.getVectorElementType().getStoreSizeInBits() / 8;
        else
          SizeSoFar += TheStoreType.getStoreSizeInBits()/8;
      }
    }
  }

  return DAG.getNode(NVPTXISD::RET_FLAG, dl, MVT::Other, Chain);
}


void NVPTXTargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, std::string &Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  if (Constraint.length() > 1)
    return;
  else
    TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

// NVPTX suuport vector of legal types of any length in Intrinsics because the
// NVPTX specific type legalizer
// will legalize them to the PTX supported length.
bool NVPTXTargetLowering::isTypeSupportedInIntrinsic(MVT VT) const {
  if (isTypeLegal(VT))
    return true;
  if (VT.isVector()) {
    MVT eVT = VT.getVectorElementType();
    if (isTypeLegal(eVT))
      return true;
  }
  return false;
}

// llvm.ptx.memcpy.const and llvm.ptx.memmove.const need to be modeled as
// TgtMemIntrinsic
// because we need the information that is only available in the "Value" type
// of destination
// pointer. In particular, the address space information.
bool NVPTXTargetLowering::getTgtMemIntrinsic(
    IntrinsicInfo &Info, const CallInst &I, unsigned Intrinsic) const {
  switch (Intrinsic) {
  default:
    return false;

  case Intrinsic::nvvm_atomic_load_add_f32:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::f32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.vol = 0;
    Info.readMem = true;
    Info.writeMem = true;
    Info.align = 0;
    return true;

  case Intrinsic::nvvm_atomic_load_inc_32:
  case Intrinsic::nvvm_atomic_load_dec_32:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.vol = 0;
    Info.readMem = true;
    Info.writeMem = true;
    Info.align = 0;
    return true;

  case Intrinsic::nvvm_ldu_global_i:
  case Intrinsic::nvvm_ldu_global_f:
  case Intrinsic::nvvm_ldu_global_p:

    Info.opc = ISD::INTRINSIC_W_CHAIN;
    if (Intrinsic == Intrinsic::nvvm_ldu_global_i)
      Info.memVT = getValueType(I.getType());
    else if (Intrinsic == Intrinsic::nvvm_ldu_global_p)
      Info.memVT = getValueType(I.getType());
    else
      Info.memVT = MVT::f32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.vol = 0;
    Info.readMem = true;
    Info.writeMem = false;
    Info.align = 0;
    return true;

  }
  return false;
}

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
/// Used to guide target specific optimizations, like loop strength reduction
/// (LoopStrengthReduce.cpp) and memory optimization for address mode
/// (CodeGenPrepare.cpp)
bool NVPTXTargetLowering::isLegalAddressingMode(const AddrMode &AM,
                                                Type *Ty) const {

  // AddrMode - This represents an addressing mode of:
  //    BaseGV + BaseOffs + BaseReg + Scale*ScaleReg
  //
  // The legal address modes are
  // - [avar]
  // - [areg]
  // - [areg+immoff]
  // - [immAddr]

  if (AM.BaseGV) {
    if (AM.BaseOffs || AM.HasBaseReg || AM.Scale)
      return false;
    return true;
  }

  switch (AM.Scale) {
  case 0: // "r", "r+i" or "i" is allowed
    break;
  case 1:
    if (AM.HasBaseReg) // "r+r+i" or "r+r" is not allowed.
      return false;
    // Otherwise we have r+i.
    break;
  default:
    // No scale > 1 is allowed
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
//                         NVPTX Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
NVPTXTargetLowering::ConstraintType
NVPTXTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      break;
    case 'r':
    case 'h':
    case 'c':
    case 'l':
    case 'f':
    case 'd':
    case '0':
    case 'N':
      return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass *>
NVPTXTargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                  MVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'c':
      return std::make_pair(0U, &NVPTX::Int16RegsRegClass);
    case 'h':
      return std::make_pair(0U, &NVPTX::Int16RegsRegClass);
    case 'r':
      return std::make_pair(0U, &NVPTX::Int32RegsRegClass);
    case 'l':
    case 'N':
      return std::make_pair(0U, &NVPTX::Int64RegsRegClass);
    case 'f':
      return std::make_pair(0U, &NVPTX::Float32RegsRegClass);
    case 'd':
      return std::make_pair(0U, &NVPTX::Float64RegsRegClass);
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

/// getFunctionAlignment - Return the Log2 alignment of this function.
unsigned NVPTXTargetLowering::getFunctionAlignment(const Function *) const {
  return 4;
}

/// ReplaceVectorLoad - Convert vector loads into multi-output scalar loads.
static void ReplaceLoadVector(SDNode *N, SelectionDAG &DAG,
                              SmallVectorImpl<SDValue> &Results) {
  EVT ResVT = N->getValueType(0);
  SDLoc DL(N);

  assert(ResVT.isVector() && "Vector load must have vector type");

  // We only handle "native" vector sizes for now, e.g. <4 x double> is not
  // legal.  We can (and should) split that into 2 loads of <2 x double> here
  // but I'm leaving that as a TODO for now.
  assert(ResVT.isSimple() && "Can only handle simple types");
  switch (ResVT.getSimpleVT().SimpleTy) {
  default:
    return;
  case MVT::v2i8:
  case MVT::v2i16:
  case MVT::v2i32:
  case MVT::v2i64:
  case MVT::v2f32:
  case MVT::v2f64:
  case MVT::v4i8:
  case MVT::v4i16:
  case MVT::v4i32:
  case MVT::v4f32:
    // This is a "native" vector type
    break;
  }

  EVT EltVT = ResVT.getVectorElementType();
  unsigned NumElts = ResVT.getVectorNumElements();

  // Since LoadV2 is a target node, we cannot rely on DAG type legalization.
  // Therefore, we must ensure the type is legal.  For i1 and i8, we set the
  // loaded type to i16 and propogate the "real" type as the memory type.
  bool NeedTrunc = false;
  if (EltVT.getSizeInBits() < 16) {
    EltVT = MVT::i16;
    NeedTrunc = true;
  }

  unsigned Opcode = 0;
  SDVTList LdResVTs;

  switch (NumElts) {
  default:
    return;
  case 2:
    Opcode = NVPTXISD::LoadV2;
    LdResVTs = DAG.getVTList(EltVT, EltVT, MVT::Other);
    break;
  case 4: {
    Opcode = NVPTXISD::LoadV4;
    EVT ListVTs[] = { EltVT, EltVT, EltVT, EltVT, MVT::Other };
    LdResVTs = DAG.getVTList(ListVTs, 5);
    break;
  }
  }

  SmallVector<SDValue, 8> OtherOps;

  // Copy regular operands
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    OtherOps.push_back(N->getOperand(i));

  LoadSDNode *LD = cast<LoadSDNode>(N);

  // The select routine does not have access to the LoadSDNode instance, so
  // pass along the extension information
  OtherOps.push_back(DAG.getIntPtrConstant(LD->getExtensionType()));

  SDValue NewLD = DAG.getMemIntrinsicNode(Opcode, DL, LdResVTs, &OtherOps[0],
                                          OtherOps.size(), LD->getMemoryVT(),
                                          LD->getMemOperand());

  SmallVector<SDValue, 4> ScalarRes;

  for (unsigned i = 0; i < NumElts; ++i) {
    SDValue Res = NewLD.getValue(i);
    if (NeedTrunc)
      Res = DAG.getNode(ISD::TRUNCATE, DL, ResVT.getVectorElementType(), Res);
    ScalarRes.push_back(Res);
  }

  SDValue LoadChain = NewLD.getValue(NumElts);

  SDValue BuildVec =
      DAG.getNode(ISD::BUILD_VECTOR, DL, ResVT, &ScalarRes[0], NumElts);

  Results.push_back(BuildVec);
  Results.push_back(LoadChain);
}

static void ReplaceINTRINSIC_W_CHAIN(SDNode *N, SelectionDAG &DAG,
                                     SmallVectorImpl<SDValue> &Results) {
  SDValue Chain = N->getOperand(0);
  SDValue Intrin = N->getOperand(1);
  SDLoc DL(N);

  // Get the intrinsic ID
  unsigned IntrinNo = cast<ConstantSDNode>(Intrin.getNode())->getZExtValue();
  switch (IntrinNo) {
  default:
    return;
  case Intrinsic::nvvm_ldg_global_i:
  case Intrinsic::nvvm_ldg_global_f:
  case Intrinsic::nvvm_ldg_global_p:
  case Intrinsic::nvvm_ldu_global_i:
  case Intrinsic::nvvm_ldu_global_f:
  case Intrinsic::nvvm_ldu_global_p: {
    EVT ResVT = N->getValueType(0);

    if (ResVT.isVector()) {
      // Vector LDG/LDU

      unsigned NumElts = ResVT.getVectorNumElements();
      EVT EltVT = ResVT.getVectorElementType();

      // Since LDU/LDG are target nodes, we cannot rely on DAG type
      // legalization.
      // Therefore, we must ensure the type is legal.  For i1 and i8, we set the
      // loaded type to i16 and propogate the "real" type as the memory type.
      bool NeedTrunc = false;
      if (EltVT.getSizeInBits() < 16) {
        EltVT = MVT::i16;
        NeedTrunc = true;
      }

      unsigned Opcode = 0;
      SDVTList LdResVTs;

      switch (NumElts) {
      default:
        return;
      case 2:
        switch (IntrinNo) {
        default:
          return;
        case Intrinsic::nvvm_ldg_global_i:
        case Intrinsic::nvvm_ldg_global_f:
        case Intrinsic::nvvm_ldg_global_p:
          Opcode = NVPTXISD::LDGV2;
          break;
        case Intrinsic::nvvm_ldu_global_i:
        case Intrinsic::nvvm_ldu_global_f:
        case Intrinsic::nvvm_ldu_global_p:
          Opcode = NVPTXISD::LDUV2;
          break;
        }
        LdResVTs = DAG.getVTList(EltVT, EltVT, MVT::Other);
        break;
      case 4: {
        switch (IntrinNo) {
        default:
          return;
        case Intrinsic::nvvm_ldg_global_i:
        case Intrinsic::nvvm_ldg_global_f:
        case Intrinsic::nvvm_ldg_global_p:
          Opcode = NVPTXISD::LDGV4;
          break;
        case Intrinsic::nvvm_ldu_global_i:
        case Intrinsic::nvvm_ldu_global_f:
        case Intrinsic::nvvm_ldu_global_p:
          Opcode = NVPTXISD::LDUV4;
          break;
        }
        EVT ListVTs[] = { EltVT, EltVT, EltVT, EltVT, MVT::Other };
        LdResVTs = DAG.getVTList(ListVTs, 5);
        break;
      }
      }

      SmallVector<SDValue, 8> OtherOps;

      // Copy regular operands

      OtherOps.push_back(Chain); // Chain
                                 // Skip operand 1 (intrinsic ID)
      // Others
      for (unsigned i = 2, e = N->getNumOperands(); i != e; ++i)
        OtherOps.push_back(N->getOperand(i));

      MemIntrinsicSDNode *MemSD = cast<MemIntrinsicSDNode>(N);

      SDValue NewLD = DAG.getMemIntrinsicNode(
          Opcode, DL, LdResVTs, &OtherOps[0], OtherOps.size(),
          MemSD->getMemoryVT(), MemSD->getMemOperand());

      SmallVector<SDValue, 4> ScalarRes;

      for (unsigned i = 0; i < NumElts; ++i) {
        SDValue Res = NewLD.getValue(i);
        if (NeedTrunc)
          Res =
              DAG.getNode(ISD::TRUNCATE, DL, ResVT.getVectorElementType(), Res);
        ScalarRes.push_back(Res);
      }

      SDValue LoadChain = NewLD.getValue(NumElts);

      SDValue BuildVec =
          DAG.getNode(ISD::BUILD_VECTOR, DL, ResVT, &ScalarRes[0], NumElts);

      Results.push_back(BuildVec);
      Results.push_back(LoadChain);
    } else {
      // i8 LDG/LDU
      assert(ResVT.isSimple() && ResVT.getSimpleVT().SimpleTy == MVT::i8 &&
             "Custom handling of non-i8 ldu/ldg?");

      // Just copy all operands as-is
      SmallVector<SDValue, 4> Ops;
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
        Ops.push_back(N->getOperand(i));

      // Force output to i16
      SDVTList LdResVTs = DAG.getVTList(MVT::i16, MVT::Other);

      MemIntrinsicSDNode *MemSD = cast<MemIntrinsicSDNode>(N);

      // We make sure the memory type is i8, which will be used during isel
      // to select the proper instruction.
      SDValue NewLD =
          DAG.getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, LdResVTs, &Ops[0],
                                  Ops.size(), MVT::i8, MemSD->getMemOperand());

      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i8,
                                    NewLD.getValue(0)));
      Results.push_back(NewLD.getValue(1));
    }
  }
  }
}

void NVPTXTargetLowering::ReplaceNodeResults(
    SDNode *N, SmallVectorImpl<SDValue> &Results, SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  default:
    report_fatal_error("Unhandled custom legalization");
  case ISD::LOAD:
    ReplaceLoadVector(N, DAG, Results);
    return;
  case ISD::INTRINSIC_W_CHAIN:
    ReplaceINTRINSIC_W_CHAIN(N, DAG, Results);
    return;
  }
}
