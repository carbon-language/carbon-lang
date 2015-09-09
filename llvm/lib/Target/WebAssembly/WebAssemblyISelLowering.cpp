//=- WebAssemblyISelLowering.cpp - WebAssembly DAG Lowering Implementation -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the WebAssemblyTargetLowering class.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyISelLowering.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyTargetMachine.h"
#include "WebAssemblyTargetObjectFile.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-lower"

namespace {
// Diagnostic information for unimplemented or unsupported feature reporting.
// FIXME copied from BPF and AMDGPU.
class DiagnosticInfoUnsupported : public DiagnosticInfo {
private:
  // Debug location where this diagnostic is triggered.
  DebugLoc DLoc;
  const Twine &Description;
  const Function &Fn;
  SDValue Value;

  static int KindID;

  static int getKindID() {
    if (KindID == 0)
      KindID = llvm::getNextAvailablePluginDiagnosticKind();
    return KindID;
  }

public:
  DiagnosticInfoUnsupported(SDLoc DLoc, const Function &Fn, const Twine &Desc,
                            SDValue Value)
      : DiagnosticInfo(getKindID(), DS_Error), DLoc(DLoc.getDebugLoc()),
        Description(Desc), Fn(Fn), Value(Value) {}

  void print(DiagnosticPrinter &DP) const override {
    std::string Str;
    raw_string_ostream OS(Str);

    if (DLoc) {
      auto DIL = DLoc.get();
      StringRef Filename = DIL->getFilename();
      unsigned Line = DIL->getLine();
      unsigned Column = DIL->getColumn();
      OS << Filename << ':' << Line << ':' << Column << ' ';
    }

    OS << "in function " << Fn.getName() << ' ' << *Fn.getFunctionType() << '\n'
       << Description;
    if (Value)
      Value->print(OS);
    OS << '\n';
    OS.flush();
    DP << Str;
  }

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == getKindID();
  }
};

int DiagnosticInfoUnsupported::KindID = 0;
} // end anonymous namespace

WebAssemblyTargetLowering::WebAssemblyTargetLowering(
    const TargetMachine &TM, const WebAssemblySubtarget &STI)
    : TargetLowering(TM), Subtarget(&STI) {
  auto MVTPtr = Subtarget->hasAddr64() ? MVT::i64 : MVT::i32;

  // Booleans always contain 0 or 1.
  setBooleanContents(ZeroOrOneBooleanContent);
  // WebAssembly does not produce floating-point exceptions on normal floating
  // point operations.
  setHasFloatingPointExceptions(false);
  // We don't know the microarchitecture here, so just reduce register pressure.
  setSchedulingPreference(Sched::RegPressure);
  // Tell ISel that we have a stack pointer.
  setStackPointerRegisterToSaveRestore(
      Subtarget->hasAddr64() ? WebAssembly::SP64 : WebAssembly::SP32);
  // Set up the register classes.
  addRegisterClass(MVT::i32, &WebAssembly::Int32RegClass);
  addRegisterClass(MVT::i64, &WebAssembly::Int64RegClass);
  addRegisterClass(MVT::f32, &WebAssembly::Float32RegClass);
  addRegisterClass(MVT::f64, &WebAssembly::Float64RegClass);
  // Compute derived properties from the register classes.
  computeRegisterProperties(Subtarget->getRegisterInfo());

  setOperationAction(ISD::GlobalAddress, MVTPtr, Custom);

  for (auto T : {MVT::f32, MVT::f64}) {
    // Don't expand the floating-point types to constant pools.
    setOperationAction(ISD::ConstantFP, T, Legal);
    // Expand floating-point comparisons.
    for (auto CC : {ISD::SETO, ISD::SETUO, ISD::SETUEQ, ISD::SETONE,
                    ISD::SETULT, ISD::SETULE, ISD::SETUGT, ISD::SETUGE})
      setCondCodeAction(CC, T, Expand);
    // Expand floating-point library function operators.
    for (auto Op : {ISD::FSIN, ISD::FCOS, ISD::FSINCOS, ISD::FPOWI, ISD::FPOW})
      setOperationAction(Op, T, Expand);
    // Note supported floating-point library function operators that otherwise
    // default to expand.
    for (auto Op : {ISD::FCEIL, ISD::FFLOOR, ISD::FTRUNC, ISD::FNEARBYINT,
                    ISD::FRINT})
      setOperationAction(Op, T, Legal);
  }

  for (auto T : {MVT::i32, MVT::i64}) {
    // Expand unavailable integer operations.
    for (auto Op : {ISD::BSWAP, ISD::ROTL, ISD::ROTR,
                    ISD::SMUL_LOHI, ISD::UMUL_LOHI,
                    ISD::MULHS, ISD::MULHU, ISD::SDIVREM, ISD::UDIVREM,
                    ISD::SHL_PARTS, ISD::SRA_PARTS, ISD::SRL_PARTS,
                    ISD::ADDC, ISD::ADDE, ISD::SUBC, ISD::SUBE}) {
      setOperationAction(Op, T, Expand);
    }
  }

  // As a special case, these operators use the type to mean the type to
  // sign-extend from.
  for (auto T : {MVT::i1, MVT::i8, MVT::i16})
    setOperationAction(ISD::SIGN_EXTEND_INREG, T, Expand);

  // Dynamic stack allocation: use the default expansion.
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVTPtr, Expand);

  // WebAssembly doesn't have:
  //  - Floating-point extending loads.
  //  - Floating-point truncating stores.
  //  - i1 extending loads.
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, MVT::f64, Expand);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  for (auto T : MVT::integer_valuetypes())
    for (auto Ext : {ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD})
      setLoadExtAction(Ext, T, MVT::i1, Promote);
}

FastISel *WebAssemblyTargetLowering::createFastISel(
    FunctionLoweringInfo &FuncInfo, const TargetLibraryInfo *LibInfo) const {
  return WebAssembly::createFastISel(FuncInfo, LibInfo);
}

bool WebAssemblyTargetLowering::isOffsetFoldingLegal(
    const GlobalAddressSDNode *GA) const {
  // The WebAssembly target doesn't support folding offsets into global
  // addresses.
  return false;
}

MVT WebAssemblyTargetLowering::getScalarShiftAmountTy(const DataLayout &DL,
                                                      EVT VT) const {
  return VT.getSimpleVT();
}

const char *
WebAssemblyTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (static_cast<WebAssemblyISD::NodeType>(Opcode)) {
  case WebAssemblyISD::FIRST_NUMBER:
    break;
#define HANDLE_NODETYPE(NODE)                                                  \
  case WebAssemblyISD::NODE:                                                   \
    return "WebAssemblyISD::" #NODE;
#include "WebAssemblyISD.def"
#undef HANDLE_NODETYPE
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// WebAssembly Lowering private implementation.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Lowering Code
//===----------------------------------------------------------------------===//

static void fail(SDLoc DL, SelectionDAG &DAG, const char *msg) {
  MachineFunction &MF = DAG.getMachineFunction();
  DAG.getContext()->diagnose(
      DiagnosticInfoUnsupported(DL, *MF.getFunction(), msg, SDValue()));
}

SDValue
WebAssemblyTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                     SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc DL = CLI.DL;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  MachineFunction &MF = DAG.getMachineFunction();

  CallingConv::ID CallConv = CLI.CallConv;
  if (CallConv != CallingConv::C)
    fail(DL, DAG, "WebAssembly doesn't support non-C calling conventions");
  if (CLI.IsTailCall || MF.getTarget().Options.GuaranteedTailCallOpt)
    fail(DL, DAG, "WebAssembly doesn't support tail call yet");
  if (CLI.IsPatchPoint)
    fail(DL, DAG, "WebAssembly doesn't support patch point yet");

  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;

  bool IsStructRet = (Outs.empty()) ? false : Outs[0].Flags.isSRet();
  if (IsStructRet)
    fail(DL, DAG, "WebAssembly doesn't support struct return yet");

  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  if (Ins.size() > 1)
    fail(DL, DAG, "WebAssembly doesn't support more than 1 returned value yet");

  bool IsVarArg = CLI.IsVarArg;
  if (IsVarArg)
    fail(DL, DAG, "WebAssembly doesn't support varargs yet");

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  unsigned NumBytes = CCInfo.getNextStackOffset();

  auto PtrVT = getPointerTy(MF.getDataLayout());
  auto Zero = DAG.getConstant(0, DL, PtrVT, true);
  auto NB = DAG.getConstant(NumBytes, DL, PtrVT, true);
  Chain = DAG.getCALLSEQ_START(Chain, NB, DL);

  SmallVector<SDValue, 16> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  Ops.append(OutVals.begin(), OutVals.end());

  SmallVector<EVT, 8> Tys;
  for (const auto &In : Ins)
    Tys.push_back(In.VT);
  Tys.push_back(MVT::Other);
  SDVTList TyList = DAG.getVTList(Tys);
  SDValue Res = DAG.getNode(WebAssemblyISD::CALL, DL, TyList, Ops);
  if (Ins.empty()) {
    Chain = Res;
  } else {
    InVals.push_back(Res);
    Chain = Res.getValue(1);
  }

  // FIXME: handle CLI.RetSExt and CLI.RetZExt?

  Chain = DAG.getCALLSEQ_END(Chain, NB, Zero, SDValue(), DL);

  return Chain;
}

bool WebAssemblyTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  // WebAssembly can't currently handle returning tuples.
  return Outs.size() <= 1;
}

SDValue WebAssemblyTargetLowering::LowerReturn(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals, SDLoc DL,
    SelectionDAG &DAG) const {

  assert(Outs.size() <= 1 && "WebAssembly can only return up to one value");
  if (CallConv != CallingConv::C)
    fail(DL, DAG, "WebAssembly doesn't support non-C calling conventions");
  if (IsVarArg)
    fail(DL, DAG, "WebAssembly doesn't support varargs yet");

  SmallVector<SDValue, 4> RetOps(1, Chain);
  RetOps.append(OutVals.begin(), OutVals.end());
  Chain = DAG.getNode(WebAssemblyISD::RETURN, DL, MVT::Other, RetOps);

  return Chain;
}

SDValue WebAssemblyTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, SDLoc DL, SelectionDAG &DAG,
    SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();

  if (CallConv != CallingConv::C)
    fail(DL, DAG, "WebAssembly doesn't support non-C calling conventions");
  if (IsVarArg)
    fail(DL, DAG, "WebAssembly doesn't support varargs yet");
  if (MF.getFunction()->hasStructRetAttr())
    fail(DL, DAG, "WebAssembly doesn't support struct return yet");

  unsigned ArgNo = 0;
  for (const ISD::InputArg &In : Ins) {
    if (In.Flags.isZExt())
      fail(DL, DAG, "WebAssembly hasn't implemented zext arguments");
    if (In.Flags.isSExt())
      fail(DL, DAG, "WebAssembly hasn't implemented sext arguments");
    if (In.Flags.isInReg())
      fail(DL, DAG, "WebAssembly hasn't implemented inreg arguments");
    if (In.Flags.isSRet())
      fail(DL, DAG, "WebAssembly hasn't implemented sret arguments");
    if (In.Flags.isByVal())
      fail(DL, DAG, "WebAssembly hasn't implemented byval arguments");
    if (In.Flags.isInAlloca())
      fail(DL, DAG, "WebAssembly hasn't implemented inalloca arguments");
    if (In.Flags.isNest())
      fail(DL, DAG, "WebAssembly hasn't implemented nest arguments");
    if (In.Flags.isReturned())
      fail(DL, DAG, "WebAssembly hasn't implemented returned arguments");
    if (In.Flags.isInConsecutiveRegs())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs arguments");
    if (In.Flags.isInConsecutiveRegsLast())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs last arguments");
    if (In.Flags.isSplit())
      fail(DL, DAG, "WebAssembly hasn't implemented split arguments");
    // FIXME Do something with In.getOrigAlign()?
    InVals.push_back(
        In.Used
            ? DAG.getNode(WebAssemblyISD::ARGUMENT, DL, In.VT,
                          DAG.getTargetConstant(ArgNo, DL, MVT::i32))
            : DAG.getNode(ISD::UNDEF, DL, In.VT));
    ++ArgNo;
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
//  Custom lowering hooks.
//===----------------------------------------------------------------------===//

SDValue WebAssemblyTargetLowering::LowerOperation(SDValue Op,
                                                  SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("unimplemented operation lowering");
    return SDValue();
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  }
}

SDValue WebAssemblyTargetLowering::LowerGlobalAddress(SDValue Op,
                                                      SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const auto *GA = cast<GlobalAddressSDNode>(Op);
  EVT VT = Op.getValueType();
  assert(GA->getOffset() == 0 &&
         "offsets on global addresses are forbidden by isOffsetFoldingLegal");
  assert(GA->getTargetFlags() == 0 && "WebAssembly doesn't set target flags");
  if (GA->getAddressSpace() != 0)
    fail(DL, DAG, "WebAssembly only expects the 0 address space");
  return DAG.getNode(WebAssemblyISD::Wrapper, DL, VT,
                     DAG.getTargetGlobalAddress(GA->getGlobal(), DL, VT));
}

//===----------------------------------------------------------------------===//
//                          WebAssembly Optimization Hooks
//===----------------------------------------------------------------------===//

MCSection *WebAssemblyTargetObjectFile::SelectSectionForGlobal(
    const GlobalValue *GV, SectionKind Kind, Mangler &Mang,
    const TargetMachine &TM) const {
  return getDataSection();
}
