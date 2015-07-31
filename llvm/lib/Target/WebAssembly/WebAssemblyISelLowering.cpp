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

  // FIXME: setOperationAction...
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
    if (In.VT != MVT::i32)
      fail(DL, DAG, "WebAssembly hasn't implemented non-i32 arguments");
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
//  Other Lowering Code
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                          WebAssembly Optimization Hooks
//===----------------------------------------------------------------------===//

MCSection *WebAssemblyTargetObjectFile::SelectSectionForGlobal(
    const GlobalValue *GV, SectionKind Kind, Mangler &Mang,
    const TargetMachine &TM) const {
  return getDataSection();
}
