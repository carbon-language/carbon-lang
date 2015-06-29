//- WebAssemblyISelDAGToDAG.cpp - A dag to dag inst selector for WebAssembly -//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines an instruction selector for the WebAssembly target.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyTargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Function.h" // To access function attributes.
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-isel"

//===--------------------------------------------------------------------===//
/// WebAssembly-specific code to select WebAssembly machine instructions for
/// SelectionDAG operations.
///
namespace {
class WebAssemblyDAGToDAGISel final : public SelectionDAGISel {
  /// Keep a pointer to the WebAssemblySubtarget around so that we can make the
  /// right decision when generating code for different targets.
  const WebAssemblySubtarget *Subtarget;

  bool ForCodeSize;

public:
  WebAssemblyDAGToDAGISel(WebAssemblyTargetMachine &tm,
                          CodeGenOpt::Level OptLevel)
      : SelectionDAGISel(tm, OptLevel), Subtarget(nullptr), ForCodeSize(false) {
  }

  const char *getPassName() const override {
    return "WebAssembly Instruction Selection";
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    ForCodeSize =
        MF.getFunction()->hasFnAttribute(Attribute::OptimizeForSize) ||
        MF.getFunction()->hasFnAttribute(Attribute::MinSize);
    Subtarget = &MF.getSubtarget<WebAssemblySubtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

  SDNode *Select(SDNode *Node) override;

private:
  // add select functions here...
};
} // end anonymous namespace

SDNode *WebAssemblyDAGToDAGISel::Select(SDNode *Node) {
  llvm_unreachable("TODO: implement Select");
}

/// This pass converts a legalized DAG into a WebAssembly-specific DAG, ready
/// for instruction scheduling.
FunctionPass *llvm::createWebAssemblyISelDag(WebAssemblyTargetMachine &TM,
                                             CodeGenOpt::Level OptLevel) {
  return new WebAssemblyDAGToDAGISel(TM, OptLevel);
}
