//===-- NVPTXTargetTransformInfo.cpp - NVPTX specific TTI pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements a TargetTransformInfo analysis pass specific to the
// NVPTX target machine. It uses the target's detailed information to provide
// more precise answers to certain TTI queries, while letting the target
// independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#include "NVPTXTargetMachine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "NVPTXtti"

// Declare the pass initialization routine locally as target-specific passes
// don't have a target-wide initialization entry point, and so we rely on the
// pass constructor initialization.
namespace llvm {
void initializeNVPTXTTIPass(PassRegistry &);
}

namespace {

class NVPTXTTI final : public ImmutablePass, public TargetTransformInfo {
  const NVPTXTargetLowering *TLI;
public:
  NVPTXTTI() : ImmutablePass(ID), TLI(nullptr) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  NVPTXTTI(const NVPTXTargetMachine *TM)
      : ImmutablePass(ID), TLI(TM->getSubtargetImpl()->getTargetLowering()) {
    initializeNVPTXTTIPass(*PassRegistry::getPassRegistry());
  }

  void initializePass() override { pushTTIStack(this); }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    TargetTransformInfo::getAnalysisUsage(AU);
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  void *getAdjustedAnalysisPointer(const void *ID) override {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo *)this;
    return this;
  }

  bool hasBranchDivergence() const override;

  unsigned getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, OperandValueKind Opd1Info = OK_AnyValue,
      OperandValueKind Opd2Info = OK_AnyValue,
      OperandValueProperties Opd1PropInfo = OP_None,
      OperandValueProperties Opd2PropInfo = OP_None) const override;
};

} // end anonymous namespace

INITIALIZE_AG_PASS(NVPTXTTI, TargetTransformInfo, "NVPTXtti",
                   "NVPTX Target Transform Info", true, true, false)
char NVPTXTTI::ID = 0;

ImmutablePass *
llvm::createNVPTXTargetTransformInfoPass(const NVPTXTargetMachine *TM) {
  return new NVPTXTTI(TM);
}

bool NVPTXTTI::hasBranchDivergence() const { return true; }

unsigned NVPTXTTI::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, OperandValueKind Opd1Info,
    OperandValueKind Opd2Info, OperandValueProperties Opd1PropInfo,
    OperandValueProperties Opd2PropInfo) const {
  // Legalize the type.
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Ty);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);

  switch (ISD) {
  default:
    return TargetTransformInfo::getArithmeticInstrCost(
        Opcode, Ty, Opd1Info, Opd2Info, Opd1PropInfo, Opd2PropInfo);
  case ISD::ADD:
  case ISD::MUL:
  case ISD::XOR:
  case ISD::OR:
  case ISD::AND:
    // The machine code (SASS) simulates an i64 with two i32. Therefore, we
    // estimate that arithmetic operations on i64 are twice as expensive as
    // those on types that can fit into one machine register.
    if (LT.second.SimpleTy == MVT::i64)
      return 2 * LT.first;
    // Delegate other cases to the basic TTI.
    return TargetTransformInfo::getArithmeticInstrCost(
        Opcode, Ty, Opd1Info, Opd2Info, Opd1PropInfo, Opd2PropInfo);
  }
}
