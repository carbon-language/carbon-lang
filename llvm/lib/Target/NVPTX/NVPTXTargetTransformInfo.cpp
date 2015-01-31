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
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "NVPTXtti"

namespace {

class NVPTXTTIImpl : public BasicTTIImplBase<NVPTXTTIImpl> {
  typedef BasicTTIImplBase<NVPTXTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;

  const NVPTXTargetLowering *TLI;

public:
  explicit NVPTXTTIImpl(const NVPTXTargetMachine *TM = nullptr)
      : BaseT(TM),
        TLI(TM ? TM->getSubtargetImpl()->getTargetLowering() : nullptr) {}

  // Provide value semantics. MSVC requires that we spell all of these out.
  NVPTXTTIImpl(const NVPTXTTIImpl &Arg)
      : BaseT(static_cast<const BaseT &>(Arg)), TLI(Arg.TLI) {}
  NVPTXTTIImpl(NVPTXTTIImpl &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))), TLI(std::move(Arg.TLI)) {}
  NVPTXTTIImpl &operator=(const NVPTXTTIImpl &RHS) {
    BaseT::operator=(static_cast<const BaseT &>(RHS));
    TLI = RHS.TLI;
    return *this;
  }
  NVPTXTTIImpl &operator=(NVPTXTTIImpl &&RHS) {
    BaseT::operator=(std::move(static_cast<BaseT &>(RHS)));
    TLI = std::move(RHS.TLI);
    return *this;
  }

  bool hasBranchDivergence() { return true; }

  unsigned getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None);
};

} // end anonymous namespace

ImmutablePass *
llvm::createNVPTXTargetTransformInfoPass(const NVPTXTargetMachine *TM) {
  return new TargetTransformInfoWrapperPass(NVPTXTTIImpl(TM));
}

unsigned NVPTXTTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::OperandValueKind Opd1Info,
    TTI::OperandValueKind Opd2Info, TTI::OperandValueProperties Opd1PropInfo,
    TTI::OperandValueProperties Opd2PropInfo) {
  // Legalize the type.
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Ty);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);

  switch (ISD) {
  default:
    return BaseT::getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                         Opd1PropInfo, Opd2PropInfo);
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
    return BaseT::getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                         Opd1PropInfo, Opd2PropInfo);
  }
}
