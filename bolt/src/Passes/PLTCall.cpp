//===--- Passes/PLTCall.h - PLT call optimization -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Replace calls to PLT entries with indirect calls against GOT.
//
//===----------------------------------------------------------------------===//

#include "PLTCall.h"
#include "llvm/Support/Options.h"

#define DEBUG_TYPE "bolt-plt"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

cl::opt<bolt::PLTCall::OptType>
PLT("plt",
  cl::desc("optimize PLT calls (requires linking with -znow)"),
  cl::init(bolt::PLTCall::OT_NONE),
  cl::values(clEnumValN(bolt::PLTCall::OT_NONE,
      "none",
      "do not optimize PLT calls"),
    clEnumValN(bolt::PLTCall::OT_HOT,
      "hot",
      "optimize executed (hot) PLT calls"),
    clEnumValN(bolt::PLTCall::OT_ALL,
      "all",
      "optimize all PLT calls")),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

}

namespace llvm {
namespace bolt {

void PLTCall::runOnFunctions(BinaryContext &BC) {
  if (opts::PLT == OT_NONE)
    return;

  uint64_t NumCallsOptimized = 0;
  for (auto &It : BC.getBinaryFunctions()) {
    auto &Function = It.second;
    if (!shouldOptimize(Function))
      continue;

    if (opts::PLT == OT_HOT &&
        Function.getExecutionCount() == BinaryFunction::COUNT_NO_PROFILE)
      continue;

    for (auto *BB : Function.layout()) {
      if (opts::PLT == OT_HOT && !BB->getKnownExecutionCount())
        continue;

      for (auto &Instr : *BB) {
        if (!BC.MIB->isCall(Instr))
          continue;
        const auto *CallSymbol = BC.MIB->getTargetSymbol(Instr);
        if (!CallSymbol)
          continue;
        const auto *CalleeBF = BC.getFunctionForSymbol(CallSymbol);
        if (!CalleeBF || !CalleeBF->isPLTFunction())
          continue;
        BC.MIB->convertCallToIndirectCall(Instr,
                                          CalleeBF->getPLTSymbol(),
                                          BC.Ctx.get());
        BC.MIB->addAnnotation(Instr, "PLTCall", true);
        ++NumCallsOptimized;
      }
    }
  }

  if (NumCallsOptimized) {
    BC.RequiresZNow = true;
    outs() << "BOLT-INFO: " << NumCallsOptimized
           << " PLT calls in the binary were optimized.\n";
  }
}


} // namespace bolt
} // namespace llvm
