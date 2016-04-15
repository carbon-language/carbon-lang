//===--- BinaryPass.h - Binary-level analysis/optimization passes ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryPass.h"

#define DEBUG_TYPE "bolt"

namespace opts {

static llvm::cl::opt<bool>
OptimizeBodylessFunctions(
    "optimize-bodyless-functions",
    llvm::cl::desc("optimize functions that just do a tail call"),
    llvm::cl::Optional);

} // namespace opts

namespace llvm {
namespace bolt {

/// Detects functions that simply do a tail call when they are called and
/// optimizes calls to these functions.
class OptimizeBodylessFunctions : public BinaryFunctionPass {
private:
  /// Maps function name to BinaryFunction.
  std::map<std::string, const BinaryFunction *> FunctionByName;

  /// EquivalentCallTarget[F] = G ==> function F is simply a tail call to G,
  /// thus calls to F can be optimized to calls to G.
  std::map<std::string, const BinaryFunction *> EquivalentCallTarget;

  void analyze(BinaryFunction &BF,
               BinaryContext &BC) {
    if (BF.size() != 1 || BF.begin()->size() == 0)
      return;

    auto &BB = *BF.begin();
    const auto &FirstInst = *BB.begin();

    if (!BC.MIA->isTailCall(FirstInst))
      return;

    auto &Op1 = FirstInst.getOperand(0);
    if (!Op1.isExpr())
      return;

    if (auto Expr = dyn_cast<MCSymbolRefExpr>(Op1.getExpr())) {
      auto CalleeIt = FunctionByName.find(Expr->getSymbol().getName());
      if (CalleeIt != FunctionByName.end()) {
        assert(Expr->getSymbol().getName() == CalleeIt->second->getName());
        EquivalentCallTarget[BF.getName()] = CalleeIt->second;
      }
    }
  }

  void optimizeCalls(BinaryFunction &BF,
                     BinaryContext &BC) {
    for (auto BBIt = BF.begin(), BBEnd = BF.end(); BBIt != BBEnd; ++BBIt) {
      for (auto InstIt = BBIt->begin(), InstEnd = BBIt->end();
           InstIt != InstEnd; ++InstIt) {
        auto &Inst = *InstIt;
        if (BC.MIA->isCall(Inst)) {
          auto &Op1 = Inst.getOperand(0);
          if (Op1.isExpr()) {
            if (auto Expr = dyn_cast<MCSymbolRefExpr>(Op1.getExpr())) {
              auto OriginalTarget = Expr->getSymbol().getName();
              errs() << "BOLT-OPT: " << BF.getName() << " calls " << OriginalTarget << "\n";
              auto Target = OriginalTarget;
              while (EquivalentCallTarget.count(Target)) {
                Target = EquivalentCallTarget.find(Target)->second->getName();
              }
              if (Target != OriginalTarget) {
                DEBUG(errs() << "BOLT-DEBUG: Optimizing " << BF.getName()
                             << ": replacing call to "
                             << OriginalTarget
                             << " by call to " << Target << "\n");
                Inst.clear();
                Inst.addOperand(MCOperand::createExpr(
                  MCSymbolRefExpr::create(
                      BC.Ctx->getOrCreateSymbol(Target), *BC.Ctx)));
              }
            }
          }
        }
      }
    }
  }

public:
  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs) override {
    for (auto &It : BFs) {
      FunctionByName[It.second.getName()] = &It.second;
    }
    for (auto &It : BFs) {
      analyze(It.second, BC);
    }
    for (auto &It : BFs) {
      optimizeCalls(It.second, BC);
    }
  }
};

void BinaryFunctionPassManager::runAllPasses(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &Functions) {
  BinaryFunctionPassManager PassManager(BC, Functions);

  PassManager.registerPass(make_unique<OptimizeBodylessFunctions>(),
                           opts::OptimizeBodylessFunctions);

  PassManager.runPasses();
}

} // namespace bolt
} // namespace llvm
