//===------ CodeGeneration.cpp - Code generate the Scops using ISL. ----======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The CodeGeneration pass takes a Scop created by ScopInfo and translates it
// back to LLVM-IR using the ISL code generator.
//
// The Scop describes the high level memory behaviour of a control flow region.
// Transformation passes can update the schedule (execution order) of statements
// in the Scop. ISL is used to generate an abstract syntax tree that reflects
// the updated execution order. This clast is used to create new LLVM-IR that is
// computationally equivalent to the original control flow region, but executes
// its code in the new execution order defined by the changed schedule.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/IslNodeBuilder.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/CodeGen/Utils.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"
#include "polly/Support/ScopHelper.h"
#include "polly/TempScopInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"

using namespace polly;
using namespace llvm;

#define DEBUG_TYPE "polly-codegen"

namespace {
class CodeGeneration : public ScopPass {
public:
  static char ID;

  CodeGeneration() : ScopPass(ID) {}

  /// @brief The datalayout used
  const DataLayout *DL;

  /// @name The analysis passes we need to generate code.
  ///
  ///{
  LoopInfo *LI;
  IslAstInfo *AI;
  DominatorTree *DT;
  ScalarEvolution *SE;
  ///}

  /// @brief The loop annotator to generate llvm.loop metadata.
  ScopAnnotator Annotator;

  /// @brief Build the runtime condition.
  ///
  /// Build the condition that evaluates at run-time to true iff all
  /// assumptions taken for the SCoP hold, and to false otherwise.
  ///
  /// @return A value evaluating to true/false if execution is save/unsafe.
  Value *buildRTC(PollyIRBuilder &Builder, IslExprBuilder &ExprBuilder) {
    Builder.SetInsertPoint(Builder.GetInsertBlock()->getTerminator());
    Value *RTC = ExprBuilder.create(AI->getRunCondition());
    if (!RTC->getType()->isIntegerTy(1))
      RTC = Builder.CreateIsNotNull(RTC);
    return RTC;
  }

  bool verifyGeneratedFunction(Scop &S, Function &F) {
    if (!verifyFunction(F))
      return false;

    DEBUG({
      errs() << "== ISL Codegen created an invalid function ==\n\n== The "
                "SCoP ==\n";
      S.print(errs());
      errs() << "\n== The isl AST ==\n";
      AI->printScop(errs(), S);
      errs() << "\n== The invalid function ==\n";
      F.print(errs());
      errs() << "\n== The errors ==\n";
      verifyFunction(F, &errs());
    });

    return true;
  }

  bool runOnScop(Scop &S) override {
    AI = &getAnalysis<IslAstInfo>();

    // Check if we created an isl_ast root node, otherwise exit.
    isl_ast_node *AstRoot = AI->getAst();
    if (!AstRoot)
      return false;

    LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    SE = &getAnalysis<ScalarEvolution>();
    DL = &S.getRegion().getEntry()->getParent()->getParent()->getDataLayout();

    assert(!S.getRegion().isTopLevelRegion() &&
           "Top level regions are not supported");

    Annotator.buildAliasScopes(S);

    BasicBlock *EnteringBB = simplifyRegion(&S, this);
    PollyIRBuilder Builder = createPollyIRBuilder(EnteringBB, Annotator);

    IslNodeBuilder NodeBuilder(Builder, Annotator, this, *DL, *LI, *SE, *DT, S);

    // Only build the run-time condition and parameters _after_ having
    // introduced the conditional branch. This is important as the conditional
    // branch will guard the original scop from new induction variables that
    // the SCEVExpander may introduce while code generating the parameters and
    // which may introduce scalar dependences that prevent us from correctly
    // code generating this scop.
    BasicBlock *StartBlock =
        executeScopConditionally(S, this, Builder.getTrue());
    auto SplitBlock = StartBlock->getSinglePredecessor();
    Builder.SetInsertPoint(SplitBlock->getTerminator());
    NodeBuilder.addParameters(S.getContext());
    Value *RTC = buildRTC(Builder, NodeBuilder.getExprBuilder());
    SplitBlock->getTerminator()->setOperand(0, RTC);
    Builder.SetInsertPoint(StartBlock->begin());

    NodeBuilder.create(AstRoot);

    assert(!verifyGeneratedFunction(S, *EnteringBB->getParent()) &&
           "Verification of generated function failed");
    return true;
  }

  void printScop(raw_ostream &, Scop &) const override {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<IslAstInfo>();
    AU.addRequired<RegionInfoPass>();
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<ScopDetection>();
    AU.addRequired<ScopInfo>();
    AU.addRequired<LoopInfoWrapperPass>();

    AU.addPreserved<DependenceInfo>();

    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<IslAstInfo>();
    AU.addPreserved<ScopDetection>();
    AU.addPreserved<ScalarEvolution>();

    // FIXME: We do not yet add regions for the newly generated code to the
    //        region tree.
    AU.addPreserved<RegionInfoPass>();
    AU.addPreserved<TempScopInfo>();
    AU.addPreserved<ScopInfo>();
    AU.addPreservedID(IndependentBlocksID);
  }
};
}

char CodeGeneration::ID = 1;

Pass *polly::createCodeGenerationPass() { return new CodeGeneration(); }

INITIALIZE_PASS_BEGIN(CodeGeneration, "polly-codegen",
                      "Polly - Create LLVM-IR from SCoPs", false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution);
INITIALIZE_PASS_DEPENDENCY(ScopDetection);
INITIALIZE_PASS_END(CodeGeneration, "polly-codegen",
                    "Polly - Create LLVM-IR from SCoPs", false, false)
