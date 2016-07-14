//===------ PPCGCodeGeneration.cpp - Polly Accelerator Code Generation. ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Take a scop created by ScopInfo and map it to GPU code using the ppcg
// GPU mapping strategy.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/IslNodeBuilder.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"

extern "C" {
#include "gpu.h"
#include "ppcg.h"
}

#include "llvm/Support/Debug.h"

using namespace polly;
using namespace llvm;

#define DEBUG_TYPE "polly-codegen-ppcg"

namespace {
class PPCGCodeGeneration : public ScopPass {
public:
  static char ID;

  /// The scop that is currently processed.
  Scop *S;

  PPCGCodeGeneration() : ScopPass(ID) {}

  /// Construct compilation options for PPCG.
  ///
  /// @returns The compilation options.
  ppcg_options *createPPCGOptions() {
    auto DebugOptions =
        (ppcg_debug_options *)malloc(sizeof(ppcg_debug_options));
    auto Options = (ppcg_options *)malloc(sizeof(ppcg_options));

    DebugOptions->dump_schedule_constraints = false;
    DebugOptions->dump_schedule = false;
    DebugOptions->dump_final_schedule = false;
    DebugOptions->dump_sizes = false;

    Options->debug = DebugOptions;

    Options->reschedule = true;
    Options->scale_tile_loops = false;
    Options->wrap = false;

    Options->non_negative_parameters = false;
    Options->ctx = nullptr;
    Options->sizes = nullptr;

    Options->use_private_memory = false;
    Options->use_shared_memory = false;
    Options->max_shared_memory = 0;

    Options->target = PPCG_TARGET_CUDA;
    Options->openmp = false;
    Options->linearize_device_arrays = true;
    Options->live_range_reordering = false;

    Options->opencl_compiler_options = nullptr;
    Options->opencl_use_gpu = false;
    Options->opencl_n_include_file = 0;
    Options->opencl_include_files = nullptr;
    Options->opencl_print_kernel_types = false;
    Options->opencl_embed_kernel_code = false;

    Options->save_schedule_file = nullptr;
    Options->load_schedule_file = nullptr;

    return Options;
  }

  /// Create a new PPCG scop from the current scop.
  ///
  /// For now the created scop is initialized to 'zero' and does not contain
  /// any scop-specific information.
  ///
  /// @returns A new ppcg scop.
  ppcg_scop *createPPCGScop() {
    auto PPCGScop = (ppcg_scop *)malloc(sizeof(ppcg_scop));

    PPCGScop->options = createPPCGOptions();

    PPCGScop->start = 0;
    PPCGScop->end = 0;

    PPCGScop->context = nullptr;
    PPCGScop->domain = nullptr;
    PPCGScop->call = nullptr;
    PPCGScop->tagged_reads = nullptr;
    PPCGScop->reads = nullptr;
    PPCGScop->live_in = nullptr;
    PPCGScop->tagged_may_writes = nullptr;
    PPCGScop->may_writes = nullptr;
    PPCGScop->tagged_must_writes = nullptr;
    PPCGScop->must_writes = nullptr;
    PPCGScop->live_out = nullptr;
    PPCGScop->tagged_must_kills = nullptr;
    PPCGScop->tagger = nullptr;

    PPCGScop->independence = nullptr;
    PPCGScop->dep_flow = nullptr;
    PPCGScop->tagged_dep_flow = nullptr;
    PPCGScop->dep_false = nullptr;
    PPCGScop->dep_forced = nullptr;
    PPCGScop->dep_order = nullptr;
    PPCGScop->tagged_dep_order = nullptr;

    PPCGScop->schedule = nullptr;
    PPCGScop->names = nullptr;

    PPCGScop->pet = nullptr;

    return PPCGScop;
  }

  /// Create a default-initialized PPCG GPU program.
  ///
  /// @returns A new gpu grogram description.
  gpu_prog *createPPCGProg(ppcg_scop *PPCGScop) {

    if (!PPCGScop)
      return nullptr;

    auto PPCGProg = isl_calloc_type(S->getIslCtx(), struct gpu_prog);

    PPCGProg->ctx = S->getIslCtx();
    PPCGProg->scop = PPCGScop;
    PPCGProg->context = nullptr;
    PPCGProg->read = nullptr;
    PPCGProg->may_write = nullptr;
    PPCGProg->must_write = nullptr;
    PPCGProg->tagged_must_kill = nullptr;
    PPCGProg->may_persist = nullptr;
    PPCGProg->to_outer = nullptr;
    PPCGProg->to_inner = nullptr;
    PPCGProg->any_to_outer = nullptr;
    PPCGProg->array_order = nullptr;
    PPCGProg->n_stmts = 0;
    PPCGProg->stmts = nullptr;
    PPCGProg->n_array = 0;
    PPCGProg->array = nullptr;

    return PPCGProg;
  }

  bool runOnScop(Scop &CurrentScop) override {
    S = &CurrentScop;

    auto PPCGScop = createPPCGScop();
    auto PPCGProg = createPPCGProg(PPCGScop);
    gpu_prog_free(PPCGProg);
    ppcg_scop_free(PPCGScop);

    return true;
  }

  void printScop(raw_ostream &, Scop &) const override {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<RegionInfoPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<ScopDetection>();
    AU.addRequired<ScopInfoRegionPass>();
    AU.addRequired<LoopInfoWrapperPass>();

    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<BasicAAWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<PostDominatorTreeWrapperPass>();
    AU.addPreserved<ScopDetection>();
    AU.addPreserved<ScalarEvolutionWrapperPass>();
    AU.addPreserved<SCEVAAWrapperPass>();

    // FIXME: We do not yet add regions for the newly generated code to the
    //        region tree.
    AU.addPreserved<RegionInfoPass>();
    AU.addPreserved<ScopInfoRegionPass>();
  }
};
}

char PPCGCodeGeneration::ID = 1;

Pass *polly::createPPCGCodeGenerationPass() { return new PPCGCodeGeneration(); }

INITIALIZE_PASS_BEGIN(PPCGCodeGeneration, "polly-codegen-ppcg",
                      "Polly - Apply PPCG translation to SCOP", false, false)
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass);
INITIALIZE_PASS_DEPENDENCY(ScopDetection);
INITIALIZE_PASS_END(PPCGCodeGeneration, "polly-codegen-ppcg",
                    "Polly - Apply PPCG translation to SCOP", false, false)
