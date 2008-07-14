//===- InlineCost.cpp - Cost analysis for inliner ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements bottom-up inlining of functions into callees.
//
//===----------------------------------------------------------------------===//

#ifndef INLINECOST_H
#define INLINECOST_H

#include "llvm/ADT/SmallPtrSet.h"
#include <map>
#include <vector>

namespace llvm {

  class Value;
  class Function;
  class CallSite;

  /// InlineCostAnalyzer - Cost analyzer used by inliner.
  class InlineCostAnalyzer {
    struct ArgInfo {
    public:
      unsigned ConstantWeight;
      unsigned AllocaWeight;
      
      ArgInfo(unsigned CWeight, unsigned AWeight)
        : ConstantWeight(CWeight), AllocaWeight(AWeight) {}
    };
    
    // FunctionInfo - For each function, calculate the size of it in blocks and
    // instructions.
    struct FunctionInfo {
      /// NeverInline - True if this callee should never be inlined into a
      /// caller.
      bool NeverInline;
      
      /// NumInsts, NumBlocks - Keep track of how large each function is, which
      /// is used to estimate the code size cost of inlining it.
      unsigned NumInsts, NumBlocks;

      /// NumVectorInsts - Keep track how many instrctions produce vector
      /// values.  The inliner is being more aggressive with inlining vector
      /// kernels.
      unsigned NumVectorInsts;
      
      /// ArgumentWeights - Each formal argument of the function is inspected to
      /// see if it is used in any contexts where making it a constant or alloca
      /// would reduce the code size.  If so, we add some value to the argument
      /// entry here.
      std::vector<ArgInfo> ArgumentWeights;
      
      FunctionInfo() : NeverInline(false), NumInsts(0), NumBlocks(0),
                       NumVectorInsts(0) {}
      
      /// analyzeFunction - Fill in the current structure with information
      /// gleaned from the specified function.
      void analyzeFunction(Function *F);

      /// CountCodeReductionForConstant - Figure out an approximation for how
      /// many instructions will be constant folded if the specified value is
      /// constant.
      unsigned CountCodeReductionForConstant(Value *V);
      
      /// CountCodeReductionForAlloca - Figure out an approximation of how much
      /// smaller the function will be if it is inlined into a context where an
      /// argument becomes an alloca.
      ///
      unsigned CountCodeReductionForAlloca(Value *V);
    };

    std::map<const Function *, FunctionInfo> CachedFunctionInfo;

  public:

    /// getInlineCost - The heuristic used to determine if we should inline the
    /// function call or not.
    ///
    int getInlineCost(CallSite CS,
                      SmallPtrSet<const Function *, 16> &NeverInline);

    /// getInlineFudgeFactor - Return a > 1.0 factor if the inliner should use a
    /// higher threshold to determine if the function call should be inlined.
    float getInlineFudgeFactor(CallSite CS);
  };
}

#endif
