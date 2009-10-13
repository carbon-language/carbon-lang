//===- InlineCost.cpp - Cost analysis for inliner ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements heuristics for inlining decisions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_INLINECOST_H
#define LLVM_ANALYSIS_INLINECOST_H

#include <cassert>
#include <climits>
#include <map>
#include <vector>

namespace llvm {

  class Value;
  class Function;
  class BasicBlock;
  class CallSite;
  template<class PtrType, unsigned SmallSize>
  class SmallPtrSet;

  // CodeMetrics - Calculate size and a few similar metrics for a set of
  // basic blocks.
  struct CodeMetrics {
    /// NeverInline - True if this callee should never be inlined into a
    /// caller.
    bool NeverInline;
    
    /// usesDynamicAlloca - True if this function calls alloca (in the C sense).
    bool usesDynamicAlloca;

    /// NumInsts, NumBlocks - Keep track of how large each function is, which
    /// is used to estimate the code size cost of inlining it.
    unsigned NumInsts, NumBlocks;

    /// NumVectorInsts - Keep track of how many instructions produce vector
    /// values.  The inliner is being more aggressive with inlining vector
    /// kernels.
    unsigned NumVectorInsts;
    
    /// NumRets - Keep track of how many Ret instructions the block contains.
    unsigned NumRets;

    CodeMetrics() : NeverInline(false), usesDynamicAlloca(false), NumInsts(0),
                    NumBlocks(0), NumVectorInsts(0), NumRets(0) {}
    
    /// analyzeBasicBlock - Add information about the specified basic block
    /// to the current structure.
    void analyzeBasicBlock(const BasicBlock *BB);

    /// analyzeFunction - Add information about the specified function
    /// to the current structure.
    void analyzeFunction(Function *F);
  };

  namespace InlineConstants {
    // Various magic constants used to adjust heuristics.
    const int CallPenalty = 5;
    const int LastCallToStaticBonus = -15000;
    const int ColdccPenalty = 2000;
    const int NoreturnPenalty = 10000;
  }

  /// InlineCost - Represent the cost of inlining a function. This
  /// supports special values for functions which should "always" or
  /// "never" be inlined. Otherwise, the cost represents a unitless
  /// amount; smaller values increase the likelyhood of the function
  /// being inlined.
  class InlineCost {
    enum Kind {
      Value,
      Always,
      Never
    };

    // This is a do-it-yourself implementation of
    //   int Cost : 30;
    //   unsigned Type : 2;
    // We used to use bitfields, but they were sometimes miscompiled (PR3822).
    enum { TYPE_BITS = 2 };
    enum { COST_BITS = unsigned(sizeof(unsigned)) * CHAR_BIT - TYPE_BITS };
    unsigned TypedCost; // int Cost : COST_BITS; unsigned Type : TYPE_BITS;

    Kind getType() const {
      return Kind(TypedCost >> COST_BITS);
    }

    int getCost() const {
      // Sign-extend the bottom COST_BITS bits.
      return (int(TypedCost << TYPE_BITS)) >> TYPE_BITS;
    }

    InlineCost(int C, int T) {
      TypedCost = (unsigned(C << TYPE_BITS) >> TYPE_BITS) | (T << COST_BITS);
      assert(getCost() == C && "Cost exceeds InlineCost precision");
    }
  public:
    static InlineCost get(int Cost) { return InlineCost(Cost, Value); }
    static InlineCost getAlways() { return InlineCost(0, Always); }
    static InlineCost getNever() { return InlineCost(0, Never); }

    bool isVariable() const { return getType() == Value; }
    bool isAlways() const { return getType() == Always; }
    bool isNever() const { return getType() == Never; }

    /// getValue() - Return a "variable" inline cost's amount. It is
    /// an error to call this on an "always" or "never" InlineCost.
    int getValue() const {
      assert(getType() == Value && "Invalid access of InlineCost");
      return getCost();
    }
  };
  
  /// InlineCostAnalyzer - Cost analyzer used by inliner.
  class InlineCostAnalyzer {
    struct ArgInfo {
    public:
      unsigned ConstantWeight;
      unsigned AllocaWeight;
      
      ArgInfo(unsigned CWeight, unsigned AWeight)
        : ConstantWeight(CWeight), AllocaWeight(AWeight) {}
    };
    
    struct FunctionInfo {
      CodeMetrics Metrics;

      /// ArgumentWeights - Each formal argument of the function is inspected to
      /// see if it is used in any contexts where making it a constant or alloca
      /// would reduce the code size.  If so, we add some value to the argument
      /// entry here.
      std::vector<ArgInfo> ArgumentWeights;
    
      /// CountCodeReductionForConstant - Figure out an approximation for how
      /// many instructions will be constant folded if the specified value is
      /// constant.
      unsigned CountCodeReductionForConstant(Value *V);
    
      /// CountCodeReductionForAlloca - Figure out an approximation of how much
      /// smaller the function will be if it is inlined into a context where an
      /// argument becomes an alloca.
      ///
      unsigned CountCodeReductionForAlloca(Value *V);

      /// analyzeFunction - Add information about the specified function
      /// to the current structure.
      void analyzeFunction(Function *F);
    };

    std::map<const Function *, FunctionInfo> CachedFunctionInfo;

  public:

    /// getInlineCost - The heuristic used to determine if we should inline the
    /// function call or not.
    ///
    InlineCost getInlineCost(CallSite CS,
                             SmallPtrSet<const Function *, 16> &NeverInline);

    /// getInlineFudgeFactor - Return a > 1.0 factor if the inliner should use a
    /// higher threshold to determine if the function call should be inlined.
    float getInlineFudgeFactor(CallSite CS);

    /// resetCachedFunctionInfo - erase any cached cost info for this function.
    void resetCachedCostInfo(Function* Caller) {
      CachedFunctionInfo[Caller].Metrics.NumBlocks = 0;
    }
  };
}

#endif
