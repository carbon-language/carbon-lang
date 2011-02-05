//===- InlineCost.h - Cost analysis for inliner -----------------*- C++ -*-===//
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
#include <vector>
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ValueMap.h"
#include "llvm/Analysis/CodeMetrics.h"

namespace llvm {

  class Value;
  class Function;
  class BasicBlock;
  class CallSite;
  template<class PtrType, unsigned SmallSize>
  class SmallPtrSet;

  namespace InlineConstants {
    // Various magic constants used to adjust heuristics.
    const int InstrCost = 5;
    const int IndirectCallBonus = -100;
    const int CallPenalty = 25;
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
        : ConstantWeight(CWeight), AllocaWeight(AWeight)
          {}
    };

    struct FunctionInfo {
      CodeMetrics Metrics;

      /// ArgumentWeights - Each formal argument of the function is inspected to
      /// see if it is used in any contexts where making it a constant or alloca
      /// would reduce the code size.  If so, we add some value to the argument
      /// entry here.
      std::vector<ArgInfo> ArgumentWeights;

      /// analyzeFunction - Add information about the specified function
      /// to the current structure.
      void analyzeFunction(Function *F);

      /// NeverInline - Returns true if the function should never be
      /// inlined into any caller.
      bool NeverInline();
    };

    // The Function* for a function can be changed (by ArgumentPromotion);
    // the ValueMap will update itself when this happens.
    ValueMap<const Function *, FunctionInfo> CachedFunctionInfo;

    int CountBonusForConstant(Value *V, Constant *C = NULL);
    int ConstantFunctionBonus(CallSite CS, Constant *C);
    int getInlineSize(CallSite CS, Function *Callee);
    int getInlineBonuses(CallSite CS, Function *Callee);
  public:

    /// getInlineCost - The heuristic used to determine if we should inline the
    /// function call or not.
    ///
    InlineCost getInlineCost(CallSite CS,
                             SmallPtrSet<const Function *, 16> &NeverInline);
    /// getCalledFunction - The heuristic used to determine if we should inline
    /// the function call or not.  The callee is explicitly specified, to allow
    /// you to calculate the cost of inlining a function via a pointer.  The
    /// result assumes that the inlined version will always be used.  You should
    /// weight it yourself in cases where this callee will not always be called.
    InlineCost getInlineCost(CallSite CS,
                             Function *Callee,
                             SmallPtrSet<const Function *, 16> &NeverInline);

    /// getSpecializationBonus - The heuristic used to determine the per-call
    /// performance boost for using a specialization of Callee with argument
    /// SpecializedArgNos replaced by a constant.
    int getSpecializationBonus(Function *Callee,
             SmallVectorImpl<unsigned> &SpecializedArgNo);

    /// getSpecializationCost - The heuristic used to determine the code-size
    /// impact of creating a specialized version of Callee with argument
    /// SpecializedArgNo replaced by a constant.
    InlineCost getSpecializationCost(Function *Callee,
               SmallVectorImpl<unsigned> &SpecializedArgNo);

    /// getInlineFudgeFactor - Return a > 1.0 factor if the inliner should use a
    /// higher threshold to determine if the function call should be inlined.
    float getInlineFudgeFactor(CallSite CS);

    /// resetCachedFunctionInfo - erase any cached cost info for this function.
    void resetCachedCostInfo(Function* Caller) {
      CachedFunctionInfo[Caller] = FunctionInfo();
    }

    /// growCachedCostInfo - update the cached cost info for Caller after Callee
    /// has been inlined. If Callee is NULL it means a dead call has been
    /// eliminated.
    void growCachedCostInfo(Function* Caller, Function* Callee);

    /// clear - empty the cache of inline costs
    void clear();
  };

  /// callIsSmall - If a call is likely to lower to a single target instruction,
  /// or is otherwise deemed small return true.
  bool callIsSmall(const Function *Callee);
}

#endif
