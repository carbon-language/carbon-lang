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

#include "llvm/Function.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/ValueMap.h"
#include "llvm/Analysis/CodeMetrics.h"
#include <cassert>
#include <climits>
#include <vector>

namespace llvm {

  class CallSite;
  class TargetData;

  namespace InlineConstants {
    // Various magic constants used to adjust heuristics.
    const int InstrCost = 5;
    const int IndirectCallThreshold = 100;
    const int CallPenalty = 25;
    const int LastCallToStaticBonus = -15000;
    const int ColdccPenalty = 2000;
    const int NoreturnPenalty = 10000;
  }

  /// \brief Represents the cost of inlining a function.
  ///
  /// This supports special values for functions which should "always" or
  /// "never" be inlined. Otherwise, the cost represents a unitless amount;
  /// smaller values increase the likelihood of the function being inlined.
  ///
  /// Objects of this type also provide the adjusted threshold for inlining
  /// based on the information available for a particular callsite. They can be
  /// directly tested to determine if inlining should occur given the cost and
  /// threshold for this cost metric.
  class InlineCost {
    enum CostKind {
      CK_Variable,
      CK_Always,
      CK_Never
    };

    const int      Cost : 30; // The inlining cost if neither always nor never.
    const unsigned Kind : 2;  // The type of cost, one of CostKind above.

    /// \brief The adjusted threshold against which this cost should be tested.
    const int Threshold;

    // Trivial constructor, interesting logic in the factory functions below.
    InlineCost(int Cost, CostKind Kind, int Threshold)
      : Cost(Cost), Kind(Kind), Threshold(Threshold) {}

  public:
    static InlineCost get(int Cost, int Threshold) {
      InlineCost Result(Cost, CK_Variable, Threshold);
      assert(Result.Cost == Cost && "Cost exceeds InlineCost precision");
      return Result;
    }
    static InlineCost getAlways() {
      return InlineCost(0, CK_Always, 0);
    }
    static InlineCost getNever() {
      return InlineCost(0, CK_Never, 0);
    }

    /// \brief Test whether the inline cost is low enough for inlining.
    operator bool() const {
      if (isAlways()) return true;
      if (isNever()) return false;
      return Cost < Threshold;
    }

    bool isVariable() const { return Kind == CK_Variable; }
    bool isAlways() const   { return Kind == CK_Always; }
    bool isNever() const    { return Kind == CK_Never; }

    /// getCost() - Return a "variable" inline cost's amount. It is
    /// an error to call this on an "always" or "never" InlineCost.
    int getCost() const {
      assert(Kind == CK_Variable && "Invalid access of InlineCost");
      return Cost;
    }

    /// \brief Get the cost delta from the threshold for inlining.
    /// Only valid if the cost is of the variable kind. Returns a negative
    /// value if the cost is too high to inline.
    int getCostDelta() const {
      return Threshold - getCost();
    }
  };

  /// InlineCostAnalyzer - Cost analyzer used by inliner.
  class InlineCostAnalyzer {
    // TargetData if available, or null.
    const TargetData *TD;

  public:
    InlineCostAnalyzer(): TD(0) {}

    void setTargetData(const TargetData *TData) { TD = TData; }

    /// \brief Get an InlineCost object representing the cost of inlining this
    /// callsite.
    ///
    /// Note that threshold is passed into this function. Only costs below the
    /// threshold are computed with any accuracy. The threshold can be used to
    /// bound the computation necessary to determine whether the cost is
    /// sufficiently low to warrant inlining.
    InlineCost getInlineCost(CallSite CS, int Threshold);

    /// resetCachedFunctionInfo - erase any cached cost info for this function.
    void resetCachedCostInfo(Function* Caller) {
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
