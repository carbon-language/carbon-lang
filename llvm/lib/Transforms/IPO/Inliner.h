//===- InlineCommon.h - Code common to all inliners -------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a simple policy-based bottom-up inliner.  This file
// implements all of the boring mechanics of the bottom-up inlining, while the
// subclass determines WHAT to inline, which is the much more interesting
// component.
//
//===----------------------------------------------------------------------===//

#ifndef INLINER_H
#define INLINER_H

#define DEBUG_TYPE "inline"
#include "llvm/CallGraphSCCPass.h"
#include <set>
class CallSite;

/// Inliner - This class contains all of the helper code which is used to
/// perform the inlining operations that does not depend on the policy.
///
struct Inliner : public CallGraphSCCPass {
  Inliner();

  // Main run interface method, this implements the interface required by the
  // Pass class.
  virtual bool runOnSCC(const std::vector<CallGraphNode *> &SCC);

  /// This method returns the value specified by the -inline-threshold value,
  /// specified on the command line.  This is typically not directly needed.
  ///
  unsigned getInlineThreshold() const { return InlineThreshold; }

  /// getInlineCost - This method must be implemented by the subclass to
  /// determine the cost of inlining the specified call site.  If the cost
  /// returned is greater than the current inline threshold, the call site is
  /// not inlined.
  ///
  virtual int getInlineCost(CallSite CS) = 0;
  
  /// getRecursiveInlineCost - This method can be implemented by subclasses if
  /// it wants to treat calls to functions within the current SCC specially.  If
  /// this method is not overloaded, it just chains to getInlineCost().
  ///
  virtual int getRecursiveInlineCost(CallSite CS);

private:
  unsigned InlineThreshold;
  bool performInlining(CallSite CS, std::set<Function*> &SCC);
};


#endif
