//===- DAGISelMatcherOpt.cpp - Optimize a DAG Matcher ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DAG Matcher optimizer.
//
//===----------------------------------------------------------------------===//

#include "DAGISelMatcher.h"
using namespace llvm;

static void ContractNodes(OwningPtr<MatcherNode> &Matcher) {
  // If we reached the end of the chain, we're done.
  MatcherNode *N = Matcher.get();
  if (N == 0) return;
  
  // If we have a push node, walk down both edges.
  if (PushMatcherNode *Push = dyn_cast<PushMatcherNode>(N))
    ContractNodes(Push->getFailurePtr());
  
  // If we found a movechild node with a node that comes in a 'foochild' form,
  // transform it.
  if (MoveChildMatcherNode *MC = dyn_cast<MoveChildMatcherNode>(N)) {
    MatcherNode *New = 0;
    if (RecordMatcherNode *RM = dyn_cast<RecordMatcherNode>(MC->getNext()))
      New = new RecordChildMatcherNode(MC->getChildNo(), RM->getWhatFor());
    
    if (CheckTypeMatcherNode *CT= dyn_cast<CheckTypeMatcherNode>(MC->getNext()))
      New = new CheckChildTypeMatcherNode(MC->getChildNo(), CT->getType());
    
    if (New) {
      // Insert the new node.
      New->setNext(Matcher.take());
      Matcher.reset(New);
      // Remove the old one.
      MC->setNext(MC->getNext()->takeNext());
      return ContractNodes(Matcher);
    }
  }
  
  if (MoveChildMatcherNode *MC = dyn_cast<MoveChildMatcherNode>(N))
    if (MoveParentMatcherNode *MP = 
          dyn_cast<MoveParentMatcherNode>(MC->getNext())) {
      Matcher.reset(MP->takeNext());
      return ContractNodes(Matcher);
    }
  
  ContractNodes(N->getNextPtr());
}


MatcherNode *llvm::OptimizeMatcher(MatcherNode *Matcher) {
  OwningPtr<MatcherNode> MatcherPtr(Matcher);
  ContractNodes(MatcherPtr);
  return MatcherPtr.take();
}
