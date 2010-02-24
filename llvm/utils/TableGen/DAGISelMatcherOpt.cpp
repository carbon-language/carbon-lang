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


static void FormRecordChildNodes(OwningPtr<MatcherNode> &Matcher) {
  // If we reached the end of the chain, we're done.
  MatcherNode *N = Matcher.get();
  if (N == 0) return;
  
  // If we have a push node, walk down both edges.
  if (PushMatcherNode *Push = dyn_cast<PushMatcherNode>(N))
    FormRecordChildNodes(Push->getFailurePtr());
  
  // If we found a movechild node, check to see if our pattern matches.
  if (MoveChildMatcherNode *MC = dyn_cast<MoveChildMatcherNode>(N)) {
    if (RecordMatcherNode *RM = dyn_cast<RecordMatcherNode>(MC->getNext()))
      if (MoveParentMatcherNode *MP = 
                 dyn_cast<MoveParentMatcherNode>(RM->getNext())) {
        MatcherNode *New
          = new RecordChildMatcherNode(MC->getChildNo(), RM->getWhatFor());
        New->setNext(MP->takeNext());
        Matcher.reset(New);
        return FormRecordChildNodes(Matcher);
      }
  }

  FormRecordChildNodes(N->getNextPtr());
}


MatcherNode *llvm::OptimizeMatcher(MatcherNode *Matcher) {
  OwningPtr<MatcherNode> MatcherPtr(Matcher);
  FormRecordChildNodes(MatcherPtr);
  return MatcherPtr.take();
}
