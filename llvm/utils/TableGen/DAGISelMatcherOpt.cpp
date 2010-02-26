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
#include "llvm/ADT/DenseMap.h"
#include <vector>
using namespace llvm;

static void ContractNodes(OwningPtr<Matcher> &MatcherPtr) {
  // If we reached the end of the chain, we're done.
  Matcher *N = MatcherPtr.get();
  if (N == 0) return;
  
  // If we have a scope node, walk down all of the children.
  if (ScopeMatcher *Scope = dyn_cast<ScopeMatcher>(N)) {
    for (unsigned i = 0, e = Scope->getNumChildren(); i != e; ++i) {
      OwningPtr<Matcher> Child(Scope->takeChild(i));
      ContractNodes(Child);
      Scope->resetChild(i, Child.take());
    }
    return;
  }
  
  // If we found a movechild node with a node that comes in a 'foochild' form,
  // transform it.
  if (MoveChildMatcher *MC = dyn_cast<MoveChildMatcher>(N)) {
    Matcher *New = 0;
    if (RecordMatcher *RM = dyn_cast<RecordMatcher>(MC->getNext()))
      New = new RecordChildMatcher(MC->getChildNo(), RM->getWhatFor());
    
    if (CheckTypeMatcher *CT= dyn_cast<CheckTypeMatcher>(MC->getNext()))
      New = new CheckChildTypeMatcher(MC->getChildNo(), CT->getType());
    
    if (New) {
      // Insert the new node.
      New->setNext(MatcherPtr.take());
      MatcherPtr.reset(New);
      // Remove the old one.
      MC->setNext(MC->getNext()->takeNext());
      return ContractNodes(MatcherPtr);
    }
  }
  
  if (MoveChildMatcher *MC = dyn_cast<MoveChildMatcher>(N))
    if (MoveParentMatcher *MP = 
          dyn_cast<MoveParentMatcher>(MC->getNext())) {
      MatcherPtr.reset(MP->takeNext());
      return ContractNodes(MatcherPtr);
    }
  
  ContractNodes(N->getNextPtr());
}

static void FactorNodes(OwningPtr<Matcher> &MatcherPtr) {
  // If we reached the end of the chain, we're done.
  Matcher *N = MatcherPtr.get();
  if (N == 0) return;
  
  // If this is not a push node, just scan for one.
  ScopeMatcher *Scope = dyn_cast<ScopeMatcher>(N);
  if (Scope == 0)
    return FactorNodes(N->getNextPtr());
  
  // Okay, pull together the children of the scope node into a vector so we can
  // inspect it more easily.  While we're at it, bucket them up by the hash
  // code of their first predicate.
  SmallVector<Matcher*, 32> OptionsToMatch;
  
  for (unsigned i = 0, e = Scope->getNumChildren(); i != e; ++i) {
    // Factor the subexpression.
    OwningPtr<Matcher> Child(Scope->takeChild(i));
    FactorNodes(Child);
    
    if (Matcher *N = Child.take())
      OptionsToMatch.push_back(N);
  }
  
  SmallVector<Matcher*, 32> NewOptionsToMatch;

  // Loop over options to match, merging neighboring patterns with identical
  // starting nodes into a shared matcher.
  for (unsigned i = 0, e = OptionsToMatch.size(); i != e;) {
    // Find the set of matchers that start with this node.
    Matcher *Optn = OptionsToMatch[i++];
 
    // See if the next option starts with the same matcher, if not, no sharing.
    if (i == e || !OptionsToMatch[i]->isEqual(Optn)) {
      // TODO: Skip over mutually exclusive patterns.
      NewOptionsToMatch.push_back(Optn);
      continue;
    }
    
    // If the two neighbors *do* start with the same matcher, we can factor the
    // matcher out of at least these two patterns.  See what the maximal set we
    // can merge together is.
    SmallVector<Matcher*, 8> EqualMatchers;
    EqualMatchers.push_back(Optn);
    EqualMatchers.push_back(OptionsToMatch[i++]);
    
    while (i != e && OptionsToMatch[i]->isEqual(Optn))
      EqualMatchers.push_back(OptionsToMatch[i++]);
    
    // Factor these checks by pulling the first node off each entry and
    // discarding it.  Take the first one off the first entry to reuse.
    Matcher *Shared = Optn;
    Optn = Optn->takeNext();
    EqualMatchers[0] = Optn;

    // Remove and delete the first node from the other matchers we're factoring.
    for (unsigned i = 1, e = EqualMatchers.size(); i != e; ++i) {
      Matcher *Tmp = EqualMatchers[i]->takeNext();
      delete EqualMatchers[i];
      EqualMatchers[i] = Tmp;
    }
    
    Shared->setNext(new ScopeMatcher(&EqualMatchers[0], EqualMatchers.size()));

    // Recursively factor the newly created node.
    FactorNodes(Shared->getNextPtr());
    
    NewOptionsToMatch.push_back(Shared);
  }

  // Reassemble a new Scope node.
  assert(!NewOptionsToMatch.empty() && "where'd all our children go?");
  if (NewOptionsToMatch.size() == 1)
    MatcherPtr.reset(NewOptionsToMatch[0]);
  else {
    Scope->setNumChildren(NewOptionsToMatch.size());
    for (unsigned i = 0, e = NewOptionsToMatch.size(); i != e; ++i)
      Scope->resetChild(i, NewOptionsToMatch[i]);
  }
}

Matcher *llvm::OptimizeMatcher(Matcher *TheMatcher) {
  OwningPtr<Matcher> MatcherPtr(TheMatcher);
  ContractNodes(MatcherPtr);
  FactorNodes(MatcherPtr);
  return MatcherPtr.take();
}
