//===-- Analysis/Writer.cpp - Printing routines for analyses -----*- C++ -*--=//
//
// This library file implements analysis result printing support for 
// llvm/Analysis/Writer.h
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Writer.h"
#include "llvm/Analysis/Interval.h"
#include "llvm/Analysis/Dominators.h"
#include <iterator>
#include <algorithm>

void cfg::WriteToOutput(const Interval *I, ostream &o) {
  o << "-------------------------------------------------------------\n"
       << "Interval Contents:\n";
  
  // Print out all of the basic blocks in the interval...
  copy(I->Nodes.begin(), I->Nodes.end(), 
       ostream_iterator<BasicBlock*>(o, "\n"));

  o << "Interval Predecessors:\n";
  copy(I->Predecessors.begin(), I->Predecessors.end(), 
       ostream_iterator<BasicBlock*>(o, "\n"));
  
  o << "Interval Successors:\n";
  copy(I->Successors.begin(), I->Successors.end(), 
       ostream_iterator<BasicBlock*>(o, "\n"));
}

ostream &operator<<(ostream &o, const set<const BasicBlock*> &BBs) {
  copy(BBs.begin(), BBs.end(), ostream_iterator<const BasicBlock*>(o, "\n"));
  return o;
}

void cfg::WriteToOutput(const DominatorSet &DS, ostream &o) {
  for (DominatorSet::const_iterator I = DS.begin(), E = DS.end(); I != E; ++I) {
    o << "=============================--------------------------------\n"
      << "\nDominator Set For Basic Block\n" << I->first
      << "-------------------------------\n" << I->second << endl;
  }
}


void cfg::WriteToOutput(const ImmediateDominators &ID, ostream &o) {
  for (ImmediateDominators::const_iterator I = ID.begin(), E = ID.end();
       I != E; ++I) {
    o << "=============================--------------------------------\n"
      << "\nImmediate Dominator For Basic Block\n" << I->first
      << "is: \n" << I->second << endl;
  }
}


static ostream &operator<<(ostream &o, const cfg::DominatorTree::Node *Node) {
  return o << Node->getNode() << "\n------------------------------------------\n";
	   
}

static void PrintDomTree(const cfg::DominatorTree::Node *N, ostream &o,
			 unsigned Lev) {
  o << "Level #" << Lev << ":  " << N;
  for (cfg::DominatorTree::Node::const_iterator I = N->begin(), E = N->end(); 
       I != E; ++I) {
    PrintDomTree(*I, o, Lev+1);
  }
}

void cfg::WriteToOutput(const DominatorTree &DT, ostream &o) {
  o << "=============================--------------------------------\n"
    << "Inorder Dominator Tree:\n";
  PrintDomTree(DT[DT.getRoot()], o, 1);
}

void cfg::WriteToOutput(const DominanceFrontier &DF, ostream &o) {
  for (DominanceFrontier::const_iterator I = DF.begin(), E = DF.end();
       I != E; ++I) {
    o << "=============================--------------------------------\n"
      << "\nDominance Frontier For Basic Block\n" << I->first
      << "is: \n" << I->second << endl;
  }
}

