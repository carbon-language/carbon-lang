//===-- Analysis/Writer.cpp - Printing routines for analyses -----*- C++ -*--=//
//
// This library file implements analysis result printing support for 
// llvm/Analysis/Writer.h
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Writer.h"
#include "llvm/Analysis/IntervalPartition.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/InductionVariable.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Module.h"
#include <iterator>
#include <algorithm>
#include <string>
#include <iostream>
using std::ostream;
using std::set;
using std::vector;
using std::string;

//===----------------------------------------------------------------------===//
//  Interval Printing Routines
//===----------------------------------------------------------------------===//

void WriteToOutput(const Interval *I, ostream &o) {
  o << "-------------------------------------------------------------\n"
       << "Interval Contents:\n";
  
  // Print out all of the basic blocks in the interval...
  copy(I->Nodes.begin(), I->Nodes.end(), 
       std::ostream_iterator<BasicBlock*>(o, "\n"));

  o << "Interval Predecessors:\n";
  copy(I->Predecessors.begin(), I->Predecessors.end(), 
       std::ostream_iterator<BasicBlock*>(o, "\n"));
  
  o << "Interval Successors:\n";
  copy(I->Successors.begin(), I->Successors.end(), 
       std::ostream_iterator<BasicBlock*>(o, "\n"));
}

void WriteToOutput(const IntervalPartition &IP, ostream &o) {
  copy(IP.begin(), IP.end(), std::ostream_iterator<const Interval *>(o, "\n"));
}



//===----------------------------------------------------------------------===//
//  Dominator Printing Routines
//===----------------------------------------------------------------------===//

ostream &operator<<(ostream &o, const set<BasicBlock*> &BBs) {
  for (set<BasicBlock*>::const_iterator I = BBs.begin(), E = BBs.end();
       I != E; ++I) {
    o << "  ";
    WriteAsOperand(o, (Value*)*I, false);
    o << "\n";
   }
  return o;
}

void WriteToOutput(const DominatorSet &DS, ostream &o) {
  for (DominatorSet::const_iterator I = DS.begin(), E = DS.end(); I != E; ++I) {
    o << "=============================--------------------------------\n"
      << "\nDominator Set For Basic Block\n" << I->first
      << "-------------------------------\n" << I->second << "\n";
  }
}


void WriteToOutput(const ImmediateDominators &ID, ostream &o) {
  for (ImmediateDominators::const_iterator I = ID.begin(), E = ID.end();
       I != E; ++I) {
    o << "=============================--------------------------------\n"
      << "\nImmediate Dominator For Basic Block\n" << *I->first
      << "is: \n" << *I->second << "\n";
  }
}


static ostream &operator<<(ostream &o, const DominatorTree::Node *Node) {
  return o << Node->getNode() << "\n------------------------------------------\n";
	   
}

static void PrintDomTree(const DominatorTree::Node *N, ostream &o,
                         unsigned Lev) {
  o << "Level #" << Lev << ":  " << N;
  for (DominatorTree::Node::const_iterator I = N->begin(), E = N->end(); 
       I != E; ++I) {
    PrintDomTree(*I, o, Lev+1);
  }
}

void WriteToOutput(const DominatorTree &DT, ostream &o) {
  o << "=============================--------------------------------\n"
    << "Inorder Dominator Tree:\n";
  PrintDomTree(DT[DT.getRoot()], o, 1);
}

void WriteToOutput(const DominanceFrontier &DF, ostream &o) {
  for (DominanceFrontier::const_iterator I = DF.begin(), E = DF.end();
       I != E; ++I) {
    o << "=============================--------------------------------\n"
      << "\nDominance Frontier For Basic Block\n";
    WriteAsOperand(o, (Value*)I->first, false);
    o << " is: \n" << I->second << "\n";
  }
}


//===----------------------------------------------------------------------===//
//  Loop Printing Routines
//===----------------------------------------------------------------------===//

void WriteToOutput(const Loop *L, ostream &o) {
  o << string(L->getLoopDepth()*2, ' ') << "Loop Containing: ";

  for (unsigned i = 0; i < L->getBlocks().size(); ++i) {
    if (i) o << ",";
    WriteAsOperand(o, (const Value*)L->getBlocks()[i]);
  }
  o << "\n";

  copy(L->getSubLoops().begin(), L->getSubLoops().end(),
       std::ostream_iterator<const Loop*>(o, "\n"));
}

void WriteToOutput(const LoopInfo &LI, ostream &o) {
  copy(LI.getTopLevelLoops().begin(), LI.getTopLevelLoops().end(),
       std::ostream_iterator<const Loop*>(o, "\n"));
}



//===----------------------------------------------------------------------===//
//  Induction Variable Printing Routines
//===----------------------------------------------------------------------===//

void WriteToOutput(const InductionVariable &IV, ostream &o) {
  switch (IV.InductionType) {
  case InductionVariable::Cannonical:   o << "Cannonical ";   break;
  case InductionVariable::SimpleLinear: o << "SimpleLinear "; break;
  case InductionVariable::Linear:       o << "Linear ";       break;
  case InductionVariable::Unknown:      o << "Unrecognized "; break;
  }
  o << "Induction Variable";
  if (IV.Phi) {
    WriteAsOperand(o, (const Value*)IV.Phi);
    o << ":\n" << (const Value*)IV.Phi;
  } else {
    o << "\n";
  }
  if (IV.InductionType == InductionVariable::Unknown) return;

  o << "  Start ="; WriteAsOperand(o, IV.Start);
  o << "  Step =" ; WriteAsOperand(o, IV.Step);
  o << "\n";
}
