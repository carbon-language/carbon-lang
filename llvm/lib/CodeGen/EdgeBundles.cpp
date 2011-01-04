//===-------- EdgeBundles.cpp - Bundles of CFG edges ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the EdgeBundles analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/EdgeBundles.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/GraphWriter.h"

using namespace llvm;

char EdgeBundles::ID = 0;

INITIALIZE_PASS(EdgeBundles, "edge-bundles", "Bundle Machine CFG Edges",
                /* cfg = */true, /* analysis = */ true)

char &llvm::EdgeBundlesID = EdgeBundles::ID;

void EdgeBundles::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool EdgeBundles::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  EC.clear();
  EC.grow(2 * MF->size());

  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end(); I != E;
       ++I) {
    const MachineBasicBlock &MBB = *I;
    unsigned OutE = 2 * MBB.getNumber() + 1;
    // Join the outgoing bundle with the ingoing bundles of all successors.
    for (MachineBasicBlock::const_succ_iterator SI = MBB.succ_begin(),
           SE = MBB.succ_end(); SI != SE; ++SI)
      EC.join(OutE, 2 * (*SI)->getNumber());
  }
  EC.compress();
  return false;
}

/// view - Visualize the annotated bipartite CFG with Graphviz.
void EdgeBundles::view() const {
  ViewGraph(*this, "EdgeBundles");
}

/// Specialize WriteGraph, the standard implementation won't work.
raw_ostream &llvm::WriteGraph(raw_ostream &O, const EdgeBundles &G,
                              bool ShortNames,
                              const std::string &Title) {
  const MachineFunction *MF = G.getMachineFunction();

  O << "digraph {\n";
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    unsigned BB = I->getNumber();
    O << "\t\"BB#" << BB << "\" [ shape=box ]\n"
      << '\t' << G.getBundle(BB, false) << " -> \"BB#" << BB << "\"\n"
      << "\t\"BB#" << BB << "\" -> " << G.getBundle(BB, true) << '\n';
    for (MachineBasicBlock::const_succ_iterator SI = I->succ_begin(),
           SE = I->succ_end(); SI != SE; ++SI)
      O << "\t\"BB#" << BB << "\" -> \"BB#" << (*SI)->getNumber()
        << "\" [ color=lightgray ]\n";
  }
  O << "}\n";
  return O;
}


