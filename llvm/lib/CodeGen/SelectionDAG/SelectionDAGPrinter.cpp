//===-- SelectionDAGPrinter.cpp - Implement SelectionDAG::viewGraph() -----===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAG::viewGraph method.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Function.h"
#include "llvm/Support/GraphWriter.h"
#include <fstream>
using namespace llvm;

namespace llvm {
  template<>
  struct DOTGraphTraits<SelectionDAG*> : public DefaultDOTGraphTraits {
    static std::string getGraphName(const SelectionDAG *G) {
      return G->getMachineFunction().getFunction()->getName();
    }
    static std::string getNodeLabel(const SDNode *Node,
                                    const SelectionDAG *Graph) {
      return Node->getOperationName();
    }

    static std::string getNodeAttributes(const SDNode *N) {
      return "shape=Mrecord";
    }
  };
}

/// viewGraph - Pop up a ghostview window with the reachable parts of the DAG
/// rendered using 'dot'.
///
void SelectionDAG::viewGraph() {
  std::string Filename = "/tmp/dag." +
    getMachineFunction().getFunction()->getName() + ".dot";
  std::cerr << "Writing '" << Filename << "'... ";
  std::ofstream F(Filename.c_str());

  if (!F) {
    std::cerr << "  error opening file for writing!\n";
    return;
  }

  WriteGraph(F, this);
  F.close();
  std::cerr << "\n";

  std::cerr << "Running 'dot' program... " << std::flush;
  if (system(("dot -Tps -Nfontname=Courier -Gsize=7.5,10 " + Filename
              + " > /tmp/dag.tempgraph.ps").c_str())) {
    std::cerr << "Error running dot: 'dot' not in path?\n";
  } else {
    std::cerr << "\n";
    system("gv /tmp/dag.tempgraph.ps");
  }
  system(("rm " + Filename + " /tmp/dag.tempgraph.ps").c_str());
}
