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
#include "llvm/ADT/StringExtras.h"
#include <fstream>
using namespace llvm;

namespace llvm {
  template<>
  struct DOTGraphTraits<SelectionDAG*> : public DefaultDOTGraphTraits {
    static std::string getGraphName(const SelectionDAG *G) {
      return G->getMachineFunction().getFunction()->getName();
    }

    static bool renderGraphFromBottomUp() {
      return true;
    }

    static std::string getNodeLabel(const SDNode *Node,
                                    const SelectionDAG *Graph);
    static std::string getNodeAttributes(const SDNode *N) {
      return "shape=Mrecord";
    }

    static void addCustomGraphFeatures(SelectionDAG *G,
                                       GraphWriter<SelectionDAG*> &GW) {
      GW.emitSimpleNode(0, "plaintext=circle", "GraphRoot");
      GW.emitEdge(0, -1, G->getRoot().Val, -1, "");
    }
  };
}

std::string DOTGraphTraits<SelectionDAG*>::getNodeLabel(const SDNode *Node,
                                                        const SelectionDAG *G) {
  std::string Op = Node->getOperationName();

  for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i) {
    switch (Node->getValueType(i)) {
    default: Op += ":unknownvt!"; break;
    case MVT::Other: Op += ":ch"; break;
    case MVT::i1:    Op += ":i1"; break;
    case MVT::i8:    Op += ":i8"; break;
    case MVT::i16:   Op += ":i16"; break;
    case MVT::i32:   Op += ":i32"; break;
    case MVT::i64:   Op += ":i64"; break;
    case MVT::i128:  Op += ":i128"; break;
    case MVT::f32:   Op += ":f32"; break;
    case MVT::f64:   Op += ":f64"; break;
    case MVT::f80:   Op += ":f80"; break;
    case MVT::f128:  Op += ":f128"; break;
    case MVT::isVoid: Op += ":void"; break;
    }
  }

  if (const ConstantSDNode *CSDN = dyn_cast<ConstantSDNode>(Node)) {
    Op += ": " + utostr(CSDN->getValue());
  } else if (const ConstantFPSDNode *CSDN = dyn_cast<ConstantFPSDNode>(Node)) {
    Op += ": " + ftostr(CSDN->getValue());
  } else if (const GlobalAddressSDNode *GADN =
             dyn_cast<GlobalAddressSDNode>(Node)) {
    Op += ": " + GADN->getGlobal()->getName();
  } else if (const FrameIndexSDNode *FIDN = dyn_cast<FrameIndexSDNode>(Node)) {
    Op += " " + itostr(FIDN->getIndex());
  } else if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Node)){
    Op += "<" + utostr(CP->getIndex()) + ">";
  } else if (const BasicBlockSDNode *BBDN = dyn_cast<BasicBlockSDNode>(Node)) {
    Op = "BB: ";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      Op += LBB->getName();
    //Op += " " + (const void*)BBDN->getBasicBlock();
  } else if (const RegSDNode *C2V = dyn_cast<RegSDNode>(Node)) {
    Op += " #" + utostr(C2V->getReg());
  } else if (const ExternalSymbolSDNode *ES =
             dyn_cast<ExternalSymbolSDNode>(Node)) {
    Op += "'" + std::string(ES->getSymbol()) + "'";
  } else if (const MVTSDNode *M = dyn_cast<MVTSDNode>(Node)) {
    Op = Op + " ty=" + MVT::getValueTypeString(M->getExtraValueType());
  }
  return Op;
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
