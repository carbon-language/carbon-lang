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

#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/config.h"
#include <fstream>
#include <sstream>
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
    
    static bool hasNodeAddressLabel(const SDNode *Node,
                                    const SelectionDAG *Graph) {
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
  std::string Op = Node->getOperationName(G);

  for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
    if (Node->getValueType(i) == MVT::Other)
      Op += ":ch";
    else
      Op = Op + ":" + MVT::getValueTypeString(Node->getValueType(i));
    
  if (const ConstantSDNode *CSDN = dyn_cast<ConstantSDNode>(Node)) {
    Op += ": " + utostr(CSDN->getValue());
  } else if (const ConstantFPSDNode *CSDN = dyn_cast<ConstantFPSDNode>(Node)) {
    Op += ": " + ftostr(CSDN->getValue());
  } else if (const GlobalAddressSDNode *GADN =
             dyn_cast<GlobalAddressSDNode>(Node)) {
    int offset = GADN->getOffset();
    Op += ": " + GADN->getGlobal()->getName();
    if (offset > 0)
      Op += "+" + itostr(offset);
    else
      Op += itostr(offset);
  } else if (const FrameIndexSDNode *FIDN = dyn_cast<FrameIndexSDNode>(Node)) {
    Op += " " + itostr(FIDN->getIndex());
  } else if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Node)){
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(CP->get()))
      Op += "<" + ftostr(CFP->getValue()) + ">";
    else if (ConstantInt *CI = dyn_cast<ConstantInt>(CP->get()))
      Op += "<" + utostr(CI->getZExtValue()) + ">";
    else {
      std::ostringstream SS;
      WriteAsOperand(SS, CP->get(), false);
      Op += "<" + SS.str() + ">";
    }
  } else if (const BasicBlockSDNode *BBDN = dyn_cast<BasicBlockSDNode>(Node)) {
    Op = "BB: ";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      Op += LBB->getName();
    //Op += " " + (const void*)BBDN->getBasicBlock();
  } else if (const RegisterSDNode *R = dyn_cast<RegisterSDNode>(Node)) {
    if (G && R->getReg() != 0 && MRegisterInfo::isPhysicalRegister(R->getReg())) {
      Op = Op + " " + G->getTarget().getRegisterInfo()->getName(R->getReg());
    } else {
      Op += " #" + utostr(R->getReg());
    }
  } else if (const ExternalSymbolSDNode *ES =
             dyn_cast<ExternalSymbolSDNode>(Node)) {
    Op += "'" + std::string(ES->getSymbol()) + "'";
  } else if (const SrcValueSDNode *M = dyn_cast<SrcValueSDNode>(Node)) {
    if (M->getValue())
      Op += "<" + M->getValue()->getName() + ":" + itostr(M->getOffset()) + ">";
    else
      Op += "<null:" + itostr(M->getOffset()) + ">";
  } else if (const VTSDNode *N = dyn_cast<VTSDNode>(Node)) {
    Op = Op + " VT=" + getValueTypeString(N->getVT());
  } else if (const StringSDNode *N = dyn_cast<StringSDNode>(Node)) {
    Op = Op + "\"" + N->getValue() + "\"";
  }
  
  return Op;
}


/// viewGraph - Pop up a ghostview window with the reachable parts of the DAG
/// rendered using 'dot'.
///
void SelectionDAG::viewGraph() {
// This code is only for debugging!
#ifndef NDEBUG
  ViewGraph(this, "dag." + getMachineFunction().getFunction()->getName());
#else
  std::cerr << "SelectionDAG::viewGraph is only available in debug builds on "
            << "systems with Graphviz or gv!\n";
#endif  // NDEBUG
}
