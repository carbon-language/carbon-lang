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
#include "llvm/CodeGen/MachineConstantPool.h"
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
    
    /// If you want to override the dot attributes printed for a particular
    /// edge, override this method.
    template<typename EdgeIter>
    static std::string getEdgeAttributes(const void *Node, EdgeIter EI) {
      SDOperand Op = EI.getNode()->getOperand(EI.getOperand());
      MVT::ValueType VT = Op.getValueType();
      if (VT == MVT::Flag)
        return "color=red,style=bold";
      else if (VT == MVT::Other)
        return "style=dashed";
      return "";
    }
    

    static std::string getNodeLabel(const SDNode *Node,
                                    const SelectionDAG *Graph);
    static std::string getNodeAttributes(const SDNode *N,
                                         const SelectionDAG *Graph) {
#ifndef NDEBUG
      const std::string &Attrs = Graph->getGraphAttrs(N);
      if (!Attrs.empty()) {
        if (Attrs.find("shape=") == std::string::npos)
          return std::string("shape=Mrecord,") + Attrs;
        else
          return Attrs;
      }
#endif
      return "shape=Mrecord";
    }

    static void addCustomGraphFeatures(SelectionDAG *G,
                                       GraphWriter<SelectionDAG*> &GW) {
      GW.emitSimpleNode(0, "plaintext=circle", "GraphRoot");
      if (G->getRoot().Val)
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
  } else if (const JumpTableSDNode *JTDN = dyn_cast<JumpTableSDNode>(Node)) {
    Op += " " + itostr(JTDN->getIndex());
  } else if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Node)){
    if (CP->isMachineConstantPoolEntry()) {
      std::ostringstream SS;
      CP->getMachineCPVal()->print(SS);
      Op += "<" + SS.str() + ">";
    } else {
      if (ConstantFP *CFP = dyn_cast<ConstantFP>(CP->getConstVal()))
        Op += "<" + ftostr(CFP->getValue()) + ">";
      else if (ConstantInt *CI = dyn_cast<ConstantInt>(CP->getConstVal()))
        Op += "<" + utostr(CI->getZExtValue()) + ">";
      else {
        std::ostringstream SS;
        WriteAsOperand(SS, CP->getConstVal(), false);
        Op += "<" + SS.str() + ">";
      }
    }
  } else if (const BasicBlockSDNode *BBDN = dyn_cast<BasicBlockSDNode>(Node)) {
    Op = "BB: ";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      Op += LBB->getName();
    //Op += " " + (const void*)BBDN->getBasicBlock();
  } else if (const RegisterSDNode *R = dyn_cast<RegisterSDNode>(Node)) {
    if (G && R->getReg() != 0 &&
        MRegisterInfo::isPhysicalRegister(R->getReg())) {
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
  } else if (const LoadSDNode *LD = dyn_cast<LoadSDNode>(Node)) {
    bool doExt = true;
    switch (LD->getExtensionType()) {
    default: doExt = false; break;
    case ISD::EXTLOAD:
      Op = Op + "<anyext ";
      break;
    case ISD::SEXTLOAD:
      Op = Op + " <sext ";
      break;
    case ISD::ZEXTLOAD:
      Op = Op + " <zext ";
      break;
    }
    if (doExt)
      Op = Op + MVT::getValueTypeString(LD->getLoadedVT()) + ">";

    Op += LD->getIndexedModeName(LD->getAddressingMode());
  } else if (const StoreSDNode *ST = dyn_cast<StoreSDNode>(Node)) {
    if (ST->isTruncatingStore())
      Op = Op + "<trunc " + MVT::getValueTypeString(ST->getStoredVT()) + ">";
    Op += ST->getIndexedModeName(ST->getAddressingMode());
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
  cerr << "SelectionDAG::viewGraph is only available in debug builds on "
       << "systems with Graphviz or gv!\n";
#endif  // NDEBUG
}


/// clearGraphAttrs - Clear all previously defined node graph attributes.
/// Intended to be used from a debugging tool (eg. gdb).
void SelectionDAG::clearGraphAttrs() {
#ifndef NDEBUG
  NodeGraphAttrs.clear();
#else
  cerr << "SelectionDAG::clearGraphAttrs is only available in debug builds"
       << " on systems with Graphviz or gv!\n";
#endif
}


/// setGraphAttrs - Set graph attributes for a node. (eg. "color=red".)
///
void SelectionDAG::setGraphAttrs(const SDNode *N, const char *Attrs) {
#ifndef NDEBUG
  NodeGraphAttrs[N] = Attrs;
#else
  cerr << "SelectionDAG::setGraphAttrs is only available in debug builds"
       << " on systems with Graphviz or gv!\n";
#endif
}


/// getGraphAttrs - Get graph attributes for a node. (eg. "color=red".)
/// Used from getNodeAttributes.
const std::string SelectionDAG::getGraphAttrs(const SDNode *N) const {
#ifndef NDEBUG
  std::map<const SDNode *, std::string>::const_iterator I =
    NodeGraphAttrs.find(N);
    
  if (I != NodeGraphAttrs.end())
    return I->second;
  else
    return "";
#else
  cerr << "SelectionDAG::getGraphAttrs is only available in debug builds"
       << " on systems with Graphviz or gv!\n";
  return std::string("");
#endif
}

/// setGraphColor - Convenience for setting node color attribute.
///
void SelectionDAG::setGraphColor(const SDNode *N, const char *Color) {
#ifndef NDEBUG
  NodeGraphAttrs[N] = std::string("color=") + Color;
#else
  cerr << "SelectionDAG::setGraphColor is only available in debug builds"
       << " on systems with Graphviz or gv!\n";
#endif
}

