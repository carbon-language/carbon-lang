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
#include "llvm/System/Path.h"
#include "llvm/System/Program.h"
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
  char pathsuff[9];

  sprintf(pathsuff, "%06u", unsigned(rand()));

  sys::Path TempDir = sys::Path::GetTemporaryDirectory();
  sys::Path Filename = TempDir;
  Filename.appendComponent("dag." + getMachineFunction().getFunction()->getName() + "." + pathsuff + ".dot");
  std::cerr << "Writing '" << Filename.toString() << "'... ";
  std::ofstream F(Filename.toString().c_str());

  if (!F) {
    std::cerr << "  error opening file for writing!\n";
    return;
  }

  WriteGraph(F, this);
  F.close();
  std::cerr << "\n";

#if HAVE_GRAPHVIZ
  sys::Path Graphviz(LLVM_PATH_GRAPHVIZ);
  std::vector<const char*> args;
  args.push_back(Graphviz.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);
  
  std::cerr << "Running 'Graphviz' program... " << std::flush;
  if (sys::Program::ExecuteAndWait(Graphviz, &args[0])) {
    std::cerr << "Error viewing graph: 'Graphviz' not in path?\n";
  } else {
    Filename.eraseFromDisk();
    return;
  }
#elif (HAVE_GV && HAVE_DOT)
  sys::Path PSFilename = TempDir;
  PSFilename.appendComponent(std::string("dag.tempgraph") + "." + pathsuff + ".ps");

  sys::Path dot(LLVM_PATH_DOT);
  std::vector<const char*> args;
  args.push_back(dot.c_str());
  args.push_back("-Tps");
  args.push_back("-Nfontname=Courier");
  args.push_back("-Gsize=7.5,10");
  args.push_back(Filename.c_str());
  args.push_back("-o");
  args.push_back(PSFilename.c_str());
  args.push_back(0);
  
  std::cerr << "Running 'dot' program... " << std::flush;
  if (sys::Program::ExecuteAndWait(dot, &args[0])) {
    std::cerr << "Error viewing graph: 'dot' not in path?\n";
  } else {
    std::cerr << "\n";

    sys::Path gv(LLVM_PATH_GV);
    args.clear();
    args.push_back(gv.c_str());
    args.push_back(PSFilename.c_str());
    args.push_back(0);
    
    sys::Program::ExecuteAndWait(gv, &args[0]);
  }
  Filename.eraseFromDisk();
  PSFilename.eraseFromDisk();
  return;
#elif HAVE_DOTTY
  sys::Path dotty(LLVM_PATH_DOTTY);
  std::vector<const char*> args;
  args.push_back(dotty.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);
  
  std::cerr << "Running 'dotty' program... " << std::flush;
  if (sys::Program::ExecuteAndWait(dotty, &args[0])) {
    std::cerr << "Error viewing graph: 'dotty' not in path?\n";
  } else {
#ifndef __MINGW32__ // Dotty spawns another app and doesn't wait until it returns
    Filename.eraseFromDisk();
#endif
    return;
  }
#endif
  
#endif  // NDEBUG
  std::cerr << "SelectionDAG::viewGraph is only available in debug builds on "
            << "systems with Graphviz or gv!\n";

#ifndef NDEBUG
  Filename.eraseFromDisk();
  TempDir.eraseFromDisk(true);
#endif
}
