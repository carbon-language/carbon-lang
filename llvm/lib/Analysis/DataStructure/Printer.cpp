//===- Printer.cpp - Code for printing data structure graphs nicely -------===//
//
// This file implements the 'dot' graph printer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Analysis/DSGraphTraits.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "Support/CommandLine.h"
#include "Support/GraphWriter.h"
#include "Support/Statistic.h"
#include <fstream>
#include <sstream>

// OnlyPrintMain - The DataStructure printer exposes this option to allow
// printing of only the graph for "main".
//
namespace {
  cl::opt<bool> OnlyPrintMain("only-print-main-ds", cl::ReallyHidden);
  Statistic<> MaxGraphSize   ("dsnode", "Maximum graph size");
  Statistic<> NumFoldedNodes ("dsnode", "Number of folded nodes (in final graph)");
}


void DSNode::dump() const { print(std::cerr, 0); }

static std::string getCaption(const DSNode *N, const DSGraph *G) {
  std::stringstream OS;
  Module *M = G && G->hasFunction() ? G->getFunction().getParent() : 0;

  if (N->isNodeCompletelyFolded())
    OS << "FOLDED";
  else {
    WriteTypeSymbolic(OS, N->getType(), M);
    if (N->isArray())
      OS << " array";
  }
  if (unsigned NodeType = N->getNodeFlags()) {
    OS << ": ";
    if (NodeType & DSNode::AllocaNode ) OS << "S";
    if (NodeType & DSNode::HeapNode   ) OS << "H";
    if (NodeType & DSNode::GlobalNode ) OS << "G";
    if (NodeType & DSNode::UnknownNode) OS << "U";
    if (NodeType & DSNode::Incomplete ) OS << "I";
    if (NodeType & DSNode::Modified   ) OS << "M";
    if (NodeType & DSNode::Read       ) OS << "R";
#ifndef NDEBUG
    if (NodeType & DSNode::DEAD       ) OS << "<dead>";
#endif
    OS << "\n";
  }

  for (unsigned i = 0, e = N->getGlobals().size(); i != e; ++i) {
    WriteAsOperand(OS, N->getGlobals()[i], false, true, M);
    OS << "\n";
  }

  return OS.str();
}

template<>
struct DOTGraphTraits<const DSGraph*> : public DefaultDOTGraphTraits {
  static std::string getGraphName(const DSGraph *G) {
    if (G->hasFunction())
      return "Function " + G->getFunction().getName();
    else
      return "Global graph";
  }

  static const char *getGraphProperties(const DSGraph *G) {
    return "\tsize=\"10,7.5\";\n"
           "\trotate=\"90\";\n";
  }

  static std::string getNodeLabel(const DSNode *Node, const DSGraph *Graph) {
    return getCaption(Node, Graph);
  }

  static std::string getNodeAttributes(const DSNode *N) {
    return "shape=Mrecord";//fontname=Courier";
  }
  
  /// addCustomGraphFeatures - Use this graph writing hook to emit call nodes
  /// and the return node.
  ///
  static void addCustomGraphFeatures(const DSGraph *G,
                                     GraphWriter<const DSGraph*> &GW) {
    Module *CurMod = G->hasFunction() ? G->getFunction().getParent() : 0;

    // Add scalar nodes to the graph...
    const hash_map<Value*, DSNodeHandle> &VM = G->getScalarMap();
    for (hash_map<Value*, DSNodeHandle>::const_iterator I = VM.begin();
         I != VM.end(); ++I)
      if (!isa<GlobalValue>(I->first)) {
        std::stringstream OS;
        WriteAsOperand(OS, I->first, false, true, CurMod);
        GW.emitSimpleNode(I->first, "", OS.str());
        
        // Add edge from return node to real destination
        int EdgeDest = I->second.getOffset() >> DS::PointerShift;
        if (EdgeDest == 0) EdgeDest = -1;
        GW.emitEdge(I->first, -1, I->second.getNode(),
                    EdgeDest, "arrowtail=tee,color=gray63");
      }


    // Output the returned value pointer...
    if (G->getRetNode().getNode() != 0) {
      // Output the return node...
      GW.emitSimpleNode((void*)1, "plaintext=circle", "returning");

      // Add edge from return node to real destination
      int RetEdgeDest = G->getRetNode().getOffset() >> DS::PointerShift;;
      if (RetEdgeDest == 0) RetEdgeDest = -1;
      GW.emitEdge((void*)1, -1, G->getRetNode().getNode(),
                  RetEdgeDest, "arrowtail=tee,color=gray63");
    }

    // Output all of the call nodes...
    const std::vector<DSCallSite> &FCs =
      G->shouldPrintAuxCalls() ? G->getAuxFunctionCalls()
      : G->getFunctionCalls();
    for (unsigned i = 0, e = FCs.size(); i != e; ++i) {
      const DSCallSite &Call = FCs[i];
      std::vector<std::string> EdgeSourceCaptions(Call.getNumPtrArgs()+2);
      EdgeSourceCaptions[0] = "r";
      if (Call.isDirectCall())
        EdgeSourceCaptions[1] = Call.getCalleeFunc()->getName();
      else
        EdgeSourceCaptions[1] = "f";

      GW.emitSimpleNode(&Call, "shape=record", "call", Call.getNumPtrArgs()+2,
                        &EdgeSourceCaptions);

      if (DSNode *N = Call.getRetVal().getNode()) {
        int EdgeDest = Call.getRetVal().getOffset() >> DS::PointerShift;
        if (EdgeDest == 0) EdgeDest = -1;
        GW.emitEdge(&Call, 0, N, EdgeDest, "color=gray63,tailclip=false");
      }

      // Print out the callee...
      if (Call.isIndirectCall()) {
        DSNode *N = Call.getCalleeNode();
        assert(N && "Null call site callee node!");
        GW.emitEdge(&Call, 1, N, -1, "color=gray63,tailclip=false");
      }

      for (unsigned j = 0, e = Call.getNumPtrArgs(); j != e; ++j)
        if (DSNode *N = Call.getPtrArg(j).getNode()) {
          int EdgeDest = Call.getPtrArg(j).getOffset() >> DS::PointerShift;
          if (EdgeDest == 0) EdgeDest = -1;
          GW.emitEdge(&Call, j+2, N, EdgeDest, "color=gray63,tailclip=false");
        }
    }
  }
};

void DSNode::print(std::ostream &O, const DSGraph *G) const {
  GraphWriter<const DSGraph *> W(O, G);
  W.writeNode(this);
}

void DSGraph::print(std::ostream &O) const {
  WriteGraph(O, this, "DataStructures");
}

void DSGraph::writeGraphToFile(std::ostream &O,
                               const std::string &GraphName) const {
  std::string Filename = GraphName + ".dot";
  O << "Writing '" << Filename << "'...";
  std::ofstream F(Filename.c_str());
  
  if (F.good()) {
    print(F);
    unsigned NumCalls = shouldPrintAuxCalls() ?
      getAuxFunctionCalls().size() : getFunctionCalls().size();
    O << " [" << getGraphSize() << "+" << NumCalls << "]\n";
  } else {
    O << "  error opening file for writing!\n";
  }
}

/// viewGraph - Emit a dot graph, run 'dot', run gv on the postscript file,
/// then cleanup.  For use from the debugger.
///
void DSGraph::viewGraph() const {
  std::ofstream F("/tmp/tempgraph.dot");
  if (!F.good()) {
    std::cerr << "Error opening '/tmp/tempgraph.dot' for temporary graph!\n";
    return;
  }
  print(F);
  F.close();
  if (system("dot -Tps /tmp/tempgraph.dot > /tmp/tempgraph.ps"))
    std::cerr << "Error running dot: 'dot' not in path?\n";
  system("gv /tmp/tempgraph.ps");
  system("rm /tmp/tempgraph.dot /tmp/tempgraph.ps");
}


template <typename Collection>
static void printCollection(const Collection &C, std::ostream &O,
                            const Module *M, const std::string &Prefix) {
  if (M == 0) {
    O << "Null Module pointer, cannot continue!\n";
    return;
  }

  unsigned TotalNumNodes = 0, TotalCallNodes = 0;
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (C.hasGraph(*I)) {
      DSGraph &Gr = C.getDSGraph((Function&)*I);
      TotalNumNodes += Gr.getGraphSize();
      unsigned NumCalls = Gr.shouldPrintAuxCalls() ?
        Gr.getAuxFunctionCalls().size() : Gr.getFunctionCalls().size();

      TotalCallNodes += NumCalls;
      if (I->getName() == "main" || !OnlyPrintMain)
        Gr.writeGraphToFile(O, Prefix+I->getName());
      else {
        O << "Skipped Writing '" << Prefix+I->getName() << ".dot'... ["
          << Gr.getGraphSize() << "+" << NumCalls << "]\n";
      }

      if (MaxGraphSize < Gr.getNodes().size())
        MaxGraphSize = Gr.getNodes().size();
      for (unsigned i = 0, e = Gr.getNodes().size(); i != e; ++i)
        if (Gr.getNodes()[i]->isNodeCompletelyFolded())
          ++NumFoldedNodes;
    }

  DSGraph &GG = C.getGlobalsGraph();
  TotalNumNodes  += GG.getGraphSize();
  TotalCallNodes += GG.getFunctionCalls().size();
  if (!OnlyPrintMain) {
    GG.writeGraphToFile(O, Prefix+"GlobalsGraph");
  } else {
    O << "Skipped Writing '" << Prefix << "GlobalsGraph.dot'... ["
      << GG.getGraphSize() << "+" << GG.getFunctionCalls().size() << "]\n";
  }

  O << "\nGraphs contain [" << TotalNumNodes << "+" << TotalCallNodes 
    << "] nodes total" << std::endl;
}


// print - Print out the analysis results...
void LocalDataStructures::print(std::ostream &O, const Module *M) const {
  printCollection(*this, O, M, "ds.");
}

void BUDataStructures::print(std::ostream &O, const Module *M) const {
  printCollection(*this, O, M, "bu.");
}

void TDDataStructures::print(std::ostream &O, const Module *M) const {
  printCollection(*this, O, M, "td.");
}
