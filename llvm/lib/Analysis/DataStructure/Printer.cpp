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
#include <fstream>
#include <sstream>
using std::string;

// OnlyPrintMain - The DataStructure printer exposes this option to allow
// printing of only the graph for "main".
//
static cl::opt<bool> OnlyPrintMain("only-print-main-ds", cl::ReallyHidden);


void DSNode::dump() const { print(std::cerr, 0); }

static string getCaption(const DSNode *N, const DSGraph *G) {
  std::stringstream OS;
  Module *M = G && &G->getFunction() ? G->getFunction().getParent() : 0;

  for (unsigned i = 0, e = N->getTypeEntries().size(); i != e; ++i) {
    WriteTypeSymbolic(OS, N->getTypeEntries()[i].first, M);
    if (N->getTypeEntries()[i].second)
      OS << "@" << N->getTypeEntries()[i].second;
    OS << "\n";
  }

  if (N->NodeType & DSNode::ScalarNode) OS << "S";
  if (N->NodeType & DSNode::AllocaNode) OS << "A";
  if (N->NodeType & DSNode::NewNode   ) OS << "N";
  if (N->NodeType & DSNode::GlobalNode) OS << "G";
  if (N->NodeType & DSNode::Incomplete) OS << "I";

  for (unsigned i = 0, e = N->getGlobals().size(); i != e; ++i) {
    WriteAsOperand(OS, N->getGlobals()[i], false, true, M);
    OS << "\n";
  }

  if ((N->NodeType & DSNode::ScalarNode) && G) {
    const std::map<Value*, DSNodeHandle> &VM = G->getValueMap();
    for (std::map<Value*, DSNodeHandle>::const_iterator I = VM.begin(),
           E = VM.end(); I != E; ++I)
      if (I->second.getNode() == N) {
        WriteAsOperand(OS, I->first, false, true, M);
        OS << "\n";
      }
  }
  return OS.str();
}

static string getValueName(Value *V, Function &F) {
  std::stringstream OS;
  WriteAsOperand(OS, V, true, true, F.getParent());
  return OS.str();
}



static void replaceIn(string &S, char From, const string &To) {
  for (unsigned i = 0; i < S.size(); )
    if (S[i] == From) {
      S.replace(S.begin()+i, S.begin()+i+1,
                To.begin(), To.end());
      i += To.size();
    } else {
      ++i;
    }
}

static std::string escapeLabel(const std::string &In) {
  std::string Label(In);
  replaceIn(Label, '\\', "\\\\");  // Escape caption...
  replaceIn(Label, '\n', "\\n");
  replaceIn(Label, ' ', "\\ ");
  replaceIn(Label, '{', "\\{");
  replaceIn(Label, '}', "\\}");
  return Label;
}

static void writeEdge(std::ostream &O, const void *SrcNode,
                      const char *SrcNodePortName, int SrcNodeIdx,
                      const DSNodeHandle &VS,
                      const std::string &EdgeAttr = "") {
  O << "\tNode" << SrcNode << SrcNodePortName;
  if (SrcNodeIdx != -1) O << SrcNodeIdx;
  O << " -> Node" << (void*)VS.getNode();
  if (VS.getOffset()) O << ":g" << VS.getOffset();

  if (!EdgeAttr.empty())
    O << "[" << EdgeAttr << "]";
  O << ";\n";
}

void DSNode::print(std::ostream &O, const DSGraph *G) const {
  std::string Caption = escapeLabel(getCaption(this, G));

  O << "\tNode" << (void*)this << " [ label =\"{" << Caption;

  unsigned Size = getSize();
  if (Size > 64) Size = 64;   // Don't print out HUGE graph nodes!

  if (getSize() != 0) {
    O << "|{";
    for (unsigned i = 0; i < Size; ++i) {
      if (i) O << "|";
      O << "<g" << i << ">" << (int)MergeMap[i];
    }
    if (Size != getSize())
      O << "|truncated...";
    O << "}";
  }
  O << "}\"];\n";

  for (unsigned i = 0; i != Size; ++i)
    if (const DSNodeHandle *DSN = getLink(i))
      writeEdge(O, this, ":g", i, *DSN);
}

void DSGraph::print(std::ostream &O) const {
  O << "digraph DataStructures {\n"
    << "\tnode [shape=Mrecord];\n"
    << "\tedge [arrowtail=\"dot\"];\n"
    << "\tsize=\"10,7.5\";\n"
    << "\trotate=\"90\";\n";

  if (Func != 0)
    O << "\tlabel=\"Function\\ " << Func->getName() << "\";\n\n";

  // Output all of the nodes...
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    Nodes[i]->print(O, this);

  O << "\n";

  // Output the returned value pointer...
  if (RetNode != 0) {
    O << "\tNode0x1" << "[ plaintext=circle, label =\""
      << escapeLabel("returning") << "\"];\n";
    writeEdge(O, (void*)1, "", -1, RetNode, "arrowtail=tee,color=gray63");
  }

  // Output all of the call nodes...
  for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i) {
    const std::vector<DSNodeHandle> &Call = FunctionCalls[i];
    O << "\tNode" << (void*)&Call << " [shape=record,label=\"{call|{";
    for (unsigned j = 0, e = Call.size(); j != e; ++j) {
      if (j) O << "|";
      O << "<g" << j << ">";
    }
    O << "}}\"];\n";

    for (unsigned j = 0, e = Call.size(); j != e; ++j)
      if (Call[j].getNode())
        writeEdge(O, &Call, ":g", j, Call[j], "color=gray63");
  }


  O << "}\n";
}

template<>
struct DOTGraphTraits<DSGraph*> : public DefaultDOTGraphTraits {
  static std::string getGraphName(DSGraph *G) {
    if (G->hasFunction())
      return "Function " + G->getFunction().getName();
    else
      return "Non-function graph";
  }

  static const char *getGraphProperties(DSGraph *G) {
    return "\tedge [arrowtail=\"dot\"];\n"
           "\tsize=\"10,7.5\";\n"
           "\trotate=\"90\";\n";
  }

  static std::string getNodeLabel(DSNode *Node, DSGraph *Graph) {
    return getCaption(Node, Graph);
  }

  static std::string getNodeAttributes(DSNode *N) {
    return "shape=Mrecord";//fontname=Courier";
  }
  
  static int getEdgeSourceLabel(DSNode *Node, DSNode::iterator I) {
    assert(Node == I.getNode() && "Iterator not for this node!");
    return Node->getMergeMapLabel(I.getOffset());
  }

  /// addCustomGraphFeatures - Use this graph writing hook to emit call nodes
  /// and the return node.
  ///
  static void addCustomGraphFeatures(DSGraph *G, GraphWriter<DSGraph*> &GW) {
    // Output the returned value pointer...
    if (G->getRetNode().getNode() != 0) {
      // Output the return node...
      GW.emitSimpleNode((void*)1, "plaintext=circle", "returning");

      // Add edge from return node to real destination
      int RetEdgeDest = G->getRetNode().getOffset();
      if (RetEdgeDest == 0) RetEdgeDest = -1;
      GW.emitEdge((void*)1, -1, G->getRetNode().getNode(),
                  RetEdgeDest, "arrowtail=tee,color=gray63");
    }

    // Output all of the call nodes...
    const std::vector<std::vector<DSNodeHandle> > &FCs = G->getFunctionCalls();
    for (unsigned i = 0, e = FCs.size(); i != e; ++i) {
      const std::vector<DSNodeHandle> &Call = FCs[i];
      GW.emitSimpleNode(&Call, "shape=record", "call", Call.size());

      for (unsigned j = 0, e = Call.size(); j != e; ++j)
        if (Call[j].getNode()) {
          int EdgeDest = Call[j].getOffset();
          if (EdgeDest == 0) EdgeDest = -1;
          GW.emitEdge(&Call, j, Call[j].getNode(), EdgeDest, "color=gray63");
        }
    }
  }
};


void DSGraph::writeGraphToFile(std::ostream &O, const string &GraphName) {
  string Filename = GraphName + ".dot";
  O << "Writing '" << Filename << "'...";
  std::ofstream F(Filename.c_str());
  
  if (F.good()) {
    WriteGraph(F, this, "DataStructures");
    //print(F);
    O << " [" << getGraphSize() << "+" << getFunctionCalls().size() << "]\n";
  } else {
    O << "  error opening file for writing!\n";
  }
}

template <typename Collection>
static void printCollection(const Collection &C, std::ostream &O,
                            const Module *M, const string &Prefix) {
  if (M == 0) {
    O << "Null Module pointer, cannot continue!\n";
    return;
  }

  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isExternal() && (I->getName() == "main" || !OnlyPrintMain))
      C.getDSGraph((Function&)*I).writeGraphToFile(O, Prefix+I->getName());
}


// print - Print out the analysis results...
void LocalDataStructures::print(std::ostream &O, const Module *M) const {
  printCollection(*this, O, M, "ds.");
}

void BUDataStructures::print(std::ostream &O, const Module *M) const {
  printCollection(*this, O, M, "bu.");
#if 0
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isExternal()) {
      (*getDSGraph(*I).GlobalsGraph)->writeGraphToFile(O, "gg.program");
      break;
    }
#endif
}

#if 0
void TDDataStructures::print(std::ostream &O, const Module *M) const {
  printCollection(*this, O, M, "td.");

  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isExternal()) {
      (*getDSGraph(*I).GlobalsGraph)->writeGraphToFile(O, "gg.program");
      break;
    }
}
#endif
