//===- Printer.cpp - Code for printing data structure graphs nicely -------===//
//
// This file implements the 'dot' graph printer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "Support/CommandLine.h"
#include <fstream>
#include <sstream>
using std::string;

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

  if (getSize() != 0) {
    O << "|{";
    for (unsigned i = 0; i < getSize(); ++i) {
      if (i) O << "|";
      O << "<g" << i << ">" << (int)MergeMap[i];
    }
    O << "}";
  }
  O << "}\"];\n";

  for (unsigned i = 0; i != getSize(); ++i)
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


void DSGraph::writeGraphToFile(std::ostream &O, const string &GraphName) {
  string Filename = GraphName + ".dot";
  O << "Writing '" << Filename << "'...";
  std::ofstream F(Filename.c_str());
  
  if (F.good()) {
    print(F);
    O << " [" << getGraphSize() << "+" << getFunctionCalls().size() << "]\n";
  } else {
    O << "  error opening file for writing!\n";
  }
}

static cl::opt<bool> OnlyPrintMain("only-print-main-ds", cl::ReallyHidden);

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

#if 0
void BUDataStructures::print(std::ostream &O, const Module *M) const {
  printCollection(*this, O, M, "bu.");

  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isExternal()) {
      (*getDSGraph(*I).GlobalsGraph)->writeGraphToFile(O, "gg.program");
      break;
    }
}

void TDDataStructures::print(std::ostream &O, const Module *M) const {
  printCollection(*this, O, M, "td.");

  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isExternal()) {
      (*getDSGraph(*I).GlobalsGraph)->writeGraphToFile(O, "gg.program");
      break;
    }
}
#endif
