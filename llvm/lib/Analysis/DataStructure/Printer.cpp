//===- Printer.cpp - Code for printing data structure graphs nicely -------===//
//
// This file implements the 'dot' graph printer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include <fstream>
#include <sstream>

void DSNode::dump() const { print(std::cerr, 0); }

std::string DSNode::getCaption(const DSGraph *G) const {
  std::stringstream OS;
  Module *M = G ? G->getFunction().getParent() : 0;
  WriteTypeSymbolic(OS, getType(), M);

  OS << " ";
  if (NodeType & ScalarNode) OS << "S";
  if (NodeType & AllocaNode) OS << "A";
  if (NodeType & NewNode   ) OS << "N";
  if (NodeType & GlobalNode) OS << "G";
  if (NodeType & SubElement) OS << "E";
  if (NodeType & CastNode  ) OS << "C";

  for (unsigned i = 0, e = Globals.size(); i != e; ++i) {
    OS << "\n";
    WriteAsOperand(OS, Globals[i], false, true, M);
  }

  if ((NodeType & ScalarNode) && G) {
    const std::map<Value*, DSNodeHandle> &VM = G->getValueMap();
    for (std::map<Value*, DSNodeHandle>::const_iterator I = VM.begin(),
           E = VM.end(); I != E; ++I)
      if (I->second == this) {
        OS << "\n";
        WriteAsOperand(OS, I->first, false, true, M);
      }
  }

  return OS.str();
}

static std::string getValueName(Value *V, Function &F) {
  std::stringstream OS;
  WriteAsOperand(OS, V, true, true, F.getParent());
  return OS.str();
}



static void replaceIn(std::string &S, char From, const std::string &To) {
  for (unsigned i = 0; i < S.size(); )
    if (S[i] == From) {
      S.replace(S.begin()+i, S.begin()+i+1,
                To.begin(), To.end());
      i += To.size();
    } else {
      ++i;
    }
}

static string escapeLabel(const string &In) {
  string Label(In);
  replaceIn(Label, '\\', "\\\\");  // Escape caption...
  replaceIn(Label, '\n', "\\n");
  replaceIn(Label, ' ', "\\ ");
  replaceIn(Label, '{', "\\{");
  replaceIn(Label, '}', "\\}");
  return Label;
}

static void writeEdge(std::ostream &O, const void *SrcNode,
                      const char *SrcNodePortName, int SrcNodeIdx,
                      const DSNode *VS, const string &EdgeAttr = "") {
  O << "\tNode" << SrcNode << SrcNodePortName;
  if (SrcNodeIdx != -1) O << SrcNodeIdx;
  O << " -> Node" << (void*)VS;

  if (!EdgeAttr.empty())
    O << "[" << EdgeAttr << "]";
  O << ";\n";
}

void DSNode::print(std::ostream &O, const DSGraph *G) const {
  string Caption = escapeLabel(getCaption(G));

  O << "\tNode" << (void*)this << " [ label =\"{" << Caption;

  if (!Links.empty()) {
    O << "|{";
    for (unsigned i = 0; i < Links.size(); ++i) {
      if (i) O << "|";
      O << "<g" << i << ">";
    }
    O << "}";
  }
  O << "}\"];\n";

  for (unsigned i = 0; i < Links.size(); ++i)
    if (Links[i])
      writeEdge(O, this, ":g", i, Links[i]);
}

void DSGraph::print(std::ostream &O) const {
  O << "digraph DataStructures {\n"
    << "\tnode [shape=Mrecord];\n"
    << "\tedge [arrowtail=\"dot\"];\n"
    << "\tsize=\"10,7.5\";\n"
    << "\trotate=\"90\";\n"
    << "\tlabel=\"Function\\ " << Func.getName() << "\";\n\n";

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
      if (Call[j])
        writeEdge(O, &Call, ":g", j, Call[j], "color=gray63");
  }


  O << "}\n";
}




// print - Print out the analysis results...
void LocalDataStructures::print(std::ostream &O, Module *M) const {
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isExternal()) {
      std::string Filename = "ds." + I->getName() + ".dot";
      O << "Writing '" << Filename << "'...";
      std::ofstream F(Filename.c_str());

      if (F.good()) {
        DSGraph &Graph = getDSGraph(*I);
        Graph.print(F);
        O << " [" << Graph.getGraphSize() << "]\n";
      } else {
        O << "  error opening file for writing!\n";
      }
    }
}
