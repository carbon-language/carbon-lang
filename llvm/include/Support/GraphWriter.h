//===-- Support/GraphWriter.h - Write a graph to a .dot file ---*- C++ -*--===//
//
// This file defines a simple interface that can be used to print out generic
// LLVM graphs to ".dot" files.  "dot" is a tool that is part of the AT&T
// graphviz package (http://www.research.att.com/sw/tools/graphviz/) which can
// be used to turn the files output by this interface into a variety of
// different graphics formats.
//
// Graphs do not need to implement any interface past what is already required
// by the GraphTraits template, but they can choose to implement specializations
// of the DOTGraphTraits template if they want to customize the graphs output in
// any way.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_GRAPHWRITER_H
#define SUPPORT_GRAPHWRITER_H

#include "Support/DOTGraphTraits.h"
#include "Support/GraphTraits.h"
#include <ostream>

namespace DOT {  // Private functions...
  inline std::string EscapeString(const std::string &Label) {
    std::string Str(Label);
    for (unsigned i = 0; i != Str.length(); ++i)
      switch (Str[i]) {
      case '\n':
        Str.insert(Str.begin()+i, '\\');  // Escape character...
        ++i;
        Str[i] = 'n';
        break;
      case '\t':
        Str.insert(Str.begin()+i, ' ');  // Convert to two spaces
        ++i;
        Str[i] = ' ';
        break;
      case '\\':
        if (i+1 != Str.length() && Str[i+1] == 'l')
          break;  // don't disturb \l
      case '{': case '}':
      case '<': case '>':
        Str.insert(Str.begin()+i, '\\');  // Escape character...
        ++i;  // don't infinite loop
        break;
      }
    return Str;
  }
}

template<typename GraphType>
std::ostream &WriteGraph(std::ostream &O, const GraphType &G) {
  typedef DOTGraphTraits<GraphType>  DOTTraits;
  typedef GraphTraits<GraphType>     GTraits;
  typedef typename GTraits::NodeType NodeType;
  typedef typename GTraits::nodes_iterator node_iterator;

  O << "digraph foo {\n";        // Graph name doesn't matter
  std::string GraphName = DOTTraits::getGraphName(G);
  if (!GraphName.empty())
    O << "\tlabel=\"" << DOT::EscapeString(GraphName) << "\";\n";
  O << DOTTraits::getGraphProperties(G);
  O << "\n";

  // Loop over the graph in DFO, printing it out...
  for (node_iterator I = GTraits::nodes_begin(G), E = GTraits::nodes_end(G);
       I != E; ++I) {
    NodeType *Node = &*I;

    std::string NodeAttributes = DOTTraits::getNodeAttributes(Node);

    O << "\tNode" << (void*)Node << " [";
    if (!NodeAttributes.empty()) O << NodeAttributes << ",";
    O << "shape=record,label=\"{"
      << DOT::EscapeString(DOTTraits::getNodeLabel(Node, G));
    
    // Print out the fields of the current node...
    typename GTraits::ChildIteratorType EI = GTraits::child_begin(Node);
    typename GTraits::ChildIteratorType EE = GTraits::child_end(Node);
    if (EI != EE) {
      O << "|{";

      for (unsigned i = 0; EI != EE && i != 64; ++EI, ++i) {
        if (i) O << "|";
        O << "<g" << i << ">" << DOTTraits::getEdgeSourceLabel(Node, EI);
      }

      if (EI != EE)
        O << "|truncated...";
      O << "}";
    }
    O << "}\"];\n";   // Finish printing the "node" line

    // Output all of the edges now
    EI = GTraits::child_begin(Node);
    for (unsigned i = 0; EI != EE && i != 64; ++EI, ++i) {
      NodeType *TargetNode = *EI;
      O << "\tNode" << (void*)Node << ":g" << i << " -> Node"
        << (void*)TargetNode;
      if (DOTTraits::edgeTargetsEdgeSource(Node, EI)) {
        typename GTraits::ChildIteratorType TargetIt =
          DOTTraits::getEdgeTarget(Node, EI);
        // Figure out which edge this targets...
        unsigned Offset = std::distance(GTraits::child_begin(TargetNode),
                                        TargetIt);
        if (Offset > 64) Offset = 64;  // Targetting the trancated part?
        O << ":g" << Offset;        
      }

      std::string EdgeAttributes = DOTTraits::getEdgeAttributes(Node, EI);
      if (!EdgeAttributes.empty())
        O << "[" << EdgeAttributes << "]";
      O << ";\n";
    }
  }

  // Finish off the graph
  O << "}\n";
  return O;
}

#endif
