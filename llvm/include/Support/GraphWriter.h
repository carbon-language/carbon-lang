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
#include <vector>
#include <iostream>

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
class GraphWriter {
  std::ostream &O;
  const GraphType &G;

  typedef DOTGraphTraits<GraphType>           DOTTraits;
  typedef GraphTraits<GraphType>              GTraits;
  typedef typename GTraits::NodeType          NodeType;
  typedef typename GTraits::nodes_iterator    node_iterator;
  typedef typename GTraits::ChildIteratorType child_iterator;
public:
  GraphWriter(std::ostream &o, const GraphType &g) : O(o), G(g) {}

  void writeHeader(const std::string &Name) {
    if (Name.empty())
      O << "digraph foo {\n";        // Graph name doesn't matter
    else
      O << "digraph " << Name << " {\n";

    std::string GraphName = DOTTraits::getGraphName(G);
    if (!GraphName.empty())
      O << "\tlabel=\"" << DOT::EscapeString(GraphName) << "\";\n";
    O << DOTTraits::getGraphProperties(G);
    O << "\n";
  }

  void writeFooter() {
    // Finish off the graph
    O << "}\n";
  }

  void writeNodes() {
    // Loop over the graph, printing it out...
    for (node_iterator I = GTraits::nodes_begin(G), E = GTraits::nodes_end(G);
         I != E; ++I)
      writeNode(&*I);
  }

  void writeNode(NodeType *Node) {
    std::string NodeAttributes = DOTTraits::getNodeAttributes(Node);
      
    O << "\tNode" << (void*)Node << " [shape=record,";
    if (!NodeAttributes.empty()) O << NodeAttributes << ",";
    O << "label=\"{"
      << DOT::EscapeString(DOTTraits::getNodeLabel(Node, G));
    
    // Print out the fields of the current node...
    child_iterator EI = GTraits::child_begin(Node);
    child_iterator EE = GTraits::child_end(Node);
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
    for (unsigned i = 0; EI != EE && i != 64; ++EI, ++i)
      writeEdge(Node, i, EI);
  }

  void writeEdge(NodeType *Node, unsigned edgeidx, child_iterator EI) {
    if (NodeType *TargetNode = *EI) {
      int DestPort = -1;
      if (DOTTraits::edgeTargetsEdgeSource(Node, EI)) {
        child_iterator TargetIt = DOTTraits::getEdgeTarget(Node, EI);

        // Figure out which edge this targets...
        unsigned Offset = std::distance(GTraits::child_begin(TargetNode),
                                        TargetIt);
        DestPort = (int)Offset;
      }

      emitEdge((void *)Node, edgeidx, (void*)TargetNode, DestPort,
               DOTTraits::getEdgeAttributes(Node, EI));
    }
  }

  /// emitSimpleNode - Outputs a simple (non-record) node
  void emitSimpleNode(const void *ID, const std::string &Attr,
                      const std::string &Label, unsigned NumEdgeSources = 0,
                      const std::vector<std::string> *EdgeSourceLabels = 0) {
    O << "\tNode" << ID << "[ ";
    if (!Attr.empty())
      O << Attr << ",";
    O << " label =\"";
    if (NumEdgeSources) O << "{";
    O << DOT::EscapeString(Label);
    if (NumEdgeSources) {
      O << "|{";
      
      for (unsigned i = 0; i != NumEdgeSources; ++i) {
        if (i) O << "|";
        O << "<g" << i << ">";
        if (EdgeSourceLabels) O << (*EdgeSourceLabels)[i];
      }
      O << "}}";
    }
    O << "\"];\n";
  }

  /// emitEdge - Output an edge from a simple node into the graph...
  void emitEdge(const void *SrcNodeID, int SrcNodePort,
                const void *DestNodeID, int DestNodePort,
                const std::string &Attrs) {
    if (SrcNodePort  > 64) return;             // Eminating from truncated part?
    if (DestNodePort > 64) DestNodePort = 64;  // Targetting the truncated part?

    O << "\tNode" << SrcNodeID;
    if (SrcNodePort >= 0)
      O << ":g" << SrcNodePort;
    O << " -> Node" << (void*)DestNodeID;
    if (DestNodePort >= 0)
      O << ":g" << DestNodePort;    

    if (!Attrs.empty())
      O << "[" << Attrs << "]";
    O << ";\n";
  }
};

template<typename GraphType>
std::ostream &WriteGraph(std::ostream &O, const GraphType &G,
                         const std::string &Name = "") {
  // Start the graph emission process...
  GraphWriter<GraphType> W(O, G);

  // Output the header for the graph...
  W.writeHeader(Name);

  // Emit all of the nodes in the graph...
  W.writeNodes();

  // Output any customizations on the graph
  DOTGraphTraits<GraphType>::addCustomGraphFeatures(G, W);

  // Output the end of the graph
  W.writeFooter();
  return O;
}

#endif
