//===- GraphPrinters.cpp - DOT printers for various graph types -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines several printers for various different types of graphs used
// by the LLVM infrastructure.  It uses the generic graph interface to convert
// the graph into a .dot graph.  These graphs can then be processed with the
// "dot" tool to convert them to postscript or some other suitable format.
//
//===----------------------------------------------------------------------===//

#include "Support/GraphWriter.h"
#include "llvm/Pass.h"
#include "llvm/Value.h"
#include "llvm/Analysis/CallGraph.h"
#include <fstream>
using namespace llvm;

template<typename GraphType>
static void WriteGraphToFile(std::ostream &O, const std::string &GraphName,
                             const GraphType &GT) {
  std::string Filename = GraphName + ".dot";
  O << "Writing '" << Filename << "'...";
  std::ofstream F(Filename.c_str());
  
  if (F.good())
    WriteGraph(F, GT);
  else
    O << "  error opening file for writing!";
  O << "\n";
}


//===----------------------------------------------------------------------===//
//                              Call Graph Printer
//===----------------------------------------------------------------------===//

namespace llvm {
  template<>
  struct DOTGraphTraits<CallGraph*> : public DefaultDOTGraphTraits {
    static std::string getGraphName(CallGraph *F) {
      return "Call Graph";
    }
    
    static std::string getNodeLabel(CallGraphNode *Node, CallGraph *Graph) {
      if (Node->getFunction())
        return ((Value*)Node->getFunction())->getName();
      else
        return "Indirect call node";
    }
  };
}


namespace {
  struct CallGraphPrinter : public Pass {
    virtual bool run(Module &M) {
      WriteGraphToFile(std::cerr, "callgraph", &getAnalysis<CallGraph>());
      return false;
    }

    void print(std::ostream &OS) const {}
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<CallGraph>();
      AU.setPreservesAll();
    }
  };

  RegisterAnalysis<CallGraphPrinter> P2("print-callgraph",
                                        "Print Call Graph to 'dot' file");
};
