//===- GraphPrinters.cpp - DOT printers for various graph types -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
//
// This file defines several printers for various different types of graphs used
// by the LLVM infrastructure.  It uses the generic graph interface to convert
// the graph into a .dot graph.  These graphs can then be processed with the
// "dot" tool to convert them to postscript or some other suitable format.
//
//===----------------------------------------------------------------------===//

#include "Support/GraphWriter.h"
#include "llvm/Pass.h"
#include "llvm/iTerminators.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/CFG.h"
#include <sstream>
#include <fstream>

//===----------------------------------------------------------------------===//
//                         Control Flow Graph Printer
//===----------------------------------------------------------------------===//

template<>
struct DOTGraphTraits<Function*> : public DefaultDOTGraphTraits {
  static std::string getGraphName(Function *F) {
    return "CFG for '" + F->getName() + "' function";
  }

  static std::string getNodeLabel(BasicBlock *Node, Function *Graph) {
    std::ostringstream Out;
    Out << Node;
    std::string OutStr = Out.str();
    if (OutStr[0] == '\n') OutStr.erase(OutStr.begin());

    // Process string output to make it nicer...
    for (unsigned i = 0; i != OutStr.length(); ++i)
      if (OutStr[i] == '\n') {                            // Left justify
        OutStr[i] = '\\';
        OutStr.insert(OutStr.begin()+i+1, 'l');
      } else if (OutStr[i] == ';') {                      // Delete comments!
        unsigned Idx = OutStr.find('\n', i+1);            // Find end of line
        OutStr.erase(OutStr.begin()+i, OutStr.begin()+Idx);
        --i;
      }

    return OutStr;
  }

  static std::string getNodeAttributes(BasicBlock *N) {
    return "fontname=Courier";
  }
  
  static std::string getEdgeSourceLabel(BasicBlock *Node, succ_iterator I) {
    // Label source of conditional branches with "T" or "F"
    if (BranchInst *BI = dyn_cast<BranchInst>(Node->getTerminator()))
      if (BI->isConditional())
        return (I == succ_begin(Node)) ? "T" : "F";
    return "";
  }
};

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


namespace {
  struct CFGPrinter : public FunctionPass {
    virtual bool runOnFunction(Function &Func) {
      WriteGraphToFile(std::cerr, "cfg."+Func.getName(), &Func);
      return false;
    }

    void print(std::ostream &OS) const {}
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  RegisterAnalysis<CFGPrinter> P1("print-cfg",
                                  "Print CFG of function to 'dot' file");
};



//===----------------------------------------------------------------------===//
//                              Call Graph Printer
//===----------------------------------------------------------------------===//

template<>
struct DOTGraphTraits<CallGraph*> : public DefaultDOTGraphTraits {
  static std::string getGraphName(CallGraph *F) {
    return "Call Graph";
  }

  static std::string getNodeLabel(CallGraphNode *Node, CallGraph *Graph) {
    if (Node->getFunction())
      return Node->getFunction()->getName();
    else
      return "Indirect call node";
  }
};


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
