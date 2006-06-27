//===- CFGPrinter.cpp - DOT printer for the control flow graph ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a '-print-cfg' analysis pass, which emits the
// cfg.<fnname>.dot file for each function in the program, with a graph of the
// CFG for that function.
//
// The other main feature of this file is that it implements the
// Function::viewCFG method, which is useful for debugging passes which operate
// on the CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Config/config.h"
#include <sstream>
#include <fstream>
using namespace llvm;

/// CFGOnly flag - This is used to control whether or not the CFG graph printer
/// prints out the contents of basic blocks or not.  This is acceptable because
/// this code is only really used for debugging purposes.
///
static bool CFGOnly = false;

namespace llvm {
template<>
struct DOTGraphTraits<const Function*> : public DefaultDOTGraphTraits {
  static std::string getGraphName(const Function *F) {
    return "CFG for '" + F->getName() + "' function";
  }

  static std::string getNodeLabel(const BasicBlock *Node,
                                  const Function *Graph) {
    if (CFGOnly && !Node->getName().empty())
      return Node->getName() + ":";

    std::ostringstream Out;
    if (CFGOnly) {
      WriteAsOperand(Out, Node, false, true);
      return Out.str();
    }

    if (Node->getName().empty()) {
      WriteAsOperand(Out, Node, false, true);
      Out << ":";
    }

    Out << *Node;
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

  static std::string getEdgeSourceLabel(const BasicBlock *Node,
                                        succ_const_iterator I) {
    // Label source of conditional branches with "T" or "F"
    if (const BranchInst *BI = dyn_cast<BranchInst>(Node->getTerminator()))
      if (BI->isConditional())
        return (I == succ_begin(Node)) ? "T" : "F";
    return "";
  }
};
}

namespace {
  struct CFGPrinter : public FunctionPass {
    virtual bool runOnFunction(Function &F) {
      std::string Filename = "cfg." + F.getName() + ".dot";
      std::cerr << "Writing '" << Filename << "'...";
      std::ofstream File(Filename.c_str());

      if (File.good())
        WriteGraph(File, (const Function*)&F);
      else
        std::cerr << "  error opening file for writing!";
      std::cerr << "\n";
      return false;
    }

    void print(std::ostream &OS, const Module* = 0) const {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  RegisterAnalysis<CFGPrinter> P1("print-cfg",
                                  "Print CFG of function to 'dot' file");

  struct CFGOnlyPrinter : public CFGPrinter {
    virtual bool runOnFunction(Function &F) {
      bool OldCFGOnly = CFGOnly;
      CFGOnly = true;
      CFGPrinter::runOnFunction(F);
      CFGOnly = OldCFGOnly;
      return false;
    }
    void print(std::ostream &OS, const Module* = 0) const {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  RegisterAnalysis<CFGOnlyPrinter>
  P2("print-cfg-only",
     "Print CFG of function to 'dot' file (with no function bodies)");
}

/// viewCFG - This function is meant for use from the debugger.  You can just
/// say 'call F->viewCFG()' and a ghostview window should pop up from the
/// program, displaying the CFG of the current function.  This depends on there
/// being a 'dot' and 'gv' program in your path.
///
void Function::viewCFG() const {
  ViewGraph(this, "cfg" + getName());
}

/// viewCFGOnly - This function is meant for use from the debugger.  It works
/// just like viewCFG, but it does not include the contents of basic blocks
/// into the nodes, just the label.  If you are only interested in the CFG t
/// his can make the graph smaller.
///
void Function::viewCFGOnly() const {
  CFGOnly = true;
  viewCFG();
  CFGOnly = false;
}

FunctionPass *llvm::createCFGPrinterPass () {
  return new CFGPrinter();
}

FunctionPass *llvm::createCFGOnlyPrinterPass () {
  return new CFGOnlyPrinter();
}

