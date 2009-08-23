//===- CFGPrinter.cpp - DOT printer for the control flow graph ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a '-dot-cfg' analysis pass, which emits the
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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Config/config.h"
#include <iosfwd>
#include <sstream>
#include <fstream>
using namespace llvm;

namespace llvm {
template<>
struct DOTGraphTraits<const Function*> : public DefaultDOTGraphTraits {
  static std::string getGraphName(const Function *F) {
    return "CFG for '" + F->getNameStr() + "' function";
  }

  static std::string getNodeLabel(const BasicBlock *Node,
                                  const Function *Graph,
                                  bool ShortNames) {
    if (ShortNames && !Node->getName().empty())
      return Node->getNameStr() + ":";

    std::string Str;
    raw_string_ostream OS(Str);

    if (ShortNames) {
      WriteAsOperand(OS, Node, false);
      return OS.str();
    }

    if (Node->getName().empty()) {
      WriteAsOperand(OS, Node, false);
      OS << ":";
    }
    
    OS << *Node;
    std::string OutStr = OS.str();
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
  struct VISIBILITY_HIDDEN CFGViewer : public FunctionPass {
    static char ID; // Pass identifcation, replacement for typeid
    CFGViewer() : FunctionPass(&ID) {}

    virtual bool runOnFunction(Function &F) {
      F.viewCFG();
      return false;
    }

    void print(std::ostream &OS, const Module* = 0) const {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char CFGViewer::ID = 0;
static RegisterPass<CFGViewer>
V0("view-cfg", "View CFG of function", false, true);

namespace {
  struct VISIBILITY_HIDDEN CFGOnlyViewer : public FunctionPass {
    static char ID; // Pass identifcation, replacement for typeid
    CFGOnlyViewer() : FunctionPass(&ID) {}

    virtual bool runOnFunction(Function &F) {
      F.viewCFG();
      return false;
    }

    void print(std::ostream &OS, const Module* = 0) const {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char CFGOnlyViewer::ID = 0;
static RegisterPass<CFGOnlyViewer>
V1("view-cfg-only",
   "View CFG of function (with no function bodies)", false, true);

namespace {
  struct VISIBILITY_HIDDEN CFGPrinter : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CFGPrinter() : FunctionPass(&ID) {}
    explicit CFGPrinter(void *pid) : FunctionPass(pid) {}

    virtual bool runOnFunction(Function &F) {
      std::string Filename = "cfg." + F.getNameStr() + ".dot";
      cerr << "Writing '" << Filename << "'...";
      std::ofstream File(Filename.c_str());

      if (File.good())
        WriteGraph(File, (const Function*)&F);
      else
        cerr << "  error opening file for writing!";
      cerr << "\n";
      return false;
    }

    void print(std::ostream &OS, const Module* = 0) const {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char CFGPrinter::ID = 0;
static RegisterPass<CFGPrinter>
P1("dot-cfg", "Print CFG of function to 'dot' file", false, true);

namespace {
  struct VISIBILITY_HIDDEN CFGOnlyPrinter : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CFGOnlyPrinter() : FunctionPass(&ID) {}
    explicit CFGOnlyPrinter(void *pid) : FunctionPass(pid) {}
    virtual bool runOnFunction(Function &F) {
      std::string Filename = "cfg." + F.getNameStr() + ".dot";
      cerr << "Writing '" << Filename << "'...";
      std::ofstream File(Filename.c_str());

      if (File.good())
        WriteGraph(File, (const Function*)&F, true);
      else
        cerr << "  error opening file for writing!";
      cerr << "\n";
      return false;
    }
    void print(std::ostream &OS, const Module* = 0) const {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char CFGOnlyPrinter::ID = 0;
static RegisterPass<CFGOnlyPrinter>
P2("dot-cfg-only",
   "Print CFG of function to 'dot' file (with no function bodies)", false, true);

/// viewCFG - This function is meant for use from the debugger.  You can just
/// say 'call F->viewCFG()' and a ghostview window should pop up from the
/// program, displaying the CFG of the current function.  This depends on there
/// being a 'dot' and 'gv' program in your path.
///
void Function::viewCFG() const {
  ViewGraph(this, "cfg" + getNameStr());
}

/// viewCFGOnly - This function is meant for use from the debugger.  It works
/// just like viewCFG, but it does not include the contents of basic blocks
/// into the nodes, just the label.  If you are only interested in the CFG t
/// his can make the graph smaller.
///
void Function::viewCFGOnly() const {
  ViewGraph(this, "cfg" + getNameStr(), true);
}

FunctionPass *llvm::createCFGPrinterPass () {
  return new CFGPrinter();
}

FunctionPass *llvm::createCFGOnlyPrinterPass () {
  return new CFGOnlyPrinter();
}

