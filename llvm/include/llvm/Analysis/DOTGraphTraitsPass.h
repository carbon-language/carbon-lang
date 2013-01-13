//===-- DOTGraphTraitsPass.h - Print/View dotty graphs-----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Templates to create dotty viewer and printer passes for GraphTraits graphs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOTGRAPHTRAITSPASS_H
#define LLVM_ANALYSIS_DOTGRAPHTRAITSPASS_H

#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Pass.h"

namespace llvm {

template <class Analysis, bool Simple>
class DOTGraphTraitsViewer : public FunctionPass {
public:
  DOTGraphTraitsViewer(StringRef GraphName, char &ID)
    : FunctionPass(ID), Name(GraphName) {}

  virtual bool runOnFunction(Function &F) {
    Analysis *Graph = &getAnalysis<Analysis>();
    std::string GraphName = DOTGraphTraits<Analysis*>::getGraphName(Graph);
    std::string Title = GraphName + " for '" + F.getName().str() + "' function";

    ViewGraph(Graph, Name, Simple, Title);

    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<Analysis>();
  }

private:
  std::string Name;
};

template <class Analysis, bool Simple>
class DOTGraphTraitsPrinter : public FunctionPass {
public:
  DOTGraphTraitsPrinter(StringRef GraphName, char &ID)
    : FunctionPass(ID), Name(GraphName) {}

  virtual bool runOnFunction(Function &F) {
    Analysis *Graph = &getAnalysis<Analysis>();
    std::string Filename = Name + "." + F.getName().str() + ".dot";
    std::string ErrorInfo;

    errs() << "Writing '" << Filename << "'...";

    raw_fd_ostream File(Filename.c_str(), ErrorInfo);
    std::string GraphName = DOTGraphTraits<Analysis*>::getGraphName(Graph);
    std::string Title = GraphName + " for '" + F.getName().str() + "' function";

    if (ErrorInfo.empty())
      WriteGraph(File, Graph, Simple, Title);
    else
      errs() << "  error opening file for writing!";
    errs() << "\n";

    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<Analysis>();
  }

private:
  std::string Name;
};

template <class Analysis, bool Simple>
class DOTGraphTraitsModuleViewer : public ModulePass {
public:
  DOTGraphTraitsModuleViewer(StringRef GraphName, char &ID)
    : ModulePass(ID), Name(GraphName) {}

  virtual bool runOnModule(Module &M) {
    Analysis *Graph = &getAnalysis<Analysis>();
    std::string Title = DOTGraphTraits<Analysis*>::getGraphName(Graph);

    ViewGraph(Graph, Name, Simple, Title);

    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<Analysis>();
  }

private:
  std::string Name;
};

template <class Analysis, bool Simple>
class DOTGraphTraitsModulePrinter : public ModulePass {
public:
  DOTGraphTraitsModulePrinter(StringRef GraphName, char &ID)
    : ModulePass(ID), Name(GraphName) {}

  virtual bool runOnModule(Module &M) {
    Analysis *Graph = &getAnalysis<Analysis>();
    std::string Filename = Name + ".dot";
    std::string ErrorInfo;

    errs() << "Writing '" << Filename << "'...";

    raw_fd_ostream File(Filename.c_str(), ErrorInfo);
    std::string Title = DOTGraphTraits<Analysis*>::getGraphName(Graph);

    if (ErrorInfo.empty())
      WriteGraph(File, Graph, Simple, Title);
    else
      errs() << "  error opening file for writing!";
    errs() << "\n";

    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<Analysis>();
  }

private:
  std::string Name;
};

} // end namespace llvm

#endif
