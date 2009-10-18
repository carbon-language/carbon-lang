//===- DomPrinter.cpp - DOT printer for the dominance trees    ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines '-dot-dom' and '-dot-postdom' analysis passes, which emit
// a dom.<fnname>.dot or postdom.<fnname>.dot file for each function in the
// program, with a graph of the dominance/postdominance tree of that
// function.
//
// There are also passes available to directly call dotty ('-view-dom' or
// '-view-postdom'). By appending '-only' like '-dot-dom-only' only the
// names of the bbs are printed, but the content is hidden.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DomPrinter.h"

#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/PostDominators.h"

using namespace llvm;

namespace llvm {
template<>
struct DOTGraphTraits<DomTreeNode*> : public DefaultDOTGraphTraits {
  static std::string getNodeLabel(DomTreeNode *Node, DomTreeNode *Graph,
                                  bool ShortNames) {

    BasicBlock *BB = Node->getBlock();

    if (!BB)
      return "Post dominance root node";

    return DOTGraphTraits<const Function*>::getNodeLabel(BB, BB->getParent(),
                                                         ShortNames);
  }
};

template<>
struct DOTGraphTraits<DominatorTree*> : public DOTGraphTraits<DomTreeNode*> {

  static std::string getGraphName(DominatorTree *DT) {
    return "Dominator tree";
  }

  static std::string getNodeLabel(DomTreeNode *Node,
                                  DominatorTree *G,
                                  bool ShortNames) {
    return DOTGraphTraits<DomTreeNode*>::getNodeLabel(Node, G->getRootNode(),
                                                      ShortNames);
  }
};

template<>
struct DOTGraphTraits<PostDominatorTree*>
  : public DOTGraphTraits<DomTreeNode*> {
  static std::string getGraphName(PostDominatorTree *DT) {
    return "Post dominator tree";
  }
  static std::string getNodeLabel(DomTreeNode *Node,
                                  PostDominatorTree *G,
                                  bool ShortNames) {
    return DOTGraphTraits<DomTreeNode*>::getNodeLabel(Node,
                                                      G->getRootNode(),
                                                      ShortNames);
  }
};
}

namespace {
template <class Analysis, bool OnlyBBS>
struct GenericGraphViewer : public FunctionPass {
  std::string Name;

  GenericGraphViewer(std::string GraphName, const void *ID) : FunctionPass(ID) {
    Name = GraphName;
  }

  virtual bool runOnFunction(Function &F) {
    Analysis *Graph;

    Graph = &getAnalysis<Analysis>();
    ViewGraph(Graph, Name, OnlyBBS);

    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<Analysis>();
  }
};

struct DomViewer
  : public GenericGraphViewer<DominatorTree, false> {
  static char ID;
  DomViewer() : GenericGraphViewer<DominatorTree, false>("dom", &ID){}
};

struct DomOnlyViewer
  : public GenericGraphViewer<DominatorTree, true> {
  static char ID;
  DomOnlyViewer() : GenericGraphViewer<DominatorTree, true>("domonly", &ID){}
};

struct PostDomViewer
  : public GenericGraphViewer<PostDominatorTree, false> {
  static char ID;
  PostDomViewer() :
    GenericGraphViewer<PostDominatorTree, false>("postdom", &ID){}
};

struct PostDomOnlyViewer
  : public GenericGraphViewer<PostDominatorTree, true> {
  static char ID;
  PostDomOnlyViewer() :
    GenericGraphViewer<PostDominatorTree, true>("postdomonly", &ID){}
};
} // end anonymous namespace

char DomViewer::ID = 0;
RegisterPass<DomViewer> A("view-dom",
                          "View dominance tree of function");

char DomOnlyViewer::ID = 0;
RegisterPass<DomOnlyViewer> B("view-dom-only",
                              "View dominance tree of function "
                              "(with no function bodies)");

char PostDomViewer::ID = 0;
RegisterPass<PostDomViewer> C("view-postdom",
                              "View postdominance tree of function");

char PostDomOnlyViewer::ID = 0;
RegisterPass<PostDomOnlyViewer> D("view-postdom-only",
                                  "View postdominance tree of function "
                                  "(with no function bodies)");

namespace {
template <class Analysis, bool OnlyBBS>
struct GenericGraphPrinter : public FunctionPass {

  static char ID;
  std::string Name;

  GenericGraphPrinter(std::string GraphName) : FunctionPass(&ID) {
    Name = GraphName;
  }

  virtual bool runOnFunction(Function &F) {
    Analysis *Graph;
    std::string Filename = Name + "." + F.getNameStr() + ".dot";
    errs() << "Writing '" << Filename << "'...";

    std::string ErrorInfo;
    raw_fd_ostream File(Filename.c_str(), ErrorInfo);
    Graph = &getAnalysis<Analysis>();

    if (ErrorInfo.empty())
      WriteGraph(File, Graph, OnlyBBS);
    else
      errs() << "  error opening file for writing!";
    errs() << "\n";
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<Analysis>();
  }
};

struct DomPrinter
  : public GenericGraphPrinter<DominatorTree, false> {
  static char ID;
  DomPrinter() : GenericGraphPrinter<DominatorTree, false>("dom"){}
};

struct DomOnlyPrinter
  : public GenericGraphPrinter<DominatorTree, true> {
  static char ID;
  DomOnlyPrinter() : GenericGraphPrinter<DominatorTree, true>("domonly"){}
};

struct PostDomPrinter
  : public GenericGraphPrinter<PostDominatorTree, false> {
  static char ID;
  PostDomPrinter() :
    GenericGraphPrinter<PostDominatorTree, false>("postdom"){}
};

struct PostDomOnlyPrinter
  : public GenericGraphPrinter<PostDominatorTree, true> {
  static char ID;
  PostDomOnlyPrinter() :
    GenericGraphPrinter<PostDominatorTree, true>("postdomonly"){}
};
} // end anonymous namespace



char DomPrinter::ID = 0;
RegisterPass<DomPrinter> E("dot-dom",
                           "Print dominance tree of function "
                           "to 'dot' file");

char DomOnlyPrinter::ID = 0;
RegisterPass<DomOnlyPrinter> F("dot-dom-only",
                               "Print dominance tree of function "
                               "to 'dot' file "
                               "(with no function bodies)");

char PostDomPrinter::ID = 0;
RegisterPass<PostDomPrinter> G("dot-postdom",
                               "Print postdominance tree of function "
                               "to 'dot' file");

char PostDomOnlyPrinter::ID = 0;
RegisterPass<PostDomOnlyPrinter> H("dot-postdom-only",
                                   "Print postdominance tree of function "
                                   "to 'dot' file "
                                   "(with no function bodies)");

// Create methods available outside of this file, to use them
// "include/llvm/LinkAllPasses.h". Otherwise the pass would be deleted by
// the link time optimization.

FunctionPass *llvm::createDomPrinterPass() {
  return new DomPrinter();
}

FunctionPass *llvm::createDomOnlyPrinterPass() {
  return new DomOnlyPrinter();
}

FunctionPass *llvm::createDomViewerPass() {
  return new DomViewer();
}

FunctionPass *llvm::createDomOnlyViewerPass() {
  return new DomOnlyViewer();
}

FunctionPass *llvm::createPostDomPrinterPass() {
  return new PostDomPrinter();
}

FunctionPass *llvm::createPostDomOnlyPrinterPass() {
  return new PostDomOnlyPrinter();
}

FunctionPass *llvm::createPostDomViewerPass() {
  return new PostDomViewer();
}

FunctionPass *llvm::createPostDomOnlyViewerPass() {
  return new PostDomOnlyViewer();
}
