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

#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/DOTGraphTraitsPass.h"
#include "llvm/Analysis/PostDominators.h"

using namespace llvm;

namespace llvm {
template<>
struct DOTGraphTraits<DomTreeNode*> : public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false)
    : DefaultDOTGraphTraits(isSimple) {}

  std::string getNodeLabel(DomTreeNode *Node, DomTreeNode *Graph) {

    BasicBlock *BB = Node->getBlock();

    if (!BB)
      return "Post dominance root node";


    if (isSimple())
      return DOTGraphTraits<const Function*>
	       ::getSimpleNodeLabel(BB, BB->getParent());
    else
      return DOTGraphTraits<const Function*>
	       ::getCompleteNodeLabel(BB, BB->getParent());
  }
};

template<>
struct DOTGraphTraits<DominatorTree*> : public DOTGraphTraits<DomTreeNode*> {

  DOTGraphTraits (bool isSimple=false)
    : DOTGraphTraits<DomTreeNode*>(isSimple) {}

  static std::string getGraphName(DominatorTree *DT) {
    return "Dominator tree";
  }

  std::string getNodeLabel(DomTreeNode *Node, DominatorTree *G) {
    return DOTGraphTraits<DomTreeNode*>::getNodeLabel(Node, G->getRootNode());
  }
};

template<>
struct DOTGraphTraits<PostDominatorTree*>
  : public DOTGraphTraits<DomTreeNode*> {

  DOTGraphTraits (bool isSimple=false)
    : DOTGraphTraits<DomTreeNode*>(isSimple) {}

  static std::string getGraphName(PostDominatorTree *DT) {
    return "Post dominator tree";
  }

  std::string getNodeLabel(DomTreeNode *Node, PostDominatorTree *G ) {
    return DOTGraphTraits<DomTreeNode*>::getNodeLabel(Node, G->getRootNode());
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
    std::string Title, GraphName;
    Graph = &getAnalysis<Analysis>();
    GraphName = DOTGraphTraits<Analysis*>::getGraphName(Graph);
    Title = GraphName + " for '" + F.getNameStr() + "' function";
    ViewGraph(Graph, Name, OnlyBBS, Title);

    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<Analysis>();
  }
};

struct DomViewer
  : public DOTGraphTraitsViewer<DominatorTree, false> {
  static char ID;
  DomViewer() : DOTGraphTraitsViewer<DominatorTree, false>("dom", &ID){}
};

struct DomOnlyViewer
  : public DOTGraphTraitsViewer<DominatorTree, true> {
  static char ID;
  DomOnlyViewer() : DOTGraphTraitsViewer<DominatorTree, true>("domonly", &ID){}
};

struct PostDomViewer
  : public DOTGraphTraitsViewer<PostDominatorTree, false> {
  static char ID;
  PostDomViewer() :
    DOTGraphTraitsViewer<PostDominatorTree, false>("postdom", &ID){}
};

struct PostDomOnlyViewer
  : public DOTGraphTraitsViewer<PostDominatorTree, true> {
  static char ID;
  PostDomOnlyViewer() :
    DOTGraphTraitsViewer<PostDominatorTree, true>("postdomonly", &ID){}
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
struct DomPrinter
  : public DOTGraphTraitsPrinter<DominatorTree, false> {
  static char ID;
  DomPrinter() : DOTGraphTraitsPrinter<DominatorTree, false>("dom", &ID) {}
};

struct DomOnlyPrinter
  : public DOTGraphTraitsPrinter<DominatorTree, true> {
  static char ID;
  DomOnlyPrinter() : DOTGraphTraitsPrinter<DominatorTree, true>("domonly", &ID) {}
};

struct PostDomPrinter
  : public DOTGraphTraitsPrinter<PostDominatorTree, false> {
  static char ID;
  PostDomPrinter() :
    DOTGraphTraitsPrinter<PostDominatorTree, false>("postdom", &ID) {}
};

struct PostDomOnlyPrinter
  : public DOTGraphTraitsPrinter<PostDominatorTree, true> {
  static char ID;
  PostDomOnlyPrinter() :
    DOTGraphTraitsPrinter<PostDominatorTree, true>("postdomonly", &ID) {}
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
