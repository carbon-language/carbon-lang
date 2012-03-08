//== CallGraph.cpp - AST-based Call graph  ----------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the AST-based CallGraph.
//
//===----------------------------------------------------------------------===//
#include "clang/Analysis/CallGraph.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/StmtVisitor.h"

#include "llvm/Support/GraphWriter.h"

using namespace clang;

/// Determine if a declaration should be included in the graph.
static bool includeInGraph(const Decl *D) {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // We skip function template definitions, as their semantics is
    // only determined when they are instantiated.
    if (!FD->isThisDeclarationADefinition() ||
        FD->isDependentContext())
      return false;

    IdentifierInfo *II = FD->getIdentifier();
    if (II && II->getName().startswith("__inline"))
      return false;
  }

  if (const ObjCMethodDecl *ID = dyn_cast<ObjCMethodDecl>(D)) {
    if (!ID->isThisDeclarationADefinition())
      return false;
  }

  return true;
}

namespace {
/// A helper class, which walks the AST and locates all the call sites in the
/// given function body.
class CGBuilder : public StmtVisitor<CGBuilder> {
  CallGraph *G;
  const Decl *FD;
  CallGraphNode *CallerNode;

public:
  CGBuilder(CallGraph *g, const Decl *D, CallGraphNode *N)
    : G(g), FD(D), CallerNode(N) {}

  void VisitStmt(Stmt *S) { VisitChildren(S); }

  void VisitCallExpr(CallExpr *CE) {
    // TODO: We need to handle ObjC method calls as well.
    if (FunctionDecl *CalleeDecl = CE->getDirectCallee())
      if (includeInGraph(CalleeDecl)) {
        CallGraphNode *CalleeNode = G->getOrInsertFunction(CalleeDecl);
        CallerNode->addCallee(CalleeNode, G);
      }
  };

  void VisitChildren(Stmt *S) {
    for (Stmt::child_range I = S->children(); I; ++I)
      if (*I)
        static_cast<CGBuilder*>(this)->Visit(*I);
  }
};
} // end anonymous namespace

CallGraph::CallGraph() {
  Root = getOrInsertFunction(0);
}

CallGraph::~CallGraph() {
  if (!FunctionMap.empty()) {
    for (FunctionMapTy::iterator I = FunctionMap.begin(), E = FunctionMap.end();
        I != E; ++I)
      delete I->second;
    FunctionMap.clear();
  }
}

void CallGraph::addToCallGraph(Decl* D, bool IsGlobal) {
  CallGraphNode *Node = getOrInsertFunction(D);

  if (IsGlobal)
    Root->addCallee(Node, this);

  // Process all the calls by this function as well.
  CGBuilder builder(this, D, Node);
  builder.Visit(D->getBody());
}

void CallGraph::addToCallGraph(DeclContext *DC) {
  for (DeclContext::decl_iterator I = DC->decls_begin(), E = DC->decls_end();
       I != E; ++I) {
    Decl *D = *I;
    switch (D->getKind()) {
    case Decl::CXXConstructor:
    case Decl::CXXDestructor:
    case Decl::CXXConversion:
    case Decl::CXXMethod:
    case Decl::Function: {
      FunctionDecl *FD = cast<FunctionDecl>(D);
      // We skip function template definitions, as their semantics is
      // only determined when they are instantiated.
      if (includeInGraph(FD))
        // If this function has external linkage, anything could call it.
        // Note, we are not precise here. For example, the function could have
        // its address taken.
        addToCallGraph(FD, FD->isGlobal());
      break;
    }

    case Decl::ObjCCategoryImpl:
    case Decl::ObjCImplementation: {
      ObjCImplDecl *ID = cast<ObjCImplDecl>(D);
      for (ObjCContainerDecl::method_iterator MI = ID->meth_begin(),
          ME = ID->meth_end(); MI != ME; ++MI) {
        if (includeInGraph(*MI))
          addToCallGraph(*MI, true);
      }
      break;
    }

    default:
      break;
    }
  }
}

CallGraphNode *CallGraph::getOrInsertFunction(Decl *F) {
  CallGraphNode *&Node = FunctionMap[F];
  if (Node)
    return Node;

  Node = new CallGraphNode(F);
  ParentlessNodes.insert(Node);
  return Node;
}

void CallGraph::print(raw_ostream &OS) const {
  OS << " --- Call graph Dump --- \n";
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    OS << "  Function: ";
    if (I->second == Root)
      OS << "< root >";
    else
      I->second->print(OS);
    OS << " calls: ";
    for (CallGraphNode::iterator CI = I->second->begin(),
        CE = I->second->end(); CI != CE; ++CI) {
      assert(*CI != Root && "No one can call the root node.");
      (*CI)->print(OS);
      OS << " ";
    }
    OS << '\n';
  }
  OS.flush();
}

void CallGraph::dump() const {
  print(llvm::errs());
}

void CallGraph::viewGraph() const {
  llvm::ViewGraph(this, "CallGraph");
}

StringRef CallGraphNode::getName() const {
  if (const FunctionDecl *D = dyn_cast_or_null<FunctionDecl>(FD))
    if (const IdentifierInfo *II = D->getIdentifier())
      return II->getName();
    return "< >";
}

void CallGraphNode::print(raw_ostream &os) const {
  os << getName();
}

void CallGraphNode::dump() const {
  print(llvm::errs());
}

namespace llvm {

template <>
struct DOTGraphTraits<const CallGraph*> : public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false) : DefaultDOTGraphTraits(isSimple) {}

  static std::string getNodeLabel(const CallGraphNode *Node,
                                  const CallGraph *CG) {
    if (CG->getRoot() == Node) {
      return "< root >";
    }
    return Node->getName();
  }

};
}
