//== CallGraph.cpp - Call graph building ------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the CallGraph and CGBuilder classes.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/CallGraph.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtVisitor.h"

#include "llvm/Support/GraphWriter.h"

using namespace clang;
using namespace idx;

namespace {
class CGBuilder : public StmtVisitor<CGBuilder> {

  CallGraph &G;
  FunctionDecl *FD;

  Entity CallerEnt;

  CallGraphNode *CallerNode;

public:
  CGBuilder(CallGraph &g, FunctionDecl *fd, Entity E, CallGraphNode *N)
    : G(g), FD(fd), CallerEnt(E), CallerNode(N) {}

  void VisitStmt(Stmt *S) { VisitChildren(S); }

  void VisitCallExpr(CallExpr *CE);

  void VisitChildren(Stmt *S) {
    for (Stmt::child_iterator I=S->child_begin(), E=S->child_end(); I != E;++I)
      if (*I)
        static_cast<CGBuilder*>(this)->Visit(*I);
  }
};
}

void CGBuilder::VisitCallExpr(CallExpr *CE) {
  if (FunctionDecl *CalleeDecl = CE->getDirectCallee()) {
    Entity Ent = Entity::get(CalleeDecl, G.getProgram());
    CallGraphNode *CalleeNode = G.getOrInsertFunction(Ent);
    CallerNode->addCallee(ASTLocation(FD, CE), CalleeNode);
  }
}

CallGraph::CallGraph() : Root(0) {
  ExternalCallingNode = getOrInsertFunction(Entity());
}

CallGraph::~CallGraph() {
  if (!FunctionMap.empty()) {
    for (FunctionMapTy::iterator I = FunctionMap.begin(), E = FunctionMap.end();
        I != E; ++I)
      delete I->second;
    FunctionMap.clear();
  }
}

void CallGraph::addTU(ASTContext& Ctx) {
  DeclContext *DC = Ctx.getTranslationUnitDecl();
  for (DeclContext::decl_iterator I = DC->decls_begin(), E = DC->decls_end();
       I != E; ++I) {

    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(*I)) {
      if (FD->isThisDeclarationADefinition()) {
        // Set caller's ASTContext.
        Entity Ent = Entity::get(FD, Prog);
        CallGraphNode *Node = getOrInsertFunction(Ent);
        CallerCtx[Node] = &Ctx;

        // If this function has external linkage, anything could call it.
        if (FD->isGlobal())
          ExternalCallingNode->addCallee(idx::ASTLocation(), Node);

        // Set root node to 'main' function.
        if (FD->getNameAsString() == "main")
          Root = Node;

        CGBuilder builder(*this, FD, Ent, Node);
        builder.Visit(FD->getBody());
      }
    }
  }
}

CallGraphNode *CallGraph::getOrInsertFunction(Entity F) {
  CallGraphNode *&Node = FunctionMap[F];
  if (Node)
    return Node;

  return Node = new CallGraphNode(F);
}

Decl *CallGraph::getDecl(CallGraphNode *Node) {
  // Get the function's context.
  ASTContext *Ctx = CallerCtx[Node];

  return Node->getDecl(*Ctx);
}

void CallGraph::print(llvm::raw_ostream &os) {
  for (iterator I = begin(), E = end(); I != E; ++I) {
    if (I->second->hasCallee()) {
      os << "function: " << I->first.getPrintableName()
         << " calls:\n";
      for (CallGraphNode::iterator CI = I->second->begin(),
             CE = I->second->end(); CI != CE; ++CI) {
        os << "    " << CI->second->getName();
      }
      os << '\n';
    }
  }
}

void CallGraph::dump() {
  print(llvm::errs());
}

void CallGraph::ViewCallGraph() const {
  llvm::ViewGraph(*this, "CallGraph");
}

namespace llvm {

template <>
struct DOTGraphTraits<CallGraph> : public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false) : DefaultDOTGraphTraits(isSimple) {}

  static std::string getNodeLabel(const CallGraphNode *Node,
                                  const CallGraph &CG) {
    return Node->getName();

  }

};

}
