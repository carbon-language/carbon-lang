//== CallGraph.cpp - Call graph building ------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the CallGraph and CallGraphNode classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_CALLGRAPH
#define LLVM_CLANG_ANALYSIS_CALLGRAPH

#include "clang/Index/ASTLocation.h"
#include "clang/Index/Entity.h"
#include "clang/Index/Program.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include <vector>
#include <map>

namespace clang {

class CallGraphNode {
  idx::Entity F;
  typedef std::pair<idx::ASTLocation, CallGraphNode*> CallRecord;
  std::vector<CallRecord> CalledFunctions;

public:
  CallGraphNode(idx::Entity f) : F(f) {}

  typedef std::vector<CallRecord>::iterator iterator;
  typedef std::vector<CallRecord>::const_iterator const_iterator;

  iterator begin() { return CalledFunctions.begin(); }
  iterator end()   { return CalledFunctions.end(); }
  const_iterator begin() const { return CalledFunctions.begin(); }
  const_iterator end()   const { return CalledFunctions.end();   }

  void addCallee(idx::ASTLocation L, CallGraphNode *Node) {
    CalledFunctions.push_back(std::make_pair(L, Node));
  }

  bool hasCallee() const { return begin() != end(); }

  std::string getName() const { return F.getPrintableName(); }

  Decl *getDecl(ASTContext &Ctx) const { return F.getDecl(Ctx); }
};

class CallGraph {
  /// Program manages all Entities.
  idx::Program Prog;

  typedef std::map<idx::Entity, CallGraphNode *> FunctionMapTy;

  /// FunctionMap owns all CallGraphNodes.
  FunctionMapTy FunctionMap;

  /// CallerCtx maps a caller to its ASTContext.
  llvm::DenseMap<CallGraphNode *, ASTContext *> CallerCtx;

  /// Root node is the 'main' function or 0.
  CallGraphNode *Root;

  /// ExternalCallingNode has edges to all external functions.
  CallGraphNode *ExternalCallingNode;

public:
  CallGraph();
  ~CallGraph();

  typedef FunctionMapTy::iterator iterator;
  typedef FunctionMapTy::const_iterator const_iterator;

  iterator begin() { return FunctionMap.begin(); }
  iterator end()   { return FunctionMap.end();   }
  const_iterator begin() const { return FunctionMap.begin(); }
  const_iterator end()   const { return FunctionMap.end();   }

  CallGraphNode *getRoot() { return Root; }

  CallGraphNode *getExternalCallingNode() { return ExternalCallingNode; }

  void addTU(ASTContext &AST);

  idx::Program &getProgram() { return Prog; }

  CallGraphNode *getOrInsertFunction(idx::Entity F);

  Decl *getDecl(CallGraphNode *Node);

  void print(llvm::raw_ostream &os);
  void dump();

  void ViewCallGraph() const;
};

} // end clang namespace

namespace llvm {

template <> struct GraphTraits<clang::CallGraph> {
  typedef clang::CallGraph GraphType;
  typedef clang::CallGraphNode NodeType;

  typedef std::pair<clang::idx::ASTLocation, NodeType*> CGNPairTy;
  typedef std::pointer_to_unary_function<CGNPairTy, NodeType*> CGNDerefFun;

  typedef mapped_iterator<NodeType::iterator, CGNDerefFun> ChildIteratorType;

  static NodeType *getEntryNode(GraphType *CG) {
    return CG->getExternalCallingNode();
  }

  static ChildIteratorType child_begin(NodeType *N) {
    return map_iterator(N->begin(), CGNDerefFun(CGNDeref));
  }
  static ChildIteratorType child_end(NodeType *N) {
    return map_iterator(N->end(), CGNDerefFun(CGNDeref));
  }

  typedef std::pair<clang::idx::Entity, NodeType*> PairTy;
  typedef std::pointer_to_unary_function<PairTy, NodeType*> DerefFun;

  typedef mapped_iterator<GraphType::const_iterator, DerefFun> nodes_iterator;

  static nodes_iterator nodes_begin(const GraphType &CG) {
    return map_iterator(CG.begin(), DerefFun(CGDeref));
  }
  static nodes_iterator nodes_end(const GraphType &CG) {
    return map_iterator(CG.end(), DerefFun(CGDeref));
  }

  static NodeType *CGNDeref(CGNPairTy P) { return P.second; }

  static NodeType *CGDeref(PairTy P) { return P.second; }
};

} // end llvm namespace

#endif
