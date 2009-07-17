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
#include "clang/Frontend/ASTUnit.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>

namespace clang {

class CallGraphNode {
  idx::Entity *F;
  typedef std::pair<idx::ASTLocation, CallGraphNode*> CallRecord;
  std::vector<CallRecord> CalledFunctions;

public:
  CallGraphNode(idx::Entity *f) : F(f) {}

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

  const char *getName(ASTContext &Ctx) { return F->getName(Ctx); }
};

class CallGraph {
  /// Program manages all Entities.
  idx::Program Prog;

  typedef llvm::DenseMap<idx::Entity *, CallGraphNode *> FunctionMapTy;

  /// FunctionMap owns all CallGraphNodes.
  FunctionMapTy FunctionMap;

  /// CallerCtx maps a caller to its ASTContext.
  llvm::DenseMap<CallGraphNode *, ASTContext *> CallerCtx;

public:
  ~CallGraph();

  typedef FunctionMapTy::iterator iterator;
  typedef FunctionMapTy::const_iterator const_iterator;

  iterator begin() { return FunctionMap.begin(); }
  iterator end()   { return FunctionMap.end();   }
  const_iterator begin() const { return FunctionMap.begin(); }
  const_iterator end()   const { return FunctionMap.end();   }

  void addTU(ASTUnit &AST);

  idx::Program &getProgram() { return Prog; }

  CallGraphNode *getOrInsertFunction(idx::Entity *F);

  void print(llvm::raw_ostream &os);
  void dump();
};

}

#endif
