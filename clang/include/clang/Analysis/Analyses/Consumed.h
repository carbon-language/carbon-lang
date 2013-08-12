//===- Consumed.h ----------------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A intra-procedural analysis for checking consumed properties.  This is based,
// in part, on research on linear types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CONSUMED_H
#define LLVM_CLANG_CONSUMED_H

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/ConsumedWarningsHandler.h"
#include "clang/Sema/Sema.h"

namespace clang {
namespace consumed {
  
  enum ConsumedState {
    // No state information for the given variable.
    CS_None,
    
    CS_Unknown,
    CS_Unconsumed,
    CS_Consumed
  };

  class ConsumedStateMap {
    
    typedef llvm::DenseMap<const VarDecl *, ConsumedState> MapType;
    typedef std::pair<const VarDecl *, ConsumedState> PairType;
    
  protected:
    
    MapType Map;
    
  public:
    /// \brief Get the consumed state of a given variable.
    ConsumedState getState(const VarDecl *Var);
    
    /// \brief Merge this state map with another map.
    void intersect(const ConsumedStateMap *Other);
    
    /// \brief Mark all variables as unknown.
    void makeUnknown();
    
    /// \brief Set the consumed state of a given variable.
    void setState(const VarDecl *Var, ConsumedState State);
  };
  
  class ConsumedBlockInfo {
    
    ConsumedStateMap **StateMapsArray;
    PostOrderCFGView::CFGBlockSet VisitedBlocks;
    
  public:
    
    ConsumedBlockInfo() : StateMapsArray(NULL) {}
    
    ConsumedBlockInfo(const CFG *CFGraph)
      : StateMapsArray(new ConsumedStateMap*[CFGraph->getNumBlockIDs()]()),
        VisitedBlocks(CFGraph) {}
    
    void addInfo(const CFGBlock *Block, ConsumedStateMap *StateMap,
                 bool &AlreadyOwned);
    void addInfo(const CFGBlock *Block, ConsumedStateMap *StateMap);
    
    ConsumedStateMap* getInfo(const CFGBlock *Block);
    
    void markVisited(const CFGBlock *Block);
  };
  
  struct VarTestResult {
    const VarDecl *Var;
    SourceLocation Loc;
    bool UnconsumedInTrueBranch;
    
    VarTestResult() : Var(NULL), Loc(), UnconsumedInTrueBranch(true) {}
    
    VarTestResult(const VarDecl *Var, SourceLocation Loc,
                  bool UnconsumedInTrueBranch)
      : Var(Var), Loc(Loc), UnconsumedInTrueBranch(UnconsumedInTrueBranch) {}
  };

  /// A class that handles the analysis of uniqueness violations.
  class ConsumedAnalyzer {
    
    typedef llvm::DenseMap<const CXXRecordDecl *, bool> CacheMapType;
    typedef std::pair<const CXXRecordDecl *, bool> CachePairType;
    
    Sema &S;
    
    ConsumedBlockInfo BlockInfo;
    ConsumedStateMap *CurrStates;
    
    CacheMapType ConsumableTypeCache;
    
    bool hasConsumableAttributes(const CXXRecordDecl *RD);
    void splitState(const CFGBlock *CurrBlock, const IfStmt *Terminator);
    
  public:
    
    ConsumedWarningsHandlerBase &WarningsHandler;
    
    ConsumedAnalyzer(Sema &S, ConsumedWarningsHandlerBase &WarningsHandler)
        : S(S), WarningsHandler(WarningsHandler) {}
    
    /// \brief Get a constant reference to the Sema object.
    const Sema & getSema(void);
    
    /// \brief Check to see if the type is a consumable type.
    bool isConsumableType(QualType Type);
    
    /// \brief Check a function's CFG for consumed violations.
    ///
    /// We traverse the blocks in the CFG, keeping track of the state of each
    /// value who's type has uniquness annotations.  If methods are invoked in
    /// the wrong state a warning is issued.  Each block in the CFG is traversed
    /// exactly once.
    void run(AnalysisDeclContext &AC);
  };
  
  unsigned checkEnabled(DiagnosticsEngine &D);
  /// \brief Check to see if a function tests an object's validity.
  bool isTestingFunction(const CXXMethodDecl *MethodDecl);
  
}} // end namespace clang::consumed

#endif
