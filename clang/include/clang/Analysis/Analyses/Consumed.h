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

namespace clang {
namespace consumed {

  typedef SmallVector<PartialDiagnosticAt, 1> OptionalNotes;
  typedef std::pair<PartialDiagnosticAt, OptionalNotes> DelayedDiag;
  typedef std::list<DelayedDiag> DiagList;

  class ConsumedWarningsHandlerBase {

  public:

    virtual ~ConsumedWarningsHandlerBase();

    /// \brief Emit the warnings and notes left by the analysis.
    virtual void emitDiagnostics() {}

    /// Warn about unnecessary-test errors.
    /// \param VariableName -- The name of the variable that holds the unique
    /// value.
    ///
    /// \param Loc -- The SourceLocation of the unnecessary test.
    virtual void warnUnnecessaryTest(StringRef VariableName,
                                     StringRef VariableState,
                                     SourceLocation Loc) {}

    /// Warn about use-while-consumed errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseOfTempWhileConsumed(StringRef MethodName,
                                            SourceLocation Loc) {}

    /// Warn about use-in-unknown-state errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseOfTempInUnknownState(StringRef MethodName,
                                             SourceLocation Loc) {}

    /// Warn about use-while-consumed errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    ///
    /// \param VariableName -- The name of the variable that holds the unique
    /// value.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseWhileConsumed(StringRef MethodName,
                                      StringRef VariableName,
                                      SourceLocation Loc) {}

    /// Warn about use-in-unknown-state errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    ///
    /// \param VariableName -- The name of the variable that holds the unique
    /// value.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseInUnknownState(StringRef MethodName,
                                       StringRef VariableName,
                                       SourceLocation Loc) {}
  };

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
    
    ConsumedBlockInfo BlockInfo;
    ConsumedStateMap *CurrStates;
    
    CacheMapType ConsumableTypeCache;
    
    bool hasConsumableAttributes(const CXXRecordDecl *RD);
    void splitState(const CFGBlock *CurrBlock, const IfStmt *Terminator);
    
  public:
    
    ConsumedWarningsHandlerBase &WarningsHandler;

    ConsumedAnalyzer(ConsumedWarningsHandlerBase &WarningsHandler)
        : WarningsHandler(WarningsHandler) {}

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
  
  /// \brief Check to see if a function tests an object's validity.
  bool isTestingFunction(const CXXMethodDecl *MethodDecl);
  
}} // end namespace clang::consumed

#endif
