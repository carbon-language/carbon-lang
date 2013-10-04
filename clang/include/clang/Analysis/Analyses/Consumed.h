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
  
  enum ConsumedState {
    // No state information for the given variable.
    CS_None,
    
    CS_Unknown,
    CS_Unconsumed,
    CS_Consumed
  };
  
  class ConsumedStmtVisitor;
  
  typedef SmallVector<PartialDiagnosticAt, 1> OptionalNotes;
  typedef std::pair<PartialDiagnosticAt, OptionalNotes> DelayedDiag;
  typedef std::list<DelayedDiag> DiagList;

  class ConsumedWarningsHandlerBase {

  public:

    virtual ~ConsumedWarningsHandlerBase();

    /// \brief Emit the warnings and notes left by the analysis.
    virtual void emitDiagnostics() {}
    
    // FIXME: This can be removed when the attr propagation fix for templated
    //        classes lands.
    /// \brief Warn about return typestates set for unconsumable types.
    /// 
    /// \param Loc -- The location of the attributes.
    ///
    /// \param TypeName -- The name of the unconsumable type.
    virtual void warnReturnTypestateForUnconsumableType(SourceLocation Loc,
                                                        StringRef TypeName) {}
    
    /// \brief Warn about return typestate mismatches.
    /// \param Loc -- The SourceLocation of the return statement.
    virtual void warnReturnTypestateMismatch(SourceLocation Loc,
                                             StringRef ExpectedState,
                                             StringRef ObservedState) {}
    
    /// \brief Warn about unnecessary-test errors.
    /// \param VariableName -- The name of the variable that holds the unique
    /// value.
    ///
    /// \param VariableState -- The known state of the value.
    ///
    /// \param Loc -- The SourceLocation of the unnecessary test.
    virtual void warnUnnecessaryTest(StringRef VariableName,
                                     StringRef VariableState,
                                     SourceLocation Loc) {}

    /// \brief Warn about use-while-consumed errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    ///
    /// \param State -- The state the object was used in.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseOfTempInInvalidState(StringRef MethodName,
                                             StringRef State,
                                             SourceLocation Loc) {}

    /// \brief Warn about use-while-consumed errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    ///
    /// \param State -- The state the object was used in.
    ///
    /// \param VariableName -- The name of the variable that holds the unique
    /// value.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseInInvalidState(StringRef MethodName,
                                       StringRef VariableName,
                                       StringRef State,
                                       SourceLocation Loc) {}
  };

  class ConsumedStateMap {
    
    typedef llvm::DenseMap<const VarDecl *, ConsumedState> MapType;
    typedef std::pair<const VarDecl *, ConsumedState> PairType;
    
  protected:
    
    bool Reachable;
    const Stmt *From;
    MapType Map;
    
  public:
    ConsumedStateMap() : Reachable(true), From(NULL) {}
    ConsumedStateMap(const ConsumedStateMap &Other)
      : Reachable(Other.Reachable), From(Other.From), Map(Other.Map) {}
    
    /// \brief Get the consumed state of a given variable.
    ConsumedState getState(const VarDecl *Var);
    
    /// \brief Merge this state map with another map.
    void intersect(const ConsumedStateMap *Other);
    
    /// \brief Return true if this block is reachable.
    bool isReachable() const { return Reachable; }
    
    /// \brief Mark all variables as unknown.
    void makeUnknown();
    
    /// \brief Mark the block as unreachable.
    void markUnreachable();
    
    /// \brief Set the source for a decision about the branching of states.
    /// \param Source -- The statement that was the origin of a branching
    /// decision.
    void setSource(const Stmt *Source) { this->From = Source; }
    
    /// \brief Set the consumed state of a given variable.
    void setState(const VarDecl *Var, ConsumedState State);
    
    /// \brief Remove the variable from our state map.
    void remove(const VarDecl *Var);
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

  /// A class that handles the analysis of uniqueness violations.
  class ConsumedAnalyzer {
    
    ConsumedBlockInfo BlockInfo;
    ConsumedStateMap *CurrStates;
    
    ConsumedState ExpectedReturnState;
    
    void determineExpectedReturnState(AnalysisDeclContext &AC,
                                      const FunctionDecl *D);
    bool hasConsumableAttributes(const CXXRecordDecl *RD);
    bool splitState(const CFGBlock *CurrBlock,
                    const ConsumedStmtVisitor &Visitor);
    
  public:
    
    ConsumedWarningsHandlerBase &WarningsHandler;

    ConsumedAnalyzer(ConsumedWarningsHandlerBase &WarningsHandler)
        : WarningsHandler(WarningsHandler) {}

    ConsumedState getExpectedReturnState() const { return ExpectedReturnState; }
    
    /// \brief Check a function's CFG for consumed violations.
    ///
    /// We traverse the blocks in the CFG, keeping track of the state of each
    /// value who's type has uniquness annotations.  If methods are invoked in
    /// the wrong state a warning is issued.  Each block in the CFG is traversed
    /// exactly once.
    void run(AnalysisDeclContext &AC);
  };
}} // end namespace clang::consumed

#endif
