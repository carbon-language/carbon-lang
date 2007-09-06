//===- LiveVariables.h - Live Variable Analysis for Source CFGs -*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Live Variables analysis for source-level CFGs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIVEVARIABLES_H
#define LLVM_CLANG_LIVEVARIABLES_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>

namespace clang {

  class Stmt;
  class DeclRefExpr;
  class Decl;
  class CFG;
  class CFGBlock;
  class SourceManager;
  class LiveVariables;
  
class LiveVariablesAuditor {
public:
  virtual ~LiveVariablesAuditor();

  /// AuditStmt - A callback invoked right before invoking the liveness
  ///  transfer function on the given statement.  If the liveness information
  ///  has been previously calculated by running LiveVariables::runOnCFG,
  ///  then V contains the liveness information after the execution of
  ///  the given statement.
  virtual void AuditStmt(Stmt* S, LiveVariables& L, llvm::BitVector& V) = 0;

  /// AuditBlockExit - A callback invoked right before invoking the liveness
  ///  transfer function on the given block.  If the liveness information
  ///  has been previously calculated by running LiveVariables::runOnCFG,
  ///  then V contains the liveness information after the execution of
  ///  the given block.
  virtual void AuditBlockExit(const CFGBlock* B, LiveVariables& L,
                              llvm::BitVector& V) = 0;  
};

class LiveVariables {
public:

  struct VarInfo {
    /// AliveBlocks - Set of blocks of which this value is alive completely
    /// through.  This is a bit set which uses the basic block number as an
    /// index.
    llvm::BitVector AliveBlocks;
    
    /// Kills - List of statements which are the last use of a variable
    ///  (kill it) in their basic block.  The first pointer in the pair
    ///  is the statement in the list of statements of a basic block where
    ///  this occurs, while the DeclRefExpr is the subexpression of this
    ///  statement where the actual last reference takes place.
    typedef std::vector< std::pair<const Stmt*,const DeclRefExpr*> > KillsSet;
    KillsSet Kills;
    
    // AddKill - Adds a kill site to the list of places where a
    //  a variable is killed.
    void AddKill(Stmt* S, DeclRefExpr* DR) {
      Kills.push_back(std::make_pair(const_cast<const Stmt*>(S),
                                     const_cast<const DeclRefExpr*>(DR)));
    }
  };
  
  struct VPair {
    VarInfo V;
    unsigned Idx;
  };
  
  typedef llvm::DenseMap<const Decl*, VPair > VarInfoMap;
  typedef llvm::DenseMap<const CFGBlock*, llvm::BitVector > BlockLivenessMap;

public:

  LiveVariables() : NumDecls(0) {}

  /// runOnCFG - Computes live variable information for a given CFG.
  void runOnCFG(const CFG& cfg, LiveVariablesAuditor* A = NULL);
  
  /// runOnBlock - Computes live variable information for a given block.
  ///  This should usually be invoked only after previously computing
  ///  live variable information using runOnCFG, and is intended to
  ///  only be used for auditing the liveness within a block.
  void runOnBlock(const CFGBlock* B, LiveVariablesAuditor* A);
  
  /// KillsVar - Return true if the specified statement kills the
  ///  specified variable.
  bool KillsVar(const Stmt* S, const Decl* D) const;
  
  /// IsLive - Return true if a variable is live at beginning of a specified
  //    block.
  bool IsLive(const CFGBlock* B, const Decl* D) const;
  
  /// getVarInfo - Return the liveness information associated with a given
  ///  variable.
  VarInfo& getVarInfo(const Decl* D);

  const VarInfo& getVarInfo(const Decl* D) const;
  
  /// getVarInfoMap
  VarInfoMap& getVarInfoMap() { return VarInfos; }

  const VarInfoMap& getVarInfoMap() const { return VarInfos; }
  
  // dumpLiveness
  void dumpLiveness(const llvm::BitVector& V, SourceManager& M) const;
  
  // dumpBlockLiveness
  void dumpBlockLiveness(SourceManager& M) const;
  
  // getLiveAtBlockEntryMap
  BlockLivenessMap& getLiveAtBlockEntryMap() { return LiveAtBlockEntryMap; }

  const BlockLivenessMap& getLiveAtBlockEntryMap() const {
    return LiveAtBlockEntryMap; 
  }
  
  // getNumDecls
  unsigned& getNumDecls() { return NumDecls; }
  unsigned getNumDecls() const { return NumDecls; }
  
protected:

  unsigned NumDecls;
  VarInfoMap VarInfos;
  BlockLivenessMap LiveAtBlockEntryMap;
  
};

} // end namespace clang

#endif
