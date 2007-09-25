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

#include "clang/AST/Decl.h"
#include "clang/Analysis/DataflowValues.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

class Stmt;
class DeclRefExpr;
class SourceManager;
  
struct LiveVariables_ValueTypes {
  //===-----------------------------------------------------===//
  // AnalysisDataTy - Whole-function analysis meta data.
  //===-----------------------------------------------------===//

  class ObserverTy;
  
  class AnalysisDataTy {
    typedef llvm::DenseMap<const VarDecl*, unsigned> VMapTy;
    VMapTy M;
    unsigned NDecls;
   public:
    ObserverTy* Observer;
    
    AnalysisDataTy() : NDecls(0), Observer(NULL) {}
    
    bool Tracked(const VarDecl* VD) const { return M.find(VD) != M.end(); }
    void RegisterDecl(const VarDecl* VD) { if (!Tracked(VD)) M[VD] = NDecls++; }

    unsigned operator[](const VarDecl* VD) const {
      VMapTy::const_iterator I = M.find(VD);
      assert (I != M.end());
      return I->second;
    }
    
    unsigned operator[](const ScopedDecl* S) const {
      return (*this)[cast<VarDecl>(S)];
    }

    unsigned getNumDecls() const { return NDecls; }
    
    typedef VMapTy::const_iterator iterator;
    iterator begin() const { return M.begin(); }
    iterator end() const { return M.end(); }
  };
  
  //===-----------------------------------------------------===//
  // ValTy - Dataflow value.
  //===-----------------------------------------------------===//
  class ValTy {
    llvm::BitVector V;
  public:
    void copyValues(const ValTy& RHS) { V = RHS.V; }

    bool operator==(const ValTy& RHS) const { return V == RHS.V; }
    
    void resetValues(const AnalysisDataTy& AD) {
      V.resize(AD.getNumDecls());
      V.reset();
    }
          
    llvm::BitVector::reference operator[](unsigned i) {
      assert (i < V.size() && "Liveness bitvector access is out-of-bounds.");
      return V[i];
    }
    
    const llvm::BitVector::reference operator[](unsigned i) const {
      return const_cast<ValTy&>(*this)[i];
    }
    
    bool operator()(const AnalysisDataTy& AD, const ScopedDecl* D) const {
      return (*this)[AD[D]];
    }
    
    ValTy& operator|=(ValTy& RHS) { V |= RHS.V; return *this; }
    
    void set(unsigned i) { V.set(i); }
    void reset(unsigned i) { V.reset(i); }
  };
  
  //===-----------------------------------------------------===//
  // ObserverTy - Observer for uninitialized values queries.
  //===-----------------------------------------------------===//
  class ObserverTy {
  public:
    virtual ~ObserverTy() {}
    
    /// ObserveStmt - A callback invoked right before invoking the
    ///  liveness transfer function on the given statement.
    virtual void ObserveStmt(Stmt* S, const AnalysisDataTy& AD, 
                             const ValTy& V) {}
    
    virtual void ObserverKill(DeclRefExpr* DR) {}
  };
};

class LiveVariables : public DataflowValues<LiveVariables_ValueTypes,
                                            dataflow::backward_analysis_tag> {
public:
  typedef LiveVariables_ValueTypes::ObserverTy ObserverTy;
    
  LiveVariables() {}
  
  /// IsLive - Return true if a variable is live at beginning of a
  /// specified block.
  bool isLive(const CFGBlock* B, const VarDecl* D) const;
  
  /// IsLive - Return true if a variable is live according to the
  ///  provided livness bitvector.
  bool isLive(const ValTy& V, const VarDecl* D) const;
  
  /// dumpLiveness - Print to stderr the liveness information encoded
  ///  by a specified bitvector.
  void dumpLiveness(const ValTy& V, SourceManager& M) const;
  
  /// dumpBlockLiveness - Print to stderr the liveness information
  ///  associated with each basic block.
  void dumpBlockLiveness(SourceManager& M) const;
  
  /// getNumDecls - Return the number of variables (declarations) that
  ///  whose liveness status is being tracked by the dataflow
  ///  analysis.
  unsigned getNumDecls() const { return getAnalysisData().getNumDecls(); }
    
  /// IntializeValues - Create initial dataflow values and meta data for
  ///  a given CFG.  This is intended to be called by the dataflow solver.
  void InitializeValues(const CFG& cfg);
  
  void runOnCFG(const CFG& cfg);
  void runOnAllBlocks(const CFG& cfg, ObserverTy& Obs);
};

} // end namespace clang

#endif
