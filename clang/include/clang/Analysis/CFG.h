//===--- CFG.h - Classes for representing and building CFGs------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CFG and CFGBuilder classes for representing and
//  building Control-Flow Graphs (CFGs) from ASTs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CFG_H
#define LLVM_CLANG_CFG_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "clang/Analysis/Support/BumpVector.h"
#include "clang/Basic/SourceLocation.h"
#include <cassert>

namespace llvm {
  class raw_ostream;
}
namespace clang {
  class Decl;
  class Stmt;
  class Expr;
  class CFG;
  class PrinterHelper;
  class LangOptions;
  class ASTContext;

/// CFGElement - Represents a top-level expression in a basic block.
class CFGElement {
  llvm::PointerIntPair<Stmt *, 2> Data;
public:
  enum Type { StartScope, EndScope };
  explicit CFGElement() {}
  CFGElement(Stmt *S, bool lvalue) : Data(S, lvalue ? 1 : 0) {}
  CFGElement(Stmt *S, Type t) : Data(S, t == StartScope ? 2 : 3) {}
  Stmt *getStmt() const { return Data.getPointer(); }
  bool asLValue() const { return Data.getInt() == 1; }
  bool asStartScope() const { return Data.getInt() == 2; }
  bool asEndScope() const { return Data.getInt() == 3; }
  bool asDtor() const { return Data.getInt() == 4; }
  operator Stmt*() const { return getStmt(); }
  operator bool() const { return getStmt() != 0; }
};

/// CFGBlock - Represents a single basic block in a source-level CFG.
///  It consists of:
///
///  (1) A set of statements/expressions (which may contain subexpressions).
///  (2) A "terminator" statement (not in the set of statements).
///  (3) A list of successors and predecessors.
///
/// Terminator: The terminator represents the type of control-flow that occurs
/// at the end of the basic block.  The terminator is a Stmt* referring to an
/// AST node that has control-flow: if-statements, breaks, loops, etc.
/// If the control-flow is conditional, the condition expression will appear
/// within the set of statements in the block (usually the last statement).
///
/// Predecessors: the order in the set of predecessors is arbitrary.
///
/// Successors: the order in the set of successors is NOT arbitrary.  We
///  currently have the following orderings based on the terminator:
///
///     Terminator       Successor Ordering
///  -----------------------------------------------------
///       if            Then Block;  Else Block
///     ? operator      LHS expression;  RHS expression
///     &&, ||          expression that uses result of && or ||, RHS
///
class CFGBlock {
  class StatementList {
    typedef BumpVector<CFGElement> ImplTy;
    ImplTy Impl;
  public:
    StatementList(BumpVectorContext &C) : Impl(C, 4) {}
    
    typedef std::reverse_iterator<ImplTy::iterator>       iterator;
    typedef std::reverse_iterator<ImplTy::const_iterator> const_iterator;
    typedef ImplTy::iterator                              reverse_iterator;
    typedef ImplTy::const_iterator                        const_reverse_iterator;
  
    void push_back(CFGElement e, BumpVectorContext &C) { Impl.push_back(e, C); }
    CFGElement front() const { return Impl.back(); }
    CFGElement back() const { return Impl.front(); }
    
    iterator begin() { return Impl.rbegin(); }
    iterator end() { return Impl.rend(); }
    const_iterator begin() const { return Impl.rbegin(); }
    const_iterator end() const { return Impl.rend(); }
    reverse_iterator rbegin() { return Impl.begin(); }
    reverse_iterator rend() { return Impl.end(); }
    const_reverse_iterator rbegin() const { return Impl.begin(); }
    const_reverse_iterator rend() const { return Impl.end(); }

   CFGElement operator[](size_t i) const  {
     assert(i < Impl.size());
     return Impl[Impl.size() - 1 - i];
   }
    
    size_t size() const { return Impl.size(); }
    bool empty() const { return Impl.empty(); }
  };

  /// Stmts - The set of statements in the basic block.
  StatementList Stmts;

  /// Label - An (optional) label that prefixes the executable
  ///  statements in the block.  When this variable is non-NULL, it is
  ///  either an instance of LabelStmt, SwitchCase or CXXCatchStmt.
  Stmt *Label;

  /// Terminator - The terminator for a basic block that
  ///  indicates the type of control-flow that occurs between a block
  ///  and its successors.
  Stmt *Terminator;

  /// LoopTarget - Some blocks are used to represent the "loop edge" to
  ///  the start of a loop from within the loop body.  This Stmt* will be
  ///  refer to the loop statement for such blocks (and be null otherwise).
  const Stmt *LoopTarget;

  /// BlockID - A numerical ID assigned to a CFGBlock during construction
  ///   of the CFG.
  unsigned BlockID;

  /// Predecessors/Successors - Keep track of the predecessor / successor
  /// CFG blocks.
  typedef BumpVector<CFGBlock*> AdjacentBlocks;
  AdjacentBlocks Preds;
  AdjacentBlocks Succs;

public:
  explicit CFGBlock(unsigned blockid, BumpVectorContext &C)
    : Stmts(C), Label(NULL), Terminator(NULL), LoopTarget(NULL),
      BlockID(blockid), Preds(C, 1), Succs(C, 1) {}
  ~CFGBlock() {}

  // Statement iterators
  typedef StatementList::iterator                      iterator;
  typedef StatementList::const_iterator                const_iterator;
  typedef StatementList::reverse_iterator              reverse_iterator;
  typedef StatementList::const_reverse_iterator        const_reverse_iterator;

  CFGElement                   front()       const { return Stmts.front();   }
  CFGElement                   back()        const { return Stmts.back();    }

  iterator                     begin()             { return Stmts.begin();   }
  iterator                     end()               { return Stmts.end();     }
  const_iterator               begin()       const { return Stmts.begin();   }
  const_iterator               end()         const { return Stmts.end();     }

  reverse_iterator             rbegin()            { return Stmts.rbegin();  }
  reverse_iterator             rend()              { return Stmts.rend();    }
  const_reverse_iterator       rbegin()      const { return Stmts.rbegin();  }
  const_reverse_iterator       rend()        const { return Stmts.rend();    }

  unsigned                     size()        const { return Stmts.size();    }
  bool                         empty()       const { return Stmts.empty();   }

  CFGElement operator[](size_t i) const  { return Stmts[i]; }

  // CFG iterators
  typedef AdjacentBlocks::iterator                              pred_iterator;
  typedef AdjacentBlocks::const_iterator                  const_pred_iterator;
  typedef AdjacentBlocks::reverse_iterator              pred_reverse_iterator;
  typedef AdjacentBlocks::const_reverse_iterator  const_pred_reverse_iterator;

  typedef AdjacentBlocks::iterator                              succ_iterator;
  typedef AdjacentBlocks::const_iterator                  const_succ_iterator;
  typedef AdjacentBlocks::reverse_iterator              succ_reverse_iterator;
  typedef AdjacentBlocks::const_reverse_iterator  const_succ_reverse_iterator;

  pred_iterator                pred_begin()        { return Preds.begin();   }
  pred_iterator                pred_end()          { return Preds.end();     }
  const_pred_iterator          pred_begin()  const { return Preds.begin();   }
  const_pred_iterator          pred_end()    const { return Preds.end();     }

  pred_reverse_iterator        pred_rbegin()       { return Preds.rbegin();  }
  pred_reverse_iterator        pred_rend()         { return Preds.rend();    }
  const_pred_reverse_iterator  pred_rbegin() const { return Preds.rbegin();  }
  const_pred_reverse_iterator  pred_rend()   const { return Preds.rend();    }

  succ_iterator                succ_begin()        { return Succs.begin();   }
  succ_iterator                succ_end()          { return Succs.end();     }
  const_succ_iterator          succ_begin()  const { return Succs.begin();   }
  const_succ_iterator          succ_end()    const { return Succs.end();     }

  succ_reverse_iterator        succ_rbegin()       { return Succs.rbegin();  }
  succ_reverse_iterator        succ_rend()         { return Succs.rend();    }
  const_succ_reverse_iterator  succ_rbegin() const { return Succs.rbegin();  }
  const_succ_reverse_iterator  succ_rend()   const { return Succs.rend();    }

  unsigned                     succ_size()   const { return Succs.size();    }
  bool                         succ_empty()  const { return Succs.empty();   }

  unsigned                     pred_size()   const { return Preds.size();    }
  bool                         pred_empty()  const { return Preds.empty();   }

  // Manipulation of block contents

  void setTerminator(Stmt* Statement) { Terminator = Statement; }
  void setLabel(Stmt* Statement) { Label = Statement; }
  void setLoopTarget(const Stmt *loopTarget) { LoopTarget = loopTarget; }

  Stmt* getTerminator() { return Terminator; }
  const Stmt* getTerminator() const { return Terminator; }

  Stmt* getTerminatorCondition();

  const Stmt* getTerminatorCondition() const {
    return const_cast<CFGBlock*>(this)->getTerminatorCondition();
  }

  const Stmt *getLoopTarget() const { return LoopTarget; }

  bool hasBinaryBranchTerminator() const;

  Stmt* getLabel() { return Label; }
  const Stmt* getLabel() const { return Label; }

  unsigned getBlockID() const { return BlockID; }

  void dump(const CFG *cfg, const LangOptions &LO) const;
  void print(llvm::raw_ostream &OS, const CFG* cfg, const LangOptions &LO) const;
  void printTerminator(llvm::raw_ostream &OS, const LangOptions &LO) const;
  
  void addSuccessor(CFGBlock* Block, BumpVectorContext &C) {
    if (Block)
      Block->Preds.push_back(this, C);
    Succs.push_back(Block, C);
  }
  
  void appendStmt(Stmt* Statement, BumpVectorContext &C, bool asLValue) {
      Stmts.push_back(CFGElement(Statement, asLValue), C);
  }  
  void StartScope(Stmt* S, BumpVectorContext &C) {
    Stmts.push_back(CFGElement(S, CFGElement::StartScope), C);
  }
  void EndScope(Stmt* S, BumpVectorContext &C) {
    Stmts.push_back(CFGElement(S, CFGElement::EndScope), C);
  }
};


/// CFG - Represents a source-level, intra-procedural CFG that represents the
///  control-flow of a Stmt.  The Stmt can represent an entire function body,
///  or a single expression.  A CFG will always contain one empty block that
///  represents the Exit point of the CFG.  A CFG will also contain a designated
///  Entry block.  The CFG solely represents control-flow; it consists of
///  CFGBlocks which are simply containers of Stmt*'s in the AST the CFG
///  was constructed from.
class CFG {
public:
  //===--------------------------------------------------------------------===//
  // CFG Construction & Manipulation.
  //===--------------------------------------------------------------------===//

  /// buildCFG - Builds a CFG from an AST.  The responsibility to free the
  ///   constructed CFG belongs to the caller.
  static CFG* buildCFG(const Decl *D, Stmt* AST, ASTContext *C,
                       bool pruneTriviallyFalseEdges = true,
                       bool AddEHEdges = false,
                       bool AddScopes = false /* NOT FULLY IMPLEMENTED.
                                                 NOT READY FOR GENERAL USE. */);

  /// createBlock - Create a new block in the CFG.  The CFG owns the block;
  ///  the caller should not directly free it.
  CFGBlock* createBlock();

  /// setEntry - Set the entry block of the CFG.  This is typically used
  ///  only during CFG construction.  Most CFG clients expect that the
  ///  entry block has no predecessors and contains no statements.
  void setEntry(CFGBlock *B) { Entry = B; }

  /// setIndirectGotoBlock - Set the block used for indirect goto jumps.
  ///  This is typically used only during CFG construction.
  void setIndirectGotoBlock(CFGBlock* B) { IndirectGotoBlock = B; }

  //===--------------------------------------------------------------------===//
  // Block Iterators
  //===--------------------------------------------------------------------===//

  typedef BumpVector<CFGBlock*>                    CFGBlockListTy;    
  typedef CFGBlockListTy::iterator                 iterator;
  typedef CFGBlockListTy::const_iterator           const_iterator;
  typedef std::reverse_iterator<iterator>          reverse_iterator;
  typedef std::reverse_iterator<const_iterator>    const_reverse_iterator;

  CFGBlock&                 front()                { return *Blocks.front(); }
  CFGBlock&                 back()                 { return *Blocks.back(); }

  iterator                  begin()                { return Blocks.begin(); }
  iterator                  end()                  { return Blocks.end(); }
  const_iterator            begin()       const    { return Blocks.begin(); }
  const_iterator            end()         const    { return Blocks.end(); }

  reverse_iterator          rbegin()               { return Blocks.rbegin(); }
  reverse_iterator          rend()                 { return Blocks.rend(); }
  const_reverse_iterator    rbegin()      const    { return Blocks.rbegin(); }
  const_reverse_iterator    rend()        const    { return Blocks.rend(); }

  CFGBlock&                 getEntry()             { return *Entry; }
  const CFGBlock&           getEntry()    const    { return *Entry; }
  CFGBlock&                 getExit()              { return *Exit; }
  const CFGBlock&           getExit()     const    { return *Exit; }

  CFGBlock*        getIndirectGotoBlock() { return IndirectGotoBlock; }
  const CFGBlock*  getIndirectGotoBlock() const { return IndirectGotoBlock; }

  //===--------------------------------------------------------------------===//
  // Member templates useful for various batch operations over CFGs.
  //===--------------------------------------------------------------------===//

  template <typename CALLBACK>
  void VisitBlockStmts(CALLBACK& O) const {
    for (const_iterator I=begin(), E=end(); I != E; ++I)
      for (CFGBlock::const_iterator BI=(*I)->begin(), BE=(*I)->end();
           BI != BE; ++BI)
        O(*BI);
  }

  //===--------------------------------------------------------------------===//
  // CFG Introspection.
  //===--------------------------------------------------------------------===//

  struct   BlkExprNumTy {
    const signed Idx;
    explicit BlkExprNumTy(signed idx) : Idx(idx) {}
    explicit BlkExprNumTy() : Idx(-1) {}
    operator bool() const { return Idx >= 0; }
    operator unsigned() const { assert(Idx >=0); return (unsigned) Idx; }
  };

  bool          isBlkExpr(const Stmt* S) { return getBlkExprNum(S); }
  BlkExprNumTy  getBlkExprNum(const Stmt* S);
  unsigned      getNumBlkExprs();

  /// getNumBlockIDs - Returns the total number of BlockIDs allocated (which
  /// start at 0).
  unsigned getNumBlockIDs() const { return NumBlockIDs; }

  //===--------------------------------------------------------------------===//
  // CFG Debugging: Pretty-Printing and Visualization.
  //===--------------------------------------------------------------------===//

  void viewCFG(const LangOptions &LO) const;
  void print(llvm::raw_ostream& OS, const LangOptions &LO) const;
  void dump(const LangOptions &LO) const;

  //===--------------------------------------------------------------------===//
  // Internal: constructors and data.
  //===--------------------------------------------------------------------===//

  CFG() : Entry(NULL), Exit(NULL), IndirectGotoBlock(NULL), NumBlockIDs(0),
          BlkExprMap(NULL), Blocks(BlkBVC, 10) {}

  ~CFG();

  llvm::BumpPtrAllocator& getAllocator() {
    return BlkBVC.getAllocator();
  }
  
  BumpVectorContext &getBumpVectorContext() {
    return BlkBVC;
  }

private:
  CFGBlock* Entry;
  CFGBlock* Exit;
  CFGBlock* IndirectGotoBlock;  // Special block to contain collective dispatch
                                // for indirect gotos
  unsigned  NumBlockIDs;

  // BlkExprMap - An opaque pointer to prevent inclusion of DenseMap.h.
  //  It represents a map from Expr* to integers to record the set of
  //  block-level expressions and their "statement number" in the CFG.
  void*     BlkExprMap;
  
  BumpVectorContext BlkBVC;
  
  CFGBlockListTy Blocks;

};
} // end namespace clang

//===----------------------------------------------------------------------===//
// GraphTraits specializations for CFG basic block graphs (source-level CFGs)
//===----------------------------------------------------------------------===//

namespace llvm {

/// Implement simplify_type for CFGElement, so that we can dyn_cast from
/// CFGElement to a specific Stmt class.
template <> struct simplify_type<const ::clang::CFGElement> {
  typedef ::clang::Stmt* SimpleType;
  static SimpleType getSimplifiedValue(const ::clang::CFGElement &Val) {
    return Val.getStmt();
  }
};
  
template <> struct simplify_type< ::clang::CFGElement> 
  : public simplify_type<const ::clang::CFGElement> {};
  
// Traits for: CFGBlock

template <> struct GraphTraits< ::clang::CFGBlock* > {
  typedef ::clang::CFGBlock NodeType;
  typedef ::clang::CFGBlock::succ_iterator ChildIteratorType;

  static NodeType* getEntryNode(::clang::CFGBlock* BB)
  { return BB; }

  static inline ChildIteratorType child_begin(NodeType* N)
  { return N->succ_begin(); }

  static inline ChildIteratorType child_end(NodeType* N)
  { return N->succ_end(); }
};

template <> struct GraphTraits< const ::clang::CFGBlock* > {
  typedef const ::clang::CFGBlock NodeType;
  typedef ::clang::CFGBlock::const_succ_iterator ChildIteratorType;

  static NodeType* getEntryNode(const clang::CFGBlock* BB)
  { return BB; }

  static inline ChildIteratorType child_begin(NodeType* N)
  { return N->succ_begin(); }

  static inline ChildIteratorType child_end(NodeType* N)
  { return N->succ_end(); }
};

template <> struct GraphTraits<Inverse<const ::clang::CFGBlock*> > {
  typedef const ::clang::CFGBlock NodeType;
  typedef ::clang::CFGBlock::const_pred_iterator ChildIteratorType;

  static NodeType *getEntryNode(Inverse<const ::clang::CFGBlock*> G)
  { return G.Graph; }

  static inline ChildIteratorType child_begin(NodeType* N)
  { return N->pred_begin(); }

  static inline ChildIteratorType child_end(NodeType* N)
  { return N->pred_end(); }
};

// Traits for: CFG

template <> struct GraphTraits< ::clang::CFG* >
    : public GraphTraits< ::clang::CFGBlock* >  {

  typedef ::clang::CFG::iterator nodes_iterator;

  static NodeType *getEntryNode(::clang::CFG* F) { return &F->getEntry(); }
  static nodes_iterator nodes_begin(::clang::CFG* F) { return F->begin(); }
  static nodes_iterator nodes_end(::clang::CFG* F) { return F->end(); }
};

template <> struct GraphTraits<const ::clang::CFG* >
    : public GraphTraits<const ::clang::CFGBlock* >  {

  typedef ::clang::CFG::const_iterator nodes_iterator;

  static NodeType *getEntryNode( const ::clang::CFG* F) {
    return &F->getEntry();
  }
  static nodes_iterator nodes_begin( const ::clang::CFG* F) {
    return F->begin();
  }
  static nodes_iterator nodes_end( const ::clang::CFG* F) {
    return F->end();
  }
};

template <> struct GraphTraits<Inverse<const ::clang::CFG*> >
  : public GraphTraits<Inverse<const ::clang::CFGBlock*> > {

  typedef ::clang::CFG::const_iterator nodes_iterator;

  static NodeType *getEntryNode(const ::clang::CFG* F) { return &F->getExit(); }
  static nodes_iterator nodes_begin(const ::clang::CFG* F) { return F->begin();}
  static nodes_iterator nodes_end(const ::clang::CFG* F) { return F->end(); }
};
} // end llvm namespace
#endif
