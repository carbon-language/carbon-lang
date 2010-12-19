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
#include <iterator>

namespace llvm {
  class raw_ostream;
}

namespace clang {
  class Decl;
  class Stmt;
  class Expr;
  class FieldDecl;
  class VarDecl;
  class CXXBaseOrMemberInitializer;
  class CXXBaseSpecifier;
  class CXXBindTemporaryExpr;
  class CFG;
  class PrinterHelper;
  class LangOptions;
  class ASTContext;

/// CFGElement - Represents a top-level expression in a basic block.
class CFGElement {
public:
  enum Kind {
    // main kind
    Statement,
    Initializer,
    ImplicitDtor,
    // dtor kind
    AutomaticObjectDtor,
    BaseDtor,
    MemberDtor,
    TemporaryDtor,
    DTOR_BEGIN = AutomaticObjectDtor
  };

protected:
  // The int bits are used to mark the main kind.
  llvm::PointerIntPair<void *, 2> Data1;
  // The int bits are used to mark the dtor kind.
  llvm::PointerIntPair<void *, 2> Data2;

  CFGElement(void *Ptr, unsigned Int) : Data1(Ptr, Int) {}
  CFGElement(void *Ptr1, unsigned Int1, void *Ptr2, unsigned Int2)
      : Data1(Ptr1, Int1), Data2(Ptr2, Int2) {}

public:
  CFGElement() {}

  Kind getKind() const { return static_cast<Kind>(Data1.getInt()); }

  Kind getDtorKind() const {
    assert(getKind() == ImplicitDtor);
    return static_cast<Kind>(Data2.getInt() + DTOR_BEGIN);
  }

  bool isValid() const { return Data1.getPointer(); }

  operator bool() const { return isValid(); }

  template<class ElemTy> ElemTy getAs() const {
    if (llvm::isa<ElemTy>(this))
      return *static_cast<const ElemTy*>(this);
    return ElemTy();
  }

  static bool classof(const CFGElement *E) { return true; }
};

class CFGStmt : public CFGElement {
public:
  CFGStmt() {}
  CFGStmt(Stmt *S) : CFGElement(S, 0) {}

  Stmt *getStmt() const { return static_cast<Stmt *>(Data1.getPointer()); }

  operator Stmt*() const { return getStmt(); }

  static bool classof(const CFGElement *E) {
    return E->getKind() == Statement;
  }
};

/// CFGInitializer - Represents C++ base or member initializer from
/// constructor's initialization list.
class CFGInitializer : public CFGElement {
public:
  CFGInitializer() {}
  CFGInitializer(CXXBaseOrMemberInitializer* I)
      : CFGElement(I, Initializer) {}

  CXXBaseOrMemberInitializer* getInitializer() const {
    return static_cast<CXXBaseOrMemberInitializer*>(Data1.getPointer());
  }
  operator CXXBaseOrMemberInitializer*() const { return getInitializer(); }

  static bool classof(const CFGElement *E) {
    return E->getKind() == Initializer;
  }
};

/// CFGImplicitDtor - Represents C++ object destructor implicitly generated
/// by compiler on various occasions.
class CFGImplicitDtor : public CFGElement {
protected:
  CFGImplicitDtor(unsigned K, void* P, void* S)
      : CFGElement(P, ImplicitDtor, S, K - DTOR_BEGIN) {}

public:
  CFGImplicitDtor() {}

  static bool classof(const CFGElement *E) {
    return E->getKind() == ImplicitDtor;
  }
};

/// CFGAutomaticObjDtor - Represents C++ object destructor implicitly generated
/// for automatic object or temporary bound to const reference at the point
/// of leaving its local scope.
class CFGAutomaticObjDtor: public CFGImplicitDtor {
public:
  CFGAutomaticObjDtor() {}
  CFGAutomaticObjDtor(VarDecl* VD, Stmt* S)
      : CFGImplicitDtor(AutomaticObjectDtor, VD, S) {}

  VarDecl* getVarDecl() const {
    return static_cast<VarDecl*>(Data1.getPointer());
  }

  // Get statement end of which triggered the destructor call.
  Stmt* getTriggerStmt() const {
    return static_cast<Stmt*>(Data2.getPointer());
  }

  static bool classof(const CFGElement *E) {
    return E->getKind() == ImplicitDtor && 
           E->getDtorKind() == AutomaticObjectDtor;
  }
};

/// CFGBaseDtor - Represents C++ object destructor implicitly generated for
/// base object in destructor.
class CFGBaseDtor : public CFGImplicitDtor {
public:
  CFGBaseDtor() {}
  CFGBaseDtor(const CXXBaseSpecifier *BS)
      : CFGImplicitDtor(BaseDtor, const_cast<CXXBaseSpecifier*>(BS), NULL) {}

  const CXXBaseSpecifier *getBaseSpecifier() const {
    return static_cast<const CXXBaseSpecifier*>(Data1.getPointer());
  }

  static bool classof(const CFGElement *E) {
    return E->getKind() == ImplicitDtor && E->getDtorKind() == BaseDtor;
  }
};

/// CFGMemberDtor - Represents C++ object destructor implicitly generated for
/// member object in destructor.
class CFGMemberDtor : public CFGImplicitDtor {
public:
  CFGMemberDtor() {}
  CFGMemberDtor(FieldDecl *FD)
      : CFGImplicitDtor(MemberDtor, FD, NULL) {}

  FieldDecl *getFieldDecl() const {
    return static_cast<FieldDecl*>(Data1.getPointer());
  }

  static bool classof(const CFGElement *E) {
    return E->getKind() == ImplicitDtor && E->getDtorKind() == MemberDtor;
  }
};

/// CFGTemporaryDtor - Represents C++ object destructor implicitly generated
/// at the end of full expression for temporary object.
class CFGTemporaryDtor : public CFGImplicitDtor {
public:
  CFGTemporaryDtor() {}
  CFGTemporaryDtor(CXXBindTemporaryExpr *E)
      : CFGImplicitDtor(TemporaryDtor, E, NULL) {}

  CXXBindTemporaryExpr *getBindTemporaryExpr() const {
    return static_cast<CXXBindTemporaryExpr *>(Data1.getPointer());
  }

  static bool classof(const CFGElement *E) {
    return E->getKind() == ImplicitDtor && E->getDtorKind() == TemporaryDtor;
  }
};

/// CFGTerminator - Represents CFGBlock terminator statement.
///
/// TemporaryDtorsBranch bit is set to true if the terminator marks a branch
/// in control flow of destructors of temporaries. In this case terminator
/// statement is the same statement that branches control flow in evaluation
/// of matching full expression.
class CFGTerminator {
  llvm::PointerIntPair<Stmt *, 1> Data;
public:
  CFGTerminator() {}
  CFGTerminator(Stmt *S, bool TemporaryDtorsBranch = false)
      : Data(S, TemporaryDtorsBranch) {}

  Stmt *getStmt() { return Data.getPointer(); }
  const Stmt *getStmt() const { return Data.getPointer(); }

  bool isTemporaryDtorsBranch() const { return Data.getInt(); }

  operator Stmt *() { return getStmt(); }
  operator const Stmt *() const { return getStmt(); }

  Stmt *operator->() { return getStmt(); }
  const Stmt *operator->() const { return getStmt(); }

  Stmt &operator*() { return *getStmt(); }
  const Stmt &operator*() const { return *getStmt(); }

  operator bool() const { return getStmt(); }
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
  class ElementList {
    typedef BumpVector<CFGElement> ImplTy;
    ImplTy Impl;
  public:
    ElementList(BumpVectorContext &C) : Impl(C, 4) {}
    
    typedef std::reverse_iterator<ImplTy::iterator>       iterator;
    typedef std::reverse_iterator<ImplTy::const_iterator> const_iterator;
    typedef ImplTy::iterator                              reverse_iterator;
    typedef ImplTy::const_iterator                        const_reverse_iterator;
  
    void push_back(CFGElement e, BumpVectorContext &C) { Impl.push_back(e, C); }
    reverse_iterator insert(reverse_iterator I, size_t Cnt, CFGElement E,
        BumpVectorContext& C) {
      return Impl.insert(I, Cnt, E, C);
    }

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
  ElementList Elements;

  /// Label - An (optional) label that prefixes the executable
  ///  statements in the block.  When this variable is non-NULL, it is
  ///  either an instance of LabelStmt, SwitchCase or CXXCatchStmt.
  Stmt *Label;

  /// Terminator - The terminator for a basic block that
  ///  indicates the type of control-flow that occurs between a block
  ///  and its successors.
  CFGTerminator Terminator;

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
    : Elements(C), Label(NULL), Terminator(NULL), LoopTarget(NULL),
      BlockID(blockid), Preds(C, 1), Succs(C, 1) {}
  ~CFGBlock() {}

  // Statement iterators
  typedef ElementList::iterator                      iterator;
  typedef ElementList::const_iterator                const_iterator;
  typedef ElementList::reverse_iterator              reverse_iterator;
  typedef ElementList::const_reverse_iterator        const_reverse_iterator;

  CFGElement                 front()       const { return Elements.front();   }
  CFGElement                 back()        const { return Elements.back();    }

  iterator                   begin()             { return Elements.begin();   }
  iterator                   end()               { return Elements.end();     }
  const_iterator             begin()       const { return Elements.begin();   }
  const_iterator             end()         const { return Elements.end();     }

  reverse_iterator           rbegin()            { return Elements.rbegin();  }
  reverse_iterator           rend()              { return Elements.rend();    }
  const_reverse_iterator     rbegin()      const { return Elements.rbegin();  }
  const_reverse_iterator     rend()        const { return Elements.rend();    }

  unsigned                   size()        const { return Elements.size();    }
  bool                       empty()       const { return Elements.empty();   }

  CFGElement operator[](size_t i) const  { return Elements[i]; }

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


  class FilterOptions {
  public:
    FilterOptions() {
      IgnoreDefaultsWithCoveredEnums = 0;
    }

    unsigned IgnoreDefaultsWithCoveredEnums : 1;
  };

  static bool FilterEdge(const FilterOptions &F, const CFGBlock *Src,
       const CFGBlock *Dst);

  template <typename IMPL, bool IsPred>
  class FilteredCFGBlockIterator {
  private:
    IMPL I, E;
    const FilterOptions F;
    const CFGBlock *From;
   public:
    explicit FilteredCFGBlockIterator(const IMPL &i, const IMPL &e,
              const CFGBlock *from,
              const FilterOptions &f)
      : I(i), E(e), F(f), From(from) {}

    bool hasMore() const { return I != E; }

    FilteredCFGBlockIterator &operator++() {
      do { ++I; } while (hasMore() && Filter(*I));
      return *this;
    }

    const CFGBlock *operator*() const { return *I; }
  private:
    bool Filter(const CFGBlock *To) {
      return IsPred ? FilterEdge(F, To, From) : FilterEdge(F, From, To);
    }
  };

  typedef FilteredCFGBlockIterator<const_pred_iterator, true>
          filtered_pred_iterator;

  typedef FilteredCFGBlockIterator<const_succ_iterator, false>
          filtered_succ_iterator;

  filtered_pred_iterator filtered_pred_start_end(const FilterOptions &f) const {
    return filtered_pred_iterator(pred_begin(), pred_end(), this, f);
  }

  filtered_succ_iterator filtered_succ_start_end(const FilterOptions &f) const {
    return filtered_succ_iterator(succ_begin(), succ_end(), this, f);
  }

  // Manipulation of block contents

  void setTerminator(Stmt* Statement) { Terminator = Statement; }
  void setLabel(Stmt* Statement) { Label = Statement; }
  void setLoopTarget(const Stmt *loopTarget) { LoopTarget = loopTarget; }

  CFGTerminator getTerminator() { return Terminator; }
  const CFGTerminator getTerminator() const { return Terminator; }

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
  
  void appendStmt(Stmt* statement, BumpVectorContext &C) {
    Elements.push_back(CFGStmt(statement), C);
  }

  void appendInitializer(CXXBaseOrMemberInitializer *initializer,
                        BumpVectorContext& C) {
    Elements.push_back(CFGInitializer(initializer), C);
  }

  void appendBaseDtor(const CXXBaseSpecifier *BS, BumpVectorContext &C) {
    Elements.push_back(CFGBaseDtor(BS), C);
  }

  void appendMemberDtor(FieldDecl *FD, BumpVectorContext &C) {
    Elements.push_back(CFGMemberDtor(FD), C);
  }
  
  void appendTemporaryDtor(CXXBindTemporaryExpr *E, BumpVectorContext &C) {
    Elements.push_back(CFGTemporaryDtor(E), C);
  }

  // Destructors must be inserted in reversed order. So insertion is in two
  // steps. First we prepare space for some number of elements, then we insert
  // the elements beginning at the last position in prepared space.
  iterator beginAutomaticObjDtorsInsert(iterator I, size_t Cnt,
      BumpVectorContext& C) {
    return iterator(Elements.insert(I.base(), Cnt, CFGElement(), C));
  }
  iterator insertAutomaticObjDtor(iterator I, VarDecl* VD, Stmt* S) {
    *I = CFGAutomaticObjDtor(VD, S);
    return ++I;
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

  class BuildOptions {
  public:
    bool PruneTriviallyFalseEdges:1;
    bool AddEHEdges:1;
    bool AddInitializers:1;
    bool AddImplicitDtors:1;

    BuildOptions()
        : PruneTriviallyFalseEdges(true)
        , AddEHEdges(false)
        , AddInitializers(false)
        , AddImplicitDtors(false) {}
  };

  /// buildCFG - Builds a CFG from an AST.  The responsibility to free the
  ///   constructed CFG belongs to the caller.
  static CFG* buildCFG(const Decl *D, Stmt* AST, ASTContext *C,
      BuildOptions BO = BuildOptions());

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
           BI != BE; ++BI) {
        if (CFGStmt S = BI->getAs<CFGStmt>())
          O(S);
      }
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

/// Implement simplify_type for CFGTerminator, so that we can dyn_cast from
/// CFGTerminator to a specific Stmt class.
template <> struct simplify_type<const ::clang::CFGTerminator> {
  typedef const ::clang::Stmt *SimpleType;
  static SimpleType getSimplifiedValue(const ::clang::CFGTerminator &Val) {
    return Val.getStmt();
  }
};

template <> struct simplify_type< ::clang::CFGTerminator> {
  typedef ::clang::Stmt *SimpleType;
  static SimpleType getSimplifiedValue(const ::clang::CFGTerminator &Val) {
    return const_cast<SimpleType>(Val.getStmt());
  }
};

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
