//===--- CFG.h - Classes for representing and building CFGs------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CFG and CFGBuilder classes for representing and
//  building Control-Flow Graphs (CFGs) from ASTs.
//
//===----------------------------------------------------------------------===//

#include <list>
#include <vector>
#include <iosfwd>

namespace clang {

class Stmt;
  
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
///
class CFGBlock {
  typedef std::vector<Stmt*> StatementListTy;
  /// Stmts - The set of statements in the basic block.
  StatementListTy Stmts;
  /// ControlFlowStmt - The terminator for a basic block that
  ///  indicates the type of control-flow that occurs between a block
  ///  and its successors.
  Stmt* ControlFlowStmt;
  /// BlockID - A numerical ID assigned to a CFGBlock during construction
  ///   of the CFG.
  unsigned BlockID;
  
  /// Predecessors/Successors - Keep track of the predecessor / successor
  /// CFG blocks.
  typedef std::vector<CFGBlock*> AdjacentBlocks;
  AdjacentBlocks Preds;
  AdjacentBlocks Succs;
  
public:
  explicit CFGBlock(unsigned blockid) : ControlFlowStmt(NULL), 
                                        BlockID(blockid) {}
  ~CFGBlock() {};

  // Statement iterators
  typedef StatementListTy::iterator                                  iterator;
  typedef StatementListTy::const_iterator                      const_iterator;
  typedef std::reverse_iterator<const_iterator>        const_reverse_iterator;
  typedef std::reverse_iterator<iterator>                    reverse_iterator;
  
  Stmt*                        front()             { return Stmts.front();   }
  Stmt*                        back()              { return Stmts.back();    }
  
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
  
  void appendStmt(Stmt* Statement) { Stmts.push_back(Statement); }
  void setTerminator(Stmt* Statement) { ControlFlowStmt = Statement; }
  void reverseStmts();
  
  void addSuccessor(CFGBlock* Block) {
    Block->Preds.push_back(this);
    Succs.push_back(Block);
  }
  
  unsigned getBlockID() const { return BlockID; }
  
  void dump();
  void print(std::ostream& OS);
};
  

/// CFG - Represents a source-level, intra-procedural CFG that represents the
///  control-flow of a Stmt.  The Stmt can represent an entire function body,
///  or a single expression.  A CFG will always contain one empty block that
///  represents the Exit point of the CFG.  A CFG will also contain a designated
///  Entry block.  The CFG solely represents control-flow; it consists of
///  CFGBlocks which are simply containers of Stmt*'s in the AST the CFG
///  was constructed from.
class CFG {
  typedef std::list<CFGBlock> CFGBlockListTy;
  CFGBlockListTy Blocks;

public:

  CFG() {};
  ~CFG() {};
  
  CFGBlock* createBlock(unsigned blockID) { 
    Blocks.push_front(CFGBlock(blockID));
    return front();
  }
  
  // Block iterators
  typedef CFGBlockListTy::iterator                                    iterator;
  typedef CFGBlockListTy::const_iterator                        const_iterator;
  typedef std::reverse_iterator<iterator>                     reverse_iterator;
  typedef std::reverse_iterator<const_iterator>         const_reverse_iterator;

  CFGBlock*                    front()             { return &Blocks.front();  }
  CFGBlock*                    back()              { return &Blocks.back();   }
  
  iterator                     begin()             { return Blocks.begin();   }
  iterator                     end()               { return Blocks.end();     }
  const_iterator               begin()       const { return Blocks.begin();   }
  const_iterator               end()         const { return Blocks.end();     } 
  
  reverse_iterator             rbegin()            { return Blocks.rbegin();  }
  reverse_iterator             rend()              { return Blocks.rend();    }
  const_reverse_iterator       rbegin()      const { return Blocks.rbegin();  }
  const_reverse_iterator       rend()        const { return Blocks.rend();    }
  
  CFGBlock*                    getEntry()          { return front();          }
  CFGBlock*                    getExit()           { return back();           }
  
  // Utility
  
  static CFG* BuildCFG(Stmt* AST);
  void print(std::ostream& OS);
  void dump();
      
};

} // end namespace clang