//===--- CFG.cpp - Classes for representing and building CFGs----*- C++ -*-===//
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

#include "clang/AST/CFG.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace clang;

namespace {

  // SaveAndRestore - A utility class that uses RIIA to save and restore
  //  the value of a variable.
  template<typename T>
  struct SaveAndRestore {
    SaveAndRestore(T& x) : X(x), old_value(x) {}
    ~SaveAndRestore() { X = old_value; }
    
    T& X;
    T old_value;
  };
}
  
/// CFGBuilder - This class is implements CFG construction from an AST.
///   The builder is stateful: an instance of the builder should be used to only
///   construct a single CFG.
///
///   Example usage:
///
///     CFGBuilder builder;
///     CFG* cfg = builder.BuildAST(stmt1);
///
///  CFG construction is done via a recursive walk of an AST.
///  We actually parse the AST in reverse order so that the successor
///  of a basic block is constructed prior to its predecessor.  This
///  allows us to nicely capture implicit fall-throughs without extra
///  basic blocks.
///
class CFGBuilder : public StmtVisitor<CFGBuilder,CFGBlock*> {    
  CFG* cfg;
  CFGBlock* Block;
  CFGBlock* Exit;
  CFGBlock* Succ;
  unsigned NumBlocks;
  
public:  
  explicit CFGBuilder() : cfg(NULL), Block(NULL), Exit(NULL), Succ(NULL), 
                          NumBlocks(0) {
    // Create an empty CFG.
    cfg = new CFG();                        
  }
  
  ~CFGBuilder() { delete cfg; }
  
  /// buildCFG - Constructs a CFG from an AST (a Stmt*).  The AST can
  ///  represent an arbitrary statement.  Examples include a single expression
  ///  or a function body (compound statement).  The ownership of the returned
  ///  CFG is transferred to the caller.  If CFG construction fails, this method
  ///  returns NULL.
  CFG* buildCFG(Stmt* Statement) {
    if (!Statement) return NULL;
  
    assert (cfg && "CFGBuilder should only be used to construct one CFG");

    // Create the exit block.
    Block = createBlock();
    Exit = Block;
    
    // Visit the statements and create the CFG.
    if (CFGBlock* B = Visit(Statement)) {
      // Reverse the statements in the last constructed block.  Statements
      // are inserted into the blocks in reverse order.
      B->reverseStmts();
      // NULL out cfg so that repeated calls
      CFG* t = cfg;
      cfg = NULL;
      return t;
    }
    else {
      // Error occured while building CFG: Delete the partially constructed CFG.
      delete cfg;
      cfg = NULL;
      return NULL;
    }
  }
  
  // createBlock - Used to lazily create blocks that are connected
  //  to the current (global) succcessor.
  CFGBlock* createBlock( bool add_successor = true ) { 
    CFGBlock* B = cfg->createBlock(NumBlocks++);
    if (add_successor && Succ) B->addSuccessor(Succ);
    return B;
  }

  /// Here we handle statements with no branching control flow.
  CFGBlock* VisitStmt(Stmt* Statement) {
    // We cannot assume that we are in the middle of a basic block, since
    // the CFG might only be constructed for this single statement.  If
    // we have no current basic block, just create one lazily.
    if (!Block) Block = createBlock();
    
    // Simply add the statement to the current block.  We actually
    // insert statements in reverse order; this order is reversed later
    // when processing the containing element in the AST.
    Block->appendStmt(Statement);
    
    return Block;
  }
  
  CFGBlock* VisitCompoundStmt(CompoundStmt* C) {
    //   The value returned from this function is the last created CFGBlock
    //   that represents the "entry" point for the translated AST node.
    for (CompoundStmt::reverse_body_iterator I = C->body_rbegin(),
          E = C->body_rend(); I != E; ++I )
      // Add the statement to the current block.
      if (!Visit(*I)) return NULL;

    return Block;
  }
  
  CFGBlock* VisitIfStmt(IfStmt* I) {
  
    // We may see an if statement in the middle of a basic block, or
    // it may be the first statement we are processing.  In either case,
    // we create a new basic block.  First, we create the blocks for
    // the then...else statements, and then we create the block containing
    // the if statement.  If we were in the middle of a block, we
    // stop processing that block and reverse its statements.  That block
    // is then the implicit successor for the "then" and "else" clauses.
    
    // The block we were proccessing is now finished.  Make it the
    // successor block.
    if (Block) { 
      Succ = Block;
      Block->reverseStmts();
    }
    
    // Process the false branch.  NULL out Block so that the recursive
    // call to Visit will create a new basic block.
    // Null out Block so that all successor
    CFGBlock* ElseBlock = Succ;
    
    if (Stmt* Else = I->getElse()) {
      SaveAndRestore<CFGBlock*> sv(Succ);
      
      // NULL out Block so that the recursive call to Visit will
      // create a new basic block.          
      Block = NULL;
      ElseBlock = Visit(Else);          
      if (!ElseBlock) return NULL;
      ElseBlock->reverseStmts();        
    }
    
    // Process the true branch.  NULL out Block so that the recursive
    // call to Visit will create a new basic block.
    // Null out Block so that all successor
    CFGBlock* ThenBlock;
    {
      Stmt* Then = I->getThen();
      assert (Then);
      SaveAndRestore<CFGBlock*> sv(Succ);
      Block = NULL;        
      ThenBlock = Visit(Then);        
      if (!ThenBlock) return NULL;
      ThenBlock->reverseStmts();
    }

    // Now create a new block containing the if statement.        
    Block = createBlock(false);
  
    // Add the condition as the last statement in the new block.
    Block->appendStmt(I->getCond());
    
    // Set the terminator of the new block to the If statement.
    Block->setTerminator(I);
    
    // Now add the successors.
    Block->addSuccessor(ThenBlock);
    Block->addSuccessor(ElseBlock);

    return Block;
  }
      
  CFGBlock* VisitReturnStmt(ReturnStmt* R) {
    // If we were in the middle of a block we stop processing that block
    // and reverse its statements.
    //
    // NOTE: If a "return" appears in the middle of a block, this means
    //       that the code afterwards is DEAD (unreachable).  We still
    //       keep a basic block for that code; a simple "mark-and-sweep"
    //       from the entry block will be able to report such dead
    //       blocks.
    if (Block) Block->reverseStmts();        

    // Create the new block.
    Block = createBlock(false);
    
    // The Exit block is the only successor.
    Block->addSuccessor(Exit);
    
    // Add the return expression to the block.
    Block->appendStmt(R);
    
    // Add the return statement itself to the block.
    if (R->getRetValue()) Block->appendStmt(R->getRetValue());
    
    return Block;
  }  
};

// BuildCFG - A helper function that builds CFGs from ASTS.
CFG* CFG::BuildCFG(Stmt* Statement) {
  CFGBuilder Builder;
  return Builder.buildCFG(Statement);
}

// reverseStmts - A method that reverses the order of the statements within
//  a CFGBlock.
void CFGBlock::reverseStmts() { std::reverse(Stmts.begin(),Stmts.end()); }

// dump - A simple pretty printer of a CFG that outputs to stderr.
void CFG::dump() { print(std::cerr); }

// print - A simple pretty printer of a CFG that outputs to an ostream.
void CFG::print(std::ostream& OS) {
  // Iterate through the CFGBlocks and print them one by one.  Specially
  // designate the Entry and Exit blocks.
  for (iterator I = Blocks.begin(), E = Blocks.end() ; I != E ; ++I) {
    OS << "\n  [ B" << I->getBlockID();
    if (&(*I) == getExit()) OS << " (EXIT) ]\n";
    else if (&(*I) == getEntry()) OS << " (ENTRY) ]\n";
    else OS << " ]\n";
    I->print(OS);
  }
  OS << "\n";
}

// dump - A simply pretty printer of a CFGBlock that outputs to stderr.
void CFGBlock::dump() { print(std::cerr); }

// print - A simple pretty printer of a CFGBlock that outputs to an ostream.
//   Generally this will only be called from CFG::print.
void CFGBlock::print(std::ostream& OS) {

  // Iterate through the statements in the block and print them.
  OS << "    ------------------------\n";
  unsigned j = 1;
  for (iterator I = Stmts.begin(), E = Stmts.end() ; I != E ; ++I, ++j ) {
    OS << "    " << std::setw(3) << j << ": ";
    (*I)->printPretty(OS);
    if (isa<Expr>(*I)) OS << '\n';
  }
  OS << "    ------------------------\n";

  // Print the predecessors of this block.
  OS << "    Predecessors (" << pred_size() << "):";
  unsigned i = 0;
  for (pred_iterator I = pred_begin(), E = pred_end(); I != E; ++I, ++i ) {
    if (i == 8 || (i-8) == 0) {
      OS << "\n     ";
    }
    OS << " B" << (*I)->getBlockID();
  }
  
  // Print the terminator of this block.
  OS << "\n    Terminator: ";
  if (ControlFlowStmt) {
    switch (ControlFlowStmt->getStmtClass()) {
      case Stmt::IfStmtClass: {
        IfStmt* I = cast<IfStmt>(ControlFlowStmt);
        OS << "if ";
        I->getCond()->printPretty(std::cerr);
        OS << "\n";
        break;
      }
      
      case Stmt::ReturnStmtClass: {
        ReturnStmt* R = cast<ReturnStmt>(ControlFlowStmt);
        R->printPretty(std::cerr);
        break;
      }
      
      default:
        assert(false && "terminator print not fully implemented");
    }
  }
  else OS << "<NULL>\n";

  // Print the successors of this block.
  OS << "    Successors (" << succ_size() << "):";
  i = 0;
  for (succ_iterator I = succ_begin(), E = succ_end(); I != E; ++I, ++i ) {
    if (i == 8 || (i-8) % 10 == 0) {
      OS << "\n    ";
    }
    OS << " B" << (*I)->getBlockID();
  }
  OS << '\n';
}