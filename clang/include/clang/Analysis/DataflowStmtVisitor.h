//===--- DataFlowStmtVisitor.h - StmtVisitor for Dataflow -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the DataflowStmtVisitor interface, which extends 
//  CFGStmtVisitor.  This interface is useful for visiting statements in a CFG
//  with the understanding that statements are walked in order of the analysis
//  traversal.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_DATAFLOW_STMTVISITOR_H
#define LLVM_CLANG_ANALYSIS_DATAFLOW_STMTVISITOR_H

#include "clang/Analysis/CFGStmtVisitor.h"

namespace clang {

// Tag classes describing what direction the dataflow analysis goes.
namespace dataflow {
  struct forward_analysis_tag {};
  struct backward_analysis_tag {};
}  

template < typename ImplClass, 
           typename AnalysisTag=dataflow::forward_analysis_tag >
class DataflowStmtVisitor : public CFGStmtVisitor<ImplClass,void> {
public:
  //===--------------------------------------------------------------------===//
  // Observer methods.  These are called before a statement is visited, and
  //  there is no special dispatch on statement type.  This allows subclasses
  //  to inject extra functionality (e.g. monitoring) that applies to all
  //  visited statements.
  //===--------------------------------------------------------------------===//
  
  void ObserveStmt(Stmt* S) {}
  
  void ObserveBlockStmt(Stmt* S) {
    static_cast<ImplClass*>(this)->ObserveStmt(S);
  }
  
  //===--------------------------------------------------------------------===//
  // Statment visitor methods.  These modify the behavior of CFGVisitor::Visit
  //  and CFGVisitor::BlockStmt_Visit by performing a traversal of substatements
  //  depending on the direction of the dataflow analysis.  For forward
  //  analyses, the traversal is postorder (representing evaluation order)
  //  and for backward analysis it is preorder (reverse-evaluation order).
  //===--------------------------------------------------------------------===//

  void BlockStmt_Visit(Stmt* S) { BlockStmt_Visit(S,AnalysisTag()); }

  void BlockStmt_Visit(Stmt* S, dataflow::forward_analysis_tag) {      
    // Process statements in a postorder traversal of the AST. 
    if (!CFG::hasImplicitControlFlow(S) &&
        S->getStmtClass() != Stmt::CallExprClass)
      static_cast<ImplClass*>(this)->VisitChildren(S);
      
    static_cast<ImplClass*>(this)->ObserveBlockStmt(S);
    static_cast<CFGStmtVisitor<ImplClass,void>*>(this)->BlockStmt_Visit(S);
  }
  
  void BlockStmt_Visit(Stmt* S, dataflow::backward_analysis_tag) {
    // Process statements in a preorder traversal of the AST.
    static_cast<ImplClass*>(this)->ObserveBlockStmt(S);
    static_cast<CFGStmtVisitor<ImplClass,void>*>(this)->BlockStmt_Visit(S);
    
    if (!CFG::hasImplicitControlFlow(S) &&
        S->getStmtClass() != Stmt::CallExprClass)
      static_cast<ImplClass*>(this)->VisitChildren(S);
  }
  
  void Visit(Stmt* S) { Visit(S,AnalysisTag()); }
  
  void Visit(Stmt* S, dataflow::forward_analysis_tag) {
    if (CFG::hasImplicitControlFlow(S))
      return;
      
    // Process statements in a postorder traversal of the AST.
    static_cast<ImplClass*>(this)->VisitChildren(S);
    static_cast<ImplClass*>(this)->ObserveStmt(S);
    static_cast<CFGStmtVisitor<ImplClass,void>*>(this)->Visit(S);
  }
  
  void Visit(Stmt* S, dataflow::backward_analysis_tag) {
    if (CFG::hasImplicitControlFlow(S))
      return;
      
    // Process statements in a preorder traversal of the AST.
    static_cast<ImplClass*>(this)->ObserveStmt(S);
    static_cast<CFGStmtVisitor<ImplClass,void>*>(this)->Visit(S);
    static_cast<ImplClass*>(this)->VisitChildren(S);
  }    
  
  //===--------------------------------------------------------------------===//
  // Methods for visiting entire CFGBlocks.
  //===--------------------------------------------------------------------===//
  
  void VisitBlockEntry(const CFGBlock* B) {}
  void VisitBlockExit(const CFGBlock* B) {}
  
  void VisitBlock(const CFGBlock* B) { VisitBlock(B,AnalysisTag()); }
  
  void VisitBlock(const CFGBlock* B, dataflow::forward_analysis_tag ) {
    static_cast<ImplClass*>(this)->VisitBlockEntry(B);
    
    for (CFGBlock::const_iterator I=B->begin(), E=B->end(); I!=E; ++I)
      static_cast<ImplClass*>(this)->BlockStmt_Visit(const_cast<Stmt*>(*I));
    
    static_cast<ImplClass*>(this)->VisitBlockExit(B);
  }
  
  void VisitBlock(const CFGBlock* B, dataflow::backward_analysis_tag ) {
    static_cast<ImplClass*>(this)->VisitBlockExit(B);
        
    for (CFGBlock::const_reverse_iterator I=B->rbegin(), E=B->rend(); I!=E; ++I)
      static_cast<ImplClass*>(this)->BlockStmt_Visit(const_cast<Stmt*>(*I));
   
    static_cast<ImplClass*>(this)->VisitBlockEntry(B);
  }
  
};  

}  // end namespace clang

#endif
