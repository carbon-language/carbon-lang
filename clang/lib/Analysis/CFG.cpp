//===--- CFG.cpp - Classes for representing and building CFGs----*- C++ -*-===//
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

#include "clang/Analysis/Support/SaveAndRestore.h"
#include "clang/Analysis/CFG.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Format.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/OwningPtr.h"

using namespace clang;

namespace {

static SourceLocation GetEndLoc(Decl* D) {
  if (VarDecl* VD = dyn_cast<VarDecl>(D))
    if (Expr* Ex = VD->getInit())
      return Ex->getSourceRange().getEnd();

  return D->getLocation();
}
  
class AddStmtChoice {
public:
  enum Kind { NotAlwaysAdd = 0, AlwaysAdd, AlwaysAddAsLValue };
public:
  AddStmtChoice(Kind kind) : k(kind) {}  
  bool alwaysAdd() const { return k != NotAlwaysAdd; }
  bool asLValue() const { return k == AlwaysAddAsLValue; }
private:
  Kind k;
};

/// CFGBuilder - This class implements CFG construction from an AST.
///   The builder is stateful: an instance of the builder should be used to only
///   construct a single CFG.
///
///   Example usage:
///
///     CFGBuilder builder;
///     CFG* cfg = builder.BuildAST(stmt1);
///
///  CFG construction is done via a recursive walk of an AST.  We actually parse
///  the AST in reverse order so that the successor of a basic block is
///  constructed prior to its predecessor.  This allows us to nicely capture
///  implicit fall-throughs without extra basic blocks.
///
class CFGBuilder {
  ASTContext *Context;
  llvm::OwningPtr<CFG> cfg;

  CFGBlock* Block;
  CFGBlock* Succ;
  CFGBlock* ContinueTargetBlock;
  CFGBlock* BreakTargetBlock;
  CFGBlock* SwitchTerminatedBlock;
  CFGBlock* DefaultCaseBlock;
  CFGBlock* TryTerminatedBlock;

  // LabelMap records the mapping from Label expressions to their blocks.
  typedef llvm::DenseMap<LabelStmt*,CFGBlock*> LabelMapTy;
  LabelMapTy LabelMap;

  // A list of blocks that end with a "goto" that must be backpatched to their
  // resolved targets upon completion of CFG construction.
  typedef std::vector<CFGBlock*> BackpatchBlocksTy;
  BackpatchBlocksTy BackpatchBlocks;

  // A list of labels whose address has been taken (for indirect gotos).
  typedef llvm::SmallPtrSet<LabelStmt*,5> LabelSetTy;
  LabelSetTy AddressTakenLabels;

public:
  explicit CFGBuilder() : cfg(new CFG()), // crew a new CFG
                          Block(NULL), Succ(NULL),
                          ContinueTargetBlock(NULL), BreakTargetBlock(NULL),
                          SwitchTerminatedBlock(NULL), DefaultCaseBlock(NULL),
                          TryTerminatedBlock(NULL) {}

  // buildCFG - Used by external clients to construct the CFG.
  CFG* buildCFG(const Decl *D, Stmt *Statement, ASTContext *C, bool AddEHEdges,
                bool AddScopes);

private:
  // Visitors to walk an AST and construct the CFG.
  CFGBlock *VisitAddrLabelExpr(AddrLabelExpr *A, AddStmtChoice asc);
  CFGBlock *VisitBinaryOperator(BinaryOperator *B, AddStmtChoice asc);
  CFGBlock *VisitBlockExpr(BlockExpr* E, AddStmtChoice asc);
  CFGBlock *VisitBreakStmt(BreakStmt *B);
  CFGBlock *VisitCallExpr(CallExpr *C, AddStmtChoice asc);
  CFGBlock *VisitCaseStmt(CaseStmt *C);
  CFGBlock *VisitChooseExpr(ChooseExpr *C, AddStmtChoice asc);
  CFGBlock *VisitCompoundStmt(CompoundStmt *C);
  CFGBlock *VisitConditionalOperator(ConditionalOperator *C,
                                     AddStmtChoice asc);
  CFGBlock *VisitContinueStmt(ContinueStmt *C);
  CFGBlock *VisitCXXCatchStmt(CXXCatchStmt *S);
  CFGBlock *VisitCXXThrowExpr(CXXThrowExpr *T);
  CFGBlock *VisitCXXTryStmt(CXXTryStmt *S);
  CFGBlock *VisitDeclStmt(DeclStmt *DS);
  CFGBlock *VisitDeclSubExpr(Decl* D);
  CFGBlock *VisitDefaultStmt(DefaultStmt *D);
  CFGBlock *VisitDoStmt(DoStmt *D);
  CFGBlock *VisitForStmt(ForStmt *F);
  CFGBlock *VisitGotoStmt(GotoStmt* G);
  CFGBlock *VisitIfStmt(IfStmt *I);
  CFGBlock *VisitIndirectGotoStmt(IndirectGotoStmt *I);
  CFGBlock *VisitLabelStmt(LabelStmt *L);
  CFGBlock *VisitObjCAtCatchStmt(ObjCAtCatchStmt *S);
  CFGBlock *VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S);
  CFGBlock *VisitObjCAtThrowStmt(ObjCAtThrowStmt *S);
  CFGBlock *VisitObjCAtTryStmt(ObjCAtTryStmt *S);
  CFGBlock *VisitObjCForCollectionStmt(ObjCForCollectionStmt *S);
  CFGBlock *VisitReturnStmt(ReturnStmt* R);
  CFGBlock *VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E, AddStmtChoice asc);
  CFGBlock *VisitStmtExpr(StmtExpr *S, AddStmtChoice asc);
  CFGBlock *VisitSwitchStmt(SwitchStmt *S);
  CFGBlock *VisitWhileStmt(WhileStmt *W);

  CFGBlock *Visit(Stmt *S, AddStmtChoice asc = AddStmtChoice::NotAlwaysAdd);
  CFGBlock *VisitStmt(Stmt *S, AddStmtChoice asc);
  CFGBlock *VisitChildren(Stmt* S);

  // NYS == Not Yet Supported
  CFGBlock* NYS() {
    badCFG = true;
    return Block;
  }

  CFGBlock *StartScope(Stmt *S, CFGBlock *B) {
    if (!AddScopes)
      return B;

    if (B == 0)
      B = createBlock();
    B->StartScope(S, cfg->getBumpVectorContext());
    return B;
  }

  void EndScope(Stmt *S) {
    if (!AddScopes)
      return;

    if (Block == 0)
      Block = createBlock();
    Block->EndScope(S, cfg->getBumpVectorContext());
  }

  void autoCreateBlock() { if (!Block) Block = createBlock(); }
  CFGBlock *createBlock(bool add_successor = true);
  bool FinishBlock(CFGBlock* B);
  CFGBlock *addStmt(Stmt *S, AddStmtChoice asc = AddStmtChoice::AlwaysAdd) {
    return Visit(S, asc);
  }
  
  void AppendStmt(CFGBlock *B, Stmt *S,
                  AddStmtChoice asc = AddStmtChoice::AlwaysAdd) {
    B->appendStmt(S, cfg->getBumpVectorContext(), asc.asLValue());
  }
  
  void AddSuccessor(CFGBlock *B, CFGBlock *S) {
    B->addSuccessor(S, cfg->getBumpVectorContext());
  }

  /// TryResult - a class representing a variant over the values
  ///  'true', 'false', or 'unknown'.  This is returned by TryEvaluateBool,
  ///  and is used by the CFGBuilder to decide if a branch condition
  ///  can be decided up front during CFG construction.
  class TryResult {
    int X;
  public:
    TryResult(bool b) : X(b ? 1 : 0) {}
    TryResult() : X(-1) {}

    bool isTrue() const { return X == 1; }
    bool isFalse() const { return X == 0; }
    bool isKnown() const { return X >= 0; }
    void negate() {
      assert(isKnown());
      X ^= 0x1;
    }
  };

  /// TryEvaluateBool - Try and evaluate the Stmt and return 0 or 1
  /// if we can evaluate to a known value, otherwise return -1.
  TryResult TryEvaluateBool(Expr *S) {
    Expr::EvalResult Result;
    if (!S->isTypeDependent() && !S->isValueDependent() &&
        S->Evaluate(Result, *Context) && Result.Val.isInt())
      return Result.Val.getInt().getBoolValue();

    return TryResult();
  }

  bool badCFG;

  // True iff EH edges on CallExprs should be added to the CFG.
  bool AddEHEdges;

  // True iff scope start and scope end notes should be added to the CFG.
  bool AddScopes;
};

// FIXME: Add support for dependent-sized array types in C++?
// Does it even make sense to build a CFG for an uninstantiated template?
static VariableArrayType* FindVA(Type* t) {
  while (ArrayType* vt = dyn_cast<ArrayType>(t)) {
    if (VariableArrayType* vat = dyn_cast<VariableArrayType>(vt))
      if (vat->getSizeExpr())
        return vat;

    t = vt->getElementType().getTypePtr();
  }

  return 0;
}

/// BuildCFG - Constructs a CFG from an AST (a Stmt*).  The AST can represent an
///  arbitrary statement.  Examples include a single expression or a function
///  body (compound statement).  The ownership of the returned CFG is
///  transferred to the caller.  If CFG construction fails, this method returns
///  NULL.
CFG* CFGBuilder::buildCFG(const Decl *D, Stmt* Statement, ASTContext* C,
                          bool addehedges, bool AddScopes) {
  AddEHEdges = addehedges;
  Context = C;
  assert(cfg.get());
  if (!Statement)
    return NULL;

  this->AddScopes = AddScopes;
  badCFG = false;

  // Create an empty block that will serve as the exit block for the CFG.  Since
  // this is the first block added to the CFG, it will be implicitly registered
  // as the exit block.
  Succ = createBlock();
  assert(Succ == &cfg->getExit());
  Block = NULL;  // the EXIT block is empty.  Create all other blocks lazily.

  // Visit the statements and create the CFG.
  CFGBlock* B = addStmt(Statement);

  if (const CXXConstructorDecl *CD = dyn_cast_or_null<CXXConstructorDecl>(D)) {
    // FIXME: Add code for base initializers and member initializers.
    (void)CD;
  }
  if (!B)
    B = Succ;

  if (B) {
    // Finalize the last constructed block.  This usually involves reversing the
    // order of the statements in the block.
    if (Block) FinishBlock(B);

    // Backpatch the gotos whose label -> block mappings we didn't know when we
    // encountered them.
    for (BackpatchBlocksTy::iterator I = BackpatchBlocks.begin(),
         E = BackpatchBlocks.end(); I != E; ++I ) {

      CFGBlock* B = *I;
      GotoStmt* G = cast<GotoStmt>(B->getTerminator());
      LabelMapTy::iterator LI = LabelMap.find(G->getLabel());

      // If there is no target for the goto, then we are looking at an
      // incomplete AST.  Handle this by not registering a successor.
      if (LI == LabelMap.end()) continue;

      AddSuccessor(B, LI->second);
    }

    // Add successors to the Indirect Goto Dispatch block (if we have one).
    if (CFGBlock* B = cfg->getIndirectGotoBlock())
      for (LabelSetTy::iterator I = AddressTakenLabels.begin(),
           E = AddressTakenLabels.end(); I != E; ++I ) {

        // Lookup the target block.
        LabelMapTy::iterator LI = LabelMap.find(*I);

        // If there is no target block that contains label, then we are looking
        // at an incomplete AST.  Handle this by not registering a successor.
        if (LI == LabelMap.end()) continue;

        AddSuccessor(B, LI->second);
      }

    Succ = B;
  }

  // Create an empty entry block that has no predecessors.
  cfg->setEntry(createBlock());

  return badCFG ? NULL : cfg.take();
}

/// createBlock - Used to lazily create blocks that are connected
///  to the current (global) succcessor.
CFGBlock* CFGBuilder::createBlock(bool add_successor) {
  CFGBlock* B = cfg->createBlock();
  if (add_successor && Succ)
    AddSuccessor(B, Succ);
  return B;
}

/// FinishBlock - "Finalize" the block by checking if we have a bad CFG.
bool CFGBuilder::FinishBlock(CFGBlock* B) {
  if (badCFG)
    return false;

  assert(B);
  return true;
}

/// Visit - Walk the subtree of a statement and add extra
///   blocks for ternary operators, &&, and ||.  We also process "," and
///   DeclStmts (which may contain nested control-flow).
CFGBlock* CFGBuilder::Visit(Stmt * S, AddStmtChoice asc) {
tryAgain:
  switch (S->getStmtClass()) {
    default:
      return VisitStmt(S, asc);

    case Stmt::AddrLabelExprClass:
      return VisitAddrLabelExpr(cast<AddrLabelExpr>(S), asc);

    case Stmt::BinaryOperatorClass:
      return VisitBinaryOperator(cast<BinaryOperator>(S), asc);

    case Stmt::BlockExprClass:
      return VisitBlockExpr(cast<BlockExpr>(S), asc);

    case Stmt::BreakStmtClass:
      return VisitBreakStmt(cast<BreakStmt>(S));

    case Stmt::CallExprClass:
      return VisitCallExpr(cast<CallExpr>(S), asc);

    case Stmt::CaseStmtClass:
      return VisitCaseStmt(cast<CaseStmt>(S));

    case Stmt::ChooseExprClass:
      return VisitChooseExpr(cast<ChooseExpr>(S), asc);

    case Stmt::CompoundStmtClass:
      return VisitCompoundStmt(cast<CompoundStmt>(S));

    case Stmt::ConditionalOperatorClass:
      return VisitConditionalOperator(cast<ConditionalOperator>(S), asc);

    case Stmt::ContinueStmtClass:
      return VisitContinueStmt(cast<ContinueStmt>(S));

    case Stmt::CXXCatchStmtClass:
      return VisitCXXCatchStmt(cast<CXXCatchStmt>(S));

    case Stmt::CXXThrowExprClass:
      return VisitCXXThrowExpr(cast<CXXThrowExpr>(S));
      
    case Stmt::CXXTryStmtClass:
      return VisitCXXTryStmt(cast<CXXTryStmt>(S));
      
    case Stmt::DeclStmtClass:
      return VisitDeclStmt(cast<DeclStmt>(S));

    case Stmt::DefaultStmtClass:
      return VisitDefaultStmt(cast<DefaultStmt>(S));

    case Stmt::DoStmtClass:
      return VisitDoStmt(cast<DoStmt>(S));

    case Stmt::ForStmtClass:
      return VisitForStmt(cast<ForStmt>(S));

    case Stmt::GotoStmtClass:
      return VisitGotoStmt(cast<GotoStmt>(S));

    case Stmt::IfStmtClass:
      return VisitIfStmt(cast<IfStmt>(S));

    case Stmt::IndirectGotoStmtClass:
      return VisitIndirectGotoStmt(cast<IndirectGotoStmt>(S));

    case Stmt::LabelStmtClass:
      return VisitLabelStmt(cast<LabelStmt>(S));

    case Stmt::ObjCAtCatchStmtClass:
      return VisitObjCAtCatchStmt(cast<ObjCAtCatchStmt>(S));

    case Stmt::ObjCAtSynchronizedStmtClass:
      return VisitObjCAtSynchronizedStmt(cast<ObjCAtSynchronizedStmt>(S));

    case Stmt::ObjCAtThrowStmtClass:
      return VisitObjCAtThrowStmt(cast<ObjCAtThrowStmt>(S));

    case Stmt::ObjCAtTryStmtClass:
      return VisitObjCAtTryStmt(cast<ObjCAtTryStmt>(S));

    case Stmt::ObjCForCollectionStmtClass:
      return VisitObjCForCollectionStmt(cast<ObjCForCollectionStmt>(S));

    case Stmt::ParenExprClass:
      S = cast<ParenExpr>(S)->getSubExpr();
      goto tryAgain;

    case Stmt::NullStmtClass:
      return Block;

    case Stmt::ReturnStmtClass:
      return VisitReturnStmt(cast<ReturnStmt>(S));

    case Stmt::SizeOfAlignOfExprClass:
      return VisitSizeOfAlignOfExpr(cast<SizeOfAlignOfExpr>(S), asc);

    case Stmt::StmtExprClass:
      return VisitStmtExpr(cast<StmtExpr>(S), asc);

    case Stmt::SwitchStmtClass:
      return VisitSwitchStmt(cast<SwitchStmt>(S));

    case Stmt::WhileStmtClass:
      return VisitWhileStmt(cast<WhileStmt>(S));
  }
}

CFGBlock *CFGBuilder::VisitStmt(Stmt *S, AddStmtChoice asc) {
  if (asc.alwaysAdd()) {
    autoCreateBlock();
    AppendStmt(Block, S, asc);
  }

  return VisitChildren(S);
}

/// VisitChildren - Visit the children of a Stmt.
CFGBlock *CFGBuilder::VisitChildren(Stmt* Terminator) {
  CFGBlock *B = Block;
  for (Stmt::child_iterator I = Terminator->child_begin(),
         E = Terminator->child_end(); I != E; ++I) {
    if (*I) B = Visit(*I);
  }
  return B;
}

CFGBlock *CFGBuilder::VisitAddrLabelExpr(AddrLabelExpr *A,
                                         AddStmtChoice asc) {
  AddressTakenLabels.insert(A->getLabel());

  if (asc.alwaysAdd()) {
    autoCreateBlock();
    AppendStmt(Block, A, asc);
  }

  return Block;
}

CFGBlock *CFGBuilder::VisitBinaryOperator(BinaryOperator *B,
                                          AddStmtChoice asc) {
  if (B->isLogicalOp()) { // && or ||
    CFGBlock* ConfluenceBlock = Block ? Block : createBlock();
    AppendStmt(ConfluenceBlock, B, asc);

    if (!FinishBlock(ConfluenceBlock))
      return 0;

    // create the block evaluating the LHS
    CFGBlock* LHSBlock = createBlock(false);
    LHSBlock->setTerminator(B);

    // create the block evaluating the RHS
    Succ = ConfluenceBlock;
    Block = NULL;
    CFGBlock* RHSBlock = addStmt(B->getRHS());
    if (!FinishBlock(RHSBlock))
      return 0;

    // See if this is a known constant.
    TryResult KnownVal = TryEvaluateBool(B->getLHS());
    if (KnownVal.isKnown() && (B->getOpcode() == BinaryOperator::LOr))
      KnownVal.negate();

    // Now link the LHSBlock with RHSBlock.
    if (B->getOpcode() == BinaryOperator::LOr) {
      AddSuccessor(LHSBlock, KnownVal.isTrue() ? NULL : ConfluenceBlock);
      AddSuccessor(LHSBlock, KnownVal.isFalse() ? NULL : RHSBlock);
    } else {
      assert(B->getOpcode() == BinaryOperator::LAnd);
      AddSuccessor(LHSBlock, KnownVal.isFalse() ? NULL : RHSBlock);
      AddSuccessor(LHSBlock, KnownVal.isTrue() ? NULL : ConfluenceBlock);
    }

    // Generate the blocks for evaluating the LHS.
    Block = LHSBlock;
    return addStmt(B->getLHS());
  }
  else if (B->getOpcode() == BinaryOperator::Comma) { // ,
    autoCreateBlock();
    AppendStmt(Block, B, asc);
    addStmt(B->getRHS());
    return addStmt(B->getLHS());
  }

  return VisitStmt(B, asc);
}

CFGBlock *CFGBuilder::VisitBlockExpr(BlockExpr *E, AddStmtChoice asc) {
  if (asc.alwaysAdd()) {
    autoCreateBlock();
    AppendStmt(Block, E, asc);
  }
  return Block;
}

CFGBlock *CFGBuilder::VisitBreakStmt(BreakStmt *B) {
  // "break" is a control-flow statement.  Thus we stop processing the current
  // block.
  if (Block && !FinishBlock(Block))
      return 0;

  // Now create a new block that ends with the break statement.
  Block = createBlock(false);
  Block->setTerminator(B);

  // If there is no target for the break, then we are looking at an incomplete
  // AST.  This means that the CFG cannot be constructed.
  if (BreakTargetBlock)
    AddSuccessor(Block, BreakTargetBlock);
  else
    badCFG = true;


  return Block;
}

static bool CanThrow(Expr *E) {
  QualType Ty = E->getType();
  if (Ty->isFunctionPointerType())
    Ty = Ty->getAs<PointerType>()->getPointeeType();
  else if (Ty->isBlockPointerType())
    Ty = Ty->getAs<BlockPointerType>()->getPointeeType();
    
  const FunctionType *FT = Ty->getAs<FunctionType>();
  if (FT) {
    if (const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(FT))
      if (Proto->hasEmptyExceptionSpec())
        return false;
  }
  return true;
}

CFGBlock *CFGBuilder::VisitCallExpr(CallExpr *C, AddStmtChoice asc) {
  // If this is a call to a no-return function, this stops the block here.
  bool NoReturn = false;
  if (C->getCallee()->getType().getNoReturnAttr()) {
    NoReturn = true;
  }

  bool AddEHEdge = false;

  // Languages without exceptions are assumed to not throw.
  if (Context->getLangOptions().Exceptions) {
    if (AddEHEdges)
      AddEHEdge = true;
  }

  if (FunctionDecl *FD = C->getDirectCallee()) {
    if (FD->hasAttr<NoReturnAttr>())
      NoReturn = true;
    if (FD->hasAttr<NoThrowAttr>())
      AddEHEdge = false;
  }

  if (!CanThrow(C->getCallee()))
    AddEHEdge = false;

  if (!NoReturn && !AddEHEdge)
    return VisitStmt(C, asc);

  if (Block) {
    Succ = Block;
    if (!FinishBlock(Block))
      return 0;
  }

  Block = createBlock(!NoReturn);
  AppendStmt(Block, C, asc);

  if (NoReturn) {
    // Wire this to the exit block directly.
    AddSuccessor(Block, &cfg->getExit());
  }
  if (AddEHEdge) {
    // Add exceptional edges.
    if (TryTerminatedBlock)
      AddSuccessor(Block, TryTerminatedBlock);
    else
      AddSuccessor(Block, &cfg->getExit());
  }

  return VisitChildren(C);
}

CFGBlock *CFGBuilder::VisitChooseExpr(ChooseExpr *C,
                                      AddStmtChoice asc) {
  CFGBlock* ConfluenceBlock = Block ? Block : createBlock();
  AppendStmt(ConfluenceBlock, C, asc);
  if (!FinishBlock(ConfluenceBlock))
    return 0;

  Succ = ConfluenceBlock;
  Block = NULL;
  CFGBlock* LHSBlock = addStmt(C->getLHS());
  if (!FinishBlock(LHSBlock))
    return 0;

  Succ = ConfluenceBlock;
  Block = NULL;
  CFGBlock* RHSBlock = addStmt(C->getRHS());
  if (!FinishBlock(RHSBlock))
    return 0;

  Block = createBlock(false);
  // See if this is a known constant.
  const TryResult& KnownVal = TryEvaluateBool(C->getCond());
  AddSuccessor(Block, KnownVal.isFalse() ? NULL : LHSBlock);
  AddSuccessor(Block, KnownVal.isTrue() ? NULL : RHSBlock);
  Block->setTerminator(C);
  return addStmt(C->getCond());
}


CFGBlock* CFGBuilder::VisitCompoundStmt(CompoundStmt* C) {
  EndScope(C);

  CFGBlock* LastBlock = Block;

  for (CompoundStmt::reverse_body_iterator I=C->body_rbegin(), E=C->body_rend();
       I != E; ++I ) {
    LastBlock = addStmt(*I);

    if (badCFG)
      return NULL;
  }

  LastBlock = StartScope(C, LastBlock);

  return LastBlock;
}

CFGBlock *CFGBuilder::VisitConditionalOperator(ConditionalOperator *C,
                                               AddStmtChoice asc) {
  // Create the confluence block that will "merge" the results of the ternary
  // expression.
  CFGBlock* ConfluenceBlock = Block ? Block : createBlock();
  AppendStmt(ConfluenceBlock, C, asc);
  if (!FinishBlock(ConfluenceBlock))
    return 0;

  // Create a block for the LHS expression if there is an LHS expression.  A
  // GCC extension allows LHS to be NULL, causing the condition to be the
  // value that is returned instead.
  //  e.g: x ?: y is shorthand for: x ? x : y;
  Succ = ConfluenceBlock;
  Block = NULL;
  CFGBlock* LHSBlock = NULL;
  if (C->getLHS()) {
    LHSBlock = addStmt(C->getLHS());
    if (!FinishBlock(LHSBlock))
      return 0;
    Block = NULL;
  }

  // Create the block for the RHS expression.
  Succ = ConfluenceBlock;
  CFGBlock* RHSBlock = addStmt(C->getRHS());
  if (!FinishBlock(RHSBlock))
    return 0;

  // Create the block that will contain the condition.
  Block = createBlock(false);

  // See if this is a known constant.
  const TryResult& KnownVal = TryEvaluateBool(C->getCond());
  if (LHSBlock) {
    AddSuccessor(Block, KnownVal.isFalse() ? NULL : LHSBlock);
  } else {
    if (KnownVal.isFalse()) {
      // If we know the condition is false, add NULL as the successor for
      // the block containing the condition.  In this case, the confluence
      // block will have just one predecessor.
      AddSuccessor(Block, 0);
      assert(ConfluenceBlock->pred_size() == 1);
    } else {
      // If we have no LHS expression, add the ConfluenceBlock as a direct
      // successor for the block containing the condition.  Moreover, we need to
      // reverse the order of the predecessors in the ConfluenceBlock because
      // the RHSBlock will have been added to the succcessors already, and we
      // want the first predecessor to the the block containing the expression
      // for the case when the ternary expression evaluates to true.
      AddSuccessor(Block, ConfluenceBlock);
      assert(ConfluenceBlock->pred_size() == 2);
      std::reverse(ConfluenceBlock->pred_begin(),
                   ConfluenceBlock->pred_end());
    }
  }

  AddSuccessor(Block, KnownVal.isTrue() ? NULL : RHSBlock);
  Block->setTerminator(C);
  return addStmt(C->getCond());
}

CFGBlock *CFGBuilder::VisitDeclStmt(DeclStmt *DS) {
  autoCreateBlock();

  if (DS->isSingleDecl()) {
    AppendStmt(Block, DS);
    return VisitDeclSubExpr(DS->getSingleDecl());
  }

  CFGBlock *B = 0;

  // FIXME: Add a reverse iterator for DeclStmt to avoid this extra copy.
  typedef llvm::SmallVector<Decl*,10> BufTy;
  BufTy Buf(DS->decl_begin(), DS->decl_end());

  for (BufTy::reverse_iterator I = Buf.rbegin(), E = Buf.rend(); I != E; ++I) {
    // Get the alignment of the new DeclStmt, padding out to >=8 bytes.
    unsigned A = llvm::AlignOf<DeclStmt>::Alignment < 8
               ? 8 : llvm::AlignOf<DeclStmt>::Alignment;

    // Allocate the DeclStmt using the BumpPtrAllocator.  It will get
    // automatically freed with the CFG.
    DeclGroupRef DG(*I);
    Decl *D = *I;
    void *Mem = cfg->getAllocator().Allocate(sizeof(DeclStmt), A);
    DeclStmt *DSNew = new (Mem) DeclStmt(DG, D->getLocation(), GetEndLoc(D));

    // Append the fake DeclStmt to block.
    AppendStmt(Block, DSNew);
    B = VisitDeclSubExpr(D);
  }

  return B;
}

/// VisitDeclSubExpr - Utility method to add block-level expressions for
///  initializers in Decls.
CFGBlock *CFGBuilder::VisitDeclSubExpr(Decl* D) {
  assert(Block);

  VarDecl *VD = dyn_cast<VarDecl>(D);

  if (!VD)
    return Block;

  Expr *Init = VD->getInit();

  if (Init) {
    // Optimization: Don't create separate block-level statements for literals.
    switch (Init->getStmtClass()) {
      case Stmt::IntegerLiteralClass:
      case Stmt::CharacterLiteralClass:
      case Stmt::StringLiteralClass:
        break;
      default:
        Block = addStmt(Init,
                        VD->getType()->isReferenceType()
                        ? AddStmtChoice::AlwaysAddAsLValue
                        : AddStmtChoice::AlwaysAdd);
    }
  }

  // If the type of VD is a VLA, then we must process its size expressions.
  for (VariableArrayType* VA = FindVA(VD->getType().getTypePtr()); VA != 0;
       VA = FindVA(VA->getElementType().getTypePtr()))
    Block = addStmt(VA->getSizeExpr());

  return Block;
}

CFGBlock* CFGBuilder::VisitIfStmt(IfStmt* I) {
  // We may see an if statement in the middle of a basic block, or it may be the
  // first statement we are processing.  In either case, we create a new basic
  // block.  First, we create the blocks for the then...else statements, and
  // then we create the block containing the if statement.  If we were in the
  // middle of a block, we stop processing that block.  That block is then the
  // implicit successor for the "then" and "else" clauses.

  // The block we were proccessing is now finished.  Make it the successor
  // block.
  if (Block) {
    Succ = Block;
    if (!FinishBlock(Block))
      return 0;
  }

  // Process the false branch.
  CFGBlock* ElseBlock = Succ;

  if (Stmt* Else = I->getElse()) {
    SaveAndRestore<CFGBlock*> sv(Succ);

    // NULL out Block so that the recursive call to Visit will
    // create a new basic block.
    Block = NULL;
    ElseBlock = addStmt(Else);

    if (!ElseBlock) // Can occur when the Else body has all NullStmts.
      ElseBlock = sv.get();
    else if (Block) {
      if (!FinishBlock(ElseBlock))
        return 0;
    }
  }

  // Process the true branch.
  CFGBlock* ThenBlock;
  {
    Stmt* Then = I->getThen();
    assert(Then);
    SaveAndRestore<CFGBlock*> sv(Succ);
    Block = NULL;
    ThenBlock = addStmt(Then);

    if (!ThenBlock) {
      // We can reach here if the "then" body has all NullStmts.
      // Create an empty block so we can distinguish between true and false
      // branches in path-sensitive analyses.
      ThenBlock = createBlock(false);
      AddSuccessor(ThenBlock, sv.get());
    } else if (Block) {
      if (!FinishBlock(ThenBlock))
        return 0;
    }
  }

  // Now create a new block containing the if statement.
  Block = createBlock(false);

  // Set the terminator of the new block to the If statement.
  Block->setTerminator(I);

  // See if this is a known constant.
  const TryResult &KnownVal = TryEvaluateBool(I->getCond());

  // Now add the successors.
  AddSuccessor(Block, KnownVal.isFalse() ? NULL : ThenBlock);
  AddSuccessor(Block, KnownVal.isTrue()? NULL : ElseBlock);

  // Add the condition as the last statement in the new block.  This may create
  // new blocks as the condition may contain control-flow.  Any newly created
  // blocks will be pointed to be "Block".
  Block = addStmt(I->getCond());
  
  // Finally, if the IfStmt contains a condition variable, add both the IfStmt
  // and the condition variable initialization to the CFG.
  if (VarDecl *VD = I->getConditionVariable()) {
    if (Expr *Init = VD->getInit()) {
      autoCreateBlock();
      AppendStmt(Block, I, AddStmtChoice::AlwaysAdd);
      addStmt(Init);
    }
  }
  
  return Block;
}


CFGBlock* CFGBuilder::VisitReturnStmt(ReturnStmt* R) {
  // If we were in the middle of a block we stop processing that block.
  //
  // NOTE: If a "return" appears in the middle of a block, this means that the
  //       code afterwards is DEAD (unreachable).  We still keep a basic block
  //       for that code; a simple "mark-and-sweep" from the entry block will be
  //       able to report such dead blocks.
  if (Block)
    FinishBlock(Block);

  // Create the new block.
  Block = createBlock(false);

  // The Exit block is the only successor.
  AddSuccessor(Block, &cfg->getExit());

  // Add the return statement to the block.  This may create new blocks if R
  // contains control-flow (short-circuit operations).
  return VisitStmt(R, AddStmtChoice::AlwaysAdd);
}

CFGBlock* CFGBuilder::VisitLabelStmt(LabelStmt* L) {
  // Get the block of the labeled statement.  Add it to our map.
  addStmt(L->getSubStmt());
  CFGBlock* LabelBlock = Block;

  if (!LabelBlock)              // This can happen when the body is empty, i.e.
    LabelBlock = createBlock(); // scopes that only contains NullStmts.

  assert(LabelMap.find(L) == LabelMap.end() && "label already in map");
  LabelMap[ L ] = LabelBlock;

  // Labels partition blocks, so this is the end of the basic block we were
  // processing (L is the block's label).  Because this is label (and we have
  // already processed the substatement) there is no extra control-flow to worry
  // about.
  LabelBlock->setLabel(L);
  if (!FinishBlock(LabelBlock))
    return 0;

  // We set Block to NULL to allow lazy creation of a new block (if necessary);
  Block = NULL;

  // This block is now the implicit successor of other blocks.
  Succ = LabelBlock;

  return LabelBlock;
}

CFGBlock* CFGBuilder::VisitGotoStmt(GotoStmt* G) {
  // Goto is a control-flow statement.  Thus we stop processing the current
  // block and create a new one.
  if (Block)
    FinishBlock(Block);

  Block = createBlock(false);
  Block->setTerminator(G);

  // If we already know the mapping to the label block add the successor now.
  LabelMapTy::iterator I = LabelMap.find(G->getLabel());

  if (I == LabelMap.end())
    // We will need to backpatch this block later.
    BackpatchBlocks.push_back(Block);
  else
    AddSuccessor(Block, I->second);

  return Block;
}

CFGBlock* CFGBuilder::VisitForStmt(ForStmt* F) {
  CFGBlock* LoopSuccessor = NULL;

  // "for" is a control-flow statement.  Thus we stop processing the current
  // block.
  if (Block) {
    if (!FinishBlock(Block))
      return 0;
    LoopSuccessor = Block;
  } else
    LoopSuccessor = Succ;

  // Because of short-circuit evaluation, the condition of the loop can span
  // multiple basic blocks.  Thus we need the "Entry" and "Exit" blocks that
  // evaluate the condition.
  CFGBlock* ExitConditionBlock = createBlock(false);
  CFGBlock* EntryConditionBlock = ExitConditionBlock;

  // Set the terminator for the "exit" condition block.
  ExitConditionBlock->setTerminator(F);

  // Now add the actual condition to the condition block.  Because the condition
  // itself may contain control-flow, new blocks may be created.
  if (Stmt* C = F->getCond()) {
    Block = ExitConditionBlock;
    EntryConditionBlock = addStmt(C);
    assert(Block == EntryConditionBlock);

    // If this block contains a condition variable, add both the condition
    // variable and initializer to the CFG.
    if (VarDecl *VD = F->getConditionVariable()) {
      if (Expr *Init = VD->getInit()) {
        autoCreateBlock();
        AppendStmt(Block, F, AddStmtChoice::AlwaysAdd);
        EntryConditionBlock = addStmt(Init);
        assert(Block == EntryConditionBlock);
      }
    }
    
    if (Block) {
      if (!FinishBlock(EntryConditionBlock))
        return 0;
    }
  }

  // The condition block is the implicit successor for the loop body as well as
  // any code above the loop.
  Succ = EntryConditionBlock;

  // See if this is a known constant.
  TryResult KnownVal(true);

  if (F->getCond())
    KnownVal = TryEvaluateBool(F->getCond());

  // Now create the loop body.
  {
    assert(F->getBody());

    // Save the current values for Block, Succ, and continue and break targets
    SaveAndRestore<CFGBlock*> save_Block(Block), save_Succ(Succ),
      save_continue(ContinueTargetBlock),
      save_break(BreakTargetBlock);

    // Create a new block to contain the (bottom) of the loop body.
    Block = NULL;

    if (Stmt* I = F->getInc()) {
      // Generate increment code in its own basic block.  This is the target of
      // continue statements.
      Succ = addStmt(I);
    } else {
      // No increment code.  Create a special, empty, block that is used as the
      // target block for "looping back" to the start of the loop.
      assert(Succ == EntryConditionBlock);
      Succ = createBlock();
    }

    // Finish up the increment (or empty) block if it hasn't been already.
    if (Block) {
      assert(Block == Succ);
      if (!FinishBlock(Block))
        return 0;
      Block = 0;
    }

    ContinueTargetBlock = Succ;

    // The starting block for the loop increment is the block that should
    // represent the 'loop target' for looping back to the start of the loop.
    ContinueTargetBlock->setLoopTarget(F);

    // All breaks should go to the code following the loop.
    BreakTargetBlock = LoopSuccessor;

    // Now populate the body block, and in the process create new blocks as we
    // walk the body of the loop.
    CFGBlock* BodyBlock = addStmt(F->getBody());

    if (!BodyBlock)
      BodyBlock = ContinueTargetBlock; // can happen for "for (...;...;...) ;"
    else if (Block && !FinishBlock(BodyBlock))
      return 0;

    // This new body block is a successor to our "exit" condition block.
    AddSuccessor(ExitConditionBlock, KnownVal.isFalse() ? NULL : BodyBlock);
  }

  // Link up the condition block with the code that follows the loop.  (the
  // false branch).
  AddSuccessor(ExitConditionBlock, KnownVal.isTrue() ? NULL : LoopSuccessor);

  // If the loop contains initialization, create a new block for those
  // statements.  This block can also contain statements that precede the loop.
  if (Stmt* I = F->getInit()) {
    Block = createBlock();
    return addStmt(I);
  } else {
    // There is no loop initialization.  We are thus basically a while loop.
    // NULL out Block to force lazy block construction.
    Block = NULL;
    Succ = EntryConditionBlock;
    return EntryConditionBlock;
  }
}

CFGBlock* CFGBuilder::VisitObjCForCollectionStmt(ObjCForCollectionStmt* S) {
  // Objective-C fast enumeration 'for' statements:
  //  http://developer.apple.com/documentation/Cocoa/Conceptual/ObjectiveC
  //
  //  for ( Type newVariable in collection_expression ) { statements }
  //
  //  becomes:
  //
  //   prologue:
  //     1. collection_expression
  //     T. jump to loop_entry
  //   loop_entry:
  //     1. side-effects of element expression
  //     1. ObjCForCollectionStmt [performs binding to newVariable]
  //     T. ObjCForCollectionStmt  TB, FB  [jumps to TB if newVariable != nil]
  //   TB:
  //     statements
  //     T. jump to loop_entry
  //   FB:
  //     what comes after
  //
  //  and
  //
  //  Type existingItem;
  //  for ( existingItem in expression ) { statements }
  //
  //  becomes:
  //
  //   the same with newVariable replaced with existingItem; the binding works
  //   the same except that for one ObjCForCollectionStmt::getElement() returns
  //   a DeclStmt and the other returns a DeclRefExpr.
  //

  CFGBlock* LoopSuccessor = 0;

  if (Block) {
    if (!FinishBlock(Block))
      return 0;
    LoopSuccessor = Block;
    Block = 0;
  } else
    LoopSuccessor = Succ;

  // Build the condition blocks.
  CFGBlock* ExitConditionBlock = createBlock(false);
  CFGBlock* EntryConditionBlock = ExitConditionBlock;

  // Set the terminator for the "exit" condition block.
  ExitConditionBlock->setTerminator(S);

  // The last statement in the block should be the ObjCForCollectionStmt, which
  // performs the actual binding to 'element' and determines if there are any
  // more items in the collection.
  AppendStmt(ExitConditionBlock, S);
  Block = ExitConditionBlock;

  // Walk the 'element' expression to see if there are any side-effects.  We
  // generate new blocks as necesary.  We DON'T add the statement by default to
  // the CFG unless it contains control-flow.
  EntryConditionBlock = Visit(S->getElement(), AddStmtChoice::NotAlwaysAdd);
  if (Block) {
    if (!FinishBlock(EntryConditionBlock))
      return 0;
    Block = 0;
  }

  // The condition block is the implicit successor for the loop body as well as
  // any code above the loop.
  Succ = EntryConditionBlock;

  // Now create the true branch.
  {
    // Save the current values for Succ, continue and break targets.
    SaveAndRestore<CFGBlock*> save_Succ(Succ),
      save_continue(ContinueTargetBlock), save_break(BreakTargetBlock);

    BreakTargetBlock = LoopSuccessor;
    ContinueTargetBlock = EntryConditionBlock;

    CFGBlock* BodyBlock = addStmt(S->getBody());

    if (!BodyBlock)
      BodyBlock = EntryConditionBlock; // can happen for "for (X in Y) ;"
    else if (Block) {
      if (!FinishBlock(BodyBlock))
        return 0;
    }

    // This new body block is a successor to our "exit" condition block.
    AddSuccessor(ExitConditionBlock, BodyBlock);
  }

  // Link up the condition block with the code that follows the loop.
  // (the false branch).
  AddSuccessor(ExitConditionBlock, LoopSuccessor);

  // Now create a prologue block to contain the collection expression.
  Block = createBlock();
  return addStmt(S->getCollection());
}

CFGBlock* CFGBuilder::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt* S) {
  // FIXME: Add locking 'primitives' to CFG for @synchronized.

  // Inline the body.
  CFGBlock *SyncBlock = addStmt(S->getSynchBody());

  // The sync body starts its own basic block.  This makes it a little easier
  // for diagnostic clients.
  if (SyncBlock) {
    if (!FinishBlock(SyncBlock))
      return 0;

    Block = 0;
  }

  Succ = SyncBlock;

  // Inline the sync expression.
  return addStmt(S->getSynchExpr());
}

CFGBlock* CFGBuilder::VisitObjCAtTryStmt(ObjCAtTryStmt* S) {
  // FIXME
  return NYS();
}

CFGBlock* CFGBuilder::VisitWhileStmt(WhileStmt* W) {
  CFGBlock* LoopSuccessor = NULL;

  // "while" is a control-flow statement.  Thus we stop processing the current
  // block.
  if (Block) {
    if (!FinishBlock(Block))
      return 0;
    LoopSuccessor = Block;
  } else
    LoopSuccessor = Succ;

  // Because of short-circuit evaluation, the condition of the loop can span
  // multiple basic blocks.  Thus we need the "Entry" and "Exit" blocks that
  // evaluate the condition.
  CFGBlock* ExitConditionBlock = createBlock(false);
  CFGBlock* EntryConditionBlock = ExitConditionBlock;

  // Set the terminator for the "exit" condition block.
  ExitConditionBlock->setTerminator(W);

  // Now add the actual condition to the condition block.  Because the condition
  // itself may contain control-flow, new blocks may be created.  Thus we update
  // "Succ" after adding the condition.
  if (Stmt* C = W->getCond()) {
    Block = ExitConditionBlock;
    EntryConditionBlock = addStmt(C);
    assert(Block == EntryConditionBlock);
    
    // If this block contains a condition variable, add both the condition
    // variable and initializer to the CFG.
    if (VarDecl *VD = W->getConditionVariable()) {
      if (Expr *Init = VD->getInit()) {
        autoCreateBlock();
        AppendStmt(Block, W, AddStmtChoice::AlwaysAdd);
        EntryConditionBlock = addStmt(Init);
        assert(Block == EntryConditionBlock);
      }
    }

    if (Block) {
      if (!FinishBlock(EntryConditionBlock))
        return 0;
    }
  }

  // The condition block is the implicit successor for the loop body as well as
  // any code above the loop.
  Succ = EntryConditionBlock;

  // See if this is a known constant.
  const TryResult& KnownVal = TryEvaluateBool(W->getCond());

  // Process the loop body.
  {
    assert(W->getBody());

    // Save the current values for Block, Succ, and continue and break targets
    SaveAndRestore<CFGBlock*> save_Block(Block), save_Succ(Succ),
                              save_continue(ContinueTargetBlock),
                              save_break(BreakTargetBlock);

    // Create an empty block to represent the transition block for looping back
    // to the head of the loop.
    Block = 0;
    assert(Succ == EntryConditionBlock);
    Succ = createBlock();
    Succ->setLoopTarget(W);
    ContinueTargetBlock = Succ;

    // All breaks should go to the code following the loop.
    BreakTargetBlock = LoopSuccessor;

    // NULL out Block to force lazy instantiation of blocks for the body.
    Block = NULL;

    // Create the body.  The returned block is the entry to the loop body.
    CFGBlock* BodyBlock = addStmt(W->getBody());

    if (!BodyBlock)
      BodyBlock = ContinueTargetBlock; // can happen for "while(...) ;"
    else if (Block) {
      if (!FinishBlock(BodyBlock))
        return 0;
    }

    // Add the loop body entry as a successor to the condition.
    AddSuccessor(ExitConditionBlock, KnownVal.isFalse() ? NULL : BodyBlock);
  }

  // Link up the condition block with the code that follows the loop.  (the
  // false branch).
  AddSuccessor(ExitConditionBlock, KnownVal.isTrue() ? NULL : LoopSuccessor);

  // There can be no more statements in the condition block since we loop back
  // to this block.  NULL out Block to force lazy creation of another block.
  Block = NULL;

  // Return the condition block, which is the dominating block for the loop.
  Succ = EntryConditionBlock;
  return EntryConditionBlock;
}


CFGBlock *CFGBuilder::VisitObjCAtCatchStmt(ObjCAtCatchStmt* S) {
  // FIXME: For now we pretend that @catch and the code it contains does not
  //  exit.
  return Block;
}

CFGBlock* CFGBuilder::VisitObjCAtThrowStmt(ObjCAtThrowStmt* S) {
  // FIXME: This isn't complete.  We basically treat @throw like a return
  //  statement.

  // If we were in the middle of a block we stop processing that block.
  if (Block && !FinishBlock(Block))
    return 0;

  // Create the new block.
  Block = createBlock(false);

  // The Exit block is the only successor.
  AddSuccessor(Block, &cfg->getExit());

  // Add the statement to the block.  This may create new blocks if S contains
  // control-flow (short-circuit operations).
  return VisitStmt(S, AddStmtChoice::AlwaysAdd);
}

CFGBlock* CFGBuilder::VisitCXXThrowExpr(CXXThrowExpr* T) {
  // If we were in the middle of a block we stop processing that block.
  if (Block && !FinishBlock(Block))
    return 0;

  // Create the new block.
  Block = createBlock(false);

  if (TryTerminatedBlock)
    // The current try statement is the only successor.
    AddSuccessor(Block, TryTerminatedBlock);
  else 
    // otherwise the Exit block is the only successor.
    AddSuccessor(Block, &cfg->getExit());

  // Add the statement to the block.  This may create new blocks if S contains
  // control-flow (short-circuit operations).
  return VisitStmt(T, AddStmtChoice::AlwaysAdd);
}

CFGBlock *CFGBuilder::VisitDoStmt(DoStmt* D) {
  CFGBlock* LoopSuccessor = NULL;

  // "do...while" is a control-flow statement.  Thus we stop processing the
  // current block.
  if (Block) {
    if (!FinishBlock(Block))
      return 0;
    LoopSuccessor = Block;
  } else
    LoopSuccessor = Succ;

  // Because of short-circuit evaluation, the condition of the loop can span
  // multiple basic blocks.  Thus we need the "Entry" and "Exit" blocks that
  // evaluate the condition.
  CFGBlock* ExitConditionBlock = createBlock(false);
  CFGBlock* EntryConditionBlock = ExitConditionBlock;

  // Set the terminator for the "exit" condition block.
  ExitConditionBlock->setTerminator(D);

  // Now add the actual condition to the condition block.  Because the condition
  // itself may contain control-flow, new blocks may be created.
  if (Stmt* C = D->getCond()) {
    Block = ExitConditionBlock;
    EntryConditionBlock = addStmt(C);
    if (Block) {
      if (!FinishBlock(EntryConditionBlock))
        return 0;
    }
  }

  // The condition block is the implicit successor for the loop body.
  Succ = EntryConditionBlock;

  // See if this is a known constant.
  const TryResult &KnownVal = TryEvaluateBool(D->getCond());

  // Process the loop body.
  CFGBlock* BodyBlock = NULL;
  {
    assert(D->getBody());

    // Save the current values for Block, Succ, and continue and break targets
    SaveAndRestore<CFGBlock*> save_Block(Block), save_Succ(Succ),
      save_continue(ContinueTargetBlock),
      save_break(BreakTargetBlock);

    // All continues within this loop should go to the condition block
    ContinueTargetBlock = EntryConditionBlock;

    // All breaks should go to the code following the loop.
    BreakTargetBlock = LoopSuccessor;

    // NULL out Block to force lazy instantiation of blocks for the body.
    Block = NULL;

    // Create the body.  The returned block is the entry to the loop body.
    BodyBlock = addStmt(D->getBody());

    if (!BodyBlock)
      BodyBlock = EntryConditionBlock; // can happen for "do ; while(...)"
    else if (Block) {
      if (!FinishBlock(BodyBlock))
        return 0;
    }

    // Add an intermediate block between the BodyBlock and the
    // ExitConditionBlock to represent the "loop back" transition.  Create an
    // empty block to represent the transition block for looping back to the
    // head of the loop.
    // FIXME: Can we do this more efficiently without adding another block?
    Block = NULL;
    Succ = BodyBlock;
    CFGBlock *LoopBackBlock = createBlock();
    LoopBackBlock->setLoopTarget(D);

    // Add the loop body entry as a successor to the condition.
    AddSuccessor(ExitConditionBlock, KnownVal.isFalse() ? NULL : LoopBackBlock);
  }

  // Link up the condition block with the code that follows the loop.
  // (the false branch).
  AddSuccessor(ExitConditionBlock, KnownVal.isTrue() ? NULL : LoopSuccessor);

  // There can be no more statements in the body block(s) since we loop back to
  // the body.  NULL out Block to force lazy creation of another block.
  Block = NULL;

  // Return the loop body, which is the dominating block for the loop.
  Succ = BodyBlock;
  return BodyBlock;
}

CFGBlock* CFGBuilder::VisitContinueStmt(ContinueStmt* C) {
  // "continue" is a control-flow statement.  Thus we stop processing the
  // current block.
  if (Block && !FinishBlock(Block))
      return 0;

  // Now create a new block that ends with the continue statement.
  Block = createBlock(false);
  Block->setTerminator(C);

  // If there is no target for the continue, then we are looking at an
  // incomplete AST.  This means the CFG cannot be constructed.
  if (ContinueTargetBlock)
    AddSuccessor(Block, ContinueTargetBlock);
  else
    badCFG = true;

  return Block;
}

CFGBlock *CFGBuilder::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E,
                                             AddStmtChoice asc) {

  if (asc.alwaysAdd()) {
    autoCreateBlock();
    AppendStmt(Block, E);
  }

  // VLA types have expressions that must be evaluated.
  if (E->isArgumentType()) {
    for (VariableArrayType* VA = FindVA(E->getArgumentType().getTypePtr());
         VA != 0; VA = FindVA(VA->getElementType().getTypePtr()))
      addStmt(VA->getSizeExpr());
  }

  return Block;
}

/// VisitStmtExpr - Utility method to handle (nested) statement
///  expressions (a GCC extension).
CFGBlock* CFGBuilder::VisitStmtExpr(StmtExpr *SE, AddStmtChoice asc) {
  if (asc.alwaysAdd()) {
    autoCreateBlock();
    AppendStmt(Block, SE);
  }
  return VisitCompoundStmt(SE->getSubStmt());
}

CFGBlock* CFGBuilder::VisitSwitchStmt(SwitchStmt* Terminator) {
  // "switch" is a control-flow statement.  Thus we stop processing the current
  // block.
  CFGBlock* SwitchSuccessor = NULL;

  if (Block) {
    if (!FinishBlock(Block))
      return 0;
    SwitchSuccessor = Block;
  } else SwitchSuccessor = Succ;

  // Save the current "switch" context.
  SaveAndRestore<CFGBlock*> save_switch(SwitchTerminatedBlock),
                            save_break(BreakTargetBlock),
                            save_default(DefaultCaseBlock);

  // Set the "default" case to be the block after the switch statement.  If the
  // switch statement contains a "default:", this value will be overwritten with
  // the block for that code.
  DefaultCaseBlock = SwitchSuccessor;

  // Create a new block that will contain the switch statement.
  SwitchTerminatedBlock = createBlock(false);

  // Now process the switch body.  The code after the switch is the implicit
  // successor.
  Succ = SwitchSuccessor;
  BreakTargetBlock = SwitchSuccessor;

  // When visiting the body, the case statements should automatically get linked
  // up to the switch.  We also don't keep a pointer to the body, since all
  // control-flow from the switch goes to case/default statements.
  assert(Terminator->getBody() && "switch must contain a non-NULL body");
  Block = NULL;
  CFGBlock *BodyBlock = addStmt(Terminator->getBody());
  if (Block) {
    if (!FinishBlock(BodyBlock))
      return 0;
  }

  // If we have no "default:" case, the default transition is to the code
  // following the switch body.
  AddSuccessor(SwitchTerminatedBlock, DefaultCaseBlock);

  // Add the terminator and condition in the switch block.
  SwitchTerminatedBlock->setTerminator(Terminator);
  assert(Terminator->getCond() && "switch condition must be non-NULL");
  Block = SwitchTerminatedBlock;
  Block = addStmt(Terminator->getCond());
  
  // Finally, if the SwitchStmt contains a condition variable, add both the
  // SwitchStmt and the condition variable initialization to the CFG.
  if (VarDecl *VD = Terminator->getConditionVariable()) {
    if (Expr *Init = VD->getInit()) {
      autoCreateBlock();
      AppendStmt(Block, Terminator, AddStmtChoice::AlwaysAdd);
      addStmt(Init);
    }
  }
  
  return Block;
}

CFGBlock* CFGBuilder::VisitCaseStmt(CaseStmt* CS) {
  // CaseStmts are essentially labels, so they are the first statement in a
  // block.

  if (CS->getSubStmt())
    addStmt(CS->getSubStmt());

  CFGBlock* CaseBlock = Block;
  if (!CaseBlock)
    CaseBlock = createBlock();

  // Cases statements partition blocks, so this is the top of the basic block we
  // were processing (the "case XXX:" is the label).
  CaseBlock->setLabel(CS);

  if (!FinishBlock(CaseBlock))
    return 0;

  // Add this block to the list of successors for the block with the switch
  // statement.
  assert(SwitchTerminatedBlock);
  AddSuccessor(SwitchTerminatedBlock, CaseBlock);

  // We set Block to NULL to allow lazy creation of a new block (if necessary)
  Block = NULL;

  // This block is now the implicit successor of other blocks.
  Succ = CaseBlock;

  return CaseBlock;
}

CFGBlock* CFGBuilder::VisitDefaultStmt(DefaultStmt* Terminator) {
  if (Terminator->getSubStmt())
    addStmt(Terminator->getSubStmt());

  DefaultCaseBlock = Block;

  if (!DefaultCaseBlock)
    DefaultCaseBlock = createBlock();

  // Default statements partition blocks, so this is the top of the basic block
  // we were processing (the "default:" is the label).
  DefaultCaseBlock->setLabel(Terminator);

  if (!FinishBlock(DefaultCaseBlock))
    return 0;

  // Unlike case statements, we don't add the default block to the successors
  // for the switch statement immediately.  This is done when we finish
  // processing the switch statement.  This allows for the default case
  // (including a fall-through to the code after the switch statement) to always
  // be the last successor of a switch-terminated block.

  // We set Block to NULL to allow lazy creation of a new block (if necessary)
  Block = NULL;

  // This block is now the implicit successor of other blocks.
  Succ = DefaultCaseBlock;

  return DefaultCaseBlock;
}

CFGBlock *CFGBuilder::VisitCXXTryStmt(CXXTryStmt *Terminator) {
  // "try"/"catch" is a control-flow statement.  Thus we stop processing the
  // current block.
  CFGBlock* TrySuccessor = NULL;

  if (Block) {
    if (!FinishBlock(Block))
      return 0;
    TrySuccessor = Block;
  } else TrySuccessor = Succ;

  CFGBlock *PrevTryTerminatedBlock = TryTerminatedBlock;

  // Create a new block that will contain the try statement.
  CFGBlock *NewTryTerminatedBlock = createBlock(false);
  // Add the terminator in the try block.
  NewTryTerminatedBlock->setTerminator(Terminator);

  bool HasCatchAll = false;
  for (unsigned h = 0; h <Terminator->getNumHandlers(); ++h) {
    // The code after the try is the implicit successor.
    Succ = TrySuccessor;
    CXXCatchStmt *CS = Terminator->getHandler(h);
    if (CS->getExceptionDecl() == 0) {
      HasCatchAll = true;
    }
    Block = NULL;
    CFGBlock *CatchBlock = VisitCXXCatchStmt(CS);
    if (CatchBlock == 0)
      return 0;
    // Add this block to the list of successors for the block with the try
    // statement.
    AddSuccessor(NewTryTerminatedBlock, CatchBlock);
  }
  if (!HasCatchAll) {
    if (PrevTryTerminatedBlock)
      AddSuccessor(NewTryTerminatedBlock, PrevTryTerminatedBlock);
    else
      AddSuccessor(NewTryTerminatedBlock, &cfg->getExit());
  }

  // The code after the try is the implicit successor.
  Succ = TrySuccessor;

  // Save the current "try" context.
  SaveAndRestore<CFGBlock*> save_try(TryTerminatedBlock);
  TryTerminatedBlock = NewTryTerminatedBlock;

  assert(Terminator->getTryBlock() && "try must contain a non-NULL body");
  Block = NULL;
  Block = addStmt(Terminator->getTryBlock());
  return Block;
}

CFGBlock* CFGBuilder::VisitCXXCatchStmt(CXXCatchStmt* CS) {
  // CXXCatchStmt are treated like labels, so they are the first statement in a
  // block.

  if (CS->getHandlerBlock())
    addStmt(CS->getHandlerBlock());

  CFGBlock* CatchBlock = Block;
  if (!CatchBlock)
    CatchBlock = createBlock();

  CatchBlock->setLabel(CS);

  if (!FinishBlock(CatchBlock))
    return 0;

  // We set Block to NULL to allow lazy creation of a new block (if necessary)
  Block = NULL;

  return CatchBlock;
}

CFGBlock* CFGBuilder::VisitIndirectGotoStmt(IndirectGotoStmt* I) {
  // Lazily create the indirect-goto dispatch block if there isn't one already.
  CFGBlock* IBlock = cfg->getIndirectGotoBlock();

  if (!IBlock) {
    IBlock = createBlock(false);
    cfg->setIndirectGotoBlock(IBlock);
  }

  // IndirectGoto is a control-flow statement.  Thus we stop processing the
  // current block and create a new one.
  if (Block && !FinishBlock(Block))
    return 0;

  Block = createBlock(false);
  Block->setTerminator(I);
  AddSuccessor(Block, IBlock);
  return addStmt(I->getTarget());
}

} // end anonymous namespace

/// createBlock - Constructs and adds a new CFGBlock to the CFG.  The block has
///  no successors or predecessors.  If this is the first block created in the
///  CFG, it is automatically set to be the Entry and Exit of the CFG.
CFGBlock* CFG::createBlock() {
  bool first_block = begin() == end();

  // Create the block.
  CFGBlock *Mem = getAllocator().Allocate<CFGBlock>();
  new (Mem) CFGBlock(NumBlockIDs++, BlkBVC);
  Blocks.push_back(Mem, BlkBVC);

  // If this is the first block, set it as the Entry and Exit.
  if (first_block)
    Entry = Exit = &back();

  // Return the block.
  return &back();
}

/// buildCFG - Constructs a CFG from an AST.  Ownership of the returned
///  CFG is returned to the caller.
CFG* CFG::buildCFG(const Decl *D, Stmt* Statement, ASTContext *C,
                   bool AddEHEdges, bool AddScopes) {
  CFGBuilder Builder;
  return Builder.buildCFG(D, Statement, C, AddEHEdges, AddScopes);
}

//===----------------------------------------------------------------------===//
// CFG: Queries for BlkExprs.
//===----------------------------------------------------------------------===//

namespace {
  typedef llvm::DenseMap<const Stmt*,unsigned> BlkExprMapTy;
}

static void FindSubExprAssignments(Stmt *S,
                                   llvm::SmallPtrSet<Expr*,50>& Set) {
  if (!S)
    return;

  for (Stmt::child_iterator I=S->child_begin(), E=S->child_end(); I!=E; ++I) {
    Stmt *child = *I;    
    if (!child)
      continue;
    
    if (BinaryOperator* B = dyn_cast<BinaryOperator>(child))
      if (B->isAssignmentOp()) Set.insert(B);

    FindSubExprAssignments(child, Set);
  }
}

static BlkExprMapTy* PopulateBlkExprMap(CFG& cfg) {
  BlkExprMapTy* M = new BlkExprMapTy();

  // Look for assignments that are used as subexpressions.  These are the only
  // assignments that we want to *possibly* register as a block-level
  // expression.  Basically, if an assignment occurs both in a subexpression and
  // at the block-level, it is a block-level expression.
  llvm::SmallPtrSet<Expr*,50> SubExprAssignments;

  for (CFG::iterator I=cfg.begin(), E=cfg.end(); I != E; ++I)
    for (CFGBlock::iterator BI=(*I)->begin(), EI=(*I)->end(); BI != EI; ++BI)
      FindSubExprAssignments(*BI, SubExprAssignments);

  for (CFG::iterator I=cfg.begin(), E=cfg.end(); I != E; ++I) {

    // Iterate over the statements again on identify the Expr* and Stmt* at the
    // block-level that are block-level expressions.

    for (CFGBlock::iterator BI=(*I)->begin(), EI=(*I)->end(); BI != EI; ++BI)
      if (Expr* Exp = dyn_cast<Expr>(*BI)) {

        if (BinaryOperator* B = dyn_cast<BinaryOperator>(Exp)) {
          // Assignment expressions that are not nested within another
          // expression are really "statements" whose value is never used by
          // another expression.
          if (B->isAssignmentOp() && !SubExprAssignments.count(Exp))
            continue;
        } else if (const StmtExpr* Terminator = dyn_cast<StmtExpr>(Exp)) {
          // Special handling for statement expressions.  The last statement in
          // the statement expression is also a block-level expr.
          const CompoundStmt* C = Terminator->getSubStmt();
          if (!C->body_empty()) {
            unsigned x = M->size();
            (*M)[C->body_back()] = x;
          }
        }

        unsigned x = M->size();
        (*M)[Exp] = x;
      }

    // Look at terminators.  The condition is a block-level expression.

    Stmt* S = (*I)->getTerminatorCondition();

    if (S && M->find(S) == M->end()) {
        unsigned x = M->size();
        (*M)[S] = x;
    }
  }

  return M;
}

CFG::BlkExprNumTy CFG::getBlkExprNum(const Stmt* S) {
  assert(S != NULL);
  if (!BlkExprMap) { BlkExprMap = (void*) PopulateBlkExprMap(*this); }

  BlkExprMapTy* M = reinterpret_cast<BlkExprMapTy*>(BlkExprMap);
  BlkExprMapTy::iterator I = M->find(S);
  return (I == M->end()) ? CFG::BlkExprNumTy() : CFG::BlkExprNumTy(I->second);
}

unsigned CFG::getNumBlkExprs() {
  if (const BlkExprMapTy* M = reinterpret_cast<const BlkExprMapTy*>(BlkExprMap))
    return M->size();
  else {
    // We assume callers interested in the number of BlkExprs will want
    // the map constructed if it doesn't already exist.
    BlkExprMap = (void*) PopulateBlkExprMap(*this);
    return reinterpret_cast<BlkExprMapTy*>(BlkExprMap)->size();
  }
}

//===----------------------------------------------------------------------===//
// Cleanup: CFG dstor.
//===----------------------------------------------------------------------===//

CFG::~CFG() {
  delete reinterpret_cast<const BlkExprMapTy*>(BlkExprMap);
}

//===----------------------------------------------------------------------===//
// CFG pretty printing
//===----------------------------------------------------------------------===//

namespace {

class StmtPrinterHelper : public PrinterHelper  {
  typedef llvm::DenseMap<Stmt*,std::pair<unsigned,unsigned> > StmtMapTy;
  StmtMapTy StmtMap;
  signed CurrentBlock;
  unsigned CurrentStmt;
  const LangOptions &LangOpts;
public:

  StmtPrinterHelper(const CFG* cfg, const LangOptions &LO)
    : CurrentBlock(0), CurrentStmt(0), LangOpts(LO) {
    for (CFG::const_iterator I = cfg->begin(), E = cfg->end(); I != E; ++I ) {
      unsigned j = 1;
      for (CFGBlock::const_iterator BI = (*I)->begin(), BEnd = (*I)->end() ;
           BI != BEnd; ++BI, ++j )
        StmtMap[*BI] = std::make_pair((*I)->getBlockID(),j);
      }
  }

  virtual ~StmtPrinterHelper() {}

  const LangOptions &getLangOpts() const { return LangOpts; }
  void setBlockID(signed i) { CurrentBlock = i; }
  void setStmtID(unsigned i) { CurrentStmt = i; }

  virtual bool handledStmt(Stmt* Terminator, llvm::raw_ostream& OS) {

    StmtMapTy::iterator I = StmtMap.find(Terminator);

    if (I == StmtMap.end())
      return false;

    if (CurrentBlock >= 0 && I->second.first == (unsigned) CurrentBlock
                          && I->second.second == CurrentStmt) {
      return false;
    }

    OS << "[B" << I->second.first << "." << I->second.second << "]";
    return true;
  }
};
} // end anonymous namespace


namespace {
class CFGBlockTerminatorPrint
  : public StmtVisitor<CFGBlockTerminatorPrint,void> {

  llvm::raw_ostream& OS;
  StmtPrinterHelper* Helper;
  PrintingPolicy Policy;
public:
  CFGBlockTerminatorPrint(llvm::raw_ostream& os, StmtPrinterHelper* helper,
                          const PrintingPolicy &Policy)
    : OS(os), Helper(helper), Policy(Policy) {}

  void VisitIfStmt(IfStmt* I) {
    OS << "if ";
    I->getCond()->printPretty(OS,Helper,Policy);
  }

  // Default case.
  void VisitStmt(Stmt* Terminator) {
    Terminator->printPretty(OS, Helper, Policy);
  }

  void VisitForStmt(ForStmt* F) {
    OS << "for (" ;
    if (F->getInit())
      OS << "...";
    OS << "; ";
    if (Stmt* C = F->getCond())
      C->printPretty(OS, Helper, Policy);
    OS << "; ";
    if (F->getInc())
      OS << "...";
    OS << ")";
  }

  void VisitWhileStmt(WhileStmt* W) {
    OS << "while " ;
    if (Stmt* C = W->getCond())
      C->printPretty(OS, Helper, Policy);
  }

  void VisitDoStmt(DoStmt* D) {
    OS << "do ... while ";
    if (Stmt* C = D->getCond())
      C->printPretty(OS, Helper, Policy);
  }

  void VisitSwitchStmt(SwitchStmt* Terminator) {
    OS << "switch ";
    Terminator->getCond()->printPretty(OS, Helper, Policy);
  }

  void VisitCXXTryStmt(CXXTryStmt* CS) {
    OS << "try ...";
  }

  void VisitConditionalOperator(ConditionalOperator* C) {
    C->getCond()->printPretty(OS, Helper, Policy);
    OS << " ? ... : ...";
  }

  void VisitChooseExpr(ChooseExpr* C) {
    OS << "__builtin_choose_expr( ";
    C->getCond()->printPretty(OS, Helper, Policy);
    OS << " )";
  }

  void VisitIndirectGotoStmt(IndirectGotoStmt* I) {
    OS << "goto *";
    I->getTarget()->printPretty(OS, Helper, Policy);
  }

  void VisitBinaryOperator(BinaryOperator* B) {
    if (!B->isLogicalOp()) {
      VisitExpr(B);
      return;
    }

    B->getLHS()->printPretty(OS, Helper, Policy);

    switch (B->getOpcode()) {
      case BinaryOperator::LOr:
        OS << " || ...";
        return;
      case BinaryOperator::LAnd:
        OS << " && ...";
        return;
      default:
        assert(false && "Invalid logical operator.");
    }
  }

  void VisitExpr(Expr* E) {
    E->printPretty(OS, Helper, Policy);
  }
};
} // end anonymous namespace


static void print_stmt(llvm::raw_ostream &OS, StmtPrinterHelper* Helper,
                       const CFGElement &E) {
  Stmt *Terminator = E;

  if (E.asStartScope()) {
    OS << "start scope\n";
    return;
  }
  if (E.asEndScope()) {
    OS << "end scope\n";
    return;
  }

  if (Helper) {
    // special printing for statement-expressions.
    if (StmtExpr* SE = dyn_cast<StmtExpr>(Terminator)) {
      CompoundStmt* Sub = SE->getSubStmt();

      if (Sub->child_begin() != Sub->child_end()) {
        OS << "({ ... ; ";
        Helper->handledStmt(*SE->getSubStmt()->body_rbegin(),OS);
        OS << " })\n";
        return;
      }
    }

    // special printing for comma expressions.
    if (BinaryOperator* B = dyn_cast<BinaryOperator>(Terminator)) {
      if (B->getOpcode() == BinaryOperator::Comma) {
        OS << "... , ";
        Helper->handledStmt(B->getRHS(),OS);
        OS << '\n';
        return;
      }
    }
  }

  Terminator->printPretty(OS, Helper, PrintingPolicy(Helper->getLangOpts()));

  // Expressions need a newline.
  if (isa<Expr>(Terminator)) OS << '\n';
}

static void print_block(llvm::raw_ostream& OS, const CFG* cfg,
                        const CFGBlock& B,
                        StmtPrinterHelper* Helper, bool print_edges) {

  if (Helper) Helper->setBlockID(B.getBlockID());

  // Print the header.
  OS << "\n [ B" << B.getBlockID();

  if (&B == &cfg->getEntry())
    OS << " (ENTRY) ]\n";
  else if (&B == &cfg->getExit())
    OS << " (EXIT) ]\n";
  else if (&B == cfg->getIndirectGotoBlock())
    OS << " (INDIRECT GOTO DISPATCH) ]\n";
  else
    OS << " ]\n";

  // Print the label of this block.
  if (Stmt* Label = const_cast<Stmt*>(B.getLabel())) {

    if (print_edges)
      OS << "    ";

    if (LabelStmt* L = dyn_cast<LabelStmt>(Label))
      OS << L->getName();
    else if (CaseStmt* C = dyn_cast<CaseStmt>(Label)) {
      OS << "case ";
      C->getLHS()->printPretty(OS, Helper,
                               PrintingPolicy(Helper->getLangOpts()));
      if (C->getRHS()) {
        OS << " ... ";
        C->getRHS()->printPretty(OS, Helper,
                                 PrintingPolicy(Helper->getLangOpts()));
      }
    } else if (isa<DefaultStmt>(Label))
      OS << "default";
    else if (CXXCatchStmt *CS = dyn_cast<CXXCatchStmt>(Label)) {
      OS << "catch (";
      if (CS->getExceptionDecl())
        CS->getExceptionDecl()->print(OS, PrintingPolicy(Helper->getLangOpts()),
                                      0);
      else
        OS << "...";
      OS << ")";

    } else
      assert(false && "Invalid label statement in CFGBlock.");

    OS << ":\n";
  }

  // Iterate through the statements in the block and print them.
  unsigned j = 1;

  for (CFGBlock::const_iterator I = B.begin(), E = B.end() ;
       I != E ; ++I, ++j ) {

    // Print the statement # in the basic block and the statement itself.
    if (print_edges)
      OS << "    ";

    OS << llvm::format("%3d", j) << ": ";

    if (Helper)
      Helper->setStmtID(j);

    print_stmt(OS,Helper,*I);
  }

  // Print the terminator of this block.
  if (B.getTerminator()) {
    if (print_edges)
      OS << "    ";

    OS << "  T: ";

    if (Helper) Helper->setBlockID(-1);

    CFGBlockTerminatorPrint TPrinter(OS, Helper,
                                     PrintingPolicy(Helper->getLangOpts()));
    TPrinter.Visit(const_cast<Stmt*>(B.getTerminator()));
    OS << '\n';
  }

  if (print_edges) {
    // Print the predecessors of this block.
    OS << "    Predecessors (" << B.pred_size() << "):";
    unsigned i = 0;

    for (CFGBlock::const_pred_iterator I = B.pred_begin(), E = B.pred_end();
         I != E; ++I, ++i) {

      if (i == 8 || (i-8) == 0)
        OS << "\n     ";

      OS << " B" << (*I)->getBlockID();
    }

    OS << '\n';

    // Print the successors of this block.
    OS << "    Successors (" << B.succ_size() << "):";
    i = 0;

    for (CFGBlock::const_succ_iterator I = B.succ_begin(), E = B.succ_end();
         I != E; ++I, ++i) {

      if (i == 8 || (i-8) % 10 == 0)
        OS << "\n    ";

      if (*I)
        OS << " B" << (*I)->getBlockID();
      else
        OS  << " NULL";
    }

    OS << '\n';
  }
}


/// dump - A simple pretty printer of a CFG that outputs to stderr.
void CFG::dump(const LangOptions &LO) const { print(llvm::errs(), LO); }

/// print - A simple pretty printer of a CFG that outputs to an ostream.
void CFG::print(llvm::raw_ostream &OS, const LangOptions &LO) const {
  StmtPrinterHelper Helper(this, LO);

  // Print the entry block.
  print_block(OS, this, getEntry(), &Helper, true);

  // Iterate through the CFGBlocks and print them one by one.
  for (const_iterator I = Blocks.begin(), E = Blocks.end() ; I != E ; ++I) {
    // Skip the entry block, because we already printed it.
    if (&(**I) == &getEntry() || &(**I) == &getExit())
      continue;

    print_block(OS, this, **I, &Helper, true);
  }

  // Print the exit block.
  print_block(OS, this, getExit(), &Helper, true);
  OS.flush();
}

/// dump - A simply pretty printer of a CFGBlock that outputs to stderr.
void CFGBlock::dump(const CFG* cfg, const LangOptions &LO) const {
  print(llvm::errs(), cfg, LO);
}

/// print - A simple pretty printer of a CFGBlock that outputs to an ostream.
///   Generally this will only be called from CFG::print.
void CFGBlock::print(llvm::raw_ostream& OS, const CFG* cfg,
                     const LangOptions &LO) const {
  StmtPrinterHelper Helper(cfg, LO);
  print_block(OS, cfg, *this, &Helper, true);
}

/// printTerminator - A simple pretty printer of the terminator of a CFGBlock.
void CFGBlock::printTerminator(llvm::raw_ostream &OS,
                               const LangOptions &LO) const {
  CFGBlockTerminatorPrint TPrinter(OS, NULL, PrintingPolicy(LO));
  TPrinter.Visit(const_cast<Stmt*>(getTerminator()));
}

Stmt* CFGBlock::getTerminatorCondition() {

  if (!Terminator)
    return NULL;

  Expr* E = NULL;

  switch (Terminator->getStmtClass()) {
    default:
      break;

    case Stmt::ForStmtClass:
      E = cast<ForStmt>(Terminator)->getCond();
      break;

    case Stmt::WhileStmtClass:
      E = cast<WhileStmt>(Terminator)->getCond();
      break;

    case Stmt::DoStmtClass:
      E = cast<DoStmt>(Terminator)->getCond();
      break;

    case Stmt::IfStmtClass:
      E = cast<IfStmt>(Terminator)->getCond();
      break;

    case Stmt::ChooseExprClass:
      E = cast<ChooseExpr>(Terminator)->getCond();
      break;

    case Stmt::IndirectGotoStmtClass:
      E = cast<IndirectGotoStmt>(Terminator)->getTarget();
      break;

    case Stmt::SwitchStmtClass:
      E = cast<SwitchStmt>(Terminator)->getCond();
      break;

    case Stmt::ConditionalOperatorClass:
      E = cast<ConditionalOperator>(Terminator)->getCond();
      break;

    case Stmt::BinaryOperatorClass: // '&&' and '||'
      E = cast<BinaryOperator>(Terminator)->getLHS();
      break;

    case Stmt::ObjCForCollectionStmtClass:
      return Terminator;
  }

  return E ? E->IgnoreParens() : NULL;
}

bool CFGBlock::hasBinaryBranchTerminator() const {

  if (!Terminator)
    return false;

  Expr* E = NULL;

  switch (Terminator->getStmtClass()) {
    default:
      return false;

    case Stmt::ForStmtClass:
    case Stmt::WhileStmtClass:
    case Stmt::DoStmtClass:
    case Stmt::IfStmtClass:
    case Stmt::ChooseExprClass:
    case Stmt::ConditionalOperatorClass:
    case Stmt::BinaryOperatorClass:
      return true;
  }

  return E ? E->IgnoreParens() : NULL;
}


//===----------------------------------------------------------------------===//
// CFG Graphviz Visualization
//===----------------------------------------------------------------------===//


#ifndef NDEBUG
static StmtPrinterHelper* GraphHelper;
#endif

void CFG::viewCFG(const LangOptions &LO) const {
#ifndef NDEBUG
  StmtPrinterHelper H(this, LO);
  GraphHelper = &H;
  llvm::ViewGraph(this,"CFG");
  GraphHelper = NULL;
#endif
}

namespace llvm {
template<>
struct DOTGraphTraits<const CFG*> : public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false) : DefaultDOTGraphTraits(isSimple) {}

  static std::string getNodeLabel(const CFGBlock* Node, const CFG* Graph) {

#ifndef NDEBUG
    std::string OutSStr;
    llvm::raw_string_ostream Out(OutSStr);
    print_block(Out,Graph, *Node, GraphHelper, false);
    std::string& OutStr = Out.str();

    if (OutStr[0] == '\n') OutStr.erase(OutStr.begin());

    // Process string output to make it nicer...
    for (unsigned i = 0; i != OutStr.length(); ++i)
      if (OutStr[i] == '\n') {                            // Left justify
        OutStr[i] = '\\';
        OutStr.insert(OutStr.begin()+i+1, 'l');
      }

    return OutStr;
#else
    return "";
#endif
  }
};
} // end namespace llvm
