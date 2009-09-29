//===--- ResolveLocation.cpp - Source location resolver ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This defines the ResolveLocationInAST function, which resolves a
//  source location into a ASTLocation.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/Utils.h"
#include "clang/Index/ASTLocation.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Compiler.h"
using namespace clang;
using namespace idx;

namespace {

/// \brief Base for the LocResolver classes. Mostly does source range checking.
class VISIBILITY_HIDDEN LocResolverBase {
protected:
  ASTContext &Ctx;
  SourceLocation Loc;

  enum RangePos {
    BeforeLoc,
    ContainsLoc,
    AfterLoc
  };

  RangePos CheckRange(SourceRange Range);
  RangePos CheckRange(Decl *D) { return CheckRange(D->getSourceRange()); }
  RangePos CheckRange(Stmt *Node) { return CheckRange(Node->getSourceRange()); }

  template <typename T>
  bool isBeforeLocation(T *Node) {
    return CheckRange(Node) == BeforeLoc;
  }

  template <typename T>
  bool ContainsLocation(T *Node) {
    return CheckRange(Node) == ContainsLoc;
  }

  template <typename T>
  bool isAfterLocation(T *Node) {
    return CheckRange(Node) == AfterLoc;
  }

public:
  LocResolverBase(ASTContext &ctx, SourceLocation loc)
    : Ctx(ctx), Loc(loc) {}

#ifndef NDEBUG
  /// \brief Debugging output.
  void print(Decl *D);
  /// \brief Debugging output.
  void print(Stmt *Node);
#endif
};

/// \brief Searches a statement for the ASTLocation that corresponds to a source
/// location.
class VISIBILITY_HIDDEN StmtLocResolver : public LocResolverBase,
                                          public StmtVisitor<StmtLocResolver,
                                                             ASTLocation     > {
  Decl * const Parent;

public:
  StmtLocResolver(ASTContext &ctx, SourceLocation loc, Decl *parent)
    : LocResolverBase(ctx, loc), Parent(parent) {}

  ASTLocation VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node);
  ASTLocation VisitDeclStmt(DeclStmt *Node);
  ASTLocation VisitStmt(Stmt *Node);
};

/// \brief Searches a declaration for the ASTLocation that corresponds to a
/// source location.
class VISIBILITY_HIDDEN DeclLocResolver : public LocResolverBase,
                                          public DeclVisitor<DeclLocResolver,
                                                             ASTLocation     > {
public:
  DeclLocResolver(ASTContext &ctx, SourceLocation loc)
    : LocResolverBase(ctx, loc) {}

  ASTLocation VisitDeclContext(DeclContext *DC);
  ASTLocation VisitTranslationUnitDecl(TranslationUnitDecl *TU);
  ASTLocation VisitVarDecl(VarDecl *D);
  ASTLocation VisitFunctionDecl(FunctionDecl *D);
  ASTLocation VisitObjCMethodDecl(ObjCMethodDecl *D);
  ASTLocation VisitDecl(Decl *D);
};

} // anonymous namespace

ASTLocation
StmtLocResolver::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
  assert(ContainsLocation(Node) &&
         "Should visit only after verifying that loc is in range");

  if (Node->getNumArgs() == 1)
    // Unary operator. Let normal child traversal handle it.
    return VisitCallExpr(Node);

  assert(Node->getNumArgs() == 2 &&
         "Wrong args for the C++ operator call expr ?");

  llvm::SmallVector<Expr *, 3> Nodes;
  // Binary operator. Check in order of 1-left arg, 2-callee, 3-right arg.
  Nodes.push_back(Node->getArg(0));
  Nodes.push_back(Node->getCallee());
  Nodes.push_back(Node->getArg(1));

  for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
    RangePos RP = CheckRange(Nodes[i]);
    if (RP == AfterLoc)
      break;
    if (RP == ContainsLoc)
      return Visit(Nodes[i]);
  }

  return ASTLocation(Parent, Node);
}

ASTLocation StmtLocResolver::VisitDeclStmt(DeclStmt *Node) {
  assert(ContainsLocation(Node) &&
         "Should visit only after verifying that loc is in range");

  // Search all declarations of this DeclStmt.
  for (DeclStmt::decl_iterator
         I = Node->decl_begin(), E = Node->decl_end(); I != E; ++I) {
    RangePos RP = CheckRange(*I);
    if (RP == AfterLoc)
      break;
    if (RP == ContainsLoc)
      return DeclLocResolver(Ctx, Loc).Visit(*I);
  }

  return ASTLocation(Parent, Node);
}

ASTLocation StmtLocResolver::VisitStmt(Stmt *Node) {
  assert(ContainsLocation(Node) &&
         "Should visit only after verifying that loc is in range");

  // Search the child statements.
  for (Stmt::child_iterator
         I = Node->child_begin(), E = Node->child_end(); I != E; ++I) {
    if (*I == NULL)
      continue;

    RangePos RP = CheckRange(*I);
    if (RP == AfterLoc)
      break;
    if (RP == ContainsLoc)
      return Visit(*I);
  }

  return ASTLocation(Parent, Node);
}

ASTLocation DeclLocResolver::VisitDeclContext(DeclContext *DC) {
  for (DeclContext::decl_iterator
         I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I) {
    RangePos RP = CheckRange(*I);
    if (RP == AfterLoc)
      break;
    if (RP == ContainsLoc)
      return Visit(*I);
  }

  return ASTLocation(cast<Decl>(DC));
}

ASTLocation DeclLocResolver::VisitTranslationUnitDecl(TranslationUnitDecl *TU) {
  ASTLocation ASTLoc = VisitDeclContext(TU);
  if (ASTLoc.getParentDecl() == TU)
    return ASTLocation();
  return ASTLoc;
}

ASTLocation DeclLocResolver::VisitFunctionDecl(FunctionDecl *D) {
  assert(ContainsLocation(D) &&
         "Should visit only after verifying that loc is in range");

  // First, search through the parameters of the function.
  for (FunctionDecl::param_iterator
         I = D->param_begin(), E = D->param_end(); I != E; ++I) {
    RangePos RP = CheckRange(*I);
    if (RP == AfterLoc)
      return ASTLocation(D);
    if (RP == ContainsLoc)
      return Visit(*I);
  }

  // We didn't find the location in the parameters and we didn't get passed it.

  if (!D->isThisDeclarationADefinition())
    return ASTLocation(D);

  // Second, search through the declarations that are part of the function.
  // If we find he location there, we won't have to search through its body.

  for (DeclContext::decl_iterator
         I = D->decls_begin(), E = D->decls_end(); I != E; ++I) {
    if (isa<ParmVarDecl>(*I))
      continue; // We already searched through the parameters.

    RangePos RP = CheckRange(*I);
    if (RP == AfterLoc)
      break;
    if (RP == ContainsLoc)
      return Visit(*I);
  }

  // We didn't find a declaration that corresponds to the source location.

  // Finally, search through the body of the function.
  Stmt *Body = D->getBody();
  assert(Body && "Expected definition");
  assert(!isBeforeLocation(Body) &&
         "This function is supposed to contain the loc");
  if (isAfterLocation(Body))
    return ASTLocation(D);

  // The body contains the location.
  assert(ContainsLocation(Body));
  return StmtLocResolver(Ctx, Loc, D).Visit(Body);
}

ASTLocation DeclLocResolver::VisitVarDecl(VarDecl *D) {
  assert(ContainsLocation(D) &&
         "Should visit only after verifying that loc is in range");

  // Check whether the location points to the init expression.
  Expr *Init = D->getInit();
  if (Init && ContainsLocation(Init))
    return StmtLocResolver(Ctx, Loc, D).Visit(Init);

  return ASTLocation(D);
}

ASTLocation DeclLocResolver::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  assert(ContainsLocation(D) &&
         "Should visit only after verifying that loc is in range");

  // First, search through the parameters of the method.
  for (ObjCMethodDecl::param_iterator
         I = D->param_begin(), E = D->param_end(); I != E; ++I) {
    RangePos RP = CheckRange(*I);
    if (RP == AfterLoc)
      return ASTLocation(D);
    if (RP == ContainsLoc)
      return Visit(*I);
  }

  // We didn't find the location in the parameters and we didn't get passed it.

  if (!D->getBody())
    return ASTLocation(D);

  // Second, search through the declarations that are part of the method.
  // If we find he location there, we won't have to search through its body.

  for (DeclContext::decl_iterator
         I = D->decls_begin(), E = D->decls_end(); I != E; ++I) {
    if (isa<ParmVarDecl>(*I))
      continue; // We already searched through the parameters.

    RangePos RP = CheckRange(*I);
    if (RP == AfterLoc)
      break;
    if (RP == ContainsLoc)
      return Visit(*I);
  }

  // We didn't find a declaration that corresponds to the source location.

  // Finally, search through the body of the method.
  Stmt *Body = D->getBody();
  assert(Body && "Expected definition");
  assert(!isBeforeLocation(Body) &&
         "This method is supposed to contain the loc");
  if (isAfterLocation(Body))
    return ASTLocation(D);

  // The body contains the location.
  assert(ContainsLocation(Body));
  return StmtLocResolver(Ctx, Loc, D).Visit(Body);
}

ASTLocation DeclLocResolver::VisitDecl(Decl *D) {
  assert(ContainsLocation(D) &&
         "Should visit only after verifying that loc is in range");
  if (DeclContext *DC = dyn_cast<DeclContext>(D))
    return VisitDeclContext(DC);
  return ASTLocation(D);
}

LocResolverBase::RangePos LocResolverBase::CheckRange(SourceRange Range) {
  if (!Range.isValid())
    return BeforeLoc; // Keep looking.

  // Update the end source range to cover the full length of the token
  // positioned at the end of the source range.
  //
  // e.g.,
  //   int foo
  //   ^   ^
  //
  // will be updated to
  //   int foo
  //   ^     ^
  unsigned TokSize = Lexer::MeasureTokenLength(Range.getEnd(),
                                               Ctx.getSourceManager(),
                                               Ctx.getLangOptions());
  Range.setEnd(Range.getEnd().getFileLocWithOffset(TokSize-1));

  SourceManager &SourceMgr = Ctx.getSourceManager();
  if (SourceMgr.isBeforeInTranslationUnit(Range.getEnd(), Loc))
    return BeforeLoc;

  if (SourceMgr.isBeforeInTranslationUnit(Loc, Range.getBegin()))
    return AfterLoc;

  return ContainsLoc;
}

#ifndef NDEBUG
void LocResolverBase::print(Decl *D) {
  llvm::raw_ostream &OS = llvm::outs();
  OS << "#### DECL " << D->getDeclKindName() << " ####\n";
  D->print(OS);
  OS << " <";
  D->getLocStart().print(OS, Ctx.getSourceManager());
  OS << " > - <";
  D->getLocEnd().print(OS, Ctx.getSourceManager());
  OS << ">\n\n";
  OS.flush();
}

void LocResolverBase::print(Stmt *Node) {
  llvm::raw_ostream &OS = llvm::outs();
  OS << "#### STMT " << Node->getStmtClassName() << " ####\n";
  Node->printPretty(OS, Ctx, 0, PrintingPolicy(Ctx.getLangOptions()));
  OS << " <";
  Node->getLocStart().print(OS, Ctx.getSourceManager());
  OS << " > - <";
  Node->getLocEnd().print(OS, Ctx.getSourceManager());
  OS << ">\n\n";
  OS.flush();
}
#endif


/// \brief Returns the AST node that a source location points to.
///
ASTLocation idx::ResolveLocationInAST(ASTContext &Ctx, SourceLocation Loc) {
  if (Loc.isInvalid())
    return ASTLocation();

  return DeclLocResolver(Ctx, Loc).Visit(Ctx.getTranslationUnitDecl());
}
