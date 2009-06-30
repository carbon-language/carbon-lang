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
//  source location into a <Decl *, Stmt *> pair.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/Utils.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Compiler.h"
using namespace clang;

namespace {

/// \brief Base for the LocResolver classes. Mostly does source range checking.
class VISIBILITY_HIDDEN LocResolverBase {
protected:
  ASTContext &Ctx;
  SourceLocation Loc;
  
  Decl *Dcl;
  Stmt *Stm;
  bool PassedLoc;

  /// \brief Checks whether Loc is in the source range of 'D'.
  ///
  /// If it is, updates Dcl. If Loc is passed the source range, it sets
  /// PassedLoc, otherwise it does nothing.
  void CheckRange(Decl *D);

  /// \brief Checks whether Loc is in the source range of 'Node'.
  ///
  /// If it is, updates Stm. If Loc is passed the source range, it sets
  /// PassedLoc, otherwise it does nothing.
  void CheckRange(Stmt *Node);
  
  /// \brief Updates the end source range to cover the full length of the token
  /// positioned at the end of the source range.
  ///
  /// e.g.,
  /// @code
  ///   int foo
  ///   ^   ^
  /// @endcode
  /// will be updated to
  /// @code
  ///   int foo
  ///   ^     ^ 
  /// @endcode
  void FixRange(SourceRange &Range);

public:
  LocResolverBase(ASTContext &ctx, SourceLocation loc)
    : Ctx(ctx), Loc(loc), Dcl(0), Stm(0), PassedLoc(0) {}
  
  /// \brief We found a AST node that corresponds to the source location.
  bool FoundIt() const { return Dcl != 0 || Stm != 0; }

  /// \brief We either found a AST node or we passed the source location while
  /// searching.
  bool Finished() const { return FoundIt() || PassedLoc; }
  
  Decl *getDecl() const { return Dcl; }
  Stmt *getStmt() const { return Stm; }
  
  std::pair<Decl *, Stmt *> getResult() const {
    return std::make_pair(getDecl(), getStmt());
  }
  
  /// \brief Debugging output.
  void print(Decl *D);
  /// \brief Debugging output.
  void print(Stmt *Node);
};

/// \brief Searches a statement for the AST node that corresponds to a source
/// location.
class VISIBILITY_HIDDEN StmtLocResolver : public LocResolverBase,
                                          public StmtVisitor<StmtLocResolver> {
public:
  StmtLocResolver(ASTContext &ctx, SourceLocation loc)
    : LocResolverBase(ctx, loc) {}

  void VisitDeclStmt(DeclStmt *Node);
  void VisitStmt(Stmt *Node);
};

/// \brief Searches a declaration for the AST node that corresponds to a source
/// location.
class VISIBILITY_HIDDEN DeclLocResolver : public LocResolverBase,
                                          public DeclVisitor<DeclLocResolver> {
public:
  DeclLocResolver(ASTContext &ctx, SourceLocation loc)
    : LocResolverBase(ctx, loc) {}

  void VisitDeclContext(DeclContext *DC);
  void VisitTranslationUnitDecl(TranslationUnitDecl *TU);
  void VisitVarDecl(VarDecl *D);
  void VisitFunctionDecl(FunctionDecl *D);
  void VisitDecl(Decl *D);
};

} // anonymous namespace

void StmtLocResolver::VisitDeclStmt(DeclStmt *Node) {
  CheckRange(Node);
  if (!FoundIt())
    return;
  assert(Stm == Node && "Result not updated ?");

  // Search all declarations of this DeclStmt. If we found the one corresponding
  // to the source location, update this StmtLocResolver's result.
  DeclLocResolver DLR(Ctx, Loc);
  for (DeclStmt::decl_iterator
         I = Node->decl_begin(), E = Node->decl_end(); I != E; ++I) {
    DLR.Visit(*I);
    if (DLR.Finished()) {
      if (DLR.FoundIt())
        llvm::tie(Dcl, Stm) = DLR.getResult();
      return;
    }
  }
}

void StmtLocResolver::VisitStmt(Stmt *Node) {
  CheckRange(Node);
  if (!FoundIt())
    return;
  assert(Stm == Node && "Result not updated ?");
  
  // Search the child statements.
  StmtLocResolver SLR(Ctx, Loc);
  for (Stmt::child_iterator
         I = Node->child_begin(), E = Node->child_end(); I != E; ++I) {
    SLR.Visit(*I);
    if (!SLR.Finished())
      continue;

    // We either found it or we passed the source location.
    
    if (SLR.FoundIt()) {
      // Only update Dcl if we found another more immediate 'parent' Decl for
      // the statement.
      if (SLR.getDecl())
        Dcl = SLR.getDecl();
      Stm = SLR.getStmt();
    }
    
    return;
  }
}

void DeclLocResolver::VisitDeclContext(DeclContext *DC) {
  DeclLocResolver DLR(Ctx, Loc);
  for (DeclContext::decl_iterator
         I = DC->decls_begin(Ctx), E = DC->decls_end(Ctx); I != E; ++I) {
    DLR.Visit(*I);
    if (DLR.Finished()) {
      if (DLR.FoundIt())
        llvm::tie(Dcl, Stm) = DLR.getResult();
      return;
    }
  }
}

void DeclLocResolver::VisitTranslationUnitDecl(TranslationUnitDecl *TU) {
  VisitDeclContext(TU);
}

void DeclLocResolver::VisitFunctionDecl(FunctionDecl *D) {
  CheckRange(D);
  if (!FoundIt())
    return;
  assert(Dcl == D && "Result not updated ?");

  // First, search through the parameters of the function.
  DeclLocResolver ParmRes(Ctx, Loc);
  for (FunctionDecl::param_iterator
         I = D->param_begin(), E = D->param_end(); I != E; ++I) {
    ParmRes.Visit(*I);
    if (ParmRes.Finished()) {
      if (ParmRes.FoundIt())
        llvm::tie(Dcl, Stm) = ParmRes.getResult();
      return;
    }
  }
  
  // We didn't found the location in the parameters and we didn't get passed it.
  
  // Second, search through the declarations that are part of the function.
  // If we find he location there, we won't have to search through its body.
  DeclLocResolver DLR(Ctx, Loc);
  DLR.VisitDeclContext(D);
  if (DLR.FoundIt()) {
    llvm::tie(Dcl, Stm) = DLR.getResult();
    return;
  }
  
  // We didn't find a declaration that corresponds to the source location.
  
  // Finally, search through the body of the function.
  if (D->isThisDeclarationADefinition()) {
    StmtLocResolver SLR(Ctx, Loc);
    SLR.Visit(D->getBody(Ctx));
    if (SLR.FoundIt()) {
      llvm::tie(Dcl, Stm) = SLR.getResult();
      // If we didn't find a more immediate 'parent' declaration for the
      // statement, set the function as the parent.
      if (Dcl == 0)
        Dcl = D;
    }
  }
}

void DeclLocResolver::VisitVarDecl(VarDecl *D) {
  CheckRange(D);
  if (!FoundIt())
    return;
  assert(Dcl == D && "Result not updated ?");
  
  // Check whether the location points to the init expression.
  if (D->getInit()) {
    StmtLocResolver SLR(Ctx, Loc);
    SLR.Visit(D->getInit());
    Stm = SLR.getStmt();
  }
}

void DeclLocResolver::VisitDecl(Decl *D) {
  CheckRange(D);
}

void LocResolverBase::CheckRange(Decl *D) {
  SourceRange Range = D->getSourceRange();
  if (!Range.isValid())
    return;

  FixRange(Range);

  SourceManager &SourceMgr = Ctx.getSourceManager(); 
  if (SourceMgr.isBeforeInTranslationUnit(Range.getEnd(), Loc))
    return;
  
  if (SourceMgr.isBeforeInTranslationUnit(Loc, Range.getBegin()))
    PassedLoc = true;
  else
    Dcl = D;
}

void LocResolverBase::CheckRange(Stmt *Node) {
  SourceRange Range = Node->getSourceRange();
  if (!Range.isValid())
    return;

  FixRange(Range);

  SourceManager &SourceMgr = Ctx.getSourceManager(); 
  if (SourceMgr.isBeforeInTranslationUnit(Range.getEnd(), Loc))
    return;
  
  if (SourceMgr.isBeforeInTranslationUnit(Loc, Range.getBegin()))
    PassedLoc = true;
  else
    Stm = Node;
}

void LocResolverBase::FixRange(SourceRange &Range) {
  if (!Range.isValid())
    return;
  
  unsigned TokSize = Lexer::MeasureTokenLength(Range.getEnd(),
                                               Ctx.getSourceManager(),
                                               Ctx.getLangOptions());
  Range.setEnd(Range.getEnd().getFileLocWithOffset(TokSize-1));
}

void LocResolverBase::print(Decl *D) {
  llvm::raw_ostream &OS = llvm::outs();
  OS << "#### DECL ####\n";
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
  OS << "#### STMT ####\n";
  Node->printPretty(OS, Ctx, 0, PrintingPolicy(Ctx.getLangOptions()));
  OS << " <";
  Node->getLocStart().print(OS, Ctx.getSourceManager());
  OS << " > - <";
  Node->getLocEnd().print(OS, Ctx.getSourceManager());
  OS << ">\n\n";
  OS.flush();
}


/// \brief Returns the AST node that a source location points to.
///
std::pair<Decl *, Stmt *>
clang::ResolveLocationInAST(ASTContext &Ctx, SourceLocation Loc) {
  if (Loc.isInvalid())
    return std::make_pair((Decl*)0, (Stmt*)0);
  
  DeclLocResolver DLR(Ctx, Loc);
  DLR.Visit(Ctx.getTranslationUnitDecl());
  return DLR.getResult();
}
