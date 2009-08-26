//===--- StmtDumper.cpp - Dumping implementation for Stmt ASTs ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt::dump/Stmt::print methods, which dump out the
// AST in a form that exposes type details and other fields.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Compiler.h"
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//
// StmtDumper Visitor
//===----------------------------------------------------------------------===//

namespace  {
  class VISIBILITY_HIDDEN StmtDumper : public StmtVisitor<StmtDumper> {
    SourceManager *SM;
    FILE *F;
    unsigned IndentLevel;
    
    /// MaxDepth - When doing a normal dump (not dumpAll) we only want to dump
    /// the first few levels of an AST.  This keeps track of how many ast levels
    /// are left.
    unsigned MaxDepth;
    
    /// LastLocFilename/LastLocLine - Keep track of the last location we print
    /// out so that we can print out deltas from then on out.
    const char *LastLocFilename;
    unsigned LastLocLine;

  public:
    StmtDumper(SourceManager *sm, FILE *f, unsigned maxDepth)
      : SM(sm), F(f), IndentLevel(0-1), MaxDepth(maxDepth) {
      LastLocFilename = "";
      LastLocLine = ~0U;
    }
    
    void DumpSubTree(Stmt *S) {
      // Prune the recursion if not using dump all.
      if (MaxDepth == 0) return;
      
      ++IndentLevel;
      if (S) {
        if (DeclStmt* DS = dyn_cast<DeclStmt>(S))
          VisitDeclStmt(DS);
        else {        
          Visit(S);
          
          // Print out children.
          Stmt::child_iterator CI = S->child_begin(), CE = S->child_end();
          if (CI != CE) {
            while (CI != CE) {
              fprintf(F, "\n");
              DumpSubTree(*CI++);
            }
          }
          fprintf(F, ")");
        }
      } else {
        Indent();
        fprintf(F, "<<<NULL>>>");
      }
      --IndentLevel;
    }
    
    void DumpDeclarator(Decl *D);
    
    void Indent() const {
      for (int i = 0, e = IndentLevel; i < e; ++i)
        fprintf(F, "  ");
    }
    
    void DumpType(QualType T) {
      fprintf(F, "'%s'", T.getAsString().c_str());

      if (!T.isNull()) {
        // If the type is directly a typedef, strip off typedefness to give at
        // least one level of concreteness.
        if (TypedefType *TDT = dyn_cast<TypedefType>(T)) {
          QualType Simplified = 
            TDT->LookThroughTypedefs().getQualifiedType(T.getCVRQualifiers());
          fprintf(F, ":'%s'", Simplified.getAsString().c_str());
        }
      }
    }
    void DumpStmt(const Stmt *Node) {
      Indent();
      fprintf(F, "(%s %p", Node->getStmtClassName(), (void*)Node);
      DumpSourceRange(Node);
    }
    void DumpExpr(const Expr *Node) {
      DumpStmt(Node);
      fprintf(F, " ");
      DumpType(Node->getType());
    }
    void DumpSourceRange(const Stmt *Node);
    void DumpLocation(SourceLocation Loc);
    
    // Stmts.
    void VisitStmt(Stmt *Node);
    void VisitDeclStmt(DeclStmt *Node);
    void VisitLabelStmt(LabelStmt *Node);
    void VisitGotoStmt(GotoStmt *Node);
    
    // Exprs
    void VisitExpr(Expr *Node);
    void VisitCastExpr(CastExpr *Node);
    void VisitDeclRefExpr(DeclRefExpr *Node);
    void VisitPredefinedExpr(PredefinedExpr *Node);
    void VisitCharacterLiteral(CharacterLiteral *Node);
    void VisitIntegerLiteral(IntegerLiteral *Node);
    void VisitFloatingLiteral(FloatingLiteral *Node);
    void VisitStringLiteral(StringLiteral *Str);
    void VisitUnaryOperator(UnaryOperator *Node);
    void VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node);
    void VisitMemberExpr(MemberExpr *Node);
    void VisitExtVectorElementExpr(ExtVectorElementExpr *Node);
    void VisitBinaryOperator(BinaryOperator *Node);
    void VisitCompoundAssignOperator(CompoundAssignOperator *Node);
    void VisitAddrLabelExpr(AddrLabelExpr *Node);
    void VisitTypesCompatibleExpr(TypesCompatibleExpr *Node);

    // C++
    void VisitCXXNamedCastExpr(CXXNamedCastExpr *Node);
    void VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node);
    void VisitCXXThisExpr(CXXThisExpr *Node);
    void VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node);
    void VisitCXXConstructExpr(CXXConstructExpr *Node);
    void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node);
    void VisitCXXExprWithTemporaries(CXXExprWithTemporaries *Node);
    void DumpCXXTemporary(CXXTemporary *Temporary);
    
    // ObjC
    void VisitObjCEncodeExpr(ObjCEncodeExpr *Node);
    void VisitObjCMessageExpr(ObjCMessageExpr* Node);
    void VisitObjCSelectorExpr(ObjCSelectorExpr *Node);
    void VisitObjCProtocolExpr(ObjCProtocolExpr *Node);
    void VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node);
    void VisitObjCImplicitSetterGetterRefExpr(
                                          ObjCImplicitSetterGetterRefExpr *Node);
    void VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node);
    void VisitObjCSuperExpr(ObjCSuperExpr *Node);
  };
}

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

void StmtDumper::DumpLocation(SourceLocation Loc) {
  SourceLocation SpellingLoc = SM->getSpellingLoc(Loc);
  
  if (SpellingLoc.isInvalid()) {
    fprintf(stderr, "<invalid sloc>");
    return;
  }

  // The general format we print out is filename:line:col, but we drop pieces
  // that haven't changed since the last loc printed.
  PresumedLoc PLoc = SM->getPresumedLoc(SpellingLoc);

  if (strcmp(PLoc.getFilename(), LastLocFilename) != 0) {
    fprintf(stderr, "%s:%u:%u", PLoc.getFilename(), PLoc.getLine(),
            PLoc.getColumn());
    LastLocFilename = PLoc.getFilename();
    LastLocLine = PLoc.getLine();
  } else if (PLoc.getLine() != LastLocLine) {
    fprintf(stderr, "line:%u:%u", PLoc.getLine(), PLoc.getColumn());
    LastLocLine = PLoc.getLine();
  } else {
    fprintf(stderr, "col:%u", PLoc.getColumn());
  }
}

void StmtDumper::DumpSourceRange(const Stmt *Node) {
  // Can't translate locations if a SourceManager isn't available.
  if (SM == 0) return;
  
  // TODO: If the parent expression is available, we can print a delta vs its
  // location.
  SourceRange R = Node->getSourceRange();
  
  fprintf(stderr, " <");
  DumpLocation(R.getBegin());
  if (R.getBegin() != R.getEnd()) {
    fprintf(stderr, ", ");
    DumpLocation(R.getEnd());
  }
  fprintf(stderr, ">");
    
  // <t2.c:123:421[blah], t2.c:412:321>

}


//===----------------------------------------------------------------------===//
//  Stmt printing methods.
//===----------------------------------------------------------------------===//

void StmtDumper::VisitStmt(Stmt *Node) {
  DumpStmt(Node);
}

void StmtDumper::DumpDeclarator(Decl *D) {
  // FIXME: Need to complete/beautify this... this code simply shows the
  // nodes are where they need to be.
  if (TypedefDecl *localType = dyn_cast<TypedefDecl>(D)) {
    fprintf(F, "\"typedef %s %s\"",
            localType->getUnderlyingType().getAsString().c_str(),
            localType->getNameAsString().c_str());
  } else if (ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    fprintf(F, "\"");
    // Emit storage class for vardecls.
    if (VarDecl *V = dyn_cast<VarDecl>(VD)) {
      if (V->getStorageClass() != VarDecl::None)
        fprintf(F, "%s ", 
                VarDecl::getStorageClassSpecifierString(V->getStorageClass()));
    }
    
    std::string Name = VD->getNameAsString();
    VD->getType().getAsStringInternal(Name, 
                          PrintingPolicy(VD->getASTContext().getLangOptions()));
    fprintf(F, "%s", Name.c_str());
    
    // If this is a vardecl with an initializer, emit it.
    if (VarDecl *V = dyn_cast<VarDecl>(VD)) {
      if (V->getInit()) {
        fprintf(F, " =\n");
        DumpSubTree(V->getInit());
      }
    }
    fprintf(F, "\"");
  } else if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    // print a free standing tag decl (e.g. "struct x;").
    const char *tagname;
    if (const IdentifierInfo *II = TD->getIdentifier())
      tagname = II->getName();
    else
      tagname = "<anonymous>";
    fprintf(F, "\"%s %s;\"", TD->getKindName(), tagname);
    // FIXME: print tag bodies.
  } else if (UsingDirectiveDecl *UD = dyn_cast<UsingDirectiveDecl>(D)) {
    // print using-directive decl (e.g. "using namespace x;")
    const char *ns;
    if (const IdentifierInfo *II = UD->getNominatedNamespace()->getIdentifier())
      ns = II->getName();
    else
      ns = "<anonymous>";
    fprintf(F, "\"%s %s;\"",UD->getDeclKindName(), ns);
  } else {
    assert(0 && "Unexpected decl");
  }
}

void StmtDumper::VisitDeclStmt(DeclStmt *Node) {
  DumpStmt(Node);
  fprintf(F,"\n");
  for (DeclStmt::decl_iterator DI = Node->decl_begin(), DE = Node->decl_end();
       DI != DE; ++DI) {
    Decl* D = *DI;
    ++IndentLevel;
    Indent();
    fprintf(F, "%p ", (void*) D);
    DumpDeclarator(D);
    if (DI+1 != DE)
      fprintf(F,"\n");
    --IndentLevel;
  }
}

void StmtDumper::VisitLabelStmt(LabelStmt *Node) {
  DumpStmt(Node);
  fprintf(F, " '%s'", Node->getName());
}

void StmtDumper::VisitGotoStmt(GotoStmt *Node) {
  DumpStmt(Node);
  fprintf(F, " '%s':%p", Node->getLabel()->getName(), (void*)Node->getLabel());
}

//===----------------------------------------------------------------------===//
//  Expr printing methods.
//===----------------------------------------------------------------------===//

void StmtDumper::VisitExpr(Expr *Node) {
  DumpExpr(Node);
}

void StmtDumper::VisitCastExpr(CastExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " ");
  switch (Node->getCastKind()) {
  case CastExpr::CK_Unknown:
    fprintf(F, "<Unknown>");
    break;
  case CastExpr::CK_BitCast:
    fprintf(F, "<BitCast>");
    break;
  case CastExpr::CK_NoOp:
    fprintf(F, "<NoOp>");
    break;
  case CastExpr::CK_DerivedToBase:
    fprintf(F, "<DerivedToBase>");
    break;
  case CastExpr::CK_Dynamic:
    fprintf(F, "<Dynamic>");
    break;
  case CastExpr::CK_ToUnion:
    fprintf(F, "<ToUnion>");
    break;
  case CastExpr::CK_ArrayToPointerDecay:
    fprintf(F, "<ArrayToPointerDecay>");
    break;
  case CastExpr::CK_NullToMemberPointer:
    fprintf(F, "<NullToMemberPointer>");
    break;
  case CastExpr::CK_BaseToDerivedMemberPointer:
    fprintf(F, "<BaseToDerivedMemberPointer>");
    break;
  case CastExpr::CK_UserDefinedConversion:
    fprintf(F, "<UserDefinedConversion>");
    break;
  }
}

void StmtDumper::VisitDeclRefExpr(DeclRefExpr *Node) {
  DumpExpr(Node);

  fprintf(F, " ");
  switch (Node->getDecl()->getKind()) {
  default: fprintf(F,"Decl"); break;
  case Decl::Function: fprintf(F,"FunctionDecl"); break;
  case Decl::Var: fprintf(F,"Var"); break;
  case Decl::ParmVar: fprintf(F,"ParmVar"); break;
  case Decl::EnumConstant: fprintf(F,"EnumConstant"); break;
  case Decl::Typedef: fprintf(F,"Typedef"); break;
  case Decl::Record: fprintf(F,"Record"); break;
  case Decl::Enum: fprintf(F,"Enum"); break;
  case Decl::CXXRecord: fprintf(F,"CXXRecord"); break;
  case Decl::ObjCInterface: fprintf(F,"ObjCInterface"); break;
  case Decl::ObjCClass: fprintf(F,"ObjCClass"); break;
  }
  
  fprintf(F, "='%s' %p", Node->getDecl()->getNameAsString().c_str(), 
          (void*)Node->getDecl());
}

void StmtDumper::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
  DumpExpr(Node);

  fprintf(F, " %sDecl='%s' %p", Node->getDecl()->getDeclKindName(), 
          Node->getDecl()->getNameAsString().c_str(), (void*)Node->getDecl());
  if (Node->isFreeIvar())
    fprintf(F, " isFreeIvar");
}

void StmtDumper::VisitPredefinedExpr(PredefinedExpr *Node) {
  DumpExpr(Node);
  switch (Node->getIdentType()) {
  default: assert(0 && "unknown case");
  case PredefinedExpr::Func:           fprintf(F, " __func__"); break;
  case PredefinedExpr::Function:       fprintf(F, " __FUNCTION__"); break;
  case PredefinedExpr::PrettyFunction: fprintf(F, " __PRETTY_FUNCTION__");break;
  }
}

void StmtDumper::VisitCharacterLiteral(CharacterLiteral *Node) {
  DumpExpr(Node);
  fprintf(F, " %d", Node->getValue());
}

void StmtDumper::VisitIntegerLiteral(IntegerLiteral *Node) {
  DumpExpr(Node);

  bool isSigned = Node->getType()->isSignedIntegerType();
  fprintf(F, " %s", Node->getValue().toString(10, isSigned).c_str());
}
void StmtDumper::VisitFloatingLiteral(FloatingLiteral *Node) {
  DumpExpr(Node);
  fprintf(F, " %f", Node->getValueAsApproximateDouble());
}

void StmtDumper::VisitStringLiteral(StringLiteral *Str) {
  DumpExpr(Str);
  // FIXME: this doesn't print wstrings right.
  fprintf(F, " %s\"", Str->isWide() ? "L" : "");

  for (unsigned i = 0, e = Str->getByteLength(); i != e; ++i) {
    switch (char C = Str->getStrData()[i]) {
    default:
      if (isprint(C))
        fputc(C, F); 
      else
        fprintf(F, "\\%03o", C);
      break;
    // Handle some common ones to make dumps prettier.
    case '\\': fprintf(F, "\\\\"); break;
    case '"':  fprintf(F, "\\\""); break;
    case '\n': fprintf(F, "\\n"); break;
    case '\t': fprintf(F, "\\t"); break;
    case '\a': fprintf(F, "\\a"); break;
    case '\b': fprintf(F, "\\b"); break;
    }
  }
  fprintf(F, "\"");
}

void StmtDumper::VisitUnaryOperator(UnaryOperator *Node) {
  DumpExpr(Node);
  fprintf(F, " %s '%s'", Node->isPostfix() ? "postfix" : "prefix",
          UnaryOperator::getOpcodeStr(Node->getOpcode()));
}
void StmtDumper::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " %s ", Node->isSizeOf() ? "sizeof" : "alignof");
  if (Node->isArgumentType())
    DumpType(Node->getArgumentType());
}

void StmtDumper::VisitMemberExpr(MemberExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " %s%s %p", Node->isArrow() ? "->" : ".",
          Node->getMemberDecl()->getNameAsString().c_str(), 
          (void*)Node->getMemberDecl());
}
void StmtDumper::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " %s", Node->getAccessor().getName());
}
void StmtDumper::VisitBinaryOperator(BinaryOperator *Node) {
  DumpExpr(Node);
  fprintf(F, " '%s'", BinaryOperator::getOpcodeStr(Node->getOpcode()));
}
void StmtDumper::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  DumpExpr(Node);
  fprintf(F, " '%s' ComputeLHSTy=",
          BinaryOperator::getOpcodeStr(Node->getOpcode()));
  DumpType(Node->getComputationLHSType());
  fprintf(F, " ComputeResultTy=");
  DumpType(Node->getComputationResultType());
}

// GNU extensions.

void StmtDumper::VisitAddrLabelExpr(AddrLabelExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " %s %p", Node->getLabel()->getName(), (void*)Node->getLabel());
}

void StmtDumper::VisitTypesCompatibleExpr(TypesCompatibleExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " ");
  DumpType(Node->getArgType1());
  fprintf(F, " ");
  DumpType(Node->getArgType2());
}

//===----------------------------------------------------------------------===//
// C++ Expressions
//===----------------------------------------------------------------------===//

void StmtDumper::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " %s<%s>", Node->getCastName(),
          Node->getTypeAsWritten().getAsString().c_str());
}

void StmtDumper::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " %s", Node->getValue() ? "true" : "false");
}

void StmtDumper::VisitCXXThisExpr(CXXThisExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " this");
}

void StmtDumper::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " functional cast to %s", 
          Node->getTypeAsWritten().getAsString().c_str());
}

void StmtDumper::VisitCXXConstructExpr(CXXConstructExpr *Node) {
  DumpExpr(Node);
  if (Node->isElidable())
    fprintf(F, " elidable");
}

void StmtDumper::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " ");
  DumpCXXTemporary(Node->getTemporary());
}

void StmtDumper::VisitCXXExprWithTemporaries(CXXExprWithTemporaries *Node) {
  DumpExpr(Node);
  ++IndentLevel;
  for (unsigned i = 0, e = Node->getNumTemporaries(); i != e; ++i) {
    fprintf(F, "\n");
    Indent();
    DumpCXXTemporary(Node->getTemporary(i));
  }
  --IndentLevel;
}

void StmtDumper::DumpCXXTemporary(CXXTemporary *Temporary) {
  fprintf(F, "(CXXTemporary %p)", (void *)Temporary);
}

//===----------------------------------------------------------------------===//
// Obj-C Expressions
//===----------------------------------------------------------------------===//

void StmtDumper::VisitObjCMessageExpr(ObjCMessageExpr* Node) {
  DumpExpr(Node);
  fprintf(F, " selector=%s", Node->getSelector().getAsString().c_str());
  IdentifierInfo* clsName = Node->getClassName();
  if (clsName) fprintf(F, " class=%s", clsName->getName());
}

void StmtDumper::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
  DumpExpr(Node);
 
  fprintf(F, " ");
  DumpType(Node->getEncodedType());
}

void StmtDumper::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
  DumpExpr(Node);
  
  fprintf(F, " ");
  fprintf(F, "%s", Node->getSelector().getAsString().c_str());
}

void StmtDumper::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
  DumpExpr(Node);
  
  fprintf(F, " ");
  fprintf(F, "%s", Node->getProtocol()->getNameAsString().c_str());
}

void StmtDumper::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
  DumpExpr(Node);

  fprintf(F, " Kind=PropertyRef Property=\"%s\"", 
          Node->getProperty()->getNameAsString().c_str());
}

void StmtDumper::VisitObjCImplicitSetterGetterRefExpr(
                                        ObjCImplicitSetterGetterRefExpr *Node) {
  DumpExpr(Node);
  
  ObjCMethodDecl *Getter = Node->getGetterMethod();
  ObjCMethodDecl *Setter = Node->getSetterMethod();
  fprintf(F, " Kind=MethodRef Getter=\"%s\" Setter=\"%s\"", 
          Getter->getSelector().getAsString().c_str(),
          Setter ? Setter->getSelector().getAsString().c_str() : "(null)");
}

void StmtDumper::VisitObjCSuperExpr(ObjCSuperExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " super");
}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

/// dump - This does a local dump of the specified AST fragment.  It dumps the
/// specified node and a few nodes underneath it, but not the whole subtree.
/// This is useful in a debugger.
void Stmt::dump(SourceManager &SM) const {
  StmtDumper P(&SM, stderr, 4);
  P.DumpSubTree(const_cast<Stmt*>(this));
  fprintf(stderr, "\n");
}

/// dump - This does a local dump of the specified AST fragment.  It dumps the
/// specified node and a few nodes underneath it, but not the whole subtree.
/// This is useful in a debugger.
void Stmt::dump() const {
  StmtDumper P(0, stderr, 4);
  P.DumpSubTree(const_cast<Stmt*>(this));
  fprintf(stderr, "\n");
}

/// dumpAll - This does a dump of the specified AST fragment and all subtrees.
void Stmt::dumpAll(SourceManager &SM) const {
  StmtDumper P(&SM, stderr, ~0U);
  P.DumpSubTree(const_cast<Stmt*>(this));
  fprintf(stderr, "\n");
}

/// dumpAll - This does a dump of the specified AST fragment and all subtrees.
void Stmt::dumpAll() const {
  StmtDumper P(0, stderr, ~0U);
  P.DumpSubTree(const_cast<Stmt*>(this));
  fprintf(stderr, "\n");
}
