//===--- StmtSerialization.cpp - Serialization of Statements --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the type-specific methods for serializing statements
// and expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Expr.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using namespace clang;

void Stmt::Emit(llvm::Serializer& S) const {
  S.EmitInt(getStmtClass());
  directEmit(S);
}  

Stmt* Stmt::Materialize(llvm::Deserializer& D) {
  StmtClass SC = static_cast<StmtClass>(D.ReadInt());
  
  switch (SC) {
    default:  
      assert (false && "Not implemented.");
      return NULL;
    
    case BinaryOperatorClass:
      return BinaryOperator::directMaterialize(D);
      
    case BreakStmtClass:
      return BreakStmt::directMaterialize(D);
      
    case CaseStmtClass:
      return CaseStmt::directMaterialize(D);
      
    case CompoundStmtClass:
      return CompoundStmt::directMaterialize(D);
      
    case DeclRefExprClass:
      return DeclRefExpr::directMaterialize(D);
      
    case DeclStmtClass:
      return DeclStmt::directMaterialize(D);
      
    case DefaultStmtClass:
      return DefaultStmt::directMaterialize(D);
    
    case DoStmtClass:
      return DoStmt::directMaterialize(D);

    case ForStmtClass:
      return ForStmt::directMaterialize(D);
    
    case GotoStmtClass:
      return GotoStmt::directMaterialize(D);
      
    case IfStmtClass:
      return IfStmt::directMaterialize(D);
      
    case IndirectGotoStmtClass:
      return IndirectGotoStmt::directMaterialize(D);      
      
    case IntegerLiteralClass:
      return IntegerLiteral::directMaterialize(D);
      
    case LabelStmtClass:
      return LabelStmt::directMaterialize(D);
      
    case NullStmtClass:
      return NullStmt::directMaterialize(D);
      
    case ParenExprClass:
      return ParenExpr::directMaterialize(D);
      
    case ReturnStmtClass:
      return ReturnStmt::directMaterialize(D);
      
    case SwitchStmtClass:
      return SwitchStmt::directMaterialize(D);
      
    case WhileStmtClass:
      return WhileStmt::directMaterialize(D);
  }
}

void BinaryOperator::directEmit(llvm::Serializer& S) const {
  S.EmitInt(Opc);
  S.Emit(OpLoc);;
  S.Emit(getType());
  S.EmitOwnedPtr(getLHS());
  S.EmitOwnedPtr(getRHS());
}

BinaryOperator* BinaryOperator::directMaterialize(llvm::Deserializer& D) {
  Opcode Opc = static_cast<Opcode>(D.ReadInt());
  SourceLocation OpLoc = SourceLocation::ReadVal(D);
  QualType Result = QualType::ReadVal(D);
  Expr* LHS = D.ReadOwnedPtr<Expr>();
  Expr* RHS = D.ReadOwnedPtr<Expr>();
  return new BinaryOperator(LHS,RHS,Opc,Result,OpLoc);
}

void BreakStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(BreakLoc);
}

BreakStmt* BreakStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  return new BreakStmt(Loc);
}
  
void CaseStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(CaseLoc);
  S.EmitOwnedPtr(getLHS());
  S.EmitOwnedPtr(getRHS());
  S.EmitOwnedPtr(getSubStmt());
  S.EmitPtr(getNextSwitchCase());
}

CaseStmt* CaseStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation CaseLoc = SourceLocation::ReadVal(D);
  Expr* LHS = D.ReadOwnedPtr<Expr>();
  Expr* RHS = D.ReadOwnedPtr<Expr>();
  Stmt* SubStmt = D.ReadOwnedPtr<Stmt>();
  
  CaseStmt* stmt = new CaseStmt(LHS,RHS,SubStmt,CaseLoc);
  stmt->setNextSwitchCase(D.ReadPtr<SwitchCase>());
  
  return stmt;
}

void CompoundStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(LBracLoc);
  S.Emit(RBracLoc);
  S.Emit(Body.size());
  
  for (const_body_iterator I=body_begin(), E=body_end(); I!=E; ++I)
    S.EmitOwnedPtr(*I);
}

CompoundStmt* CompoundStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation LB = SourceLocation::ReadVal(D);
  SourceLocation RB = SourceLocation::ReadVal(D);
  unsigned size = D.ReadInt();
  
  CompoundStmt* stmt = new CompoundStmt(NULL,0,LB,RB);
  
  stmt->Body.reserve(size);
  
  for (unsigned i = 0; i < size; ++i)
    stmt->Body.push_back(D.ReadOwnedPtr<Stmt>());
  
  return stmt;
}

void DeclStmt::directEmit(llvm::Serializer& S) const {
  // FIXME: special handling for struct decls.
  S.EmitOwnedPtr(getDecl());  
}

void DeclRefExpr::directEmit(llvm::Serializer& S) const {
  S.Emit(Loc);
  S.Emit(getType());
  S.EmitPtr(getDecl());
}

DeclRefExpr* DeclRefExpr::directMaterialize(llvm::Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  QualType T = QualType::ReadVal(D);
  DeclRefExpr* dr = new DeclRefExpr(NULL,T,Loc);
  D.ReadPtr(dr->D,false);  
  return dr;
}

DeclStmt* DeclStmt::directMaterialize(llvm::Deserializer& D) {
  ScopedDecl* decl = cast<ScopedDecl>(D.ReadOwnedPtr<Decl>());
  return new DeclStmt(decl);
}

void DefaultStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(DefaultLoc);
  S.EmitOwnedPtr(getSubStmt());
  S.EmitPtr(getNextSwitchCase());
}

DefaultStmt* DefaultStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  Stmt* SubStmt = D.ReadOwnedPtr<Stmt>();
  
  DefaultStmt* stmt = new DefaultStmt(Loc,SubStmt);
  stmt->setNextSwitchCase(D.ReadPtr<SwitchCase>());
  
  return stmt;
}

void DoStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(DoLoc);
  S.EmitOwnedPtr(getCond());
  S.EmitOwnedPtr(getBody());
}

DoStmt* DoStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation DoLoc = SourceLocation::ReadVal(D);
  Expr* Cond = D.ReadOwnedPtr<Expr>();
  Stmt* Body = D.ReadOwnedPtr<Stmt>();
  return new DoStmt(Body,Cond,DoLoc);
}

void ForStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(ForLoc);
  S.EmitOwnedPtr(getInit());
  S.EmitOwnedPtr(getCond());
  S.EmitOwnedPtr(getInc());
  S.EmitOwnedPtr(getBody());
}

ForStmt* ForStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation ForLoc = SourceLocation::ReadVal(D);
  Stmt* Init = D.ReadOwnedPtr<Stmt>();
  Expr* Cond = D.ReadOwnedPtr<Expr>();
  Expr* Inc = D.ReadOwnedPtr<Expr>();
  Stmt* Body = D.ReadOwnedPtr<Stmt>();
  return new ForStmt(Init,Cond,Inc,Body,ForLoc);
}

void GotoStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(GotoLoc);
  S.Emit(LabelLoc);
  S.EmitPtr(Label);
}

GotoStmt* GotoStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation GotoLoc = SourceLocation::ReadVal(D);
  SourceLocation LabelLoc = SourceLocation::ReadVal(D);
  GotoStmt* stmt = new GotoStmt(NULL,GotoLoc,LabelLoc);
  D.ReadPtr(stmt->Label); // This pointer may be backpatched later.
  return stmt;  
}

void IfStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(IfLoc);
  S.EmitOwnedPtr(getCond());
  S.EmitOwnedPtr(getThen());
  S.EmitOwnedPtr(getElse());
}

IfStmt* IfStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  Expr* Cond = D.ReadOwnedPtr<Expr>();
  Stmt* Then = D.ReadOwnedPtr<Stmt>();
  Stmt* Else = D.ReadOwnedPtr<Stmt>();
  return new IfStmt(L,Cond,Then,Else);
}

void IndirectGotoStmt::directEmit(llvm::Serializer& S) const {
  S.EmitPtr(Target);  
}

IndirectGotoStmt* IndirectGotoStmt::directMaterialize(llvm::Deserializer& D) {
  IndirectGotoStmt* stmt = new IndirectGotoStmt(NULL);
  D.ReadPtr(stmt->Target); // The target may be backpatched.
  return stmt;
}

void IntegerLiteral::directEmit(llvm::Serializer& S) const {
  S.Emit(Loc);
  S.Emit(getType());
  S.Emit(getValue());
}

IntegerLiteral* IntegerLiteral::directMaterialize(llvm::Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  QualType T = QualType::ReadVal(D);
  
  // Create a dummy APInt because it is more efficient to deserialize
  // it in place with the deserialized IntegerLiteral. (fewer copies)
  llvm::APInt temp;  
  IntegerLiteral* expr = new IntegerLiteral(temp,T,Loc);
  D.Read(expr->Value);
  
  return expr;
}

void LabelStmt::directEmit(llvm::Serializer& S) const {
  S.EmitPtr(Label);
  S.Emit(IdentLoc);
  S.EmitOwnedPtr(SubStmt);
}

LabelStmt* LabelStmt::directMaterialize(llvm::Deserializer& D) {
  IdentifierInfo* Label = D.ReadPtr<IdentifierInfo>();
  SourceLocation IdentLoc = SourceLocation::ReadVal(D);
  Stmt* SubStmt = D.ReadOwnedPtr<Stmt>();
  return new LabelStmt(IdentLoc,Label,SubStmt);
}

void NullStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(SemiLoc);
}

NullStmt* NullStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation SemiLoc = SourceLocation::ReadVal(D);
  return new NullStmt(SemiLoc);
}

void ParenExpr::directEmit(llvm::Serializer& S) const {
  S.Emit(L);
  S.Emit(R);
  S.EmitOwnedPtr(Val);
}

ParenExpr* ParenExpr::directMaterialize(llvm::Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  SourceLocation R = SourceLocation::ReadVal(D);
  Expr* val = D.ReadOwnedPtr<Expr>();  
  return new ParenExpr(L,R,val);
}  

void ReturnStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(RetLoc);
  S.EmitOwnedPtr(RetExpr);
}

ReturnStmt* ReturnStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation RetLoc = SourceLocation::ReadVal(D);
  Expr* RetExpr = D.ReadOwnedPtr<Expr>();  
  return new ReturnStmt(RetLoc,RetExpr);
}

void SwitchStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(SwitchLoc);
  S.EmitOwnedPtr(getCond());
  S.EmitOwnedPtr(getBody());
  S.EmitPtr(FirstCase);  
}

SwitchStmt* SwitchStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  Stmt* Cond = D.ReadOwnedPtr<Stmt>();
  Stmt* Body = D.ReadOwnedPtr<Stmt>();
  SwitchCase* FirstCase = cast<SwitchCase>(D.ReadPtr<Stmt>());
  
  SwitchStmt* stmt = new SwitchStmt(cast<Expr>(Cond));
  stmt->setBody(Body,Loc);
  stmt->FirstCase = FirstCase;
  
  return stmt;
}

void WhileStmt::directEmit(llvm::Serializer& S) const {
  S.Emit(WhileLoc);
  S.EmitOwnedPtr(getCond());
  S.EmitOwnedPtr(getBody());
}

WhileStmt* WhileStmt::directMaterialize(llvm::Deserializer& D) {
  SourceLocation WhileLoc = SourceLocation::ReadVal(D);
  Expr* Cond = D.ReadOwnedPtr<Expr>();
  Stmt* Body = D.ReadOwnedPtr<Stmt>();
  return new WhileStmt(Cond,Body,WhileLoc);
}
