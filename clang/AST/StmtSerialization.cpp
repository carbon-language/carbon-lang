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
using llvm::Serializer;
using llvm::Deserializer;

void Stmt::Emit(Serializer& S) const {
  S.FlushRecord();
  S.EmitInt(getStmtClass());
  EmitImpl(S);
  S.FlushRecord();
}  

Stmt* Stmt::Materialize(Deserializer& D) {
  StmtClass SC = static_cast<StmtClass>(D.ReadInt());
  
  switch (SC) {
    default:  
      assert (false && "Not implemented.");
      return NULL;
    
    case AddrLabelExprClass:
      return AddrLabelExpr::CreateImpl(D);
      
    case ArraySubscriptExprClass:
      return ArraySubscriptExpr::CreateImpl(D);
      
    case BinaryOperatorClass:
      return BinaryOperator::CreateImpl(D);
      
    case BreakStmtClass:
      return BreakStmt::CreateImpl(D);
     
    case CallExprClass:
      return CallExpr::CreateImpl(D);
      
    case CaseStmtClass:
      return CaseStmt::CreateImpl(D);
    
    case CastExprClass:
      return CastExpr::CreateImpl(D);
      
    case CharacterLiteralClass:
      return CharacterLiteral::CreateImpl(D);
      
    case CompoundAssignOperatorClass:
      return CompoundAssignOperator::CreateImpl(D);
      
    case CompoundStmtClass:
      return CompoundStmt::CreateImpl(D);
    
    case ConditionalOperatorClass:
      return ConditionalOperator::CreateImpl(D);
      
    case ContinueStmtClass:
      return ContinueStmt::CreateImpl(D);
      
    case DeclRefExprClass:
      return DeclRefExpr::CreateImpl(D);
      
    case DeclStmtClass:
      return DeclStmt::CreateImpl(D);
      
    case DefaultStmtClass:
      return DefaultStmt::CreateImpl(D);
    
    case DoStmtClass:
      return DoStmt::CreateImpl(D);
      
    case FloatingLiteralClass:
      return FloatingLiteral::CreateImpl(D);

    case ForStmtClass:
      return ForStmt::CreateImpl(D);
    
    case GotoStmtClass:
      return GotoStmt::CreateImpl(D);
      
    case IfStmtClass:
      return IfStmt::CreateImpl(D);
    
    case ImaginaryLiteralClass:
      return ImaginaryLiteral::CreateImpl(D);
      
    case ImplicitCastExprClass:
      return ImplicitCastExpr::CreateImpl(D);
      
    case IndirectGotoStmtClass:
      return IndirectGotoStmt::CreateImpl(D);      
      
    case IntegerLiteralClass:
      return IntegerLiteral::CreateImpl(D);
      
    case LabelStmtClass:
      return LabelStmt::CreateImpl(D);
      
    case NullStmtClass:
      return NullStmt::CreateImpl(D);
      
    case ParenExprClass:
      return ParenExpr::CreateImpl(D);
      
    case PreDefinedExprClass:
      return PreDefinedExpr::CreateImpl(D);
      
    case ReturnStmtClass:
      return ReturnStmt::CreateImpl(D);
    
    case StmtExprClass:
      return StmtExpr::CreateImpl(D);
      
    case StringLiteralClass:
      return StringLiteral::CreateImpl(D);
      
    case SwitchStmtClass:
      return SwitchStmt::CreateImpl(D);
      
    case UnaryOperatorClass:
      return UnaryOperator::CreateImpl(D);
      
    case WhileStmtClass:
      return WhileStmt::CreateImpl(D);
  }
}

void AddrLabelExpr::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.Emit(AmpAmpLoc);
  S.Emit(LabelLoc);
  S.EmitPtr(Label);
}

AddrLabelExpr* AddrLabelExpr::CreateImpl(Deserializer& D) {
  QualType t = QualType::ReadVal(D);
  SourceLocation AALoc = SourceLocation::ReadVal(D);
  SourceLocation LLoc = SourceLocation::ReadVal(D);
  AddrLabelExpr* expr = new AddrLabelExpr(AALoc,LLoc,NULL,t);
  D.ReadPtr(expr->Label); // Pointer may be backpatched.
  return expr;
}

void ArraySubscriptExpr::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.Emit(RBracketLoc);
  S.BatchEmitOwnedPtrs(getLHS(),getRHS());
}

ArraySubscriptExpr* ArraySubscriptExpr::CreateImpl(Deserializer& D) {
  QualType t = QualType::ReadVal(D);
  SourceLocation L = SourceLocation::ReadVal(D);
  Expr *LHS, *RHS;
  D.BatchReadOwnedPtrs(LHS,RHS);
  return new ArraySubscriptExpr(LHS,RHS,t,L);  
}

void BinaryOperator::EmitImpl(Serializer& S) const {
  S.EmitInt(Opc);
  S.Emit(OpLoc);;
  S.Emit(getType());
  S.BatchEmitOwnedPtrs(getLHS(),getRHS());
}

BinaryOperator* BinaryOperator::CreateImpl(Deserializer& D) {
  Opcode Opc = static_cast<Opcode>(D.ReadInt());
  SourceLocation OpLoc = SourceLocation::ReadVal(D);
  QualType Result = QualType::ReadVal(D);
  Expr *LHS, *RHS;
  D.BatchReadOwnedPtrs(LHS,RHS);

  return new BinaryOperator(LHS,RHS,Opc,Result,OpLoc);
}

void BreakStmt::EmitImpl(Serializer& S) const {
  S.Emit(BreakLoc);
}

BreakStmt* BreakStmt::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  return new BreakStmt(Loc);
}

void CallExpr::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.Emit(RParenLoc);
  S.EmitInt(NumArgs);
  S.BatchEmitOwnedPtrs(NumArgs+1,SubExprs);  
}

CallExpr* CallExpr::CreateImpl(Deserializer& D) {
  QualType t = QualType::ReadVal(D);
  SourceLocation L = SourceLocation::ReadVal(D);
  unsigned NumArgs = D.ReadInt();
  Expr** SubExprs = new Expr*[NumArgs+1];
  D.BatchReadOwnedPtrs(NumArgs+1,SubExprs);

  return new CallExpr(SubExprs,NumArgs,t,L);  
}

void CaseStmt::EmitImpl(Serializer& S) const {
  S.Emit(CaseLoc);
  S.EmitPtr(getNextSwitchCase());
  S.BatchEmitOwnedPtrs((unsigned) END_EXPR,&SubExprs[0]);
}

CaseStmt* CaseStmt::CreateImpl(Deserializer& D) {
  SourceLocation CaseLoc = SourceLocation::ReadVal(D);
  CaseStmt* stmt = new CaseStmt(NULL,NULL,NULL,CaseLoc);  
  D.ReadPtr(stmt->NextSwitchCase);
  D.BatchReadOwnedPtrs((unsigned) END_EXPR,&stmt->SubExprs[0]);
  return stmt;
}

void CastExpr::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.Emit(Loc);
  S.EmitOwnedPtr(Op);
}

CastExpr* CastExpr::CreateImpl(Deserializer& D) {
  QualType t = QualType::ReadVal(D);
  SourceLocation Loc = SourceLocation::ReadVal(D);
  Expr* Op = D.ReadOwnedPtr<Expr>();
  return new CastExpr(t,Op,Loc);
}
  

void CharacterLiteral::EmitImpl(Serializer& S) const {
  S.Emit(Value);
  S.Emit(Loc);
  S.Emit(getType());
}

CharacterLiteral* CharacterLiteral::CreateImpl(Deserializer& D) {
  unsigned value = D.ReadInt();
  SourceLocation Loc = SourceLocation::ReadVal(D);
  QualType T = QualType::ReadVal(D);
  return new CharacterLiteral(value,T,Loc);
}

void CompoundAssignOperator::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.Emit(ComputationType);
  S.Emit(getOperatorLoc());
  S.EmitInt(getOpcode());
  S.BatchEmitOwnedPtrs(getLHS(),getRHS());
}

CompoundAssignOperator* 
CompoundAssignOperator::CreateImpl(Deserializer& D) {
  QualType t = QualType::ReadVal(D);
  QualType c = QualType::ReadVal(D);
  SourceLocation L = SourceLocation::ReadVal(D);
  Opcode Opc = static_cast<Opcode>(D.ReadInt());
  Expr* LHS, *RHS;
  D.BatchReadOwnedPtrs(LHS,RHS);
  
  return new CompoundAssignOperator(LHS,RHS,Opc,t,c,L);
}

void CompoundStmt::EmitImpl(Serializer& S) const {
  S.Emit(LBracLoc);
  S.Emit(RBracLoc);
  S.Emit(Body.size());
  
  for (const_body_iterator I=body_begin(), E=body_end(); I!=E; ++I)
    S.EmitOwnedPtr(*I);
}

CompoundStmt* CompoundStmt::CreateImpl(Deserializer& D) {
  SourceLocation LB = SourceLocation::ReadVal(D);
  SourceLocation RB = SourceLocation::ReadVal(D);
  unsigned size = D.ReadInt();
  
  CompoundStmt* stmt = new CompoundStmt(NULL,0,LB,RB);
  
  stmt->Body.reserve(size);
  
  for (unsigned i = 0; i < size; ++i)
    stmt->Body.push_back(D.ReadOwnedPtr<Stmt>());
  
  return stmt;
}

void ConditionalOperator::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.BatchEmitOwnedPtrs((unsigned) END_EXPR, SubExprs);
}

ConditionalOperator* ConditionalOperator::CreateImpl(Deserializer& D) {
  QualType t = QualType::ReadVal(D);
  ConditionalOperator* c = new ConditionalOperator(NULL,NULL,NULL,t);
  D.BatchReadOwnedPtrs((unsigned) END_EXPR, c->SubExprs);
  return c;
}

void ContinueStmt::EmitImpl(Serializer& S) const {
  S.Emit(ContinueLoc);
}

ContinueStmt* ContinueStmt::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  return new ContinueStmt(Loc);
}

void DeclStmt::EmitImpl(Serializer& S) const {
  // FIXME: special handling for struct decls.
  S.EmitOwnedPtr(getDecl());  
}

void DeclRefExpr::EmitImpl(Serializer& S) const {
  S.Emit(Loc);
  S.Emit(getType());
  S.EmitPtr(getDecl());
}

DeclRefExpr* DeclRefExpr::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  QualType T = QualType::ReadVal(D);
  DeclRefExpr* dr = new DeclRefExpr(NULL,T,Loc);
  D.ReadPtr(dr->D,false);  
  return dr;
}

void ObjCIvarRefExpr::EmitImpl(Serializer& S) const {
  S.Emit(Loc);
  S.Emit(getType());
  S.EmitPtr(getDecl());
}

ObjCIvarRefExpr* ObjCIvarRefExpr::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  QualType T = QualType::ReadVal(D);
  ObjCIvarRefExpr* dr = new ObjCIvarRefExpr(NULL,T,Loc);
  D.ReadPtr(dr->D,false);  
  return dr;
}

DeclStmt* DeclStmt::CreateImpl(Deserializer& D) {
  ScopedDecl* decl = cast<ScopedDecl>(D.ReadOwnedPtr<Decl>());
  return new DeclStmt(decl);
}

void DefaultStmt::EmitImpl(Serializer& S) const {
  S.Emit(DefaultLoc);
  S.EmitOwnedPtr(getSubStmt());
  S.EmitPtr(getNextSwitchCase());
}

DefaultStmt* DefaultStmt::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  Stmt* SubStmt = D.ReadOwnedPtr<Stmt>();
  
  DefaultStmt* stmt = new DefaultStmt(Loc,SubStmt);
  stmt->setNextSwitchCase(D.ReadPtr<SwitchCase>());
  
  return stmt;
}

void DoStmt::EmitImpl(Serializer& S) const {
  S.Emit(DoLoc);
  S.EmitOwnedPtr(getCond());
  S.EmitOwnedPtr(getBody());
}

DoStmt* DoStmt::CreateImpl(Deserializer& D) {
  SourceLocation DoLoc = SourceLocation::ReadVal(D);
  Expr* Cond = D.ReadOwnedPtr<Expr>();
  Stmt* Body = D.ReadOwnedPtr<Stmt>();
  return new DoStmt(Body,Cond,DoLoc);
}

void FloatingLiteral::EmitImpl(Serializer& S) const {
  S.Emit(Loc);
  S.Emit(getType());
  S.Emit(Value);
}

FloatingLiteral* FloatingLiteral::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  QualType t = QualType::ReadVal(D);
  llvm::APFloat Val = llvm::APFloat::ReadVal(D);
  FloatingLiteral* expr = new FloatingLiteral(Val,t,Loc);
  return expr;
}

void ForStmt::EmitImpl(Serializer& S) const {
  S.Emit(ForLoc);
  S.EmitOwnedPtr(getInit());
  S.EmitOwnedPtr(getCond());
  S.EmitOwnedPtr(getInc());
  S.EmitOwnedPtr(getBody());
}

ForStmt* ForStmt::CreateImpl(Deserializer& D) {
  SourceLocation ForLoc = SourceLocation::ReadVal(D);
  Stmt* Init = D.ReadOwnedPtr<Stmt>();
  Expr* Cond = D.ReadOwnedPtr<Expr>();
  Expr* Inc = D.ReadOwnedPtr<Expr>();
  Stmt* Body = D.ReadOwnedPtr<Stmt>();
  return new ForStmt(Init,Cond,Inc,Body,ForLoc);
}

void GotoStmt::EmitImpl(Serializer& S) const {
  S.Emit(GotoLoc);
  S.Emit(LabelLoc);
  S.EmitPtr(Label);
}

GotoStmt* GotoStmt::CreateImpl(Deserializer& D) {
  SourceLocation GotoLoc = SourceLocation::ReadVal(D);
  SourceLocation LabelLoc = SourceLocation::ReadVal(D);
  GotoStmt* stmt = new GotoStmt(NULL,GotoLoc,LabelLoc);
  D.ReadPtr(stmt->Label); // This pointer may be backpatched later.
  return stmt;  
}

void IfStmt::EmitImpl(Serializer& S) const {
  S.Emit(IfLoc);
  S.EmitOwnedPtr(getCond());
  S.EmitOwnedPtr(getThen());
  S.EmitOwnedPtr(getElse());
}

IfStmt* IfStmt::CreateImpl(Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  Expr* Cond = D.ReadOwnedPtr<Expr>();
  Stmt* Then = D.ReadOwnedPtr<Stmt>();
  Stmt* Else = D.ReadOwnedPtr<Stmt>();
  return new IfStmt(L,Cond,Then,Else);
}

void ImaginaryLiteral::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.EmitOwnedPtr(Val);    
}

ImaginaryLiteral* ImaginaryLiteral::CreateImpl(Deserializer& D) {
  QualType t = QualType::ReadVal(D);
  Expr* expr = D.ReadOwnedPtr<Expr>();
  assert (isa<FloatingLiteral>(expr) || isa<IntegerLiteral>(expr));
  return new ImaginaryLiteral(expr,t);
}

void ImplicitCastExpr::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.EmitOwnedPtr(Op);
}

ImplicitCastExpr* ImplicitCastExpr::CreateImpl(Deserializer& D) {
  QualType t = QualType::ReadVal(D);
  Expr* Op = D.ReadOwnedPtr<Expr>();
  return new ImplicitCastExpr(t,Op);
}

void IndirectGotoStmt::EmitImpl(Serializer& S) const {
  S.EmitOwnedPtr(Target);  
}

IndirectGotoStmt* IndirectGotoStmt::CreateImpl(Deserializer& D) {
  Expr* Target = D.ReadOwnedPtr<Expr>();
  return new IndirectGotoStmt(Target);
}

void IntegerLiteral::EmitImpl(Serializer& S) const {
  S.Emit(Loc);
  S.Emit(getType());
  S.Emit(getValue());
}

IntegerLiteral* IntegerLiteral::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  QualType T = QualType::ReadVal(D);
  
  // Create a dummy APInt because it is more efficient to deserialize
  // it in place with the deserialized IntegerLiteral. (fewer copies)
  llvm::APInt temp;  
  IntegerLiteral* expr = new IntegerLiteral(temp,T,Loc);
  D.Read(expr->Value);
  
  return expr;
}

void LabelStmt::EmitImpl(Serializer& S) const {
  S.EmitPtr(Label);
  S.Emit(IdentLoc);
  S.EmitOwnedPtr(SubStmt);
}

LabelStmt* LabelStmt::CreateImpl(Deserializer& D) {
  IdentifierInfo* Label = D.ReadPtr<IdentifierInfo>();
  SourceLocation IdentLoc = SourceLocation::ReadVal(D);
  Stmt* SubStmt = D.ReadOwnedPtr<Stmt>();
  return new LabelStmt(IdentLoc,Label,SubStmt);
}

void NullStmt::EmitImpl(Serializer& S) const {
  S.Emit(SemiLoc);
}

NullStmt* NullStmt::CreateImpl(Deserializer& D) {
  SourceLocation SemiLoc = SourceLocation::ReadVal(D);
  return new NullStmt(SemiLoc);
}

void ParenExpr::EmitImpl(Serializer& S) const {
  S.Emit(L);
  S.Emit(R);
  S.EmitOwnedPtr(Val);
}

ParenExpr* ParenExpr::CreateImpl(Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  SourceLocation R = SourceLocation::ReadVal(D);
  Expr* val = D.ReadOwnedPtr<Expr>();  
  return new ParenExpr(L,R,val);
}

void PreDefinedExpr::EmitImpl(Serializer& S) const {
  S.Emit(Loc);
  S.EmitInt(getIdentType());
  S.Emit(getType());  
}

PreDefinedExpr* PreDefinedExpr::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  IdentType it = static_cast<IdentType>(D.ReadInt());
  QualType Q = QualType::ReadVal(D);
  return new PreDefinedExpr(Loc,Q,it);
}

void ReturnStmt::EmitImpl(Serializer& S) const {
  S.Emit(RetLoc);
  S.EmitOwnedPtr(RetExpr);
}

ReturnStmt* ReturnStmt::CreateImpl(Deserializer& D) {
  SourceLocation RetLoc = SourceLocation::ReadVal(D);
  Expr* RetExpr = D.ReadOwnedPtr<Expr>();  
  return new ReturnStmt(RetLoc,RetExpr);
}

void StmtExpr::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.Emit(LParenLoc);
  S.Emit(RParenLoc);
  S.EmitOwnedPtr(SubStmt);
}

StmtExpr* StmtExpr::CreateImpl(Deserializer& D) {
  QualType t = QualType::ReadVal(D);
  SourceLocation L = SourceLocation::ReadVal(D);
  SourceLocation R = SourceLocation::ReadVal(D);
  CompoundStmt* SubStmt = cast<CompoundStmt>(D.ReadOwnedPtr<Stmt>());
  return new StmtExpr(SubStmt,t,L,R);
}

void StringLiteral::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.Emit(firstTokLoc);
  S.Emit(lastTokLoc);
  S.EmitBool(isWide());
  S.Emit(getByteLength());

  for (unsigned i = 0 ; i < ByteLength; ++i)
    S.EmitInt(StrData[i]);
}

StringLiteral* StringLiteral::CreateImpl(Deserializer& D) {
  QualType t = QualType::ReadVal(D);
  SourceLocation firstTokLoc = SourceLocation::ReadVal(D);
  SourceLocation lastTokLoc = SourceLocation::ReadVal(D);
  bool isWide = D.ReadBool();
  unsigned ByteLength = D.ReadInt();
  
  StringLiteral* sl = new StringLiteral(NULL,0,isWide,t,firstTokLoc,lastTokLoc);

  char* StrData = new char[ByteLength];
  for (unsigned i = 0; i < ByteLength; ++i)
    StrData[i] = (char) D.ReadInt();

  sl->ByteLength = ByteLength;
  sl->StrData = StrData;
  
  return sl;
}

void SwitchStmt::EmitImpl(Serializer& S) const {
  S.Emit(SwitchLoc);
  S.EmitOwnedPtr(getCond());
  S.EmitOwnedPtr(getBody());
  S.EmitPtr(FirstCase);  
}

SwitchStmt* SwitchStmt::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  Stmt* Cond = D.ReadOwnedPtr<Stmt>();
  Stmt* Body = D.ReadOwnedPtr<Stmt>();
  SwitchCase* FirstCase = cast<SwitchCase>(D.ReadPtr<Stmt>());
  
  SwitchStmt* stmt = new SwitchStmt(cast<Expr>(Cond));
  stmt->setBody(Body,Loc);
  stmt->FirstCase = FirstCase;
  
  return stmt;
}

void UnaryOperator::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.Emit(Loc);
  S.EmitInt(Opc);
  S.EmitOwnedPtr(Val);
}

UnaryOperator* UnaryOperator::CreateImpl(Deserializer& D) {
  QualType t = QualType::ReadVal(D);
  SourceLocation L = SourceLocation::ReadVal(D);
  Opcode Opc = static_cast<Opcode>(D.ReadInt());
  Expr* Val = D.ReadOwnedPtr<Expr>();
  return new UnaryOperator(Val,Opc,t,L);
}

void WhileStmt::EmitImpl(Serializer& S) const {
  S.Emit(WhileLoc);
  S.EmitOwnedPtr(getCond());
  S.EmitOwnedPtr(getBody());
}

WhileStmt* WhileStmt::CreateImpl(Deserializer& D) {
  SourceLocation WhileLoc = SourceLocation::ReadVal(D);
  Expr* Cond = D.ReadOwnedPtr<Expr>();
  Stmt* Body = D.ReadOwnedPtr<Stmt>();
  return new WhileStmt(Cond,Body,WhileLoc);
}
