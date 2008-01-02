//===--- StmtSerialization.cpp - Serialization of Statements --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

Stmt* Stmt::Create(Deserializer& D) {
  StmtClass SC = static_cast<StmtClass>(D.ReadInt());
  
  switch (SC) {
    default:  
      assert (false && "Not implemented.");
      return NULL;
    
    case AddrLabelExprClass:
      return AddrLabelExpr::CreateImpl(D);
      
    case ArraySubscriptExprClass:
      return ArraySubscriptExpr::CreateImpl(D);
      
    case AsmStmtClass:
      return AsmStmt::CreateImpl(D);
      
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
      
    case CompoundLiteralExprClass:
      return CompoundLiteralExpr::CreateImpl(D);
      
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
      
    case InitListExprClass:
      return InitListExpr::CreateImpl(D);
      
    case IntegerLiteralClass:
      return IntegerLiteral::CreateImpl(D);
      
    case LabelStmtClass:
      return LabelStmt::CreateImpl(D);
      
    case MemberExprClass:
      return MemberExpr::CreateImpl(D);
      
    case NullStmtClass:
      return NullStmt::CreateImpl(D);
      
    case ParenExprClass:
      return ParenExpr::CreateImpl(D);
      
    case PreDefinedExprClass:
      return PreDefinedExpr::CreateImpl(D);
      
    case ReturnStmtClass:
      return ReturnStmt::CreateImpl(D);
    
    case SizeOfAlignOfTypeExprClass:
      return SizeOfAlignOfTypeExpr::CreateImpl(D);
      
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
      
    //==--------------------------------------==//
    //    Objective C
    //==--------------------------------------==//
    
    case ObjcAtCatchStmtClass:
      return ObjcAtCatchStmt::CreateImpl(D);
      
    case ObjcAtFinallyStmtClass:
      return ObjcAtFinallyStmt::CreateImpl(D);
      
    case ObjcAtThrowStmtClass:
      return ObjcAtThrowStmt::CreateImpl(D);
      
    case ObjcAtTryStmtClass:
      return ObjcAtTryStmt::CreateImpl(D);
    
    case ObjCEncodeExprClass:
      return ObjCEncodeExpr::CreateImpl(D);
      
    case ObjCIvarRefExprClass:
      return ObjCIvarRefExpr::CreateImpl(D);
      
    case ObjCSelectorExprClass:
      return ObjCSelectorExpr::CreateImpl(D);
      
    case ObjCStringLiteralClass:
      return ObjCStringLiteral::CreateImpl(D);
  }
}

//===----------------------------------------------------------------------===//
//   C Serialization
//===----------------------------------------------------------------------===//

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

void AsmStmt::EmitImpl(Serializer& S) const {
  S.Emit(AsmLoc);
  
  getAsmString()->EmitImpl(S);
  S.Emit(RParenLoc);  

  S.EmitBool(IsVolatile);
  S.EmitInt(NumOutputs);
  S.EmitInt(NumInputs);

  unsigned size = NumOutputs + NumInputs;

  for (unsigned i = 0; i < size; ++i)
    S.EmitCStr(Names[i].c_str());
  
  for (unsigned i = 0; i < size; ++i)
    Constraints[i]->EmitImpl(S);

  for (unsigned i = 0; i < size; ++i)
    S.EmitOwnedPtr(Exprs[i]);
  
  S.EmitInt(Clobbers.size());
  for (unsigned i = 0, e = Clobbers.size(); i != e; ++i)
    Clobbers[i]->EmitImpl(S);
}

AsmStmt* AsmStmt::CreateImpl(Deserializer& D) {
  SourceLocation ALoc = SourceLocation::ReadVal(D);
  StringLiteral *AsmStr = StringLiteral::CreateImpl(D);
  SourceLocation PLoc = SourceLocation::ReadVal(D);
  
  bool IsVolatile = D.ReadBool();
  AsmStmt *Stmt = new AsmStmt(ALoc, IsVolatile, 0, 0, 0, 0, 0,  
                              AsmStr, 
                              0, 0, PLoc);  

  Stmt->NumOutputs = D.ReadInt();
  Stmt->NumInputs = D.ReadInt();
  
  unsigned size = Stmt->NumOutputs + Stmt->NumInputs;

  Stmt->Names.reserve(size);
  for (unsigned i = 0; i < size; ++i) {
    std::vector<char> data;
    D.ReadCStr(data, false);
    
    Stmt->Names.push_back(std::string(&data[0], data.size()));
  }    

  Stmt->Constraints.reserve(size);
  for (unsigned i = 0; i < size; ++i)
    Stmt->Constraints.push_back(StringLiteral::CreateImpl(D));
  
  Stmt->Exprs.reserve(size);
  for (unsigned i = 0; i < size; ++i)
    Stmt->Exprs.push_back(D.ReadOwnedPtr<Expr>());
  
  unsigned NumClobbers = D.ReadInt();
  Stmt->Clobbers.reserve(NumClobbers);
  for (unsigned i = 0; i < NumClobbers; ++i)
    Stmt->Clobbers.push_back(StringLiteral::CreateImpl(D));
  
  return Stmt;
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

void CompoundLiteralExpr::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  S.Emit(getLParenLoc());
  S.EmitOwnedPtr(Init);
}

CompoundLiteralExpr* CompoundLiteralExpr::CreateImpl(Deserializer& D) {
  QualType Q = QualType::ReadVal(D);
  SourceLocation L = SourceLocation::ReadVal(D);
  Expr* Init = D.ReadOwnedPtr<Expr>();
  return new CompoundLiteralExpr(L, Q, Init);
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
  
  // Some DeclRefExprs can actually hold the owning reference to a FunctionDecl.
  // This occurs when an implicitly defined function is called, and
  // the decl does not appear in the source file.  We thus check if the
  // decl pointer has been registered, and if not, emit an owned pointer.

  // FIXME: While this will work for serialization, it won't work for
  //  memory management.  The only reason this works for serialization is
  //  because we are tracking all serialized pointers.  Either DeclRefExpr
  //  needs an explicit bit indicating that it owns the the object,
  //  or we need a different ownership model.
  
  const Decl* d = getDecl();
  
  if (!S.isRegistered(d)) {
    assert (isa<FunctionDecl>(d) 
     && "DeclRefExpr can only own FunctionDecls for implicitly def. funcs.");

    S.EmitBool(true);
    S.EmitOwnedPtr(d);
  }
  else {
    S.EmitBool(false);
    S.EmitPtr(d);
  }
}

DeclRefExpr* DeclRefExpr::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  QualType T = QualType::ReadVal(D);  
  bool OwnsDecl = D.ReadBool();
  ValueDecl* decl;
  
  if (!OwnsDecl)
    D.ReadPtr(decl,false); // No backpatching.
  else
    decl = cast<ValueDecl>(D.ReadOwnedPtr<Decl>());
  
  return new DeclRefExpr(decl,T,Loc);
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
  S.EmitBool(isExact());
  S.Emit(Value);
}

FloatingLiteral* FloatingLiteral::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  QualType t = QualType::ReadVal(D);
  bool isExact = D.ReadBool();
  llvm::APFloat Val = llvm::APFloat::ReadVal(D);
  FloatingLiteral* expr = new FloatingLiteral(Val,&isExact,t,Loc);
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

void InitListExpr::EmitImpl(Serializer& S) const {
  S.Emit(LBraceLoc);
  S.Emit(RBraceLoc);
  S.EmitInt(NumInits);
  S.BatchEmitOwnedPtrs(NumInits,InitExprs);
}

InitListExpr* InitListExpr::CreateImpl(Deserializer& D) {
  InitListExpr* expr = new InitListExpr();
  expr->LBraceLoc = SourceLocation::ReadVal(D);
  expr->RBraceLoc = SourceLocation::ReadVal(D);
  expr->NumInits = D.ReadInt();
  assert(expr->NumInits);
  expr->InitExprs = new Expr*[expr->NumInits];
  D.BatchReadOwnedPtrs(expr->NumInits,expr->InitExprs);
  return expr;
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

void MemberExpr::EmitImpl(Serializer& S) const {
  S.Emit(MemberLoc);
  S.EmitPtr(MemberDecl);
  S.EmitBool(IsArrow);
  S.EmitOwnedPtr(Base);
}

MemberExpr* MemberExpr::CreateImpl(Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  FieldDecl* MemberDecl = cast<FieldDecl>(D.ReadPtr<Decl>());
  bool IsArrow = D.ReadBool();
  Expr* base = D.ReadOwnedPtr<Expr>();
  
  return new MemberExpr(base,IsArrow,MemberDecl,L); 
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

void SizeOfAlignOfTypeExpr::EmitImpl(Serializer& S) const {
  S.EmitBool(isSizeof);
  S.Emit(Ty);
  S.Emit(getType());
  S.Emit(OpLoc);
  S.Emit(RParenLoc);
}

SizeOfAlignOfTypeExpr* SizeOfAlignOfTypeExpr::CreateImpl(Deserializer& D) {
  bool isSizeof = D.ReadBool();
  QualType Ty = QualType::ReadVal(D);
  QualType Res = QualType::ReadVal(D);
  SourceLocation OpLoc = SourceLocation::ReadVal(D);
  SourceLocation RParenLoc = SourceLocation::ReadVal(D);
  
  return new SizeOfAlignOfTypeExpr(isSizeof,Ty,Res,OpLoc,RParenLoc);  
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

//===----------------------------------------------------------------------===//
//   Objective C Serialization
//===----------------------------------------------------------------------===//

void ObjcAtCatchStmt::EmitImpl(Serializer& S) const {
  S.Emit(AtCatchLoc);
  S.Emit(RParenLoc);
  S.EmitPtr(NextAtCatchStmt);
  S.BatchEmitOwnedPtrs((unsigned) END_EXPR,&SubExprs[0]);
}

ObjcAtCatchStmt* ObjcAtCatchStmt::CreateImpl(Deserializer& D) {
  SourceLocation AtCatchLoc = SourceLocation::ReadVal(D);
  SourceLocation RParenLoc = SourceLocation::ReadVal(D);
  
  ObjcAtCatchStmt* stmt = new ObjcAtCatchStmt(AtCatchLoc,RParenLoc);
  
  D.ReadPtr(stmt->NextAtCatchStmt); // Allows backpatching.
  D.BatchReadOwnedPtrs((unsigned) END_EXPR, &stmt->SubExprs[0]);

  return stmt;
}

void ObjcAtFinallyStmt::EmitImpl(Serializer& S) const {
  S.Emit(AtFinallyLoc);
  S.EmitOwnedPtr(AtFinallyStmt);
}

ObjcAtFinallyStmt* ObjcAtFinallyStmt::CreateImpl(Deserializer& D) {
  SourceLocation Loc = SourceLocation::ReadVal(D);
  Stmt* AtFinallyStmt = D.ReadOwnedPtr<Stmt>();
  return new ObjcAtFinallyStmt(Loc,AtFinallyStmt);  
}

void ObjcAtThrowStmt::EmitImpl(Serializer& S) const {
  S.Emit(AtThrowLoc);
  S.EmitOwnedPtr(Throw);
}

ObjcAtThrowStmt* ObjcAtThrowStmt::CreateImpl(Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  Stmt* Throw = D.ReadOwnedPtr<Stmt>();
  return new ObjcAtThrowStmt(L,Throw);  
}
  
void ObjcAtTryStmt::EmitImpl(Serializer& S) const {
  S.Emit(AtTryLoc);
  S.BatchEmitOwnedPtrs((unsigned) END_EXPR, &SubStmts[0]);
}

ObjcAtTryStmt* ObjcAtTryStmt::CreateImpl(Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  ObjcAtTryStmt* stmt = new ObjcAtTryStmt(L,NULL,NULL,NULL);
  D.BatchReadOwnedPtrs((unsigned) END_EXPR, &stmt->SubStmts[0]);
  return stmt;
}

void ObjCEncodeExpr::EmitImpl(Serializer& S) const {
  S.Emit(AtLoc);
  S.Emit(RParenLoc);
  S.Emit(getType());
  S.Emit(EncType);
}

ObjCEncodeExpr* ObjCEncodeExpr::CreateImpl(Deserializer& D) {
  SourceLocation AtLoc = SourceLocation::ReadVal(D);
  SourceLocation RParenLoc = SourceLocation::ReadVal(D);  
  QualType T = QualType::ReadVal(D);
  QualType ET = QualType::ReadVal(D);
  return new ObjCEncodeExpr(T,ET,AtLoc,RParenLoc);
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

void ObjCSelectorExpr::EmitImpl(Serializer& S) const {
  S.Emit(AtLoc);
  S.Emit(RParenLoc);
  S.Emit(getType());
  S.Emit(SelName);
}

ObjCSelectorExpr* ObjCSelectorExpr::CreateImpl(Deserializer& D) {
  SourceLocation AtLoc = SourceLocation::ReadVal(D);
  SourceLocation RParenLoc = SourceLocation::ReadVal(D);
  QualType T = QualType::ReadVal(D);
  Selector SelName = Selector::ReadVal(D);
  
  return new ObjCSelectorExpr(T,SelName,AtLoc,RParenLoc);
}

void ObjCStringLiteral::EmitImpl(Serializer& S) const {
  S.Emit(AtLoc);
  S.Emit(getType());
  S.EmitOwnedPtr(String);
}

ObjCStringLiteral* ObjCStringLiteral::CreateImpl(Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  QualType T = QualType::ReadVal(D);
  StringLiteral* String = cast<StringLiteral>(D.ReadOwnedPtr<Stmt>());
  return new ObjCStringLiteral(String,T,L);
}
