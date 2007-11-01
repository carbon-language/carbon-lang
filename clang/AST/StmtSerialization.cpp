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

#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using llvm::Serializer;
using llvm::Deserializer;
using llvm::SerializeTrait;

using namespace clang;


namespace {
class StmtSerializer : public StmtVisitor<StmtSerializer> {
  Serializer& S;
public:  
  StmtSerializer(Serializer& S) : S(S) {}

  void VisitDeclStmt(DeclStmt* Stmt);
  void VisitNullStmt(NullStmt* Stmt);
  void VisitCompoundStmt(CompoundStmt* Stmt);
  
  void VisitStmt(Stmt*) { assert("Not Implemented yet"); }  
};
} // end anonymous namespace


void SerializeTrait<Stmt>::Emit(Serializer& S, const Stmt& stmt) {
  S.EmitInt(stmt.getStmtClass());
  
  StmtSerializer SS(S);
  SS.Visit(const_cast<Stmt*>(&stmt));
}

void StmtSerializer::VisitDeclStmt(DeclStmt* Stmt) {
  // FIXME
 // S.EmitOwnedPtr(Stmt->getDecl());  
}

void StmtSerializer::VisitNullStmt(NullStmt* Stmt) {
  S.Emit(Stmt->getSemiLoc());
}

void StmtSerializer::VisitCompoundStmt(CompoundStmt* Stmt) {
  S.Emit(Stmt->getLBracLoc());
  S.Emit(Stmt->getRBracLoc());

  CompoundStmt::body_iterator I=Stmt->body_begin(), E=Stmt->body_end();

  S.EmitInt(E-I);  
  
  for ( ; I != E; ++I )
    S.EmitOwnedPtr(*I);
}

Stmt* SerializeTrait<Stmt>::Materialize(Deserializer& D) {
  unsigned sClass = D.ReadInt();
  
  switch (sClass) {
    default:
      assert(false && "No matching statement class.");
      return NULL;
    
    case Stmt::DeclStmtClass:
      return NULL; // FIXME
//      return new DeclStmt(D.ReadOwnedPtr<ScopedDecl>());

    case Stmt::NullStmtClass:
      return new NullStmt(SourceLocation::ReadVal(D));
    
    case Stmt::CompoundStmtClass: {
      SourceLocation LBracLoc = SourceLocation::ReadVal(D);
      SourceLocation RBracLoc = SourceLocation::ReadVal(D);
      unsigned NumStmts = D.ReadInt();
      llvm::SmallVector<Stmt*, 16> Body;
      
      for (unsigned i = 0 ; i <  NumStmts; ++i)
        Body.push_back(D.ReadOwnedPtr<Stmt>());  
      
      return new CompoundStmt(&Body[0],NumStmts,LBracLoc,RBracLoc);
    }
  }
}
