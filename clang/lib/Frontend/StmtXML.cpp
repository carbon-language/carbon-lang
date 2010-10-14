//===--- StmtXML.cpp - XML implementation for Stmt ASTs ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt::dumpXML methods, which dump out the
// AST to an XML document.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/DocumentXML.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/SourceManager.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// StmtXML Visitor
//===----------------------------------------------------------------------===//

namespace  {
  class StmtXML : public StmtVisitor<StmtXML> {
    DocumentXML&  Doc;

    //static const char *getOpcodeStr(UnaryOperator::Opcode Op);
    //static const char *getOpcodeStr(BinaryOperator::Opcode Op);


  void addSpecialAttribute(const char* pName, StringLiteral* Str) {
    Doc.addAttribute(pName, Doc.escapeString(Str->getString().data(),
                                             Str->getString().size()));
  }

  void addSpecialAttribute(const char* pName, SizeOfAlignOfExpr* S) {
    if (S->isArgumentType())
      Doc.addAttribute(pName, S->getArgumentType());
  }

  void addSpecialAttribute(const char* pName, CXXTypeidExpr* S) {
    if (S->isTypeOperand())
      Doc.addAttribute(pName, S->getTypeOperand());
  }


  public:
    StmtXML(DocumentXML& doc)
      : Doc(doc) {
    }

    void DumpSubTree(Stmt *S) {
      if (S) {
        Visit(S);
        if (DeclStmt* DS = dyn_cast<DeclStmt>(S)) {
          for (DeclStmt::decl_iterator DI = DS->decl_begin(),
                 DE = DS->decl_end(); DI != DE; ++DI) {
            Doc.PrintDecl(*DI);
          }
        } else {
          for (Stmt::child_iterator i = S->child_begin(), e = S->child_end();
               i != e; ++i)
            DumpSubTree(*i);
        }
        Doc.toParent();
      } else {
        Doc.addSubNode("NULL").toParent();
      }
    }


#define NODE_XML( CLASS, NAME )          \
  void Visit##CLASS(CLASS* S)            \
  {                                      \
    typedef CLASS tStmtType;             \
    Doc.addSubNode(NAME);

#define ATTRIBUTE_XML( FN, NAME )         Doc.addAttribute(NAME, S->FN);
#define TYPE_ATTRIBUTE_XML( FN )          ATTRIBUTE_XML(FN, "type")
#define ATTRIBUTE_OPT_XML( FN, NAME )     Doc.addAttributeOptional(NAME, S->FN);
#define ATTRIBUTE_SPECIAL_XML( FN, NAME ) addSpecialAttribute(NAME, S);
#define ATTRIBUTE_FILE_LOCATION_XML       Doc.addLocationRange(S->getSourceRange());


#define ATTRIBUTE_ENUM_XML( FN, NAME )  \
  {                                     \
    const char* pAttributeName = NAME;  \
    const bool optional = false;        \
    switch (S->FN) {                    \
    default: assert(0 && "unknown enum value");

#define ATTRIBUTE_ENUM_OPT_XML( FN, NAME )  \
  {                                         \
    const char* pAttributeName = NAME;      \
    const bool optional = true;             \
    switch (S->FN) {                        \
    default: assert(0 && "unknown enum value");

#define ENUM_XML( VALUE, NAME )         case VALUE: if ((!optional) || NAME[0]) Doc.addAttribute(pAttributeName, NAME); break;
#define END_ENUM_XML                    } }
#define END_NODE_XML                    }

#define ID_ATTRIBUTE_XML                Doc.addAttribute("id", S);
#define SUB_NODE_XML( CLASS )
#define SUB_NODE_SEQUENCE_XML( CLASS )
#define SUB_NODE_OPT_XML( CLASS )

#include "clang/Frontend/StmtXML.def"

#if (0)
    // Stmts.
    void VisitStmt(Stmt *Node);
    void VisitDeclStmt(DeclStmt *Node);
    void VisitLabelStmt(LabelStmt *Node);
    void VisitGotoStmt(GotoStmt *Node);

    // Exprs
    void VisitExpr(Expr *Node);
    void VisitDeclRefExpr(DeclRefExpr *Node);
    void VisitPredefinedExpr(PredefinedExpr *Node);
    void VisitCharacterLiteral(CharacterLiteral *Node);
    void VisitIntegerLiteral(IntegerLiteral *Node);
    void VisitFloatingLiteral(FloatingLiteral *Node);
    void VisitStringLiteral(StringLiteral *Str);
    void VisitUnaryOperator(UnaryOperator *Node);
    void VisitOffsetOfExpr(OffsetOfExpr *Node);
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

    // ObjC
    void VisitObjCEncodeExpr(ObjCEncodeExpr *Node);
    void VisitObjCMessageExpr(ObjCMessageExpr* Node);
    void VisitObjCSelectorExpr(ObjCSelectorExpr *Node);
    void VisitObjCProtocolExpr(ObjCProtocolExpr *Node);
    void VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node);
    void VisitObjCImplicitSetterGetterRefExpr(
                        ObjCImplicitSetterGetterRefExpr *Node);
    void VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node);
#endif
  };
}

//===----------------------------------------------------------------------===//
//  Stmt printing methods.
//===----------------------------------------------------------------------===//
#if (0)
void StmtXML::VisitStmt(Stmt *Node) {
  // nothing special to do
}

void StmtXML::VisitDeclStmt(DeclStmt *Node) {
  for (DeclStmt::decl_iterator DI = Node->decl_begin(), DE = Node->decl_end();
       DI != DE; ++DI) {
    Doc.PrintDecl(*DI);
  }
}

void StmtXML::VisitLabelStmt(LabelStmt *Node) {
  Doc.addAttribute("name", Node->getName());
}

void StmtXML::VisitGotoStmt(GotoStmt *Node) {
  Doc.addAttribute("name", Node->getLabel()->getName());
}

//===----------------------------------------------------------------------===//
//  Expr printing methods.
//===----------------------------------------------------------------------===//

void StmtXML::VisitExpr(Expr *Node) {
  DumpExpr(Node);
}

void StmtXML::VisitDeclRefExpr(DeclRefExpr *Node) {
  DumpExpr(Node);

  const char* pKind;
  switch (Node->getDecl()->getKind()) {
  case Decl::Function: pKind = "FunctionDecl"; break;
  case Decl::Var: pKind = "Var"; break;
  case Decl::ParmVar: pKind = "ParmVar"; break;
  case Decl::EnumConstant: pKind = "EnumConstant"; break;
  case Decl::Typedef: pKind = "Typedef"; break;
  case Decl::Record: pKind = "Record"; break;
  case Decl::Enum: pKind = "Enum"; break;
  case Decl::CXXRecord: pKind = "CXXRecord"; break;
  case Decl::ObjCInterface: pKind = "ObjCInterface"; break;
  case Decl::ObjCClass: pKind = "ObjCClass"; break;
  default: pKind = "Decl"; break;
  }

  Doc.addAttribute("kind", pKind);
  Doc.addAttribute("name", Node->getDecl()->getNameAsString());
  Doc.addRefAttribute(Node->getDecl());
}

void StmtXML::VisitPredefinedExpr(PredefinedExpr *Node) {
  DumpExpr(Node);
  switch (Node->getIdentType()) {
  default: assert(0 && "unknown case");
  case PredefinedExpr::Func:           Doc.addAttribute("predefined", " __func__"); break;
  case PredefinedExpr::Function:       Doc.addAttribute("predefined", " __FUNCTION__"); break;
  case PredefinedExpr::PrettyFunction: Doc.addAttribute("predefined", " __PRETTY_FUNCTION__");break;
  }
}

void StmtXML::VisitCharacterLiteral(CharacterLiteral *Node) {
  DumpExpr(Node);
  Doc.addAttribute("value", Node->getValue());
}

void StmtXML::VisitIntegerLiteral(IntegerLiteral *Node) {
  DumpExpr(Node);
  bool isSigned = Node->getType()->isSignedIntegerType();
  Doc.addAttribute("value", Node->getValue().toString(10, isSigned));
}

void StmtXML::VisitFloatingLiteral(FloatingLiteral *Node) {
  DumpExpr(Node);
  // FIXME: output float as written in source (no approximation or the like)
  //Doc.addAttribute("value", Node->getValueAsApproximateDouble()));
  Doc.addAttribute("value", "FIXME");
}

void StmtXML::VisitStringLiteral(StringLiteral *Str) {
  DumpExpr(Str);
  if (Str->isWide())
    Doc.addAttribute("is_wide", "1");

  Doc.addAttribute("value", Doc.escapeString(Str->getStrData(), Str->getByteLength()));
}


const char *StmtXML::getOpcodeStr(UnaryOperator::Opcode Op) {
  switch (Op) {
  default: assert(0 && "Unknown unary operator");
  case UnaryOperator::PostInc: return "postinc";
  case UnaryOperator::PostDec: return "postdec";
  case UnaryOperator::PreInc:  return "preinc";
  case UnaryOperator::PreDec:  return "predec";
  case UnaryOperator::AddrOf:  return "addrof";
  case UnaryOperator::Deref:   return "deref";
  case UnaryOperator::Plus:    return "plus";
  case UnaryOperator::Minus:   return "minus";
  case UnaryOperator::Not:     return "not";
  case UnaryOperator::LNot:    return "lnot";
  case UnaryOperator::Real:    return "__real";
  case UnaryOperator::Imag:    return "__imag";
  case UnaryOperator::Extension: return "__extension__";
  }
}


const char *StmtXML::getOpcodeStr(BinaryOperator::Opcode Op) {
  switch (Op) {
  default: assert(0 && "Unknown binary operator");
  case BinaryOperator::PtrMemD:   return "ptrmemd";
  case BinaryOperator::PtrMemI:   return "ptrmemi";
  case BinaryOperator::Mul:       return "mul";
  case BinaryOperator::Div:       return "div";
  case BinaryOperator::Rem:       return "rem";
  case BinaryOperator::Add:       return "add";
  case BinaryOperator::Sub:       return "sub";
  case BinaryOperator::Shl:       return "shl";
  case BinaryOperator::Shr:       return "shr";
  case BinaryOperator::LT:        return "lt";
  case BinaryOperator::GT:        return "gt";
  case BinaryOperator::LE:        return "le";
  case BinaryOperator::GE:        return "ge";
  case BinaryOperator::EQ:        return "eq";
  case BinaryOperator::NE:        return "ne";
  case BinaryOperator::And:       return "and";
  case BinaryOperator::Xor:       return "xor";
  case BinaryOperator::Or:        return "or";
  case BinaryOperator::LAnd:      return "land";
  case BinaryOperator::LOr:       return "lor";
  case BinaryOperator::Assign:    return "assign";
  case BinaryOperator::MulAssign: return "mulassign";
  case BinaryOperator::DivAssign: return "divassign";
  case BinaryOperator::RemAssign: return "remassign";
  case BinaryOperator::AddAssign: return "addassign";
  case BinaryOperator::SubAssign: return "subassign";
  case BinaryOperator::ShlAssign: return "shlassign";
  case BinaryOperator::ShrAssign: return "shrassign";
  case BinaryOperator::AndAssign: return "andassign";
  case BinaryOperator::XorAssign: return "xorassign";
  case BinaryOperator::OrAssign:  return "orassign";
  case BinaryOperator::Comma:     return "comma";
  }
}

void StmtXML::VisitUnaryOperator(UnaryOperator *Node) {
  DumpExpr(Node);
  Doc.addAttribute("op_code", getOpcodeStr(Node->getOpcode()));
}

void StmtXML::OffsetOfExpr(OffsetOfExpr *Node) {
  DumpExpr(Node);
}

void StmtXML::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  DumpExpr(Node);
  Doc.addAttribute("is_sizeof", Node->isSizeOf() ? "sizeof" : "alignof");
  Doc.addAttribute("is_type", Node->isArgumentType() ? "1" : "0");
  if (Node->isArgumentType())
    DumpTypeExpr(Node->getArgumentType());
}

void StmtXML::VisitMemberExpr(MemberExpr *Node) {
  DumpExpr(Node);
  Doc.addAttribute("is_deref", Node->isArrow() ? "1" : "0");
  Doc.addAttribute("name", Node->getMemberDecl()->getNameAsString());
  Doc.addRefAttribute(Node->getMemberDecl());
}

void StmtXML::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  DumpExpr(Node);
  Doc.addAttribute("name", Node->getAccessor().getName());
}

void StmtXML::VisitBinaryOperator(BinaryOperator *Node) {
  DumpExpr(Node);
  Doc.addAttribute("op_code", getOpcodeStr(Node->getOpcode()));
}

void StmtXML::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  VisitBinaryOperator(Node);
/* FIXME: is this needed in the AST?
  DumpExpr(Node);
  CurrentNode = CurrentNode->addSubNode("ComputeLHSTy");
  DumpType(Node->getComputationLHSType());
  CurrentNode = CurrentNode->Parent->addSubNode("ComputeResultTy");
  DumpType(Node->getComputationResultType());
  Doc.toParent();
*/
}

// GNU extensions.

void StmtXML::VisitAddrLabelExpr(AddrLabelExpr *Node) {
  DumpExpr(Node);
  Doc.addAttribute("name", Node->getLabel()->getName());
}

void StmtXML::VisitTypesCompatibleExpr(TypesCompatibleExpr *Node) {
  DumpExpr(Node);
  DumpTypeExpr(Node->getArgType1());
  DumpTypeExpr(Node->getArgType2());
}

//===----------------------------------------------------------------------===//
// C++ Expressions
//===----------------------------------------------------------------------===//

void StmtXML::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
  DumpExpr(Node);
  Doc.addAttribute("kind", Node->getCastName());
  DumpTypeExpr(Node->getTypeAsWritten());
}

void StmtXML::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
  DumpExpr(Node);
  Doc.addAttribute("value", Node->getValue() ? "true" : "false");
}

void StmtXML::VisitCXXThisExpr(CXXThisExpr *Node) {
  DumpExpr(Node);
}

void StmtXML::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
  DumpExpr(Node);
  DumpTypeExpr(Node->getTypeAsWritten());
}

//===----------------------------------------------------------------------===//
// Obj-C Expressions
//===----------------------------------------------------------------------===//

void StmtXML::VisitObjCMessageExpr(ObjCMessageExpr* Node) {
  DumpExpr(Node);
  Doc.addAttribute("selector", Node->getSelector().getAsString());
  IdentifierInfo* clsName = Node->getClassName();
  if (clsName)
    Doc.addAttribute("class", clsName->getName());
}

void StmtXML::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
  DumpExpr(Node);
  DumpTypeExpr(Node->getEncodedType());
}

void StmtXML::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
  DumpExpr(Node);
  Doc.addAttribute("selector", Node->getSelector().getAsString());
}

void StmtXML::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
  DumpExpr(Node);
  Doc.addAttribute("protocol", Node->getProtocol()->getNameAsString());
}

void StmtXML::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
  DumpExpr(Node);
  Doc.addAttribute("property", Node->getProperty()->getNameAsString());
}

void StmtXML::VisitObjCImplicitSetterGetterRefExpr(
                             ObjCImplicitSetterGetterRefExpr *Node) {
  DumpExpr(Node);
  ObjCMethodDecl *Getter = Node->getGetterMethod();
  ObjCMethodDecl *Setter = Node->getSetterMethod();
  Doc.addAttribute("Getter", Getter->getSelector().getAsString());
  Doc.addAttribute("Setter", Setter ? Setter->getSelector().getAsString().c_str() : "(null)");
}

void StmtXML::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
  DumpExpr(Node);
  Doc.addAttribute("kind", Node->getDecl()->getDeclKindName());
  Doc.addAttribute("decl", Node->getDecl()->getNameAsString());
  if (Node->isFreeIvar())
    Doc.addAttribute("isFreeIvar", "1");
}
#endif
//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

/// dumpAll - This does a dump of the specified AST fragment and all subtrees.
void DocumentXML::PrintStmt(const Stmt *S) {
  StmtXML P(*this);
  P.DumpSubTree(const_cast<Stmt*>(S));
}

