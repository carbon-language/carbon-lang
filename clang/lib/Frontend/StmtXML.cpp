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
          for (Stmt::child_range i = S->children(); i; ++i)
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
  };
}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

/// dumpAll - This does a dump of the specified AST fragment and all subtrees.
void DocumentXML::PrintStmt(const Stmt *S) {
  StmtXML P(*this);
  P.DumpSubTree(const_cast<Stmt*>(S));
}

