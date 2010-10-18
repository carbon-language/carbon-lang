//===--- DeclXML.cpp - XML implementation for Decl ASTs -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the XML document class, which provides the means to
// dump out the AST in a XML form that exposes type details and other fields.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/DocumentXML.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"

namespace clang {

//---------------------------------------------------------
class DocumentXML::DeclPrinter : public DeclVisitor<DocumentXML::DeclPrinter> {
  DocumentXML& Doc;

  void addSubNodes(FunctionDecl* FD) {
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
      Visit(FD->getParamDecl(i));
      Doc.toParent();
    }
  }

  void addFunctionBody(FunctionDecl* FD) {
    if (FD->isThisDeclarationADefinition()) {
      Doc.addSubNode("Body");
      Doc.PrintStmt(FD->getBody());
      Doc.toParent();
    }
  }

  void addSubNodes(RecordDecl* RD) {
    for (RecordDecl::decl_iterator i = RD->decls_begin(),
                                   e = RD->decls_end(); i != e; ++i) {
      if (!(*i)->isImplicit()) {
        Visit(*i);
        Doc.toParent();
      }
    }
  }

  void addSubNodes(CXXRecordDecl* RD) {
    addSubNodes(cast<RecordDecl>(RD));

    if (RD->isDefinition()) {
      // FIXME: This breaks XML generation
      //Doc.addAttribute("num_bases", RD->getNumBases());

      for (CXXRecordDecl::base_class_iterator 
             base = RD->bases_begin(),
             bend = RD->bases_end();
           base != bend;
           ++base) {
        Doc.addSubNode("Base");
        Doc.addAttribute("id", base->getType());
        AccessSpecifier as = base->getAccessSpecifierAsWritten();
        const char* as_name = "";
        switch(as) {
        case AS_none:      as_name = ""; break;
        case AS_public:    as_name = "public"; break;
        case AS_protected: as_name = "protected"; break;
        case AS_private:   as_name = "private"; break;
        }
        Doc.addAttributeOptional("access", as_name);
        Doc.addAttribute("is_virtual", base->isVirtual());
        Doc.toParent();
      }
    }
  }

  void addSubNodes(EnumDecl* ED) {
    for (EnumDecl::enumerator_iterator i = ED->enumerator_begin(),
                                       e = ED->enumerator_end(); i != e; ++i) {
      Visit(*i);
      Doc.toParent();
    }
  }

  void addSubNodes(EnumConstantDecl* ECD) {
    if (ECD->getInitExpr())
      Doc.PrintStmt(ECD->getInitExpr());
  }

  void addSubNodes(FieldDecl* FdD)  {
    if (FdD->isBitField())
      Doc.PrintStmt(FdD->getBitWidth());
  }

  void addSubNodes(VarDecl* V) {
    if (V->getInit())
      Doc.PrintStmt(V->getInit());
  }

  void addSubNodes(ParmVarDecl* argDecl) {
    if (argDecl->getDefaultArg())
      Doc.PrintStmt(argDecl->getDefaultArg());
  }

  void addSubNodes(DeclContext* ns) {

    for (DeclContext::decl_iterator 
           d    = ns->decls_begin(), 
           dend = ns->decls_end();
         d != dend;
         ++d) {
      Visit(*d);
      Doc.toParent();
    }
  }

  void addSpecialAttribute(const char* pName, EnumDecl* ED) {
    const QualType& enumType = ED->getIntegerType();
    if (!enumType.isNull())
      Doc.addAttribute(pName, enumType);
  }

  void addIdAttribute(LinkageSpecDecl* ED) {
    Doc.addAttribute("id", ED);
  }

  void addIdAttribute(NamedDecl* ND) {
    Doc.addAttribute("id", ND);
  }

public:
  DeclPrinter(DocumentXML& doc) : Doc(doc) {}

#define NODE_XML( CLASS, NAME )          \
  void Visit##CLASS(CLASS* T)            \
  {                                      \
    Doc.addSubNode(NAME);

#define ID_ATTRIBUTE_XML                  addIdAttribute(T);
#define ATTRIBUTE_XML( FN, NAME )         Doc.addAttribute(NAME, T->FN);
#define ATTRIBUTE_OPT_XML( FN, NAME )     Doc.addAttributeOptional(NAME, T->FN);
#define ATTRIBUTE_FILE_LOCATION_XML       Doc.addLocation(T->getLocation());
#define ATTRIBUTE_SPECIAL_XML( FN, NAME ) addSpecialAttribute(NAME, T);

#define ATTRIBUTE_ENUM_XML( FN, NAME )  \
  {                                     \
    const char* pAttributeName = NAME;  \
    const bool optional = false;             \
    switch (T->FN) {                    \
    default: assert(0 && "unknown enum value");

#define ATTRIBUTE_ENUM_OPT_XML( FN, NAME )  \
  {                                     \
    const char* pAttributeName = NAME;  \
    const bool optional = true;              \
    switch (T->FN) {                    \
    default: assert(0 && "unknown enum value");

#define ENUM_XML( VALUE, NAME )         case VALUE: if ((!optional) || NAME[0]) Doc.addAttribute(pAttributeName, NAME); break;
#define END_ENUM_XML                    } }
#define END_NODE_XML                    }

#define SUB_NODE_XML( CLASS )           addSubNodes(T);
#define SUB_NODE_SEQUENCE_XML( CLASS )  addSubNodes(T);
#define SUB_NODE_OPT_XML( CLASS )       addSubNodes(T);

#define SUB_NODE_FN_BODY_XML            addFunctionBody(T);

#include "clang/Frontend/DeclXML.def"
};


//---------------------------------------------------------
void DocumentXML::writeDeclToXML(Decl *D) {
  DeclPrinter(*this).Visit(D);
  toParent();
}

//---------------------------------------------------------
} // NS clang

