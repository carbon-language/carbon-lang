//===--- DocumentXML.cpp - XML document for ASTs --------------------------===//
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
#include "clang/AST/TypeVisitor.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"

namespace clang {
  namespace XML {
    namespace {

//---------------------------------------------------------
class TypeWriter : public TypeVisitor<TypeWriter> {
  DocumentXML& Doc;

public:
  TypeWriter(DocumentXML& doc) : Doc(doc) {}

#define NODE_XML( CLASS, NAME )          \
  void Visit##CLASS(const CLASS* T) {          \
    Doc.addSubNode(NAME);

#define ID_ATTRIBUTE_XML                // done by the Document class itself
#define ATTRIBUTE_XML( FN, NAME )       Doc.addAttribute(NAME, T->FN);
#define TYPE_ATTRIBUTE_XML( FN )        ATTRIBUTE_XML(FN, "type")
#define CONTEXT_ATTRIBUTE_XML( FN )     ATTRIBUTE_XML(FN, "context")
#define ATTRIBUTE_OPT_XML( FN, NAME )   Doc.addAttributeOptional(NAME, T->FN);

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

#include "clang/Frontend/TypeXML.def"

};

//---------------------------------------------------------
    } // anon clang
  } // NS XML

//---------------------------------------------------------
class DocumentXML::TypeAdder : public TypeVisitor<DocumentXML::TypeAdder> {
  DocumentXML& Doc;

  void addIfType(const Type* pType) {
    Doc.addTypeRecursively(pType);
  }

  void addIfType(const QualType& pType) {
    Doc.addTypeRecursively(pType);
  }

  template<class T> void addIfType(T) {}

public:
  TypeAdder(DocumentXML& doc) : Doc(doc) {}

#define NODE_XML( CLASS, NAME )          \
  void Visit##CLASS(const CLASS* T) \
  {

#define ID_ATTRIBUTE_XML
#define TYPE_ATTRIBUTE_XML( FN )        Doc.addTypeRecursively(T->FN);
#define CONTEXT_ATTRIBUTE_XML( FN )
#define ATTRIBUTE_XML( FN, NAME )       addIfType(T->FN);
#define ATTRIBUTE_OPT_XML( FN, NAME )
#define ATTRIBUTE_ENUM_XML( FN, NAME )
#define ATTRIBUTE_ENUM_OPT_XML( FN, NAME )
#define ENUM_XML( VALUE, NAME )
#define END_ENUM_XML
#define END_NODE_XML                    }

#include "clang/Frontend/TypeXML.def"
};

//---------------------------------------------------------
void DocumentXML::addParentTypes(const Type* pType) {
  TypeAdder(*this).Visit(pType);
}

//---------------------------------------------------------
void DocumentXML::writeTypeToXML(const Type* pType) {
  XML::TypeWriter(*this).Visit(const_cast<Type*>(pType));
}

//---------------------------------------------------------
void DocumentXML::writeTypeToXML(const QualType& pType) {
  XML::TypeWriter(*this).VisitQualType(const_cast<QualType*>(&pType));
}

//---------------------------------------------------------
} // NS clang

