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
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringExtras.h"

namespace clang {

//---------------------------------------------------------
DocumentXML::DocumentXML(const std::string& rootName, llvm::raw_ostream& out) :
  Out(out),
  Ctx(0),
  HasCurrentNodeSubNodes(false) {
  NodeStack.push(rootName);
  Out << "<?xml version=\"1.0\"?>\n<" << rootName;
}

//---------------------------------------------------------
DocumentXML& DocumentXML::addSubNode(const std::string& name) {
  if (!HasCurrentNodeSubNodes)
    Out << ">\n";
  NodeStack.push(name);
  HasCurrentNodeSubNodes = false;
  Indent();
  Out << "<" << NodeStack.top();
  return *this;
}

//---------------------------------------------------------
void DocumentXML::Indent() {
  for (size_t i = 0, e = (NodeStack.size() - 1) * 2; i < e; ++i)
    Out << ' ';
}

//---------------------------------------------------------
DocumentXML& DocumentXML::toParent() {
  assert(NodeStack.size() > 1 && "too much backtracking");

  if (HasCurrentNodeSubNodes) {
    Indent();
    Out << "</" << NodeStack.top() << ">\n";
  } else
    Out << "/>\n";
  NodeStack.pop();
  HasCurrentNodeSubNodes = true;
  return *this;
}

//---------------------------------------------------------
namespace {

enum tIdType { ID_NORMAL, ID_FILE, ID_LABEL, ID_LAST };

unsigned getNewId(tIdType idType) {
  static unsigned int idCounts[ID_LAST] = { 0 };
  return ++idCounts[idType];
}

//---------------------------------------------------------
inline std::string getPrefixedId(unsigned uId, tIdType idType) {
  static const char idPrefix[ID_LAST] = { '_', 'f', 'l' };
  char buffer[20];
  char* BufPtr = llvm::utohex_buffer(uId, buffer + 20);
  *--BufPtr = idPrefix[idType];
  return BufPtr;
}

//---------------------------------------------------------
template<class T, class V>
bool addToMap(T& idMap, const V& value, tIdType idType = ID_NORMAL) {
  typename T::iterator i = idMap.find(value);
  bool toAdd = i == idMap.end();
  if (toAdd)
    idMap.insert(typename T::value_type(value, getNewId(idType)));
  return toAdd;
}

} // anon NS


//---------------------------------------------------------
std::string DocumentXML::escapeString(const char* pStr,
                                      std::string::size_type len) {
  std::string value;
  value.reserve(len + 1);
  char buffer[16];
  for (unsigned i = 0; i < len; ++i) {
    switch (char C = pStr[i]) {
    default:
      if (isprint(C))
        value += C;
      else {
        sprintf(buffer, "\\%03o", C);
        value += buffer;
      }
      break;

    case '\n': value += "\\n"; break;
    case '\t': value += "\\t"; break;
    case '\a': value += "\\a"; break;
    case '\b': value += "\\b"; break;
    case '\r': value += "\\r"; break;

    case '&': value += "&amp;"; break;
    case '<': value += "&lt;"; break;
    case '>': value += "&gt;"; break;
    case '"': value += "&quot;"; break;
    case '\'':  value += "&apos;"; break;

    }
  }
  return value;
}

//---------------------------------------------------------
void DocumentXML::finalize() {
  assert(NodeStack.size() == 1 && "not completely backtracked");

  addSubNode("ReferenceSection");
  addSubNode("Types");

  for (XML::IdMap<QualType>::iterator i = Types.begin(), e = Types.end();
       i != e; ++i) {
    if (i->first.getCVRQualifiers() != 0) {
      writeTypeToXML(i->first);
      addAttribute("id", getPrefixedId(i->second, ID_NORMAL));
      toParent();
    }
  }

  for (XML::IdMap<const Type*>::iterator i = BasicTypes.begin(),
         e = BasicTypes.end(); i != e; ++i) {
    writeTypeToXML(i->first);
    addAttribute("id", getPrefixedId(i->second, ID_NORMAL));
    toParent();
  }


  toParent().addSubNode("Contexts");

  for (XML::IdMap<const DeclContext*>::iterator i = Contexts.begin(),
         e = Contexts.end(); i != e; ++i) {
    addSubNode(i->first->getDeclKindName());
    addAttribute("id", getPrefixedId(i->second, ID_NORMAL));
    if (const NamedDecl *ND = dyn_cast<NamedDecl>(i->first))
      addAttribute("name", ND->getNameAsString());
    if (const TagDecl *TD = dyn_cast<TagDecl>(i->first))
      addAttribute("type", getPrefixedId(BasicTypes[TD->getTypeForDecl()], ID_NORMAL));
    else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(i->first))
      addAttribute("type", getPrefixedId(BasicTypes[FD->getType()->getAsFunctionType()], ID_NORMAL));

    if (const DeclContext* parent = i->first->getParent())
      addAttribute("context", parent);
    toParent();
  }

  toParent().addSubNode("Files");

  for (XML::IdMap<std::string>::iterator i = SourceFiles.begin(),
         e = SourceFiles.end(); i != e; ++i) {
    addSubNode("File");
    addAttribute("id", getPrefixedId(i->second, ID_FILE));
    addAttribute("name", escapeString(i->first.c_str(), i->first.size()));
    toParent();
  }

  toParent().toParent();

  // write the root closing node (which has always subnodes)
  Out << "</" << NodeStack.top() << ">\n";
}

//---------------------------------------------------------
void DocumentXML::addAttribute(const char* pAttributeName,
                               const QualType& pType) {
  addTypeRecursively(pType);
  addAttribute(pAttributeName, getPrefixedId(Types[pType], ID_NORMAL));
}

//---------------------------------------------------------
void DocumentXML::addPtrAttribute(const char* pAttributeName,
                                  const Type* pType) {
  addTypeRecursively(pType);
  addAttribute(pAttributeName, getPrefixedId(BasicTypes[pType], ID_NORMAL));
}

//---------------------------------------------------------
void DocumentXML::addTypeRecursively(const QualType& pType)
{
  if (addToMap(Types, pType))
  {
    addTypeRecursively(pType.getTypePtr());
    // beautifier: a non-qualified type shall be transparent
    if (pType.getCVRQualifiers() == 0)
    {
      Types[pType] = BasicTypes[pType.getTypePtr()];
    }
  }
}

//---------------------------------------------------------
void DocumentXML::addTypeRecursively(const Type* pType)
{
  if (addToMap(BasicTypes, pType))
  {
    addParentTypes(pType);
/*
    // FIXME: doesn't work in the immediate streaming approach
    if (const VariableArrayType *VAT = dyn_cast<VariableArrayType>(pType))
    {
      addSubNode("VariableArraySizeExpression");
      PrintStmt(VAT->getSizeExpr());
      toParent();
    }
*/
  }
}

//---------------------------------------------------------
void DocumentXML::addPtrAttribute(const char* pName, const DeclContext* DC)
{
  addContextsRecursively(DC);
  addAttribute(pName, getPrefixedId(Contexts[DC], ID_NORMAL));
}

//---------------------------------------------------------
void DocumentXML::addPtrAttribute(const char* pAttributeName, const NamedDecl* D)
{
  if (const DeclContext* DC = dyn_cast<DeclContext>(D))
  {
    addContextsRecursively(DC);
    addAttribute(pAttributeName, getPrefixedId(Contexts[DC], ID_NORMAL));
  }
  else
  {
    addToMap(Decls, D);
    addAttribute(pAttributeName, getPrefixedId(Decls[D], ID_NORMAL));
  }
}

//---------------------------------------------------------
void DocumentXML::addPtrAttribute(const char* pName, const NamespaceDecl* D)
{
  addPtrAttribute(pName, static_cast<const DeclContext*>(D));
}

//---------------------------------------------------------
void DocumentXML::addContextsRecursively(const DeclContext *DC)
{
  if (DC != 0 && addToMap(Contexts, DC))
  {
    addContextsRecursively(DC->getParent());
  }
}

//---------------------------------------------------------
void DocumentXML::addSourceFileAttribute(const std::string& fileName)
{
  addToMap(SourceFiles, fileName, ID_FILE);
  addAttribute("file", getPrefixedId(SourceFiles[fileName], ID_FILE));
}


//---------------------------------------------------------
void DocumentXML::addPtrAttribute(const char* pName, const LabelStmt* L)
{
  addToMap(Labels, L, ID_LABEL);
  addAttribute(pName, getPrefixedId(Labels[L], ID_LABEL));
}


//---------------------------------------------------------
PresumedLoc DocumentXML::addLocation(const SourceLocation& Loc)
{
  SourceManager& SM = Ctx->getSourceManager();
  SourceLocation SpellingLoc = SM.getSpellingLoc(Loc);
  PresumedLoc PLoc;
  if (!SpellingLoc.isInvalid())
  {
    PLoc = SM.getPresumedLoc(SpellingLoc);
    addSourceFileAttribute(PLoc.getFilename());
    addAttribute("line", PLoc.getLine());
    addAttribute("col", PLoc.getColumn());
  }
  // else there is no error in some cases (eg. CXXThisExpr)
  return PLoc;
}

//---------------------------------------------------------
void DocumentXML::addLocationRange(const SourceRange& R)
{
  PresumedLoc PStartLoc = addLocation(R.getBegin());
  if (R.getBegin() != R.getEnd())
  {
    SourceManager& SM = Ctx->getSourceManager();
    SourceLocation SpellingLoc = SM.getSpellingLoc(R.getEnd());
    if (!SpellingLoc.isInvalid())
    {
      PresumedLoc PLoc = SM.getPresumedLoc(SpellingLoc);
      if (PStartLoc.isInvalid() ||
          strcmp(PLoc.getFilename(), PStartLoc.getFilename()) != 0) {
        addToMap(SourceFiles, PLoc.getFilename(), ID_FILE);
        addAttribute("endfile", PLoc.getFilename());
        addAttribute("endline", PLoc.getLine());
        addAttribute("endcol", PLoc.getColumn());
      } else if (PLoc.getLine() != PStartLoc.getLine()) {
        addAttribute("endline", PLoc.getLine());
        addAttribute("endcol", PLoc.getColumn());
      } else {
        addAttribute("endcol", PLoc.getColumn());
      }
    }
  }
}

//---------------------------------------------------------
void DocumentXML::PrintDecl(Decl *D)
{
  writeDeclToXML(D);
}

//---------------------------------------------------------
} // NS clang

