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
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringExtras.h"

namespace clang {

//--------------------------------------------------------- 
struct DocumentXML::NodeXML
{
  std::string               Name;
  NodeXML*                  Parent;

  NodeXML(const std::string& name, NodeXML* parent) :
    Name(name),
    Parent(parent)
  {}
};

//--------------------------------------------------------- 
DocumentXML::DocumentXML(const std::string& rootName, llvm::raw_ostream& out) :
  Root(new NodeXML(rootName, 0)),
  CurrentNode(Root),
  Out(out),
  Ctx(0),
  CurrentIndent(0),
  HasCurrentNodeSubNodes(false)
{
  Out << "<?xml version=\"1.0\"?>\n<" << rootName;
}

//--------------------------------------------------------- 
DocumentXML::~DocumentXML()
{
  assert(CurrentNode == Root && "not completely backtracked");
  delete Root;
}

//--------------------------------------------------------- 
DocumentXML& DocumentXML::addSubNode(const std::string& name)
{
  if (!HasCurrentNodeSubNodes)
  {
    Out << ">\n";
  }
  CurrentNode = new NodeXML(name, CurrentNode);
  HasCurrentNodeSubNodes = false;
  CurrentIndent += 2;
  Indent();
  Out << "<" << CurrentNode->Name;
  return *this;
}

//--------------------------------------------------------- 
void DocumentXML::Indent() 
{
  for (int i = 0; i < CurrentIndent; ++i)
    Out << ' ';
}

//--------------------------------------------------------- 
DocumentXML& DocumentXML::toParent() 
{ 
  assert(CurrentNode != Root && "to much backtracking");

  if (HasCurrentNodeSubNodes)
  {
    Indent();
    Out << "</" << CurrentNode->Name << ">\n";
  }
  else
  {
    Out << "/>\n";
  }
  NodeXML* NodeToDelete = CurrentNode;
  CurrentNode = CurrentNode->Parent; 
  delete NodeToDelete;
  HasCurrentNodeSubNodes = true;
  CurrentIndent -= 2;
  return *this; 
}

//--------------------------------------------------------- 
namespace {

enum tIdType { ID_NORMAL, ID_FILE, ID_LAST };

unsigned getNewId(tIdType idType)
{
  static unsigned int idCounts[ID_LAST] = { 0 };
  return ++idCounts[idType];
}

//--------------------------------------------------------- 
inline std::string getPrefixedId(unsigned uId, tIdType idType)
{
  static const char idPrefix[ID_LAST] = { '_', 'f' };
  char buffer[20];
  char* BufPtr = llvm::utohex_buffer(uId, buffer + 20);
  *--BufPtr = idPrefix[idType];
  return BufPtr;
}

//--------------------------------------------------------- 
template<class T, class V>
bool addToMap(T& idMap, const V& value, tIdType idType = ID_NORMAL)
{
  typename T::iterator i = idMap.find(value);
  bool toAdd = i == idMap.end();
  if (toAdd) 
  {
    idMap.insert(typename T::value_type(value, getNewId(idType)));
  }
  return toAdd;
}

} // anon NS

//--------------------------------------------------------- 
std::string DocumentXML::escapeString(const char* pStr, std::string::size_type len)
{
  std::string value;
  value.reserve(len + 1);
  char buffer[16];
  for (unsigned i = 0; i < len; ++i) {
    switch (char C = pStr[i]) {
    default:
      if (isprint(C))
        value += C;
      else
      {
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
void DocumentXML::finalize()
{
  assert(CurrentNode == Root && "not completely backtracked");

  addSubNode("ReferenceSection");
  addSubNode("Types");

  for (XML::IdMap<QualType>::iterator i = Types.begin(), e = Types.end(); i != e; ++i)
  {
    if (i->first.getCVRQualifiers() != 0)
    {
      addSubNode("CvQualifiedType");
      addAttribute("id", getPrefixedId(i->second, ID_NORMAL));
      addAttribute("type", getPrefixedId(BasicTypes[i->first.getTypePtr()], ID_NORMAL));
      if (i->first.isConstQualified()) addAttribute("const", "1");
      if (i->first.isVolatileQualified()) addAttribute("volatile", "1");
      if (i->first.isRestrictQualified()) addAttribute("restrict", "1");
      toParent();
    }
  }

  for (XML::IdMap<const Type*>::iterator i = BasicTypes.begin(), e = BasicTypes.end(); i != e; ++i)
  {
    // don't use the get methods as they strip of typedef infos
    if (const BuiltinType *BT = dyn_cast<BuiltinType>(i->first)) {
      addSubNode("FundamentalType");
      addAttribute("name", BT->getName(Ctx->getLangOptions().CPlusPlus));
    }
    else if (const PointerType *PT = dyn_cast<PointerType>(i->first)) {
      addSubNode("PointerType");
      addTypeAttribute(PT->getPointeeType());
    }
    else if (dyn_cast<FunctionType>(i->first) != 0) {
      addSubNode("FunctionType");
    }
    else if (const ReferenceType *RT = dyn_cast<ReferenceType>(i->first)) {
      addSubNode("ReferenceType");
      addTypeAttribute(RT->getPointeeType());
    }
    else if (const TypedefType * TT = dyn_cast<TypedefType>(i->first)) {
      addSubNode("Typedef");
      addAttribute("name", TT->getDecl()->getNameAsString());
      addTypeAttribute(TT->getDecl()->getUnderlyingType());
      addContextAttribute(TT->getDecl()->getDeclContext());
    }
    else if (const QualifiedNameType *QT = dyn_cast<QualifiedNameType>(i->first)) {
      addSubNode("QualifiedNameType");
      addTypeAttribute(QT->getNamedType());
    }
    else if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(i->first)) {
      addSubNode("ArrayType");
      addAttribute("min", 0);
      addAttribute("max", (CAT->getSize() - 1).toString(10, false));
      addTypeAttribute(CAT->getElementType());
    }
    else if (const VariableArrayType *VAT = dyn_cast<VariableArrayType>(i->first)) {
      addSubNode("VariableArrayType");
      addTypeAttribute(VAT->getElementType());
    } 
    else if (const TagType *RET = dyn_cast<TagType>(i->first)) {
      const TagDecl *tagDecl = RET->getDecl();
      std::string tagKind = tagDecl->getKindName();
      tagKind[0] = std::toupper(tagKind[0]);
      addSubNode(tagKind);
      addAttribute("name", tagDecl->getNameAsString());
      addContextAttribute(tagDecl->getDeclContext());
    }
    else if (const VectorType* VT = dyn_cast<VectorType>(i->first)) {
      addSubNode("VectorType");
      addTypeAttribute(VT->getElementType());
      addAttribute("num_elements", VT->getNumElements());
    }
    else 
    {
      addSubNode("FIXMEType");
    }
    addAttribute("id", getPrefixedId(i->second, ID_NORMAL));
    toParent();
  }


  toParent().addSubNode("Contexts");

  for (XML::IdMap<const DeclContext*>::iterator i = Contexts.begin(), e = Contexts.end(); i != e; ++i)
  {
    addSubNode(i->first->getDeclKindName());
    addAttribute("id", getPrefixedId(i->second, ID_NORMAL));
    if (const NamedDecl *ND = dyn_cast<NamedDecl>(i->first)) {
      addAttribute("name", ND->getNameAsString());
    }
    if (const TagDecl *TD = dyn_cast<TagDecl>(i->first)) {
      addAttribute("type", getPrefixedId(BasicTypes[TD->getTypeForDecl()], ID_NORMAL));
    }
    else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(i->first)) {
      addAttribute("type", getPrefixedId(BasicTypes[FD->getType()->getAsFunctionType()], ID_NORMAL));
    }

    if (const DeclContext* parent = i->first->getParent())
    {
      addContextAttribute(parent);
    } 
    toParent();
  }

  toParent().addSubNode("Files");

  for (XML::IdMap<std::string>::iterator i = SourceFiles.begin(), e = SourceFiles.end(); i != e; ++i)
  {
    addSubNode("File");
    addAttribute("id", getPrefixedId(i->second, ID_FILE));
    addAttribute("name", escapeString(i->first.c_str(), i->first.size()));
    toParent();
  }

  toParent().toParent();
  
  // write the root closing node (which has always subnodes)
  Out << "</" << CurrentNode->Name << ">\n";
}

//--------------------------------------------------------- 
void DocumentXML::addTypeAttribute(const QualType& pType)
{
  addTypeRecursively(pType);
  addAttribute("type", getPrefixedId(Types[pType], ID_NORMAL));
}

//--------------------------------------------------------- 
void DocumentXML::addTypeIdAttribute(const Type* pType)
{
  addBasicTypeRecursively(pType);
  addAttribute("id", getPrefixedId(BasicTypes[pType], ID_NORMAL));
}

//--------------------------------------------------------- 
void DocumentXML::addTypeRecursively(const QualType& pType)
{
  if (addToMap(Types, pType))
  {
    addBasicTypeRecursively(pType.getTypePtr());
    // beautifier: a non-qualified type shall be transparent
    if (pType.getCVRQualifiers() == 0)
    {
      Types[pType] = BasicTypes[pType.getTypePtr()];   
    }
  }
}

//--------------------------------------------------------- 
void DocumentXML::addBasicTypeRecursively(const Type* pType)
{
  if (addToMap(BasicTypes, pType))
  {
    if (const PointerType *PT = dyn_cast<PointerType>(pType)) {
      addTypeRecursively(PT->getPointeeType());
    }
    else if (const ReferenceType *RT = dyn_cast<ReferenceType>(pType)) {
      addTypeRecursively(RT->getPointeeType());
    }
    else if (const TypedefType *TT = dyn_cast<TypedefType>(pType)) {
      addTypeRecursively(TT->getDecl()->getUnderlyingType());
      addContextsRecursively(TT->getDecl()->getDeclContext());
    }
    else if (const QualifiedNameType *QT = dyn_cast<QualifiedNameType>(pType)) {
      addTypeRecursively(QT->getNamedType());
      // FIXME: what to do with NestedNameSpecifier or shall this type be transparent?
    }
    else if (const ArrayType *AT = dyn_cast<ArrayType>(pType)) {
      addTypeRecursively(AT->getElementType());
      // FIXME: doesn't work in the immediate streaming approach
      /*if (const VariableArrayType *VAT = dyn_cast<VariableArrayType>(AT)) 
      {
        addSubNode("VariableArraySizeExpression");
        PrintStmt(VAT->getSizeExpr());
        toParent();
      }*/
    }
  }
}

//--------------------------------------------------------- 
void DocumentXML::addContextAttribute(const DeclContext *DC, tContextUsage usage)
{
  addContextsRecursively(DC);
  const char* pAttributeTags[2] = { "context", "id" };
  addAttribute(pAttributeTags[usage], getPrefixedId(Contexts[DC], ID_NORMAL));
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
void DocumentXML::PrintFunctionDecl(FunctionDecl *FD) 
{
  switch (FD->getStorageClass()) {
  default: assert(0 && "Unknown storage class");
  case FunctionDecl::None: break;
  case FunctionDecl::Extern: addAttribute("storage_class", "extern"); break;
  case FunctionDecl::Static: addAttribute("storage_class", "static"); break;
  case FunctionDecl::PrivateExtern: addAttribute("storage_class", "__private_extern__"); break;
  }

  if (FD->isInline())
    addAttribute("inline", "1");
  
  const FunctionType *AFT = FD->getType()->getAsFunctionType();
  addTypeAttribute(AFT->getResultType());
  addBasicTypeRecursively(AFT);  

  if (const FunctionProtoType *FT = dyn_cast<FunctionProtoType>(AFT)) {
    addAttribute("num_args", FD->getNumParams());
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
      addSubNode("Argument");
      ParmVarDecl *argDecl = FD->getParamDecl(i);
      addAttribute("name", argDecl->getNameAsString());
      addTypeAttribute(FT->getArgType(i));
      addDeclIdAttribute(argDecl);
      if (argDecl->getDefaultArg())
      {
        addAttribute("default_arg", "1");
        PrintStmt(argDecl->getDefaultArg());
      }
      toParent();
    }
    
    if (FT->isVariadic()) {
      addSubNode("Ellipsis").toParent();
    }
  } else {
    assert(isa<FunctionNoProtoType>(AFT));
  }
}

//--------------------------------------------------------- 
void DocumentXML::addRefAttribute(const NamedDecl* D)
{
  // FIXME: in case of CXX inline member functions referring to a member defined 
  // after the function it needs to be tested, if the ids are already there
  // (should work, but I couldn't test it)
  if (const DeclContext* DC = dyn_cast<DeclContext>(D))
  {
    addAttribute("ref", getPrefixedId(Contexts[DC], ID_NORMAL));
  }
  else
  {
    addAttribute("ref", getPrefixedId(Decls[D], ID_NORMAL));
  }
}

//--------------------------------------------------------- 
void DocumentXML::addDeclIdAttribute(const NamedDecl* D)
{
  addToMap(Decls, D);
  addAttribute("id", getPrefixedId(Decls[D], ID_NORMAL));
}

//--------------------------------------------------------- 
void DocumentXML::PrintDecl(Decl *D)
{
  addSubNode(D->getDeclKindName());
  addContextAttribute(D->getDeclContext());
  addLocation(D->getLocation());
  if (DeclContext* DC = dyn_cast<DeclContext>(D))
  {
    addContextAttribute(DC, CONTEXT_AS_ID);
  }

  if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    addAttribute("name", ND->getNameAsString());

    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      PrintFunctionDecl(FD);
      if (Stmt *Body = FD->getBody(*Ctx)) {
        addSubNode("Body");
        PrintStmt(Body);
        toParent();
      }
    } else if (RecordDecl *RD = dyn_cast<RecordDecl>(D)) {
      addBasicTypeRecursively(RD->getTypeForDecl());
      addAttribute("type", getPrefixedId(BasicTypes[RD->getTypeForDecl()], ID_NORMAL));
      if (!RD->isDefinition())
      {
        addAttribute("forward", "1");
      }

      for (RecordDecl::field_iterator i = RD->field_begin(*Ctx), e = RD->field_end(*Ctx); i != e; ++i)
      {
        PrintDecl(*i);
      }
    } else if (EnumDecl *ED = dyn_cast<EnumDecl>(D)) {
      const QualType& enumType = ED->getIntegerType();
      if (!enumType.isNull())
      {
        addTypeAttribute(enumType);
        for (EnumDecl::enumerator_iterator i = ED->enumerator_begin(*Ctx), e = ED->enumerator_end(*Ctx); i != e; ++i)
        {
          PrintDecl(*i);
        }
      }
    } else if (EnumConstantDecl* ECD = dyn_cast<EnumConstantDecl>(D)) {
      addTypeAttribute(ECD->getType());
      addAttribute("value", ECD->getInitVal().toString(10, true));
      if (ECD->getInitExpr()) 
      {
        PrintStmt(ECD->getInitExpr());
      }
    } else if (FieldDecl *FdD = dyn_cast<FieldDecl>(D)) {
      addTypeAttribute(FdD->getType());
      addDeclIdAttribute(ND);
      if (FdD->isMutable())
        addAttribute("mutable", "1");
      if (FdD->isBitField())
      {
        addAttribute("bitfield", "1");
        PrintStmt(FdD->getBitWidth());
      }
    } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
      addTypeIdAttribute(Ctx->getTypedefType(TD).getTypePtr());
      addTypeAttribute(TD->getUnderlyingType());
    } else if (ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
      addTypeAttribute(VD->getType());
      addDeclIdAttribute(ND);

      VarDecl *V = dyn_cast<VarDecl>(VD);
      if (V && V->getStorageClass() != VarDecl::None)
      {
        addAttribute("storage_class", VarDecl::getStorageClassSpecifierString(V->getStorageClass()));
      }
      
      if (V && V->getInit()) 
      {
        PrintStmt(V->getInit());
      }
    }
  } else if (LinkageSpecDecl* LSD = dyn_cast<LinkageSpecDecl>(D)) {
    switch (LSD->getLanguage())
    {
      case LinkageSpecDecl::lang_c:    addAttribute("lang", "C");  break;
      case LinkageSpecDecl::lang_cxx:  addAttribute("lang", "CXX");  break;
      default:                         assert(0 && "Unexpected lang id");
    }
  } else {
    assert(0 && "Unexpected decl");
  }
  toParent();
}

//--------------------------------------------------------- 
} // NS clang

