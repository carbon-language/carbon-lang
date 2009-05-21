//===--- DocumentXML.h - XML document for ASTs ------------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_FRONTEND_DOCUMENTXML_H
#define LLVM_CLANG_FRONTEND_DOCUMENTXML_H

#include <string>
#include <map>
#include "clang/AST/Type.h"
#include "clang/AST/TypeOrdering.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

//--------------------------------------------------------- forwards
class DeclContext;
class Decl;
class NamedDecl;
class FunctionDecl;
class ASTContext;
 
//--------------------------------------------------------- 
namespace XML
{
  // id maps:
  template<class T>
  struct IdMap : llvm::DenseMap<T, unsigned> {};

  template<>
  struct IdMap<QualType> : std::map<QualType, unsigned, QualTypeOrdering> {};

  template<>
  struct IdMap<std::string> : std::map<std::string, unsigned> {};
}

//--------------------------------------------------------- 
class DocumentXML
{
public:
  DocumentXML(const std::string& rootName, llvm::raw_ostream& out);
  ~DocumentXML();

  void initialize(ASTContext &Context);
  void PrintDecl(Decl *D);
  void PrintStmt(const Stmt *S);    // defined in StmtXML.cpp

  void finalize();


  DocumentXML& addSubNode(const std::string& name);   // also enters the sub node, returns *this
  DocumentXML& toParent();                            // returns *this

  template<class T>
  void addAttribute(const char* pName, const T& value);
  
  void addTypeAttribute(const QualType& pType);  
  void addRefAttribute(const NamedDecl* D);

  enum tContextUsage { CONTEXT_AS_CONTEXT, CONTEXT_AS_ID };
  void addContextAttribute(const DeclContext *DC, tContextUsage usage = CONTEXT_AS_CONTEXT);
  void addSourceFileAttribute(const std::string& fileName);

  PresumedLoc addLocation(const SourceLocation& Loc);
  void addLocationRange(const SourceRange& R);

  static std::string escapeString(const char* pStr, std::string::size_type len);

private:
  DocumentXML(const DocumentXML&);              // not defined
  DocumentXML& operator=(const DocumentXML&);   // not defined

  struct NodeXML;

  NodeXML*  Root;
  NodeXML*  CurrentNode;   // always after Root
  llvm::raw_ostream& Out;
  ASTContext *Ctx;
  int       CurrentIndent;
  bool      HasCurrentNodeSubNodes;


  XML::IdMap<QualType>                 Types;
  XML::IdMap<const DeclContext*>       Contexts;
  XML::IdMap<const Type*>              BasicTypes;
  XML::IdMap<std::string>              SourceFiles;
  XML::IdMap<const NamedDecl*>         Decls;

  void addContextsRecursively(const DeclContext *DC);
  void addBasicTypeRecursively(const Type* pType);
  void addTypeRecursively(const QualType& pType);

  void PrintFunctionDecl(FunctionDecl *FD);
  void addDeclIdAttribute(const NamedDecl* D);
  void addTypeIdAttribute(const Type* pType);  
  void Indent();
};

//--------------------------------------------------------- inlines

inline void DocumentXML::initialize(ASTContext &Context) 
{ 
  Ctx = &Context; 
}

//--------------------------------------------------------- 
template<class T>
inline void DocumentXML::addAttribute(const char* pName, const T& value)
{
  Out << ' ' << pName << "=\"" << value << "\"";
}

//--------------------------------------------------------- 

} //namespace clang 

#endif //LLVM_CLANG_DOCUMENTXML_H
