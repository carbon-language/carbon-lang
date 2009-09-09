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
#include <stack>
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
class LabelStmt;

//---------------------------------------------------------
namespace XML {
  // id maps:
  template<class T>
  struct IdMap : llvm::DenseMap<T, unsigned> {};

  template<>
  struct IdMap<QualType> : std::map<QualType, unsigned, QualTypeOrdering> {};

  template<>
  struct IdMap<std::string> : std::map<std::string, unsigned> {};
}

//---------------------------------------------------------
class DocumentXML {
public:
  DocumentXML(const std::string& rootName, llvm::raw_ostream& out);

  void initialize(ASTContext &Context);
  void PrintDecl(Decl *D);
  void PrintStmt(const Stmt *S);    // defined in StmtXML.cpp
  void finalize();


  DocumentXML& addSubNode(const std::string& name);   // also enters the sub node, returns *this
  DocumentXML& toParent();                            // returns *this

  void addAttribute(const char* pName, const QualType& pType);
  void addAttribute(const char* pName, bool value);

  template<class T>
  void addAttribute(const char* pName, const T* value)   {
    addPtrAttribute(pName, value);
  }

  template<class T>
  void addAttribute(const char* pName, T* value) {
    addPtrAttribute(pName, value);
  }

  template<class T>
  void addAttribute(const char* pName, const T& value);

  template<class T>
  void addAttributeOptional(const char* pName, const T& value);

  void addSourceFileAttribute(const std::string& fileName);

  PresumedLoc addLocation(const SourceLocation& Loc);
  void addLocationRange(const SourceRange& R);

  static std::string escapeString(const char* pStr, std::string::size_type len);

private:
  DocumentXML(const DocumentXML&);              // not defined
  DocumentXML& operator=(const DocumentXML&);   // not defined

  std::stack<std::string> NodeStack;
  llvm::raw_ostream& Out;
  ASTContext *Ctx;
  bool      HasCurrentNodeSubNodes;


  XML::IdMap<QualType>                 Types;
  XML::IdMap<const DeclContext*>       Contexts;
  XML::IdMap<const Type*>              BasicTypes;
  XML::IdMap<std::string>              SourceFiles;
  XML::IdMap<const NamedDecl*>         Decls;
  XML::IdMap<const LabelStmt*>         Labels;

  void addContextsRecursively(const DeclContext *DC);
  void addTypeRecursively(const Type* pType);
  void addTypeRecursively(const QualType& pType);

  void Indent();

  // forced pointer dispatch:
  void addPtrAttribute(const char* pName, const Type* pType);
  void addPtrAttribute(const char* pName, const NamedDecl* D);
  void addPtrAttribute(const char* pName, const DeclContext* D);
  void addPtrAttribute(const char* pName, const NamespaceDecl* D);    // disambiguation
  void addPtrAttribute(const char* pName, const LabelStmt* L);
  void addPtrAttribute(const char* pName, const char* text);

  // defined in TypeXML.cpp:
  void addParentTypes(const Type* pType);
  void writeTypeToXML(const Type* pType);
  void writeTypeToXML(const QualType& pType);
  class TypeAdder;
  friend class TypeAdder;

  // defined in DeclXML.cpp:
  void writeDeclToXML(Decl *D);
  class DeclPrinter;
  friend class DeclPrinter;

  // for addAttributeOptional:
  static bool isDefault(unsigned value)           { return value == 0; }
  static bool isDefault(bool value)               { return !value; }
  static bool isDefault(const std::string& value) { return value.empty(); }
};

//--------------------------------------------------------- inlines

inline void DocumentXML::initialize(ASTContext &Context) {
  Ctx = &Context;
}

//---------------------------------------------------------
template<class T>
inline void DocumentXML::addAttribute(const char* pName, const T& value) {
  Out << ' ' << pName << "=\"" << value << "\"";
}

//---------------------------------------------------------
inline void DocumentXML::addPtrAttribute(const char* pName, const char* text) {
  Out << ' ' << pName << "=\"" << text << "\"";
}

//---------------------------------------------------------
inline void DocumentXML::addAttribute(const char* pName, bool value) {
  addPtrAttribute(pName, value ? "1" : "0");
}

//---------------------------------------------------------
template<class T>
inline void DocumentXML::addAttributeOptional(const char* pName,
                                              const T& value) {
  if (!isDefault(value)) {
    addAttribute(pName, value);
  }
}

//---------------------------------------------------------

} //namespace clang

#endif //LLVM_CLANG_DOCUMENTXML_H
