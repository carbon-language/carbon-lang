//===--- TranslationUnit.h - Abstraction for Translation Units  -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
// FIXME: This should eventually be moved out of the driver, or replaced
//        with its eventual successor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TRANSLATION_UNIT_H
#define LLVM_CLANG_TRANSLATION_UNIT_H

#include "clang/Basic/LangOptions.h"
#include "llvm/Bitcode/SerializationFwd.h"
#include "llvm/System/Path.h"
#include <vector>

namespace clang {
 
class FileManager;
class SourceManager;
class TargetInfo;
class IdentifierTable;
class SelectorTable;
class ASTContext;
class Decl;
  
class TranslationUnit {
  LangOptions LangOpts;
  ASTContext* Context;
  std::vector<Decl*> TopLevelDecls;

  explicit TranslationUnit() : Context(NULL) {}
  
public:
  explicit TranslationUnit(const LangOptions& lopt)
    : LangOpts(lopt), Context(NULL) {}
  
  explicit TranslationUnit(const LangOptions& lopt, ASTContext& context)
    : LangOpts(lopt), Context(&context) {}

  void setContext(ASTContext* context) { Context = context; }
  ASTContext* getContext() const { return Context; }
  const LangOptions& getLangOpts() const { return LangOpts; }
  
  /// EmitBitcodeFile - Emit the translation unit to a bitcode file.
  bool EmitBitcodeFile(const llvm::sys::Path& Filename) const;
  
  /// Emit - Emit the translation unit to an arbitray bitcode stream.
  void Emit(llvm::Serializer& S) const;
  
  /// Create - Reconsititute a translation unit from a bitcode stream.
  static TranslationUnit* Create(llvm::Deserializer& D, FileManager& FMgr);
  
  /// ReadBitcodeFile - Reconsitute a translation unit from a bitcode file.
  static TranslationUnit* ReadBitcodeFile(const llvm::sys::Path& Filename,
                                          FileManager& FMgr); 
  
  // Accessors
  const LangOptions& getLangOptions() const { return LangOpts; }
  ASTContext*        getASTContext() { return Context; }
  
  /// AddTopLevelDecl - Add a top-level declaration to the translation unit.
  void AddTopLevelDecl(Decl* d) {
    TopLevelDecls.push_back(d);
  }
  
  typedef std::vector<Decl*>::iterator iterator;  
  iterator begin() { return TopLevelDecls.begin(); }
  iterator end() { return TopLevelDecls.end(); }
  
  typedef std::vector<Decl*>::const_iterator const_iterator;  
  const_iterator begin() const { return TopLevelDecls.begin(); }
  const_iterator end() const { return TopLevelDecls.end(); }  
};
  
} // end namespace clang

#endif
