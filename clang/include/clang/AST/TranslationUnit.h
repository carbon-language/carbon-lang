//===--- TranslationUnit.h - Abstraction for Translation Units  -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include <string>

namespace clang {
 
class FileManager;
class SourceManager;
class TargetInfo;
class IdentifierTable;
class SelectorTable;
class ASTContext;
class Decl;
class FileEntry;
  
class TranslationUnit {
  LangOptions LangOpts;
  ASTContext* Context;
  std::vector<Decl*> TopLevelDecls;
  bool OwnsMetaData;
  bool OwnsDecls;

  // The default ctor is only invoked during deserialization.
  explicit TranslationUnit() : Context(NULL), OwnsMetaData(true),
                               OwnsDecls(true) {}
  
public:
  explicit TranslationUnit(ASTContext& Ctx, const LangOptions& lopt)
    : LangOpts(lopt), Context(&Ctx), OwnsMetaData(false),
      OwnsDecls(true) {}

  void SetOwnsDecls(bool val) { OwnsDecls = val; }

  ~TranslationUnit();

  const LangOptions& getLangOpts() const { return LangOpts; }
  const std::string& getSourceFile() const;
  
  /// Emit - Emit the translation unit to an arbitray bitcode stream.
  void Emit(llvm::Serializer& S) const;
  
  /// Create - Reconsititute a translation unit from a bitcode stream.
  static TranslationUnit* Create(llvm::Deserializer& D, FileManager& FMgr);
  
  // Accessors
  const LangOptions& getLangOptions() const { return LangOpts; }

  ASTContext&        getContext() { return *Context; }
  const ASTContext&  getContext() const { return *Context; }
  
  /// AddTopLevelDecl - Add a top-level declaration to the translation unit.
  ///  Ownership of the Decl is transfered to the TranslationUnit object.
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
  
/// EmitASTBitcodeFile - Emit a translation unit to a bitcode file.
bool EmitASTBitcodeFile(const TranslationUnit& TU, 
                        const llvm::sys::Path& Filename);
  
bool EmitASTBitcodeFile(const TranslationUnit* TU, 
                        const llvm::sys::Path& Filename);
                     
/// ReadASTBitcodeFile - Reconsitute a translation unit from a bitcode file.
TranslationUnit* ReadASTBitcodeFile(const llvm::sys::Path& Filename,
                                    FileManager& FMgr); 
                

} // end namespace clang

#endif
