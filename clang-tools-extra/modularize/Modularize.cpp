//===- tools/clang/Modularize.cpp - Check modularized headers -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a tool that checks whether a set of headers provides
// the consistent definitions required to use modules. For example, it detects
// whether the same entity (say, a NULL macro or size_t typedef) is defined in
// multiple headers or whether a header produces different definitions under
// different circumstances. These conditions cause modules built from the
// headers to behave poorly, and should be fixed before introducing a module 
// map.
//
// Modularize takes as argument a file name for a file containing the
// newline-separated list of headers to check with respect to each other.
// Modularize also accepts regular front-end arguments.
//
// Usage:   modularize (include-files_list) [(front-end-options) ...]
//
// Modularize will do normal parsing, reporting normal errors and warnings,
// but will also report special error messages like the following:
//
// error: '(symbol)' defined at both (file):(row):(column) and
//  (file):(row):(column)
//
// error: header '(file)' has different contents dependening on how it was
//   included
//
// The latter might be followed by messages like the following:
//
// note: '(symbol)' in (file) at (row):(column) not always provided
//
//===----------------------------------------------------------------------===//
 
#include "llvm/Config/config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/ADT/StringRef.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iterator>

using namespace clang::tooling;
using namespace clang;
using llvm::StringRef;

struct Location {
  const FileEntry *File;
  unsigned Line, Column;
  
  Location() : File(), Line(), Column() { }
  
  Location(SourceManager &SM, SourceLocation Loc) : File(), Line(), Column() {
    Loc = SM.getExpansionLoc(Loc);
    if (Loc.isInvalid())
      return;
    
    std::pair<FileID, unsigned> Decomposed = SM.getDecomposedLoc(Loc);
    File = SM.getFileEntryForID(Decomposed.first);
    if (!File)
      return;
    
    Line = SM.getLineNumber(Decomposed.first, Decomposed.second);
    Column = SM.getColumnNumber(Decomposed.first, Decomposed.second);
  }
  
  operator bool() const { return File != 0; }
  
  friend bool operator==(const Location &X, const Location &Y) {
    return X.File == Y.File && X.Line == Y.Line && X.Column == Y.Column;
  }

  friend bool operator!=(const Location &X, const Location &Y) {
    return !(X == Y);
  }
  
  friend bool operator<(const Location &X, const Location &Y) {
    if (X.File != Y.File)
      return X.File < Y.File;
    if (X.Line != Y.Line)
      return X.Line < Y.Line;
    return X.Column < Y.Column;
  }
  friend bool operator>(const Location &X, const Location &Y) {
    return Y < X;
  }
  friend bool operator<=(const Location &X, const Location &Y) {
    return !(Y < X);
  }
  friend bool operator>=(const Location &X, const Location &Y) {
    return !(X < Y);
  }

};


struct Entry {
  enum Kind {
    Tag,
    Value,
    Macro
  } Kind;
  
  Location Loc;
};

struct HeaderEntry {
  std::string Name;
  Location Loc;
  
  friend bool operator==(const HeaderEntry &X, const HeaderEntry &Y) {
    return X.Loc == Y.Loc && X.Name == Y.Name;
  }
  friend bool operator!=(const HeaderEntry &X, const HeaderEntry &Y) {
    return !(X == Y);
  }
  friend bool operator<(const HeaderEntry &X, const HeaderEntry &Y) {
    return X.Loc < Y.Loc || (X.Loc == Y.Loc && X.Name < Y.Name);
  }
  friend bool operator>(const HeaderEntry &X, const HeaderEntry &Y) {
    return Y < X;
  }
  friend bool operator<=(const HeaderEntry &X, const HeaderEntry &Y) {
    return !(Y < X);
  }
  friend bool operator>=(const HeaderEntry &X, const HeaderEntry &Y) {
    return !(X < Y);
  }
};

typedef std::vector<HeaderEntry> HeaderContents;

class EntityMap : public llvm::StringMap<llvm::SmallVector<Entry, 2> > {
  llvm::DenseMap<const FileEntry *, HeaderContents> CurHeaderContents;
  llvm::DenseMap<const FileEntry *, HeaderContents> AllHeaderContents;
  
public:
  llvm::DenseMap<const FileEntry *, HeaderContents> HeaderContentMismatches;
    
  void add(const std::string &Name, enum Entry::Kind Kind, Location Loc) {
    // Record this entity in its header.
    HeaderEntry HE = { Name, Loc };
    CurHeaderContents[Loc.File].push_back(HE);
    
    // Check whether we've seen this entry before.
    llvm::SmallVector<Entry, 2> &Entries = (*this)[Name];
    for (unsigned I = 0, N = Entries.size(); I != N; ++I) {
      if (Entries[I].Kind == Kind && Entries[I].Loc == Loc)
        return;
    }
    
    // We have not seen this entry before; record it.
    Entry E = { Kind, Loc };
    Entries.push_back(E);
  }
  
  void mergeCurHeaderContents() {
    for (llvm::DenseMap<const FileEntry *, HeaderContents>::iterator
           H = CurHeaderContents.begin(), HEnd = CurHeaderContents.end();
         H != HEnd; ++H) {
      // Sort contents.
      std::sort(H->second.begin(), H->second.end());

      // Check whether we've seen this header before.
      llvm::DenseMap<const FileEntry *, HeaderContents>::iterator KnownH
        = AllHeaderContents.find(H->first);
      if (KnownH == AllHeaderContents.end()) {
        // We haven't seen this header before; record its contents.
        AllHeaderContents.insert(*H);
        continue;
      }
      
      // If the header contents are the same, we're done.
      if (H->second == KnownH->second)
        continue;
      
      // Determine what changed.
      std::set_symmetric_difference(H->second.begin(), H->second.end(),
        KnownH->second.begin(),
        KnownH->second.end(),
        std::back_inserter(HeaderContentMismatches[H->first]));
    }
    
    CurHeaderContents.clear();
  }
};

class CollectEntitiesVisitor
  : public RecursiveASTVisitor<CollectEntitiesVisitor>
{
  SourceManager &SM;
  EntityMap &Entities;
  
public:
  CollectEntitiesVisitor(SourceManager &SM, EntityMap &Entities)
    : SM(SM), Entities(Entities) { }
  
  bool TraverseStmt(Stmt *S) { return true; }
  bool TraverseType(QualType T) { return true; }
  bool TraverseTypeLoc(TypeLoc TL) { return true; }
  bool TraverseNestedNameSpecifier(NestedNameSpecifier *NNS) { return true; }
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS) { return true; }
  bool TraverseDeclarationNameInfo(DeclarationNameInfo NameInfo) { return true; }
  bool TraverseTemplateName(TemplateName Template) { return true; }
  bool TraverseTemplateArgument(const TemplateArgument &Arg) { return true; }
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &ArgLoc) { return true; }
  bool TraverseTemplateArguments(const TemplateArgument *Args,
                                 unsigned NumArgs) { return true; }
  bool TraverseConstructorInitializer(CXXCtorInitializer *Init) { return true; }
  bool TraverseLambdaCapture(LambdaExpr::Capture C) { return true; }
  
  bool VisitNamedDecl(NamedDecl *ND) {
    // We only care about file-context variables.
    if (!ND->getDeclContext()->isFileContext())
      return true;
    
    // Skip declarations that tend to be properly multiply-declared.
    if (isa<NamespaceDecl>(ND) || isa<UsingDirectiveDecl>(ND) ||
        isa<NamespaceAliasDecl>(ND) || 
        isa<ClassTemplateSpecializationDecl>(ND) ||
        isa<UsingDecl>(ND) || isa<UsingShadowDecl>(ND) || 
        isa<FunctionDecl>(ND) || isa<FunctionTemplateDecl>(ND) ||
        (isa<TagDecl>(ND) &&
         !cast<TagDecl>(ND)->isThisDeclarationADefinition()))
      return true;
    
    std::string Name = ND->getNameAsString();
    if (Name.empty())
      return true;
    
    Location Loc(SM, ND->getLocation());
    if (!Loc)
      return true;
    
    Entities.add(Name, isa<TagDecl>(ND)? Entry::Tag : Entry::Value, Loc);
    return true;
  }
};

class CollectEntitiesConsumer : public ASTConsumer {
  EntityMap &Entities;
  Preprocessor &PP;
  
public:
  CollectEntitiesConsumer(EntityMap &Entities, Preprocessor &PP)
    : Entities(Entities), PP(PP) { }
  
  virtual void HandleTranslationUnit(ASTContext &Ctx) {
    SourceManager &SM = Ctx.getSourceManager();
    
    // Collect declared entities.
    CollectEntitiesVisitor(SM, Entities)
      .TraverseDecl(Ctx.getTranslationUnitDecl());
    
    // Collect macro definitions.
    for (Preprocessor::macro_iterator M = PP.macro_begin(),
                                   MEnd = PP.macro_end();
         M != MEnd; ++M) {
      Location Loc(SM, M->second->getLocation());
      if (!Loc)
        continue;

      Entities.add(M->first->getName().str(), Entry::Macro, Loc);
    }
    
    // Merge header contents.
    Entities.mergeCurHeaderContents();
  }
};

class CollectEntitiesAction : public SyntaxOnlyAction {
  EntityMap &Entities;
  
protected:
  virtual clang::ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                                StringRef InFile) {
    return new CollectEntitiesConsumer(Entities, CI.getPreprocessor());
  }
  
public:
  CollectEntitiesAction(EntityMap &Entities) : Entities(Entities) { }
};

class ModularizeFrontendActionFactory : public FrontendActionFactory {
  EntityMap &Entities;

public:
  ModularizeFrontendActionFactory(EntityMap &Entities) : Entities(Entities) { }

  virtual CollectEntitiesAction *create() {
    return new CollectEntitiesAction(Entities);
  }
};

int main(int argc, const char **argv) {
  // Figure out command-line arguments.
  if (argc < 2) {
    llvm::errs() << "Usage: modularize <file containing header names> <arguments>\n";
    return 1;
  }
  
  // Load the list of headers.
  std::string File = argv[1];
  llvm::SmallVector<std::string, 8> Headers;
  {
    std::ifstream In(File.c_str());
    if (!In) {
      llvm::errs() << "Unable to open header list file \"" << File.c_str() << "\"\n";
      return 2;
    }
    
    std::string Line;
    while (std::getline(In, Line)) {
      if (Line.empty() || Line[0] == '#')
        continue;
      
      Headers.push_back(Line);
    }
  }
  
  // Create the compilation database.
  llvm::OwningPtr<CompilationDatabase> Compilations;
  {
    std::vector<std::string> Arguments;
    for (int I = 2; I < argc; ++I)
      Arguments.push_back(argv[I]);
    SmallString<256> PathBuf;
    llvm::sys::fs::current_path(PathBuf);
    Compilations.reset(new FixedCompilationDatabase(Twine(PathBuf), Arguments));
  }
  
  // Parse all of the headers, detecting duplicates.
  EntityMap Entities;
  ClangTool Tool(*Compilations, Headers);
  int HadErrors = Tool.run(new ModularizeFrontendActionFactory(Entities));
  
  // Check for the same entity being defined in multiple places.
  for (EntityMap::iterator E = Entities.begin(), EEnd = Entities.end();
       E != EEnd; ++E) {
    Location Tag, Value, Macro;
    for (unsigned I = 0, N = E->second.size(); I != N; ++I) {
      Location *Which;
      switch (E->second[I].Kind) {
      case Entry::Tag: Which = &Tag; break;
      case Entry::Value: Which = &Value; break;
      case Entry::Macro: Which = &Macro; break;
      }
      
      if (!Which->File) {
        *Which = E->second[I].Loc;
        continue;
      }
      
      llvm::errs() << "error: '" << E->first().str().c_str()
        << "' defined at both " << Which->File->getName()
        << ":" << Which->Line << ":" << Which->Column
        << " and " << E->second[I].Loc.File->getName() << ":" 
        << E->second[I].Loc.Line << ":" << E->second[I].Loc.Column << "\n";
      HadErrors = 1;
    }
  }
  
  // Complain about any headers that have contents that differ based on how
  // they are included.
  for (llvm::DenseMap<const FileEntry *, HeaderContents>::iterator
            H = Entities.HeaderContentMismatches.begin(),
         HEnd = Entities.HeaderContentMismatches.end();
       H != HEnd; ++H) {
    if (H->second.empty()) {
      llvm::errs() << "internal error: phantom header content mismatch\n";
      continue;
    }
    
    HadErrors = 1;
    llvm::errs() << "error: header '" << H->first->getName()
      << "' has different contents dependening on how it was included\n";
    for (unsigned I = 0, N = H->second.size(); I != N; ++I) {
      llvm::errs() << "note: '" << H->second[I].Name.c_str()
        << "' in " << H->second[I].Loc.File->getName() << " at "
        << H->second[I].Loc.Line << ":" << H->second[I].Loc.Column
        << " not always provided\n";
    }
  }
  
  return HadErrors;
}
