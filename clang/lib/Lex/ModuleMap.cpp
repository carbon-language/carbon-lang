//===--- ModuleMap.cpp - Describe the layout of modules ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ModuleMap implementation, which describes the layout
// of a module as it relates to headers.
//
//===----------------------------------------------------------------------===//
#include "clang/Lex/ModuleMap.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/PathV2.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
using namespace clang;

ModuleMap::ModuleMap(FileManager &FileMgr, const DiagnosticConsumer &DC) {
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagIDs(new DiagnosticIDs);
  Diags = llvm::IntrusiveRefCntPtr<DiagnosticsEngine>(
            new DiagnosticsEngine(DiagIDs));
  Diags->setClient(DC.clone(*Diags), /*ShouldOwnClient=*/true);
  SourceMgr = new SourceManager(*Diags, FileMgr);
}

ModuleMap::~ModuleMap() {
  for (llvm::StringMap<Module *>::iterator I = Modules.begin(), 
                                        IEnd = Modules.end();
       I != IEnd; ++I) {
    delete I->getValue();
  }
  
  delete SourceMgr;
}

Module *ModuleMap::findModuleForHeader(const FileEntry *File) {
  llvm::DenseMap<const FileEntry *, Module *>::iterator Known
    = Headers.find(File);
  if (Known != Headers.end())
    return Known->second;
  
  const DirectoryEntry *Dir = File->getDir();
  llvm::DenseMap<const DirectoryEntry *, Module *>::iterator KnownDir
    = UmbrellaDirs.find(Dir);
  if (KnownDir != UmbrellaDirs.end())
    return KnownDir->second;

  // Walk up the directory hierarchy looking for umbrella headers.
  llvm::SmallVector<const DirectoryEntry *, 2> SkippedDirs;
  StringRef DirName = Dir->getName();
  do {
    // Retrieve our parent path.
    DirName = llvm::sys::path::parent_path(DirName);
    if (DirName.empty())
      break;
    
    // Resolve the parent path to a directory entry.
    Dir = SourceMgr->getFileManager().getDirectory(DirName);
    if (!Dir)
      break;
    
    KnownDir = UmbrellaDirs.find(Dir);
    if (KnownDir != UmbrellaDirs.end()) {
      Module *Result = KnownDir->second;
      
      // Record each of the directories we stepped through as being part of
      // the module we found, since the umbrella header covers them all.
      for (unsigned I = 0, N = SkippedDirs.size(); I != N; ++I)
        UmbrellaDirs[SkippedDirs[I]] = Result;
      
      return Result;
    }
    
    SkippedDirs.push_back(Dir);
  } while (true);
  
  return 0;
}

Module *ModuleMap::findModule(StringRef Name) {
  llvm::StringMap<Module *>::iterator Known = Modules.find(Name);
  if (Known != Modules.end())
    return Known->getValue();
  
  return 0;
}

std::pair<Module *, bool> 
ModuleMap::findOrCreateModule(StringRef Name, Module *Parent, bool IsFramework,
                              bool IsExplicit) {
  // Try to find an existing module with this name.
  if (Module *Found = Parent? Parent->SubModules[Name] : Modules[Name])
    return std::make_pair(Found, false);
  
  // Create a new module with this name.
  Module *Result = new Module(Name, SourceLocation(), Parent, IsFramework, 
                              IsExplicit);
  if (Parent)
    Parent->SubModules[Name] = Result;
  else
    Modules[Name] = Result;
  return std::make_pair(Result, true);
}

Module *
ModuleMap::inferFrameworkModule(StringRef ModuleName, 
                                const DirectoryEntry *FrameworkDir) {
  // Check whether we've already found this module.
  if (Module *Module = findModule(ModuleName))
    return Module;
  
  // Look for an umbrella header.
  llvm::SmallString<128> UmbrellaName = StringRef(FrameworkDir->getName());
  llvm::sys::path::append(UmbrellaName, "Headers");
  llvm::sys::path::append(UmbrellaName, ModuleName + ".h");
  const FileEntry *UmbrellaHeader
    = SourceMgr->getFileManager().getFile(UmbrellaName);
  
  // FIXME: If there's no umbrella header, we could probably scan the
  // framework to load *everything*. But, it's not clear that this is a good
  // idea.
  if (!UmbrellaHeader)
    return 0;
  
  Module *Result = new Module(ModuleName, SourceLocation(), 
                              /*IsFramework=*/true);
  Result->UmbrellaHeader = UmbrellaHeader;
  Headers[UmbrellaHeader] = Result;
  UmbrellaDirs[FrameworkDir] = Result;
  Modules[ModuleName] = Result;
  return Result;
}

const FileEntry *
ModuleMap::getContainingModuleMapFile(Module *Module) {
  if (Module->DefinitionLoc.isInvalid() || !SourceMgr)
    return 0;

  return SourceMgr->getFileEntryForID(
           SourceMgr->getFileID(Module->DefinitionLoc));
}

void ModuleMap::dump() {
  llvm::errs() << "Modules:";
  for (llvm::StringMap<Module *>::iterator M = Modules.begin(), 
                                        MEnd = Modules.end(); 
       M != MEnd; ++M)
    M->getValue()->print(llvm::errs(), 2);
  
  llvm::errs() << "Headers:";
  for (llvm::DenseMap<const FileEntry *, Module *>::iterator 
            H = Headers.begin(),
         HEnd = Headers.end();
       H != HEnd; ++H) {
    llvm::errs() << "  \"" << H->first->getName() << "\" -> " 
                 << H->second->getFullModuleName() << "\n";
  }
}

//----------------------------------------------------------------------------//
// Module map file parser
//----------------------------------------------------------------------------//

namespace clang {
  /// \brief A token in a module map file.
  struct MMToken {
    enum TokenKind {
      EndOfFile,
      HeaderKeyword,
      Identifier,
      ExplicitKeyword,
      FrameworkKeyword,
      ModuleKeyword,
      UmbrellaKeyword,
      StringLiteral,
      LBrace,
      RBrace
    } Kind;
    
    unsigned Location;
    unsigned StringLength;
    const char *StringData;
    
    void clear() {
      Kind = EndOfFile;
      Location = 0;
      StringLength = 0;
      StringData = 0;
    }
    
    bool is(TokenKind K) const { return Kind == K; }
    
    SourceLocation getLocation() const {
      return SourceLocation::getFromRawEncoding(Location);
    }
    
    StringRef getString() const {
      return StringRef(StringData, StringLength);
    }
  };
  
  class ModuleMapParser {
    Lexer &L;
    SourceManager &SourceMgr;
    DiagnosticsEngine &Diags;
    ModuleMap &Map;
    
    /// \brief The directory that this module map resides in.
    const DirectoryEntry *Directory;
    
    /// \brief Whether an error occurred.
    bool HadError;
    
    /// \brief Default target information, used only for string literal
    /// parsing.
    TargetInfo *Target;
    
    /// \brief Stores string data for the various string literals referenced
    /// during parsing.
    llvm::BumpPtrAllocator StringData;
    
    /// \brief The current token.
    MMToken Tok;
    
    /// \brief The active module.
    Module *ActiveModule;
    
    /// \brief Consume the current token and return its location.
    SourceLocation consumeToken();
    
    /// \brief Skip tokens until we reach the a token with the given kind
    /// (or the end of the file).
    void skipUntil(MMToken::TokenKind K);
    
    void parseModuleDecl();
    void parseUmbrellaDecl();
    void parseHeaderDecl();
    
  public:
    typedef Module Module;
    
    explicit ModuleMapParser(Lexer &L, SourceManager &SourceMgr, 
                             DiagnosticsEngine &Diags,
                             ModuleMap &Map,
                             const DirectoryEntry *Directory)
      : L(L), SourceMgr(SourceMgr), Diags(Diags), Map(Map), 
        Directory(Directory), HadError(false), ActiveModule(0)
    {
      TargetOptions TargetOpts;
      TargetOpts.Triple = llvm::sys::getDefaultTargetTriple();
      Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
      
      Tok.clear();
      consumeToken();
    }
    
    bool parseModuleMapFile();
  };
}

SourceLocation ModuleMapParser::consumeToken() {
retry:
  SourceLocation Result = Tok.getLocation();
  Tok.clear();
  
  Token LToken;
  L.LexFromRawLexer(LToken);
  Tok.Location = LToken.getLocation().getRawEncoding();
  switch (LToken.getKind()) {
  case tok::raw_identifier:
    Tok.StringData = LToken.getRawIdentifierData();
    Tok.StringLength = LToken.getLength();
    Tok.Kind = llvm::StringSwitch<MMToken::TokenKind>(Tok.getString())
                 .Case("header", MMToken::HeaderKeyword)
                 .Case("explicit", MMToken::ExplicitKeyword)
                 .Case("framework", MMToken::FrameworkKeyword)
                 .Case("module", MMToken::ModuleKeyword)
                 .Case("umbrella", MMToken::UmbrellaKeyword)
                 .Default(MMToken::Identifier);
    break;
      
  case tok::eof:
    Tok.Kind = MMToken::EndOfFile;
    break;
      
  case tok::l_brace:
    Tok.Kind = MMToken::LBrace;
    break;

  case tok::r_brace:
    Tok.Kind = MMToken::RBrace;
    break;
      
  case tok::string_literal: {
    // Parse the string literal.
    LangOptions LangOpts;
    StringLiteralParser StringLiteral(&LToken, 1, SourceMgr, LangOpts, *Target);
    if (StringLiteral.hadError)
      goto retry;
    
    // Copy the string literal into our string data allocator.
    unsigned Length = StringLiteral.GetStringLength();
    char *Saved = StringData.Allocate<char>(Length + 1);
    memcpy(Saved, StringLiteral.GetString().data(), Length);
    Saved[Length] = 0;
    
    // Form the token.
    Tok.Kind = MMToken::StringLiteral;
    Tok.StringData = Saved;
    Tok.StringLength = Length;
    break;
  }
      
  case tok::comment:
    goto retry;
      
  default:
    Diags.Report(LToken.getLocation(), diag::err_mmap_unknown_token);
    HadError = true;
    goto retry;
  }
  
  return Result;
}

void ModuleMapParser::skipUntil(MMToken::TokenKind K) {
  unsigned braceDepth = 0;
  do {
    switch (Tok.Kind) {
    case MMToken::EndOfFile:
      return;

    case MMToken::LBrace:
      if (Tok.is(K) && braceDepth == 0)
        return;
        
      ++braceDepth;
      break;
    
    case MMToken::RBrace:
      if (braceDepth > 0)
        --braceDepth;
      else if (Tok.is(K))
        return;
      break;
        
    default:
      if (braceDepth == 0 && Tok.is(K))
        return;
      break;
    }
    
   consumeToken();
  } while (true);
}

/// \brief Parse a module declaration.
///
///   module-declaration:
///     'framework'[opt] 'module' identifier { module-member* }
///
///   module-member:
///     umbrella-declaration
///     header-declaration
///     'explicit'[opt] module-declaration
void ModuleMapParser::parseModuleDecl() {
  assert(Tok.is(MMToken::ExplicitKeyword) || Tok.is(MMToken::ModuleKeyword) ||
         Tok.is(MMToken::FrameworkKeyword));

  // Parse 'framework' or 'explicit' keyword, if present.
  bool Framework = false;
  bool Explicit = false;

  if (Tok.is(MMToken::FrameworkKeyword)) {
    consumeToken();
    Framework = true;
  } 
  // Parse 'explicit' keyword, if present.
  else if (Tok.is(MMToken::ExplicitKeyword)) {
    consumeToken();
    Explicit = true;
  }
  
  // Parse 'module' keyword.
  if (!Tok.is(MMToken::ModuleKeyword)) {
    Diags.Report(Tok.getLocation(), 
                 diag::err_mmap_expected_module_after_explicit);
    consumeToken();
    HadError = true;
    return;
  }
  consumeToken(); // 'module' keyword
  
  // Parse the module name.
  if (!Tok.is(MMToken::Identifier)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_module_name);
    HadError = true;
    return;    
  }
  StringRef ModuleName = Tok.getString();
  SourceLocation ModuleNameLoc = consumeToken();
  
  // Parse the opening brace.
  if (!Tok.is(MMToken::LBrace)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_lbrace)
      << ModuleName;
    HadError = true;
    return;
  }  
  SourceLocation LBraceLoc = consumeToken();
  
  // Determine whether this (sub)module has already been defined.
  llvm::StringMap<Module *> &ModuleSpace
    = ActiveModule? ActiveModule->SubModules : Map.Modules;
  llvm::StringMap<Module *>::iterator ExistingModule
    = ModuleSpace.find(ModuleName);
  if (ExistingModule != ModuleSpace.end()) {
    Diags.Report(ModuleNameLoc, diag::err_mmap_module_redefinition)
      << ModuleName;
    Diags.Report(ExistingModule->getValue()->DefinitionLoc,
                 diag::note_mmap_prev_definition);
    
    // Skip the module definition.
    skipUntil(MMToken::RBrace);
    if (Tok.is(MMToken::RBrace))
      consumeToken();
    
    HadError = true;
    return;
  }

  // Start defining this module.
  ActiveModule = new Module(ModuleName, ModuleNameLoc, ActiveModule, Framework,
                            Explicit);
  ModuleSpace[ModuleName] = ActiveModule;
  
  bool Done = false;
  do {
    switch (Tok.Kind) {
    case MMToken::EndOfFile:
    case MMToken::RBrace:
      Done = true;
      break;
        
    case MMToken::ExplicitKeyword:
    case MMToken::ModuleKeyword:
      parseModuleDecl();
      break;
        
    case MMToken::HeaderKeyword:
      parseHeaderDecl();
      break;
        
    case MMToken::UmbrellaKeyword:
      parseUmbrellaDecl();
      break;
        
    default:
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_member);
      consumeToken();
      break;        
    }
  } while (!Done);

  if (Tok.is(MMToken::RBrace))
    consumeToken();
  else {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_rbrace);
    Diags.Report(LBraceLoc, diag::note_mmap_lbrace_match);
    HadError = true;
  }

  // We're done parsing this module. Pop back to our parent scope.
  ActiveModule = ActiveModule->Parent;
}
 
/// \brief Parse an umbrella header declaration.
///
///   umbrella-declaration:
///     'umbrella' string-literal
void ModuleMapParser::parseUmbrellaDecl() {
  assert(Tok.is(MMToken::UmbrellaKeyword));
  SourceLocation UmbrellaLoc = consumeToken();
  
  // Parse the header name.
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_header) 
      << "umbrella";
    HadError = true;
    return;
  }
  StringRef FileName = Tok.getString();
  SourceLocation FileNameLoc = consumeToken();

  // Check whether we already have an umbrella header.
  if (ActiveModule->UmbrellaHeader) {
    Diags.Report(FileNameLoc, diag::err_mmap_umbrella_header_conflict)
      << ActiveModule->getFullModuleName() 
      << ActiveModule->UmbrellaHeader->getName();
    HadError = true;
    return;
  }
  
  // Only top-level modules can have umbrella headers.
  if (ActiveModule->Parent) {
    Diags.Report(UmbrellaLoc, diag::err_mmap_umbrella_header_submodule)
      << ActiveModule->getFullModuleName();
    HadError = true;
    return;
  }
  
  // Look for this file.
  llvm::SmallString<128> PathName;
  const FileEntry *File = 0;
  
  if (llvm::sys::path::is_absolute(FileName)) {
    PathName = FileName;
    File = SourceMgr.getFileManager().getFile(PathName);
  } else {
    // Search for the header file within the search directory.
    PathName += Directory->getName();
    unsigned PathLength = PathName.size();
    if (ActiveModule->isPartOfFramework()) {
      // Check whether this file is in the public headers.
      llvm::sys::path::append(PathName, "Headers");
      llvm::sys::path::append(PathName, FileName);
      File = SourceMgr.getFileManager().getFile(PathName);

      if (!File) {
        // Check whether this file is in the private headers.
        PathName.resize(PathLength);
        llvm::sys::path::append(PathName, "PrivateHeaders");
        llvm::sys::path::append(PathName, FileName);
        File = SourceMgr.getFileManager().getFile(PathName);
      }
      
      // FIXME: Deal with subframeworks.
    } else {
      // Lookup for normal headers.
      llvm::sys::path::append(PathName, FileName);
      File = SourceMgr.getFileManager().getFile(PathName);
    }
  }
  
  // FIXME: We shouldn't be eagerly stat'ing every file named in a module map.
  // Come up with a lazy way to do this.
  if (File) {
    if (const Module *OwningModule = Map.Headers[File]) {
      Diags.Report(FileNameLoc, diag::err_mmap_header_conflict)
        << FileName << OwningModule->getFullModuleName();
      HadError = true;
    } else if ((OwningModule = Map.UmbrellaDirs[Directory])) {
      Diags.Report(UmbrellaLoc, diag::err_mmap_umbrella_clash)
        << OwningModule->getFullModuleName();
      HadError = true;
    } else {
      // Record this umbrella header.
      ActiveModule->UmbrellaHeader = File;
      Map.Headers[File] = ActiveModule;
      Map.UmbrellaDirs[Directory] = ActiveModule;
    }
  } else {
    Diags.Report(FileNameLoc, diag::err_mmap_header_not_found)
      << true << FileName;
    HadError = true;    
  }
}

/// \brief Parse a header declaration.
///
///   header-declaration:
///     'header' string-literal
void ModuleMapParser::parseHeaderDecl() {
  assert(Tok.is(MMToken::HeaderKeyword));
  consumeToken();

  // Parse the header name.
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_header) 
      << "header";
    HadError = true;
    return;
  }
  StringRef FileName = Tok.getString();
  SourceLocation FileNameLoc = consumeToken();
  
  // Look for this file.
  llvm::SmallString<128> PathName;
  if (llvm::sys::path::is_relative(FileName)) {
    // FIXME: Change this search to also look for private headers!
    PathName += Directory->getName();
    
    if (ActiveModule->isPartOfFramework())
      llvm::sys::path::append(PathName, "Headers");
  }
  
  llvm::sys::path::append(PathName, FileName);
  
  // FIXME: We shouldn't be eagerly stat'ing every file named in a module map.
  // Come up with a lazy way to do this.
  if (const FileEntry *File = SourceMgr.getFileManager().getFile(PathName)) {
    if (const Module *OwningModule = Map.Headers[File]) {
      Diags.Report(FileNameLoc, diag::err_mmap_header_conflict)
        << FileName << OwningModule->getFullModuleName();
      HadError = true;
    } else {
      // Record this file.
      ActiveModule->Headers.push_back(File);      
      Map.Headers[File] = ActiveModule;
    }
  } else {
    Diags.Report(FileNameLoc, diag::err_mmap_header_not_found)
      << false << FileName;
    HadError = true;
  }
}

/// \brief Parse a module map file.
///
///   module-map-file:
///     module-declaration*
bool ModuleMapParser::parseModuleMapFile() {
  do {
    switch (Tok.Kind) {
    case MMToken::EndOfFile:
      return HadError;
      
    case MMToken::ModuleKeyword:
    case MMToken::FrameworkKeyword:
      parseModuleDecl();
      break;
      
    case MMToken::ExplicitKeyword:
    case MMToken::HeaderKeyword:
    case MMToken::Identifier:
    case MMToken::LBrace:
    case MMToken::RBrace:
    case MMToken::StringLiteral:
    case MMToken::UmbrellaKeyword:
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_module);
      HadError = true;
      consumeToken();
      break;
    }
  } while (true);
  
  return HadError;
}

bool ModuleMap::parseModuleMapFile(const FileEntry *File) {
  FileID ID = SourceMgr->createFileID(File, SourceLocation(), SrcMgr::C_User);
  const llvm::MemoryBuffer *Buffer = SourceMgr->getBuffer(ID);
  if (!Buffer)
    return true;
  
  // Parse this module map file.
  Lexer L(ID, SourceMgr->getBuffer(ID), *SourceMgr, LangOpts);
  Diags->getClient()->BeginSourceFile(LangOpts);
  ModuleMapParser Parser(L, *SourceMgr, *Diags, *this, File->getDir());
  bool Result = Parser.parseModuleMapFile();
  Diags->getClient()->EndSourceFile();
  
  return Result;
}
