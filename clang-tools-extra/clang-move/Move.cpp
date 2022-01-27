//===-- Move.cpp - Implement ClangMove functationalities --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Move.h"
#include "HelperDeclRefGraph.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "clang-move"

using namespace clang::ast_matchers;

namespace clang {
namespace move {
namespace {

// FIXME: Move to ASTMatchers.
AST_MATCHER(VarDecl, isStaticDataMember) { return Node.isStaticDataMember(); }

AST_MATCHER(NamedDecl, notInMacro) { return !Node.getLocation().isMacroID(); }

AST_MATCHER_P(Decl, hasOutermostEnclosingClass,
              ast_matchers::internal::Matcher<Decl>, InnerMatcher) {
  const auto *Context = Node.getDeclContext();
  if (!Context)
    return false;
  while (const auto *NextContext = Context->getParent()) {
    if (isa<NamespaceDecl>(NextContext) ||
        isa<TranslationUnitDecl>(NextContext))
      break;
    Context = NextContext;
  }
  return InnerMatcher.matches(*Decl::castFromDeclContext(Context), Finder,
                              Builder);
}

AST_MATCHER_P(CXXMethodDecl, ofOutermostEnclosingClass,
              ast_matchers::internal::Matcher<CXXRecordDecl>, InnerMatcher) {
  const CXXRecordDecl *Parent = Node.getParent();
  if (!Parent)
    return false;
  while (const auto *NextParent =
             dyn_cast<CXXRecordDecl>(Parent->getParent())) {
    Parent = NextParent;
  }

  return InnerMatcher.matches(*Parent, Finder, Builder);
}

std::string CleanPath(StringRef PathRef) {
  llvm::SmallString<128> Path(PathRef);
  llvm::sys::path::remove_dots(Path, /*remove_dot_dot=*/true);
  // FIXME: figure out why this is necessary.
  llvm::sys::path::native(Path);
  return std::string(Path.str());
}

// Make the Path absolute using the CurrentDir if the Path is not an absolute
// path. An empty Path will result in an empty string.
std::string MakeAbsolutePath(StringRef CurrentDir, StringRef Path) {
  if (Path.empty())
    return "";
  llvm::SmallString<128> InitialDirectory(CurrentDir);
  llvm::SmallString<128> AbsolutePath(Path);
  llvm::sys::fs::make_absolute(InitialDirectory, AbsolutePath);
  return CleanPath(std::move(AbsolutePath));
}

// Make the Path absolute using the current working directory of the given
// SourceManager if the Path is not an absolute path.
//
// The Path can be a path relative to the build directory, or retrieved from
// the SourceManager.
std::string MakeAbsolutePath(const SourceManager &SM, StringRef Path) {
  llvm::SmallString<128> AbsolutePath(Path);
  if (std::error_code EC =
          SM.getFileManager().getVirtualFileSystem().makeAbsolute(AbsolutePath))
    llvm::errs() << "Warning: could not make absolute file: '" << EC.message()
                 << '\n';
  // Handle symbolic link path cases.
  // We are trying to get the real file path of the symlink.
  auto Dir = SM.getFileManager().getDirectory(
      llvm::sys::path::parent_path(AbsolutePath.str()));
  if (Dir) {
    StringRef DirName = SM.getFileManager().getCanonicalName(*Dir);
    // FIXME: getCanonicalName might fail to get real path on VFS.
    if (llvm::sys::path::is_absolute(DirName)) {
      SmallString<128> AbsoluteFilename;
      llvm::sys::path::append(AbsoluteFilename, DirName,
                              llvm::sys::path::filename(AbsolutePath.str()));
      return CleanPath(AbsoluteFilename);
    }
  }
  return CleanPath(AbsolutePath);
}

// Matches AST nodes that are expanded within the given AbsoluteFilePath.
AST_POLYMORPHIC_MATCHER_P(isExpansionInFile,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(Decl, Stmt, TypeLoc),
                          std::string, AbsoluteFilePath) {
  auto &SourceManager = Finder->getASTContext().getSourceManager();
  auto ExpansionLoc = SourceManager.getExpansionLoc(Node.getBeginLoc());
  if (ExpansionLoc.isInvalid())
    return false;
  auto *FileEntry =
      SourceManager.getFileEntryForID(SourceManager.getFileID(ExpansionLoc));
  if (!FileEntry)
    return false;
  return MakeAbsolutePath(SourceManager, FileEntry->getName()) ==
         AbsoluteFilePath;
}

class FindAllIncludes : public PPCallbacks {
public:
  explicit FindAllIncludes(SourceManager *SM, ClangMoveTool *const MoveTool)
      : SM(*SM), MoveTool(MoveTool) {}

  void InclusionDirective(SourceLocation HashLoc, const Token & /*IncludeTok*/,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          const FileEntry * /*File*/, StringRef SearchPath,
                          StringRef /*RelativePath*/,
                          const Module * /*Imported*/,
                          SrcMgr::CharacteristicKind /*FileType*/) override {
    if (const auto *FileEntry = SM.getFileEntryForID(SM.getFileID(HashLoc)))
      MoveTool->addIncludes(FileName, IsAngled, SearchPath,
                            FileEntry->getName(), FilenameRange, SM);
  }

private:
  const SourceManager &SM;
  ClangMoveTool *const MoveTool;
};

/// Add a declaration being moved to new.h/cc. Note that the declaration will
/// also be deleted in old.h/cc.
void MoveDeclFromOldFileToNewFile(ClangMoveTool *MoveTool, const NamedDecl *D) {
  MoveTool->getMovedDecls().push_back(D);
  MoveTool->addRemovedDecl(D);
  MoveTool->getUnremovedDeclsInOldHeader().erase(D);
}

class FunctionDeclarationMatch : public MatchFinder::MatchCallback {
public:
  explicit FunctionDeclarationMatch(ClangMoveTool *MoveTool)
      : MoveTool(MoveTool) {}

  void run(const MatchFinder::MatchResult &Result) override {
    const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("function");
    assert(FD);
    const NamedDecl *D = FD;
    if (const auto *FTD = FD->getDescribedFunctionTemplate())
      D = FTD;
    MoveDeclFromOldFileToNewFile(MoveTool, D);
  }

private:
  ClangMoveTool *MoveTool;
};

class VarDeclarationMatch : public MatchFinder::MatchCallback {
public:
  explicit VarDeclarationMatch(ClangMoveTool *MoveTool)
      : MoveTool(MoveTool) {}

  void run(const MatchFinder::MatchResult &Result) override {
    const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var");
    assert(VD);
    MoveDeclFromOldFileToNewFile(MoveTool, VD);
  }

private:
  ClangMoveTool *MoveTool;
};

class TypeAliasMatch : public MatchFinder::MatchCallback {
public:
  explicit TypeAliasMatch(ClangMoveTool *MoveTool)
      : MoveTool(MoveTool) {}

  void run(const MatchFinder::MatchResult &Result) override {
    if (const auto *TD = Result.Nodes.getNodeAs<TypedefDecl>("typedef"))
      MoveDeclFromOldFileToNewFile(MoveTool, TD);
    else if (const auto *TAD =
                 Result.Nodes.getNodeAs<TypeAliasDecl>("type_alias")) {
      const NamedDecl * D = TAD;
      if (const auto * TD = TAD->getDescribedAliasTemplate())
        D = TD;
      MoveDeclFromOldFileToNewFile(MoveTool, D);
    }
  }

private:
  ClangMoveTool *MoveTool;
};

class EnumDeclarationMatch : public MatchFinder::MatchCallback {
public:
  explicit EnumDeclarationMatch(ClangMoveTool *MoveTool)
      : MoveTool(MoveTool) {}

  void run(const MatchFinder::MatchResult &Result) override {
    const auto *ED = Result.Nodes.getNodeAs<EnumDecl>("enum");
    assert(ED);
    MoveDeclFromOldFileToNewFile(MoveTool, ED);
  }

private:
  ClangMoveTool *MoveTool;
};

class ClassDeclarationMatch : public MatchFinder::MatchCallback {
public:
  explicit ClassDeclarationMatch(ClangMoveTool *MoveTool)
      : MoveTool(MoveTool) {}
  void run(const MatchFinder::MatchResult &Result) override {
    SourceManager *SM = &Result.Context->getSourceManager();
    if (const auto *CMD = Result.Nodes.getNodeAs<CXXMethodDecl>("class_method"))
      MatchClassMethod(CMD, SM);
    else if (const auto *VD =
                 Result.Nodes.getNodeAs<VarDecl>("class_static_var_decl"))
      MatchClassStaticVariable(VD, SM);
    else if (const auto *CD =
                 Result.Nodes.getNodeAs<CXXRecordDecl>("moved_class"))
      MatchClassDeclaration(CD, SM);
  }

private:
  void MatchClassMethod(const CXXMethodDecl *CMD, SourceManager *SM) {
    // Skip inline class methods. isInline() ast matcher doesn't ignore this
    // case.
    if (!CMD->isInlined()) {
      MoveTool->getMovedDecls().push_back(CMD);
      MoveTool->addRemovedDecl(CMD);
      // Get template class method from its method declaration as
      // UnremovedDecls stores template class method.
      if (const auto *FTD = CMD->getDescribedFunctionTemplate())
        MoveTool->getUnremovedDeclsInOldHeader().erase(FTD);
      else
        MoveTool->getUnremovedDeclsInOldHeader().erase(CMD);
    }
  }

  void MatchClassStaticVariable(const NamedDecl *VD, SourceManager *SM) {
    MoveDeclFromOldFileToNewFile(MoveTool, VD);
  }

  void MatchClassDeclaration(const CXXRecordDecl *CD, SourceManager *SM) {
    // Get class template from its class declaration as UnremovedDecls stores
    // class template.
    if (const auto *TC = CD->getDescribedClassTemplate())
      MoveTool->getMovedDecls().push_back(TC);
    else
      MoveTool->getMovedDecls().push_back(CD);
    MoveTool->addRemovedDecl(MoveTool->getMovedDecls().back());
    MoveTool->getUnremovedDeclsInOldHeader().erase(
        MoveTool->getMovedDecls().back());
  }

  ClangMoveTool *MoveTool;
};

// Expand to get the end location of the line where the EndLoc of the given
// Decl.
SourceLocation getLocForEndOfDecl(const Decl *D,
                                  const LangOptions &LangOpts = LangOptions()) {
  const auto &SM = D->getASTContext().getSourceManager();
  // If the expansion range is a character range, this is the location of
  // the first character past the end. Otherwise it's the location of the
  // first character in the final token in the range.
  auto EndExpansionLoc = SM.getExpansionRange(D->getEndLoc()).getEnd();
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(EndExpansionLoc);
  // Try to load the file buffer.
  bool InvalidTemp = false;
  llvm::StringRef File = SM.getBufferData(LocInfo.first, &InvalidTemp);
  if (InvalidTemp)
    return SourceLocation();

  const char *TokBegin = File.data() + LocInfo.second;
  // Lex from the start of the given location.
  Lexer Lex(SM.getLocForStartOfFile(LocInfo.first), LangOpts, File.begin(),
            TokBegin, File.end());

  llvm::SmallVector<char, 16> Line;
  // FIXME: this is a bit hacky to get ReadToEndOfLine work.
  Lex.setParsingPreprocessorDirective(true);
  Lex.ReadToEndOfLine(&Line);
  SourceLocation EndLoc = EndExpansionLoc.getLocWithOffset(Line.size());
  // If we already reach EOF, just return the EOF SourceLocation;
  // otherwise, move 1 offset ahead to include the trailing newline character
  // '\n'.
  return SM.getLocForEndOfFile(LocInfo.first) == EndLoc
             ? EndLoc
             : EndLoc.getLocWithOffset(1);
}

// Get full range of a Decl including the comments associated with it.
CharSourceRange getFullRange(const Decl *D,
                             const LangOptions &options = LangOptions()) {
  const auto &SM = D->getASTContext().getSourceManager();
  SourceRange Full(SM.getExpansionLoc(D->getBeginLoc()), getLocForEndOfDecl(D));
  // Expand to comments that are associated with the Decl.
  if (const auto *Comment = D->getASTContext().getRawCommentForDeclNoCache(D)) {
    if (SM.isBeforeInTranslationUnit(Full.getEnd(), Comment->getEndLoc()))
      Full.setEnd(Comment->getEndLoc());
    // FIXME: Don't delete a preceding comment, if there are no other entities
    // it could refer to.
    if (SM.isBeforeInTranslationUnit(Comment->getBeginLoc(), Full.getBegin()))
      Full.setBegin(Comment->getBeginLoc());
  }

  return CharSourceRange::getCharRange(Full);
}

std::string getDeclarationSourceText(const Decl *D) {
  const auto &SM = D->getASTContext().getSourceManager();
  llvm::StringRef SourceText =
      Lexer::getSourceText(getFullRange(D), SM, LangOptions());
  return SourceText.str();
}

bool isInHeaderFile(const Decl *D, llvm::StringRef OriginalRunningDirectory,
                    llvm::StringRef OldHeader) {
  const auto &SM = D->getASTContext().getSourceManager();
  if (OldHeader.empty())
    return false;
  auto ExpansionLoc = SM.getExpansionLoc(D->getBeginLoc());
  if (ExpansionLoc.isInvalid())
    return false;

  if (const auto *FE = SM.getFileEntryForID(SM.getFileID(ExpansionLoc))) {
    return MakeAbsolutePath(SM, FE->getName()) ==
           MakeAbsolutePath(OriginalRunningDirectory, OldHeader);
  }

  return false;
}

std::vector<std::string> getNamespaces(const Decl *D) {
  std::vector<std::string> Namespaces;
  for (const auto *Context = D->getDeclContext(); Context;
       Context = Context->getParent()) {
    if (llvm::isa<TranslationUnitDecl>(Context) ||
        llvm::isa<LinkageSpecDecl>(Context))
      break;

    if (const auto *ND = llvm::dyn_cast<NamespaceDecl>(Context))
      Namespaces.push_back(ND->getName().str());
  }
  std::reverse(Namespaces.begin(), Namespaces.end());
  return Namespaces;
}

tooling::Replacements
createInsertedReplacements(const std::vector<std::string> &Includes,
                           const std::vector<const NamedDecl *> &Decls,
                           llvm::StringRef FileName, bool IsHeader = false,
                           StringRef OldHeaderInclude = "") {
  std::string NewCode;
  std::string GuardName(FileName);
  if (IsHeader) {
    for (size_t i = 0; i < GuardName.size(); ++i) {
      if (!isAlphanumeric(GuardName[i]))
        GuardName[i] = '_';
    }
    GuardName = StringRef(GuardName).upper();
    NewCode += "#ifndef " + GuardName + "\n";
    NewCode += "#define " + GuardName + "\n\n";
  }

  NewCode += OldHeaderInclude;
  // Add #Includes.
  for (const auto &Include : Includes)
    NewCode += Include;

  if (!Includes.empty())
    NewCode += "\n";

  // Add moved class definition and its related declarations. All declarations
  // in same namespace are grouped together.
  //
  // Record namespaces where the current position is in.
  std::vector<std::string> CurrentNamespaces;
  for (const auto *MovedDecl : Decls) {
    // The namespaces of the declaration being moved.
    std::vector<std::string> DeclNamespaces = getNamespaces(MovedDecl);
    auto CurrentIt = CurrentNamespaces.begin();
    auto DeclIt = DeclNamespaces.begin();
    // Skip the common prefix.
    while (CurrentIt != CurrentNamespaces.end() &&
           DeclIt != DeclNamespaces.end()) {
      if (*CurrentIt != *DeclIt)
        break;
      ++CurrentIt;
      ++DeclIt;
    }
    // Calculate the new namespaces after adding MovedDecl in CurrentNamespace,
    // which is used for next iteration of this loop.
    std::vector<std::string> NextNamespaces(CurrentNamespaces.begin(),
                                            CurrentIt);
    NextNamespaces.insert(NextNamespaces.end(), DeclIt, DeclNamespaces.end());


    // End with CurrentNamespace.
    bool HasEndCurrentNamespace = false;
    auto RemainingSize = CurrentNamespaces.end() - CurrentIt;
    for (auto It = CurrentNamespaces.rbegin(); RemainingSize > 0;
         --RemainingSize, ++It) {
      assert(It < CurrentNamespaces.rend());
      NewCode += "} // namespace " + *It + "\n";
      HasEndCurrentNamespace = true;
    }
    // Add trailing '\n' after the nested namespace definition.
    if (HasEndCurrentNamespace)
      NewCode += "\n";

    // If the moved declaration is not in CurrentNamespace, add extra namespace
    // definitions.
    bool IsInNewNamespace = false;
    while (DeclIt != DeclNamespaces.end()) {
      NewCode += "namespace " + *DeclIt + " {\n";
      IsInNewNamespace = true;
      ++DeclIt;
    }
    // If the moved declaration is in same namespace CurrentNamespace, add
    // a preceeding `\n' before the moved declaration.
    // FIXME: Don't add empty lines between using declarations.
    if (!IsInNewNamespace)
      NewCode += "\n";
    NewCode += getDeclarationSourceText(MovedDecl);
    CurrentNamespaces = std::move(NextNamespaces);
  }
  std::reverse(CurrentNamespaces.begin(), CurrentNamespaces.end());
  for (const auto &NS : CurrentNamespaces)
    NewCode += "} // namespace " + NS + "\n";

  if (IsHeader)
    NewCode += "\n#endif // " + GuardName + "\n";
  return tooling::Replacements(tooling::Replacement(FileName, 0, 0, NewCode));
}

// Return a set of all decls which are used/referenced by the given Decls.
// Specifically, given a class member declaration, this method will return all
// decls which are used by the whole class.
llvm::DenseSet<const Decl *>
getUsedDecls(const HelperDeclRefGraph *RG,
             const std::vector<const NamedDecl *> &Decls) {
  assert(RG);
  llvm::DenseSet<const CallGraphNode *> Nodes;
  for (const auto *D : Decls) {
    auto Result = RG->getReachableNodes(
        HelperDeclRGBuilder::getOutmostClassOrFunDecl(D));
    Nodes.insert(Result.begin(), Result.end());
  }
  llvm::DenseSet<const Decl *> Results;
  for (const auto *Node : Nodes)
    Results.insert(Node->getDecl());
  return Results;
}

} // namespace

std::unique_ptr<ASTConsumer>
ClangMoveAction::CreateASTConsumer(CompilerInstance &Compiler,
                                   StringRef /*InFile*/) {
  Compiler.getPreprocessor().addPPCallbacks(std::make_unique<FindAllIncludes>(
      &Compiler.getSourceManager(), &MoveTool));
  return MatchFinder.newASTConsumer();
}

ClangMoveTool::ClangMoveTool(ClangMoveContext *const Context,
                             DeclarationReporter *const Reporter)
    : Context(Context), Reporter(Reporter) {
  if (!Context->Spec.NewHeader.empty())
    CCIncludes.push_back("#include \"" + Context->Spec.NewHeader + "\"\n");
}

void ClangMoveTool::addRemovedDecl(const NamedDecl *Decl) {
  const auto &SM = Decl->getASTContext().getSourceManager();
  auto Loc = Decl->getLocation();
  StringRef FilePath = SM.getFilename(Loc);
  FilePathToFileID[FilePath] = SM.getFileID(Loc);
  RemovedDecls.push_back(Decl);
}

void ClangMoveTool::registerMatchers(ast_matchers::MatchFinder *Finder) {
  auto InOldHeader =
      isExpansionInFile(makeAbsolutePath(Context->Spec.OldHeader));
  auto InOldCC = isExpansionInFile(makeAbsolutePath(Context->Spec.OldCC));
  auto InOldFiles = anyOf(InOldHeader, InOldCC);
  auto classTemplateForwardDecls =
      classTemplateDecl(unless(has(cxxRecordDecl(isDefinition()))));
  auto ForwardClassDecls = namedDecl(
      anyOf(cxxRecordDecl(unless(anyOf(isImplicit(), isDefinition()))),
            classTemplateForwardDecls));
  auto TopLevelDecl =
      hasDeclContext(anyOf(namespaceDecl(), translationUnitDecl()));

  //============================================================================
  // Matchers for old header
  //============================================================================
  // Match all top-level named declarations (e.g. function, variable, enum) in
  // old header, exclude forward class declarations and namespace declarations.
  //
  // We consider declarations inside a class belongs to the class. So these
  // declarations will be ignored.
  auto AllDeclsInHeader = namedDecl(
      unless(ForwardClassDecls), unless(namespaceDecl()),
      unless(usingDirectiveDecl()), // using namespace decl.
      notInMacro(),
      InOldHeader,
      hasParent(decl(anyOf(namespaceDecl(), translationUnitDecl()))),
      hasDeclContext(decl(anyOf(namespaceDecl(), translationUnitDecl()))));
  Finder->addMatcher(AllDeclsInHeader.bind("decls_in_header"), this);

  // Don't register other matchers when dumping all declarations in header.
  if (Context->DumpDeclarations)
    return;

  // Match forward declarations in old header.
  Finder->addMatcher(namedDecl(ForwardClassDecls, InOldHeader).bind("fwd_decl"),
                     this);

  //============================================================================
  // Matchers for old cc
  //============================================================================
  auto IsOldCCTopLevelDecl = allOf(
      hasParent(decl(anyOf(namespaceDecl(), translationUnitDecl()))), InOldCC);
  // Matching using decls/type alias decls which are in named/anonymous/global
  // namespace, these decls are always copied to new.h/cc. Those in classes,
  // functions are covered in other matchers.
  Finder->addMatcher(namedDecl(anyOf(usingDecl(IsOldCCTopLevelDecl),
                                     usingDirectiveDecl(unless(isImplicit()),
                                                        IsOldCCTopLevelDecl),
                                     typeAliasDecl(IsOldCCTopLevelDecl)),
                               notInMacro())
                         .bind("using_decl"),
                     this);

  // Match static functions/variable definitions which are defined in named
  // namespaces.
  SmallVector<std::string, 4> QualNames;
  QualNames.reserve(Context->Spec.Names.size());
  for (StringRef SymbolName : Context->Spec.Names) {
    QualNames.push_back(("::" + SymbolName.trim().ltrim(':')).str());
  }

  if (QualNames.empty()) {
    llvm::errs() << "No symbols being moved.\n";
    return;
  }

  ast_matchers::internal::Matcher<NamedDecl> HasAnySymbolNames =
      hasAnyName(SmallVector<StringRef, 4>(QualNames.begin(), QualNames.end()));

  auto InMovedClass =
      hasOutermostEnclosingClass(cxxRecordDecl(HasAnySymbolNames));

  // Matchers for helper declarations in old.cc.
  auto InAnonymousNS = hasParent(namespaceDecl(isAnonymous()));
  auto NotInMovedClass= allOf(unless(InMovedClass), InOldCC);
  auto IsOldCCHelper =
      allOf(NotInMovedClass, anyOf(isStaticStorageClass(), InAnonymousNS));
  // Match helper classes separately with helper functions/variables since we
  // want to reuse these matchers in finding helpers usage below.
  //
  // There could be forward declarations usage for helpers, especially for
  // classes and functions. We need include these forward declarations.
  //
  // Forward declarations for variable helpers will be excluded as these
  // declarations (with "extern") are not supposed in cpp file.
   auto HelperFuncOrVar =
      namedDecl(notInMacro(), anyOf(functionDecl(IsOldCCHelper),
                                    varDecl(isDefinition(), IsOldCCHelper)));
  auto HelperClasses =
      cxxRecordDecl(notInMacro(), NotInMovedClass, InAnonymousNS);
  // Save all helper declarations in old.cc.
  Finder->addMatcher(
      namedDecl(anyOf(HelperFuncOrVar, HelperClasses)).bind("helper_decls"),
      this);

  // Construct an AST-based call graph of helper declarations in old.cc.
  // In the following matcheres, "dc" is a caller while "helper_decls" and
  // "used_class" is a callee, so a new edge starting from caller to callee will
  // be add in the graph.
  //
  // Find helper function/variable usages.
  Finder->addMatcher(
      declRefExpr(to(HelperFuncOrVar), hasAncestor(decl().bind("dc")))
          .bind("func_ref"),
      &RGBuilder);
  // Find helper class usages.
  Finder->addMatcher(
      typeLoc(loc(recordType(hasDeclaration(HelperClasses.bind("used_class")))),
              hasAncestor(decl().bind("dc"))),
      &RGBuilder);

  //============================================================================
  // Matchers for old files, including old.h/old.cc
  //============================================================================
  // Create a MatchCallback for class declarations.
  MatchCallbacks.push_back(std::make_unique<ClassDeclarationMatch>(this));
  // Match moved class declarations.
  auto MovedClass =
      cxxRecordDecl(InOldFiles, HasAnySymbolNames, isDefinition(), TopLevelDecl)
          .bind("moved_class");
  Finder->addMatcher(MovedClass, MatchCallbacks.back().get());
  // Match moved class methods (static methods included) which are defined
  // outside moved class declaration.
  Finder->addMatcher(cxxMethodDecl(InOldFiles,
                                   ofOutermostEnclosingClass(HasAnySymbolNames),
                                   isDefinition())
                         .bind("class_method"),
                     MatchCallbacks.back().get());
  // Match static member variable definition of the moved class.
  Finder->addMatcher(
      varDecl(InMovedClass, InOldFiles, isDefinition(), isStaticDataMember())
          .bind("class_static_var_decl"),
      MatchCallbacks.back().get());

  MatchCallbacks.push_back(std::make_unique<FunctionDeclarationMatch>(this));
  Finder->addMatcher(functionDecl(InOldFiles, HasAnySymbolNames, TopLevelDecl)
                         .bind("function"),
                     MatchCallbacks.back().get());

  MatchCallbacks.push_back(std::make_unique<VarDeclarationMatch>(this));
  Finder->addMatcher(
      varDecl(InOldFiles, HasAnySymbolNames, TopLevelDecl).bind("var"),
      MatchCallbacks.back().get());

  // Match enum definition in old.h. Enum helpers (which are defined in old.cc)
  // will not be moved for now no matter whether they are used or not.
  MatchCallbacks.push_back(std::make_unique<EnumDeclarationMatch>(this));
  Finder->addMatcher(
      enumDecl(InOldHeader, HasAnySymbolNames, isDefinition(), TopLevelDecl)
          .bind("enum"),
      MatchCallbacks.back().get());

  // Match type alias in old.h, this includes "typedef" and "using" type alias
  // declarations. Type alias helpers (which are defined in old.cc) will not be
  // moved for now no matter whether they are used or not.
  MatchCallbacks.push_back(std::make_unique<TypeAliasMatch>(this));
  Finder->addMatcher(namedDecl(anyOf(typedefDecl().bind("typedef"),
                                     typeAliasDecl().bind("type_alias")),
                               InOldHeader, HasAnySymbolNames, TopLevelDecl),
                     MatchCallbacks.back().get());
}

void ClangMoveTool::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const auto *D = Result.Nodes.getNodeAs<NamedDecl>("decls_in_header")) {
    UnremovedDeclsInOldHeader.insert(D);
  } else if (const auto *FWD =
                 Result.Nodes.getNodeAs<CXXRecordDecl>("fwd_decl")) {
    // Skip all forward declarations which appear after moved class declaration.
    if (RemovedDecls.empty()) {
      if (const auto *DCT = FWD->getDescribedClassTemplate())
        MovedDecls.push_back(DCT);
      else
        MovedDecls.push_back(FWD);
    }
  } else if (const auto *ND =
                 Result.Nodes.getNodeAs<NamedDecl>("helper_decls")) {
    MovedDecls.push_back(ND);
    HelperDeclarations.push_back(ND);
    LLVM_DEBUG(llvm::dbgs()
               << "Add helper : " << ND->getDeclName() << " (" << ND << ")\n");
  } else if (const auto *UD = Result.Nodes.getNodeAs<NamedDecl>("using_decl")) {
    MovedDecls.push_back(UD);
  }
}

std::string ClangMoveTool::makeAbsolutePath(StringRef Path) {
  return MakeAbsolutePath(Context->OriginalRunningDirectory, Path);
}

void ClangMoveTool::addIncludes(llvm::StringRef IncludeHeader, bool IsAngled,
                                llvm::StringRef SearchPath,
                                llvm::StringRef FileName,
                                CharSourceRange IncludeFilenameRange,
                                const SourceManager &SM) {
  SmallString<128> HeaderWithSearchPath;
  llvm::sys::path::append(HeaderWithSearchPath, SearchPath, IncludeHeader);
  std::string AbsoluteIncludeHeader =
      MakeAbsolutePath(SM, HeaderWithSearchPath);
  std::string IncludeLine =
      IsAngled ? ("#include <" + IncludeHeader + ">\n").str()
               : ("#include \"" + IncludeHeader + "\"\n").str();

  std::string AbsoluteOldHeader = makeAbsolutePath(Context->Spec.OldHeader);
  std::string AbsoluteCurrentFile = MakeAbsolutePath(SM, FileName);
  if (AbsoluteOldHeader == AbsoluteCurrentFile) {
    // Find old.h includes "old.h".
    if (AbsoluteOldHeader == AbsoluteIncludeHeader) {
      OldHeaderIncludeRangeInHeader = IncludeFilenameRange;
      return;
    }
    HeaderIncludes.push_back(IncludeLine);
  } else if (makeAbsolutePath(Context->Spec.OldCC) == AbsoluteCurrentFile) {
    // Find old.cc includes "old.h".
    if (AbsoluteOldHeader == AbsoluteIncludeHeader) {
      OldHeaderIncludeRangeInCC = IncludeFilenameRange;
      return;
    }
    CCIncludes.push_back(IncludeLine);
  }
}

void ClangMoveTool::removeDeclsInOldFiles() {
  if (RemovedDecls.empty()) return;

  // If old_header is not specified (only move declarations from old.cc), remain
  // all the helper function declarations in old.cc as UnremovedDeclsInOldHeader
  // is empty in this case, there is no way to verify unused/used helpers.
  if (!Context->Spec.OldHeader.empty()) {
    std::vector<const NamedDecl *> UnremovedDecls;
    for (const auto *D : UnremovedDeclsInOldHeader)
      UnremovedDecls.push_back(D);

    auto UsedDecls = getUsedDecls(RGBuilder.getGraph(), UnremovedDecls);

    // We remove the helper declarations which are not used in the old.cc after
    // moving the given declarations.
    for (const auto *D : HelperDeclarations) {
      LLVM_DEBUG(llvm::dbgs() << "Check helper is used: " << D->getDeclName()
                              << " (" << D << ")\n");
      if (!UsedDecls.count(HelperDeclRGBuilder::getOutmostClassOrFunDecl(
              D->getCanonicalDecl()))) {
        LLVM_DEBUG(llvm::dbgs() << "Helper removed in old.cc: "
                                << D->getDeclName() << " (" << D << ")\n");
        RemovedDecls.push_back(D);
      }
    }
  }

  for (const auto *RemovedDecl : RemovedDecls) {
    const auto &SM = RemovedDecl->getASTContext().getSourceManager();
    auto Range = getFullRange(RemovedDecl);
    tooling::Replacement RemoveReplacement(
        SM, CharSourceRange::getCharRange(Range.getBegin(), Range.getEnd()),
        "");
    std::string FilePath = RemoveReplacement.getFilePath().str();
    auto Err = Context->FileToReplacements[FilePath].add(RemoveReplacement);
    if (Err)
      llvm::errs() << llvm::toString(std::move(Err)) << "\n";
  }
  const auto &SM = RemovedDecls[0]->getASTContext().getSourceManager();

  // Post process of cleanup around all the replacements.
  for (auto &FileAndReplacements : Context->FileToReplacements) {
    StringRef FilePath = FileAndReplacements.first;
    // Add #include of new header to old header.
    if (Context->Spec.OldDependOnNew &&
        MakeAbsolutePath(SM, FilePath) ==
            makeAbsolutePath(Context->Spec.OldHeader)) {
      // FIXME: Minimize the include path like clang-include-fixer.
      std::string IncludeNewH =
          "#include \"" + Context->Spec.NewHeader + "\"\n";
      // This replacement for inserting header will be cleaned up at the end.
      auto Err = FileAndReplacements.second.add(
          tooling::Replacement(FilePath, UINT_MAX, 0, IncludeNewH));
      if (Err)
        llvm::errs() << llvm::toString(std::move(Err)) << "\n";
    }

    auto SI = FilePathToFileID.find(FilePath);
    // Ignore replacements for new.h/cc.
    if (SI == FilePathToFileID.end()) continue;
    llvm::StringRef Code = SM.getBufferData(SI->second);
    auto Style = format::getStyle(format::DefaultFormatStyle, FilePath,
                                  Context->FallbackStyle);
    if (!Style) {
      llvm::errs() << llvm::toString(Style.takeError()) << "\n";
      continue;
    }
    auto CleanReplacements = format::cleanupAroundReplacements(
        Code, Context->FileToReplacements[std::string(FilePath)], *Style);

    if (!CleanReplacements) {
      llvm::errs() << llvm::toString(CleanReplacements.takeError()) << "\n";
      continue;
    }
    Context->FileToReplacements[std::string(FilePath)] = *CleanReplacements;
  }
}

void ClangMoveTool::moveDeclsToNewFiles() {
  std::vector<const NamedDecl *> NewHeaderDecls;
  std::vector<const NamedDecl *> NewCCDecls;
  for (const auto *MovedDecl : MovedDecls) {
    if (isInHeaderFile(MovedDecl, Context->OriginalRunningDirectory,
                       Context->Spec.OldHeader))
      NewHeaderDecls.push_back(MovedDecl);
    else
      NewCCDecls.push_back(MovedDecl);
  }

  auto UsedDecls = getUsedDecls(RGBuilder.getGraph(), RemovedDecls);
  std::vector<const NamedDecl *> ActualNewCCDecls;

  // Filter out all unused helpers in NewCCDecls.
  // We only move the used helpers (including transitively used helpers) and the
  // given symbols being moved.
  for (const auto *D : NewCCDecls) {
    if (llvm::is_contained(HelperDeclarations, D) &&
        !UsedDecls.count(HelperDeclRGBuilder::getOutmostClassOrFunDecl(
            D->getCanonicalDecl())))
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Helper used in new.cc: " << D->getDeclName()
                            << " " << D << "\n");
    ActualNewCCDecls.push_back(D);
  }

  if (!Context->Spec.NewHeader.empty()) {
    std::string OldHeaderInclude =
        Context->Spec.NewDependOnOld
            ? "#include \"" + Context->Spec.OldHeader + "\"\n"
            : "";
    Context->FileToReplacements[Context->Spec.NewHeader] =
        createInsertedReplacements(HeaderIncludes, NewHeaderDecls,
                                   Context->Spec.NewHeader, /*IsHeader=*/true,
                                   OldHeaderInclude);
  }
  if (!Context->Spec.NewCC.empty())
    Context->FileToReplacements[Context->Spec.NewCC] =
        createInsertedReplacements(CCIncludes, ActualNewCCDecls,
                                   Context->Spec.NewCC);
}

// Move all contents from OldFile to NewFile.
void ClangMoveTool::moveAll(SourceManager &SM, StringRef OldFile,
                            StringRef NewFile) {
  auto FE = SM.getFileManager().getFile(makeAbsolutePath(OldFile));
  if (!FE) {
    llvm::errs() << "Failed to get file: " << OldFile << "\n";
    return;
  }
  FileID ID = SM.getOrCreateFileID(*FE, SrcMgr::C_User);
  auto Begin = SM.getLocForStartOfFile(ID);
  auto End = SM.getLocForEndOfFile(ID);
  tooling::Replacement RemoveAll(SM, CharSourceRange::getCharRange(Begin, End),
                                 "");
  std::string FilePath = RemoveAll.getFilePath().str();
  Context->FileToReplacements[FilePath] = tooling::Replacements(RemoveAll);

  StringRef Code = SM.getBufferData(ID);
  if (!NewFile.empty()) {
    auto AllCode =
        tooling::Replacements(tooling::Replacement(NewFile, 0, 0, Code));
    auto ReplaceOldInclude = [&](CharSourceRange OldHeaderIncludeRange) {
      AllCode = AllCode.merge(tooling::Replacements(tooling::Replacement(
          SM, OldHeaderIncludeRange, '"' + Context->Spec.NewHeader + '"')));
    };
    // Fix the case where old.h/old.cc includes "old.h", we replace the
    // `#include "old.h"` with `#include "new.h"`.
    if (Context->Spec.NewCC == NewFile && OldHeaderIncludeRangeInCC.isValid())
      ReplaceOldInclude(OldHeaderIncludeRangeInCC);
    else if (Context->Spec.NewHeader == NewFile &&
             OldHeaderIncludeRangeInHeader.isValid())
      ReplaceOldInclude(OldHeaderIncludeRangeInHeader);
    Context->FileToReplacements[std::string(NewFile)] = std::move(AllCode);
  }
}

void ClangMoveTool::onEndOfTranslationUnit() {
  if (Context->DumpDeclarations) {
    assert(Reporter);
    for (const auto *Decl : UnremovedDeclsInOldHeader) {
      auto Kind = Decl->getKind();
      bool Templated = Decl->isTemplated();
      const std::string QualifiedName = Decl->getQualifiedNameAsString();
      if (Kind == Decl::Kind::Var)
        Reporter->reportDeclaration(QualifiedName, "Variable", Templated);
      else if (Kind == Decl::Kind::Function ||
               Kind == Decl::Kind::FunctionTemplate)
        Reporter->reportDeclaration(QualifiedName, "Function", Templated);
      else if (Kind == Decl::Kind::ClassTemplate ||
               Kind == Decl::Kind::CXXRecord)
        Reporter->reportDeclaration(QualifiedName, "Class", Templated);
      else if (Kind == Decl::Kind::Enum)
        Reporter->reportDeclaration(QualifiedName, "Enum", Templated);
      else if (Kind == Decl::Kind::Typedef || Kind == Decl::Kind::TypeAlias ||
               Kind == Decl::Kind::TypeAliasTemplate)
        Reporter->reportDeclaration(QualifiedName, "TypeAlias", Templated);
    }
    return;
  }

  if (RemovedDecls.empty())
    return;
  // Ignore symbols that are not supported when checking if there is unremoved
  // symbol in old header. This makes sure that we always move old files to new
  // files when all symbols produced from dump_decls are moved.
  auto IsSupportedKind = [](const NamedDecl *Decl) {
    switch (Decl->getKind()) {
    case Decl::Kind::Function:
    case Decl::Kind::FunctionTemplate:
    case Decl::Kind::ClassTemplate:
    case Decl::Kind::CXXRecord:
    case Decl::Kind::Enum:
    case Decl::Kind::Typedef:
    case Decl::Kind::TypeAlias:
    case Decl::Kind::TypeAliasTemplate:
    case Decl::Kind::Var:
      return true;
    default:
      return false;
    }
  };
  if (std::none_of(UnremovedDeclsInOldHeader.begin(),
                   UnremovedDeclsInOldHeader.end(), IsSupportedKind) &&
      !Context->Spec.OldHeader.empty()) {
    auto &SM = RemovedDecls[0]->getASTContext().getSourceManager();
    moveAll(SM, Context->Spec.OldHeader, Context->Spec.NewHeader);
    moveAll(SM, Context->Spec.OldCC, Context->Spec.NewCC);
    return;
  }
  LLVM_DEBUG(RGBuilder.getGraph()->dump());
  moveDeclsToNewFiles();
  removeDeclsInOldFiles();
}

} // namespace move
} // namespace clang
