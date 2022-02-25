//===--- TestTU.cpp - Scratch source files for testing --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestTU.h"
#include "CompileCommands.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "TestFS.h"
#include "index/FileIndex.h"
#include "index/MemIndex.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/Utils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

namespace clang {
namespace clangd {

ParseInputs TestTU::inputs(MockFS &FS) const {
  std::string FullFilename = testPath(Filename),
              FullHeaderName = testPath(HeaderFilename),
              ImportThunk = testPath("import_thunk.h");
  // We want to implicitly include HeaderFilename without messing up offsets.
  // -include achieves this, but sometimes we want #import (to simulate a header
  // guard without messing up offsets). In this case, use an intermediate file.
  std::string ThunkContents = "#import \"" + FullHeaderName + "\"\n";

  FS.Files = AdditionalFiles;
  FS.Files[FullFilename] = Code;
  FS.Files[FullHeaderName] = HeaderCode;
  FS.Files[ImportThunk] = ThunkContents;

  ParseInputs Inputs;
  Inputs.FeatureModules = FeatureModules;
  auto &Argv = Inputs.CompileCommand.CommandLine;
  Argv = {"clang"};
  // FIXME: this shouldn't need to be conditional, but it breaks a
  // GoToDefinition test for some reason (getMacroArgExpandedLocation fails).
  if (!HeaderCode.empty()) {
    Argv.push_back("-include");
    Argv.push_back(ImplicitHeaderGuard ? ImportThunk : FullHeaderName);
    // ms-compatibility changes the meaning of #import.
    // The default is OS-dependent (on on windows), ensure it's off.
    if (ImplicitHeaderGuard)
      Inputs.CompileCommand.CommandLine.push_back("-fno-ms-compatibility");
  }
  Argv.insert(Argv.end(), ExtraArgs.begin(), ExtraArgs.end());
  // Put the file name at the end -- this allows the extra arg (-xc++) to
  // override the language setting.
  Argv.push_back(FullFilename);

  auto Mangler = CommandMangler::forTests();
  Mangler.adjust(Inputs.CompileCommand.CommandLine, FullFilename);
  Inputs.CompileCommand.Filename = FullFilename;
  Inputs.CompileCommand.Directory = testRoot();
  Inputs.Contents = Code;
  if (OverlayRealFileSystemForModules)
    FS.OverlayRealFileSystemForModules = true;
  Inputs.TFS = &FS;
  Inputs.Opts = ParseOptions();
  if (ClangTidyProvider)
    Inputs.ClangTidyProvider = ClangTidyProvider;
  Inputs.Index = ExternalIndex;
  return Inputs;
}

void initializeModuleCache(CompilerInvocation &CI) {
  llvm::SmallString<128> ModuleCachePath;
  if (llvm::sys::fs::createUniqueDirectory("module-cache", ModuleCachePath)) {
    llvm::errs() << "Failed to create temp directory for module-cache";
    std::abort();
  }
  CI.getHeaderSearchOpts().ModuleCachePath = ModuleCachePath.c_str();
}

void deleteModuleCache(const std::string ModuleCachePath) {
  if (!ModuleCachePath.empty()) {
    if (llvm::sys::fs::remove_directories(ModuleCachePath)) {
      llvm::errs() << "Failed to delete temp directory for module-cache";
      std::abort();
    }
  }
}

std::shared_ptr<const PreambleData>
TestTU::preamble(PreambleParsedCallback PreambleCallback) const {
  MockFS FS;
  auto Inputs = inputs(FS);
  IgnoreDiagnostics Diags;
  auto CI = buildCompilerInvocation(Inputs, Diags);
  assert(CI && "Failed to build compilation invocation.");
  if (OverlayRealFileSystemForModules)
    initializeModuleCache(*CI);
  auto ModuleCacheDeleter = llvm::make_scope_exit(
      std::bind(deleteModuleCache, CI->getHeaderSearchOpts().ModuleCachePath));
  return clang::clangd::buildPreamble(testPath(Filename), *CI, Inputs,
                                      /*StoreInMemory=*/true, PreambleCallback);
}

ParsedAST TestTU::build() const {
  MockFS FS;
  auto Inputs = inputs(FS);
  StoreDiags Diags;
  auto CI = buildCompilerInvocation(Inputs, Diags);
  assert(CI && "Failed to build compilation invocation.");
  if (OverlayRealFileSystemForModules)
    initializeModuleCache(*CI);
  auto ModuleCacheDeleter = llvm::make_scope_exit(
      std::bind(deleteModuleCache, CI->getHeaderSearchOpts().ModuleCachePath));

  auto Preamble = clang::clangd::buildPreamble(testPath(Filename), *CI, Inputs,
                                               /*StoreInMemory=*/true,
                                               /*PreambleCallback=*/nullptr);
  auto AST = ParsedAST::build(testPath(Filename), Inputs, std::move(CI),
                              Diags.take(), Preamble);
  if (!AST.hasValue()) {
    llvm::errs() << "Failed to build code:\n" << Code;
    std::abort();
  }
  assert(AST->getDiagnostics() &&
         "TestTU should always build an AST with a fresh Preamble");
  // Check for error diagnostics and report gtest failures (unless expected).
  // This guards against accidental syntax errors silently subverting tests.
  // error-ok is awfully primitive - using clang -verify would be nicer.
  // Ownership and layering makes it pretty hard.
  bool ErrorOk = [&, this] {
    llvm::StringLiteral Marker = "error-ok";
    if (llvm::StringRef(Code).contains(Marker) ||
        llvm::StringRef(HeaderCode).contains(Marker))
      return true;
    for (const auto &KV : this->AdditionalFiles)
      if (llvm::StringRef(KV.second).contains(Marker))
        return true;
    return false;
  }();
  if (!ErrorOk) {
    // We always build AST with a fresh preamble in TestTU.
    for (const auto &D : *AST->getDiagnostics())
      if (D.Severity >= DiagnosticsEngine::Error) {
        llvm::errs()
            << "TestTU failed to build (suppress with /*error-ok*/): \n"
            << D << "\n\nFor code:\n"
            << Code;
        std::abort(); // Stop after first error for simplicity.
      }
  }
  return std::move(*AST);
}

SymbolSlab TestTU::headerSymbols() const {
  auto AST = build();
  return std::get<0>(indexHeaderSymbols(/*Version=*/"null", AST.getASTContext(),
                                        AST.getPreprocessor(),
                                        AST.getCanonicalIncludes()));
}

RefSlab TestTU::headerRefs() const {
  auto AST = build();
  return std::get<1>(indexMainDecls(AST));
}

std::unique_ptr<SymbolIndex> TestTU::index() const {
  auto AST = build();
  auto Idx = std::make_unique<FileIndex>();
  Idx->updatePreamble(testPath(Filename), /*Version=*/"null",
                      AST.getASTContext(), AST.getPreprocessor(),
                      AST.getCanonicalIncludes());
  Idx->updateMain(testPath(Filename), AST);
  return std::move(Idx);
}

const Symbol &findSymbol(const SymbolSlab &Slab, llvm::StringRef QName) {
  const Symbol *Result = nullptr;
  for (const Symbol &S : Slab) {
    if (QName != (S.Scope + S.Name).str())
      continue;
    if (Result) {
      llvm::errs() << "Multiple symbols named " << QName << ":\n"
                   << *Result << "\n---\n"
                   << S;
      assert(false && "QName is not unique");
    }
    Result = &S;
  }
  if (!Result) {
    llvm::errs() << "No symbol named " << QName << " in "
                 << llvm::to_string(Slab);
    assert(false && "No symbol with QName");
  }
  return *Result;
}

// RAII scoped class to disable TraversalScope for a ParsedAST.
class TraverseHeadersToo {
  ASTContext &Ctx;
  std::vector<Decl *> ScopeToRestore;

public:
  TraverseHeadersToo(ParsedAST &AST)
      : Ctx(AST.getASTContext()), ScopeToRestore(Ctx.getTraversalScope()) {
    Ctx.setTraversalScope({Ctx.getTranslationUnitDecl()});
  }
  ~TraverseHeadersToo() { Ctx.setTraversalScope(std::move(ScopeToRestore)); }
};

const NamedDecl &findDecl(ParsedAST &AST, llvm::StringRef QName) {
  auto &Ctx = AST.getASTContext();
  auto LookupDecl = [&Ctx](const DeclContext &Scope,
                           llvm::StringRef Name) -> const NamedDecl & {
    auto LookupRes = Scope.lookup(DeclarationName(&Ctx.Idents.get(Name)));
    assert(!LookupRes.empty() && "Lookup failed");
    assert(LookupRes.isSingleResult() && "Lookup returned multiple results");
    return *LookupRes.front();
  };

  const DeclContext *Scope = Ctx.getTranslationUnitDecl();

  StringRef Cur, Rest;
  for (std::tie(Cur, Rest) = QName.split("::"); !Rest.empty();
       std::tie(Cur, Rest) = Rest.split("::")) {
    Scope = &cast<DeclContext>(LookupDecl(*Scope, Cur));
  }
  return LookupDecl(*Scope, Cur);
}

const NamedDecl &findDecl(ParsedAST &AST,
                          std::function<bool(const NamedDecl &)> Filter) {
  TraverseHeadersToo Too(AST);
  struct Visitor : RecursiveASTVisitor<Visitor> {
    decltype(Filter) F;
    llvm::SmallVector<const NamedDecl *, 1> Decls;
    bool VisitNamedDecl(const NamedDecl *ND) {
      if (F(*ND))
        Decls.push_back(ND);
      return true;
    }
  } Visitor;
  Visitor.F = Filter;
  Visitor.TraverseDecl(AST.getASTContext().getTranslationUnitDecl());
  if (Visitor.Decls.size() != 1) {
    llvm::errs() << Visitor.Decls.size() << " symbols matched.";
    assert(Visitor.Decls.size() == 1);
  }
  return *Visitor.Decls.front();
}

const NamedDecl &findUnqualifiedDecl(ParsedAST &AST, llvm::StringRef Name) {
  return findDecl(AST, [Name](const NamedDecl &ND) {
    if (auto *ID = ND.getIdentifier())
      if (ID->getName() == Name)
        return true;
    return false;
  });
}

} // namespace clangd
} // namespace clang
