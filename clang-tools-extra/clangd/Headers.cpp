//===--- Headers.cpp - Include headers ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Headers.h"
#include "Compiler.h"
#include "Preamble.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/HeaderSearch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {
namespace {

class RecordHeaders : public PPCallbacks {
public:
  RecordHeaders(const SourceManager &SM, IncludeStructure *Out)
      : SM(SM), Out(Out) {}

  // Record existing #includes - both written and resolved paths. Only #includes
  // in the main file are collected.
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          llvm::StringRef FileName, bool IsAngled,
                          CharSourceRange /*FilenameRange*/,
                          const FileEntry *File, llvm::StringRef /*SearchPath*/,
                          llvm::StringRef /*RelativePath*/,
                          const clang::Module * /*Imported*/,
                          SrcMgr::CharacteristicKind FileKind) override {
    auto MainFID = SM.getMainFileID();
    // If an include is part of the preamble patch, translate #line directives.
    if (InBuiltinFile)
      HashLoc = translatePreamblePatchLocation(HashLoc, SM);

    // Record main-file inclusions (including those mapped from the preamble
    // patch).
    if (isInsideMainFile(HashLoc, SM)) {
      Out->MainFileIncludes.emplace_back();
      auto &Inc = Out->MainFileIncludes.back();
      Inc.Written =
          (IsAngled ? "<" + FileName + ">" : "\"" + FileName + "\"").str();
      Inc.Resolved = std::string(File ? File->tryGetRealPathName() : "");
      Inc.HashOffset = SM.getFileOffset(HashLoc);
      Inc.HashLine =
          SM.getLineNumber(SM.getFileID(HashLoc), Inc.HashOffset) - 1;
      Inc.FileKind = FileKind;
      Inc.Directive = IncludeTok.getIdentifierInfo()->getPPKeywordID();
    }

    // Record include graph (not just for main-file includes)
    if (File) {
      auto *IncludingFileEntry = SM.getFileEntryForID(SM.getFileID(HashLoc));
      if (!IncludingFileEntry) {
        assert(SM.getBufferName(HashLoc).startswith("<") &&
               "Expected #include location to be a file or <built-in>");
        // Treat as if included from the main file.
        IncludingFileEntry = SM.getFileEntryForID(MainFID);
      }
      Out->recordInclude(IncludingFileEntry->getName(), File->getName(),
                         File->tryGetRealPathName());
    }
  }

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override {
    switch (Reason) {
    case PPCallbacks::EnterFile:
      if (BuiltinFile.isInvalid() && SM.isWrittenInBuiltinFile(Loc)) {
        BuiltinFile = SM.getFileID(Loc);
        InBuiltinFile = true;
      }
      break;
    case PPCallbacks::ExitFile:
      if (PrevFID == BuiltinFile)
        InBuiltinFile = false;
      break;
    case PPCallbacks::RenameFile:
    case PPCallbacks::SystemHeaderPragma:
      break;
    }
  }

private:
  const SourceManager &SM;
  // Set after entering the <built-in> file.
  FileID BuiltinFile;
  // Indicates whether <built-in> file is part of include stack.
  bool InBuiltinFile = false;

  IncludeStructure *Out;
};

} // namespace

bool isLiteralInclude(llvm::StringRef Include) {
  return Include.startswith("<") || Include.startswith("\"");
}

bool HeaderFile::valid() const {
  return (Verbatim && isLiteralInclude(File)) ||
         (!Verbatim && llvm::sys::path::is_absolute(File));
}

llvm::Expected<HeaderFile> toHeaderFile(llvm::StringRef Header,
                                        llvm::StringRef HintPath) {
  if (isLiteralInclude(Header))
    return HeaderFile{Header.str(), /*Verbatim=*/true};
  auto U = URI::parse(Header);
  if (!U)
    return U.takeError();

  auto IncludePath = URI::includeSpelling(*U);
  if (!IncludePath)
    return IncludePath.takeError();
  if (!IncludePath->empty())
    return HeaderFile{std::move(*IncludePath), /*Verbatim=*/true};

  auto Resolved = URI::resolve(*U, HintPath);
  if (!Resolved)
    return Resolved.takeError();
  return HeaderFile{std::move(*Resolved), /*Verbatim=*/false};
}

llvm::SmallVector<llvm::StringRef, 1> getRankedIncludes(const Symbol &Sym) {
  auto Includes = Sym.IncludeHeaders;
  // Sort in descending order by reference count and header length.
  llvm::sort(Includes, [](const Symbol::IncludeHeaderWithReferences &LHS,
                          const Symbol::IncludeHeaderWithReferences &RHS) {
    if (LHS.References == RHS.References)
      return LHS.IncludeHeader.size() < RHS.IncludeHeader.size();
    return LHS.References > RHS.References;
  });
  llvm::SmallVector<llvm::StringRef, 1> Headers;
  for (const auto &Include : Includes)
    Headers.push_back(Include.IncludeHeader);
  return Headers;
}

std::unique_ptr<PPCallbacks>
collectIncludeStructureCallback(const SourceManager &SM,
                                IncludeStructure *Out) {
  return std::make_unique<RecordHeaders>(SM, Out);
}

void IncludeStructure::recordInclude(llvm::StringRef IncludingName,
                                     llvm::StringRef IncludedName,
                                     llvm::StringRef IncludedRealName) {
  auto Child = fileIndex(IncludedName);
  if (!IncludedRealName.empty() && RealPathNames[Child].empty())
    RealPathNames[Child] = std::string(IncludedRealName);
  auto Parent = fileIndex(IncludingName);
  IncludeChildren[Parent].push_back(Child);
}

unsigned IncludeStructure::fileIndex(llvm::StringRef Name) {
  auto R = NameToIndex.try_emplace(Name, RealPathNames.size());
  if (R.second)
    RealPathNames.emplace_back();
  return R.first->getValue();
}

llvm::StringMap<unsigned>
IncludeStructure::includeDepth(llvm::StringRef Root) const {
  // Include depth 0 is the main file only.
  llvm::StringMap<unsigned> Result;
  Result[Root] = 0;
  std::vector<unsigned> CurrentLevel;
  llvm::DenseSet<unsigned> Seen;
  auto It = NameToIndex.find(Root);
  if (It != NameToIndex.end()) {
    CurrentLevel.push_back(It->second);
    Seen.insert(It->second);
  }

  // Each round of BFS traversal finds the next depth level.
  std::vector<unsigned> PreviousLevel;
  for (unsigned Level = 1; !CurrentLevel.empty(); ++Level) {
    PreviousLevel.clear();
    PreviousLevel.swap(CurrentLevel);
    for (const auto &Parent : PreviousLevel) {
      for (const auto &Child : IncludeChildren.lookup(Parent)) {
        if (Seen.insert(Child).second) {
          CurrentLevel.push_back(Child);
          const auto &Name = RealPathNames[Child];
          // Can't include files if we don't have their real path.
          if (!Name.empty())
            Result[Name] = Level;
        }
      }
    }
  }
  return Result;
}

void IncludeInserter::addExisting(const Inclusion &Inc) {
  IncludedHeaders.insert(Inc.Written);
  if (!Inc.Resolved.empty())
    IncludedHeaders.insert(Inc.Resolved);
}

/// FIXME(ioeric): we might not want to insert an absolute include path if the
/// path is not shortened.
bool IncludeInserter::shouldInsertInclude(
    PathRef DeclaringHeader, const HeaderFile &InsertedHeader) const {
  assert(InsertedHeader.valid());
  if (!HeaderSearchInfo && !InsertedHeader.Verbatim)
    return false;
  if (FileName == DeclaringHeader || FileName == InsertedHeader.File)
    return false;
  auto Included = [&](llvm::StringRef Header) {
    return IncludedHeaders.find(Header) != IncludedHeaders.end();
  };
  return !Included(DeclaringHeader) && !Included(InsertedHeader.File);
}

llvm::Optional<std::string>
IncludeInserter::calculateIncludePath(const HeaderFile &InsertedHeader,
                                      llvm::StringRef IncludingFile) const {
  assert(InsertedHeader.valid());
  if (InsertedHeader.Verbatim)
    return InsertedHeader.File;
  bool IsSystem = false;
  std::string Suggested;
  if (HeaderSearchInfo) {
    Suggested = HeaderSearchInfo->suggestPathToFileForDiagnostics(
        InsertedHeader.File, BuildDir, IncludingFile, &IsSystem);
  } else {
    // Calculate include relative to including file only.
    StringRef IncludingDir = llvm::sys::path::parent_path(IncludingFile);
    SmallString<256> RelFile(InsertedHeader.File);
    // Replacing with "" leaves "/RelFile" if IncludingDir doesn't end in "/".
    llvm::sys::path::replace_path_prefix(RelFile, IncludingDir, "./");
    Suggested = llvm::sys::path::convert_to_slash(
        llvm::sys::path::remove_leading_dotslash(RelFile));
  }
  // FIXME: should we allow (some limited number of) "../header.h"?
  if (llvm::sys::path::is_absolute(Suggested))
    return None;
  if (IsSystem)
    Suggested = "<" + Suggested + ">";
  else
    Suggested = "\"" + Suggested + "\"";
  return Suggested;
}

llvm::Optional<TextEdit>
IncludeInserter::insert(llvm::StringRef VerbatimHeader) const {
  llvm::Optional<TextEdit> Edit = None;
  if (auto Insertion = Inserter.insert(VerbatimHeader.trim("\"<>"),
                                       VerbatimHeader.startswith("<")))
    Edit = replacementToEdit(Code, *Insertion);
  return Edit;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Inclusion &Inc) {
  return OS << Inc.Written << " = "
            << (!Inc.Resolved.empty() ? Inc.Resolved : "[unresolved]")
            << " at line" << Inc.HashLine;
}

bool operator==(const Inclusion &LHS, const Inclusion &RHS) {
  return std::tie(LHS.Directive, LHS.FileKind, LHS.HashOffset, LHS.HashLine,
                  LHS.Resolved, LHS.Written) ==
         std::tie(RHS.Directive, RHS.FileKind, RHS.HashOffset, RHS.HashLine,
                  RHS.Resolved, RHS.Written);
}
} // namespace clangd
} // namespace clang
