//===-- ClangApplyReplacementsMain.cpp - Main file for the tool -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the main function for the
/// clang-apply-replacements tool.
///
//===----------------------------------------------------------------------===//

#include "clang-apply-replacements/Tooling/ApplyReplacements.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Format/Format.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace clang;
using namespace clang::replace;

static cl::opt<std::string> Directory(cl::Positional, cl::Required,
                                      cl::desc("<Search Root Directory>"));

static cl::OptionCategory FormattingCategory("Formatting Options");

static cl::opt<bool> RemoveTUReplacementFiles(
    "remove-change-desc-files",
    cl::desc("Remove the change description files regardless of successful\n"
             "merging/replacing."),
    cl::init(false));

// Update this list of options to show in -help as new options are added.
// Should add even those options marked as 'Hidden'. Any option not listed
// here will get marked 'ReallyHidden' so they don't appear in any -help text.
const char *OptionsToShow[] = { "help",                     "version",
                                "remove-change-desc-files", "format",
                                "style-config",             "style" };

static cl::opt<bool> DoFormat(
    "format",
    cl::desc("Enable formatting of code changed by applying replacements.\n"
             "Use -style to choose formatting style.\n"),
    cl::cat(FormattingCategory));

// FIXME: Consider making the default behaviour for finding a style
// configuration file to start the search anew for every file being changed to
// handle situations where the style is different for different parts of a
// project.

static cl::opt<std::string> FormatStyleConfig(
    "style-config",
    cl::desc("Path to a directory containing a .clang-format file\n"
             "describing a formatting style to use for formatting\n"
             "code when -style=file.\n"),
    cl::init(""), cl::cat(FormattingCategory));

static cl::opt<std::string>
FormatStyleOpt("style", cl::desc(format::StyleOptionHelpDescription),
               cl::init("LLVM"), cl::cat(FormattingCategory));

// Helper object to remove the TUReplacement files (triggered by
// "remove-change-desc-files" command line option) when exiting current scope.
class ScopedFileRemover {
public:
  ScopedFileRemover(const TUReplacementFiles &Files,
                    clang::DiagnosticsEngine &Diagnostics)
      : TURFiles(Files), Diag(Diagnostics) {}

  ~ScopedFileRemover() {
    deleteReplacementFiles(TURFiles, Diag);
  }

private:
  const TUReplacementFiles &TURFiles;
  clang::DiagnosticsEngine &Diag;
};

void printVersion() {
  outs() << "clang-apply-replacements version " CLANG_VERSION_STRING << "\n";
}

/// \brief Convenience function to get rewritten content for \c Filename from
/// \c Rewrites.
///
/// \pre Replacements[i].getFilePath() == Replacements[i+1].getFilePath().
/// \post Replacements.empty() -> Result.empty()
///
/// \param[in] Replacements Replacements to apply
/// \param[in] Rewrites Rewriter to use to apply replacements.
/// \param[out] Result Contents of the file after applying replacements if
/// replacements were provided.
///
/// \returns \li true if all replacements were applied successfully.
///          \li false if at least one replacement failed to apply.
static bool
getRewrittenData(const std::vector<tooling::Replacement> &Replacements,
                 Rewriter &Rewrites, std::string &Result) {
  if (Replacements.empty()) return true;

  if (!tooling::applyAllReplacements(Replacements, Rewrites))
    return false;

  SourceManager &SM = Rewrites.getSourceMgr();
  FileManager &Files = SM.getFileManager();

  StringRef FileName = Replacements.begin()->getFilePath();
  const clang::FileEntry *Entry = Files.getFile(FileName);
  assert(Entry && "Expected an existing file");
  FileID ID = SM.translateFile(Entry);
  assert(!ID.isInvalid() && "Expected a valid FileID");
  const RewriteBuffer *Buffer = Rewrites.getRewriteBufferFor(ID);
  Result = std::string(Buffer->begin(), Buffer->end());

  return true;
}

/// \brief Apply \c Replacements and return the new file contents.
///
/// \pre Replacements[i].getFilePath() == Replacements[i+1].getFilePath().
/// \post Replacements.empty() -> Result.empty()
///
/// \param[in] Replacements Replacements to apply.
/// \param[out] Result Contents of the file after applying replacements if
/// replacements were provided.
/// \param[in] Diagnostics For diagnostic output.
///
/// \returns \li true if all replacements applied successfully.
///          \li false if at least one replacement failed to apply.
bool applyReplacements(const std::vector<tooling::Replacement> &Replacements,
                       std::string &Result,
                       DiagnosticsEngine &Diagnostics) {
  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);
  Rewriter Rewrites(SM, LangOptions());

  return getRewrittenData(Replacements, Rewrites, Result);
}

/// \brief Apply code formatting to all places where replacements were made.
///
/// \pre !Replacements.empty().
/// \pre Replacements[i].getFilePath() == Replacements[i+1].getFilePath().
/// \pre Replacements[i].getOffset() <= Replacements[i+1].getOffset().
///
/// \param[in] Replacements Replacements that were made to the file. Provided
/// to indicate where changes were made.
/// \param[in] FileData The contents of the file \b after \c Replacements have
/// been applied.
/// \param[out] FormattedFileData The contents of the file after reformatting.
/// \param[in] FormatStyle Style to apply.
/// \param[in] Diagnostics For diagnostic output.
///
/// \returns \li true if reformatting replacements were all successfully
///          applied.
///          \li false if at least one reformatting replacement failed to apply.
bool applyFormatting(const std::vector<tooling::Replacement> &Replacements,
                     const StringRef FileData,
                     std::string &FormattedFileData,
                     const format::FormatStyle &FormatStyle,
                     DiagnosticsEngine &Diagnostics) {
  assert(!Replacements.empty() && "Need at least one replacement");

  RangeVector Ranges = calculateChangedRanges(Replacements);

  StringRef FileName = Replacements.begin()->getFilePath();
  tooling::Replacements R =
      format::reformat(FormatStyle, FileData, Ranges, FileName);

  // FIXME: Remove this copy when tooling::Replacements is implemented as a
  // vector instead of a set.
  std::vector<tooling::Replacement> FormattingReplacements;
  std::copy(R.begin(), R.end(), back_inserter(FormattingReplacements));

  if (FormattingReplacements.empty()) {
    FormattedFileData = FileData;
    return true;
  }

  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);
  SM.overrideFileContents(Files.getFile(FileName),
                          llvm::MemoryBuffer::getMemBufferCopy(FileData));
  Rewriter Rewrites(SM, LangOptions());

  return getRewrittenData(FormattingReplacements, Rewrites, FormattedFileData);
}

int main(int argc, char **argv) {
  // Only include our options in -help output.
  StringMap<cl::Option*> OptMap;
  cl::getRegisteredOptions(OptMap);
  const char **EndOpts = OptionsToShow + array_lengthof(OptionsToShow);
  for (StringMap<cl::Option *>::iterator I = OptMap.begin(), E = OptMap.end();
       I != E; ++I) {
    if (std::find(OptionsToShow, EndOpts, I->getKey()) == EndOpts)
      I->getValue()->setHiddenFlag(cl::ReallyHidden);
  }

  cl::SetVersionPrinter(&printVersion);
  cl::ParseCommandLineOptions(argc, argv);

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      DiagOpts.getPtr());

  // Determine a formatting style from options.
  format::FormatStyle FormatStyle;
  if (DoFormat)
    FormatStyle = format::getStyle(FormatStyleOpt, FormatStyleConfig, "LLVM");

  TUReplacements TUs;
  TUReplacementFiles TURFiles;

  error_code ErrorCode =
      collectReplacementsFromDirectory(Directory, TUs, TURFiles, Diagnostics);

  if (ErrorCode) {
    errs() << "Trouble iterating over directory '" << Directory
           << "': " << ErrorCode.message() << "\n";
    return 1;
  }

  // Remove the TUReplacementFiles (triggered by "remove-change-desc-files"
  // command line option) when exiting main().
  OwningPtr<ScopedFileRemover> Remover;
  if (RemoveTUReplacementFiles)
    Remover.reset(new ScopedFileRemover(TURFiles, Diagnostics));

  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);

  FileToReplacementsMap GroupedReplacements;
  if (!mergeAndDeduplicate(TUs, GroupedReplacements, SM))
    return 1;

  Rewriter ReplacementsRewriter(SM, LangOptions());

  for (FileToReplacementsMap::const_iterator I = GroupedReplacements.begin(),
                                             E = GroupedReplacements.end();
       I != E; ++I) {

    std::string NewFileData;

    // This shouldn't happen but if a file somehow has no replacements skip to
    // next file.
    if (I->getValue().empty())
      continue;

    if (!applyReplacements(I->getValue(), NewFileData, Diagnostics)) {
      errs() << "Failed to apply replacements to " << I->getKey() << "\n";
      continue;
    }

    // Apply formatting if requested.
    if (DoFormat && !applyFormatting(I->getValue(), NewFileData, NewFileData,
                                     FormatStyle, Diagnostics)) {
      errs() << "Failed to apply reformatting replacements for " << I->getKey()
             << "\n";
      continue;
    }

    // Write new file to disk
    std::string ErrorInfo;
    llvm::raw_fd_ostream FileStream(I->getKey().str().c_str(), ErrorInfo,
                                    llvm::sys::fs::F_Text);
    if (!ErrorInfo.empty()) {
      llvm::errs() << "Could not open " << I->getKey() << " for writing\n";
      continue;
    }

    FileStream << NewFileData;
  }

  return 0;
}
