//===-- Core/ReplacementHandling.cpp --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides implementations for the ReplacementHandling class.
///
//===----------------------------------------------------------------------===//

#include "Core/ReplacementHandling.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/system_error.h"

using namespace llvm;
using namespace llvm::sys;
using namespace clang::tooling;

bool ReplacementHandling::findClangApplyReplacements(const char *Argv0) {
  CARPath = FindProgramByName("clang-apply-replacements");

  if (!CARPath.empty())
    return true;

  static int StaticSymbol;
  CARPath = fs::getMainExecutable(Argv0, &StaticSymbol);
  SmallString<128> TestPath = path::parent_path(CARPath);
  path::append(TestPath, "clang-apply-replacements");
  if (fs::can_execute(Twine(TestPath)))
    CARPath = TestPath.str();

  return !CARPath.empty();
}

StringRef ReplacementHandling::useTempDestinationDir() {
  DestinationDir = generateTempDir();
  return DestinationDir;
}

void ReplacementHandling::enableFormatting(StringRef Style,
                                           StringRef StyleConfigDir) {
  DoFormat = true;
  FormatStyle = Style;
  this->StyleConfigDir = StyleConfigDir;
}

bool ReplacementHandling::serializeReplacements(
    const TUReplacementsMap &Replacements) {
  assert(!DestinationDir.empty() && "Destination directory not set");

  bool Errors = false;

  for (TUReplacementsMap::const_iterator I = Replacements.begin(),
                                         E = Replacements.end();
       I != E; ++I) {
    SmallString<128> ReplacementsFileName;
    SmallString<64> Error;
    bool Result = generateReplacementsFileName(DestinationDir,
                                               I->getValue().MainSourceFile,
                                               ReplacementsFileName, Error);
    if (!Result) {
      errs() << "Failed to generate replacements filename:" << Error << "\n";
      Errors = true;
      continue;
    }

    std::string ErrorInfo;
    raw_fd_ostream ReplacementsFile(ReplacementsFileName.c_str(), ErrorInfo,
                                    fs::F_None);
    if (!ErrorInfo.empty()) {
      errs() << "Error opening file: " << ErrorInfo << "\n";
      Errors = true;
      continue;
    }
    yaml::Output YAML(ReplacementsFile);
    YAML << const_cast<TranslationUnitReplacements &>(I->getValue());
  }
  return !Errors;
}

bool ReplacementHandling::applyReplacements() {
  SmallVector<const char *, 8> Argv;
  Argv.push_back(CARPath.c_str());
  std::string Style = "--style=" + FormatStyle;
  std::string StyleConfig = "--style-config=" + StyleConfigDir;
  if (DoFormat) {
    Argv.push_back("--format");
    Argv.push_back(Style.c_str());
    if (!StyleConfigDir.empty())
      Argv.push_back(StyleConfig.c_str());
  }
  Argv.push_back("--remove-change-desc-files");
  Argv.push_back(DestinationDir.c_str());

  // Argv array needs to be null terminated.
  Argv.push_back(0);

  std::string ErrorMsg;
  bool ExecutionFailed = false;
  int ReturnCode = ExecuteAndWait(CARPath.c_str(), Argv.data(), /* env */ 0,
                                  /* redirects */ 0,
                                  /* secondsToWait */ 0, /* memoryLimit */ 0,
                                  &ErrorMsg, &ExecutionFailed);
  if (ExecutionFailed || !ErrorMsg.empty()) {
    errs() << "Failed to launch clang-apply-replacements: " << ErrorMsg << "\n";
    errs() << "Command Line:\n";
    for (const char **I = Argv.begin(), **E = Argv.end(); I != E; ++I) {
      if (*I)
        errs() << *I << "\n";
    }
    return false;
  }

  if (ReturnCode != 0) {
    errs() << "clang-apply-replacements failed with return code " << ReturnCode
           << "\n";
    return false;
  }

  return true;
}

std::string ReplacementHandling::generateTempDir() {
  SmallString<128> Prefix;
  path::system_temp_directory(true, Prefix);
  path::append(Prefix, "clang-modernize");
  SmallString<128> Result;
  fs::createUniqueDirectory(Twine(Prefix), Result);
  return Result.str();
}

bool ReplacementHandling::generateReplacementsFileName(
    StringRef DestinationDir, StringRef MainSourceFile,
    SmallVectorImpl<char> &Result, SmallVectorImpl<char> &Error) {

  Error.clear();
  SmallString<128> Prefix = DestinationDir;
  path::append(Prefix, path::filename(MainSourceFile));
  if (error_code EC =
          fs::createUniqueFile(Prefix + "_%%_%%_%%_%%_%%_%%.yaml", Result)) {
    const std::string &Msg = EC.message();
    Error.append(Msg.begin(), Msg.end());
    return false;
  }

  return true;
}
