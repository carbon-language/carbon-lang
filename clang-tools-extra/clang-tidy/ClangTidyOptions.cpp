//===--- ClangTidyOptions.cpp - clang-tidy ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangTidyOptions.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/YAMLTraits.h"
#include <utility>

#define DEBUG_TYPE "clang-tidy-options"

using clang::tidy::ClangTidyOptions;
using clang::tidy::FileFilter;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(FileFilter)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(FileFilter::LineRange)

namespace llvm {
namespace yaml {

// Map std::pair<int, int> to a JSON array of size 2.
template <> struct SequenceTraits<FileFilter::LineRange> {
  static size_t size(IO &IO, FileFilter::LineRange &Range) {
    return Range.first == 0 ? 0 : Range.second == 0 ? 1 : 2;
  }
  static unsigned &element(IO &IO, FileFilter::LineRange &Range, size_t Index) {
    if (Index > 1)
      IO.setError("Too many elements in line range.");
    return Index == 0 ? Range.first : Range.second;
  }
};

template <> struct MappingTraits<FileFilter> {
  static void mapping(IO &IO, FileFilter &File) {
    IO.mapRequired("name", File.Name);
    IO.mapOptional("lines", File.LineRanges);
  }
  static StringRef validate(IO &io, FileFilter &File) {
    if (File.Name.empty())
      return "No file name specified";
    for (const FileFilter::LineRange &Range : File.LineRanges) {
      if (Range.first <= 0 || Range.second <= 0)
        return "Invalid line range";
    }
    return StringRef();
  }
};

template <> struct MappingTraits<ClangTidyOptions> {
  static void mapping(IO &IO, ClangTidyOptions &Options) {
    IO.mapOptional("Checks", Options.Checks);
    IO.mapOptional("HeaderFilterRegex", Options.HeaderFilterRegex);
    IO.mapOptional("AnalyzeTemporaryDtors", Options.AnalyzeTemporaryDtors);
  }
};

} // namespace yaml
} // namespace llvm

namespace clang {
namespace tidy {

ClangTidyOptions
ClangTidyOptions::mergeWith(const ClangTidyOptions &Other) const {
  ClangTidyOptions Result = *this;

  // Merge comma-separated glob lists by appending the new value after a comma.
  if (Other.Checks)
    Result.Checks =
        (Result.Checks && !Result.Checks->empty() ? *Result.Checks + "," : "") +
        *Other.Checks;

  if (Other.HeaderFilterRegex)
    Result.HeaderFilterRegex = Other.HeaderFilterRegex;
  if (Other.AnalyzeTemporaryDtors)
    Result.AnalyzeTemporaryDtors = Other.AnalyzeTemporaryDtors;
  return Result;
}

FileOptionsProvider::FileOptionsProvider(
    const ClangTidyGlobalOptions &GlobalOptions,
    const ClangTidyOptions &FallbackOptions,
    const ClangTidyOptions &OverrideOptions)
    : DefaultOptionsProvider(GlobalOptions, FallbackOptions),
      OverrideOptions(OverrideOptions) {
  CachedOptions[""] = FallbackOptions.mergeWith(OverrideOptions);
}

static const char ConfigFileName[] = ".clang-tidy";

// FIXME: This method has some common logic with clang::format::getStyle().
// Consider pulling out common bits to a findParentFileWithName function or
// similar.
const ClangTidyOptions &FileOptionsProvider::getOptions(StringRef FileName) {
  DEBUG(llvm::dbgs() << "Getting options for file " << FileName << "...\n");
  SmallString<256> FilePath(FileName);

  if (std::error_code EC = llvm::sys::fs::make_absolute(FilePath)) {
    llvm::errs() << "Can't make absolute path from " << FileName << ": "
                 << EC.message() << "\n";
    // FIXME: Figure out what to do.
  } else {
    FileName = FilePath;
  }

  // Look for a suitable configuration file in all parent directories of the
  // file. Start with the immediate parent directory and move up.
  StringRef Path = llvm::sys::path::parent_path(FileName);
  for (StringRef CurrentPath = Path;;
       CurrentPath = llvm::sys::path::parent_path(CurrentPath)) {
    llvm::ErrorOr<ClangTidyOptions> Result = std::error_code();

    auto Iter = CachedOptions.find(CurrentPath);
    if (Iter != CachedOptions.end())
      Result = Iter->second;

    if (!Result)
      Result = TryReadConfigFile(CurrentPath);

    if (Result) {
      // Store cached value for all intermediate directories.
      while (Path != CurrentPath) {
        DEBUG(llvm::dbgs() << "Caching configuration for path " << Path
                           << ".\n");
        CachedOptions.GetOrCreateValue(Path, *Result);
        Path = llvm::sys::path::parent_path(Path);
      }
      return CachedOptions.GetOrCreateValue(Path, *Result).getValue();
    }
    if (Result.getError() != llvm::errc::no_such_file_or_directory) {
      llvm::errs() << "Error reading " << ConfigFileName << " from " << Path
                   << ": " << Result.getError().message() << "\n";
    }
  }
}

llvm::ErrorOr<ClangTidyOptions>
FileOptionsProvider::TryReadConfigFile(StringRef Directory) {
  assert(!Directory.empty());

  ClangTidyOptions Options = DefaultOptionsProvider::getOptions(Directory);
  if (!llvm::sys::fs::is_directory(Directory))
    return make_error_code(llvm::errc::not_a_directory);

  SmallString<128> ConfigFile(Directory);
  llvm::sys::path::append(ConfigFile, ".clang-tidy");
  DEBUG(llvm::dbgs() << "Trying " << ConfigFile << "...\n");

  bool IsFile = false;
  // Ignore errors from is_regular_file: we only need to know if we can read
  // the file or not.
  llvm::sys::fs::is_regular_file(Twine(ConfigFile), IsFile);

  if (!IsFile)
    return make_error_code(llvm::errc::no_such_file_or_directory);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
      llvm::MemoryBuffer::getFile(ConfigFile.c_str());
  if (std::error_code EC = Text.getError())
    return EC;
  if (std::error_code EC = parseConfiguration((*Text)->getBuffer(), Options))
    return EC;
  return Options.mergeWith(OverrideOptions);
}

/// \brief Parses -line-filter option and stores it to the \c Options.
std::error_code parseLineFilter(StringRef LineFilter,
                                clang::tidy::ClangTidyGlobalOptions &Options) {
  llvm::yaml::Input Input(LineFilter);
  Input >> Options.LineFilter;
  return Input.error();
}

std::error_code parseConfiguration(StringRef Config,
                                   clang::tidy::ClangTidyOptions &Options) {
  llvm::yaml::Input Input(Config);
  Input >> Options;
  return Input.error();
}

std::string configurationAsText(const ClangTidyOptions &Options) {
  std::string Text;
  llvm::raw_string_ostream Stream(Text);
  llvm::yaml::Output Output(Stream);
  // We use the same mapping method for input and output, so we need a non-const
  // reference here.
  ClangTidyOptions NonConstValue = Options;
  Output << NonConstValue;
  return Stream.str();
}

} // namespace tidy
} // namespace clang
