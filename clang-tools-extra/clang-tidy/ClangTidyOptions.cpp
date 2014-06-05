//===--- ClangTidyOptions.cpp - clang-tidy ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangTidyOptions.h"
#include "llvm/Support/YAMLTraits.h"

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

/// \brief Parses -line-filter option and stores it to the \c Options.
llvm::error_code parseLineFilter(const std::string &LineFilter,
                                 clang::tidy::ClangTidyGlobalOptions &Options) {
  llvm::yaml::Input Input(LineFilter);
  Input >> Options.LineFilter;
  return Input.error();
}

llvm::error_code parseConfiguration(const std::string &Config,
                                    clang::tidy::ClangTidyOptions &Options) {
  llvm::yaml::Input Input(Config);
  Input >> Options;
  return Input.error();
}

} // namespace tidy
} // namespace clang
