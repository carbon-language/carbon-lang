//===- CodeCoverage.cpp - Coverage tool based on profiling instrumentation-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The 'CodeCoverageTool' class implements a command line tool to analyze and
// report coverage information using the profiling instrumentation and code
// coverage mapping.
//
//===----------------------------------------------------------------------===//

#include "RenderingSupport.h"
#include "CoverageViewOptions.h"
#include "CoverageFilters.h"
#include "SourceCoverageDataManager.h"
#include "SourceCoverageView.h"
#include "CoverageSummary.h"
#include "CoverageReport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/CoverageMapping.h"
#include "llvm/ProfileData/CoverageMappingReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/PrettyStackTrace.h"
#include <functional>
#include <system_error>
#include <unordered_map>

using namespace llvm;
using namespace coverage;

namespace {
/// \brief Distribute the functions into instantiation sets.
/// An instantiation set is a collection of functions
/// that have the same source code, e.g.
/// template functions specializations.
class FunctionInstantiationSetCollector {
  ArrayRef<FunctionCoverageMapping> FunctionMappings;
  typedef uint64_t KeyType;
  typedef std::vector<const FunctionCoverageMapping *> SetType;
  std::unordered_map<uint64_t, SetType> InstantiatedFunctions;

  static KeyType getKey(const CountedRegion &R) {
    return uint64_t(R.LineStart) | uint64_t(R.ColumnStart) << 32;
  }

public:
  void insert(const FunctionCoverageMapping &Function, unsigned FileID) {
    KeyType Key = 0;
    for (const auto &R : Function.CountedRegions) {
      if (R.FileID == FileID) {
        Key = getKey(R);
        break;
      }
    }
    auto I = InstantiatedFunctions.find(Key);
    if (I == InstantiatedFunctions.end()) {
      SetType Set;
      Set.push_back(&Function);
      InstantiatedFunctions.insert(std::make_pair(Key, Set));
    } else
      I->second.push_back(&Function);
  }

  std::unordered_map<KeyType, SetType>::iterator begin() {
    return InstantiatedFunctions.begin();
  }

  std::unordered_map<KeyType, SetType>::iterator end() {
    return InstantiatedFunctions.end();
  }
};

/// \brief The implementation of the coverage tool.
class CodeCoverageTool {
public:
  enum Command {
    /// \brief The show command.
    Show,
    /// \brief The report command.
    Report
  };

  /// \brief Print the error message to the error output stream.
  void error(const Twine &Message, StringRef Whence = "");

  /// \brief Return a memory buffer for the given source file.
  ErrorOr<const MemoryBuffer &> getSourceFile(StringRef SourceFile);

  /// \brief Return true if two filepaths refer to the same file.
  bool equivalentFiles(StringRef A, StringRef B);

  /// \brief Collect a set of function's file ids which correspond to the
  /// given source file. Return false if the set is empty.
  bool gatherInterestingFileIDs(StringRef SourceFile,
                                const FunctionCoverageMapping &Function,
                                SmallSet<unsigned, 8> &InterestingFileIDs);

  /// \brief Find the file id which is not an expanded file id.
  bool findMainViewFileID(StringRef SourceFile,
                          const FunctionCoverageMapping &Function,
                          unsigned &MainViewFileID);

  bool findMainViewFileID(const FunctionCoverageMapping &Function,
                          unsigned &MainViewFileID);

  /// \brief Create a source view which shows coverage for an expansion
  /// of a file.
  void createExpansionSubView(const CountedRegion &ExpandedRegion,
                              const FunctionCoverageMapping &Function,
                              SourceCoverageView &Parent);

  void createExpansionSubViews(SourceCoverageView &View, unsigned ViewFileID,
                               const FunctionCoverageMapping &Function);

  /// \brief Create a source view which shows coverage for an instantiation
  /// of a funciton.
  void createInstantiationSubView(StringRef SourceFile,
                                  const FunctionCoverageMapping &Function,
                                  SourceCoverageView &View);

  /// \brief Create the main source view of a particular source file.
  /// Return true if this particular source file is not covered.
  bool
  createSourceFileView(StringRef SourceFile, SourceCoverageView &View,
                       ArrayRef<FunctionCoverageMapping> FunctionMappingRecords,
                       bool UseOnlyRegionsInMainFile = false);

  /// \brief Load the coverage mapping data. Return true if an error occured.
  bool load();

  int run(Command Cmd, int argc, const char **argv);

  typedef std::function<int(int, const char **)> CommandLineParserType;

  int show(int argc, const char **argv,
           CommandLineParserType commandLineParser);

  int report(int argc, const char **argv,
             CommandLineParserType commandLineParser);

  StringRef ObjectFilename;
  CoverageViewOptions ViewOpts;
  std::unique_ptr<IndexedInstrProfReader> PGOReader;
  CoverageFiltersMatchAll Filters;
  std::vector<std::string> SourceFiles;
  std::vector<std::pair<std::string, std::unique_ptr<MemoryBuffer>>>
      LoadedSourceFiles;
  std::vector<FunctionCoverageMapping> FunctionMappingRecords;
  bool CompareFilenamesOnly;
};
}

void CodeCoverageTool::error(const Twine &Message, StringRef Whence) {
  errs() << "error: ";
  if (!Whence.empty())
    errs() << Whence << ": ";
  errs() << Message << "\n";
}

ErrorOr<const MemoryBuffer &>
CodeCoverageTool::getSourceFile(StringRef SourceFile) {
  SmallString<256> Path(SourceFile);
  sys::fs::make_absolute(Path);
  for (const auto &Files : LoadedSourceFiles) {
    if (equivalentFiles(Path.str(), Files.first)) {
      return *Files.second;
    }
  }
  auto Buffer = MemoryBuffer::getFile(SourceFile);
  if (auto EC = Buffer.getError()) {
    error(EC.message(), SourceFile);
    return EC;
  }
  LoadedSourceFiles.push_back(std::make_pair(
      std::string(Path.begin(), Path.end()), std::move(Buffer.get())));
  return *LoadedSourceFiles.back().second;
}

/// \brief Return a line start - line end range which contains
/// all the mapping regions of a given function with a particular file id.
std::pair<unsigned, unsigned>
findExpandedFileInterestingLineRange(unsigned FileID,
                                     const FunctionCoverageMapping &Function) {
  unsigned LineStart = std::numeric_limits<unsigned>::max();
  unsigned LineEnd = 0;
  for (const auto &CR : Function.CountedRegions) {
    if (CR.FileID != FileID)
      continue;
    LineStart = std::min(CR.LineStart, LineStart);
    LineEnd = std::max(CR.LineEnd, LineEnd);
  }
  return std::make_pair(LineStart, LineEnd);
}

bool CodeCoverageTool::equivalentFiles(StringRef A, StringRef B) {
  if (CompareFilenamesOnly)
    return sys::path::filename(A).equals_lower(sys::path::filename(B));
  return sys::fs::equivalent(A, B);
}

bool CodeCoverageTool::gatherInterestingFileIDs(
    StringRef SourceFile, const FunctionCoverageMapping &Function,
    SmallSet<unsigned, 8> &InterestingFileIDs) {
  bool Interesting = false;
  for (unsigned I = 0, E = Function.Filenames.size(); I < E; ++I) {
    if (equivalentFiles(SourceFile, Function.Filenames[I])) {
      InterestingFileIDs.insert(I);
      Interesting = true;
    }
  }
  return Interesting;
}

bool
CodeCoverageTool::findMainViewFileID(StringRef SourceFile,
                                     const FunctionCoverageMapping &Function,
                                     unsigned &MainViewFileID) {
  llvm::SmallVector<bool, 8> IsExpandedFile(Function.Filenames.size(), false);
  llvm::SmallVector<bool, 8> FilenameEquivalence(Function.Filenames.size(),
                                                 false);
  for (unsigned I = 0, E = Function.Filenames.size(); I < E; ++I) {
    if (equivalentFiles(SourceFile, Function.Filenames[I]))
      FilenameEquivalence[I] = true;
  }
  for (const auto &CR : Function.CountedRegions) {
    if (CR.Kind == CounterMappingRegion::ExpansionRegion &&
        FilenameEquivalence[CR.FileID])
      IsExpandedFile[CR.ExpandedFileID] = true;
  }
  for (unsigned I = 0, E = Function.Filenames.size(); I < E; ++I) {
    if (!FilenameEquivalence[I] || IsExpandedFile[I])
      continue;
    MainViewFileID = I;
    return false;
  }
  return true;
}

bool
CodeCoverageTool::findMainViewFileID(const FunctionCoverageMapping &Function,
                                     unsigned &MainViewFileID) {
  llvm::SmallVector<bool, 8> IsExpandedFile(Function.Filenames.size(), false);
  for (const auto &CR : Function.CountedRegions) {
    if (CR.Kind == CounterMappingRegion::ExpansionRegion)
      IsExpandedFile[CR.ExpandedFileID] = true;
  }
  for (unsigned I = 0, E = Function.Filenames.size(); I < E; ++I) {
    if (IsExpandedFile[I])
      continue;
    MainViewFileID = I;
    return false;
  }
  return true;
}

void CodeCoverageTool::createExpansionSubView(
    const CountedRegion &ExpandedRegion,
    const FunctionCoverageMapping &Function, SourceCoverageView &Parent) {
  auto ExpandedLines = findExpandedFileInterestingLineRange(
      ExpandedRegion.ExpandedFileID, Function);
  if (ViewOpts.Debug)
    llvm::errs() << "Expansion of " << ExpandedRegion.ExpandedFileID << ":"
                 << ExpandedLines.first << " -> " << ExpandedLines.second
                 << " @ " << ExpandedRegion.FileID << ", "
                 << ExpandedRegion.LineStart << ":"
                 << ExpandedRegion.ColumnStart << "\n";
  auto SourceBuffer =
      getSourceFile(Function.Filenames[ExpandedRegion.ExpandedFileID]);
  if (!SourceBuffer)
    return;
  auto SubView = llvm::make_unique<SourceCoverageView>(
      SourceBuffer.get(), Parent.getOptions(), ExpandedRegion);
  SourceCoverageDataManager RegionManager;
  for (const auto &CR : Function.CountedRegions) {
    if (CR.FileID == ExpandedRegion.ExpandedFileID)
      RegionManager.insert(CR);
  }
  SubView->load(RegionManager);
  createExpansionSubViews(*SubView, ExpandedRegion.ExpandedFileID, Function);
  Parent.addChild(std::move(SubView));
}

void CodeCoverageTool::createExpansionSubViews(
    SourceCoverageView &View, unsigned ViewFileID,
    const FunctionCoverageMapping &Function) {
  if (!ViewOpts.ShowExpandedRegions)
    return;
  for (const auto &CR : Function.CountedRegions) {
    if (CR.Kind != CounterMappingRegion::ExpansionRegion)
      continue;
    if (CR.FileID != ViewFileID)
      continue;
    createExpansionSubView(CR, Function, View);
  }
}

void CodeCoverageTool::createInstantiationSubView(
    StringRef SourceFile, const FunctionCoverageMapping &Function,
    SourceCoverageView &View) {
  SourceCoverageDataManager RegionManager;
  SmallSet<unsigned, 8> InterestingFileIDs;
  if (!gatherInterestingFileIDs(SourceFile, Function, InterestingFileIDs))
    return;
  // Get the interesting regions
  for (const auto &CR : Function.CountedRegions) {
    if (InterestingFileIDs.count(CR.FileID))
      RegionManager.insert(CR);
  }
  View.load(RegionManager);
  unsigned MainFileID;
  if (findMainViewFileID(SourceFile, Function, MainFileID))
    return;
  createExpansionSubViews(View, MainFileID, Function);
}

bool CodeCoverageTool::createSourceFileView(
    StringRef SourceFile, SourceCoverageView &View,
    ArrayRef<FunctionCoverageMapping> FunctionMappingRecords,
    bool UseOnlyRegionsInMainFile) {
  SourceCoverageDataManager RegionManager;
  FunctionInstantiationSetCollector InstantiationSetCollector;

  for (const auto &Function : FunctionMappingRecords) {
    unsigned MainFileID;
    if (findMainViewFileID(SourceFile, Function, MainFileID))
      continue;
    SmallSet<unsigned, 8> InterestingFileIDs;
    if (UseOnlyRegionsInMainFile) {
      InterestingFileIDs.insert(MainFileID);
    } else if (!gatherInterestingFileIDs(SourceFile, Function,
                                         InterestingFileIDs))
      continue;
    // Get the interesting regions
    for (const auto &CR : Function.CountedRegions) {
      if (InterestingFileIDs.count(CR.FileID))
        RegionManager.insert(CR);
    }
    InstantiationSetCollector.insert(Function, MainFileID);
    createExpansionSubViews(View, MainFileID, Function);
  }
  if (RegionManager.getSourceRegions().empty())
    return true;
  View.load(RegionManager);
  // Show instantiations
  if (!ViewOpts.ShowFunctionInstantiations)
    return false;
  for (const auto &InstantiationSet : InstantiationSetCollector) {
    if (InstantiationSet.second.size() < 2)
      continue;
    for (auto Function : InstantiationSet.second) {
      auto SubView =
          llvm::make_unique<SourceCoverageView>(View, Function->Name);
      createInstantiationSubView(SourceFile, *Function, *SubView);
      View.addChild(std::move(SubView));
    }
  }
  return false;
}

bool CodeCoverageTool::load() {
  auto CounterMappingBuff = MemoryBuffer::getFileOrSTDIN(ObjectFilename);
  if (auto EC = CounterMappingBuff.getError()) {
    error(EC.message(), ObjectFilename);
    return true;
  }
  ObjectFileCoverageMappingReader MappingReader(CounterMappingBuff.get());
  if (auto EC = MappingReader.readHeader()) {
    error(EC.message(), ObjectFilename);
    return true;
  }

  std::vector<uint64_t> Counts;
  for (const auto &I : MappingReader) {
    FunctionCoverageMapping Function(I.FunctionName, I.Filenames);

    // Create the mapping regions with evaluated execution counts
    Counts.clear();
    PGOReader->getFunctionCounts(Function.Name, I.FunctionHash, Counts);

    // Get the biggest referenced counters
    bool RegionError = false;
    CounterMappingContext Ctx(I.Expressions, Counts);
    for (const auto &R : I.MappingRegions) {
      // Compute the values of mapped regions
      if (ViewOpts.Debug) {
        errs() << "File " << R.FileID << "| " << R.LineStart << ":"
               << R.ColumnStart << " -> " << R.LineEnd << ":" << R.ColumnEnd
               << " = ";
        Ctx.dump(R.Count);
        if (R.Kind == CounterMappingRegion::ExpansionRegion) {
          errs() << " (Expanded file id = " << R.ExpandedFileID << ") ";
        }
        errs() << "\n";
      }
      ErrorOr<int64_t> ExecutionCount = Ctx.evaluate(R.Count);
      if (ExecutionCount) {
        Function.CountedRegions.push_back(CountedRegion(R, *ExecutionCount));
      } else if (!RegionError) {
        colored_ostream(errs(), raw_ostream::RED)
            << "error: Regions and counters don't match in a function '"
            << Function.Name << "' (re-run the instrumented binary).";
        errs() << "\n";
        RegionError = true;
      }
    }

    if (RegionError || !Filters.matches(Function))
      continue;

    FunctionMappingRecords.push_back(Function);
  }
  return false;
}

int CodeCoverageTool::run(Command Cmd, int argc, const char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  cl::list<std::string> InputSourceFiles(
      cl::Positional, cl::desc("<Source files>"), cl::ZeroOrMore);

  cl::opt<std::string> PGOFilename(
      "instr-profile", cl::Required,
      cl::desc(
          "File with the profile data obtained after an instrumented run"));

  cl::opt<bool> DebugDump("dump", cl::Optional,
                          cl::desc("Show internal debug dump"));

  cl::opt<bool> FilenameEquivalence(
      "filename-equivalence", cl::Optional,
      cl::desc("Compare the filenames instead of full filepaths"));

  cl::OptionCategory FilteringCategory("Function filtering options");

  cl::list<std::string> NameFilters(
      "name", cl::Optional,
      cl::desc("Show code coverage only for functions with the given name"),
      cl::ZeroOrMore, cl::cat(FilteringCategory));

  cl::list<std::string> NameRegexFilters(
      "name-regex", cl::Optional,
      cl::desc("Show code coverage only for functions that match the given "
               "regular expression"),
      cl::ZeroOrMore, cl::cat(FilteringCategory));

  cl::opt<double> RegionCoverageLtFilter(
      "region-coverage-lt", cl::Optional,
      cl::desc("Show code coverage only for functions with region coverage "
               "less than the given threshold"),
      cl::cat(FilteringCategory));

  cl::opt<double> RegionCoverageGtFilter(
      "region-coverage-gt", cl::Optional,
      cl::desc("Show code coverage only for functions with region coverage "
               "greater than the given threshold"),
      cl::cat(FilteringCategory));

  cl::opt<double> LineCoverageLtFilter(
      "line-coverage-lt", cl::Optional,
      cl::desc("Show code coverage only for functions with line coverage less "
               "than the given threshold"),
      cl::cat(FilteringCategory));

  cl::opt<double> LineCoverageGtFilter(
      "line-coverage-gt", cl::Optional,
      cl::desc("Show code coverage only for functions with line coverage "
               "greater than the given threshold"),
      cl::cat(FilteringCategory));

  auto commandLineParser = [&, this](int argc, const char **argv) -> int {
    cl::ParseCommandLineOptions(argc, argv, "LLVM code coverage tool\n");
    ViewOpts.Debug = DebugDump;
    CompareFilenamesOnly = FilenameEquivalence;

    if (auto EC = IndexedInstrProfReader::create(PGOFilename, PGOReader)) {
      error(EC.message(), PGOFilename);
      return 1;
    }

    // Create the function filters
    if (!NameFilters.empty() || !NameRegexFilters.empty()) {
      auto NameFilterer = new CoverageFilters;
      for (const auto &Name : NameFilters)
        NameFilterer->push_back(llvm::make_unique<NameCoverageFilter>(Name));
      for (const auto &Regex : NameRegexFilters)
        NameFilterer->push_back(
            llvm::make_unique<NameRegexCoverageFilter>(Regex));
      Filters.push_back(std::unique_ptr<CoverageFilter>(NameFilterer));
    }
    if (RegionCoverageLtFilter.getNumOccurrences() ||
        RegionCoverageGtFilter.getNumOccurrences() ||
        LineCoverageLtFilter.getNumOccurrences() ||
        LineCoverageGtFilter.getNumOccurrences()) {
      auto StatFilterer = new CoverageFilters;
      if (RegionCoverageLtFilter.getNumOccurrences())
        StatFilterer->push_back(llvm::make_unique<RegionCoverageFilter>(
            RegionCoverageFilter::LessThan, RegionCoverageLtFilter));
      if (RegionCoverageGtFilter.getNumOccurrences())
        StatFilterer->push_back(llvm::make_unique<RegionCoverageFilter>(
            RegionCoverageFilter::GreaterThan, RegionCoverageGtFilter));
      if (LineCoverageLtFilter.getNumOccurrences())
        StatFilterer->push_back(llvm::make_unique<LineCoverageFilter>(
            LineCoverageFilter::LessThan, LineCoverageLtFilter));
      if (LineCoverageGtFilter.getNumOccurrences())
        StatFilterer->push_back(llvm::make_unique<LineCoverageFilter>(
            RegionCoverageFilter::GreaterThan, LineCoverageGtFilter));
      Filters.push_back(std::unique_ptr<CoverageFilter>(StatFilterer));
    }

    SourceFiles = InputSourceFiles;
    return 0;
  };

  // Parse the object filename
  if (argc > 1) {
    StringRef Arg(argv[1]);
    if (Arg.equals_lower("-help") || Arg.equals_lower("-version")) {
      cl::ParseCommandLineOptions(2, argv, "LLVM code coverage tool\n");
      return 0;
    }
    ObjectFilename = Arg;

    argv[1] = argv[0];
    --argc;
    ++argv;
  } else {
    errs() << sys::path::filename(argv[0]) << ": No executable file given!\n";
    return 1;
  }

  switch (Cmd) {
  case Show:
    return show(argc, argv, commandLineParser);
  case Report:
    return report(argc, argv, commandLineParser);
  }
  return 0;
}

int CodeCoverageTool::show(int argc, const char **argv,
                           CommandLineParserType commandLineParser) {

  cl::OptionCategory ViewCategory("Viewing options");

  cl::opt<bool> ShowLineExecutionCounts(
      "show-line-counts", cl::Optional,
      cl::desc("Show the execution counts for each line"), cl::init(true),
      cl::cat(ViewCategory));

  cl::opt<bool> ShowRegions(
      "show-regions", cl::Optional,
      cl::desc("Show the execution counts for each region"),
      cl::cat(ViewCategory));

  cl::opt<bool> ShowBestLineRegionsCounts(
      "show-line-counts-or-regions", cl::Optional,
      cl::desc("Show the execution counts for each line, or the execution "
               "counts for each region on lines that have multiple regions"),
      cl::cat(ViewCategory));

  cl::opt<bool> ShowExpansions("show-expansions", cl::Optional,
                               cl::desc("Show expanded source regions"),
                               cl::cat(ViewCategory));

  cl::opt<bool> ShowInstantiations("show-instantiations", cl::Optional,
                                   cl::desc("Show function instantiations"),
                                   cl::cat(ViewCategory));

  cl::opt<bool> NoColors("no-colors", cl::Optional,
                         cl::desc("Don't show text colors"), cl::init(false),
                         cl::cat(ViewCategory));

  auto Err = commandLineParser(argc, argv);
  if (Err)
    return Err;

  ViewOpts.Colors = !NoColors;
  ViewOpts.ShowLineNumbers = true;
  ViewOpts.ShowLineStats = ShowLineExecutionCounts.getNumOccurrences() != 0 ||
                           !ShowRegions || ShowBestLineRegionsCounts;
  ViewOpts.ShowRegionMarkers = ShowRegions || ShowBestLineRegionsCounts;
  ViewOpts.ShowLineStatsOrRegionMarkers = ShowBestLineRegionsCounts;
  ViewOpts.ShowExpandedRegions = ShowExpansions;
  ViewOpts.ShowFunctionInstantiations = ShowInstantiations;

  if (load())
    return 1;

  if (!Filters.empty()) {
    // Show functions
    for (const auto &Function : FunctionMappingRecords) {
      unsigned MainFileID;
      if (findMainViewFileID(Function, MainFileID))
        continue;
      StringRef SourceFile = Function.Filenames[MainFileID];
      auto SourceBuffer = getSourceFile(SourceFile);
      if (!SourceBuffer)
        return 1;
      SourceCoverageView mainView(SourceBuffer.get(), ViewOpts);
      createSourceFileView(SourceFile, mainView, Function, true);
      ViewOpts.colored_ostream(outs(), raw_ostream::CYAN)
          << Function.Name << " from " << SourceFile << ":";
      outs() << "\n";
      mainView.render(outs());
      if (FunctionMappingRecords.size() > 1)
        outs() << "\n";
    }
    return 0;
  }

  // Show files
  bool ShowFilenames = SourceFiles.size() != 1;

  if (SourceFiles.empty()) {
    // Get the source files from the function coverage mapping
    std::set<StringRef> UniqueFilenames;
    for (const auto &Function : FunctionMappingRecords) {
      for (const auto &Filename : Function.Filenames)
        UniqueFilenames.insert(Filename);
    }
    for (const auto &Filename : UniqueFilenames)
      SourceFiles.push_back(Filename);
  }

  for (const auto &SourceFile : SourceFiles) {
    auto SourceBuffer = getSourceFile(SourceFile);
    if (!SourceBuffer)
      return 1;
    SourceCoverageView mainView(SourceBuffer.get(), ViewOpts);
    if (createSourceFileView(SourceFile, mainView, FunctionMappingRecords)) {
      ViewOpts.colored_ostream(outs(), raw_ostream::RED)
          << "warning: The file '" << SourceFile << "' isn't covered.";
      outs() << "\n";
      continue;
    }

    if (ShowFilenames) {
      ViewOpts.colored_ostream(outs(), raw_ostream::CYAN) << SourceFile << ":";
      outs() << "\n";
    }
    mainView.render(outs());
    if (SourceFiles.size() > 1)
      outs() << "\n";
  }

  return 0;
}

int CodeCoverageTool::report(int argc, const char **argv,
                             CommandLineParserType commandLineParser) {
  cl::opt<bool> NoColors("no-colors", cl::Optional,
                         cl::desc("Don't show text colors"), cl::init(false));

  auto Err = commandLineParser(argc, argv);
  if (Err)
    return Err;

  ViewOpts.Colors = !NoColors;

  if (load())
    return 1;

  CoverageSummary Summarizer;
  Summarizer.createSummaries(FunctionMappingRecords);
  CoverageReport Report(ViewOpts, Summarizer);
  if (SourceFiles.empty() && Filters.empty()) {
    Report.renderFileReports(llvm::outs());
    return 0;
  }

  Report.renderFunctionReports(llvm::outs());
  return 0;
}

int show_main(int argc, const char **argv) {
  CodeCoverageTool Tool;
  return Tool.run(CodeCoverageTool::Show, argc, argv);
}

int report_main(int argc, const char **argv) {
  CodeCoverageTool Tool;
  return Tool.run(CodeCoverageTool::Report, argc, argv);
}
