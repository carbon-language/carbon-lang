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
#include "llvm/ADT/DenseMap.h"
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

using namespace llvm;
using namespace coverage;

namespace {
/// \brief Distribute the functions into instantiation sets.
///
/// An instantiation set is a collection of functions that have the same source
/// code, ie, template functions specializations.
class FunctionInstantiationSetCollector {
  typedef DenseMap<std::pair<unsigned, unsigned>,
                   std::vector<const FunctionCoverageMapping *>> MapT;
  MapT InstantiatedFunctions;

public:
  void insert(const FunctionCoverageMapping &Function, unsigned FileID) {
    auto I = Function.CountedRegions.begin(), E = Function.CountedRegions.end();
    while (I != E && I->FileID != FileID)
      ++I;
    assert(I != E && "function does not cover the given file");
    auto &Functions = InstantiatedFunctions[I->startLoc()];
    Functions.push_back(&Function);
  }

  MapT::iterator begin() {
    return InstantiatedFunctions.begin();
  }

  MapT::iterator end() {
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
  std::unique_ptr<SourceCoverageView>
  createExpansionSubView(const CountedRegion &ExpandedRegion,
                         const FunctionCoverageMapping &Function);

  void attachExpansionSubViews(SourceCoverageView &View, unsigned ViewFileID,
                               const FunctionCoverageMapping &Function);

  /// \brief Create a source view which shows coverage for an instantiation
  /// of a funciton.
  std::unique_ptr<SourceCoverageView>
  createInstantiationSubView(StringRef SourceFile,
                             const FunctionCoverageMapping &Function);

  /// \brief Create the main source view of a particular source file.
  std::unique_ptr<SourceCoverageView>
  createSourceFileView(StringRef SourceFile,
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
  StringMap<std::string> RemappedFilenames;
};
}

static std::vector<StringRef>
getUniqueFilenames(ArrayRef<FunctionCoverageMapping> FunctionMappingRecords) {
  std::vector<StringRef> Filenames;
  for (const auto &Function : FunctionMappingRecords)
    for (const auto &Filename : Function.Filenames)
      Filenames.push_back(Filename);
  std::sort(Filenames.begin(), Filenames.end());
  auto Last = std::unique(Filenames.begin(), Filenames.end());
  Filenames.erase(Last, Filenames.end());
  return Filenames;
}

void CodeCoverageTool::error(const Twine &Message, StringRef Whence) {
  errs() << "error: ";
  if (!Whence.empty())
    errs() << Whence << ": ";
  errs() << Message << "\n";
}

ErrorOr<const MemoryBuffer &>
CodeCoverageTool::getSourceFile(StringRef SourceFile) {
  // If we've remapped filenames, look up the real location for this file.
  if (!RemappedFilenames.empty()) {
    auto Loc = RemappedFilenames.find(SourceFile);
    if (Loc != RemappedFilenames.end())
      SourceFile = Loc->second;
  }
  for (const auto &Files : LoadedSourceFiles)
    if (sys::fs::equivalent(SourceFile, Files.first))
      return *Files.second;
  auto Buffer = MemoryBuffer::getFile(SourceFile);
  if (auto EC = Buffer.getError()) {
    error(EC.message(), SourceFile);
    return EC;
  }
  LoadedSourceFiles.push_back(
      std::make_pair(SourceFile, std::move(Buffer.get())));
  return *LoadedSourceFiles.back().second;
}

bool CodeCoverageTool::gatherInterestingFileIDs(
    StringRef SourceFile, const FunctionCoverageMapping &Function,
    SmallSet<unsigned, 8> &InterestingFileIDs) {
  bool Interesting = false;
  for (unsigned I = 0, E = Function.Filenames.size(); I < E; ++I) {
    if (SourceFile == Function.Filenames[I]) {
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
    if (SourceFile == Function.Filenames[I])
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

std::unique_ptr<SourceCoverageView> CodeCoverageTool::createExpansionSubView(
    const CountedRegion &ExpandedRegion,
    const FunctionCoverageMapping &Function) {
  auto SourceBuffer =
      getSourceFile(Function.Filenames[ExpandedRegion.ExpandedFileID]);
  if (!SourceBuffer)
    return nullptr;
  auto RegionManager = llvm::make_unique<SourceCoverageDataManager>();
  for (const auto &CR : Function.CountedRegions) {
    if (CR.FileID == ExpandedRegion.ExpandedFileID)
      RegionManager->insert(CR);
  }
  auto SubView = llvm::make_unique<SourceCoverageView>(SourceBuffer.get(),
                                                       ViewOpts);
  SubView->load(std::move(RegionManager));
  attachExpansionSubViews(*SubView, ExpandedRegion.ExpandedFileID, Function);
  return SubView;
}

void CodeCoverageTool::attachExpansionSubViews(
    SourceCoverageView &View, unsigned ViewFileID,
    const FunctionCoverageMapping &Function) {
  if (!ViewOpts.ShowExpandedRegions)
    return;
  for (const auto &CR : Function.CountedRegions) {
    if (CR.Kind != CounterMappingRegion::ExpansionRegion)
      continue;
    if (CR.FileID != ViewFileID)
      continue;
    auto SubView = createExpansionSubView(CR, Function);
    if (SubView)
      View.addExpansion(CR, std::move(SubView));
  }
}

std::unique_ptr<SourceCoverageView>
CodeCoverageTool::createInstantiationSubView(
    StringRef SourceFile, const FunctionCoverageMapping &Function) {
  auto RegionManager = llvm::make_unique<SourceCoverageDataManager>();
  SmallSet<unsigned, 8> InterestingFileIDs;
  if (!gatherInterestingFileIDs(SourceFile, Function, InterestingFileIDs))
    return nullptr;
  // Get the interesting regions
  for (const auto &CR : Function.CountedRegions) {
    if (InterestingFileIDs.count(CR.FileID))
      RegionManager->insert(CR);
  }

  auto SourceBuffer = getSourceFile(SourceFile);
  if (!SourceBuffer)
    return nullptr;
  auto SubView = llvm::make_unique<SourceCoverageView>(SourceBuffer.get(),
                                                       ViewOpts);
  SubView->load(std::move(RegionManager));
  unsigned MainFileID;
  if (!findMainViewFileID(SourceFile, Function, MainFileID))
    attachExpansionSubViews(*SubView, MainFileID, Function);
  return SubView;
}

std::unique_ptr<SourceCoverageView> CodeCoverageTool::createSourceFileView(
    StringRef SourceFile,
    ArrayRef<FunctionCoverageMapping> FunctionMappingRecords,
    bool UseOnlyRegionsInMainFile) {
  auto RegionManager = llvm::make_unique<SourceCoverageDataManager>();
  FunctionInstantiationSetCollector InstantiationSetCollector;

  auto SourceBuffer = getSourceFile(SourceFile);
  if (!SourceBuffer)
    return nullptr;
  auto View =
      llvm::make_unique<SourceCoverageView>(SourceBuffer.get(), ViewOpts);

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
        RegionManager->insert(CR);
    }
    InstantiationSetCollector.insert(Function, MainFileID);
    attachExpansionSubViews(*View, MainFileID, Function);
  }
  if (RegionManager->getCoverageSegments().empty())
    return nullptr;
  View->load(std::move(RegionManager));
  // Show instantiations
  if (!ViewOpts.ShowFunctionInstantiations)
    return View;
  for (const auto &InstantiationSet : InstantiationSetCollector) {
    if (InstantiationSet.second.size() < 2)
      continue;
    for (auto Function : InstantiationSet.second) {
      unsigned FileID = Function->CountedRegions.front().FileID;
      unsigned Line = 0;
      for (const auto &CR : Function->CountedRegions)
        if (CR.FileID == FileID)
          Line = std::max(CR.LineEnd, Line);
      auto SubView = createInstantiationSubView(SourceFile, *Function);
      if (SubView)
        View->addInstantiation(Function->Name, Line, std::move(SubView));
    }
  }
  return View;
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

  if (CompareFilenamesOnly) {
    auto CoveredFiles = getUniqueFilenames(FunctionMappingRecords);
    for (auto &SF : SourceFiles) {
      StringRef SFBase = sys::path::filename(SF);
      for (const auto &CF : CoveredFiles)
        if (SFBase == sys::path::filename(CF)) {
          RemappedFilenames[CF] = SF;
          SF = CF;
          break;
        }
    }
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
      cl::desc("Treat source files as equivalent to paths in the coverage data "
               "when the file names match, even if the full paths do not"));

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

    for (const auto &File : InputSourceFiles) {
      SmallString<128> Path(File);
      if (std::error_code EC = sys::fs::make_absolute(Path)) {
        errs() << "error: " << File << ": " << EC.message();
        return 1;
      }
      SourceFiles.push_back(Path.str());
    }
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
      auto mainView = createSourceFileView(SourceFile, Function, true);
      if (!mainView) {
        ViewOpts.colored_ostream(outs(), raw_ostream::RED)
            << "warning: Could not read coverage for '" << Function.Name
            << " from " << SourceFile;
        outs() << "\n";
        continue;
      }
      ViewOpts.colored_ostream(outs(), raw_ostream::CYAN)
          << Function.Name << " from " << SourceFile << ":";
      outs() << "\n";
      mainView->render(outs(), /*WholeFile=*/false);
      if (FunctionMappingRecords.size() > 1)
        outs() << "\n";
    }
    return 0;
  }

  // Show files
  bool ShowFilenames = SourceFiles.size() != 1;

  if (SourceFiles.empty())
    // Get the source files from the function coverage mapping
    for (StringRef Filename : getUniqueFilenames(FunctionMappingRecords))
      SourceFiles.push_back(Filename);

  for (const auto &SourceFile : SourceFiles) {
    auto mainView = createSourceFileView(SourceFile, FunctionMappingRecords);
    if (!mainView) {
      ViewOpts.colored_ostream(outs(), raw_ostream::RED)
          << "warning: The file '" << SourceFile << "' isn't covered.";
      outs() << "\n";
      continue;
    }

    if (ShowFilenames) {
      ViewOpts.colored_ostream(outs(), raw_ostream::CYAN) << SourceFile << ":";
      outs() << "\n";
    }
    mainView->render(outs(), /*Wholefile=*/true);
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
