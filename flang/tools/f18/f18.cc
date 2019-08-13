// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Temporary Fortran front end driver main program for development scaffolding.

#include "../../lib/common/default-kinds.h"
#include "../../lib/evaluate/expression.h"
#include "../../lib/parser/characters.h"
#include "../../lib/parser/dump-parse-tree.h"
#include "../../lib/parser/features.h"
#include "../../lib/parser/message.h"
#include "../../lib/parser/parse-tree-visitor.h"
#include "../../lib/parser/parse-tree.h"
#include "../../lib/parser/parsing.h"
#include "../../lib/parser/provenance.h"
#include "../../lib/parser/unparse.h"
#include "../../lib/semantics/expression.h"
#include "../../lib/semantics/semantics.h"
#include "../../lib/semantics/unparse-with-symbols.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <stdlib.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static std::list<std::string> argList(int argc, char *const argv[]) {
  std::list<std::string> result;
  for (int j = 0; j < argc; ++j) {
    result.emplace_back(argv[j]);
  }
  return result;
}

struct MeasurementVisitor {
  template<typename A> bool Pre(const A &) { return true; }
  template<typename A> void Post(const A &) {
    ++objects;
    bytes += sizeof(A);
  }
  size_t objects{0}, bytes{0};
};

void MeasureParseTree(const Fortran::parser::Program &program) {
  MeasurementVisitor visitor;
  Fortran::parser::Walk(program, visitor);
  std::cout << "Parse tree comprises " << visitor.objects
            << " objects and occupies " << visitor.bytes << " total bytes.\n";
}

std::vector<std::string> filesToDelete;

void CleanUpAtExit() {
  for (const auto &path : filesToDelete) {
    if (!path.empty()) {
      unlink(path.data());
    }
  }
}

struct DriverOptions {
  DriverOptions() {}
  bool verbose{false};  // -v
  bool compileOnly{false};  // -c
  std::string outputPath;  // -o path
  std::vector<std::string> searchDirectories{"."s};  // -I dir
  std::string moduleDirectory{"."s};  // -module dir
  std::string moduleFileSuffix{".mod"};  // -moduleSuffix suff
  bool forcedForm{false};  // -Mfixed or -Mfree appeared
  bool warnOnNonstandardUsage{false};  // -Mstandard
  bool warningsAreErrors{false};  // -Werror
  Fortran::parser::Encoding encoding{Fortran::parser::Encoding::UTF_8};
  bool parseOnly{false};
  bool dumpProvenance{false};
  bool dumpCookedChars{false};
  bool dumpUnparse{false};
  bool dumpUnparseWithSymbols{false};
  bool dumpParseTree{false};
  bool dumpSymbols{false};
  bool debugResolveNames{false};
  bool debugSemantics{false};
  bool measureTree{false};
  bool unparseTypedExprsToPGF90{false};
  std::vector<std::string> pgf90Args;
  const char *prefix{nullptr};
};

bool ParentProcess() {
  if (fork() == 0) {
    return false;  // in child process
  }
  int childStat{0};
  wait(&childStat);
  if (!WIFEXITED(childStat) || WEXITSTATUS(childStat) != 0) {
    exit(EXIT_FAILURE);
  }
  return true;
}

void Exec(std::vector<char *> &argv, bool verbose = false) {
  if (verbose) {
    for (size_t j{0}; j < argv.size(); ++j) {
      std::cerr << (j > 0 ? " " : "") << argv[j];
    }
    std::cerr << '\n';
  }
  argv.push_back(nullptr);
  execvp(argv[0], &argv[0]);
  std::cerr << "execvp(" << argv[0] << ") failed: " << std::strerror(errno)
            << '\n';
  exit(EXIT_FAILURE);
}

void RunOtherCompiler(DriverOptions &driver, char *source, char *relo) {
  std::vector<char *> argv;
  for (size_t j{0}; j < driver.pgf90Args.size(); ++j) {
    argv.push_back(driver.pgf90Args[j].data());
  }
  char dashC[3] = "-c", dashO[3] = "-o";
  argv.push_back(dashC);
  argv.push_back(dashO);
  argv.push_back(relo);
  argv.push_back(source);
  Exec(argv, driver.verbose);
}

std::string RelocatableName(const DriverOptions &driver, std::string path) {
  if (driver.compileOnly && !driver.outputPath.empty()) {
    return driver.outputPath;
  }
  std::string base{path};
  auto slash{base.rfind("/")};
  if (slash != std::string::npos) {
    base = base.substr(slash + 1);
  }
  std::string relo{base};
  auto dot{base.rfind(".")};
  if (dot != std::string::npos) {
    relo = base.substr(0, dot);
  }
  relo += ".o";
  return relo;
}

int exitStatus{EXIT_SUCCESS};

std::string CompileFortran(std::string path, Fortran::parser::Options options,
    DriverOptions &driver,
    const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds) {
  Fortran::parser::AllSources allSources;
  allSources.set_encoding(driver.encoding);
  Fortran::semantics::SemanticsContext semanticsContext{
      defaultKinds, options.features, allSources};
  semanticsContext.set_moduleDirectory(driver.moduleDirectory)
      .set_moduleFileSuffix(driver.moduleFileSuffix)
      .set_searchDirectories(driver.searchDirectories)
      .set_warnOnNonstandardUsage(driver.warnOnNonstandardUsage)
      .set_warningsAreErrors(driver.warningsAreErrors);
  if (!driver.forcedForm) {
    auto dot{path.rfind(".")};
    if (dot != std::string::npos) {
      std::string suffix{path.substr(dot + 1)};
      options.isFixedForm = suffix == "f" || suffix == "F" || suffix == "ff";
    }
  }
  options.searchDirectories = driver.searchDirectories;
  Fortran::parser::Parsing parsing{semanticsContext.allSources()};
  parsing.Prescan(path, options);
  if (!parsing.messages().empty() &&
      (driver.warningsAreErrors || parsing.messages().AnyFatalError())) {
    std::cerr << driver.prefix << "could not scan " << path << '\n';
    parsing.messages().Emit(std::cerr, parsing.cooked());
    exitStatus = EXIT_FAILURE;
    return {};
  }
  if (driver.dumpProvenance) {
    parsing.DumpProvenance(std::cout);
    return {};
  }
  if (driver.dumpCookedChars) {
    parsing.messages().Emit(std::cerr, parsing.cooked());
    parsing.DumpCookedChars(std::cout);
    return {};
  }
  parsing.Parse(&std::cout);
  if (options.instrumentedParse) {
    parsing.DumpParsingLog(std::cout);
    return {};
  }
  parsing.ClearLog();
  parsing.messages().Emit(std::cerr, parsing.cooked());
  if (!parsing.consumedWholeFile()) {
    parsing.EmitMessage(
        std::cerr, parsing.finalRestingPlace(), "parser FAIL (final position)");
    exitStatus = EXIT_FAILURE;
    return {};
  }
  if ((!parsing.messages().empty() &&
          (driver.warningsAreErrors || parsing.messages().AnyFatalError())) ||
      !parsing.parseTree().has_value()) {
    std::cerr << driver.prefix << "could not parse " << path << '\n';
    exitStatus = EXIT_FAILURE;
    return {};
  }
  auto &parseTree{*parsing.parseTree()};
  if (driver.measureTree) {
    MeasureParseTree(parseTree);
  }
  // TODO: Change this predicate to just "if (!driver.debugNoSemantics)"
  if (driver.debugSemantics || driver.debugResolveNames || driver.dumpSymbols ||
      driver.dumpUnparseWithSymbols) {
    Fortran::semantics::Semantics semantics{
        semanticsContext, parseTree, parsing.cooked()};
    semantics.Perform();
    semantics.EmitMessages(std::cerr);
    if (driver.dumpSymbols) {
      semantics.DumpSymbols(std::cout);
    }
    if (semantics.AnyFatalError()) {
      std::cerr << driver.prefix << "semantic errors in " << path << '\n';
      exitStatus = EXIT_FAILURE;
      if (driver.dumpParseTree) {
        Fortran::parser::DumpTree(std::cout, parseTree);
      }
      return {};
    }
    if (driver.dumpUnparseWithSymbols) {
      Fortran::semantics::UnparseWithSymbols(
          std::cout, parseTree, driver.encoding);
      return {};
    }
  }
  if (driver.dumpParseTree) {
    Fortran::parser::DumpTree(std::cout, parseTree);
  }

  Fortran::parser::TypedExprAsFortran unparseExpression{
      [](std::ostream &o, const Fortran::evaluate::GenericExprWrapper &x) {
        if (x.v.has_value()) {
          o << *x.v;
        } else {
          o << "(bad expression)";
        }
      }};

  if (driver.dumpUnparse) {
    Unparse(std::cout, parseTree, driver.encoding, true /*capitalize*/,
        options.features.IsEnabled(
            Fortran::parser::LanguageFeature::BackslashEscapes),
        nullptr /* action before each statement */, &unparseExpression);
    return {};
  }
  if (driver.parseOnly) {
    return {};
  }

  std::string relo{RelocatableName(driver, path)};

  char tmpSourcePath[32];
  std::snprintf(tmpSourcePath, sizeof tmpSourcePath, "/tmp/f18-%lx.f90",
      static_cast<unsigned long>(getpid()));
  {
    std::ofstream tmpSource;
    tmpSource.open(tmpSourcePath);
    Fortran::evaluate::formatForPGF90 = true;
    Unparse(tmpSource, parseTree, driver.encoding, true /*capitalize*/,
        options.features.IsEnabled(
            Fortran::parser::LanguageFeature::BackslashEscapes),
        nullptr /* action before each statement */,
        driver.unparseTypedExprsToPGF90 ? &unparseExpression : nullptr);
    Fortran::evaluate::formatForPGF90 = false;
  }

  if (ParentProcess()) {
    filesToDelete.push_back(tmpSourcePath);
    if (!driver.compileOnly && driver.outputPath.empty()) {
      filesToDelete.push_back(relo);
    }
    return relo;
  }
  RunOtherCompiler(driver, tmpSourcePath, relo.data());
  return {};
}

std::string CompileOtherLanguage(std::string path, DriverOptions &driver) {
  std::string relo{RelocatableName(driver, path)};
  if (ParentProcess()) {
    if (!driver.compileOnly && driver.outputPath.empty()) {
      filesToDelete.push_back(relo);
    }
    return relo;
  }
  RunOtherCompiler(driver, path.data(), relo.data());
  return {};
}

void Link(std::vector<std::string> &relocatables, DriverOptions &driver) {
  if (!ParentProcess()) {
    std::vector<char *> argv;
    for (size_t j{0}; j < driver.pgf90Args.size(); ++j) {
      argv.push_back(driver.pgf90Args[j].data());
    }
    for (auto &relo : relocatables) {
      argv.push_back(relo.data());
    }
    if (!driver.outputPath.empty()) {
      char dashO[3] = "-o";
      argv.push_back(dashO);
      argv.push_back(driver.outputPath.data());
    }
    Exec(argv, driver.verbose);
  }
}

int main(int argc, char *const argv[]) {

  atexit(CleanUpAtExit);

  DriverOptions driver;
  const char *pgf90{getenv("F18_FC")};
  driver.pgf90Args.push_back(pgf90 ? pgf90 : "pgf90");
  bool isPGF90{driver.pgf90Args.back().rfind("pgf90") != std::string::npos};

  std::list<std::string> args{argList(argc, argv)};
  std::string prefix{args.front()};
  args.pop_front();
  prefix += ": ";
  driver.prefix = prefix.data();

  Fortran::parser::Options options;
  options.predefinitions.emplace_back("__F18", "1");
  options.predefinitions.emplace_back("__F18_MAJOR__", "1");
  options.predefinitions.emplace_back("__F18_MINOR__", "1");
  options.predefinitions.emplace_back("__F18_PATCHLEVEL__", "1");
#if __x86_64__
  options.predefinitions.emplace_back("__x86_64__", "1");
#endif

  Fortran::common::IntrinsicTypeDefaultKinds defaultKinds;

  std::vector<std::string> fortranSources, otherSources, relocatables;
  bool anyFiles{false};

  while (!args.empty()) {
    std::string arg{std::move(args.front())};
    args.pop_front();
    if (arg.empty()) {
    } else if (arg.at(0) != '-') {
      anyFiles = true;
      auto dot{arg.rfind(".")};
      if (dot == std::string::npos) {
        driver.pgf90Args.push_back(arg);
      } else {
        std::string suffix{arg.substr(dot + 1)};
        if (suffix == "f" || suffix == "F" || suffix == "ff" ||
            suffix == "f90" || suffix == "F90" || suffix == "ff90" ||
            suffix == "f95" || suffix == "F95" || suffix == "ff95" ||
            suffix == "cuf" || suffix == "CUF" || suffix == "f18" ||
            suffix == "F18" || suffix == "ff18") {
          fortranSources.push_back(arg);
        } else if (suffix == "o" || suffix == "a") {
          relocatables.push_back(arg);
        } else {
          otherSources.push_back(arg);
        }
      }
    } else if (arg == "-") {
      fortranSources.push_back("-");
    } else if (arg == "--") {
      while (!args.empty()) {
        fortranSources.emplace_back(std::move(args.front()));
        args.pop_front();
      }
      break;
    } else if (arg == "-Mfixed") {
      driver.forcedForm = true;
      options.isFixedForm = true;
    } else if (arg == "-Mfree") {
      driver.forcedForm = true;
      options.isFixedForm = false;
    } else if (arg == "-Mextend") {
      options.fixedFormColumns = 132;
    } else if (arg == "-Mbackslash") {
      options.features.Enable(
          Fortran::parser::LanguageFeature::BackslashEscapes, false);
    } else if (arg == "-Mnobackslash") {
      options.features.Enable(
          Fortran::parser::LanguageFeature::BackslashEscapes, true);
    } else if (arg == "-Mstandard") {
      driver.warnOnNonstandardUsage = true;
    } else if (arg == "-fopenmp") {
      options.features.Enable(Fortran::parser::LanguageFeature::OpenMP);
      options.predefinitions.emplace_back("_OPENMP", "201511");
    } else if (arg == "-Werror") {
      driver.warningsAreErrors = true;
    } else if (arg == "-ed") {
      options.features.Enable(Fortran::parser::LanguageFeature::OldDebugLines);
    } else if (arg == "-E") {
      driver.dumpCookedChars = true;
    } else if (arg == "-fbackslash" || arg == "-fno-backslash") {
      options.features.Enable(
          Fortran::parser::LanguageFeature::BackslashEscapes,
          arg == "-fbackslash");
    } else if (arg == "-fxor-operator" || arg == "-fno-xor-operator") {
      options.features.Enable(Fortran::parser::LanguageFeature::XOROperator,
          arg == "-fxor-operator");
    } else if (arg == "-fdebug-dump-provenance") {
      driver.dumpProvenance = true;
    } else if (arg == "-fdebug-dump-parse-tree") {
      driver.dumpParseTree = true;
    } else if (arg == "-fdebug-dump-symbols") {
      driver.dumpSymbols = true;
    } else if (arg == "-fdebug-resolve-names") {
      driver.debugResolveNames = true;
    } else if (arg == "-fdebug-measure-parse-tree") {
      driver.measureTree = true;
    } else if (arg == "-fdebug-instrumented-parse") {
      options.instrumentedParse = true;
    } else if (arg == "-fdebug-semantics") {
      // TODO: Enable by default once basic tests pass
      driver.debugSemantics = true;
    } else if (arg == "-funparse") {
      driver.dumpUnparse = true;
    } else if (arg == "-funparse-with-symbols") {
      driver.dumpUnparseWithSymbols = true;
    } else if (arg == "-funparse-typed-exprs-to-pgf90") {
      driver.unparseTypedExprsToPGF90 = true;
    } else if (arg == "-fparse-only") {
      driver.parseOnly = true;
    } else if (arg == "-c") {
      driver.compileOnly = true;
    } else if (arg == "-o") {
      driver.outputPath = args.front();
      args.pop_front();
    } else if (arg.substr(0, 2) == "-D") {
      auto eq{arg.find('=')};
      if (eq == std::string::npos) {
        options.predefinitions.emplace_back(arg.substr(2), "1");
      } else {
        options.predefinitions.emplace_back(
            arg.substr(2, eq - 2), arg.substr(eq + 1));
      }
    } else if (arg.substr(0, 2) == "-U") {
      options.predefinitions.emplace_back(
          arg.substr(2), std::optional<std::string>{});
    } else if (arg == "-r8" || arg == "-fdefault-real-8") {
      defaultKinds.set_defaultRealKind(8);
    } else if (arg == "-i8" || arg == "-fdefault-integer-8") {
      defaultKinds.set_defaultIntegerKind(8);
    } else if (arg == "-module") {
      driver.moduleDirectory = args.front();
      args.pop_front();
    } else if (arg == "-module-suffix") {
      driver.moduleFileSuffix = args.front();
      args.pop_front();
    } else if (arg == "-intrinsic-module-directory") {
      driver.searchDirectories.push_back(args.front());
      args.pop_front();
    } else if (arg == "-futf-8") {
      driver.encoding = Fortran::parser::Encoding::UTF_8;
    } else if (arg == "-flatin") {
      driver.encoding = Fortran::parser::Encoding::LATIN_1;
    } else if (arg == "-help" || arg == "--help" || arg == "-?") {
      std::cerr
          << "f18 options:\n"
          << "  -Mfixed | -Mfree     force the source form\n"
          << "  -Mextend             132-column fixed form\n"
          << "  -f[no-]backslash     enable[disable] \\escapes in literals\n"
          << "  -M[no]backslash      disable[enable] \\escapes in literals\n"
          << "  -Mstandard           enable conformance warnings\n"
          << "  -r8 | -fdefault-real-8 | -i8 | -fdefault-integer-8  "
             "change default kinds of intrinsic types\n"
          << "  -Werror              treat warnings as errors\n"
          << "  -ed                  enable fixed form D lines\n"
          << "  -E                   prescan & preprocess only\n"
          << "  -module dir          module output directory (default .)\n"
          << "  -flatin              interpret source as Latin-1 (ISO 8859-1) "
             "rather than UTF-8\n"
          << "  -fparse-only         parse only, no output except messages\n"
          << "  -funparse            parse & reformat only, no code "
             "generation\n"
          << "  -funparse-with-symbols  parse, resolve symbols, and unparse\n"
          << "  -fdebug-measure-parse-tree\n"
          << "  -fdebug-dump-provenance\n"
          << "  -fdebug-dump-parse-tree\n"
          << "  -fdebug-dump-symbols\n"
          << "  -fdebug-resolve-names\n"
          << "  -fdebug-instrumented-parse\n"
          << "  -fdebug-semantics    perform semantic checks\n"
          << "  -v -c -o -I -D -U    have their usual meanings\n"
          << "  -help                print this again\n"
          << "Other options are passed through to the compiler.\n";
      return exitStatus;
    } else if (arg == "-V") {
      std::cerr << "\nf18 compiler (under development)\n";
      return exitStatus;
    } else {
      driver.pgf90Args.push_back(arg);
      if (arg == "-v") {
        driver.verbose = true;
      } else if (arg == "-I") {
        driver.pgf90Args.push_back(args.front());
        driver.searchDirectories.push_back(args.front());
        args.pop_front();
      } else if (arg.substr(0, 2) == "-I") {
        driver.searchDirectories.push_back(arg.substr(2));
      }
    }
  }

  if (driver.warnOnNonstandardUsage) {
    options.features.WarnOnAllNonstandard();
  }
  if (options.features.IsEnabled(Fortran::parser::LanguageFeature::OpenMP)) {
    driver.pgf90Args.push_back("-mp");
  }
  if (isPGF90) {
    if (!options.features.IsEnabled(
            Fortran::parser::LanguageFeature::BackslashEscapes)) {
      driver.pgf90Args.push_back(
          "-Mbackslash");  // yes, this *disables* them in pgf90
    }
  } else {
    // TODO: equivalents for other Fortran compilers
  }

  if (!anyFiles) {
    driver.measureTree = true;
    driver.dumpUnparse = true;
    CompileFortran("-", options, driver, defaultKinds);
    return exitStatus;
  }
  for (const auto &path : fortranSources) {
    std::string relo{CompileFortran(path, options, driver, defaultKinds)};
    if (!driver.compileOnly && !relo.empty()) {
      relocatables.push_back(relo);
    }
  }
  for (const auto &path : otherSources) {
    std::string relo{CompileOtherLanguage(path, driver)};
    if (!driver.compileOnly && !relo.empty()) {
      relocatables.push_back(relo);
    }
  }
  if (!relocatables.empty()) {
    Link(relocatables, driver);
  }
  return exitStatus;
}
