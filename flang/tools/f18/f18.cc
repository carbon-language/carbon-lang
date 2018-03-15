// Temporary Fortran front end driver main program for development scaffolding.

#include "../../lib/parser/characters.h"
#include "../../lib/parser/message.h"
#include "../../lib/parser/parse-tree.h"
#include "../../lib/parser/parse-tree-visitor.h"
#include "../../lib/parser/parsing.h"
#include "../../lib/parser/provenance.h"
#include "../../lib/parser/unparse.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

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
            << " objects and occupies " << visitor.bytes
            << " total bytes.\n";
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
  bool forcedForm{false};  // -Mfixed or -Mfree appeared
  std::vector<std::string> searchPath;  // -I path
  Fortran::parser::Encoding encoding{Fortran::parser::Encoding::UTF8};
  bool parseOnly{false};
  bool dumpProvenance{false};
  bool dumpCookedChars{false};
  bool dumpUnparse{false};
  bool measureTree{false};
  std::vector<std::string> pgf90Args;
  const char *prefix{nullptr};
};

bool ParentProcess() {
  if (fork() == 0) {
    return false;  // in child process
  }
  int childStat{0};
  wait(&childStat);
  if (!WIFEXITED(childStat) ||
      WEXITSTATUS(childStat) != 0) {
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
  std::cerr << "execvp(" << argv[0] << ") failed: "
            << std::strerror(errno) << '\n';
  exit(EXIT_FAILURE);
}

std::string Compile(std::string path, Fortran::parser::Options options,
                    DriverOptions &driver) {
  if (!driver.forcedForm) {
    auto dot = path.rfind(".");
    if (dot != std::string::npos) {
      std::string suffix{path.substr(dot + 1, std::string::npos)};
      options.isFixedForm = suffix == "f" || suffix == "F";
    }
  }
  Fortran::parser::Parsing parsing;
  for (const auto &searchPath : driver.searchPath) {
    parsing.PushSearchPathDirectory(searchPath);
  }
  if (!parsing.Prescan(path, options)) {
    parsing.messages().Emit(std::cerr, driver.prefix);
    exit(EXIT_FAILURE);
  }
  if (driver.dumpProvenance) {
    parsing.DumpProvenance(std::cout);
    return {};
  }
  if (driver.dumpCookedChars) {
    parsing.DumpCookedChars(std::cout);
    return {};
  }
  if (!parsing.Parse()) {
    if (!parsing.consumedWholeFile()) {
      std::cerr << "f18 FAIL; final position: ";
      parsing.Identify(std::cerr, parsing.finalRestingPlace(), "   ");
    }
    std::cerr << driver.prefix << "could not parse " << path << '\n';
    parsing.messages().Emit(std::cerr, driver.prefix);
    exit(EXIT_FAILURE);
  }
  if (driver.measureTree) {
    MeasureParseTree(parsing.parseTree());
  }
  if (driver.dumpUnparse) {
    Unparse(std::cout, parsing.parseTree(), driver.encoding,
            true /*capitalize*/);
    return {};
  }

  parsing.messages().Emit(std::cerr, driver.prefix);
  if (driver.parseOnly) {
    return {};
  }

  std::string relo;
  bool deleteReloAfterLink{false};
  if (driver.compileOnly && !driver.outputPath.empty()) {
    relo = driver.outputPath;
  } else {
    std::string base{path};
    auto slash = base.rfind("/");
    if (slash != std::string::npos) {
      base = base.substr(slash + 1);
    }
    auto dot = base.rfind(".");
    if (dot == std::string::npos) {
      relo = base;
    } else {
      relo = base.substr(0, dot);
    }
    relo += ".o";
    deleteReloAfterLink = !driver.compileOnly;
  }

  char tmpSourcePath[32];
  std::snprintf(tmpSourcePath, sizeof tmpSourcePath, "/tmp/f18-%lx.f90",
                static_cast<unsigned long>(getpid()));
  { std::ofstream tmpSource;
    tmpSource.open(tmpSourcePath);
    Unparse(tmpSource, parsing.parseTree(), driver.encoding);
  }

  if (ParentProcess()) {
    filesToDelete.push_back(tmpSourcePath);
    if (deleteReloAfterLink) {
      filesToDelete.push_back(relo);
    }
    return relo;
  }

  std::vector<char *> argv;
  for (size_t j{0}; j < driver.pgf90Args.size(); ++j) {
    argv.push_back(driver.pgf90Args[j].data());
  }
  char dashC[3] = "-c", dashO[3] = "-o";
  argv.push_back(dashC);
  argv.push_back(dashO);
  argv.push_back(relo.data());
  argv.push_back(tmpSourcePath);
  Exec(argv, driver.verbose);
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

  std::list<std::string> args{argList(argc, argv)};
  std::string prefix{args.front()};
  args.pop_front();
  prefix += ": ";
  driver.prefix = prefix.data();

  Fortran::parser::Options options;
  std::vector<std::string> sources, relocatables;
  bool anyFiles{false};

  while (!args.empty()) {
    std::string arg{std::move(args.front())};
    args.pop_front();
    if (arg.empty()) {
    } else if (arg.at(0) != '-') {
      anyFiles = true;
      auto dot = arg.rfind(".");
      if (dot == std::string::npos) {
        driver.pgf90Args.push_back(arg);
      } else {
        std::string suffix{arg.substr(dot + 1, std::string::npos)};
        if (suffix == "f" || suffix == "F" ||
            suffix == "f90" || suffix == "F90" ||
            suffix == "cuf" || suffix == "CUF" ||
            suffix == "f18" || suffix == "F18") {
          sources.push_back(arg);
        } else if (suffix == "o" || suffix == "a") {
          relocatables.push_back(arg);
        } else {
          driver.pgf90Args.push_back(arg);
        }
      }
    } else if (arg == "-") {
      sources.push_back("-");
    } else if (arg == "--") {
      while (!args.empty()) {
        sources.emplace_back(std::move(args.front()));
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
      options.enableBackslashEscapes = false;
    } else if (arg == "-Mnobackslash") {
      options.enableBackslashEscapes = true;
    } else if (arg == "-Mstandard") {
      options.isStrictlyStandard = true;
    } else if (arg == "-ed") {
      options.enableOldDebugLines = true;
    } else if (arg == "-E") {
      driver.dumpCookedChars = true;
    } else if (arg == "-fdebug-dump-provenance") {
      driver.dumpProvenance = true;
    } else if (arg == "-fdebug-measure-parse-tree") {
      driver.measureTree = true;
    } else if (arg == "-funparse") {
      driver.dumpUnparse = true;
    } else if (arg == "-fparse-only") {
      driver.parseOnly = true;
    } else if (arg == "-c") {
      driver.compileOnly = true;
    } else if (arg == "-o") {
      driver.outputPath = args.front();
      args.pop_front();
    } else if (arg == "-help" || arg == "--help" || arg == "-?") {
      std::cerr << "f18 options:\n"
        << "  -Mfixed | -Mfree     force the source form\n"
        << "  -Mextend             132-column fixed form\n"
        << "  -M[no]backslash      disable[enable] \\escapes in literals\n"
        << "  -Mstandard           enable conformance warnings\n"
        << "  -Mx,125,4            set bit 2 in xflag[125] (all Kanji mode)\n"
        << "  -ed                  enable fixed form D lines\n"
        << "  -E                   prescan & preprocess only\n"
        << "  -fparse-only         parse only, no output except messages\n"
        << "  -funparse            parse & reformat only, no code generation\n"
        << "  -fdebug-measure-parse-tree\n"
        << "  -fdebug-dump-provenance\n"
        << "  -v, -c, -o, -I       have their usual meanings\n"
        << "  -help                print this again\n"
        << "Other options are passed through to the compiler.\n";
      return EXIT_SUCCESS;
    } else {
      driver.pgf90Args.push_back(arg);
      if (arg == "-v") {
        driver.verbose = true;
      } else if (arg == "-I") {
        driver.pgf90Args.push_back(args.front());
        driver.searchPath.push_back(args.front());
        args.pop_front();
      } else if (arg.substr(0, 2) == "-I") {
        driver.searchPath.push_back(arg.substr(2, std::string::npos));
      } else if (arg == "-Mx,125,4") {  // PGI "all Kanji" mode
        options.encoding = Fortran::parser::Encoding::EUC_JP;
      }
    }
  }
  driver.encoding = options.encoding;

  if (!anyFiles) {
    driver.measureTree = true;
    driver.dumpUnparse = true;
    Compile("-", options, driver);
    return EXIT_SUCCESS;
  }
  for (const auto &path : sources) {
    std::string relo{Compile(path, options, driver)};
    if (!driver.compileOnly && !relo.empty()) {
      relocatables.push_back(relo);
    }
  }
  if (!relocatables.empty()) {
    Link(relocatables, driver);
  }
  return EXIT_SUCCESS;
}
