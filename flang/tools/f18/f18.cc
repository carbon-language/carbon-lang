// Temporary Fortran front end driver main program for development scaffolding.

#include "../../lib/parser/basic-parsers.h"
#include "../../lib/parser/char-buffer.h"
#include "../../lib/parser/cooked-chars.h"
#include "../../lib/parser/grammar.h"
#include "../../lib/parser/idioms.h"
#include "../../lib/parser/message.h"
#include "../../lib/parser/parse-tree.h"
#include "../../lib/parser/prescan.h"
#include "../../lib/parser/provenance.h"
#include "../../lib/parser/source.h"
#include "../../lib/parser/user-state.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {

std::list<std::string> argList(int argc, char *const argv[]) {
  std::list<std::string> result;
  for (int j = 0; j < argc; ++j) {
    result.emplace_back(argv[j]);
  }
  return result;
}

}  // namespace

namespace Fortran {
namespace parser {
constexpr auto grammar = program;
}  // namespace parser
}  // namespace Fortran
using Fortran::parser::grammar;

int main(int argc, char *const argv[]) {

  auto args = argList(argc, argv);
  std::string progName{args.front()};
  args.pop_front();

  bool dumpCookedChars{false};
  bool fixedForm{false};
  bool backslashEscapes{true};
  bool standard{false};
  bool enableOldDebugLines{false};
  int columns{72};

  while (!args.empty()) {
    if (args.front().empty()) {
      args.pop_front();
    } else if (args.front().at(0) != '-' || args.front() == "-") {
      break;
    } else if (args.front() == "--") {
      args.pop_front();
      break;
    } else {
      std::string flag{std::move(args.front())};
      args.pop_front();
      if (flag == "-Mfixed") {
        fixedForm = true;
      } else if (flag == "-Mbackslash") {
        backslashEscapes = false;
      } else if (flag == "-Mstandard") {
        standard = false;
      } else if (flag == "-Mextend") {
        columns = 132;
      } else if (flag == "-fdebug-dump-cooked-chars") {
        dumpCookedChars = true;
      } else if (flag == "-ed") {
        enableOldDebugLines = true;
      } else {
        std::cerr << "unknown flag: '" << flag << "'\n";
        return 1;
      }
    }
  }

  std::string path{"-"};
  if (!args.empty()) {
    path = std::move(args.front());
    args.pop_front();
    if (!args.empty()) {
      std::cerr << "multiple input files\n";
      ;
      return 1;
    }
  }

  Fortran::parser::SourceFile sourceFile;
  std::stringstream error;
  if (!sourceFile.Open(path, &error)) {
    std::cerr << error.str() << '\n';
    return 1;
  }

  Fortran::parser::AllSources allSources{sourceFile};
  Fortran::parser::Messages messages;
  Fortran::parser::Prescanner prescanner{messages, allSources};
  Fortran::parser::CookedSource cooked{
      prescanner.set_fixedForm(fixedForm)
          .set_enableBackslashEscapesInCharLiterals(backslashEscapes)
          .set_fixedFormColumnLimit(columns)
          .set_enableOldDebugLines(enableOldDebugLines)
          .Prescan()};
  messages.Emit(std::cerr, allSources);
  if (prescanner.anyFatalErrors()) {
    return 1;
  }
  columns = std::numeric_limits<int>::max();

  Fortran::parser::ParseState state{cooked};
  state.set_inFixedForm(fixedForm);
  state.set_enableBackslashEscapesInCharLiterals(backslashEscapes);
  state.set_strictConformance(standard);
  state.set_columns(columns);
  state.set_enableOldDebugLines(enableOldDebugLines);
  state.PushContext("source file '"s + path + "'");
  Fortran::parser::UserState ustate;
  state.set_userState(&ustate);

  if (dumpCookedChars) {
    while (std::optional<char> och{
        Fortran::parser::cookedNextChar.Parse(&state)}) {
      std::cout << *och;
    }
    return 0;
  }

  std::optional<typename decltype(grammar)::resultType> result;
#if 0
  for (int j = 0; j < 1000; ++j) {
    Fortran::parser::ParseState state1{state};
    result = grammar.Parse(&state1);
    if (!result) {
      std::cerr << "demo FAIL in timing loop\n";
      break;
    }
  }
#endif
  result = grammar.Parse(&state);
  if (result.has_value() && !state.anyErrorRecovery()) {
    std::cout << "demo PASS\n" << *result << '\n';
  } else {
    std::cerr << "demo FAIL\n";
    allSources.Identify(std::cerr, state.GetProvenance(), "   ");
    state.messages()->Emit(std::cerr, allSources);
    return EXIT_FAILURE;
  }
}
