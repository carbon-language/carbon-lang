//===- llvm-cov.cpp - LLVM coverage tool ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// llvm-cov is a command line tools to analyze and report coverage information.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
#include <string>

using namespace llvm;

/// \brief The main entry point for the 'show' subcommand.
int show_main(int argc, const char **argv);

/// \brief The main entry point for the 'report' subcommand.
int report_main(int argc, const char **argv);

/// \brief The main entry point for the 'convert-for-testing' subcommand.
int convert_for_testing_main(int argc, const char **argv);

/// \brief The main entry point for the gcov compatible coverage tool.
int gcov_main(int argc, const char **argv);

int main(int argc, const char **argv) {
  // If argv[0] is or ends with 'gcov', always be gcov compatible
  if (sys::path::stem(argv[0]).endswith_lower("gcov"))
    return gcov_main(argc, argv);

  // Check if we are invoking a specific tool command.
  if (argc > 1) {
    int (*func)(int, const char **) = nullptr;

    StringRef command = argv[1];
    if (command.equals_lower("show"))
      func = show_main;
    else if (command.equals_lower("report"))
      func = report_main;
    else if (command.equals_lower("convert-for-testing"))
      func = convert_for_testing_main;
    else if (command.equals_lower("gcov"))
      func = gcov_main;

    if (func) {
      std::string Invocation = std::string(argv[0]) + " " + argv[1];
      argv[1] = Invocation.c_str();
      return func(argc - 1, argv + 1);
    }
  }

  // Give a warning and fall back to gcov
  errs().changeColor(raw_ostream::RED);
  errs() << "warning:";
  // Assume that argv[1] wasn't a command when it stats with a '-' or is a
  // filename (i.e. contains a '.')
  if (argc > 1 && !StringRef(argv[1]).startswith("-") &&
      StringRef(argv[1]).find(".") == StringRef::npos)
    errs() << " Unrecognized command '" << argv[1] << "'.";
  errs() << " Using the gcov compatible mode "
            "(this behaviour may be dropped in the future).";
  errs().resetColor();
  errs() << "\n";

  return gcov_main(argc, argv);
}
