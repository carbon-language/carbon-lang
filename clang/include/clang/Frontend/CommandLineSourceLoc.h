//===--- CommandLineSourceLoc.h - Parsing for source locations-*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Command line parsing for source locations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_COMMANDLINESOURCELOC_H
#define LLVM_CLANG_FRONTEND_COMMANDLINESOURCELOC_H

#include "llvm/Support/CommandLine.h"
#include <cstdio>

namespace clang {

/// \brief A source location that has been parsed on the command line.
struct ParsedSourceLocation {
  std::string FileName;
  unsigned Line;
  unsigned Column;
};

}

namespace llvm {
  namespace cl {
    /// \brief Command-line option parser that parses source locations.
    ///
    /// Source locations are of the form filename:line:column.
    template<>
    class parser<clang::ParsedSourceLocation>
      : public basic_parser<clang::ParsedSourceLocation> {
    public:
      bool parse(Option &O, const char *ArgName,
                 const std::string &ArgValue,
                 clang::ParsedSourceLocation &Val);
    };

    bool
    parser<clang::ParsedSourceLocation>::
    parse(Option &O, const char *ArgName, const std::string &ArgValue,
          clang::ParsedSourceLocation &Val) {
      using namespace clang;

      const char *ExpectedFormat
        = "source location must be of the form filename:line:column";
      std::string::size_type SecondColon = ArgValue.rfind(':');
      if (SecondColon == std::string::npos) {
        std::fprintf(stderr, "%s\n", ExpectedFormat);
        return true;
      }
      char *EndPtr;
      long Column
        = std::strtol(ArgValue.c_str() + SecondColon + 1, &EndPtr, 10);
      if (EndPtr != ArgValue.c_str() + ArgValue.size()) {
        std::fprintf(stderr, "%s\n", ExpectedFormat);
        return true;
      }

      std::string::size_type FirstColon = ArgValue.rfind(':', SecondColon-1);
      if (FirstColon == std::string::npos) {
        std::fprintf(stderr, "%s\n", ExpectedFormat);
        return true;
      }
      long Line = std::strtol(ArgValue.c_str() + FirstColon + 1, &EndPtr, 10);
      if (EndPtr != ArgValue.c_str() + SecondColon) {
        std::fprintf(stderr, "%s\n", ExpectedFormat);
        return true;
      }

      Val.FileName = ArgValue.substr(0, FirstColon);
      Val.Line = Line;
      Val.Column = Column;
      return false;
    }
  }
}

#endif
