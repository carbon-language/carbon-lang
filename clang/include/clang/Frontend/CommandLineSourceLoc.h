
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
      inline bool parse(Option &O, StringRef ArgName, StringRef ArgValue,
                 clang::ParsedSourceLocation &Val);
    };

    bool
    parser<clang::ParsedSourceLocation>::
    parse(Option &O, StringRef ArgName, StringRef ArgValue,
          clang::ParsedSourceLocation &Val) {
      using namespace clang;

      const char *ExpectedFormat
        = "source location must be of the form filename:line:column";
      StringRef::size_type SecondColon = ArgValue.rfind(':');
      if (SecondColon == std::string::npos) {
        std::fprintf(stderr, "%s\n", ExpectedFormat);
        return true;
      }

      unsigned Column;
      if (ArgValue.substr(SecondColon + 1).getAsInteger(10, Column)) {
        std::fprintf(stderr, "%s\n", ExpectedFormat);
        return true;
      }
      ArgValue = ArgValue.substr(0, SecondColon);

      StringRef::size_type FirstColon = ArgValue.rfind(':');
      if (FirstColon == std::string::npos) {
        std::fprintf(stderr, "%s\n", ExpectedFormat);
        return true;
      }
      unsigned Line;
      if (ArgValue.substr(FirstColon + 1).getAsInteger(10, Line)) {
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
