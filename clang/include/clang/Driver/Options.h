//===--- Options.h - Option info & table ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_OPTIONS_H
#define CLANG_DRIVER_OPTIONS_H

namespace llvm {
namespace opt {
class OptTable;
}
}

namespace clang {
namespace driver {

namespace options {
/// Flags specifically for clang options.  Must not overlap with
/// llvm::opt::DriverFlag.
enum ClangFlags {
  DriverOption = (1 << 4),
  LinkerInput = (1 << 5),
  NoArgumentUnused = (1 << 6),
  NoForward = (1 << 7),
  Unsupported = (1 << 8),
  CC1Option = (1 << 9),
  NoDriverOption = (1 << 10)
};

enum ID {
    OPT_INVALID = 0, // This is not an option ID.
#define PREFIX(NAME, VALUE)
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR) OPT_##ID,
#include "clang/Driver/Options.inc"
    LastOption
#undef OPTION
#undef PREFIX
  };
}

llvm::opt::OptTable *createDriverOptTable();
}
}

#endif
