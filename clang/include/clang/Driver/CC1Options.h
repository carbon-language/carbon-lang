//===--- CC1Options.h - Clang CC1 Options Table -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_CC1OPTIONS_H
#define CLANG_DRIVER_CC1OPTIONS_H

namespace clang {
namespace driver {
  class OptTable;

namespace cc1options {
  enum ID {
    OPT_INVALID = 0, // This is not an option ID.
#define OPTION(NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR) OPT_##ID,
#include "clang/Driver/CC1Options.inc"
    LastOption
#undef OPTION
  };
}

  OptTable *createCC1OptTable();
}
}

#endif
