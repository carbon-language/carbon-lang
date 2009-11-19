//===--- DriverOptions.cpp - Driver Options Table -----------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Options.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Option.h"

using namespace clang::driver;
using namespace clang::driver::options;

static OptTable::Info InfoTable[] = {
  // The InputOption info
  { "<input>", 0, 0, Option::InputClass, DriverOption, 0, OPT_INVALID, OPT_INVALID },
  // The UnknownOption info
  { "<unknown>", 0, 0, Option::UnknownClass, 0, 0, OPT_INVALID, OPT_INVALID },

#define OPTION(NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { NAME, HELPTEXT, METAVAR, Option::KIND##Class, FLAGS, PARAM, \
    OPT_##GROUP, OPT_##ALIAS },
#include "clang/Driver/Options.inc"
};

namespace {

class DriverOptTable : public OptTable {
public:
  DriverOptTable()
    : OptTable(InfoTable, sizeof(InfoTable) / sizeof(InfoTable[0])) {}
};

}

OptTable *clang::driver::createDriverOptTable() {
  return new DriverOptTable();
}
