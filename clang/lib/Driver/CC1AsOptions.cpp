//===--- CC1AsOptions.cpp - Clang Assembler Options Table -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/CC1AsOptions.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/OptTable.h"
using namespace clang;
using namespace clang::driver;
using namespace clang::driver::options;
using namespace clang::driver::cc1asoptions;

static const OptTable::Info CC1AsInfoTable[] = {
#define OPTION(NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { NAME, HELPTEXT, METAVAR, Option::KIND##Class, PARAM, FLAGS, \
    OPT_##GROUP, OPT_##ALIAS },
#include "clang/Driver/CC1AsOptions.inc"
};

namespace {

class CC1AsOptTable : public OptTable {
public:
  CC1AsOptTable()
    : OptTable(CC1AsInfoTable,
               sizeof(CC1AsInfoTable) / sizeof(CC1AsInfoTable[0])) {}
};

}

OptTable *clang::driver::createCC1AsOptTable() {
  return new CC1AsOptTable();
}
