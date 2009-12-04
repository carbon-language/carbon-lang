//===- OptParserEmitter.cpp - Table Driven Command Line Parsing -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OptParserEmitter.h"
#include "Record.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

static int StrCmpOptionName(const char *A, const char *B) {
  char a = *A, b = *B;
  while (a == b) {
    if (a == '\0')
      return 0;

    a = *++A;
    b = *++B;
  }

  if (a == '\0') // A is a prefix of B.
    return 1;
  if (b == '\0') // B is a prefix of A.
    return -1;

  // Otherwise lexicographic.
  return (a < b) ? -1 : 1;
}

static int CompareOptionRecords(const void *Av, const void *Bv) {
  const Record *A = *(Record**) Av;
  const Record *B = *(Record**) Bv;

  // Sentinel options preceed all others and are only ordered by precedence.
  bool ASent = A->getValueAsDef("Kind")->getValueAsBit("Sentinel");
  bool BSent = B->getValueAsDef("Kind")->getValueAsBit("Sentinel");
  if (ASent != BSent)
    return ASent ? -1 : 1;

  // Compare options by name, unless they are sentinels.
  if (!ASent)
    if (int Cmp = StrCmpOptionName(A->getValueAsString("Name").c_str(),
                                   B->getValueAsString("Name").c_str()))
    return Cmp;

  // Then by the kind precedence;
  int APrec = A->getValueAsDef("Kind")->getValueAsInt("Precedence");
  int BPrec = B->getValueAsDef("Kind")->getValueAsInt("Precedence");
  assert(APrec != BPrec && "Options are equivalent!");
  return APrec < BPrec ? -1 : 1;
}

static const std::string getOptionName(const Record &R) {
  // Use the record name unless EnumName is defined.
  if (dynamic_cast<UnsetInit*>(R.getValueInit("EnumName")))
    return R.getName();

  return R.getValueAsString("EnumName");
}

static raw_ostream &write_cstring(raw_ostream &OS, llvm::StringRef Str) {
  OS << '"';
  OS.write_escaped(Str);
  OS << '"';
  return OS;
}

void OptParserEmitter::run(raw_ostream &OS) {
  // Get the option groups and options.
  const std::vector<Record*> &Groups =
    Records.getAllDerivedDefinitions("OptionGroup");
  std::vector<Record*> Opts = Records.getAllDerivedDefinitions("Option");

  if (GenDefs) {
    OS << "\
//=== TableGen'erated File - Option Parsing Definitions ---------*- C++ -*-===//\n \
//\n\
// Option Parsing Definitions\n\
//\n\
// Automatically generated file, do not edit!\n\
//\n\
//===----------------------------------------------------------------------===//\n";
  } else {
    OS << "\
//=== TableGen'erated File - Option Parsing Table ---------------*- C++ -*-===//\n \
//\n\
// Option Parsing Definitions\n\
//\n\
// Automatically generated file, do not edit!\n\
//\n\
//===----------------------------------------------------------------------===//\n";
  }
  OS << "\n";

  array_pod_sort(Opts.begin(), Opts.end(), CompareOptionRecords);
  if (GenDefs) {
    OS << "#ifndef OPTION\n";
    OS << "#error \"Define OPTION prior to including this file!\"\n";
    OS << "#endif\n\n";

    OS << "/////////\n";
    OS << "// Groups\n\n";
    for (unsigned i = 0, e = Groups.size(); i != e; ++i) {
      const Record &R = *Groups[i];

      // Start a single option entry.
      OS << "OPTION(";

      // The option string.
      OS << '"' << R.getValueAsString("Name") << '"';

      // The option identifier name.
      OS  << ", "<< getOptionName(R);

      // The option kind.
      OS << ", Group";

      // The containing option group (if any).
      OS << ", ";
      if (const DefInit *DI = dynamic_cast<DefInit*>(R.getValueInit("Group")))
        OS << getOptionName(*DI->getDef());
      else
        OS << "INVALID";

      // The other option arguments (unused for groups).
      OS << ", INVALID, 0, 0";

      // The option help text.
      if (!dynamic_cast<UnsetInit*>(R.getValueInit("HelpText"))) {
        OS << ",\n";
        OS << "       ";
        write_cstring(OS, R.getValueAsString("HelpText"));
      } else
        OS << ", 0";

      // The option meta-variable name (unused).
      OS << ", 0)\n";
    }
    OS << "\n";

    OS << "//////////\n";
    OS << "// Options\n\n";
    for (unsigned i = 0, e = Opts.size(); i != e; ++i) {
      const Record &R = *Opts[i];

      // Start a single option entry.
      OS << "OPTION(";

      // The option string.
      write_cstring(OS, R.getValueAsString("Name"));

      // The option identifier name.
      OS  << ", "<< getOptionName(R);

      // The option kind.
      OS << ", " << R.getValueAsDef("Kind")->getValueAsString("Name");

      // The containing option group (if any).
      OS << ", ";
      if (const DefInit *DI = dynamic_cast<DefInit*>(R.getValueInit("Group")))
        OS << getOptionName(*DI->getDef());
      else
        OS << "INVALID";

      // The option alias (if any).
      OS << ", ";
      if (const DefInit *DI = dynamic_cast<DefInit*>(R.getValueInit("Alias")))
        OS << getOptionName(*DI->getDef());
      else
        OS << "INVALID";

      // The option flags.
      const ListInit *LI = R.getValueAsListInit("Flags");
      if (LI->empty()) {
        OS << ", 0";
      } else {
        OS << ", ";
        for (unsigned i = 0, e = LI->size(); i != e; ++i) {
          if (i)
            OS << " | ";
          OS << dynamic_cast<DefInit*>(LI->getElement(i))->getDef()->getName();
        }
      }

      // The option parameter field.
      OS << ", " << R.getValueAsInt("NumArgs");

      // The option help text.
      if (!dynamic_cast<UnsetInit*>(R.getValueInit("HelpText"))) {
        OS << ",\n";
        OS << "       ";
        write_cstring(OS, R.getValueAsString("HelpText"));
      } else
        OS << ", 0";

      // The option meta-variable name.
      OS << ", ";
      if (!dynamic_cast<UnsetInit*>(R.getValueInit("MetaVarName")))
        write_cstring(OS, R.getValueAsString("MetaVarName"));
      else
        OS << "0";

      OS << ")\n";
    }
  }
}
