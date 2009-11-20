//===--- CC1Options.cpp - Clang CC1 Options Table -----------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/CC1Options.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Option.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Host.h"

using namespace clang::driver;
using namespace clang::driver::options;
using namespace clang::driver::cc1options;

static OptTable::Info CC1InfoTable[] = {
#define OPTION(NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { NAME, HELPTEXT, METAVAR, Option::KIND##Class, FLAGS, PARAM, \
    OPT_##GROUP, OPT_##ALIAS },
#include "clang/Driver/CC1Options.inc"
};

namespace {

class CC1OptTable : public OptTable {
public:
  CC1OptTable()
    : OptTable(CC1InfoTable, sizeof(CC1InfoTable) / sizeof(CC1InfoTable[0])) {}
};

}

OptTable *clang::driver::createCC1OptTable() {
  return new CC1OptTable();
}

//

using namespace clang;

static llvm::StringRef getLastArgValue(ArgList &Args, cc1options::ID ID,
                                       llvm::StringRef Default = "") {
  if (Arg *A = Args.getLastArg(ID))
    return A->getValue(Args);
  return Default;
}

static std::vector<std::string>
getAllArgValues(ArgList &Args, cc1options::ID ID) {
  llvm::SmallVector<const char *, 16> Values;
  Args.AddAllArgValues(Values, ID);
  return std::vector<std::string>(Values.begin(), Values.end());
}

static void ParseTargetArgs(TargetOptions &Opts, ArgList &Args) {
  Opts.ABI = getLastArgValue(Args, cc1options::OPT_target_abi);
  Opts.CPU = getLastArgValue(Args, cc1options::OPT_mcpu);
  Opts.Triple = getLastArgValue(Args, cc1options::OPT_triple);
  Opts.Features = getAllArgValues(Args, cc1options::OPT_target_feature);

  // Use the host triple if unspecified.
  if (Opts.Triple.empty())
    Opts.Triple = llvm::sys::getHostTriple();
}

void CompilerInvocation::CreateFromArgs(CompilerInvocation &Res,
                           const llvm::SmallVectorImpl<llvm::StringRef> &Args) {
  // This is gratuitous, but until we switch the driver to using StringRe we
  // need to get C strings.
  llvm::SmallVector<std::string, 16> StringArgs(Args.begin(), Args.end());
  llvm::SmallVector<const char *, 16> CStringArgs;
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
    CStringArgs.push_back(StringArgs[i].c_str());

  // Parse the arguments.
  llvm::OwningPtr<OptTable> Opts(createCC1OptTable());
  unsigned MissingArgIndex, MissingArgCount;
  llvm::OwningPtr<InputArgList> InputArgs(
    Opts->ParseArgs(CStringArgs.begin(), CStringArgs.end(),
                    MissingArgIndex, MissingArgCount));

  // Check for missing argument error.
  if (MissingArgCount) {
    // FIXME: Use proper diagnostics!
    llvm::errs() << "error: argument to '"
                 << InputArgs->getArgString(MissingArgIndex)
                 << "' is missing (expected " << MissingArgCount
                 << " value )\n";
  }

  ParseTargetArgs(Res.getTargetOpts(), *InputArgs);
}
