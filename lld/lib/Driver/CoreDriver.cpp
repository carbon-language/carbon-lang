//===- lib/Driver/CoreDriver.cpp ------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Driver/Driver.h"
#include "lld/Driver/CoreInputGraph.h"
#include "lld/ReaderWriter/CoreLinkingContext.h"
#include "lld/ReaderWriter/Reader.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"

using namespace lld;

namespace {

// Create enum with OPT_xxx values for each option in DarwinOptions.td
enum {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELP, META) \
          OPT_##ID,
#include "CoreOptions.inc"
#undef OPTION
};

// Create prefix string literals used in CoreOptions.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "CoreOptions.inc"
#undef PREFIX

// Create table mapping all options defined in CoreOptions.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, llvm::opt::Option::KIND##Class, \
    PARAM, FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS },
#include "CoreOptions.inc"
#undef OPTION
};

// Create OptTable class for parsing actual command line arguments
class CoreOptTable : public llvm::opt::OptTable {
public:
  CoreOptTable() : OptTable(infoTable, llvm::array_lengthof(infoTable)){}
};

} // namespace anonymous


namespace lld {

bool CoreDriver::link(int argc, const char *argv[], raw_ostream &diagnostics) {
  CoreLinkingContext info;
  if (!parse(argc, argv, info))
    return false;
  return Driver::link(info);
}

bool CoreDriver::parse(int argc, const char *argv[], CoreLinkingContext &ctx,
                       raw_ostream &diagnostics) {
  // Parse command line options using CoreOptions.td
  std::unique_ptr<llvm::opt::InputArgList> parsedArgs;
  CoreOptTable table;
  unsigned missingIndex;
  unsigned missingCount;
  parsedArgs.reset(
      table.ParseArgs(&argv[1], &argv[argc], missingIndex, missingCount));
  if (missingCount) {
    diagnostics << "error: missing arg value for '"
                << parsedArgs->getArgString(missingIndex) << "' expected "
                << missingCount << " argument(s).\n";
    return false;
  }

  std::unique_ptr<InputGraph> inputGraph(new InputGraph());

  // Set default options
  ctx.setOutputPath("-");
  ctx.setDeadStripping(false);
  ctx.setGlobalsAreDeadStripRoots(false);
  ctx.setPrintRemainingUndefines(false);
  ctx.setAllowRemainingUndefines(true);
  ctx.setSearchArchivesToOverrideTentativeDefinitions(false);

  // Process all the arguments and create Input Elements
  for (auto inputArg : *parsedArgs) {
    switch (inputArg->getOption().getID()) {
    case OPT_mllvm:
      ctx.appendLLVMOption(inputArg->getValue());
      break;

    case OPT_entry:
      ctx.setEntrySymbolName(inputArg->getValue());
      break;

    case OPT_output:
      ctx.setOutputPath(inputArg->getValue());
      break;

    case OPT_dead_strip:
      ctx.setDeadStripping(true);
      break;

    case OPT_keep_globals:
      ctx.setGlobalsAreDeadStripRoots(true);
      break;

    case OPT_undefines_are_errors:
      ctx.setPrintRemainingUndefines(true);
      ctx.setAllowRemainingUndefines(false);
      break;

    case OPT_commons_search_archives:
      ctx.setSearchArchivesToOverrideTentativeDefinitions(true);
      break;

    case OPT_add_pass:
      ctx.addPassNamed(inputArg->getValue());
      break;

    case OPT_INPUT:
      inputGraph->addInputElement(std::unique_ptr<InputElement>(
          new COREFileNode(ctx, inputArg->getValue())));
      break;

    default:
      break;
    }
  }

  if (!inputGraph->size()) {
    diagnostics << "No input files\n";
    return false;
  }

  ctx.setInputGraph(std::move(inputGraph));

  // Validate the combination of options used.
  return ctx.validate(diagnostics);
}

} // namespace lld
