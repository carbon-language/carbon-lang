//===- lib/Driver/CoreDriver.cpp ------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Driver/Driver.h"
#include "lld/ReaderWriter/CoreTargetInfo.h"
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
enum CoreOpt {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, HELP, META) \
          OPT_##ID,
#include "CoreOptions.inc"
  LastOption
#undef OPTION
};

// Create prefix string literals used in CoreOptions.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "CoreOptions.inc"
#undef PREFIX

// Create table mapping all options defined in CoreOptions.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, llvm::opt::Option::KIND##Class, \
    PARAM, FLAGS, OPT_##GROUP, OPT_##ALIAS },
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
  CoreTargetInfo info;
  if (parse(argc, argv, info))
    return true;
  
  return Driver::link(info);
}


bool CoreDriver::parse(int argc, const char *argv[],  
                          CoreTargetInfo &info, raw_ostream &diagnostics) {
  // Parse command line options using CoreOptions.td
  std::unique_ptr<llvm::opt::InputArgList> parsedArgs;
  CoreOptTable table;
  unsigned missingIndex;
  unsigned missingCount;
  parsedArgs.reset(table.ParseArgs(&argv[1], &argv[argc], 
                                                missingIndex, missingCount));
  if (missingCount) {
    diagnostics  << "error: missing arg value for '"
                 << parsedArgs->getArgString(missingIndex)
                 << "' expected " << missingCount << " argument(s).\n";
    return true;
  }

  for (auto it = parsedArgs->filtered_begin(OPT_UNKNOWN),
            ie = parsedArgs->filtered_end(); it != ie; ++it) {
    diagnostics  << "warning: ignoring unknown argument: "
                 << (*it)->getAsString(*parsedArgs) << "\n";
  }
  
  
  // Handle -e xxx
  if (llvm::opt::Arg *entry = parsedArgs->getLastArg(OPT_entry))
    info.setEntrySymbolName(entry->getValue());
    
  // Handle -o xxx
  if (llvm::opt::Arg *outpath = parsedArgs->getLastArg(OPT_output))
    info.setOutputPath(outpath->getValue());
  else
    info.setOutputPath("-");
    
  // Handle --dead_strip
  if (parsedArgs->getLastArg(OPT_dead_strip))
    info.setDeadStripping(true);
  else
    info.setDeadStripping(false);
 
  // Handle --keep-globals
  if (parsedArgs->getLastArg(OPT_keep_globals))
    info.setGlobalsAreDeadStripRoots(true);
  else
    info.setGlobalsAreDeadStripRoots(false);
  
  // Handle --undefines-are-errors
  if (parsedArgs->getLastArg(OPT_undefines_are_errors)) {
    info.setPrintRemainingUndefines(true);
    info.setAllowRemainingUndefines(false);
  }
  else {
    info.setPrintRemainingUndefines(false);
    info.setAllowRemainingUndefines(true);
  }

  // Handle --commons-search-archives
  if (parsedArgs->getLastArg(OPT_commons_search_archives))
    info.setSearchArchivesToOverrideTentativeDefinitions(true);
  else
    info.setSearchArchivesToOverrideTentativeDefinitions(false);
  
  // Handle --add-pass xxx option
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_add_pass),
                               ie = parsedArgs->filtered_end();
                              it != ie; ++it) {
    info.addPassNamed((*it)->getValue());
  }

  // Handle input files
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_INPUT),
                               ie = parsedArgs->filtered_end();
                              it != ie; ++it) {
    info.appendInputFile((*it)->getValue());
  }
  
  return false;
}

} // namespace lld

