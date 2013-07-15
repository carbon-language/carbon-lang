//===- lib/Driver/DarwinLdDriver.cpp --------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Concrete instance of the Driver for darwin's ld.
///
//===----------------------------------------------------------------------===//

#include "lld/Driver/Driver.h"
#include "lld/ReaderWriter/MachOTargetInfo.h"
#include "../ReaderWriter/MachO/MachOFormat.hpp"

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


namespace {

// Create enum with OPT_xxx values for each option in DarwinOptions.td
enum DarwinOpt {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, HELP, META) \
          OPT_##ID,
#include "DarwinOptions.inc"
  LastOption
#undef OPTION
};

// Create prefix string literals used in DarwinOptions.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "DarwinOptions.inc"
#undef PREFIX

// Create table mapping all options defined in DarwinOptions.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, llvm::opt::Option::KIND##Class, \
    PARAM, FLAGS, OPT_##GROUP, OPT_##ALIAS },
#include "DarwinOptions.inc"
#undef OPTION
};

// Create OptTable class for parsing actual command line arguments
class DarwinLdOptTable : public llvm::opt::OptTable {
public:
  DarwinLdOptTable() : OptTable(infoTable, llvm::array_lengthof(infoTable)){}
};


} // namespace anonymous

namespace lld {

bool DarwinLdDriver::linkMachO(int argc, const char *argv[], 
                                                    raw_ostream &diagnostics) {
  MachOTargetInfo info;
  if (parse(argc, argv, info, diagnostics))
    return true;
    
  return link(info, diagnostics);
}



bool DarwinLdDriver::parse(int argc, const char *argv[],  
                          MachOTargetInfo &info, raw_ostream &diagnostics) {
  // Parse command line options using DarwinOptions.td
  std::unique_ptr<llvm::opt::InputArgList> parsedArgs;
  DarwinLdOptTable table;
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
  
  // Figure out output kind ( -dylib, -r, -bundle, -preload, or -static )
  if ( llvm::opt::Arg *kind = parsedArgs->getLastArg(OPT_dylib, OPT_relocatable,
                                      OPT_bundle, OPT_static, OPT_preload)) {
    switch (kind->getOption().getID()) {
    case OPT_dylib:
      info.setOutputFileType(mach_o::MH_DYLIB);
      break;
    case OPT_relocatable:
      info.setPrintRemainingUndefines(false);
      info.setAllowRemainingUndefines(true);
      info.setOutputFileType(mach_o::MH_OBJECT);
      break;
    case OPT_bundle:
      info.setOutputFileType(mach_o::MH_BUNDLE);
      break;
    case OPT_static:
      info.setOutputFileType(mach_o::MH_EXECUTE);
      break;
    case OPT_preload:
       info.setOutputFileType(mach_o::MH_PRELOAD);
      break;
    }
  }
  
  // Handle -e xxx
  if (llvm::opt::Arg *entry = parsedArgs->getLastArg(OPT_entry))
    info.setEntrySymbolName(entry->getValue());

  // Handle -o xxx
  if (llvm::opt::Arg *outpath = parsedArgs->getLastArg(OPT_output))
    info.setOutputPath(outpath->getValue());
    
  // Handle -dead_strip
  if (parsedArgs->getLastArg(OPT_dead_strip))
    info.setDeadStripping(true);
  
  // Handle -arch xxx
  if (llvm::opt::Arg *archStr = parsedArgs->getLastArg(OPT_arch)) {
    info.setArch(llvm::StringSwitch<MachOTargetInfo::Arch>(archStr->getValue())
           .Case("x86_64",  MachOTargetInfo::arch_x86_64)
           .Case("i386",    MachOTargetInfo::arch_x86)
           .Case("armv6",   MachOTargetInfo::arch_armv6)
           .Case("armv7",   MachOTargetInfo::arch_armv7)
           .Case("armv7s",  MachOTargetInfo::arch_armv7s)
           .Default(MachOTargetInfo::arch_unknown));
  }

  // Handle -macosx_version_min or -ios_version_min
  if (llvm::opt::Arg *minOS = parsedArgs->getLastArg(
                                               OPT_macosx_version_min,
                                               OPT_ios_version_min,
                                               OPT_ios_simulator_version_min)) {
    switch (minOS->getOption().getID()) {
    case OPT_macosx_version_min:
      if (info.setOS(MachOTargetInfo::OS::macOSX, minOS->getValue())) {
        diagnostics << "error: malformed macosx_version_min value\n";
        return true;
      }
      break;
    case OPT_ios_version_min:
      if (info.setOS(MachOTargetInfo::OS::iOS, minOS->getValue())) {
        diagnostics << "error: malformed ios_version_min value\n";
        return true;
      }
      break;
    case OPT_ios_simulator_version_min:
      if (info.setOS(MachOTargetInfo::OS::iOS_simulator, minOS->getValue())) {
        diagnostics << "error: malformed ios_simulator_version_min value\n";
        return true;
      }
      break;
    }
  }
  else {
    // No min-os version on command line, check environment variables
  
  }
  
  // Handle input files
  for (llvm::opt::arg_iterator it = parsedArgs->filtered_begin(OPT_INPUT),
                               ie = parsedArgs->filtered_end();
                              it != ie; ++it) {
    info.appendInputFile((*it)->getValue());
  }
  
  // Validate the combination of options used.
  if (info.validate(diagnostics))
    return true;

  return false;
}


} // namespace lld


