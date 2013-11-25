//===- lib/Driver/GnuLdDriver.cpp -----------------------------------------===//
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
/// Concrete instance of the Driver for GNU's ld.
///
//===----------------------------------------------------------------------===//

#include "lld/Driver/Driver.h"
#include "lld/Driver/GnuLdInputGraph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
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

// Create enum with OPT_xxx values for each option in GnuLdOptions.td
enum {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELP, META) \
          OPT_##ID,
#include "GnuLdOptions.inc"
#undef OPTION
};

// Create prefix string literals used in GnuLdOptions.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "GnuLdOptions.inc"
#undef PREFIX

// Create table mapping all options defined in GnuLdOptions.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, llvm::opt::Option::KIND##Class, \
    PARAM, FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS },
#include "GnuLdOptions.inc"
#undef OPTION
};


// Create OptTable class for parsing actual command line arguments
class GnuLdOptTable : public llvm::opt::OptTable {
public:
  GnuLdOptTable() : OptTable(infoTable, llvm::array_lengthof(infoTable)){}
};

// Get the Input file magic for creating appropriate InputGraph nodes.
error_code getFileMagic(ELFLinkingContext &ctx, StringRef path,
                        llvm::sys::fs::file_magic &magic) {
  error_code ec = llvm::sys::fs::identify_magic(path, magic);
  if (ec)
    return ec;
  switch (magic) {
  case llvm::sys::fs::file_magic::archive:
  case llvm::sys::fs::file_magic::elf_relocatable:
  case llvm::sys::fs::file_magic::elf_shared_object:
  case llvm::sys::fs::file_magic::unknown:
    return error_code::success();
  default:
    break;
  }
  return make_error_code(ReaderError::unknown_file_format);
}

} // namespace

llvm::ErrorOr<StringRef> ELFFileNode::getPath(const LinkingContext &) const {
  if (!_isDashlPrefix)
    return _path;
  return _elfLinkingContext.searchLibrary(_path);
}

std::string ELFFileNode::errStr(error_code errc) {
  if (errc == llvm::errc::no_such_file_or_directory) {
    if (_isDashlPrefix)
      return (Twine("Unable to find library -l") + _path).str();
    return (Twine("Unable to find file ") + _path).str();
  }
  return FileNode::errStr(errc);
}

bool GnuLdDriver::linkELF(int argc, const char *argv[],
                          raw_ostream &diagnostics) {
  std::unique_ptr<ELFLinkingContext> options;
  if (!parse(argc, argv, options, diagnostics))
    return false;
  if (!options)
    return true;

  return link(*options, diagnostics);
}

bool GnuLdDriver::parse(int argc, const char *argv[],
                        std::unique_ptr<ELFLinkingContext> &context,
                        raw_ostream &diagnostics) {
  // Parse command line options using GnuLdOptions.td
  std::unique_ptr<llvm::opt::InputArgList> parsedArgs;
  GnuLdOptTable table;
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

  // Handle --help
  if (parsedArgs->getLastArg(OPT_help)) {
    table.PrintHelp(llvm::outs(), argv[0], "LLVM Linker", false);
    return true;
  }

  // Use -target or use default target triple to instantiate LinkingContext
  llvm::Triple triple;
  if (llvm::opt::Arg *trip = parsedArgs->getLastArg(OPT_target))
    triple = llvm::Triple(trip->getValue());
  else
    triple = getDefaultTarget(argv[0]);
  std::unique_ptr<ELFLinkingContext> ctx(ELFLinkingContext::create(triple));

  if (!ctx) {
    diagnostics << "unknown target triple\n";
    return false;
  }

  std::unique_ptr<InputGraph> inputGraph(new InputGraph());
  std::stack<InputElement *> controlNodeStack;

  // Positional options for an Input File
  bool isWholeArchive = false;
  bool asNeeded = false;
  bool _outputOptionSet = false;

  int index = 0;

  // Set sys root path.
  if (llvm::opt::Arg *sysRootPath = parsedArgs->getLastArg(OPT_sysroot))
    ctx->setSysroot(sysRootPath->getValue());

  // Add all search paths.
  for (auto it = parsedArgs->filtered_begin(OPT_L),
            ie = parsedArgs->filtered_end();
       it != ie; ++it)
    ctx->addSearchPath((*it)->getValue());

  // Create a dynamic executable by default
  ctx->setOutputELFType(llvm::ELF::ET_EXEC);
  ctx->setIsStaticExecutable(false);
  ctx->setAllowShlibUndefines(false);
  ctx->setUseShlibUndefines(true);

  // Figure out output kind ( -r, -static, -shared)
  if ( llvm::opt::Arg *kind = parsedArgs->getLastArg(OPT_relocatable, OPT_static,
                                      OPT_shared, OPT_nmagic,
                                      OPT_omagic, OPT_no_omagic)) {
    switch (kind->getOption().getID()) {
    case OPT_relocatable:
      ctx->setOutputELFType(llvm::ELF::ET_REL);
      ctx->setPrintRemainingUndefines(false);
      ctx->setAllowRemainingUndefines(true);
      break;

    case OPT_static:
      ctx->setOutputELFType(llvm::ELF::ET_EXEC);
      ctx->setIsStaticExecutable(true);
      break;
    case OPT_shared:
      ctx->setOutputELFType(llvm::ELF::ET_DYN);
      ctx->setAllowShlibUndefines(true);
      ctx->setUseShlibUndefines(false);
      break;
    }
  }

  // Figure out if the output type is nmagic/omagic
  if ( llvm::opt::Arg *kind = parsedArgs->getLastArg(OPT_nmagic, OPT_omagic,
                                                     OPT_no_omagic)) {
    switch (kind->getOption().getID()) {
    case OPT_nmagic:
      ctx->setOutputMagic(ELFLinkingContext::OutputMagic::NMAGIC);
      ctx->setIsStaticExecutable(true);
      break;

    case OPT_omagic:
      ctx->setOutputMagic(ELFLinkingContext::OutputMagic::OMAGIC);
      ctx->setIsStaticExecutable(true);
      break;

    case OPT_no_omagic:
      ctx->setOutputMagic(ELFLinkingContext::OutputMagic::DEFAULT);
      ctx->setNoAllowDynamicLibraries();
      break;
    }
  }

  // Process all the arguments and create Input Elements
  for (auto inputArg : *parsedArgs) {
    switch (inputArg->getOption().getID()) {
    case OPT_mllvm:
      ctx->appendLLVMOption(inputArg->getValue());
      break;
    case OPT_e:
      ctx->setEntrySymbolName(inputArg->getValue());
      break;

    case OPT_output:
      _outputOptionSet = true;
      ctx->setOutputPath(inputArg->getValue());
      break;

    case OPT_noinhibit_exec:
      ctx->setAllowRemainingUndefines(true);
      break;

    case OPT_merge_strings:
      ctx->setMergeCommonStrings(true);
      break;

    case OPT_t:
      ctx->setLogInputFiles(true);
      break;

    case OPT_no_allow_shlib_undefs:
      ctx->setAllowShlibUndefines(false);
      break;

    case OPT_allow_shlib_undefs:
      ctx->setAllowShlibUndefines(true);
      break;

    case OPT_use_shlib_undefs:
      ctx->setUseShlibUndefines(true);
      break;

    case OPT_dynamic_linker:
      ctx->setInterpreter(inputArg->getValue());
      break;

    case OPT_u:
      ctx->addInitialUndefinedSymbol(inputArg->getValue());
      break;

    case OPT_init:
      ctx->addInitFunction(inputArg->getValue());
      break;

    case OPT_fini:
      ctx->addFiniFunction(inputArg->getValue());
      break;

    case OPT_output_filetype:
      ctx->setOutputFileType(inputArg->getValue());
      break;

    case OPT_no_whole_archive:
      isWholeArchive = false;
      break;

    case OPT_whole_archive:
      isWholeArchive = true;
      break;

    case OPT_as_needed:
      asNeeded = true;
      break;

    case OPT_no_as_needed:
      asNeeded = false;
      break;

    case OPT_start_group: {
      std::unique_ptr<InputElement> controlStart(new ELFGroup(*ctx, index++));
      controlNodeStack.push(controlStart.get());
      dyn_cast<ControlNode>(controlNodeStack.top())->processControlEnter();
      inputGraph->addInputElement(std::move(controlStart));
      break;
    }

    case OPT_end_group:
      dyn_cast<ControlNode>(controlNodeStack.top())->processControlExit();
      controlNodeStack.pop();
      break;

    case OPT_INPUT:
    case OPT_l: {
      bool isDashlPrefix = (inputArg->getOption().getID() == OPT_l);
      bool isELFFileNode = true;
      StringRef userPath = inputArg->getValue();
      std::string resolvedInputPath = userPath;

      // If the path was referred to by using a -l argument, lets search
      // for the file in the search path.
      if (isDashlPrefix) {
        ErrorOr<StringRef> resolvedPath = ctx->searchLibrary(userPath);
        if (!resolvedPath) {
          diagnostics << " Unable to find library -l" << userPath << "\n";
          return false;
        }
        resolvedInputPath = resolvedPath->str();
      }
      llvm::sys::fs::file_magic magic = llvm::sys::fs::file_magic::unknown;
      error_code ec = getFileMagic(*ctx, resolvedInputPath, magic);
      if (ec) {
        diagnostics << "lld: unknown input file format for file " << userPath
                    << "\n";
        return false;
      }
      if ((!userPath.endswith(".objtxt")) &&
          (magic == llvm::sys::fs::file_magic::unknown))
        isELFFileNode = false;

      FileNode *inputNode = nullptr;
      if (isELFFileNode)
        inputNode = new ELFFileNode(*ctx, userPath, index++, isWholeArchive,
                                    asNeeded, isDashlPrefix);
      else {
        inputNode = new ELFGNULdScript(*ctx, resolvedInputPath, index++);
        ec = inputNode->parse(*ctx, diagnostics);
        if (ec) {
          diagnostics << userPath << ": Error parsing linker script"
                      << "\n";
          return false;
        }
      }
      std::unique_ptr<InputElement> inputFile(inputNode);
      if (controlNodeStack.empty())
        inputGraph->addInputElement(std::move(inputFile));
      else
        dyn_cast<ControlNode>(controlNodeStack.top())
            ->processInputElement(std::move(inputFile));
      break;
    }

    case OPT_rpath: {
      SmallVector<StringRef, 2> rpaths;
      StringRef(inputArg->getValue()).split(rpaths, ":");
      for (auto path : rpaths)
        ctx->addRpath(path);
      break;
    }

    case OPT_rpath_link: {
      SmallVector<StringRef, 2> rpaths;
      StringRef(inputArg->getValue()).split(rpaths, ":");
      for (auto path : rpaths)
        ctx->addRpathLink(path);
      break;
    }

    case OPT_soname:
      ctx->setSharedObjectName(inputArg->getValue());
      break;

    default:
      break;
    } // end switch on option ID
  }   // end for

  if (!inputGraph->size()) {
    diagnostics << "No input files\n";
    return false;
  }

  // Set default output file name if the output file was not
  // specified.
  if (!_outputOptionSet) {
    switch (ctx->outputFileType()) {
    case LinkingContext::OutputFileType::YAML:
      ctx->setOutputPath("-");
      break;
    case LinkingContext::OutputFileType::Native:
      ctx->setOutputPath("a.native");
      break;
    default:
      ctx->setOutputPath("a.out");
      break;
    }
  }

  if (ctx->outputFileType() == LinkingContext::OutputFileType::YAML)
    inputGraph->dump(diagnostics);

  // Validate the combination of options used.
  if (!ctx->validate(diagnostics))
    return false;

  // Normalize the InputGraph.
  inputGraph->normalize();

  ctx->setInputGraph(std::move(inputGraph));

  context.swap(ctx);

  return true;
}

/// Get the default target triple based on either the program name
/// (e.g. "x86-ibm-linux-lld") or the primary target llvm was configured for.
llvm::Triple GnuLdDriver::getDefaultTarget(const char *progName) {
  SmallVector<StringRef, 4> components;
  llvm::SplitString(llvm::sys::path::stem(progName), components, "-");
  // If has enough parts to be start with a triple.
  if (components.size() >= 4) {
    llvm::Triple triple(components[0], components[1], components[2],
                        components[3]);
    // If first component looks like an arch.
    if (triple.getArch() != llvm::Triple::UnknownArch)
      return triple;
  }

  // Fallback to use whatever default triple llvm was configured for.
  return llvm::Triple(llvm::sys::getDefaultTargetTriple());
}
