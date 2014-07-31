//===-- gold-plugin.cpp - Plugin to gold for Link Time Optimization  ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a gold plugin for LLVM. It provides an LLVM implementation of the
// interface described in http://gcc.gnu.org/wiki/whopr/driver .
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h" // plugin-api.h requires HAVE_STDINT_H
#include "llvm-c/lto.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include <list>
#include <plugin-api.h>
#include <system_error>
#include <vector>

#ifndef LDPO_PIE
// FIXME: remove this declaration when we stop maintaining Ubuntu Quantal and
// Precise and Debian Wheezy (binutils 2.23 is required)
# define LDPO_PIE 3
#endif

using namespace llvm;

namespace {
struct claimed_file {
  void *handle;
  std::vector<ld_plugin_symbol> syms;
};
}

static ld_plugin_status discard_message(int level, const char *format, ...) {
  // Die loudly. Recent versions of Gold pass ld_plugin_message as the first
  // callback in the transfer vector. This should never be called.
  abort();
}

static ld_plugin_add_symbols add_symbols = nullptr;
static ld_plugin_get_symbols get_symbols = nullptr;
static ld_plugin_add_input_file add_input_file = nullptr;
static ld_plugin_set_extra_library_path set_extra_library_path = nullptr;
static ld_plugin_get_view get_view = nullptr;
static ld_plugin_message message = discard_message;
static lto_codegen_model output_type = LTO_CODEGEN_PIC_MODEL_STATIC;
static std::string output_name = "";
static std::list<claimed_file> Modules;
static std::vector<std::string> Cleanup;
static LTOCodeGenerator *CodeGen = nullptr;
static StringSet<> CannotBeHidden;
static llvm::TargetOptions TargetOpts;

namespace options {
  enum generate_bc { BC_NO, BC_ALSO, BC_ONLY };
  static bool generate_api_file = false;
  static generate_bc generate_bc_file = BC_NO;
  static std::string bc_path;
  static std::string obj_path;
  static std::string extra_library_path;
  static std::string triple;
  static std::string mcpu;
  // Additional options to pass into the code generator.
  // Note: This array will contain all plugin options which are not claimed
  // as plugin exclusive to pass to the code generator.
  // For example, "generate-api-file" and "as"options are for the plugin
  // use only and will not be passed.
  static std::vector<const char *> extra;

  static void process_plugin_option(const char* opt_)
  {
    if (opt_ == nullptr)
      return;
    llvm::StringRef opt = opt_;

    if (opt == "generate-api-file") {
      generate_api_file = true;
    } else if (opt.startswith("mcpu=")) {
      mcpu = opt.substr(strlen("mcpu="));
    } else if (opt.startswith("extra-library-path=")) {
      extra_library_path = opt.substr(strlen("extra_library_path="));
    } else if (opt.startswith("mtriple=")) {
      triple = opt.substr(strlen("mtriple="));
    } else if (opt.startswith("obj-path=")) {
      obj_path = opt.substr(strlen("obj-path="));
    } else if (opt == "emit-llvm") {
      generate_bc_file = BC_ONLY;
    } else if (opt == "also-emit-llvm") {
      generate_bc_file = BC_ALSO;
    } else if (opt.startswith("also-emit-llvm=")) {
      llvm::StringRef path = opt.substr(strlen("also-emit-llvm="));
      generate_bc_file = BC_ALSO;
      if (!bc_path.empty()) {
        message(LDPL_WARNING, "Path to the output IL file specified twice. "
                              "Discarding %s",
                opt_);
      } else {
        bc_path = path;
      }
    } else {
      // Save this option to pass to the code generator.
      extra.push_back(opt_);
    }
  }
}

static ld_plugin_status claim_file_hook(const ld_plugin_input_file *file,
                                        int *claimed);
static ld_plugin_status all_symbols_read_hook(void);
static ld_plugin_status cleanup_hook(void);

extern "C" ld_plugin_status onload(ld_plugin_tv *tv);
ld_plugin_status onload(ld_plugin_tv *tv) {
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  // We're given a pointer to the first transfer vector. We read through them
  // until we find one where tv_tag == LDPT_NULL. The REGISTER_* tagged values
  // contain pointers to functions that we need to call to register our own
  // hooks. The others are addresses of functions we can use to call into gold
  // for services.

  bool registeredClaimFile = false;
  bool RegisteredAllSymbolsRead = false;

  for (; tv->tv_tag != LDPT_NULL; ++tv) {
    switch (tv->tv_tag) {
      case LDPT_OUTPUT_NAME:
        output_name = tv->tv_u.tv_string;
        break;
      case LDPT_LINKER_OUTPUT:
        switch (tv->tv_u.tv_val) {
          case LDPO_REL:  // .o
          case LDPO_DYN:  // .so
          case LDPO_PIE:  // position independent executable
            output_type = LTO_CODEGEN_PIC_MODEL_DYNAMIC;
            break;
          case LDPO_EXEC:  // .exe
            output_type = LTO_CODEGEN_PIC_MODEL_STATIC;
            break;
          default:
            message(LDPL_ERROR, "Unknown output file type %d", tv->tv_u.tv_val);
            return LDPS_ERR;
        }
        break;
      case LDPT_OPTION:
        options::process_plugin_option(tv->tv_u.tv_string);
        break;
      case LDPT_REGISTER_CLAIM_FILE_HOOK: {
        ld_plugin_register_claim_file callback;
        callback = tv->tv_u.tv_register_claim_file;

        if (callback(claim_file_hook) != LDPS_OK)
          return LDPS_ERR;

        registeredClaimFile = true;
      } break;
      case LDPT_REGISTER_ALL_SYMBOLS_READ_HOOK: {
        ld_plugin_register_all_symbols_read callback;
        callback = tv->tv_u.tv_register_all_symbols_read;

        if (callback(all_symbols_read_hook) != LDPS_OK)
          return LDPS_ERR;

        RegisteredAllSymbolsRead = true;
      } break;
      case LDPT_REGISTER_CLEANUP_HOOK: {
        ld_plugin_register_cleanup callback;
        callback = tv->tv_u.tv_register_cleanup;

        if (callback(cleanup_hook) != LDPS_OK)
          return LDPS_ERR;
      } break;
      case LDPT_ADD_SYMBOLS:
        add_symbols = tv->tv_u.tv_add_symbols;
        break;
      case LDPT_GET_SYMBOLS_V2:
        get_symbols = tv->tv_u.tv_get_symbols;
        break;
      case LDPT_ADD_INPUT_FILE:
        add_input_file = tv->tv_u.tv_add_input_file;
        break;
      case LDPT_SET_EXTRA_LIBRARY_PATH:
        set_extra_library_path = tv->tv_u.tv_set_extra_library_path;
        break;
      case LDPT_GET_VIEW:
        get_view = tv->tv_u.tv_get_view;
        break;
      case LDPT_MESSAGE:
        message = tv->tv_u.tv_message;
        break;
      default:
        break;
    }
  }

  if (!registeredClaimFile) {
    message(LDPL_ERROR, "register_claim_file not passed to LLVMgold.");
    return LDPS_ERR;
  }
  if (!add_symbols) {
    message(LDPL_ERROR, "add_symbols not passed to LLVMgold.");
    return LDPS_ERR;
  }

  if (!RegisteredAllSymbolsRead)
    return LDPS_OK;

  CodeGen = new LTOCodeGenerator();

  // Pass through extra options to the code generator.
  if (!options::extra.empty()) {
    for (const char *Opt : options::extra)
      CodeGen->setCodeGenDebugOptions(Opt);
  }

  CodeGen->parseCodeGenDebugOptions();
  if (MAttrs.size()) {
    std::string Attrs;
    for (unsigned I = 0; I < MAttrs.size(); ++I) {
      if (I > 0)
        Attrs.append(",");
      Attrs.append(MAttrs[I]);
    }
    CodeGen->setAttr(Attrs.c_str());
  }

  TargetOpts = InitTargetOptionsFromCodeGenFlags();
  CodeGen->setTargetOptions(TargetOpts);

  return LDPS_OK;
}

/// Called by gold to see whether this file is one that our plugin can handle.
/// We'll try to open it and register all the symbols with add_symbol if
/// possible.
static ld_plugin_status claim_file_hook(const ld_plugin_input_file *file,
                                        int *claimed) {
  const void *view;
  std::unique_ptr<MemoryBuffer> buffer;
  if (get_view) {
    if (get_view(file->handle, &view) != LDPS_OK) {
      message(LDPL_ERROR, "Failed to get a view of %s", file->name);
      return LDPS_ERR;
    }
  } else {
    int64_t offset = 0;
    // Gold has found what might be IR part-way inside of a file, such as
    // an .a archive.
    if (file->offset) {
      offset = file->offset;
    }
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getOpenFileSlice(file->fd, file->name, file->filesize,
                                       offset);
    if (std::error_code EC = BufferOrErr.getError()) {
      message(LDPL_ERROR, EC.message().c_str());
      return LDPS_ERR;
    }
    buffer = std::move(BufferOrErr.get());
    view = buffer->getBufferStart();
  }

  if (!LTOModule::isBitcodeFile(view, file->filesize))
    return LDPS_OK;

  *claimed = 1;

  std::string Error;
  LTOModule *M =
      LTOModule::createFromBuffer(view, file->filesize, TargetOpts, Error);
  if (!M) {
    message(LDPL_ERROR, "LLVM gold plugin has failed to create LTO module: %s",
            Error.c_str());
    return LDPS_ERR;
  }

  Modules.resize(Modules.size() + 1);
  claimed_file &cf = Modules.back();

  if (!options::triple.empty())
    M->setTargetTriple(options::triple.c_str());

  cf.handle = file->handle;
  unsigned sym_count = M->getSymbolCount();
  cf.syms.reserve(sym_count);

  for (unsigned i = 0; i != sym_count; ++i) {
    lto_symbol_attributes attrs = M->getSymbolAttributes(i);
    if ((attrs & LTO_SYMBOL_SCOPE_MASK) == LTO_SYMBOL_SCOPE_INTERNAL)
      continue;

    cf.syms.push_back(ld_plugin_symbol());
    ld_plugin_symbol &sym = cf.syms.back();
    sym.name = strdup(M->getSymbolName(i));
    sym.version = nullptr;

    int scope = attrs & LTO_SYMBOL_SCOPE_MASK;
    bool CanBeHidden = scope == LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN;
    if (!CanBeHidden)
      CannotBeHidden.insert(sym.name);
    switch (scope) {
      case LTO_SYMBOL_SCOPE_HIDDEN:
        sym.visibility = LDPV_HIDDEN;
        break;
      case LTO_SYMBOL_SCOPE_PROTECTED:
        sym.visibility = LDPV_PROTECTED;
        break;
      case 0: // extern
      case LTO_SYMBOL_SCOPE_DEFAULT:
      case LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN:
        sym.visibility = LDPV_DEFAULT;
        break;
      default:
        message(LDPL_ERROR, "Unknown scope attribute: %d", scope);
        return LDPS_ERR;
    }

    int definition = attrs & LTO_SYMBOL_DEFINITION_MASK;
    sym.comdat_key = nullptr;
    switch (definition) {
      case LTO_SYMBOL_DEFINITION_REGULAR:
        sym.def = LDPK_DEF;
        break;
      case LTO_SYMBOL_DEFINITION_UNDEFINED:
        sym.def = LDPK_UNDEF;
        break;
      case LTO_SYMBOL_DEFINITION_TENTATIVE:
        sym.def = LDPK_COMMON;
        break;
      case LTO_SYMBOL_DEFINITION_WEAK:
        sym.comdat_key = sym.name;
        sym.def = LDPK_WEAKDEF;
        break;
      case LTO_SYMBOL_DEFINITION_WEAKUNDEF:
        sym.def = LDPK_WEAKUNDEF;
        break;
      default:
        message(LDPL_ERROR, "Unknown definition attribute: %d", definition);
        return LDPS_ERR;
    }

    sym.size = 0;

    sym.resolution = LDPR_UNKNOWN;
  }

  cf.syms.reserve(cf.syms.size());

  if (!cf.syms.empty()) {
    if (add_symbols(cf.handle, cf.syms.size(), &cf.syms[0]) != LDPS_OK) {
      message(LDPL_ERROR, "Unable to add symbols!");
      return LDPS_ERR;
    }
  }

  if (CodeGen) {
    std::string Error;
    if (!CodeGen->addModule(M, Error)) {
      message(LDPL_ERROR, "Error linking module: %s", Error.c_str());
      return LDPS_ERR;
    }
  }

  delete M;

  return LDPS_OK;
}

static bool mustPreserve(ld_plugin_symbol &Sym) {
  if (Sym.resolution == LDPR_PREVAILING_DEF)
    return true;
  if (Sym.resolution == LDPR_PREVAILING_DEF_IRONLY_EXP)
    return CannotBeHidden.count(Sym.name);
  return false;
}

/// gold informs us that all symbols have been read. At this point, we use
/// get_symbols to see if any of our definitions have been overridden by a
/// native object file. Then, perform optimization and codegen.
static ld_plugin_status all_symbols_read_hook(void) {
  // FIXME: raw_fd_ostream should be able to represent an unopened file.
  std::unique_ptr<raw_fd_ostream> api_file;

  assert(CodeGen);

  if (options::generate_api_file) {
    std::string Error;
    api_file.reset(new raw_fd_ostream("apifile.txt", Error, sys::fs::F_None));
    if (!Error.empty())
      message(LDPL_FATAL, "Unable to open apifile.txt for writing: %s",
              Error.c_str());
  }

  for (claimed_file &F : Modules) {
    if (F.syms.empty())
      continue;
    get_symbols(F.handle, F.syms.size(), &F.syms[0]);
    for (ld_plugin_symbol &Sym : F.syms) {
      if (mustPreserve(Sym)) {
        CodeGen->addMustPreserveSymbol(Sym.name);

        if (options::generate_api_file)
          (*api_file) << Sym.name << "\n";
      }
    }
  }

  CodeGen->setCodePICModel(output_type);
  CodeGen->setDebugInfo(LTO_DEBUG_MODEL_DWARF);
  if (!options::mcpu.empty())
    CodeGen->setCpu(options::mcpu.c_str());

  if (options::generate_bc_file != options::BC_NO) {
    std::string path;
    if (options::generate_bc_file == options::BC_ONLY)
      path = output_name;
    else if (!options::bc_path.empty())
      path = options::bc_path;
    else
      path = output_name + ".bc";
    std::string Error;
    if (!CodeGen->writeMergedModules(path.c_str(), Error))
      message(LDPL_FATAL, "Failed to write the output file.");
    if (options::generate_bc_file == options::BC_ONLY) {
      delete CodeGen;
      exit(0);
    }
  }

  std::string ObjPath;
  {
    const char *Temp;
    std::string Error;
    if (!CodeGen->compile_to_file(&Temp, /*DisableOpt*/ false, /*DisableInline*/
                                  false, /*DisableGVNLoadPRE*/ false, Error))
      message(LDPL_ERROR, "Could not produce a combined object file\n");
    ObjPath = Temp;
  }

  delete CodeGen;
  for (claimed_file &F : Modules) {
    for (ld_plugin_symbol &Sym : F.syms)
      free(Sym.name);
  }

  if (add_input_file(ObjPath.c_str()) != LDPS_OK) {
    message(LDPL_ERROR, "Unable to add .o file to the link.");
    message(LDPL_ERROR, "File left behind in: %s", ObjPath.c_str());
    return LDPS_ERR;
  }

  if (!options::extra_library_path.empty() &&
      set_extra_library_path(options::extra_library_path.c_str()) != LDPS_OK) {
    message(LDPL_ERROR, "Unable to set the extra library path.");
    return LDPS_ERR;
  }

  if (options::obj_path.empty())
    Cleanup.push_back(ObjPath);

  return LDPS_OK;
}

static ld_plugin_status cleanup_hook(void) {
  for (std::string &Name : Cleanup) {
    std::error_code EC = sys::fs::remove(Name);
    if (EC)
      message(LDPL_ERROR, "Failed to delete '%s': %s", Name.c_str(),
              EC.message().c_str());
  }

  return LDPS_OK;
}
