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

#include "llvm/Config/config.h"
#include "plugin-api.h"

#include "llvm-c/lto.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Errno.h"
#include "llvm/System/Path.h"
#include "llvm/System/Program.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <vector>

using namespace llvm;

namespace {
  ld_plugin_status discard_message(int level, const char *format, ...) {
    // Die loudly. Recent versions of Gold pass ld_plugin_message as the first
    // callback in the transfer vector. This should never be called.
    abort();
  }

  ld_plugin_add_symbols add_symbols = NULL;
  ld_plugin_get_symbols get_symbols = NULL;
  ld_plugin_add_input_file add_input_file = NULL;
  ld_plugin_message message = discard_message;

  int api_version = 0;
  int gold_version = 0;

  bool generate_api_file = false;
  const char *as_path = NULL;

  struct claimed_file {
    lto_module_t M;
    void *handle;
    std::vector<ld_plugin_symbol> syms;
  };

  lto_codegen_model output_type = LTO_CODEGEN_PIC_MODEL_STATIC;
  std::list<claimed_file> Modules;
  std::vector<sys::Path> Cleanup;
}

ld_plugin_status claim_file_hook(const ld_plugin_input_file *file,
                                 int *claimed);
ld_plugin_status all_symbols_read_hook(void);
ld_plugin_status cleanup_hook(void);

extern "C" ld_plugin_status onload(ld_plugin_tv *tv);
ld_plugin_status onload(ld_plugin_tv *tv) {
  // We're given a pointer to the first transfer vector. We read through them
  // until we find one where tv_tag == LDPT_NULL. The REGISTER_* tagged values
  // contain pointers to functions that we need to call to register our own
  // hooks. The others are addresses of functions we can use to call into gold
  // for services.

  bool registeredClaimFile = false;
  bool registeredAllSymbolsRead = false;
  bool registeredCleanup = false;

  for (; tv->tv_tag != LDPT_NULL; ++tv) {
    switch (tv->tv_tag) {
      case LDPT_API_VERSION:
        api_version = tv->tv_u.tv_val;
        break;
      case LDPT_GOLD_VERSION:  // major * 100 + minor
        gold_version = tv->tv_u.tv_val;
        break;
      case LDPT_LINKER_OUTPUT:
        switch (tv->tv_u.tv_val) {
          case LDPO_REL:  // .o
          case LDPO_DYN:  // .so
            output_type = LTO_CODEGEN_PIC_MODEL_DYNAMIC;
            break;
          case LDPO_EXEC:  // .exe
            output_type = LTO_CODEGEN_PIC_MODEL_STATIC;
            break;
          default:
            (*message)(LDPL_ERROR, "Unknown output file type %d",
                       tv->tv_u.tv_val);
            return LDPS_ERR;
        }
        // TODO: add an option to disable PIC.
        //output_type = LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC;
        break;
      case LDPT_OPTION:
        if (strcmp("generate-api-file", tv->tv_u.tv_string) == 0) {
          generate_api_file = true;
        } else if (strncmp("as=", tv->tv_u.tv_string, 3) == 0) {
          if (as_path) {
            (*message)(LDPL_WARNING, "Path to as specified twice. "
                       "Discarding %s", tv->tv_u.tv_string);
          } else {
            as_path = strdup(tv->tv_u.tv_string + 3);
          }
        } else {
          (*message)(LDPL_WARNING, "Ignoring flag %s", tv->tv_u.tv_string);
        }
        break;
      case LDPT_REGISTER_CLAIM_FILE_HOOK: {
        ld_plugin_register_claim_file callback;
        callback = tv->tv_u.tv_register_claim_file;

        if ((*callback)(claim_file_hook) != LDPS_OK)
          return LDPS_ERR;

        registeredClaimFile = true;
      } break;
      case LDPT_REGISTER_ALL_SYMBOLS_READ_HOOK: {
        ld_plugin_register_all_symbols_read callback;
        callback = tv->tv_u.tv_register_all_symbols_read;

        if ((*callback)(all_symbols_read_hook) != LDPS_OK)
          return LDPS_ERR;

        registeredAllSymbolsRead = true;
      } break;
      case LDPT_REGISTER_CLEANUP_HOOK: {
        ld_plugin_register_cleanup callback;
        callback = tv->tv_u.tv_register_cleanup;

        if ((*callback)(cleanup_hook) != LDPS_OK)
          return LDPS_ERR;

        registeredCleanup = true;
      } break;
      case LDPT_ADD_SYMBOLS:
        add_symbols = tv->tv_u.tv_add_symbols;
        break;
      case LDPT_GET_SYMBOLS:
        get_symbols = tv->tv_u.tv_get_symbols;
        break;
      case LDPT_ADD_INPUT_FILE:
        add_input_file = tv->tv_u.tv_add_input_file;
        break;
      case LDPT_MESSAGE:
        message = tv->tv_u.tv_message;
        break;
      default:
        break;
    }
  }

  if (!registeredClaimFile) {
    (*message)(LDPL_ERROR, "register_claim_file not passed to LLVMgold.");
    return LDPS_ERR;
  }
  if (!add_symbols) {
    (*message)(LDPL_ERROR, "add_symbols not passed to LLVMgold.");
    return LDPS_ERR;
  }

  return LDPS_OK;
}

/// claim_file_hook - called by gold to see whether this file is one that
/// our plugin can handle. We'll try to open it and register all the symbols
/// with add_symbol if possible.
ld_plugin_status claim_file_hook(const ld_plugin_input_file *file,
                                 int *claimed) {
  void *buf = NULL;
  if (file->offset) {
    // Gold has found what might be IR part-way inside of a file, such as
    // an .a archive.
    if (lseek(file->fd, file->offset, SEEK_SET) == -1) {
      (*message)(LDPL_ERROR,
                 "Failed to seek to archive member of %s at offset %d: %s\n", 
                 file->name,
                 file->offset, sys::StrError(errno).c_str());
      return LDPS_ERR;
    }
    buf = malloc(file->filesize);
    if (!buf) {
      (*message)(LDPL_ERROR,
                 "Failed to allocate buffer for archive member of size: %d\n", 
                 file->filesize);
      return LDPS_ERR;
    }
    if (read(file->fd, buf, file->filesize) != file->filesize) {
      (*message)(LDPL_ERROR,
                 "Failed to read archive member of %s at offset %d: %s\n",
                 file->name,
                 file->offset,
                 sys::StrError(errno).c_str());
      free(buf);
      return LDPS_ERR;
    }
    if (!lto_module_is_object_file_in_memory(buf, file->filesize)) {
      free(buf);
      return LDPS_OK;
    }
  } else if (!lto_module_is_object_file(file->name))
    return LDPS_OK;

  *claimed = 1;
  Modules.resize(Modules.size() + 1);
  claimed_file &cf = Modules.back();

  cf.M = buf ? lto_module_create_from_memory(buf, file->filesize) :
               lto_module_create(file->name);
  free(buf);
  if (!cf.M) {
    (*message)(LDPL_ERROR, "Failed to create LLVM module: %s",
               lto_get_error_message());
    return LDPS_ERR;
  }
  cf.handle = file->handle;
  unsigned sym_count = lto_module_get_num_symbols(cf.M);
  cf.syms.reserve(sym_count);

  for (unsigned i = 0; i != sym_count; ++i) {
    lto_symbol_attributes attrs = lto_module_get_symbol_attribute(cf.M, i);
    if ((attrs & LTO_SYMBOL_SCOPE_MASK) == LTO_SYMBOL_SCOPE_INTERNAL)
      continue;

    cf.syms.push_back(ld_plugin_symbol());
    ld_plugin_symbol &sym = cf.syms.back();
    sym.name = const_cast<char *>(lto_module_get_symbol_name(cf.M, i));
    sym.version = NULL;

    int scope = attrs & LTO_SYMBOL_SCOPE_MASK;
    switch (scope) {
      case LTO_SYMBOL_SCOPE_HIDDEN:
        sym.visibility = LDPV_HIDDEN;
        break;
      case LTO_SYMBOL_SCOPE_PROTECTED:
        sym.visibility = LDPV_PROTECTED;
        break;
      case 0: // extern
      case LTO_SYMBOL_SCOPE_DEFAULT:
        sym.visibility = LDPV_DEFAULT;
        break;
      default:
        (*message)(LDPL_ERROR, "Unknown scope attribute: %d", scope);
        return LDPS_ERR;
    }

    int definition = attrs & LTO_SYMBOL_DEFINITION_MASK;
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
        sym.def = LDPK_WEAKDEF;
        break;
      case LTO_SYMBOL_DEFINITION_WEAKUNDEF:
        sym.def = LDPK_WEAKUNDEF;
        break;
      default:
        (*message)(LDPL_ERROR, "Unknown definition attribute: %d", definition);
        return LDPS_ERR;
    }

    // LLVM never emits COMDAT.
    sym.size = 0;
    sym.comdat_key = NULL;

    sym.resolution = LDPR_UNKNOWN;
  }

  cf.syms.reserve(cf.syms.size());

  if (!cf.syms.empty()) {
    if ((*add_symbols)(cf.handle, cf.syms.size(), &cf.syms[0]) != LDPS_OK) {
      (*message)(LDPL_ERROR, "Unable to add symbols!");
      return LDPS_ERR;
    }
  }

  return LDPS_OK;
}

/// all_symbols_read_hook - gold informs us that all symbols have been read.
/// At this point, we use get_symbols to see if any of our definitions have
/// been overridden by a native object file. Then, perform optimization and
/// codegen.
ld_plugin_status all_symbols_read_hook(void) {
  lto_code_gen_t cg = lto_codegen_create();

  for (std::list<claimed_file>::iterator I = Modules.begin(),
       E = Modules.end(); I != E; ++I)
    lto_codegen_add_module(cg, I->M);

  std::ofstream api_file;
  if (generate_api_file) {
    api_file.open("apifile.txt", std::ofstream::out | std::ofstream::trunc);
    if (!api_file.is_open()) {
      (*message)(LDPL_FATAL, "Unable to open apifile.txt for writing.");
      abort();
    }
  }

  // If we don't preserve any symbols, libLTO will assume that all symbols are
  // needed. Keep all symbols unless we're producing a final executable.
  if (output_type == LTO_CODEGEN_PIC_MODEL_STATIC) {
    bool anySymbolsPreserved = false;
    for (std::list<claimed_file>::iterator I = Modules.begin(),
         E = Modules.end(); I != E; ++I) {
      (*get_symbols)(I->handle, I->syms.size(), &I->syms[0]);
      for (unsigned i = 0, e = I->syms.size(); i != e; i++) {
        if (I->syms[i].resolution == LDPR_PREVAILING_DEF ||
            (I->syms[i].def == LDPK_COMMON &&
             I->syms[i].resolution == LDPR_RESOLVED_IR)) {
          lto_codegen_add_must_preserve_symbol(cg, I->syms[i].name);
          anySymbolsPreserved = true;

          if (generate_api_file)
            api_file << I->syms[i].name << "\n";
        }
      }
    }

    if (generate_api_file)
      api_file.close();

    if (!anySymbolsPreserved) {
      // This entire file is unnecessary!
      lto_codegen_dispose(cg);
      return LDPS_OK;
    }
  }

  lto_codegen_set_pic_model(cg, output_type);
  lto_codegen_set_debug_model(cg, LTO_DEBUG_MODEL_DWARF);
  if (as_path) {
    sys::Path p = sys::Program::FindProgramByName(as_path);
    lto_codegen_set_assembler_path(cg, p.c_str());
  }

  size_t bufsize = 0;
  const char *buffer = static_cast<const char *>(lto_codegen_compile(cg,
                                                                     &bufsize));

  std::string ErrMsg;

  sys::Path uniqueObjPath("/tmp/llvmgold.o");
  if (uniqueObjPath.createTemporaryFileOnDisk(true, &ErrMsg)) {
    (*message)(LDPL_ERROR, "%s", ErrMsg.c_str());
    return LDPS_ERR;
  }
  raw_fd_ostream *objFile = 
    new raw_fd_ostream(uniqueObjPath.c_str(), ErrMsg,
                       raw_fd_ostream::F_Binary);
  if (!ErrMsg.empty()) {
    delete objFile;
    (*message)(LDPL_ERROR, "%s", ErrMsg.c_str());
    return LDPS_ERR;
  }

  objFile->write(buffer, bufsize);
  objFile->close();

  lto_codegen_dispose(cg);

  if ((*add_input_file)(const_cast<char*>(uniqueObjPath.c_str())) != LDPS_OK) {
    (*message)(LDPL_ERROR, "Unable to add .o file to the link.");
    (*message)(LDPL_ERROR, "File left behind in: %s", uniqueObjPath.c_str());
    return LDPS_ERR;
  }

  Cleanup.push_back(uniqueObjPath);

  return LDPS_OK;
}

ld_plugin_status cleanup_hook(void) {
  std::string ErrMsg;

  for (int i = 0, e = Cleanup.size(); i != e; ++i)
    if (Cleanup[i].eraseFromDisk(false, &ErrMsg))
      (*message)(LDPL_ERROR, "Failed to delete '%s': %s", Cleanup[i].c_str(),
                 ErrMsg.c_str());

  return LDPS_OK;
}
