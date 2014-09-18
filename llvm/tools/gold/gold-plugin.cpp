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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/PassManager.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
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

static ld_plugin_get_input_file get_input_file = nullptr;
static ld_plugin_release_input_file release_input_file = nullptr;
static ld_plugin_add_symbols add_symbols = nullptr;
static ld_plugin_get_symbols get_symbols = nullptr;
static ld_plugin_add_input_file add_input_file = nullptr;
static ld_plugin_set_extra_library_path set_extra_library_path = nullptr;
static ld_plugin_get_view get_view = nullptr;
static ld_plugin_message message = discard_message;
static Reloc::Model RelocationModel = Reloc::Default;
static std::string output_name = "";
static std::list<claimed_file> Modules;
static std::vector<std::string> Cleanup;
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
      // ParseCommandLineOptions() expects argv[0] to be program name. Lazily
      // add that.
      if (extra.empty())
        extra.push_back("LLVMgold");

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
            RelocationModel = Reloc::PIC_;
            break;
          case LDPO_EXEC:  // .exe
            RelocationModel = Reloc::Static;
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
      case LDPT_GET_INPUT_FILE:
        get_input_file = tv->tv_u.tv_get_input_file;
        break;
      case LDPT_RELEASE_INPUT_FILE:
        release_input_file = tv->tv_u.tv_release_input_file;
        break;
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

  if (!get_input_file) {
    message(LDPL_ERROR, "get_input_file not passed to LLVMgold.");
    return LDPS_ERR;
  }
  if (!release_input_file) {
    message(LDPL_ERROR, "relesase_input_file not passed to LLVMgold.");
    return LDPS_ERR;
  }

  return LDPS_OK;
}

static const GlobalObject *getBaseObject(const GlobalValue &GV) {
  if (auto *GA = dyn_cast<GlobalAlias>(&GV))
    return GA->getBaseObject();
  return cast<GlobalObject>(&GV);
}

/// Called by gold to see whether this file is one that our plugin can handle.
/// We'll try to open it and register all the symbols with add_symbol if
/// possible.
static ld_plugin_status claim_file_hook(const ld_plugin_input_file *file,
                                        int *claimed) {
  LLVMContext Context;
  MemoryBufferRef BufferRef;
  std::unique_ptr<MemoryBuffer> Buffer;
  if (get_view) {
    const void *view;
    if (get_view(file->handle, &view) != LDPS_OK) {
      message(LDPL_ERROR, "Failed to get a view of %s", file->name);
      return LDPS_ERR;
    }
    BufferRef = MemoryBufferRef(StringRef((char *)view, file->filesize), "");
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
    Buffer = std::move(BufferOrErr.get());
    BufferRef = Buffer->getMemBufferRef();
  }

  ErrorOr<std::unique_ptr<object::IRObjectFile>> ObjOrErr =
      object::IRObjectFile::createIRObjectFile(BufferRef, Context);
  std::error_code EC = ObjOrErr.getError();
  if (EC == BitcodeError::InvalidBitcodeSignature ||
      EC == object::object_error::invalid_file_type ||
      EC == object::object_error::bitcode_section_not_found)
    return LDPS_OK;

  *claimed = 1;

  if (EC) {
    message(LDPL_ERROR, "LLVM gold plugin has failed to create LTO module: %s",
            EC.message().c_str());
    return LDPS_ERR;
  }
  std::unique_ptr<object::IRObjectFile> Obj = std::move(*ObjOrErr);

  Modules.resize(Modules.size() + 1);
  claimed_file &cf = Modules.back();

  cf.handle = file->handle;

  for (auto &Sym : Obj->symbols()) {
    uint32_t Symflags = Sym.getFlags();
    if (!(Symflags & object::BasicSymbolRef::SF_Global))
      continue;

    if (Symflags & object::BasicSymbolRef::SF_FormatSpecific)
      continue;

    cf.syms.push_back(ld_plugin_symbol());
    ld_plugin_symbol &sym = cf.syms.back();
    sym.version = nullptr;

    SmallString<64> Name;
    {
      raw_svector_ostream OS(Name);
      Sym.printName(OS);
    }
    sym.name = strdup(Name.c_str());

    const GlobalValue *GV = Obj->getSymbolGV(Sym.getRawDataRefImpl());

    sym.visibility = LDPV_DEFAULT;
    if (GV) {
      switch (GV->getVisibility()) {
      case GlobalValue::DefaultVisibility:
        sym.visibility = LDPV_DEFAULT;
        break;
      case GlobalValue::HiddenVisibility:
        sym.visibility = LDPV_HIDDEN;
        break;
      case GlobalValue::ProtectedVisibility:
        sym.visibility = LDPV_PROTECTED;
        break;
      }
    }

    if (Symflags & object::BasicSymbolRef::SF_Undefined) {
      sym.def = LDPK_UNDEF;
      if (GV && GV->hasExternalWeakLinkage())
        sym.def = LDPK_WEAKUNDEF;
    } else {
      sym.def = LDPK_DEF;
      if (GV) {
        assert(!GV->hasExternalWeakLinkage() &&
               !GV->hasAvailableExternallyLinkage() && "Not a declaration!");
        if (GV->hasCommonLinkage())
          sym.def = LDPK_COMMON;
        else if (GV->isWeakForLinker())
          sym.def = LDPK_WEAKDEF;
      }
    }

    sym.size = 0;
    sym.comdat_key = nullptr;
    if (GV) {
      const GlobalObject *Base = getBaseObject(*GV);
      if (!Base)
        message(LDPL_FATAL, "Unable to determine comdat of alias!");
      const Comdat *C = Base->getComdat();
      if (C)
        sym.comdat_key = strdup(C->getName().str().c_str());
      else if (Base->hasWeakLinkage() || Base->hasLinkOnceLinkage())
        sym.comdat_key = strdup(sym.name);
    }

    sym.resolution = LDPR_UNKNOWN;
  }

  if (!cf.syms.empty()) {
    if (add_symbols(cf.handle, cf.syms.size(), &cf.syms[0]) != LDPS_OK) {
      message(LDPL_ERROR, "Unable to add symbols!");
      return LDPS_ERR;
    }
  }

  return LDPS_OK;
}

static void keepGlobalValue(GlobalValue &GV,
                            std::vector<GlobalAlias *> &KeptAliases) {
  assert(!GV.hasLocalLinkage());

  if (auto *GA = dyn_cast<GlobalAlias>(&GV))
    KeptAliases.push_back(GA);

  switch (GV.getLinkage()) {
  default:
    break;
  case GlobalValue::LinkOnceAnyLinkage:
    GV.setLinkage(GlobalValue::WeakAnyLinkage);
    break;
  case GlobalValue::LinkOnceODRLinkage:
    GV.setLinkage(GlobalValue::WeakODRLinkage);
    break;
  }

  assert(!GV.isDiscardableIfUnused());
}

static bool isDeclaration(const GlobalValue &V) {
  if (V.hasAvailableExternallyLinkage())
    return true;

  if (V.isMaterializable())
    return false;

  return V.isDeclaration();
}

static void internalize(GlobalValue &GV) {
  if (isDeclaration(GV))
    return; // We get here if there is a matching asm definition.
  if (!GV.hasLocalLinkage())
    GV.setLinkage(GlobalValue::InternalLinkage);
}

static void drop(GlobalValue &GV) {
  if (auto *F = dyn_cast<Function>(&GV)) {
    F->deleteBody();
    F->setComdat(nullptr); // Should deleteBody do this?
    return;
  }

  if (auto *Var = dyn_cast<GlobalVariable>(&GV)) {
    Var->setInitializer(nullptr);
    Var->setLinkage(
        GlobalValue::ExternalLinkage); // Should setInitializer do this?
    Var->setComdat(nullptr); // and this?
    return;
  }

  auto &Alias = cast<GlobalAlias>(GV);
  Module &M = *Alias.getParent();
  PointerType &Ty = *cast<PointerType>(Alias.getType());
  GlobalValue::LinkageTypes L = Alias.getLinkage();
  auto *Var =
      new GlobalVariable(M, Ty.getElementType(), /*isConstant*/ false, L,
                         /*Initializer*/ nullptr);
  Var->takeName(&Alias);
  Alias.replaceAllUsesWith(Var);
  Alias.eraseFromParent();
}

static const char *getResolutionName(ld_plugin_symbol_resolution R) {
  switch (R) {
  case LDPR_UNKNOWN:
    return "UNKNOWN";
  case LDPR_UNDEF:
    return "UNDEF";
  case LDPR_PREVAILING_DEF:
    return "PREVAILING_DEF";
  case LDPR_PREVAILING_DEF_IRONLY:
    return "PREVAILING_DEF_IRONLY";
  case LDPR_PREEMPTED_REG:
    return "PREEMPTED_REG";
  case LDPR_PREEMPTED_IR:
    return "PREEMPTED_IR";
  case LDPR_RESOLVED_IR:
    return "RESOLVED_IR";
  case LDPR_RESOLVED_EXEC:
    return "RESOLVED_EXEC";
  case LDPR_RESOLVED_DYN:
    return "RESOLVED_DYN";
  case LDPR_PREVAILING_DEF_IRONLY_EXP:
    return "PREVAILING_DEF_IRONLY_EXP";
  }
}

static GlobalObject *makeInternalReplacement(GlobalObject *GO) {
  Module *M = GO->getParent();
  GlobalObject *Ret;
  if (auto *F = dyn_cast<Function>(GO)) {
    auto *NewF = Function::Create(
        F->getFunctionType(), GlobalValue::InternalLinkage, F->getName(), M);
    NewF->getBasicBlockList().splice(NewF->end(), F->getBasicBlockList());
    Ret = NewF;
    F->deleteBody();
  } else {
    auto *Var = cast<GlobalVariable>(GO);
    Ret = new GlobalVariable(
        *M, Var->getType()->getElementType(), Var->isConstant(),
        GlobalValue::InternalLinkage, Var->getInitializer(), Var->getName(),
        nullptr, Var->getThreadLocalMode(), Var->getType()->getAddressSpace(),
        Var->isExternallyInitialized());
    Var->setInitializer(nullptr);
  }
  Ret->copyAttributesFrom(GO);
  Ret->setComdat(GO->getComdat());

  return Ret;
}

namespace {
class LocalValueMaterializer : public ValueMaterializer {
  DenseSet<GlobalValue *> &Dropped;

public:
  LocalValueMaterializer(DenseSet<GlobalValue *> &Dropped) : Dropped(Dropped) {}
  Value *materializeValueFor(Value *V) override;
};
}

Value *LocalValueMaterializer::materializeValueFor(Value *V) {
  auto *GV = dyn_cast<GlobalValue>(V);
  if (!GV)
    return nullptr;
  if (!Dropped.count(GV))
    return nullptr;
  assert(!isa<GlobalAlias>(GV) && "Found alias point to weak alias.");
  return makeInternalReplacement(cast<GlobalObject>(GV));
}

static Constant *mapConstantToLocalCopy(Constant *C, ValueToValueMapTy &VM,
                                        LocalValueMaterializer *Materializer) {
  return MapValue(C, VM, RF_IgnoreMissingEntries, nullptr, Materializer);
}

static std::unique_ptr<Module>
getModuleForFile(LLVMContext &Context, claimed_file &F, raw_fd_ostream *ApiFile,
                 StringSet<> &Internalize, StringSet<> &Maybe) {
  ld_plugin_input_file File;
  if (get_input_file(F.handle, &File) != LDPS_OK)
    message(LDPL_FATAL, "Failed to get file information");

  if (get_symbols(F.handle, F.syms.size(), &F.syms[0]) != LDPS_OK)
    message(LDPL_FATAL, "Failed to get symbol information");

  const void *View;
  if (get_view(F.handle, &View) != LDPS_OK)
    message(LDPL_FATAL, "Failed to get a view of file");

  llvm::ErrorOr<MemoryBufferRef> MBOrErr =
      object::IRObjectFile::findBitcodeInMemBuffer(
          MemoryBufferRef(StringRef((const char *)View, File.filesize), ""));
  if (std::error_code EC = MBOrErr.getError())
    message(LDPL_FATAL, "Could not read bitcode from file : %s",
            EC.message().c_str());

  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer(MBOrErr->getBuffer(), "", false);

  if (release_input_file(F.handle) != LDPS_OK)
    message(LDPL_FATAL, "Failed to release file information");

  ErrorOr<Module *> MOrErr = getLazyBitcodeModule(std::move(Buffer), Context);

  if (std::error_code EC = MOrErr.getError())
    message(LDPL_FATAL, "Could not read bitcode from file : %s",
            EC.message().c_str());

  std::unique_ptr<Module> M(MOrErr.get());

  SmallPtrSet<GlobalValue *, 8> Used;
  collectUsedGlobalVariables(*M, Used, /*CompilerUsed*/ false);

  DenseSet<GlobalValue *> Drop;
  std::vector<GlobalAlias *> KeptAliases;
  for (ld_plugin_symbol &Sym : F.syms) {
    ld_plugin_symbol_resolution Resolution =
        (ld_plugin_symbol_resolution)Sym.resolution;

    if (options::generate_api_file)
      *ApiFile << Sym.name << ' ' << getResolutionName(Resolution) << '\n';

    GlobalValue *GV = M->getNamedValue(Sym.name);
    if (!GV)
      continue; // Asm symbol.

    if (Resolution != LDPR_PREVAILING_DEF_IRONLY && GV->hasCommonLinkage()) {
      // Common linkage is special. There is no single symbol that wins the
      // resolution. Instead we have to collect the maximum alignment and size.
      // The IR linker does that for us if we just pass it every common GV.
      // We still have to keep track of LDPR_PREVAILING_DEF_IRONLY so we
      // internalize once the IR linker has done its job.
      continue;
    }

    switch (Resolution) {
    case LDPR_UNKNOWN:
      llvm_unreachable("Unexpected resolution");

    case LDPR_RESOLVED_IR:
    case LDPR_RESOLVED_EXEC:
    case LDPR_RESOLVED_DYN:
    case LDPR_UNDEF:
      assert(isDeclaration(*GV));
      break;

    case LDPR_PREVAILING_DEF_IRONLY: {
      keepGlobalValue(*GV, KeptAliases);
      if (!Used.count(GV)) {
        // Since we use the regular lib/Linker, we cannot just internalize GV
        // now or it will not be copied to the merged module. Instead we force
        // it to be copied and then internalize it.
        Internalize.insert(Sym.name);
      }
      break;
    }

    case LDPR_PREVAILING_DEF:
      keepGlobalValue(*GV, KeptAliases);
      break;

    case LDPR_PREEMPTED_REG:
    case LDPR_PREEMPTED_IR:
      Drop.insert(GV);
      break;

    case LDPR_PREVAILING_DEF_IRONLY_EXP: {
      // We can only check for address uses after we merge the modules. The
      // reason is that this GV might have a copy in another module
      // and in that module the address might be significant, but that
      // copy will be LDPR_PREEMPTED_IR.
      if (GV->hasLinkOnceODRLinkage())
        Maybe.insert(Sym.name);
      keepGlobalValue(*GV, KeptAliases);
      break;
    }
    }

    free(Sym.name);
    free(Sym.comdat_key);
    Sym.name = nullptr;
    Sym.comdat_key = nullptr;
  }

  if (!Drop.empty())
    // This is horrible. Given how lazy loading is implemented, dropping
    // the body while there is a materializer present doesn't work, the
    // linker will just read the body back.
    M->materializeAllPermanently();

  ValueToValueMapTy VM;
  LocalValueMaterializer Materializer(Drop);
  for (GlobalAlias *GA : KeptAliases) {
    // Gold told us to keep GA. It is possible that a GV usied in the aliasee
    // expression is being dropped. If that is the case, that GV must be copied.
    Constant *Aliasee = GA->getAliasee();
    Constant *Replacement = mapConstantToLocalCopy(Aliasee, VM, &Materializer);
    if (Aliasee != Replacement)
      GA->setAliasee(Replacement);
  }

  for (auto *GV : Drop)
    drop(*GV);

  return M;
}

static void runLTOPasses(Module &M, TargetMachine &TM) {
  PassManager passes;
  PassManagerBuilder PMB;
  PMB.LibraryInfo = new TargetLibraryInfo(Triple(TM.getTargetTriple()));
  PMB.Inliner = createFunctionInliningPass();
  PMB.VerifyInput = true;
  PMB.VerifyOutput = true;
  PMB.populateLTOPassManager(passes, &TM);
  passes.run(M);
}

static void codegen(Module &M) {
  const std::string &TripleStr = M.getTargetTriple();
  Triple TheTriple(TripleStr);

  std::string ErrMsg;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleStr, ErrMsg);
  if (!TheTarget)
    message(LDPL_FATAL, "Target not found: %s", ErrMsg.c_str());

  if (unsigned NumOpts = options::extra.size())
    cl::ParseCommandLineOptions(NumOpts, &options::extra[0]);

  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(TheTriple);
  for (const std::string &A : MAttrs)
    Features.AddFeature(A);

  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine(
      TripleStr, options::mcpu, Features.getString(), Options, RelocationModel,
      CodeModel::Default, CodeGenOpt::Aggressive));

  runLTOPasses(M, *TM);

  PassManager CodeGenPasses;
  CodeGenPasses.add(new DataLayoutPass());

  SmallString<128> Filename;
  int FD;
  if (options::obj_path.empty()) {
    std::error_code EC =
        sys::fs::createTemporaryFile("lto-llvm", "o", FD, Filename);
    if (EC)
      message(LDPL_FATAL, "Could not create temorary file: %s",
              EC.message().c_str());
  } else {
    Filename = options::obj_path;
    std::error_code EC =
        sys::fs::openFileForWrite(Filename.c_str(), FD, sys::fs::F_None);
    if (EC)
      message(LDPL_FATAL, "Could not open file: %s", EC.message().c_str());
  }

  {
    raw_fd_ostream OS(FD, true);
    formatted_raw_ostream FOS(OS);

    if (TM->addPassesToEmitFile(CodeGenPasses, FOS,
                                TargetMachine::CGFT_ObjectFile))
      message(LDPL_FATAL, "Failed to setup codegen");
    CodeGenPasses.run(M);
  }

  if (add_input_file(Filename.c_str()) != LDPS_OK)
    message(LDPL_FATAL,
            "Unable to add .o file to the link. File left behind in: %s",
            Filename.c_str());

  if (options::obj_path.empty())
    Cleanup.push_back(Filename.c_str());
}

/// gold informs us that all symbols have been read. At this point, we use
/// get_symbols to see if any of our definitions have been overridden by a
/// native object file. Then, perform optimization and codegen.
static ld_plugin_status allSymbolsReadHook(raw_fd_ostream *ApiFile) {
  if (Modules.empty())
    return LDPS_OK;

  LLVMContext Context;
  std::unique_ptr<Module> Combined(new Module("ld-temp.o", Context));
  Linker L(Combined.get());

  std::string DefaultTriple = sys::getDefaultTargetTriple();

  StringSet<> Internalize;
  StringSet<> Maybe;
  for (claimed_file &F : Modules) {
    std::unique_ptr<Module> M =
        getModuleForFile(Context, F, ApiFile, Internalize, Maybe);
    if (!options::triple.empty())
      M->setTargetTriple(options::triple.c_str());
    else if (M->getTargetTriple().empty()) {
      M->setTargetTriple(DefaultTriple);
    }

    std::string ErrMsg;
    if (L.linkInModule(M.get(), &ErrMsg))
      message(LDPL_FATAL, "Failed to link module: %s", ErrMsg.c_str());
  }

  for (const auto &Name : Internalize) {
    GlobalValue *GV = Combined->getNamedValue(Name.first());
    if (GV)
      internalize(*GV);
  }

  for (const auto &Name : Maybe) {
    GlobalValue *GV = Combined->getNamedValue(Name.first());
    if (!GV)
      continue;
    GV->setLinkage(GlobalValue::LinkOnceODRLinkage);
    if (canBeOmittedFromSymbolTable(GV))
      internalize(*GV);
  }

  if (options::generate_bc_file != options::BC_NO) {
    std::string path;
    if (options::generate_bc_file == options::BC_ONLY)
      path = output_name;
    else if (!options::bc_path.empty())
      path = options::bc_path;
    else
      path = output_name + ".bc";
    {
      std::error_code EC;
      raw_fd_ostream OS(path, EC, sys::fs::OpenFlags::F_None);
      if (EC)
        message(LDPL_FATAL, "Failed to write the output file.");
      WriteBitcodeToFile(L.getModule(), OS);
    }
    if (options::generate_bc_file == options::BC_ONLY)
      return LDPS_OK;
  }

  codegen(*L.getModule());

  if (!options::extra_library_path.empty() &&
      set_extra_library_path(options::extra_library_path.c_str()) != LDPS_OK)
    message(LDPL_FATAL, "Unable to set the extra library path.");

  return LDPS_OK;
}

static ld_plugin_status all_symbols_read_hook(void) {
  ld_plugin_status Ret;
  if (!options::generate_api_file) {
    Ret = allSymbolsReadHook(nullptr);
  } else {
    std::error_code EC;
    raw_fd_ostream ApiFile("apifile.txt", EC, sys::fs::F_None);
    if (EC)
      message(LDPL_FATAL, "Unable to open apifile.txt for writing: %s",
              EC.message().c_str());
    Ret = allSymbolsReadHook(&ApiFile);
  }

  if (options::generate_bc_file == options::BC_ONLY)
    exit(0);

  return Ret;
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
