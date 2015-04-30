//===------ OrcLazyJIT.cpp - Basic Orc-based JIT for lazy execution -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcLazyJIT.h"
#include "llvm/ExecutionEngine/Orc/OrcTargetSupport.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include <cstdio>
#include <system_error>

using namespace llvm;

namespace {

  enum class DumpKind { NoDump, DumpFuncsToStdOut, DumpModsToStdErr,
                        DumpModsToDisk };

  cl::opt<DumpKind> OrcDumpKind("orc-lazy-debug",
                                cl::desc("Debug dumping for the orc-lazy JIT."),
                                cl::init(DumpKind::NoDump),
                                cl::values(
                                  clEnumValN(DumpKind::NoDump, "no-dump",
                                             "Don't dump anything."),
                                  clEnumValN(DumpKind::DumpFuncsToStdOut,
                                             "funcs-to-stdout",
                                             "Dump function names to stdout."),
                                  clEnumValN(DumpKind::DumpModsToStdErr,
                                             "mods-to-stderr",
                                             "Dump modules to stderr."),
                                  clEnumValN(DumpKind::DumpModsToDisk,
                                             "mods-to-disk",
                                             "Dump modules to the current "
                                             "working directory. (WARNING: "
                                             "will overwrite existing files)."),
                                  clEnumValEnd));
}

OrcLazyJIT::CallbackManagerBuilder
OrcLazyJIT::createCallbackManagerBuilder(Triple T) {
  switch (T.getArch()) {
    default: return nullptr;

    case Triple::x86_64: {
      typedef orc::JITCompileCallbackManager<IRDumpLayerT,
                                             orc::OrcX86_64> CCMgrT;
      return [](IRDumpLayerT &IRDumpLayer, RuntimeDyld::MemoryManager &MemMgr,
                LLVMContext &Context) {
               return llvm::make_unique<CCMgrT>(IRDumpLayer, MemMgr, Context, 0,
                                                64);
             };
    }
  }
}

OrcLazyJIT::TransformFtor OrcLazyJIT::createDebugDumper() {

  switch (OrcDumpKind) {
  case DumpKind::NoDump:
    return [](std::unique_ptr<Module> M) { return M; };

  case DumpKind::DumpFuncsToStdOut:
    return [](std::unique_ptr<Module> M) {
      printf("[ ");

      for (const auto &F : *M) {
        if (F.isDeclaration())
          continue;

        if (F.hasName()) {
          std::string Name(F.getName());
          printf("%s ", Name.c_str());
        } else
          printf("<anon> ");
      }

      printf("]\n");
      return M;
    };

  case DumpKind::DumpModsToStdErr:
    return [](std::unique_ptr<Module> M) {
             dbgs() << "----- Module Start -----\n" << *M
                    << "----- Module End -----\n";

             return M;
           };

  case DumpKind::DumpModsToDisk:
    return [](std::unique_ptr<Module> M) {
             std::error_code EC;
             raw_fd_ostream Out(M->getModuleIdentifier() + ".ll", EC,
                                sys::fs::F_Text);
             if (EC) {
               errs() << "Couldn't open " << M->getModuleIdentifier()
                      << " for dumping.\nError:" << EC.message() << "\n";
               exit(1);
             }
             Out << *M;
             return M;
           };
  }
  llvm_unreachable("Unknown DumpKind");
}

int llvm::runOrcLazyJIT(std::unique_ptr<Module> M, int ArgC, char* ArgV[]) {
  // Add the program's symbols into the JIT's search space.
  if (sys::DynamicLibrary::LoadLibraryPermanently(nullptr)) {
    errs() << "Error loading program symbols.\n";
    return 1;
  }

  // Grab a target machine and try to build a factory function for the
  // target-specific Orc callback manager.
  auto TM = std::unique_ptr<TargetMachine>(EngineBuilder().selectTarget());
  auto &Context = getGlobalContext();
  auto CallbackMgrBuilder =
    OrcLazyJIT::createCallbackManagerBuilder(Triple(TM->getTargetTriple()));

  // If we couldn't build the factory function then there must not be a callback
  // manager for this target. Bail out.
  if (!CallbackMgrBuilder) {
    errs() << "No callback manager available for target '"
           << TM->getTargetTriple() << "'.\n";
    return 1;
  }

  // Everything looks good. Build the JIT.
  OrcLazyJIT J(std::move(TM), Context, CallbackMgrBuilder);

  // Add the module, look up main and run it.
  auto MainHandle = J.addModule(std::move(M));
  auto MainSym = J.findSymbolIn(MainHandle, "main");

  if (!MainSym) {
    errs() << "Could not find main function.\n";
    return 1;
  }

  typedef int (*MainFnPtr)(int, char*[]);
  auto Main = OrcLazyJIT::fromTargetAddress<MainFnPtr>(MainSym.getAddress());
  return Main(ArgC, ArgV);
}
