// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "third_party/llvm/clang_cc1.h"

#include <stdlib.h>

#include <memory>
#include <utility>

#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/CodeGen/ObjectFilePCHContainerWriter.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "common/check.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/install_paths.h"

namespace Carbon {

auto RunClangCC1(const InstallPaths& installation,
                 llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                 llvm::SmallVectorImpl<const char*>& cc1_args,
                 bool enable_leaking) -> int {
  llvm::BumpPtrAllocator allocator;
  llvm::cl::ExpansionContext expansion_context(
      allocator, llvm::cl::TokenizeGNUCommandLine);
  if (llvm::Error error = expansion_context.expandResponseFiles(cc1_args)) {
    llvm::errs() << toString(std::move(error)) << '\n';
    return 1;
  }
  CARBON_CHECK(cc1_args[1] == llvm::StringRef("-cc1"));

  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids =
      clang::DiagnosticIDs::create();

  // Register the support for object-file-wrapped Clang modules.
  auto pch_ops = std::make_shared<clang::PCHContainerOperations>();
  pch_ops->registerWriter(
      std::make_unique<clang::ObjectFilePCHContainerWriter>());
  pch_ops->registerReader(
      std::make_unique<clang::ObjectFilePCHContainerReader>());

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  clang::DiagnosticOptions diag_opts;
  clang::TextDiagnosticBuffer diag_buffer;
  clang::DiagnosticsEngine diags(diag_ids, diag_opts, &diag_buffer,
                                 /*ShouldOwnClient=*/false);

  // Setup round-trip remarks for the DiagnosticsEngine used in CreateFromArgs.
  if (llvm::find(cc1_args, llvm::StringRef("-Rround-trip-cc1-args")) !=
      cc1_args.end()) {
    diags.setSeverity(clang::diag::remark_cc1_round_trip_generated,
                      clang::diag::Severity::Remark, {});
  }

  auto invocation = std::make_shared<clang::CompilerInvocation>();
  bool success = clang::CompilerInvocation::CreateFromArgs(
      *invocation, llvm::ArrayRef(cc1_args).slice(1), diags, cc1_args[0]);

  // Heap allocate the compiler instance so that if we disable freeing we can
  // discard the pointer without destroying or deallocating it.
  auto clang_instance = std::make_unique<clang::CompilerInstance>(
      std::move(invocation), std::move(pch_ops));

  // Override the disabling of free when we don't want to leak memory.
  if (!enable_leaking) {
    clang_instance->getFrontendOpts().DisableFree = false;
    clang_instance->getCodeGenOpts().DisableFree = false;
  }

  if (!clang_instance->getFrontendOpts().TimeTracePath.empty()) {
    llvm::timeTraceProfilerInitialize(
        clang_instance->getFrontendOpts().TimeTraceGranularity, cc1_args[0],
        clang_instance->getFrontendOpts().TimeTraceVerbose);
  }

  // TODO: These options should take priority over the actual compilation.
  // However, their implementation is currently not accessible from a library.
  // We should factor the implementation into a reusable location and then use
  // that here.
  CARBON_CHECK(!clang_instance->getFrontendOpts().PrintSupportedCPUs &&
               !clang_instance->getFrontendOpts().PrintSupportedExtensions &&
               !clang_instance->getFrontendOpts().PrintEnabledExtensions);

  // Infer the builtin include path if unspecified.
  if (clang_instance->getHeaderSearchOpts().UseBuiltinIncludes &&
      clang_instance->getHeaderSearchOpts().ResourceDir.empty()) {
    clang_instance->getHeaderSearchOpts().ResourceDir =
        installation.clang_resource_path();
  }

  // Create the filesystem.
  clang_instance->createVirtualFileSystem(std::move(fs), &diag_buffer);

  // Create the actual diagnostics engine.
  clang_instance->createDiagnostics();
  if (!clang_instance->hasDiagnostics()) {
    return EXIT_FAILURE;
  }

  // Now flush the buffered diagnostics into the Clang instance's diagnostic
  // engine. If we've already hit an error, we can exit early once that's done.
  diag_buffer.FlushDiagnostics(clang_instance->getDiagnostics());
  if (!success) {
    return EXIT_FAILURE;
  }

  // Execute the frontend actions.
  {
    llvm::TimeTraceScope time_scope("ExecuteCompiler");
    bool time_passes = clang_instance->getCodeGenOpts().TimePasses;
    if (time_passes) {
      clang_instance->createFrontendTimer();
    }
    llvm::TimeRegion timer(time_passes ? &clang_instance->getFrontendTimer()
                                       : nullptr);
    success = clang::ExecuteCompilerInvocation(clang_instance.get());
  }

  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  std::unique_ptr<llvm::raw_ostream> io_file = llvm::CreateInfoOutputFile();
  if (clang_instance->getCodeGenOpts().TimePassesJson) {
    *io_file << "{\n";
    llvm::TimerGroup::printAllJSONValues(*io_file, "");
    *io_file << "\n}\n";
  } else if (!clang_instance->getCodeGenOpts().TimePassesStatsFile) {
    llvm::TimerGroup::printAll(*io_file);
  }
  llvm::TimerGroup::clearAll();

  if (llvm::timeTraceProfilerEnabled()) {
    // It is possible that the compiler instance doesn't own a file manager here
    // if we're compiling a module unit, since the file manager is owned by the
    // AST when we're compiling a module unit. So the file manager may be
    // invalid here.
    //
    // It should be fine to create file manager here since the file system
    // options are stored in the compiler invocation and we can recreate the VFS
    // from the compiler invocation.
    if (!clang_instance->hasFileManager()) {
      clang_instance->createFileManager();
    }

    if (auto profiler_output = clang_instance->createOutputFile(
            clang_instance->getFrontendOpts().TimeTracePath, /*Binary=*/false,
            /*RemoveFileOnSignal=*/false,
            /*useTemporary=*/false)) {
      llvm::timeTraceProfilerWrite(*profiler_output);
      profiler_output.reset();
      llvm::timeTraceProfilerCleanup();
      clang_instance->clearOutputFiles(false);
    }
  }

  // When running with -disable-free, don't do any destruction or shutdown.
  if (clang_instance->getFrontendOpts().DisableFree) {
    llvm::BuryPointer(std::move(clang_instance));
  }
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace Carbon
