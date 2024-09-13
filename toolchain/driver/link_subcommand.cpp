// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/link_subcommand.h"

#include "llvm/TargetParser/Triple.h"
#include "toolchain/driver/clang_runner.h"

// TODO: Remove this include.
#include "toolchain/driver/driver.h"

namespace Carbon {

constexpr CommandLine::CommandInfo LinkOptions::Info = {
    .name = "link",
    .help = R"""(
Link Carbon executables.

This subcommand links Carbon executables by combining object files.

TODO: Support linking binary libraries, both archives and shared libraries.
TODO: Support linking against binary libraries.
)""",
};

auto LinkOptions::Build(CommandLine::CommandBuilder& b,
                        CodegenOptions& codegen_options) -> void {
  b.AddStringPositionalArg(
      {
          .name = "OBJECT_FILE",
          .help = R"""(
The input object files.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Required(true);
        arg_b.Append(&object_filenames);
      });

  b.AddStringOption(
      {
          .name = "output",
          .value_name = "FILE",
          .help = R"""(
The linked file name. The output is always a linked binary.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Required(true);
        arg_b.Set(&output_filename);
      });

  codegen_options.Build(b);
}

static void AddOSFlags(llvm::StringRef target,
                       llvm::SmallVectorImpl<llvm::StringRef>& args) {
  llvm::Triple triple(target);
  switch (triple.getOS()) {
    case llvm::Triple::Darwin:
    case llvm::Triple::MacOSX:
      // On macOS we need to set the sysroot to a viable SDK. Currently, this
      // hard codes the path to be the unversioned symlink. The prefix is also
      // hard coded in Homebrew and so this seems likely to work reasonably
      // well. Homebrew and I suspect the Xcode Clang both have this hard coded
      // at build time, so this seems reasonably safe but we can revisit if/when
      // needed.
      args.push_back(
          "--sysroot=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk");
      // We also need to insist on a modern linker, otherwise the driver tries
      // too old and deprecated flags. The specific number here comes from an
      // inspection of the Clang driver source code to understand where features
      // were enabled, and this appears to be the latest version to control
      // driver behavior.
      //
      // TODO: We should replace this with use of `lld` eventually.
      args.push_back("-mlinker-version=705");
      break;

    default:
      // By default, just let the Clang driver handle everything.
      break;
  }
}

auto Driver::Link(const LinkOptions& options,
                  const CodegenOptions& codegen_options) -> RunResult {
  // TODO: Currently we use the Clang driver to link. This works well on Unix
  // OSes but we likely need to directly build logic to invoke `link.exe` on
  // Windows where `cl.exe` doesn't typically cover that logic.

  // Use a reasonably large small vector here to minimize allocations. We expect
  // to link reasonably large numbers of object files.
  llvm::SmallVector<llvm::StringRef, 128> clang_args;

  // We link using a C++ mode of the driver.
  clang_args.push_back("--driver-mode=g++");

  // Use LLD, which we provide in our install directory, for linking.
  clang_args.push_back("-fuse-ld=lld");

  // Disable linking the C++ standard library until can build and ship it as
  // part of the Carbon toolchain. This clearly won't work once we get into
  // interop, but for now it avoids spurious failures and distraction. The plan
  // is to build and bundle libc++ at which point we can replace this with
  // pointing at our bundled library.
  // TODO: Replace this when ready.
  clang_args.push_back("-nostdlib++");

  // Add OS-specific flags based on the target.
  AddOSFlags(codegen_options.target, clang_args);

  clang_args.push_back("-o");
  clang_args.push_back(options.output_filename);
  clang_args.append(options.object_filenames.begin(),
                    options.object_filenames.end());

  ClangRunner runner(driver_env_.installation, codegen_options.target,
                     driver_env_.vlog_stream);
  return {.success = runner.Run(clang_args)};
}

}  // namespace Carbon
