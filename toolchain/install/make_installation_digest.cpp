// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <unistd.h>

#include <cstdlib>
#include <string>

#include "common/bazel_working_dir.h"
#include "common/error.h"
#include "common/exe_path.h"
#include "common/filesystem.h"
#include "common/init_llvm.h"
#include "common/map.h"
#include "common/vlog.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

namespace Carbon {
namespace {

// A class implementing our digest program.
//
// The program is started with a call to `Run`, and either returns an error or
// an exit code for `main`. It has a very simple command line interface:
// - An optional flag `--verbose` that must be the first argument if provided.
// - A required positional argument of a manifest file of all the files in a
//   Carbon installation that should be added to the digest.
// - A required positional argument of an output file for the digest.
//
// The program reads the manifest of all the files in the Carbon installation,
// and adds each of those files to a running cryptographic digest. Once
// complete, it writes this cryptographic digest to the provided output digest
// file.
//
// The exact digest format is unspecified, but should provide a strong guarantee
// that changes to any of the files in the manifest of the install produce
// different digests.
class DigestProgram {
 public:
  auto Run(int argc, char** argv) -> ErrorOr<int>;

 private:
  auto ComputeFileDigest(Filesystem::ReadFileRef file)
      -> std::array<uint8_t, 32>;

  llvm::raw_ostream* vlog_stream_ = nullptr;
  Map<uint64_t, std::array<uint8_t, 32>> file_digests_;
};

auto DigestProgram::Run(int argc, char** argv) -> ErrorOr<int> {
  InitLLVM init_llvm(argc, argv);
  SetWorkingDirForBazel();
  llvm::SHA256 sha256;

  // If the first argument is `--verbose`, enable verbose logging.
  int num_args = 2;
  if (argc > 1 && llvm::StringRef(argv[1]) == "--verbose") {
    vlog_stream_ = &llvm::errs();
    ++num_args;
  }
  // The last two arguments are required and positional.
  if (argc <= num_args) {
    return Error(
        "Usage: make-installation-digest [--verbose] MANIFEST_FILE "
        "OUTPUT_FILE");
  }
  std::filesystem::path manifest_path = argv[num_args - 1];
  std::filesystem::path digest_path = argv[num_args];

  CARBON_ASSIGN_OR_RETURN(std::string manifest,
                          Filesystem::Cwd().ReadFileToString(manifest_path));
  llvm::SmallVector<llvm::StringRef> manifest_lines;
  llvm::StringRef(manifest).split(manifest_lines, '\n');

  // Walk all the install data files in the manifest.
  for (llvm::StringRef manifest_line : manifest_lines) {
    if (manifest_line.empty()) {
      continue;
    }
    // Compute the full path and installed path for each file. The installed
    // path comes from the path components below the `prefix_root` component.
    std::filesystem::path full_path = manifest_line.trim().str();
    std::filesystem::path install_path;
    bool append = false;
    for (const auto& component : full_path) {
      if (append) {
        install_path /= component;
        continue;
      }

      if (component == "prefix_root") {
        append = true;
      }
    }

    CARBON_VLOG("Digesting file: {0}\n", install_path);
    // Add the install path itself to the digest to track the layout of the
    // installation data.
    sha256.update(install_path.native());

    // Open the file and compute its digest to add as well. We use a memoizing
    // helper here to avoid re-examining the same file even if there are
    // multiple paths to reach that file.
    CARBON_ASSIGN_OR_RETURN(
        Filesystem::ReadFile file,
        Filesystem::Cwd().OpenReadOnly(full_path, Filesystem::OpenExisting));
    sha256.update(ComputeFileDigest(file));
  }

  auto digest = sha256.final();
  CARBON_VLOG("Digest: {0}\n", llvm::toHex(digest));

  CARBON_RETURN_IF_ERROR(Filesystem::Cwd().WriteFileFromString(
      digest_path, llvm::toHex(digest, /*LowerCase=*/true) + "\n"));

  return EXIT_SUCCESS;
}

auto DigestProgram::ComputeFileDigest(Filesystem::ReadFileRef file)
    -> std::array<uint8_t, 32> {
  // Filesystem errors are unlikely here, and the library will check them just
  // in case.
  Filesystem::FileStatus stat = *file.Stat();
  auto result = file_digests_.Insert(stat.unix_inode(), [file]() mutable {
    llvm::SHA256 sha256;
    // TODO: We could do this more efficiently by using a fixed buffer.
    sha256.update(*file.ReadFileToString());
    return sha256.final();
  });
  return result.value();
}

}  // namespace
}  // namespace Carbon

auto main(int argc, char** argv) -> int {
  Carbon::DigestProgram program;
  auto result = program.Run(argc, argv);
  if (result.ok()) {
    return *result;
  } else {
    llvm::errs() << "error: " << result.error() << "\n";
    return EXIT_FAILURE;
  }
}
