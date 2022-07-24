#include "explorer/main.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;

auto main(int argc, char** argv) -> int {
  llvm::StringRef bin = path::filename(argv[0]);

  if (bin == "carbon-explorer") {
    static int static_for_main_address;

    std::string exe = fs::getMainExecutable(
            argv[0], static_cast<void *>(&static_for_main_address)
    );
    llvm::StringRef install_path = path::parent_path(exe);
    llvm::SmallString<256> prelude_file(install_path);
    path::append(prelude_file, "data", "prelude.carbon");
    
    return Carbon::ExplorerMain(prelude_file, argc, argv);
  } else {
    return fprintf(stderr, "Unrecognized Carbon binary requested: %s", argv[0]), 1;
  }
}
