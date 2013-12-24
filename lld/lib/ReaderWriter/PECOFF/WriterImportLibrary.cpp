//===- lib/ReaderWriter/PECOFF/WriterImportLibrary.cpp --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This file is reponsible for creating the Import Library file.
///
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace pecoff {

static std::string getOutputPath(const PECOFFLinkingContext &ctx) {
  SmallString<128> path = ctx.outputPath();
  llvm::sys::path::replace_extension(path, ".lib");
  return path.str();
}

/// Creates a .def file containing the list of exported symbols.
static std::string
createModuleDefinitionFile(const PECOFFLinkingContext &ctx,
                           llvm::FileRemover &fileRemover) {
  SmallString<128> defFile;
  int fd;
  if (llvm::sys::fs::createTemporaryFile("tmp", "def", fd, defFile)) {
    llvm::errs() << "Failed to create temporary file\n";
    return "";
  }

  llvm::raw_fd_ostream os(fd, /*shouldClose*/ true);
  os << "LIBRARY \"" << llvm::sys::path::filename(ctx.outputPath()) << "\"\n"
     << "EXPORTS\n";

  for (const PECOFFLinkingContext::ExportDesc &desc : ctx.getDllExports()) {
    os << "  " << ctx.undecorateSymbol(desc.name) << " @" << desc.ordinal;
    if (desc.noname)
      os << " NONAME";
    if (desc.isData)
      os << " DATA";
    os << "\n";
  }
  return defFile.str();
}

/// Creates a .def file and runs lib.exe on it to create an import library.
void writeImportLibrary(const PECOFFLinkingContext &ctx) {
  std::string program = "lib.exe";
  std::string programPath = llvm::sys::FindProgramByName(program);
  if (programPath.empty()) {
    llvm::errs() << "Unable to find " << program << " in PATH\n";
    return;
  }

  llvm::FileRemover tmpFile;
  std::string defArg = "/def:";
  defArg.append(createModuleDefinitionFile(ctx, tmpFile));
  std::string outputArg = "/out:";
  outputArg.append(getOutputPath(ctx));

  std::vector<const char *> args;
  args.push_back(programPath.c_str());
  args.push_back("/nologo");
  args.push_back("/machine:x86");
  args.push_back(defArg.c_str());
  args.push_back(outputArg.c_str());
  args.push_back(nullptr);
  if (llvm::sys::ExecuteAndWait(programPath.c_str(), &args[0]) != 0)
    llvm::errs() << program << " failed\n";
}

} // end namespace pecoff
} // end namespace lld
