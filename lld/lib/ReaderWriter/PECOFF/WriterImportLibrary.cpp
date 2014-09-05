//===- lib/ReaderWriter/PECOFF/WriterImportLibrary.cpp --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This file is responsible for creating the Import Library file.
///
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace pecoff {

/// Creates a .def file containing the list of exported symbols.
static std::string
createModuleDefinitionFile(const PECOFFLinkingContext &ctx) {
  std::string ret;
  llvm::raw_string_ostream os(ret);
  os << "LIBRARY \"" << llvm::sys::path::filename(ctx.outputPath()) << "\"\n"
     << "EXPORTS\n";

  for (const PECOFFLinkingContext::ExportDesc &desc : ctx.getDllExports()) {
    os << "  " << desc.externalName << " @" << desc.ordinal;
    if (desc.noname)
      os << " NONAME";
    if (desc.isData)
      os << " DATA";
    os << "\n";
  }
  os.flush();
  return ret;
}

static std::string writeToTempFile(StringRef contents) {
  SmallString<128> path;
  int fd;
  if (llvm::sys::fs::createTemporaryFile("tmp", "def", fd, path)) {
    llvm::errs() << "Failed to create temporary file\n";
    return "";
  }
  llvm::raw_fd_ostream os(fd, /*shouldClose*/ true);
  os << contents;
  return path.str();
}

static void writeTo(StringRef path, StringRef contents) {
  int fd;
  if (llvm::sys::fs::openFileForWrite(path, fd, llvm::sys::fs::F_Text)) {
    llvm::errs() << "Failed to open " << path << "\n";
    return;
  }
  llvm::raw_fd_ostream os(fd, /*shouldClose*/ true);
  os << contents;
}

/// Creates a .def file and runs lib.exe on it to create an import library.
void writeImportLibrary(const PECOFFLinkingContext &ctx) {
  std::string program = "lib.exe";
  std::string programPath = llvm::sys::FindProgramByName(program);

  std::string fileContents = createModuleDefinitionFile(ctx);
  std::string defPath = writeToTempFile(fileContents);
  llvm::FileRemover tmpFile(defPath);

  std::string defArg = "/def:";
  defArg.append(defPath);
  std::string outputArg = "/out:";
  outputArg.append(ctx.getOutputImportLibraryPath());

  std::vector<const char *> args;
  args.push_back(programPath.c_str());
  args.push_back("/nologo");
  args.push_back(ctx.is64Bit() ? "/machine:x64" : "/machine:x86");
  args.push_back(defArg.c_str());
  args.push_back(outputArg.c_str());
  args.push_back(nullptr);

  if (programPath.empty()) {
    llvm::errs() << "Unable to find " << program << " in PATH\n";
  } else if (llvm::sys::ExecuteAndWait(programPath.c_str(), &args[0]) != 0) {
    llvm::errs() << program << " failed\n";
  }

  // If /lldmoduledeffile:<filename> is given, make a copy of the
  // temporary module definition file. This feature is for unit tests.
  if (!ctx.getModuleDefinitionFile().empty())
    writeTo(ctx.getModuleDefinitionFile(), fileContents);
}

} // end namespace pecoff
} // end namespace lld
