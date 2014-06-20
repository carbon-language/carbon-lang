//===--- ModuleDependencyCollector.cpp - Collect module dependencies ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect the dependencies of a set of modules.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/Utils.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {
/// Private implementation for ModuleDependencyCollector
class ModuleDependencyListener : public ASTReaderListener {
  ModuleDependencyCollector &Collector;

  std::error_code copyToRoot(StringRef Src);
public:
  ModuleDependencyListener(ModuleDependencyCollector &Collector)
      : Collector(Collector) {}
  bool needsInputFileVisitation() override { return true; }
  bool needsSystemInputFileVisitation() override { return true; }
  bool visitInputFile(StringRef Filename, bool IsSystem,
                      bool IsOverridden) override;
};
}

void ModuleDependencyCollector::attachToASTReader(ASTReader &R) {
  R.addListener(new ModuleDependencyListener(*this));
}

void ModuleDependencyCollector::writeFileMap() {
  if (Seen.empty())
    return;

  SmallString<256> Dest = getDest();
  llvm::sys::path::append(Dest, "vfs.yaml");

  std::string ErrorInfo;
  llvm::raw_fd_ostream OS(Dest.c_str(), ErrorInfo, llvm::sys::fs::F_Text);
  if (!ErrorInfo.empty()) {
    setHasErrors();
    return;
  }
  VFSWriter.write(OS);
}

/// Remove traversal (ie, . or ..) from the given absolute path.
static void removePathTraversal(SmallVectorImpl<char> &Path) {
  using namespace llvm::sys;
  SmallVector<StringRef, 16> ComponentStack;
  StringRef P(Path.data(), Path.size());

  // Skip the root path, then look for traversal in the components.
  StringRef Rel = path::relative_path(P);
  for (StringRef C : llvm::make_range(path::begin(Rel), path::end(Rel))) {
    if (C == ".")
      continue;
    if (C == "..") {
      assert(ComponentStack.size() && "Path traverses out of parent");
      ComponentStack.pop_back();
    } else
      ComponentStack.push_back(C);
  }

  // The stack is now the path without any directory traversal.
  SmallString<256> Buffer = path::root_path(P);
  for (StringRef C : ComponentStack)
    path::append(Buffer, C);

  // Put the result in Path.
  Path.swap(Buffer);
}

std::error_code ModuleDependencyListener::copyToRoot(StringRef Src) {
  using namespace llvm::sys;

  // We need an absolute path to append to the root.
  SmallString<256> AbsoluteSrc = Src;
  fs::make_absolute(AbsoluteSrc);
  removePathTraversal(AbsoluteSrc);

  // Build the destination path.
  SmallString<256> Dest = Collector.getDest();
  path::append(Dest, path::relative_path(AbsoluteSrc));

  // Copy the file into place.
  if (std::error_code EC = fs::create_directories(path::parent_path(Dest),
                                                   /*IgnoreExisting=*/true))
    return EC;
  if (std::error_code EC = fs::copy_file(AbsoluteSrc.str(), Dest.str()))
    return EC;
  // Use the absolute path under the root for the file mapping.
  Collector.addFileMapping(AbsoluteSrc.str(), Dest.str());
  return std::error_code();
}

bool ModuleDependencyListener::visitInputFile(StringRef Filename, bool IsSystem,
                                              bool IsOverridden) {
  if (Collector.insertSeen(Filename))
    if (copyToRoot(Filename))
      Collector.setHasErrors();
  return true;
}
