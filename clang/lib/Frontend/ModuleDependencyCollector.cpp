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
#include "llvm/Support/Filesystem.h"
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

/// Append the absolute path in Nested to the path given by Root. This will
/// remove directory traversal from the resulting nested path.
static void appendNestedPath(SmallVectorImpl<char> &Root, StringRef Nested) {
  using namespace llvm::sys;
  SmallVector<StringRef, 16> ComponentStack;

  StringRef Rel = path::relative_path(Nested);
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
  for (StringRef C : ComponentStack)
    path::append(Root, C);
}

std::error_code ModuleDependencyListener::copyToRoot(StringRef Src) {
  using namespace llvm::sys;

  // We need an absolute path to append to the root.
  SmallString<256> AbsoluteSrc = Src;
  fs::make_absolute(AbsoluteSrc);
  // Build the destination path.
  SmallString<256> Dest = Collector.getDest();
  size_t RootLen = Dest.size();
  appendNestedPath(Dest, AbsoluteSrc);

  // Copy the file into place.
  if (std::error_code EC = fs::create_directories(path::parent_path(Dest),
                                                   /*IgnoreExisting=*/true))
    return EC;
  if (std::error_code EC = fs::copy_file(AbsoluteSrc.str(), Dest.str()))
    return EC;
  // Use the absolute path under the root for the file mapping.
  Collector.addFileMapping(Dest.substr(RootLen), Dest.str());
  return std::error_code();
}

bool ModuleDependencyListener::visitInputFile(StringRef Filename, bool IsSystem,
                                              bool IsOverridden) {
  if (Collector.insertSeen(Filename))
    if (copyToRoot(Filename))
      Collector.setHasErrors();
  return true;
}
