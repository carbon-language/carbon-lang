//===- BuildSystem.cpp - Utilities for use by build systems ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements various utilities for use by build systems.
//
//===----------------------------------------------------------------------===//

#include "clang-c/BuildSystem.h"
#include "CXString.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TimeValue.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm::sys;

unsigned long long clang_getBuildSessionTimestamp(void) {
  return llvm::sys::TimeValue::now().toEpochTime();
}

struct CXVirtualFileOverlayImpl {
  std::vector<std::pair<std::string, std::string> > Mappings;
};

CXVirtualFileOverlay clang_VirtualFileOverlay_create(unsigned) {
  return new CXVirtualFileOverlayImpl();
}

enum CXErrorCode
clang_VirtualFileOverlay_addFileMapping(CXVirtualFileOverlay VFO,
                                        const char *virtualPath,
                                        const char *realPath) {
  if (!VFO || !virtualPath || !realPath)
    return CXError_InvalidArguments;
  if (!path::is_absolute(virtualPath))
    return CXError_InvalidArguments;
  if (!path::is_absolute(realPath))
    return CXError_InvalidArguments;

  for (path::const_iterator
         PI = path::begin(virtualPath),
         PE = path::end(virtualPath); PI != PE; ++PI) {
    StringRef Comp = *PI;
    if (Comp == "." || Comp == "..")
      return CXError_InvalidArguments;
  }

  VFO->Mappings.push_back(std::make_pair(virtualPath, realPath));
  return CXError_Success;
}

namespace {
struct EntryTy {
  std::string VPath;
  std::string RPath;

  friend bool operator < (const EntryTy &LHS, const EntryTy &RHS) {
    return LHS.VPath < RHS.VPath;
  }
};

class JSONVFSPrinter {
  llvm::raw_ostream &OS;

public:
  JSONVFSPrinter(llvm::raw_ostream &OS) : OS(OS) {}

  /// Entries must be sorted.
  void print(ArrayRef<EntryTy> Entries) {
    OS << "{\n"
          "  'version': 0,\n"
          "  'roots': [\n";
    printDirNodes(Entries, "", 4);
    OS << "  ]\n"
          "}\n";
  }

private:
  ArrayRef<EntryTy> printDirNodes(ArrayRef<EntryTy> Entries,
                                  StringRef ParentPath,
                                  unsigned Indent) {
    while (!Entries.empty()) {
      const EntryTy &Entry = Entries.front();
      OS.indent(Indent) << "{\n";
      Indent += 2;
      OS.indent(Indent) << "'type': 'directory',\n";
      OS.indent(Indent) << "'name': \"";
      StringRef DirName = containedPart(ParentPath,
                                        path::parent_path(Entry.VPath));
      OS.write_escaped(DirName) << "\",\n";
      OS.indent(Indent) << "'contents': [\n";
      Entries = printContents(Entries, Indent + 2);
      OS.indent(Indent) << "]\n";
      Indent -= 2;
      OS.indent(Indent) << '}';
      if (Entries.empty()) {
        OS << '\n';
        break;
      }
      StringRef NextVPath = Entries.front().VPath;
      if (!containedIn(ParentPath, NextVPath)) {
        OS << '\n';
        break;
      }
      OS << ",\n";
    }
    return Entries;
  }

  ArrayRef<EntryTy> printContents(ArrayRef<EntryTy> Entries,
                                  unsigned Indent) {
    while (!Entries.empty()) {
      const EntryTy &Entry = Entries.front();
      Entries = Entries.slice(1);
      StringRef ParentPath = path::parent_path(Entry.VPath);
      StringRef VName = path::filename(Entry.VPath);
      OS.indent(Indent) << "{\n";
      Indent += 2;
      OS.indent(Indent) << "'type': 'file',\n";
      OS.indent(Indent) << "'name': \"";
      OS.write_escaped(VName) << "\",\n";
      OS.indent(Indent) << "'external-contents': \"";
      OS.write_escaped(Entry.RPath) << "\"\n";
      Indent -= 2;
      OS.indent(Indent) << '}';
      if (Entries.empty()) {
        OS << '\n';
        break;
      }
      StringRef NextVPath = Entries.front().VPath;
      if (!containedIn(ParentPath, NextVPath)) {
        OS << '\n';
        break;
      }
      OS << ",\n";
      if (path::parent_path(NextVPath) != ParentPath) {
        Entries = printDirNodes(Entries, ParentPath, Indent);
      }
    }
    return Entries;
  }

  bool containedIn(StringRef Parent, StringRef Path) {
    return Path.startswith(Parent);
  }

  StringRef containedPart(StringRef Parent, StringRef Path) {
    assert(containedIn(Parent, Path));
    if (Parent.empty())
      return Path;
    return Path.slice(Parent.size()+1, StringRef::npos);
  }
};
}

enum CXErrorCode
clang_VirtualFileOverlay_writeToBuffer(CXVirtualFileOverlay VFO, unsigned,
                                       char **out_buffer_ptr,
                                       unsigned *out_buffer_size) {
  if (!VFO || !out_buffer_ptr || !out_buffer_size)
    return CXError_InvalidArguments;

  llvm::SmallVector<EntryTy, 16> Entries;
  for (unsigned i = 0, e = VFO->Mappings.size(); i != e; ++i) {
    EntryTy Entry;
    Entry.VPath = VFO->Mappings[i].first;
    Entry.RPath = VFO->Mappings[i].second;
    Entries.push_back(Entry);
  }

  // FIXME: We should add options to determine if the paths are case sensitive
  // or not. The following assumes that if paths are case-insensitive the caller
  // did not mix cases in the virtual paths it provided.

  std::sort(Entries.begin(), Entries.end());

  llvm::SmallString<256> Buf;
  llvm::raw_svector_ostream OS(Buf);
  JSONVFSPrinter Printer(OS);
  Printer.print(Entries);

  StringRef Data = OS.str();
  *out_buffer_ptr = (char*)malloc(Data.size());
  *out_buffer_size = Data.size();
  memcpy(*out_buffer_ptr, Data.data(), Data.size());
  return CXError_Success;
}

void clang_VirtualFileOverlay_dispose(CXVirtualFileOverlay VFO) {
  delete VFO;
}


struct CXModuleMapDescriptorImpl {
  std::string ModuleName;
  std::string UmbrellaHeader;
};

CXModuleMapDescriptor clang_ModuleMapDescriptor_create(unsigned) {
  return new CXModuleMapDescriptorImpl();
}

enum CXErrorCode
clang_ModuleMapDescriptor_setFrameworkModuleName(CXModuleMapDescriptor MMD,
                                                 const char *name) {
  if (!MMD || !name)
    return CXError_InvalidArguments;

  MMD->ModuleName = name;
  return CXError_Success;
}

enum CXErrorCode
clang_ModuleMapDescriptor_setUmbrellaHeader(CXModuleMapDescriptor MMD,
                                            const char *name) {
  if (!MMD || !name)
    return CXError_InvalidArguments;

  MMD->UmbrellaHeader = name;
  return CXError_Success;
}

enum CXErrorCode
clang_ModuleMapDescriptor_writeToBuffer(CXModuleMapDescriptor MMD, unsigned,
                                       char **out_buffer_ptr,
                                       unsigned *out_buffer_size) {
  if (!MMD || !out_buffer_ptr || !out_buffer_size)
    return CXError_InvalidArguments;

  llvm::SmallString<256> Buf;
  llvm::raw_svector_ostream OS(Buf);
  OS << "framework module " << MMD->ModuleName << " {\n";
  OS << "  umbrella header \"";
  OS.write_escaped(MMD->UmbrellaHeader) << "\"\n";
  OS << '\n';
  OS << "  export *\n";
  OS << "  module * { export * }\n";
  OS << "}\n";

  StringRef Data = OS.str();
  *out_buffer_ptr = (char*)malloc(Data.size());
  *out_buffer_size = Data.size();
  memcpy(*out_buffer_ptr, Data.data(), Data.size());
  return CXError_Success;
}

void clang_ModuleMapDescriptor_dispose(CXModuleMapDescriptor MMD) {
  delete MMD;
}
