//===- VirtualFileSystem.cpp - Virtual File System Layer --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file implements the VirtualFileSystem interface.
//===----------------------------------------------------------------------===//

#include "clang/Basic/VirtualFileSystem.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLParser.h"
#include <atomic>
#include <memory>

using namespace clang;
using namespace clang::vfs;
using namespace llvm;
using llvm::sys::fs::file_status;
using llvm::sys::fs::file_type;
using llvm::sys::fs::perms;
using llvm::sys::fs::UniqueID;

Status::Status(const file_status &Status)
    : UID(Status.getUniqueID()), MTime(Status.getLastModificationTime()),
      User(Status.getUser()), Group(Status.getGroup()), Size(Status.getSize()),
      Type(Status.type()), Perms(Status.permissions()), IsVFSMapped(false)  {}

Status::Status(StringRef Name, StringRef ExternalName, UniqueID UID,
               sys::TimeValue MTime, uint32_t User, uint32_t Group,
               uint64_t Size, file_type Type, perms Perms)
    : Name(Name), UID(UID), MTime(MTime), User(User), Group(Group), Size(Size),
      Type(Type), Perms(Perms), IsVFSMapped(false) {}

bool Status::equivalent(const Status &Other) const {
  return getUniqueID() == Other.getUniqueID();
}
bool Status::isDirectory() const {
  return Type == file_type::directory_file;
}
bool Status::isRegularFile() const {
  return Type == file_type::regular_file;
}
bool Status::isOther() const {
  return exists() && !isRegularFile() && !isDirectory() && !isSymlink();
}
bool Status::isSymlink() const {
  return Type == file_type::symlink_file;
}
bool Status::isStatusKnown() const {
  return Type != file_type::status_error;
}
bool Status::exists() const {
  return isStatusKnown() && Type != file_type::file_not_found;
}

File::~File() {}

FileSystem::~FileSystem() {}

ErrorOr<std::unique_ptr<MemoryBuffer>>
FileSystem::getBufferForFile(const llvm::Twine &Name, int64_t FileSize,
                             bool RequiresNullTerminator, bool IsVolatile) {
  auto F = openFileForRead(Name);
  if (!F)
    return F.getError();

  return (*F)->getBuffer(Name, FileSize, RequiresNullTerminator, IsVolatile);
}

//===-----------------------------------------------------------------------===/
// RealFileSystem implementation
//===-----------------------------------------------------------------------===/

namespace {
/// \brief Wrapper around a raw file descriptor.
class RealFile : public File {
  int FD;
  Status S;
  friend class RealFileSystem;
  RealFile(int FD) : FD(FD) {
    assert(FD >= 0 && "Invalid or inactive file descriptor");
  }

public:
  ~RealFile() override;
  ErrorOr<Status> status() override;
  ErrorOr<std::unique_ptr<MemoryBuffer>>
  getBuffer(const Twine &Name, int64_t FileSize = -1,
            bool RequiresNullTerminator = true,
            bool IsVolatile = false) override;
  std::error_code close() override;
  void setName(StringRef Name) override;
};
} // end anonymous namespace
RealFile::~RealFile() { close(); }

ErrorOr<Status> RealFile::status() {
  assert(FD != -1 && "cannot stat closed file");
  if (!S.isStatusKnown()) {
    file_status RealStatus;
    if (std::error_code EC = sys::fs::status(FD, RealStatus))
      return EC;
    Status NewS(RealStatus);
    NewS.setName(S.getName());
    S = std::move(NewS);
  }
  return S;
}

ErrorOr<std::unique_ptr<MemoryBuffer>>
RealFile::getBuffer(const Twine &Name, int64_t FileSize,
                    bool RequiresNullTerminator, bool IsVolatile) {
  assert(FD != -1 && "cannot get buffer for closed file");
  return MemoryBuffer::getOpenFile(FD, Name, FileSize, RequiresNullTerminator,
                                   IsVolatile);
}

// FIXME: This is terrible, we need this for ::close.
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#include <sys/uio.h>
#else
#include <io.h>
#ifndef S_ISFIFO
#define S_ISFIFO(x) (0)
#endif
#endif
std::error_code RealFile::close() {
  if (::close(FD))
    return std::error_code(errno, std::generic_category());
  FD = -1;
  return std::error_code();
}

void RealFile::setName(StringRef Name) {
  S.setName(Name);
}

namespace {
/// \brief The file system according to your operating system.
class RealFileSystem : public FileSystem {
public:
  ErrorOr<Status> status(const Twine &Path) override;
  ErrorOr<std::unique_ptr<File>> openFileForRead(const Twine &Path) override;
  directory_iterator dir_begin(const Twine &Dir, std::error_code &EC) override;
};
} // end anonymous namespace

ErrorOr<Status> RealFileSystem::status(const Twine &Path) {
  sys::fs::file_status RealStatus;
  if (std::error_code EC = sys::fs::status(Path, RealStatus))
    return EC;
  Status Result(RealStatus);
  Result.setName(Path.str());
  return Result;
}

ErrorOr<std::unique_ptr<File>>
RealFileSystem::openFileForRead(const Twine &Name) {
  int FD;
  if (std::error_code EC = sys::fs::openFileForRead(Name, FD))
    return EC;
  std::unique_ptr<File> Result(new RealFile(FD));
  Result->setName(Name.str());
  return std::move(Result);
}

IntrusiveRefCntPtr<FileSystem> vfs::getRealFileSystem() {
  static IntrusiveRefCntPtr<FileSystem> FS = new RealFileSystem();
  return FS;
}

namespace {
class RealFSDirIter : public clang::vfs::detail::DirIterImpl {
  std::string Path;
  llvm::sys::fs::directory_iterator Iter;
public:
  RealFSDirIter(const Twine &_Path, std::error_code &EC)
      : Path(_Path.str()), Iter(Path, EC) {
    if (!EC && Iter != llvm::sys::fs::directory_iterator()) {
      llvm::sys::fs::file_status S;
      EC = Iter->status(S);
      if (!EC) {
        CurrentEntry = Status(S);
        CurrentEntry.setName(Iter->path());
      }
    }
  }

  std::error_code increment() override {
    std::error_code EC;
    Iter.increment(EC);
    if (EC) {
      return EC;
    } else if (Iter == llvm::sys::fs::directory_iterator()) {
      CurrentEntry = Status();
    } else {
      llvm::sys::fs::file_status S;
      EC = Iter->status(S);
      CurrentEntry = Status(S);
      CurrentEntry.setName(Iter->path());
    }
    return EC;
  }
};
}

directory_iterator RealFileSystem::dir_begin(const Twine &Dir,
                                             std::error_code &EC) {
  return directory_iterator(std::make_shared<RealFSDirIter>(Dir, EC));
}

//===-----------------------------------------------------------------------===/
// OverlayFileSystem implementation
//===-----------------------------------------------------------------------===/
OverlayFileSystem::OverlayFileSystem(IntrusiveRefCntPtr<FileSystem> BaseFS) {
  pushOverlay(BaseFS);
}

void OverlayFileSystem::pushOverlay(IntrusiveRefCntPtr<FileSystem> FS) {
  FSList.push_back(FS);
}

ErrorOr<Status> OverlayFileSystem::status(const Twine &Path) {
  // FIXME: handle symlinks that cross file systems
  for (iterator I = overlays_begin(), E = overlays_end(); I != E; ++I) {
    ErrorOr<Status> Status = (*I)->status(Path);
    if (Status || Status.getError() != llvm::errc::no_such_file_or_directory)
      return Status;
  }
  return make_error_code(llvm::errc::no_such_file_or_directory);
}

ErrorOr<std::unique_ptr<File>>
OverlayFileSystem::openFileForRead(const llvm::Twine &Path) {
  // FIXME: handle symlinks that cross file systems
  for (iterator I = overlays_begin(), E = overlays_end(); I != E; ++I) {
    auto Result = (*I)->openFileForRead(Path);
    if (Result || Result.getError() != llvm::errc::no_such_file_or_directory)
      return Result;
  }
  return make_error_code(llvm::errc::no_such_file_or_directory);
}

clang::vfs::detail::DirIterImpl::~DirIterImpl() { }

namespace {
class OverlayFSDirIterImpl : public clang::vfs::detail::DirIterImpl {
  OverlayFileSystem &Overlays;
  std::string Path;
  OverlayFileSystem::iterator CurrentFS;
  directory_iterator CurrentDirIter;
  llvm::StringSet<> SeenNames;

  std::error_code incrementFS() {
    assert(CurrentFS != Overlays.overlays_end() && "incrementing past end");
    ++CurrentFS;
    for (auto E = Overlays.overlays_end(); CurrentFS != E; ++CurrentFS) {
      std::error_code EC;
      CurrentDirIter = (*CurrentFS)->dir_begin(Path, EC);
      if (EC && EC != errc::no_such_file_or_directory)
        return EC;
      if (CurrentDirIter != directory_iterator())
        break; // found
    }
    return std::error_code();
  }

  std::error_code incrementDirIter(bool IsFirstTime) {
    assert((IsFirstTime || CurrentDirIter != directory_iterator()) &&
           "incrementing past end");
    std::error_code EC;
    if (!IsFirstTime)
      CurrentDirIter.increment(EC);
    if (!EC && CurrentDirIter == directory_iterator())
      EC = incrementFS();
    return EC;
  }

  std::error_code incrementImpl(bool IsFirstTime) {
    while (true) {
      std::error_code EC = incrementDirIter(IsFirstTime);
      if (EC || CurrentDirIter == directory_iterator()) {
        CurrentEntry = Status();
        return EC;
      }
      CurrentEntry = *CurrentDirIter;
      StringRef Name = llvm::sys::path::filename(CurrentEntry.getName());
      if (SeenNames.insert(Name).second)
        return EC; // name not seen before
    }
    llvm_unreachable("returned above");
  }

public:
  OverlayFSDirIterImpl(const Twine &Path, OverlayFileSystem &FS,
                       std::error_code &EC)
      : Overlays(FS), Path(Path.str()), CurrentFS(Overlays.overlays_begin()) {
    CurrentDirIter = (*CurrentFS)->dir_begin(Path, EC);
    EC = incrementImpl(true);
  }

  std::error_code increment() override { return incrementImpl(false); }
};
} // end anonymous namespace

directory_iterator OverlayFileSystem::dir_begin(const Twine &Dir,
                                                std::error_code &EC) {
  return directory_iterator(
      std::make_shared<OverlayFSDirIterImpl>(Dir, *this, EC));
}

//===-----------------------------------------------------------------------===/
// VFSFromYAML implementation
//===-----------------------------------------------------------------------===/

// Allow DenseMap<StringRef, ...>.  This is useful below because we know all the
// strings are literals and will outlive the map, and there is no reason to
// store them.
namespace llvm {
  template<>
  struct DenseMapInfo<StringRef> {
    // This assumes that "" will never be a valid key.
    static inline StringRef getEmptyKey() { return StringRef(""); }
    static inline StringRef getTombstoneKey() { return StringRef(); }
    static unsigned getHashValue(StringRef Val) { return HashString(Val); }
    static bool isEqual(StringRef LHS, StringRef RHS) { return LHS == RHS; }
  };
}

namespace {

enum EntryKind {
  EK_Directory,
  EK_File
};

/// \brief A single file or directory in the VFS.
class Entry {
  EntryKind Kind;
  std::string Name;

public:
  virtual ~Entry();
  Entry(EntryKind K, StringRef Name) : Kind(K), Name(Name) {}
  StringRef getName() const { return Name; }
  EntryKind getKind() const { return Kind; }
};

class DirectoryEntry : public Entry {
  std::vector<Entry *> Contents;
  Status S;

public:
  ~DirectoryEntry() override;
  DirectoryEntry(StringRef Name, std::vector<Entry *> Contents, Status S)
      : Entry(EK_Directory, Name), Contents(std::move(Contents)),
        S(std::move(S)) {}
  Status getStatus() { return S; }
  typedef std::vector<Entry *>::iterator iterator;
  iterator contents_begin() { return Contents.begin(); }
  iterator contents_end() { return Contents.end(); }
  static bool classof(const Entry *E) { return E->getKind() == EK_Directory; }
};

class FileEntry : public Entry {
public:
  enum NameKind {
    NK_NotSet,
    NK_External,
    NK_Virtual
  };
private:
  std::string ExternalContentsPath;
  NameKind UseName;
public:
  FileEntry(StringRef Name, StringRef ExternalContentsPath, NameKind UseName)
      : Entry(EK_File, Name), ExternalContentsPath(ExternalContentsPath),
        UseName(UseName) {}
  StringRef getExternalContentsPath() const { return ExternalContentsPath; }
  /// \brief whether to use the external path as the name for this file.
  bool useExternalName(bool GlobalUseExternalName) const {
    return UseName == NK_NotSet ? GlobalUseExternalName
                                : (UseName == NK_External);
  }
  static bool classof(const Entry *E) { return E->getKind() == EK_File; }
};

class VFSFromYAML;

class VFSFromYamlDirIterImpl : public clang::vfs::detail::DirIterImpl {
  std::string Dir;
  VFSFromYAML &FS;
  DirectoryEntry::iterator Current, End;
public:
  VFSFromYamlDirIterImpl(const Twine &Path, VFSFromYAML &FS,
                         DirectoryEntry::iterator Begin,
                         DirectoryEntry::iterator End, std::error_code &EC);
  std::error_code increment() override;
};

/// \brief A virtual file system parsed from a YAML file.
///
/// Currently, this class allows creating virtual directories and mapping
/// virtual file paths to existing external files, available in \c ExternalFS.
///
/// The basic structure of the parsed file is:
/// \verbatim
/// {
///   'version': <version number>,
///   <optional configuration>
///   'roots': [
///              <directory entries>
///            ]
/// }
/// \endverbatim
///
/// All configuration options are optional.
///   'case-sensitive': <boolean, default=true>
///   'use-external-names': <boolean, default=true>
///
/// Virtual directories are represented as
/// \verbatim
/// {
///   'type': 'directory',
///   'name': <string>,
///   'contents': [ <file or directory entries> ]
/// }
/// \endverbatim
///
/// The default attributes for virtual directories are:
/// \verbatim
/// MTime = now() when created
/// Perms = 0777
/// User = Group = 0
/// Size = 0
/// UniqueID = unspecified unique value
/// \endverbatim
///
/// Re-mapped files are represented as
/// \verbatim
/// {
///   'type': 'file',
///   'name': <string>,
///   'use-external-name': <boolean> # Optional
///   'external-contents': <path to external file>)
/// }
/// \endverbatim
///
/// and inherit their attributes from the external contents.
///
/// In both cases, the 'name' field may contain multiple path components (e.g.
/// /path/to/file). However, any directory that contains more than one child
/// must be uniquely represented by a directory entry.
class VFSFromYAML : public vfs::FileSystem {
  std::vector<Entry *> Roots; ///< The root(s) of the virtual file system.
  /// \brief The file system to use for external references.
  IntrusiveRefCntPtr<FileSystem> ExternalFS;

  /// @name Configuration
  /// @{

  /// \brief Whether to perform case-sensitive comparisons.
  ///
  /// Currently, case-insensitive matching only works correctly with ASCII.
  bool CaseSensitive;

  /// \brief Whether to use to use the value of 'external-contents' for the
  /// names of files.  This global value is overridable on a per-file basis.
  bool UseExternalNames;
  /// @}

  friend class VFSFromYAMLParser;

private:
  VFSFromYAML(IntrusiveRefCntPtr<FileSystem> ExternalFS)
      : ExternalFS(ExternalFS), CaseSensitive(true), UseExternalNames(true) {}

  /// \brief Looks up \p Path in \c Roots.
  ErrorOr<Entry *> lookupPath(const Twine &Path);

  /// \brief Looks up the path <tt>[Start, End)</tt> in \p From, possibly
  /// recursing into the contents of \p From if it is a directory.
  ErrorOr<Entry *> lookupPath(sys::path::const_iterator Start,
                              sys::path::const_iterator End, Entry *From);

  /// \brief Get the status of a given an \c Entry.
  ErrorOr<Status> status(const Twine &Path, Entry *E);

public:
  ~VFSFromYAML() override;

  /// \brief Parses \p Buffer, which is expected to be in YAML format and
  /// returns a virtual file system representing its contents.
  static VFSFromYAML *create(std::unique_ptr<MemoryBuffer> Buffer,
                             SourceMgr::DiagHandlerTy DiagHandler,
                             void *DiagContext,
                             IntrusiveRefCntPtr<FileSystem> ExternalFS);

  ErrorOr<Status> status(const Twine &Path) override;
  ErrorOr<std::unique_ptr<File>> openFileForRead(const Twine &Path) override;

  directory_iterator dir_begin(const Twine &Dir, std::error_code &EC) override{
    ErrorOr<Entry *> E = lookupPath(Dir);
    if (!E) {
      EC = E.getError();
      return directory_iterator();
    }
    ErrorOr<Status> S = status(Dir, *E);
    if (!S) {
      EC = S.getError();
      return directory_iterator();
    }
    if (!S->isDirectory()) {
      EC = std::error_code(static_cast<int>(errc::not_a_directory),
                           std::system_category());
      return directory_iterator();
    }

    DirectoryEntry *D = cast<DirectoryEntry>(*E);
    return directory_iterator(std::make_shared<VFSFromYamlDirIterImpl>(Dir,
        *this, D->contents_begin(), D->contents_end(), EC));
  }
};

/// \brief A helper class to hold the common YAML parsing state.
class VFSFromYAMLParser {
  yaml::Stream &Stream;

  void error(yaml::Node *N, const Twine &Msg) {
    Stream.printError(N, Msg);
  }

  // false on error
  bool parseScalarString(yaml::Node *N, StringRef &Result,
                         SmallVectorImpl<char> &Storage) {
    yaml::ScalarNode *S = dyn_cast<yaml::ScalarNode>(N);
    if (!S) {
      error(N, "expected string");
      return false;
    }
    Result = S->getValue(Storage);
    return true;
  }

  // false on error
  bool parseScalarBool(yaml::Node *N, bool &Result) {
    SmallString<5> Storage;
    StringRef Value;
    if (!parseScalarString(N, Value, Storage))
      return false;

    if (Value.equals_lower("true") || Value.equals_lower("on") ||
        Value.equals_lower("yes") || Value == "1") {
      Result = true;
      return true;
    } else if (Value.equals_lower("false") || Value.equals_lower("off") ||
               Value.equals_lower("no") || Value == "0") {
      Result = false;
      return true;
    }

    error(N, "expected boolean value");
    return false;
  }

  struct KeyStatus {
    KeyStatus(bool Required=false) : Required(Required), Seen(false) {}
    bool Required;
    bool Seen;
  };
  typedef std::pair<StringRef, KeyStatus> KeyStatusPair;

  // false on error
  bool checkDuplicateOrUnknownKey(yaml::Node *KeyNode, StringRef Key,
                                  DenseMap<StringRef, KeyStatus> &Keys) {
    if (!Keys.count(Key)) {
      error(KeyNode, "unknown key");
      return false;
    }
    KeyStatus &S = Keys[Key];
    if (S.Seen) {
      error(KeyNode, Twine("duplicate key '") + Key + "'");
      return false;
    }
    S.Seen = true;
    return true;
  }

  // false on error
  bool checkMissingKeys(yaml::Node *Obj, DenseMap<StringRef, KeyStatus> &Keys) {
    for (DenseMap<StringRef, KeyStatus>::iterator I = Keys.begin(),
         E = Keys.end();
         I != E; ++I) {
      if (I->second.Required && !I->second.Seen) {
        error(Obj, Twine("missing key '") + I->first + "'");
        return false;
      }
    }
    return true;
  }

  Entry *parseEntry(yaml::Node *N) {
    yaml::MappingNode *M = dyn_cast<yaml::MappingNode>(N);
    if (!M) {
      error(N, "expected mapping node for file or directory entry");
      return nullptr;
    }

    KeyStatusPair Fields[] = {
      KeyStatusPair("name", true),
      KeyStatusPair("type", true),
      KeyStatusPair("contents", false),
      KeyStatusPair("external-contents", false),
      KeyStatusPair("use-external-name", false),
    };

    DenseMap<StringRef, KeyStatus> Keys(
        &Fields[0], Fields + sizeof(Fields)/sizeof(Fields[0]));

    bool HasContents = false; // external or otherwise
    std::vector<Entry *> EntryArrayContents;
    std::string ExternalContentsPath;
    std::string Name;
    FileEntry::NameKind UseExternalName = FileEntry::NK_NotSet;
    EntryKind Kind;

    for (yaml::MappingNode::iterator I = M->begin(), E = M->end(); I != E;
         ++I) {
      StringRef Key;
      // Reuse the buffer for key and value, since we don't look at key after
      // parsing value.
      SmallString<256> Buffer;
      if (!parseScalarString(I->getKey(), Key, Buffer))
        return nullptr;

      if (!checkDuplicateOrUnknownKey(I->getKey(), Key, Keys))
        return nullptr;

      StringRef Value;
      if (Key == "name") {
        if (!parseScalarString(I->getValue(), Value, Buffer))
          return nullptr;
        Name = Value;
      } else if (Key == "type") {
        if (!parseScalarString(I->getValue(), Value, Buffer))
          return nullptr;
        if (Value == "file")
          Kind = EK_File;
        else if (Value == "directory")
          Kind = EK_Directory;
        else {
          error(I->getValue(), "unknown value for 'type'");
          return nullptr;
        }
      } else if (Key == "contents") {
        if (HasContents) {
          error(I->getKey(),
                "entry already has 'contents' or 'external-contents'");
          return nullptr;
        }
        HasContents = true;
        yaml::SequenceNode *Contents =
            dyn_cast<yaml::SequenceNode>(I->getValue());
        if (!Contents) {
          // FIXME: this is only for directories, what about files?
          error(I->getValue(), "expected array");
          return nullptr;
        }

        for (yaml::SequenceNode::iterator I = Contents->begin(),
                                          E = Contents->end();
             I != E; ++I) {
          if (Entry *E = parseEntry(&*I))
            EntryArrayContents.push_back(E);
          else
            return nullptr;
        }
      } else if (Key == "external-contents") {
        if (HasContents) {
          error(I->getKey(),
                "entry already has 'contents' or 'external-contents'");
          return nullptr;
        }
        HasContents = true;
        if (!parseScalarString(I->getValue(), Value, Buffer))
          return nullptr;
        ExternalContentsPath = Value;
      } else if (Key == "use-external-name") {
        bool Val;
        if (!parseScalarBool(I->getValue(), Val))
          return nullptr;
        UseExternalName = Val ? FileEntry::NK_External : FileEntry::NK_Virtual;
      } else {
        llvm_unreachable("key missing from Keys");
      }
    }

    if (Stream.failed())
      return nullptr;

    // check for missing keys
    if (!HasContents) {
      error(N, "missing key 'contents' or 'external-contents'");
      return nullptr;
    }
    if (!checkMissingKeys(N, Keys))
      return nullptr;

    // check invalid configuration
    if (Kind == EK_Directory && UseExternalName != FileEntry::NK_NotSet) {
      error(N, "'use-external-name' is not supported for directories");
      return nullptr;
    }

    // Remove trailing slash(es), being careful not to remove the root path
    StringRef Trimmed(Name);
    size_t RootPathLen = sys::path::root_path(Trimmed).size();
    while (Trimmed.size() > RootPathLen &&
           sys::path::is_separator(Trimmed.back()))
      Trimmed = Trimmed.slice(0, Trimmed.size()-1);
    // Get the last component
    StringRef LastComponent = sys::path::filename(Trimmed);

    Entry *Result = nullptr;
    switch (Kind) {
    case EK_File:
      Result = new FileEntry(LastComponent, std::move(ExternalContentsPath),
                             UseExternalName);
      break;
    case EK_Directory:
      Result = new DirectoryEntry(LastComponent, std::move(EntryArrayContents),
          Status("", "", getNextVirtualUniqueID(), sys::TimeValue::now(), 0, 0,
                 0, file_type::directory_file, sys::fs::all_all));
      break;
    }

    StringRef Parent = sys::path::parent_path(Trimmed);
    if (Parent.empty())
      return Result;

    // if 'name' contains multiple components, create implicit directory entries
    for (sys::path::reverse_iterator I = sys::path::rbegin(Parent),
                                     E = sys::path::rend(Parent);
         I != E; ++I) {
      Result = new DirectoryEntry(*I, llvm::makeArrayRef(Result),
          Status("", "", getNextVirtualUniqueID(), sys::TimeValue::now(), 0, 0,
                 0, file_type::directory_file, sys::fs::all_all));
    }
    return Result;
  }

public:
  VFSFromYAMLParser(yaml::Stream &S) : Stream(S) {}

  // false on error
  bool parse(yaml::Node *Root, VFSFromYAML *FS) {
    yaml::MappingNode *Top = dyn_cast<yaml::MappingNode>(Root);
    if (!Top) {
      error(Root, "expected mapping node");
      return false;
    }

    KeyStatusPair Fields[] = {
      KeyStatusPair("version", true),
      KeyStatusPair("case-sensitive", false),
      KeyStatusPair("use-external-names", false),
      KeyStatusPair("roots", true),
    };

    DenseMap<StringRef, KeyStatus> Keys(
        &Fields[0], Fields + sizeof(Fields)/sizeof(Fields[0]));

    // Parse configuration and 'roots'
    for (yaml::MappingNode::iterator I = Top->begin(), E = Top->end(); I != E;
         ++I) {
      SmallString<10> KeyBuffer;
      StringRef Key;
      if (!parseScalarString(I->getKey(), Key, KeyBuffer))
        return false;

      if (!checkDuplicateOrUnknownKey(I->getKey(), Key, Keys))
        return false;

      if (Key == "roots") {
        yaml::SequenceNode *Roots = dyn_cast<yaml::SequenceNode>(I->getValue());
        if (!Roots) {
          error(I->getValue(), "expected array");
          return false;
        }

        for (yaml::SequenceNode::iterator I = Roots->begin(), E = Roots->end();
             I != E; ++I) {
          if (Entry *E = parseEntry(&*I))
            FS->Roots.push_back(E);
          else
            return false;
        }
      } else if (Key == "version") {
        StringRef VersionString;
        SmallString<4> Storage;
        if (!parseScalarString(I->getValue(), VersionString, Storage))
          return false;
        int Version;
        if (VersionString.getAsInteger<int>(10, Version)) {
          error(I->getValue(), "expected integer");
          return false;
        }
        if (Version < 0) {
          error(I->getValue(), "invalid version number");
          return false;
        }
        if (Version != 0) {
          error(I->getValue(), "version mismatch, expected 0");
          return false;
        }
      } else if (Key == "case-sensitive") {
        if (!parseScalarBool(I->getValue(), FS->CaseSensitive))
          return false;
      } else if (Key == "use-external-names") {
        if (!parseScalarBool(I->getValue(), FS->UseExternalNames))
          return false;
      } else {
        llvm_unreachable("key missing from Keys");
      }
    }

    if (Stream.failed())
      return false;

    if (!checkMissingKeys(Top, Keys))
      return false;
    return true;
  }
};
} // end of anonymous namespace

Entry::~Entry() {}
DirectoryEntry::~DirectoryEntry() { llvm::DeleteContainerPointers(Contents); }

VFSFromYAML::~VFSFromYAML() { llvm::DeleteContainerPointers(Roots); }

VFSFromYAML *VFSFromYAML::create(std::unique_ptr<MemoryBuffer> Buffer,
                                 SourceMgr::DiagHandlerTy DiagHandler,
                                 void *DiagContext,
                                 IntrusiveRefCntPtr<FileSystem> ExternalFS) {

  SourceMgr SM;
  yaml::Stream Stream(Buffer->getMemBufferRef(), SM);

  SM.setDiagHandler(DiagHandler, DiagContext);
  yaml::document_iterator DI = Stream.begin();
  yaml::Node *Root = DI->getRoot();
  if (DI == Stream.end() || !Root) {
    SM.PrintMessage(SMLoc(), SourceMgr::DK_Error, "expected root node");
    return nullptr;
  }

  VFSFromYAMLParser P(Stream);

  std::unique_ptr<VFSFromYAML> FS(new VFSFromYAML(ExternalFS));
  if (!P.parse(Root, FS.get()))
    return nullptr;

  return FS.release();
}

ErrorOr<Entry *> VFSFromYAML::lookupPath(const Twine &Path_) {
  SmallString<256> Path;
  Path_.toVector(Path);

  // Handle relative paths
  if (std::error_code EC = sys::fs::make_absolute(Path))
    return EC;

  if (Path.empty())
    return make_error_code(llvm::errc::invalid_argument);

  sys::path::const_iterator Start = sys::path::begin(Path);
  sys::path::const_iterator End = sys::path::end(Path);
  for (std::vector<Entry *>::iterator I = Roots.begin(), E = Roots.end();
       I != E; ++I) {
    ErrorOr<Entry *> Result = lookupPath(Start, End, *I);
    if (Result || Result.getError() != llvm::errc::no_such_file_or_directory)
      return Result;
  }
  return make_error_code(llvm::errc::no_such_file_or_directory);
}

ErrorOr<Entry *> VFSFromYAML::lookupPath(sys::path::const_iterator Start,
                                         sys::path::const_iterator End,
                                         Entry *From) {
  if (Start->equals("."))
    ++Start;

  // FIXME: handle ..
  if (CaseSensitive ? !Start->equals(From->getName())
                    : !Start->equals_lower(From->getName()))
    // failure to match
    return make_error_code(llvm::errc::no_such_file_or_directory);

  ++Start;

  if (Start == End) {
    // Match!
    return From;
  }

  DirectoryEntry *DE = dyn_cast<DirectoryEntry>(From);
  if (!DE)
    return make_error_code(llvm::errc::not_a_directory);

  for (DirectoryEntry::iterator I = DE->contents_begin(),
                                E = DE->contents_end();
       I != E; ++I) {
    ErrorOr<Entry *> Result = lookupPath(Start, End, *I);
    if (Result || Result.getError() != llvm::errc::no_such_file_or_directory)
      return Result;
  }
  return make_error_code(llvm::errc::no_such_file_or_directory);
}

ErrorOr<Status> VFSFromYAML::status(const Twine &Path, Entry *E) {
  assert(E != nullptr);
  std::string PathStr(Path.str());
  if (FileEntry *F = dyn_cast<FileEntry>(E)) {
    ErrorOr<Status> S = ExternalFS->status(F->getExternalContentsPath());
    assert(!S || S->getName() == F->getExternalContentsPath());
    if (S && !F->useExternalName(UseExternalNames))
      S->setName(PathStr);
    if (S)
      S->IsVFSMapped = true;
    return S;
  } else { // directory
    DirectoryEntry *DE = cast<DirectoryEntry>(E);
    Status S = DE->getStatus();
    S.setName(PathStr);
    return S;
  }
}

ErrorOr<Status> VFSFromYAML::status(const Twine &Path) {
  ErrorOr<Entry *> Result = lookupPath(Path);
  if (!Result)
    return Result.getError();
  return status(Path, *Result);
}

ErrorOr<std::unique_ptr<File>> VFSFromYAML::openFileForRead(const Twine &Path) {
  ErrorOr<Entry *> E = lookupPath(Path);
  if (!E)
    return E.getError();

  FileEntry *F = dyn_cast<FileEntry>(*E);
  if (!F) // FIXME: errc::not_a_file?
    return make_error_code(llvm::errc::invalid_argument);

  auto Result = ExternalFS->openFileForRead(F->getExternalContentsPath());
  if (!Result)
    return Result;

  if (!F->useExternalName(UseExternalNames))
    (*Result)->setName(Path.str());

  return Result;
}

IntrusiveRefCntPtr<FileSystem>
vfs::getVFSFromYAML(std::unique_ptr<MemoryBuffer> Buffer,
                    SourceMgr::DiagHandlerTy DiagHandler, void *DiagContext,
                    IntrusiveRefCntPtr<FileSystem> ExternalFS) {
  return VFSFromYAML::create(std::move(Buffer), DiagHandler, DiagContext,
                             ExternalFS);
}

UniqueID vfs::getNextVirtualUniqueID() {
  static std::atomic<unsigned> UID;
  unsigned ID = ++UID;
  // The following assumes that uint64_t max will never collide with a real
  // dev_t value from the OS.
  return UniqueID(std::numeric_limits<uint64_t>::max(), ID);
}

#ifndef NDEBUG
static bool pathHasTraversal(StringRef Path) {
  using namespace llvm::sys;
  for (StringRef Comp : llvm::make_range(path::begin(Path), path::end(Path)))
    if (Comp == "." || Comp == "..")
      return true;
  return false;
}
#endif

void YAMLVFSWriter::addFileMapping(StringRef VirtualPath, StringRef RealPath) {
  assert(sys::path::is_absolute(VirtualPath) && "virtual path not absolute");
  assert(sys::path::is_absolute(RealPath) && "real path not absolute");
  assert(!pathHasTraversal(VirtualPath) && "path traversal is not supported");
  Mappings.emplace_back(VirtualPath, RealPath);
}

namespace {
class JSONWriter {
  llvm::raw_ostream &OS;
  SmallVector<StringRef, 16> DirStack;
  inline unsigned getDirIndent() { return 4 * DirStack.size(); }
  inline unsigned getFileIndent() { return 4 * (DirStack.size() + 1); }
  bool containedIn(StringRef Parent, StringRef Path);
  StringRef containedPart(StringRef Parent, StringRef Path);
  void startDirectory(StringRef Path);
  void endDirectory();
  void writeEntry(StringRef VPath, StringRef RPath);

public:
  JSONWriter(llvm::raw_ostream &OS) : OS(OS) {}
  void write(ArrayRef<YAMLVFSEntry> Entries, Optional<bool> IsCaseSensitive);
};
}

bool JSONWriter::containedIn(StringRef Parent, StringRef Path) {
  using namespace llvm::sys;
  // Compare each path component.
  auto IParent = path::begin(Parent), EParent = path::end(Parent);
  for (auto IChild = path::begin(Path), EChild = path::end(Path);
       IParent != EParent && IChild != EChild; ++IParent, ++IChild) {
    if (*IParent != *IChild)
      return false;
  }
  // Have we exhausted the parent path?
  return IParent == EParent;
}

StringRef JSONWriter::containedPart(StringRef Parent, StringRef Path) {
  assert(!Parent.empty());
  assert(containedIn(Parent, Path));
  return Path.slice(Parent.size() + 1, StringRef::npos);
}

void JSONWriter::startDirectory(StringRef Path) {
  StringRef Name =
      DirStack.empty() ? Path : containedPart(DirStack.back(), Path);
  DirStack.push_back(Path);
  unsigned Indent = getDirIndent();
  OS.indent(Indent) << "{\n";
  OS.indent(Indent + 2) << "'type': 'directory',\n";
  OS.indent(Indent + 2) << "'name': \"" << llvm::yaml::escape(Name) << "\",\n";
  OS.indent(Indent + 2) << "'contents': [\n";
}

void JSONWriter::endDirectory() {
  unsigned Indent = getDirIndent();
  OS.indent(Indent + 2) << "]\n";
  OS.indent(Indent) << "}";

  DirStack.pop_back();
}

void JSONWriter::writeEntry(StringRef VPath, StringRef RPath) {
  unsigned Indent = getFileIndent();
  OS.indent(Indent) << "{\n";
  OS.indent(Indent + 2) << "'type': 'file',\n";
  OS.indent(Indent + 2) << "'name': \"" << llvm::yaml::escape(VPath) << "\",\n";
  OS.indent(Indent + 2) << "'external-contents': \""
                        << llvm::yaml::escape(RPath) << "\"\n";
  OS.indent(Indent) << "}";
}

void JSONWriter::write(ArrayRef<YAMLVFSEntry> Entries,
                       Optional<bool> IsCaseSensitive) {
  using namespace llvm::sys;

  OS << "{\n"
        "  'version': 0,\n";
  if (IsCaseSensitive.hasValue())
    OS << "  'case-sensitive': '"
       << (IsCaseSensitive.getValue() ? "true" : "false") << "',\n";
  OS << "  'roots': [\n";

  if (!Entries.empty()) {
    const YAMLVFSEntry &Entry = Entries.front();
    startDirectory(path::parent_path(Entry.VPath));
    writeEntry(path::filename(Entry.VPath), Entry.RPath);

    for (const auto &Entry : Entries.slice(1)) {
      StringRef Dir = path::parent_path(Entry.VPath);
      if (Dir == DirStack.back())
        OS << ",\n";
      else {
        while (!DirStack.empty() && !containedIn(DirStack.back(), Dir)) {
          OS << "\n";
          endDirectory();
        }
        OS << ",\n";
        startDirectory(Dir);
      }
      writeEntry(path::filename(Entry.VPath), Entry.RPath);
    }

    while (!DirStack.empty()) {
      OS << "\n";
      endDirectory();
    }
    OS << "\n";
  }

  OS << "  ]\n"
     << "}\n";
}

void YAMLVFSWriter::write(llvm::raw_ostream &OS) {
  std::sort(Mappings.begin(), Mappings.end(),
            [](const YAMLVFSEntry &LHS, const YAMLVFSEntry &RHS) {
    return LHS.VPath < RHS.VPath;
  });

  JSONWriter(OS).write(Mappings, IsCaseSensitive);
}

VFSFromYamlDirIterImpl::VFSFromYamlDirIterImpl(const Twine &_Path,
                                               VFSFromYAML &FS,
                                               DirectoryEntry::iterator Begin,
                                               DirectoryEntry::iterator End,
                                               std::error_code &EC)
    : Dir(_Path.str()), FS(FS), Current(Begin), End(End) {
  if (Current != End) {
    SmallString<128> PathStr(Dir);
    llvm::sys::path::append(PathStr, (*Current)->getName());
    llvm::ErrorOr<vfs::Status> S = FS.status(PathStr);
    if (S)
      CurrentEntry = *S;
    else
      EC = S.getError();
  }
}

std::error_code VFSFromYamlDirIterImpl::increment() {
  assert(Current != End && "cannot iterate past end");
  if (++Current != End) {
    SmallString<128> PathStr(Dir);
    llvm::sys::path::append(PathStr, (*Current)->getName());
    llvm::ErrorOr<vfs::Status> S = FS.status(PathStr);
    if (!S)
      return S.getError();
    CurrentEntry = *S;
  } else {
    CurrentEntry = Status();
  }
  return std::error_code();
}

vfs::recursive_directory_iterator::recursive_directory_iterator(FileSystem &FS_,
                                                           const Twine &Path,
                                                           std::error_code &EC)
    : FS(&FS_) {
  directory_iterator I = FS->dir_begin(Path, EC);
  if (!EC && I != directory_iterator()) {
    State = std::make_shared<IterState>();
    State->push(I);
  }
}

vfs::recursive_directory_iterator &
recursive_directory_iterator::increment(std::error_code &EC) {
  assert(FS && State && !State->empty() && "incrementing past end");
  assert(State->top()->isStatusKnown() && "non-canonical end iterator");
  vfs::directory_iterator End;
  if (State->top()->isDirectory()) {
    vfs::directory_iterator I = FS->dir_begin(State->top()->getName(), EC);
    if (EC)
      return *this;
    if (I != End) {
      State->push(I);
      return *this;
    }
  }

  while (!State->empty() && State->top().increment(EC) == End)
    State->pop();

  if (State->empty())
    State.reset(); // end iterator

  return *this;
}
