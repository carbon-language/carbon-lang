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
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Atomic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLParser.h"

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
      Type(Status.type()), Perms(Status.permissions()) {}

Status::Status(StringRef Name, StringRef ExternalName, UniqueID UID,
               sys::TimeValue MTime, uint32_t User, uint32_t Group,
               uint64_t Size, file_type Type, perms Perms)
    : Name(Name), UID(UID), MTime(MTime), User(User), Group(Group), Size(Size),
      Type(Type), Perms(Perms) {}

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

error_code FileSystem::getBufferForFile(const llvm::Twine &Name,
                                        OwningPtr<MemoryBuffer> &Result,
                                        int64_t FileSize,
                                        bool RequiresNullTerminator) {
  llvm::OwningPtr<File> F;
  if (error_code EC = openFileForRead(Name, F))
    return EC;

  error_code EC = F->getBuffer(Name, Result, FileSize, RequiresNullTerminator);
  return EC;
}

//===-----------------------------------------------------------------------===/
// RealFileSystem implementation
//===-----------------------------------------------------------------------===/

/// \brief Wrapper around a raw file descriptor.
class RealFile : public File {
  int FD;
  friend class RealFileSystem;
  RealFile(int FD) : FD(FD) {
    assert(FD >= 0 && "Invalid or inactive file descriptor");
  }

public:
  ~RealFile();
  ErrorOr<Status> status() LLVM_OVERRIDE;
  error_code getBuffer(const Twine &Name, OwningPtr<MemoryBuffer> &Result,
                       int64_t FileSize = -1,
                       bool RequiresNullTerminator = true) LLVM_OVERRIDE;
  error_code close() LLVM_OVERRIDE;
};
RealFile::~RealFile() { close(); }

ErrorOr<Status> RealFile::status() {
  assert(FD != -1 && "cannot stat closed file");
  file_status RealStatus;
  if (error_code EC = sys::fs::status(FD, RealStatus))
    return EC;
  return Status(RealStatus);
}

error_code RealFile::getBuffer(const Twine &Name,
                               OwningPtr<MemoryBuffer> &Result,
                               int64_t FileSize, bool RequiresNullTerminator) {
  assert(FD != -1 && "cannot get buffer for closed file");
  return MemoryBuffer::getOpenFile(FD, Name.str().c_str(), Result, FileSize,
                                   RequiresNullTerminator);
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
error_code RealFile::close() {
  if (::close(FD))
    return error_code(errno, system_category());
  FD = -1;
  return error_code::success();
}

/// \brief The file system according to your operating system.
class RealFileSystem : public FileSystem {
public:
  ErrorOr<Status> status(const Twine &Path) LLVM_OVERRIDE;
  error_code openFileForRead(const Twine &Path,
                             OwningPtr<File> &Result) LLVM_OVERRIDE;
};

ErrorOr<Status> RealFileSystem::status(const Twine &Path) {
  sys::fs::file_status RealStatus;
  if (error_code EC = sys::fs::status(Path, RealStatus))
    return EC;
  Status Result(RealStatus);
  Result.setName(Path.str());
  return Result;
}

error_code RealFileSystem::openFileForRead(const Twine &Name,
                                           OwningPtr<File> &Result) {
  int FD;
  if (error_code EC = sys::fs::openFileForRead(Name, FD))
    return EC;
  Result.reset(new RealFile(FD));
  return error_code::success();
}

IntrusiveRefCntPtr<FileSystem> vfs::getRealFileSystem() {
  static IntrusiveRefCntPtr<FileSystem> FS = new RealFileSystem();
  return FS;
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
    if (Status || Status.getError() != errc::no_such_file_or_directory)
      return Status;
  }
  return error_code(errc::no_such_file_or_directory, system_category());
}

error_code OverlayFileSystem::openFileForRead(const llvm::Twine &Path,
                                              OwningPtr<File> &Result) {
  // FIXME: handle symlinks that cross file systems
  for (iterator I = overlays_begin(), E = overlays_end(); I != E; ++I) {
    error_code EC = (*I)->openFileForRead(Path, Result);
    if (!EC || EC != errc::no_such_file_or_directory)
      return EC;
  }
  return error_code(errc::no_such_file_or_directory, system_category());
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
  virtual ~DirectoryEntry();
#if LLVM_HAS_RVALUE_REFERENCES
  DirectoryEntry(StringRef Name, std::vector<Entry *> Contents, Status S)
      : Entry(EK_Directory, Name), Contents(std::move(Contents)),
        S(std::move(S)) {}
#endif
  DirectoryEntry(StringRef Name, ArrayRef<Entry *> Contents, const Status &S)
      : Entry(EK_Directory, Name), Contents(Contents), S(S) {}
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
  NameKind useName() const { return UseName; }
  static bool classof(const Entry *E) { return E->getKind() == EK_File; }
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

public:
  ~VFSFromYAML();

  /// \brief Parses \p Buffer, which is expected to be in YAML format and
  /// returns a virtual file system representing its contents.
  ///
  /// Takes ownership of \p Buffer.
  static VFSFromYAML *create(MemoryBuffer *Buffer,
                             SourceMgr::DiagHandlerTy DiagHandler,
                             void *DiagContext,
                             IntrusiveRefCntPtr<FileSystem> ExternalFS);

  ErrorOr<Status> status(const Twine &Path) LLVM_OVERRIDE;
  error_code openFileForRead(const Twine &Path,
                             OwningPtr<File> &Result) LLVM_OVERRIDE;
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
      return NULL;
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
        return NULL;

      if (!checkDuplicateOrUnknownKey(I->getKey(), Key, Keys))
        return NULL;

      StringRef Value;
      if (Key == "name") {
        if (!parseScalarString(I->getValue(), Value, Buffer))
          return NULL;
        Name = Value;
      } else if (Key == "type") {
        if (!parseScalarString(I->getValue(), Value, Buffer))
          return NULL;
        if (Value == "file")
          Kind = EK_File;
        else if (Value == "directory")
          Kind = EK_Directory;
        else {
          error(I->getValue(), "unknown value for 'type'");
          return NULL;
        }
      } else if (Key == "contents") {
        if (HasContents) {
          error(I->getKey(),
                "entry already has 'contents' or 'external-contents'");
          return NULL;
        }
        HasContents = true;
        yaml::SequenceNode *Contents =
            dyn_cast<yaml::SequenceNode>(I->getValue());
        if (!Contents) {
          // FIXME: this is only for directories, what about files?
          error(I->getValue(), "expected array");
          return NULL;
        }

        for (yaml::SequenceNode::iterator I = Contents->begin(),
                                          E = Contents->end();
             I != E; ++I) {
          if (Entry *E = parseEntry(&*I))
            EntryArrayContents.push_back(E);
          else
            return NULL;
        }
      } else if (Key == "external-contents") {
        if (HasContents) {
          error(I->getKey(),
                "entry already has 'contents' or 'external-contents'");
          return NULL;
        }
        HasContents = true;
        if (!parseScalarString(I->getValue(), Value, Buffer))
          return NULL;
        ExternalContentsPath = Value;
      } else if (Key == "use-external-name") {
        bool Val;
        if (!parseScalarBool(I->getValue(), Val))
          return NULL;
        UseExternalName = Val ? FileEntry::NK_External : FileEntry::NK_Virtual;
      } else {
        llvm_unreachable("key missing from Keys");
      }
    }

    if (Stream.failed())
      return NULL;

    // check for missing keys
    if (!HasContents) {
      error(N, "missing key 'contents' or 'external-contents'");
      return NULL;
    }
    if (!checkMissingKeys(N, Keys))
      return NULL;

    // check invalid configuration
    if (Kind == EK_Directory && UseExternalName != FileEntry::NK_NotSet) {
      error(N, "'use-external-name' is not supported for directories");
      return NULL;
    }

    // Remove trailing slash(es)
    StringRef Trimmed(Name);
    while (Trimmed.size() > 1 && sys::path::is_separator(Trimmed.back()))
      Trimmed = Trimmed.slice(0, Trimmed.size()-1);
    // Get the last component
    StringRef LastComponent = sys::path::filename(Trimmed);

    Entry *Result = 0;
    switch (Kind) {
    case EK_File:
      Result = new FileEntry(LastComponent, llvm_move(ExternalContentsPath),
                             UseExternalName);
      break;
    case EK_Directory:
      Result = new DirectoryEntry(LastComponent, llvm_move(EntryArrayContents),
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
      Result = new DirectoryEntry(*I, Result,
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

VFSFromYAML *VFSFromYAML::create(MemoryBuffer *Buffer,
                                 SourceMgr::DiagHandlerTy DiagHandler,
                                 void *DiagContext,
                                 IntrusiveRefCntPtr<FileSystem> ExternalFS) {

  SourceMgr SM;
  yaml::Stream Stream(Buffer, SM);

  SM.setDiagHandler(DiagHandler, DiagContext);
  yaml::document_iterator DI = Stream.begin();
  yaml::Node *Root = DI->getRoot();
  if (DI == Stream.end() || !Root) {
    SM.PrintMessage(SMLoc(), SourceMgr::DK_Error, "expected root node");
    return NULL;
  }

  VFSFromYAMLParser P(Stream);

  OwningPtr<VFSFromYAML> FS(new VFSFromYAML(ExternalFS));
  if (!P.parse(Root, FS.get()))
    return NULL;

  return FS.take();
}

ErrorOr<Entry *> VFSFromYAML::lookupPath(const Twine &Path_) {
  SmallVector<char, 256> Storage;
  StringRef Path = Path_.toNullTerminatedStringRef(Storage);

  if (Path.empty())
    return error_code(errc::invalid_argument, system_category());

  sys::path::const_iterator Start = sys::path::begin(Path);
  sys::path::const_iterator End = sys::path::end(Path);
  for (std::vector<Entry *>::iterator I = Roots.begin(), E = Roots.end();
       I != E; ++I) {
    ErrorOr<Entry *> Result = lookupPath(Start, End, *I);
    if (Result || Result.getError() != errc::no_such_file_or_directory)
      return Result;
  }
  return error_code(errc::no_such_file_or_directory, system_category());
}

ErrorOr<Entry *> VFSFromYAML::lookupPath(sys::path::const_iterator Start,
                                         sys::path::const_iterator End,
                                         Entry *From) {
  // FIXME: handle . and ..
  if (CaseSensitive ? !Start->equals(From->getName())
                    : !Start->equals_lower(From->getName()))
    // failure to match
    return error_code(errc::no_such_file_or_directory, system_category());

  ++Start;

  if (Start == End) {
    // Match!
    return From;
  }

  DirectoryEntry *DE = dyn_cast<DirectoryEntry>(From);
  if (!DE)
    return error_code(errc::not_a_directory, system_category());

  for (DirectoryEntry::iterator I = DE->contents_begin(),
                                E = DE->contents_end();
       I != E; ++I) {
    ErrorOr<Entry *> Result = lookupPath(Start, End, *I);
    if (Result || Result.getError() != errc::no_such_file_or_directory)
      return Result;
  }
  return error_code(errc::no_such_file_or_directory, system_category());
}

ErrorOr<Status> VFSFromYAML::status(const Twine &Path) {
  ErrorOr<Entry *> Result = lookupPath(Path);
  if (!Result)
    return Result.getError();

  std::string PathStr(Path.str());
  if (FileEntry *F = dyn_cast<FileEntry>(*Result)) {
    ErrorOr<Status> S = ExternalFS->status(F->getExternalContentsPath());
    assert(!S || S->getName() == F->getExternalContentsPath());
    if (S && (F->useName() == FileEntry::NK_Virtual ||
              (F->useName() == FileEntry::NK_NotSet && !UseExternalNames)))
      S->setName(PathStr);
    return S;
  } else { // directory
    DirectoryEntry *DE = cast<DirectoryEntry>(*Result);
    Status S = DE->getStatus();
    S.setName(PathStr);
    return S;
  }
}

error_code VFSFromYAML::openFileForRead(const Twine &Path,
                                        OwningPtr<vfs::File> &Result) {
  ErrorOr<Entry *> E = lookupPath(Path);
  if (!E)
    return E.getError();

  FileEntry *F = dyn_cast<FileEntry>(*E);
  if (!F) // FIXME: errc::not_a_file?
    return error_code(errc::invalid_argument, system_category());

  return ExternalFS->openFileForRead(F->getExternalContentsPath(), Result);
}

IntrusiveRefCntPtr<FileSystem>
vfs::getVFSFromYAML(MemoryBuffer *Buffer, SourceMgr::DiagHandlerTy DiagHandler,
                    void *DiagContext,
                    IntrusiveRefCntPtr<FileSystem> ExternalFS) {
  return VFSFromYAML::create(Buffer, DiagHandler, DiagContext, ExternalFS);
}

UniqueID vfs::getNextVirtualUniqueID() {
  static volatile sys::cas_flag UID = 0;
  sys::cas_flag ID = llvm::sys::AtomicIncrement(&UID);
  // The following assumes that uint64_t max will never collide with a real
  // dev_t value from the OS.
  return UniqueID(std::numeric_limits<uint64_t>::max(), ID);
}
