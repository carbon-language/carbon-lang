// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_FILESYSTEM_H_
#define CARBON_COMMON_FILESYSTEM_H_

#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <concepts>
#include <filesystem>
#include <iterator>
#include <string>

#include "common/check.h"
#include "common/error.h"
#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "common/template_string.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

// Provides a filesystem library for use in the Carbon project.
//
// This library provides an API designed to support modern Unix / Linux / POSIX
// style filesystem operations, often called "Unix-like"[1] here, efficiently
// and securely, while also carefully staying to a set of abstractions and
// operations that can be reasonably implemented even on Windows platforms.
//
// TODO: Currently, there is not a Windows implementation, but this is actively
// desired when we have testing infrastructure in place for Windows development.
// Lacking that testing infrastructure and a full Windows port, the operations
// here are manually compared with LLVM's filesystem library to ensure a
// reasonable Windows implementation is possible.
//
// The library uses C++'s `std::filesystem::path` as its abstraction for
// paths. This library provides two core APIs: open directories and files.
//
// Open directories provide relative- and absolute-path based operations to open
// other directories or files. This allows secure creation of directories even
// in the face of adversarial operations for example in a shared `/tmp`
// directory. There is a `constexpr` current working directory available as
// `Cwd()` that models normal filesystem operations with paths.
//
// Open files provide read, write, and other operations on the file. There are
// separate types for read-only, write-only, and read-write files to model the
// different APIs available.
//
// The APIs for both directories and files are primarily on `*Ref` types that
// model a non-owning reference to the directory or file. These types are the
// preferred types to use on an API boundary. Owning versions are provided that
// ensure the file or directory is closed on destruction. Files support explicit
// closing in order to observe any close-specific errors.
//
// Where APIs require flag parameters of some form, this library provides
// enumerations that model those flags. The enumeration values are in turn
// chosen to simplify passing these to specific native APIs. This means the
// enumeration *values* should not be expected to be portable across platforms.
// Customizing the values is part of the larger TODO to port the implementation
// to Windows.
//
// [1]: Note that we refer to platforms as "Unix-like" rather than POSIX as we
//      want to group together all the OSes where the Unix-derived APIs are the
//      primary and expected way to interact with the filesystem, regardless of
//      whether a POSIX conforming API happens to exist. For example, both macOS
//      and WSL (Windows Subsystem for Linux) _are_ Unix-like as those are the
//      primary APIs used to access files in those environments. But Windows
//      itself _isn't_ Unix-like, even considering things like the defunct NT
//      POSIX subsystem or modern WSL, as those aren't the primary filesystem
//      APIs for the (non-WSL) Windows platform. This also matches the rough OS
//      classification used in LLVM.
namespace Carbon::Filesystem {

// The different creation options available when opening a file or directory.
//
// Because these are by far the most common parameters and they have unambiguous
// names, the enumerators are also available directly within the namespace.
enum class CreationOptions {
  // Requires an existing file or directory.
  OpenExisting = 0,

  // Opens an existing file or directory, and create one otherwise.
  OpenAlways = O_CREAT,

  // Opens and truncates an existing file or creates a new file. Provides
  // consistent behavior of an empty file regardless of the starting state. This
  // cannot be used for directories as they cannot be truncated on open. This is
  // essentially a short-cut for using `OpenAlways` and passing the
  // `OpenFlags::Truncate` below.
  CreateAlways = O_CREAT | O_TRUNC,

  // Requires no existing file or directory and will error if one is found. Only
  // succeeds when it creates a new file or directory.
  CreateNew = O_CREAT | O_EXCL,
};
using enum CreationOptions;

// General flags to control the behavior of opening files that aren't covered by
// other more specific flags.
//
// These can be combined using the `|` operator where the semantics are
// compatible, although not all are.
enum class OpenFlags : int {
  None = 0,

  // Open the file for appending rather than with the position at the start.
  //
  // An error to combine with `Truncate` or to use with `CreateAlways`.
  Append = O_APPEND,

  // Open the file and truncate its contents to be empty.
  Truncate = O_TRUNC,

  // Don't follow a symlink in the final path component being opened.
  NoFollow = O_NOFOLLOW,
};
inline auto operator|(OpenFlags lhs, OpenFlags rhs) -> OpenFlags {
  return static_cast<OpenFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
}
inline auto operator|=(OpenFlags& lhs, OpenFlags rhs) -> OpenFlags& {
  lhs = lhs | rhs;
  return lhs;
}

// Flags controlling which permissions should be checked in an `Access` call.
//
// These permissions can also be combined with the `|` operator, so
// `AccessCheckFlags::Read | AccessCheckFlags::Write` checks for both read and
// write access.
enum class AccessCheckFlags : int {
  Exists = F_OK,
  Read = R_OK,
  Write = W_OK,
  Execute = X_OK,
};
inline auto operator|(AccessCheckFlags lhs, AccessCheckFlags rhs)
    -> AccessCheckFlags {
  return static_cast<AccessCheckFlags>(static_cast<int>(lhs) |
                                       static_cast<int>(rhs));
}
inline auto operator|=(AccessCheckFlags& lhs, AccessCheckFlags rhs)
    -> AccessCheckFlags& {
  lhs = lhs | rhs;
  return lhs;
}

// The underlying integer type that should be used to model the mode of a file.
//
// The mode is used in this API to represent both the permission bit mask and
// special properties of a file. For example, on Unix-like systems, it combines
// permissions with set-user-ID, set-group-ID, and sticky bits.
//
// The permission bits in the mode are represented using the Unix-style bit
// pattern that facilitates octal modeling:
// - Owner bit mask: 0700
// - Group bit mask: 0070
// - All bit mask:   0007
//
// For each, read is an octal value of `1`, write `2`, and execute `4`.
//
// Windows gracefully degrades to the effective permissions modeled using
// these values.
using ModeType = mode_t;

// Enumeration of the different file types recognized.
//
// In addition to the specific type values being arranged for ease of use with
// the POSIX APIs, the underlying type of the enum is arranged to use the common
// mode type.
enum class FileType : ModeType {
  // Portable file types that need to be supported across platform
  // implementations.
  Directory = S_IFDIR,
  RegularFile = S_IFREG,
  SymbolicLink = S_IFLNK,

  // Non-portable Unix-like platform specific types.
  UnixFifo = S_IFIFO,
  UnixCharDevice = S_IFCHR,
  UnixBlockDevice = S_IFBLK,
  UnixSocket = S_IFSOCK,

  // Mask for the Unix-like types to allow easy extraction.
  UnixMask = S_IFMT,
};

using Clock = std::chrono::file_clock;
using Duration = std::chrono::duration<__int128, std::nano>;
using TimePoint = std::chrono::time_point<Clock, Duration>;

// Enumerates the different open access modes available.
//
// These are largely used to parameterize types in order to constrain which API
// subset is available, and rarely needed directly.
enum class OpenAccess {
  ReadOnly = O_RDONLY,
  WriteOnly = O_WRONLY,
  ReadWrite = O_RDWR,
};

// Forward declarations of various types that appear in APIs.
class DirRef;
class Dir;
class RemovingDir;
template <OpenAccess A>
class FileRef;
template <OpenAccess A>
class File;
class FdError;
class PathError;
namespace Internal {
class FileRefBase;
}  // namespace Internal

// Returns a constant `Dir` object that models the open current working
// directory.
//
// Whatever the working directory of the process is will be used as the base for
// any relative path operations on this object. For example, on Unix-like
// systems, `Cwd().Stat("some/path")` is equivalent to `stat("some/path")`.
consteval auto Cwd() -> Dir;

// Creates a temporary directory and returns a removing directory handle to it.
//
// Each directory created will be unique and newly created by the call. It is
// the caller's responsibility to clean up this directory.
auto MakeTmpDir() -> ErrorOr<RemovingDir, Error>;

// Creates a temporary directory and returns a removing directory handle to it.
//
// The path of the temporary directory will use the provided prefix, and will
// not add any additional directory separators. Every component but the last in
// the prefix path must exist and be a directory, the last directory must be
// writable.
auto MakeTmpDirWithPrefix(std::filesystem::path prefix)
    -> ErrorOr<RemovingDir, Error>;

// Class modeling a file (or directory) status information structure.
//
// This provides a largely-portable model that callers can use, as well as a few
// APIs to access non-portable implementation details when necessary.
class FileStatus {
 public:
  // The size of the file in bytes.
  auto size() const -> int64_t { return stat_buf_.st_size; }

  auto type() const -> FileType {
    return static_cast<FileType>(stat_buf_.st_mode &
                                 static_cast<ModeType>(FileType::UnixMask));
  }

  // Convenience predicates to test for specific values of `type()`.
  auto is_dir() const -> bool { return type() == FileType::Directory; }
  auto is_file() const -> bool { return type() == FileType::RegularFile; }
  auto is_symlink() const -> bool { return type() == FileType::SymbolicLink; }

  // The read, write, and execute permissions for user, group, and others. See
  // the `ModeType` documentation for how to interpret the result.
  auto permissions() const -> ModeType { return stat_buf_.st_mode & 0777; }

  auto mtime() const -> TimePoint {
    // Linux, FreeBSD, and OpenBSD all use `st_mtim`, but macOS uses a different
    // spelling.
#if __APPLE__
    timespec ts = stat_buf_.st_mtimespec;
#else
    timespec ts = stat_buf_.st_mtim;
#endif
    TimePoint t(std::chrono::seconds(ts.tv_sec));
    return t + std::chrono::nanoseconds(ts.tv_nsec);
  }

  // Non-portable APIs only available on Unix-like systems. See the
  // documentation of the Unix `stat` structure fields they expose for their
  // meaning.
  auto unix_inode() const -> uint64_t { return stat_buf_.st_ino; }
  auto unix_uid() const -> uid_t { return stat_buf_.st_uid; }

 private:
  friend DirRef;
  friend Internal::FileRefBase;

  FileStatus() = default;

  struct stat stat_buf_ = {};
};

// Models a held lock on a file or directory.
//
// Can model either a shared or exclusive lock with respect to the file, but the
// held lock is unique.
//
// Must be released prior to the underlying file or directory being closed as it
// contains a non-owning reference to that underlying file.
class FileLock {
 public:
  enum class Kind {
    Shared = LOCK_SH,
    Exclusive = LOCK_EX,
  };
  using enum Kind;

  FileLock() = default;
  FileLock(FileLock&& arg) noexcept : fd_(std::exchange(arg.fd_, -1)) {}
  auto operator=(FileLock&& arg) noexcept -> FileLock& {
    Destroy();
    fd_ = std::exchange(arg.fd_, -1);
    return *this;
  }
  ~FileLock() { Destroy(); }

  // Returns true if the lock is currently held.
  auto is_locked() const -> bool { return fd_ != -1; }

 private:
  friend Internal::FileRefBase;

  explicit FileLock(int fd) : fd_(fd) {}

  auto Destroy() -> void;

  int fd_ = -1;
};

// The base class defining the core `File` API.
//
// While not used directly, this is the base class used to implement all of the
// main `File` types: `ReadFileRef`, `WriteFileRef`, and `ReadWriteFileRef`.
//
// Objects using this type have access to an open file handle to a specific file
// and expose operations on that open file. These operations may fail directly
// with their `ErrorOr` return, but some errors may be deferred until the
// underlying owning file is closed.
//
// The type provides reference semantics to the underlying file, but is
// rebindable, movable, and copyable unlike a C++ language reference.
class Internal::FileRefBase {
 public:
  // This object can be default constructed, but will hold an invalid file
  // handle in that case. This is to support rebinding operations.
  FileRefBase() = default;

  // Returns true if this refers to a valid open file, and false otherwise.
  auto is_valid() const -> bool { return fd_ != -1; }

  // Reads the file status.
  //
  // Analogous to the Unix-like `fstat` call.
  auto Stat() -> ErrorOr<FileStatus, FdError>;

  // Updates the access and modification times for the open file.
  //
  // If no explicit `time_point` is provided, sets both times to the current
  // time. If an explicit `time_point` is provided, the times are updated to
  // that time.
  auto UpdateTimes(std::optional<TimePoint> time_point = std::nullopt)
      -> ErrorOr<Success, FdError>;

  // Methods to seek the current file position, with various semantics for the
  // offset.
  auto Seek(int64_t delta) -> ErrorOr<int64_t, FdError>;
  auto SeekFromBeginning(int64_t delta_from_beginning)
      -> ErrorOr<int64_t, FdError>;
  auto SeekFromEnd(int64_t delta_from_end) -> ErrorOr<int64_t, FdError>;

  // Truncates or extends the size of the file to the provided size.
  //
  // If the new size is smaller, all bytes beyond this size will be unavailable.
  // If the new size is larger, the file will be zero-filled to the new size.
  // The position of reads and writes does not change.
  auto Truncate(int64_t new_size) -> ErrorOr<Success, FdError>;

  // Reads as much data as is available and fits into the provided buffer.
  //
  // On success, this returns a new slice from the start to the end of the
  // successfully read bytes. These will always be located in the passed-in
  // buffer, but not all of the buffer may be filled. A partial read does not
  // mean that the end of the file has been reached.
  //
  // When a successful read with an *empty* slice is returned, that represents
  // reaching EOF on the underlying file successfully and there is no more data
  // to read.
  //
  // This method retries `EINTR` on Unix-like systems and returns
  // other errors to the caller.
  auto ReadToBuffer(llvm::MutableArrayRef<std::byte> buffer)
      -> ErrorOr<llvm::MutableArrayRef<std::byte>, FdError>;

  // Writes as much data as possible from the provided buffer.
  //
  // On success, this returns a new slice of the *unwritten* bytes still present
  // in the buffer. An empty return represents a successful write of all bytes
  // in the buffer. A non-empty return does not represent an error or the
  // inability to finish writing.
  //
  // This method retries `EINTR` on Unix-like systems and returns
  // other errors to the caller.
  auto WriteFromBuffer(llvm::ArrayRef<std::byte> buffer)
      -> ErrorOr<llvm::ArrayRef<std::byte>, FdError>;

  // Returns an LLVM `raw_fd_ostream` that writes to this file.
  //
  // Note that this doesn't expose any write errors here, those will surface
  // through the `raw_fd_ostream` API. The stream will also not close the file
  // which remains owned by the owning `File` object.
  auto WriteStream() -> llvm::raw_fd_ostream;

  // Reads the file from its start until EOF into the returned string.
  //
  // This method will retry any recoverable errors and work to completely read
  // the file contents from its beginning up to first encountering EOF. This
  // will seek the file to ensure it starts at the beginning regardless of
  // previous read or write operations.
  //
  // Any non-recoverable errors are returned to the caller.
  auto ReadFileToString() -> ErrorOr<std::string, FdError>;

  // Writes the provided string to the file from the start and truncating to the
  // written size.
  //
  // This method will retry any recoverable errors and work to completely write
  // the provided content into the file from its beginning, and truncate the
  // file's size to the provided string size. Essentially, this function
  // replaces the file contents with the string's contents.
  //
  // Any non-recoverable errors are returned to the caller.
  auto WriteFileFromString(llvm::StringRef str) -> ErrorOr<Success, FdError>;

  // Attempt to acquire an advisory shared lock on this directory.
  //
  // This is always done as a non-blocking operation, as blocking on advisory
  // locks without a deadline can easily result in build systems essentially
  // "fork-bombing" a machine. However, a `deadline` duration can be provided
  // and this routine will block and attempt to acquire the requested lock for a
  // bounded amount of time approximately based on that duration. Further, a
  // `poll_interval` can be provided to control how quickly the lock will be
  // polled during that duration. There is no scaling of the poll intervals at
  // this layer, if a back-off heuristic is desired the caller should manage the
  // polling themselves. The goal is to allow simple cases of spurious failures
  // to be easily handled without unbounded blocking calls. Typically, callers
  // should use a duration that is a reasonable upper bound on the latency to
  // begin the locked operation and a poll interval that is a reasonably low
  // median latency to begin the operation as 1-2 polls is expected to be
  // common. These should not be set anywhere near the cost of acquiring a file
  // lock, and in general file locks should only be used for expensive
  // operations where it is worth significant delays to avoid duplicate work.
  //
  // If the lock cannot be acquired, the most recent lock-attempt error is
  // returned.
  auto TryLock(FileLock::Kind kind, Duration deadline = {},
               Duration poll_interval = {}) -> ErrorOr<FileLock, FdError>;

 protected:
  explicit FileRefBase(int fd) : fd_(fd) {}

  // Note: this should only be used or made part of the public API by subclasses
  // that provide *ownership* of the open file. It is implemented here to
  // provide a single, non-templated implementation.
  auto Close() && -> ErrorOr<Success, FdError>;

  // Factored out code to destroy an open read-only file. This calls `Close`
  // above but ignores any errors as there is no risk of data loss for a
  // read-only file.
  //
  // Note: this is a private API that should not be made public, and should only
  // be used by the implementation of subclass destructors. It should also only
  // be called for subclasses with *ownership* of the file reference, and is
  // provided here as a single non-template implementation.
  auto ReadOnlyDestroy() -> void;

  // Factored out code to destroy an open writable file. This _requires_ the
  // file to have already been closed with an explicit `Close` call, where it
  // can report any errors. Without that, destroying a writable file can easily
  // result in unnoticed data loss.
  //
  // Note: this is a private API that should not be made public, and should only
  // be used by the implementation of subclass destructors. It should also only
  // be called for subclasses with *ownership* of the file reference, and is
  // provided here as a single non-template implementation.
  auto WriteableDestroy() -> void;

  // State representing a potentially open file.
  //
  // On POSIX systems, this will be a file descriptor. For moved-from and
  // default-constructed file objects this may be an invalid negative value to
  // signal that state.
  //
  // TODO: This should be customized on non-POSIX systems.
  //
  // This member is made protected rather than private as the derived classes
  // need direct access to it in several contexts.
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  int fd_ = -1;
};

// A non-owning reference to an open file.
//
// Instances model a reference to an open file. Generally, rather than using a
// `WriteFile&`, code should use a `WriteFileRef`.
//
// A specific instance provides the subset of the file API suitable for its
// access based on its template parameter: read, write, or both.
//
// The API for file references is factored into a base class
// `Internal::FileRefBase` to avoid duplication for each access instantiation.
// Only the methods that are constrained by access are defined here, and they
// are defined as wrappers around methods in the base where the documentation
// and implementation live.
template <OpenAccess A>
class FileRef : public Internal::FileRefBase {
 public:
  static constexpr bool Readable =
      A == OpenAccess::ReadOnly || A == OpenAccess::ReadWrite;
  static constexpr bool Writeable =
      A == OpenAccess::WriteOnly || A == OpenAccess::ReadWrite;

  // This object can be default constructed, but will hold an invalid file
  // handle in that case. This is to support rebinding operations.
  FileRef() = default;

  // Read and Write methods that delegate to the `FileRefBase` implementations,
  // but require the relevant access. See the methods on `FileRefBase` for full
  // documentation.
  auto Truncate(int64_t new_size) -> ErrorOr<Success, FdError>
    requires Writeable;
  auto ReadToBuffer(llvm::MutableArrayRef<std::byte> buffer)
      -> ErrorOr<llvm::MutableArrayRef<std::byte>, FdError>
    requires Readable;
  auto WriteFromBuffer(llvm::ArrayRef<std::byte> buffer)
      -> ErrorOr<llvm::ArrayRef<std::byte>, FdError>
    requires Writeable;
  auto WriteStream() -> llvm::raw_fd_ostream
    requires Writeable;
  auto ReadFileToString() -> ErrorOr<std::string, FdError>
    requires Readable;
  auto WriteFileFromString(llvm::StringRef str) -> ErrorOr<Success, FdError>
    requires Writeable;

 protected:
  friend File<A>;
  friend DirRef;

  // Other constructors from the base are also available, but remain protected.
  using FileRefBase::FileRefBase;
};

// Convenience type defs for the three access combinations.
using ReadFileRef = FileRef<OpenAccess::ReadOnly>;
using WriteFileRef = FileRef<OpenAccess::WriteOnly>;
using ReadWriteFileRef = FileRef<OpenAccess::ReadWrite>;

// An owning handle to an open file.
//
// This extends the `FileRef` API to provide ownership of the file handle. Most
// of the API is defined by `FileRef`.
//
// The file will be closed when the object is destroyed, and must close without
// errors. If there is a chance of errors on close, and that is often where
// errors are reported, code must use the `Close` API to directly handle them or
// it must be correct to check-fail on them.
//
// This type allows intentional "slicing" to the `FileRef` base class as that is
// a correct and safe conversion to pass a non-owning reference to a file to
// another function, much like binding a reference to an owning type is
// implicit.
template <OpenAccess A>
class File : public FileRef<A> {
 public:
  static constexpr bool Readable =
      A == OpenAccess::ReadOnly || A == OpenAccess::ReadWrite;
  static constexpr bool Writeable =
      A == OpenAccess::WriteOnly || A == OpenAccess::ReadWrite;

  // Default constructs an invalid file.
  //
  // This can be destroyed or assigned safely, but no other operations are
  // correct.
  File() = default;

  // File objects are move-only as they model ownership.
  File(File&& arg) noexcept : FileRef<A>(std::exchange(arg.fd_, -1)) {}
  auto operator=(File&& arg) noexcept -> File& {
    Destroy();
    this->fd_ = std::exchange(arg.fd_, -1);
    return *this;
  }
  File(const File&) = delete;
  auto operator=(const File&) -> File& = delete;
  ~File() { Destroy(); }

  // Closes the open file and leaves the file in a moved-from state.
  //
  // The signature is `auto Close() && -> ErrorOr<Success, FdError>`.
  //
  // This type provides ownership of the file, so expose the `Close` method to
  // allow checked destruction and release of the file resources.
  //
  // If any errors are encountered during closing, returns them. Note that the
  // file should still be considered closed, and the object is moved-from even
  // if errors occur.
  using Internal::FileRefBase::Close;

 private:
  friend DirRef;

  // Destroy the file.
  //
  // This dispatches to non-template code in `FileRefBase` based on whether the
  // file is writable or readonly. The core logic is in the non-template
  // methods.
  auto Destroy() -> void;

  explicit File(int fd) : FileRef<A>(fd) {}
};

// Convenience type defs for the three access combinations.
using ReadFile = File<OpenAccess::ReadOnly>;
using WriteFile = File<OpenAccess::WriteOnly>;
using ReadWriteFile = File<OpenAccess::ReadWrite>;

// A non-owning reference to an open directory.
//
// This is the main API for accessing and opening files and other directories.
// Conceptually, every open file or directory is relative to some other
// directory. The symbolic current working directory object is available via the
// `Cwd()` function. When on a Unix-like platform, this is intended to provide
// the semantics of `openat` and related functions, including the ability to
// write secure filesystem operations in the face of adversarial parallel
// filesystem operations.
//
// Relative path parameters are always relative to this directory. Absolute path
// parameters are also allowed and are treated as absolute paths. This parallels
// the behavior of `/` for path concatenation where an absolute path ignores all
// preceding components.
//
// Errors for directory operations retain the path parameter used in order to
// print helpful detail when unhandled, but otherwise work to be lazy and
// lightweight to support low-overhead expected error patterns.
//
// The names are designed to mirror the underlying Unix-like APIs that implement
// them, with extensions to add clarity. However, the set of operations is
// expected to be reasonable to implement on Windows with reasonable fidelity.
class DirRef {
 public:
  class Entry;
  class Iterator;
  class Reader;

  constexpr DirRef() = default;

  // Returns true if this refers to a valid open directory, and false otherwise.
  auto is_valid() const -> bool { return dfd_ != -1; }

  // Begin reading the entries in a directory.
  //
  // This returns a `Reader` object that can be iterated to walk over all the
  // entries in this directory. Note that the returned `Reader` owns a newly
  // allocated handle to this directory, and provides the full `DirRef` API. If
  // it isn't necessary to keep both open, the `Dir` class offers a
  // move-qualified method `TakeAndRead` that optimizes this case.
  //
  // Note that it is unspecified whether added and removed files during the
  // lifetime of the reader will be included when iterating, but otherwise
  // concurrent mutations are well defined.
  auto Read() -> ErrorOr<Reader, FdError>;

  // Reads all of the entries in this directory into a vector.
  //
  // A helper function wrapping `Read` and walking the resulting reader's
  // entries to produce a container.
  //
  // For details on errors, see the documentation of `Read` and `Reader`.
  auto ReadEntries()
      -> ErrorOr<llvm::SmallVector<std::filesystem::path>, FdError>;

  // Reads the directory and appends entries to a container if they pass a
  // predicate. The predicate can be null, which is treated as if it always
  // returns true.
  //
  // For details on errors, see the documentation of `Read` and `Reader`.
  auto AppendEntriesIf(
      llvm::SmallVectorImpl<std::filesystem::path>& entries,
      llvm::function_ref<auto(llvm::StringRef name)->bool> predicate = {})
      -> ErrorOr<Success, FdError>;

  // Reads the directory and appends entries to one of two containers if they
  // pass a predicate.  The predicate can be null, which is treated as if it
  // always returns true.
  //
  // Which container an entry is appended to depends on its kind -- directories
  // go to the first and non-directories go to the second. This turns out to be
  // a very common split with subdirectories often needing separate handling
  // from other entries.
  //
  // For details on errors, see the documentation of `Read` and `Reader`. This
  // may also `Stat` entries if necessary to determine whether they are
  // directories.
  auto AppendEntriesIf(
      llvm::SmallVectorImpl<std::filesystem::path>& dir_entries,
      llvm::SmallVectorImpl<std::filesystem::path>& non_dir_entries,
      llvm::function_ref<auto(llvm::StringRef name)->bool> predicate = {})
      -> ErrorOr<Success, FdError>;

  // Checks that the provided path can be accessed.
  auto Access(const std::filesystem::path& path,
              AccessCheckFlags check = AccessCheckFlags::Exists)
      -> ErrorOr<bool, PathError>;

  // Reads the `FileStatus` for the open directory.
  auto Stat() -> ErrorOr<FileStatus, FdError>;

  // Reads the `FileStatus` for the provided path (without opening it).
  //
  // Like the `stat` system call on Unix-like platforms, this will follow any
  // symlinks and provide the status of the underlying file or directory.
  auto Stat(const std::filesystem::path& path)
      -> ErrorOr<FileStatus, PathError>;

  // Reads the `FileStatus` for the provided path (without opening it).
  //
  // Like the `lstat` system call on Unix-like platforms, this will *not* follow
  // symlinks, and instead will return the status of the symlink itself.
  auto Lstat(const std::filesystem::path& path)
      -> ErrorOr<FileStatus, PathError>;

  // Updates the access and modification times for the provided path.
  //
  // If no explicit `time_point` is provided, sets both times to the current
  // time. If an explicit `time_point` is provided, the times are updated to
  // that time.
  auto UpdateTimes(const std::filesystem::path& path,
                   std::optional<TimePoint> time_point = std::nullopt)
      -> ErrorOr<Success, PathError>;

  // Reads the target string of the symlink at the provided path.
  //
  // This does not follow the symlink, and does not require the symlink target
  // to be valid or exist. It merely reads the textual string.
  //
  // Returns an error if called with a path that is not a symlink.
  auto Readlink(const std::filesystem::path& path)
      -> ErrorOr<std::string, PathError>;

  // Opens the provided path as a read-only file.
  //
  // The interaction with an existing file is governed by `creation_options` and
  // defaults to error unless opening an existing file. When creating a file,
  // only the leaf component in the provided path can be created with this call.
  //
  // If creating a file, the file is created with `creation_mode` which defaults
  // to a restrictive `0600`. The creation permission bits are also completely
  // independent of the access provided via the opened file. For example,
  // creating with write permissions doesn't impact whether write access is
  // available via the returned file. And creating _without_ write permission
  // bits is compatible with opening the file for writing.
  //
  // Additional flags can be provided to `flags` to control other aspects of
  // behavior on open.
  //
  // This is an error if the path exists and is a directory. If the path is a
  // symlink, it will follow the symlink.
  auto OpenReadOnly(const std::filesystem::path& path,
                    CreationOptions creation_options = OpenExisting,
                    ModeType creation_mode = 0600,
                    OpenFlags flags = OpenFlags::None)
      -> ErrorOr<ReadFile, PathError>;

  // Opens the provided path as a write-only file. Otherwise, behaves as
  // `OpenReadOnly`.
  auto OpenWriteOnly(const std::filesystem::path& path,
                     CreationOptions creation_options = OpenExisting,
                     ModeType creation_mode = 0600,
                     OpenFlags flags = OpenFlags::None)
      -> ErrorOr<WriteFile, PathError>;

  // Opens the provided path as a read-and-write file. Otherwise, behaves as
  // `OpenReadOnly`.
  auto OpenReadWrite(const std::filesystem::path& path,
                     CreationOptions creation_options = OpenExisting,
                     ModeType creation_mode = 0600,
                     OpenFlags flags = OpenFlags::None)
      -> ErrorOr<ReadWriteFile, PathError>;

  // Opens the provided path as a directory.
  //
  // Similar to `OpenReadOnly` and other file opening APIs, accepts
  // `creation_options` to control the interaction with any existing directory.
  // However, `CreateAlways` is not implementable for directories and an error
  // if passed. The default permissions in the `creation_mode` are `0700` which
  // is more suitable for directories. There are no extra flags that can be
  // passed.
  //
  // As with other open routines, when creating a directory, only the leaf
  // component can be created by the call to this routine.
  //
  // When creating a directory with `CreateNew`, this routine works to be safe
  // even in the presence of adversarial, concurrent operations that attempt to
  // replace the created directory with one that is controlled by the adversary.
  //
  // Specifically, for `CreateNew` we ensure that the last component is a
  // created directory in its parent, and cannot be replaced by a symlink into
  // an attacker-controlled directory. We further ensure it cannot have been
  // replaced by a directory with a different owner or with wider permissions
  // than the created directory.
  //
  // However, no validation is done on any prefix path components leading to the
  // leaf component created. When securely creating directories, the initial
  // creation should have a single component from an opened existing parent
  // directory. Also, no validation of the owning _group_ is performed. When
  // securely creating a directory, the caller should either ensure the parent
  // directory does not have a malicious setgid bit set, or restrict the
  // created mode to not give group access, or both. In general, the lack of
  // control over the owning group motivates our choice to make the default mode
  // permissions restrictive and not include any group access.
  //
  // To securely achieve a result similar to `OpenAlways` instead of
  // `CreateNew`, callers can directly `CreateNew` and handle failures with an
  // explicit `OpenExisting` that also blocks following symlinks with
  // `OpenFlags::NoFollow` and performs any needed validation.
  auto OpenDir(const std::filesystem::path& path,
               CreationOptions creation_options = OpenExisting,
               ModeType creation_mode = 0700, OpenFlags flags = OpenFlags::None)
      -> ErrorOr<Dir, PathError>;

  // Reads the file at the provided path to a string.
  //
  // This is a convenience wrapper for opening the path, reading the returned
  // file to a string, and closing it. Errors from any step are returned.
  auto ReadFileToString(const std::filesystem::path& path)
      -> ErrorOr<std::string, PathError>;

  // Writes the provided `content` to the provided path.
  //
  // This is a convenience wrapper for opening the path, creating it according
  // to `creation_options` as necessary, writing `content` to it, and closing
  // it. Errors from any step are returned.
  auto WriteFileFromString(const std::filesystem::path& path,
                           llvm::StringRef content,
                           CreationOptions creation_options = CreateAlways)
      -> ErrorOr<Success, PathError>;

  // Renames an entry from one directory to another directory, replacing any
  // existing entry with the target path.
  //
  // Note that this is *not* a general purpose move! It must be possible for
  // this operation to be performed as a metadata-only change, and so without
  // moving any actual data. This means it will not work across devices, mounts,
  // or filesystems. However, these restrictions make this an *atomic* rename.
  //
  // The most common usage is to rename an entry within a single directory, by
  // passing `*this` as `target_dir`.
  auto Rename(const std::filesystem::path& path, DirRef target_dir,
              const std::filesystem::path& target_path)
      -> ErrorOr<Success, PathError>;

  // Changes the current working directory to this directory.
  auto Chdir() -> ErrorOr<Success, FdError>;

  // Changes the current working directory to the provided path.
  //
  // An error if the provided path is not a directory. Does not open the
  // provided path as a directory, but it will be available as the current
  // working directory via `Cwd()`.
  auto Chdir(const std::filesystem::path& path) -> ErrorOr<Success, PathError>;

  // Creates a symlink at the provided path with the contents of `target`.
  //
  // Note that the target of a symlink is an arbitrary string and there is no
  // error checking on whether it exists or is sensible. Also, the target string
  // set will be up to the first null byte in `target`, regardless of its
  // `size`. This will not overwrite an existing symlink at the provided path.
  //
  // Also note that the written symlink will be the null-terminated string
  // `target.c_str()`, ignoring everything past any embedded null bytes.
  auto Symlink(const std::filesystem::path& path, const std::string& target)
      -> ErrorOr<Success, PathError>;

  // Creates the directories in the provided path, using the permissions in
  // `creation_mode`.
  //
  // This will create any missing directory components in `path`. Relative paths
  // will be created relative to this directory, and without re-resolving its
  // path. The leaf created directory is opened and returned.
  //
  // The implementation allows for concurrent creation of the same directory (or
  // a prefix) without error or corruption and optimizes for performance of
  // creating the requested path. As a consequence, this creation is _unsafe_ in
  // the face of adversarial concurrent manipulation of components of the path.
  // If you need to create directories securely, first create an initial
  // directory securely using `OpenDir` and `CreateNew` with restricted
  // permissions that preclude any adversarial behavior, then use this API to
  // create tree components within that root.
  auto CreateDirectories(const std::filesystem::path& path,
                         ModeType creation_mode = 0700)
      -> ErrorOr<Dir, PathError>;

  // Unlink the last component of the path, removing that name from its parent
  // directory.
  //
  // If this was the last link to the underlying file its contents will be
  // removed when the last open file handle to it is closed.
  //
  // The path must not be a directory. If the path is a symbolic link, the link
  // will be removed, not the target. Models the behavior of `unlinkat(2)` on
  // Unix-like platforms.
  auto Unlink(const std::filesystem::path& path) -> ErrorOr<Success, PathError>;

  // Remove the directory entry of the last component of the path.
  //
  // The path must be a directory, and that directory must be empty. Models
  // `rmdirat(2)` on Unix-like platforms.
  auto Rmdir(const std::filesystem::path& path) -> ErrorOr<Success, PathError>;

  // Remove the directory tree identified by the last component of the path.
  //
  // The provided path must name a directory. This removes all files and
  // subdirectories contained within that named directory and then removes the
  // directory itself once empty.
  auto Rmtree(const std::filesystem::path& path) -> ErrorOr<Success, PathError>;

 protected:
  constexpr explicit DirRef(int dfd) : dfd_(dfd) {}

  // Slow-path fallback when unable to read the symlink target into a small
  // stack buffer.
  auto ReadlinkSlow(const std::filesystem::path& path)
      -> ErrorOr<std::string, PathError>;

  // Generic implementation of the various `Open*` variants using the
  // `OpenAccess` enumerator.
  template <OpenAccess A>
  auto OpenImpl(const std::filesystem::path& path,
                CreationOptions creation_options, ModeType creation_mode,
                OpenFlags flags) -> ErrorOr<File<A>, PathError>;

  // State representing an open directory.
  //
  // On POSIX systems, this will be a file descriptor. For moved-from and
  // default-constructed file objects this may be an invalid negative value to
  // signal that state.
  //
  // TODO: This should be customized on non-POSIX systems.
  //
  // The directory's file descriptor is part of the protected API.
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes):
  int dfd_ = -1;
};

// An owning handle to an open directory.
//
// This extends the `DirRef` API to provide ownership of the directory. Most of
// the API is defined by `DirRef`. It additionally provides optimized move-based
// variations on those APIs where relevant.
//
// The directory will be closed when the object is destroyed. Closing an open
// directory isn't an interesting error reporting path and so no direct close
// API is provided.
//
// This type allows intentional "slicing" to the `DirRef` base class as that is
// a correct and safe conversion to pass a non-owning reference to a directory
// to another function, much like binding a reference to an owning type is
// implicit.
class Dir : public DirRef {
 public:
  Dir() = default;

  // Dir objects are move-only as they model ownership.
  Dir(Dir&& arg) noexcept : DirRef(std::exchange(arg.dfd_, -1)) {}
  auto operator=(Dir&& arg) noexcept -> Dir& {
    Destroy();
    dfd_ = std::exchange(arg.dfd_, -1);
    return *this;
  }
  Dir(const Dir&) = delete;
  auto operator=(const Dir&) -> Dir& = delete;
  constexpr ~Dir();

  // An optimized way to read the entries in a directory when moving from an
  // owning `Dir` object.
  //
  // This avoids creating a duplicate file handle for the returned `Reader`.
  // That `Reader` also supports the full `DirRef` API and so can often be used
  // without retaining the original `Dir`.
  //
  // For more details about reading, see the documentation on `DirRef::Read`.
  auto TakeAndRead() && -> ErrorOr<Reader, FdError>;

  // Also include `DirRef`'s read API.
  using DirRef::Read;

 private:
  friend consteval auto Cwd() -> Dir;
  friend DirRef;
  friend RemovingDir;

  explicit constexpr Dir(int dfd) : DirRef(dfd) {}

  // Prevent implicit creation of a `Dir` object from a `RemovingDir` which will
  // end up as a subclass below and represent harmful implicit slicing. Instead,
  // require friendship and an explicit construction on an _intended_ release of
  // the removing semantics.
  explicit Dir(RemovingDir&& arg) noexcept;

  constexpr auto Destroy() -> void;
};

// An owning handle to an open directory and its absolute path that will be
// removed recursively when destroyed.
//
// This can be used to ensure removal of a directory, and also exposes the
// absolute path of the directory.
//
// As removal may encounter errors, unless the desired behavior is a
// check-failure, users should explicitly move and call `Remove` at the end of
// lifetime and handle any resultant errors.
class RemovingDir : public Dir {
 public:
  // Takes ownership of the open directory `d` and wraps it in a `RemovingDir`
  // that will remove it on destruction using `abs_path`. Requires `abs_path` to
  // be an absolute path and the desired path to remove on destruction.
  //
  // Note that there is no way for the implementation to validate what directory
  // `abs_path` refers to, that is the responsibility of the caller.
  explicit RemovingDir(Dir d, std::filesystem::path abs_path)
      : Dir(std::move(d)), abs_path_(std::move(abs_path)) {
    CARBON_CHECK(abs_path_.is_absolute(), "Relative path used for removal: {0}",
                 abs_path_);
  }

  RemovingDir() = default;
  RemovingDir(RemovingDir&& arg) = default;
  auto operator=(RemovingDir&& rhs) -> RemovingDir& = default;
  ~RemovingDir();

  auto abs_path() const [[clang::lifetimebound]]
  -> const std::filesystem::path& {
    return abs_path_;
  }

  // Releases the directory from being removed and returns just the underlying
  // owning handle.
  auto Release() && -> Dir { return std::move(*this); }

  // Removes the directory immediately and surfaces any errors encountered.
  auto Remove() && -> ErrorOr<Success, PathError>;

 private:
  friend Dir;

  std::filesystem::path abs_path_;
};

// A named entry in a directory.
//
// This provides access to the scanned data when reading the entries of the
// directory. It can only be produced by iterating over a `DirRef::Reader`.
class DirRef::Entry {
 public:
  // The name of the entry.
  //
  // This is exposed as a null-terminated C-string as that is the most common
  // representation.
  auto name() const -> const char* { return dent_->d_name; }

  // Test if the entry has an unknown type. In this case, all other type
  // predicates will return false and the caller will have to directly `Lstat()`
  // the entry to determine its type.
  auto is_unknown_type() const -> bool { return dent_->d_type == DT_UNKNOWN; }

  // Predicates to test for known entry types.
  //
  // Note that we don't provide an enumerator here as we don't have any reliable
  // way to predict the set of possible values or narrow to that set. Different
  // platforms and even different versions of the same header may change the set
  // of types surfaced here.
  auto is_known_dir() const -> bool { return dent_->d_type == DT_DIR; }
  auto is_known_regular_file() const -> bool { return dent_->d_type == DT_REG; }
  auto is_known_symlink() const -> bool { return dent_->d_type == DT_LNK; }

 private:
  friend Dir::Reader;
  friend Dir::Iterator;

  Entry() = default;
  explicit Entry(dirent* dent) : dent_(dent) {}

  dirent* dent_ = nullptr;
};

// An iterator into a `DirRef::Reader`, used for walking the entries in a
// directory.
//
// Most of the work of iterating a directory is done when constructing the
// `Reader`, when constructing the beginning iterator, or when incrementing the
// iterator.
class DirRef::Iterator
    : public llvm::iterator_facade_base<Iterator, std::input_iterator_tag,
                                        const Entry> {
 public:
  // Default construct a general end iterator.
  Iterator() = default;

  auto operator==(const Iterator& rhs) const -> bool {
    CARBON_DCHECK(dirp_ == nullptr || rhs.dirp_ == nullptr ||
                  dirp_ == rhs.dirp_);
    return entry_.dent_ == rhs.entry_.dent_;
  }

  auto operator*() const [[clang::lifetimebound]] -> const Entry& {
    return entry_;
  }
  auto operator++() -> Iterator&;

 private:
  friend Dir::Reader;

  // Construct a begin iterator for a specific directory stream.
  explicit Iterator(DIR* dirp) : dirp_(dirp) {
    // Increment immediately to populate the initial entry.
    ++*this;
  }

  DIR* dirp_ = nullptr;
  Entry entry_;
};

// A reader for a directory.
//
// This class owns a handle to a directory that is set up for reading the
// entries within the directory. Because it owns a handle to the directory, it
// also implements the full `DirRef` API for convenience.
//
// Beyond the `DirRef` API, this object can be iterated as a range to visit all
// the entries in the directory.
//
// Note that it is unspecified whether entries added or removed prior to being
// visited while iterating. Iterating also cannot be re-started once begun --
// this models an input iterable range, not even a forward iterable range.
//
// This type allows intentional "slicing" to the `DirRef` base class as that is
// a correct and safe conversion to pass a non-owning reference to a directory
// to another function, much like binding a reference to an owning type is
// implicit.
class DirRef::Reader : public DirRef {
 public:
  Reader() = default;
  Reader(Reader&& arg) noexcept
      // The directory file descriptor isn't owning, but clear it for clarity.
      : DirRef(std::exchange(arg.dfd_, -1)),
        dirp_(std::exchange(arg.dirp_, nullptr)) {}
  Reader(const Reader&) = delete;
  auto operator=(Reader&& arg) noexcept -> Reader& {
    Destroy();
    // The directory file descriptor isn't owning, but clear it for clarity.
    dfd_ = std::exchange(arg.dfd_, -1);
    dirp_ = std::exchange(arg.dirp_, nullptr);
    return *this;
  }
  ~Reader() { Destroy(); }

  // Compute the begin and end iterators for reading the entries of the
  // directory.
  auto begin() -> Iterator;
  auto end() -> Iterator;

 private:
  friend DirRef;
  friend Dir;

  explicit Reader(DIR* dirp) : DirRef(dirfd(dirp)), dirp_(dirp) {}
  auto Destroy() -> void;

  DIR* dirp_ = nullptr;
};

namespace Internal {
// Base class for `errno` errors.
//
// This is where we extract common APIs and logic for querying the specific
// `errno`-based error.
template <typename ErrorT>
class ErrnoErrorBase : public ErrorBase<ErrorT> {
 public:
  // Accessors to test for specific kinds of errors that are portably available.
  auto already_exists() const -> bool { return errnum_ == EEXIST; }
  auto is_dir() const -> bool { return errnum_ == EISDIR; }
  auto no_entity() const -> bool { return errnum_ == ENOENT; }
  auto not_dir() const -> bool { return errnum_ == ENOTDIR; }
  auto access_denied() const -> bool { return errnum_ == EACCES; }
  auto would_block() const -> bool { return errnum_ == EWOULDBLOCK; }

  // Specific to `Rmdir` and `Rename` operations that remove a directory name,
  // two different error values can be used.
  auto not_empty() const -> bool {
    return errnum_ == ENOTEMPTY || errnum_ == EEXIST;
  }

  // Accessor for the `errno` based error number. This is not a portable API,
  // code using it will need to be ported to use a different API on Windows.
  // TODO: Add a Windows-specific API for its low-level error information.
  auto unix_errnum() const -> int { return errnum_; }

 protected:
  // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility):
  explicit ErrnoErrorBase(int errnum) : errnum_(errnum) {}

 private:
  int errnum_;
};
}  // namespace Internal

// Error from a file-descriptor operation.
//
// This is the implementation of the file-descriptor-based error type. When
// operations on a file descriptor fail, they use this object to convey the
// error plus the descriptor in question.
//
// Specific context on the exact point or nature of the operation that failed
// can be included in the custom format string. The format string should include
// a placeholder for the file descriptor to be substituted into. The format
// string should describe the _operation_ that failed, once rendered it will
// have `failed: ` and a description of the `errno`-indicated failure appended.
//
// For example:
//
//   `FdError(EPERM, "Read of file '{0}'", 42)`
//
// Will be rendered similarly to:
//
//   "Read of file '42' failed: EPERM: ..."
class FdError : public Internal::ErrnoErrorBase<FdError> {
 public:
  FdError(FdError&&) noexcept = default;
  auto operator=(FdError&&) noexcept -> FdError& = default;

  // Prints this error to the provided string.
  //
  // Works to render the `errno` in a friendly way and includes the file
  // descriptor for context.
  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  friend FileLock;
  friend Internal::FileRefBase;
  friend ReadFile;
  friend WriteFile;
  friend ReadWriteFile;
  friend DirRef;
  friend Dir;

  explicit FdError(int errnum, llvm::StringLiteral format, int fd)
      : ErrnoErrorBase(errnum), fd_(fd), format_(format) {}

  int fd_;
  llvm::StringLiteral format_;
};

// Error from a path-based operation.
//
// This is the implementation of the path-based error type. When operations on a
// path fail, they use this object to convey the error plus both the path and
// relevant directory FD leading to the failure.
//
// Specific context on the exact point or nature of the operation that failed
// can be included in the custom format string. The format string should include
// placeholders for the path and the directory file descriptor to be substituted
// into. The format string should describe the _operation_ that failed, once
// rendered it will have `failed: ` and a description of the `errno`-indicated
// failure appended.
//
// For example:
//
//   `PathError(EPERM, "Open of '{0}' relative to '{1}'", "filename", 42)`
//
// Will be rendered similarly to:
//
//   "Open of 'filename' relative to '42' failed: EPERM: ..."
class PathError : public Internal::ErrnoErrorBase<PathError> {
 public:
  PathError(PathError&&) noexcept = default;
  auto operator=(PathError&&) noexcept -> PathError& = default;

  // Prints this error to the provided string.
  //
  // Works to render the `errno` in a friendly way and includes the path and
  // directory file descriptor for context.
  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  friend DirRef;
  friend Dir;

  explicit PathError(int errnum, llvm::StringLiteral format,
                     std::filesystem::path path, int dir_fd)
      : ErrnoErrorBase(errnum),
        dir_fd_(dir_fd),
        path_(std::move(path)),
        format_(format) {}

  int dir_fd_;
  std::filesystem::path path_;
  llvm::StringLiteral format_;
};

// Implementation details only below.

namespace Internal {

inline auto DurationToTimespec(Duration d) -> timespec {
  timespec ts = {};
  ts.tv_sec = std::chrono::duration_cast<std::chrono::seconds>(d).count();
  d -= std::chrono::seconds(ts.tv_sec);
  ts.tv_nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
  return ts;
}

}  // namespace Internal

consteval auto Cwd() -> Dir { return Dir(AT_FDCWD); }

inline auto FileLock::Destroy() -> void {
  if (fd_ == -1) {
    // Nothing to unlock.
    return;
  }

  // We always try to unlock in a non-blocking way as there should never be a
  // reason to block here.
  int result = flock(fd_, LOCK_UN | LOCK_NB);

  // The only realistic error is `EBADF` that would represent a programming
  // error the type system should prevent. We conservatively check-fail if an
  // error occurs here.
  CARBON_CHECK(result == 0, "{0}",
               FdError(errno, "Unexpected error while _unlocking_ '{0}'", fd_));
}

inline auto Internal::FileRefBase::Stat() -> ErrorOr<FileStatus, FdError> {
  FileStatus status;
  if (fstat(fd_, &status.stat_buf_) == 0) {
    return status;
  }
  return FdError(errno, "File::Stat on '{0}'", fd_);
}

inline auto Internal::FileRefBase::UpdateTimes(
    std::optional<TimePoint> time_point) -> ErrorOr<Success, FdError> {
  if (!time_point) {
    if (futimens(fd_, nullptr) == -1) {
      return FdError(errno, "File::UpdateTimes to now on '{0}'", fd_);
    }
    return Success();
  }

  timespec times[2];
  times[0] = Internal::DurationToTimespec(time_point->time_since_epoch());
  times[1] = times[0];
  if (futimens(fd_, times) == -1) {
    return FdError(errno, "File::UpdateTimes to a specific time on '{0}'", fd_);
  }
  return Success();
}

inline auto Internal::FileRefBase::Seek(int64_t delta)
    -> ErrorOr<int64_t, FdError> {
  int64_t byte_offset = lseek(fd_, delta, SEEK_CUR);
  if (byte_offset == -1) {
    return FdError(errno, "File::Seek on '{0}'", fd_);
  }
  return byte_offset;
}

inline auto Internal::FileRefBase::SeekFromBeginning(
    int64_t delta_from_beginning) -> ErrorOr<int64_t, FdError> {
  int64_t byte_offset = lseek(fd_, delta_from_beginning, SEEK_SET);
  if (byte_offset == -1) {
    return FdError(errno, "File::SeekTo on '{0}'", fd_);
  }
  return byte_offset;
}

inline auto Internal::FileRefBase::SeekFromEnd(int64_t delta_from_end)
    -> ErrorOr<int64_t, FdError> {
  int64_t byte_offset = lseek(fd_, delta_from_end, SEEK_END);
  if (byte_offset == -1) {
    return FdError(errno, "File::SeekFromEnd on '{0}'", fd_);
  }
  return byte_offset;
}

inline auto Internal::FileRefBase::Truncate(int64_t new_size)
    -> ErrorOr<Success, FdError> {
  int64_t result = ftruncate(fd_, new_size);
  if (result == -1) {
    return FdError(errno, "File::Truncate on '{0}'", fd_);
  }
  return Success();
}

inline auto Internal::FileRefBase::ReadToBuffer(
    llvm::MutableArrayRef<std::byte> buffer)
    -> ErrorOr<llvm::MutableArrayRef<std::byte>, FdError> {
  for (;;) {
    ssize_t read_bytes = read(fd_, buffer.data(), buffer.size());
    if (read_bytes == -1) {
      if (errno == EINTR) {
        continue;
      }
      return FdError(errno, "File::Read on '{0}'", fd_);
    }
    return buffer.slice(0, read_bytes);
  }
}

inline auto Internal::FileRefBase::WriteFromBuffer(
    llvm::ArrayRef<std::byte> buffer)
    -> ErrorOr<llvm::ArrayRef<std::byte>, FdError> {
  for (;;) {
    ssize_t written_bytes = write(fd_, buffer.data(), buffer.size());
    if (written_bytes == -1) {
      if (errno == EINTR) {
        continue;
      }
      return FdError(errno, "File::Write on '{0}'", fd_);
    }
    return buffer.drop_front(written_bytes);
  }
}

inline auto Internal::FileRefBase::WriteStream() -> llvm::raw_fd_ostream {
  return llvm::raw_fd_ostream(fd_, /*shouldClose=*/false);
}

inline auto Internal::FileRefBase::Close() && -> ErrorOr<Success, FdError> {
  // Put the file in a moved-from state immediately as it is invalid to
  // retry closing or use the file in any way even if the close fails.
  int fd = std::exchange(fd_, -1);

  int result = close(fd);
  if (result == 0) {
    return Success();
  }

  return FdError(errno, "File::Close on '{0}'", fd);
}

inline auto Internal::FileRefBase::ReadOnlyDestroy() -> void {
  if (fd_ >= 0) {
    auto result = std::move(*this).Close();
    // Intentionally drop errors, as there is no interesting error here. There
    // is no risk of data loss, and the least bad thing we can do is to just
    // leak the file descriptor.
    static_cast<void>(result);
  }
}

inline auto Internal::FileRefBase::WriteableDestroy() -> void {
  CARBON_CHECK(
      fd_ == -1,
      "Cannot destroy an open writable file, they _must_ be destroyed by "
      "calling `Close` and handling any errors to avoid data loss.");
}

template <OpenAccess A>
auto FileRef<A>::Truncate(int64_t new_size) -> ErrorOr<Success, FdError>
  requires Writeable
{
  return FileRefBase::Truncate(new_size);
}

template <OpenAccess A>
auto FileRef<A>::ReadToBuffer(llvm::MutableArrayRef<std::byte> buffer)
    -> ErrorOr<llvm::MutableArrayRef<std::byte>, FdError>
  requires Readable
{
  return FileRefBase::ReadToBuffer(buffer);
}

template <OpenAccess A>
auto FileRef<A>::ReadFileToString() -> ErrorOr<std::string, FdError>
  requires Readable
{
  return FileRefBase::ReadFileToString();
}

template <OpenAccess A>
auto FileRef<A>::WriteFromBuffer(llvm::ArrayRef<std::byte> buffer)
    -> ErrorOr<llvm::ArrayRef<std::byte>, FdError>
  requires Writeable
{
  return FileRefBase::WriteFromBuffer(buffer);
}

template <OpenAccess A>
auto FileRef<A>::WriteStream() -> llvm::raw_fd_ostream
  requires Writeable
{
  return FileRefBase::WriteStream();
}

template <OpenAccess A>
auto FileRef<A>::WriteFileFromString(llvm::StringRef str)
    -> ErrorOr<Success, FdError>
  requires Writeable
{
  return FileRefBase::WriteFileFromString(str);
}

template <OpenAccess A>
auto File<A>::Destroy() -> void {
  if constexpr (Writeable) {
    this->WriteableDestroy();
  } else {
    this->ReadOnlyDestroy();
  }
}

inline auto DirRef::Read() -> ErrorOr<Reader, FdError> {
  int dup_dfd = dup(dfd_);
  if (dup_dfd == -1) {
    // There are very few plausible errors here, but we can return one so it
    // doesn't hurt to do so. While `EINTR` and `EBUSY` are mentioned in some
    // documentation, there is no indication that for just `dup` it is useful to
    // loop and retry.
    return FdError(errno, "Dir::Read on '{0}'", dfd_);
  }
  return Dir(dup_dfd).TakeAndRead();
}

inline auto DirRef::ReadEntries()
    -> ErrorOr<llvm::SmallVector<std::filesystem::path>, FdError> {
  llvm::SmallVector<std::filesystem::path> entries;
  CARBON_RETURN_IF_ERROR(AppendEntriesIf(entries));
  return entries;
}

inline auto DirRef::Access(const std::filesystem::path& path,
                           AccessCheckFlags check) -> ErrorOr<bool, PathError> {
  if (faccessat(dfd_, path.c_str(), static_cast<int>(check), /*flags=*/0) ==
      0) {
    return true;
  }
  return PathError(errno, "Dir::Access on '{0}' relative to '{1}'", path, dfd_);
}

inline auto DirRef::Stat() -> ErrorOr<FileStatus, FdError> {
  FileStatus status;
  if (fstat(dfd_, &status.stat_buf_) == 0) {
    return status;
  }
  return FdError(errno, "Dir::Stat on '{0}': ", dfd_);
}

inline auto DirRef::Stat(const std::filesystem::path& path)
    -> ErrorOr<FileStatus, PathError> {
  FileStatus status;
  if (fstatat(dfd_, path.c_str(), &status.stat_buf_, /*flags=*/0) == 0) {
    return status;
  }
  return PathError(errno, "Dir::Stat on '{0}' relative to '{1}'", path, dfd_);
}

inline auto DirRef::Lstat(const std::filesystem::path& path)
    -> ErrorOr<FileStatus, PathError> {
  FileStatus status;
  if (fstatat(dfd_, path.c_str(), &status.stat_buf_,
              /*flags=*/AT_SYMLINK_NOFOLLOW) == 0) {
    return status;
  }
  return PathError(errno, "Dir::Lstat on '{0}' relative to '{1}'", path, dfd_);
}

inline auto DirRef::UpdateTimes(const std::filesystem::path& path,
                                std::optional<TimePoint> time_point)
    -> ErrorOr<Success, PathError> {
  if (!time_point) {
    if (utimensat(dfd_, path.c_str(), nullptr, /*flags*/ 0) == -1) {
      return PathError(errno,
                       "Dir::UpdateTimes to now on '{0}' relative to '{1}'",
                       path, dfd_);
    }
    return Success();
  }

  timespec times[2];
  times[0] = Internal::DurationToTimespec(time_point->time_since_epoch());
  times[1] = times[0];
  if (utimensat(dfd_, path.c_str(), times, /*flags*/ 0) == -1) {
    return PathError(
        errno, "Dir::UpdateTimes to a specific time on '{0}' relative to '{1}'",
        path, dfd_);
  }
  return Success();
}

inline auto DirRef::Readlink(const std::filesystem::path& path)
    -> ErrorOr<std::string, PathError> {
  // On the fast path, we read into a small stack buffer and get the whole
  // contents.
  constexpr ssize_t BufferSize = 256;
  char buffer[BufferSize];
  ssize_t read_bytes = readlinkat(dfd_, path.c_str(), buffer, BufferSize);
  if (read_bytes == -1) {
    return PathError(errno, "Dir::Readlink on '{0}' relative to '{1}'", path,
                     dfd_);
  }
  if (read_bytes < BufferSize) {
    // We got the whole contents in one shot, return it.
    return std::string(buffer, read_bytes);
  }

  // Otherwise, fallback to an out-of-line function to handle the slow path.
  return ReadlinkSlow(path);
}

inline auto DirRef::OpenReadOnly(const std::filesystem::path& path,
                                 CreationOptions creation_options,
                                 ModeType creation_mode, OpenFlags flags)
    -> ErrorOr<ReadFile, PathError> {
  return OpenImpl<OpenAccess::ReadOnly>(path, creation_options, creation_mode,
                                        flags);
}

inline auto DirRef::OpenWriteOnly(const std::filesystem::path& path,
                                  CreationOptions creation_options,
                                  ModeType creation_mode, OpenFlags flags)
    -> ErrorOr<WriteFile, PathError> {
  return OpenImpl<OpenAccess::WriteOnly>(path, creation_options, creation_mode,
                                         flags);
}

inline auto DirRef::OpenReadWrite(const std::filesystem::path& path,
                                  CreationOptions creation_options,
                                  ModeType creation_mode, OpenFlags flags)
    -> ErrorOr<ReadWriteFile, PathError> {
  return OpenImpl<OpenAccess::ReadWrite>(path, creation_options, creation_mode,
                                         flags);
}

inline auto DirRef::Rename(const std::filesystem::path& path, DirRef target_dir,
                           const std::filesystem::path& target_path)
    -> ErrorOr<Success, PathError> {
  if (renameat(dfd_, path.c_str(), target_dir.dfd_, target_path.c_str()) ==
      -1) {
    return PathError(errno, "Dir::Rename on '{0}' relative to '{1}'", path,
                     dfd_);
  }
  return Success();
}

inline auto DirRef::Chdir() -> ErrorOr<Success, FdError> {
  if (fchdir(dfd_) == -1) {
    return FdError(errno, "Dir::Chdir on '{0}'", dfd_);
  }
  return Success();
}

inline auto DirRef::Chdir(const std::filesystem::path& path)
    -> ErrorOr<Success, PathError> {
  if (path.is_absolute()) {
    if (chdir(path.c_str()) == -1) {
      return PathError(errno, "Dir::Chdir on '{0}' relative to '{1}'", path,
                       dfd_);
    }
    return Success();
  }

  CARBON_ASSIGN_OR_RETURN(Dir d, OpenDir(path));
  auto result = d.Chdir();
  if (result.ok()) {
    return Success();
  }
  return PathError(result.error().unix_errnum(),
                   "Dir::Chdir on '{0}' relative to '{1}'", path, dfd_);
}

inline auto DirRef::Symlink(const std::filesystem::path& path,
                            const std::string& target)
    -> ErrorOr<Success, PathError> {
  if (symlinkat(target.c_str(), dfd_, path.c_str()) == -1) {
    return PathError(errno, "Dir::Symlink on '{0}' relative to '{1}'", path,
                     dfd_);
  }
  return Success();
}

inline auto DirRef::Unlink(const std::filesystem::path& path)
    -> ErrorOr<Success, PathError> {
  if (unlinkat(dfd_, path.c_str(), /*flags=*/0) == -1) {
    return PathError(errno, "Dir::Unlink on '{0}' relative to '{1}'", path,
                     dfd_);
  }
  return Success();
}

inline auto DirRef::Rmdir(const std::filesystem::path& path)
    -> ErrorOr<Success, PathError> {
  if (unlinkat(dfd_, path.c_str(), AT_REMOVEDIR) == -1) {
    return PathError(errno, "Dir::Rmdir on '{0}' relative to '{1}'", path,
                     dfd_);
  }
  return Success();
}

template <OpenAccess A>
inline auto DirRef::OpenImpl(const std::filesystem::path& path,
                             CreationOptions creation_options,
                             ModeType creation_mode, OpenFlags flags)
    -> ErrorOr<File<A>, PathError> {
  for (;;) {
    int fd = openat(dfd_, path.c_str(),
                    static_cast<int>(A) | static_cast<int>(creation_options) |
                        static_cast<int>(flags),
                    creation_mode);
    if (fd == -1) {
      // May need to retry on `EINTR` when opening FIFOs on Linux.
      if (errno == EINTR) {
        continue;
      }
      return PathError(errno, "Dir::Open on '{0}' relative to '{1}'", path,
                       dfd_);
    }
    return File<A>(fd);
  }
}

constexpr Dir::~Dir() { Destroy(); }

inline auto Dir::TakeAndRead() && -> ErrorOr<Reader, FdError> {
  // Transition our file descriptor into a directory stream, clearing it in the
  // process.
  int dfd = std::exchange(dfd_, -1);
  DIR* dirp = fdopendir(dfd);
  if (dirp == nullptr) {
    return FdError(errno, "Dir::Read on '{0}'", dfd);
  }
  return Dir::Reader(dirp);
}

inline Dir::Dir(RemovingDir&& arg) noexcept : Dir(static_cast<Dir&&>(arg)) {
  arg.abs_path_.clear();
}

constexpr auto Dir::Destroy() -> void {
  if (dfd_ != -1 && dfd_ != AT_FDCWD) {
    auto result = close(dfd_);
    // Closing a directory shouldn't produce errors, directly check fail on any.
    //
    // This is a very different case from `close` on a file producing an error.
    // We don't actually write through the directory file descriptor, and for
    // most platforms `closedir` (the closest thing in documentation and
    // exclusively about directories), only provides a very few possible errors
    // here:
    //
    // EBADF: This should be precluded by the types here, and so we consider
    //        it a programming error.
    //
    // EINTR: Technically, a system could fail here. We have good evidence
    //        that systems we practically support don't as there also is nothing
    //        useful to *do* in the face of this: retrying on almost all systems
    //        is not allowed as the file descriptor is immediately released. And
    //        here, there is no potentially dropped data to report.
    //
    // If we ever discover a platform that fails here, we should adjust this
    // code to not fail in the face of that, likely by dropping the error. If we
    // end up supporting a platform that actually requires well-specified
    // retries, this code should handle that. Until then, we require these to
    // succeed so we will learn about any issues during porting to new
    // platforms.
    CARBON_CHECK(result == 0, "{0}",
                 FdError(errno, "Dir::Destroy on '{0}'", dfd_));
  }
  dfd_ = -1;
}

inline RemovingDir::~RemovingDir() {
  if (dfd_ != -1) {
    auto result = std::move(*this).Remove();
    CARBON_CHECK(result.ok(), "{0}", result.error());
  }
}

inline auto RemovingDir::Remove() && -> ErrorOr<Success, PathError> {
  CARBON_CHECK(dfd_ != -1,
               "Unexpected explicit remove on a `RemovingDir` with no owned "
               "directory!");

  // Close the directory base object prior to removing it.
  static_cast<Dir&>(*this) = Dir();
  return Cwd().Rmtree(abs_path_);
}

inline auto Dir::Iterator::operator++() -> Iterator& {
  CARBON_CHECK(dirp_, "Cannot increment an end-iterator");

  errno = 0;
  entry_.dent_ = readdir(dirp_);
  // There are no documented errors beyond an erroneous `dirp_` which would be
  // a programming error and not due to any recoverable failure of the
  // filesystem.
  CARBON_CHECK(entry_.dent_ != nullptr || errno == 0,
               "Using a directory iterator with a non-directory, errno '{0}'",
               errno);
  if (entry_.dent_ == nullptr) {
    // Clear the directory pointer to ease debugging increments past the end.
    dirp_ = nullptr;
  }
  return *this;
}

inline auto Dir::Reader::begin() -> Iterator {
  // Reset the position of the directory stream to get the actual beginning.
  rewinddir(dirp_);
  return Iterator(dirp_);
}

inline auto Dir::Reader::end() -> Iterator { return Iterator(); }

inline auto Dir::Reader::Destroy() -> void {
  if (dirp_) {
    int result = closedir(dirp_);
    // Closing a directory shouldn't produce interesting errors, so check fail
    // on them directly.
    //
    // See the detailed comment on `Dir::Destroy` for more context on closing of
    // directories, why we check-fail, and what we should do if we discover
    // platforms where an error needs to be handled here.
    CARBON_CHECK(result == 0, "{0}",
                 FdError(errno, "Dir::Reader::Destroy on '{0}'", dfd_));
    dirp_ = nullptr;
    dfd_ = -1;
  }
}

}  // namespace Carbon::Filesystem

#endif  // CARBON_COMMON_FILESYSTEM_H_
