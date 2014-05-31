//===- llvm/Support/Unix/Path.inc - Unix Path Implementation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific implementation of the Path API.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include <limits.h>
#include <stdio.h>
#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#if HAVE_FCNTL_H
#include <fcntl.h>
#endif
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif
#if HAVE_DIRENT_H
# include <dirent.h>
# define NAMLEN(dirent) strlen((dirent)->d_name)
#else
# define dirent direct
# define NAMLEN(dirent) (dirent)->d_namlen
# if HAVE_SYS_NDIR_H
#  include <sys/ndir.h>
# endif
# if HAVE_SYS_DIR_H
#  include <sys/dir.h>
# endif
# if HAVE_NDIR_H
#  include <ndir.h>
# endif
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

// Both stdio.h and cstdio are included via different pathes and
// stdcxx's cstdio doesn't include stdio.h, so it doesn't #undef the macros
// either.
#undef ferror
#undef feof

// For GNU Hurd
#if defined(__GNU__) && !defined(PATH_MAX)
# define PATH_MAX 4096
#endif

using namespace llvm;

namespace {
  /// This class automatically closes the given file descriptor when it goes out
  /// of scope. You can take back explicit ownership of the file descriptor by
  /// calling take(). The destructor does not verify that close was successful.
  /// Therefore, never allow this class to call close on a file descriptor that
  /// has been read from or written to.
  struct AutoFD {
    int FileDescriptor;

    AutoFD(int fd) : FileDescriptor(fd) {}
    ~AutoFD() {
      if (FileDescriptor >= 0)
        ::close(FileDescriptor);
    }

    int take() {
      int ret = FileDescriptor;
      FileDescriptor = -1;
      return ret;
    }

    operator int() const {return FileDescriptor;}
  };
}

static error_code TempDir(SmallVectorImpl<char> &result) {
  // FIXME: Don't use TMPDIR if program is SUID or SGID enabled.
  const char *dir = nullptr;
  (dir = std::getenv("TMPDIR")) || (dir = std::getenv("TMP")) ||
      (dir = std::getenv("TEMP")) || (dir = std::getenv("TEMPDIR")) ||
#ifdef P_tmpdir
      (dir = P_tmpdir) ||
#endif
      (dir = "/tmp");

  result.clear();
  StringRef d(dir);
  result.append(d.begin(), d.end());
  return error_code::success();
}

namespace llvm {
namespace sys  {
namespace fs {
#if defined(__FreeBSD__) || defined (__NetBSD__) || defined(__Bitrig__) || \
    defined(__OpenBSD__) || defined(__minix) || defined(__FreeBSD_kernel__) || \
    defined(__linux__) || defined(__CYGWIN__) || defined(__DragonFly__)
static int
test_dir(char ret[PATH_MAX], const char *dir, const char *bin)
{  
  struct stat sb;
  char fullpath[PATH_MAX];

  snprintf(fullpath, PATH_MAX, "%s/%s", dir, bin);
  if (realpath(fullpath, ret) == NULL)
    return (1);
  if (stat(fullpath, &sb) != 0)
    return (1);

  return (0);
}

static char *
getprogpath(char ret[PATH_MAX], const char *bin)
{
  char *pv, *s, *t;

  /* First approach: absolute path. */
  if (bin[0] == '/') {
    if (test_dir(ret, "/", bin) == 0)
      return (ret);
    return (NULL);
  }

  /* Second approach: relative path. */
  if (strchr(bin, '/') != NULL) {
    char cwd[PATH_MAX];
    if (getcwd(cwd, PATH_MAX) == NULL)
      return (NULL);
    if (test_dir(ret, cwd, bin) == 0)
      return (ret);
    return (NULL);
  }

  /* Third approach: $PATH */
  if ((pv = getenv("PATH")) == NULL)
    return (NULL);
  s = pv = strdup(pv);
  if (pv == NULL)
    return (NULL);
  while ((t = strsep(&s, ":")) != NULL) {
    if (test_dir(ret, t, bin) == 0) {
      free(pv);
      return (ret);
    }
  }
  free(pv);
  return (NULL);
}
#endif // __FreeBSD__ || __NetBSD__ || __FreeBSD_kernel__

/// GetMainExecutable - Return the path to the main executable, given the
/// value of argv[0] from program startup.
std::string getMainExecutable(const char *argv0, void *MainAddr) {
#if defined(__APPLE__)
  // On OS X the executable path is saved to the stack by dyld. Reading it
  // from there is much faster than calling dladdr, especially for large
  // binaries with symbols.
  char exe_path[MAXPATHLEN];
  uint32_t size = sizeof(exe_path);
  if (_NSGetExecutablePath(exe_path, &size) == 0) {
    char link_path[MAXPATHLEN];
    if (realpath(exe_path, link_path))
      return link_path;
  }
#elif defined(__FreeBSD__) || defined (__NetBSD__) || defined(__Bitrig__) || \
      defined(__OpenBSD__) || defined(__minix) || defined(__DragonFly__) || \
      defined(__FreeBSD_kernel__)
  char exe_path[PATH_MAX];

  if (getprogpath(exe_path, argv0) != NULL)
    return exe_path;
#elif defined(__linux__) || defined(__CYGWIN__)
  char exe_path[MAXPATHLEN];
  StringRef aPath("/proc/self/exe");
  if (sys::fs::exists(aPath)) {
      // /proc is not always mounted under Linux (chroot for example).
      ssize_t len = readlink(aPath.str().c_str(), exe_path, sizeof(exe_path));
      if (len >= 0)
          return StringRef(exe_path, len);
  } else {
      // Fall back to the classical detection.
      if (getprogpath(exe_path, argv0) != NULL)
          return exe_path;
  }
#elif defined(HAVE_DLFCN_H)
  // Use dladdr to get executable path if available.
  Dl_info DLInfo;
  int err = dladdr(MainAddr, &DLInfo);
  if (err == 0)
    return "";

  // If the filename is a symlink, we need to resolve and return the location of
  // the actual executable.
  char link_path[MAXPATHLEN];
  if (realpath(DLInfo.dli_fname, link_path))
    return link_path;
#else
#error GetMainExecutable is not implemented on this host yet.
#endif
  return "";
}

TimeValue file_status::getLastModificationTime() const {
  TimeValue Ret;
  Ret.fromEpochTime(fs_st_mtime);
  return Ret;
}

UniqueID file_status::getUniqueID() const {
  return UniqueID(fs_st_dev, fs_st_ino);
}

error_code current_path(SmallVectorImpl<char> &result) {
  result.clear();

  const char *pwd = ::getenv("PWD");
  llvm::sys::fs::file_status PWDStatus, DotStatus;
  if (pwd && llvm::sys::path::is_absolute(pwd) &&
      !llvm::sys::fs::status(pwd, PWDStatus) &&
      !llvm::sys::fs::status(".", DotStatus) &&
      PWDStatus.getUniqueID() == DotStatus.getUniqueID()) {
    result.append(pwd, pwd + strlen(pwd));
    return error_code::success();
  }

#ifdef MAXPATHLEN
  result.reserve(MAXPATHLEN);
#else
// For GNU Hurd
  result.reserve(1024);
#endif

  while (true) {
    if (::getcwd(result.data(), result.capacity()) == nullptr) {
      // See if there was a real error.
      if (errno != errc::not_enough_memory)
        return error_code(errno, system_category());
      // Otherwise there just wasn't enough space.
      result.reserve(result.capacity() * 2);
    } else
      break;
  }

  result.set_size(strlen(result.data()));
  return error_code::success();
}

error_code create_directory(const Twine &path, bool IgnoreExisting) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::mkdir(p.begin(), S_IRWXU | S_IRWXG) == -1) {
    if (errno != errc::file_exists || !IgnoreExisting)
      return error_code(errno, system_category());
  }

  return error_code::success();
}

error_code normalize_separators(SmallVectorImpl<char> &Path) {
  for (auto PI = Path.begin(), PE = Path.end(); PI < PE; ++PI) {
    if (*PI == '\\') {
      auto PN = PI + 1;
      if (PN < PE && *PN == '\\')
        ++PI; // increment once, the for loop will move over the escaped slash
      else
        *PI = '/';
    }
  }
  return error_code::success();
}

// Note that we are using symbolic link because hard links are not supported by
// all filesystems (SMB doesn't).
error_code create_link(const Twine &to, const Twine &from) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::symlink(t.begin(), f.begin()) == -1)
    return error_code(errno, system_category());

  return error_code::success();
}

error_code remove(const Twine &path, bool IgnoreNonExisting) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  struct stat buf;
  if (lstat(p.begin(), &buf) != 0) {
    if (errno != errc::no_such_file_or_directory || !IgnoreNonExisting)
      return error_code(errno, system_category());
    return error_code::success();
  }

  // Note: this check catches strange situations. In all cases, LLVM should
  // only be involved in the creation and deletion of regular files.  This
  // check ensures that what we're trying to erase is a regular file. It
  // effectively prevents LLVM from erasing things like /dev/null, any block
  // special file, or other things that aren't "regular" files.
  if (!S_ISREG(buf.st_mode) && !S_ISDIR(buf.st_mode) && !S_ISLNK(buf.st_mode))
    return make_error_code(errc::operation_not_permitted);

  if (::remove(p.begin()) == -1) {
    if (errno != errc::no_such_file_or_directory || !IgnoreNonExisting)
      return error_code(errno, system_category());
  }

  return error_code::success();
}

error_code rename(const Twine &from, const Twine &to) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::rename(f.begin(), t.begin()) == -1)
    return error_code(errno, system_category());

  return error_code::success();
}

error_code resize_file(const Twine &path, uint64_t size) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::truncate(p.begin(), size) == -1)
    return error_code(errno, system_category());

  return error_code::success();
}

error_code exists(const Twine &path, bool &result) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::access(p.begin(), F_OK) == -1) {
    if (errno != errc::no_such_file_or_directory)
      return error_code(errno, system_category());
    result = false;
  } else
    result = true;

  return error_code::success();
}

bool can_write(const Twine &Path) {
  SmallString<128> PathStorage;
  StringRef P = Path.toNullTerminatedStringRef(PathStorage);
  return 0 == access(P.begin(), W_OK);
}

bool can_execute(const Twine &Path) {
  SmallString<128> PathStorage;
  StringRef P = Path.toNullTerminatedStringRef(PathStorage);

  if (0 != access(P.begin(), R_OK | X_OK))
    return false;
  struct stat buf;
  if (0 != stat(P.begin(), &buf))
    return false;
  if (!S_ISREG(buf.st_mode))
    return false;
  return true;
}

bool equivalent(file_status A, file_status B) {
  assert(status_known(A) && status_known(B));
  return A.fs_st_dev == B.fs_st_dev &&
         A.fs_st_ino == B.fs_st_ino;
}

error_code equivalent(const Twine &A, const Twine &B, bool &result) {
  file_status fsA, fsB;
  if (error_code ec = status(A, fsA)) return ec;
  if (error_code ec = status(B, fsB)) return ec;
  result = equivalent(fsA, fsB);
  return error_code::success();
}

static error_code fillStatus(int StatRet, const struct stat &Status,
                             file_status &Result) {
  if (StatRet != 0) {
    error_code ec(errno, system_category());
    if (ec == errc::no_such_file_or_directory)
      Result = file_status(file_type::file_not_found);
    else
      Result = file_status(file_type::status_error);
    return ec;
  }

  file_type Type = file_type::type_unknown;

  if (S_ISDIR(Status.st_mode))
    Type = file_type::directory_file;
  else if (S_ISREG(Status.st_mode))
    Type = file_type::regular_file;
  else if (S_ISBLK(Status.st_mode))
    Type = file_type::block_file;
  else if (S_ISCHR(Status.st_mode))
    Type = file_type::character_file;
  else if (S_ISFIFO(Status.st_mode))
    Type = file_type::fifo_file;
  else if (S_ISSOCK(Status.st_mode))
    Type = file_type::socket_file;

  perms Perms = static_cast<perms>(Status.st_mode);
  Result =
      file_status(Type, Perms, Status.st_dev, Status.st_ino, Status.st_mtime,
                  Status.st_uid, Status.st_gid, Status.st_size);

  return error_code::success();
}

error_code status(const Twine &Path, file_status &Result) {
  SmallString<128> PathStorage;
  StringRef P = Path.toNullTerminatedStringRef(PathStorage);

  struct stat Status;
  int StatRet = ::stat(P.begin(), &Status);
  return fillStatus(StatRet, Status, Result);
}

error_code status(int FD, file_status &Result) {
  struct stat Status;
  int StatRet = ::fstat(FD, &Status);
  return fillStatus(StatRet, Status, Result);
}

error_code setLastModificationAndAccessTime(int FD, TimeValue Time) {
#if defined(HAVE_FUTIMENS)
  timespec Times[2];
  Times[0].tv_sec = Time.toEpochTime();
  Times[0].tv_nsec = 0;
  Times[1] = Times[0];
  if (::futimens(FD, Times))
    return error_code(errno, system_category());
  return error_code::success();
#elif defined(HAVE_FUTIMES)
  timeval Times[2];
  Times[0].tv_sec = Time.toEpochTime();
  Times[0].tv_usec = 0;
  Times[1] = Times[0];
  if (::futimes(FD, Times))
    return error_code(errno, system_category());
  return error_code::success();
#else
#warning Missing futimes() and futimens()
  return make_error_code(errc::not_supported);
#endif
}

error_code mapped_file_region::init(int FD, bool CloseFD, uint64_t Offset) {
  AutoFD ScopedFD(FD);
  if (!CloseFD)
    ScopedFD.take();

  // Figure out how large the file is.
  struct stat FileInfo;
  if (fstat(FD, &FileInfo) == -1)
    return error_code(errno, system_category());
  uint64_t FileSize = FileInfo.st_size;

  if (Size == 0)
    Size = FileSize;
  else if (FileSize < Size) {
    // We need to grow the file.
    if (ftruncate(FD, Size) == -1)
      return error_code(errno, system_category());
  }

  int flags = (Mode == readwrite) ? MAP_SHARED : MAP_PRIVATE;
  int prot = (Mode == readonly) ? PROT_READ : (PROT_READ | PROT_WRITE);
#ifdef MAP_FILE
  flags |= MAP_FILE;
#endif
  Mapping = ::mmap(nullptr, Size, prot, flags, FD, Offset);
  if (Mapping == MAP_FAILED)
    return error_code(errno, system_category());
  return error_code::success();
}

mapped_file_region::mapped_file_region(const Twine &path,
                                       mapmode mode,
                                       uint64_t length,
                                       uint64_t offset,
                                       error_code &ec)
  : Mode(mode)
  , Size(length)
  , Mapping() {
  // Make sure that the requested size fits within SIZE_T.
  if (length > std::numeric_limits<size_t>::max()) {
    ec = make_error_code(errc::invalid_argument);
    return;
  }

  SmallString<128> path_storage;
  StringRef name = path.toNullTerminatedStringRef(path_storage);
  int oflags = (mode == readonly) ? O_RDONLY : O_RDWR;
  int ofd = ::open(name.begin(), oflags);
  if (ofd == -1) {
    ec = error_code(errno, system_category());
    return;
  }

  ec = init(ofd, true, offset);
  if (ec)
    Mapping = nullptr;
}

mapped_file_region::mapped_file_region(int fd,
                                       bool closefd,
                                       mapmode mode,
                                       uint64_t length,
                                       uint64_t offset,
                                       error_code &ec)
  : Mode(mode)
  , Size(length)
  , Mapping() {
  // Make sure that the requested size fits within SIZE_T.
  if (length > std::numeric_limits<size_t>::max()) {
    ec = make_error_code(errc::invalid_argument);
    return;
  }

  ec = init(fd, closefd, offset);
  if (ec)
    Mapping = nullptr;
}

mapped_file_region::~mapped_file_region() {
  if (Mapping)
    ::munmap(Mapping, Size);
}

mapped_file_region::mapped_file_region(mapped_file_region &&other)
  : Mode(other.Mode), Size(other.Size), Mapping(other.Mapping) {
  other.Mapping = nullptr;
}

mapped_file_region::mapmode mapped_file_region::flags() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return Mode;
}

uint64_t mapped_file_region::size() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return Size;
}

char *mapped_file_region::data() const {
  assert(Mapping && "Mapping failed but used anyway!");
  assert(Mode != readonly && "Cannot get non-const data for readonly mapping!");
  return reinterpret_cast<char*>(Mapping);
}

const char *mapped_file_region::const_data() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return reinterpret_cast<const char*>(Mapping);
}

int mapped_file_region::alignment() {
  return process::get_self()->page_size();
}

error_code detail::directory_iterator_construct(detail::DirIterState &it,
                                                StringRef path){
  SmallString<128> path_null(path);
  DIR *directory = ::opendir(path_null.c_str());
  if (!directory)
    return error_code(errno, system_category());

  it.IterationHandle = reinterpret_cast<intptr_t>(directory);
  // Add something for replace_filename to replace.
  path::append(path_null, ".");
  it.CurrentEntry = directory_entry(path_null.str());
  return directory_iterator_increment(it);
}

error_code detail::directory_iterator_destruct(detail::DirIterState &it) {
  if (it.IterationHandle)
    ::closedir(reinterpret_cast<DIR *>(it.IterationHandle));
  it.IterationHandle = 0;
  it.CurrentEntry = directory_entry();
  return error_code::success();
}

error_code detail::directory_iterator_increment(detail::DirIterState &it) {
  errno = 0;
  dirent *cur_dir = ::readdir(reinterpret_cast<DIR *>(it.IterationHandle));
  if (cur_dir == nullptr && errno != 0) {
    return error_code(errno, system_category());
  } else if (cur_dir != nullptr) {
    StringRef name(cur_dir->d_name, NAMLEN(cur_dir));
    if ((name.size() == 1 && name[0] == '.') ||
        (name.size() == 2 && name[0] == '.' && name[1] == '.'))
      return directory_iterator_increment(it);
    it.CurrentEntry.replace_filename(name);
  } else
    return directory_iterator_destruct(it);

  return error_code::success();
}

error_code get_magic(const Twine &path, uint32_t len,
                     SmallVectorImpl<char> &result) {
  SmallString<128> PathStorage;
  StringRef Path = path.toNullTerminatedStringRef(PathStorage);
  result.set_size(0);

  // Open path.
  std::FILE *file = std::fopen(Path.data(), "rb");
  if (!file)
    return error_code(errno, system_category());

  // Reserve storage.
  result.reserve(len);

  // Read magic!
  size_t size = std::fread(result.data(), 1, len, file);
  if (std::ferror(file) != 0) {
    std::fclose(file);
    return error_code(errno, system_category());
  } else if (size != len) {
    if (std::feof(file) != 0) {
      std::fclose(file);
      result.set_size(size);
      return make_error_code(errc::value_too_large);
    }
  }
  std::fclose(file);
  result.set_size(size);
  return error_code::success();
}

error_code openFileForRead(const Twine &Name, int &ResultFD) {
  SmallString<128> Storage;
  StringRef P = Name.toNullTerminatedStringRef(Storage);
  while ((ResultFD = open(P.begin(), O_RDONLY)) < 0) {
    if (errno != EINTR)
      return error_code(errno, system_category());
  }
  return error_code::success();
}

error_code openFileForWrite(const Twine &Name, int &ResultFD,
                            sys::fs::OpenFlags Flags, unsigned Mode) {
  // Verify that we don't have both "append" and "excl".
  assert((!(Flags & sys::fs::F_Excl) || !(Flags & sys::fs::F_Append)) &&
         "Cannot specify both 'excl' and 'append' file creation flags!");

  int OpenFlags = O_CREAT;

  if (Flags & F_RW)
    OpenFlags |= O_RDWR;
  else
    OpenFlags |= O_WRONLY;

  if (Flags & F_Append)
    OpenFlags |= O_APPEND;
  else
    OpenFlags |= O_TRUNC;

  if (Flags & F_Excl)
    OpenFlags |= O_EXCL;

  SmallString<128> Storage;
  StringRef P = Name.toNullTerminatedStringRef(Storage);
  while ((ResultFD = open(P.begin(), OpenFlags, Mode)) < 0) {
    if (errno != EINTR)
      return error_code(errno, system_category());
  }
  return error_code::success();
}

} // end namespace fs

namespace path {

bool home_directory(SmallVectorImpl<char> &result) {
  if (char *RequestedDir = getenv("HOME")) {
    result.clear();
    result.append(RequestedDir, RequestedDir + strlen(RequestedDir));
    return true;
  }

  return false;
}

} // end namespace path

} // end namespace sys
} // end namespace llvm
