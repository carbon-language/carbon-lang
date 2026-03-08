#ifdef _WIN32
#ifndef CARBON_COMMON_FILESYSTEM_WIN32_H_
#define CARBON_COMMON_FILESYSTEM_WIN32_H_

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vector>
#include <winioctl.h>
#include <string>
#include <io.h>
#include <direct.h>
#include <process.h>

#ifndef AT_FDCWD
#define AT_FDCWD -100
#endif
#ifndef PIPE_BUF
#define PIPE_BUF 4096
#endif
#ifndef O_DIRECTORY
#define O_DIRECTORY 0x10000
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#ifndef EWOULDBLOCK
#define EWOULDBLOCK EAGAIN
#endif
#ifndef TIMER_ABSTIME
#define TIMER_ABSTIME 1
#endif
#ifndef ssize_t
typedef intptr_t ssize_t;
#endif
#ifndef uid_t
typedef int uid_t;
#endif

// Directory fd table: maps fake fd values (1000+) to real HANDLEs
#define _CARBON_DIR_FD_BASE 1000
#define _CARBON_DIR_FD_MAX  512
// Defined once in filesystem.cpp
extern HANDLE _carbon_dir_handles[_CARBON_DIR_FD_MAX];
extern volatile LONG _carbon_dir_next;

static inline int _carbon_dir_add(HANDLE h) {
  LONG idx = InterlockedIncrement(&_carbon_dir_next) - 1;
  if (idx >= _CARBON_DIR_FD_MAX) return -1;
  _carbon_dir_handles[idx] = h;
  return (int)(idx + _CARBON_DIR_FD_BASE);
}
static inline HANDLE _carbon_dir_get(int fd) {
  int idx = fd - _CARBON_DIR_FD_BASE;
  if (idx < 0 || idx >= _CARBON_DIR_FD_MAX) return INVALID_HANDLE_VALUE;
  return _carbon_dir_handles[idx];
}
static inline void _carbon_dir_remove(int fd) {
  int idx = fd - _CARBON_DIR_FD_BASE;
  if (idx >= 0 && idx < _CARBON_DIR_FD_MAX) _carbon_dir_handles[idx] = INVALID_HANDLE_VALUE;
}
static inline bool _carbon_is_dir_fd(int fd) { return fd >= _CARBON_DIR_FD_BASE; }

// mkdirat
#ifdef mkdirat
#undef mkdirat
#endif
static inline int mkdirat(int dfd, const wchar_t* path, int) {
  // Resolve path relative to dfd
  std::wstring norm(path);
  for (auto& c : norm) { if (c == L'/') c = L'\\'; }
  bool is_abs = (norm.size() > 1 && norm[1] == L':') || (!norm.empty() && norm[0] == L'\\');
  if (dfd == -100 || is_abs) {
    return _wmkdir(norm.c_str());
  }
  HANDLE h = _carbon_is_dir_fd(dfd) ? _carbon_dir_get(dfd) : (HANDLE)_get_osfhandle(dfd);
  if (h != INVALID_HANDLE_VALUE) {
    std::vector<wchar_t> _dir_buf_vec(32768); wchar_t* dir_buf = _dir_buf_vec.data();
    DWORD len = GetFinalPathNameByHandleW(h, dir_buf, 32768, FILE_NAME_NORMALIZED);
    if (len > 0) {
      std::wstring full(dir_buf, len);
      if (full.size() >= 4 && full[0]==L'\\' && full[1]==L'\\' && full[2]==L'?' && full[3]==L'\\')
        full = full.substr(4);
      if (!full.empty() && full.back() != L'\\') full += L'\\';
      full += norm;
      return _wmkdir(full.c_str());
    }
  }
  return _wmkdir(norm.c_str());
}

// openat
#ifdef openat
#undef openat
#endif
static inline std::wstring _carbon_resolve_path(int dfd, const wchar_t* path) {
  // Normalize forward slashes to backslashes
  std::wstring norm(path);
  for (auto& c : norm) { if (c == L'/') c = L'\\'; }
  // If absolute path or AT_FDCWD, use as-is
  bool is_abs = (norm.size() > 1 && norm[1] == L':') || (!norm.empty() && norm[0] == L'\\');
  if (dfd == -100 || is_abs) return norm;
  // Resolve relative to dfd
  HANDLE h = _carbon_is_dir_fd(dfd) ? _carbon_dir_get(dfd) : (HANDLE)_get_osfhandle(dfd);
  if (h == INVALID_HANDLE_VALUE) return norm;
  std::vector<wchar_t> _dir_buf_vec(32768); wchar_t* dir_buf = _dir_buf_vec.data();
  DWORD len = GetFinalPathNameByHandleW(h, dir_buf, 32768, FILE_NAME_NORMALIZED);
  if (len == 0) return norm;
  std::wstring full(dir_buf, len);
  // Strip \\?\ prefix
  if (full.size() >= 4 && full[0]==L'\\' && full[1]==L'\\' && full[2]==L'?' && full[3]==L'\\')
    full = full.substr(4);
  if (!full.empty() && full.back() != L'\\') full += L'\\';
  full += norm;
  return full;
}
static inline int openat(int dfd, const wchar_t* path, int flags, int mode) {
  std::wstring full = _carbon_resolve_path(dfd, path);
  if (flags & O_DIRECTORY) {
    // If O_CREAT is set, try to create the directory first using resolved full path
    if (flags & O_CREAT) {
      _wmkdir(full.c_str());  // ignore error - may already exist, full path already resolved
    }
    HANDLE h = CreateFileW(full.c_str(), GENERIC_READ,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) { errno = ENOENT; return -1; }
    int fd = _carbon_dir_add(h);
    return fd;
  }
  return _wopen(full.c_str(), flags & ~O_DIRECTORY, mode);
}
static inline int openat(int dfd, const wchar_t* path, int flags) {
  return openat(dfd, path, flags, 0);
}

// readlinkat
#ifdef readlinkat
#undef readlinkat
#endif
static inline ssize_t readlinkat(int, const wchar_t* path, char* buf, size_t bufsiz) {
  HANDLE h = CreateFileW(path, 0,
    FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
    nullptr, OPEN_EXISTING,
    FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (h == INVALID_HANDLE_VALUE) { errno = ENOENT; return -1; }
  struct { DWORD ReparseTag; WORD ReparseDataLength; WORD Reserved;
    WORD SubstituteNameOffset; WORD SubstituteNameLength;
    WORD PrintNameOffset; WORD PrintNameLength;
    ULONG Flags; WCHAR PathBuffer[MAX_PATH*2]; } rbuf;
  DWORD n = 0;
  BOOL ok = DeviceIoControl(h, FSCTL_GET_REPARSE_POINT, nullptr, 0, &rbuf, sizeof(rbuf), &n, nullptr);
  CloseHandle(h);
  if (!ok || rbuf.ReparseTag != IO_REPARSE_TAG_SYMLINK) { errno = EINVAL; return -1; }
  WCHAR* name = rbuf.PathBuffer + rbuf.PrintNameOffset/sizeof(WCHAR);
  int nlen = rbuf.PrintNameLength/sizeof(WCHAR);
  int r = WideCharToMultiByte(CP_UTF8, 0, name, nlen, buf, (int)bufsiz, nullptr, nullptr);
  if (r == 0) { errno = ENOMEM; return -1; }
  return r;
}

// symlinkat - char version
#ifdef symlinkat
#undef symlinkat
#endif
static inline int symlinkat(const char* target, int, const char* path) {
  DWORD flags = 0x2;
  DWORD attrs = GetFileAttributesA(target);
  if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY))
    flags |= SYMBOLIC_LINK_FLAG_DIRECTORY;
  return CreateSymbolicLinkA(path, target, flags) ? 0 : (errno = EPERM, -1);
}

// symlinkat - mixed wchar_t* path version (target is char* on Windows call sites)
static inline int symlinkat(const wchar_t* target, int, const wchar_t* path) {
  DWORD flags = 0x2;
  DWORD attrs = GetFileAttributesW(target);
  if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY))
    flags |= SYMBOLIC_LINK_FLAG_DIRECTORY;
  return CreateSymbolicLinkW(path, target, flags) ? 0 : (errno = EPERM, -1);
}
static inline int symlinkat(const char* target, int dfd, const wchar_t* path) {
  std::wstring wtarget(target, target + strlen(target));
  return symlinkat(wtarget.c_str(), dfd, path);
}
// mkdtemp
static inline char* mkdtemp(char* tmpl) {
  size_t len = strlen(tmpl);
  if (len < 6 || strcmp(tmpl+len-6,"XXXXXX") != 0) { errno = EINVAL; return nullptr; }
  char* x = tmpl+len-6;
  for (int i = 0; i < 100; i++) {
    unsigned r = (unsigned)GetTickCount() ^ ((unsigned)i*1234567u);
    for (int j = 0; j < 6; j++) { x[j] = "abcdefghijklmnopqrstuvwxyz0123456789"[r%36]; r/=36; }
    if (_mkdir(tmpl) == 0) return tmpl;
    if (errno != EEXIST) return nullptr;
  }
  errno = EEXIST; return nullptr;
}


// faccessat
#ifdef faccessat
#undef faccessat
#endif
static inline int faccessat(int dfd, const wchar_t* path, int mode, int) {
  // Normalize forward slashes to backslashes
  std::wstring norm(path);
  for (auto& c : norm) { if (c == L'/') c = L'\\'; }
  std::wstring full_path;
  // If dfd is AT_FDCWD or path is absolute, use path directly
  if (dfd == -100 || (norm.size() > 1 && norm[1] == L':') || (!norm.empty() && norm[0] == L'\\')) {
    full_path = norm;
  } else {
    // Resolve the directory fd to a path
    HANDLE h = _carbon_is_dir_fd(dfd) ? _carbon_dir_get(dfd) : (HANDLE)_get_osfhandle(dfd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    std::vector<wchar_t> _dir_buf_vec(32768); wchar_t* dir_buf = _dir_buf_vec.data();
    DWORD len = GetFinalPathNameByHandleW(h, dir_buf, 32768, FILE_NAME_NORMALIZED);
    if (len == 0) { errno = ENOENT; return -1; }
    full_path = std::wstring(dir_buf, len);
    // Strip \\?\ prefix if present
    if (full_path.substr(0, 4) == L"\\\\?\\") full_path = full_path.substr(4);
    if (!full_path.empty() && full_path.back() != L'\\') full_path += L'\\';
    full_path += norm;
  }
  DWORD attrs = GetFileAttributesW(full_path.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    errno = ENOENT;
    return -1;
  }
  if ((mode & 2) && (attrs & FILE_ATTRIBUTE_READONLY)) {
    errno = EACCES;
    return -1;
  }
  return 0;
}


// Map struct stat to _stat64 for large file support and add missing fields
#ifndef _CARBON_STAT_WIN32_DEFINED
#define _CARBON_STAT_WIN32_DEFINED
#include <sys/stat.h>
#include <sys/types.h>
// Remap struct stat to _stat64
#ifdef stat
#undef stat
#endif
#define stat _stat64
#ifdef fstat
#undef fstat
#endif
static inline int fstat(int fd, struct _stat64* buf) {
  return _fstat64(fd, buf);
}
#ifdef fstatat
#undef fstatat
#endif
static inline int fstatat(int dfd, const wchar_t* path, struct _stat64* buf, int /*flags*/) {
  std::wstring full = _carbon_resolve_path(dfd, path);
  return _wstat64(full.c_str(), buf);
}
// futimens - no-op on Windows
#ifdef futimens
#undef futimens
#endif
static inline int futimens(int, const struct timespec*) { return 0; }
// unix_uid stub
static inline int unix_uid() { return 0; }
#endif // _CARBON_STAT_WIN32_DEFINED


// lseek - use _lseeki64 for large file support
#ifdef lseek
#undef lseek
#endif
#define lseek _lseeki64

// pread - read at offset without seeking
#ifdef pread
#undef pread
#endif
static inline ssize_t pread(int fd, void* buf, size_t count, int64_t offset) {
  int64_t old_pos = _lseeki64(fd, 0, SEEK_CUR);
  if (old_pos == -1) return -1;
  if (_lseeki64(fd, offset, SEEK_SET) == -1) return -1;
  ssize_t result = _read(fd, buf, (unsigned int)count);
  _lseeki64(fd, old_pos, SEEK_SET);
  return result;
}

// pwrite - write at offset without seeking
#ifdef pwrite
#undef pwrite
#endif
static inline ssize_t pwrite(int fd, const void* buf, size_t count, int64_t offset) {
  int64_t old_pos = _lseeki64(fd, 0, SEEK_CUR);
  if (old_pos == -1) return -1;
  if (_lseeki64(fd, offset, SEEK_SET) == -1) return -1;
  ssize_t result = _write(fd, buf, (unsigned int)count);
  _lseeki64(fd, old_pos, SEEK_SET);
  return result;
}

// ftruncate
#ifdef ftruncate
#undef ftruncate
#endif
static inline int ftruncate(int fd, int64_t length) {
  return _chsize_s(fd, length);
}




// close - safe wrapper that handles directory HANDLE-based fds on Windows
#define unlink _unlink

  if (_carbon_is_dir_fd(fd)) {
    HANDLE h = _carbon_dir_get(fd);
    _carbon_dir_remove(fd);
    return 0;
  }
  return _close(fd);
}
#ifdef close
#undef close
#endif
#define close _carbon_safe_close

// dup / dup2 - duplicate file descriptors
#ifdef dup
#undef dup
#endif
static inline int dup(int fd) {
  return _dup(fd);
}

#ifdef dup2
#undef dup2
#endif
static inline int dup2(int fd, int fd2) {
  return _dup2(fd, fd2);
}

// geteuid
static inline uid_t geteuid() { return 0; }

// flock
#undef flock
static inline int flock(int, int) { return 0; }

// clock_nanosleep
static inline int clock_nanosleep(int, int, const struct timespec* ts, struct timespec*) {
  if (ts) { DWORD ms = (DWORD)(ts->tv_sec*1000 + ts->tv_nsec/1000000); if (ms>0) ::Sleep(ms); }
  return 0;
}


// DIR / dirent / opendir / readdir / closedir / fdopendir
#undef DIR
#undef dirent
#ifndef _CARBON_DIR_IMPL
#define _CARBON_DIR_IMPL
struct _CarbonDirent {
  char d_name[32768];
  unsigned char d_type;
};
#ifndef dirent
#define dirent _CarbonDirent
#endif
#define DT_DIR  4
#define DT_REG  8
#define DT_UNKNOWN 0
#define DT_LNK 10

struct _CarbonDIR {
  HANDLE hFind;
  WIN32_FIND_DATAW ffd;
  _CarbonDirent entry;
  bool first;
  wchar_t* path;
};
#define DIR _CarbonDIR

  int wlen = MultiByteToWideChar(CP_UTF8, 0, name, -1, nullptr, 0);
  wchar_t* wname = new wchar_t[wlen + 4];
  MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, wlen);
  // Append \* for FindFirstFile
  size_t l = wcslen(wname);
  if (l > 0 && wname[l-1] != L'\\') wname[l++] = L'\\';
  wname[l++] = L'*'; wname[l] = 0;
  _CarbonDIR* d = new _CarbonDIR();
  d->path = wname;
  d->hFind = INVALID_HANDLE_VALUE;
  d->first = true;
  return d;
}

  HANDLE h = _carbon_is_dir_fd(fd) ? _carbon_dir_get(fd) : (HANDLE)_get_osfhandle(fd);
  if (h == INVALID_HANDLE_VALUE) return nullptr;
  std::vector<wchar_t> _dir_buf_vec(32768); wchar_t* dir_buf = _dir_buf_vec.data();
  DWORD len = GetFinalPathNameByHandleW(h, dir_buf, 32768, FILE_NAME_NORMALIZED);
  if (len == 0) return nullptr;
  std::wstring full(dir_buf, len);
  if (full.size() >= 4 && full[0]==L'\\' && full[1]==L'\\' && full[2]==L'?' && full[3]==L'\\')
    full = full.substr(4);
  if (!full.empty() && full.back() != L'\\') full += L'\\';
  full += L'*';
  _CarbonDIR* d = new _CarbonDIR();
  d->path = new wchar_t[full.size()+1];
  wcsncpy(d->path, full.c_str(), full.size());
  d->hFind = INVALID_HANDLE_VALUE;
  d->first = true;
  return d;
}

static inline struct _CarbonDirent* readdir(_CarbonDIR* d) {
  if (d->first) {
    d->hFind = FindFirstFileW(d->path, &d->ffd);
    d->first = false;
    if (d->hFind == INVALID_HANDLE_VALUE) return nullptr;
  } else {
    if (!FindNextFileW(d->hFind, &d->ffd)) return nullptr;
  }
  WideCharToMultiByte(CP_UTF8, 0, d->ffd.cFileName, -1, d->entry.d_name, 32767, nullptr, nullptr);
  d->entry.d_type = (d->ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? DT_DIR : DT_REG;
  return &d->entry;
}

static inline int closedir(_CarbonDIR* d) {
  if (d->hFind != INVALID_HANDLE_VALUE) FindClose(d->hFind);
  delete[] d->path;
  delete d;
  return 0;
}
static inline void rewinddir(_CarbonDIR* d) { if (d->hFind != INVALID_HANDLE_VALUE) { FindClose(d->hFind); d->hFind = INVALID_HANDLE_VALUE; } d->first = true; }
static inline int dirfd(_CarbonDIR* d) { (void)d; return -1; }
#endif // _CARBON_DIR_IMPL

#endif // CARBON_COMMON_FILESYSTEM_WIN32_H_
#endif // _WIN32
