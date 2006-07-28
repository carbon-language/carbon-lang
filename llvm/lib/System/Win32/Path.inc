//===- llvm/System/Linux/Path.cpp - Linux Path Implementation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
// Modified by Henrik Bach to comply with at least MinGW.
// Ported to Win32 by Jeff Cohen.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the Path class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Win32 code that
//===          is guaranteed to work on *all* Win32 variants.
//===----------------------------------------------------------------------===//

#include "Win32.h"
#include <malloc.h>

// We need to undo a macro defined in Windows.h, otherwise we won't compile:
#undef CopyFile

// Windows happily accepts either forward or backward slashes, though any path
// returned by a Win32 API will have backward slashes.  As LLVM code basically
// assumes forward slashes are used, backward slashs are converted where they
// can be introduced into a path.
//
// Another invariant is that a path ends with a slash if and only if the path
// is a root directory.  Any other use of a trailing slash is stripped.  Unlike
// in Unix, Windows has a rather complicated notion of a root path and this
// invariant helps simply the code.

static void FlipBackSlashes(std::string& s) {
  for (size_t i = 0; i < s.size(); i++)
    if (s[i] == '\\')
      s[i] = '/';
}

namespace llvm {
namespace sys {

bool
Path::isValid() const {
  if (path.empty())
    return false;

  // If there is a colon, it must be the second character, preceded by a letter
  // and followed by something.
  size_t len = path.size();
  size_t pos = path.rfind(':',len);
  size_t rootslash = 0;
  if (pos != std::string::npos) {
    if (pos != 1 || !isalpha(path[0]) || len < 3)
      return false;
      rootslash = 2;
  }

  // Look for a UNC path, and if found adjust our notion of the root slash.
  if (len > 3 && path[0] == '/' && path[1] == '/') {
    rootslash = path.find('/', 2);
    if (rootslash == std::string::npos)
      rootslash = 0;
  }

  // Check for illegal characters.
  if (path.find_first_of("\\<>\"|\001\002\003\004\005\006\007\010\011\012"
                         "\013\014\015\016\017\020\021\022\023\024\025\026"
                         "\027\030\031\032\033\034\035\036\037")
      != std::string::npos)
    return false;

  // Remove trailing slash, unless it's a root slash.
  if (len > rootslash+1 && path[len-1] == '/')
    path.erase(--len);

  // Check each component for legality.
  for (pos = 0; pos < len; ++pos) {
    // A component may not end in a space.
    if (path[pos] == ' ') {
      if (path[pos+1] == '/' || path[pos+1] == '\0')
        return false;
    }

    // A component may not end in a period.
    if (path[pos] == '.') {
      if (path[pos+1] == '/' || path[pos+1] == '\0') {
        // Unless it is the pseudo-directory "."...
        if (pos == 0 || path[pos-1] == '/' || path[pos-1] == ':')
          return true;
        // or "..".
        if (pos > 0 && path[pos-1] == '.') {
          if (pos == 1 || path[pos-2] == '/' || path[pos-2] == ':')
            return true;
        }
        return false;
      }
    }
  }

  return true;
}

static Path *TempDirectory = NULL;

Path
Path::GetTemporaryDirectory() {
  if (TempDirectory)
    return *TempDirectory;

  char pathname[MAX_PATH];
  if (!GetTempPath(MAX_PATH, pathname))
    throw std::string("Can't determine temporary directory");

  Path result;
  result.set(pathname);

  // Append a subdirectory passed on our process id so multiple LLVMs don't
  // step on each other's toes.
  sprintf(pathname, "LLVM_%u", unsigned(GetCurrentProcessId()));
  result.appendComponent(pathname);

  // If there's a directory left over from a previous LLVM execution that
  // happened to have the same process id, get rid of it.
  result.eraseFromDisk(true);

  // And finally (re-)create the empty directory.
  result.createDirectoryOnDisk(false);
  TempDirectory = new Path(result);
  return *TempDirectory;
}

Path::Path(const std::string& unverified_path)
  : path(unverified_path)
{
  FlipBackSlashes(path);
  if (unverified_path.empty())
    return;
  if (this->isValid())
    return;
  // oops, not valid.
  path.clear();
  throw std::string(unverified_path + ": path is not valid");
}

// FIXME: the following set of functions don't map to Windows very well.
Path
Path::GetRootDirectory() {
  Path result;
  result.set("C:/");
  return result;
}

static void getPathList(const char*path, std::vector<sys::Path>& Paths) {
  const char* at = path;
  const char* delim = strchr(at, ';');
  Path tmpPath;
  while (delim != 0) {
    std::string tmp(at, size_t(delim-at));
    if (tmpPath.set(tmp))
      if (tmpPath.canRead())
        Paths.push_back(tmpPath);
    at = delim + 1;
    delim = strchr(at, ';');
  }

  if (*at != 0)
    if (tmpPath.set(std::string(at)))
      if (tmpPath.canRead())
        Paths.push_back(tmpPath);
}

void
Path::GetSystemLibraryPaths(std::vector<sys::Path>& Paths) {
  Paths.push_back(sys::Path("C:/WINDOWS/SYSTEM32"));
  Paths.push_back(sys::Path("C:/WINDOWS"));
}

void
Path::GetBytecodeLibraryPaths(std::vector<sys::Path>& Paths) {
  char * env_var = getenv("LLVM_LIB_SEARCH_PATH");
  if (env_var != 0) {
    getPathList(env_var,Paths);
  }
#ifdef LLVM_LIBDIR
  {
    Path tmpPath;
    if (tmpPath.set(LLVM_LIBDIR))
      if (tmpPath.canRead())
        Paths.push_back(tmpPath);
  }
#endif
  GetSystemLibraryPaths(Paths);
}

Path
Path::GetLLVMDefaultConfigDir() {
  // TODO: this isn't going to fly on Windows
  return Path("/etc/llvm");
}

Path
Path::GetUserHomeDirectory() {
  // TODO: Typical Windows setup doesn't define HOME.
  const char* home = getenv("HOME");
  if (home) {
    Path result;
    if (result.set(home))
      return result;
  }
  return GetRootDirectory();
}
// FIXME: the above set of functions don't map to Windows very well.

bool
Path::isFile() const {
  WIN32_FILE_ATTRIBUTE_DATA fi;
  BOOL rc = GetFileAttributesEx(path.c_str(), GetFileExInfoStandard, &fi);
  if (rc)
    return !(fi.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
  else if (GetLastError() != ERROR_FILE_NOT_FOUND) {
    ThrowError("isFile(): " + std::string(path) + ": Can't get status: ");
  }
  return false;
}

bool
Path::isDirectory() const {
  WIN32_FILE_ATTRIBUTE_DATA fi;
  BOOL rc = GetFileAttributesEx(path.c_str(), GetFileExInfoStandard, &fi);
  if (rc)
    return fi.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY;
  else if (GetLastError() != ERROR_FILE_NOT_FOUND)
    ThrowError("isDirectory(): " + std::string(path) + ": Can't get status: ");
  return false;
}

bool
Path::isHidden() const {
  WIN32_FILE_ATTRIBUTE_DATA fi;
  BOOL rc = GetFileAttributesEx(path.c_str(), GetFileExInfoStandard, &fi);
  if (rc)
    return fi.dwFileAttributes & FILE_ATTRIBUTE_HIDDEN;
  else if (GetLastError() != ERROR_FILE_NOT_FOUND)
    ThrowError("isHidden(): " + std::string(path) + ": Can't get status: ");
  return false;
}

bool
Path::isRootDirectory() const {
  size_t len = path.size();
  return len > 0 && path[len-1] == '/';
}

std::string
Path::getBasename() const {
  // Find the last slash
  size_t slash = path.rfind('/');
  if (slash == std::string::npos)
    slash = 0;
  else
    slash++;

  size_t dot = path.rfind('.');
  if (dot == std::string::npos || dot < slash)
    return path.substr(slash);
  else
    return path.substr(slash, dot - slash);
}

bool Path::hasMagicNumber(const std::string &Magic) const {
  std::string actualMagic;
  if (getMagicNumber(actualMagic, Magic.size()))
    return Magic == actualMagic;
  return false;
}

bool
Path::isBytecodeFile() const {
  if (!isFile())
    return false;
  std::string actualMagic;
  if (!getMagicNumber(actualMagic, 4))
    return false;
  return actualMagic == "llvc" || actualMagic == "llvm";
}

bool
Path::exists() const {
  DWORD attr = GetFileAttributes(path.c_str());
  return attr != INVALID_FILE_ATTRIBUTES;
}

bool
Path::canRead() const {
  // FIXME: take security attributes into account.
  DWORD attr = GetFileAttributes(path.c_str());
  return attr != INVALID_FILE_ATTRIBUTES;
}

bool
Path::canWrite() const {
  // FIXME: take security attributes into account.
  DWORD attr = GetFileAttributes(path.c_str());
  return (attr != INVALID_FILE_ATTRIBUTES) && !(attr & FILE_ATTRIBUTE_READONLY);
}

bool
Path::canExecute() const {
  // FIXME: take security attributes into account.
  DWORD attr = GetFileAttributes(path.c_str());
  return attr != INVALID_FILE_ATTRIBUTES;
}

std::string
Path::getLast() const {
  // Find the last slash
  size_t pos = path.rfind('/');

  // Handle the corner cases
  if (pos == std::string::npos)
    return path;

  // If the last character is a slash, we have a root directory
  if (pos == path.length()-1)
    return path;

  // Return everything after the last slash
  return path.substr(pos+1);
}

bool
Path::getStatusInfo(FileStatus &info, std::string *ErrStr) const {
  WIN32_FILE_ATTRIBUTE_DATA fi;
  if (!GetFileAttributesEx(path.c_str(), GetFileExInfoStandard, &fi))
    return GetError("getStatusInfo():" + std::string(path) +
                    ": Can't get status: ", ErrStr);

  info.fileSize = fi.nFileSizeHigh;
  info.fileSize <<= sizeof(fi.nFileSizeHigh)*8;
  info.fileSize += fi.nFileSizeLow;

  info.mode = fi.dwFileAttributes & FILE_ATTRIBUTE_READONLY ? 0555 : 0777;
  info.user = 9999;    // Not applicable to Windows, so...
  info.group = 9999;   // Not applicable to Windows, so...

  __int64 ft = *reinterpret_cast<__int64*>(&fi.ftLastWriteTime);
  info.modTime.fromWin32Time(ft);

  info.isDir = fi.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY;
  return false;
}

static bool AddPermissionBits(const std::string& Filename, int bits) {
  DWORD attr = GetFileAttributes(Filename.c_str());

  // If it doesn't exist, we're done.
  if (attr == INVALID_FILE_ATTRIBUTES)
    return false;

  // The best we can do to interpret Unix permission bits is to use
  // the owner writable bit.
  if ((attr & FILE_ATTRIBUTE_READONLY) && (bits & 0200)) {
    if (!SetFileAttributes(Filename.c_str(), attr & ~FILE_ATTRIBUTE_READONLY))
      ThrowError(Filename + ": SetFileAttributes: ");
  }
  return true;
}

void Path::makeReadableOnDisk() {
  // All files are readable on Windows (ignoring security attributes).
}

void Path::makeWriteableOnDisk() {
  DWORD attr = GetFileAttributes(path.c_str());

  // If it doesn't exist, we're done.
  if (attr == INVALID_FILE_ATTRIBUTES)
    return;

  if (attr & FILE_ATTRIBUTE_READONLY) {
    if (!SetFileAttributes(path.c_str(), attr & ~FILE_ATTRIBUTE_READONLY))
      ThrowError(std::string(path) + ": Can't make file writable: ");
  }
}

void Path::makeExecutableOnDisk() {
  // All files are executable on Windows (ignoring security attributes).
}

bool
Path::getDirectoryContents(std::set<Path>& result) const {
  if (!isDirectory())
    return false;

  result.clear();
  WIN32_FIND_DATA fd;
  std::string searchpath = path;
  if (path.size() == 0 || searchpath[path.size()-1] == '/')
    searchpath += "*";
  else
    searchpath += "/*";

  HANDLE h = FindFirstFile(searchpath.c_str(), &fd);
  if (h == INVALID_HANDLE_VALUE) {
    if (GetLastError() == ERROR_FILE_NOT_FOUND)
      return true; // not really an error, now is it?
    ThrowError(path + ": Can't read directory: ");
  }

  do {
    if (fd.cFileName[0] == '.')
      continue;
    Path aPath(path);
    aPath.appendComponent(&fd.cFileName[0]);
    result.insert(aPath);
  } while (FindNextFile(h, &fd));

  DWORD err = GetLastError();
  FindClose(h);
  if (err != ERROR_NO_MORE_FILES) {
    SetLastError(err);
    ThrowError(path + ": Can't read directory: ");
  }
  return true;
}

bool
Path::set(const std::string& a_path) {
  if (a_path.size() == 0)
    return false;
  std::string save(path);
  path = a_path;
  FlipBackSlashes(path);
  if (!isValid()) {
    path = save;
    return false;
  }
  return true;
}

bool
Path::appendComponent(const std::string& name) {
  if (name.empty())
    return false;
  std::string save(path);
  if (!path.empty()) {
    size_t last = path.size() - 1;
    if (path[last] != '/')
      path += '/';
  }
  path += name;
  if (!isValid()) {
    path = save;
    return false;
  }
  return true;
}

bool
Path::eraseComponent() {
  size_t slashpos = path.rfind('/',path.size());
  if (slashpos == path.size() - 1 || slashpos == std::string::npos)
    return false;
  std::string save(path);
  path.erase(slashpos);
  if (!isValid()) {
    path = save;
    return false;
  }
  return true;
}

bool
Path::appendSuffix(const std::string& suffix) {
  std::string save(path);
  path.append(".");
  path.append(suffix);
  if (!isValid()) {
    path = save;
    return false;
  }
  return true;
}

bool
Path::eraseSuffix() {
  size_t dotpos = path.rfind('.',path.size());
  size_t slashpos = path.rfind('/',path.size());
  if (dotpos != std::string::npos) {
    if (slashpos == std::string::npos || dotpos > slashpos+1) {
      std::string save(path);
      path.erase(dotpos, path.size()-dotpos);
      if (!isValid()) {
        path = save;
        return false;
      }
      return true;
    }
  }
  return false;
}

bool
Path::createDirectoryOnDisk(bool create_parents) {
  // Get a writeable copy of the path name
  size_t len = path.length();
  char *pathname = reinterpret_cast<char *>(_alloca(len+2));
  path.copy(pathname, len);
  pathname[len] = 0;

  // Make sure it ends with a slash.
  if (len == 0 || pathname[len - 1] != '/') {
    pathname[len] = '/';
    pathname[++len] = 0;
  }

  // Determine starting point for initial / search.
  char *next = pathname;
  if (pathname[0] == '/' && pathname[1] == '/') {
    // Skip host name.
    next = strchr(pathname+2, '/');
    if (next == NULL)
      throw std::string(pathname) + ": badly formed remote directory";
    // Skip share name.
    next = strchr(next+1, '/');
    if (next == NULL)
      throw std::string(pathname) + ": badly formed remote directory";
    next++;
    if (*next == 0)
      throw std::string(pathname) + ": badly formed remote directory";
  } else {
    if (pathname[1] == ':')
      next += 2;    // skip drive letter
    if (*next == '/')
      next++;       // skip root directory
  }

  // If we're supposed to create intermediate directories
  if (create_parents) {
    // Loop through the directory components until we're done
    while (*next) {
      next = strchr(next, '/');
      *next = 0;
      if (!CreateDirectory(pathname, NULL))
          ThrowError(std::string(pathname) + ": Can't create directory: ");
      *next++ = '/';
    }
  } else {
    // Drop trailing slash.
    pathname[len-1] = 0;
    if (!CreateDirectory(pathname, NULL)) {
      ThrowError(std::string(pathname) + ": Can't create directory: ");
    }
  }
  return true;
}

bool
Path::createFileOnDisk() {
  // Create the file
  HANDLE h = CreateFile(path.c_str(), GENERIC_WRITE, 0, NULL, CREATE_NEW,
                        FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE)
    ThrowError(path + ": Can't create file: ");

  CloseHandle(h);
  return true;
}

bool
Path::eraseFromDisk(bool remove_contents, std::string *ErrStr) const {
  if (isFile()) {
    DWORD attr = GetFileAttributes(path.c_str());

    // If it doesn't exist, we're done.
    if (attr == INVALID_FILE_ATTRIBUTES)
      return false;

    // Read-only files cannot be deleted on Windows.  Must remove the read-only
    // attribute first.
    if (attr & FILE_ATTRIBUTE_READONLY) {
      if (!SetFileAttributes(path.c_str(), attr & ~FILE_ATTRIBUTE_READONLY))
        return GetError(path + ": Can't destroy file: ", ErrStr);
    }

    if (!DeleteFile(path.c_str()))
      ThrowError(path + ": Can't destroy file: ");
    return true;
  } else if (isDirectory()) {
    // If it doesn't exist, we're done.
    if (!exists())
      return false;

    char *pathname = reinterpret_cast<char *>(_alloca(path.length()+3));
    int lastchar = path.length() - 1 ;
    path.copy(pathname, lastchar+1);

    // Make path end with '/*'.
    if (pathname[lastchar] != '/')
      pathname[++lastchar] = '/';
    pathname[lastchar+1] = '*';
    pathname[lastchar+2] = 0;

    if (remove_contents) {
      WIN32_FIND_DATA fd;
      HANDLE h = FindFirstFile(pathname, &fd);

      // It's a bad idea to alter the contents of a directory while enumerating
      // its contents. So build a list of its contents first, then destroy them.

      if (h != INVALID_HANDLE_VALUE) {
        std::vector<Path> list;

        do {
          if (strcmp(fd.cFileName, ".") == 0)
            continue;
          if (strcmp(fd.cFileName, "..") == 0)
            continue;

          Path aPath(path);
          aPath.appendComponent(&fd.cFileName[0]);
          list.push_back(aPath);
        } while (FindNextFile(h, &fd));

        DWORD err = GetLastError();
        FindClose(h);
        if (err != ERROR_NO_MORE_FILES) {
          SetLastError(err);
          return GetError(path + ": Can't read directory: ", ErrStr);
        }

        for (std::vector<Path>::iterator I = list.begin(); I != list.end();
             ++I) {
          Path &aPath = *I;
          aPath.eraseFromDisk(true);
        }
      } else {
        if (GetLastError() != ERROR_FILE_NOT_FOUND)
          return GetError(path + ": Can't read directory: ", ErrStr);
      }
    }

    pathname[lastchar] = 0;
    if (!RemoveDirectory(pathname))
      return GetError(std::string(pathname) + ": Can't destroy directory: ",
                      ErrStr);
    return false;
  } else {
    // It appears the path doesn't exist.
    return true;
  }
}

bool Path::getMagicNumber(std::string& Magic, unsigned len) const {
  if (!isFile())
    return false;
  assert(len < 1024 && "Request for magic string too long");
  char* buf = (char*) alloca(1 + len);

  HANDLE h = CreateFile(path.c_str(),
                        GENERIC_READ,
                        FILE_SHARE_READ,
                        NULL,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL,
                        NULL);
  if (h == INVALID_HANDLE_VALUE)
    return false;

  DWORD nRead = 0;
  BOOL ret = ReadFile(h, buf, len, &nRead, NULL);
  CloseHandle(h);

  if (!ret || nRead != len)
    return false;

  buf[len] = '\0';
  Magic = buf;
  return true;
}

bool
Path::renamePathOnDisk(const Path& newName) {
  if (!MoveFileEx(path.c_str(), newName.c_str(), MOVEFILE_REPLACE_EXISTING))
    ThrowError("Can't move '" + path +
               "' to '" + newName.path + "': ");
  return true;
}

bool
Path::setStatusInfoOnDisk(const FileStatus &si) const {
  // FIXME: should work on directories also.
  if (!isFile()) return false;

  HANDLE h = CreateFile(path.c_str(),
                        FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES,
                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                        NULL,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL,
                        NULL);
  if (h == INVALID_HANDLE_VALUE)
    return false;

  BY_HANDLE_FILE_INFORMATION bhfi;
  if (!GetFileInformationByHandle(h, &bhfi)) {
    DWORD err = GetLastError();
    CloseHandle(h);
    SetLastError(err);
    ThrowError(path + ": GetFileInformationByHandle: ");
  }

  FILETIME ft;
  (uint64_t&)ft = si.modTime.toWin32Time();
  BOOL ret = SetFileTime(h, NULL, &ft, &ft);
  DWORD err = GetLastError();
  CloseHandle(h);
  if (!ret) {
    SetLastError(err);
    ThrowError(path + ": SetFileTime: ");
  }

  // Best we can do with Unix permission bits is to interpret the owner
  // writable bit.
  if (si.mode & 0200) {
    if (bhfi.dwFileAttributes & FILE_ATTRIBUTE_READONLY) {
      if (!SetFileAttributes(path.c_str(),
              bhfi.dwFileAttributes & ~FILE_ATTRIBUTE_READONLY))
        ThrowError(path + ": SetFileAttributes: ");
    }
  } else {
    if (!(bhfi.dwFileAttributes & FILE_ATTRIBUTE_READONLY)) {
      if (!SetFileAttributes(path.c_str(),
              bhfi.dwFileAttributes | FILE_ATTRIBUTE_READONLY))
        ThrowError(path + ": SetFileAttributes: ");
    }
  }

  return true;
}

void
CopyFile(const sys::Path &Dest, const sys::Path &Src) {
  // Can't use CopyFile macro defined in Windows.h because it would mess up the
  // above line.  We use the expansion it would have in a non-UNICODE build.
  if (!::CopyFileA(Src.c_str(), Dest.c_str(), false))
    ThrowError("Can't copy '" + Src.toString() +
               "' to '" + Dest.toString() + "': ");
}

void
Path::makeUnique(bool reuse_current) {
  if (reuse_current && !exists())
    return; // File doesn't exist already, just use it!

  // Reserve space for -XXXXXX at the end.
  char *FNBuffer = (char*) alloca(path.size()+8);
  unsigned offset = path.size();
  path.copy(FNBuffer, offset);

  // Find a numeric suffix that isn't used by an existing file.  Assume there
  // won't be more than 1 million files with the same prefix.  Probably a safe
  // bet.
  static unsigned FCounter = 0;
  do {
    sprintf(FNBuffer+offset, "-%06u", FCounter);
    if (++FCounter > 999999)
      FCounter = 0;
    path = FNBuffer;
  } while (exists());
}

bool
Path::createTemporaryFileOnDisk(bool reuse_current) {
  // Make this into a unique file name
  makeUnique(reuse_current);

  // Now go and create it
  HANDLE h = CreateFile(path.c_str(), GENERIC_WRITE, 0, NULL, CREATE_NEW,
                        FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE)
    return false;

  CloseHandle(h);
  return true;
}

}
}
