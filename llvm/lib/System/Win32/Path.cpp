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
#include <fstream>
#include <malloc.h>

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
  if (pos != std::string::npos) {
    if (pos != 1 || !isalpha(path[0]) || len < 3)
      return false;
  }

  // Check for illegal characters.
  if (path.find_first_of("\\<>\"|\001\002\003\004\005\006\007\010\011\012"
                         "\013\014\015\016\017\020\021\022\023\024\025\026"
                         "\027\030\031\032\033\034\035\036\037")
      != std::string::npos)
    return false;

  // A file or directory name may not end in a period.
  if (path[len-1] == '.')
    return false;
  if (len >= 2 && path[len-2] == '.' && path[len-1] == '/')
    return false;

  // A file or directory name may not end in a space.
  if (path[len-1] == ' ')
    return false;
  if (len >= 2 && path[len-2] == ' ' && path[len-1] == '/')
    return false;

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
  result.setDirectory(pathname);

  // Append a subdirectory passed on our process id so multiple LLVMs don't
  // step on each other's toes.
  sprintf(pathname, "LLVM_%u", GetCurrentProcessId());
  result.appendDirectory(pathname);

  // If there's a directory left over from a previous LLVM execution that
  // happened to have the same process id, get rid of it.
  result.destroyDirectory(true);

  // And finally (re-)create the empty directory.
  result.createDirectory(false);
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
  result.setDirectory("/");
  return result;
}

static void getPathList(const char*path, std::vector<sys::Path>& Paths) {
  const char* at = path;
  const char* delim = strchr(at, ';');
  Path tmpPath;
  while( delim != 0 ) {
    std::string tmp(at, size_t(delim-at));
    if (tmpPath.setDirectory(tmp))
      if (tmpPath.readable())
        Paths.push_back(tmpPath);
    at = delim + 1;
    delim = strchr(at, ';');
  }
  if (*at != 0)
    if (tmpPath.setDirectory(std::string(at)))
      if (tmpPath.readable())
        Paths.push_back(tmpPath);

}

void 
Path::GetSystemLibraryPaths(std::vector<sys::Path>& Paths) {
#ifdef LTDL_SHLIBPATH_VAR
  char* env_var = getenv(LTDL_SHLIBPATH_VAR);
  if (env_var != 0) {
    getPathList(env_var,Paths);
  }
#endif
  // FIXME: Should this look at LD_LIBRARY_PATH too?
  Paths.push_back(sys::Path("C:\\WINDOWS\\SYSTEM32\\"));
  Paths.push_back(sys::Path("C:\\WINDOWS\\"));
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
    if (tmpPath.setDirectory(LLVM_LIBDIR))
      if (tmpPath.readable())
        Paths.push_back(tmpPath);
  }
#endif
  GetSystemLibraryPaths(Paths);
}

Path
Path::GetLLVMDefaultConfigDir() {
  return Path("/etc/llvm/");
}

Path
Path::GetUserHomeDirectory() {
  const char* home = getenv("HOME");
  if (home) {
    Path result;
    if (result.setDirectory(home))
      return result;
  }
  return GetRootDirectory();
}
// FIXME: the above set of functions don't map to Windows very well.

bool
Path::isFile() const {
  return (isValid() && path[path.length()-1] != '/');
}

bool
Path::isDirectory() const {
  return (isValid() && path[path.length()-1] == '/');
}

std::string
Path::getBasename() const {
  // Find the last slash
  size_t slash = path.rfind('/');
  if (slash == std::string::npos)
    slash = 0;
  else
    slash++;

  return path.substr(slash, path.rfind('.'));
}

bool Path::hasMagicNumber(const std::string &Magic) const {
  size_t len = Magic.size();
  char *buf = reinterpret_cast<char *>(_alloca(len+1));
  std::ifstream f(path.c_str());
  f.read(buf, len);
  buf[len] = '\0';
  return Magic == buf;
}

bool 
Path::isBytecodeFile() const {
  char buffer[ 4];
  buffer[0] = 0;
  std::ifstream f(path.c_str());
  f.read(buffer, 4);
  if (f.bad())
    ThrowErrno("can't read file signature");
  return 0 == memcmp(buffer,"llvc",4) || 0 == memcmp(buffer,"llvm",4);
}

bool
Path::exists() const {
  DWORD attr = GetFileAttributes(path.c_str());
  return attr != INVALID_FILE_ATTRIBUTES;
}

bool
Path::readable() const {
  // FIXME: take security attributes into account.
  DWORD attr = GetFileAttributes(path.c_str());
  return attr != INVALID_FILE_ATTRIBUTES;
}

bool
Path::writable() const {
  // FIXME: take security attributes into account.
  DWORD attr = GetFileAttributes(path.c_str());
  return (attr != INVALID_FILE_ATTRIBUTES) && !(attr & FILE_ATTRIBUTE_READONLY);
}

bool
Path::executable() const {
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

  // If the last character is a slash
  if (pos == path.length()-1) {
    // Find the second to last slash
    size_t pos2 = path.rfind('/', pos-1);
    if (pos2 == std::string::npos)
      return path.substr(0,pos);
    else
      return path.substr(pos2+1,pos-pos2-1);
  }
  // Return everything after the last slash
  return path.substr(pos+1);
}

void
Path::getStatusInfo(StatusInfo& info) const {
  WIN32_FILE_ATTRIBUTE_DATA fi;
  if (!GetFileAttributesEx(path.c_str(), GetFileExInfoStandard, &fi))
    ThrowError(std::string(path) + ": Can't get status: ");

  info.fileSize = fi.nFileSizeHigh;
  info.fileSize <<= 32;
  info.fileSize += fi.nFileSizeLow;

  info.mode = 0777;    // Not applicable to Windows, so...
  info.user = 9999;    // Not applicable to Windows, so...
  info.group = 9999;   // Not applicable to Windows, so...

  __int64 ft = *reinterpret_cast<__int64*>(&fi.ftLastWriteTime);
  info.modTime.fromWin32Time(ft);

  info.isDir = fi.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY;
  if (info.isDir && path[path.length() - 1] != '/')
    path += '/';
  else if (!info.isDir && path[path.length() - 1] == '/')
    path.erase(path.length() - 1);
}

void Path::makeReadable() {
  // All files are readable on Windows (ignoring security attributes).
}

void Path::makeWriteable() {
  DWORD attr = GetFileAttributes(path.c_str());

  // If it doesn't exist, we're done.
  if (attr == INVALID_FILE_ATTRIBUTES)
    return;

  if (attr & FILE_ATTRIBUTE_READONLY) {
    if (!SetFileAttributes(path.c_str(), attr & ~FILE_ATTRIBUTE_READONLY))
      ThrowError(std::string(path) + ": Can't make file writable: ");
  }
}

void Path::makeExecutable() {
  // All files are executable on Windows (ignoring security attributes).
}

bool
Path::setDirectory(const std::string& a_path) {
  if (a_path.size() == 0)
    return false;
  Path save(*this);
  path = a_path;
  FlipBackSlashes(path);
  size_t last = a_path.size() -1;
  if (a_path[last] != '/')
    path += '/';
  if (!isValid()) {
    path = save.path;
    return false;
  }
  return true;
}

bool
Path::setFile(const std::string& a_path) {
  if (a_path.size() == 0)
    return false;
  Path save(*this);
  path = a_path;
  FlipBackSlashes(path);
  size_t last = a_path.size() - 1;
  while (last > 0 && a_path[last] == '/')
    last--;
  path.erase(last+1);
  if (!isValid()) {
    path = save.path;
    return false;
  }
  return true;
}

bool
Path::appendDirectory(const std::string& dir) {
  if (isFile())
    return false;
  Path save(*this);
  path += dir;
  path += "/";
  if (!isValid()) {
    path = save.path;
    return false;
  }
  return true;
}

bool
Path::elideDirectory() {
  if (isFile())
    return false;
  size_t slashpos = path.rfind('/',path.size());
  if (slashpos == 0 || slashpos == std::string::npos)
    return false;
  if (slashpos == path.size() - 1)
    slashpos = path.rfind('/',slashpos-1);
  if (slashpos == std::string::npos)
    return false;
  path.erase(slashpos);
  return true;
}

bool
Path::appendFile(const std::string& file) {
  if (!isDirectory())
    return false;
  Path save(*this);
  path += file;
  if (!isValid()) {
    path = save.path;
    return false;
  }
  return true;
}

bool
Path::elideFile() {
  if (isDirectory())
    return false;
  size_t slashpos = path.rfind('/',path.size());
  if (slashpos == std::string::npos)
    return false;
  path.erase(slashpos+1);
  return true;
}

bool
Path::appendSuffix(const std::string& suffix) {
  if (isDirectory())
    return false;
  Path save(*this);
  path.append(".");
  path.append(suffix);
  if (!isValid()) {
    path = save.path;
    return false;
  }
  return true;
}

bool
Path::elideSuffix() {
  if (isDirectory()) return false;
  size_t dotpos = path.rfind('.',path.size());
  size_t slashpos = path.rfind('/',path.size());
  if (slashpos != std::string::npos && dotpos != std::string::npos &&
      dotpos > slashpos) {
    path.erase(dotpos, path.size()-dotpos);
    return true;
  }
  return false;
}


bool
Path::createDirectory( bool create_parents) {
  // Make sure we're dealing with a directory
  if (!isDirectory()) return false;

  // Get a writeable copy of the path name
  char *pathname = reinterpret_cast<char *>(_alloca(path.length()+1));
  path.copy(pathname,path.length());
  pathname[path.length()] = 0;

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
    pathname[path.size()-1] = 0;
    if (!CreateDirectory(pathname, NULL)) {
      ThrowError(std::string(pathname) + ": Can't create directory: ");
    }
  }
  return true;
}

bool
Path::createFile() {
  // Make sure we're dealing with a file
  if (!isFile()) return false;

  // Create the file
  HANDLE h = CreateFile(path.c_str(), GENERIC_WRITE, 0, NULL, CREATE_NEW,
                        FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE)
    ThrowError(std::string(path.c_str()) + ": Can't create file: ");

  CloseHandle(h);
  return true;
}

bool
Path::destroyDirectory(bool remove_contents) const {
  // Make sure we're dealing with a directory
  if (!isDirectory()) return false;

  // If it doesn't exist, we're done.
  if (!exists()) return true;

  char *pathname = reinterpret_cast<char *>(_alloca(path.length()+1));
  path.copy(pathname,path.length()+1);
  int lastchar = path.length() - 1 ;
  if (pathname[lastchar] == '/')
    pathname[lastchar] = 0;

  if (remove_contents) {
    // Recursively descend the directory to remove its content
    // FIXME: The correct way of doing this on Windows isn't pretty...
    // but this may work if unix-like utils are present.
    std::string cmd("rm -rf ");
    cmd += path;
    system(cmd.c_str());
  } else {
    // Otherwise, try to just remove the one directory
    if (!RemoveDirectory(pathname))
      ThrowError(std::string(pathname) + ": Can't destroy directory: ");
  }
  return true;
}

bool
Path::destroyFile() const {
  if (!isFile()) return false;

  DWORD attr = GetFileAttributes(path.c_str());

  // If it doesn't exist, we're done.
  if (attr == INVALID_FILE_ATTRIBUTES)
    return true;

  // Read-only files cannot be deleted on Windows.  Must remove the read-only
  // attribute first.
  if (attr & FILE_ATTRIBUTE_READONLY) {
    if (!SetFileAttributes(path.c_str(), attr & ~FILE_ATTRIBUTE_READONLY))
      ThrowError(std::string(path.c_str()) + ": Can't destroy file: ");
  }

  if (!DeleteFile(path.c_str()))
    ThrowError(std::string(path.c_str()) + ": Can't destroy file: ");
  return true;
}

bool Path::getMagicNumber(std::string& Magic, unsigned len) const {
  if (!isFile())
    return false;
  assert(len < 1024 && "Request for magic string too long");
  char* buf = (char*) alloca(1 + len);
  std::ofstream ofs(path.c_str(),std::ofstream::in);
  if (!ofs.is_open())
    return false;
  std::ifstream ifs(path.c_str());
  if (!ifs.is_open())
    return false;
  ifs.read(buf, len);
  ofs.close();
  ifs.close();
  buf[len] = '\0';
  Magic = buf;
  return true;
}

void 
CopyFile(const sys::Path &Dest, const sys::Path &Src) {
  if (!::CopyFile(Src.c_str(), Dest.c_str(), false))
    ThrowError("Can't copy '" + Src.toString() + 
               "' to '" + Dest.toString() + "'");
}

void 
Path::makeUnique( bool reuse_current ) {
  if (reuse_current && !exists())
    return; // File doesn't exist already, just use it!

  Path dir (*this);
  dir.elideFile();
  std::string fname = this->getLast();

  char newName[MAX_PATH + 1];
  if (!GetTempFileName(dir.c_str(), fname.c_str(), 0, newName))
    ThrowError("Cannot make unique filename for '" + path + "'");

  path = newName;
}

bool
Path::createTemporaryFile(bool reuse_current) {
  // Make sure we're dealing with a file
  if (!isFile()) 
    return false;

  // Make this into a unique file name
  makeUnique( reuse_current );
  return true;
}

}
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab

