//===- llvm/System/Unix/Path.cpp - Unix Path Implementation -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific portion of the Path class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include <llvm/Config/config.h>
#include "Unix.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>

namespace llvm {
using namespace sys;

Path::Path(std::string unverified_path) 
  : path(unverified_path)
{
  if (unverified_path.empty())
    return;
  if (this->isValid()) 
    return;
  // oops, not valid.
  path.clear();
  ThrowErrno(unverified_path + ": path is not valid");
}

Path
Path::GetRootDirectory() {
  Path result;
  result.setDirectory("/");
  return result;
}

static inline bool IsLibrary(Path& path, const std::string& basename) {
  if (path.appendFile(std::string("lib") + basename)) {
    if (path.appendSuffix(Path::GetDLLSuffix()) && path.readable())
      return true;
    else if (path.elideSuffix() && path.appendSuffix("a") && path.readable())
      return true;
    else if (path.elideSuffix() && path.appendSuffix("o") && path.readable())
      return true;
    else if (path.elideSuffix() && path.appendSuffix("bc") && path.readable())
      return true;
  } else if (path.elideFile() && path.appendFile(basename)) {
    if (path.appendSuffix(Path::GetDLLSuffix()) && path.readable())
      return true;
    else if (path.elideSuffix() && path.appendSuffix("a") && path.readable())
      return true;
    else if (path.elideSuffix() && path.appendSuffix("o") && path.readable())
      return true;
    else if (path.elideSuffix() && path.appendSuffix("bc") && path.readable())
      return true;
  }
  path.clear();
  return false;
}

Path 
Path::GetLibraryPath(const std::string& basename, 
                     const std::vector<std::string>& LibPaths) {
  Path result;

  // Try the paths provided
  for (std::vector<std::string>::const_iterator I = LibPaths.begin(),
       E = LibPaths.end(); I != E; ++I ) {
    if (result.setDirectory(*I) && IsLibrary(result,basename))
      return result;
  }

  // Try the LLVM lib directory in the LLVM install area
  if (result.setDirectory(LLVM_LIBDIR) && IsLibrary(result,basename))
    return result;

  // Try /usr/lib
  if (result.setDirectory("/usr/lib/") && IsLibrary(result,basename))
    return result;

  // Try /lib
  if (result.setDirectory("/lib/") && IsLibrary(result,basename))
    return result;

  // Can't find it, give up and return invalid path.
  result.clear();
  return result;
}

Path 
Path::GetSystemLibraryPath1() {
  return Path("/lib/");
}

Path 
Path::GetSystemLibraryPath2() {
  return Path("/usr/lib/");
}

Path 
Path::GetLLVMDefaultConfigDir() {
  return Path("/etc/llvm/");
}

Path 
Path::GetLLVMConfigDir() {
  Path result;
  if (result.setDirectory(LLVM_ETCDIR))
    return result;
  return GetLLVMDefaultConfigDir();
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
  char buf[ 1 + len];
  std::ifstream f(path.c_str());
  f.read(buf, len);
  buf[len] = '\0';
  return Magic == buf;
}

bool 
Path::isBytecodeFile() const {
  if (readable()) {
    return hasMagicNumber("llvm");
  }
  return false;
}

bool
Path::isArchive() const {
  if (readable()) {
    return hasMagicNumber("!<arch>\012");
  }
  return false;
}

bool
Path::exists() const {
  return 0 == access(path.c_str(), F_OK );
}

bool
Path::readable() const {
  return 0 == access(path.c_str(), F_OK | R_OK );
}

bool
Path::writable() const {
  return 0 == access(path.c_str(), F_OK | W_OK );
}

bool
Path::executable() const {
  return 0 == access(path.c_str(), R_OK | X_OK );
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

bool
Path::setDirectory(const std::string& a_path) {
  if (a_path.size() == 0)
    return false;
  Path save(*this);
  path = a_path;
  size_t last = a_path.size() -1;
  if (last != 0 && a_path[last] != '/')
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
  char pathname[MAXPATHLEN];
  path.copy(pathname,MAXPATHLEN);

  // Null-terminate the last component
  int lastchar = path.length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;

  // If we're supposed to create intermediate directories
  if ( create_parents ) {
    // Find the end of the initial name component
    char * next = strchr(pathname,'/');
    if ( pathname[0] == '/') 
      next = strchr(&pathname[1],'/');

    // Loop through the directory components until we're done 
    while ( next != 0 ) {
      *next = 0;
      if (0 != access(pathname, F_OK | R_OK | W_OK))
        if (0 != mkdir(pathname, S_IRWXU | S_IRWXG))
          ThrowErrno(std::string(pathname) + ": Can't create directory");
      char* save = next;
      next = strchr(pathname,'/');
      *save = '/';
    }
  } else if (0 != mkdir(pathname, S_IRWXU | S_IRWXG)) {
    ThrowErrno(std::string(pathname) + ": Can't create directory");
  } 
  return true;
}

bool
Path::createFile() {
  // Make sure we're dealing with a file
  if (!isFile()) return false; 

  // Create the file
  int fd = ::creat(path.c_str(), S_IRUSR | S_IWUSR);
  if (fd < 0)
    ThrowErrno(std::string(path.c_str()) + ": Can't create file");
  ::close(fd);

  return true;
}

bool
Path::destroyDirectory(bool remove_contents) {
  // Make sure we're dealing with a directory
  if (!isDirectory()) return false;

  // If it doesn't exist, we're done.
  if (!exists()) return true;

  if (remove_contents) {
    // Recursively descend the directory to remove its content
    std::string cmd("/bin/rm -rf ");
    cmd += path;
    system(cmd.c_str());
  } else {
    // Otherwise, try to just remove the one directory
    char pathname[MAXPATHLEN];
    path.copy(pathname,MAXPATHLEN);
    int lastchar = path.length() - 1 ; 
    if (pathname[lastchar] == '/') 
      pathname[lastchar] = 0;
    if ( 0 != rmdir(pathname))
      ThrowErrno(std::string(pathname) + ": Can't destroy directory");
  }
  return true;
}

bool
Path::destroyFile() {
  if (!isFile()) return false;
  if (0 != unlink(path.c_str()))
    ThrowErrno(std::string(path.c_str()) + ": Can't destroy file");
  return true;
}

}

// vim: sw=2
