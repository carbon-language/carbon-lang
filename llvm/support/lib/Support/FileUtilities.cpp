//===- Support/FileUtilities.cpp - File System Utilities ------------------===//
//
// This file implements a family of utility functions which are useful for doing
// various things with files.
//
//===----------------------------------------------------------------------===//

#include "Support/FileUtilities.h"
#include "Config/unistd.h"
#include "Config/sys/stat.h"
#include "Config/sys/types.h"
#include <fstream>
#include <iostream>
#include <cstdio>

/// DiffFiles - Compare the two files specified, returning true if they are
/// different or if there is a file error.  If you specify a string to fill in
/// for the error option, it will set the string to an error message if an error
/// occurs, allowing the caller to distinguish between a failed diff and a file
/// system error.
///
bool DiffFiles(const std::string &FileA, const std::string &FileB,
               std::string *Error) {
  std::ifstream FileAStream(FileA.c_str());
  if (!FileAStream) {
    if (Error) *Error = "Couldn't open file '" + FileA + "'";
    return true;
  }

  std::ifstream FileBStream(FileB.c_str());
  if (!FileBStream) {
    if (Error) *Error = "Couldn't open file '" + FileB + "'";
    return true;
  }

  // Compare the two files...
  int C1, C2;
  do {
    C1 = FileAStream.get();
    C2 = FileBStream.get();
    if (C1 != C2) return true;
  } while (C1 != EOF);

  return false;
}


/// MoveFileOverIfUpdated - If the file specified by New is different than Old,
/// or if Old does not exist, move the New file over the Old file.  Otherwise,
/// remove the New file.
///
void MoveFileOverIfUpdated(const std::string &New, const std::string &Old) {
  if (DiffFiles(New, Old)) {
    if (std::rename(New.c_str(), Old.c_str()))
      std::cerr << "Error renaming '" << New << "' to '" << Old << "'!\n";
  } else {
    std::remove(New.c_str());
  }  
}

/// removeFile - Delete the specified file
///
void removeFile(const std::string &Filename) {
  std::remove(Filename.c_str());
}

/// getUniqueFilename - Return a filename with the specified prefix.  If the
/// file does not exist yet, return it, otherwise add a suffix to make it
/// unique.
///
std::string getUniqueFilename(const std::string &FilenameBase) {
  if (!std::ifstream(FilenameBase.c_str()))
    return FilenameBase;    // Couldn't open the file? Use it!

  // Create a pattern for mkstemp...
  char *FNBuffer = new char[FilenameBase.size()+8];
  strcpy(FNBuffer, FilenameBase.c_str());
  strcpy(FNBuffer+FilenameBase.size(), "-XXXXXX");

  // Agree on a temporary file name to use....
  int TempFD;
  if ((TempFD = mkstemp(FNBuffer)) == -1) {
    std::cerr << "bugpoint: ERROR: Cannot create temporary file in the current "
	      << " directory!\n";
    exit(1);
  }

  // We don't need to hold the temp file descriptor... we will trust that noone
  // will overwrite/delete the file while we are working on it...
  close(TempFD);
  std::string Result(FNBuffer);
  delete[] FNBuffer;
  return Result;
}

///
/// Method: MakeFileExecutable ()
///
/// Description:
///	This method makes the specified filename executable by giving it
///	execute permission.
///
///	For the UNIX version of this method, we turn on all of the read and
///	execute bits and then turn off anything specified in the umask.  This
///	should help ensure that access to the file remains at the level that
///	the user desires.
///
bool
MakeFileExecutable (const std::string & Filename)
{
  // Permissions masking value of the user
  mode_t mask;

  // Permissions currently enabled on the file
  struct stat fstat;

  //
  // Grab the umask value from the operating system.  We want to use it when
  // changing the file's permissions.
  //
  // Note:
  //  Umask() is one of those annoying system calls.  You have to call it
  //  to get the current value and then set it back.
  //
  mask = umask (0x777);
  umask (mask);

  //
  // Go fetch the file's current permission bits.  We want to *add* execute
  // access to the file.
  //
  if ((stat (Filename.c_str(), &fstat)) == -1)
  {
    return false;
  }

  // Make the script executable...
  chmod(Filename.c_str(), (fstat.st_mode | (0111 & ~mask)));

  return true;
}

