//===- Support/FileUtilities.h - File System Utilities ----------*- C++ -*-===//
//
// This file defines a family of utility functions which are useful for doing
// various things with files.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_FILEUTILITIES_H
#define SUPPORT_FILEUTILITIES_H

#include <string>

/// DiffFiles - Compare the two files specified, returning true if they are
/// different or if there is a file error.  If you specify a string to fill in
/// for the error option, it will set the string to an error message if an error
/// occurs, allowing the caller to distinguish between a failed diff and a file
/// system error.
///
bool DiffFiles(const std::string &FileA, const std::string &FileB,
               std::string *Error = 0);


/// MoveFileOverIfUpdated - If the file specified by New is different than Old,
/// or if Old does not exist, move the New file over the Old file.  Otherwise,
/// remove the New file.
///
void MoveFileOverIfUpdated(const std::string &New, const std::string &Old);
 
/// removeFile - Delete the specified file
///
void removeFile(const std::string &Filename);

/// getUniqueFilename - Return a filename with the specified prefix.  If the
/// file does not exist yet, return it, otherwise add a suffix to make it
/// unique.
///
std::string getUniqueFilename(const std::string &FilenameBase);

///
/// Method: MakeFileExecutable()
///
/// Description:
///	This method turns on whatever access attributes are needed to make the
///	specified file executable.
///
/// Return value:
///	True  - The operation succeeded.
///	False - The operation failed.
///
/// Notes:
///	In case of failure, the file's access attributes are unspecified.
///
bool MakeFileExecutable (const std::string & Filename);
#endif
