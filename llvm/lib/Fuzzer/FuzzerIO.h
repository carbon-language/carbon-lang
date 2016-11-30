//===- FuzzerIO.h - Internal header for IO utils ----------------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// IO interface.
//===----------------------------------------------------------------------===//
#ifndef LLVM_FUZZER_IO_H
#define LLVM_FUZZER_IO_H

#include "FuzzerDefs.h"

namespace fuzzer {

bool IsFile(const std::string &Path);

long GetEpoch(const std::string &Path);

Unit FileToVector(const std::string &Path, size_t MaxSize = 0,
                  bool ExitOnError = true);

void DeleteFile(const std::string &Path);

std::string FileToString(const std::string &Path);

void CopyFileToErr(const std::string &Path);

void WriteToFile(const Unit &U, const std::string &Path);

void ReadDirToVectorOfUnits(const char *Path, std::vector<Unit> *V,
                            long *Epoch, size_t MaxSize, bool ExitOnError);

// Returns "Dir/FileName" or equivalent for the current OS.
std::string DirPlusFile(const std::string &DirPath,
                        const std::string &FileName);

void DupAndCloseStderr();

void CloseStdout();

void Printf(const char *Fmt, ...);

}  // namespace fuzzer
#endif  // LLVM_FUZZER_IO_H
