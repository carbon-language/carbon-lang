//===- FuzzerIO.cpp - IO utils. -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// IO functions.
//===----------------------------------------------------------------------===//
#include "FuzzerInternal.h"
#include <fstream>
#include <dirent.h>
namespace fuzzer {

static std::vector<std::string> ListFilesInDir(const std::string &Dir) {
  std::vector<std::string> V;
  DIR *D = opendir(Dir.c_str());
  if (!D) return V;
  while (auto E = readdir(D)) {
    if (E->d_type == DT_REG || E->d_type == DT_LNK)
      V.push_back(E->d_name);
  }
  closedir(D);
  return V;
}

Unit FileToVector(const std::string &Path) {
  std::ifstream T(Path);
  return Unit((std::istreambuf_iterator<char>(T)),
              std::istreambuf_iterator<char>());
}

void WriteToFile(const Unit &U, const std::string &Path) {
  std::ofstream OF(Path);
  OF.write((const char*)U.data(), U.size());
}

void ReadDirToVectorOfUnits(const char *Path, std::vector<Unit> *V) {
  for (auto &X : ListFilesInDir(Path))
    V->push_back(FileToVector(DirPlusFile(Path, X)));
}

std::string DirPlusFile(const std::string &DirPath,
                        const std::string &FileName) {
  return DirPath + "/" + FileName;
}

}  // namespace fuzzer
