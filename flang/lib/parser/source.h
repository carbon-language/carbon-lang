// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_PARSER_SOURCE_H_
#define FORTRAN_PARSER_SOURCE_H_

// Source file content is lightly normalized when the file is read.
//  - Line ending markers are converted to single newline characters
//  - A newline character is added to the last line of the file if one is needed

#include <cstddef>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace Fortran::parser {

std::string DirectoryName(std::string path);
std::string LocateSourceFile(
    std::string name, const std::vector<std::string> &searchPath);

class SourceFile {
public:
  SourceFile() {}
  ~SourceFile();
  std::string path() const { return path_; }
  const char *content() const { return content_; }
  std::size_t bytes() const { return bytes_; }
  std::size_t lines() const { return lineStart_.size(); }

  bool Open(std::string path, std::stringstream *error);
  bool ReadStandardInput(std::stringstream *error);
  void Close();
  std::pair<int, int> FindOffsetLineAndColumn(std::size_t) const;
  std::size_t GetLineStartOffset(int lineNumber) const {
    return lineStart_.at(lineNumber - 1);
  }

private:
  bool ReadFile(std::string errorPath, std::stringstream *error);

  std::string path_;
  int fileDescriptor_{-1};
  bool isMemoryMapped_{false};
  const char *content_{nullptr};
  std::size_t bytes_{0};
  std::vector<std::size_t> lineStart_;
  std::string normalized_;
};

}  // namespace Fortran::parser
#endif  // FORTRAN_PARSER_SOURCE_H_
