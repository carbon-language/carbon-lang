#ifndef FORTRAN_SOURCE_H_
#define FORTRAN_SOURCE_H_

// Source file content is lightly normalized when the file is read.
//  - Line ending markers are converted to single newline characters
//  - A newline character is added to the last line of the file if one is needed

#include <sstream>
#include <string>

namespace Fortran {

class SourceFile {
 public:
  SourceFile() {}
  ~SourceFile();
  bool Open(std::string path, std::stringstream *error);
  void Close();
  std::string path() const { return path_; }
  const char *content() const { return content_; }
  size_t bytes() const { return bytes_; }
 private:
  std::string path_;
  int fileDescriptor_{-1};
  bool isMemoryMapped_{false};
  const char *content_{nullptr};
  size_t bytes_{0};
};
}  // namespace Fortran
#endif  // FORTRAN_SOURCE_H_
