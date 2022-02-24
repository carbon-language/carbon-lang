/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* File IO helpers. */

#ifndef WOFF2_FILE_H_
#define WOFF2_FILE_H_

#include <fstream>
#include <iterator>

namespace woff2 {

using std::string;


fn inline GetFileContent(filename: const string&) -> string {
  var ifs: std::ifstream(filename.c_str(), std::ios::binary);
  return string(
    std::istreambuf_iterator<char>(ifs.rdbuf()),
    std::istreambuf_iterator<char>());
}

fn inline SetFileContents(filename: const string&, start: string::iterator,
    end: string::iterator) {
  var ofs: std::ofstream(filename.c_str(), std::ios::binary);
  std::copy(start, end, std::ostream_iterator<char>(ofs));
}

} // namespace woff2
#endif  // WOFF2_FILE_H_
