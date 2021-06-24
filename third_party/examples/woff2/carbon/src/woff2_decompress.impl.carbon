/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* A very simple commandline tool for decompressing woff2 format files to true
   type font files. */

#include <string>

#include "./file.h"
#include <woff2/decode.h>


fn main(argc: int, argv: char**) -> int {
  using std::string;

  if (argc != 2) {
    fprintf(stderr, "One argument, the input filename, must be provided.\n");
    return 1;
  }

  var filename: string;
  var outfilename: string;

  // Note: update woff2_dec_fuzzer_new_entry.cc if this pattern changes.
  var input: string;
  var raw_input: auto*;
  var output: string;
  var out: woff2::WOFF2StringOut;

  let ok: bool;

  if (ok) {
    woff2::SetFileContents(outfilename, output.begin(),
        output.begin() + out.Size());
  }
  return ok ? 0 : 1;
}
