/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* A commandline tool for compressing ttf format files to woff2. */

#include <string>

#include "file.h"
#include <woff2/encode.h>


fn main(argc: int, argv: char**) -> int {
  using std::string;

  if (argc != 2) {
    fprintf(stderr, "One argument, the input filename, must be provided.\n");
    return 1;
  }

  var filename: string;
  var outfilename: string;
  fprintf(stdout, "Processing %s => %s\n",
    filename.c_str(), outfilename.c_str());
  var input: string;

  var input_data: const auto*;
  var output_size: size_t;
  var output: string;
  var output_data: auto*;

  var params: woff2::WOFF2Params;
  if (!woff2::ConvertTTFToWOFF2(input_data, input.size(),
                                output_data, &output_size, params)) {
    fprintf(stderr, "Compression failed.\n");
    return 1;
  }
  output.resize(output_size);

  woff2::SetFileContents(outfilename, output.begin(), output.end());

  return 0;
}
