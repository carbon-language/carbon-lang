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

  var filename: string(argv[1]);
  var outfilename: string = filename.substr(0, filename.find_last_of('.')) + ".woff2";
  fprintf(stdout, "Processing %s => %s\n",
    filename.c_str(), outfilename.c_str());
  var input: string = woff2::GetFileContent(filename);

  var input_data: const auto* = reinterpret_cast<const uint8_t*>(input.data());
  var output_size: size_t = woff2::MaxWOFF2CompressedSize(input_data, input.size());
  var output: string(output_size, 0);
  var output_data: auto* = reinterpret_cast<uint8_t*>(&output[0]);

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
