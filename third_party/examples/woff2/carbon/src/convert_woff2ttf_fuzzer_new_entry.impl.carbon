#include <string>
#include <woff2/decode.h>

extern "C" fn LLVMFuzzerTestOneInput(data: uint8_t*, data_size: size_t) -> int {
  // Decode using newer entry pattern.
  // Same pattern as woff2_decompress.
  var output: stringstd::;
  var out: WOFF2StringOutwoff2::;
  woff2::ConvertWOFF2ToTTF(data, data_size, &out);
  return 0;
}
