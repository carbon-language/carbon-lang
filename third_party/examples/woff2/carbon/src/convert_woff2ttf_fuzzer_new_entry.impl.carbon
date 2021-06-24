#include <string>
#include <woff2/decode.h>

extern "C" fn LLVMFuzzerTestOneInput(data: const uint8_t *, data_size: size_t) -> int {
  // Decode using newer entry pattern.
  // Same pattern as woff2_decompress.
  var output: std::string;
  var out: woff2::WOFF2StringOut;
  woff2::ConvertWOFF2ToTTF(data, data_size, &out);
  return 0;
}
