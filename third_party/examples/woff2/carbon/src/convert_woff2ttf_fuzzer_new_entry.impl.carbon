#include <string>
#include <woff2/decode.h>

extern "C" fn LLVMFuzzerTestOneInput(data: const uint8_t*, data_size: size_t) -> int {
  // Decode using newer entry pattern.
  // Same pattern as woff2_decompress.
  var output: std::string(std::min(woff2::ComputeWOFF2FinalSize(data, data_size),
                              woff2::kDefaultMaxSize), 0);
  var out: woff2::WOFF2StringOut(&output);
  woff2::ConvertWOFF2ToTTF(data, data_size, &out);
  return 0;
}
