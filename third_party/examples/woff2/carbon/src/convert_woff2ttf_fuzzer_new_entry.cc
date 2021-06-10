#include <string>
#include <woff2/decode.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t data_size) {
  // Decode using newer entry pattern.
  // Same pattern as woff2_decompress.
  std::string output(std::min(woff2::ComputeWOFF2FinalSize(data, data_size),
                              woff2::kDefaultMaxSize), 0);
  woff2::WOFF2StringOut out(&output);
  woff2::ConvertWOFF2ToTTF(data, data_size, &out);
  return 0;
}
