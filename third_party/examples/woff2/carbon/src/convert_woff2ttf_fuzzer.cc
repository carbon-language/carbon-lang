#include <cstddef>
#include <cstdint>

#include <woff2/decode.h>

// Entry point for LibFuzzer.
extern "C" auto LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) -> int {
  std::string buf;
  woff2::WOFF2StringOut out(&buf);
  out.SetMaxSize(30 * 1024 * 1024);
  woff2::ConvertWOFF2ToTTF(data, size, &out);
  return 0;
}
