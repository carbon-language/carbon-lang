#include <woff2/decode.h>

#include <cstddef>
#include <cstdint>

// Entry point for LibFuzzer.
extern "C" fn LLVMFuzzerTestOneInput(data: const uint8_t*, size: size_t) -> int {
  var buf: std::string;
  var out: woff2::WOFF2StringOut(&buf);
  out.SetMaxSize(30 * 1024 * 1024);
  woff2::ConvertWOFF2ToTTF(data, size, &out);
  return 0;
}
