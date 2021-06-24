#include <cstddef>
#include <cstdint>

#include <woff2/decode.h>

// Entry point for LibFuzzer.
extern "C" fn LLVMFuzzerTestOneInput(data: uint8_t*, size: size_t) -> int {
  var buf: stringstd::;
  var out: WOFF2StringOutwoff2::;
  out.SetMaxSize(30 * 1024 * 1024);
  woff2::ConvertWOFF2ToTTF(data, size, &out);
  return 0;
}
