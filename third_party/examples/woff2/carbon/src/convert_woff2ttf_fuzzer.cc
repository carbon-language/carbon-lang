<<<<<<< HEAD
#include <cstddef>
#include <cstdint>
=======
#include <stddef.h>
#include <stdint.h>
>>>>>>> trunk

#include <woff2/decode.h>

// Entry point for LibFuzzer.
<<<<<<< HEAD
extern "C" auto LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) -> int {
=======
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
>>>>>>> trunk
  std::string buf;
  woff2::WOFF2StringOut out(&buf);
  out.SetMaxSize(30 * 1024 * 1024);
  woff2::ConvertWOFF2ToTTF(data, size, &out);
  return 0;
}
