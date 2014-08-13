// MSVC header files have compilation issues when compiling with exceptions disabled.  Notably,
// this function is compiled out when _HAS_EXCEPTIONS=0, but this function is called from another
// place even when _HAS_EXCEPTIONS=0.  So we define a dummy implementation as a workaround and
// force include this header file.
static void *__uncaught_exception() { return nullptr; }
