-- mach_override.c is taken from upstream version at
 https://github.com/rentzsch/mach_star/tree/f8e0c424b5be5cb641ded67c265e616157ae4bcf
-- Added debugging code under DEBUG_DISASM.
-- The files are guarded with #ifdef __APPLE__
-- some opcodes are added in order to parse the library functions on Lion
-- fixupInstructions() is extended to relocate relative calls, not only jumps
-- mach_override_ptr is renamed to __asan_mach_override_ptr and
 other functions are marked as hidden.

