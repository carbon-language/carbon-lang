// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR6382 {
  int foo()
  {
    goto error;
    {
      struct BitPacker {
        BitPacker() {}
      };
      BitPacker packer;
    }

  error:
    return -1;
  }
}
