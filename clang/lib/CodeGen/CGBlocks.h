//===-- CGBlocks.h - state for LLVM CodeGen for blocks ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the internal state used for llvm translation for block literals.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGBLOCKS_H
#define CLANG_CODEGEN_CGBLOCKS_H

namespace clang {

namespace CodeGen {

class BlockBase {
public:
    enum {
        BLOCK_NEEDS_FREE =        (1 << 24),
        BLOCK_HAS_COPY_DISPOSE =  (1 << 25),
        BLOCK_HAS_CXX_OBJ =       (1 << 26),
        BLOCK_IS_GC =             (1 << 27),
        BLOCK_IS_GLOBAL =         (1 << 28),
        BLOCK_HAS_DESCRIPTOR =    (1 << 29)
    };
};

class BlockModule : public BlockBase {
};

class BlockFunction : public BlockBase {
public:
    enum {
    BLOCK_FIELD_IS_OBJECT   =  3,  /* id, NSObject, __attribute__((NSObject)),
                                      block, ... */
    BLOCK_FIELD_IS_BLOCK    =  7,  /* a block variable */
    BLOCK_FIELD_IS_BYREF    =  8,  /* the on stack structure holding the __block
                                      variable */
    BLOCK_FIELD_IS_WEAK     = 16,  /* declared __weak, only used in byref copy
                                      helpers */
    BLOCK_BYREF_CALLER      = 128  /* called from __block (byref) copy/dispose
                                      support routines */
  };
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
