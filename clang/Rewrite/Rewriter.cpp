//===--- Rewriter.cpp - Code rewriting interface --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Rewriter class, which is used for code
//  transformations.
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/Rewriter.h"
using namespace clang;


void RewriteBuffer::RemoveText(unsigned OrigOffset, unsigned Size) {
  // FIXME:
}

void RewriteBuffer::InsertText(unsigned OrigOffset,
                               const char *StrData, unsigned StrLen) {
  // FIXME:
}
