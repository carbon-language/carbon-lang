//===-- LevelChange.h - Passes for raising/lowering llvm code ----*- C++ -*--=//
//
// This family of passes is useful for changing the 'level' of a module. This
// can either be raising (f.e. converting direct addressing to use getelementptr
// for structs and arrays), or lowering (for instruction selection).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_LEVELCHANGE_H
#define LLVM_TRANSFORMS_LEVELCHANGE_H

class Pass;

// RaisePointerReferences - Try to eliminate as many pointer arithmetic
// expressions as possible, by converting expressions to use getelementptr and
// friends.
//
Pass *createRaisePointerReferencesPass();

#endif
