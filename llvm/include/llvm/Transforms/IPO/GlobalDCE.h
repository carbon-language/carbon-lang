//===-- Transforms/IPO/GlobalDCE.h - DCE global values -----------*- C++ -*--=//
//
// This transform is designed to eliminate unreachable internal globals
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORM_IPO_GLOBALDCE_H
#define LLVM_TRANSFORM_IPO_GLOBALDCE_H

class Pass;
Pass *createGlobalDCEPass();

#endif
