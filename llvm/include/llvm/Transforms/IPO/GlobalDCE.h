//===-- Transforms/IPO/GlobalDCE.h - DCE global values -----------*- C++ -*--=//
//
// This transform is designed to eliminate unreachable internal globals
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORM_IPO_GLOBALDCE_H
#define LLVM_TRANSFORM_IPO_GLOBALDCE_H

#include "llvm/Pass.h"

namespace cfg { class CallGraph; }
class Module;

struct GlobalDCE : public Pass {

  // run - Do the GlobalDCE pass on the specified module, optionally updating
  // the specified callgraph to reflect the changes.
  //
  bool run(Module *M);
};

#endif
