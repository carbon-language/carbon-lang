

#ifndef LLVM_TRANSFORMS_CLONE_FUNCTION_H
#define LLVM_TRANSFORMS_CLONE_FUNCTION_H

#include <vector>
class Function;
class Value;

// Clone OldFunc into NewFunc, transforming the old arguments into references to
// ArgMap values.  Note that if NewFunc already has basic blocks, the ones
// cloned into it will be added to the end of the function.
//
void CloneFunctionInto(Function *NewFunc, const Function *OldFunc,
                       const std::vector<Value*> &ArgMap);

#endif
