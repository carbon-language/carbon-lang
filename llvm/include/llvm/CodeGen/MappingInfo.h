#ifndef LLVM_CODEGEN_MAPPINGINFO_H
#define LLVM_CODEGEN_MAPPINGINFO_H

#include <iosfwd>
class Pass;

Pass *MappingInfoForFunction(std::ostream &out);

#endif


