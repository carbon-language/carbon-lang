//===- ValueMapper.h - Interface shared by lib/Transforms/Utils -*- C++ -*-===//
//
// This file defines the MapValue interface which is used by various parts of
// the Transforms/Utils library to implement cloning and linking facilities.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_UTILS_VALUE_MAPPER_H
#define LIB_TRANSFORMS_UTILS_VALUE_MAPPER_H

#include <map>
class Value;

Value *MapValue(const Value *V, std::map<const Value*, Value*> &VM);

#endif
