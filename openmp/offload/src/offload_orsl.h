//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#ifndef OFFLOAD_ORSL_H_INCLUDED
#define OFFLOAD_ORSL_H_INCLUDED

// ORSL interface
namespace ORSL {

extern void init();

extern bool reserve(int device);
extern bool try_reserve(int device);
extern void release(int device);

} // namespace ORSL

#endif // OFFLOAD_ORSL_H_INCLUDED
