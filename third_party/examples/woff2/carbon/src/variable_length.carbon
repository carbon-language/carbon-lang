/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Helper functions for woff2 variable length types: 255UInt16 and UIntBase128 */

#ifndef WOFF2_VARIABLE_LENGTH_H_
#define WOFF2_VARIABLE_LENGTH_H_

#include <cinttypes>
#include <vector>

#include "./buffer.h"

namespace woff2 {

fn Size255UShort(value: uint16_t) -> size_t;
fn Read255UShort(buf: Buffer*, value: unsigned int*) -> bool;
fn Write255UShort(out: std::vector<uint8_t>*, value: int);
fn Store255UShort(val: int, offset: size_t*, dst: uint8_t*);

fn Base128Size(n: size_t) -> size_t;
fn ReadBase128(buf: Buffer*, value: uint32_t*) -> bool;
fn StoreBase128(len: size_t, offset: size_t*, dst: uint8_t*);

} // namespace woff2

#endif  // WOFF2_VARIABLE_LENGTH_H_

