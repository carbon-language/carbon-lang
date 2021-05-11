/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Helper functions for woff2 variable length types: 255UInt16 and UIntBase128 */

#ifndef WOFF2_VARIABLE_LENGTH_H_
#define WOFF2_VARIABLE_LENGTH_H_

<<<<<<< HEAD
#include <cinttypes>
=======
#include <inttypes.h>
>>>>>>> trunk
#include <vector>
#include "./buffer.h"

namespace woff2 {

<<<<<<< HEAD
auto Size255UShort(uint16_t value) -> size_t;
auto Read255UShort(Buffer* buf, unsigned int* value) -> bool;
void Write255UShort(std::vector<uint8_t>* out, int value);
void Store255UShort(int val, size_t* offset, uint8_t* dst);

auto Base128Size(size_t n) -> size_t;
auto ReadBase128(Buffer* buf, uint32_t* value) -> bool;
=======
size_t Size255UShort(uint16_t value);
bool Read255UShort(Buffer* buf, unsigned int* value);
void Write255UShort(std::vector<uint8_t>* out, int value);
void Store255UShort(int val, size_t* offset, uint8_t* dst);

size_t Base128Size(size_t n);
bool ReadBase128(Buffer* buf, uint32_t* value);
>>>>>>> trunk
void StoreBase128(size_t len, size_t* offset, uint8_t* dst);

} // namespace woff2

#endif  // WOFF2_VARIABLE_LENGTH_H_

