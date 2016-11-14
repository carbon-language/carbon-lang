//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <ObjectType T> T* addressof(T&& r) = delete;

#include <memory>
#include <cassert>

int main()
{
	const int *p = std::addressof<const int>(0);
}
