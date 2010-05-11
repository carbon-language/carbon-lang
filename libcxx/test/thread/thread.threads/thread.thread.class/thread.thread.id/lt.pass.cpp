//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <thread>

// class thread::id

// bool operator< (thread::id x, thread::id y);
// bool operator<=(thread::id x, thread::id y);
// bool operator> (thread::id x, thread::id y);
// bool operator>=(thread::id x, thread::id y);

#include <thread>
#include <cassert>

int main()
{
    std::thread::id id0;
    std::thread::id id1;
    std::thread::id id2 = std::this_thread::get_id();
    assert(!(id0 <  id1));
    assert( (id0 <= id1));
    assert(!(id0 >  id1));
    assert( (id0 >= id1));
    assert( (id0 <  id2));
    assert( (id0 <= id2));
    assert(!(id0 >  id2));
    assert(!(id0 >= id2));
}
