//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MOVEONLY_H
#define MOVEONLY_H

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

#include <cstddef>
#include <functional>

class MoveOnly
{
    MoveOnly(const MoveOnly&);
    MoveOnly& operator=(const MoveOnly&);

    int data_;
public:
    MoveOnly(int data = 0) : data_(data) { assert(data != -1); }
    MoveOnly(MoveOnly &&x) : data_(x.data_) {
        assert(x.data_ != -1);
        x.data_ = -1;
    }
    MoveOnly &operator=(MoveOnly &&x) {
        assert(x.data_ != -1);
        data_ = x.data_;
        x.data_ = -1;
        return *this;
    }

    int get() const {return data_;}

    bool operator==(const MoveOnly& x) const {return data_ == x.data_;}
    bool operator< (const MoveOnly& x) const {return data_ <  x.data_;}
};

namespace std {

template <>
struct hash<MoveOnly>
    : public std::unary_function<MoveOnly, std::size_t>
{
    std::size_t operator()(const MoveOnly& x) const {return x.get();}
};

}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

#endif  // MOVEONLY_H
