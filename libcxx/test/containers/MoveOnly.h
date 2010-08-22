#ifndef MOVEONLY_H
#define MOVEONLY_H

#ifdef _LIBCPP_MOVE

#include <cstddef>
#include <functional>

class MoveOnly
{
    MoveOnly(const MoveOnly&);
    MoveOnly& operator=(const MoveOnly&);

    int data_;
public:
    MoveOnly(int data = 1) : data_(data) {}
    MoveOnly(MoveOnly&& x)
        : data_(x.data_) {x.data_ = 0;}
    MoveOnly& operator=(MoveOnly&& x)
        {data_ = x.data_; x.data_ = 0; return *this;}

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

#endif  // _LIBCPP_MOVE

#endif  // MOVEONLY_H
