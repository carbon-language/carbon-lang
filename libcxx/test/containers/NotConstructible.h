//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef NOTCONSTRUCTIBLE_H
#define NOTCONSTRUCTIBLE_H

#include <functional>

class NotConstructible
{
    NotConstructible(const NotConstructible&);
    NotConstructible& operator=(const NotConstructible&);
public:
};

inline
bool
operator==(const NotConstructible&, const NotConstructible&)
{return true;}

namespace std
{

template <>
struct hash<NotConstructible>
    : public std::unary_function<NotConstructible, std::size_t>
{
    std::size_t operator()(const NotConstructible&) const {return 0;}
};

}

#endif  // NOTCONSTRUCTIBLE_H
