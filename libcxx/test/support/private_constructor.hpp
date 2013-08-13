#ifndef __PRIVATE_CONSTRUCTOR__H
#define __PRIVATE_CONSTRUCTOR__H

#include <iostream>

struct PrivateConstructor {

    PrivateConstructor static make ( int v ) { return PrivateConstructor(v); }
    int get () const { return val; }
private:
    PrivateConstructor ( int v ) : val(v) {}
    int val;
    };

bool operator < ( const PrivateConstructor &lhs, const PrivateConstructor &rhs ) { return lhs.get() < rhs.get(); }

bool operator < ( const PrivateConstructor &lhs, int rhs ) { return lhs.get() < rhs; }
bool operator < ( int lhs, const PrivateConstructor &rhs ) { return lhs < rhs.get(); }

std::ostream & operator << ( std::ostream &os, const PrivateConstructor &foo ) { return os << foo.get (); }

#endif
