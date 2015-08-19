//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Example move-only deleter

#ifndef DELETER_H
#define DELETER_H

#include <type_traits>
#include <utility>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER >= 11

template <class T>
class Deleter
{
    int state_;

    Deleter(const Deleter&);
    Deleter& operator=(const Deleter&);

public:
    Deleter(Deleter&& r) : state_(r.state_) {r.state_ = 0;}
    Deleter& operator=(Deleter&& r)
    {
        state_ = r.state_;
        r.state_ = 0;
        return *this;
    }


    Deleter() : state_(0) {}
    explicit Deleter(int s) : state_(s) {}
    ~Deleter() {assert(state_ >= 0); state_ = -1;}

    template <class U>
        Deleter(Deleter<U>&& d,
            typename std::enable_if<!std::is_same<U, T>::value>::type* = 0)
            : state_(d.state()) {d.set_state(0);}

private:
    template <class U>
        Deleter(const Deleter<U>& d,
            typename std::enable_if<!std::is_same<U, T>::value>::type* = 0);
public:
    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete p;}
};

template <class T>
class Deleter<T[]>
{
    int state_;

    Deleter(const Deleter&);
    Deleter& operator=(const Deleter&);

public:

    Deleter(Deleter&& r) : state_(r.state_) {r.state_ = 0;}
    Deleter& operator=(Deleter&& r)
    {
        state_ = r.state_;
        r.state_ = 0;
        return *this;
    }

    Deleter() : state_(0) {}
    explicit Deleter(int s) : state_(s) {}
    ~Deleter() {assert(state_ >= 0); state_ = -1;}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete [] p;}
};

#else // TEST_STD_VER < 11

template <class T>
class Deleter
{
    mutable int state_;

public:
    Deleter() : state_(0) {}
    explicit Deleter(int s) : state_(s) {}

    Deleter(Deleter const & other) : state_(other.state_) {
        other.state_ = 0;
    }
    Deleter& operator=(Deleter const& other) {
        state_ = other.state_;
        other.state_ = 0;
        return *this;
    }

    ~Deleter() {assert(state_ >= 0); state_ = -1;}

    template <class U>
        Deleter(Deleter<U> d,
            typename std::enable_if<!std::is_same<U, T>::value>::type* = 0)
            : state_(d.state()) {}

public:
    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete p;}
};

template <class T>
class Deleter<T[]>
{
    mutable int state_;

public:

    Deleter(Deleter const& other) : state_(other.state_) {
        other.state_ = 0;
    }
    Deleter& operator=(Deleter const& other) {
        state_ = other.state_;
        other.state_ = 0;
        return *this;
    }

    Deleter() : state_(0) {}
    explicit Deleter(int s) : state_(s) {}
    ~Deleter() {assert(state_ >= 0); state_ = -1;}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete [] p;}
};

#endif

template <class T>
void
swap(Deleter<T>& x, Deleter<T>& y)
{
    Deleter<T> t(std::move(x));
    x = std::move(y);
    y = std::move(t);
}


template <class T>
class CDeleter
{
    int state_;

public:

    CDeleter() : state_(0) {}
    explicit CDeleter(int s) : state_(s) {}
    ~CDeleter() {assert(state_ >= 0); state_ = -1;}

    template <class U>
        CDeleter(const CDeleter<U>& d)
            : state_(d.state()) {}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete p;}
};

template <class T>
class CDeleter<T[]>
{
    int state_;

public:

    CDeleter() : state_(0) {}
    explicit CDeleter(int s) : state_(s) {}
    ~CDeleter() {assert(state_ >= 0); state_ = -1;}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete [] p;}
};

template <class T>
void
swap(CDeleter<T>& x, CDeleter<T>& y)
{
    CDeleter<T> t(std::move(x));
    x = std::move(y);
    y = std::move(t);
}

// Non-copyable deleter
template <class T>
class NCDeleter
{
    int state_;
    NCDeleter(NCDeleter const&);
    NCDeleter& operator=(NCDeleter const&);
public:

    NCDeleter() : state_(0) {}
    explicit NCDeleter(int s) : state_(s) {}
    ~NCDeleter() {assert(state_ >= 0); state_ = -1;}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete p;}
};


template <class T>
class NCDeleter<T[]>
{
    int state_;
    NCDeleter(NCDeleter const&);
    NCDeleter& operator=(NCDeleter const&);
public:

    NCDeleter() : state_(0) {}
    explicit NCDeleter(int s) : state_(s) {}
    ~NCDeleter() {assert(state_ >= 0); state_ = -1;}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete [] p;}
};


// Non-copyable deleter
template <class T>
class NCConstDeleter
{
    int state_;
    NCConstDeleter(NCConstDeleter const&);
    NCConstDeleter& operator=(NCConstDeleter const&);
public:

    NCConstDeleter() : state_(0) {}
    explicit NCConstDeleter(int s) : state_(s) {}
    ~NCConstDeleter() {assert(state_ >= 0); state_ = -1;}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) const {delete p;}
};


template <class T>
class NCConstDeleter<T[]>
{
    int state_;
    NCConstDeleter(NCConstDeleter const&);
    NCConstDeleter& operator=(NCConstDeleter const&);
public:

    NCConstDeleter() : state_(0) {}
    explicit NCConstDeleter(int s) : state_(s) {}
    ~NCConstDeleter() {assert(state_ >= 0); state_ = -1;}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) const {delete [] p;}
};


// Non-copyable deleter
template <class T>
class CopyDeleter
{
    int state_;
public:

    CopyDeleter() : state_(0) {}
    explicit CopyDeleter(int s) : state_(s) {}
    ~CopyDeleter() {assert(state_ >= 0); state_ = -1;}

    CopyDeleter(CopyDeleter const& other) : state_(other.state_) {}
    CopyDeleter& operator=(CopyDeleter const& other) {
        state_ = other.state_;
        return *this;
    }

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete p;}
};


template <class T>
class CopyDeleter<T[]>
{
    int state_;

public:

    CopyDeleter() : state_(0) {}
    explicit CopyDeleter(int s) : state_(s) {}
    ~CopyDeleter() {assert(state_ >= 0); state_ = -1;}

    CopyDeleter(CopyDeleter const& other) : state_(other.state_) {}
    CopyDeleter& operator=(CopyDeleter const& other) {
        state_ = other.state_;
        return *this;
    }

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {delete [] p;}
};


#endif  // DELETER_H
