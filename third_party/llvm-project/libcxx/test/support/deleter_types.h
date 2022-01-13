//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Example move-only deleter

#ifndef SUPPORT_DELETER_TYPES_H
#define SUPPORT_DELETER_TYPES_H

#include <type_traits>
#include <utility>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

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
    template <class U>
        CDeleter(const CDeleter<U>& d)
            : state_(d.state()) {}

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


struct test_deleter_base
{
    static int count;
    static int dealloc_count;
};

int test_deleter_base::count = 0;
int test_deleter_base::dealloc_count = 0;

template <class T>
class test_deleter
    : public test_deleter_base
{
    int state_;

public:

    test_deleter() : state_(0) {++count;}
    explicit test_deleter(int s) : state_(s) {++count;}
    test_deleter(const test_deleter& d)
        : state_(d.state_) {++count;}
    ~test_deleter() {assert(state_ >= 0); --count; state_ = -1;}

    int state() const {return state_;}
    void set_state(int i) {state_ = i;}

    void operator()(T* p) {assert(state_ >= 0); ++dealloc_count; delete p;}
#if TEST_STD_VER >= 11
    test_deleter* operator&() const = delete;
#else
private:
  test_deleter* operator&() const;
#endif
};

template <class T>
void
swap(test_deleter<T>& x, test_deleter<T>& y)
{
    test_deleter<T> t(std::move(x));
    x = std::move(y);
    y = std::move(t);
}

#if TEST_STD_VER >= 11

template <class T, size_t ID = 0>
class PointerDeleter
{
    PointerDeleter(const PointerDeleter&);
    PointerDeleter& operator=(const PointerDeleter&);

public:
    typedef min_pointer<T, std::integral_constant<size_t, ID>> pointer;

    PointerDeleter() = default;
    PointerDeleter(PointerDeleter&&) = default;
    PointerDeleter& operator=(PointerDeleter&&) = default;
    explicit PointerDeleter(int) {}

    template <class U>
        PointerDeleter(PointerDeleter<U, ID>&&,
            typename std::enable_if<!std::is_same<U, T>::value>::type* = 0)
    {}

    void operator()(pointer p) { if (p) { delete std::addressof(*p); }}

private:
    template <class U>
        PointerDeleter(const PointerDeleter<U, ID>&,
            typename std::enable_if<!std::is_same<U, T>::value>::type* = 0);
};


template <class T, size_t ID>
class PointerDeleter<T[], ID>
{
    PointerDeleter(const PointerDeleter&);
    PointerDeleter& operator=(const PointerDeleter&);

public:
    typedef min_pointer<T, std::integral_constant<size_t, ID> > pointer;

    PointerDeleter() = default;
    PointerDeleter(PointerDeleter&&) = default;
    PointerDeleter& operator=(PointerDeleter&&) = default;
    explicit PointerDeleter(int) {}

    template <class U>
        PointerDeleter(PointerDeleter<U, ID>&&,
            typename std::enable_if<!std::is_same<U, T>::value>::type* = 0)
    {}

    void operator()(pointer p) { if (p) { delete [] std::addressof(*p); }}

private:
    template <class U>
        PointerDeleter(const PointerDeleter<U, ID>&,
            typename std::enable_if<!std::is_same<U, T>::value>::type* = 0);
};

#endif // TEST_STD_VER >= 11

template <class T>
class DefaultCtorDeleter
{
    int state_;

public:
    int state() const {return state_;}
    void operator()(T* p) {delete p;}
};

template <class T>
class DefaultCtorDeleter<T[]>
{
    int state_;

public:
    int state() const {return state_;}
    void operator()(T* p) {delete [] p;}
};

#endif // SUPPORT_DELETER_TYPES_H
