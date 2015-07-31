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

// Test unique_ptr converting move ctor

// NOTE: unique_ptr does not provide converting constructors in c++03
// XFAIL: c++98, c++03


#include <memory>
#include <type_traits>
#include <utility>
#include <cassert>

#include "../../deleter.h"

// test converting move ctor.  Should only require a MoveConstructible deleter, or if
//    deleter is a reference, not even that.
// Explicit version

struct A
{
    static int count;
    A() {++count;}
    A(const A&) {++count;}
    virtual ~A() {--count;}
};

int A::count = 0;

struct B
    : public A
{
    static int count;
    B() {++count;}
    B(const B&) {++count;}
    virtual ~B() {--count;}
};

int B::count = 0;

template <class OrigPtr, class NewPtr>
void checkDeleter(OrigPtr& O, NewPtr& N, int OrigState, int NewState) {
    typedef typename NewPtr::deleter_type NewDel;
    if (std::is_reference<NewDel>::value) {
      O.get_deleter().set_state(42);
      assert(O.get_deleter().state() == 42);
      assert(N.get_deleter().state() == 42);
      O.get_deleter().set_state(99);
      assert(O.get_deleter().state() == 99);
      assert(N.get_deleter().state() == 99);
    } else {
      // TODO(EricWF) Enable this?
      // assert(OrigState != NewState);
      assert(O.get_deleter().state() == OrigState);
      assert(N.get_deleter().state() == NewState);
    }
}

template <class APtr, class BPtr>
void testMoveConvertExplicit()
{
    { // Test explicit constructor
        BPtr s(new B);
        A* p = s.get();
        APtr s2(std::move(s));
        assert(s2.get() == p);
        assert(s.get() == 0);
        assert(A::count == 1);
        assert(B::count == 1);
    }
    assert(A::count == 0);
    assert(B::count == 0);
}

template <class APtr, class BPtr>
void testMoveConvertImplicit() {

    { // Test implicit constructor
        BPtr s(new B);
        A* p = s.get();
        APtr s2 = std::move(s);
        assert(s2.get() == p);
        assert(s.get() == 0);
        assert(A::count == 1);
        assert(B::count == 1);
    }
    assert(A::count == 0);
    assert(B::count == 0);
}

template <class APtr, class BPtr, class Deleter>
#if TEST_STD_VER >= 11
void testMoveConvertExplicit(Deleter&& del) {
#else
void testMoveConvert(Deleter& del) {
#endif
    int old_val = del.state();
    { // Test Explicit constructor
        BPtr s(new B, std::forward<Deleter>(del));
        A* p = s.get();
        APtr s2(std::move(s));
        assert(s2.get() == p);
        assert(s.get() == 0);
        checkDeleter(s, s2, del.state(), old_val);
        assert(A::count == 1);
        assert(B::count == 1);
    }
    assert(A::count == 0);
    assert(B::count == 0);
}


template <class APtr, class BPtr, class Deleter>
#if TEST_STD_VER >= 11
void testMoveConvertImplicit(Deleter&& del) {
#else
void testMoveConvertImplicit(Deleter& del) {
#endif
    int old_val = del.state();
    { // Test Implicit constructor
        BPtr s(new B, std::forward<Deleter>(del));
        A* p = s.get();
        APtr s2 = std::move(s);
        assert(s2.get() == p);
        assert(s.get() == 0);
        checkDeleter(s, s2, del.state(), old_val);
        assert(A::count == 1);
        assert(B::count == 1);
    }
    assert(A::count == 0);
    assert(B::count == 0);
}
int main()
{
    {
        typedef std::unique_ptr<A> APtr;
        typedef std::unique_ptr<B> BPtr;
        testMoveConvertExplicit<APtr, BPtr>();
        testMoveConvertImplicit<APtr, BPtr>();
    }
    {
        typedef std::unique_ptr<A, Deleter<A> > APtr;
        typedef std::unique_ptr<B, Deleter<B> > BPtr;
        Deleter<B> del(5);
        testMoveConvertExplicit<APtr, BPtr>(std::move(del));
        del.set_state(5);
        testMoveConvertImplicit<APtr, BPtr>(std::move(del));
    }
    {
        typedef std::unique_ptr<A, NCDeleter<A>& > APtr;
        typedef std::unique_ptr<B, NCDeleter<A>& > BPtr;
        NCDeleter<A> del(5);
        testMoveConvertExplicit<APtr, BPtr>(del);
        testMoveConvertImplicit<APtr, BPtr>(del);
    }
    {
        typedef std::unique_ptr<A, CDeleter<A> > APtr;
        typedef std::unique_ptr<B, CDeleter<B>& > BPtr;
        CDeleter<B> del(5);
        testMoveConvertImplicit<APtr, BPtr>(del);
        testMoveConvertExplicit<APtr, BPtr>(del);
    }
}
