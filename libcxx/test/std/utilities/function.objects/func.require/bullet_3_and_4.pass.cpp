//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// INVOKE (f, t1, t2, ..., tN)

//------------------------------------------------------------------------------
// TESTING INVOKE(f, t1, t2, ..., tN)
//   - Bullet 3 -- t1.*f
//   - Bullet 4 -- (*t1).*f
//
// Overview:
//    Bullets 3 and 4 handle the case where 'f' is a pointer to member object.
//    Bullet 3 only handles the cases where t1 is an object of type T or a
//    type derived from 'T'. Bullet 4 handles all other cases.
//
// Concerns:
//   1) The return type is always an lvalue reference.
//   2) The return type is not less cv-qualified that the object that contains it.
//   3) The return type is not less cv-qualified than object type.
//   4) The call object is perfectly forwarded.
//   5) Classes that are publicly derived from 'T' are accepted as the call object
//   6) All types that dereference to T or a type derived from T can be used
//      as the call object.
//   7) Pointers to T or a type derived from T can be used as the call object.

#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "invoke_helpers.h"

template <class Tp>
struct TestMemberObject {
    TestMemberObject() : object() {}
    Tp object;
private:
    TestMemberObject(TestMemberObject const&);
    TestMemberObject& operator=(TestMemberObject const&);
};

template <class ObjectType>
struct TestCase {
    public:

    static void run() { TestCase().doTest(); }

private:
    typedef TestMemberObject<ObjectType> TestType;

    //==========================================================================
    // TEST DISPATCH
    void doTest() {
        typedef DerivedFromType<TestType> Derived;
        TestType obj;
        TestType* obj_ptr = &obj;
        Derived der;
        Derived* der_ptr = &der;
        DerefToType<TestType>   dref;
        DerefPropType<TestType> dref2;

        {
            typedef ObjectType (TestType::*MemPtr);
            typedef ObjectType E;
            MemPtr M = &TestType::object;
            runTestDispatch<E>(M, obj, &obj.object);
            runTestDispatch<E>(M, der, &der.object);
            runTestDispatch<E>(M, dref2, &dref2.object.object);
            runTestPointerDispatch<E>(M, obj_ptr, &obj_ptr->object);
            runTestPointerDispatch<E>(M, der_ptr, &der_ptr->object);
            runTestPointerDispatch<E>(M, dref, &dref.object.object);
        }
        {
            typedef ObjectType const (TestType::*CMemPtr);
            typedef ObjectType const E;
            CMemPtr M = &TestType::object;
            runTestDispatch<E>(M, obj, &obj.object);
            runTestDispatch<E>(M, der, &der.object);
            runTestDispatch<E>(M, dref2, &dref2.object.object);
            runTestPointerDispatch<E>(M, obj_ptr, &obj_ptr->object);
            runTestPointerDispatch<E>(M, der_ptr, &der_ptr->object);
            runTestPointerDispatch<E>(M, dref, &dref.object.object);
        }
        {
            typedef ObjectType volatile (TestType::*VMemPtr);
            typedef ObjectType volatile E;
            VMemPtr M = &TestType::object;
            runTestDispatch<E>(M, obj,  &obj.object);
            runTestDispatch<E>(M, der,  &der.object);
            runTestDispatch<E>(M, dref2, &dref2.object.object);
            runTestPointerDispatch<E>(M, obj_ptr, &obj_ptr->object);
            runTestPointerDispatch<E>(M, der_ptr, &der_ptr->object);
            runTestPointerDispatch<E>(M, dref,    &dref.object.object);
        }
        {
            typedef ObjectType const volatile (TestType::*CVMemPtr);
            typedef ObjectType const volatile E;
            CVMemPtr M = &TestType::object;
            runTestDispatch<E>(M, obj,   &obj.object);
            runTestDispatch<E>(M, der,   &der.object);
            runTestDispatch<E>(M, dref2, &dref2.object.object);
            runTestPointerDispatch<E>(M, obj_ptr, &obj_ptr->object);
            runTestPointerDispatch<E>(M, der_ptr, &der_ptr->object);
            runTestPointerDispatch<E>(M, dref,    &dref.object.object);
        }
    }

    template <class Expect, class Fn, class T>
    void runTestDispatch(Fn M, T& obj, ObjectType* expect) {
        runTest<Expect &>              (M, C_<T&>(obj),                expect);
        runTest<Expect const&>         (M, C_<T const&>(obj),          expect);
        runTest<Expect volatile&>      (M, C_<T volatile&>(obj),       expect);
        runTest<Expect const volatile&>(M, C_<T const volatile&>(obj), expect);
#if TEST_STD_VER >= 11
        runTest<Expect&&>               (M, C_<T&&>(obj),                expect);
        runTest<Expect const&&>         (M, C_<T const&&>(obj),          expect);
        runTest<Expect volatile&&>      (M, C_<T volatile&&>(obj),       expect);
        runTest<Expect const volatile&&>(M, C_<T const volatile&&>(obj), expect);
#endif
    }

    template <class Expect, class Fn, class T>
    void runTestPointerDispatch(Fn M, T& obj, ObjectType* expect) {
        runTest<Expect&>(M, C_<T &>(obj),               expect);
        runTest<Expect&>(M, C_<T const&>(obj),          expect);
        runTest<Expect&>(M, C_<T volatile&>(obj),       expect);
        runTest<Expect&>(M, C_<T const volatile&>(obj), expect);
#if TEST_STD_VER >= 11
        runTest<Expect&>(M, C_<T&&>(obj),                expect);
        runTest<Expect&>(M, C_<T const&&>(obj),          expect);
        runTest<Expect&>(M, C_<T volatile&&>(obj),       expect);
        runTest<Expect&>(M, C_<T const volatile&&>(obj), expect);
#endif
    }

    template <class Expect, class Fn, class T>
#if TEST_STD_VER >= 11
    void runTest(Fn M, T&& obj, ObjectType* expect) {
#else
    void runTest(Fn M, T& obj, ObjectType* expect ) {
#endif
        static_assert((std::is_same<
            decltype(std::__invoke(M, std::forward<T>(obj))), Expect
          >::value), "");
        Expect e = std::__invoke(M, std::forward<T>(obj));
        assert(&e == expect);
    }
};

int main() {
    TestCase<ArgType>::run();
    TestCase<ArgType const>::run();
    TestCase<ArgType volatile>::run();
    TestCase<ArgType const volatile>::run();
    TestCase<ArgType*>::run();
}