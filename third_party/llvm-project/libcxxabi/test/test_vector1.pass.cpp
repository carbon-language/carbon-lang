//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cxxabi.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "test_macros.h"

//  Wrapper routines
void *my_alloc2 ( size_t sz ) {
    void *p = std::malloc ( sz );
//  std::printf ( "Allocated %ld bytes at %lx\n", sz, (unsigned long) p );
    return p;
}

void my_dealloc2 ( void *p ) {
//  std::printf ( "Freeing %lx\n", (unsigned long) p );
    std::free ( p );
}

void my_dealloc3 ( void *p, size_t ) {
//  std::printf ( "Freeing %lx (size %ld)\n", (unsigned long) p, sz );
    std::free ( p );
}

void my_construct ( void * ) {
//  std::printf ( "Constructing %lx\n", (unsigned long) p );
}

void my_destruct  ( void * ) {
//  std::printf ( "Destructing  %lx\n", (unsigned long) p );
}

int gCounter;
void count_construct ( void * ) { ++gCounter; }
void count_destruct  ( void * ) { --gCounter; }


int gConstructorCounter;
int gConstructorThrowTarget;
int gDestructorCounter;
int gDestructorThrowTarget;
void throw_construct ( void * ) {
#ifndef TEST_HAS_NO_EXCEPTIONS
    if ( gConstructorCounter   == gConstructorThrowTarget )
        throw 1;
    ++gConstructorCounter;
#endif
}
void throw_destruct  ( void * ) {
#ifndef TEST_HAS_NO_EXCEPTIONS
    if ( ++gDestructorCounter  == gDestructorThrowTarget  )
        throw 2;
#endif
}

#if __cplusplus >= 201103L
#   define CAN_THROW noexcept(false)
#else
#   define CAN_THROW
#endif

struct vec_on_stack {
    void *storage;
    vec_on_stack () : storage ( __cxxabiv1::__cxa_vec_new    (            10, 40, 8, throw_construct, throw_destruct )) {}
    ~vec_on_stack () CAN_THROW {__cxxabiv1::__cxa_vec_delete ( storage,       40, 8,                  throw_destruct );  }
};

//  Test calls with empty constructors and destructors
int test_empty ( ) {
    void *one, *two, *three;

//  Try with no padding and no con/destructors
    one     = __cxxabiv1::__cxa_vec_new ( 10, 40, 0, NULL, NULL );
    two     = __cxxabiv1::__cxa_vec_new2( 10, 40, 0, NULL, NULL, my_alloc2, my_dealloc2 );
    three   = __cxxabiv1::__cxa_vec_new3( 10, 40, 0, NULL, NULL, my_alloc2, my_dealloc3 );

    __cxxabiv1::__cxa_vec_delete ( one,       40, 0, NULL );
    __cxxabiv1::__cxa_vec_delete2( two,       40, 0, NULL, my_dealloc2 );
    __cxxabiv1::__cxa_vec_delete3( three,     40, 0, NULL, my_dealloc3 );

//  Try with no padding
    one     = __cxxabiv1::__cxa_vec_new ( 10, 40, 0, my_construct, my_destruct );
    two     = __cxxabiv1::__cxa_vec_new2( 10, 40, 0, my_construct, my_destruct, my_alloc2, my_dealloc2 );
    three   = __cxxabiv1::__cxa_vec_new3( 10, 40, 0, my_construct, my_destruct, my_alloc2, my_dealloc3 );

    __cxxabiv1::__cxa_vec_delete ( one,       40, 0, my_destruct );
    __cxxabiv1::__cxa_vec_delete2( two,       40, 0, my_destruct, my_dealloc2 );
    __cxxabiv1::__cxa_vec_delete3( three,     40, 0, my_destruct, my_dealloc3 );

//  Padding and no con/destructors
    one     = __cxxabiv1::__cxa_vec_new ( 10, 40, 8, NULL, NULL );
    two     = __cxxabiv1::__cxa_vec_new2( 10, 40, 8, NULL, NULL, my_alloc2, my_dealloc2 );
    three   = __cxxabiv1::__cxa_vec_new3( 10, 40, 8, NULL, NULL, my_alloc2, my_dealloc3 );

    __cxxabiv1::__cxa_vec_delete ( one,       40, 8, NULL );
    __cxxabiv1::__cxa_vec_delete2( two,       40, 8, NULL, my_dealloc2 );
    __cxxabiv1::__cxa_vec_delete3( three,     40, 8, NULL, my_dealloc3 );

//  Padding with con/destructors
    one     = __cxxabiv1::__cxa_vec_new ( 10, 40, 8, my_construct, my_destruct );
    two     = __cxxabiv1::__cxa_vec_new2( 10, 40, 8, my_construct, my_destruct, my_alloc2, my_dealloc2 );
    three   = __cxxabiv1::__cxa_vec_new3( 10, 40, 8, my_construct, my_destruct, my_alloc2, my_dealloc3 );

    __cxxabiv1::__cxa_vec_delete ( one,       40, 8, my_destruct );
    __cxxabiv1::__cxa_vec_delete2( two,       40, 8, my_destruct, my_dealloc2 );
    __cxxabiv1::__cxa_vec_delete3( three,     40, 8, my_destruct, my_dealloc3 );

    return 0;
}

//  Make sure the constructors and destructors are matched
int test_counted ( ) {
    int retVal = 0;
    void *one, *two, *three;

//  Try with no padding
    gCounter = 0;
    one     = __cxxabiv1::__cxa_vec_new ( 10, 40, 0, count_construct, count_destruct );
    two     = __cxxabiv1::__cxa_vec_new2( 10, 40, 0, count_construct, count_destruct, my_alloc2, my_dealloc2 );
    three   = __cxxabiv1::__cxa_vec_new3( 10, 40, 0, count_construct, count_destruct, my_alloc2, my_dealloc3 );

    __cxxabiv1::__cxa_vec_delete ( one,       40, 0, count_destruct );
    __cxxabiv1::__cxa_vec_delete2( two,       40, 0, count_destruct, my_dealloc2 );
    __cxxabiv1::__cxa_vec_delete3( three,     40, 0, count_destruct, my_dealloc3 );

//  Since there was no padding, the # of elements in the array are not stored
//  and the destructors are not called.
    if ( gCounter != 30 ) {
        std::printf("Mismatched Constructor/Destructor calls (1)\n");
        std::printf("  Expected 30, got %d\n", gCounter);
        retVal = 1;
    }

    gCounter = 0;
    one     = __cxxabiv1::__cxa_vec_new ( 10, 40, 8, count_construct, count_destruct );
    two     = __cxxabiv1::__cxa_vec_new2( 10, 40, 8, count_construct, count_destruct, my_alloc2, my_dealloc2 );
    three   = __cxxabiv1::__cxa_vec_new3( 10, 40, 8, count_construct, count_destruct, my_alloc2, my_dealloc3 );

    __cxxabiv1::__cxa_vec_delete ( one,       40, 8, count_destruct );
    __cxxabiv1::__cxa_vec_delete2( two,       40, 8, count_destruct, my_dealloc2 );
    __cxxabiv1::__cxa_vec_delete3( three,     40, 8, count_destruct, my_dealloc3 );

    if ( gCounter != 0 ) {
        std::printf("Mismatched Constructor/Destructor calls (2)\n");
        std::printf("  Expected 0, got %d\n", gCounter);
        retVal = 1;
    }

    return retVal;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
//  Make sure the constructors and destructors are matched
int test_exception_in_constructor ( ) {
    int retVal = 0;
    void *one, *two, *three;

//  Try with no padding
    gConstructorCounter = gDestructorCounter = 0;
    gConstructorThrowTarget = 15;
    gDestructorThrowTarget  = -1;
    try {
        one = two = three = NULL;
        one     = __cxxabiv1::__cxa_vec_new ( 10, 40, 0, throw_construct, throw_destruct );
        two     = __cxxabiv1::__cxa_vec_new2( 10, 40, 0, throw_construct, throw_destruct, my_alloc2, my_dealloc2 );
        three   = __cxxabiv1::__cxa_vec_new3( 10, 40, 0, throw_construct, throw_destruct, my_alloc2, my_dealloc3 );
    }
    catch ( int i ) {}

    __cxxabiv1::__cxa_vec_delete ( one,       40, 0, throw_destruct );
    __cxxabiv1::__cxa_vec_delete2( two,       40, 0, throw_destruct, my_dealloc2 );
    __cxxabiv1::__cxa_vec_delete3( three,     40, 0, throw_destruct, my_dealloc3 );

//  Since there was no padding, the # of elements in the array are not stored
//  and the destructors are not called.
//  Since we threw after 15 calls to the constructor, we should see 5 calls to
//      the destructor from the partially constructed array.
    if ( gConstructorCounter - gDestructorCounter != 10 ) {
        std::printf("Mismatched Constructor/Destructor calls (1C)\n");
        std::printf("%d constructors, but %d destructors\n", gConstructorCounter, gDestructorCounter);
        retVal = 1;
    }

    gConstructorCounter = gDestructorCounter = 0;
    gConstructorThrowTarget = 15;
    gDestructorThrowTarget  = -1;
    try {
        one = two = three = NULL;
        one     = __cxxabiv1::__cxa_vec_new ( 10, 40, 8, throw_construct, throw_destruct );
        two     = __cxxabiv1::__cxa_vec_new2( 10, 40, 8, throw_construct, throw_destruct, my_alloc2, my_dealloc2 );
        three   = __cxxabiv1::__cxa_vec_new3( 10, 40, 8, throw_construct, throw_destruct, my_alloc2, my_dealloc3 );
    }
    catch ( int i ) {}

    __cxxabiv1::__cxa_vec_delete ( one,       40, 8, throw_destruct );
    __cxxabiv1::__cxa_vec_delete2( two,       40, 8, throw_destruct, my_dealloc2 );
    __cxxabiv1::__cxa_vec_delete3( three,     40, 8, throw_destruct, my_dealloc3 );

    if ( gConstructorCounter != gDestructorCounter ) {
        std::printf("Mismatched Constructor/Destructor calls (2C)\n");
        std::printf("%d constructors, but %d destructors\n", gConstructorCounter, gDestructorCounter);
        retVal = 1;
    }

    return retVal;
}
#endif

#ifndef TEST_HAS_NO_EXCEPTIONS
//  Make sure the constructors and destructors are matched
int test_exception_in_destructor ( ) {
    int retVal = 0;
    void *one, *two, *three;
    one = two = three = NULL;

//  Throw from within a destructor
    gConstructorCounter = gDestructorCounter = 0;
    gConstructorThrowTarget = -1;
    gDestructorThrowTarget  = 15;
    try {
        one = two = NULL;
        one     = __cxxabiv1::__cxa_vec_new ( 10, 40, 8, throw_construct, throw_destruct );
        two     = __cxxabiv1::__cxa_vec_new2( 10, 40, 8, throw_construct, throw_destruct, my_alloc2, my_dealloc2 );
    }
    catch ( int i ) {}

    try {
        __cxxabiv1::__cxa_vec_delete ( one,       40, 8, throw_destruct );
        __cxxabiv1::__cxa_vec_delete2( two,       40, 8, throw_destruct, my_dealloc2 );
        assert(false);
    }
    catch ( int i ) {}

//  We should have thrown in the middle of cleaning up "two", which means that
//  there should be 20 calls to the destructor and the try block should exit
//  before the assertion.
    if ( gConstructorCounter != 20 || gDestructorCounter != 20 ) {
        std::printf("Unexpected Constructor/Destructor calls (1D)\n");
        std::printf("Expected (20, 20), but got (%d, %d)\n", gConstructorCounter, gDestructorCounter);
        retVal = 1;
    }

//  Try throwing from a destructor - should be fine.
    gConstructorCounter = gDestructorCounter = 0;
    gConstructorThrowTarget = -1;
    gDestructorThrowTarget  = 5;
    try { vec_on_stack v; }
    catch ( int i ) {}

    if ( gConstructorCounter != gDestructorCounter ) {
        std::printf("Mismatched Constructor/Destructor calls (2D)\n");
        std::printf("%d constructors, but %d destructors\n", gConstructorCounter, gDestructorCounter);
        retVal = 1;
    }

    return retVal;
}
#endif

int main(int, char**) {
    int retVal = 0;
    retVal += test_empty ();
    retVal += test_counted ();
#ifndef TEST_HAS_NO_EXCEPTIONS
    retVal += test_exception_in_constructor ();
    retVal += test_exception_in_destructor ();
#endif
    return retVal;
}
