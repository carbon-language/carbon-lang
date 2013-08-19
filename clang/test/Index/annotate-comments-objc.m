// Run lines are sensitive to line numbers and come below the code.

#ifndef HEADER
#define HEADER

@class NSString;

//===---
// rdar://14258334
// Check that we attach comments to properties correctly.
//===---

@interface MyClass {
}

/// property1_isdoxy1 IS_DOXYGEN_SINGLE
@property (nonatomic, copy, readwrite) NSString *property1_isdoxy1;
@property (nonatomic, copy, readwrite) NSString *property1_isdoxy2; ///< property1_isdoxy2 IS_DOXYGEN_SINGLE
@property (nonatomic, copy, readwrite) NSString *property1_isdoxy3; /**< property1_isdoxy3 IS_DOXYGEN_SINGLE */
@property (nonatomic, copy, readwrite) NSString *property1_isdoxy4; /*!< property1_isdoxy4 IS_DOXYGEN_SINGLE */

/// method1_isdoxy1 IS_DOXYGEN_SINGLE
- (void)method1_isdoxy1;
- (void)method1_isdoxy2; /*!< method1_isdoxy2 IS_DOXYGEN_SINGLE */
- (void)method1_isdoxy3; /*!< method1_isdoxy3 IS_DOXYGEN_SINGLE */
- (void)method1_isdoxy4; /*!< method1_isdoxy4 IS_DOXYGEN_SINGLE */
@end


#endif

// RUN: rm -rf %t
// RUN: mkdir %t

// Check that we serialize comment source locations properly.
// RUN: %clang_cc1 -emit-pch -o %t/out.pch %s
// RUN: %clang_cc1 -include-pch %t/out.pch -fsyntax-only %s

// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out.c-index-direct
// RUN: c-index-test -test-load-tu %t/out.pch all > %t/out.c-index-pch

// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-direct
// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-pch

// Declarations without Doxygen comments should not pick up some Doxygen comments.
// WRONG-NOT: notdoxy{{.*}}Comment=
// WRONG-NOT: test{{.*}}Comment=

// Non-Doxygen comments should not be attached to anything.
// WRONG-NOT: NOT_DOXYGEN

// Some Doxygen comments are not attached to anything.
// WRONG-NOT: IS_DOXYGEN_NOT_ATTACHED

// Ensure we don't pick up extra comments.
// WRONG-NOT: IS_DOXYGEN_START{{.*}}IS_DOXYGEN_START{{.*}}BriefComment=
// WRONG-NOT: IS_DOXYGEN_END{{.*}}IS_DOXYGEN_END{{.*}}BriefComment=
//
// Ensure that XML is not invalid
// WRONG-NOT: CommentXMLInvalid

// RUN: FileCheck %s < %t/out.c-index-direct
// RUN: FileCheck %s < %t/out.c-index-pch

// These CHECK lines are not located near the code on purpose.  This test
// checks that documentation comments are attached to declarations correctly.
// Adding a non-documentation comment with CHECK line between every two
// documentation comments will only test a single code path.
//
// CHECK: annotate-comments-objc.m:17:50: ObjCPropertyDecl=property1_isdoxy1:{{.*}} property1_isdoxy1 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:18:50: ObjCPropertyDecl=property1_isdoxy2:{{.*}} property1_isdoxy2 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:19:50: ObjCPropertyDecl=property1_isdoxy3:{{.*}} property1_isdoxy3 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:20:50: ObjCPropertyDecl=property1_isdoxy4:{{.*}} property1_isdoxy4 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:23:9: ObjCInstanceMethodDecl=method1_isdoxy1:{{.*}} method1_isdoxy1 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:24:9: ObjCInstanceMethodDecl=method1_isdoxy2:{{.*}} method1_isdoxy2 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:25:9: ObjCInstanceMethodDecl=method1_isdoxy3:{{.*}} method1_isdoxy3 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:26:9: ObjCInstanceMethodDecl=method1_isdoxy4:{{.*}} method1_isdoxy4 IS_DOXYGEN_SINGLE

