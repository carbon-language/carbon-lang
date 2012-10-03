// rdar://12378879

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s < %t/out

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

/// Aaa.
void attr_availability_1() __attribute__((availability(macosx,obsoleted=10.0,introduced=8.0,deprecated=9.0, message="use availability_test in <foo.h>")))
                           __attribute__((availability(ios,unavailable, message="not for iOS")));

// CHECK: annotate-comments-availability-attrs.cpp:13:6: FunctionDecl=attr_availability_1:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments-availability-attrs.cpp" line="13" column="6"><Name>attr_availability_1</Name><USR>c:@F@attr_availability_1#</USR><Abstract><Para> Aaa.</Para></Abstract><Availability distribution="iOS"><DeprecationSummary>not for iOS</DeprecationSummary><Unavailable/></Availability><Availability distribution="OS X"><IntroducedInVersion>8.0</IntroducedInVersion><DeprecatedInVersion>9.0</DeprecatedInVersion><RemovedAfterVersion>10.0</RemovedAfterVersion><DeprecationSummary>use availability_test in &lt;foo.h&gt;</DeprecationSummary></Availability></Function>]

/// Aaa.
void attr_availability_2() __attribute__((availability(macosx,obsoleted=10.0.1,introduced=8.0.1,deprecated=9.0.1)));

// CHECK: annotate-comments-availability-attrs.cpp:19:6: FunctionDecl=attr_availability_2:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments-availability-attrs.cpp" line="19" column="6"><Name>attr_availability_2</Name><USR>c:@F@attr_availability_2#</USR><Abstract><Para> Aaa.</Para></Abstract><Availability distribution="OS X"><IntroducedInVersion>8.0.1</IntroducedInVersion><DeprecatedInVersion>9.0.1</DeprecatedInVersion><RemovedAfterVersion>10.0.1</RemovedAfterVersion></Availability></Function>]

/// Aaa.
void attr_deprecated_1() __attribute__((deprecated));

// CHECK: annotate-comments-availability-attrs.cpp:24:6: FunctionDecl=attr_deprecated_1:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments-availability-attrs.cpp" line="24" column="6"><Name>attr_deprecated_1</Name><USR>c:@F@attr_deprecated_1#</USR><Abstract><Para> Aaa.</Para></Abstract><Deprecated/></Function>]

/// Aaa.
void attr_deprecated_2() __attribute__((deprecated("message 1 <foo.h>")));

// CHECK: annotate-comments-availability-attrs.cpp:29:6: FunctionDecl=attr_deprecated_2:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments-availability-attrs.cpp" line="29" column="6"><Name>attr_deprecated_2</Name><USR>c:@F@attr_deprecated_2#</USR><Abstract><Para> Aaa.</Para></Abstract><Deprecated>message 1 &lt;foo.h&gt;</Deprecated></Function>]

/// Aaa.
void attr_unavailable_1() __attribute__((unavailable));

// CHECK: annotate-comments-availability-attrs.cpp:34:6: FunctionDecl=attr_unavailable_1:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments-availability-attrs.cpp" line="34" column="6"><Name>attr_unavailable_1</Name><USR>c:@F@attr_unavailable_1#</USR><Abstract><Para> Aaa.</Para></Abstract><Unavailable/></Function>]

/// Aaa.
void attr_unavailable_2() __attribute__((unavailable("message 2 <foo.h>")));

// CHECK: annotate-comments-availability-attrs.cpp:39:6: FunctionDecl=attr_unavailable_2:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments-availability-attrs.cpp" line="39" column="6"><Name>attr_unavailable_2</Name><USR>c:@F@attr_unavailable_2#</USR><Abstract><Para> Aaa.</Para></Abstract><Unavailable>message 2 &lt;foo.h&gt;</Unavailable></Function>]

